// baracuda_atomic.cuh
//
// Phase 11.3 (Fuel team feedback #6) — uniform `atomicAdd` helpers for
// every value dtype the BW kernels need to scatter-add into.
//
// **Why this header exists.** CUDA's native `atomicAdd` is only:
//   * f32 from sm_20+ (universally available),
//   * f64 from sm_60+ (universally available on baracuda's sm_80 baseline),
//   * `__half` / `__nv_bfloat16` from sm_70+ / sm_80+ respectively, BUT
//     only via the *paired* `__half2` / `__nv_bfloat162` form for many
//     CUDA versions, AND the single-element scalar form has had
//     intermittent availability across `cuda_fp16.h` / `cuda_bf16.h`
//     header revisions.
//
// Fuel team's reproducer for indexing/segment BW (gather_backward,
// scatter_add for autograd, embedding_backward, segment_sum_backward)
// found that the previous half / bf16 path silently round-tripped
// through f32 — correct, but loses determinism guarantees and adds 2×
// memory traffic.
//
// **Policy.** We always use a 32-bit `atomicCAS` loop for `__half` and
// `__nv_bfloat16`, regardless of `__CUDA_ARCH__`. This:
//   * gives universal availability (works on every arch baracuda
//     supports — sm_70+),
//   * provides a stable memory-traffic profile across versions,
//   * matches the algorithmic determinism profile the rest of the
//     family advertises (atomic ordering is still non-deterministic
//     across launches, but the per-thread arithmetic is bit-stable).
//
// For `float` / `double` / `int32_t` / `int64_t` we still route to
// native `atomicAdd` (universally supported on sm_80, no CAS dance
// needed).
//
// **CAS pattern.** The 16-bit half / bf16 value lives inside a 32-bit
// aligned slot. We:
//   1. round the address down to a 4-byte boundary,
//   2. detect which 16-bit half is ours (`addr & 2`),
//   3. read-modify-CAS in a loop until the 32-bit slot updates
//      atomically.
// The other 16 bits of the slot are preserved across the CAS.
//
// Used by:
//   * baracuda_indexing.cuh — gather_backward / scatter_add /
//     index_select_backward (atomicAdd into the dgrad tensor)
//   * baracuda_embedding.cuh — embedding_backward / embedding_bag_backward
//   * baracuda_segment.cuh — segment_sum / segment_mean BW (currently
//     f32/f64 only; the helpers are here so a future half/bf16
//     extension drops in cleanly)

#ifndef BARACUDA_ATOMIC_CUH
#define BARACUDA_ATOMIC_CUH

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace atomic {

// =============================================================================
// Generic forwarder — uses native `atomicAdd` for every dtype that has
// one available on baracuda's sm_80+ baseline (f32, f64, int32_t,
// int64_t / uint32_t / uint64_t).
// =============================================================================

template <typename T>
__device__ __forceinline__ void add(T* addr, T val) {
    atomicAdd(addr, val);
}

// =============================================================================
// `__half` — always CAS, regardless of arch.
// =============================================================================

template <>
__device__ __forceinline__ void add<__half>(__half* addr, __half val) {
    // Align down to the 32-bit slot containing this half.
    uintptr_t addr_int = reinterpret_cast<uintptr_t>(addr);
    unsigned int* aligned = reinterpret_cast<unsigned int*>(addr_int & ~uintptr_t(3));
    bool is_high = (addr_int & 2) != 0;

    unsigned int old = *aligned;
    unsigned int assumed;
    do {
        assumed = old;
        // Extract our 16-bit slot from the 32-bit word.
        unsigned short half_bits = is_high
            ? static_cast<unsigned short>((assumed >> 16) & 0xFFFFu)
            : static_cast<unsigned short>(assumed & 0xFFFFu);
        __half cur = __ushort_as_half(half_bits);
        // Sum in half (matches the native atomicAdd's semantics).
        __half sum = __hadd(cur, val);
        unsigned short sum_bits = __half_as_ushort(sum);
        // Reassemble the 32-bit word with our updated half.
        unsigned int new_word = is_high
            ? ((assumed & 0x0000FFFFu)
                  | (static_cast<unsigned int>(sum_bits) << 16))
            : ((assumed & 0xFFFF0000u)
                  | static_cast<unsigned int>(sum_bits));
        old = atomicCAS(aligned, assumed, new_word);
    } while (assumed != old);
}

// =============================================================================
// `__nv_bfloat16` — always CAS, regardless of arch. Accumulation is
// done in f32 then narrowed back to bf16 (bf16's `__hadd` was added
// in cuda_bf16.h late, but f32 conversion is universally available).
// =============================================================================

template <>
__device__ __forceinline__ void add<__nv_bfloat16>(__nv_bfloat16* addr, __nv_bfloat16 val) {
    uintptr_t addr_int = reinterpret_cast<uintptr_t>(addr);
    unsigned int* aligned = reinterpret_cast<unsigned int*>(addr_int & ~uintptr_t(3));
    bool is_high = (addr_int & 2) != 0;

    float vf = __bfloat162float(val);
    unsigned int old = *aligned;
    unsigned int assumed;
    do {
        assumed = old;
        unsigned short bf_bits = is_high
            ? static_cast<unsigned short>((assumed >> 16) & 0xFFFFu)
            : static_cast<unsigned short>(assumed & 0xFFFFu);
        // Reinterpret the 16-bit pattern as a bf16, then promote to f32
        // for the add. `__ushort_as_bfloat16` exists in cuda_bf16.h on
        // CUDA 11.0+; if a future CUDA drops it, fall back to memcpy.
        __nv_bfloat16 cur;
        // Bit-cast 16-bit word into bf16. Avoid relying on
        // __ushort_as_bfloat16 (not always available) — use memcpy.
        unsigned short cur_bits_le = bf_bits;
        memcpy(&cur, &cur_bits_le, sizeof(cur));
        float sum_f = __bfloat162float(cur) + vf;
        __nv_bfloat16 sum = __float2bfloat16(sum_f);
        unsigned short sum_bits;
        memcpy(&sum_bits, &sum, sizeof(sum));
        unsigned int new_word = is_high
            ? ((assumed & 0x0000FFFFu)
                  | (static_cast<unsigned int>(sum_bits) << 16))
            : ((assumed & 0xFFFF0000u)
                  | static_cast<unsigned int>(sum_bits));
        old = atomicCAS(aligned, assumed, new_word);
    } while (assumed != old);
}

// =============================================================================
// `int64_t` — CUDA only provides `atomicAdd(unsigned long long*, ull)`.
// We reinterpret the destination pointer and add as `ull`; the
// two's-complement wraparound semantics match for signed/unsigned long
// long inside `atomicAdd`. Used by Phase 40 (Fuel 6c.4 Gap 6b spillover)
// for `index_add` on integer value-dtypes.
// =============================================================================

template <>
__device__ __forceinline__ void add<int64_t>(int64_t* addr, int64_t val) {
    atomicAdd(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(val));
}

}} // namespace baracuda::atomic

#endif // BARACUDA_ATOMIC_CUH
