// baracuda_block_atomic.cuh
//
// Phase 67c — unified cross-block atomic-merge API across every numeric
// dtype baracuda's reduction / scatter kernels touch.
//
// **Why this header exists.** Many baracuda kernels accumulate a partial
// result per block and then atomically merge it into a single shared
// output buffer (cross-block reduction). The merge primitives —
// atomic add / max / min / mul — are duplicated across the segment,
// embedding-bag, index-add, scatter and batch-norm BW kernels, each
// with its own hand-rolled `atomicCAS` loop for the dtypes CUDA lacks a
// native intrinsic for. This header is the single home for all four
// operations, generic over `T`.
//
// **Relationship to `baracuda_atomic.cuh`.** The `add` operation was
// already factored out into `baracuda_atomic.cuh` in Phase 11.3 (Fuel
// team feedback #6), living in this same `baracuda::atomic` namespace.
// Rather than duplicate it (which would be an ODR clash if both headers
// land in one translation unit) we `#include` it here and re-export it,
// then add `max` / `min` / `mul` alongside. Include this header to get
// the whole family; include just `baracuda_atomic.cuh` if you only need
// `add`.
//
// **Native vs CAS coverage.**
//   * `add` — see `baracuda_atomic.cuh` (native for f32/f64/i32/i64/
//     u32/u64; 32-bit `atomicCAS` loop for `__half` / `__nv_bfloat16`).
//   * `max` / `min` — native `atomicMax` / `atomicMin` for the integer
//     types (int / unsigned / long long / unsigned long long, all
//     available on baracuda's sm_80+ baseline). f32 / f64 go through an
//     `atomicCAS` loop on the bit pattern; `__half` / `__nv_bfloat16`
//     through a 16-bit-in-32 `atomicCAS` loop with f32 compute.
//   * `mul` — CUDA has no native atomic multiply for any dtype, so every
//     specialization is a CAS loop. Rare (only `unsorted_segment_prod`
//     uses it today) and non-deterministic across launches, but the
//     per-thread arithmetic is bit-stable.
//
// **CAS pattern for half / bf16.** The 16-bit value lives inside a
// 32-bit aligned slot. We (1) round the address down to a 4-byte
// boundary, (2) detect which 16-bit half is ours (`addr & 2`),
// (3) read-modify-CAS the 32-bit slot in a loop, preserving the other
// 16 bits. Compute happens in f32 then narrows back — matching the
// determinism profile the rest of the family advertises.
//
// **Early-out.** Each CAS loop computes the would-be new 32-bit word and
// returns immediately if it equals the observed word (e.g. `max` with a
// value already <= the slot, or `mul` by 1). This avoids a redundant
// atomic write under contention.
//
// Lifted / generalized from:
//   * baracuda_atomic.cuh        — `add` (re-exported via #include)
//   * baracuda_segment.cuh       — `atomic_{max,min,mul}_f{32,64}`
//                                  (these become migration candidates)
//
// Migration of existing kernels onto this header is a SEPARATE phase
// (1-kernel-per-commit audit) — see the commit message for the list.

#ifndef BARACUDA_BLOCK_ATOMIC_CUH
#define BARACUDA_BLOCK_ATOMIC_CUH

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Re-export the unified `add` family (f32/f64/i32/i64/u32/u64/half/bf16).
#include "baracuda_atomic.cuh"

namespace baracuda { namespace atomic {

// =============================================================================
// Internal CAS-body macros. COMBINE_F is an expression in `cur_f` (the
// current slot value, promoted to its compute type) and `val_f` (the
// incoming value) producing the new value. Every macro carries the
// no-change early-out described in the header docstring.
// =============================================================================

// f32: CAS on the int bit pattern.
#define BARACUDA_ATOMIC_F32_(OP, COMBINE)                                     \
    template <>                                                               \
    __device__ __forceinline__ void OP<float>(float* addr, float val) {       \
        int* iaddr = reinterpret_cast<int*>(addr);                            \
        int old = __float_as_int(*addr);                                      \
        int assumed;                                                          \
        const float val_f = val;                                              \
        do {                                                                  \
            assumed = old;                                                    \
            const float cur_f = __int_as_float(assumed);                      \
            const int newbits = __float_as_int(COMBINE);                      \
            if (newbits == assumed) return;                                   \
            old = atomicCAS(iaddr, assumed, newbits);                         \
        } while (assumed != old);                                             \
    }

// f64: CAS on the unsigned long long bit pattern.
#define BARACUDA_ATOMIC_F64_(OP, COMBINE)                                     \
    template <>                                                               \
    __device__ __forceinline__ void OP<double>(double* addr, double val) {    \
        unsigned long long* uaddr =                                           \
            reinterpret_cast<unsigned long long*>(addr);                      \
        unsigned long long old = __double_as_longlong(*addr);                 \
        unsigned long long assumed;                                           \
        const double val_f = val;                                             \
        do {                                                                  \
            assumed = old;                                                    \
            const double cur_f = __longlong_as_double(assumed);               \
            const unsigned long long newbits = __double_as_longlong(COMBINE); \
            if (newbits == assumed) return;                                   \
            old = atomicCAS(uaddr, assumed, newbits);                         \
        } while (assumed != old);                                             \
    }

// __half: 16-bit-in-32 CAS, compute promoted to f32. `__ushort_as_half`
// is available on the same CUDA versions baracuda_atomic.cuh relies on.
#define BARACUDA_ATOMIC_HALF_(OP, COMBINE)                                    \
    template <>                                                               \
    __device__ __forceinline__ void OP<__half>(__half* addr, __half val) {    \
        const uintptr_t addr_int = reinterpret_cast<uintptr_t>(addr);         \
        unsigned int* aligned =                                               \
            reinterpret_cast<unsigned int*>(addr_int & ~uintptr_t(3));        \
        const bool is_high = (addr_int & 2) != 0;                             \
        const float val_f = __half2float(val);                                \
        unsigned int old = *aligned;                                          \
        unsigned int assumed;                                                 \
        do {                                                                  \
            assumed = old;                                                    \
            const unsigned short cur_bits = is_high                           \
                ? static_cast<unsigned short>((assumed >> 16) & 0xFFFFu)       \
                : static_cast<unsigned short>(assumed & 0xFFFFu);             \
            const float cur_f = __half2float(__ushort_as_half(cur_bits));     \
            const unsigned short res_bits =                                   \
                __half_as_ushort(__float2half(COMBINE));                      \
            const unsigned int new_word = is_high                             \
                ? ((assumed & 0x0000FFFFu)                                    \
                       | (static_cast<unsigned int>(res_bits) << 16))         \
                : ((assumed & 0xFFFF0000u)                                    \
                       | static_cast<unsigned int>(res_bits));                \
            if (new_word == assumed) return;                                  \
            old = atomicCAS(aligned, assumed, new_word);                      \
        } while (assumed != old);                                             \
    }

// __nv_bfloat16: 16-bit-in-32 CAS, compute promoted to f32. Bit-casts
// via memcpy to avoid relying on `__ushort_as_bfloat16` (mirrors the
// existing bf16 `add` in baracuda_atomic.cuh).
#define BARACUDA_ATOMIC_BF16_(OP, COMBINE)                                    \
    template <>                                                               \
    __device__ __forceinline__ void OP<__nv_bfloat16>(                        \
            __nv_bfloat16* addr, __nv_bfloat16 val) {                         \
        const uintptr_t addr_int = reinterpret_cast<uintptr_t>(addr);         \
        unsigned int* aligned =                                               \
            reinterpret_cast<unsigned int*>(addr_int & ~uintptr_t(3));        \
        const bool is_high = (addr_int & 2) != 0;                             \
        const float val_f = __bfloat162float(val);                            \
        unsigned int old = *aligned;                                          \
        unsigned int assumed;                                                 \
        do {                                                                  \
            assumed = old;                                                    \
            const unsigned short cur_bits = is_high                           \
                ? static_cast<unsigned short>((assumed >> 16) & 0xFFFFu)       \
                : static_cast<unsigned short>(assumed & 0xFFFFu);             \
            __nv_bfloat16 cur;                                                 \
            memcpy(&cur, &cur_bits, sizeof(cur));                             \
            const float cur_f = __bfloat162float(cur);                        \
            const __nv_bfloat16 res = __float2bfloat16(COMBINE);              \
            unsigned short res_bits;                                          \
            memcpy(&res_bits, &res, sizeof(res));                             \
            const unsigned int new_word = is_high                             \
                ? ((assumed & 0x0000FFFFu)                                    \
                       | (static_cast<unsigned int>(res_bits) << 16))         \
                : ((assumed & 0xFFFF0000u)                                    \
                       | static_cast<unsigned int>(res_bits));                \
            if (new_word == assumed) return;                                  \
            old = atomicCAS(aligned, assumed, new_word);                      \
        } while (assumed != old);                                             \
    }

// Integer CAS multiply (no native atomic multiply exists). CAST is the
// unsigned word type CUDA's `atomicCAS` accepts for this width.
#define BARACUDA_ATOMIC_INT_MUL_(T, CAST)                                     \
    template <>                                                               \
    __device__ __forceinline__ void mul<T>(T* addr, T val) {                  \
        CAST* caddr = reinterpret_cast<CAST*>(addr);                          \
        CAST old = *caddr;                                                    \
        CAST assumed;                                                         \
        do {                                                                  \
            assumed = old;                                                    \
            const T cur = static_cast<T>(assumed);                            \
            const CAST newbits =                                              \
                static_cast<CAST>(static_cast<T>(cur * val));                 \
            if (newbits == assumed) return;                                   \
            old = atomicCAS(caddr, assumed, newbits);                         \
        } while (assumed != old);                                             \
    }

// =============================================================================
// max — native for integers, CAS for floating point.
// =============================================================================

template <typename T>
__device__ __forceinline__ void max(T* addr, T val) {
    atomicMax(addr, val);  // int / unsigned / (unsigned) long long on sm_80+
}

BARACUDA_ATOMIC_F32_(max, fmaxf(cur_f, val_f))
BARACUDA_ATOMIC_F64_(max, fmax(cur_f, val_f))
BARACUDA_ATOMIC_HALF_(max, fmaxf(cur_f, val_f))
BARACUDA_ATOMIC_BF16_(max, fmaxf(cur_f, val_f))

// =============================================================================
// min — native for integers, CAS for floating point.
// =============================================================================

template <typename T>
__device__ __forceinline__ void min(T* addr, T val) {
    atomicMin(addr, val);  // int / unsigned / (unsigned) long long on sm_80+
}

BARACUDA_ATOMIC_F32_(min, fminf(cur_f, val_f))
BARACUDA_ATOMIC_F64_(min, fmin(cur_f, val_f))
BARACUDA_ATOMIC_HALF_(min, fminf(cur_f, val_f))
BARACUDA_ATOMIC_BF16_(min, fminf(cur_f, val_f))

// =============================================================================
// mul — always CAS (no native atomic multiply for any dtype). The
// primary template is intentionally undefined: a link error flags any
// dtype without a specialization below.
// =============================================================================

template <typename T>
__device__ __forceinline__ void mul(T* addr, T val);

BARACUDA_ATOMIC_F32_(mul, cur_f * val_f)
BARACUDA_ATOMIC_F64_(mul, cur_f * val_f)
BARACUDA_ATOMIC_HALF_(mul, cur_f * val_f)
BARACUDA_ATOMIC_BF16_(mul, cur_f * val_f)

BARACUDA_ATOMIC_INT_MUL_(int, int)
BARACUDA_ATOMIC_INT_MUL_(unsigned int, unsigned int)
BARACUDA_ATOMIC_INT_MUL_(long long, unsigned long long)
BARACUDA_ATOMIC_INT_MUL_(unsigned long long, unsigned long long)

#undef BARACUDA_ATOMIC_F32_
#undef BARACUDA_ATOMIC_F64_
#undef BARACUDA_ATOMIC_HALF_
#undef BARACUDA_ATOMIC_BF16_
#undef BARACUDA_ATOMIC_INT_MUL_

}} // namespace baracuda::atomic

#endif // BARACUDA_BLOCK_ATOMIC_CUH
