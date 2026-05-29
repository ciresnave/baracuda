// baracuda_cp_async.cuh
//
// Phase 44b — templated `cp.async` (Ampere+) helpers, with an automatic
// fallback to a normal memcpy on older arches so the same source can
// build for sm_70. Baracuda's existing Flash SDPA sm_89 sibling
// (Phase 10) uses raw `cp.async` PTX inline; future kernels that want
// the same async-copy + commit-group + wait-group pattern can pull in
// this header instead of re-rolling the asm.
//
// **Provenance.** Derived from Hiroyuki Ootomo's `cutf/cp_async.hpp`
// (https://gitlab.momo86.net/mutsuki/cutf — now offline). Same MIT
// license as the rest of `cutf`; folded into baracuda during the
// Phase 44b cutf-submodule retirement. Re-styled to match the
// `baracuda_*.cuh` convention and given baracuda doc-comments.
//
// **API.**
//
//   - `baracuda::cp_async::cp_async<N>(smem, gmem)` — async-copy `N`
//     bytes from global memory to shared memory. `N` must be 4, 8, or
//     16. On sm_80+ this maps to `cp.async.ca.shared.global`; on
//     older arches it degrades to a plain pointer-store.
//   - `baracuda::cp_async::commit()` — boundary marker. Groups all
//     pending `cp_async` calls since the last `commit()` into one
//     "group" that can be waited on individually.
//   - `baracuda::cp_async::wait_group<N>()` — wait until all but the
//     last `N` groups have completed. `N == 0` waits for everything.
//   - `baracuda::cp_async::wait_all()` — synonym for `wait_group<0>()`
//     plus an implicit final group commit. Convenient for kernels
//     that only have one async-copy phase.
//
// The header has no runtime cost when included by a TU that doesn't
// use the helpers — every entry point is `inline __device__`. The
// fallback path on sm_70 is single-element stores; for kernels that
// can't tolerate the loss of latency-hiding on pre-sm_80 hardware,
// that's the cue to write an explicit `#if __CUDA_ARCH__ >= 800`
// fast-path rather than rely on this header to bridge the gap.

#pragma once

#include <cstdint>

namespace baracuda {
namespace cp_async {
namespace detail {

/// Convert a generic shared-memory pointer to the 32-bit shared
/// address that PTX `cp.async` accepts. Uses the documented
/// `cvta.to.shared.u64` → `cvt.u32.u64` pair. NVCC generates the
/// same code as the inline asm, but emitting it here keeps the
/// templated entry points clean.
__device__ inline uint32_t get_smem_ptr_uint(const void *const ptr) {
    uint32_t smem_ptr;
    asm volatile("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
                 : "=r"(smem_ptr) : "l"(ptr));
    return smem_ptr;
}

}  // namespace detail

#if defined(BARACUDA_CP_ASYNC_DISABLE) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)
// fall-through to scalar fallback
#else
#define BARACUDA_CP_ASYNC_INTERNAL_USE_CP_ASYNC
#endif

/// Async-copy `Size` bytes from `gmem` (a generic-memory pointer
/// pointing into global memory) into `smem` (a generic-memory
/// pointer pointing into shared memory). Returns immediately on
/// sm_80+; the copy completes some time later, observable via
/// `commit()` + `wait_group<N>()`.
///
/// `Size` must be 4, 8, or 16 bytes. Other values fail at compile
/// time. The asm encoding `cp.async.ca.shared.global` is the
/// cacheable-all variant — appropriate for most matrix-tile loads.
/// Kernels that want the cacheable-global-only variant
/// (`cp.async.cg`) should write inline asm directly.
template <unsigned Size>
__device__ inline void cp_async(void *const smem, const void *const gmem) {
    static_assert(Size == 4 || Size == 8 || Size == 16,
                  "baracuda::cp_async::cp_async<Size>: Size must be 4, 8, or 16");
#ifdef BARACUDA_CP_ASYNC_INTERNAL_USE_CP_ASYNC
    const unsigned smem_int_ptr = detail::get_smem_ptr_uint(smem);
    asm volatile("{cp.async.ca.shared.global [%0], [%1], %2;}"
                 :: "r"(smem_int_ptr), "l"(gmem), "n"(Size));
#else
    if (Size == 4) {
        *(reinterpret_cast<uint32_t *>(smem)) = *(reinterpret_cast<const uint32_t *>(gmem));
    } else if (Size == 8) {
        *(reinterpret_cast<uint64_t *>(smem)) = *(reinterpret_cast<const uint64_t *>(gmem));
    } else {
        *(reinterpret_cast<ulong2 *>(smem)) = *(reinterpret_cast<const ulong2 *>(gmem));
    }
#endif
}

/// Commit all pending `cp_async` calls since the last commit into one
/// "group" — that group can be individually waited on via
/// `wait_group<N>()`. No-op on pre-sm_80 (the fallback `cp_async`
/// path is already synchronous).
__device__ inline void commit() {
#ifdef BARACUDA_CP_ASYNC_INTERNAL_USE_CP_ASYNC
    asm volatile("{cp.async.commit_group;}\n");
#endif
}

/// Wait until every committed `cp_async` group has retired. Synonym
/// for `wait_group<0>()`. No-op on pre-sm_80.
__device__ inline void wait_all() {
#ifdef BARACUDA_CP_ASYNC_INTERNAL_USE_CP_ASYNC
    asm volatile("{cp.async.wait_all;}");
#endif
}

/// Wait until all but the last `N` committed groups have retired.
/// Lets the kernel start consuming the oldest tile of a multi-tile
/// double-buffered pipeline while later tiles are still in flight.
template <int N>
__device__ inline void wait_group() {
#ifdef BARACUDA_CP_ASYNC_INTERNAL_USE_CP_ASYNC
    asm volatile("{cp.async.wait_group %0;}" :: "n"(N));
#endif
}

}  // namespace cp_async
}  // namespace baracuda
