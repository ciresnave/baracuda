// baracuda_smem_row_stager.cuh — cooperative load / store helpers for the
// "one block per row" + "all threads cooperatively stage a row in SMEM"
// kernel pattern. Phase 65 — used to retrofit baracuda's normalization
// kernels (RMSNorm, LayerNorm, Softmax, LogSoftmax, BatchNorm,
// GroupNorm) for **in-place safety** (`y_ptr == x_ptr` aliasing) AND
// for the perf benefit of one global read + one global write per cell
// instead of the previous 2-3 reads from global.
//
// The helpers are deliberately simple and dynamic-length (any `n` at
// runtime). For tiled patterns with compile-time `ITEMS_PER_THREAD`,
// CUB's `cub::BlockLoad` / `cub::BlockStore` are faster — these
// helpers fill the gap where the per-row length is a runtime
// parameter (e.g. `d_model` varies per call).
//
// Usage pattern:
//
//     extern __shared__ float smem_row[];  // dynamic SMEM allocation
//     int64_t b = blockIdx.x;
//     const T* x_row = x + b * stride_b;
//     T*       y_row = y + b * stride_b;
//
//     // Phase 1: cooperative load (includes sync).
//     baracuda::smem_stage_row(smem_row, x_row, n);
//
//     // Phase 2: kernel-specific multi-pass arithmetic over smem_row.
//     // ...compute mean, variance, normalize, etc...
//
//     // Phase 3: cooperative store (no trailing sync; caller adds if needed).
//     baracuda::smem_unstage_row(y_row, smem_row, n);
//
// With this pattern, the kernel reads `x_row` ONCE from global into
// SMEM and writes `y_row` ONCE to global. If `x_row == y_row` (in-place
// dispatch), the staging insulates the multi-pass compute from the
// final write — same-pointer aliasing is structurally safe.

#ifndef BARACUDA_SMEM_ROW_STAGER_CUH
#define BARACUDA_SMEM_ROW_STAGER_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda {

// Cooperative load: `blockDim.x` threads cooperatively load `n` elements
// from `global` into `smem` using a grid-stride pattern. Synchronizes
// before returning so all threads see the staged data.
//
// `smem` must be sized to at least `n * sizeof(SmemT)` bytes. If `SmemT`
// differs from `GlobalT` (e.g. f32 SMEM for f16 / bf16 storage to
// promote to compute precision in one pass), the load includes the
// implicit conversion.
template <typename SmemT, typename GlobalT>
__device__ __forceinline__ void smem_stage_row(
    SmemT* __restrict__ smem,
    const GlobalT* __restrict__ global,
    int64_t n)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < n;
         i += static_cast<int64_t>(blockDim.x))
    {
        smem[i] = static_cast<SmemT>(global[i]);
    }
    __syncthreads();
}

// Strided variant — for non-contiguous source rows (each axis-step in
// the source is `stride` elements apart in global memory). Includes
// the sync.
template <typename SmemT, typename GlobalT>
__device__ __forceinline__ void smem_stage_row_strided(
    SmemT* __restrict__ smem,
    const GlobalT* __restrict__ global,
    int64_t n,
    int64_t stride)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < n;
         i += static_cast<int64_t>(blockDim.x))
    {
        smem[i] = static_cast<SmemT>(global[i * stride]);
    }
    __syncthreads();
}

// Cooperative store: writes `n` elements from `smem` back to `global`.
// No leading or trailing sync (caller adds before if SMEM was just
// written by another phase; adds after only if a subsequent phase
// needs to read the global write).
template <typename SmemT, typename GlobalT>
__device__ __forceinline__ void smem_unstage_row(
    GlobalT* __restrict__ global,
    const SmemT* __restrict__ smem,
    int64_t n)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < n;
         i += static_cast<int64_t>(blockDim.x))
    {
        global[i] = static_cast<GlobalT>(smem[i]);
    }
}

// Strided variant — write to a non-contiguous destination row.
template <typename SmemT, typename GlobalT>
__device__ __forceinline__ void smem_unstage_row_strided(
    GlobalT* __restrict__ global,
    const SmemT* __restrict__ smem,
    int64_t n,
    int64_t stride)
{
    for (int64_t i = static_cast<int64_t>(threadIdx.x);
         i < n;
         i += static_cast<int64_t>(blockDim.x))
    {
        global[i * stride] = static_cast<GlobalT>(smem[i]);
    }
}

// Per-architecture SMEM budget (bytes) for the per-block dynamic SMEM
// allocation. These are the OPT-IN limits (caller is responsible for
// calling `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
// to actually unlock the opt-in limit on devices that support it).
// Returns a conservative value with ~16 KB headroom for kernel-internal
// scratch (warp-reduce SMEM buffers, register spills, etc.).
__host__ __device__ inline std::size_t smem_budget_for_arch(int compute_capability) {
    // Reserve 16 KB headroom for cross-warp reduction buffers + spills.
    if (compute_capability >= 90) return (228u - 16u) * 1024u;  // sm_90 / Hopper
    if (compute_capability >= 89) return ( 99u - 16u) * 1024u;  // sm_89 / Ada
    if (compute_capability >= 80) return (164u - 16u) * 1024u;  // sm_80 / Ampere
    return ( 48u - 16u) * 1024u;                                 // sm_75 / Turing default
}

// Helper: given an element type size in bytes + per-block scratch
// requirement, return the maximum `n` that can be SMEM-staged on a
// given compute capability. Used by callers to decide whether to
// dispatch to the SMEM-staged kernel or fall back to the legacy
// global-read kernel.
//
// `bytes_per_element` is the SMEM staging element size — typically
// `sizeof(float)` when promoting to f32 compute precision regardless
// of storage dtype, OR `sizeof(T)` when SMEM stages in the operand
// dtype directly.
//
// `extra_smem_bytes` is the kernel's additional fixed SMEM use (e.g.
// 32 floats for a cross-warp reduction buffer = 128 bytes). Pass 0
// if the kernel uses only the row stage.
__host__ __device__ inline int64_t smem_stage_max_n(
    int bytes_per_element,
    int extra_smem_bytes,
    int compute_capability)
{
    std::size_t budget = smem_budget_for_arch(compute_capability);
    if (budget <= static_cast<std::size_t>(extra_smem_bytes)) return 0;
    std::size_t available = budget - static_cast<std::size_t>(extra_smem_bytes);
    return static_cast<int64_t>(available / static_cast<std::size_t>(bytes_per_element));
}

}  // namespace baracuda

#endif  // BARACUDA_SMEM_ROW_STAGER_CUH
