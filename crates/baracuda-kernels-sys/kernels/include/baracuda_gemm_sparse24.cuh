// baracuda_gemm_sparse24.cuh
//
// Phase 54 — 2:4 Structured Sparsity GEMM (clean-room hand-port of
// facebookresearch/xformers `sparse24/` algorithmic reference; see
// `vendor/xformers/VENDOR.md` for attribution).
//
// 2:4 pattern: in every 4 consecutive weight cells, AT MOST 2 are
// non-zero. The compressed format is:
//   W_compressed: [M, K/2]   (the up-to-2 non-zero values per 4-group,
//                             packed contiguously per row)
//   W_metadata:   [M, K/8]   (uint16 — 2 bits per 4-group identifying
//                             which 2 of the 4 positions are non-zero,
//                             stored 8 groups per uint16 = 16 bits)
// The dense `W` reconstruction is:
//   for m in 0..M:
//     for k_group in 0..K/4:
//       meta = metadata bits for this group (2 nybbles, 4 bits = pos0,pos1)
//       w_dense[m, k_group*4 + meta.pos0] = W_compressed[m, k_group*2 + 0]
//       w_dense[m, k_group*4 + meta.pos1] = W_compressed[m, k_group*2 + 1]
//       (other 2 positions in the 4-group are 0)
//
// Output: Y[N, M] = X[N, K] @ W_dense^T[K, M] ===> Y = X @ W^T
//
// (Following PyTorch/xFormers convention: weight is `[M, K]`, the GEMM
// is `Y = X @ W^T` where X is `[N, K]` and Y is `[N, M]`. M = out_features,
// K = in_features, N = batch * seq.)
//
// Tier-1 implementation strategy:
//   We **inflate-then-multiply**: kernel 1 reconstructs the dense W in
//   a caller-owned workspace buffer, then we route the actual matmul
//   through the standard dense path. This is correctness-first; the
//   sparse-tensor-core (`mma.sp.sync.aligned`) hardware speedup is
//   deferred to Tier 2 (which would use `cusparseLt` or a bespoke
//   inline-PTX `mma.sp` kernel).
//
// Rationale: the sparse-tensor-core path involves complex metadata
// formatting (cuSPARSELt's pruning + reorder + permute is non-trivial
// and tightly coupled to cuSPARSELt's opaque descriptor handles), and
// `mma.sp.sync.aligned` requires register-level metadata layout that
// is highly arch-specific. Phase 54's value is the **API surface +
// compression format**; the hardware-accelerated backend lands in a
// follow-up alongside cuSPARSELt integration.
//
// The inflated dense W is in the caller-supplied workspace (size
// `M * K * sizeof(T)`); the standard GEMM kernel (not in this file —
// the Rust plan composes it via `cublasGemmEx` or `GemmPlan`) consumes
// the inflated weight directly. This file ships ONLY the inflation
// kernel and a thin self-contained reference GEMM for smoke tests.
//
// Layout contract:
//   W_compressed: [M, K/2]   row-major contiguous
//   W_metadata:   [M, K/8]   row-major contiguous (uint16)
//   X:            [N, K]     row-major contiguous
//   Y:            [N, M]     row-major contiguous
//
// Constraints:
//   K must be a multiple of 8 (a uint16 metadata word covers 8 4-groups
//   = 32 K-positions; we don't support partial trailing metadata words
//   in the trailblazer).
//
// Status codes:
//   0 success, 2 invalid problem, 3 unsupported,
//   1000+e launch failure.

#ifndef BARACUDA_GEMM_SPARSE24_CUH
#define BARACUDA_GEMM_SPARSE24_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace gemm_sparse24 {

// dtype conversion helpers — match the flash family's compute-type
// convention.
template <typename T>
__device__ __forceinline__ float load_as_f32(T x) { return (float)x; }
template <>
__device__ __forceinline__ float load_as_f32<__half>(__half x) {
    return __half2float(x);
}
template <>
__device__ __forceinline__ float load_as_f32<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T store_from_f32(float v) { return (T)v; }
template <>
__device__ __forceinline__ __half store_from_f32<__half>(float v) {
    return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 store_from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// =============================================================================
// Inflation kernel: reconstruct dense W [M, K] from compressed format.
// Each thread handles a 4-group of K (one (m, k_group) pair).
// =============================================================================

template <typename T>
__global__ void sparse24_inflate_kernel(
    const T*        __restrict__ w_compressed,   // [M, K/2]
    const uint16_t* __restrict__ w_metadata,     // [M, K/8]
    T*              __restrict__ w_dense,        // [M, K]
    int32_t M, int32_t K)
{
    // Each thread handles one (m, k_group_idx) where k_group_idx ranges
    // over [0, K/4). The compressed row stride is K/2; metadata row
    // stride is K/8.
    const int K_groups = K / 4;
    const int total = M * K_groups;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;
    for (int idx = tid; idx < total; idx += step) {
        const int m = idx / K_groups;
        const int kg = idx - m * K_groups;
        // Metadata: 2 bits per 4-group; 8 4-groups per uint16 (16 bits).
        // Metadata stride is K/8 uint16 per row.
        const int meta_word_idx = kg / 8;
        const int meta_word_off = kg & 7;     // 0..7
        // Per 4-group: 4 bits encode the 2 non-zero positions
        // (pos0: 2 bits, pos1: 2 bits). 8 groups × 4 bits = 32 bits.
        // We store 2 groups per uint16 (8 bits each — wasting half the
        // bit budget for clarity in the trailblazer; xFormers' real
        // format is denser at 2 bits-per-pos × 2 pos = 4 bits-per-group
        // packed 4 groups per uint16. We use 8-bits-per-group for
        // simplicity).
        // **Trailblazer convention**: 1 byte per 4-group, low 2 bits =
        // pos0, bits [2:3] = pos1. So uint16 carries 2 groups
        // (low byte = group g, high byte = group g+1).
        //
        // meta_word_idx selects pairs-of-groups; meta_word_off ∈ [0, 1]
        // picks low/high byte.
        //
        // Re-derive: K/8 uint16s × 2 bytes-each × 1 group-per-byte =
        // K/4 groups. ✓
        const int meta_pair_idx = kg / 2;
        const int meta_pair_off = kg & 1;
        const uint16_t mword = w_metadata[(int64_t)m * (K / 8) + meta_pair_idx];
        const uint8_t mbyte = (meta_pair_off == 0)
            ? (uint8_t)(mword & 0xFF)
            : (uint8_t)((mword >> 8) & 0xFF);
        const int pos0 = (int)(mbyte & 0x3);
        const int pos1 = (int)((mbyte >> 2) & 0x3);
        // Read the 2 non-zero values for this group.
        const int compressed_base = m * (K / 2) + kg * 2;
        const T v0 = w_compressed[compressed_base + 0];
        const T v1 = w_compressed[compressed_base + 1];
        const int dense_base = m * K + kg * 4;
        // Zero the 4-cell group then place v0, v1.
        w_dense[dense_base + 0] = store_from_f32<T>(0.0f);
        w_dense[dense_base + 1] = store_from_f32<T>(0.0f);
        w_dense[dense_base + 2] = store_from_f32<T>(0.0f);
        w_dense[dense_base + 3] = store_from_f32<T>(0.0f);
        // Caller is responsible for ensuring pos0 != pos1 and both are
        // in [0, 3]; an invalid metadata byte would overwrite the same
        // cell twice. The kernel doesn't validate (perf path).
        w_dense[dense_base + pos0] = v0;
        w_dense[dense_base + pos1] = v1;
    }
}

// =============================================================================
// Reference dense GEMM: Y[N, M] = X[N, K] @ W^T[K, M]
// (so W is consumed as [M, K] row-major; we compute Y[n, m] = Σ_k X[n, k] * W[m, k])
//
// This is a naive triple-loop kernel that bypasses tensor cores — it's
// here as a self-contained smoke-test reference. Production callers
// should use baracuda's `GemmPlan` / `cublasGemmEx` directly on the
// inflated dense W.
// =============================================================================

template <typename T>
__global__ void sparse24_reference_gemm_kernel(
    const T* __restrict__ x,        // [N, K]
    const T* __restrict__ w_dense,  // [M, K]
    T*       __restrict__ y,        // [N, M]
    int32_t N, int32_t M, int32_t K)
{
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || m >= M) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        float xv = load_as_f32<T>(x[(int64_t)n * K + k]);
        float wv = load_as_f32<T>(w_dense[(int64_t)m * K + k]);
        acc += xv * wv;
    }
    y[(int64_t)n * M + m] = store_from_f32<T>(acc);
}

// =============================================================================
// Launcher: inflate + reference matmul. The bespoke launcher is for
// smoke-test validation; production callers will use a separate
// inflate-only launcher and route the matmul through `cublasGemmEx`.
// =============================================================================

template <typename T>
__host__ inline int32_t launch_sparse24_inflate(
    const T* w_compressed, const uint16_t* w_metadata,
    T* w_dense,
    int32_t M, int32_t K,
    cudaStream_t stream)
{
    if (M < 0 || K < 0) return 2;
    if ((K & 7) != 0) return 3;
    if (M == 0 || K == 0) return 0;
    const int K_groups = K / 4;
    const int total = M * K_groups;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    sparse24_inflate_kernel<T><<<grid, block, 0, stream>>>(
        w_compressed, w_metadata, w_dense, M, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;
    return 0;
}

template <typename T>
__host__ inline int32_t launch_sparse24_gemm_reference(
    const T* x, const T* w_compressed, const uint16_t* w_metadata,
    T* y,
    void* workspace, uint64_t workspace_bytes,
    int32_t N, int32_t M, int32_t K,
    cudaStream_t stream)
{
    if (N < 0 || M < 0 || K < 0) return 2;
    if ((K & 7) != 0) return 3;
    if (N == 0 || M == 0 || K == 0) return 0;
    // Need workspace ≥ M * K * sizeof(T) for the inflated dense W.
    const uint64_t needed = (uint64_t)M * (uint64_t)K * sizeof(T);
    if (workspace == nullptr || workspace_bytes < needed) return 4;
    T* w_dense = reinterpret_cast<T*>(workspace);
    int32_t rc = launch_sparse24_inflate<T>(
        w_compressed, w_metadata, w_dense, M, K, stream);
    if (rc != 0) return rc;
    dim3 block(16, 16);
    dim3 grid((unsigned)((M + 15) / 16), (unsigned)((N + 15) / 16));
    sparse24_reference_gemm_kernel<T><<<grid, block, 0, stream>>>(
        x, w_dense, y, N, M, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return 1000 + (int32_t)err;
    return 0;
}

} } // namespace baracuda::gemm_sparse24

// =============================================================================
// INSTANTIATE macro — emits the C-ABI symbols for a given (name, T) pair.
// =============================================================================

#define BARACUDA_KERNELS_GEMM_SPARSE24_INSTANTIATE(NAME, T)                      \
    extern "C" int32_t baracuda_kernels_##NAME##_sparse24_inflate(               \
        int32_t M, int32_t K,                                                     \
        const void* w_compressed, const void* w_metadata,                         \
        void* w_dense,                                                            \
        void* stream)                                                             \
    {                                                                             \
        return ::baracuda::gemm_sparse24::launch_sparse24_inflate<T>(            \
            reinterpret_cast<const T*>(w_compressed),                             \
            reinterpret_cast<const uint16_t*>(w_metadata),                        \
            reinterpret_cast<T*>(w_dense),                                        \
            M, K,                                                                 \
            reinterpret_cast<cudaStream_t>(stream));                              \
    }                                                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_sparse24_gemm_run(              \
        int32_t N, int32_t M, int32_t K,                                          \
        const void* x, const void* w_compressed, const void* w_metadata,          \
        void* y,                                                                  \
        void* workspace, uint64_t workspace_bytes,                                \
        void* stream)                                                             \
    {                                                                             \
        return ::baracuda::gemm_sparse24::launch_sparse24_gemm_reference<T>(     \
            reinterpret_cast<const T*>(x),                                        \
            reinterpret_cast<const T*>(w_compressed),                             \
            reinterpret_cast<const uint16_t*>(w_metadata),                        \
            reinterpret_cast<T*>(y),                                              \
            workspace, workspace_bytes,                                           \
            N, M, K,                                                              \
            reinterpret_cast<cudaStream_t>(stream));                              \
    }                                                                             \
    extern "C" int32_t baracuda_kernels_##NAME##_sparse24_gemm_can_implement(    \
        int32_t N, int32_t M, int32_t K)                                          \
    {                                                                             \
        if (N < 0 || M < 0 || K < 0) return 2;                                    \
        if ((K & 7) != 0) return 3;                                               \
        return 0;                                                                 \
    }                                                                             \
    extern "C" uint64_t baracuda_kernels_##NAME##_sparse24_gemm_workspace_bytes( \
        int32_t /*N*/, int32_t M, int32_t K)                                      \
    {                                                                             \
        if (M < 0 || K < 0) return 0;                                             \
        return (uint64_t)M * (uint64_t)K * sizeof(T);                             \
    }

#endif // BARACUDA_GEMM_SPARSE24_CUH
