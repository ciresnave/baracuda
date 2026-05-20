// baracuda_triu_tril.cuh
//
// Phase 13.4 — Triu / Tril (upper / lower triangular matrix masks).
//
// `torch.triu(input, diagonal)`: output[..., i, j] = input[..., i, j] if
//      j >= i + diagonal else 0.
// `torch.tril(input, diagonal)`: output[..., i, j] = input[..., i, j] if
//      j <= i + diagonal else 0.
//
// Both ops operate on the **last two dimensions** (`M = shape[rank-2]`,
// `N = shape[rank-1]`) of an arbitrary-rank-≥2 tensor. Anything before
// the matrix axes (the "batch" prefix) is iterated transparently — each
// batch slice is masked independently with the same `diagonal`.
//
// One templated kernel body parameterized on a Predicate functor covers
// both ops; the per-dtype symbols flow from the INSTANTIATE macros at
// the bottom of this header.
//
// Kernel design:
//   * One thread per output element. Thread `tid` writes
//     `output[tid] = predicate(i, j, diagonal) ? input[tid] : 0`.
//   * Coordinates `(i, j)` decode from `tid` via `(tid / N) % M` and
//     `tid % N`. The batch prefix doesn't influence the mask — it just
//     selects which slice we're in, and all slices share the same mask.
//   * Output buffer is contiguous, zero-offset, row-major (same shape
//     as input).
//   * Input is read at `input[tid]` (contiguous).
//
// Differentiable: `d_input = triu(d_output, diagonal)` for the triu fw,
// and `d_input = tril(d_output, diagonal)` for the tril fw. The
// backward plans on the Rust side dispatch back to these same launch
// symbols with `dy → input` and `dx → output`.
//
// Status codes mirror the rest of baracuda-kernels-sys:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch error.

#ifndef BARACUDA_TRIU_TRIL_CUH
#define BARACUDA_TRIU_TRIL_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace triu_tril {

inline constexpr int MAX_RANK = 8;

// Predicate functors. `true` means the element is KEPT (input passes
// through); `false` means the element is zeroed.
struct TriuPredicate {
    // triu keeps elements where j >= i + diagonal.
    __host__ __device__ __forceinline__
    bool operator()(int64_t i, int64_t j, int64_t diagonal) const {
        return j >= i + diagonal;
    }
};

struct TrilPredicate {
    // tril keeps elements where j <= i + diagonal.
    __host__ __device__ __forceinline__
    bool operator()(int64_t i, int64_t j, int64_t diagonal) const {
        return j <= i + diagonal;
    }
};

// "Type-correct zero" — needed because `__half` / `__nv_bfloat16` don't
// implicitly convert from `int(0)`.
template <typename T>
__host__ __device__ __forceinline__ T zero_of() { return T(0); }

template <>
__host__ __device__ __forceinline__ __half zero_of<__half>() {
    return __float2half(0.0f);
}

template <>
__host__ __device__ __forceinline__ __nv_bfloat16 zero_of<__nv_bfloat16>() {
    return __float2bfloat16(0.0f);
}

// One thread per output element. The batch prefix is implicit — `tid`
// indexes the flat row-major buffer and the mask depends only on the
// `(i, j)` coords within the last two dims (`M`, `N`).
template <typename T, typename Predicate>
__global__ void triu_tril_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t numel,
    int32_t M,
    int32_t N,
    int32_t diagonal)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    const int64_t MN = (int64_t)M * (int64_t)N;
    Predicate pred;
    for (int64_t k = tid; k < numel; k += step) {
        // Decompose the flat index into (batch_lin, i, j) — the batch
        // prefix is anything outside the last two dims, but we only need
        // (i, j) for the mask.
        int64_t within_matrix = (MN == 0) ? 0 : (k % MN);
        int64_t i = (N == 0) ? 0 : (within_matrix / (int64_t)N);
        int64_t j = (N == 0) ? 0 : (within_matrix % (int64_t)N);
        bool keep = pred(i, j, (int64_t)diagonal);
        output[k] = keep ? input[k] : zero_of<T>();
    }
}

template <typename T, typename Predicate>
__host__ inline int32_t launch_triu_tril(
    const void* input, void* output,
    const int32_t* shape_host,
    int32_t rank,
    int32_t diagonal,
    cudaStream_t stream)
{
    if (rank < 2 || rank > MAX_RANK) return 2;
    if (input == nullptr || output == nullptr) return 2;

    // Compute numel and extract M / N from the last two dims.
    int64_t numel = 1;
    for (int d = 0; d < rank; ++d) {
        if (shape_host[d] < 0) return 2;
        numel *= (int64_t)shape_host[d];
    }
    if (numel == 0) return 0;

    int32_t M = shape_host[rank - 2];
    int32_t N = shape_host[rank - 1];

    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    triu_tril_kernel<T, Predicate><<<blocks, kBlock, 0, stream>>>(
        static_cast<const T*>(input),
        static_cast<T*>(output),
        numel, M, N, diagonal);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::triu_tril

// Emit one Triu launcher (per-dtype).
//
// NAME : symbol suffix — e.g. `f32`, `f16`, `bf16`, `f64`, `i32`,
//        `i64`, `bool`.
// T    : on-device element type.
#define BARACUDA_KERNELS_TRIU_INSTANTIATE(NAME, T)                                                 \
    extern "C" int32_t baracuda_kernels_triu_##NAME##_run(                                          \
        const void* input, void* output,                                                            \
        const int32_t* shape, int32_t rank,                                                         \
        int32_t diagonal,                                                                            \
        void* stream_ptr)                                                                            \
    {                                                                                                \
        if (shape == nullptr) return 2;                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                 \
        return baracuda::triu_tril::launch_triu_tril<T, baracuda::triu_tril::TriuPredicate>(         \
            input, output, shape, rank, diagonal, stream);                                           \
    }

// Emit one Tril launcher (per-dtype).
#define BARACUDA_KERNELS_TRIL_INSTANTIATE(NAME, T)                                                 \
    extern "C" int32_t baracuda_kernels_tril_##NAME##_run(                                          \
        const void* input, void* output,                                                            \
        const int32_t* shape, int32_t rank,                                                         \
        int32_t diagonal,                                                                            \
        void* stream_ptr)                                                                            \
    {                                                                                                \
        if (shape == nullptr) return 2;                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                 \
        return baracuda::triu_tril::launch_triu_tril<T, baracuda::triu_tril::TrilPredicate>(         \
            input, output, shape, rank, diagonal, stream);                                           \
    }

#endif // BARACUDA_TRIU_TRIL_CUH
