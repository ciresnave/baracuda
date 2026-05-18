// baracuda_embedding.cuh
//
// Templated kernels and INSTANTIATE macros for the embedding op family
// (Phase 7 Milestone 7.5 — Category M from the comprehensive plan).
//
// Ops shipped here:
//   embedding             — `out[n, :] = weight[indices[n], :]` with an
//                            optional padding_idx that zeros rows where
//                            `indices[n] == padding_idx`.
//   embedding_backward    — `dweight[indices[n], :] += dout[n, :]`
//                            (atomicAdd), skipping rows where
//                            `indices[n] == padding_idx`.
//   embedding_bag         — `out[b, :] = reduce(weight[indices[k], :]
//                            for k in offsets[b]..offsets[b+1])` with
//                            mode ∈ {Sum, Mean}. Empty bags
//                            (`start == end`) emit a zero row.
//   embedding_bag_backward — for each (b, k) and feature d:
//                            atomicAdd(dweight[indices[k], d],
//                                      dout[b, d] / divisor) where
//                            divisor = 1 (Sum) or bag_size (Mean).
//
// Index dtype is `i32` only (i64 deferred). `padding_idx` is `int32_t`
// with the sentinel `-1` meaning "disabled" (matches PyTorch). Negative
// or out-of-range `indices` entries are silently skipped (no PyTorch-
// style negative-wrap).
//
// Status codes mirror the indexing family:
//   0 success
//   1 misaligned operand (reserved)
//   2 invalid problem
//   3 unsupported (reserved)
//   4 workspace too small (reserved — these ops are workspace-free)
//   5 internal kernel error (typically a launch failure)

#ifndef BARACUDA_EMBEDDING_CUH
#define BARACUDA_EMBEDDING_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_indexing.cuh"  // for scatter_atomic_add<T>

namespace baracuda { namespace embedding {

// `kPaddingDisabled` is the sentinel returned to the kernel when the
// caller did not pass a `padding_idx`. Chosen as `INT32_MIN` so that no
// legitimate index in `[0, V)` collides (V is i32-bounded).
inline constexpr int32_t kPaddingDisabled = (int32_t)(-2147483647 - 1);

// =============================================================================
// embedding forward — one thread per (n, d) output cell.
// =============================================================================

// Phase 11.5 / Fuel team feedback #7: templated on `IndexT` (i32 or i64).
// `padding_idx` is sized i64 so the same parameter slot covers both
// index dtypes (the kPaddingDisabled sentinel is INT32_MIN and stays
// distinct under int64 promotion).
template <typename T, typename IndexT>
__global__ void embedding_kernel(
    const T* __restrict__ weight,
    const IndexT* __restrict__ indices,
    T* __restrict__ out,
    int64_t out_numel,        // == N * D
    int32_t num_embeddings,   // V
    int32_t embedding_dim,    // D
    int64_t padding_idx)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    T zero;
    // `T{}` is value-initialized; for arithmetic types this is zero.
    // Use memset on the byte buffer because __half / __nv_bfloat16 don't
    // have a `T(0)` constructor that's portable across nvcc versions.
    {
        unsigned char* p = reinterpret_cast<unsigned char*>(&zero);
        for (size_t i = 0; i < sizeof(T); ++i) p[i] = 0;
    }
    for (int64_t i = tid; i < out_numel; i += step) {
        int64_t n = i / (int64_t)embedding_dim;
        int64_t d = i - n * (int64_t)embedding_dim;
        int64_t idx = (int64_t)indices[n];
        if (idx == padding_idx || idx < 0 || idx >= (int64_t)num_embeddings) {
            out[i] = zero;
        } else {
            out[i] = weight[idx * (int64_t)embedding_dim + d];
        }
    }
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_embedding(
    const T* weight, const IndexT* indices, T* out,
    int64_t num_indices,
    int32_t num_embeddings,
    int32_t embedding_dim,
    int64_t padding_idx,
    cudaStream_t stream)
{
    if (num_indices < 0 || num_embeddings < 0 || embedding_dim < 0) return 2;
    int64_t out_numel = num_indices * (int64_t)embedding_dim;
    if (out_numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (out_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    embedding_kernel<T, IndexT><<<blocks, kBlock, 0, stream>>>(
        weight, indices, out, out_numel,
        num_embeddings, embedding_dim, padding_idx);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// embedding backward — scatter-add dout into dweight along row dim,
// skipping the padding_idx row. One thread per (n, d) cell; atomicAdd
// into dweight.
// =============================================================================

template <typename T, typename IndexT>
__global__ void embedding_backward_kernel(
    const T* __restrict__ dout,
    const IndexT* __restrict__ indices,
    T* __restrict__ dweight,
    int64_t out_numel,
    int32_t num_embeddings,
    int32_t embedding_dim,
    int64_t padding_idx)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < out_numel; i += step) {
        int64_t n = i / (int64_t)embedding_dim;
        int64_t d = i - n * (int64_t)embedding_dim;
        int64_t idx = (int64_t)indices[n];
        if (idx == padding_idx || idx < 0 || idx >= (int64_t)num_embeddings) {
            continue;
        }
        int64_t off = idx * (int64_t)embedding_dim + d;
        baracuda::indexing::scatter_atomic_add<T>(&dweight[off], dout[i]);
    }
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_embedding_backward(
    const T* dout, const IndexT* indices, T* dweight,
    int64_t num_indices,
    int32_t num_embeddings,
    int32_t embedding_dim,
    int64_t padding_idx,
    cudaStream_t stream)
{
    if (num_indices < 0 || num_embeddings < 0 || embedding_dim < 0) return 2;
    int64_t out_numel = num_indices * (int64_t)embedding_dim;
    if (out_numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (out_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    embedding_backward_kernel<T, IndexT><<<blocks, kBlock, 0, stream>>>(
        dout, indices, dweight, out_numel,
        num_embeddings, embedding_dim, padding_idx);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Mode tag for embedding_bag — kept in-header so launchers can take a
// scalar int and route to the same template body.
// =============================================================================

inline constexpr int32_t kModeSum  = 0;
inline constexpr int32_t kModeMean = 1;

// =============================================================================
// f32 accumulator selection: f32, f16, bf16 → float; f64 → double.
// =============================================================================

template <typename T> struct AccumOf { using type = float; };
template <> struct AccumOf<double> { using type = double; };

template <typename T> __device__ inline typename AccumOf<T>::type to_accum(T v) {
    return (typename AccumOf<T>::type)v;
}
template <> __device__ inline float to_accum<__half>(__half v) { return __half2float(v); }
template <> __device__ inline float to_accum<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T, typename Acc> __device__ inline T from_accum(Acc v) {
    return (T)v;
}
template <> __device__ inline __half from_accum<__half, float>(float v) {
    return __float2half(v);
}
template <> __device__ inline __nv_bfloat16 from_accum<__nv_bfloat16, float>(float v) {
    return __float2bfloat16(v);
}

// =============================================================================
// embedding_bag forward — one thread per (b, d) output cell. Walks the
// bag's index range and accumulates `weight[indices[k], d]`. If mode is
// Mean, divides by `(end - start)` (after skipping any padding_idx rows).
// Empty bags emit zero.
// =============================================================================

// Phase 11.5: templated on `IndexT` (i32 or i64) for `indices`. `offsets`
// remains i32 — bag boundaries fit comfortably in int32 because the
// total-indices count itself is i32-bounded.
template <typename T, typename IndexT>
__global__ void embedding_bag_kernel(
    const T* __restrict__ weight,
    const IndexT* __restrict__ indices,
    const int32_t* __restrict__ offsets,
    T* __restrict__ out,
    int64_t out_numel,         // == B * D
    int32_t total_indices,
    int32_t num_embeddings,
    int32_t embedding_dim,
    int32_t num_bags,
    int32_t mode,
    int64_t padding_idx)
{
    using Acc = typename AccumOf<T>::type;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < out_numel; i += step) {
        int64_t b = i / (int64_t)embedding_dim;
        int32_t d = (int32_t)(i - b * (int64_t)embedding_dim);
        int32_t start = offsets[b];
        int32_t end = (b + 1 < num_bags) ? offsets[b + 1] : total_indices;
        Acc acc = (Acc)0;
        int32_t counted = 0;
        for (int32_t k = start; k < end; ++k) {
            int64_t idx = (int64_t)indices[k];
            if (idx == padding_idx || idx < 0 || idx >= (int64_t)num_embeddings) {
                continue;
            }
            acc += to_accum<T>(weight[idx * (int64_t)embedding_dim + (int64_t)d]);
            counted++;
        }
        if (mode == kModeMean && counted > 0) {
            acc = acc / (Acc)counted;
        }
        out[i] = from_accum<T, Acc>(acc);
    }
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_embedding_bag(
    const T* weight, const IndexT* indices, const int32_t* offsets, T* out,
    int32_t total_indices,
    int32_t num_embeddings,
    int32_t embedding_dim,
    int32_t num_bags,
    int32_t mode,
    int64_t padding_idx,
    cudaStream_t stream)
{
    if (total_indices < 0 || num_embeddings < 0 || embedding_dim < 0 || num_bags < 0) return 2;
    if (mode != kModeSum && mode != kModeMean) return 3;
    int64_t out_numel = (int64_t)num_bags * (int64_t)embedding_dim;
    if (out_numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (out_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    embedding_bag_kernel<T, IndexT><<<blocks, kBlock, 0, stream>>>(
        weight, indices, offsets, out, out_numel,
        total_indices, num_embeddings, embedding_dim, num_bags, mode, padding_idx);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// embedding_bag backward — one thread per (b, k) pair. Each thread
// looks at its bag's index range and, if (k - start) < (end - start),
// fans out to all D features and atomicAdds dout[b, d] / divisor into
// dweight[indices[k], d].
//
// We launch (num_bags * D) threads laid out as one thread per (b, d)
// output cell. Each cell walks its bag's [start, end) range, reads the
// shared divisor once per (b, d), and atomicAdds the per-row entries.
// =============================================================================

template <typename T, typename IndexT>
__global__ void embedding_bag_backward_kernel(
    const T* __restrict__ dout,
    const IndexT* __restrict__ indices,
    const int32_t* __restrict__ offsets,
    T* __restrict__ dweight,
    int64_t out_numel,         // == B * D
    int32_t total_indices,
    int32_t num_embeddings,
    int32_t embedding_dim,
    int32_t num_bags,
    int32_t mode,
    int64_t padding_idx)
{
    using Acc = typename AccumOf<T>::type;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < out_numel; i += step) {
        int64_t b = i / (int64_t)embedding_dim;
        int32_t d = (int32_t)(i - b * (int64_t)embedding_dim);
        int32_t start = offsets[b];
        int32_t end = (b + 1 < num_bags) ? offsets[b + 1] : total_indices;
        if (end <= start) continue;
        // Compute bag_size (counted = non-padded, non-OOB indices).
        int32_t counted = 0;
        if (mode == kModeMean) {
            for (int32_t k = start; k < end; ++k) {
                int64_t idx = (int64_t)indices[k];
                if (idx == padding_idx || idx < 0 || idx >= (int64_t)num_embeddings) continue;
                counted++;
            }
            if (counted == 0) continue;
        }
        Acc up = to_accum<T>(dout[i]);
        if (mode == kModeMean) {
            up = up / (Acc)counted;
        }
        T contrib = from_accum<T, Acc>(up);
        for (int32_t k = start; k < end; ++k) {
            int64_t idx = (int64_t)indices[k];
            if (idx == padding_idx || idx < 0 || idx >= (int64_t)num_embeddings) continue;
            int64_t off = idx * (int64_t)embedding_dim + (int64_t)d;
            baracuda::indexing::scatter_atomic_add<T>(&dweight[off], contrib);
        }
    }
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_embedding_bag_backward(
    const T* dout, const IndexT* indices, const int32_t* offsets, T* dweight,
    int32_t total_indices,
    int32_t num_embeddings,
    int32_t embedding_dim,
    int32_t num_bags,
    int32_t mode,
    int64_t padding_idx,
    cudaStream_t stream)
{
    if (total_indices < 0 || num_embeddings < 0 || embedding_dim < 0 || num_bags < 0) return 2;
    if (mode != kModeSum && mode != kModeMean) return 3;
    int64_t out_numel = (int64_t)num_bags * (int64_t)embedding_dim;
    if (out_numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (out_numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    embedding_bag_backward_kernel<T, IndexT><<<blocks, kBlock, 0, stream>>>(
        dout, indices, offsets, dweight, out_numel,
        total_indices, num_embeddings, embedding_dim, num_bags, mode, padding_idx);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::embedding

// =============================================================================
// INSTANTIATE macros.
// =============================================================================

// Phase 11.5: `INDEX_T` parameter selects the index dtype (`int32_t`
// or `int64_t`). `padding_idx` is sized `int64_t` in the FFI so the
// same parameter slot covers both index dtypes — i32 callers cast
// their `padding_idx` (or `kPaddingDisabled` sentinel) on the way in.
#define BARACUDA_KERNELS_EMBEDDING_INSTANTIATE(NAME, T, INDEX_T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t num_indices,                                                                       \
        int32_t num_embeddings,                                                                    \
        int32_t embedding_dim,                                                                     \
        int64_t padding_idx,                                                                       \
        const void* weight,                                                                        \
        const void* indices,                                                                       \
        void* out,                                                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (num_indices < 0) return 2;                                                             \
        if (num_indices == 0) return 0;                                                            \
        if (weight == nullptr || indices == nullptr || out == nullptr) return 2;                  \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::embedding::launch_embedding<T, INDEX_T>(                                  \
            static_cast<const T*>(weight),                                                         \
            static_cast<const INDEX_T*>(indices),                                                  \
            static_cast<T*>(out),                                                                  \
            num_indices, num_embeddings, embedding_dim, padding_idx, stream);                      \
    }

#define BARACUDA_KERNELS_EMBEDDING_BACKWARD_INSTANTIATE(NAME, T, INDEX_T)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t num_indices,                                                                       \
        int32_t num_embeddings,                                                                    \
        int32_t embedding_dim,                                                                     \
        int64_t padding_idx,                                                                       \
        const void* dout,                                                                          \
        const void* indices,                                                                       \
        void* dweight,                                                                             \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (num_indices < 0) return 2;                                                             \
        if (num_indices == 0) return 0;                                                            \
        if (dout == nullptr || indices == nullptr || dweight == nullptr) return 2;                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::embedding::launch_embedding_backward<T, INDEX_T>(                         \
            static_cast<const T*>(dout),                                                           \
            static_cast<const INDEX_T*>(indices),                                                  \
            static_cast<T*>(dweight),                                                              \
            num_indices, num_embeddings, embedding_dim, padding_idx, stream);                      \
    }

#define BARACUDA_KERNELS_EMBEDDING_BAG_INSTANTIATE(NAME, T, INDEX_T)                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t total_indices,                                                                     \
        int32_t num_embeddings,                                                                    \
        int32_t embedding_dim,                                                                     \
        int32_t num_bags,                                                                          \
        int32_t mode,                                                                              \
        int64_t padding_idx,                                                                       \
        const void* weight,                                                                        \
        const void* indices,                                                                       \
        const void* offsets,                                                                       \
        void* out,                                                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (num_bags < 0) return 2;                                                                \
        if (num_bags == 0) return 0;                                                               \
        if (weight == nullptr || indices == nullptr || offsets == nullptr || out == nullptr)      \
            return 2;                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::embedding::launch_embedding_bag<T, INDEX_T>(                              \
            static_cast<const T*>(weight),                                                         \
            static_cast<const INDEX_T*>(indices),                                                  \
            static_cast<const int32_t*>(offsets),                                                  \
            static_cast<T*>(out),                                                                  \
            total_indices, num_embeddings, embedding_dim, num_bags, mode, padding_idx, stream);   \
    }

#define BARACUDA_KERNELS_EMBEDDING_BAG_BACKWARD_INSTANTIATE(NAME, T, INDEX_T)                      \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t total_indices,                                                                     \
        int32_t num_embeddings,                                                                    \
        int32_t embedding_dim,                                                                     \
        int32_t num_bags,                                                                          \
        int32_t mode,                                                                              \
        int64_t padding_idx,                                                                       \
        const void* dout,                                                                          \
        const void* indices,                                                                       \
        const void* offsets,                                                                       \
        void* dweight,                                                                             \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (num_bags < 0) return 2;                                                                \
        if (num_bags == 0) return 0;                                                               \
        if (dout == nullptr || indices == nullptr || offsets == nullptr || dweight == nullptr)    \
            return 2;                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::embedding::launch_embedding_bag_backward<T, INDEX_T>(                     \
            static_cast<const T*>(dout),                                                           \
            static_cast<const INDEX_T*>(indices),                                                  \
            static_cast<const int32_t*>(offsets),                                                  \
            static_cast<T*>(dweight),                                                              \
            total_indices, num_embeddings, embedding_dim, num_bags, mode, padding_idx, stream);   \
    }

#endif // BARACUDA_EMBEDDING_CUH
