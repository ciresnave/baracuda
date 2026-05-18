// baracuda_topk.cuh
//
// Block-level top-k kernel for the Phase 9 Category O sorting family.
//
// Trailblazer algorithm — partial bitonic top-k:
//   * Re-uses the block-bitonic sort from baracuda_sort.cuh (one block
//     per row, indices sorted in shared memory).
//   * After a full sort, the first k cells of the sorted order are
//     the top-k. We could short-circuit (only sort the top-k bitonic
//     layers) at large `row_len`, but at the trailblazer scope
//     (`row_len <= 1024`, `k <= 64`) the simpler "full sort + take
//     first k" pattern is the same asymptotic cost as a careful
//     partial bitonic and a lot easier to get right.
//
// `descending == 1` is the default sort order for top-k FW (largest
// values first); pass `descending == 0` for bottom-k / "smallest first".
//
// kthvalue is computed by calling topk with k=`k` and returning the
// (k-1)-th cell (descending: takes the k-th largest; ascending: takes
// the k-th smallest). The Rust plan layer composes; this header
// emits only the topk launcher.
//
// Trailblazer dtype coverage: f32, f64. Trailblazer limits:
//   * row_len <= 1024 (one block per row, bitonic).
//   * k <= 64 (the common LLM-inference range).
//   * k must be <= row_len (validated by the launcher).
//
// Status codes mirror baracuda_sort.cuh.

#ifndef BARACUDA_TOPK_CUH
#define BARACUDA_TOPK_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "baracuda_sort.cuh"

namespace baracuda { namespace topk {

inline constexpr int MAX_ROW = baracuda::sort::MAX_ROW;
inline constexpr int MAX_K   = 64;

template <typename T, int ORDER>
__global__ void topk_block_kernel(
    const T*  __restrict__ x,
    T*        __restrict__ y_vals,    // [batch, k]
    int32_t*  __restrict__ y_idx,     // [batch, k]
    int32_t   row_len,
    int32_t   row_len_pad,
    int32_t   k)
{
    int row = blockIdx.x;
    const T* x_row = x + (int64_t)row * (int64_t)row_len;

    extern __shared__ int32_t s_idx[];

    for (int col = threadIdx.x; col < row_len_pad; col += blockDim.x) {
        s_idx[col] = col;
    }
    __syncthreads();

    // Bitonic sort the full row (same ladder as baracuda::sort).
    for (int32_t kk = 2; kk <= row_len_pad; kk <<= 1) {
        for (int32_t j = kk >> 1; j > 0; j >>= 1) {
            for (int col = threadIdx.x; col < row_len_pad; col += blockDim.x) {
                int ixj = col ^ j;
                if (ixj > col) {
                    bool ascending_block = ((col & kk) == 0);
                    int32_t a_idx = s_idx[col];
                    int32_t b_idx = s_idx[ixj];
                    bool a_is_pad = (a_idx >= row_len);
                    bool b_is_pad = (b_idx >= row_len);

                    bool swap;
                    if (a_is_pad && b_is_pad) {
                        swap = false;
                    } else if (a_is_pad) {
                        swap = ascending_block;
                    } else if (b_is_pad) {
                        swap = !ascending_block;
                    } else {
                        T a = x_row[a_idx];
                        T b = x_row[b_idx];
                        swap = baracuda::sort::cmp_swap_needed<T, ORDER, 0>(
                            a, b, a_idx, b_idx, ascending_block);
                    }
                    if (swap) {
                        s_idx[col] = b_idx;
                        s_idx[ixj] = a_idx;
                    }
                }
            }
            __syncthreads();
        }
    }

    int64_t row_off = (int64_t)row * (int64_t)k;
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        int32_t src = s_idx[col];
        y_idx [row_off + col] = src;
        y_vals[row_off + col] = x_row[src];
    }
}

template <typename T, int ORDER>
__host__ inline int32_t launch_topk_block(
    const T* x, T* y_vals, int32_t* y_idx,
    int32_t batch, int32_t row_len, int32_t k,
    cudaStream_t stream)
{
    if (batch < 0 || row_len < 0 || k < 0) return 2;
    if (row_len > MAX_ROW) return 3;
    if (k > MAX_K) return 3;
    if (k > row_len) return 2;
    if (batch == 0 || k == 0) return 0;
    if (x == nullptr || y_vals == nullptr || y_idx == nullptr) return 2;
    int32_t row_pad = baracuda::sort::next_pow2_i32(row_len);
    int threads = row_pad;
    if (threads > 1024) threads = 1024;
    if (threads < 32) threads = 32;
    size_t smem = (size_t)row_pad * sizeof(int32_t);
    topk_block_kernel<T, ORDER><<<batch, threads, smem, stream>>>(
        x, y_vals, y_idx, row_len, row_pad, k);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// topk BW — scatter dy back to the original positions via the saved
// indices. Reuses baracuda::sort::sort_backward_kernel logic.
template <typename T>
__global__ void topk_backward_kernel(
    const T*       __restrict__ dy,         // [batch, k]
    const int32_t* __restrict__ indices,    // [batch, k]
    T*             __restrict__ dx,         // [batch, row_len] — zero-init
    int32_t        batch,
    int32_t        k,
    int32_t        row_len)
{
    int64_t total = (int64_t)batch * (int64_t)k;
    int64_t tid   = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step  = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t row = (int32_t)(i / (int64_t)k);
        int32_t src = indices[i];
        if (src >= 0 && src < row_len) {
            dx[(int64_t)row * (int64_t)row_len + (int64_t)src] = dy[i];
        }
    }
}

template <typename T>
__host__ inline int32_t launch_topk_backward(
    const T* dy, const int32_t* indices, T* dx,
    int32_t batch, int32_t k, int32_t row_len,
    cudaStream_t stream)
{
    if (batch < 0 || k < 0 || row_len < 0) return 2;
    int64_t dx_total = (int64_t)batch * (int64_t)row_len;
    if (dx_total == 0) return 0;
    if (dx == nullptr) return 2;
    cudaError_t merr = cudaMemsetAsync(dx, 0, (size_t)dx_total * sizeof(T), stream);
    if (merr != cudaSuccess) return 5;
    int64_t total = (int64_t)batch * (int64_t)k;
    if (total == 0) return 0;
    if (dy == nullptr || indices == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    topk_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, indices, dx, batch, k, row_len);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::topk

#define BARACUDA_KERNELS_TOPK_INSTANTIATE(NAME, T)                                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t batch, int32_t row_len, int32_t k, int32_t largest,                               \
        const void* x, void* y_vals, void* y_idx,                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        if (largest != 0) {                                                                       \
            return baracuda::topk::launch_topk_block<T, 0>(                                       \
                static_cast<const T*>(x), static_cast<T*>(y_vals),                                \
                static_cast<int32_t*>(y_idx), batch, row_len, k, stream);                         \
        } else {                                                                                  \
            return baracuda::topk::launch_topk_block<T, 1>(                                       \
                static_cast<const T*>(x), static_cast<T*>(y_vals),                                \
                static_cast<int32_t*>(y_idx), batch, row_len, k, stream);                         \
        }                                                                                          \
    }

#define BARACUDA_KERNELS_TOPK_BACKWARD_INSTANTIATE(NAME, T)                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t batch, int32_t k, int32_t row_len,                                                \
        const void* dy, const void* indices, void* dx,                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::topk::launch_topk_backward<T>(                                           \
            static_cast<const T*>(dy),                                                            \
            static_cast<const int32_t*>(indices),                                                 \
            static_cast<T*>(dx),                                                                  \
            batch, k, row_len, stream);                                                           \
    }

#endif // BARACUDA_TOPK_CUH
