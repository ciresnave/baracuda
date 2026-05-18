// baracuda_searchsorted.cuh
//
// `searchsorted` — binary search of `values[i]` in a sorted 1-D
// `sorted_seq` array. PyTorch `torch.searchsorted` semantics:
//   * `right == false` (default): return lower_bound — the leftmost
//     position `j` such that `sorted_seq[j] >= values[i]`.
//   * `right == true`: return upper_bound — the leftmost position `j`
//     such that `sorted_seq[j] > values[i]`.
// Returned positions are in `[0, len_sorted]` (i.e. can equal len when
// all elements are less than the query).
//
// One thread per query value. Result is i32.
//
// Trailblazer dtype coverage: f32, f64, i32, i64.
//
// Trailblazer scope: 1-D `sorted_seq` shared across ALL query rows. The
// batched-per-row version (PyTorch's batched-searchsorted) is a
// follow-up; today the Rust plan layer requires `sorted_seq.shape ==
// [len_sorted]`.

#ifndef BARACUDA_SEARCHSORTED_CUH
#define BARACUDA_SEARCHSORTED_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda { namespace searchsorted {

template <typename T>
__device__ inline int32_t lower_bound(const T* arr, int32_t n, T target) {
    int32_t lo = 0, hi = n;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else                   hi = mid;
    }
    return lo;
}

template <typename T>
__device__ inline int32_t upper_bound(const T* arr, int32_t n, T target) {
    int32_t lo = 0, hi = n;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] <= target) lo = mid + 1;
        else                    hi = mid;
    }
    return lo;
}

template <typename T>
__global__ void searchsorted_kernel(
    const T* __restrict__ sorted_seq,   // [len_sorted]
    const T* __restrict__ values,       // [num_queries]
    int32_t* __restrict__ output,       // [num_queries]
    int64_t  num_queries,
    int32_t  len_sorted,
    int32_t  right)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < num_queries; i += step) {
        T q = values[i];
        int32_t pos = (right != 0) ? upper_bound<T>(sorted_seq, len_sorted, q)
                                   : lower_bound<T>(sorted_seq, len_sorted, q);
        output[i] = pos;
    }
}

template <typename T>
__host__ inline int32_t launch_searchsorted(
    const T* sorted_seq, const T* values, int32_t* output,
    int64_t num_queries, int32_t len_sorted, int32_t right,
    cudaStream_t stream)
{
    if (num_queries < 0 || len_sorted < 0) return 2;
    if (num_queries == 0) return 0;
    if (values == nullptr || output == nullptr) return 2;
    if (len_sorted > 0 && sorted_seq == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (num_queries + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    searchsorted_kernel<T><<<blocks, kBlock, 0, stream>>>(
        sorted_seq, values, output, num_queries, len_sorted, right);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::searchsorted

#define BARACUDA_KERNELS_SEARCHSORTED_INSTANTIATE(NAME, T)                                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t num_queries, int32_t len_sorted, int32_t right,                                   \
        const void* sorted_seq, const void* values, void* output,                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::searchsorted::launch_searchsorted<T>(                                    \
            static_cast<const T*>(sorted_seq), static_cast<const T*>(values),                     \
            static_cast<int32_t*>(output),                                                         \
            num_queries, len_sorted, right, stream);                                              \
    }

#endif // BARACUDA_SEARCHSORTED_CUH
