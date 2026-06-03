// baracuda_unique.cuh
//
// `unique` / `unique_consecutive` FW (no BW — set-valued op).
//
// `unique_consecutive` is the trailblazer primitive — it walks the
// already-sorted (or assumed-sorted by the caller) input row and emits
// one cell per "run start" (position 0 OR a position where the value
// differs from its predecessor). One global atomic counter assigns
// output slots; the launcher zeroes it on the stream before launch.
//
// `unique` is the same kernel with the contract that the caller
// pre-sorted the row (calls SortPlan first). Since the per-row sort
// then a per-row consecutive-dedup are two separate launches at the
// Rust plan layer, the kernel here only implements the consecutive
// variant. The Plan's `unique` op chains sort + this kernel.
//
// Output ordering caveat (matches the nonzero family): atomic-counter
// assigns slots in CUDA-block race order, NOT in input order. Callers
// that need input-order output should run unique_consecutive
// single-threaded per row (deferred — set `unique_consecutive_blocked
// (...)` below if needed). The trailblazer accepts the unordered output.
//
// Status codes mirror baracuda_sort.cuh.
//
// Trailblazer dtype coverage: f32, f64, i32.

#ifndef BARACUDA_UNIQUE_CUH
#define BARACUDA_UNIQUE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda { namespace unique_ns {

template <typename T>
__device__ inline bool eq_val(T a, T b) { return a == b; }

template <typename T>
__global__ void unique_consecutive_kernel(
    const T*  __restrict__ x,           // [batch, row_len], sorted per row
    T*        __restrict__ y_vals,      // [batch, max_unique] flat
    int32_t*  __restrict__ y_counts,    // [batch, max_unique] flat (1 per
                                        //   detected run start; further
                                        //   accumulation is the caller's
                                        //   job since slot order is racy)
    int32_t*  __restrict__ counter,     // [batch], one atomic counter
                                        //   per row
    int32_t   row_len,
    int32_t   max_unique)
{
    int row = blockIdx.x;
    const T* x_row = x + (int64_t)row * (int64_t)row_len;
    int64_t out_off = (int64_t)row * (int64_t)max_unique;

    for (int col = threadIdx.x; col < row_len; col += blockDim.x) {
        bool is_run_start;
        if (col == 0) {
            is_run_start = true;
        } else {
            is_run_start = !eq_val<T>(x_row[col], x_row[col - 1]);
        }
        if (is_run_start) {
            int32_t slot = atomicAdd(&counter[row], 1);
            if (slot < max_unique) {
                y_vals  [out_off + (int64_t)slot] = x_row[col];
                if (y_counts) y_counts[out_off + (int64_t)slot] = 1;
            }
        }
    }
}

template <typename T>
__host__ inline int32_t launch_unique_consecutive(
    const T* x, T* y_vals, int32_t* y_counts, int32_t* counter,
    int32_t batch, int32_t row_len, int32_t max_unique,
    cudaStream_t stream)
{
    if (batch < 0 || row_len < 0 || max_unique < 0) return 2;
    if (batch == 0) return 0;
    if (x == nullptr || y_vals == nullptr || counter == nullptr) return 2;
    // Zero the per-row counters.
    cudaError_t err = cudaMemsetAsync(counter, 0,
                                      (size_t)batch * sizeof(int32_t), stream);
    if (err != cudaSuccess) return 5;
    if (row_len == 0 || max_unique == 0) return 0;
    int threads = 256;
    if (threads > row_len) threads = row_len;
    if (threads < 32) threads = 32;
    unique_consecutive_kernel<T><<<batch, threads, 0, stream>>>(
        x, y_vals, y_counts, counter, row_len, max_unique);
    err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::unique_ns

#define BARACUDA_KERNELS_UNIQUE_CONSECUTIVE_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t batch, int32_t row_len, int32_t max_unique,                                       \
        const void* x, void* y_vals, void* y_counts, void* counter,                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::unique_ns::launch_unique_consecutive<T>(                                 \
            static_cast<const T*>(x),                                                              \
            static_cast<T*>(y_vals),                                                               \
            static_cast<int32_t*>(y_counts),                                                       \
            static_cast<int32_t*>(counter),                                                        \
            batch, row_len, max_unique, stream);                                                  \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t batch, int32_t row_len, int32_t max_unique,                                       \
        const void* /*x*/, const void* /*y_vals*/,                                                \
        const void* /*y_counts*/, const void* /*counter*/)                                        \
    {                                                                                              \
        if (batch < 0 || row_len < 0 || max_unique < 0) return 2;                                 \
        return 0;                                                                                 \
    }

#endif // BARACUDA_UNIQUE_CUH
