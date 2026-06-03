// baracuda_histogram.cuh
//
// `histogram` / `histogramdd` / `bincount` — Phase 9 Category O.
//
// `bincount(x[N], minlength)` — x is integer; output[i] = count of
//   occurrences of `i` in x, for i in [0, max(max(x)+1, minlength)).
//   Trailblazer: caller pre-computes `num_bins = max(max(x)+1, minlength)`
//   and passes it; this kernel just sweeps x and atomicAdd's into
//   output[x[i]] for x[i] in range.
//
// `histogram(x[N], num_bins, lo, hi)` — float x; bin index is
//   `floor((x[i] - lo) / (hi - lo) * num_bins)` clipped to
//   [0, num_bins). Inputs outside [lo, hi] are SKIPPED (matches PyTorch
//   semantics for `weights=None`).
//
// `histogramdd` — N-D histogram; we ship 1-D only in trailblazer and
//   leave the multi-axis version as a follow-up. The Rust plan layer
//   raises `Unsupported` for ndim > 1.
//
// All three accumulate INT32 counts into `output[num_bins]`. They take
// FP input (f32 / f64) for histogram, INT input (i32 / i64) for
// bincount.
//
// Status codes mirror baracuda_sort.cuh.

#ifndef BARACUDA_HISTOGRAM_CUH
#define BARACUDA_HISTOGRAM_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda { namespace histogram {

// ---------- bincount ----------

template <typename TIdx>
__global__ void bincount_kernel(
    const TIdx* __restrict__ x,           // [N]
    int32_t*    __restrict__ output,      // [num_bins]
    int64_t     n,
    int32_t     num_bins)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < n; i += step) {
        TIdx v = x[i];
        if (v >= 0 && (int64_t)v < (int64_t)num_bins) {
            atomicAdd(&output[(int32_t)v], 1);
        }
    }
}

template <typename TIdx>
__host__ inline int32_t launch_bincount(
    const TIdx* x, int32_t* output,
    int64_t n, int32_t num_bins,
    cudaStream_t stream)
{
    if (n < 0 || num_bins < 0) return 2;
    if (num_bins > 0) {
        if (output == nullptr) return 2;
        cudaError_t err = cudaMemsetAsync(output, 0,
                                          (size_t)num_bins * sizeof(int32_t), stream);
        if (err != cudaSuccess) return 5;
    }
    if (n == 0) return 0;
    if (x == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (n + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    bincount_kernel<TIdx><<<blocks, kBlock, 0, stream>>>(
        x, output, n, num_bins);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// ---------- histogram (1-D, uniform bins) ----------

template <typename T>
__global__ void histogram_kernel(
    const T*  __restrict__ x,
    int32_t*  __restrict__ output,
    int64_t   n,
    int32_t   num_bins,
    T         lo,
    T         hi)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    T scale = (T)((double)num_bins / ((double)hi - (double)lo));
    for (int64_t i = tid; i < n; i += step) {
        T v = x[i];
        if (v < lo || v > hi) continue;
        // Bin index = floor((v - lo) * scale). Cells exactly == hi go in
        // the last bin.
        int32_t b = (int32_t)((double)(v - lo) * (double)scale);
        if (b >= num_bins) b = num_bins - 1;
        if (b < 0) b = 0;
        atomicAdd(&output[b], 1);
    }
}

template <typename T>
__host__ inline int32_t launch_histogram(
    const T* x, int32_t* output,
    int64_t n, int32_t num_bins, T lo, T hi,
    cudaStream_t stream)
{
    if (n < 0 || num_bins < 0) return 2;
    if (num_bins > 0) {
        if (output == nullptr) return 2;
        cudaError_t err = cudaMemsetAsync(output, 0,
                                          (size_t)num_bins * sizeof(int32_t), stream);
        if (err != cudaSuccess) return 5;
    }
    if (n == 0) return 0;
    if (!(hi > lo)) return 2;
    if (x == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (n + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    histogram_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, output, n, num_bins, lo, hi);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::histogram

#define BARACUDA_KERNELS_BINCOUNT_INSTANTIATE(NAME, TIDX)                                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t n, int32_t num_bins,                                                              \
        const void* x, void* output,                                                              \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::histogram::launch_bincount<TIDX>(                                        \
            static_cast<const TIDX*>(x), static_cast<int32_t*>(output),                           \
            n, num_bins, stream);                                                                 \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int64_t n, int32_t num_bins,                                                              \
        const void* /*x*/, const void* /*output*/)                                                \
    {                                                                                              \
        if (n < 0 || num_bins < 0) return 2;                                                      \
        return 0;                                                                                 \
    }

#define BARACUDA_KERNELS_HISTOGRAM_INSTANTIATE(NAME, T)                                           \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t n, int32_t num_bins,                                                              \
        double lo_d, double hi_d,                                                                 \
        const void* x, void* output,                                                              \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        T lo = (T)lo_d;                                                                            \
        T hi = (T)hi_d;                                                                            \
        return baracuda::histogram::launch_histogram<T>(                                          \
            static_cast<const T*>(x), static_cast<int32_t*>(output),                              \
            n, num_bins, lo, hi, stream);                                                         \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int64_t n, int32_t num_bins,                                                              \
        double /*lo_d*/, double /*hi_d*/,                                                         \
        const void* /*x*/, const void* /*output*/)                                                \
    {                                                                                              \
        if (n < 0 || num_bins < 0) return 2;                                                      \
        return 0;                                                                                 \
    }

#endif // BARACUDA_HISTOGRAM_CUH
