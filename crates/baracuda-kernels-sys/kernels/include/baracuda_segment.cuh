// baracuda_segment.cuh
//
// Templated kernels and INSTANTIATE macros for the segment / scatter-
// reduce op family (Phase 7 Milestone 7.6 — Category S from the
// comprehensive plan).
//
// Algorithm choices:
//
//  Sorted family — segment_ids[i] is monotonically non-decreasing in i.
//    One thread per output cell (segment_id, d). Each thread binary-
//    searches the segment_ids array for the half-open range
//    [start, end) covering its segment_id, then sweeps input[i, d] for
//    i in [start, end) accumulating the reduction. Sum / Mean / Max /
//    Min / Prod share one template (kernel functor parameter).
//
//  Unsorted family — segment_ids[i] in any order in [0, num_segments).
//    One thread per (n, d) input cell. Thread reads seg = segment_ids[n]
//    and emits an atomic into output[seg, d] (atomicAdd for sum,
//    atomicMax / atomicMin for max / min). Mean is implemented as
//    {sum kernel + count kernel + post-divide kernel}.
//
// Conventions:
//
//   input         [N, D]  fp32 or fp64
//   segment_ids   [N]     i32, values in [0, num_segments)
//   output        [num_segments, D]  fp32 or fp64
//
// Out-of-range segment IDs (< 0 or >= num_segments) are SKIPPED — the
// kernel writes nothing into output for that input row. PyTorch /
// TF / JAX all treat this as undefined behavior; we silently drop.
//
// Status codes returned by the launchers mirror the indexing family:
//   0 success
//   1 misaligned operand
//   2 invalid problem
//   3 unsupported
//   4 workspace too small
//   5 internal kernel error

#ifndef BARACUDA_SEGMENT_CUH
#define BARACUDA_SEGMENT_CUH

#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <cuda_runtime.h>

#include "baracuda_atomic.cuh"

namespace baracuda { namespace segment {

// =============================================================================
// Reduction helpers — per-dtype +∞ / -∞ initial values for max / min.
// =============================================================================

// These identity constants are called from BOTH device kernels and
// host-side launcher code (e.g. `launch_unsorted_segment_max` passes
// `seg_max_init<T>()` as the fill value to a device kernel via
// cudaMemcpyToSymbol-equivalent host wrapper). Marking them
// `__host__ __device__` so NVCC emits both versions — otherwise the
// host call sites silently link against an unresolved host stub and
// the process aborts at launch time.
template <typename T> __host__ __device__ inline T seg_max_init();
template <> __host__ __device__ inline float  seg_max_init<float>()  { return -FLT_MAX; }
template <> __host__ __device__ inline double seg_max_init<double>() { return -DBL_MAX; }

template <typename T> __host__ __device__ inline T seg_min_init();
template <> __host__ __device__ inline float  seg_min_init<float>()  { return  FLT_MAX; }
template <> __host__ __device__ inline double seg_min_init<double>() { return  DBL_MAX; }

template <typename T> __host__ __device__ inline T seg_zero();
template <> __host__ __device__ inline float  seg_zero<float>()  { return 0.0f; }
template <> __host__ __device__ inline double seg_zero<double>() { return 0.0; }

template <typename T> __host__ __device__ inline T seg_one();
template <> __host__ __device__ inline float  seg_one<float>()  { return 1.0f; }
template <> __host__ __device__ inline double seg_one<double>() { return 1.0; }

// Atomic-max / atomic-min for f32 / f64 (sign-aware bit-trick on the
// underlying integer representation). Native atomicMax / atomicMin only
// exist for integer types; for FP we go through atomicCAS on the bit
// pattern. The standard "sign trick" is:
//   - if val >= 0: signed-int atomicMax on the bits
//   - if val <  0: unsigned-int atomicMin on the bits
// (and dual for atomicMin → swap signed/unsigned + min/max).

__device__ inline void atomic_max_f32(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old = __float_as_int(*addr);
    int assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (val <= cur) return;
        int newbits = __float_as_int(val);
        old = atomicCAS(iaddr, assumed, newbits);
    } while (assumed != old);
}

__device__ inline void atomic_min_f32(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old = __float_as_int(*addr);
    int assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        if (val >= cur) return;
        int newbits = __float_as_int(val);
        old = atomicCAS(iaddr, assumed, newbits);
    } while (assumed != old);
}

__device__ inline void atomic_max_f64(double* addr, double val) {
    unsigned long long* uaddr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = __double_as_longlong(*addr);
    unsigned long long assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        if (val <= cur) return;
        unsigned long long newbits = __double_as_longlong(val);
        old = atomicCAS(uaddr, assumed, newbits);
    } while (assumed != old);
}

__device__ inline void atomic_min_f64(double* addr, double val) {
    unsigned long long* uaddr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = __double_as_longlong(*addr);
    unsigned long long assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        if (val >= cur) return;
        unsigned long long newbits = __double_as_longlong(val);
        old = atomicCAS(uaddr, assumed, newbits);
    } while (assumed != old);
}

// `seg_atomic_add` routes to the unified `baracuda::atomic::add<T>`
// helper from `baracuda_atomic.cuh` (Phase 11.3 / Fuel team feedback
// #6) — generic over every dtype with a native or CAS-emulated
// atomicAdd. f32 / f64 fall through to the native intrinsic; half /
// bf16 use a 32-bit `atomicCAS` loop. The kernels here today only
// instantiate f32 / f64 but the helper covers the future half/bf16
// extension automatically.
template <typename T>
__device__ __forceinline__ void seg_atomic_add(T* addr, T val) {
    baracuda::atomic::add<T>(addr, val);
}

template <typename T> __device__ inline void seg_atomic_max(T* addr, T val);
template <> __device__ inline void seg_atomic_max<float >(float*  a, float  v) { atomic_max_f32(a, v); }
template <> __device__ inline void seg_atomic_max<double>(double* a, double v) { atomic_max_f64(a, v); }

template <typename T> __device__ inline void seg_atomic_min(T* addr, T val);
template <> __device__ inline void seg_atomic_min<float >(float*  a, float  v) { atomic_min_f32(a, v); }
template <> __device__ inline void seg_atomic_min<double>(double* a, double v) { atomic_min_f64(a, v); }

// =============================================================================
// SORTED FAMILY — one thread per (segment_id, d) output cell.
// =============================================================================
//
// Sweeps the half-open range [lo, hi) of `segment_ids` covering this
// segment_id (found by binary search) and accumulates input[i, d] with
// the requested reduction. The total launch is
// `num_segments * D` threads.

enum SegReduceOp : int { SEG_SUM = 0, SEG_MEAN = 1, SEG_MAX = 2, SEG_MIN = 3, SEG_PROD = 4 };

// Lower bound on monotonically non-decreasing array — first index i
// with segment_ids[i] >= target. Returns N if none. Phase 11.5:
// templated on the segment-id dtype.
template <typename IndexT>
__device__ inline int32_t seg_lower_bound(
    const IndexT* __restrict__ seg_ids, int32_t n, int32_t target)
{
    int32_t lo = 0, hi = n;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if ((int64_t)seg_ids[mid] < (int64_t)target) lo = mid + 1;
        else                                         hi = mid;
    }
    return lo;
}

template <typename T, int OP, typename IndexT>
__global__ void segment_sorted_kernel(
    const T*       __restrict__ input,        // [N, D]
    const IndexT*  __restrict__ segment_ids,  // [N]
    T*             __restrict__ output,       // [num_segments, D]
    int32_t        N,
    int32_t        D,
    int32_t        num_segments)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)num_segments * (int64_t)D;
    for (int64_t i = tid; i < total; i += step) {
        int32_t s = (int32_t)(i / (int64_t)D);
        int32_t d = (int32_t)(i - (int64_t)s * (int64_t)D);
        int32_t lo = seg_lower_bound(segment_ids, N, s);
        int32_t hi = seg_lower_bound(segment_ids, N, s + 1);
        int32_t count = hi - lo;

        T acc;
        if      (OP == SEG_SUM)  acc = seg_zero<T>();
        else if (OP == SEG_MEAN) acc = seg_zero<T>();
        else if (OP == SEG_MAX)  acc = seg_max_init<T>();
        else if (OP == SEG_MIN)  acc = seg_min_init<T>();
        else /* SEG_PROD */      acc = seg_one<T>();

        for (int32_t k = lo; k < hi; ++k) {
            T v = input[(int64_t)k * (int64_t)D + (int64_t)d];
            if      (OP == SEG_SUM)  acc = acc + v;
            else if (OP == SEG_MEAN) acc = acc + v;
            else if (OP == SEG_MAX)  acc = (v > acc) ? v : acc;
            else if (OP == SEG_MIN)  acc = (v < acc) ? v : acc;
            else /* SEG_PROD */      acc = acc * v;
        }

        T result;
        if (OP == SEG_MEAN) {
            // Empty segment → 0 (TF convention).
            result = (count > 0) ? (acc / (T)(double)count) : seg_zero<T>();
        } else if ((OP == SEG_MAX || OP == SEG_MIN) && count == 0) {
            // Empty max/min — leave the initial sentinel; PyTorch /
            // TF behavior here is implementation-defined. We emit 0
            // so the output isn't a sentinel that downstream
            // arithmetic can blow up on.
            result = seg_zero<T>();
        } else if (OP == SEG_PROD && count == 0) {
            result = seg_one<T>();
        } else {
            result = acc;
        }
        output[i] = result;
    }
}

template <typename T, int OP, typename IndexT>
__host__ inline int32_t launch_segment_sorted(
    const T* input, const IndexT* segment_ids, T* output,
    int32_t N, int32_t D, int32_t num_segments,
    cudaStream_t stream)
{
    if (N < 0 || D < 0 || num_segments < 0) return 2;
    if (N > 0 && (input == nullptr || segment_ids == nullptr)) return 2;
    if (num_segments > 0 && D > 0 && output == nullptr) return 2;
    int64_t total = (int64_t)num_segments * (int64_t)D;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    segment_sorted_kernel<T, OP, IndexT><<<blocks, kBlock, 0, stream>>>(
        input, segment_ids, output, N, D, num_segments);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// UNSORTED FAMILY — one thread per (n, d) input cell, atomic into output.
// =============================================================================
//
// Output buffer MUST be pre-initialized by the caller (or by the
// init-output kernel below) to the reduction identity:
//   sum → 0, max → -∞, min → +∞.
// The launcher does NOT call cudaMemset (we don't know the dtype
// memset value generically); the safe-layer plan zeroes / fills as
// part of its run().

template <typename T, typename IndexT>
__global__ void unsorted_segment_sum_kernel(
    const T*       __restrict__ input,
    const IndexT*  __restrict__ segment_ids,
    T*             __restrict__ output,
    int32_t        N,
    int32_t        D,
    int32_t        num_segments)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        int32_t d = (int32_t)(i - (int64_t)n * (int64_t)D);
        int64_t s = (int64_t)segment_ids[n];
        if (s < 0 || s >= (int64_t)num_segments) continue;
        seg_atomic_add<T>(&output[s * (int64_t)D + (int64_t)d], input[i]);
    }
}

template <typename T, typename IndexT>
__global__ void unsorted_segment_max_kernel(
    const T*       __restrict__ input,
    const IndexT*  __restrict__ segment_ids,
    T*             __restrict__ output,
    int32_t        N,
    int32_t        D,
    int32_t        num_segments)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        int32_t d = (int32_t)(i - (int64_t)n * (int64_t)D);
        int64_t s = (int64_t)segment_ids[n];
        if (s < 0 || s >= (int64_t)num_segments) continue;
        seg_atomic_max<T>(&output[s * (int64_t)D + (int64_t)d], input[i]);
    }
}

template <typename T, typename IndexT>
__global__ void unsorted_segment_min_kernel(
    const T*       __restrict__ input,
    const IndexT*  __restrict__ segment_ids,
    T*             __restrict__ output,
    int32_t        N,
    int32_t        D,
    int32_t        num_segments)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        int32_t d = (int32_t)(i - (int64_t)n * (int64_t)D);
        int64_t s = (int64_t)segment_ids[n];
        if (s < 0 || s >= (int64_t)num_segments) continue;
        seg_atomic_min<T>(&output[s * (int64_t)D + (int64_t)d], input[i]);
    }
}

// Init-output kernel — fill output with a per-op identity value.
// Used by the unsorted launchers because we can't cudaMemset an arbitrary
// FP bit pattern (0.0 works for sum, but max needs -∞ and min needs +∞).
template <typename T>
__global__ void init_fill_kernel(T* __restrict__ out, int64_t n, T value) {
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < n; i += step) out[i] = value;
}

template <typename T>
__host__ inline void launch_init_fill(T* out, int64_t n, T value, cudaStream_t stream) {
    if (n <= 0) return;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (n + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    init_fill_kernel<T><<<blocks, kBlock, 0, stream>>>(out, n, value);
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_unsorted_segment_sum(
    const T* input, const IndexT* segment_ids, T* output,
    int32_t N, int32_t D, int32_t num_segments,
    cudaStream_t stream)
{
    if (N < 0 || D < 0 || num_segments < 0) return 2;
    int64_t out_total = (int64_t)num_segments * (int64_t)D;
    if (out_total > 0) {
        if (output == nullptr) return 2;
        // SUM identity is 0; IEEE-754 f32/f64 zero is bytewise zero so
        // cudaMemsetAsync is safe (no need for init_fill_kernel here).
        cudaError_t merr = cudaMemsetAsync(
            output, 0, (size_t)out_total * sizeof(T), stream);
        if (merr != cudaSuccess) return 5;
    }
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (input == nullptr || segment_ids == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unsorted_segment_sum_kernel<T, IndexT><<<blocks, kBlock, 0, stream>>>(
        input, segment_ids, output, N, D, num_segments);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_unsorted_segment_max(
    const T* input, const IndexT* segment_ids, T* output,
    int32_t N, int32_t D, int32_t num_segments,
    cudaStream_t stream)
{
    if (N < 0 || D < 0 || num_segments < 0) return 2;
    int64_t out_total = (int64_t)num_segments * (int64_t)D;
    if (out_total > 0) {
        if (output == nullptr) return 2;
        launch_init_fill<T>(output, out_total, seg_max_init<T>(), stream);
    }
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (input == nullptr || segment_ids == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unsorted_segment_max_kernel<T, IndexT><<<blocks, kBlock, 0, stream>>>(
        input, segment_ids, output, N, D, num_segments);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_unsorted_segment_min(
    const T* input, const IndexT* segment_ids, T* output,
    int32_t N, int32_t D, int32_t num_segments,
    cudaStream_t stream)
{
    if (N < 0 || D < 0 || num_segments < 0) return 2;
    int64_t out_total = (int64_t)num_segments * (int64_t)D;
    if (out_total > 0) {
        if (output == nullptr) return 2;
        launch_init_fill<T>(output, out_total, seg_min_init<T>(), stream);
    }
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (input == nullptr || segment_ids == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unsorted_segment_min_kernel<T, IndexT><<<blocks, kBlock, 0, stream>>>(
        input, segment_ids, output, N, D, num_segments);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// MEAN HELPERS — per-segment integer counts + post-pass divide.
// =============================================================================
//
// Used by both sorted-mean BW and unsorted-mean (FW + BW). Sorted-mean
// FW computes the count inline from the binary-search range, so it
// doesn't need this helper.

template <typename IndexT>
__global__ void seg_count_kernel(
    const IndexT* __restrict__ segment_ids,
    int32_t*      __restrict__ counts,    // [num_segments] zero-init
    int32_t       N,
    int32_t       num_segments)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < N; i += step) {
        int64_t s = (int64_t)segment_ids[i];
        if (s < 0 || s >= (int64_t)num_segments) continue;
        atomicAdd(&counts[s], 1);
    }
}

template <typename T>
__global__ void seg_mean_divide_kernel(
    T*             __restrict__ output,    // [num_segments, D]
    const int32_t* __restrict__ counts,    // [num_segments]
    int32_t        num_segments,
    int32_t        D)
{
    int64_t total = (int64_t)num_segments * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t s = (int32_t)(i / (int64_t)D);
        int32_t c = counts[s];
        if (c > 0) output[i] = output[i] / (T)(double)c;
    }
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_unsorted_segment_mean(
    const T* input, const IndexT* segment_ids, T* output,
    int32_t* counts_workspace,            // [num_segments]
    int32_t N, int32_t D, int32_t num_segments,
    cudaStream_t stream)
{
    if (N < 0 || D < 0 || num_segments < 0) return 2;
    int32_t s = launch_unsorted_segment_sum<T, IndexT>(input, segment_ids, output,
                                                       N, D, num_segments, stream);
    if (s != 0) return s;
    if (num_segments == 0) return 0;
    if (counts_workspace == nullptr) return 4;
    cudaError_t err = cudaMemsetAsync(counts_workspace, 0,
                                      (size_t)num_segments * sizeof(int32_t), stream);
    if (err != cudaSuccess) return 5;
    if (N > 0) {
        constexpr int kBlock = 256;
        constexpr int64_t kMaxBlocks = 65535;
        int64_t blocks_i64 = ((int64_t)N + kBlock - 1) / kBlock;
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
        if (blocks <= 0) blocks = 1;
        seg_count_kernel<IndexT><<<blocks, kBlock, 0, stream>>>(
            segment_ids, counts_workspace, N, num_segments);
    }
    int64_t out_total = (int64_t)num_segments * (int64_t)D;
    if (out_total > 0) {
        constexpr int kBlock = 256;
        constexpr int64_t kMaxBlocks = 65535;
        int64_t blocks_i64 = (out_total + kBlock - 1) / kBlock;
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
        if (blocks <= 0) blocks = 1;
        seg_mean_divide_kernel<T><<<blocks, kBlock, 0, stream>>>(
            output, counts_workspace, num_segments, D);
    }
    err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// BACKWARD KERNELS — sum / mean only.
// =============================================================================
//
// Sum BW: d_input[n, d] = d_output[seg[n], d]. Pure gather along seg.
//   - sorted and unsorted share the same kernel body (the seg-ids
//     array is the only operand difference, but the access pattern is
//     identical — one thread per (n, d) reads d_output[seg[n], d]).
//
// Mean BW: d_input[n, d] = d_output[seg[n], d] / count[seg[n]].
//   - sorted: need the count → run seg_count_kernel into a workspace
//     (counts can also be computed via binary search but the count
//     kernel is simpler and amortizes across rows).
//   - unsorted: same path.

template <typename T, typename IndexT>
__global__ void segment_sum_backward_kernel(
    const T*       __restrict__ d_output,     // [num_segments, D]
    const IndexT*  __restrict__ segment_ids,  // [N]
    T*             __restrict__ d_input,      // [N, D]
    int32_t        N,
    int32_t        D,
    int32_t        num_segments)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        int32_t d = (int32_t)(i - (int64_t)n * (int64_t)D);
        int64_t s = (int64_t)segment_ids[n];
        T v = seg_zero<T>();
        if (s >= 0 && s < (int64_t)num_segments) {
            v = d_output[s * (int64_t)D + (int64_t)d];
        }
        d_input[i] = v;
    }
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_segment_sum_backward(
    const T* d_output, const IndexT* segment_ids, T* d_input,
    int32_t N, int32_t D, int32_t num_segments,
    cudaStream_t stream)
{
    if (N < 0 || D < 0 || num_segments < 0) return 2;
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (d_output == nullptr || segment_ids == nullptr || d_input == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    segment_sum_backward_kernel<T, IndexT><<<blocks, kBlock, 0, stream>>>(
        d_output, segment_ids, d_input, N, D, num_segments);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T, typename IndexT>
__global__ void segment_mean_backward_kernel(
    const T*       __restrict__ d_output,     // [num_segments, D]
    const IndexT*  __restrict__ segment_ids,  // [N]
    const int32_t* __restrict__ counts,       // [num_segments]
    T*             __restrict__ d_input,      // [N, D]
    int32_t        N,
    int32_t        D,
    int32_t        num_segments)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        int32_t d = (int32_t)(i - (int64_t)n * (int64_t)D);
        int64_t s = (int64_t)segment_ids[n];
        T v = seg_zero<T>();
        if (s >= 0 && s < (int64_t)num_segments) {
            int32_t c = counts[s];
            if (c > 0) {
                v = d_output[s * (int64_t)D + (int64_t)d] / (T)(double)c;
            }
        }
        d_input[i] = v;
    }
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_segment_mean_backward(
    const T* d_output, const IndexT* segment_ids, T* d_input,
    int32_t* counts_workspace,                     // [num_segments]
    int32_t N, int32_t D, int32_t num_segments,
    cudaStream_t stream)
{
    if (N < 0 || D < 0 || num_segments < 0) return 2;
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (d_output == nullptr || segment_ids == nullptr || d_input == nullptr) return 2;
    if (num_segments > 0 && counts_workspace == nullptr) return 4;
    if (num_segments > 0) {
        cudaError_t err = cudaMemsetAsync(counts_workspace, 0,
                                          (size_t)num_segments * sizeof(int32_t), stream);
        if (err != cudaSuccess) return 5;
        if (N > 0) {
            constexpr int kBlock = 256;
            constexpr int64_t kMaxBlocks = 65535;
            int64_t blocks_i64 = ((int64_t)N + kBlock - 1) / kBlock;
            int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
            if (blocks <= 0) blocks = 1;
            seg_count_kernel<IndexT><<<blocks, kBlock, 0, stream>>>(
                segment_ids, counts_workspace, N, num_segments);
        }
    }
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    segment_mean_backward_kernel<T, IndexT><<<blocks, kBlock, 0, stream>>>(
        d_output, segment_ids, counts_workspace, d_input, N, D, num_segments);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Phase 25 — Max / Min / Prod BW (sorted + unsorted) + Unsorted Prod FW.
//
// Design choices:
//
// - Max / Min BW: argmax / argmin is **recomputed in the BW kernel** by
//   re-scanning the segment, rather than saving an index tensor from
//   the FW. This preserves the FW API source-compat (no new output
//   shape) — see [[max-min-bw-no-new-shape]] / [[phase16-complete]] for
//   the precedent (FractionalMaxPool used the same recompute pattern).
//   Tie-break = first occurrence (lowest k). PyTorch chooses the *last*
//   occurrence; we document the divergence in the Rust plan.
//
// - Prod BW: direct `d_input[k, d] = d_output[seg, d] * (prod / x[k, d])`.
//   Numerically dangerous when `x[k, d] == 0` (yields NaN or Inf).
//   Documented as a caller-responsibility in the Rust plan. To avoid
//   re-running the forward, the BW signature takes `output` (the prod
//   from FW) as an input.
//
// - Unsorted Prod FW: `atomicCAS` retry loop on the underlying 32 / 64-bit
//   slot. Slow but allowed per OP-MATRIX. Non-deterministic.
// =============================================================================

// Per-dtype atomic-mul via CAS. f32 uses int slot; f64 uses
// unsigned long long. Same shape as atomic_max_f32 etc.
__device__ inline void atomic_mul_f32(float* addr, float val) {
    int* iaddr = reinterpret_cast<int*>(addr);
    int old = __float_as_int(*addr);
    int assumed;
    do {
        assumed = old;
        float cur = __int_as_float(assumed);
        float newv = cur * val;
        int newbits = __float_as_int(newv);
        old = atomicCAS(iaddr, assumed, newbits);
    } while (assumed != old);
}

__device__ inline void atomic_mul_f64(double* addr, double val) {
    unsigned long long* uaddr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = __double_as_longlong(*addr);
    unsigned long long assumed;
    do {
        assumed = old;
        double cur = __longlong_as_double(assumed);
        double newv = cur * val;
        unsigned long long newbits = __double_as_longlong(newv);
        old = atomicCAS(uaddr, assumed, newbits);
    } while (assumed != old);
}

template <typename T> __device__ inline void seg_atomic_mul(T* addr, T val);
template <> __device__ inline void seg_atomic_mul<float >(float*  a, float  v) { atomic_mul_f32(a, v); }
template <> __device__ inline void seg_atomic_mul<double>(double* a, double v) { atomic_mul_f64(a, v); }

// =============================================================================
// SORTED Max / Min BW — one thread per (n, d) input cell.
//
// Per input cell `(n, d)`:
//   1. seg = segment_ids[n]
//   2. locate the segment's [lo, hi) range via binary search
//   3. scan [lo, hi) to find the (first) k where input[k, d] is max
//   4. if k == n, gradient flows: d_input[n, d] = d_output[seg, d]
//      else d_input[n, d] = 0
//
// This produces the "first-occurrence" tie-break (PyTorch uses last;
// documented divergence). Empty segments / out-of-range seg ids: the
// d_input cell is left as zero.
// =============================================================================

enum SegArgOp : int { SEG_ARG_MAX = 0, SEG_ARG_MIN = 1 };

template <typename T, int OP, typename IndexT>
__global__ void segment_arg_backward_kernel(
    const T*       __restrict__ d_output,     // [num_segments, D]
    const T*       __restrict__ input,        // [N, D]
    const IndexT*  __restrict__ segment_ids,  // [N]
    T*             __restrict__ d_input,      // [N, D]
    int32_t        N,
    int32_t        D,
    int32_t        num_segments)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        int32_t d = (int32_t)(i - (int64_t)n * (int64_t)D);
        int64_t s = (int64_t)segment_ids[n];
        if (s < 0 || s >= (int64_t)num_segments) {
            d_input[i] = seg_zero<T>();
            continue;
        }
        // Find [lo, hi) for this segment via binary search.
        int32_t lo = seg_lower_bound<IndexT>(segment_ids, N, (int32_t)s);
        int32_t hi = seg_lower_bound<IndexT>(segment_ids, N, (int32_t)s + 1);
        // Find first-occurrence argmax (or argmin) at column d.
        int32_t arg = lo;
        T best = input[(int64_t)lo * (int64_t)D + (int64_t)d];
        for (int32_t k = lo + 1; k < hi; ++k) {
            T v = input[(int64_t)k * (int64_t)D + (int64_t)d];
            if (OP == SEG_ARG_MAX) {
                if (v > best) { best = v; arg = k; }
            } else {
                if (v < best) { best = v; arg = k; }
            }
        }
        if (n == arg) {
            d_input[i] = d_output[s * (int64_t)D + (int64_t)d];
        } else {
            d_input[i] = seg_zero<T>();
        }
    }
}

template <typename T, int OP, typename IndexT>
__host__ inline int32_t launch_segment_arg_backward(
    const T* d_output, const T* input, const IndexT* segment_ids, T* d_input,
    int32_t N, int32_t D, int32_t num_segments,
    cudaStream_t stream)
{
    if (N < 0 || D < 0 || num_segments < 0) return 2;
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (d_output == nullptr || input == nullptr || segment_ids == nullptr ||
        d_input == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    segment_arg_backward_kernel<T, OP, IndexT><<<blocks, kBlock, 0, stream>>>(
        d_output, input, segment_ids, d_input, N, D, num_segments);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// SORTED Prod BW — direct division.
//
// `d_input[k, d] = d_output[seg, d] * (output[seg, d] / x[k, d])`
//
// `output` is the FW `prod` result (caller must pass it in). When
// `x[k, d] == 0`, gradient is NaN or Inf — documented limitation.
// =============================================================================

template <typename T, typename IndexT>
__global__ void segment_prod_backward_kernel(
    const T*       __restrict__ d_output,     // [num_segments, D]
    const T*       __restrict__ input,        // [N, D]
    const T*       __restrict__ output,       // [num_segments, D] from FW
    const IndexT*  __restrict__ segment_ids,  // [N]
    T*             __restrict__ d_input,      // [N, D]
    int32_t        N,
    int32_t        D,
    int32_t        num_segments)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        int32_t d = (int32_t)(i - (int64_t)n * (int64_t)D);
        int64_t s = (int64_t)segment_ids[n];
        if (s < 0 || s >= (int64_t)num_segments) {
            d_input[i] = seg_zero<T>();
            continue;
        }
        int64_t out_off = s * (int64_t)D + (int64_t)d;
        T x_nd = input[i];
        T prod = output[out_off];
        T dy = d_output[out_off];
        // Direct division — yields NaN/Inf if x_nd == 0.
        d_input[i] = dy * (prod / x_nd);
    }
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_segment_prod_backward(
    const T* d_output, const T* input, const T* output,
    const IndexT* segment_ids, T* d_input,
    int32_t N, int32_t D, int32_t num_segments,
    cudaStream_t stream)
{
    if (N < 0 || D < 0 || num_segments < 0) return 2;
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (d_output == nullptr || input == nullptr || output == nullptr ||
        segment_ids == nullptr || d_input == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    segment_prod_backward_kernel<T, IndexT><<<blocks, kBlock, 0, stream>>>(
        d_output, input, output, segment_ids, d_input, N, D, num_segments);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// UNSORTED Max / Min BW — same recompute pattern but scans the full
// input array per (n, d) cell (segment ids may be in any order).
//
// One thread per (n, d). Thread reads its seg = segment_ids[n], then
// scans m ∈ [0, N) looking at input[m, d] where segment_ids[m] == seg
// to locate the first-occurrence argmax/argmin. If m == n, gradient
// flows in. O(N) work per cell — slow on big N — but unsorted seg ids
// don't admit a binary-search range, and the input layout means we
// can't shortcut.
// =============================================================================

template <typename T, int OP, typename IndexT>
__global__ void unsorted_segment_arg_backward_kernel(
    const T*       __restrict__ d_output,     // [num_segments, D]
    const T*       __restrict__ input,        // [N, D]
    const IndexT*  __restrict__ segment_ids,  // [N]
    T*             __restrict__ d_input,      // [N, D]
    int32_t        N,
    int32_t        D,
    int32_t        num_segments)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        int32_t d = (int32_t)(i - (int64_t)n * (int64_t)D);
        int64_t s = (int64_t)segment_ids[n];
        if (s < 0 || s >= (int64_t)num_segments) {
            d_input[i] = seg_zero<T>();
            continue;
        }
        // Scan full input to find first-occurrence arg-extreme in seg s.
        int32_t arg = -1;
        T best;
        if (OP == SEG_ARG_MAX) best = seg_max_init<T>();
        else                   best = seg_min_init<T>();
        for (int32_t m = 0; m < N; ++m) {
            int64_t sm = (int64_t)segment_ids[m];
            if (sm != s) continue;
            T v = input[(int64_t)m * (int64_t)D + (int64_t)d];
            if (arg < 0) {
                best = v; arg = m;
                continue;
            }
            if (OP == SEG_ARG_MAX) {
                if (v > best) { best = v; arg = m; }
            } else {
                if (v < best) { best = v; arg = m; }
            }
        }
        if (n == arg) {
            d_input[i] = d_output[s * (int64_t)D + (int64_t)d];
        } else {
            d_input[i] = seg_zero<T>();
        }
    }
}

template <typename T, int OP, typename IndexT>
__host__ inline int32_t launch_unsorted_segment_arg_backward(
    const T* d_output, const T* input, const IndexT* segment_ids, T* d_input,
    int32_t N, int32_t D, int32_t num_segments,
    cudaStream_t stream)
{
    if (N < 0 || D < 0 || num_segments < 0) return 2;
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (d_output == nullptr || input == nullptr || segment_ids == nullptr ||
        d_input == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unsorted_segment_arg_backward_kernel<T, OP, IndexT><<<blocks, kBlock, 0, stream>>>(
        d_output, input, segment_ids, d_input, N, D, num_segments);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// UNSORTED Prod FW — atomicCAS retry loop. Output pre-initialized to 1.
// =============================================================================

template <typename T, typename IndexT>
__global__ void unsorted_segment_prod_kernel(
    const T*       __restrict__ input,
    const IndexT*  __restrict__ segment_ids,
    T*             __restrict__ output,
    int32_t        N,
    int32_t        D,
    int32_t        num_segments)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        int32_t d = (int32_t)(i - (int64_t)n * (int64_t)D);
        int64_t s = (int64_t)segment_ids[n];
        if (s < 0 || s >= (int64_t)num_segments) continue;
        seg_atomic_mul<T>(&output[s * (int64_t)D + (int64_t)d], input[i]);
    }
}

template <typename T, typename IndexT>
__host__ inline int32_t launch_unsorted_segment_prod(
    const T* input, const IndexT* segment_ids, T* output,
    int32_t N, int32_t D, int32_t num_segments,
    cudaStream_t stream)
{
    if (N < 0 || D < 0 || num_segments < 0) return 2;
    int64_t out_total = (int64_t)num_segments * (int64_t)D;
    if (out_total > 0) {
        if (output == nullptr) return 2;
        launch_init_fill<T>(output, out_total, seg_one<T>(), stream);
    }
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (input == nullptr || segment_ids == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    unsorted_segment_prod_kernel<T, IndexT><<<blocks, kBlock, 0, stream>>>(
        input, segment_ids, output, N, D, num_segments);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::segment

// =============================================================================
// INSTANTIATE macros — emit `extern "C"` launcher per (op, dtype) pair.
//
// Two FFI signatures used here:
//
//   1. Sorted FW (sum / mean / max / min / prod):
//        (N, D, num_segments, input, segment_ids, output, ws, ws_bytes, stream)
//      Workspace is unused (segment_sorted_kernel doesn't need it; mean
//      computes count inline). The ws params are present for FFI shape
//      uniformity with the rest of the family.
//
//   2. Unsorted FW sum / max / min:
//        (N, D, num_segments, input, segment_ids, output, ws, ws_bytes, stream)
//      Same shape; ws unused for sum / max / min.
//
//   3. Unsorted FW mean:
//        (N, D, num_segments, input, segment_ids, output, ws, ws_bytes, stream)
//      Requires a `num_segments * sizeof(i32)` workspace for the counts
//      buffer.
//
//   4. Sum BW (sorted or unsorted — same signature, same kernel):
//        (N, D, num_segments, d_output, segment_ids, d_input, ws, ws_bytes, stream)
//      No workspace needed.
//
//   5. Mean BW (sorted or unsorted — same signature, same kernel):
//        (N, D, num_segments, d_output, segment_ids, d_input, ws, ws_bytes, stream)
//      Requires a `num_segments * sizeof(i32)` workspace for the counts
//      buffer.
// =============================================================================

// Phase 11.5: `INDEX_T` parameter selects the segment-id dtype
// (`int32_t` or `int64_t`).
#define BARACUDA_KERNELS_SEGMENT_SORTED_INSTANTIATE(NAME, T, OP, INDEX_T)                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* input,                                                                        \
        const void* segment_ids,                                                                  \
        void* output,                                                                             \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        int64_t total = (int64_t)num_segments * (int64_t)D;                                       \
        if (total == 0) return 0;                                                                 \
        if (output == nullptr) return 2;                                                          \
        if (N > 0 && (input == nullptr || segment_ids == nullptr)) return 2;                      \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::segment::launch_segment_sorted<T, OP, INDEX_T>(                          \
            static_cast<const T*>(input),                                                         \
            static_cast<const INDEX_T*>(segment_ids),                                             \
            static_cast<T*>(output),                                                              \
            N, D, num_segments, stream);                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* /*input*/,                                                                    \
        const void* /*segment_ids*/,                                                              \
        const void* /*output*/)                                                                   \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_UNSORTED_SEGMENT_SUM_INSTANTIATE(NAME, T, INDEX_T)                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* input,                                                                        \
        const void* segment_ids,                                                                  \
        void* output,                                                                             \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::segment::launch_unsorted_segment_sum<T, INDEX_T>(                        \
            static_cast<const T*>(input),                                                         \
            static_cast<const INDEX_T*>(segment_ids),                                             \
            static_cast<T*>(output),                                                              \
            N, D, num_segments, stream);                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* /*input*/,                                                                    \
        const void* /*segment_ids*/,                                                              \
        const void* /*output*/)                                                                   \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_UNSORTED_SEGMENT_MAX_INSTANTIATE(NAME, T, INDEX_T)                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* input,                                                                        \
        const void* segment_ids,                                                                  \
        void* output,                                                                             \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::segment::launch_unsorted_segment_max<T, INDEX_T>(                        \
            static_cast<const T*>(input),                                                         \
            static_cast<const INDEX_T*>(segment_ids),                                             \
            static_cast<T*>(output),                                                              \
            N, D, num_segments, stream);                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* /*input*/,                                                                    \
        const void* /*segment_ids*/,                                                              \
        const void* /*output*/)                                                                   \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_UNSORTED_SEGMENT_MIN_INSTANTIATE(NAME, T, INDEX_T)                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* input,                                                                        \
        const void* segment_ids,                                                                  \
        void* output,                                                                             \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::segment::launch_unsorted_segment_min<T, INDEX_T>(                        \
            static_cast<const T*>(input),                                                         \
            static_cast<const INDEX_T*>(segment_ids),                                             \
            static_cast<T*>(output),                                                              \
            N, D, num_segments, stream);                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* /*input*/,                                                                    \
        const void* /*segment_ids*/,                                                              \
        const void* /*output*/)                                                                   \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_UNSORTED_SEGMENT_MEAN_INSTANTIATE(NAME, T, INDEX_T)                      \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* input,                                                                        \
        const void* segment_ids,                                                                  \
        void* output,                                                                             \
        void* workspace, size_t workspace_bytes,                                                  \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        if (num_segments > 0 && workspace == nullptr) return 4;                                   \
        if ((size_t)num_segments * sizeof(int32_t) > workspace_bytes && num_segments > 0)         \
            return 4;                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::segment::launch_unsorted_segment_mean<T, INDEX_T>(                       \
            static_cast<const T*>(input),                                                         \
            static_cast<const INDEX_T*>(segment_ids),                                             \
            static_cast<T*>(output),                                                              \
            static_cast<int32_t*>(workspace),                                                     \
            N, D, num_segments, stream);                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* /*input*/,                                                                    \
        const void* /*segment_ids*/,                                                              \
        const void* /*output*/)                                                                   \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_SEGMENT_SUM_BACKWARD_INSTANTIATE(NAME, T, INDEX_T)                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* d_output,                                                                     \
        const void* segment_ids,                                                                  \
        void* d_input,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::segment::launch_segment_sum_backward<T, INDEX_T>(                        \
            static_cast<const T*>(d_output),                                                      \
            static_cast<const INDEX_T*>(segment_ids),                                             \
            static_cast<T*>(d_input),                                                             \
            N, D, num_segments, stream);                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* /*d_output*/,                                                                 \
        const void* /*segment_ids*/,                                                              \
        const void* /*d_input*/)                                                                  \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_SEGMENT_MEAN_BACKWARD_INSTANTIATE(NAME, T, INDEX_T)                      \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* d_output,                                                                     \
        const void* segment_ids,                                                                  \
        void* d_input,                                                                            \
        void* workspace, size_t workspace_bytes,                                                  \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        if (num_segments > 0 && workspace == nullptr) return 4;                                   \
        if ((size_t)num_segments * sizeof(int32_t) > workspace_bytes && num_segments > 0)         \
            return 4;                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::segment::launch_segment_mean_backward<T, INDEX_T>(                       \
            static_cast<const T*>(d_output),                                                      \
            static_cast<const INDEX_T*>(segment_ids),                                             \
            static_cast<T*>(d_input),                                                             \
            static_cast<int32_t*>(workspace),                                                     \
            N, D, num_segments, stream);                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* /*d_output*/,                                                                 \
        const void* /*segment_ids*/,                                                              \
        const void* /*d_input*/)                                                                  \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        return 0;                                                                                  \
    }

// =============================================================================
// Phase 25 INSTANTIATE macros.
//
// 1. Segment Max / Min BW (sorted + unsorted) — signature mirrors the
//    sum/mean BW family but with an extra `input` pointer (re-scan source).
//
//    (N, D, num_segments, d_output, input, segment_ids, d_input,
//     ws, ws_bytes, stream)
//
// 2. Segment Prod BW (sorted + unsorted) — signature mirrors the BW
//    family plus `input` and `output` (saved FW prod).
//
//    (N, D, num_segments, d_output, input, output, segment_ids, d_input,
//     ws, ws_bytes, stream)
//
// 3. Unsorted Segment Prod FW — same shape as the other unsorted-FW
//    INSTANTIATEs (the launcher pre-fills output with 1).
//
//    (N, D, num_segments, input, segment_ids, output, ws, ws_bytes, stream)
// =============================================================================

#define BARACUDA_KERNELS_SEGMENT_ARG_BACKWARD_INSTANTIATE(NAME, T, OP, INDEX_T)                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* d_output,                                                                     \
        const void* input,                                                                        \
        const void* segment_ids,                                                                  \
        void* d_input,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::segment::launch_segment_arg_backward<T, OP, INDEX_T>(                    \
            static_cast<const T*>(d_output),                                                      \
            static_cast<const T*>(input),                                                         \
            static_cast<const INDEX_T*>(segment_ids),                                             \
            static_cast<T*>(d_input),                                                             \
            N, D, num_segments, stream);                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* /*d_output*/,                                                                 \
        const void* /*input*/,                                                                    \
        const void* /*segment_ids*/,                                                              \
        const void* /*d_input*/)                                                                  \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_UNSORTED_SEGMENT_ARG_BACKWARD_INSTANTIATE(NAME, T, OP, INDEX_T)          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* d_output,                                                                     \
        const void* input,                                                                        \
        const void* segment_ids,                                                                  \
        void* d_input,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::segment::launch_unsorted_segment_arg_backward<T, OP, INDEX_T>(           \
            static_cast<const T*>(d_output),                                                      \
            static_cast<const T*>(input),                                                         \
            static_cast<const INDEX_T*>(segment_ids),                                             \
            static_cast<T*>(d_input),                                                             \
            N, D, num_segments, stream);                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* /*d_output*/,                                                                 \
        const void* /*input*/,                                                                    \
        const void* /*segment_ids*/,                                                              \
        const void* /*d_input*/)                                                                  \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_SEGMENT_PROD_BACKWARD_INSTANTIATE(NAME, T, INDEX_T)                      \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* d_output,                                                                     \
        const void* input,                                                                        \
        const void* output,                                                                       \
        const void* segment_ids,                                                                  \
        void* d_input,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::segment::launch_segment_prod_backward<T, INDEX_T>(                       \
            static_cast<const T*>(d_output),                                                      \
            static_cast<const T*>(input),                                                         \
            static_cast<const T*>(output),                                                        \
            static_cast<const INDEX_T*>(segment_ids),                                             \
            static_cast<T*>(d_input),                                                             \
            N, D, num_segments, stream);                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* /*d_output*/,                                                                 \
        const void* /*input*/,                                                                    \
        const void* /*output*/,                                                                   \
        const void* /*segment_ids*/,                                                              \
        const void* /*d_input*/)                                                                  \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_UNSORTED_SEGMENT_PROD_INSTANTIATE(NAME, T, INDEX_T)                      \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* input,                                                                        \
        const void* segment_ids,                                                                  \
        void* output,                                                                             \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                          \
        void* stream_ptr)                                                                         \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::segment::launch_unsorted_segment_prod<T, INDEX_T>(                       \
            static_cast<const T*>(input),                                                         \
            static_cast<const INDEX_T*>(segment_ids),                                             \
            static_cast<T*>(output),                                                              \
            N, D, num_segments, stream);                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                   \
        int32_t N, int32_t D, int32_t num_segments,                                               \
        const void* /*input*/,                                                                    \
        const void* /*segment_ids*/,                                                              \
        const void* /*output*/)                                                                   \
    {                                                                                              \
        if (N < 0 || D < 0 || num_segments < 0) return 2;                                         \
        return 0;                                                                                  \
    }

#endif // BARACUDA_SEGMENT_CUH
