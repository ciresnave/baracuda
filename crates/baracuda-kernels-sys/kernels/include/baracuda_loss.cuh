// baracuda_loss.cuh
//
// Templated kernels and INSTANTIATE macros for the loss op family
// (Phase 5 Category R of the comprehensive plan).
//
// Today's wiring (FW + BW × 4 FP dtypes: f32, f16, bf16, f64):
//
//   - MSE              `y = mean((pred - target)²)` (or sum)
//                      BW: `dpred = 2·(pred - target) / N`
//                          `dtarget = -dpred`
//   - NLLLoss          `y = -mean(input[target_idx[i]])` along feature axis
//                      target is i64 class indices.
//                      BW: `dinput[i, c] = -dy / N if c == target[i] else 0`
//   - CrossEntropyLoss `y = NLLLoss(LogSoftmax(input), target)`. Class-index
//                      target only (i64). Fused FW for stability.
//                      BW: `dinput[i, c] = (softmax(input)[i, c] - 1{c==t[i]}) / N`
//   - BCELoss          `y = -mean(target·log(pred) + (1-target)·log(1-pred))`
//                      pred ∈ (0, 1) — caller's responsibility.
//                      BW: `dpred = (pred - target) / (pred·(1-pred)) / N`
//   - KLDivLoss        `y = mean(target·(log(target) - input))`. PyTorch
//                      convention: `input` is already log-prob.
//                      BW: `dinput = -target / N`
//
// Reduction modes:
//   0 = None (output is per-cell tensor, same shape as the loss surface)
//   1 = Mean (output is scalar — single T element)
//   2 = Sum  (output is scalar — single T element)
//
// Design — fully deterministic, no atomic adds:
//
//   For Mean / Sum modes, the FW kernels split into two passes:
//   1. **per-cell pass**: every thread computes one cell's loss term in
//      f32 / f64 accumulator, casts to T, writes to a workspace buffer
//      of shape == per-cell loss shape (numel = total terms before
//      reduction).
//   2. **single-block reduction pass**: one block, kBlockReduce threads,
//      cooperative tree reduction with warp shuffles + smem; the lone
//      surviving thread casts (and divides by N if Mean) and writes the
//      final scalar into output[0].
//
//   For None mode, the per-cell kernel writes directly into the output
//   buffer; the reduction pass is skipped.
//
// Status codes match the elementwise family:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch failure.

#ifndef BARACUDA_LOSS_CUH
#define BARACUDA_LOSS_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_elementwise.cuh" // for MAX_RANK

namespace baracuda { namespace loss {

// =============================================================================
// dtype helpers — f32 detour for half / bf16, native otherwise.
// =============================================================================

template <typename T>
__device__ __forceinline__ float load_as_acc(T x) { return (float)x; }

template <>
__device__ __forceinline__ float load_as_acc<__half>(__half x) { return __half2float(x); }

template <>
__device__ __forceinline__ float load_as_acc<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T store_from_acc(float v) { return (T)v; }

template <>
__device__ __forceinline__ __half store_from_acc<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_from_acc<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ double load_as_acc_d(T x) { return (double)x; }

template <typename T>
__device__ __forceinline__ T store_from_acc_d(double v) { return (T)v; }

// =============================================================================
// Single-block tree reduction over a contiguous T[] of length `n`.
// Final thread divides by `denom` (1.0 for Sum, n for Mean) and writes
// the result to `out[0]`. Deterministic — single block, no atomics.
// =============================================================================

constexpr int kBlockReduce = 256;

template <typename T>
__global__ void loss_reduce_finalize_kernel(
    const T* __restrict__ buf,
    T* __restrict__ out,
    int64_t n,
    float denom_inv) // 1.0/n for Mean, 1.0 for Sum
{
    __shared__ float smem[kBlockReduce];
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int64_t i = tid; i < n; i += kBlockReduce) {
        acc += load_as_acc<T>(buf[i]);
    }
    smem[tid] = acc;
    __syncthreads();
    for (int s = kBlockReduce / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = store_from_acc<T>(smem[0] * denom_inv);
    }
}

template <>
__global__ void loss_reduce_finalize_kernel<double>(
    const double* __restrict__ buf,
    double* __restrict__ out,
    int64_t n,
    float denom_inv)
{
    // For f64 we ignore `denom_inv` (which is a lossy f32 of 1/N) and
    // recompute the divide in double from `n`. denom_inv == 1.0 (no
    // divide) is encoded by passing `n == 0` from the launcher… but we
    // need a separate signal. Convention: if `denom_inv` is non-zero and
    // numerically equals 1.0f then this is Sum mode; else Mean.
    __shared__ double smem[kBlockReduce];
    int tid = threadIdx.x;
    double acc = 0.0;
    for (int64_t i = tid; i < n; i += kBlockReduce) {
        acc += buf[i];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int s = kBlockReduce / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        // Sum mode: denom_inv == 1.0f exactly; Mean: < 1.0f.
        if (denom_inv >= 1.0f) {
            out[0] = smem[0];
        } else {
            out[0] = smem[0] / (double)n;
        }
    }
}

// =============================================================================
// MSE FW per-cell: `term[i] = (pred[i] - target[i])²`
// =============================================================================

template <typename T>
__global__ void mse_per_cell_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    T* __restrict__ term,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float p = load_as_acc<T>(pred[i]);
        float t = load_as_acc<T>(target[i]);
        float d = p - t;
        term[i] = store_from_acc<T>(d * d);
    }
}

template <>
__global__ void mse_per_cell_kernel<double>(
    const double* __restrict__ pred,
    const double* __restrict__ target,
    double* __restrict__ term,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        double d = pred[i] - target[i];
        term[i] = d * d;
    }
}

// MSE BW: `dpred[i] = 2·(pred[i] - target[i]) · scale`.
// Mean: `scale = dy[0] / N`; Sum: `scale = dy[0]`; None: `scale = dy[i]`.
// The kernel reads dy from device memory (no host-side sync needed).
// `inv_n_or_one` is `1.0 / N` for Mean, `1.0` for Sum, ignored for None.
template <typename T>
__global__ void mse_backward_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    const T* __restrict__ dy,
    T* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < numel; i += step) {
        float p = load_as_acc<T>(pred[i]);
        float t = load_as_acc<T>(target[i]);
        float s = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        dpred[i] = store_from_acc<T>(2.0f * (p - t) * s);
    }
}

template <>
__global__ void mse_backward_kernel<double>(
    const double* __restrict__ pred,
    const double* __restrict__ target,
    const double* __restrict__ dy,
    double* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    // f64 precision: recompute the reduction scale in double from numel
    // (the f32 `inv_n_or_one` would lose ~7 digits in the mean divisor).
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)numel; // Mean
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0]; // Sum
    }
    for (int64_t i = tid; i < numel; i += step) {
        double s = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        dpred[i] = 2.0 * (pred[i] - target[i]) * s;
    }
}

// =============================================================================
// BCE FW per-cell: `term[i] = -(target·log(pred) + (1-target)·log(1-pred))`
// =============================================================================

template <typename T>
__global__ void bce_per_cell_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    T* __restrict__ term,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float p = load_as_acc<T>(pred[i]);
        float t = load_as_acc<T>(target[i]);
        float v = -(t * logf(p) + (1.0f - t) * logf(1.0f - p));
        term[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void bce_per_cell_kernel<double>(
    const double* __restrict__ pred,
    const double* __restrict__ target,
    double* __restrict__ term,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        double p = pred[i];
        double t = target[i];
        term[i] = -(t * log(p) + (1.0 - t) * log(1.0 - p));
    }
}

// BCE BW: `dpred[i] = (pred[i] - target[i]) / (pred·(1-pred)) · scale`
template <typename T>
__global__ void bce_backward_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    const T* __restrict__ dy,
    T* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < numel; i += step) {
        float p = load_as_acc<T>(pred[i]);
        float t = load_as_acc<T>(target[i]);
        float s = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        float v = (p - t) / (p * (1.0f - p)) * s;
        dpred[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void bce_backward_kernel<double>(
    const double* __restrict__ pred,
    const double* __restrict__ target,
    const double* __restrict__ dy,
    double* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)numel;
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0];
    }
    (void)inv_n_or_one;
    for (int64_t i = tid; i < numel; i += step) {
        double p = pred[i];
        double t = target[i];
        double s = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        dpred[i] = (p - t) / (p * (1.0 - p)) * s;
    }
}

// =============================================================================
// KLDiv FW per-cell: `term[i] = target[i]·(log(target[i]) - input[i])`
// (PyTorch convention: input is already log-prob)
// =============================================================================

template <typename T>
__global__ void kl_div_per_cell_kernel(
    const T* __restrict__ input,
    const T* __restrict__ target,
    T* __restrict__ term,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float inp = load_as_acc<T>(input[i]);
        float t   = load_as_acc<T>(target[i]);
        // Convention: when target == 0, contribution is 0 (avoid log(0)).
        float v = (t > 0.0f) ? (t * (logf(t) - inp)) : 0.0f;
        term[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void kl_div_per_cell_kernel<double>(
    const double* __restrict__ input,
    const double* __restrict__ target,
    double* __restrict__ term,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        double t = target[i];
        double v = (t > 0.0) ? (t * (log(t) - input[i])) : 0.0;
        term[i] = v;
    }
}

// KLDiv BW: `dinput[i] = -target[i] · scale`
template <typename T>
__global__ void kl_div_backward_kernel(
    const T* __restrict__ target,
    const T* __restrict__ dy,
    T* __restrict__ dinput,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < numel; i += step) {
        float t = load_as_acc<T>(target[i]);
        float s = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        dinput[i] = store_from_acc<T>(-t * s);
    }
}

template <>
__global__ void kl_div_backward_kernel<double>(
    const double* __restrict__ target,
    const double* __restrict__ dy,
    double* __restrict__ dinput,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)numel;
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0];
    }
    (void)inv_n_or_one;
    for (int64_t i = tid; i < numel; i += step) {
        double s = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        dinput[i] = -target[i] * s;
    }
}

// =============================================================================
// NLLLoss FW per-row: input is [N, C] flattened by (rows = N, class_extent = C),
//                     target is i64[N]; per_row_term[i] = -input[i, target[i]]
// =============================================================================

template <typename T>
__global__ void nll_per_row_kernel(
    const T* __restrict__ input,
    const int64_t* __restrict__ target,
    T* __restrict__ term, // shape [N]
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < n_rows; i += step) {
        int64_t t = target[i];
        if (t < 0 || t >= (int64_t)class_extent) {
            term[i] = store_from_acc<T>(0.0f);
            continue;
        }
        float v = load_as_acc<T>(input[i * row_stride_input + t]);
        term[i] = store_from_acc<T>(-v);
    }
}

template <>
__global__ void nll_per_row_kernel<double>(
    const double* __restrict__ input,
    const int64_t* __restrict__ target,
    double* __restrict__ term,
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < n_rows; i += step) {
        int64_t t = target[i];
        if (t < 0 || t >= (int64_t)class_extent) {
            term[i] = 0.0;
            continue;
        }
        term[i] = -input[i * row_stride_input + t];
    }
}

// NLLLoss BW: `dinput[i, c] = -dy_or_scale if c == target[i] else 0`
// `dy` is shape [N] for None mode, shape [1] (scalar) for Mean/Sum.
// We zero the whole dinput tensor first, then write the active cell per row.
// Note: launcher does the zero via cudaMemsetAsync; this kernel only fills the
// active cells.
template <typename T>
__global__ void nll_backward_kernel(
    const T* __restrict__ dy,
    const int64_t* __restrict__ target,
    T* __restrict__ dinput,
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < n_rows; i += step) {
        int64_t t = target[i];
        if (t < 0 || t >= (int64_t)class_extent) continue;
        float s = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        dinput[i * row_stride_input + t] = store_from_acc<T>(-s);
    }
}

template <>
__global__ void nll_backward_kernel<double>(
    const double* __restrict__ dy,
    const int64_t* __restrict__ target,
    double* __restrict__ dinput,
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)n_rows;
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0];
    }
    (void)inv_n_or_one;
    for (int64_t i = tid; i < n_rows; i += step) {
        int64_t t = target[i];
        if (t < 0 || t >= (int64_t)class_extent) continue;
        double s = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        dinput[i * row_stride_input + t] = -s;
    }
}

// =============================================================================
// CrossEntropyLoss FW per-row: input [N, C], target i64[N].
// Per row: stable LogSoftmax over class axis, then loss = -log_softmax[target].
// term[i] = -(input[i, target[i]] - max - log(Σ exp(input[i, j] - max)))
//         = -(input[i, t]) + max + log(Σ exp(input[i, j] - max))
// =============================================================================

template <typename T>
__global__ void cross_entropy_per_row_kernel(
    const T* __restrict__ input,
    const int64_t* __restrict__ target,
    T* __restrict__ term,
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < n_rows; i += step) {
        int64_t t = target[i];
        const T* row = input + i * row_stride_input;
        if (t < 0 || t >= (int64_t)class_extent) {
            term[i] = store_from_acc<T>(0.0f);
            continue;
        }
        float m = -INFINITY;
        for (int j = 0; j < class_extent; ++j) {
            float v = load_as_acc<T>(row[j]);
            if (v > m) m = v;
        }
        float s = 0.0f;
        for (int j = 0; j < class_extent; ++j) {
            s += expf(load_as_acc<T>(row[j]) - m);
        }
        float xt = load_as_acc<T>(row[t]);
        // -log_softmax[t] = -(xt - m - log(s)) = -xt + m + log(s)
        float v = -xt + m + logf(s);
        term[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void cross_entropy_per_row_kernel<double>(
    const double* __restrict__ input,
    const int64_t* __restrict__ target,
    double* __restrict__ term,
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < n_rows; i += step) {
        int64_t t = target[i];
        const double* row = input + i * row_stride_input;
        if (t < 0 || t >= (int64_t)class_extent) {
            term[i] = 0.0;
            continue;
        }
        double m = -INFINITY;
        for (int j = 0; j < class_extent; ++j) {
            double v = row[j];
            if (v > m) m = v;
        }
        double s = 0.0;
        for (int j = 0; j < class_extent; ++j) {
            s += exp(row[j] - m);
        }
        term[i] = -row[t] + m + log(s);
    }
}

// CrossEntropyLoss BW: dinput[i, c] = (softmax(input)[i, c] - 1{c==t[i]}) · scale
// where scale = (dy/N) for Mean, dy for Sum, dy[i] for None.
template <typename T>
__global__ void cross_entropy_backward_kernel(
    const T* __restrict__ input,
    const int64_t* __restrict__ target,
    const T* __restrict__ dy,
    T* __restrict__ dinput,
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < n_rows; i += step) {
        int64_t t = target[i];
        const T* row = input + i * row_stride_input;
        T* drow = dinput + i * row_stride_input;
        float s = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        if (t < 0 || t >= (int64_t)class_extent) {
            // Zero out this row.
            for (int j = 0; j < class_extent; ++j) {
                drow[j] = store_from_acc<T>(0.0f);
            }
            continue;
        }
        // Compute softmax(row).
        float m = -INFINITY;
        for (int j = 0; j < class_extent; ++j) {
            float v = load_as_acc<T>(row[j]);
            if (v > m) m = v;
        }
        float sum = 0.0f;
        for (int j = 0; j < class_extent; ++j) {
            sum += expf(load_as_acc<T>(row[j]) - m);
        }
        for (int j = 0; j < class_extent; ++j) {
            float p = expf(load_as_acc<T>(row[j]) - m) / sum;
            float one_hot = (j == (int)t) ? 1.0f : 0.0f;
            drow[j] = store_from_acc<T>((p - one_hot) * s);
        }
    }
}

template <>
__global__ void cross_entropy_backward_kernel<double>(
    const double* __restrict__ input,
    const int64_t* __restrict__ target,
    const double* __restrict__ dy,
    double* __restrict__ dinput,
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)n_rows;
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0];
    }
    (void)inv_n_or_one;
    for (int64_t i = tid; i < n_rows; i += step) {
        int64_t t = target[i];
        const double* row = input + i * row_stride_input;
        double* drow = dinput + i * row_stride_input;
        double s = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        if (t < 0 || t >= (int64_t)class_extent) {
            for (int j = 0; j < class_extent; ++j) drow[j] = 0.0;
            continue;
        }
        double m = -INFINITY;
        for (int j = 0; j < class_extent; ++j) {
            if (row[j] > m) m = row[j];
        }
        double sum = 0.0;
        for (int j = 0; j < class_extent; ++j) {
            sum += exp(row[j] - m);
        }
        for (int j = 0; j < class_extent; ++j) {
            double p = exp(row[j] - m) / sum;
            double one_hot = (j == (int)t) ? 1.0 : 0.0;
            drow[j] = (p - one_hot) * s;
        }
    }
}

// =============================================================================
// Milestone 5.2 — Tier-1 regression losses (L1 / SmoothL1 / Huber)
//
// All three share the same FW shape: per-cell elementwise `x = pred - target`
// then a piecewise scalar formula. BW: per-cell piecewise scalar applied with
// the standard reduction-scale factor. SmoothL1 carries `beta`, Huber carries
// `delta`; L1 has no extra parameter (passed as a no-op `0.0f`).
// =============================================================================

// L1 FW per-cell: `term[i] = |pred - target|`.
template <typename T>
__global__ void l1_per_cell_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    T* __restrict__ term,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float p = load_as_acc<T>(pred[i]);
        float t = load_as_acc<T>(target[i]);
        float d = p - t;
        term[i] = store_from_acc<T>(d >= 0.0f ? d : -d);
    }
}

template <>
__global__ void l1_per_cell_kernel<double>(
    const double* __restrict__ pred,
    const double* __restrict__ target,
    double* __restrict__ term,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        double d = pred[i] - target[i];
        term[i] = d >= 0.0 ? d : -d;
    }
}

// L1 BW: `dpred[i] = sign(pred - target) · scale`. Subgradient at 0 = 0
// (PyTorch convention).
template <typename T>
__global__ void l1_backward_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    const T* __restrict__ dy,
    T* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < numel; i += step) {
        float p = load_as_acc<T>(pred[i]);
        float t = load_as_acc<T>(target[i]);
        float d = p - t;
        float sgn = (d > 0.0f) ? 1.0f : ((d < 0.0f) ? -1.0f : 0.0f);
        float s = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        dpred[i] = store_from_acc<T>(sgn * s);
    }
}

template <>
__global__ void l1_backward_kernel<double>(
    const double* __restrict__ pred,
    const double* __restrict__ target,
    const double* __restrict__ dy,
    double* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)numel;
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0];
    }
    (void)inv_n_or_one;
    for (int64_t i = tid; i < numel; i += step) {
        double d = pred[i] - target[i];
        double sgn = (d > 0.0) ? 1.0 : ((d < 0.0) ? -1.0 : 0.0);
        double s = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        dpred[i] = sgn * s;
    }
}

// SmoothL1 FW per-cell: piecewise; `0.5·(x/β)²·β` if `|x|<β` else `|x|-0.5·β`.
template <typename T>
__global__ void smooth_l1_per_cell_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    T* __restrict__ term,
    int64_t numel,
    float beta)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float p = load_as_acc<T>(pred[i]);
        float t = load_as_acc<T>(target[i]);
        float x = p - t;
        float ax = x >= 0.0f ? x : -x;
        float v;
        if (ax < beta) {
            v = 0.5f * x * x / beta;
        } else {
            v = ax - 0.5f * beta;
        }
        term[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void smooth_l1_per_cell_kernel<double>(
    const double* __restrict__ pred,
    const double* __restrict__ target,
    double* __restrict__ term,
    int64_t numel,
    float beta_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double beta = (double)beta_f;
    for (int64_t i = tid; i < numel; i += step) {
        double x = pred[i] - target[i];
        double ax = x >= 0.0 ? x : -x;
        double v;
        if (ax < beta) {
            v = 0.5 * x * x / beta;
        } else {
            v = ax - 0.5 * beta;
        }
        term[i] = v;
    }
}

// SmoothL1 BW: `dpred[i] = (x/β) · scale` if `|x|<β`, else `sign(x) · scale`.
template <typename T>
__global__ void smooth_l1_backward_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    const T* __restrict__ dy,
    T* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one,
    float beta)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < numel; i += step) {
        float p = load_as_acc<T>(pred[i]);
        float t = load_as_acc<T>(target[i]);
        float x = p - t;
        float ax = x >= 0.0f ? x : -x;
        float v;
        if (ax < beta) {
            v = x / beta;
        } else {
            v = (x > 0.0f) ? 1.0f : -1.0f;
        }
        float s = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        dpred[i] = store_from_acc<T>(v * s);
    }
}

template <>
__global__ void smooth_l1_backward_kernel<double>(
    const double* __restrict__ pred,
    const double* __restrict__ target,
    const double* __restrict__ dy,
    double* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one,
    float beta_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)numel;
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0];
    }
    (void)inv_n_or_one;
    double beta = (double)beta_f;
    for (int64_t i = tid; i < numel; i += step) {
        double x = pred[i] - target[i];
        double ax = x >= 0.0 ? x : -x;
        double v;
        if (ax < beta) {
            v = x / beta;
        } else {
            v = (x > 0.0) ? 1.0 : -1.0;
        }
        double s = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        dpred[i] = v * s;
    }
}

// Huber FW per-cell: `0.5·x²` if `|x|<δ`, else `δ·(|x| - 0.5·δ)`.
template <typename T>
__global__ void huber_per_cell_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    T* __restrict__ term,
    int64_t numel,
    float delta)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float p = load_as_acc<T>(pred[i]);
        float t = load_as_acc<T>(target[i]);
        float x = p - t;
        float ax = x >= 0.0f ? x : -x;
        float v;
        if (ax < delta) {
            v = 0.5f * x * x;
        } else {
            v = delta * (ax - 0.5f * delta);
        }
        term[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void huber_per_cell_kernel<double>(
    const double* __restrict__ pred,
    const double* __restrict__ target,
    double* __restrict__ term,
    int64_t numel,
    float delta_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double delta = (double)delta_f;
    for (int64_t i = tid; i < numel; i += step) {
        double x = pred[i] - target[i];
        double ax = x >= 0.0 ? x : -x;
        double v;
        if (ax < delta) {
            v = 0.5 * x * x;
        } else {
            v = delta * (ax - 0.5 * delta);
        }
        term[i] = v;
    }
}

// Huber BW: `dpred[i] = x · scale` if `|x|<δ`, else `δ·sign(x) · scale`.
template <typename T>
__global__ void huber_backward_kernel(
    const T* __restrict__ pred,
    const T* __restrict__ target,
    const T* __restrict__ dy,
    T* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one,
    float delta)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < numel; i += step) {
        float p = load_as_acc<T>(pred[i]);
        float t = load_as_acc<T>(target[i]);
        float x = p - t;
        float ax = x >= 0.0f ? x : -x;
        float v;
        if (ax < delta) {
            v = x;
        } else {
            v = (x > 0.0f) ? delta : -delta;
        }
        float s = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        dpred[i] = store_from_acc<T>(v * s);
    }
}

template <>
__global__ void huber_backward_kernel<double>(
    const double* __restrict__ pred,
    const double* __restrict__ target,
    const double* __restrict__ dy,
    double* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one,
    float delta_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)numel;
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0];
    }
    (void)inv_n_or_one;
    double delta = (double)delta_f;
    for (int64_t i = tid; i < numel; i += step) {
        double x = pred[i] - target[i];
        double ax = x >= 0.0 ? x : -x;
        double v;
        if (ax < delta) {
            v = x;
        } else {
            v = (x > 0.0) ? delta : -delta;
        }
        double s = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        dpred[i] = v * s;
    }
}

// =============================================================================
// BCEWithLogits FW per-cell: numerically stable BCE for raw logits.
//   term[i] = max(x, 0) - x · target + log(1 + exp(-|x|))
// where x = logits[i].
// =============================================================================

template <typename T>
__global__ void bce_with_logits_per_cell_kernel(
    const T* __restrict__ logits,
    const T* __restrict__ target,
    T* __restrict__ term,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float x = load_as_acc<T>(logits[i]);
        float t = load_as_acc<T>(target[i]);
        float ax = x >= 0.0f ? x : -x;
        float mx = x > 0.0f ? x : 0.0f;
        float v = mx - x * t + log1pf(expf(-ax));
        term[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void bce_with_logits_per_cell_kernel<double>(
    const double* __restrict__ logits,
    const double* __restrict__ target,
    double* __restrict__ term,
    int64_t numel)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        double x = logits[i];
        double t = target[i];
        double ax = x >= 0.0 ? x : -x;
        double mx = x > 0.0 ? x : 0.0;
        term[i] = mx - x * t + log1p(exp(-ax));
    }
}

// BCEWithLogits BW: `dpred[i] = (sigmoid(x) - target) · scale`.
// Numerically stable sigmoid: `sigmoid(x) = exp(-max(-x, 0)) / (1 + exp(-|x|))`.
template <typename T>
__global__ void bce_with_logits_backward_kernel(
    const T* __restrict__ logits,
    const T* __restrict__ target,
    const T* __restrict__ dy,
    T* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < numel; i += step) {
        float x = load_as_acc<T>(logits[i]);
        float t = load_as_acc<T>(target[i]);
        // Stable sigmoid.
        float sig;
        if (x >= 0.0f) {
            float e = expf(-x);
            sig = 1.0f / (1.0f + e);
        } else {
            float e = expf(x);
            sig = e / (1.0f + e);
        }
        float sc = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        dpred[i] = store_from_acc<T>((sig - t) * sc);
    }
}

template <>
__global__ void bce_with_logits_backward_kernel<double>(
    const double* __restrict__ logits,
    const double* __restrict__ target,
    const double* __restrict__ dy,
    double* __restrict__ dpred,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)numel;
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0];
    }
    (void)inv_n_or_one;
    for (int64_t i = tid; i < numel; i += step) {
        double x = logits[i];
        double t = target[i];
        double sig;
        if (x >= 0.0) {
            double e = exp(-x);
            sig = 1.0 / (1.0 + e);
        } else {
            double e = exp(x);
            sig = e / (1.0 + e);
        }
        double sc = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        dpred[i] = (sig - t) * sc;
    }
}

// =============================================================================
// PoissonNLL FW per-cell.
//   log_input=true:  term[i] = exp(input[i]) - target[i] · input[i]
//   log_input=false: term[i] = input[i] - target[i] · log(input[i])
// =============================================================================

template <typename T>
__global__ void poisson_nll_per_cell_kernel(
    const T* __restrict__ input,
    const T* __restrict__ target,
    T* __restrict__ term,
    int64_t numel,
    int32_t log_input_flag)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float x = load_as_acc<T>(input[i]);
        float t = load_as_acc<T>(target[i]);
        float v;
        if (log_input_flag != 0) {
            v = expf(x) - t * x;
        } else {
            v = x - t * logf(x);
        }
        term[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void poisson_nll_per_cell_kernel<double>(
    const double* __restrict__ input,
    const double* __restrict__ target,
    double* __restrict__ term,
    int64_t numel,
    int32_t log_input_flag)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        double x = input[i];
        double t = target[i];
        double v;
        if (log_input_flag != 0) {
            v = exp(x) - t * x;
        } else {
            v = x - t * log(x);
        }
        term[i] = v;
    }
}

// PoissonNLL BW.
//   log_input=true:  dinput[i] = (exp(input) - target) · scale
//   log_input=false: dinput[i] = (1 - target/input) · scale
template <typename T>
__global__ void poisson_nll_backward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ target,
    const T* __restrict__ dy,
    T* __restrict__ dinput,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one,
    int32_t log_input_flag)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < numel; i += step) {
        float x = load_as_acc<T>(input[i]);
        float t = load_as_acc<T>(target[i]);
        float v;
        if (log_input_flag != 0) {
            v = expf(x) - t;
        } else {
            v = 1.0f - t / x;
        }
        float sc = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        dinput[i] = store_from_acc<T>(v * sc);
    }
}

template <>
__global__ void poisson_nll_backward_kernel<double>(
    const double* __restrict__ input,
    const double* __restrict__ target,
    const double* __restrict__ dy,
    double* __restrict__ dinput,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one,
    int32_t log_input_flag)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)numel;
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0];
    }
    (void)inv_n_or_one;
    for (int64_t i = tid; i < numel; i += step) {
        double x = input[i];
        double t = target[i];
        double v;
        if (log_input_flag != 0) {
            v = exp(x) - t;
        } else {
            v = 1.0 - t / x;
        }
        double sc = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        dinput[i] = v * sc;
    }
}

// =============================================================================
// GaussianNLL FW per-cell.
//   v_eff = max(var, eps)
//   term[i] = 0.5 · (log(v_eff) + (input - target)² / v_eff)
// =============================================================================

template <typename T>
__global__ void gaussian_nll_per_cell_kernel(
    const T* __restrict__ input,
    const T* __restrict__ target,
    const T* __restrict__ var,
    T* __restrict__ term,
    int64_t numel,
    float eps)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float x = load_as_acc<T>(input[i]);
        float t = load_as_acc<T>(target[i]);
        float vv = load_as_acc<T>(var[i]);
        float ve = vv > eps ? vv : eps;
        float d = x - t;
        float v = 0.5f * (logf(ve) + d * d / ve);
        term[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void gaussian_nll_per_cell_kernel<double>(
    const double* __restrict__ input,
    const double* __restrict__ target,
    const double* __restrict__ var,
    double* __restrict__ term,
    int64_t numel,
    float eps_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double eps = (double)eps_f;
    for (int64_t i = tid; i < numel; i += step) {
        double x = input[i];
        double t = target[i];
        double vv = var[i];
        double ve = vv > eps ? vv : eps;
        double d = x - t;
        term[i] = 0.5 * (log(ve) + d * d / ve);
    }
}

// GaussianNLL BW.
//   dinput[i] = (input - target) / max(var, eps) · scale
template <typename T>
__global__ void gaussian_nll_backward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ target,
    const T* __restrict__ var,
    const T* __restrict__ dy,
    T* __restrict__ dinput,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one,
    float eps)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < numel; i += step) {
        float x = load_as_acc<T>(input[i]);
        float t = load_as_acc<T>(target[i]);
        float vv = load_as_acc<T>(var[i]);
        float ve = vv > eps ? vv : eps;
        float sc = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        dinput[i] = store_from_acc<T>((x - t) / ve * sc);
    }
}

template <>
__global__ void gaussian_nll_backward_kernel<double>(
    const double* __restrict__ input,
    const double* __restrict__ target,
    const double* __restrict__ var,
    const double* __restrict__ dy,
    double* __restrict__ dinput,
    int64_t numel,
    int32_t reduction_mode,
    float inv_n_or_one,
    float eps_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)numel;
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0];
    }
    (void)inv_n_or_one;
    double eps = (double)eps_f;
    for (int64_t i = tid; i < numel; i += step) {
        double vv = var[i];
        double ve = vv > eps ? vv : eps;
        double sc = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        dinput[i] = (input[i] - target[i]) / ve * sc;
    }
}

// =============================================================================
// Soft-target CrossEntropy FW per-row.
//   y[n] = -Σ_c target[n,c] · log_softmax(input)[n,c]
// target is a soft probability tensor of shape [n_rows, class_extent] (same
// dtype as input).
// =============================================================================

template <typename T>
__global__ void cross_entropy_soft_per_row_kernel(
    const T* __restrict__ input,
    const T* __restrict__ target,
    T* __restrict__ term,
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input,
    int64_t row_stride_target)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < n_rows; i += step) {
        const T* row = input + i * row_stride_input;
        const T* trow = target + i * row_stride_target;
        float m = -INFINITY;
        for (int j = 0; j < class_extent; ++j) {
            float v = load_as_acc<T>(row[j]);
            if (v > m) m = v;
        }
        float s = 0.0f;
        for (int j = 0; j < class_extent; ++j) {
            s += expf(load_as_acc<T>(row[j]) - m);
        }
        float lse = m + logf(s);
        // accumulate -sum_c target[n, c] * (row[c] - lse) = -sum_c tc·row[c] + lse·sum_c tc
        float acc = 0.0f;
        for (int j = 0; j < class_extent; ++j) {
            float tc = load_as_acc<T>(trow[j]);
            float xc = load_as_acc<T>(row[j]);
            acc += tc * (xc - lse);
        }
        term[i] = store_from_acc<T>(-acc);
    }
}

template <>
__global__ void cross_entropy_soft_per_row_kernel<double>(
    const double* __restrict__ input,
    const double* __restrict__ target,
    double* __restrict__ term,
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input,
    int64_t row_stride_target)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < n_rows; i += step) {
        const double* row = input + i * row_stride_input;
        const double* trow = target + i * row_stride_target;
        double m = -INFINITY;
        for (int j = 0; j < class_extent; ++j) {
            if (row[j] > m) m = row[j];
        }
        double s = 0.0;
        for (int j = 0; j < class_extent; ++j) {
            s += exp(row[j] - m);
        }
        double lse = m + log(s);
        double acc = 0.0;
        for (int j = 0; j < class_extent; ++j) {
            acc += trow[j] * (row[j] - lse);
        }
        term[i] = -acc;
    }
}

// Soft-target CrossEntropy BW.
//   dinput[n, c] = (softmax(input)[n, c] - target[n, c]) · scale
template <typename T>
__global__ void cross_entropy_soft_backward_kernel(
    const T* __restrict__ input,
    const T* __restrict__ target,
    const T* __restrict__ dy,
    T* __restrict__ dinput,
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input,
    int64_t row_stride_target,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < n_rows; i += step) {
        const T* row = input + i * row_stride_input;
        const T* trow = target + i * row_stride_target;
        T* drow = dinput + i * row_stride_input;
        float sc = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        float m = -INFINITY;
        for (int j = 0; j < class_extent; ++j) {
            float v = load_as_acc<T>(row[j]);
            if (v > m) m = v;
        }
        float sum = 0.0f;
        for (int j = 0; j < class_extent; ++j) {
            sum += expf(load_as_acc<T>(row[j]) - m);
        }
        for (int j = 0; j < class_extent; ++j) {
            float p = expf(load_as_acc<T>(row[j]) - m) / sum;
            float tc = load_as_acc<T>(trow[j]);
            drow[j] = store_from_acc<T>((p - tc) * sc);
        }
    }
}

template <>
__global__ void cross_entropy_soft_backward_kernel<double>(
    const double* __restrict__ input,
    const double* __restrict__ target,
    const double* __restrict__ dy,
    double* __restrict__ dinput,
    int64_t n_rows,
    int32_t class_extent,
    int64_t row_stride_input,
    int64_t row_stride_target,
    int32_t reduction_mode,
    float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) {
        dy0_scaled = dy[0] / (double)n_rows;
    } else if (reduction_mode == 2) {
        dy0_scaled = dy[0];
    }
    (void)inv_n_or_one;
    for (int64_t i = tid; i < n_rows; i += step) {
        const double* row = input + i * row_stride_input;
        const double* trow = target + i * row_stride_target;
        double* drow = dinput + i * row_stride_input;
        double sc = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        double m = -INFINITY;
        for (int j = 0; j < class_extent; ++j) {
            if (row[j] > m) m = row[j];
        }
        double sum = 0.0;
        for (int j = 0; j < class_extent; ++j) {
            sum += exp(row[j] - m);
        }
        for (int j = 0; j < class_extent; ++j) {
            double p = exp(row[j] - m) / sum;
            drow[j] = (p - trow[j]) * sc;
        }
    }
}

// =============================================================================
// Host launcher helpers.
// =============================================================================

// Common per-cell launch (elementwise loss: MSE / BCE / KLDiv).
template <typename T, typename Kernel>
__host__ inline int32_t launch_elementwise_loss_fw(
    Kernel kernel,
    const T* pred, const T* target, T* out,
    int64_t numel,
    int32_t reduction_mode,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;

    // None: write per-cell terms directly to out.
    if (reduction_mode == 0) {
        kernel<<<blocks, kBlock, 0, stream>>>(pred, target, out, numel);
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;
    }
    // Mean / Sum: per-cell -> workspace[numel*T], then single-block reduce to out[0].
    size_t need = (size_t)numel * sizeof(T);
    if (workspace == nullptr || workspace_bytes < need) return 4;
    T* term = static_cast<T*>(workspace);
    kernel<<<blocks, kBlock, 0, stream>>>(pred, target, term, numel);
    if (cudaGetLastError() != cudaSuccess) return 5;
    float denom_inv = (reduction_mode == 1) ? (1.0f / (float)numel) : 1.0f;
    loss_reduce_finalize_kernel<T><<<1, kBlockReduce, 0, stream>>>(term, out, numel, denom_inv);
    return (cudaGetLastError() == cudaSuccess) ? 0 : 5;
}

} } // namespace baracuda::loss

// =============================================================================
// INSTANTIATE macros
// =============================================================================

// Elementwise-loss FW ABI (pred, target, out, [workspace, workspace_bytes],
// numel, reduction_mode, stream).
#define BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE(NAME, T, KFN)                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t reduction_mode,                                                                     \
        const void* pred, const void* target, void* out,                                            \
        void* workspace, size_t workspace_bytes,                                                    \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (pred == nullptr || target == nullptr || out == nullptr) return 2;                       \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::loss::launch_elementwise_loss_fw<T>(                                       \
            baracuda::loss::KFN<T>,                                                                 \
            static_cast<const T*>(pred), static_cast<const T*>(target),                             \
            static_cast<T*>(out), numel, reduction_mode, workspace, workspace_bytes, stream);       \
    }

// Elementwise-loss BW ABI (pred, target, dy, dpred, numel, reduction_mode,
// scale_scalar, stream). For BCE / MSE — KLDiv has a thinner ABI (no pred).
#define BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE(NAME, T, KFN)                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t reduction_mode,                                                                     \
        float scale_scalar,                                                                         \
        const void* pred, const void* target, const void* dy, void* dpred,                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (pred == nullptr || target == nullptr || dy == nullptr || dpred == nullptr) return 2;    \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::KFN<T><<<blocks, kBlock, 0, stream>>>(                                      \
            static_cast<const T*>(pred), static_cast<const T*>(target),                             \
            static_cast<const T*>(dy), static_cast<T*>(dpred),                                      \
            numel, reduction_mode, scale_scalar);                                                   \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// KLDiv BW ABI (target, dy, dinput, numel, reduction_mode, scale_scalar, stream).
#define BARACUDA_KERNELS_LOSS_KL_DIV_BW_INSTANTIATE(NAME, T)                                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t reduction_mode,                                                                     \
        float scale_scalar,                                                                         \
        const void* target, const void* dy, void* dinput,                                           \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (target == nullptr || dy == nullptr || dinput == nullptr) return 2;                      \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::kl_div_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(                   \
            static_cast<const T*>(target), static_cast<const T*>(dy),                               \
            static_cast<T*>(dinput), numel, reduction_mode, scale_scalar);                          \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// NLL FW launcher — input [n_rows, class_extent], target i64[n_rows], output
// per-row [n_rows] for None mode or scalar [1] for Mean/Sum. Workspace
// holds the per-row term buffer for Mean/Sum reduction.
#define BARACUDA_KERNELS_LOSS_NLL_FW_INSTANTIATE(NAME, T)                                           \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows,                                                                             \
        int32_t class_extent,                                                                       \
        int64_t row_stride_input,                                                                   \
        int32_t reduction_mode,                                                                     \
        const void* input, const void* target, void* out,                                           \
        void* workspace, size_t workspace_bytes,                                                    \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0) return 2;                                              \
        if (n_rows == 0) return 0;                                                                  \
        if (input == nullptr || target == nullptr || out == nullptr) return 2;                      \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        if (reduction_mode == 0) {                                                                  \
            baracuda::loss::nll_per_row_kernel<T><<<blocks, kBlock, 0, stream>>>(                   \
                static_cast<const T*>(input),                                                       \
                static_cast<const int64_t*>(target),                                                \
                static_cast<T*>(out), n_rows, class_extent, row_stride_input);                      \
            return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                     \
        }                                                                                           \
        size_t need = (size_t)n_rows * sizeof(T);                                                   \
        if (workspace == nullptr || workspace_bytes < need) return 4;                               \
        T* term = static_cast<T*>(workspace);                                                       \
        baracuda::loss::nll_per_row_kernel<T><<<blocks, kBlock, 0, stream>>>(                       \
            static_cast<const T*>(input),                                                           \
            static_cast<const int64_t*>(target),                                                    \
            term, n_rows, class_extent, row_stride_input);                                          \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        float denom_inv = (reduction_mode == 1) ? (1.0f / (float)n_rows) : 1.0f;                    \
        baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,        \
            stream>>>(term, static_cast<T*>(out), n_rows, denom_inv);                               \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// NLL BW launcher — pre-zero dinput then write the active cells.
#define BARACUDA_KERNELS_LOSS_NLL_BW_INSTANTIATE(NAME, T)                                           \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows,                                                                             \
        int32_t class_extent,                                                                       \
        int64_t row_stride_input,                                                                   \
        int64_t dinput_numel,                                                                       \
        int32_t reduction_mode,                                                                     \
        float scale_scalar,                                                                         \
        const void* dy, const void* target, void* dinput,                                           \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0 || dinput_numel < 0) return 2;                          \
        if (n_rows == 0) return 0;                                                                  \
        if (dy == nullptr || target == nullptr || dinput == nullptr) return 2;                      \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        cudaMemsetAsync(dinput, 0, (size_t)dinput_numel * sizeof(T), stream);                       \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::nll_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(                      \
            static_cast<const T*>(dy),                                                              \
            static_cast<const int64_t*>(target),                                                    \
            static_cast<T*>(dinput),                                                                \
            n_rows, class_extent, row_stride_input, reduction_mode, scale_scalar);                  \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// CrossEntropy FW launcher.
#define BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_FW_INSTANTIATE(NAME, T)                                 \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows,                                                                             \
        int32_t class_extent,                                                                       \
        int64_t row_stride_input,                                                                   \
        int32_t reduction_mode,                                                                     \
        const void* input, const void* target, void* out,                                           \
        void* workspace, size_t workspace_bytes,                                                    \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0) return 2;                                              \
        if (n_rows == 0) return 0;                                                                  \
        if (input == nullptr || target == nullptr || out == nullptr) return 2;                      \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        if (reduction_mode == 0) {                                                                  \
            baracuda::loss::cross_entropy_per_row_kernel<T><<<blocks, kBlock, 0, stream>>>(         \
                static_cast<const T*>(input),                                                       \
                static_cast<const int64_t*>(target),                                                \
                static_cast<T*>(out), n_rows, class_extent, row_stride_input);                      \
            return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                     \
        }                                                                                           \
        size_t need = (size_t)n_rows * sizeof(T);                                                   \
        if (workspace == nullptr || workspace_bytes < need) return 4;                               \
        T* term = static_cast<T*>(workspace);                                                       \
        baracuda::loss::cross_entropy_per_row_kernel<T><<<blocks, kBlock, 0, stream>>>(             \
            static_cast<const T*>(input),                                                           \
            static_cast<const int64_t*>(target),                                                    \
            term, n_rows, class_extent, row_stride_input);                                          \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        float denom_inv = (reduction_mode == 1) ? (1.0f / (float)n_rows) : 1.0f;                    \
        baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,        \
            stream>>>(term, static_cast<T*>(out), n_rows, denom_inv);                               \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// CrossEntropy BW launcher.
#define BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_BW_INSTANTIATE(NAME, T)                                 \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows,                                                                             \
        int32_t class_extent,                                                                       \
        int64_t row_stride_input,                                                                   \
        int32_t reduction_mode,                                                                     \
        float scale_scalar,                                                                         \
        const void* input, const void* target, const void* dy, void* dinput,                        \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0) return 2;                                              \
        if (n_rows == 0) return 0;                                                                  \
        if (input == nullptr || target == nullptr || dy == nullptr || dinput == nullptr) return 2;  \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::cross_entropy_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(            \
            static_cast<const T*>(input),                                                           \
            static_cast<const int64_t*>(target),                                                    \
            static_cast<const T*>(dy),                                                              \
            static_cast<T*>(dinput),                                                                \
            n_rows, class_extent, row_stride_input, reduction_mode, scale_scalar);                  \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// =============================================================================
// Milestone 5.2 — Tier-1 loss INSTANTIATE macros.
// =============================================================================

// Parameterized elementwise-loss FW ABI: identical to the basic
// `BARACUDA_KERNELS_LOSS_ELEMENTWISE_FW_INSTANTIATE` but threads a single
// `float param` (β / δ) through to the kernel.
#define BARACUDA_KERNELS_LOSS_PARAM_FW_INSTANTIATE(NAME, T, KFN)                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t reduction_mode,                                                                     \
        float param,                                                                                \
        const void* pred, const void* target, void* out,                                            \
        void* workspace, size_t workspace_bytes,                                                    \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (pred == nullptr || target == nullptr || out == nullptr) return 2;                       \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        if (reduction_mode == 0) {                                                                  \
            baracuda::loss::KFN<T><<<blocks, kBlock, 0, stream>>>(                                  \
                static_cast<const T*>(pred), static_cast<const T*>(target),                         \
                static_cast<T*>(out), numel, param);                                                \
            return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                     \
        }                                                                                           \
        size_t need = (size_t)numel * sizeof(T);                                                    \
        if (workspace == nullptr || workspace_bytes < need) return 4;                               \
        T* term = static_cast<T*>(workspace);                                                       \
        baracuda::loss::KFN<T><<<blocks, kBlock, 0, stream>>>(                                      \
            static_cast<const T*>(pred), static_cast<const T*>(target),                             \
            term, numel, param);                                                                    \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        float denom_inv = (reduction_mode == 1) ? (1.0f / (float)numel) : 1.0f;                     \
        baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,        \
            stream>>>(term, static_cast<T*>(out), numel, denom_inv);                                \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// Parameterized elementwise-loss BW ABI: same shape as
// `BARACUDA_KERNELS_LOSS_ELEMENTWISE_BW_INSTANTIATE` plus a trailing `float param`.
#define BARACUDA_KERNELS_LOSS_PARAM_BW_INSTANTIATE(NAME, T, KFN)                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t reduction_mode,                                                                     \
        float scale_scalar,                                                                         \
        float param,                                                                                \
        const void* pred, const void* target, const void* dy, void* dpred,                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (pred == nullptr || target == nullptr || dy == nullptr || dpred == nullptr) return 2;    \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::KFN<T><<<blocks, kBlock, 0, stream>>>(                                      \
            static_cast<const T*>(pred), static_cast<const T*>(target),                             \
            static_cast<const T*>(dy), static_cast<T*>(dpred),                                      \
            numel, reduction_mode, scale_scalar, param);                                            \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// PoissonNLL FW: threads an `int32_t log_input_flag` instead of a float param.
#define BARACUDA_KERNELS_LOSS_POISSON_NLL_FW_INSTANTIATE(NAME, T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t reduction_mode,                                                                     \
        int32_t log_input_flag,                                                                     \
        const void* input, const void* target, void* out,                                           \
        void* workspace, size_t workspace_bytes,                                                    \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (input == nullptr || target == nullptr || out == nullptr) return 2;                      \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        if (reduction_mode == 0) {                                                                  \
            baracuda::loss::poisson_nll_per_cell_kernel<T><<<blocks, kBlock, 0, stream>>>(          \
                static_cast<const T*>(input), static_cast<const T*>(target),                        \
                static_cast<T*>(out), numel, log_input_flag);                                       \
            return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                     \
        }                                                                                           \
        size_t need = (size_t)numel * sizeof(T);                                                    \
        if (workspace == nullptr || workspace_bytes < need) return 4;                               \
        T* term = static_cast<T*>(workspace);                                                       \
        baracuda::loss::poisson_nll_per_cell_kernel<T><<<blocks, kBlock, 0, stream>>>(              \
            static_cast<const T*>(input), static_cast<const T*>(target),                            \
            term, numel, log_input_flag);                                                           \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        float denom_inv = (reduction_mode == 1) ? (1.0f / (float)numel) : 1.0f;                     \
        baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,        \
            stream>>>(term, static_cast<T*>(out), numel, denom_inv);                                \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// PoissonNLL BW.
#define BARACUDA_KERNELS_LOSS_POISSON_NLL_BW_INSTANTIATE(NAME, T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t reduction_mode,                                                                     \
        float scale_scalar,                                                                         \
        int32_t log_input_flag,                                                                     \
        const void* input, const void* target, const void* dy, void* dinput,                        \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (input == nullptr || target == nullptr || dy == nullptr || dinput == nullptr) return 2;  \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::poisson_nll_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(              \
            static_cast<const T*>(input), static_cast<const T*>(target),                            \
            static_cast<const T*>(dy), static_cast<T*>(dinput),                                     \
            numel, reduction_mode, scale_scalar, log_input_flag);                                   \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// GaussianNLL FW (3-tensor input: input, target, var; carries `eps`).
#define BARACUDA_KERNELS_LOSS_GAUSSIAN_NLL_FW_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t reduction_mode,                                                                     \
        float eps,                                                                                  \
        const void* input, const void* target, const void* var, void* out,                          \
        void* workspace, size_t workspace_bytes,                                                    \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (input == nullptr || target == nullptr || var == nullptr || out == nullptr) return 2;    \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        if (reduction_mode == 0) {                                                                  \
            baracuda::loss::gaussian_nll_per_cell_kernel<T><<<blocks, kBlock, 0, stream>>>(         \
                static_cast<const T*>(input), static_cast<const T*>(target),                        \
                static_cast<const T*>(var),                                                         \
                static_cast<T*>(out), numel, eps);                                                  \
            return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                     \
        }                                                                                           \
        size_t need = (size_t)numel * sizeof(T);                                                    \
        if (workspace == nullptr || workspace_bytes < need) return 4;                               \
        T* term = static_cast<T*>(workspace);                                                       \
        baracuda::loss::gaussian_nll_per_cell_kernel<T><<<blocks, kBlock, 0, stream>>>(             \
            static_cast<const T*>(input), static_cast<const T*>(target),                            \
            static_cast<const T*>(var), term, numel, eps);                                          \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        float denom_inv = (reduction_mode == 1) ? (1.0f / (float)numel) : 1.0f;                     \
        baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,        \
            stream>>>(term, static_cast<T*>(out), numel, denom_inv);                                \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// GaussianNLL BW.
#define BARACUDA_KERNELS_LOSS_GAUSSIAN_NLL_BW_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t reduction_mode,                                                                     \
        float scale_scalar,                                                                         \
        float eps,                                                                                  \
        const void* input, const void* target, const void* var, const void* dy, void* dinput,       \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (input == nullptr || target == nullptr || var == nullptr ||                              \
            dy == nullptr || dinput == nullptr) return 2;                                           \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::gaussian_nll_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(             \
            static_cast<const T*>(input), static_cast<const T*>(target),                            \
            static_cast<const T*>(var),                                                             \
            static_cast<const T*>(dy), static_cast<T*>(dinput),                                     \
            numel, reduction_mode, scale_scalar, eps);                                              \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// Soft-target CrossEntropy FW (target is `T` of shape [n_rows, class_extent]).
#define BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_SOFT_FW_INSTANTIATE(NAME, T)                            \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows,                                                                             \
        int32_t class_extent,                                                                       \
        int64_t row_stride_input,                                                                   \
        int64_t row_stride_target,                                                                  \
        int32_t reduction_mode,                                                                     \
        const void* input, const void* target, void* out,                                           \
        void* workspace, size_t workspace_bytes,                                                    \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0) return 2;                                              \
        if (n_rows == 0) return 0;                                                                  \
        if (input == nullptr || target == nullptr || out == nullptr) return 2;                      \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        if (reduction_mode == 0) {                                                                  \
            baracuda::loss::cross_entropy_soft_per_row_kernel<T><<<blocks, kBlock, 0, stream>>>(    \
                static_cast<const T*>(input), static_cast<const T*>(target),                        \
                static_cast<T*>(out), n_rows, class_extent,                                         \
                row_stride_input, row_stride_target);                                               \
            return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                     \
        }                                                                                           \
        size_t need = (size_t)n_rows * sizeof(T);                                                   \
        if (workspace == nullptr || workspace_bytes < need) return 4;                               \
        T* term = static_cast<T*>(workspace);                                                       \
        baracuda::loss::cross_entropy_soft_per_row_kernel<T><<<blocks, kBlock, 0, stream>>>(        \
            static_cast<const T*>(input), static_cast<const T*>(target),                            \
            term, n_rows, class_extent, row_stride_input, row_stride_target);                       \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        float denom_inv = (reduction_mode == 1) ? (1.0f / (float)n_rows) : 1.0f;                    \
        baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,        \
            stream>>>(term, static_cast<T*>(out), n_rows, denom_inv);                               \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// Soft-target CrossEntropy BW.
#define BARACUDA_KERNELS_LOSS_CROSS_ENTROPY_SOFT_BW_INSTANTIATE(NAME, T)                            \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows,                                                                             \
        int32_t class_extent,                                                                       \
        int64_t row_stride_input,                                                                   \
        int64_t row_stride_target,                                                                  \
        int32_t reduction_mode,                                                                     \
        float scale_scalar,                                                                         \
        const void* input, const void* target, const void* dy, void* dinput,                        \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0) return 2;                                              \
        if (n_rows == 0) return 0;                                                                  \
        if (input == nullptr || target == nullptr || dy == nullptr || dinput == nullptr) return 2;  \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        constexpr int64_t kMaxBlocks = 65535;                                                       \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);           \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::cross_entropy_soft_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(       \
            static_cast<const T*>(input), static_cast<const T*>(target),                            \
            static_cast<const T*>(dy), static_cast<T*>(dinput),                                     \
            n_rows, class_extent, row_stride_input, row_stride_target,                              \
            reduction_mode, scale_scalar);                                                          \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// =============================================================================
// Milestone 5.3 — Tier-2 margin / embedding losses.
//
// All ops follow the deterministic two-pass design: per-cell / per-row kernel
// computes the term, writes to a workspace buffer, then a single-block tree
// reduction collapses to a scalar (Mean / Sum) — or the term kernel writes
// directly to `out` for None mode.
//
// Wired ops:
//   - MarginRankingLoss      `y = mean(max(0, -t · (x1 - x2) + margin))`
//   - HingeEmbeddingLoss     `y = mean(input if t==1 else max(0, margin - input))`
//   - CosineEmbeddingLoss    `1 - cos(x1, x2)` if t==1 else `max(0, cos(x1, x2) - margin)`
//   - TripletMarginLoss      `max(0, ||a-p||_p - ||a-n||_p + margin)`
//   - MultiMarginLoss        `Σ_{j != t_i} max(0, margin - input[i, t_i] + input[i, j])^p / C`
//   - MultilabelMarginLoss   sum over pos j, neg c of max(0, 1 - input[j] + input[c]) / C
//   - MultilabelSoftMarginLoss `-mean_c(target·log(σ(x)) + (1-target)·log(1-σ(x)))`
// =============================================================================

namespace baracuda { namespace loss {

// -----------------------------------------------------------------------------
// MarginRanking per-cell: term[i] = max(0, -t[i] · (x1[i] - x2[i]) + margin)
// -----------------------------------------------------------------------------

template <typename T>
__global__ void margin_ranking_per_cell_kernel(
    const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ t,
    T* __restrict__ term, int64_t numel, float margin)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float a = load_as_acc<T>(x1[i]);
        float b = load_as_acc<T>(x2[i]);
        float tv = load_as_acc<T>(t[i]);
        float v  = -tv * (a - b) + margin;
        term[i] = store_from_acc<T>(v > 0.0f ? v : 0.0f);
    }
}

template <>
__global__ void margin_ranking_per_cell_kernel<double>(
    const double* __restrict__ x1, const double* __restrict__ x2, const double* __restrict__ t,
    double* __restrict__ term, int64_t numel, float margin_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double margin = (double)margin_f;
    for (int64_t i = tid; i < numel; i += step) {
        double v = -t[i] * (x1[i] - x2[i]) + margin;
        term[i] = v > 0.0 ? v : 0.0;
    }
}

// MarginRanking BW: dx1[i] = -t[i]/N · dy if loss > 0, dx2 = -dx1.
template <typename T>
__global__ void margin_ranking_backward_kernel(
    const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ t,
    const T* __restrict__ dy, T* __restrict__ dx1, T* __restrict__ dx2,
    int64_t numel, int32_t reduction_mode, float inv_n_or_one, float margin)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < numel; i += step) {
        float a = load_as_acc<T>(x1[i]);
        float b = load_as_acc<T>(x2[i]);
        float tv = load_as_acc<T>(t[i]);
        float loss = -tv * (a - b) + margin;
        float sc = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        float g = (loss > 0.0f) ? (-tv * sc) : 0.0f;
        if (dx1 != nullptr) dx1[i] = store_from_acc<T>(g);
        if (dx2 != nullptr) dx2[i] = store_from_acc<T>(-g);
    }
}

template <>
__global__ void margin_ranking_backward_kernel<double>(
    const double* __restrict__ x1, const double* __restrict__ x2, const double* __restrict__ t,
    const double* __restrict__ dy, double* __restrict__ dx1, double* __restrict__ dx2,
    int64_t numel, int32_t reduction_mode, float inv_n_or_one, float margin_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) dy0_scaled = dy[0] / (double)numel;
    else if (reduction_mode == 2) dy0_scaled = dy[0];
    (void)inv_n_or_one;
    double margin = (double)margin_f;
    for (int64_t i = tid; i < numel; i += step) {
        double loss = -t[i] * (x1[i] - x2[i]) + margin;
        double sc = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        double g = (loss > 0.0) ? (-t[i] * sc) : 0.0;
        if (dx1 != nullptr) dx1[i] = g;
        if (dx2 != nullptr) dx2[i] = -g;
    }
}

// -----------------------------------------------------------------------------
// HingeEmbedding per-cell: term[i] = input[i] if t[i]==1 else max(0, margin - input[i])
// target is i64.
// -----------------------------------------------------------------------------

template <typename T>
__global__ void hinge_embedding_per_cell_kernel(
    const T* __restrict__ input, const int64_t* __restrict__ t,
    T* __restrict__ term, int64_t numel, float margin)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float x = load_as_acc<T>(input[i]);
        int64_t ti = t[i];
        float v;
        if (ti == 1) v = x;
        else { float h = margin - x; v = h > 0.0f ? h : 0.0f; }
        term[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void hinge_embedding_per_cell_kernel<double>(
    const double* __restrict__ input, const int64_t* __restrict__ t,
    double* __restrict__ term, int64_t numel, float margin_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double margin = (double)margin_f;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t ti = t[i];
        double v;
        if (ti == 1) v = input[i];
        else { double h = margin - input[i]; v = h > 0.0 ? h : 0.0; }
        term[i] = v;
    }
}

// HingeEmbedding BW: dinput[i] = sc if t==1, else (-sc if margin > input else 0).
template <typename T>
__global__ void hinge_embedding_backward_kernel(
    const T* __restrict__ input, const int64_t* __restrict__ t,
    const T* __restrict__ dy, T* __restrict__ dinput,
    int64_t numel, int32_t reduction_mode, float inv_n_or_one, float margin)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t i = tid; i < numel; i += step) {
        float x = load_as_acc<T>(input[i]);
        int64_t ti = t[i];
        float sc = (reduction_mode == 0) ? load_as_acc<T>(dy[i]) : dy0_scaled;
        float g;
        if (ti == 1) g = sc;
        else g = (margin > x) ? -sc : 0.0f;
        dinput[i] = store_from_acc<T>(g);
    }
}

template <>
__global__ void hinge_embedding_backward_kernel<double>(
    const double* __restrict__ input, const int64_t* __restrict__ t,
    const double* __restrict__ dy, double* __restrict__ dinput,
    int64_t numel, int32_t reduction_mode, float inv_n_or_one, float margin_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) dy0_scaled = dy[0] / (double)numel;
    else if (reduction_mode == 2) dy0_scaled = dy[0];
    (void)inv_n_or_one;
    double margin = (double)margin_f;
    for (int64_t i = tid; i < numel; i += step) {
        int64_t ti = t[i];
        double sc = (reduction_mode == 0) ? dy[i] : dy0_scaled;
        double g;
        if (ti == 1) g = sc;
        else g = (margin > input[i]) ? -sc : 0.0;
        dinput[i] = g;
    }
}

// -----------------------------------------------------------------------------
// CosineEmbedding — per-row, input [N, D], target T scalar ±1 per row.
// Per row n: cs = dot(x1, x2) / (||x1|| * ||x2||).
// term = (t==1) ? (1 - cs) : max(0, cs - margin).
// One thread per row (D small in smoke tests; matches our other per-row patterns).
// -----------------------------------------------------------------------------

template <typename T>
__global__ void cosine_embedding_per_row_kernel(
    const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ t,
    T* __restrict__ term, int64_t n_rows, int32_t d_extent,
    int64_t row_stride_x, float margin)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t r = tid; r < n_rows; r += step) {
        const T* r1 = x1 + r * row_stride_x;
        const T* r2 = x2 + r * row_stride_x;
        float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
        for (int j = 0; j < d_extent; ++j) {
            float a = load_as_acc<T>(r1[j]);
            float b = load_as_acc<T>(r2[j]);
            dot += a * b; n1 += a * a; n2 += b * b;
        }
        float denom = sqrtf(n1) * sqrtf(n2);
        if (denom < 1e-12f) denom = 1e-12f;
        float cs = dot / denom;
        float tv = load_as_acc<T>(t[r]);
        float v;
        if (tv > 0.0f) v = 1.0f - cs;
        else { float h = cs - margin; v = h > 0.0f ? h : 0.0f; }
        term[r] = store_from_acc<T>(v);
    }
}

template <>
__global__ void cosine_embedding_per_row_kernel<double>(
    const double* __restrict__ x1, const double* __restrict__ x2, const double* __restrict__ t,
    double* __restrict__ term, int64_t n_rows, int32_t d_extent,
    int64_t row_stride_x, float margin_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double margin = (double)margin_f;
    for (int64_t r = tid; r < n_rows; r += step) {
        const double* r1 = x1 + r * row_stride_x;
        const double* r2 = x2 + r * row_stride_x;
        double dot = 0.0, n1 = 0.0, n2 = 0.0;
        for (int j = 0; j < d_extent; ++j) {
            dot += r1[j] * r2[j]; n1 += r1[j] * r1[j]; n2 += r2[j] * r2[j];
        }
        double denom = sqrt(n1) * sqrt(n2);
        if (denom < 1e-30) denom = 1e-30;
        double cs = dot / denom;
        double tv = t[r];
        double v;
        if (tv > 0.0) v = 1.0 - cs;
        else { double h = cs - margin; v = h > 0.0 ? h : 0.0; }
        term[r] = v;
    }
}

// CosineEmbedding BW.
//   per-row: cs = dot/(n1·n2); n1 = ||x1||, n2 = ||x2||.
//   d(cs)/d(x1) = (x2 / (n1*n2)) - cs · (x1 / n1²)
//   d(cs)/d(x2) = (x1 / (n1*n2)) - cs · (x2 / n2²)
//   d(term)/d(cs) = -1 if t==1, else (cs > margin ? 1 : 0).
//   dx{1,2} = d(term)/d(cs) · d(cs)/d(x{1,2}) · sc
template <typename T>
__global__ void cosine_embedding_backward_kernel(
    const T* __restrict__ x1, const T* __restrict__ x2, const T* __restrict__ t,
    const T* __restrict__ dy, T* __restrict__ dx1, T* __restrict__ dx2,
    int64_t n_rows, int32_t d_extent, int64_t row_stride_x,
    int32_t reduction_mode, float inv_n_or_one, float margin)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t r = tid; r < n_rows; r += step) {
        const T* r1 = x1 + r * row_stride_x;
        const T* r2 = x2 + r * row_stride_x;
        T* d1 = (dx1 != nullptr) ? dx1 + r * row_stride_x : nullptr;
        T* d2 = (dx2 != nullptr) ? dx2 + r * row_stride_x : nullptr;
        float dot = 0.0f, n1 = 0.0f, n2 = 0.0f;
        for (int j = 0; j < d_extent; ++j) {
            float a = load_as_acc<T>(r1[j]);
            float b = load_as_acc<T>(r2[j]);
            dot += a * b; n1 += a * a; n2 += b * b;
        }
        float s1 = sqrtf(n1), s2 = sqrtf(n2);
        float denom = s1 * s2; if (denom < 1e-12f) denom = 1e-12f;
        float cs = dot / denom;
        float tv = load_as_acc<T>(t[r]);
        float dcs;
        if (tv > 0.0f) dcs = -1.0f;
        else dcs = (cs > margin) ? 1.0f : 0.0f;
        float sc = (reduction_mode == 0) ? load_as_acc<T>(dy[r]) : dy0_scaled;
        float coef = dcs * sc;
        float inv_n1n2 = 1.0f / denom;
        float inv_n1sq = (n1 > 1e-24f) ? (1.0f / n1) : 0.0f;
        float inv_n2sq = (n2 > 1e-24f) ? (1.0f / n2) : 0.0f;
        for (int j = 0; j < d_extent; ++j) {
            float a = load_as_acc<T>(r1[j]);
            float b = load_as_acc<T>(r2[j]);
            float g1 = coef * (b * inv_n1n2 - cs * a * inv_n1sq);
            float g2 = coef * (a * inv_n1n2 - cs * b * inv_n2sq);
            if (d1 != nullptr) d1[j] = store_from_acc<T>(g1);
            if (d2 != nullptr) d2[j] = store_from_acc<T>(g2);
        }
    }
}

template <>
__global__ void cosine_embedding_backward_kernel<double>(
    const double* __restrict__ x1, const double* __restrict__ x2, const double* __restrict__ t,
    const double* __restrict__ dy, double* __restrict__ dx1, double* __restrict__ dx2,
    int64_t n_rows, int32_t d_extent, int64_t row_stride_x,
    int32_t reduction_mode, float inv_n_or_one, float margin_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) dy0_scaled = dy[0] / (double)n_rows;
    else if (reduction_mode == 2) dy0_scaled = dy[0];
    (void)inv_n_or_one;
    double margin = (double)margin_f;
    for (int64_t r = tid; r < n_rows; r += step) {
        const double* r1 = x1 + r * row_stride_x;
        const double* r2 = x2 + r * row_stride_x;
        double* d1 = (dx1 != nullptr) ? dx1 + r * row_stride_x : nullptr;
        double* d2 = (dx2 != nullptr) ? dx2 + r * row_stride_x : nullptr;
        double dot = 0.0, n1 = 0.0, n2 = 0.0;
        for (int j = 0; j < d_extent; ++j) {
            dot += r1[j] * r2[j]; n1 += r1[j] * r1[j]; n2 += r2[j] * r2[j];
        }
        double s1 = sqrt(n1), s2 = sqrt(n2);
        double denom = s1 * s2; if (denom < 1e-30) denom = 1e-30;
        double cs = dot / denom;
        double tv = t[r];
        double dcs = (tv > 0.0) ? -1.0 : ((cs > margin) ? 1.0 : 0.0);
        double sc = (reduction_mode == 0) ? dy[r] : dy0_scaled;
        double coef = dcs * sc;
        double inv_n1n2 = 1.0 / denom;
        double inv_n1sq = (n1 > 1e-60) ? (1.0 / n1) : 0.0;
        double inv_n2sq = (n2 > 1e-60) ? (1.0 / n2) : 0.0;
        for (int j = 0; j < d_extent; ++j) {
            double g1 = coef * (r2[j] * inv_n1n2 - cs * r1[j] * inv_n1sq);
            double g2 = coef * (r1[j] * inv_n1n2 - cs * r2[j] * inv_n2sq);
            if (d1 != nullptr) d1[j] = g1;
            if (d2 != nullptr) d2[j] = g2;
        }
    }
}

// -----------------------------------------------------------------------------
// TripletMargin — per-row, input [N, D]. Per row:
//   pd = (Σ |a-p|^p)^(1/p), nd = (Σ |a-n|^p)^(1/p).
//   loss = max(0, pd - nd + margin).
// BW (active rows only):
//   dpd/dp_j  = -|a_j - p_j|^(p-1) · sign(a_j-p_j) / pd^(p-1)  [note: derivative of L_p norm]
//   For p=2: dpd/dp_j = -(a_j - p_j) / pd; dpd/da_j = (a_j - p_j) / pd.
//   Same for nd / negative; but sign flipped because loss has -nd:
//     dloss/dn_j = +(a_j - n_j) / nd
//     dloss/da_j = (a_j - p_j) / pd - (a_j - n_j) / nd
// -----------------------------------------------------------------------------

template <typename T>
__global__ void triplet_margin_per_row_kernel(
    const T* __restrict__ a, const T* __restrict__ pp, const T* __restrict__ nn,
    T* __restrict__ term, int64_t n_rows, int32_t d_extent,
    int64_t row_stride, float margin, float p_norm)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t r = tid; r < n_rows; r += step) {
        const T* ra = a + r * row_stride;
        const T* rp = pp + r * row_stride;
        const T* rn = nn + r * row_stride;
        float sp = 0.0f, sn = 0.0f;
        for (int j = 0; j < d_extent; ++j) {
            float d1 = load_as_acc<T>(ra[j]) - load_as_acc<T>(rp[j]);
            float d2 = load_as_acc<T>(ra[j]) - load_as_acc<T>(rn[j]);
            sp += powf(fabsf(d1), p_norm);
            sn += powf(fabsf(d2), p_norm);
        }
        float pd = powf(sp, 1.0f / p_norm);
        float nd = powf(sn, 1.0f / p_norm);
        float v = pd - nd + margin;
        term[r] = store_from_acc<T>(v > 0.0f ? v : 0.0f);
    }
}

template <>
__global__ void triplet_margin_per_row_kernel<double>(
    const double* __restrict__ a, const double* __restrict__ pp, const double* __restrict__ nn,
    double* __restrict__ term, int64_t n_rows, int32_t d_extent,
    int64_t row_stride, float margin_f, float p_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double margin = (double)margin_f;
    double p_norm = (double)p_f;
    for (int64_t r = tid; r < n_rows; r += step) {
        const double* ra = a + r * row_stride;
        const double* rp = pp + r * row_stride;
        const double* rn = nn + r * row_stride;
        double sp = 0.0, sn = 0.0;
        for (int j = 0; j < d_extent; ++j) {
            double d1 = ra[j] - rp[j];
            double d2 = ra[j] - rn[j];
            sp += pow(fabs(d1), p_norm);
            sn += pow(fabs(d2), p_norm);
        }
        double pd = pow(sp, 1.0 / p_norm);
        double nd = pow(sn, 1.0 / p_norm);
        double v = pd - nd + margin;
        term[r] = v > 0.0 ? v : 0.0;
    }
}

// TripletMargin BW. For p=2 (the common case), per-row:
//   if loss > 0: da = (a-p)/pd - (a-n)/nd ; dp = -(a-p)/pd ; dn = (a-n)/nd
//   else all-zero.
// For general p:
//   d(pd)/d(a_j) = |a_j-p_j|^(p-1) · sign(a_j-p_j) / pd^(p-1)
//   = pow(|.|, p-1) · sign / pow(pd, p-1)
template <typename T>
__global__ void triplet_margin_backward_kernel(
    const T* __restrict__ a, const T* __restrict__ pp, const T* __restrict__ nn,
    const T* __restrict__ dy,
    T* __restrict__ da, T* __restrict__ dp, T* __restrict__ dn,
    int64_t n_rows, int32_t d_extent, int64_t row_stride,
    int32_t reduction_mode, float inv_n_or_one, float margin, float p_norm)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t r = tid; r < n_rows; r += step) {
        const T* ra = a + r * row_stride;
        const T* rp = pp + r * row_stride;
        const T* rn = nn + r * row_stride;
        T* xa = (da != nullptr) ? da + r * row_stride : nullptr;
        T* xp = (dp != nullptr) ? dp + r * row_stride : nullptr;
        T* xn = (dn != nullptr) ? dn + r * row_stride : nullptr;
        float sp = 0.0f, sn = 0.0f;
        for (int j = 0; j < d_extent; ++j) {
            float d1 = load_as_acc<T>(ra[j]) - load_as_acc<T>(rp[j]);
            float d2 = load_as_acc<T>(ra[j]) - load_as_acc<T>(rn[j]);
            sp += powf(fabsf(d1), p_norm);
            sn += powf(fabsf(d2), p_norm);
        }
        float pd = powf(sp, 1.0f / p_norm);
        float nd = powf(sn, 1.0f / p_norm);
        float loss = pd - nd + margin;
        bool active = (loss > 0.0f);
        float sc = (reduction_mode == 0) ? load_as_acc<T>(dy[r]) : dy0_scaled;
        float pd_pm1 = powf(pd, p_norm - 1.0f); if (pd_pm1 < 1e-12f) pd_pm1 = 1e-12f;
        float nd_pm1 = powf(nd, p_norm - 1.0f); if (nd_pm1 < 1e-12f) nd_pm1 = 1e-12f;
        for (int j = 0; j < d_extent; ++j) {
            float d1 = load_as_acc<T>(ra[j]) - load_as_acc<T>(rp[j]);
            float d2 = load_as_acc<T>(ra[j]) - load_as_acc<T>(rn[j]);
            float sgn1 = (d1 > 0.0f) ? 1.0f : ((d1 < 0.0f) ? -1.0f : 0.0f);
            float sgn2 = (d2 > 0.0f) ? 1.0f : ((d2 < 0.0f) ? -1.0f : 0.0f);
            float pa_pj = powf(fabsf(d1), p_norm - 1.0f) * sgn1 / pd_pm1; // d(pd)/d(a_j)
            float pa_nj = powf(fabsf(d2), p_norm - 1.0f) * sgn2 / nd_pm1; // d(nd)/d(a_j)
            float g_a = active ? (pa_pj - pa_nj) * sc : 0.0f;
            float g_p = active ? (-pa_pj) * sc : 0.0f;
            float g_n = active ? ( pa_nj) * sc : 0.0f;
            if (xa != nullptr) xa[j] = store_from_acc<T>(g_a);
            if (xp != nullptr) xp[j] = store_from_acc<T>(g_p);
            if (xn != nullptr) xn[j] = store_from_acc<T>(g_n);
        }
    }
}

template <>
__global__ void triplet_margin_backward_kernel<double>(
    const double* __restrict__ a, const double* __restrict__ pp, const double* __restrict__ nn,
    const double* __restrict__ dy,
    double* __restrict__ da, double* __restrict__ dp, double* __restrict__ dn,
    int64_t n_rows, int32_t d_extent, int64_t row_stride,
    int32_t reduction_mode, float inv_n_or_one, float margin_f, float p_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) dy0_scaled = dy[0] / (double)n_rows;
    else if (reduction_mode == 2) dy0_scaled = dy[0];
    (void)inv_n_or_one;
    double margin = (double)margin_f;
    double p_norm = (double)p_f;
    for (int64_t r = tid; r < n_rows; r += step) {
        const double* ra = a + r * row_stride;
        const double* rp = pp + r * row_stride;
        const double* rn = nn + r * row_stride;
        double* xa = (da != nullptr) ? da + r * row_stride : nullptr;
        double* xp = (dp != nullptr) ? dp + r * row_stride : nullptr;
        double* xn = (dn != nullptr) ? dn + r * row_stride : nullptr;
        double sp = 0.0, sn = 0.0;
        for (int j = 0; j < d_extent; ++j) {
            sp += pow(fabs(ra[j] - rp[j]), p_norm);
            sn += pow(fabs(ra[j] - rn[j]), p_norm);
        }
        double pd = pow(sp, 1.0 / p_norm);
        double nd = pow(sn, 1.0 / p_norm);
        double loss = pd - nd + margin;
        bool active = (loss > 0.0);
        double sc = (reduction_mode == 0) ? dy[r] : dy0_scaled;
        double pd_pm1 = pow(pd, p_norm - 1.0); if (pd_pm1 < 1e-30) pd_pm1 = 1e-30;
        double nd_pm1 = pow(nd, p_norm - 1.0); if (nd_pm1 < 1e-30) nd_pm1 = 1e-30;
        for (int j = 0; j < d_extent; ++j) {
            double d1 = ra[j] - rp[j];
            double d2 = ra[j] - rn[j];
            double sgn1 = (d1 > 0.0) ? 1.0 : ((d1 < 0.0) ? -1.0 : 0.0);
            double sgn2 = (d2 > 0.0) ? 1.0 : ((d2 < 0.0) ? -1.0 : 0.0);
            double pa_pj = pow(fabs(d1), p_norm - 1.0) * sgn1 / pd_pm1;
            double pa_nj = pow(fabs(d2), p_norm - 1.0) * sgn2 / nd_pm1;
            double g_a = active ? (pa_pj - pa_nj) * sc : 0.0;
            double g_p = active ? (-pa_pj) * sc : 0.0;
            double g_n = active ? ( pa_nj) * sc : 0.0;
            if (xa != nullptr) xa[j] = g_a;
            if (xp != nullptr) xp[j] = g_p;
            if (xn != nullptr) xn[j] = g_n;
        }
    }
}

// -----------------------------------------------------------------------------
// MultiMargin — input [N, C], target [N] i64.
//   per_row_n = (Σ_{j != t} max(0, margin - input[n, t] + input[n, j])^p) / C
// -----------------------------------------------------------------------------

template <typename T>
__global__ void multi_margin_per_row_kernel(
    const T* __restrict__ input, const int64_t* __restrict__ t,
    T* __restrict__ term, int64_t n_rows, int32_t class_extent,
    int64_t row_stride, float margin, float p_norm)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t r = tid; r < n_rows; r += step) {
        const T* row = input + r * row_stride;
        int64_t ti = t[r];
        if (ti < 0 || ti >= (int64_t)class_extent) { term[r] = store_from_acc<T>(0.0f); continue; }
        float xt = load_as_acc<T>(row[ti]);
        float acc = 0.0f;
        for (int j = 0; j < class_extent; ++j) {
            if (j == (int)ti) continue;
            float xj = load_as_acc<T>(row[j]);
            float h = margin - xt + xj;
            if (h > 0.0f) acc += (p_norm == 1.0f) ? h : powf(h, p_norm);
        }
        acc /= (float)class_extent;
        term[r] = store_from_acc<T>(acc);
    }
}

template <>
__global__ void multi_margin_per_row_kernel<double>(
    const double* __restrict__ input, const int64_t* __restrict__ t,
    double* __restrict__ term, int64_t n_rows, int32_t class_extent,
    int64_t row_stride, float margin_f, float p_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double margin = (double)margin_f;
    double p_norm = (double)p_f;
    for (int64_t r = tid; r < n_rows; r += step) {
        const double* row = input + r * row_stride;
        int64_t ti = t[r];
        if (ti < 0 || ti >= (int64_t)class_extent) { term[r] = 0.0; continue; }
        double xt = row[ti];
        double acc = 0.0;
        for (int j = 0; j < class_extent; ++j) {
            if (j == (int)ti) continue;
            double h = margin - xt + row[j];
            if (h > 0.0) acc += (p_norm == 1.0) ? h : pow(h, p_norm);
        }
        acc /= (double)class_extent;
        term[r] = acc;
    }
}

// MultiMargin BW:
//   For active j (h > 0): contribution to dinput[n, t] -= p · h^(p-1) / C · sc;
//                          contribution to dinput[n, j] += p · h^(p-1) / C · sc.
// We zero-init dinput first.
template <typename T>
__global__ void multi_margin_backward_kernel(
    const T* __restrict__ input, const int64_t* __restrict__ t,
    const T* __restrict__ dy, T* __restrict__ dinput,
    int64_t n_rows, int32_t class_extent, int64_t row_stride,
    int32_t reduction_mode, float inv_n_or_one, float margin, float p_norm)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t r = tid; r < n_rows; r += step) {
        const T* row = input + r * row_stride;
        T* drow = dinput + r * row_stride;
        int64_t ti = t[r];
        if (ti < 0 || ti >= (int64_t)class_extent) {
            for (int j = 0; j < class_extent; ++j) drow[j] = store_from_acc<T>(0.0f);
            continue;
        }
        float xt = load_as_acc<T>(row[ti]);
        float sc = (reduction_mode == 0) ? load_as_acc<T>(dy[r]) : dy0_scaled;
        float coef = sc / (float)class_extent;
        float acc_t = 0.0f;
        for (int j = 0; j < class_extent; ++j) drow[j] = store_from_acc<T>(0.0f);
        for (int j = 0; j < class_extent; ++j) {
            if (j == (int)ti) continue;
            float h = margin - xt + load_as_acc<T>(row[j]);
            if (h > 0.0f) {
                float grad_h = (p_norm == 1.0f) ? 1.0f : (p_norm * powf(h, p_norm - 1.0f));
                drow[j] = store_from_acc<T>(grad_h * coef);
                acc_t += grad_h;
            }
        }
        drow[ti] = store_from_acc<T>(-acc_t * coef);
    }
}

template <>
__global__ void multi_margin_backward_kernel<double>(
    const double* __restrict__ input, const int64_t* __restrict__ t,
    const double* __restrict__ dy, double* __restrict__ dinput,
    int64_t n_rows, int32_t class_extent, int64_t row_stride,
    int32_t reduction_mode, float inv_n_or_one, float margin_f, float p_f)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) dy0_scaled = dy[0] / (double)n_rows;
    else if (reduction_mode == 2) dy0_scaled = dy[0];
    (void)inv_n_or_one;
    double margin = (double)margin_f;
    double p_norm = (double)p_f;
    for (int64_t r = tid; r < n_rows; r += step) {
        const double* row = input + r * row_stride;
        double* drow = dinput + r * row_stride;
        int64_t ti = t[r];
        if (ti < 0 || ti >= (int64_t)class_extent) {
            for (int j = 0; j < class_extent; ++j) drow[j] = 0.0;
            continue;
        }
        double xt = row[ti];
        double sc = (reduction_mode == 0) ? dy[r] : dy0_scaled;
        double coef = sc / (double)class_extent;
        double acc_t = 0.0;
        for (int j = 0; j < class_extent; ++j) drow[j] = 0.0;
        for (int j = 0; j < class_extent; ++j) {
            if (j == (int)ti) continue;
            double h = margin - xt + row[j];
            if (h > 0.0) {
                double grad_h = (p_norm == 1.0) ? 1.0 : (p_norm * pow(h, p_norm - 1.0));
                drow[j] = grad_h * coef;
                acc_t += grad_h;
            }
        }
        drow[ti] = -acc_t * coef;
    }
}

// -----------------------------------------------------------------------------
// MultilabelMargin — input [N, C], target [N, C] i64. Target lists positive
// class indices, then -1 sentinel for unused slots.
//   per row n:
//     let pos = {target[n, k] : k < first_neg(n)}    (positive classes)
//     loss_n = (Σ_{j ∈ pos} Σ_{i ∉ pos} max(0, 1 - input[n, j] + input[n, i])) / C
// PyTorch uses fixed margin = 1.
// -----------------------------------------------------------------------------

template <typename T>
__global__ void multilabel_margin_per_row_kernel(
    const T* __restrict__ input, const int64_t* __restrict__ tgt,
    T* __restrict__ term, int64_t n_rows, int32_t class_extent,
    int64_t row_stride_in, int64_t row_stride_tgt)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t r = tid; r < n_rows; r += step) {
        const T* row = input + r * row_stride_in;
        const int64_t* trow = tgt + r * row_stride_tgt;
        // Decode positives: traverse trow until -1.
        // For each k where trow[k] >= 0, treat trow[k] as positive class j.
        // For each j: for each i in [0, C) not in pos-set: accumulate max(0, 1 - x[j] + x[i]).
        // Naive O(C^2) but fine for small C in smoke tests.
        float acc = 0.0f;
        for (int k = 0; k < class_extent; ++k) {
            int64_t j = trow[k];
            if (j < 0) break;
            if (j >= (int64_t)class_extent) continue;
            float xj = load_as_acc<T>(row[j]);
            for (int i = 0; i < class_extent; ++i) {
                // Skip if i is in positive list.
                bool in_pos = false;
                for (int kk = 0; kk < class_extent; ++kk) {
                    int64_t pp_ = trow[kk];
                    if (pp_ < 0) break;
                    if (pp_ == (int64_t)i) { in_pos = true; break; }
                }
                if (in_pos) continue;
                float h = 1.0f - xj + load_as_acc<T>(row[i]);
                if (h > 0.0f) acc += h;
            }
        }
        acc /= (float)class_extent;
        term[r] = store_from_acc<T>(acc);
    }
}

template <>
__global__ void multilabel_margin_per_row_kernel<double>(
    const double* __restrict__ input, const int64_t* __restrict__ tgt,
    double* __restrict__ term, int64_t n_rows, int32_t class_extent,
    int64_t row_stride_in, int64_t row_stride_tgt)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t r = tid; r < n_rows; r += step) {
        const double* row = input + r * row_stride_in;
        const int64_t* trow = tgt + r * row_stride_tgt;
        double acc = 0.0;
        for (int k = 0; k < class_extent; ++k) {
            int64_t j = trow[k];
            if (j < 0) break;
            if (j >= (int64_t)class_extent) continue;
            double xj = row[j];
            for (int i = 0; i < class_extent; ++i) {
                bool in_pos = false;
                for (int kk = 0; kk < class_extent; ++kk) {
                    int64_t pp_ = trow[kk];
                    if (pp_ < 0) break;
                    if (pp_ == (int64_t)i) { in_pos = true; break; }
                }
                if (in_pos) continue;
                double h = 1.0 - xj + row[i];
                if (h > 0.0) acc += h;
            }
        }
        acc /= (double)class_extent;
        term[r] = acc;
    }
}

// MultilabelMargin BW: for each (j ∈ pos, i ∉ pos) with h > 0:
//   dinput[j] -= 1 / C · sc;  dinput[i] += 1 / C · sc.
template <typename T>
__global__ void multilabel_margin_backward_kernel(
    const T* __restrict__ input, const int64_t* __restrict__ tgt,
    const T* __restrict__ dy, T* __restrict__ dinput,
    int64_t n_rows, int32_t class_extent, int64_t row_stride_in, int64_t row_stride_tgt,
    int32_t reduction_mode, float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t r = tid; r < n_rows; r += step) {
        const T* row = input + r * row_stride_in;
        T* drow = dinput + r * row_stride_in;
        const int64_t* trow = tgt + r * row_stride_tgt;
        float sc = (reduction_mode == 0) ? load_as_acc<T>(dy[r]) : dy0_scaled;
        float coef = sc / (float)class_extent;
        // Compute partials in f32 to avoid serialized writes.
        // Use thread-local accumulators size class_extent? We can't easily;
        // class_extent is runtime. Use the per-cell store approach by
        // looping twice: clear then accumulate. Since one thread per row
        // here we can use a small stack array? Risky for large C. We do
        // simple in-place accumulation: zero drow first, then iterate.
        for (int j = 0; j < class_extent; ++j) drow[j] = store_from_acc<T>(0.0f);
        for (int k = 0; k < class_extent; ++k) {
            int64_t j = trow[k];
            if (j < 0) break;
            if (j >= (int64_t)class_extent) continue;
            float xj = load_as_acc<T>(row[j]);
            for (int i = 0; i < class_extent; ++i) {
                bool in_pos = false;
                for (int kk = 0; kk < class_extent; ++kk) {
                    int64_t pp_ = trow[kk];
                    if (pp_ < 0) break;
                    if (pp_ == (int64_t)i) { in_pos = true; break; }
                }
                if (in_pos) continue;
                float h = 1.0f - xj + load_as_acc<T>(row[i]);
                if (h > 0.0f) {
                    // Accumulate via load + store.
                    float cur_j = load_as_acc<T>(drow[j]);
                    float cur_i = load_as_acc<T>(drow[i]);
                    drow[j] = store_from_acc<T>(cur_j - coef);
                    drow[i] = store_from_acc<T>(cur_i + coef);
                }
            }
        }
    }
}

template <>
__global__ void multilabel_margin_backward_kernel<double>(
    const double* __restrict__ input, const int64_t* __restrict__ tgt,
    const double* __restrict__ dy, double* __restrict__ dinput,
    int64_t n_rows, int32_t class_extent, int64_t row_stride_in, int64_t row_stride_tgt,
    int32_t reduction_mode, float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) dy0_scaled = dy[0] / (double)n_rows;
    else if (reduction_mode == 2) dy0_scaled = dy[0];
    (void)inv_n_or_one;
    for (int64_t r = tid; r < n_rows; r += step) {
        const double* row = input + r * row_stride_in;
        double* drow = dinput + r * row_stride_in;
        const int64_t* trow = tgt + r * row_stride_tgt;
        double sc = (reduction_mode == 0) ? dy[r] : dy0_scaled;
        double coef = sc / (double)class_extent;
        for (int j = 0; j < class_extent; ++j) drow[j] = 0.0;
        for (int k = 0; k < class_extent; ++k) {
            int64_t j = trow[k];
            if (j < 0) break;
            if (j >= (int64_t)class_extent) continue;
            double xj = row[j];
            for (int i = 0; i < class_extent; ++i) {
                bool in_pos = false;
                for (int kk = 0; kk < class_extent; ++kk) {
                    int64_t pp_ = trow[kk];
                    if (pp_ < 0) break;
                    if (pp_ == (int64_t)i) { in_pos = true; break; }
                }
                if (in_pos) continue;
                double h = 1.0 - xj + row[i];
                if (h > 0.0) {
                    drow[j] -= coef;
                    drow[i] += coef;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// MultilabelSoftMargin — input [N, C], target [N, C] T.
//   per row n: loss_n = -mean_c(target·log(σ(x)) + (1-target)·log(1-σ(x)))
// Numerically stable: -log σ(x) = log(1+exp(-x)) when x>=0, else -x + log(1+exp(x));
// -log(1-σ(x)) = log(1+exp(x)) when x<=0, else x + log(1+exp(-x)).
// In general: -y·log(σ(x)) - (1-y)·log(1-σ(x)) = max(x,0) - x·y + log(1+exp(-|x|))
// (same as BCEWithLogits per element). Then mean over C.
// -----------------------------------------------------------------------------

template <typename T>
__global__ void multilabel_soft_margin_per_row_kernel(
    const T* __restrict__ input, const T* __restrict__ tgt,
    T* __restrict__ term, int64_t n_rows, int32_t class_extent,
    int64_t row_stride_in, int64_t row_stride_tgt)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t r = tid; r < n_rows; r += step) {
        const T* row = input + r * row_stride_in;
        const T* trow = tgt + r * row_stride_tgt;
        float acc = 0.0f;
        for (int j = 0; j < class_extent; ++j) {
            float x = load_as_acc<T>(row[j]);
            float y = load_as_acc<T>(trow[j]);
            float ax = fabsf(x);
            float mx = x > 0.0f ? x : 0.0f;
            acc += mx - x * y + log1pf(expf(-ax));
        }
        acc /= (float)class_extent;
        term[r] = store_from_acc<T>(acc);
    }
}

template <>
__global__ void multilabel_soft_margin_per_row_kernel<double>(
    const double* __restrict__ input, const double* __restrict__ tgt,
    double* __restrict__ term, int64_t n_rows, int32_t class_extent,
    int64_t row_stride_in, int64_t row_stride_tgt)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t r = tid; r < n_rows; r += step) {
        const double* row = input + r * row_stride_in;
        const double* trow = tgt + r * row_stride_tgt;
        double acc = 0.0;
        for (int j = 0; j < class_extent; ++j) {
            double x = row[j];
            double y = trow[j];
            double ax = fabs(x);
            double mx = x > 0.0 ? x : 0.0;
            acc += mx - x * y + log1p(exp(-ax));
        }
        acc /= (double)class_extent;
        term[r] = acc;
    }
}

// MultilabelSoftMargin BW: per element, d/dx = (σ(x) - y) / C · sc.
template <typename T>
__global__ void multilabel_soft_margin_backward_kernel(
    const T* __restrict__ input, const T* __restrict__ tgt,
    const T* __restrict__ dy, T* __restrict__ dinput,
    int64_t n_rows, int32_t class_extent, int64_t row_stride_in, int64_t row_stride_tgt,
    int32_t reduction_mode, float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float dy0_scaled = (reduction_mode != 0) ? load_as_acc<T>(dy[0]) * inv_n_or_one : 0.0f;
    for (int64_t r = tid; r < n_rows; r += step) {
        const T* row = input + r * row_stride_in;
        const T* trow = tgt + r * row_stride_tgt;
        T* drow = dinput + r * row_stride_in;
        float sc = (reduction_mode == 0) ? load_as_acc<T>(dy[r]) : dy0_scaled;
        float coef = sc / (float)class_extent;
        for (int j = 0; j < class_extent; ++j) {
            float x = load_as_acc<T>(row[j]);
            float y = load_as_acc<T>(trow[j]);
            float sig;
            if (x >= 0.0f) { float e = expf(-x); sig = 1.0f / (1.0f + e); }
            else { float e = expf(x); sig = e / (1.0f + e); }
            drow[j] = store_from_acc<T>((sig - y) * coef);
        }
    }
}

template <>
__global__ void multilabel_soft_margin_backward_kernel<double>(
    const double* __restrict__ input, const double* __restrict__ tgt,
    const double* __restrict__ dy, double* __restrict__ dinput,
    int64_t n_rows, int32_t class_extent, int64_t row_stride_in, int64_t row_stride_tgt,
    int32_t reduction_mode, float inv_n_or_one)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double dy0_scaled = 0.0;
    if (reduction_mode == 1) dy0_scaled = dy[0] / (double)n_rows;
    else if (reduction_mode == 2) dy0_scaled = dy[0];
    (void)inv_n_or_one;
    for (int64_t r = tid; r < n_rows; r += step) {
        const double* row = input + r * row_stride_in;
        const double* trow = tgt + r * row_stride_tgt;
        double* drow = dinput + r * row_stride_in;
        double sc = (reduction_mode == 0) ? dy[r] : dy0_scaled;
        double coef = sc / (double)class_extent;
        for (int j = 0; j < class_extent; ++j) {
            double x = row[j];
            double y = trow[j];
            double sig;
            if (x >= 0.0) { double e = exp(-x); sig = 1.0 / (1.0 + e); }
            else { double e = exp(x); sig = e / (1.0 + e); }
            drow[j] = (sig - y) * coef;
        }
    }
}

} } // namespace baracuda::loss

// =============================================================================
// INSTANTIATE macros — Tier-2 margin / embedding losses.
// =============================================================================

// Common reduction-finalize helper for the FW path (called by the macros
// below). Mean denom_inv is 1/numel for elementwise ops, 1/n_rows for
// per-row ops.

// MarginRanking FW: 3 tensor inputs (x1, x2, t), elementwise per-cell.
#define BARACUDA_KERNELS_LOSS_MARGIN_RANKING_FW_INSTANTIATE(NAME, T)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int32_t reduction_mode,                                                                     \
        float margin,                                                                               \
        const void* x1, const void* x2, const void* t, void* out,                                   \
        void* workspace, size_t workspace_bytes,                                                    \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (x1 == nullptr || x2 == nullptr || t == nullptr || out == nullptr) return 2;             \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        T* term_buf = nullptr;                                                                      \
        if (reduction_mode == 0) {                                                                  \
            term_buf = static_cast<T*>(out);                                                        \
        } else {                                                                                    \
            size_t need = (size_t)numel * sizeof(T);                                                \
            if (workspace == nullptr || workspace_bytes < need) return 4;                           \
            term_buf = static_cast<T*>(workspace);                                                  \
        }                                                                                           \
        baracuda::loss::margin_ranking_per_cell_kernel<T><<<blocks, kBlock, 0, stream>>>(           \
            static_cast<const T*>(x1), static_cast<const T*>(x2), static_cast<const T*>(t),         \
            term_buf, numel, margin);                                                               \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        if (reduction_mode != 0) {                                                                  \
            float denom_inv = (reduction_mode == 1) ? (1.0f / (float)numel) : 1.0f;                 \
            baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,    \
                stream>>>(term_buf, static_cast<T*>(out), numel, denom_inv);                        \
        }                                                                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// MarginRanking BW.
#define BARACUDA_KERNELS_LOSS_MARGIN_RANKING_BW_INSTANTIATE(NAME, T)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel, int32_t reduction_mode,                                                      \
        float scale_scalar, float margin,                                                           \
        const void* x1, const void* x2, const void* t, const void* dy,                              \
        void* dx1, void* dx2,                                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                          \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (x1 == nullptr || x2 == nullptr || t == nullptr || dy == nullptr) return 2;              \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::margin_ranking_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(           \
            static_cast<const T*>(x1), static_cast<const T*>(x2), static_cast<const T*>(t),         \
            static_cast<const T*>(dy), static_cast<T*>(dx1), static_cast<T*>(dx2),                  \
            numel, reduction_mode, scale_scalar, margin);                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// HingeEmbedding FW: heterogeneous-dtype (input T, target i64), elementwise per-cell.
#define BARACUDA_KERNELS_LOSS_HINGE_EMBEDDING_FW_INSTANTIATE(NAME, T)                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel, int32_t reduction_mode, float margin,                                        \
        const void* input, const void* target, void* out,                                           \
        void* workspace, size_t workspace_bytes, void* stream_ptr)                                  \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (input == nullptr || target == nullptr || out == nullptr) return 2;                      \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        T* term_buf = nullptr;                                                                      \
        if (reduction_mode == 0) {                                                                  \
            term_buf = static_cast<T*>(out);                                                        \
        } else {                                                                                    \
            size_t need = (size_t)numel * sizeof(T);                                                \
            if (workspace == nullptr || workspace_bytes < need) return 4;                           \
            term_buf = static_cast<T*>(workspace);                                                  \
        }                                                                                           \
        baracuda::loss::hinge_embedding_per_cell_kernel<T><<<blocks, kBlock, 0, stream>>>(          \
            static_cast<const T*>(input), static_cast<const int64_t*>(target),                      \
            term_buf, numel, margin);                                                               \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        if (reduction_mode != 0) {                                                                  \
            float denom_inv = (reduction_mode == 1) ? (1.0f / (float)numel) : 1.0f;                 \
            baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,    \
                stream>>>(term_buf, static_cast<T*>(out), numel, denom_inv);                        \
        }                                                                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// HingeEmbedding BW.
#define BARACUDA_KERNELS_LOSS_HINGE_EMBEDDING_BW_INSTANTIATE(NAME, T)                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel, int32_t reduction_mode,                                                      \
        float scale_scalar, float margin,                                                           \
        const void* input, const void* target, const void* dy, void* dinput,                        \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                          \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (input == nullptr || target == nullptr || dy == nullptr || dinput == nullptr) return 2;  \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::hinge_embedding_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(          \
            static_cast<const T*>(input), static_cast<const int64_t*>(target),                      \
            static_cast<const T*>(dy), static_cast<T*>(dinput),                                     \
            numel, reduction_mode, scale_scalar, margin);                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// CosineEmbedding FW (per-row, [N, D]).
#define BARACUDA_KERNELS_LOSS_COSINE_EMBEDDING_FW_INSTANTIATE(NAME, T)                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows, int32_t d_extent, int64_t row_stride_x,                                     \
        int32_t reduction_mode, float margin,                                                       \
        const void* x1, const void* x2, const void* t, void* out,                                   \
        void* workspace, size_t workspace_bytes, void* stream_ptr)                                  \
    {                                                                                               \
        if (n_rows < 0 || d_extent < 0) return 2;                                                  \
        if (n_rows == 0) return 0;                                                                  \
        if (x1 == nullptr || x2 == nullptr || t == nullptr || out == nullptr) return 2;             \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        T* term_buf = nullptr;                                                                      \
        if (reduction_mode == 0) { term_buf = static_cast<T*>(out); }                               \
        else {                                                                                      \
            size_t need = (size_t)n_rows * sizeof(T);                                               \
            if (workspace == nullptr || workspace_bytes < need) return 4;                           \
            term_buf = static_cast<T*>(workspace);                                                  \
        }                                                                                           \
        baracuda::loss::cosine_embedding_per_row_kernel<T><<<blocks, kBlock, 0, stream>>>(          \
            static_cast<const T*>(x1), static_cast<const T*>(x2), static_cast<const T*>(t),         \
            term_buf, n_rows, d_extent, row_stride_x, margin);                                      \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        if (reduction_mode != 0) {                                                                  \
            float denom_inv = (reduction_mode == 1) ? (1.0f / (float)n_rows) : 1.0f;                \
            baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,    \
                stream>>>(term_buf, static_cast<T*>(out), n_rows, denom_inv);                       \
        }                                                                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// CosineEmbedding BW.
#define BARACUDA_KERNELS_LOSS_COSINE_EMBEDDING_BW_INSTANTIATE(NAME, T)                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows, int32_t d_extent, int64_t row_stride_x,                                     \
        int32_t reduction_mode, float scale_scalar, float margin,                                   \
        const void* x1, const void* x2, const void* t, const void* dy,                              \
        void* dx1, void* dx2,                                                                       \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                          \
    {                                                                                               \
        if (n_rows < 0 || d_extent < 0) return 2;                                                  \
        if (n_rows == 0) return 0;                                                                  \
        if (x1 == nullptr || x2 == nullptr || t == nullptr || dy == nullptr) return 2;              \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::cosine_embedding_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(         \
            static_cast<const T*>(x1), static_cast<const T*>(x2), static_cast<const T*>(t),         \
            static_cast<const T*>(dy), static_cast<T*>(dx1), static_cast<T*>(dx2),                  \
            n_rows, d_extent, row_stride_x, reduction_mode, scale_scalar, margin);                  \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// TripletMargin FW (per-row, [N, D]).
#define BARACUDA_KERNELS_LOSS_TRIPLET_MARGIN_FW_INSTANTIATE(NAME, T)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows, int32_t d_extent, int64_t row_stride,                                       \
        int32_t reduction_mode, float margin, float p_norm,                                         \
        const void* a, const void* p_tensor, const void* n_tensor, void* out,                       \
        void* workspace, size_t workspace_bytes, void* stream_ptr)                                  \
    {                                                                                               \
        if (n_rows < 0 || d_extent < 0) return 2;                                                  \
        if (n_rows == 0) return 0;                                                                  \
        if (a == nullptr || p_tensor == nullptr || n_tensor == nullptr || out == nullptr) return 2; \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        T* term_buf = nullptr;                                                                      \
        if (reduction_mode == 0) { term_buf = static_cast<T*>(out); }                               \
        else {                                                                                      \
            size_t need = (size_t)n_rows * sizeof(T);                                               \
            if (workspace == nullptr || workspace_bytes < need) return 4;                           \
            term_buf = static_cast<T*>(workspace);                                                  \
        }                                                                                           \
        baracuda::loss::triplet_margin_per_row_kernel<T><<<blocks, kBlock, 0, stream>>>(            \
            static_cast<const T*>(a), static_cast<const T*>(p_tensor),                              \
            static_cast<const T*>(n_tensor),                                                        \
            term_buf, n_rows, d_extent, row_stride, margin, p_norm);                                \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        if (reduction_mode != 0) {                                                                  \
            float denom_inv = (reduction_mode == 1) ? (1.0f / (float)n_rows) : 1.0f;                \
            baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,    \
                stream>>>(term_buf, static_cast<T*>(out), n_rows, denom_inv);                       \
        }                                                                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// TripletMargin BW.
#define BARACUDA_KERNELS_LOSS_TRIPLET_MARGIN_BW_INSTANTIATE(NAME, T)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows, int32_t d_extent, int64_t row_stride,                                       \
        int32_t reduction_mode, float scale_scalar, float margin, float p_norm,                    \
        const void* a, const void* p_tensor, const void* n_tensor, const void* dy,                  \
        void* da, void* dp, void* dn,                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                          \
    {                                                                                               \
        if (n_rows < 0 || d_extent < 0) return 2;                                                  \
        if (n_rows == 0) return 0;                                                                  \
        if (a == nullptr || p_tensor == nullptr || n_tensor == nullptr || dy == nullptr) return 2;  \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::triplet_margin_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(           \
            static_cast<const T*>(a), static_cast<const T*>(p_tensor),                              \
            static_cast<const T*>(n_tensor), static_cast<const T*>(dy),                             \
            static_cast<T*>(da), static_cast<T*>(dp), static_cast<T*>(dn),                          \
            n_rows, d_extent, row_stride, reduction_mode, scale_scalar, margin, p_norm);            \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// MultiMargin FW (per-row, [N, C], target i64[N]).
#define BARACUDA_KERNELS_LOSS_MULTI_MARGIN_FW_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows, int32_t class_extent, int64_t row_stride,                                   \
        int32_t reduction_mode, float margin, float p_norm,                                         \
        const void* input, const void* target, void* out,                                           \
        void* workspace, size_t workspace_bytes, void* stream_ptr)                                  \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0) return 2;                                              \
        if (n_rows == 0) return 0;                                                                  \
        if (input == nullptr || target == nullptr || out == nullptr) return 2;                      \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        T* term_buf = nullptr;                                                                      \
        if (reduction_mode == 0) { term_buf = static_cast<T*>(out); }                               \
        else {                                                                                      \
            size_t need = (size_t)n_rows * sizeof(T);                                               \
            if (workspace == nullptr || workspace_bytes < need) return 4;                           \
            term_buf = static_cast<T*>(workspace);                                                  \
        }                                                                                           \
        baracuda::loss::multi_margin_per_row_kernel<T><<<blocks, kBlock, 0, stream>>>(              \
            static_cast<const T*>(input), static_cast<const int64_t*>(target),                      \
            term_buf, n_rows, class_extent, row_stride, margin, p_norm);                            \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        if (reduction_mode != 0) {                                                                  \
            float denom_inv = (reduction_mode == 1) ? (1.0f / (float)n_rows) : 1.0f;                \
            baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,    \
                stream>>>(term_buf, static_cast<T*>(out), n_rows, denom_inv);                       \
        }                                                                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// MultiMargin BW.
#define BARACUDA_KERNELS_LOSS_MULTI_MARGIN_BW_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows, int32_t class_extent, int64_t row_stride,                                   \
        int32_t reduction_mode, float scale_scalar, float margin, float p_norm,                    \
        const void* input, const void* target, const void* dy, void* dinput,                        \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                          \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0) return 2;                                              \
        if (n_rows == 0) return 0;                                                                  \
        if (input == nullptr || target == nullptr || dy == nullptr || dinput == nullptr) return 2;  \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::multi_margin_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(             \
            static_cast<const T*>(input), static_cast<const int64_t*>(target),                      \
            static_cast<const T*>(dy), static_cast<T*>(dinput),                                     \
            n_rows, class_extent, row_stride, reduction_mode, scale_scalar, margin, p_norm);        \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// MultilabelMargin FW (per-row, [N, C], target i64[N, C]).
#define BARACUDA_KERNELS_LOSS_MULTILABEL_MARGIN_FW_INSTANTIATE(NAME, T)                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows, int32_t class_extent,                                                       \
        int64_t row_stride_in, int64_t row_stride_tgt,                                              \
        int32_t reduction_mode,                                                                     \
        const void* input, const void* target, void* out,                                           \
        void* workspace, size_t workspace_bytes, void* stream_ptr)                                  \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0) return 2;                                              \
        if (n_rows == 0) return 0;                                                                  \
        if (input == nullptr || target == nullptr || out == nullptr) return 2;                      \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        T* term_buf = nullptr;                                                                      \
        if (reduction_mode == 0) { term_buf = static_cast<T*>(out); }                               \
        else {                                                                                      \
            size_t need = (size_t)n_rows * sizeof(T);                                               \
            if (workspace == nullptr || workspace_bytes < need) return 4;                           \
            term_buf = static_cast<T*>(workspace);                                                  \
        }                                                                                           \
        baracuda::loss::multilabel_margin_per_row_kernel<T><<<blocks, kBlock, 0, stream>>>(         \
            static_cast<const T*>(input), static_cast<const int64_t*>(target),                      \
            term_buf, n_rows, class_extent, row_stride_in, row_stride_tgt);                         \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        if (reduction_mode != 0) {                                                                  \
            float denom_inv = (reduction_mode == 1) ? (1.0f / (float)n_rows) : 1.0f;                \
            baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,    \
                stream>>>(term_buf, static_cast<T*>(out), n_rows, denom_inv);                       \
        }                                                                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// MultilabelMargin BW.
#define BARACUDA_KERNELS_LOSS_MULTILABEL_MARGIN_BW_INSTANTIATE(NAME, T)                             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows, int32_t class_extent,                                                       \
        int64_t row_stride_in, int64_t row_stride_tgt,                                              \
        int32_t reduction_mode, float scale_scalar,                                                 \
        const void* input, const void* target, const void* dy, void* dinput,                        \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                          \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0) return 2;                                              \
        if (n_rows == 0) return 0;                                                                  \
        if (input == nullptr || target == nullptr || dy == nullptr || dinput == nullptr) return 2;  \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::multilabel_margin_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(        \
            static_cast<const T*>(input), static_cast<const int64_t*>(target),                      \
            static_cast<const T*>(dy), static_cast<T*>(dinput),                                     \
            n_rows, class_extent, row_stride_in, row_stride_tgt,                                    \
            reduction_mode, scale_scalar);                                                          \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// MultilabelSoftMargin FW.
#define BARACUDA_KERNELS_LOSS_MULTILABEL_SOFT_MARGIN_FW_INSTANTIATE(NAME, T)                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows, int32_t class_extent,                                                       \
        int64_t row_stride_in, int64_t row_stride_tgt,                                              \
        int32_t reduction_mode,                                                                     \
        const void* input, const void* target, void* out,                                           \
        void* workspace, size_t workspace_bytes, void* stream_ptr)                                  \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0) return 2;                                              \
        if (n_rows == 0) return 0;                                                                  \
        if (input == nullptr || target == nullptr || out == nullptr) return 2;                      \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        T* term_buf = nullptr;                                                                      \
        if (reduction_mode == 0) { term_buf = static_cast<T*>(out); }                               \
        else {                                                                                      \
            size_t need = (size_t)n_rows * sizeof(T);                                               \
            if (workspace == nullptr || workspace_bytes < need) return 4;                           \
            term_buf = static_cast<T*>(workspace);                                                  \
        }                                                                                           \
        baracuda::loss::multilabel_soft_margin_per_row_kernel<T><<<blocks, kBlock, 0, stream>>>(    \
            static_cast<const T*>(input), static_cast<const T*>(target),                            \
            term_buf, n_rows, class_extent, row_stride_in, row_stride_tgt);                         \
        if (cudaGetLastError() != cudaSuccess) return 5;                                            \
        if (reduction_mode != 0) {                                                                  \
            float denom_inv = (reduction_mode == 1) ? (1.0f / (float)n_rows) : 1.0f;                \
            baracuda::loss::loss_reduce_finalize_kernel<T><<<1, baracuda::loss::kBlockReduce, 0,    \
                stream>>>(term_buf, static_cast<T*>(out), n_rows, denom_inv);                       \
        }                                                                                           \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

// MultilabelSoftMargin BW.
#define BARACUDA_KERNELS_LOSS_MULTILABEL_SOFT_MARGIN_BW_INSTANTIATE(NAME, T)                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t n_rows, int32_t class_extent,                                                       \
        int64_t row_stride_in, int64_t row_stride_tgt,                                              \
        int32_t reduction_mode, float scale_scalar,                                                 \
        const void* input, const void* target, const void* dy, void* dinput,                        \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                          \
    {                                                                                               \
        if (n_rows < 0 || class_extent < 0) return 2;                                              \
        if (n_rows == 0) return 0;                                                                  \
        if (input == nullptr || target == nullptr || dy == nullptr || dinput == nullptr) return 2;  \
        if (reduction_mode < 0 || reduction_mode > 2) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (n_rows + kBlock - 1) / kBlock;                                        \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::loss::multilabel_soft_margin_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(   \
            static_cast<const T*>(input), static_cast<const T*>(target),                            \
            static_cast<const T*>(dy), static_cast<T*>(dinput),                                     \
            n_rows, class_extent, row_stride_in, row_stride_tgt,                                    \
            reduction_mode, scale_scalar);                                                          \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }

#endif // BARACUDA_LOSS_CUH
