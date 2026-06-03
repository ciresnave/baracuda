// baracuda_quantize.cuh
//
// Templated kernels and INSTANTIATE macros for the quantization op family
// (Phase 8 Milestone 8.1 — Category P from the comprehensive plan).
//
// Ops shipped here (trailblazer):
//   quantize_per_tensor      — q = clamp(round(x/scale) + zp, qmin, qmax)
//   quantize_per_tensor_bw   — dx = (dy / scale) * in_range_mask        (STE)
//   dequantize_per_tensor    — x = scale * (q - zp)
//   dequantize_per_tensor_bw — dq = dy * scale
//   quantize_per_channel     — same as per_tensor but scale[c]/zp[c] per axis slice
//   quantize_per_channel_bw  — dx = (dy / scale[c]) * in_range_mask[c]  (STE)
//   dequantize_per_channel   — x = scale[c] * (q - zp[c])
//   dequantize_per_channel_bw— dq = dy * scale[c]
//   fake_quantize            — y = scale * (clamp(round(x/scale)+zp, qmin, qmax) - zp)
//   fake_quantize_bw         — dx = dy * in_range_mask                  (STE, NO 1/scale)
//
// Trailblazer dtype scope:
//   Input FP : float, double, __half, __nv_bfloat16
//   Output Q : int8_t, uint8_t (sub-byte s4/u4 deferred to 8.2+)
//   scale    : same FP dtype as input
//   zero_pt  : i32 (wide enough for any byte-range qmin/qmax)
//
// STE convention (READ TWICE — easy to get wrong):
//   The "in-range mask" for BW is NOT saved during FW. BW re-computes it
//   from the original input `x` (which the caller retains for autograd)
//   plus `scale`/`zero_point`. Mask = (qmin <= round(x/scale)+zp <= qmax).
//   - quantize_bw      : dx = dy / scale (where in-range; else 0).
//   - fake_quantize_bw : dx = dy        (where in-range; else 0).
//     The 1/scale factor is omitted because the FW's dequant-side
//     multiplication by scale exactly cancels the STE 1/scale.
//
// Per-channel kernels assume the caller has padded rank to MAX_RANK (=4)
// with 1's so the kernel sees a fixed-rank tensor; the `axis` field
// selects which of the 4 dims indexes scale[]/zp[].
//
// Status codes returned by the launchers mirror the rest of the kernel
// surface:
//   0 success
//   1 misaligned operand
//   2 invalid problem
//   3 unsupported
//   4 workspace too small
//   5 internal kernel error (launch failure)

#ifndef BARACUDA_QUANTIZE_CUH
#define BARACUDA_QUANTIZE_CUH

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace baracuda { namespace quantize {

inline constexpr int MAX_RANK = 4;

// =============================================================================
// FP → float promotion helpers — every kernel does math at float and casts
// back. Matches the f32-detour convention used elsewhere in the kernel
// family for f16 / bf16.
// =============================================================================

template <typename T> __device__ __forceinline__ float to_float(T v);
template <typename T> __device__ __forceinline__ T from_float(float v);

template <> __device__ __forceinline__ float to_float<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_float<double>(double v) { return (float)v; }
template <> __device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ double from_float<double>(float v) { return (double)v; }
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// Specialized "do math at the native FP precision" variants for f64 so we
// don't lose precision through the float detour. Used inside the per-cell
// loop only for the f64 case.
__device__ __forceinline__ double to_double(double v) { return v; }
__device__ __forceinline__ double to_double(float v) { return (double)v; }
__device__ __forceinline__ double to_double(__half v) { return (double)__half2float(v); }
__device__ __forceinline__ double to_double(__nv_bfloat16 v) {
    return (double)__bfloat162float(v);
}

// =============================================================================
// Round-half-to-even helpers. Match `__float2int_rn` / `__double2int_rn`
// semantics — same convention used by the integer-GEMM saturating-cast
// helpers and by PyTorch's `torch.round`.
// =============================================================================

__device__ __forceinline__ int32_t round_to_int_f(float v) { return __float2int_rn(v); }
__device__ __forceinline__ int32_t round_to_int_d(double v) { return __double2int_rn(v); }

// =============================================================================
// quantize_per_tensor FW kernel — q = clamp(round(x/scale) + zp, qmin, qmax)
// =============================================================================

template <typename TIn, typename TOut>
__global__ void quantize_per_tensor_kernel(
    const TIn* __restrict__ x,
    TOut* __restrict__ q,
    int64_t numel,
    float scale,           // f32 scale (input dtype already promoted on host side)
    int32_t zero_point,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float inv_scale = 1.0f / scale;
    for (int64_t i = tid; i < numel; i += step) {
        float xf = to_float<TIn>(x[i]);
        int32_t r = round_to_int_f(xf * inv_scale) + zero_point;
        if (r < q_min) r = q_min;
        if (r > q_max) r = q_max;
        q[i] = static_cast<TOut>(r);
    }
}

// f64 specialization — do the divide / round at f64 precision then narrow.
template <typename TOut>
__global__ void quantize_per_tensor_kernel_f64(
    const double* __restrict__ x,
    TOut* __restrict__ q,
    int64_t numel,
    double scale,
    int32_t zero_point,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double inv_scale = 1.0 / scale;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t r = round_to_int_d(x[i] * inv_scale) + zero_point;
        if (r < q_min) r = q_min;
        if (r > q_max) r = q_max;
        q[i] = static_cast<TOut>(r);
    }
}

template <typename TIn, typename TOut>
__host__ inline int32_t launch_quantize_per_tensor(
    const TIn* x, TOut* q,
    int64_t numel, float scale, int32_t zero_point,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_tensor_kernel<TIn, TOut><<<blocks, kBlock, 0, stream>>>(
        x, q, numel, scale, zero_point, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TOut>
__host__ inline int32_t launch_quantize_per_tensor_f64(
    const double* x, TOut* q,
    int64_t numel, double scale, int32_t zero_point,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_tensor_kernel_f64<TOut><<<blocks, kBlock, 0, stream>>>(
        x, q, numel, scale, zero_point, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// quantize_per_tensor BW kernel — STE.
//   dx[i] = (dy[i] / scale) * in_range_mask(x[i])
// where in_range = (qmin <= round(x/scale) + zp <= qmax).
// Mask is recomputed from the saved input `x` — no separate mask tensor.
// =============================================================================

template <typename TIn>
__global__ void quantize_per_tensor_backward_kernel(
    const TIn* __restrict__ x,
    const TIn* __restrict__ dy,
    TIn* __restrict__ dx,
    int64_t numel,
    float scale,
    int32_t zero_point,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float inv_scale = 1.0f / scale;
    for (int64_t i = tid; i < numel; i += step) {
        float xf  = to_float<TIn>(x[i]);
        float dyf = to_float<TIn>(dy[i]);
        int32_t r = round_to_int_f(xf * inv_scale) + zero_point;
        bool in_range = (r >= q_min) && (r <= q_max);
        float gx = in_range ? (dyf * inv_scale) : 0.0f;
        dx[i] = from_float<TIn>(gx);
    }
}

template <>
__global__ void quantize_per_tensor_backward_kernel<double>(
    const double* __restrict__ x,
    const double* __restrict__ dy,
    double* __restrict__ dx,
    int64_t numel,
    float scale_f32_unused,
    int32_t zero_point,
    int32_t q_min,
    int32_t q_max)
{
    // f64 path: caller invokes the dedicated `_f64` launcher with a
    // double scale; this template specialization should not be reached.
    // Fall back to single-precision math if it is.
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double inv_scale = 1.0 / (double)scale_f32_unused;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t r = round_to_int_d(x[i] * inv_scale) + zero_point;
        bool in_range = (r >= q_min) && (r <= q_max);
        dx[i] = in_range ? (dy[i] * inv_scale) : 0.0;
    }
}

// Dedicated f64 BW kernel (preserves f64 precision in scale).
__global__ inline void quantize_per_tensor_backward_kernel_f64(
    const double* __restrict__ x,
    const double* __restrict__ dy,
    double* __restrict__ dx,
    int64_t numel,
    double scale,
    int32_t zero_point,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double inv_scale = 1.0 / scale;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t r = round_to_int_d(x[i] * inv_scale) + zero_point;
        bool in_range = (r >= q_min) && (r <= q_max);
        dx[i] = in_range ? (dy[i] * inv_scale) : 0.0;
    }
}

template <typename TIn>
__host__ inline int32_t launch_quantize_per_tensor_backward(
    const TIn* x, const TIn* dy, TIn* dx,
    int64_t numel, float scale, int32_t zero_point,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_tensor_backward_kernel<TIn><<<blocks, kBlock, 0, stream>>>(
        x, dy, dx, numel, scale, zero_point, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_quantize_per_tensor_backward_f64(
    const double* x, const double* dy, double* dx,
    int64_t numel, double scale, int32_t zero_point,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_tensor_backward_kernel_f64<<<blocks, kBlock, 0, stream>>>(
        x, dy, dx, numel, scale, zero_point, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// dequantize_per_tensor FW kernel — x = scale * (q - zp).
// =============================================================================

template <typename TIn, typename TOut>
__global__ void dequantize_per_tensor_kernel(
    const TOut* __restrict__ q,       // int input
    TIn* __restrict__ x,              // FP output
    int64_t numel,
    float scale,
    int32_t zero_point)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t qi = (int32_t)q[i];
        float xf = scale * (float)(qi - zero_point);
        x[i] = from_float<TIn>(xf);
    }
}

template <typename TOut>
__global__ void dequantize_per_tensor_kernel_f64(
    const TOut* __restrict__ q,
    double* __restrict__ x,
    int64_t numel,
    double scale,
    int32_t zero_point)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t qi = (int32_t)q[i];
        x[i] = scale * (double)(qi - zero_point);
    }
}

template <typename TIn, typename TOut>
__host__ inline int32_t launch_dequantize_per_tensor(
    const TOut* q, TIn* x,
    int64_t numel, float scale, int32_t zero_point,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_tensor_kernel<TIn, TOut><<<blocks, kBlock, 0, stream>>>(
        q, x, numel, scale, zero_point);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TOut>
__host__ inline int32_t launch_dequantize_per_tensor_f64(
    const TOut* q, double* x,
    int64_t numel, double scale, int32_t zero_point,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_tensor_kernel_f64<TOut><<<blocks, kBlock, 0, stream>>>(
        q, x, numel, scale, zero_point);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// dequantize_per_tensor BW kernel — dq = dy * scale. Output is the input
// FP gradient `dq` (cast back to FP since the quant-graph's gradient
// continues to flow in FP space; the integer `q` is non-differentiable).
// Note: PyTorch reports `dq` as same-FP-dtype as `dy` here, NOT as int.
// =============================================================================

template <typename TIn>
__global__ void dequantize_per_tensor_backward_kernel(
    const TIn* __restrict__ dy,
    TIn* __restrict__ dq,
    int64_t numel,
    float scale)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float dyf = to_float<TIn>(dy[i]);
        dq[i] = from_float<TIn>(dyf * scale);
    }
}

__global__ inline void dequantize_per_tensor_backward_kernel_f64(
    const double* __restrict__ dy,
    double* __restrict__ dq,
    int64_t numel,
    double scale)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        dq[i] = dy[i] * scale;
    }
}

template <typename TIn>
__host__ inline int32_t launch_dequantize_per_tensor_backward(
    const TIn* dy, TIn* dq,
    int64_t numel, float scale,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_tensor_backward_kernel<TIn><<<blocks, kBlock, 0, stream>>>(
        dy, dq, numel, scale);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_dequantize_per_tensor_backward_f64(
    const double* dy, double* dq,
    int64_t numel, double scale,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_tensor_backward_kernel_f64<<<blocks, kBlock, 0, stream>>>(
        dy, dq, numel, scale);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// fake_quantize FW kernel — y = scale * (clamp(round(x/scale)+zp, qmin, qmax) - zp).
// Stays in FP space; output dtype == input dtype.
// =============================================================================

template <typename TIn>
__global__ void fake_quantize_kernel(
    const TIn* __restrict__ x,
    TIn* __restrict__ y,
    int64_t numel,
    float scale,
    int32_t zero_point,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float inv_scale = 1.0f / scale;
    for (int64_t i = tid; i < numel; i += step) {
        float xf = to_float<TIn>(x[i]);
        int32_t r = round_to_int_f(xf * inv_scale) + zero_point;
        if (r < q_min) r = q_min;
        if (r > q_max) r = q_max;
        float yf = scale * (float)(r - zero_point);
        y[i] = from_float<TIn>(yf);
    }
}

__global__ inline void fake_quantize_kernel_f64(
    const double* __restrict__ x,
    double* __restrict__ y,
    int64_t numel,
    double scale,
    int32_t zero_point,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double inv_scale = 1.0 / scale;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t r = round_to_int_d(x[i] * inv_scale) + zero_point;
        if (r < q_min) r = q_min;
        if (r > q_max) r = q_max;
        y[i] = scale * (double)(r - zero_point);
    }
}

template <typename TIn>
__host__ inline int32_t launch_fake_quantize(
    const TIn* x, TIn* y,
    int64_t numel, float scale, int32_t zero_point,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fake_quantize_kernel<TIn><<<blocks, kBlock, 0, stream>>>(
        x, y, numel, scale, zero_point, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_fake_quantize_f64(
    const double* x, double* y,
    int64_t numel, double scale, int32_t zero_point,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fake_quantize_kernel_f64<<<blocks, kBlock, 0, stream>>>(
        x, y, numel, scale, zero_point, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// fake_quantize BW kernel — dx = dy * in_range_mask. NO 1/scale factor.
// =============================================================================

template <typename TIn>
__global__ void fake_quantize_backward_kernel(
    const TIn* __restrict__ x,
    const TIn* __restrict__ dy,
    TIn* __restrict__ dx,
    int64_t numel,
    float scale,
    int32_t zero_point,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    float inv_scale = 1.0f / scale;
    for (int64_t i = tid; i < numel; i += step) {
        float xf  = to_float<TIn>(x[i]);
        float dyf = to_float<TIn>(dy[i]);
        int32_t r = round_to_int_f(xf * inv_scale) + zero_point;
        bool in_range = (r >= q_min) && (r <= q_max);
        dx[i] = from_float<TIn>(in_range ? dyf : 0.0f);
    }
}

__global__ inline void fake_quantize_backward_kernel_f64(
    const double* __restrict__ x,
    const double* __restrict__ dy,
    double* __restrict__ dx,
    int64_t numel,
    double scale,
    int32_t zero_point,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    double inv_scale = 1.0 / scale;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t r = round_to_int_d(x[i] * inv_scale) + zero_point;
        bool in_range = (r >= q_min) && (r <= q_max);
        dx[i] = in_range ? dy[i] : 0.0;
    }
}

template <typename TIn>
__host__ inline int32_t launch_fake_quantize_backward(
    const TIn* x, const TIn* dy, TIn* dx,
    int64_t numel, float scale, int32_t zero_point,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fake_quantize_backward_kernel<TIn><<<blocks, kBlock, 0, stream>>>(
        x, dy, dx, numel, scale, zero_point, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_fake_quantize_backward_f64(
    const double* x, const double* dy, double* dx,
    int64_t numel, double scale, int32_t zero_point,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fake_quantize_backward_kernel_f64<<<blocks, kBlock, 0, stream>>>(
        x, dy, dx, numel, scale, zero_point, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// Per-channel helpers: caller pads the tensor rank to MAX_RANK (=4) with
// extents of 1. `axis` selects which dim indexes the per-channel scale[]
// and zero_point[] vectors. Channel index is computed from the flat
// element index by decomposing it into a 4-D coord using the contiguous
// shape, then picking coord[axis].
//
// All inputs are assumed contiguous (row-major) for the trailblazer —
// strided per-channel quantize is deferred.
// =============================================================================

struct PcShape4 { int32_t d[MAX_RANK]; };

__device__ __forceinline__ int32_t pc_axis_coord(
    int64_t linear, const PcShape4 shape, int32_t axis)
{
    // Decompose linear → coord assuming row-major contiguous shape.
    int64_t rem = linear;
    int64_t coord[MAX_RANK] = {0};
    for (int d = MAX_RANK - 1; d >= 0; --d) {
        int32_t s = shape.d[d];
        if (s <= 0) { coord[d] = 0; continue; }
        coord[d] = rem % (int64_t)s;
        rem /= (int64_t)s;
    }
    return (int32_t)coord[axis];
}

// ----- per-channel quantize FW -----

template <typename TIn, typename TOut>
__global__ void quantize_per_channel_kernel(
    const TIn* __restrict__ x,
    const TIn* __restrict__ scale,         // [C]
    const int32_t* __restrict__ zp,        // [C]
    TOut* __restrict__ q,
    int64_t numel,
    PcShape4 shape,
    int32_t axis,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t c = pc_axis_coord(i, shape, axis);
        float s_f = to_float<TIn>(scale[c]);
        float xf  = to_float<TIn>(x[i]);
        int32_t r = round_to_int_f(xf / s_f) + zp[c];
        if (r < q_min) r = q_min;
        if (r > q_max) r = q_max;
        q[i] = static_cast<TOut>(r);
    }
}

template <typename TOut>
__global__ void quantize_per_channel_kernel_f64(
    const double* __restrict__ x,
    const double* __restrict__ scale,
    const int32_t* __restrict__ zp,
    TOut* __restrict__ q,
    int64_t numel,
    PcShape4 shape,
    int32_t axis,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t c = pc_axis_coord(i, shape, axis);
        int32_t r = round_to_int_d(x[i] / scale[c]) + zp[c];
        if (r < q_min) r = q_min;
        if (r > q_max) r = q_max;
        q[i] = static_cast<TOut>(r);
    }
}

template <typename TIn, typename TOut>
__host__ inline int32_t launch_quantize_per_channel(
    const TIn* x, const TIn* scale, const int32_t* zp, TOut* q,
    int64_t numel,
    const int32_t* shape_host, int32_t axis,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    if (axis < 0 || axis >= MAX_RANK) return 2;
    PcShape4 sh = {};
    for (int i = 0; i < MAX_RANK; ++i) sh.d[i] = shape_host[i];
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_channel_kernel<TIn, TOut><<<blocks, kBlock, 0, stream>>>(
        x, scale, zp, q, numel, sh, axis, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TOut>
__host__ inline int32_t launch_quantize_per_channel_f64(
    const double* x, const double* scale, const int32_t* zp, TOut* q,
    int64_t numel,
    const int32_t* shape_host, int32_t axis,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    if (axis < 0 || axis >= MAX_RANK) return 2;
    PcShape4 sh = {};
    for (int i = 0; i < MAX_RANK; ++i) sh.d[i] = shape_host[i];
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_channel_kernel_f64<TOut><<<blocks, kBlock, 0, stream>>>(
        x, scale, zp, q, numel, sh, axis, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// ----- per-channel quantize BW (STE) -----

template <typename TIn>
__global__ void quantize_per_channel_backward_kernel(
    const TIn* __restrict__ x,
    const TIn* __restrict__ scale,
    const int32_t* __restrict__ zp,
    const TIn* __restrict__ dy,
    TIn* __restrict__ dx,
    int64_t numel,
    PcShape4 shape,
    int32_t axis,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t c = pc_axis_coord(i, shape, axis);
        float s_f = to_float<TIn>(scale[c]);
        float inv = 1.0f / s_f;
        float xf  = to_float<TIn>(x[i]);
        float dyf = to_float<TIn>(dy[i]);
        int32_t r = round_to_int_f(xf * inv) + zp[c];
        bool in_range = (r >= q_min) && (r <= q_max);
        dx[i] = from_float<TIn>(in_range ? (dyf * inv) : 0.0f);
    }
}

__global__ inline void quantize_per_channel_backward_kernel_f64(
    const double* __restrict__ x,
    const double* __restrict__ scale,
    const int32_t* __restrict__ zp,
    const double* __restrict__ dy,
    double* __restrict__ dx,
    int64_t numel,
    PcShape4 shape,
    int32_t axis,
    int32_t q_min,
    int32_t q_max)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t c = pc_axis_coord(i, shape, axis);
        double inv = 1.0 / scale[c];
        int32_t r = round_to_int_d(x[i] * inv) + zp[c];
        bool in_range = (r >= q_min) && (r <= q_max);
        dx[i] = in_range ? (dy[i] * inv) : 0.0;
    }
}

template <typename TIn>
__host__ inline int32_t launch_quantize_per_channel_backward(
    const TIn* x, const TIn* scale, const int32_t* zp,
    const TIn* dy, TIn* dx,
    int64_t numel,
    const int32_t* shape_host, int32_t axis,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    if (axis < 0 || axis >= MAX_RANK) return 2;
    PcShape4 sh = {};
    for (int i = 0; i < MAX_RANK; ++i) sh.d[i] = shape_host[i];
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_channel_backward_kernel<TIn><<<blocks, kBlock, 0, stream>>>(
        x, scale, zp, dy, dx, numel, sh, axis, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_quantize_per_channel_backward_f64(
    const double* x, const double* scale, const int32_t* zp,
    const double* dy, double* dx,
    int64_t numel,
    const int32_t* shape_host, int32_t axis,
    int32_t q_min, int32_t q_max,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    if (axis < 0 || axis >= MAX_RANK) return 2;
    PcShape4 sh = {};
    for (int i = 0; i < MAX_RANK; ++i) sh.d[i] = shape_host[i];
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_channel_backward_kernel_f64<<<blocks, kBlock, 0, stream>>>(
        x, scale, zp, dy, dx, numel, sh, axis, q_min, q_max);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// ----- per-channel dequantize FW -----

template <typename TIn, typename TOut>
__global__ void dequantize_per_channel_kernel(
    const TOut* __restrict__ q,
    const TIn* __restrict__ scale,
    const int32_t* __restrict__ zp,
    TIn* __restrict__ x,
    int64_t numel,
    PcShape4 shape,
    int32_t axis)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t c = pc_axis_coord(i, shape, axis);
        float s_f = to_float<TIn>(scale[c]);
        int32_t qi = (int32_t)q[i];
        float xf = s_f * (float)(qi - zp[c]);
        x[i] = from_float<TIn>(xf);
    }
}

template <typename TOut>
__global__ void dequantize_per_channel_kernel_f64(
    const TOut* __restrict__ q,
    const double* __restrict__ scale,
    const int32_t* __restrict__ zp,
    double* __restrict__ x,
    int64_t numel,
    PcShape4 shape,
    int32_t axis)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t c = pc_axis_coord(i, shape, axis);
        int32_t qi = (int32_t)q[i];
        x[i] = scale[c] * (double)(qi - zp[c]);
    }
}

template <typename TIn, typename TOut>
__host__ inline int32_t launch_dequantize_per_channel(
    const TOut* q, const TIn* scale, const int32_t* zp, TIn* x,
    int64_t numel,
    const int32_t* shape_host, int32_t axis,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    if (axis < 0 || axis >= MAX_RANK) return 2;
    PcShape4 sh = {};
    for (int i = 0; i < MAX_RANK; ++i) sh.d[i] = shape_host[i];
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_channel_kernel<TIn, TOut><<<blocks, kBlock, 0, stream>>>(
        q, scale, zp, x, numel, sh, axis);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TOut>
__host__ inline int32_t launch_dequantize_per_channel_f64(
    const TOut* q, const double* scale, const int32_t* zp, double* x,
    int64_t numel,
    const int32_t* shape_host, int32_t axis,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    if (axis < 0 || axis >= MAX_RANK) return 2;
    PcShape4 sh = {};
    for (int i = 0; i < MAX_RANK; ++i) sh.d[i] = shape_host[i];
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_channel_kernel_f64<TOut><<<blocks, kBlock, 0, stream>>>(
        q, scale, zp, x, numel, sh, axis);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// ----- per-channel dequantize BW -----

template <typename TIn>
__global__ void dequantize_per_channel_backward_kernel(
    const TIn* __restrict__ scale,
    const TIn* __restrict__ dy,
    TIn* __restrict__ dq,
    int64_t numel,
    PcShape4 shape,
    int32_t axis)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t c = pc_axis_coord(i, shape, axis);
        float s_f = to_float<TIn>(scale[c]);
        float dyf = to_float<TIn>(dy[i]);
        dq[i] = from_float<TIn>(dyf * s_f);
    }
}

__global__ inline void dequantize_per_channel_backward_kernel_f64(
    const double* __restrict__ scale,
    const double* __restrict__ dy,
    double* __restrict__ dq,
    int64_t numel,
    PcShape4 shape,
    int32_t axis)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        int32_t c = pc_axis_coord(i, shape, axis);
        dq[i] = dy[i] * scale[c];
    }
}

template <typename TIn>
__host__ inline int32_t launch_dequantize_per_channel_backward(
    const TIn* scale, const TIn* dy, TIn* dq,
    int64_t numel,
    const int32_t* shape_host, int32_t axis,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    if (axis < 0 || axis >= MAX_RANK) return 2;
    PcShape4 sh = {};
    for (int i = 0; i < MAX_RANK; ++i) sh.d[i] = shape_host[i];
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_channel_backward_kernel<TIn><<<blocks, kBlock, 0, stream>>>(
        scale, dy, dq, numel, sh, axis);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

__host__ inline int32_t launch_dequantize_per_channel_backward_f64(
    const double* scale, const double* dy, double* dq,
    int64_t numel,
    const int32_t* shape_host, int32_t axis,
    cudaStream_t stream)
{
    if (numel < 0) return 2;
    if (numel == 0) return 0;
    if (axis < 0 || axis >= MAX_RANK) return 2;
    PcShape4 sh = {};
    for (int i = 0; i < MAX_RANK; ++i) sh.d[i] = shape_host[i];
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_channel_backward_kernel_f64<<<blocks, kBlock, 0, stream>>>(
        scale, dy, dq, numel, sh, axis);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::quantize

// =============================================================================
// INSTANTIATE macros — emit `extern "C"` launcher per (op, dtype) pair.
// =============================================================================

// quantize_per_tensor — f32-scale variant. TIn ∈ {f32, f16, bf16}, TOut ∈ {s8, u8}.
#define BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_INSTANTIATE(NAME, TIN, TOUT)             \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                  \
        int64_t numel,                                                                  \
        float scale,                                                                    \
        int32_t zero_point,                                                             \
        int32_t q_min,                                                                  \
        int32_t q_max,                                                                  \
        const void* x,                                                                  \
        void* q,                                                                        \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                \
        void* stream_ptr)                                                               \
    {                                                                                   \
        if (x == nullptr || q == nullptr) return 2;                                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                   \
        return baracuda::quantize::launch_quantize_per_tensor<TIN, TOUT>(              \
            static_cast<const TIN*>(x),                                                 \
            static_cast<TOUT*>(q),                                                      \
            numel, scale, zero_point, q_min, q_max, stream);                            \
    }                                                                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                         \
        int64_t numel,                                                                  \
        float /*scale*/,                                                                \
        int32_t /*zero_point*/,                                                         \
        int32_t q_min,                                                                  \
        int32_t q_max,                                                                  \
        const void* /*x*/,                                                              \
        const void* /*q*/)                                                              \
    {                                                                                   \
        if (numel < 0) return 2;                                                        \
        if (q_min > q_max) return 2;                                                    \
        return 0;                                                                       \
    }

// quantize_per_tensor — f64 variant (carries f64 scale).
#define BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_F64_INSTANTIATE(NAME, TOUT)                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                   \
        int64_t numel,                                                                   \
        double scale,                                                                    \
        int32_t zero_point,                                                              \
        int32_t q_min,                                                                   \
        int32_t q_max,                                                                   \
        const void* x,                                                                   \
        void* q,                                                                         \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                 \
        void* stream_ptr)                                                                \
    {                                                                                    \
        if (x == nullptr || q == nullptr) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                    \
        return baracuda::quantize::launch_quantize_per_tensor_f64<TOUT>(                \
            static_cast<const double*>(x),                                               \
            static_cast<TOUT*>(q),                                                       \
            numel, scale, zero_point, q_min, q_max, stream);                             \
    }                                                                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                          \
        int64_t numel,                                                                   \
        double /*scale*/,                                                                \
        int32_t /*zero_point*/,                                                          \
        int32_t q_min,                                                                   \
        int32_t q_max,                                                                   \
        const void* /*x*/,                                                               \
        const void* /*q*/)                                                               \
    {                                                                                    \
        if (numel < 0) return 2;                                                         \
        if (q_min > q_max) return 2;                                                     \
        return 0;                                                                        \
    }

// quantize_per_tensor BW — f32-scale variant.
#define BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_BW_INSTANTIATE(NAME, TIN)                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                    \
        int64_t numel,                                                                    \
        float scale,                                                                      \
        int32_t zero_point,                                                               \
        int32_t q_min,                                                                    \
        int32_t q_max,                                                                    \
        const void* x,                                                                    \
        const void* dy,                                                                   \
        void* dx,                                                                         \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                  \
        void* stream_ptr)                                                                 \
    {                                                                                     \
        if (x == nullptr || dy == nullptr || dx == nullptr) return 2;                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                     \
        return baracuda::quantize::launch_quantize_per_tensor_backward<TIN>(             \
            static_cast<const TIN*>(x),                                                   \
            static_cast<const TIN*>(dy),                                                  \
            static_cast<TIN*>(dx),                                                        \
            numel, scale, zero_point, q_min, q_max, stream);                              \
    }                                                                                     \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                           \
        int64_t numel,                                                                    \
        float /*scale*/,                                                                  \
        int32_t /*zero_point*/,                                                           \
        int32_t q_min,                                                                    \
        int32_t q_max,                                                                    \
        const void* /*x*/,                                                                \
        const void* /*dy*/,                                                               \
        const void* /*dx*/)                                                               \
    {                                                                                     \
        if (numel < 0) return 2;                                                          \
        if (q_min > q_max) return 2;                                                      \
        return 0;                                                                         \
    }

// quantize_per_tensor BW — f64 variant.
#define BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_BW_F64_INSTANTIATE(NAME)                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                     \
        int64_t numel,                                                                     \
        double scale,                                                                      \
        int32_t zero_point,                                                                \
        int32_t q_min,                                                                     \
        int32_t q_max,                                                                     \
        const void* x,                                                                     \
        const void* dy,                                                                    \
        void* dx,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                   \
        void* stream_ptr)                                                                  \
    {                                                                                      \
        if (x == nullptr || dy == nullptr || dx == nullptr) return 2;                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                      \
        return baracuda::quantize::launch_quantize_per_tensor_backward_f64(               \
            static_cast<const double*>(x),                                                 \
            static_cast<const double*>(dy),                                                \
            static_cast<double*>(dx),                                                      \
            numel, scale, zero_point, q_min, q_max, stream);                               \
    }                                                                                      \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                            \
        int64_t numel,                                                                     \
        double /*scale*/,                                                                  \
        int32_t /*zero_point*/,                                                            \
        int32_t q_min,                                                                     \
        int32_t q_max,                                                                     \
        const void* /*x*/,                                                                 \
        const void* /*dy*/,                                                                \
        const void* /*dx*/)                                                                \
    {                                                                                      \
        if (numel < 0) return 2;                                                           \
        if (q_min > q_max) return 2;                                                       \
        return 0;                                                                          \
    }

// dequantize_per_tensor — f32-scale variant.
#define BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_INSTANTIATE(NAME, TIN, TOUT)              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                     \
        int64_t numel,                                                                     \
        float scale,                                                                       \
        int32_t zero_point,                                                                \
        const void* q,                                                                     \
        void* x,                                                                           \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                   \
        void* stream_ptr)                                                                  \
    {                                                                                      \
        if (q == nullptr || x == nullptr) return 2;                                       \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                      \
        return baracuda::quantize::launch_dequantize_per_tensor<TIN, TOUT>(               \
            static_cast<const TOUT*>(q),                                                   \
            static_cast<TIN*>(x),                                                          \
            numel, scale, zero_point, stream);                                             \
    }                                                                                      \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                            \
        int64_t numel,                                                                     \
        float /*scale*/,                                                                   \
        int32_t /*zero_point*/,                                                            \
        const void* /*q*/,                                                                 \
        const void* /*x*/)                                                                 \
    {                                                                                      \
        if (numel < 0) return 2;                                                           \
        return 0;                                                                          \
    }

#define BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_F64_INSTANTIATE(NAME, TOUT)                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                      \
        int64_t numel,                                                                      \
        double scale,                                                                       \
        int32_t zero_point,                                                                 \
        const void* q,                                                                      \
        void* x,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                    \
        void* stream_ptr)                                                                   \
    {                                                                                       \
        if (q == nullptr || x == nullptr) return 2;                                        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                       \
        return baracuda::quantize::launch_dequantize_per_tensor_f64<TOUT>(                 \
            static_cast<const TOUT*>(q),                                                    \
            static_cast<double*>(x),                                                        \
            numel, scale, zero_point, stream);                                              \
    }                                                                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                             \
        int64_t numel,                                                                      \
        double /*scale*/,                                                                   \
        int32_t /*zero_point*/,                                                             \
        const void* /*q*/,                                                                  \
        const void* /*x*/)                                                                  \
    {                                                                                       \
        if (numel < 0) return 2;                                                            \
        return 0;                                                                           \
    }

// dequantize_per_tensor BW — dq = dy * scale. f32-scale variant.
#define BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_BW_INSTANTIATE(NAME, TIN)                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                      \
        int64_t numel,                                                                      \
        float scale,                                                                        \
        const void* dy,                                                                     \
        void* dq,                                                                           \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                    \
        void* stream_ptr)                                                                   \
    {                                                                                       \
        if (dy == nullptr || dq == nullptr) return 2;                                      \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                       \
        return baracuda::quantize::launch_dequantize_per_tensor_backward<TIN>(            \
            static_cast<const TIN*>(dy),                                                    \
            static_cast<TIN*>(dq),                                                          \
            numel, scale, stream);                                                          \
    }                                                                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                             \
        int64_t numel,                                                                      \
        float /*scale*/,                                                                    \
        const void* /*dy*/,                                                                 \
        const void* /*dq*/)                                                                 \
    {                                                                                       \
        if (numel < 0) return 2;                                                            \
        return 0;                                                                           \
    }

#define BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_BW_F64_INSTANTIATE(NAME)                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                       \
        int64_t numel,                                                                       \
        double scale,                                                                        \
        const void* dy,                                                                      \
        void* dq,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                     \
        void* stream_ptr)                                                                    \
    {                                                                                        \
        if (dy == nullptr || dq == nullptr) return 2;                                       \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                        \
        return baracuda::quantize::launch_dequantize_per_tensor_backward_f64(              \
            static_cast<const double*>(dy),                                                  \
            static_cast<double*>(dq),                                                        \
            numel, scale, stream);                                                           \
    }                                                                                        \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                              \
        int64_t numel,                                                                       \
        double /*scale*/,                                                                    \
        const void* /*dy*/,                                                                  \
        const void* /*dq*/)                                                                  \
    {                                                                                        \
        if (numel < 0) return 2;                                                             \
        return 0;                                                                            \
    }

// fake_quantize FW — f32-scale.
#define BARACUDA_KERNELS_FAKE_QUANTIZE_INSTANTIATE(NAME, TIN)                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                       \
        int64_t numel,                                                                       \
        float scale,                                                                         \
        int32_t zero_point,                                                                  \
        int32_t q_min,                                                                       \
        int32_t q_max,                                                                       \
        const void* x,                                                                       \
        void* y,                                                                             \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                     \
        void* stream_ptr)                                                                    \
    {                                                                                        \
        if (x == nullptr || y == nullptr) return 2;                                         \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                        \
        return baracuda::quantize::launch_fake_quantize<TIN>(                               \
            static_cast<const TIN*>(x),                                                      \
            static_cast<TIN*>(y),                                                            \
            numel, scale, zero_point, q_min, q_max, stream);                                 \
    }                                                                                        \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                              \
        int64_t numel,                                                                       \
        float /*scale*/,                                                                     \
        int32_t /*zero_point*/,                                                              \
        int32_t q_min,                                                                       \
        int32_t q_max,                                                                       \
        const void* /*x*/,                                                                   \
        const void* /*y*/)                                                                   \
    {                                                                                        \
        if (numel < 0) return 2;                                                             \
        if (q_min > q_max) return 2;                                                         \
        return 0;                                                                            \
    }

// fake_quantize FW — f64.
#define BARACUDA_KERNELS_FAKE_QUANTIZE_F64_INSTANTIATE(NAME)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                        \
        int64_t numel,                                                                        \
        double scale,                                                                         \
        int32_t zero_point,                                                                   \
        int32_t q_min,                                                                        \
        int32_t q_max,                                                                        \
        const void* x,                                                                        \
        void* y,                                                                              \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                      \
        void* stream_ptr)                                                                     \
    {                                                                                         \
        if (x == nullptr || y == nullptr) return 2;                                          \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                         \
        return baracuda::quantize::launch_fake_quantize_f64(                                 \
            static_cast<const double*>(x),                                                    \
            static_cast<double*>(y),                                                          \
            numel, scale, zero_point, q_min, q_max, stream);                                  \
    }                                                                                         \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                               \
        int64_t numel,                                                                        \
        double /*scale*/,                                                                     \
        int32_t /*zero_point*/,                                                               \
        int32_t q_min,                                                                        \
        int32_t q_max,                                                                        \
        const void* /*x*/,                                                                    \
        const void* /*y*/)                                                                    \
    {                                                                                         \
        if (numel < 0) return 2;                                                              \
        if (q_min > q_max) return 2;                                                          \
        return 0;                                                                             \
    }

// fake_quantize BW.
#define BARACUDA_KERNELS_FAKE_QUANTIZE_BW_INSTANTIATE(NAME, TIN)                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int64_t numel,                                                                          \
        float scale,                                                                            \
        int32_t zero_point,                                                                     \
        int32_t q_min,                                                                          \
        int32_t q_max,                                                                          \
        const void* x,                                                                          \
        const void* dy,                                                                         \
        void* dx,                                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                        \
        void* stream_ptr)                                                                       \
    {                                                                                           \
        if (x == nullptr || dy == nullptr || dx == nullptr) return 2;                          \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::quantize::launch_fake_quantize_backward<TIN>(                         \
            static_cast<const TIN*>(x),                                                         \
            static_cast<const TIN*>(dy),                                                        \
            static_cast<TIN*>(dx),                                                              \
            numel, scale, zero_point, q_min, q_max, stream);                                    \
    }                                                                                           \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                 \
        int64_t numel,                                                                          \
        float /*scale*/,                                                                        \
        int32_t /*zero_point*/,                                                                 \
        int32_t q_min,                                                                          \
        int32_t q_max,                                                                          \
        const void* /*x*/,                                                                      \
        const void* /*dy*/,                                                                     \
        const void* /*dx*/)                                                                     \
    {                                                                                           \
        if (numel < 0) return 2;                                                                \
        if (q_min > q_max) return 2;                                                            \
        return 0;                                                                               \
    }

#define BARACUDA_KERNELS_FAKE_QUANTIZE_BW_F64_INSTANTIATE(NAME)                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                             \
        int64_t numel,                                                                             \
        double scale,                                                                              \
        int32_t zero_point,                                                                        \
        int32_t q_min,                                                                             \
        int32_t q_max,                                                                             \
        const void* x,                                                                             \
        const void* dy,                                                                            \
        void* dx,                                                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                           \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (x == nullptr || dy == nullptr || dx == nullptr) return 2;                             \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                              \
        return baracuda::quantize::launch_fake_quantize_backward_f64(                             \
            static_cast<const double*>(x),                                                         \
            static_cast<const double*>(dy),                                                        \
            static_cast<double*>(dx),                                                              \
            numel, scale, zero_point, q_min, q_max, stream);                                       \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel,                                                                             \
        double /*scale*/,                                                                          \
        int32_t /*zero_point*/,                                                                    \
        int32_t q_min,                                                                             \
        int32_t q_max,                                                                             \
        const void* /*x*/,                                                                         \
        const void* /*dy*/,                                                                        \
        const void* /*dx*/)                                                                        \
    {                                                                                              \
        if (numel < 0) return 2;                                                                   \
        if (q_min > q_max) return 2;                                                               \
        return 0;                                                                                  \
    }

// Per-channel quantize FW.
#define BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_INSTANTIATE(NAME, TIN, TOUT)                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        const int32_t* shape4,                                                                      \
        int32_t axis,                                                                               \
        int32_t q_min,                                                                              \
        int32_t q_max,                                                                              \
        const void* x,                                                                              \
        const void* scale,                                                                          \
        const void* zero_point,                                                                     \
        void* q,                                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                               \
        if (x == nullptr || scale == nullptr || zero_point == nullptr ||                           \
            q == nullptr || shape4 == nullptr) return 2;                                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::quantize::launch_quantize_per_channel<TIN, TOUT>(                         \
            static_cast<const TIN*>(x),                                                             \
            static_cast<const TIN*>(scale),                                                         \
            static_cast<const int32_t*>(zero_point),                                                \
            static_cast<TOUT*>(q),                                                                  \
            numel, shape4, axis, q_min, q_max, stream);                                             \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                     \
        int64_t numel,                                                                              \
        const int32_t* shape4,                                                                      \
        int32_t axis,                                                                               \
        int32_t q_min,                                                                              \
        int32_t q_max,                                                                              \
        const void* /*x*/,                                                                          \
        const void* /*scale*/,                                                                      \
        const void* /*zero_point*/,                                                                 \
        const void* /*q*/)                                                                          \
    {                                                                                               \
        if (numel < 0) return 2;                                                                    \
        if (q_min > q_max) return 2;                                                                \
        if (axis < 0 || axis >= baracuda::quantize::MAX_RANK) return 2;                             \
        if (numel > 0 && shape4 == nullptr) return 2;                                               \
        return 0;                                                                                   \
    }

#define BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_F64_INSTANTIATE(NAME, TOUT)                           \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                                \
        int64_t numel,                                                                                \
        const int32_t* shape4,                                                                        \
        int32_t axis,                                                                                 \
        int32_t q_min,                                                                                \
        int32_t q_max,                                                                                \
        const void* x,                                                                                \
        const void* scale,                                                                            \
        const void* zero_point,                                                                       \
        void* q,                                                                                      \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                              \
        void* stream_ptr)                                                                             \
    {                                                                                                 \
        if (x == nullptr || scale == nullptr || zero_point == nullptr ||                             \
            q == nullptr || shape4 == nullptr) return 2;                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                 \
        return baracuda::quantize::launch_quantize_per_channel_f64<TOUT>(                            \
            static_cast<const double*>(x),                                                            \
            static_cast<const double*>(scale),                                                        \
            static_cast<const int32_t*>(zero_point),                                                  \
            static_cast<TOUT*>(q),                                                                    \
            numel, shape4, axis, q_min, q_max, stream);                                               \
    }                                                                                                 \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                       \
        int64_t numel,                                                                                \
        const int32_t* shape4,                                                                        \
        int32_t axis,                                                                                 \
        int32_t q_min,                                                                                \
        int32_t q_max,                                                                                \
        const void* /*x*/,                                                                            \
        const void* /*scale*/,                                                                        \
        const void* /*zero_point*/,                                                                   \
        const void* /*q*/)                                                                            \
    {                                                                                                 \
        if (numel < 0) return 2;                                                                      \
        if (q_min > q_max) return 2;                                                                  \
        if (axis < 0 || axis >= baracuda::quantize::MAX_RANK) return 2;                               \
        if (numel > 0 && shape4 == nullptr) return 2;                                                 \
        return 0;                                                                                     \
    }

// Per-channel quantize BW (STE).
#define BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_BW_INSTANTIATE(NAME, TIN)                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                                 \
        int64_t numel,                                                                                 \
        const int32_t* shape4,                                                                         \
        int32_t axis,                                                                                  \
        int32_t q_min,                                                                                 \
        int32_t q_max,                                                                                 \
        const void* x,                                                                                 \
        const void* scale,                                                                             \
        const void* zero_point,                                                                        \
        const void* dy,                                                                                \
        void* dx,                                                                                      \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                               \
        void* stream_ptr)                                                                              \
    {                                                                                                  \
        if (x == nullptr || scale == nullptr || zero_point == nullptr ||                              \
            dy == nullptr || dx == nullptr || shape4 == nullptr) return 2;                            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::quantize::launch_quantize_per_channel_backward<TIN>(                        \
            static_cast<const TIN*>(x),                                                                \
            static_cast<const TIN*>(scale),                                                            \
            static_cast<const int32_t*>(zero_point),                                                   \
            static_cast<const TIN*>(dy),                                                               \
            static_cast<TIN*>(dx),                                                                     \
            numel, shape4, axis, q_min, q_max, stream);                                                \
    }                                                                                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                        \
        int64_t numel,                                                                                 \
        const int32_t* shape4,                                                                         \
        int32_t axis,                                                                                  \
        int32_t q_min,                                                                                 \
        int32_t q_max,                                                                                 \
        const void* /*x*/,                                                                             \
        const void* /*scale*/,                                                                         \
        const void* /*zero_point*/,                                                                    \
        const void* /*dy*/,                                                                            \
        const void* /*dx*/)                                                                            \
    {                                                                                                  \
        if (numel < 0) return 2;                                                                       \
        if (q_min > q_max) return 2;                                                                   \
        if (axis < 0 || axis >= baracuda::quantize::MAX_RANK) return 2;                                \
        if (numel > 0 && shape4 == nullptr) return 2;                                                  \
        return 0;                                                                                      \
    }

#define BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_BW_F64_INSTANTIATE(NAME)                                 \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                                   \
        int64_t numel,                                                                                   \
        const int32_t* shape4,                                                                           \
        int32_t axis,                                                                                    \
        int32_t q_min,                                                                                   \
        int32_t q_max,                                                                                   \
        const void* x,                                                                                   \
        const void* scale,                                                                               \
        const void* zero_point,                                                                          \
        const void* dy,                                                                                  \
        void* dx,                                                                                        \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                                 \
        void* stream_ptr)                                                                                \
    {                                                                                                    \
        if (x == nullptr || scale == nullptr || zero_point == nullptr ||                                \
            dy == nullptr || dx == nullptr || shape4 == nullptr) return 2;                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                    \
        return baracuda::quantize::launch_quantize_per_channel_backward_f64(                           \
            static_cast<const double*>(x),                                                               \
            static_cast<const double*>(scale),                                                           \
            static_cast<const int32_t*>(zero_point),                                                     \
            static_cast<const double*>(dy),                                                              \
            static_cast<double*>(dx),                                                                    \
            numel, shape4, axis, q_min, q_max, stream);                                                  \
    }                                                                                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                          \
        int64_t numel,                                                                                   \
        const int32_t* shape4,                                                                           \
        int32_t axis,                                                                                    \
        int32_t q_min,                                                                                   \
        int32_t q_max,                                                                                   \
        const void* /*x*/,                                                                               \
        const void* /*scale*/,                                                                           \
        const void* /*zero_point*/,                                                                      \
        const void* /*dy*/,                                                                              \
        const void* /*dx*/)                                                                              \
    {                                                                                                    \
        if (numel < 0) return 2;                                                                         \
        if (q_min > q_max) return 2;                                                                     \
        if (axis < 0 || axis >= baracuda::quantize::MAX_RANK) return 2;                                  \
        if (numel > 0 && shape4 == nullptr) return 2;                                                    \
        return 0;                                                                                        \
    }

// Per-channel dequantize FW.
#define BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_INSTANTIATE(NAME, TIN, TOUT)                            \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                                    \
        int64_t numel,                                                                                    \
        const int32_t* shape4,                                                                            \
        int32_t axis,                                                                                     \
        const void* q,                                                                                    \
        const void* scale,                                                                                \
        const void* zero_point,                                                                           \
        void* x,                                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                                  \
        void* stream_ptr)                                                                                 \
    {                                                                                                     \
        if (q == nullptr || scale == nullptr || zero_point == nullptr ||                                 \
            x == nullptr || shape4 == nullptr) return 2;                                                  \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                     \
        return baracuda::quantize::launch_dequantize_per_channel<TIN, TOUT>(                            \
            static_cast<const TOUT*>(q),                                                                  \
            static_cast<const TIN*>(scale),                                                               \
            static_cast<const int32_t*>(zero_point),                                                      \
            static_cast<TIN*>(x),                                                                         \
            numel, shape4, axis, stream);                                                                 \
    }                                                                                                     \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                           \
        int64_t numel,                                                                                    \
        const int32_t* shape4,                                                                            \
        int32_t axis,                                                                                     \
        const void* /*q*/,                                                                                \
        const void* /*scale*/,                                                                            \
        const void* /*zero_point*/,                                                                       \
        const void* /*x*/)                                                                                \
    {                                                                                                     \
        if (numel < 0) return 2;                                                                          \
        if (axis < 0 || axis >= baracuda::quantize::MAX_RANK) return 2;                                   \
        if (numel > 0 && shape4 == nullptr) return 2;                                                     \
        return 0;                                                                                         \
    }

#define BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_F64_INSTANTIATE(NAME, TOUT)                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                                     \
        int64_t numel,                                                                                     \
        const int32_t* shape4,                                                                             \
        int32_t axis,                                                                                      \
        const void* q,                                                                                     \
        const void* scale,                                                                                 \
        const void* zero_point,                                                                            \
        void* x,                                                                                           \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                                   \
        void* stream_ptr)                                                                                  \
    {                                                                                                      \
        if (q == nullptr || scale == nullptr || zero_point == nullptr ||                                  \
            x == nullptr || shape4 == nullptr) return 2;                                                   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                      \
        return baracuda::quantize::launch_dequantize_per_channel_f64<TOUT>(                              \
            static_cast<const TOUT*>(q),                                                                   \
            static_cast<const double*>(scale),                                                             \
            static_cast<const int32_t*>(zero_point),                                                       \
            static_cast<double*>(x),                                                                       \
            numel, shape4, axis, stream);                                                                  \
    }                                                                                                      \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                            \
        int64_t numel,                                                                                     \
        const int32_t* shape4,                                                                             \
        int32_t axis,                                                                                      \
        const void* /*q*/,                                                                                 \
        const void* /*scale*/,                                                                             \
        const void* /*zero_point*/,                                                                        \
        const void* /*x*/)                                                                                 \
    {                                                                                                      \
        if (numel < 0) return 2;                                                                           \
        if (axis < 0 || axis >= baracuda::quantize::MAX_RANK) return 2;                                    \
        if (numel > 0 && shape4 == nullptr) return 2;                                                      \
        return 0;                                                                                          \
    }

// Per-channel dequantize BW — dq[i] = dy[i] * scale[c].
#define BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_BW_INSTANTIATE(NAME, TIN)                                 \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                                      \
        int64_t numel,                                                                                      \
        const int32_t* shape4,                                                                              \
        int32_t axis,                                                                                       \
        const void* scale,                                                                                  \
        const void* dy,                                                                                     \
        void* dq,                                                                                           \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                                    \
        void* stream_ptr)                                                                                   \
    {                                                                                                       \
        if (scale == nullptr || dy == nullptr || dq == nullptr || shape4 == nullptr) return 2;            \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                       \
        return baracuda::quantize::launch_dequantize_per_channel_backward<TIN>(                           \
            static_cast<const TIN*>(scale),                                                                 \
            static_cast<const TIN*>(dy),                                                                    \
            static_cast<TIN*>(dq),                                                                          \
            numel, shape4, axis, stream);                                                                   \
    }                                                                                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                             \
        int64_t numel,                                                                                      \
        const int32_t* shape4,                                                                              \
        int32_t axis,                                                                                       \
        const void* /*scale*/,                                                                              \
        const void* /*dy*/,                                                                                 \
        const void* /*dq*/)                                                                                 \
    {                                                                                                       \
        if (numel < 0) return 2;                                                                            \
        if (axis < 0 || axis >= baracuda::quantize::MAX_RANK) return 2;                                     \
        if (numel > 0 && shape4 == nullptr) return 2;                                                       \
        return 0;                                                                                           \
    }

#define BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_BW_F64_INSTANTIATE(NAME)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                                       \
        int64_t numel,                                                                                       \
        const int32_t* shape4,                                                                               \
        int32_t axis,                                                                                        \
        const void* scale,                                                                                   \
        const void* dy,                                                                                      \
        void* dq,                                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                                     \
        void* stream_ptr)                                                                                    \
    {                                                                                                        \
        if (scale == nullptr || dy == nullptr || dq == nullptr || shape4 == nullptr) return 2;             \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                        \
        return baracuda::quantize::launch_dequantize_per_channel_backward_f64(                             \
            static_cast<const double*>(scale),                                                               \
            static_cast<const double*>(dy),                                                                  \
            static_cast<double*>(dq),                                                                        \
            numel, shape4, axis, stream);                                                                    \
    }                                                                                                        \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                              \
        int64_t numel,                                                                                       \
        const int32_t* shape4,                                                                               \
        int32_t axis,                                                                                        \
        const void* /*scale*/,                                                                               \
        const void* /*dy*/,                                                                                  \
        const void* /*dq*/)                                                                                  \
    {                                                                                                        \
        if (numel < 0) return 2;                                                                             \
        if (axis < 0 || axis >= baracuda::quantize::MAX_RANK) return 2;                                      \
        if (numel > 0 && shape4 == nullptr) return 2;                                                        \
        return 0;                                                                                            \
    }

#endif // BARACUDA_QUANTIZE_CUH
