// baracuda_quantize_per_token_group.cuh
//
// Templated kernels and INSTANTIATE macros for per-token + per-group
// quantize / dequantize (Phase 8 Milestone 8.2 — sibling to 8.1's
// per-tensor / per-channel / fake_quantize header).
//
// Math reference:
//
//   per-token (input [N, D], one scale + zp per token row):
//     FW: q[n, d] = clamp(round(x[n, d] / scale[n]) + zp[n], qmin, qmax)
//     BW: dx[n, d] = dy[n, d] / scale[n]  (STE) * in_range_mask[n, d]
//     Dequant: y[n, d] = (x[n, d] - zp[n]) * scale[n]
//
//   per-group (input [outer, axis_size], `axis_size = group_size * num_groups`,
//              scale + zp shape `[outer, num_groups]`):
//     FW: g_idx = j / group_size
//         q[i, j] = clamp(round(x[i, j] / scale[i, g_idx]) + zp[i, g_idx],
//                         qmin, qmax)
//     BW (STE): same divisor-by-scale[g_idx], masked by in-range.
//     Dequant: y[i, j] = (x[i, j] - zp[i, g_idx]) * scale[i, g_idx]
//
// Dtype coverage: TIn ∈ {f32, f64, f16, bf16}; TOut ∈ {s8, u8} (int8).
// f16 / bf16 accumulate in f32 (round-half-to-even via `rintf` and
// saturation by clamping to [qmin, qmax]).
//
// Status codes mirror the rest of the kernel set:
//   0 success
//   1 misaligned operand (reserved)
//   2 invalid problem
//   3 unsupported (reserved)
//   4 workspace too small (reserved — these ops are workspace-free)
//   5 internal kernel error (typically a launch failure)
//
// Coordination with sibling 8.1 header (`baracuda_quantize.cuh`): we
// keep these in distinct files so the two milestones never collide on
// helper symbols. Any future shared utility (e.g. a saturating-round
// helper) can be lifted into `baracuda_dtype.cuh` — sibling already
// owns the int8 sat-cast primitives.

#ifndef BARACUDA_QUANTIZE_PER_TOKEN_GROUP_CUH
#define BARACUDA_QUANTIZE_PER_TOKEN_GROUP_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace quantize_ptg {

// =============================================================================
// dtype detour helpers — f32 path for f16 / bf16, native otherwise.
// =============================================================================
//
// Marked `__host__ __device__` even though present call sites only need
// the device half. The reason is the Milestone 7.6 segment-launcher
// footgun: NVCC silently emits a broken host stub if a `__device__`-
// only helper is referenced (even indirectly) from a `__host__ inline`
// launcher. Keeping these dual-mode means no host-side surprises when
// the launchers grow CPU-side validation in the future.

template <typename T>
__host__ __device__ __forceinline__ float qptg_load_as_f32(T x) {
    return (float)x;
}

template <>
__host__ __device__ __forceinline__ float qptg_load_as_f32<__half>(__half x) {
#if defined(__CUDA_ARCH__)
    return __half2float(x);
#else
    return (float)x;
#endif
}

template <>
__host__ __device__ __forceinline__ float qptg_load_as_f32<__nv_bfloat16>(
    __nv_bfloat16 x)
{
#if defined(__CUDA_ARCH__)
    return __bfloat162float(x);
#else
    return (float)x;
#endif
}

template <>
__host__ __device__ __forceinline__ float qptg_load_as_f32<double>(double x) {
    return (float)x;
}

// Convert f32 back to TIn (used by BW where we need to write dx as TIn).
template <typename T>
__host__ __device__ __forceinline__ T qptg_store_from_f32(float v) {
    return (T)v;
}

template <>
__host__ __device__ __forceinline__ __half qptg_store_from_f32<__half>(float v) {
#if defined(__CUDA_ARCH__)
    return __float2half(v);
#else
    return __half(v);
#endif
}

template <>
__host__ __device__ __forceinline__ __nv_bfloat16 qptg_store_from_f32<__nv_bfloat16>(
    float v)
{
#if defined(__CUDA_ARCH__)
    return __float2bfloat16(v);
#else
    return __nv_bfloat16(v);
#endif
}

template <>
__host__ __device__ __forceinline__ double qptg_store_from_f32<double>(float v) {
    return (double)v;
}

// Saturating round-to-nearest-int from an already-f32 value, clamping to
// [qmin, qmax]. Round-half-to-even via `rintf` matches `__float2int_rn`,
// `rust::f32::round_ties_even` and the existing int-GEMM sat-cast path.
__host__ __device__ __forceinline__ int32_t qptg_round_sat(
    float x, int32_t qmin, int32_t qmax)
{
#if defined(__CUDA_ARCH__)
    int32_t r = __float2int_rn(x);
#else
    int32_t r = (int32_t)nearbyintf(x);
#endif
    if (r < qmin) r = qmin;
    if (r > qmax) r = qmax;
    return r;
}

// Narrow an int32 quantized value to the output storage type. TOut is
// either int8_t (s8) or uint8_t (u8); the value has already been clamped
// to [qmin, qmax] by qptg_round_sat so a direct narrowing cast is safe.
template <typename TOut>
__host__ __device__ __forceinline__ TOut qptg_narrow_to_out(int32_t q) {
    return static_cast<TOut>(q);
}

// =============================================================================
// PER-TOKEN — input [N, D], one (scale_n, zp_n) per row.
// =============================================================================

template <typename TIn, typename TOut>
__global__ void quantize_per_token_kernel(
    const TIn*     __restrict__ input,         // [N, D]
    const TIn*     __restrict__ scale,         // [N]
    const int32_t* __restrict__ zero_point,    // [N]
    TOut*          __restrict__ output,        // [N, D]
    int32_t        N,
    int32_t        D,
    int32_t        qmin,
    int32_t        qmax)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        float x   = qptg_load_as_f32<TIn>(input[i]);
        float s   = qptg_load_as_f32<TIn>(scale[n]);
        int32_t zp = zero_point[n];
        // Divide-by-zero defense: kernel callers should not pass scale=0
        // (the safe-layer `can_implement` is free to reject it), but we
        // still guard so the kernel doesn't emit NaN -> sentinel garbage.
        float q_f = (s != 0.0f) ? (x / s) : 0.0f;
        int32_t q = qptg_round_sat(q_f + (float)zp, qmin, qmax);
        output[i] = qptg_narrow_to_out<TOut>(q);
    }
}

template <typename TIn>
__global__ void quantize_per_token_backward_kernel(
    const TIn*     __restrict__ d_output,      // [N, D]  (dy)
    const TIn*     __restrict__ input,         // [N, D]  (x — for in-range mask)
    const TIn*     __restrict__ scale,         // [N]
    const int32_t* __restrict__ zero_point,    // [N]
    TIn*           __restrict__ d_input,       // [N, D]  (dx)
    int32_t        N,
    int32_t        D,
    int32_t        qmin,
    int32_t        qmax)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        float x   = qptg_load_as_f32<TIn>(input[i]);
        float s   = qptg_load_as_f32<TIn>(scale[n]);
        int32_t zp = zero_point[n];
        // Recompute the un-clamped pre-rounding quant value to detect
        // whether the FW path saturated this cell. STE passes the
        // gradient through only when in-range.
        float q_f = (s != 0.0f) ? (x / s) : 0.0f;
        int32_t q = (int32_t)rintf(q_f) + zp;
        bool in_range = (q > qmin) && (q < qmax);
        // PyTorch convention: gradient is also passed at the boundary
        // (q == qmin || q == qmax) when the gradient would push the
        // value back into range. The STE form `1[qmin<q<qmax]` is the
        // simpler one and matches `torch.fake_quantize_per_tensor_affine`
        // back-prop in eager mode.
        float dy = qptg_load_as_f32<TIn>(d_output[i]);
        float dx_f = (in_range && s != 0.0f) ? (dy / s) : 0.0f;
        d_input[i] = qptg_store_from_f32<TIn>(dx_f);
    }
}

template <typename TIn, typename TOut>
__global__ void dequantize_per_token_kernel(
    const TOut*    __restrict__ input,         // [N, D]   (q)
    const TIn*     __restrict__ scale,         // [N]
    const int32_t* __restrict__ zero_point,    // [N]
    TIn*           __restrict__ output,        // [N, D]   (y)
    int32_t        N,
    int32_t        D)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        float q  = (float)(int32_t)input[i];
        float s  = qptg_load_as_f32<TIn>(scale[n]);
        float zp = (float)zero_point[n];
        float y  = (q - zp) * s;
        output[i] = qptg_store_from_f32<TIn>(y);
    }
}

template <typename TIn>
__global__ void dequantize_per_token_backward_kernel(
    const TIn*     __restrict__ d_output,      // [N, D]   (dy)
    const TIn*     __restrict__ scale,         // [N]
    TIn*           __restrict__ d_input,       // [N, D]   (dq stored as TIn —
                                                //          straight-through
                                                //          since dq/dy = scale[n])
    int32_t        N,
    int32_t        D)
{
    int64_t total = (int64_t)N * (int64_t)D;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t n = (int32_t)(i / (int64_t)D);
        float dy = qptg_load_as_f32<TIn>(d_output[i]);
        float s  = qptg_load_as_f32<TIn>(scale[n]);
        d_input[i] = qptg_store_from_f32<TIn>(dy * s);
    }
}

// =============================================================================
// PER-GROUP — input [outer, axis_size], scale/zp [outer, num_groups].
// =============================================================================
//
// Last-axis-only trailblazer: the quant axis MUST be the rightmost
// axis so the layout is naturally group-contiguous. Higher-rank tensors
// are flattened by the safe-plan layer into a 2-D `[outer, axis_size]`
// view before launching the kernel.

template <typename TIn, typename TOut>
__global__ void quantize_per_group_kernel(
    const TIn*     __restrict__ input,         // [outer, axis_size]
    const TIn*     __restrict__ scale,         // [outer, num_groups]
    const int32_t* __restrict__ zero_point,    // [outer, num_groups]
    TOut*          __restrict__ output,        // [outer, axis_size]
    int32_t        outer,
    int32_t        axis_size,
    int32_t        group_size,
    int32_t        num_groups,                 // axis_size / group_size
    int32_t        qmin,
    int32_t        qmax)
{
    int64_t total = (int64_t)outer * (int64_t)axis_size;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t out_idx = (int32_t)(i / (int64_t)axis_size);
        int32_t j       = (int32_t)(i - (int64_t)out_idx * (int64_t)axis_size);
        int32_t g_idx   = j / group_size;
        int64_t sg_off  = (int64_t)out_idx * (int64_t)num_groups + (int64_t)g_idx;
        float x  = qptg_load_as_f32<TIn>(input[i]);
        float s  = qptg_load_as_f32<TIn>(scale[sg_off]);
        int32_t zp = zero_point[sg_off];
        float q_f = (s != 0.0f) ? (x / s) : 0.0f;
        int32_t q = qptg_round_sat(q_f + (float)zp, qmin, qmax);
        output[i] = qptg_narrow_to_out<TOut>(q);
    }
}

template <typename TIn>
__global__ void quantize_per_group_backward_kernel(
    const TIn*     __restrict__ d_output,      // [outer, axis_size]
    const TIn*     __restrict__ input,         // [outer, axis_size]
    const TIn*     __restrict__ scale,         // [outer, num_groups]
    const int32_t* __restrict__ zero_point,    // [outer, num_groups]
    TIn*           __restrict__ d_input,       // [outer, axis_size]
    int32_t        outer,
    int32_t        axis_size,
    int32_t        group_size,
    int32_t        num_groups,
    int32_t        qmin,
    int32_t        qmax)
{
    int64_t total = (int64_t)outer * (int64_t)axis_size;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t out_idx = (int32_t)(i / (int64_t)axis_size);
        int32_t j       = (int32_t)(i - (int64_t)out_idx * (int64_t)axis_size);
        int32_t g_idx   = j / group_size;
        int64_t sg_off  = (int64_t)out_idx * (int64_t)num_groups + (int64_t)g_idx;
        float x  = qptg_load_as_f32<TIn>(input[i]);
        float s  = qptg_load_as_f32<TIn>(scale[sg_off]);
        int32_t zp = zero_point[sg_off];
        float q_f = (s != 0.0f) ? (x / s) : 0.0f;
        int32_t q = (int32_t)rintf(q_f) + zp;
        bool in_range = (q > qmin) && (q < qmax);
        float dy = qptg_load_as_f32<TIn>(d_output[i]);
        float dx_f = (in_range && s != 0.0f) ? (dy / s) : 0.0f;
        d_input[i] = qptg_store_from_f32<TIn>(dx_f);
    }
}

template <typename TIn, typename TOut>
__global__ void dequantize_per_group_kernel(
    const TOut*    __restrict__ input,         // [outer, axis_size] (q)
    const TIn*     __restrict__ scale,         // [outer, num_groups]
    const int32_t* __restrict__ zero_point,    // [outer, num_groups]
    TIn*           __restrict__ output,        // [outer, axis_size]
    int32_t        outer,
    int32_t        axis_size,
    int32_t        group_size,
    int32_t        num_groups)
{
    int64_t total = (int64_t)outer * (int64_t)axis_size;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t out_idx = (int32_t)(i / (int64_t)axis_size);
        int32_t j       = (int32_t)(i - (int64_t)out_idx * (int64_t)axis_size);
        int32_t g_idx   = j / group_size;
        int64_t sg_off  = (int64_t)out_idx * (int64_t)num_groups + (int64_t)g_idx;
        float q  = (float)(int32_t)input[i];
        float s  = qptg_load_as_f32<TIn>(scale[sg_off]);
        float zp = (float)zero_point[sg_off];
        float y  = (q - zp) * s;
        output[i] = qptg_store_from_f32<TIn>(y);
    }
}

template <typename TIn>
__global__ void dequantize_per_group_backward_kernel(
    const TIn*     __restrict__ d_output,      // [outer, axis_size]
    const TIn*     __restrict__ scale,         // [outer, num_groups]
    TIn*           __restrict__ d_input,       // [outer, axis_size]
    int32_t        outer,
    int32_t        axis_size,
    int32_t        group_size,
    int32_t        num_groups)
{
    int64_t total = (int64_t)outer * (int64_t)axis_size;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        int32_t out_idx = (int32_t)(i / (int64_t)axis_size);
        int32_t j       = (int32_t)(i - (int64_t)out_idx * (int64_t)axis_size);
        int32_t g_idx   = j / group_size;
        int64_t sg_off  = (int64_t)out_idx * (int64_t)num_groups + (int64_t)g_idx;
        float dy = qptg_load_as_f32<TIn>(d_output[i]);
        float s  = qptg_load_as_f32<TIn>(scale[sg_off]);
        d_input[i] = qptg_store_from_f32<TIn>(dy * s);
    }
}

// =============================================================================
// LAUNCH WRAPPERS
// =============================================================================
//
// Grid sizing matches the segment / embedding family: 256 threads/block,
// stride loop over `total` cells, capped at 65535 blocks so we never
// exceed the legacy grid limit even on huge `[N, D]` problems.

template <typename TIn, typename TOut>
__host__ inline int32_t launch_quantize_per_token(
    const TIn* input, const TIn* scale, const int32_t* zp, TOut* output,
    int32_t N, int32_t D, int32_t qmin, int32_t qmax,
    cudaStream_t stream)
{
    if (N < 0 || D < 0) return 2;
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (input == nullptr || scale == nullptr || zp == nullptr || output == nullptr)
        return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_token_kernel<TIn, TOut><<<blocks, kBlock, 0, stream>>>(
        input, scale, zp, output, N, D, qmin, qmax);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TIn>
__host__ inline int32_t launch_quantize_per_token_backward(
    const TIn* d_output, const TIn* input,
    const TIn* scale, const int32_t* zp, TIn* d_input,
    int32_t N, int32_t D, int32_t qmin, int32_t qmax,
    cudaStream_t stream)
{
    if (N < 0 || D < 0) return 2;
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (d_output == nullptr || input == nullptr || scale == nullptr ||
        zp == nullptr || d_input == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_token_backward_kernel<TIn><<<blocks, kBlock, 0, stream>>>(
        d_output, input, scale, zp, d_input, N, D, qmin, qmax);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TIn, typename TOut>
__host__ inline int32_t launch_dequantize_per_token(
    const TOut* input, const TIn* scale, const int32_t* zp, TIn* output,
    int32_t N, int32_t D, cudaStream_t stream)
{
    if (N < 0 || D < 0) return 2;
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (input == nullptr || scale == nullptr || zp == nullptr || output == nullptr)
        return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_token_kernel<TIn, TOut><<<blocks, kBlock, 0, stream>>>(
        input, scale, zp, output, N, D);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TIn>
__host__ inline int32_t launch_dequantize_per_token_backward(
    const TIn* d_output, const TIn* scale, TIn* d_input,
    int32_t N, int32_t D, cudaStream_t stream)
{
    if (N < 0 || D < 0) return 2;
    int64_t total = (int64_t)N * (int64_t)D;
    if (total == 0) return 0;
    if (d_output == nullptr || scale == nullptr || d_input == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_token_backward_kernel<TIn><<<blocks, kBlock, 0, stream>>>(
        d_output, scale, d_input, N, D);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TIn, typename TOut>
__host__ inline int32_t launch_quantize_per_group(
    const TIn* input, const TIn* scale, const int32_t* zp, TOut* output,
    int32_t outer, int32_t axis_size, int32_t group_size,
    int32_t qmin, int32_t qmax, cudaStream_t stream)
{
    if (outer < 0 || axis_size < 0 || group_size <= 0) return 2;
    if (axis_size % group_size != 0) return 2;
    int32_t num_groups = axis_size / group_size;
    int64_t total = (int64_t)outer * (int64_t)axis_size;
    if (total == 0) return 0;
    if (input == nullptr || scale == nullptr || zp == nullptr || output == nullptr)
        return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_group_kernel<TIn, TOut><<<blocks, kBlock, 0, stream>>>(
        input, scale, zp, output,
        outer, axis_size, group_size, num_groups, qmin, qmax);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TIn>
__host__ inline int32_t launch_quantize_per_group_backward(
    const TIn* d_output, const TIn* input,
    const TIn* scale, const int32_t* zp, TIn* d_input,
    int32_t outer, int32_t axis_size, int32_t group_size,
    int32_t qmin, int32_t qmax, cudaStream_t stream)
{
    if (outer < 0 || axis_size < 0 || group_size <= 0) return 2;
    if (axis_size % group_size != 0) return 2;
    int32_t num_groups = axis_size / group_size;
    int64_t total = (int64_t)outer * (int64_t)axis_size;
    if (total == 0) return 0;
    if (d_output == nullptr || input == nullptr || scale == nullptr ||
        zp == nullptr || d_input == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    quantize_per_group_backward_kernel<TIn><<<blocks, kBlock, 0, stream>>>(
        d_output, input, scale, zp, d_input,
        outer, axis_size, group_size, num_groups, qmin, qmax);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TIn, typename TOut>
__host__ inline int32_t launch_dequantize_per_group(
    const TOut* input, const TIn* scale, const int32_t* zp, TIn* output,
    int32_t outer, int32_t axis_size, int32_t group_size,
    cudaStream_t stream)
{
    if (outer < 0 || axis_size < 0 || group_size <= 0) return 2;
    if (axis_size % group_size != 0) return 2;
    int32_t num_groups = axis_size / group_size;
    int64_t total = (int64_t)outer * (int64_t)axis_size;
    if (total == 0) return 0;
    if (input == nullptr || scale == nullptr || zp == nullptr || output == nullptr)
        return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_group_kernel<TIn, TOut><<<blocks, kBlock, 0, stream>>>(
        input, scale, zp, output, outer, axis_size, group_size, num_groups);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename TIn>
__host__ inline int32_t launch_dequantize_per_group_backward(
    const TIn* d_output, const TIn* scale, TIn* d_input,
    int32_t outer, int32_t axis_size, int32_t group_size,
    cudaStream_t stream)
{
    if (outer < 0 || axis_size < 0 || group_size <= 0) return 2;
    if (axis_size % group_size != 0) return 2;
    int32_t num_groups = axis_size / group_size;
    int64_t total = (int64_t)outer * (int64_t)axis_size;
    if (total == 0) return 0;
    if (d_output == nullptr || scale == nullptr || d_input == nullptr) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    dequantize_per_group_backward_kernel<TIn><<<blocks, kBlock, 0, stream>>>(
        d_output, scale, d_input, outer, axis_size, group_size, num_groups);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::quantize_ptg

// =============================================================================
// INSTANTIATE macros — one `extern "C"` launcher per (op, TIn, TOut) tuple.
// =============================================================================

#define BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_INSTANTIATE(NAME, TIN, TOUT)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t N, int32_t D, int32_t qmin, int32_t qmax,                                          \
        const void* input,                                                                          \
        const void* scale,                                                                          \
        const void* zero_point,                                                                     \
        void* output,                                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::quantize_ptg::launch_quantize_per_token<TIN, TOUT>(                        \
            static_cast<const TIN*>(input),                                                         \
            static_cast<const TIN*>(scale),                                                         \
            static_cast<const int32_t*>(zero_point),                                                \
            static_cast<TOUT*>(output),                                                             \
            N, D, qmin, qmax, stream);                                                              \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int32_t N, int32_t D, int32_t qmin, int32_t qmax,                                          \
        const void* /*input*/,                                                                     \
        const void* /*scale*/,                                                                     \
        const void* /*zero_point*/,                                                                \
        const void* /*output*/)                                                                    \
    {                                                                                              \
        if (N < 0 || D < 0) return 2;                                                              \
        if (qmin > qmax) return 2;                                                                 \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_BACKWARD_INSTANTIATE(NAME, TIN)                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t N, int32_t D, int32_t qmin, int32_t qmax,                                          \
        const void* d_output,                                                                       \
        const void* input,                                                                          \
        const void* scale,                                                                          \
        const void* zero_point,                                                                     \
        void* d_input,                                                                              \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::quantize_ptg::launch_quantize_per_token_backward<TIN>(                     \
            static_cast<const TIN*>(d_output),                                                      \
            static_cast<const TIN*>(input),                                                         \
            static_cast<const TIN*>(scale),                                                         \
            static_cast<const int32_t*>(zero_point),                                                \
            static_cast<TIN*>(d_input),                                                             \
            N, D, qmin, qmax, stream);                                                              \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int32_t N, int32_t D, int32_t qmin, int32_t qmax,                                          \
        const void* /*d_output*/,                                                                  \
        const void* /*input*/,                                                                     \
        const void* /*scale*/,                                                                     \
        const void* /*zero_point*/,                                                                \
        const void* /*d_input*/)                                                                   \
    {                                                                                              \
        if (N < 0 || D < 0) return 2;                                                              \
        if (qmin > qmax) return 2;                                                                 \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_INSTANTIATE(NAME, TIN, TOUT)                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t N, int32_t D,                                                                       \
        const void* input,                                                                          \
        const void* scale,                                                                          \
        const void* zero_point,                                                                     \
        void* output,                                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::quantize_ptg::launch_dequantize_per_token<TIN, TOUT>(                      \
            static_cast<const TOUT*>(input),                                                        \
            static_cast<const TIN*>(scale),                                                         \
            static_cast<const int32_t*>(zero_point),                                                \
            static_cast<TIN*>(output),                                                              \
            N, D, stream);                                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int32_t N, int32_t D,                                                                      \
        const void* /*input*/,                                                                     \
        const void* /*scale*/,                                                                     \
        const void* /*zero_point*/,                                                                \
        const void* /*output*/)                                                                    \
    {                                                                                              \
        if (N < 0 || D < 0) return 2;                                                              \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_BACKWARD_INSTANTIATE(NAME, TIN)                     \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t N, int32_t D,                                                                       \
        const void* d_output,                                                                       \
        const void* scale,                                                                          \
        void* d_input,                                                                              \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::quantize_ptg::launch_dequantize_per_token_backward<TIN>(                   \
            static_cast<const TIN*>(d_output),                                                      \
            static_cast<const TIN*>(scale),                                                         \
            static_cast<TIN*>(d_input),                                                             \
            N, D, stream);                                                                          \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int32_t N, int32_t D,                                                                      \
        const void* /*d_output*/,                                                                  \
        const void* /*scale*/,                                                                     \
        const void* /*d_input*/)                                                                   \
    {                                                                                              \
        if (N < 0 || D < 0) return 2;                                                              \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_QUANTIZE_PER_GROUP_INSTANTIATE(NAME, TIN, TOUT)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t outer, int32_t axis_size, int32_t group_size,                                       \
        int32_t qmin, int32_t qmax,                                                                 \
        const void* input,                                                                          \
        const void* scale,                                                                          \
        const void* zero_point,                                                                     \
        void* output,                                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::quantize_ptg::launch_quantize_per_group<TIN, TOUT>(                        \
            static_cast<const TIN*>(input),                                                         \
            static_cast<const TIN*>(scale),                                                         \
            static_cast<const int32_t*>(zero_point),                                                \
            static_cast<TOUT*>(output),                                                             \
            outer, axis_size, group_size, qmin, qmax, stream);                                      \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int32_t outer, int32_t axis_size, int32_t group_size,                                      \
        int32_t qmin, int32_t qmax,                                                                \
        const void* /*input*/,                                                                     \
        const void* /*scale*/,                                                                     \
        const void* /*zero_point*/,                                                                \
        const void* /*output*/)                                                                    \
    {                                                                                              \
        if (outer < 0 || axis_size < 0) return 2;                                                  \
        if (group_size <= 0) return 2;                                                             \
        if (axis_size % group_size != 0) return 2;                                                 \
        if (qmin > qmax) return 2;                                                                 \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_QUANTIZE_PER_GROUP_BACKWARD_INSTANTIATE(NAME, TIN)                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t outer, int32_t axis_size, int32_t group_size,                                       \
        int32_t qmin, int32_t qmax,                                                                 \
        const void* d_output,                                                                       \
        const void* input,                                                                          \
        const void* scale,                                                                          \
        const void* zero_point,                                                                     \
        void* d_input,                                                                              \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::quantize_ptg::launch_quantize_per_group_backward<TIN>(                     \
            static_cast<const TIN*>(d_output),                                                      \
            static_cast<const TIN*>(input),                                                         \
            static_cast<const TIN*>(scale),                                                         \
            static_cast<const int32_t*>(zero_point),                                                \
            static_cast<TIN*>(d_input),                                                             \
            outer, axis_size, group_size, qmin, qmax, stream);                                      \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int32_t outer, int32_t axis_size, int32_t group_size,                                      \
        int32_t qmin, int32_t qmax,                                                                \
        const void* /*d_output*/,                                                                  \
        const void* /*input*/,                                                                     \
        const void* /*scale*/,                                                                     \
        const void* /*zero_point*/,                                                                \
        const void* /*d_input*/)                                                                   \
    {                                                                                              \
        if (outer < 0 || axis_size < 0) return 2;                                                  \
        if (group_size <= 0) return 2;                                                             \
        if (axis_size % group_size != 0) return 2;                                                 \
        if (qmin > qmax) return 2;                                                                 \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_INSTANTIATE(NAME, TIN, TOUT)                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t outer, int32_t axis_size, int32_t group_size,                                       \
        const void* input,                                                                          \
        const void* scale,                                                                          \
        const void* zero_point,                                                                     \
        void* output,                                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::quantize_ptg::launch_dequantize_per_group<TIN, TOUT>(                      \
            static_cast<const TOUT*>(input),                                                        \
            static_cast<const TIN*>(scale),                                                         \
            static_cast<const int32_t*>(zero_point),                                                \
            static_cast<TIN*>(output),                                                              \
            outer, axis_size, group_size, stream);                                                  \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int32_t outer, int32_t axis_size, int32_t group_size,                                      \
        const void* /*input*/,                                                                     \
        const void* /*scale*/,                                                                     \
        const void* /*zero_point*/,                                                                \
        const void* /*output*/)                                                                    \
    {                                                                                              \
        if (outer < 0 || axis_size < 0) return 2;                                                  \
        if (group_size <= 0) return 2;                                                             \
        if (axis_size % group_size != 0) return 2;                                                 \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_BACKWARD_INSTANTIATE(NAME, TIN)                     \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int32_t outer, int32_t axis_size, int32_t group_size,                                       \
        const void* d_output,                                                                       \
        const void* scale,                                                                          \
        void* d_input,                                                                              \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                            \
        void* stream_ptr)                                                                           \
    {                                                                                              \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        return baracuda::quantize_ptg::launch_dequantize_per_group_backward<TIN>(                   \
            static_cast<const TIN*>(d_output),                                                      \
            static_cast<const TIN*>(scale),                                                         \
            static_cast<TIN*>(d_input),                                                             \
            outer, axis_size, group_size, stream);                                                  \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int32_t outer, int32_t axis_size, int32_t group_size,                                      \
        const void* /*d_output*/,                                                                  \
        const void* /*scale*/,                                                                     \
        const void* /*d_input*/)                                                                   \
    {                                                                                              \
        if (outer < 0 || axis_size < 0) return 2;                                                  \
        if (group_size <= 0) return 2;                                                             \
        if (axis_size % group_size != 0) return 2;                                                 \
        return 0;                                                                                  \
    }

#endif // BARACUDA_QUANTIZE_PER_TOKEN_GROUP_CUH
