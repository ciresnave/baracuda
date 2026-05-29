// baracuda-kernels Phase 50 — causal-conv1d forward (FP types).
//
// Hand-port of Tri Dao's causal-conv1d primitive (BSD-3-Clause).
// See `vendor/causal-conv1d/VENDOR.md` for upstream attribution.
//
// Algorithm — depthwise causal 1-D convolution:
//   For input  x : [B, C, L] (channels-second, contiguous row-major)
//   And weight w : [C, W]    (per-channel filter)
//   And bias   b : [C] or null
//   Compute output y : [B, C, L] s.t.
//     y[b, c, t] = act( sum_{k=0..W-1} w[c, k] * x_padded[b, c, t - (W - 1 - k)]
//                       + bias[c] )
//   where x_padded is x with (W-1) zeros prepended along the L axis.
//   The "causal" property: y[t] depends only on x[<=t].
//
// Width restriction: W ∈ {2, 3, 4} (matches upstream's fast path).
//
// Activation: passed via int flag (0 = none, 1 = SiLU). SiLU is the
// default in Mamba / Mamba-2.
//
// Layout: NCL (channels-second, like cuDNN's NCHW convention). One
// thread per (b, c, t) output cell — embarrassingly parallel, no
// inter-thread communication.
//
// f16 / bf16 detour through f32 for the convolution accumulation and
// the SiLU activation; f64 uses native double throughout.

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace causal_conv1d {

// =========================================================================
// dtype helpers — match the convention used in baracuda_attention.cuh.
// =========================================================================

template <typename T>
__device__ __forceinline__ float load_as_f32(T x) { return (float)x; }

template <>
__device__ __forceinline__ float load_as_f32<__half>(__half x) {
    return __half2float(x);
}

template <>
__device__ __forceinline__ float load_as_f32<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ T store_from_f32(float v) { return (T)v; }

template <>
__device__ __forceinline__ __half store_from_f32<__half>(float v) {
    return __float2half(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_from_f32<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// f64 specialisation — keep the accumulator in double.
template <>
__device__ __forceinline__ float load_as_f32<double>(double x) {
    return (float)x;  // narrowing — only used by f32 detour callers
}

template <>
__device__ __forceinline__ double store_from_f32<double>(float v) {
    return (double)v;
}

// SiLU activation: y = x * sigmoid(x).
__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ double silu_f64(double x) {
    return x / (1.0 + exp(-x));
}

// SiLU derivative: d/dx silu(x) = silu(x) + sigmoid(x) * (1 - silu(x))
//                                = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// We compute via sigma = 1 / (1 + exp(-x)); silu = x * sigma;
//   d_silu = sigma + silu * (1 - sigma)
__device__ __forceinline__ float silu_grad_f32(float x) {
    float sigma = 1.0f / (1.0f + __expf(-x));
    float silu  = x * sigma;
    return sigma + silu * (1.0f - sigma);
}

__device__ __forceinline__ double silu_grad_f64(double x) {
    double sigma = 1.0 / (1.0 + exp(-x));
    double silu  = x * sigma;
    return sigma + silu * (1.0 - sigma);
}

// =========================================================================
// FW kernel — depthwise causal conv1d
// =========================================================================
//
// One thread per (b, c, t) output cell. Total threads = B * C * L.
// We launch a 1D grid for simplicity; the unravel cost is amortised
// since each thread does O(W) work after.
//
// Numerical contract: accumulator is f32 for {f32, f16, bf16} and
// f64 for f64.

template <typename T, int W>
__global__ void causal_conv1d_fwd_kernel(
    const T* __restrict__ x,        // [B, C, L]
    const T* __restrict__ weight,   // [C, W]
    const T* __restrict__ bias,     // [C] or null
    T* __restrict__ y,              // [B, C, L]
    int32_t batch,
    int32_t channels,
    int32_t seqlen,
    int32_t use_silu)
{
    const int64_t total = (int64_t)batch * channels * seqlen;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int32_t t = (int32_t)(tid % seqlen);
    const int32_t c = (int32_t)((tid / seqlen) % channels);
    const int32_t b = (int32_t)(tid / ((int64_t)seqlen * channels));

    const int64_t bc_off = ((int64_t)b * channels + c) * seqlen;

    // Pre-load the per-channel filter into registers (W <= 4 fits easily).
    float w_f32[W];
    #pragma unroll
    for (int k = 0; k < W; ++k) {
        w_f32[k] = load_as_f32<T>(weight[c * W + k]);
    }

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < W; ++k) {
        // y[t] = sum_k w[c, k] * x_padded[t - (W - 1 - k)]
        // i.e. x index = t - (W - 1 - k). Out of bounds → 0.
        const int32_t xi = t - (W - 1 - k);
        if (xi >= 0) {
            acc += w_f32[k] * load_as_f32<T>(x[bc_off + xi]);
        }
    }

    if (bias != nullptr) {
        acc += load_as_f32<T>(bias[c]);
    }

    if (use_silu) {
        acc = silu_f32(acc);
    }

    y[bc_off + t] = store_from_f32<T>(acc);
}

// f64 specialisation — keep accumulator in double.
template <int W>
__global__ void causal_conv1d_fwd_kernel_f64(
    const double* __restrict__ x,
    const double* __restrict__ weight,
    const double* __restrict__ bias,
    double* __restrict__ y,
    int32_t batch,
    int32_t channels,
    int32_t seqlen,
    int32_t use_silu)
{
    const int64_t total = (int64_t)batch * channels * seqlen;
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int32_t t = (int32_t)(tid % seqlen);
    const int32_t c = (int32_t)((tid / seqlen) % channels);
    const int32_t b = (int32_t)(tid / ((int64_t)seqlen * channels));

    const int64_t bc_off = ((int64_t)b * channels + c) * seqlen;

    double w_f64[W];
    #pragma unroll
    for (int k = 0; k < W; ++k) {
        w_f64[k] = weight[c * W + k];
    }

    double acc = 0.0;
    #pragma unroll
    for (int k = 0; k < W; ++k) {
        const int32_t xi = t - (W - 1 - k);
        if (xi >= 0) {
            acc += w_f64[k] * x[bc_off + xi];
        }
    }

    if (bias != nullptr) {
        acc += bias[c];
    }

    if (use_silu) {
        acc = silu_f64(acc);
    }

    y[bc_off + t] = acc;
}

// Host-side launcher.
template <typename T>
int32_t launch_causal_conv1d_fwd(
    const T* x, const T* weight, const T* bias,
    T* y, int32_t batch, int32_t channels, int32_t seqlen, int32_t width,
    int32_t use_silu, cudaStream_t stream)
{
    if (width < 2 || width > 4) return 3;  // unsupported width
    if (batch == 0 || channels == 0 || seqlen == 0) return 0;

    const int64_t total = (int64_t)batch * channels * seqlen;
    const int32_t threads_per_block = 256;
    const int64_t blocks = (total + threads_per_block - 1) / threads_per_block;
    if (blocks > (int64_t)0x7FFFFFFF) return 3;
    dim3 grid((unsigned)blocks);
    dim3 block(threads_per_block);

    if constexpr (sizeof(T) == sizeof(double)) {
        // f64 path
        if (width == 2) causal_conv1d_fwd_kernel_f64<2><<<grid, block, 0, stream>>>(
            (const double*)x, (const double*)weight, (const double*)bias,
            (double*)y, batch, channels, seqlen, use_silu);
        else if (width == 3) causal_conv1d_fwd_kernel_f64<3><<<grid, block, 0, stream>>>(
            (const double*)x, (const double*)weight, (const double*)bias,
            (double*)y, batch, channels, seqlen, use_silu);
        else /* 4 */ causal_conv1d_fwd_kernel_f64<4><<<grid, block, 0, stream>>>(
            (const double*)x, (const double*)weight, (const double*)bias,
            (double*)y, batch, channels, seqlen, use_silu);
    } else {
        if (width == 2) causal_conv1d_fwd_kernel<T, 2><<<grid, block, 0, stream>>>(
            x, weight, bias, y, batch, channels, seqlen, use_silu);
        else if (width == 3) causal_conv1d_fwd_kernel<T, 3><<<grid, block, 0, stream>>>(
            x, weight, bias, y, batch, channels, seqlen, use_silu);
        else /* 4 */ causal_conv1d_fwd_kernel<T, 4><<<grid, block, 0, stream>>>(
            x, weight, bias, y, batch, channels, seqlen, use_silu);
    }

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}}  // namespace baracuda::causal_conv1d

// =========================================================================
// extern "C" FFI surface
// =========================================================================
//
// Status codes match baracuda convention:
//   0 ok, 2 invalid_problem, 3 unsupported, 5 launch_failure.

#define BARACUDA_CAUSAL_CONV1D_FWD_INSTANTIATE(NAME, T)                                       \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                         \
        int32_t batch, int32_t channels, int32_t seqlen, int32_t width,                       \
        int32_t use_silu,                                                                     \
        const void* x, const void* weight, const void* bias,                                  \
        void* y,                                                                              \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                      \
        void* stream_ptr)                                                                     \
    {                                                                                          \
        if (batch < 0 || channels < 0 || seqlen < 0) return 2;                                \
        if (width < 2 || width > 4) return 3;                                                 \
        if (batch == 0 || channels == 0 || seqlen == 0) return 0;                             \
        if (x == nullptr || weight == nullptr || y == nullptr) return 2;                      \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                          \
        return baracuda::causal_conv1d::launch_causal_conv1d_fwd<T>(                          \
            static_cast<const T*>(x),                                                         \
            static_cast<const T*>(weight),                                                    \
            static_cast<const T*>(bias),                                                      \
            static_cast<T*>(y),                                                               \
            batch, channels, seqlen, width, use_silu, stream);                                \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                               \
        int32_t batch, int32_t channels, int32_t seqlen, int32_t width)                       \
    {                                                                                          \
        if (batch < 0 || channels < 0 || seqlen < 0) return 2;                                \
        if (width < 2 || width > 4) return 3;                                                 \
        return 0;                                                                              \
    }

BARACUDA_CAUSAL_CONV1D_FWD_INSTANTIATE(causal_conv1d_f32,  float)
BARACUDA_CAUSAL_CONV1D_FWD_INSTANTIATE(causal_conv1d_f16,  __half)
BARACUDA_CAUSAL_CONV1D_FWD_INSTANTIATE(causal_conv1d_bf16, __nv_bfloat16)
BARACUDA_CAUSAL_CONV1D_FWD_INSTANTIATE(causal_conv1d_f64,  double)
