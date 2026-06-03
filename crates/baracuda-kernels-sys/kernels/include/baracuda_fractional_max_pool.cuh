// baracuda_fractional_max_pool.cuh
//
// Phase 16.3 — Fractional max-pool (2-D + 3-D) bespoke kernels.
//
// Replaces the Phase 11.8 stubs (`Error::Unsupported`) for
// `FractionalMaxPool2dPlan` / `FractionalMaxPool3dPlan`. cuDNN has no
// fractional-pool primitive — the kernel-window positions are sampled
// pseudorandomly per output cell, so a uniform-window cuDNN call doesn't
// apply.
//
// **Window-placement formula.** baracuda uses the "evenly-spaced base
// position + per-output-cell α perturbation" variant, NOT PyTorch's exact
// start_index/end_index sequence derivation. The PyTorch C++ source
// formula in `aten/src/ATen/native/AdaptivePooling.cpp` is clean
// conceptually but messy to reproduce bit-exactly (and depends on a CPU
// path's numerical reduction). The approximation we ship:
//
//     base[i]  = i * (H_in - kh) / (H_out - 1)      (for H_out > 1)
//     start[i] = floor(base[i] + α[n, c, axis])
//     start[i] = clamp(start[i], 0, H_in - kh)
//
//   where α[n, c, axis] ∈ [0, 1) is the caller-provided sample. For
//   H_out == 1 we collapse to `start[0] = floor(α * (H_in - kh))`.
//
// This produces a valid fractional-max-pool: kernel windows tile the
// input non-uniformly, each output cell sees a unique window of size
// kh × kw, and the choice is reproducible given a fixed α buffer. It
// **does not** bit-match PyTorch's `nn.FractionalMaxPool2d` — document
// that divergence in the safe-plan rustdoc.
//
// **Random-samples ABI.** Caller provides `random_samples` as a
// `[N, C, num_axes]` f32 tensor (always f32 regardless of input dtype —
// uniform[0, 1) precision past ~24 bits is meaningless). The caller is
// responsible for filling this buffer (typically via `baracuda-curand`
// `RandomKind::Uniform`). The kernel reads samples as
// `random_samples[((n * C) + c) * num_axes + axis]`.
//
// **Forward output.** Writes both `y` (the per-window max value, dtype
// `T`) and `indices` (the per-window argmax linear index into the input
// tensor, dtype `i64`). The indices tensor is consumed by the BW kernel
// — same saved-indices pattern as MaxPool BW.
//
// **Backward.** One thread per output cell. Reads `dy[out_cell]` and
// `indices[out_cell]`, atomicAdd's `dy` into `dx[indices[out_cell]]`.
// Routes through `baracuda::atomic::add<T>` so half / bf16 use the
// 32-bit atomicCAS loop (Phase 11.3 / Fuel feedback #6).
//
// Status codes mirror the indexing family:
//   0 success
//   2 invalid problem
//   5 internal kernel error

#ifndef BARACUDA_FRACTIONAL_MAX_POOL_CUH
#define BARACUDA_FRACTIONAL_MAX_POOL_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>   // CUDART_INF / CUDART_INF_F

#include "baracuda_atomic.cuh"

namespace baracuda { namespace fractional_max_pool {

// =============================================================================
// Numeric helpers — value-typed promotion to f32 for comparison /
// "negative infinity" initializer. We compare in the input dtype itself
// for f32 / f64 (no rounding loss) and promote to f32 for f16 / bf16
// (mirrors cuDNN max-pool's accumulator semantics).
// =============================================================================

template <typename T>
__device__ __forceinline__ T neg_inf();

template <>
__device__ __forceinline__ float neg_inf<float>() { return -CUDART_INF_F; }

template <>
__device__ __forceinline__ double neg_inf<double>() { return -CUDART_INF; }

template <>
__device__ __forceinline__ __half neg_inf<__half>() {
    // CUDA's cuda_fp16.h doesn't expose a constexpr neg-inf — build one
    // from the f32 path. `__float2half(-inf)` rounds to half's neg-inf
    // pattern (0xFC00).
    return __float2half(-CUDART_INF_F);
}

template <>
__device__ __forceinline__ __nv_bfloat16 neg_inf<__nv_bfloat16>() {
    return __float2bfloat16(-CUDART_INF_F);
}

template <typename T>
__device__ __forceinline__ float to_compare_f32(T v);

template <>
__device__ __forceinline__ float to_compare_f32<float>(float v) { return v; }

template <>
__device__ __forceinline__ float to_compare_f32<double>(double v) { return (float)v; }

template <>
__device__ __forceinline__ float to_compare_f32<__half>(__half v) { return __half2float(v); }

template <>
__device__ __forceinline__ float to_compare_f32<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// =============================================================================
// Position formula — compute the per-output-cell window start for one
// axis. `evenly-spaced base + α perturbation` per the header comment.
// =============================================================================

__device__ __forceinline__ int32_t compute_window_start(
    int32_t out_idx, int32_t out_size, int32_t in_size, int32_t k, float alpha)
{
    // Span over which the start can vary. If in_size == k the only
    // legal start is 0; clamp out the divide.
    const int32_t max_start = in_size - k;
    if (max_start <= 0) return 0;
    int32_t start;
    if (out_size <= 1) {
        // Single output cell: pure α-driven offset.
        float pos = alpha * (float)max_start;
        start = (int32_t)floorf(pos);
    } else {
        // base[i] = i * max_start / (out_size - 1); float math is OK
        // because max_start fits in i32 and out_size fits in i32 too.
        float base = (float)out_idx * (float)max_start / (float)(out_size - 1);
        float pos = base + alpha;  // α perturbation
        start = (int32_t)floorf(pos);
    }
    // Clamp into [0, max_start].
    if (start < 0) start = 0;
    if (start > max_start) start = max_start;
    return start;
}

// =============================================================================
// 2-D forward kernel. One thread per output cell (n, c, oh, ow).
// =============================================================================

template <typename T>
__global__ void fractional_max_pool_2d_fw_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t* __restrict__ indices,
    const float* __restrict__ random_samples,   // [N, C, 2]
    int32_t batch, int32_t channels,
    int32_t h_in, int32_t w_in,
    int32_t h_out, int32_t w_out,
    int32_t kh, int32_t kw)
{
    const int64_t total = (int64_t)batch * channels * h_out * w_out;
    const int64_t tid  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t step = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        // Decompose into (n, c, oh, ow).
        const int32_t ow = (int32_t)(i % w_out);
        int64_t t = i / w_out;
        const int32_t oh = (int32_t)(t % h_out);
        t = t / h_out;
        const int32_t c  = (int32_t)(t % channels);
        const int32_t n  = (int32_t)(t / channels);

        // Read per-(n, c) α samples.
        const int64_t sample_base = ((int64_t)n * channels + c) * 2;
        const float alpha_h = random_samples[sample_base + 0];
        const float alpha_w = random_samples[sample_base + 1];

        // Per-axis window start.
        const int32_t start_h = compute_window_start(oh, h_out, h_in, kh, alpha_h);
        const int32_t start_w = compute_window_start(ow, w_out, w_in, kw, alpha_w);

        // Walk the (kh × kw) window, track argmax.
        const int64_t in_plane_stride = (int64_t)h_in * w_in;
        const int64_t in_nc_base = ((int64_t)n * channels + c) * in_plane_stride;

        T best_val = neg_inf<T>();
        float best_f = -CUDART_INF_F;
        int64_t best_idx = in_nc_base + (int64_t)start_h * w_in + start_w;

        #pragma unroll 1
        for (int32_t dh = 0; dh < kh; ++dh) {
            const int32_t ih = start_h + dh;
            #pragma unroll 1
            for (int32_t dw = 0; dw < kw; ++dw) {
                const int32_t iw = start_w + dw;
                const int64_t in_off = in_nc_base + (int64_t)ih * w_in + iw;
                T v = x[in_off];
                float vf = to_compare_f32<T>(v);
                if (vf > best_f) {
                    best_f = vf;
                    best_val = v;
                    best_idx = in_off;
                }
            }
        }
        y[i] = best_val;
        indices[i] = best_idx;
    }
}

// =============================================================================
// 2-D backward kernel. One thread per output cell; atomicAdd dy into
// dx[indices[out_cell]]. dx must be pre-zeroed by the caller.
// =============================================================================

template <typename T>
__global__ void fractional_max_pool_2d_bw_kernel(
    const T* __restrict__ dy,
    const int64_t* __restrict__ indices,
    T* __restrict__ dx,
    int32_t batch, int32_t channels,
    int32_t h_out, int32_t w_out,
    int64_t dx_numel)
{
    const int64_t total = (int64_t)batch * channels * h_out * w_out;
    const int64_t tid  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t step = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        const int64_t idx = indices[i];
        if (idx < 0 || idx >= dx_numel) continue;
        baracuda::atomic::add<T>(&dx[idx], dy[i]);
    }
}

template <typename T>
__host__ inline int32_t launch_fractional_max_pool_2d_fw(
    const T* x, T* y, int64_t* indices, const float* random_samples,
    int32_t batch, int32_t channels,
    int32_t h_in, int32_t w_in,
    int32_t h_out, int32_t w_out,
    int32_t kh, int32_t kw,
    cudaStream_t stream)
{
    if (batch <= 0 || channels <= 0) return 2;
    if (h_in <= 0 || w_in <= 0 || h_out <= 0 || w_out <= 0) return 2;
    if (kh <= 0 || kw <= 0 || kh > h_in || kw > w_in) return 2;
    const int64_t total = (int64_t)batch * channels * h_out * w_out;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = (int)(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fractional_max_pool_2d_fw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, indices, random_samples,
        batch, channels, h_in, w_in, h_out, w_out, kh, kw);
    return (cudaGetLastError() == cudaSuccess) ? 0 : 5;
}

template <typename T>
__host__ inline int32_t launch_fractional_max_pool_2d_bw(
    const T* dy, const int64_t* indices, T* dx,
    int32_t batch, int32_t channels,
    int32_t h_in, int32_t w_in,
    int32_t h_out, int32_t w_out,
    cudaStream_t stream)
{
    if (batch <= 0 || channels <= 0) return 2;
    if (h_in <= 0 || w_in <= 0 || h_out <= 0 || w_out <= 0) return 2;
    const int64_t total = (int64_t)batch * channels * h_out * w_out;
    const int64_t dx_numel = (int64_t)batch * channels * h_in * w_in;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = (int)(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fractional_max_pool_2d_bw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, indices, dx, batch, channels, h_out, w_out, dx_numel);
    return (cudaGetLastError() == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// 3-D forward kernel. One thread per output cell (n, c, od, oh, ow).
// =============================================================================

template <typename T>
__global__ void fractional_max_pool_3d_fw_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int64_t* __restrict__ indices,
    const float* __restrict__ random_samples,   // [N, C, 3]
    int32_t batch, int32_t channels,
    int32_t d_in, int32_t h_in, int32_t w_in,
    int32_t d_out, int32_t h_out, int32_t w_out,
    int32_t kd, int32_t kh, int32_t kw)
{
    const int64_t total = (int64_t)batch * channels * d_out * h_out * w_out;
    const int64_t tid  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t step = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        const int32_t ow = (int32_t)(i % w_out);
        int64_t t = i / w_out;
        const int32_t oh = (int32_t)(t % h_out);
        t = t / h_out;
        const int32_t od = (int32_t)(t % d_out);
        t = t / d_out;
        const int32_t c  = (int32_t)(t % channels);
        const int32_t n  = (int32_t)(t / channels);

        const int64_t sample_base = ((int64_t)n * channels + c) * 3;
        const float alpha_d = random_samples[sample_base + 0];
        const float alpha_h = random_samples[sample_base + 1];
        const float alpha_w = random_samples[sample_base + 2];

        const int32_t start_d = compute_window_start(od, d_out, d_in, kd, alpha_d);
        const int32_t start_h = compute_window_start(oh, h_out, h_in, kh, alpha_h);
        const int32_t start_w = compute_window_start(ow, w_out, w_in, kw, alpha_w);

        const int64_t in_plane_stride = (int64_t)h_in * w_in;
        const int64_t in_depth_stride = (int64_t)d_in * in_plane_stride;
        const int64_t in_nc_base = ((int64_t)n * channels + c) * in_depth_stride;

        T best_val = neg_inf<T>();
        float best_f = -CUDART_INF_F;
        int64_t best_idx = in_nc_base + (int64_t)start_d * in_plane_stride
                         + (int64_t)start_h * w_in + start_w;

        #pragma unroll 1
        for (int32_t dd = 0; dd < kd; ++dd) {
            const int32_t id = start_d + dd;
            const int64_t in_d_base = in_nc_base + (int64_t)id * in_plane_stride;
            #pragma unroll 1
            for (int32_t dh = 0; dh < kh; ++dh) {
                const int32_t ih = start_h + dh;
                const int64_t in_dh_base = in_d_base + (int64_t)ih * w_in;
                #pragma unroll 1
                for (int32_t dw = 0; dw < kw; ++dw) {
                    const int32_t iw = start_w + dw;
                    const int64_t in_off = in_dh_base + iw;
                    T v = x[in_off];
                    float vf = to_compare_f32<T>(v);
                    if (vf > best_f) {
                        best_f = vf;
                        best_val = v;
                        best_idx = in_off;
                    }
                }
            }
        }
        y[i] = best_val;
        indices[i] = best_idx;
    }
}

template <typename T>
__global__ void fractional_max_pool_3d_bw_kernel(
    const T* __restrict__ dy,
    const int64_t* __restrict__ indices,
    T* __restrict__ dx,
    int32_t batch, int32_t channels,
    int32_t d_out, int32_t h_out, int32_t w_out,
    int64_t dx_numel)
{
    const int64_t total = (int64_t)batch * channels * d_out * h_out * w_out;
    const int64_t tid  = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t step = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < total; i += step) {
        const int64_t idx = indices[i];
        if (idx < 0 || idx >= dx_numel) continue;
        baracuda::atomic::add<T>(&dx[idx], dy[i]);
    }
}

template <typename T>
__host__ inline int32_t launch_fractional_max_pool_3d_fw(
    const T* x, T* y, int64_t* indices, const float* random_samples,
    int32_t batch, int32_t channels,
    int32_t d_in, int32_t h_in, int32_t w_in,
    int32_t d_out, int32_t h_out, int32_t w_out,
    int32_t kd, int32_t kh, int32_t kw,
    cudaStream_t stream)
{
    if (batch <= 0 || channels <= 0) return 2;
    if (d_in <= 0 || h_in <= 0 || w_in <= 0) return 2;
    if (d_out <= 0 || h_out <= 0 || w_out <= 0) return 2;
    if (kd <= 0 || kh <= 0 || kw <= 0) return 2;
    if (kd > d_in || kh > h_in || kw > w_in) return 2;
    const int64_t total = (int64_t)batch * channels * d_out * h_out * w_out;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = (int)(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fractional_max_pool_3d_fw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        x, y, indices, random_samples,
        batch, channels, d_in, h_in, w_in, d_out, h_out, w_out, kd, kh, kw);
    return (cudaGetLastError() == cudaSuccess) ? 0 : 5;
}

template <typename T>
__host__ inline int32_t launch_fractional_max_pool_3d_bw(
    const T* dy, const int64_t* indices, T* dx,
    int32_t batch, int32_t channels,
    int32_t d_in, int32_t h_in, int32_t w_in,
    int32_t d_out, int32_t h_out, int32_t w_out,
    cudaStream_t stream)
{
    if (batch <= 0 || channels <= 0) return 2;
    if (d_in <= 0 || h_in <= 0 || w_in <= 0) return 2;
    if (d_out <= 0 || h_out <= 0 || w_out <= 0) return 2;
    const int64_t total = (int64_t)batch * channels * d_out * h_out * w_out;
    const int64_t dx_numel = (int64_t)batch * channels * d_in * h_in * w_in;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = (int)(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    fractional_max_pool_3d_bw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dy, indices, dx, batch, channels, d_out, h_out, w_out, dx_numel);
    return (cudaGetLastError() == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::fractional_max_pool

// =============================================================================
// INSTANTIATE macros — emit `extern "C"` launchers per (op, dtype) pair.
// =============================================================================

#define BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_2D_FW_INSTANTIATE(NAME, T)                           \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        const void* x, void* y,                                                                    \
        void* indices,                                                                             \
        const float* random_samples,                                                               \
        int32_t batch, int32_t channels,                                                           \
        int32_t h_in, int32_t w_in,                                                                \
        int32_t h_out, int32_t w_out,                                                              \
        int32_t kh, int32_t kw,                                                                    \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (x == nullptr || y == nullptr || indices == nullptr) return 2;                          \
        if (random_samples == nullptr) return 2;                                                   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::fractional_max_pool::launch_fractional_max_pool_2d_fw<T>(                 \
            static_cast<const T*>(x),                                                              \
            static_cast<T*>(y),                                                                    \
            static_cast<int64_t*>(indices),                                                        \
            random_samples,                                                                        \
            batch, channels, h_in, w_in, h_out, w_out, kh, kw, stream);                            \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        const void* /*x*/, const void* /*y*/,                                                      \
        const void* /*indices*/,                                                                   \
        const float* /*random_samples*/,                                                           \
        int32_t batch, int32_t channels,                                                           \
        int32_t h_in, int32_t w_in,                                                                \
        int32_t h_out, int32_t w_out,                                                              \
        int32_t kh, int32_t kw)                                                                    \
    {                                                                                              \
        if (batch < 0 || channels < 0) return 2;                                                   \
        if (h_in < 0 || w_in < 0 || h_out < 0 || w_out < 0) return 2;                              \
        if (kh <= 0 || kw <= 0) return 2;                                                          \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_2D_BW_INSTANTIATE(NAME, T)                           \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        const void* dy,                                                                            \
        const void* indices,                                                                       \
        void* dx,                                                                                  \
        int32_t batch, int32_t channels,                                                           \
        int32_t h_in, int32_t w_in,                                                                \
        int32_t h_out, int32_t w_out,                                                              \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (dy == nullptr || indices == nullptr || dx == nullptr) return 2;                        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::fractional_max_pool::launch_fractional_max_pool_2d_bw<T>(                 \
            static_cast<const T*>(dy),                                                             \
            static_cast<const int64_t*>(indices),                                                  \
            static_cast<T*>(dx),                                                                   \
            batch, channels, h_in, w_in, h_out, w_out, stream);                                    \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        const void* /*dy*/,                                                                        \
        const void* /*indices*/,                                                                   \
        const void* /*dx*/,                                                                        \
        int32_t batch, int32_t channels,                                                           \
        int32_t h_in, int32_t w_in,                                                                \
        int32_t h_out, int32_t w_out)                                                              \
    {                                                                                              \
        if (batch < 0 || channels < 0) return 2;                                                   \
        if (h_in < 0 || w_in < 0 || h_out < 0 || w_out < 0) return 2;                              \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_3D_FW_INSTANTIATE(NAME, T)                           \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        const void* x, void* y,                                                                    \
        void* indices,                                                                             \
        const float* random_samples,                                                               \
        int32_t batch, int32_t channels,                                                           \
        int32_t d_in, int32_t h_in, int32_t w_in,                                                  \
        int32_t d_out, int32_t h_out, int32_t w_out,                                               \
        int32_t kd, int32_t kh, int32_t kw,                                                        \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (x == nullptr || y == nullptr || indices == nullptr) return 2;                          \
        if (random_samples == nullptr) return 2;                                                   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::fractional_max_pool::launch_fractional_max_pool_3d_fw<T>(                 \
            static_cast<const T*>(x),                                                              \
            static_cast<T*>(y),                                                                    \
            static_cast<int64_t*>(indices),                                                        \
            random_samples,                                                                        \
            batch, channels,                                                                       \
            d_in, h_in, w_in, d_out, h_out, w_out, kd, kh, kw, stream);                            \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        const void* /*x*/, const void* /*y*/,                                                      \
        const void* /*indices*/,                                                                   \
        const float* /*random_samples*/,                                                           \
        int32_t batch, int32_t channels,                                                           \
        int32_t d_in, int32_t h_in, int32_t w_in,                                                  \
        int32_t d_out, int32_t h_out, int32_t w_out,                                               \
        int32_t kd, int32_t kh, int32_t kw)                                                        \
    {                                                                                              \
        if (batch < 0 || channels < 0) return 2;                                                   \
        if (d_in < 0 || h_in < 0 || w_in < 0 || d_out < 0 || h_out < 0 || w_out < 0) return 2;     \
        if (kd <= 0 || kh <= 0 || kw <= 0) return 2;                                               \
        return 0;                                                                                  \
    }

#define BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_3D_BW_INSTANTIATE(NAME, T)                           \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        const void* dy,                                                                            \
        const void* indices,                                                                       \
        void* dx,                                                                                  \
        int32_t batch, int32_t channels,                                                           \
        int32_t d_in, int32_t h_in, int32_t w_in,                                                  \
        int32_t d_out, int32_t h_out, int32_t w_out,                                               \
        void* stream_ptr)                                                                          \
    {                                                                                              \
        if (dy == nullptr || indices == nullptr || dx == nullptr) return 2;                        \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                               \
        return baracuda::fractional_max_pool::launch_fractional_max_pool_3d_bw<T>(                 \
            static_cast<const T*>(dy),                                                             \
            static_cast<const int64_t*>(indices),                                                  \
            static_cast<T*>(dx),                                                                   \
            batch, channels,                                                                       \
            d_in, h_in, w_in, d_out, h_out, w_out, stream);                                        \
    }                                                                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        const void* /*dy*/,                                                                        \
        const void* /*indices*/,                                                                   \
        const void* /*dx*/,                                                                        \
        int32_t batch, int32_t channels,                                                           \
        int32_t d_in, int32_t h_in, int32_t w_in,                                                  \
        int32_t d_out, int32_t h_out, int32_t w_out)                                               \
    {                                                                                              \
        if (batch < 0 || channels < 0) return 2;                                                   \
        if (d_in < 0 || h_in < 0 || w_in < 0 || d_out < 0 || h_out < 0 || w_out < 0) return 2;     \
        return 0;                                                                                  \
    }

#endif // BARACUDA_FRACTIONAL_MAX_POOL_CUH
