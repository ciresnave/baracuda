// baracuda_im2col.cuh
//
// Phase 19.3 — im2col / im2col1d / col2im1d bespoke kernels.
//
// Building blocks for Fuel's conv-via-im2col-and-GEMM fallback
// lowering + the conv-backward filter-gradient path. PyTorch
// `torch.nn.functional.unfold` is the 2-D im2col equivalent;
// `torch.nn.functional.fold` is the 2-D inverse (`col2im`). This
// header ships:
//
//   * `im2col_2d` — 2-D im2col, NCHW → [N, C·kh·kw, h_out·w_out].
//   * `im2col_1d` — 1-D im2col, NCL  → [N, C·kl,    l_out].
//   * `col2im_1d` — 1-D col2im (inverse of im2col_1d).
//
// 2-D col2im is intentionally not provided here — Fuel routes the
// conv-2d filter-gradient through cuDNN's
// `cudnnConvolutionBackwardFilter`, which is already exposed by the
// `Conv2dPlan` family.
//
// =============================================================================
// Op semantics
// =============================================================================
//
// im2col_2d:
//   For each output column `(c·kh·kw + ki·kw + kj, oh·w_out + ow)`:
//
//     y[n, c·kh·kw + ki·kw + kj, oh·w_out + ow]
//         = x[n, c, oh·stride_h + ki·dilation_h - pad_h,
//                  ow·stride_w + kj·dilation_w - pad_w]
//
//   if both source coordinates land inside `[0, H_in) × [0, W_in)`;
//   otherwise the cell holds 0 (zero-padding semantics).
//
//   Output extents:
//     h_out = (H_in + 2·pad_h - dilation_h·(kh-1) - 1) / stride_h + 1
//     w_out = (W_in + 2·pad_w - dilation_w·(kw-1) - 1) / stride_w + 1
//
// im2col_1d: same as the 2-D case, restricted to the (kl, stride_l,
// pad_l, dilation_l) 1-D variant.
//
// col2im_1d: pure inverse — scatter each col cell back to its source
// input position. Overlapping output positions accumulate via
// atomicAdd (when stride < kernel). Caller must pre-zero the output
// buffer before launch.
//
// =============================================================================
// Kernel design
// =============================================================================
//
// FW (im2col_{1d,2d}):
//   One thread per output element. Thread decomposes its linear
//   index into [n, c·kij, output_spatial] then drills further. No
//   atomics — each output cell has exactly one source cell.
//
// BW-like (col2im_1d):
//   One thread per *input* column-entry (col-shaped tensor). Each
//   thread atomicAdds its contribution into the (n, c, l_target)
//   position of the output. half/bf16 routes through
//   `baracuda::atomic::add<T>` from `baracuda_atomic.cuh`.
//
// Status codes mirror the rest of the family:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   5 internal launch error.

#ifndef BARACUDA_IM2COL_CUH
#define BARACUDA_IM2COL_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_atomic.cuh"

namespace baracuda { namespace im2col {

// =============================================================================
// Zero helpers — dtype-uniform zero for the out-of-bounds pad cells.
// =============================================================================

template <typename T> __device__ __forceinline__ T zero_of() { return T(0); }
template <> __device__ __forceinline__ __half        zero_of<__half>()        { return __float2half(0.0f); }
template <> __device__ __forceinline__ __nv_bfloat16 zero_of<__nv_bfloat16>() { return __float2bfloat16(0.0f); }

// =============================================================================
// 2-D FW kernel — im2col over NCHW.
//
// Output layout: [N, C·kh·kw, h_out·w_out] contiguous (C-order:
// `[ni, c·kh·kw + ki·kw + kj, oh·w_out + ow]`).
// =============================================================================

template <typename T>
__global__ void im2col_2d_fw_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int32_t N, int32_t C,
    int32_t H_in, int32_t W_in,
    int32_t H_out, int32_t W_out,
    int32_t kh, int32_t kw,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t dilation_h, int32_t dilation_w)
{
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t spatial = (int64_t)H_out * (int64_t)W_out;
    int64_t col_rows = (int64_t)C * (int64_t)kh * (int64_t)kw;
    int64_t total = (int64_t)N * col_rows * spatial;
    if (tid >= total) return;

    // Decompose tid → (ni, c·kij row, output spatial cell).
    int64_t spatial_idx = tid % spatial;
    int64_t tmp = tid / spatial;
    int64_t row = tmp % col_rows;     // c·kh·kw + ki·kw + kj
    int32_t ni = (int32_t)(tmp / col_rows);

    int32_t ow = (int32_t)(spatial_idx % (int64_t)W_out);
    int32_t oh = (int32_t)(spatial_idx / (int64_t)W_out);

    int32_t kij = (int32_t)(row % (int64_t)(kh * kw));
    int32_t ci  = (int32_t)(row / (int64_t)(kh * kw));
    int32_t ki  = kij / kw;
    int32_t kj  = kij - ki * kw;

    int32_t in_h = oh * stride_h + ki * dilation_h - pad_h;
    int32_t in_w = ow * stride_w + kj * dilation_w - pad_w;

    T val;
    if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
        int64_t x_off = (((int64_t)ni * C + ci) * (int64_t)H_in + in_h) * (int64_t)W_in + in_w;
        val = x[x_off];
    } else {
        val = zero_of<T>();
    }
    y[tid] = val;
}

// =============================================================================
// 1-D FW kernel — im2col over NCL.
//
// Output layout: [N, C·kl, l_out] contiguous (`[ni, ci·kl + ki, ol]`).
// =============================================================================

template <typename T>
__global__ void im2col_1d_fw_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int32_t N, int32_t C,
    int32_t L_in, int32_t L_out,
    int32_t kl,
    int32_t stride_l, int32_t pad_l, int32_t dilation_l)
{
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t col_rows = (int64_t)C * (int64_t)kl;
    int64_t total = (int64_t)N * col_rows * (int64_t)L_out;
    if (tid >= total) return;

    int32_t ol = (int32_t)(tid % (int64_t)L_out);
    int64_t tmp = tid / (int64_t)L_out;
    int64_t row = tmp % col_rows;
    int32_t ni = (int32_t)(tmp / col_rows);

    int32_t ki = (int32_t)(row % (int64_t)kl);
    int32_t ci = (int32_t)(row / (int64_t)kl);

    int32_t in_l = ol * stride_l + ki * dilation_l - pad_l;

    T val;
    if (in_l >= 0 && in_l < L_in) {
        int64_t x_off = ((int64_t)ni * C + ci) * (int64_t)L_in + in_l;
        val = x[x_off];
    } else {
        val = zero_of<T>();
    }
    y[tid] = val;
}

// =============================================================================
// 1-D BW kernel — col2im (inverse of im2col_1d).
//
// One thread per *input* col-shaped element. Atomic-scatters into the
// output. Caller pre-zeroes the output buffer.
//
// Input shape: [N, C·kl, L_out]. Output shape: [N, C, L_in].
// =============================================================================

template <typename T>
__global__ void col2im_1d_kernel(
    const T* __restrict__ col,
    T* __restrict__ out,
    int32_t N, int32_t C,
    int32_t L_in, int32_t L_out,
    int32_t kl,
    int32_t stride_l, int32_t pad_l, int32_t dilation_l)
{
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t col_rows = (int64_t)C * (int64_t)kl;
    int64_t total = (int64_t)N * col_rows * (int64_t)L_out;
    if (tid >= total) return;

    int32_t ol = (int32_t)(tid % (int64_t)L_out);
    int64_t tmp = tid / (int64_t)L_out;
    int64_t row = tmp % col_rows;
    int32_t ni = (int32_t)(tmp / col_rows);

    int32_t ki = (int32_t)(row % (int64_t)kl);
    int32_t ci = (int32_t)(row / (int64_t)kl);

    int32_t target_l = ol * stride_l + ki * dilation_l - pad_l;
    if (target_l < 0 || target_l >= L_in) return;

    int64_t out_off = ((int64_t)ni * C + ci) * (int64_t)L_in + target_l;
    baracuda::atomic::add<T>(out + out_off, col[tid]);
}

// =============================================================================
// Launchers
// =============================================================================

template <typename T>
__host__ inline int32_t launch_im2col_2d(
    int32_t batch, int32_t channels,
    int32_t h_in, int32_t w_in,
    int32_t h_out, int32_t w_out,
    int32_t kh, int32_t kw,
    int32_t stride_h, int32_t stride_w,
    int32_t pad_h, int32_t pad_w,
    int32_t dilation_h, int32_t dilation_w,
    const void* x, void* y,
    cudaStream_t stream)
{
    if (x == nullptr || y == nullptr) return 2;
    if (batch <= 0 || channels <= 0 || h_in <= 0 || w_in <= 0) return 2;
    if (h_out <= 0 || w_out <= 0) return 2;
    if (kh <= 0 || kw <= 0) return 2;
    if (stride_h <= 0 || stride_w <= 0) return 2;
    if (dilation_h <= 0 || dilation_w <= 0) return 2;
    if (pad_h < 0 || pad_w < 0) return 2;

    int64_t total = (int64_t)batch * (int64_t)channels * (int64_t)kh * (int64_t)kw
                  * (int64_t)h_out * (int64_t)w_out;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    int64_t blocks64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks64 > 2147483647LL ? 2147483647LL : blocks64);
    if (blocks <= 0) blocks = 1;
    im2col_2d_fw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        static_cast<const T*>(x), static_cast<T*>(y),
        batch, channels, h_in, w_in, h_out, w_out,
        kh, kw, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T>
__host__ inline int32_t launch_im2col_1d(
    int32_t batch, int32_t channels,
    int32_t l_in, int32_t l_out,
    int32_t kl, int32_t stride_l, int32_t pad_l, int32_t dilation_l,
    const void* x, void* y,
    cudaStream_t stream)
{
    if (x == nullptr || y == nullptr) return 2;
    if (batch <= 0 || channels <= 0 || l_in <= 0) return 2;
    if (l_out <= 0) return 2;
    if (kl <= 0) return 2;
    if (stride_l <= 0 || dilation_l <= 0) return 2;
    if (pad_l < 0) return 2;

    int64_t total = (int64_t)batch * (int64_t)channels * (int64_t)kl * (int64_t)l_out;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    int64_t blocks64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks64 > 2147483647LL ? 2147483647LL : blocks64);
    if (blocks <= 0) blocks = 1;
    im2col_1d_fw_kernel<T><<<blocks, kBlock, 0, stream>>>(
        static_cast<const T*>(x), static_cast<T*>(y),
        batch, channels, l_in, l_out, kl, stride_l, pad_l, dilation_l);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

template <typename T>
__host__ inline int32_t launch_col2im_1d(
    int32_t batch, int32_t channels,
    int32_t l_in, int32_t l_out,
    int32_t kl, int32_t stride_l, int32_t pad_l, int32_t dilation_l,
    const void* col, void* out,
    cudaStream_t stream)
{
    if (col == nullptr || out == nullptr) return 2;
    if (batch <= 0 || channels <= 0 || l_in <= 0) return 2;
    if (l_out <= 0) return 2;
    if (kl <= 0) return 2;
    if (stride_l <= 0 || dilation_l <= 0) return 2;
    if (pad_l < 0) return 2;

    // Caller must pre-zero `out` (atomicAdd scatter).
    int64_t total = (int64_t)batch * (int64_t)channels * (int64_t)kl * (int64_t)l_out;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    int64_t blocks64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks64 > 2147483647LL ? 2147483647LL : blocks64);
    if (blocks <= 0) blocks = 1;
    col2im_1d_kernel<T><<<blocks, kBlock, 0, stream>>>(
        static_cast<const T*>(col), static_cast<T*>(out),
        batch, channels, l_in, l_out, kl, stride_l, pad_l, dilation_l);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::im2col

// =============================================================================
// Per-dtype FFI instantiation macros.
// =============================================================================

#define BARACUDA_KERNELS_IM2COL_2D_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_im2col_2d_##NAME##_run(                          \
        int32_t batch, int32_t channels,                                                 \
        int32_t h_in, int32_t w_in,                                                      \
        int32_t h_out, int32_t w_out,                                                    \
        int32_t kh, int32_t kw,                                                          \
        int32_t stride_h, int32_t stride_w,                                              \
        int32_t pad_h, int32_t pad_w,                                                    \
        int32_t dilation_h, int32_t dilation_w,                                          \
        const void* input,                                                               \
        void* output,                                                                    \
        void* stream_ptr)                                                                \
    {                                                                                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                     \
        return baracuda::im2col::launch_im2col_2d<T>(                                    \
            batch, channels, h_in, w_in, h_out, w_out,                                   \
            kh, kw, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,            \
            input, output, stream);                                                      \
    }                                                                                    \
    extern "C" int32_t baracuda_kernels_im2col_2d_##NAME##_can_implement(                \
        int32_t batch, int32_t channels,                                                 \
        int32_t h_in, int32_t w_in,                                                      \
        int32_t h_out, int32_t w_out,                                                    \
        int32_t kh, int32_t kw,                                                          \
        int32_t stride_h, int32_t stride_w,                                              \
        int32_t pad_h, int32_t pad_w,                                                    \
        int32_t dilation_h, int32_t dilation_w,                                          \
        const void* /*input*/,                                                           \
        const void* /*output*/)                                                          \
    {                                                                                    \
        if (batch < 0 || channels < 0) return 2;                                         \
        if (h_in < 0 || w_in < 0 || h_out < 0 || w_out < 0) return 2;                    \
        if (kh <= 0 || kw <= 0) return 2;                                                \
        if (stride_h <= 0 || stride_w <= 0) return 2;                                    \
        if (pad_h < 0 || pad_w < 0) return 2;                                            \
        if (dilation_h <= 0 || dilation_w <= 0) return 2;                                \
        return 0;                                                                        \
    }

#define BARACUDA_KERNELS_IM2COL_1D_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_im2col_1d_##NAME##_run(                          \
        int32_t batch, int32_t channels,                                                 \
        int32_t l_in, int32_t l_out,                                                     \
        int32_t kl, int32_t stride_l, int32_t pad_l, int32_t dilation_l,                 \
        const void* input,                                                               \
        void* output,                                                                    \
        void* stream_ptr)                                                                \
    {                                                                                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                     \
        return baracuda::im2col::launch_im2col_1d<T>(                                    \
            batch, channels, l_in, l_out,                                                \
            kl, stride_l, pad_l, dilation_l,                                             \
            input, output, stream);                                                      \
    }                                                                                    \
    extern "C" int32_t baracuda_kernels_im2col_1d_##NAME##_can_implement(                \
        int32_t batch, int32_t channels,                                                 \
        int32_t l_in, int32_t l_out,                                                     \
        int32_t kl, int32_t stride_l, int32_t pad_l, int32_t dilation_l,                 \
        const void* /*input*/,                                                           \
        const void* /*output*/)                                                          \
    {                                                                                    \
        if (batch < 0 || channels < 0) return 2;                                         \
        if (l_in < 0 || l_out < 0) return 2;                                             \
        if (kl <= 0 || stride_l <= 0 || dilation_l <= 0) return 2;                       \
        if (pad_l < 0) return 2;                                                         \
        return 0;                                                                        \
    }

#define BARACUDA_KERNELS_COL2IM_1D_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_col2im_1d_##NAME##_run(                          \
        int32_t batch, int32_t channels,                                                 \
        int32_t l_in, int32_t l_out,                                                     \
        int32_t kl, int32_t stride_l, int32_t pad_l, int32_t dilation_l,                 \
        const void* input,                                                               \
        void* output,                                                                    \
        void* stream_ptr)                                                                \
    {                                                                                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                     \
        return baracuda::im2col::launch_col2im_1d<T>(                                    \
            batch, channels, l_in, l_out,                                                \
            kl, stride_l, pad_l, dilation_l,                                             \
            input, output, stream);                                                      \
    }                                                                                    \
    extern "C" int32_t baracuda_kernels_col2im_1d_##NAME##_can_implement(                \
        int32_t batch, int32_t channels,                                                 \
        int32_t l_in, int32_t l_out,                                                     \
        int32_t kl, int32_t stride_l, int32_t pad_l, int32_t dilation_l,                 \
        const void* /*input*/,                                                           \
        const void* /*output*/)                                                          \
    {                                                                                    \
        if (batch < 0 || channels < 0) return 2;                                         \
        if (l_in < 0 || l_out < 0) return 2;                                             \
        if (kl <= 0 || stride_l <= 0 || dilation_l <= 0) return 2;                       \
        if (pad_l < 0) return 2;                                                         \
        return 0;                                                                        \
    }

#endif // BARACUDA_IM2COL_CUH
