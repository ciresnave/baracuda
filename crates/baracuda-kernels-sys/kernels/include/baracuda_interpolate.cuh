// baracuda_interpolate.cuh
//
// Templated kernels and INSTANTIATE macros for `interpolate` (Phase 9
// Category T — image / spatial transforms; Phase 21 — align_corners +
// scale-factor overrides + f16 / bf16 fanout).
//
// NCHW layout. Both `align_corners=false` (PyTorch default for new
// code, e.g. `F.interpolate`) and `align_corners=true` (PyTorch's
// `nn.Upsample(align_corners=True)`) are supported. Coordinate mapping
// (per PyTorch ATen `UpSample.h::area_pixel_compute_source_index`):
//
//   align_corners=false:
//     scale_h = (scale_h_factor != 0)
//                 ? 1.0 / scale_h_factor
//                 : (double)IH / (double)OH
//     src_y = (oh + 0.5) * scale_h - 0.5
//
//   align_corners=true:
//     scale_h = (scale_h_factor != 0)
//                 ? 1.0 / scale_h_factor
//                 : (double)(IH - 1) / (double)max(OH - 1, 1)
//     src_y = oh * scale_h
//
// `scale_h_factor` / `scale_w_factor` are the SCALE values (output_size
// / input_size) — when nonzero they override the size-derived ratio.
// `0.0` means "derive from (IH, OH) / (IW, OW)".
//
// Out-of-range samples are clamped to the input boundary (PyTorch's
// `interpolate` for bilinear uses zero-pad-equivalent via clamp — i.e.
// edge replication — which matches PyTorch's behavior).
//
// **Phase 21 breaking change**: the FFI signature gained three trailing
// params before `stream` — `align_corners: i32`, `scale_h_factor: f64`,
// `scale_w_factor: f64`. Pre-Phase-21 callers passing only the original
// 6 shape ints + 2 buffers + workspace pair + stream will need to
// rebuild. Fuel was the only known caller; this change tracks their
// `Option<f64>` / `bool` plumbing.
//
// **Phase 21 dtype fanout**: f16 + bf16 join the existing f32 + f64
// instantiation set (four FW + four BW = 8 bilinear symbols total).
// Half-precision paths follow the "cast at read, f32 accumulator, cast
// at write" pattern; f64 stays in double end-to-end for bit-stability
// with the legacy code path.
//
// Status codes (mirror the rest of the bespoke family):
//   0 success
//   1 misaligned operand (reserved)
//   2 invalid problem
//   3 unsupported (mode not implemented, etc.)
//   4 workspace too small (reserved)
//   5 internal kernel error (launch failure)

#ifndef BARACUDA_INTERPOLATE_CUH
#define BARACUDA_INTERPOLATE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "baracuda_indexing.cuh"  // scatter_atomic_add<T>

namespace baracuda { namespace image {

// Accumulator selection: f32 for {f32, f16, bf16}, f64 for f64. Matches
// `baracuda_adaptive_pool.cuh::accum_of` / `baracuda_norm.cuh` family.
template <typename T> struct interp_accum_of { using type = float; };
template <> struct interp_accum_of<double> { using type = double; };

// Cast operand to accumulator scalar.
template <typename T>
__device__ __forceinline__ typename interp_accum_of<T>::type
interp_to_accum(T v) {
    return (typename interp_accum_of<T>::type)v;
}
template <>
__device__ __forceinline__ float interp_to_accum<__half>(__half v) {
    return __half2float(v);
}
template <>
__device__ __forceinline__ float interp_to_accum<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// Cast accumulator scalar back to operand.
template <typename T>
__device__ __forceinline__ T
interp_from_accum(typename interp_accum_of<T>::type v) {
    return (T)v;
}
template <>
__device__ __forceinline__ __half interp_from_accum<__half>(float v) {
    return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 interp_from_accum<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

// PyTorch `area_pixel_compute_scale` equivalent. Returns the per-output
// step in input units. `scale_factor` of 0.0 means "derive from sizes".
//
// align_corners=true && scale_factor==0 && out_size==1 collapses to a
// scale of 0 (no movement) — matches PyTorch behavior. The kernel
// guards `out_size==0` at the launch boundary so we never enter the
// kernel with `out_size <= 0`.
__host__ __device__ __forceinline__ double interp_compute_scale(
    int in_size, int out_size, bool align_corners, double scale_factor)
{
    if (scale_factor != 0.0) {
        return 1.0 / scale_factor;
    }
    if (align_corners) {
        return (out_size > 1)
            ? (double)(in_size - 1) / (double)(out_size - 1)
            : 0.0;
    }
    // align_corners=false, no override: ratio of sizes.
    return (out_size > 0) ? (double)in_size / (double)out_size : 0.0;
}

// =============================================================================
// interpolate_bilinear_2d forward — one thread per (n, c, oh, ow) output.
// =============================================================================

template <typename T>
__global__ void interpolate_bilinear_2d_kernel(
    const T* __restrict__ input,    // [N, C, IH, IW]
    T* __restrict__ output,         // [N, C, OH, OW]
    int N, int C, int IH, int IW, int OH, int OW,
    int align_corners_i32,
    double scale_h, double scale_w)
{
    using Acc = typename interp_accum_of<T>::type;
    const bool align_corners = (align_corners_i32 != 0);
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)OH * (int64_t)OW;
    for (int64_t i = tid; i < total; i += step) {
        int ow = (int)(i % OW);
        int oh = (int)((i / OW) % OH);
        int c  = (int)((i / ((int64_t)OW * OH)) % C);
        int n  = (int)(i / ((int64_t)OW * OH * C));

        // Source-index math in the accumulator precision (double for
        // f64 operands, float for everything else).
        Acc sx, sy;
        if (align_corners) {
            sx = (Acc)((double)ow * scale_w);
            sy = (Acc)((double)oh * scale_h);
        } else {
            sx = (Acc)(((double)ow + 0.5) * scale_w - 0.5);
            sy = (Acc)(((double)oh + 0.5) * scale_h - 0.5);
        }
        int x0 = (int)floor((double)sx);
        int y0 = (int)floor((double)sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        Acc wx1 = sx - (Acc)x0;
        Acc wy1 = sy - (Acc)y0;
        Acc wx0 = (Acc)1 - wx1;
        Acc wy0 = (Acc)1 - wy1;
        // Clamp (edge replicate) for OOB samples.
        int cx0 = x0 < 0 ? 0 : (x0 >= IW ? IW - 1 : x0);
        int cx1 = x1 < 0 ? 0 : (x1 >= IW ? IW - 1 : x1);
        int cy0 = y0 < 0 ? 0 : (y0 >= IH ? IH - 1 : y0);
        int cy1 = y1 < 0 ? 0 : (y1 >= IH ? IH - 1 : y1);
        const T* in_nc = input + ((int64_t)n * C + c) * (int64_t)IH * IW;
        Acc v00 = interp_to_accum<T>(in_nc[(int64_t)cy0 * IW + cx0]);
        Acc v01 = interp_to_accum<T>(in_nc[(int64_t)cy0 * IW + cx1]);
        Acc v10 = interp_to_accum<T>(in_nc[(int64_t)cy1 * IW + cx0]);
        Acc v11 = interp_to_accum<T>(in_nc[(int64_t)cy1 * IW + cx1]);
        Acc out = wy0 * (wx0 * v00 + wx1 * v01)
                + wy1 * (wx0 * v10 + wx1 * v11);
        output[i] = interp_from_accum<T>(out);
    }
}

template <typename T>
__host__ inline int32_t launch_interpolate_bilinear_2d(
    const T* input, T* output,
    int N, int C, int IH, int IW, int OH, int OW,
    int align_corners, double scale_h_factor, double scale_w_factor,
    cudaStream_t stream)
{
    if (N < 0 || C < 0 || IH < 0 || IW < 0 || OH < 0 || OW < 0) return 2;
    int64_t total = (int64_t)N * C * OH * OW;
    if (total == 0) return 0;
    // Source data must exist to bilinearly sample.
    if (IH == 0 || IW == 0) return 2;
    const bool ac = (align_corners != 0);
    double scale_h = interp_compute_scale(IH, OH, ac, scale_h_factor);
    double scale_w = interp_compute_scale(IW, OW, ac, scale_w_factor);
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    interpolate_bilinear_2d_kernel<T><<<blocks, kBlock, 0, stream>>>(
        input, output, N, C, IH, IW, OH, OW,
        align_corners, scale_h, scale_w);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// interpolate_bilinear_2d backward — one thread per (n, c, oh, ow) output
// cell. Distributes dout[n, c, oh, ow] back across the 4 input cells it
// bilinearly sampled, weighted by the same wij. atomicAdd into dinput.
// Caller pre-zeros dinput.
// =============================================================================

template <typename T>
__global__ void interpolate_bilinear_2d_backward_kernel(
    const T* __restrict__ dout,     // [N, C, OH, OW]
    T* __restrict__ dinput,         // [N, C, IH, IW]
    int N, int C, int IH, int IW, int OH, int OW,
    int align_corners_i32,
    double scale_h, double scale_w)
{
    using Acc = typename interp_accum_of<T>::type;
    const bool align_corners = (align_corners_i32 != 0);
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)OH * (int64_t)OW;
    for (int64_t i = tid; i < total; i += step) {
        int ow = (int)(i % OW);
        int oh = (int)((i / OW) % OH);
        int c  = (int)((i / ((int64_t)OW * OH)) % C);
        int n  = (int)(i / ((int64_t)OW * OH * C));

        Acc sx, sy;
        if (align_corners) {
            sx = (Acc)((double)ow * scale_w);
            sy = (Acc)((double)oh * scale_h);
        } else {
            sx = (Acc)(((double)ow + 0.5) * scale_w - 0.5);
            sy = (Acc)(((double)oh + 0.5) * scale_h - 0.5);
        }
        int x0 = (int)floor((double)sx);
        int y0 = (int)floor((double)sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        Acc wx1 = sx - (Acc)x0;
        Acc wy1 = sy - (Acc)y0;
        Acc wx0 = (Acc)1 - wx1;
        Acc wy0 = (Acc)1 - wy1;
        int cx0 = x0 < 0 ? 0 : (x0 >= IW ? IW - 1 : x0);
        int cx1 = x1 < 0 ? 0 : (x1 >= IW ? IW - 1 : x1);
        int cy0 = y0 < 0 ? 0 : (y0 >= IH ? IH - 1 : y0);
        int cy1 = y1 < 0 ? 0 : (y1 >= IH ? IH - 1 : y1);
        T* di_nc = dinput + ((int64_t)n * C + c) * (int64_t)IH * IW;
        Acc g = interp_to_accum<T>(dout[i]);
        baracuda::indexing::scatter_atomic_add<T>(
            &di_nc[(int64_t)cy0 * IW + cx0], interp_from_accum<T>(g * wy0 * wx0));
        baracuda::indexing::scatter_atomic_add<T>(
            &di_nc[(int64_t)cy0 * IW + cx1], interp_from_accum<T>(g * wy0 * wx1));
        baracuda::indexing::scatter_atomic_add<T>(
            &di_nc[(int64_t)cy1 * IW + cx0], interp_from_accum<T>(g * wy1 * wx0));
        baracuda::indexing::scatter_atomic_add<T>(
            &di_nc[(int64_t)cy1 * IW + cx1], interp_from_accum<T>(g * wy1 * wx1));
    }
}

template <typename T>
__host__ inline int32_t launch_interpolate_bilinear_2d_backward(
    const T* dout, T* dinput,
    int N, int C, int IH, int IW, int OH, int OW,
    int align_corners, double scale_h_factor, double scale_w_factor,
    cudaStream_t stream)
{
    if (N < 0 || C < 0 || IH < 0 || IW < 0 || OH < 0 || OW < 0) return 2;
    int64_t total = (int64_t)N * C * OH * OW;
    if (total == 0) return 0;
    if (IH == 0 || IW == 0) return 2;
    const bool ac = (align_corners != 0);
    double scale_h = interp_compute_scale(IH, OH, ac, scale_h_factor);
    double scale_w = interp_compute_scale(IW, OW, ac, scale_w_factor);
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    interpolate_bilinear_2d_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dout, dinput, N, C, IH, IW, OH, OW,
        align_corners, scale_h, scale_w);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::image

// =============================================================================
// INSTANTIATE macros.
// =============================================================================
//
// FFI signature (Phase 21):
//   N, C, IH, IW, OH, OW: shape ints (i32)
//   input/output (FW) or dout/dinput (BW): NCHW row-major buffers
//   workspace, workspace_bytes: reserved (zero)
//   align_corners: i32 (0 = false / PyTorch new-code default, nonzero = true)
//   scale_h_factor, scale_w_factor: f64; 0.0 means "derive from sizes",
//     otherwise interpreted as PyTorch-style SCALE (output/input)
//   stream: cudaStream_t

#define BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_INSTANTIATE(NAME, T)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                  \
        const void* input,                                                                     \
        void* output,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        int32_t align_corners,                                                                 \
        double scale_h_factor, double scale_w_factor,                                          \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (input == nullptr || output == nullptr) return 2;                                   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_interpolate_bilinear_2d<T>(                             \
            static_cast<const T*>(input),                                                      \
            static_cast<T*>(output),                                                           \
            N, C, IH, IW, OH, OW,                                                              \
            align_corners, scale_h_factor, scale_w_factor, stream);                            \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                  \
        const void* /*input*/,                                                                 \
        const void* /*output*/,                                                                \
        int32_t /*align_corners*/,                                                             \
        double /*scale_h_factor*/, double /*scale_w_factor*/)                                  \
    {                                                                                          \
        if (N < 0 || C < 0 || IH < 0 || IW < 0 || OH < 0 || OW < 0) return 2;                  \
        return 0;                                                                              \
    }

#define BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_BACKWARD_INSTANTIATE(NAME, T)                 \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                  \
        const void* dout,                                                                      \
        void* dinput,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        int32_t align_corners,                                                                 \
        double scale_h_factor, double scale_w_factor,                                          \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (dout == nullptr || dinput == nullptr) return 2;                                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_interpolate_bilinear_2d_backward<T>(                    \
            static_cast<const T*>(dout),                                                       \
            static_cast<T*>(dinput),                                                            \
            N, C, IH, IW, OH, OW,                                                              \
            align_corners, scale_h_factor, scale_w_factor, stream);                            \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                  \
        const void* /*dout*/,                                                                  \
        const void* /*dinput*/,                                                                \
        int32_t /*align_corners*/,                                                             \
        double /*scale_h_factor*/, double /*scale_w_factor*/)                                  \
    {                                                                                          \
        if (N < 0 || C < 0 || IH < 0 || IW < 0 || OH < 0 || OW < 0) return 2;                  \
        return 0;                                                                              \
    }

#endif // BARACUDA_INTERPOLATE_CUH
