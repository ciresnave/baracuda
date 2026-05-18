// baracuda_interpolate.cuh
//
// Templated kernels and INSTANTIATE macros for `interpolate` (Phase 9
// Category T — image / spatial transforms). Trailblazer mode is
// bilinear-2D; other modes (nearest, bicubic, trilinear, linear, area)
// have INSTANTIATE-shaped stubs that return status `3` (unsupported) so
// the FFI surface is symmetric across modes.
//
// NCHW layout. `align_corners=false` (PyTorch default for new code).
// Coordinate mapping for `align_corners=false`:
//   src_x = (dst_x + 0.5) * (src_w / dst_w) - 0.5
//   src_y = (dst_y + 0.5) * (src_h / dst_h) - 0.5
// Out-of-range samples are clamped to the input boundary (PyTorch's
// `interpolate` for bilinear uses zero-pad-equivalent via clamp — i.e.
// edge replication — which matches PyTorch's behavior).
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

#include "baracuda_indexing.cuh"  // scatter_atomic_add<T>

namespace baracuda { namespace image {

// Coordinate mapping for `align_corners=false`.
//   src = (dst + 0.5) * (src_size / dst_size) - 0.5
__host__ __device__ inline float coord_dst_to_src_align_false(
    int dst, int dst_size, int src_size)
{
    return ((float)dst + 0.5f) * ((float)src_size / (float)dst_size) - 0.5f;
}

// =============================================================================
// interpolate_bilinear_2d forward — one thread per (n, c, oh, ow) output.
// =============================================================================

template <typename T>
__global__ void interpolate_bilinear_2d_kernel(
    const T* __restrict__ input,    // [N, C, IH, IW]
    T* __restrict__ output,         // [N, C, OH, OW]
    int N, int C, int IH, int IW, int OH, int OW)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)N * (int64_t)C * (int64_t)OH * (int64_t)OW;
    for (int64_t i = tid; i < total; i += step) {
        int ow = (int)(i % OW);
        int oh = (int)((i / OW) % OH);
        int c  = (int)((i / ((int64_t)OW * OH)) % C);
        int n  = (int)(i / ((int64_t)OW * OH * C));

        float sx = coord_dst_to_src_align_false(ow, OW, IW);
        float sy = coord_dst_to_src_align_false(oh, OH, IH);
        int x0 = (int)floorf(sx);
        int y0 = (int)floorf(sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float wx1 = sx - (float)x0;
        float wy1 = sy - (float)y0;
        float wx0 = 1.0f - wx1;
        float wy0 = 1.0f - wy1;
        // Clamp (edge replicate) for OOB samples.
        int cx0 = x0 < 0 ? 0 : (x0 >= IW ? IW - 1 : x0);
        int cx1 = x1 < 0 ? 0 : (x1 >= IW ? IW - 1 : x1);
        int cy0 = y0 < 0 ? 0 : (y0 >= IH ? IH - 1 : y0);
        int cy1 = y1 < 0 ? 0 : (y1 >= IH ? IH - 1 : y1);
        const T* in_nc = input + ((int64_t)n * C + c) * (int64_t)IH * IW;
        float v00 = (float)in_nc[(int64_t)cy0 * IW + cx0];
        float v01 = (float)in_nc[(int64_t)cy0 * IW + cx1];
        float v10 = (float)in_nc[(int64_t)cy1 * IW + cx0];
        float v11 = (float)in_nc[(int64_t)cy1 * IW + cx1];
        float out = wy0 * (wx0 * v00 + wx1 * v01)
                  + wy1 * (wx0 * v10 + wx1 * v11);
        output[i] = (T)out;
    }
}

template <typename T>
__host__ inline int32_t launch_interpolate_bilinear_2d(
    const T* input, T* output,
    int N, int C, int IH, int IW, int OH, int OW,
    cudaStream_t stream)
{
    if (N < 0 || C < 0 || IH < 0 || IW < 0 || OH < 0 || OW < 0) return 2;
    int64_t total = (int64_t)N * C * OH * OW;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    interpolate_bilinear_2d_kernel<T><<<blocks, kBlock, 0, stream>>>(
        input, output, N, C, IH, IW, OH, OW);
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
    int N, int C, int IH, int IW, int OH, int OW)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)N * C * OH * OW;
    for (int64_t i = tid; i < total; i += step) {
        int ow = (int)(i % OW);
        int oh = (int)((i / OW) % OH);
        int c  = (int)((i / ((int64_t)OW * OH)) % C);
        int n  = (int)(i / ((int64_t)OW * OH * C));

        float sx = coord_dst_to_src_align_false(ow, OW, IW);
        float sy = coord_dst_to_src_align_false(oh, OH, IH);
        int x0 = (int)floorf(sx);
        int y0 = (int)floorf(sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float wx1 = sx - (float)x0;
        float wy1 = sy - (float)y0;
        float wx0 = 1.0f - wx1;
        float wy0 = 1.0f - wy1;
        int cx0 = x0 < 0 ? 0 : (x0 >= IW ? IW - 1 : x0);
        int cx1 = x1 < 0 ? 0 : (x1 >= IW ? IW - 1 : x1);
        int cy0 = y0 < 0 ? 0 : (y0 >= IH ? IH - 1 : y0);
        int cy1 = y1 < 0 ? 0 : (y1 >= IH ? IH - 1 : y1);
        T* di_nc = dinput + ((int64_t)n * C + c) * (int64_t)IH * IW;
        float g = (float)dout[i];
        baracuda::indexing::scatter_atomic_add<T>(&di_nc[(int64_t)cy0 * IW + cx0], (T)(g * wy0 * wx0));
        baracuda::indexing::scatter_atomic_add<T>(&di_nc[(int64_t)cy0 * IW + cx1], (T)(g * wy0 * wx1));
        baracuda::indexing::scatter_atomic_add<T>(&di_nc[(int64_t)cy1 * IW + cx0], (T)(g * wy1 * wx0));
        baracuda::indexing::scatter_atomic_add<T>(&di_nc[(int64_t)cy1 * IW + cx1], (T)(g * wy1 * wx1));
    }
}

template <typename T>
__host__ inline int32_t launch_interpolate_bilinear_2d_backward(
    const T* dout, T* dinput,
    int N, int C, int IH, int IW, int OH, int OW,
    cudaStream_t stream)
{
    if (N < 0 || C < 0 || IH < 0 || IW < 0 || OH < 0 || OW < 0) return 2;
    int64_t total = (int64_t)N * C * OH * OW;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    interpolate_bilinear_2d_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dout, dinput, N, C, IH, IW, OH, OW);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::image

// =============================================================================
// INSTANTIATE macros.
// =============================================================================

#define BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_INSTANTIATE(NAME, T)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                  \
        const void* input,                                                                     \
        void* output,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (input == nullptr || output == nullptr) return 2;                                   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_interpolate_bilinear_2d<T>(                             \
            static_cast<const T*>(input),                                                      \
            static_cast<T*>(output),                                                           \
            N, C, IH, IW, OH, OW, stream);                                                     \
    }

#define BARACUDA_KERNELS_INTERPOLATE_BILINEAR_2D_BACKWARD_INSTANTIATE(NAME, T)                 \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                  \
        const void* dout,                                                                      \
        void* dinput,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (dout == nullptr || dinput == nullptr) return 2;                                    \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_interpolate_bilinear_2d_backward<T>(                    \
            static_cast<const T*>(dout),                                                       \
            static_cast<T*>(dinput),                                                           \
            N, C, IH, IW, OH, OW, stream);                                                     \
    }

#endif // BARACUDA_INTERPOLATE_CUH
