// baracuda_roi.cuh
//
// Templated kernels and INSTANTIATE macros for `roi_align` and
// `roi_pool` (Phase 9 Category T).
//
// Layout convention:
//   input  : [N, C, H, W] (NCHW)
//   rois   : [num_rois, 5] — each row is (batch_idx, x1, y1, x2, y2)
//                            in INPUT-image pixel coordinates.
//   output : [num_rois, C, pooled_h, pooled_w]
//
// roi_align trailblazer config (PyTorch convention):
//   sampling_ratio = 0  → adaptive: ceil(roi_h / pooled_h) × ceil(roi_w / pooled_w)
//                        sample points per output cell, averaged.
//   aligned = false     → pre-0.6 PyTorch coordinate convention.
//
// roi_pool:
//   max-pool over the (possibly non-integer) RoI bin. FW saves argmax
//   linear index (per output cell) into a u32 buffer that BW reads to
//   route the gradient.
//
// Status codes mirror the family (0/2/3/5).

#ifndef BARACUDA_ROI_CUH
#define BARACUDA_ROI_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_indexing.cuh"  // scatter_atomic_add<T>

namespace baracuda { namespace image {

// Bilinear sample at (y, x) on a [H, W] plane with zero-pad for OOB.
template <typename T>
__device__ inline float bilinear_sample_zero(
    const T* plane, int H, int W, float y, float x)
{
    if (y < -1.0f || y > (float)H || x < -1.0f || x > (float)W) {
        return 0.0f;
    }
    if (y < 0) y = 0;
    if (x < 0) x = 0;
    int y0 = (int)floorf(y);
    int x0 = (int)floorf(x);
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    float wy1 = y - (float)y0;
    float wx1 = x - (float)x0;
    if (y0 >= H - 1) { y0 = y1 = H - 1; wy1 = 0; }
    if (x0 >= W - 1) { x0 = x1 = W - 1; wx1 = 0; }
    float wy0 = 1.0f - wy1;
    float wx0 = 1.0f - wx1;
    float v00 = (float)plane[(int64_t)y0 * W + x0];
    float v01 = (float)plane[(int64_t)y0 * W + x1];
    float v10 = (float)plane[(int64_t)y1 * W + x0];
    float v11 = (float)plane[(int64_t)y1 * W + x1];
    return wy0 * (wx0 * v00 + wx1 * v01)
         + wy1 * (wx0 * v10 + wx1 * v11);
}

// Bilinear scatter at (y, x) onto a [H, W] plane — atomic-add the
// gradient contribution to each of the 4 corners weighted by wij.
template <typename T>
__device__ inline void bilinear_scatter_zero(
    T* plane, int H, int W, float y, float x, float g)
{
    if (y < -1.0f || y > (float)H || x < -1.0f || x > (float)W) return;
    if (y < 0) y = 0;
    if (x < 0) x = 0;
    int y0 = (int)floorf(y);
    int x0 = (int)floorf(x);
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    float wy1 = y - (float)y0;
    float wx1 = x - (float)x0;
    if (y0 >= H - 1) { y0 = y1 = H - 1; wy1 = 0; }
    if (x0 >= W - 1) { x0 = x1 = W - 1; wx1 = 0; }
    float wy0 = 1.0f - wy1;
    float wx0 = 1.0f - wx1;
    baracuda::indexing::scatter_atomic_add<T>(&plane[(int64_t)y0 * W + x0], (T)(g * wy0 * wx0));
    baracuda::indexing::scatter_atomic_add<T>(&plane[(int64_t)y0 * W + x1], (T)(g * wy0 * wx1));
    baracuda::indexing::scatter_atomic_add<T>(&plane[(int64_t)y1 * W + x0], (T)(g * wy1 * wx0));
    baracuda::indexing::scatter_atomic_add<T>(&plane[(int64_t)y1 * W + x1], (T)(g * wy1 * wx1));
}

// =============================================================================
// roi_align forward — one thread per (r, c, ph, pw) output cell.
// sampling_ratio = 0 means adaptive (ceil(bin_h), ceil(bin_w)).
// =============================================================================

template <typename T>
__global__ void roi_align_kernel(
    const T* __restrict__ input,    // [N, C, H, W]
    const T* __restrict__ rois,     // [num_rois, 5]
    T* __restrict__ output,         // [num_rois, C, pooled_h, pooled_w]
    int N, int C, int H, int W,
    int num_rois, int pooled_h, int pooled_w,
    float spatial_scale, int sampling_ratio, int aligned)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)num_rois * C * pooled_h * pooled_w;
    float offset = aligned ? 0.5f : 0.0f;
    for (int64_t i = tid; i < total; i += step) {
        int pw = (int)(i % pooled_w);
        int ph = (int)((i / pooled_w) % pooled_h);
        int c  = (int)((i / ((int64_t)pooled_w * pooled_h)) % C);
        int r  = (int)(i / ((int64_t)pooled_w * pooled_h * C));
        int64_t roi_off = (int64_t)r * 5;
        int n = (int)rois[roi_off + 0];
        if (n < 0 || n >= N) { output[i] = (T)0; continue; }
        float x1 = (float)rois[roi_off + 1] * spatial_scale - offset;
        float y1 = (float)rois[roi_off + 2] * spatial_scale - offset;
        float x2 = (float)rois[roi_off + 3] * spatial_scale - offset;
        float y2 = (float)rois[roi_off + 4] * spatial_scale - offset;
        float roi_w_f = x2 - x1;
        float roi_h_f = y2 - y1;
        if (!aligned) {
            roi_w_f = roi_w_f < 1.0f ? 1.0f : roi_w_f;
            roi_h_f = roi_h_f < 1.0f ? 1.0f : roi_h_f;
        }
        float bin_w = roi_w_f / (float)pooled_w;
        float bin_h = roi_h_f / (float)pooled_h;
        int sx = sampling_ratio > 0 ? sampling_ratio : (int)ceilf(bin_w);
        int sy = sampling_ratio > 0 ? sampling_ratio : (int)ceilf(bin_h);
        if (sx < 1) sx = 1;
        if (sy < 1) sy = 1;
        const T* plane = input + ((int64_t)n * C + c) * (int64_t)H * W;
        float acc = 0.0f;
        for (int iy = 0; iy < sy; ++iy) {
            float y = y1 + (float)ph * bin_h + ((float)iy + 0.5f) * bin_h / (float)sy;
            for (int ix = 0; ix < sx; ++ix) {
                float x = x1 + (float)pw * bin_w + ((float)ix + 0.5f) * bin_w / (float)sx;
                acc += bilinear_sample_zero<T>(plane, H, W, y, x);
            }
        }
        float scale = 1.0f / (float)(sy * sx);
        output[i] = (T)(acc * scale);
    }
}

template <typename T>
__host__ inline int32_t launch_roi_align(
    const T* input, const T* rois, T* output,
    int N, int C, int H, int W,
    int num_rois, int pooled_h, int pooled_w,
    float spatial_scale, int sampling_ratio, int aligned,
    cudaStream_t stream)
{
    if (N < 0 || C < 0 || H < 0 || W < 0 || num_rois < 0 || pooled_h < 0 || pooled_w < 0)
        return 2;
    int64_t total = (int64_t)num_rois * C * pooled_h * pooled_w;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    roi_align_kernel<T><<<blocks, kBlock, 0, stream>>>(
        input, rois, output, N, C, H, W,
        num_rois, pooled_h, pooled_w,
        spatial_scale, sampling_ratio, aligned);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// roi_align backward — one thread per output cell, scatters bilinear
// gradient contributions into dinput. Caller pre-zeros dinput.
// =============================================================================

template <typename T>
__global__ void roi_align_backward_kernel(
    const T* __restrict__ dout,     // [num_rois, C, pooled_h, pooled_w]
    const T* __restrict__ rois,     // [num_rois, 5]
    T* __restrict__ dinput,         // [N, C, H, W]
    int N, int C, int H, int W,
    int num_rois, int pooled_h, int pooled_w,
    float spatial_scale, int sampling_ratio, int aligned)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)num_rois * C * pooled_h * pooled_w;
    float offset = aligned ? 0.5f : 0.0f;
    for (int64_t i = tid; i < total; i += step) {
        int pw = (int)(i % pooled_w);
        int ph = (int)((i / pooled_w) % pooled_h);
        int c  = (int)((i / ((int64_t)pooled_w * pooled_h)) % C);
        int r  = (int)(i / ((int64_t)pooled_w * pooled_h * C));
        int64_t roi_off = (int64_t)r * 5;
        int n = (int)rois[roi_off + 0];
        if (n < 0 || n >= N) continue;
        float x1 = (float)rois[roi_off + 1] * spatial_scale - offset;
        float y1 = (float)rois[roi_off + 2] * spatial_scale - offset;
        float x2 = (float)rois[roi_off + 3] * spatial_scale - offset;
        float y2 = (float)rois[roi_off + 4] * spatial_scale - offset;
        float roi_w_f = x2 - x1;
        float roi_h_f = y2 - y1;
        if (!aligned) {
            roi_w_f = roi_w_f < 1.0f ? 1.0f : roi_w_f;
            roi_h_f = roi_h_f < 1.0f ? 1.0f : roi_h_f;
        }
        float bin_w = roi_w_f / (float)pooled_w;
        float bin_h = roi_h_f / (float)pooled_h;
        int sx = sampling_ratio > 0 ? sampling_ratio : (int)ceilf(bin_w);
        int sy = sampling_ratio > 0 ? sampling_ratio : (int)ceilf(bin_h);
        if (sx < 1) sx = 1;
        if (sy < 1) sy = 1;
        T* plane = dinput + ((int64_t)n * C + c) * (int64_t)H * W;
        float g = (float)dout[i];
        float scale = g / (float)(sy * sx);
        for (int iy = 0; iy < sy; ++iy) {
            float y = y1 + (float)ph * bin_h + ((float)iy + 0.5f) * bin_h / (float)sy;
            for (int ix = 0; ix < sx; ++ix) {
                float x = x1 + (float)pw * bin_w + ((float)ix + 0.5f) * bin_w / (float)sx;
                bilinear_scatter_zero<T>(plane, H, W, y, x, scale);
            }
        }
    }
}

template <typename T>
__host__ inline int32_t launch_roi_align_backward(
    const T* dout, const T* rois, T* dinput,
    int N, int C, int H, int W,
    int num_rois, int pooled_h, int pooled_w,
    float spatial_scale, int sampling_ratio, int aligned,
    cudaStream_t stream)
{
    if (N < 0 || C < 0 || H < 0 || W < 0 || num_rois < 0 || pooled_h < 0 || pooled_w < 0)
        return 2;
    int64_t total = (int64_t)num_rois * C * pooled_h * pooled_w;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    roi_align_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dout, rois, dinput, N, C, H, W,
        num_rois, pooled_h, pooled_w,
        spatial_scale, sampling_ratio, aligned);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// roi_pool forward — max-pool variant. One thread per (r, c, ph, pw)
// output cell. Bins are integer-rounded RoI rectangles in INPUT-pixel
// coords scaled by spatial_scale. Saves argmax (int32 linear cell index
// into the [H, W] plane) for BW; argmax = -1 marks an empty bin.
// =============================================================================

template <typename T>
__global__ void roi_pool_kernel(
    const T* __restrict__ input,    // [N, C, H, W]
    const T* __restrict__ rois,     // [num_rois, 5]
    T* __restrict__ output,         // [num_rois, C, pooled_h, pooled_w]
    int32_t* __restrict__ argmax,   // [num_rois, C, pooled_h, pooled_w]
    int N, int C, int H, int W,
    int num_rois, int pooled_h, int pooled_w,
    float spatial_scale)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)num_rois * C * pooled_h * pooled_w;
    for (int64_t i = tid; i < total; i += step) {
        int pw = (int)(i % pooled_w);
        int ph = (int)((i / pooled_w) % pooled_h);
        int c  = (int)((i / ((int64_t)pooled_w * pooled_h)) % C);
        int r  = (int)(i / ((int64_t)pooled_w * pooled_h * C));
        int64_t roi_off = (int64_t)r * 5;
        int n = (int)rois[roi_off + 0];
        if (n < 0 || n >= N) { output[i] = (T)0; argmax[i] = -1; continue; }
        int rx1 = (int)roundf((float)rois[roi_off + 1] * spatial_scale);
        int ry1 = (int)roundf((float)rois[roi_off + 2] * spatial_scale);
        int rx2 = (int)roundf((float)rois[roi_off + 3] * spatial_scale);
        int ry2 = (int)roundf((float)rois[roi_off + 4] * spatial_scale);
        int rh = rx2 - rx1 + 1; (void)rh;
        int roi_w = rx2 - rx1 + 1; if (roi_w < 1) roi_w = 1;
        int roi_h = ry2 - ry1 + 1; if (roi_h < 1) roi_h = 1;
        float bin_w = (float)roi_w / (float)pooled_w;
        float bin_h = (float)roi_h / (float)pooled_h;
        int hs = (int)floorf((float)ph * bin_h);
        int he = (int)ceilf((float)(ph + 1) * bin_h);
        int ws = (int)floorf((float)pw * bin_w);
        int we = (int)ceilf((float)(pw + 1) * bin_w);
        hs = ry1 + hs; he = ry1 + he;
        ws = rx1 + ws; we = rx1 + we;
        // Clip to plane.
        hs = hs < 0 ? 0 : (hs > H ? H : hs);
        he = he < 0 ? 0 : (he > H ? H : he);
        ws = ws < 0 ? 0 : (ws > W ? W : ws);
        we = we < 0 ? 0 : (we > W ? W : we);
        bool empty = (he <= hs) || (we <= ws);
        const T* plane = input + ((int64_t)n * C + c) * (int64_t)H * W;
        float best = empty ? 0.0f : -3.4e38f;
        int32_t argbest = -1;
        for (int y = hs; y < he; ++y) {
            for (int x = ws; x < we; ++x) {
                int32_t idx = y * W + x;
                float v = (float)plane[idx];
                if (v > best) { best = v; argbest = idx; }
            }
        }
        output[i] = empty ? (T)0 : (T)best;
        argmax[i] = empty ? -1 : argbest;
    }
}

template <typename T>
__host__ inline int32_t launch_roi_pool(
    const T* input, const T* rois, T* output, int32_t* argmax,
    int N, int C, int H, int W,
    int num_rois, int pooled_h, int pooled_w,
    float spatial_scale, cudaStream_t stream)
{
    if (N < 0 || C < 0 || H < 0 || W < 0 || num_rois < 0 || pooled_h < 0 || pooled_w < 0)
        return 2;
    int64_t total = (int64_t)num_rois * C * pooled_h * pooled_w;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    roi_pool_kernel<T><<<blocks, kBlock, 0, stream>>>(
        input, rois, output, argmax,
        N, C, H, W, num_rois, pooled_h, pooled_w, spatial_scale);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// roi_pool backward — atomic-add `dout` into `dinput` at the saved
// argmax cell. Caller pre-zeros dinput.
// =============================================================================

template <typename T>
__global__ void roi_pool_backward_kernel(
    const T* __restrict__ dout,         // [num_rois, C, pooled_h, pooled_w]
    const T* __restrict__ rois,         // [num_rois, 5]
    const int32_t* __restrict__ argmax, // [num_rois, C, pooled_h, pooled_w]
    T* __restrict__ dinput,             // [N, C, H, W]
    int N, int C, int H, int W,
    int num_rois, int pooled_h, int pooled_w)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)num_rois * C * pooled_h * pooled_w;
    for (int64_t i = tid; i < total; i += step) {
        int32_t arg = argmax[i];
        if (arg < 0) continue;
        int c = (int)((i / ((int64_t)pooled_w * pooled_h)) % C);
        int r = (int)(i / ((int64_t)pooled_w * pooled_h * C));
        int64_t roi_off = (int64_t)r * 5;
        int n = (int)rois[roi_off + 0];
        if (n < 0 || n >= N) continue;
        T* plane = dinput + ((int64_t)n * C + c) * (int64_t)H * W;
        baracuda::indexing::scatter_atomic_add<T>(&plane[arg], dout[i]);
    }
}

template <typename T>
__host__ inline int32_t launch_roi_pool_backward(
    const T* dout, const T* rois, const int32_t* argmax, T* dinput,
    int N, int C, int H, int W,
    int num_rois, int pooled_h, int pooled_w,
    cudaStream_t stream)
{
    if (N < 0 || C < 0 || H < 0 || W < 0 || num_rois < 0 || pooled_h < 0 || pooled_w < 0)
        return 2;
    int64_t total = (int64_t)num_rois * C * pooled_h * pooled_w;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    roi_pool_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dout, rois, argmax, dinput, N, C, H, W,
        num_rois, pooled_h, pooled_w);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::image

// =============================================================================
// INSTANTIATE macros.
// =============================================================================

#define BARACUDA_KERNELS_ROI_ALIGN_INSTANTIATE(NAME, T)                                        \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t H, int32_t W,                                            \
        int32_t num_rois, int32_t pooled_h, int32_t pooled_w,                                  \
        float spatial_scale, int32_t sampling_ratio, int32_t aligned,                          \
        const void* input,                                                                     \
        const void* rois,                                                                      \
        void* output,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (input == nullptr || rois == nullptr || output == nullptr) return 2;                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_roi_align<T>(                                           \
            static_cast<const T*>(input),                                                      \
            static_cast<const T*>(rois),                                                       \
            static_cast<T*>(output),                                                           \
            N, C, H, W, num_rois, pooled_h, pooled_w,                                          \
            spatial_scale, sampling_ratio, aligned, stream);                                   \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                \
        int32_t N, int32_t C, int32_t H, int32_t W,                                            \
        int32_t num_rois, int32_t pooled_h, int32_t pooled_w,                                  \
        float /*spatial_scale*/, int32_t /*sampling_ratio*/, int32_t /*aligned*/,              \
        const void* /*input*/,                                                                 \
        const void* /*rois*/,                                                                  \
        const void* /*output*/)                                                                \
    {                                                                                          \
        if (N < 0 || C < 0 || H < 0 || W < 0) return 2;                                        \
        if (num_rois < 0 || pooled_h <= 0 || pooled_w <= 0) return 2;                          \
        return 0;                                                                              \
    }

#define BARACUDA_KERNELS_ROI_ALIGN_BACKWARD_INSTANTIATE(NAME, T)                               \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t H, int32_t W,                                            \
        int32_t num_rois, int32_t pooled_h, int32_t pooled_w,                                  \
        float spatial_scale, int32_t sampling_ratio, int32_t aligned,                          \
        const void* dout,                                                                      \
        const void* rois,                                                                      \
        void* dinput,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (dout == nullptr || rois == nullptr || dinput == nullptr) return 2;                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_roi_align_backward<T>(                                  \
            static_cast<const T*>(dout),                                                       \
            static_cast<const T*>(rois),                                                       \
            static_cast<T*>(dinput),                                                           \
            N, C, H, W, num_rois, pooled_h, pooled_w,                                          \
            spatial_scale, sampling_ratio, aligned, stream);                                   \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                \
        int32_t N, int32_t C, int32_t H, int32_t W,                                            \
        int32_t num_rois, int32_t pooled_h, int32_t pooled_w,                                  \
        float /*spatial_scale*/, int32_t /*sampling_ratio*/, int32_t /*aligned*/,              \
        const void* /*dout*/,                                                                  \
        const void* /*rois*/,                                                                  \
        const void* /*dinput*/)                                                                \
    {                                                                                          \
        if (N < 0 || C < 0 || H < 0 || W < 0) return 2;                                        \
        if (num_rois < 0 || pooled_h <= 0 || pooled_w <= 0) return 2;                          \
        return 0;                                                                              \
    }

#define BARACUDA_KERNELS_ROI_POOL_INSTANTIATE(NAME, T)                                         \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t H, int32_t W,                                            \
        int32_t num_rois, int32_t pooled_h, int32_t pooled_w,                                  \
        float spatial_scale,                                                                   \
        const void* input,                                                                     \
        const void* rois,                                                                      \
        void* output,                                                                          \
        void* argmax,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (input == nullptr || rois == nullptr || output == nullptr || argmax == nullptr)    \
            return 2;                                                                          \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_roi_pool<T>(                                            \
            static_cast<const T*>(input),                                                      \
            static_cast<const T*>(rois),                                                       \
            static_cast<T*>(output),                                                           \
            static_cast<int32_t*>(argmax),                                                     \
            N, C, H, W, num_rois, pooled_h, pooled_w,                                          \
            spatial_scale, stream);                                                            \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                \
        int32_t N, int32_t C, int32_t H, int32_t W,                                            \
        int32_t num_rois, int32_t pooled_h, int32_t pooled_w,                                  \
        float /*spatial_scale*/,                                                               \
        const void* /*input*/,                                                                 \
        const void* /*rois*/,                                                                  \
        const void* /*output*/,                                                                \
        const void* /*argmax*/)                                                                \
    {                                                                                          \
        if (N < 0 || C < 0 || H < 0 || W < 0) return 2;                                        \
        if (num_rois < 0 || pooled_h <= 0 || pooled_w <= 0) return 2;                          \
        return 0;                                                                              \
    }

#define BARACUDA_KERNELS_ROI_POOL_BACKWARD_INSTANTIATE(NAME, T)                                \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t H, int32_t W,                                            \
        int32_t num_rois, int32_t pooled_h, int32_t pooled_w,                                  \
        const void* dout,                                                                      \
        const void* rois,                                                                      \
        const void* argmax,                                                                    \
        void* dinput,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (dout == nullptr || rois == nullptr || argmax == nullptr || dinput == nullptr)     \
            return 2;                                                                          \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_roi_pool_backward<T>(                                   \
            static_cast<const T*>(dout),                                                       \
            static_cast<const T*>(rois),                                                       \
            static_cast<const int32_t*>(argmax),                                               \
            static_cast<T*>(dinput),                                                           \
            N, C, H, W, num_rois, pooled_h, pooled_w, stream);                                 \
    }                                                                                          \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                \
        int32_t N, int32_t C, int32_t H, int32_t W,                                            \
        int32_t num_rois, int32_t pooled_h, int32_t pooled_w,                                  \
        const void* /*dout*/,                                                                  \
        const void* /*rois*/,                                                                  \
        const void* /*argmax*/,                                                                \
        const void* /*dinput*/)                                                                \
    {                                                                                          \
        if (N < 0 || C < 0 || H < 0 || W < 0) return 2;                                        \
        if (num_rois < 0 || pooled_h <= 0 || pooled_w <= 0) return 2;                          \
        return 0;                                                                              \
    }

#endif // BARACUDA_ROI_CUH
