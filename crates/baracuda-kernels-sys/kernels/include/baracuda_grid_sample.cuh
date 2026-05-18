// baracuda_grid_sample.cuh
//
// Templated kernels and INSTANTIATE macros for `grid_sample` and
// `affine_grid` (Phase 9 Category T).
//
// Trailblazer convention (PyTorch defaults):
//   - 2-D NCHW input.
//   - bilinear interpolation.
//   - padding_mode = 'zeros' (OOB samples contribute 0).
//   - align_corners = false.
//
// Coordinate mapping for `align_corners=false`:
//   norm in [-1, 1] over [-0.5, src_size - 0.5]:
//     src = ((norm + 1) * src_size - 1) / 2 = (norm + 1) * src_size / 2 - 0.5
//
// Status codes mirror the family (0/2/3/5).

#ifndef BARACUDA_GRID_SAMPLE_CUH
#define BARACUDA_GRID_SAMPLE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_indexing.cuh"  // scatter_atomic_add<T>

namespace baracuda { namespace image {

__host__ __device__ inline float grid_to_src_align_false(float norm, int src_size)
{
    return ((norm + 1.0f) * (float)src_size - 1.0f) * 0.5f;
}

// d(src) / d(norm) for align_corners=false.
__host__ __device__ inline float grid_to_src_align_false_grad(int src_size)
{
    return 0.5f * (float)src_size;
}

// =============================================================================
// grid_sample_2d forward — one thread per (n, c, oh, ow).
// `grid` is [N, OH, OW, 2] with (x, y) normalized in [-1, 1]; OOB = zero.
// =============================================================================

template <typename T>
__global__ void grid_sample_2d_kernel(
    const T* __restrict__ input,   // [N, C, IH, IW]
    const T* __restrict__ grid,    // [N, OH, OW, 2]
    T* __restrict__ output,        // [N, C, OH, OW]
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

        int64_t g_off = (((int64_t)n * OH + oh) * OW + ow) * 2;
        float nx = (float)grid[g_off + 0];
        float ny = (float)grid[g_off + 1];
        float sx = grid_to_src_align_false(nx, IW);
        float sy = grid_to_src_align_false(ny, IH);
        int x0 = (int)floorf(sx);
        int y0 = (int)floorf(sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float wx1 = sx - (float)x0;
        float wy1 = sy - (float)y0;
        float wx0 = 1.0f - wx1;
        float wy0 = 1.0f - wy1;
        // zeros padding: OOB contributes 0.
        const T* in_nc = input + ((int64_t)n * C + c) * (int64_t)IH * IW;
        auto fetch = [&](int yy, int xx) -> float {
            if (yy < 0 || yy >= IH || xx < 0 || xx >= IW) return 0.0f;
            return (float)in_nc[(int64_t)yy * IW + xx];
        };
        float v00 = fetch(y0, x0);
        float v01 = fetch(y0, x1);
        float v10 = fetch(y1, x0);
        float v11 = fetch(y1, x1);
        float out = wy0 * (wx0 * v00 + wx1 * v01)
                  + wy1 * (wx0 * v10 + wx1 * v11);
        output[i] = (T)out;
    }
}

template <typename T>
__host__ inline int32_t launch_grid_sample_2d(
    const T* input, const T* grid, T* output,
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
    grid_sample_2d_kernel<T><<<blocks, kBlock, 0, stream>>>(
        input, grid, output, N, C, IH, IW, OH, OW);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// grid_sample_2d backward — one thread per (n, c, oh, ow).
// Atomic-adds gradient to `dinput` at the 4 corners weighted by wij;
// accumulates per-(n, oh, ow) coordinate gradient `dgrid` across C.
// Caller pre-zeros dinput AND dgrid.
// =============================================================================

template <typename T>
__global__ void grid_sample_2d_backward_kernel(
    const T* __restrict__ dout,    // [N, C, OH, OW]
    const T* __restrict__ input,   // [N, C, IH, IW]
    const T* __restrict__ grid,    // [N, OH, OW, 2]
    T* __restrict__ dinput,        // [N, C, IH, IW]
    T* __restrict__ dgrid,         // [N, OH, OW, 2]
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

        int64_t g_off = (((int64_t)n * OH + oh) * OW + ow) * 2;
        float nx = (float)grid[g_off + 0];
        float ny = (float)grid[g_off + 1];
        float sx = grid_to_src_align_false(nx, IW);
        float sy = grid_to_src_align_false(ny, IH);
        int x0 = (int)floorf(sx);
        int y0 = (int)floorf(sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float wx1 = sx - (float)x0;
        float wy1 = sy - (float)y0;
        float wx0 = 1.0f - wx1;
        float wy0 = 1.0f - wy1;
        float gdy = (float)dout[i];
        const T* in_nc = input + ((int64_t)n * C + c) * (int64_t)IH * IW;
        T* di_nc = dinput + ((int64_t)n * C + c) * (int64_t)IH * IW;

        auto fetch = [&](int yy, int xx) -> float {
            if (yy < 0 || yy >= IH || xx < 0 || xx >= IW) return 0.0f;
            return (float)in_nc[(int64_t)yy * IW + xx];
        };
        auto scatter = [&](int yy, int xx, float v) {
            if (yy < 0 || yy >= IH || xx < 0 || xx >= IW) return;
            baracuda::indexing::scatter_atomic_add<T>(
                &di_nc[(int64_t)yy * IW + xx], (T)v);
        };
        // dinput contributions.
        scatter(y0, x0, gdy * wy0 * wx0);
        scatter(y0, x1, gdy * wy0 * wx1);
        scatter(y1, x0, gdy * wy1 * wx0);
        scatter(y1, x1, gdy * wy1 * wx1);

        // dgrid: chain rule through the bilinear interpolant.
        //   out = Σ wy * wx * v
        //   d out / d sx = Σ wy * dwx/dsx * v  with dwx0/dsx = -1, dwx1/dsx = +1
        //   d out / d sy = Σ dwy/dsy * wx * v
        //   d sx / d nx = src_w / 2 ; d sy / d ny = src_h / 2 (align_corners=false)
        float v00 = fetch(y0, x0);
        float v01 = fetch(y0, x1);
        float v10 = fetch(y1, x0);
        float v11 = fetch(y1, x1);
        float dout_dsx = wy0 * (-v00 + v01) + wy1 * (-v10 + v11);
        float dout_dsy = (-(wx0 * v00 + wx1 * v01)) + (wx0 * v10 + wx1 * v11);
        float dnx = gdy * dout_dsx * grid_to_src_align_false_grad(IW);
        float dny = gdy * dout_dsy * grid_to_src_align_false_grad(IH);
        baracuda::indexing::scatter_atomic_add<T>(&dgrid[g_off + 0], (T)dnx);
        baracuda::indexing::scatter_atomic_add<T>(&dgrid[g_off + 1], (T)dny);
    }
}

template <typename T>
__host__ inline int32_t launch_grid_sample_2d_backward(
    const T* dout, const T* input, const T* grid,
    T* dinput, T* dgrid,
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
    grid_sample_2d_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dout, input, grid, dinput, dgrid, N, C, IH, IW, OH, OW);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// affine_grid_2d — generate normalized coords for a 2x3 affine
// transform per batch. Output [N, OH, OW, 2] (matching grid_sample).
// Pixel centers in `align_corners=false` are at norm coords:
//   base_x = (2 * ow + 1) / OW - 1     ∈ [-1 + 1/OW, 1 - 1/OW]
//   base_y = (2 * oh + 1) / OH - 1
// Then x = theta[n, 0, 0] * base_x + theta[n, 0, 1] * base_y + theta[n, 0, 2]
//      y = theta[n, 1, 0] * base_x + theta[n, 1, 1] * base_y + theta[n, 1, 2]
// =============================================================================

template <typename T>
__global__ void affine_grid_2d_kernel(
    const T* __restrict__ theta,   // [N, 2, 3]
    T* __restrict__ grid,          // [N, OH, OW, 2]
    int N, int OH, int OW)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)N * OH * OW;
    for (int64_t i = tid; i < total; i += step) {
        int ow = (int)(i % OW);
        int oh = (int)((i / OW) % OH);
        int n  = (int)(i / ((int64_t)OW * OH));

        float bx = (OW > 1) ? (2.0f * (float)ow + 1.0f) / (float)OW - 1.0f : 0.0f;
        float by = (OH > 1) ? (2.0f * (float)oh + 1.0f) / (float)OH - 1.0f : 0.0f;

        int64_t t_base = (int64_t)n * 6;
        float t00 = (float)theta[t_base + 0];
        float t01 = (float)theta[t_base + 1];
        float t02 = (float)theta[t_base + 2];
        float t10 = (float)theta[t_base + 3];
        float t11 = (float)theta[t_base + 4];
        float t12 = (float)theta[t_base + 5];
        float x = t00 * bx + t01 * by + t02;
        float y = t10 * bx + t11 * by + t12;
        int64_t g = (((int64_t)n * OH + oh) * OW + ow) * 2;
        grid[g + 0] = (T)x;
        grid[g + 1] = (T)y;
    }
}

template <typename T>
__host__ inline int32_t launch_affine_grid_2d(
    const T* theta, T* grid, int N, int OH, int OW, cudaStream_t stream)
{
    if (N < 0 || OH < 0 || OW < 0) return 2;
    int64_t total = (int64_t)N * OH * OW;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    affine_grid_2d_kernel<T><<<blocks, kBlock, 0, stream>>>(theta, grid, N, OH, OW);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::image

// =============================================================================
// INSTANTIATE macros.
// =============================================================================

#define BARACUDA_KERNELS_GRID_SAMPLE_2D_INSTANTIATE(NAME, T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                  \
        const void* input,                                                                     \
        const void* grid,                                                                      \
        void* output,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (input == nullptr || grid == nullptr || output == nullptr) return 2;                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_grid_sample_2d<T>(                                      \
            static_cast<const T*>(input),                                                      \
            static_cast<const T*>(grid),                                                       \
            static_cast<T*>(output),                                                           \
            N, C, IH, IW, OH, OW, stream);                                                     \
    }

#define BARACUDA_KERNELS_GRID_SAMPLE_2D_BACKWARD_INSTANTIATE(NAME, T)                          \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                  \
        const void* dout,                                                                      \
        const void* input,                                                                     \
        const void* grid,                                                                      \
        void* dinput,                                                                          \
        void* dgrid,                                                                           \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (dout == nullptr || input == nullptr || grid == nullptr                             \
            || dinput == nullptr || dgrid == nullptr) return 2;                                \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_grid_sample_2d_backward<T>(                             \
            static_cast<const T*>(dout),                                                       \
            static_cast<const T*>(input),                                                      \
            static_cast<const T*>(grid),                                                       \
            static_cast<T*>(dinput),                                                           \
            static_cast<T*>(dgrid),                                                            \
            N, C, IH, IW, OH, OW, stream);                                                     \
    }

#define BARACUDA_KERNELS_AFFINE_GRID_2D_INSTANTIATE(NAME, T)                                   \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t OH, int32_t OW,                                                     \
        const void* theta,                                                                     \
        void* grid,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (theta == nullptr || grid == nullptr) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_affine_grid_2d<T>(                                      \
            static_cast<const T*>(theta),                                                      \
            static_cast<T*>(grid),                                                             \
            N, OH, OW, stream);                                                                \
    }

#endif // BARACUDA_GRID_SAMPLE_CUH
