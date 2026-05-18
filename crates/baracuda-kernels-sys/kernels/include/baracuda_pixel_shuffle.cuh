// baracuda_pixel_shuffle.cuh
//
// Templated kernels and INSTANTIATE macros for `pixel_shuffle` and
// `pixel_unshuffle` (Phase 9 Category T). Pure index permutation — no
// arithmetic. They are each other's backward.
//
// pixel_shuffle (upscale_factor = r):
//   input  [N, C * r * r, H, W]
//   output [N, C,         H*r, W*r]
//   formula:
//     oh = h * r + (idx_within_c_block / r)
//     ow = w * r + (idx_within_c_block % r)
//     where idx_within_c_block ∈ [0, r*r), packed in the C dim.
//
// pixel_unshuffle (downscale_factor = r):
//   inverse permutation.
//
// Dtype-agnostic — covers {f32, f64, f16, bf16} (memory-bound).
//
// Status codes mirror the family (0/2/5).

#ifndef BARACUDA_PIXEL_SHUFFLE_CUH
#define BARACUDA_PIXEL_SHUFFLE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace baracuda { namespace image {

// =============================================================================
// pixel_shuffle: input [N, C * r * r, H, W] → output [N, C, H*r, W*r].
// One thread per OUTPUT cell (N, C, OH, OW).
// =============================================================================

template <typename T>
__global__ void pixel_shuffle_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N, int C, int H, int W, int r)
{
    int OH = H * r;
    int OW = W * r;
    int CIn = C * r * r;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)N * C * OH * OW;
    for (int64_t i = tid; i < total; i += step) {
        int ow = (int)(i % OW);
        int oh = (int)((i / OW) % OH);
        int c  = (int)((i / ((int64_t)OW * OH)) % C);
        int n  = (int)(i / ((int64_t)OW * OH * C));
        int h = oh / r;
        int w = ow / r;
        int dh = oh - h * r;
        int dw = ow - w * r;
        int cin = c * r * r + dh * r + dw;
        int64_t in_idx = (((int64_t)n * CIn + cin) * H + h) * (int64_t)W + w;
        output[i] = input[in_idx];
    }
}

template <typename T>
__host__ inline int32_t launch_pixel_shuffle(
    const T* input, T* output,
    int N, int C, int H, int W, int r,
    cudaStream_t stream)
{
    if (N < 0 || C < 0 || H < 0 || W < 0 || r <= 0) return 2;
    int64_t total = (int64_t)N * C * (H * r) * (W * r);
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    pixel_shuffle_kernel<T><<<blocks, kBlock, 0, stream>>>(input, output, N, C, H, W, r);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// pixel_unshuffle: input [N, C, H * r, W * r] → output [N, C * r * r, H, W].
// Inverse permutation. One thread per OUTPUT cell.
// =============================================================================

template <typename T>
__global__ void pixel_unshuffle_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N, int C, int H, int W, int r)
{
    int IH = H * r;
    int IW = W * r;
    int COut = C * r * r;
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    int64_t total = (int64_t)N * COut * H * W;
    for (int64_t i = tid; i < total; i += step) {
        int w = (int)(i % W);
        int h = (int)((i / W) % H);
        int cout = (int)((i / ((int64_t)W * H)) % COut);
        int n  = (int)(i / ((int64_t)W * H * COut));
        int c = cout / (r * r);
        int rem = cout - c * r * r;
        int dh = rem / r;
        int dw = rem - dh * r;
        int ih = h * r + dh;
        int iw = w * r + dw;
        int64_t in_idx = (((int64_t)n * C + c) * IH + ih) * (int64_t)IW + iw;
        output[i] = input[in_idx];
    }
}

template <typename T>
__host__ inline int32_t launch_pixel_unshuffle(
    const T* input, T* output,
    int N, int C, int H, int W, int r,
    cudaStream_t stream)
{
    if (N < 0 || C < 0 || H < 0 || W < 0 || r <= 0) return 2;
    int64_t total = (int64_t)N * (C * r * r) * H * W;
    if (total == 0) return 0;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    pixel_unshuffle_kernel<T><<<blocks, kBlock, 0, stream>>>(input, output, N, C, H, W, r);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::image

// =============================================================================
// INSTANTIATE macros.
// =============================================================================

#define BARACUDA_KERNELS_PIXEL_SHUFFLE_INSTANTIATE(NAME, T)                                    \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t H, int32_t W, int32_t r,                                  \
        const void* input,                                                                     \
        void* output,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (input == nullptr || output == nullptr) return 2;                                   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_pixel_shuffle<T>(                                       \
            static_cast<const T*>(input),                                                      \
            static_cast<T*>(output),                                                           \
            N, C, H, W, r, stream);                                                            \
    }

#define BARACUDA_KERNELS_PIXEL_UNSHUFFLE_INSTANTIATE(NAME, T)                                  \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                          \
        int32_t N, int32_t C, int32_t H, int32_t W, int32_t r,                                  \
        const void* input,                                                                     \
        void* output,                                                                          \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                       \
        void* stream_ptr)                                                                      \
    {                                                                                          \
        if (input == nullptr || output == nullptr) return 2;                                   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                           \
        return baracuda::image::launch_pixel_unshuffle<T>(                                     \
            static_cast<const T*>(input),                                                      \
            static_cast<T*>(output),                                                           \
            N, C, H, W, r, stream);                                                            \
    }

#endif // BARACUDA_PIXEL_SHUFFLE_CUH
