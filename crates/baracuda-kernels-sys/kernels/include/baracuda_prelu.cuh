// baracuda_prelu.cuh
//
// Milestone 5.3 — PReLU FW + BW kernels.
//
// FW: y[..., c, ...] = x[..., c, ...] if x > 0 else weight[c] * x[..., c, ...].
//     weight is either per-channel (shape [C]) or a single scalar (shape [1]).
// BW: dx[..., c, ...] = dy if x > 0 else weight[c] * dy.
//     dweight[c] = Σ over non-channel cells of (dy · x) where x < 0.
//
// `dweight` reduction uses the deterministic warp-shuffle pattern from
// LayerNorm BW affine kernel — no atomicAdd.
//
// All kernels assume contiguous tensors of rank ≤ 8.

#ifndef BARACUDA_PRELU_CUH
#define BARACUDA_PRELU_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "baracuda_loss.cuh"  // for load_as_acc / store_from_acc helpers

namespace baracuda { namespace prelu {

using baracuda::loss::load_as_acc;
using baracuda::loss::store_from_acc;

constexpr int kPReluBlock = 256;

// channel_axis == -1 means scalar weight (single α). channel_extent is then 1.
// Strides are implicit contiguous row-major; we encode shape + axis as args.

// FW kernel: per-cell. We compute the channel index from the flat index by
// the row-major stride decomposition. Pass shape[] and channel_stride.
template <typename T>
__global__ void prelu_fw_kernel(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int64_t numel,
    int64_t channel_stride,  // = product of shape dims AFTER channel axis
    int32_t channel_extent,  // = shape[channel_axis], or 1 for scalar
    int32_t scalar_weight)   // 0 = per-channel, 1 = scalar
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float xv = load_as_acc<T>(x[i]);
        int c = 0;
        if (scalar_weight == 0) {
            c = (int)((i / channel_stride) % (int64_t)channel_extent);
        }
        float wv = load_as_acc<T>(weight[c]);
        float v = (xv > 0.0f) ? xv : (wv * xv);
        y[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void prelu_fw_kernel<double>(
    const double* __restrict__ x,
    const double* __restrict__ weight,
    double* __restrict__ y,
    int64_t numel,
    int64_t channel_stride,
    int32_t channel_extent,
    int32_t scalar_weight)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        double xv = x[i];
        int c = 0;
        if (scalar_weight == 0) {
            c = (int)((i / channel_stride) % (int64_t)channel_extent);
        }
        double wv = weight[c];
        double v = (xv > 0.0) ? xv : (wv * xv);
        y[i] = v;
    }
}

// dx kernel: per-cell.
template <typename T>
__global__ void prelu_dx_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ dx,
    int64_t numel,
    int64_t channel_stride,
    int32_t channel_extent,
    int32_t scalar_weight)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        float xv = load_as_acc<T>(x[i]);
        float dyv = load_as_acc<T>(dy[i]);
        int c = 0;
        if (scalar_weight == 0) {
            c = (int)((i / channel_stride) % (int64_t)channel_extent);
        }
        float wv = load_as_acc<T>(weight[c]);
        float v = (xv > 0.0f) ? dyv : (wv * dyv);
        dx[i] = store_from_acc<T>(v);
    }
}

template <>
__global__ void prelu_dx_kernel<double>(
    const double* __restrict__ dy,
    const double* __restrict__ x,
    const double* __restrict__ weight,
    double* __restrict__ dx,
    int64_t numel,
    int64_t channel_stride,
    int32_t channel_extent,
    int32_t scalar_weight)
{
    int64_t tid  = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t step = (int64_t)gridDim.x  * (int64_t)blockDim.x;
    for (int64_t i = tid; i < numel; i += step) {
        double xv = x[i];
        double dyv = dy[i];
        int c = 0;
        if (scalar_weight == 0) {
            c = (int)((i / channel_stride) % (int64_t)channel_extent);
        }
        double wv = weight[c];
        double v = (xv > 0.0) ? dyv : (wv * dyv);
        dx[i] = v;
    }
}

// dweight kernel: one block per channel index. Threads stride over the
// outer-grid cells assigned to that channel, accumulate dy·x where x<0, then
// reduce via warp-shuffle + smem (deterministic single-block).
//
// For per-channel weight: block index = c. Each thread iterates all (outer,
// inner) cells where the cell's channel-axis coordinate == c.
// outer_extent = numel / channel_extent / inner_extent? Actually we
// re-derive: for each flat index i, channel index = (i / channel_stride) % channel_extent.
// We iterate by flat index in stride of (blockDim.x) and test the channel.
// For correctness on small problems this is sufficient (smoke-test scale).
//
// For scalar weight: single block; reduce over all cells where x<0.

template <typename T>
__global__ void prelu_dweight_kernel(
    const T* __restrict__ dy,
    const T* __restrict__ x,
    T* __restrict__ dweight,
    int64_t numel,
    int64_t channel_stride,
    int32_t channel_extent,
    int32_t scalar_weight)
{
    int target_c = scalar_weight ? 0 : (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsize = (int)blockDim.x;
    float partial = 0.0f;
    for (int64_t i = (int64_t)tid; i < numel; i += (int64_t)bsize) {
        float xv = load_as_acc<T>(x[i]);
        if (xv >= 0.0f) continue;
        int c = 0;
        if (scalar_weight == 0) {
            c = (int)((i / channel_stride) % (int64_t)channel_extent);
            if (c != target_c) continue;
        }
        float dyv = load_as_acc<T>(dy[i]);
        partial += dyv * xv;
    }
    __shared__ float smem[32];
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) smem[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        int n_warps = (bsize + 31) >> 5;
        float v = (lane < n_warps) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane == 0) dweight[target_c] = store_from_acc<T>(v);
    }
}

template <>
__global__ void prelu_dweight_kernel<double>(
    const double* __restrict__ dy,
    const double* __restrict__ x,
    double* __restrict__ dweight,
    int64_t numel,
    int64_t channel_stride,
    int32_t channel_extent,
    int32_t scalar_weight)
{
    int target_c = scalar_weight ? 0 : (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int bsize = (int)blockDim.x;
    double partial = 0.0;
    for (int64_t i = (int64_t)tid; i < numel; i += (int64_t)bsize) {
        double xv = x[i];
        if (xv >= 0.0) continue;
        int c = 0;
        if (scalar_weight == 0) {
            c = (int)((i / channel_stride) % (int64_t)channel_extent);
            if (c != target_c) continue;
        }
        partial += dy[i] * xv;
    }
    __shared__ double smem[32];
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(0xffffffff, partial, offset);
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) smem[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        int n_warps = (bsize + 31) >> 5;
        double v = (lane < n_warps) ? smem[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane == 0) dweight[target_c] = v;
    }
}

} } // namespace baracuda::prelu

// INSTANTIATE macros.

// PReLU FW.
#define BARACUDA_KERNELS_PRELU_FW_INSTANTIATE(NAME, T)                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int64_t channel_stride,                                                                     \
        int32_t channel_extent,                                                                     \
        int32_t scalar_weight,                                                                      \
        const void* x, const void* weight, void* y,                                                 \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                          \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (x == nullptr || weight == nullptr || y == nullptr) return 2;                            \
        if (channel_extent < 1) return 2;                                                          \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        baracuda::prelu::prelu_fw_kernel<T><<<blocks, kBlock, 0, stream>>>(                         \
            static_cast<const T*>(x), static_cast<const T*>(weight), static_cast<T*>(y),            \
            numel, channel_stride, channel_extent, scalar_weight);                                  \
        return (cudaGetLastError() == cudaSuccess) ? 0 : 5;                                         \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel,                                                                              \
        int64_t /*channel_stride*/,                                                                 \
        int32_t channel_extent,                                                                     \
        int32_t /*scalar_weight*/,                                                                  \
        const void* /*x*/, const void* /*weight*/, const void* /*y*/)                               \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (channel_extent < 1) return 2;                                                          \
        return 0;                                                                                   \
    }

// PReLU BW. Launches dx kernel + dweight kernel. dweight is one block per
// channel (or 1 block total for scalar weight). dweight may be null to skip.
#define BARACUDA_KERNELS_PRELU_BW_INSTANTIATE(NAME, T)                                              \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                              \
        int64_t numel,                                                                              \
        int64_t channel_stride,                                                                     \
        int32_t channel_extent,                                                                     \
        int32_t scalar_weight,                                                                      \
        const void* dy, const void* x, const void* weight,                                          \
        void* dx, void* dweight,                                                                    \
        void* /*workspace*/, size_t /*workspace_bytes*/, void* stream_ptr)                          \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (numel == 0) return 0;                                                                   \
        if (dy == nullptr || x == nullptr || weight == nullptr) return 2;                           \
        if (channel_extent < 1) return 2;                                                          \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                \
        constexpr int kBlock = 256;                                                                 \
        int64_t blocks_i64 = (numel + kBlock - 1) / kBlock;                                         \
        int blocks = static_cast<int>(blocks_i64 > 65535 ? 65535 : blocks_i64);                     \
        if (blocks <= 0) blocks = 1;                                                                \
        if (dx != nullptr) {                                                                        \
            baracuda::prelu::prelu_dx_kernel<T><<<blocks, kBlock, 0, stream>>>(                     \
                static_cast<const T*>(dy), static_cast<const T*>(x),                                \
                static_cast<const T*>(weight), static_cast<T*>(dx),                                 \
                numel, channel_stride, channel_extent, scalar_weight);                              \
            if (cudaGetLastError() != cudaSuccess) return 5;                                        \
        }                                                                                           \
        if (dweight != nullptr) {                                                                   \
            int n_blocks = scalar_weight ? 1 : channel_extent;                                      \
            baracuda::prelu::prelu_dweight_kernel<T><<<n_blocks, baracuda::prelu::kPReluBlock, 0,   \
                stream>>>(                                                                          \
                static_cast<const T*>(dy), static_cast<const T*>(x),                                \
                static_cast<T*>(dweight),                                                           \
                numel, channel_stride, channel_extent, scalar_weight);                              \
            if (cudaGetLastError() != cudaSuccess) return 5;                                        \
        }                                                                                           \
        return 0;                                                                                   \
    }                                                                                               \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                    \
        int64_t numel,                                                                              \
        int64_t channel_stride,                                                                     \
        int32_t channel_extent,                                                                     \
        int32_t scalar_weight,                                                                      \
        const void* /*dy*/, const void* /*x*/, const void* /*weight*/,                              \
        const void* /*dx*/, const void* /*dweight*/)                                                \
    {                                                                                               \
        if (numel < 0) return 2;                                                                   \
        if (channel_extent < 1) return 2;                                                          \
        (void)channel_stride;                                                                       \
        (void)scalar_weight;                                                                        \
        return 0;                                                                                   \
    }

#endif // BARACUDA_PRELU_CUH
