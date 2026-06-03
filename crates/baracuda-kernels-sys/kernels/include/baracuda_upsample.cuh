// baracuda_upsample.cuh
//
// Templated kernels + INSTANTIATE macros for the `upsample` FFI family
// (Phase 19.2 — design-correction-driven FFI facade). Provides the
// nearest-2D mode (FW + BW) across {f32, f64, f16, bf16} as standalone
// `extern "C"` `_run` symbols, so downstream callers get the same
// "single unified CUDA facade" promise that bespoke kernels already
// satisfy.
//
// Bilinear-2D is intentionally NOT re-instantiated here — the existing
// `interpolate_bilinear_2d_*` symbols from `baracuda_interpolate.cuh`
// remain authoritative; lib.rs exposes thin `upsample_bilinear_2d_*`
// aliases that re-export them.
//
// NCHW layout. `align_corners=false` semantics (PyTorch default for
// new code). Coordinate mapping mirrors `baracuda_interpolate.cuh`:
//   src = (dst + 0.5) * (src_size / dst_size) - 0.5
// Nearest-mode picks `floor(src + 0.5)` with edge clamp on OOB. This
// matches PyTorch's `F.interpolate(mode='nearest')` behavior, which
// effectively computes `floor(dst * src/dst)` for `align_corners=false`
// (the additive `0.5 - 0.5` shifts cancel symbolically for nearest
// rounding when `src/dst` is rational — we use the explicit form for
// numerical-stability parity with the bilinear path).
//
// Status codes (mirror the rest of the bespoke family):
//   0 success
//   1 misaligned operand (reserved)
//   2 invalid problem
//   3 unsupported (reserved)
//   4 workspace too small (reserved)
//   5 internal kernel error (launch failure)

#ifndef BARACUDA_UPSAMPLE_CUH
#define BARACUDA_UPSAMPLE_CUH

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_indexing.cuh"  // scatter_atomic_add<T> (f16/bf16 via atomicCAS)

namespace baracuda { namespace upsample {

// PyTorch nearest mapping for `align_corners=false`:
//   src_idx = min(floor(dst * src_size / dst_size), src_size - 1)
// Computed with a stable integer division (no float drift) when
// (dst, src_size, dst_size) are small i32s, which is always the case
// for NCHW image sizes.
__host__ __device__ inline int nearest_src_idx(
    int dst, int dst_size, int src_size)
{
    // Equivalent to `floor(dst * (src_size / dst_size))` for
    // non-negative inputs. Uses i64 product to avoid i32 overflow at
    // image sizes ≥ ~46k (sqrt(2^31)).
    int64_t prod = (int64_t)dst * (int64_t)src_size;
    int idx = (int)(prod / (int64_t)dst_size);
    if (idx < 0) idx = 0;
    if (idx >= src_size) idx = src_size - 1;
    return idx;
}

// =============================================================================
// upsample_nearest_2d forward — one thread per (n, c, oh, ow) output cell.
// =============================================================================

template <typename T>
__global__ void upsample_nearest_2d_kernel(
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

        int iy = nearest_src_idx(oh, OH, IH);
        int ix = nearest_src_idx(ow, OW, IW);
        const T* in_nc = input + ((int64_t)n * C + c) * (int64_t)IH * IW;
        output[i] = in_nc[(int64_t)iy * IW + ix];
    }
}

template <typename T>
__host__ inline int32_t launch_upsample_nearest_2d(
    const T* input, T* output,
    int N, int C, int IH, int IW, int OH, int OW,
    cudaStream_t stream)
{
    if (N < 0 || C < 0 || IH < 0 || IW < 0 || OH < 0 || OW < 0) return 2;
    // Nearest sampling requires a non-empty source (otherwise there's
    // no valid src cell to clamp to). Empty output is a no-op.
    int64_t total = (int64_t)N * C * OH * OW;
    if (total == 0) return 0;
    if (IH == 0 || IW == 0) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    upsample_nearest_2d_kernel<T><<<blocks, kBlock, 0, stream>>>(
        input, output, N, C, IH, IW, OH, OW);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

// =============================================================================
// upsample_nearest_2d backward — one thread per (n, c, oh, ow) output cell.
// Scatter-adds `dout[i]` into the single input cell that this output
// sampled from (atomicAdd handles the many-to-one collapse when the
// upsample factor is > 1). Caller pre-zeros `dinput`.
// =============================================================================

template <typename T>
__global__ void upsample_nearest_2d_backward_kernel(
    const T* __restrict__ dout,     // [N, C, OH, OW]
    T* __restrict__ dinput,         // [N, C, IH, IW]
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

        int iy = nearest_src_idx(oh, OH, IH);
        int ix = nearest_src_idx(ow, OW, IW);
        T* di_nc = dinput + ((int64_t)n * C + c) * (int64_t)IH * IW;
        baracuda::indexing::scatter_atomic_add<T>(
            &di_nc[(int64_t)iy * IW + ix], dout[i]);
    }
}

template <typename T>
__host__ inline int32_t launch_upsample_nearest_2d_backward(
    const T* dout, T* dinput,
    int N, int C, int IH, int IW, int OH, int OW,
    cudaStream_t stream)
{
    if (N < 0 || C < 0 || IH < 0 || IW < 0 || OH < 0 || OW < 0) return 2;
    int64_t total = (int64_t)N * C * OH * OW;
    if (total == 0) return 0;
    if (IH == 0 || IW == 0) return 2;
    constexpr int kBlock = 256;
    constexpr int64_t kMaxBlocks = 65535;
    int64_t blocks_i64 = (total + kBlock - 1) / kBlock;
    int blocks = static_cast<int>(blocks_i64 > kMaxBlocks ? kMaxBlocks : blocks_i64);
    if (blocks <= 0) blocks = 1;
    upsample_nearest_2d_backward_kernel<T><<<blocks, kBlock, 0, stream>>>(
        dout, dinput, N, C, IH, IW, OH, OW);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : 5;
}

}} // namespace baracuda::upsample

// =============================================================================
// INSTANTIATE macros — `extern "C" int32_t baracuda_kernels_<NAME>_run(...)`.
// FW signature matches `baracuda_kernels_interpolate_bilinear_2d_*_run`
// (NCHW shape int32s, input/output void*, workspace pair, stream).
// =============================================================================

#define BARACUDA_KERNELS_UPSAMPLE_NEAREST_2D_FW_INSTANTIATE(NAME, T)                            \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                            \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                    \
        const void* input,                                                                       \
        void* output,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                         \
        void* stream_ptr)                                                                        \
    {                                                                                            \
        if (input == nullptr || output == nullptr) return 2;                                     \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                             \
        return baracuda::upsample::launch_upsample_nearest_2d<T>(                                \
            static_cast<const T*>(input),                                                        \
            static_cast<T*>(output),                                                             \
            N, C, IH, IW, OH, OW, stream);                                                       \
    }                                                                                            \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                  \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                    \
        const void* /*input*/,                                                                   \
        const void* /*output*/)                                                                  \
    {                                                                                            \
        if (N < 0 || C < 0 || IH < 0 || IW < 0 || OH < 0 || OW < 0) return 2;                    \
        return 0;                                                                                \
    }

#define BARACUDA_KERNELS_UPSAMPLE_NEAREST_2D_BW_INSTANTIATE(NAME, T)                            \
    extern "C" int32_t baracuda_kernels_##NAME##_run(                                            \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                    \
        const void* dout,                                                                        \
        void* dinput,                                                                            \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                         \
        void* stream_ptr)                                                                        \
    {                                                                                            \
        if (dout == nullptr || dinput == nullptr) return 2;                                      \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                             \
        return baracuda::upsample::launch_upsample_nearest_2d_backward<T>(                       \
            static_cast<const T*>(dout),                                                         \
            static_cast<T*>(dinput),                                                             \
            N, C, IH, IW, OH, OW, stream);                                                       \
    }                                                                                            \
    extern "C" int32_t baracuda_kernels_##NAME##_can_implement(                                  \
        int32_t N, int32_t C, int32_t IH, int32_t IW, int32_t OH, int32_t OW,                    \
        const void* /*dout*/,                                                                    \
        const void* /*dinput*/)                                                                  \
    {                                                                                            \
        if (N < 0 || C < 0 || IH < 0 || IW < 0 || OH < 0 || OW < 0) return 2;                    \
        return 0;                                                                                \
    }

#endif // BARACUDA_UPSAMPLE_CUH
