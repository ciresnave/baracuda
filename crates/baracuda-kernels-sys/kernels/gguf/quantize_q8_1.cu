// SPDX-FileCopyrightText: 2023-2024 The llama.cpp / ggml authors  (MIT)
// SPDX-FileCopyrightText: 2024-2026 Fuel project contributors      (MIT OR Apache-2.0)
// SPDX-FileCopyrightText: 2026 baracuda project contributors       (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// quantize_q8_1.cu — Phase 33 activation staging kernel.
//
// Per-32-element block: compute `d = amax / 127`, `sum = Σ x_i`, then
// quantize `q_i = round(x_i / d)` and store as `block_q8_1 { half2 ds,
// int8_t qs[32] }`. The output buffer is laid out as
// `[M_rows × (K_padded / 32) × sizeof(block_q8_1) bytes]` — one row of
// staged activations per source row. `K_padded` is the K dimension
// rounded up to a multiple of 32 (the QK8_1 block size); out-of-range
// columns are zero-padded so the dot kernel can read them without
// branching.
//
// Activation input dtypes: f32 (primary), f16 (cast in-kernel),
// bf16 (cast in-kernel). The staged buffer is always `block_q8_1`
// regardless of source dtype.

#include "../include/baracuda_mmvq_multim.cuh"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

using baracuda::mmvq_multim::block_q8_1;
using baracuda::mmvq_multim::QK8_1;

namespace {

inline int32_t status_from_launch(cudaError_t err) {
    if (err != cudaSuccess) {
        return 5;
    }
    return 0;
}

template <typename T> struct load_act;

template <> struct load_act<float> {
    __device__ __forceinline__ static float load(const float * p) { return *p; }
};
template <> struct load_act<__half> {
    __device__ __forceinline__ static float load(const __half * p) { return __half2float(*p); }
};
template <> struct load_act<__nv_bfloat16> {
    __device__ __forceinline__ static float load(const __nv_bfloat16 * p) {
        return __bfloat162float(*p);
    }
};

// Warp-restricted max/sum reductions over a 32-wide block. Each
// activation block (QK8_1 = 32 elements) is contained in a single warp
// because we launch blockDim.x = 32 (= WARP_SIZE) — the cross-thread
// reduction is therefore a single __shfl pass.
__device__ __forceinline__ float warp_reduce_max_q8_1(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}
__device__ __forceinline__ float warp_reduce_sum_q8_1(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

template <typename T>
__global__ void quantize_q8_1_kernel(
    const T * __restrict__ x, void * __restrict__ vy,
    const int kx, const int kx_padded)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    if (ix >= kx_padded) {
        return;
    }
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    const int i_padded = iy * kx_padded + ix;
    block_q8_1 * y = static_cast<block_q8_1 *>(vy);

    const int ib  = i_padded / QK8_1;     // block index
    const int iqs = i_padded % QK8_1;     // quant index inside the block

    const float xi = (ix < kx) ? load_act<T>::load(&x[iy * kx + ix]) : 0.0f;

    float amax = fabsf(xi);
    float sum  = xi;

    amax = warp_reduce_max_q8_1(amax);
    sum  = warp_reduce_sum_q8_1(sum);

    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? int8_t(0) : static_cast<int8_t>(roundf(xi / d));

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }
    // Block header written by lane 0 only.
    reinterpret_cast<__half &>(y[ib].ds.x) = __float2half(d);
    reinterpret_cast<__half &>(y[ib].ds.y) = __float2half(sum);
}

inline int ceil_div_host(int a, int b) { return (a + b - 1) / b; }

template <typename T>
int32_t launch_quantize(
    int64_t kx, int64_t ny,
    const T * x, void * dst,
    cudaStream_t stream)
{
    if (kx <= 0 || ny <= 0) {
        return 2;
    }
    // Round kx up to QK8_1 boundary; out-of-range lanes write 0 quants.
    const int kx_i  = static_cast<int>(kx);
    const int kx_padded = ((kx_i + QK8_1 - 1) / QK8_1) * QK8_1;
    const int ny_i = static_cast<int>(ny);

    // 32 threads per block in X (one QK8_1 block per warp) × (8 rows per block in Y).
    constexpr int BSX = 32;
    constexpr int BSY = 8;
    dim3 block(BSX, BSY, 1);
    dim3 grid(ceil_div_host(kx_padded, BSX), ceil_div_host(ny_i, BSY), 1);

    quantize_q8_1_kernel<T><<<grid, block, 0, stream>>>(
        x, dst, kx_i, kx_padded);
    return status_from_launch(cudaPeekAtLastError());
}

} // anonymous namespace

// ---- FFI launchers --------------------------------------------------------
//
// `kx`  = unpacked column count (= K dimension).
// `ny`  = number of activation rows (= M; >= 1).
// `dst` = output buffer, size >= ny * ceil(kx / QK8_1) * sizeof(block_q8_1).
//
// Status codes: 0 = success, 2 = invalid arg, 5 = launch error.

extern "C" int32_t baracuda_kernels_quantize_q8_1_f32_run(
    int64_t kx, int64_t ny,
    const void * __restrict__ src,
    void * __restrict__ dst_q8_1,
    void * stream_ptr)
{
    if (!src || !dst_q8_1) { return 2; }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_quantize<float>(kx, ny,
        static_cast<const float *>(src), dst_q8_1, stream);
}

extern "C" int32_t baracuda_kernels_quantize_q8_1_f16_run(
    int64_t kx, int64_t ny,
    const void * __restrict__ src,
    void * __restrict__ dst_q8_1,
    void * stream_ptr)
{
    if (!src || !dst_q8_1) { return 2; }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_quantize<__half>(kx, ny,
        static_cast<const __half *>(src), dst_q8_1, stream);
}

extern "C" int32_t baracuda_kernels_quantize_q8_1_bf16_run(
    int64_t kx, int64_t ny,
    const void * __restrict__ src,
    void * __restrict__ dst_q8_1,
    void * stream_ptr)
{
    if (!src || !dst_q8_1) { return 2; }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_quantize<__nv_bfloat16>(kx, ny,
        static_cast<const __nv_bfloat16 *>(src), dst_q8_1, stream);
}

// Helper: returns the number of bytes needed to stage `ny × kx` activations
// into Q8_1 (= ny × ceil(kx/32) × 36 B). Used by the Rust plan for workspace
// sizing.
extern "C" int64_t baracuda_kernels_quantize_q8_1_workspace_bytes(
    int64_t kx, int64_t ny)
{
    if (kx <= 0 || ny <= 0) { return 0; }
    const int64_t kx_padded = ((kx + QK8_1 - 1) / QK8_1) * QK8_1;
    return ny * (kx_padded / QK8_1) * static_cast<int64_t>(sizeof(block_q8_1));
}
