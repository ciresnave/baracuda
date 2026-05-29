// SPDX-FileCopyrightText: 2026 baracuda project contributors  (MIT OR Apache-2.0)
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// nf4_launcher.cu — Phase 53 C-ABI launchers for the vendored NF4
// dequant + GEMV kernels. The kernel templates themselves live under
// `vendor/bitsandbytes/src/nf4_*.cuh`; this TU just instantiates them
// for the supported (T_act, M) pairs and exposes one FFI symbol per
// instantiation.
//
// Status codes (matches the rest of baracuda-kernels-sys):
//   0 = success
//   2 = invalid problem (non-positive dims, bad alignment, ...)
//   5 = internal kernel error (launch failure)
//
// All shapes / strides are caller-validated up in the Rust plan layer;
// this layer only re-checks the pointer-non-null + dims-positive
// invariants the kernels themselves rely on.

#include "../../vendor/bitsandbytes/src/nf4_gemv.cuh"

#include <cstdint>
#include <cuda_runtime.h>

using baracuda::nf4::nf4_dequantize_kernel;
using baracuda::nf4::nf4_gemv_m1_kernel;
using baracuda::nf4::nf4_gemv_multim_kernel;

namespace {

inline int32_t status_from_launch(cudaError_t err) {
    return (err != cudaSuccess) ? 5 : 0;
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// -----------------------------------------------------------------------------
// Dequant launcher template.
// -----------------------------------------------------------------------------
template <typename T>
int32_t launch_nf4_dequant(
    int N, int K, int block_size,
    const void * w_packed, const void * absmax,
    void * out, cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || block_size <= 0)         { return 2; }
    if ((K % block_size) != 0)                       { return 2; }
    if ((N % 2) != 0)                                { return 2; }

    constexpr int BX = 16;
    constexpr int BY = 16;
    dim3 block(BX, BY, 1);
    dim3 grid(ceil_div(N, BX), ceil_div(K, BY), 1);

    nf4_dequantize_kernel<T><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t *>(w_packed),
        static_cast<const float   *>(absmax),
        static_cast<T *>(out),
        N, K, block_size);
    return status_from_launch(cudaPeekAtLastError());
}

// -----------------------------------------------------------------------------
// GEMV M=1 launcher template.
// -----------------------------------------------------------------------------
template <typename T>
int32_t launch_nf4_gemv_m1(
    int N, int K, int block_size,
    const void * w_packed, const void * absmax,
    const void * y, void * out,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || block_size <= 0)         { return 2; }
    if ((K % block_size) != 0)                       { return 2; }
    if ((N % 2) != 0)                                { return 2; }

    dim3 block(32, 1, 1);
    dim3 grid(N, 1, 1);

    nf4_gemv_m1_kernel<T><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t *>(w_packed),
        static_cast<const float   *>(absmax),
        static_cast<const T *>(y),
        static_cast<T *>(out),
        N, K, block_size);
    return status_from_launch(cudaPeekAtLastError());
}

// -----------------------------------------------------------------------------
// GEMV multi-M launcher template (compile-time M ∈ {2, 4, 8}).
// -----------------------------------------------------------------------------
template <typename T, int M>
int32_t launch_nf4_gemv_multim(
    int N, int K, int block_size,
    const void * w_packed, const void * absmax,
    const void * y, void * out,
    cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || block_size <= 0)         { return 2; }
    if ((K % block_size) != 0)                       { return 2; }
    if ((N % 2) != 0)                                { return 2; }

    dim3 block(32, 1, 1);
    dim3 grid(N, 1, 1);

    nf4_gemv_multim_kernel<T, M><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t *>(w_packed),
        static_cast<const float   *>(absmax),
        static_cast<const T *>(y),
        static_cast<T *>(out),
        N, K, block_size);
    return status_from_launch(cudaPeekAtLastError());
}

} // anonymous namespace

// =============================================================================
// FFI surface — NF4 dequant
// =============================================================================
//
// dequantize `[N/2, K]` u8 packed NF4 → `[N, K]` of T.
//
// `w_packed` length must be `(N/2) * K` bytes.
// `absmax`   length must be `N * (K / block_size)` f32.
// `out`      length must be `N * K` of T.
// `block_size` typically 64.

extern "C" int32_t baracuda_kernels_nf4_dequantize_f16_run(
    int32_t N, int32_t K, int32_t block_size,
    const void * w_packed, const void * absmax,
    void * out, void * stream_ptr)
{
    if (!w_packed || !absmax || !out) { return 2; }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_nf4_dequant<__half>(N, K, block_size, w_packed, absmax, out, stream);
}

extern "C" int32_t baracuda_kernels_nf4_dequantize_bf16_run(
    int32_t N, int32_t K, int32_t block_size,
    const void * w_packed, const void * absmax,
    void * out, void * stream_ptr)
{
    if (!w_packed || !absmax || !out) { return 2; }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_nf4_dequant<__nv_bfloat16>(N, K, block_size, w_packed, absmax, out, stream);
}

extern "C" int32_t baracuda_kernels_nf4_dequantize_f32_run(
    int32_t N, int32_t K, int32_t block_size,
    const void * w_packed, const void * absmax,
    void * out, void * stream_ptr)
{
    // f32 output is only used by the smoke test for the
    // quantize→dequant roundtrip; the real inference path is
    // f16/bf16. Exposed for the test layer.
    if (!w_packed || !absmax || !out) { return 2; }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_nf4_dequant<float>(N, K, block_size, w_packed, absmax, out, stream);
}

// =============================================================================
// FFI surface — NF4 GEMV M=1
// =============================================================================
//
// out `[N]` = W_q `[N/2, K]` (NF4) × y `[K]` (T)  with per-block absmax.

extern "C" int32_t baracuda_kernels_nf4_gemv_m1_f16_run(
    int32_t N, int32_t K, int32_t block_size,
    const void * w_packed, const void * absmax,
    const void * y, void * out,
    void * stream_ptr)
{
    if (!w_packed || !absmax || !y || !out) { return 2; }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_nf4_gemv_m1<__half>(N, K, block_size, w_packed, absmax, y, out, stream);
}

extern "C" int32_t baracuda_kernels_nf4_gemv_m1_bf16_run(
    int32_t N, int32_t K, int32_t block_size,
    const void * w_packed, const void * absmax,
    const void * y, void * out,
    void * stream_ptr)
{
    if (!w_packed || !absmax || !y || !out) { return 2; }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
    return launch_nf4_gemv_m1<__nv_bfloat16>(N, K, block_size, w_packed, absmax, y, out, stream);
}

// =============================================================================
// FFI surface — NF4 GEMV multi-M (M = 2 / 4 / 8)
// =============================================================================
//
// out `[M, N]` = W_q `[N/2, K]` (NF4) × y `[M, K]` (T).

#define BARACUDA_NF4_MULTIM_LAUNCHER(T_TAG, T_TY, M_VAL) \
extern "C" int32_t baracuda_kernels_nf4_gemv_m##M_VAL##_##T_TAG##_run( \
    int32_t N, int32_t K, int32_t block_size, \
    const void * w_packed, const void * absmax, \
    const void * y, void * out, \
    void * stream_ptr) \
{ \
    if (!w_packed || !absmax || !y || !out) { return 2; } \
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr); \
    return launch_nf4_gemv_multim<T_TY, M_VAL>( \
        N, K, block_size, w_packed, absmax, y, out, stream); \
}

BARACUDA_NF4_MULTIM_LAUNCHER(f16,  __half,        2)
BARACUDA_NF4_MULTIM_LAUNCHER(f16,  __half,        4)
BARACUDA_NF4_MULTIM_LAUNCHER(f16,  __half,        8)
BARACUDA_NF4_MULTIM_LAUNCHER(bf16, __nv_bfloat16, 2)
BARACUDA_NF4_MULTIM_LAUNCHER(bf16, __nv_bfloat16, 4)
BARACUDA_NF4_MULTIM_LAUNCHER(bf16, __nv_bfloat16, 8)

#undef BARACUDA_NF4_MULTIM_LAUNCHER
