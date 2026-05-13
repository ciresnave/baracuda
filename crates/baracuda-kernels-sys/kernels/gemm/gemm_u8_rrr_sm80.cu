// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_u8_rrr_sm80.cu — U8 GEMM with RRR layout, Identity epilogue,
// for sm_80 (Ampere; forward-compatible on Ada / Hopper).
//
// Computes D = saturating_cast<u8>(alpha * (A @ B) + beta * C) with
// `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.u8.u8.s32`.
//
// Sibling of `gemm_s8_rrr_sm80.cu`; only the InT / OutT template
// arguments and the underlying MMA-operand encoding differ.

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_int8_rrr_sm80.cuh"

extern "C" {

int32_t baracuda_kernels_gemm_u8_rrr_sm80_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    float alpha, float beta,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void *stream
) {
    return baracuda::launch_gemm_int8_rrr_sm80<
        uint8_t, uint8_t, float, baracuda::Activation::None
    >(
        m, n, k,
        a, lda, b, ldb, c, ldc, d, ldd,
        /*bias=*/nullptr,
        alpha, beta,
        stream
    );
}

size_t baracuda_kernels_gemm_u8_rrr_sm80_workspace_size(
    int32_t /*m*/, int32_t /*n*/, int32_t /*k*/
) {
    return 0;
}

int32_t baracuda_kernels_gemm_u8_rrr_sm80_can_implement(
    int32_t m, int32_t n, int32_t k,
    const void * /*a*/, int64_t /*lda*/,
    const void * /*b*/, int64_t /*ldb*/,
    const void * /*c*/, int64_t /*ldc*/,
    const void * /*d*/, int64_t /*ldd*/
) {
    if (m <= 0 || n <= 0 || k <= 0) return 2;
    return 0;
}

} // extern "C"
