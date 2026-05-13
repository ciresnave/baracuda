// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_s8_rrr_sm80.cu — S8 GEMM with RRR layout, Identity epilogue,
// for sm_80 (Ampere; forward-compatible on Ada / Hopper).
//
// Computes D = saturating_cast<s8>(alpha * (A @ B) + beta * C).
//
// This file is a thin instantiation of the shared templated kernel in
// `baracuda_int8_rrr_sm80.cuh` (Act = None, no bias) plus the
// extern "C" launcher and host-side helpers. The kernel body itself,
// the mma.sync wrappers, and the saturating-cast machinery live in the
// shared headers so the U8 / bias variants can reuse them.

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_int8_rrr_sm80.cuh"

extern "C" {

// Status codes (shared across all baracuda-kernels-sys entry points):
//   0 = success
//   1 = misaligned operand            (not enforced today)
//   2 = invalid problem               (M, N, or K non-positive)
//   3 = unsupported                   (this kernel doesn't implement that combo)
//   4 = workspace too small or null   (this SKU has zero workspace)
//   5 = internal kernel error         (kernel launch failure)
int32_t baracuda_kernels_gemm_s8_rrr_sm80_run(
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
        int8_t, int8_t, float, baracuda::Activation::None
    >(
        m, n, k,
        a, lda, b, ldb, c, ldc, d, ldd,
        /*bias=*/nullptr,
        alpha, beta,
        stream
    );
}

size_t baracuda_kernels_gemm_s8_rrr_sm80_workspace_size(
    int32_t /*m*/, int32_t /*n*/, int32_t /*k*/
) {
    return 0;
}

int32_t baracuda_kernels_gemm_s8_rrr_sm80_can_implement(
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
