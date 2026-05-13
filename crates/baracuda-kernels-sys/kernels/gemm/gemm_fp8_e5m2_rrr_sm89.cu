// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_fp8_e5m2_rrr_sm89.cu — FP8 E5M2 GEMM with RRR layout, Identity
// epilogue, for sm_89.
//
// Thin instantiation of `baracuda_fp8_rrr_sm89.cuh` with
// `<Fp8Encoding::E5M2, float, Activation::None>`.

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_fp8_rrr_sm89.cuh"

extern "C" {

int32_t baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    float alpha, float beta,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void *stream
) {
    return baracuda::launch_gemm_fp8_rrr_sm89<
        baracuda::Fp8Encoding::E5M2, float, baracuda::Activation::None
    >(
        m, n, k,
        a, lda, b, ldb, c, ldc, d, ldd,
        /*bias=*/nullptr,
        alpha, beta,
        stream
    );
}

size_t baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_workspace_size(
    int32_t /*m*/, int32_t /*n*/, int32_t /*k*/
) {
    return 0;
}

int32_t baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_can_implement(
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
