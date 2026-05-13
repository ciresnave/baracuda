// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_u4_rrr_sm89.cu — int4 (unsigned) GEMM with RRR layout, Identity
// epilogue, for sm_89. Thin instantiation of
// `baracuda_int4_rrr_sm89.cuh` with
// `<Int4Encoding::U4, float, Activation::None>`.

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_int4_rrr_sm89.cuh"

extern "C" {

int32_t baracuda_kernels_gemm_u4_rrr_sm89_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    float alpha, float beta,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void *stream
) {
    return baracuda::launch_gemm_int4_rrr_sm89<
        baracuda::Int4Encoding::U4, float, baracuda::Activation::None
    >(
        m, n, k,
        a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        /*bias=*/nullptr,
        alpha, beta,
        stream
    );
}

size_t baracuda_kernels_gemm_u4_rrr_sm89_workspace_size(
    int32_t /*m*/, int32_t /*n*/, int32_t /*k*/
) {
    return 0;
}

int32_t baracuda_kernels_gemm_u4_rrr_sm89_can_implement(
    int32_t m, int32_t n, int32_t k,
    const void * /*a*/, int64_t /*lda_bytes*/,
    const void * /*b*/, int64_t /*ldb_bytes*/,
    const void * /*c*/, int64_t /*ldc_bytes*/,
    const void * /*d*/, int64_t /*ldd_bytes*/
) {
    if (m <= 0 || n <= 0 || k <= 0) return 2;
    if ((k & 1) != 0) return 3;
    if ((n & 1) != 0) return 3;
    return 0;
}

} // extern "C"
