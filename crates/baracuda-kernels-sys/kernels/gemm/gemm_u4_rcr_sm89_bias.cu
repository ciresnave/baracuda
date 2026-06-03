// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_u4_rcr_sm89_bias.cu — U4 GEMM with RCR layout, bias + activation
// epilogue family, for sm_89.
//
// Sibling of `gemm_s4_rcr_sm89_bias.cu`; only the `Int4Encoding`
// template parameter differs (`U4` instead of `S4`), which switches the
// MMA operand tag to `.u4.u4` and the saturating cast on store to
// clamp `[0, 15]`.

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_int4_rcr_sm89.cuh"

namespace {

template <typename BiasT, baracuda::Activation Act>
int32_t run_u4_rcr(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    const void *bias,
    float alpha, float beta,
    void *stream
) {
    return baracuda::launch_gemm_int4_rcr_sm89<
        baracuda::Int4Encoding::U4, BiasT, Act
    >(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

} // namespace

extern "C" {

// -------- f32 bias --------

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_f32_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_u4_rcr<float, baracuda::Activation::Bias>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_relu_f32_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_u4_rcr<float, baracuda::Activation::BiasRelu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_gelu_f32_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_u4_rcr<float, baracuda::Activation::BiasGelu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_silu_f32_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_u4_rcr<float, baracuda::Activation::BiasSilu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

// -------- i32 bias --------

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_i32_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_u4_rcr<int32_t, baracuda::Activation::Bias>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_relu_i32_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_u4_rcr<int32_t, baracuda::Activation::BiasRelu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_gelu_i32_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_u4_rcr<int32_t, baracuda::Activation::BiasGelu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_silu_i32_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_u4_rcr<int32_t, baracuda::Activation::BiasSilu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}


// ---- _can_implement companions ----

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_f32_can_implement(
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

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_i32_can_implement(
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

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_relu_f32_can_implement(
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

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_relu_i32_can_implement(
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

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_gelu_f32_can_implement(
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

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_gelu_i32_can_implement(
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

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_silu_f32_can_implement(
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

int32_t baracuda_kernels_gemm_u4_rcr_sm89_bias_silu_i32_can_implement(
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
