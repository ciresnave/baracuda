// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_fp8_e4m3_rrr_sm89_bias.cu — FP8 E4M3 GEMM with RRR layout, bias +
// activation epilogue family, for sm_89.
//
// Four SKUs: {Bias, BiasRelu, BiasGelu, BiasSilu} × { f32 bias only }.

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_fp8_rrr_sm89.cuh"

namespace {

template <baracuda::Activation Act>
int32_t run_fp8_e4m3_rrr(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    const void *bias,
    float alpha, float beta,
    void *stream
) {
    return baracuda::launch_gemm_fp8_rrr_sm89<
        baracuda::Fp8Encoding::E4M3, float, Act
    >(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
        bias, alpha, beta, stream
    );
}

} // namespace

extern "C" {

int32_t baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_fp8_e4m3_rrr<baracuda::Activation::Bias>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_relu_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_fp8_e4m3_rrr<baracuda::Activation::BiasRelu>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_gelu_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_fp8_e4m3_rrr<baracuda::Activation::BiasGelu>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_silu_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    const void *bias,
    float alpha, float beta,
    void * /*ws*/, size_t /*ws_bytes*/,
    void *stream
) {
    return run_fp8_e4m3_rrr<baracuda::Activation::BiasSilu>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

} // extern "C"
