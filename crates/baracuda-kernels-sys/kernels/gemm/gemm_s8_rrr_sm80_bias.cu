// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_s8_rrr_sm80_bias.cu — S8 GEMM with RRR layout, bias + activation
// epilogue family, for sm_80.
//
// Eight SKUs:
//   {Bias, BiasRelu, BiasGelu, BiasSilu} × {f32 bias, i32 bias}
//
// Each SKU is a thin instantiation of the shared templated kernel in
// `baracuda_int8_rrr_sm80.cuh`. The epilogue chain (bias-add →
// optional scalar activation → saturating cast back to s8) is the only
// thing that varies between SKUs; the kernel body is identical to the
// Identity case.
//
// Naming convention: `baracuda_kernels_gemm_s8_rrr_sm80_<epilogue>_<bias_t>_run`.
//   epilogue ∈ {bias, bias_relu, bias_gelu, bias_silu}.
//   bias_t   ∈ {f32, i32}.

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_int8_rrr_sm80.cuh"

namespace {

template <typename BiasT, baracuda::Activation Act>
int32_t run_s8(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    const void *bias,
    float alpha, float beta,
    void *stream
) {
    return baracuda::launch_gemm_int8_rrr_sm80<int8_t, int8_t, BiasT, Act>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

} // namespace

extern "C" {

// -------- f32 bias --------

int32_t baracuda_kernels_gemm_s8_rrr_sm80_bias_f32_run(
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
    return run_s8<float, baracuda::Activation::Bias>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s8_rrr_sm80_bias_relu_f32_run(
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
    return run_s8<float, baracuda::Activation::BiasRelu>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s8_rrr_sm80_bias_gelu_f32_run(
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
    return run_s8<float, baracuda::Activation::BiasGelu>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s8_rrr_sm80_bias_silu_f32_run(
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
    return run_s8<float, baracuda::Activation::BiasSilu>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

// -------- i32 bias --------

int32_t baracuda_kernels_gemm_s8_rrr_sm80_bias_i32_run(
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
    return run_s8<int32_t, baracuda::Activation::Bias>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s8_rrr_sm80_bias_relu_i32_run(
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
    return run_s8<int32_t, baracuda::Activation::BiasRelu>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s8_rrr_sm80_bias_gelu_i32_run(
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
    return run_s8<int32_t, baracuda::Activation::BiasGelu>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s8_rrr_sm80_bias_silu_i32_run(
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
    return run_s8<int32_t, baracuda::Activation::BiasSilu>(
        m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias, alpha, beta, stream
    );
}

} // extern "C"
