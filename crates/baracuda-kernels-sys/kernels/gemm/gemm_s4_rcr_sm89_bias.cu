// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_s4_rcr_sm89_bias.cu — S4 GEMM with RCR layout, bias + activation
// epilogue family, for sm_89 (Ada Lovelace; int4 tensor cores via
// `mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32`).
//
// Eight SKUs:
//   {Bias, BiasRelu, BiasGelu, BiasSilu} × {f32 bias, i32 bias}
//
// Each SKU is a thin instantiation of `baracuda_int4_rcr_sm89.cuh`. The
// epilogue chain (bias-add → optional scalar activation → saturating
// cast back to s4) is the only thing that varies between SKUs; the
// kernel body is identical to the Identity case.
//
// Naming convention: `baracuda_kernels_gemm_s4_rcr_sm89_<epilogue>_<bias_t>_run`.
//   epilogue ∈ {bias, bias_relu, bias_gelu, bias_silu}.
//   bias_t   ∈ {f32, i32}.
//
// Identity SKU (`gemm_s4_rcr_sm89.cu`) ships the shared
// `_workspace_size` and `_can_implement` for this `(T, layout)` pair;
// bias SKUs share them and only ship the `_run` entry points.

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_int4_rcr_sm89.cuh"

namespace {

template <typename BiasT, baracuda::Activation Act>
int32_t run_s4_rcr(
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
        baracuda::Int4Encoding::S4, BiasT, Act
    >(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

} // namespace

extern "C" {

// -------- f32 bias --------

int32_t baracuda_kernels_gemm_s4_rcr_sm89_bias_f32_run(
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
    return run_s4_rcr<float, baracuda::Activation::Bias>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s4_rcr_sm89_bias_relu_f32_run(
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
    return run_s4_rcr<float, baracuda::Activation::BiasRelu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s4_rcr_sm89_bias_gelu_f32_run(
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
    return run_s4_rcr<float, baracuda::Activation::BiasGelu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s4_rcr_sm89_bias_silu_f32_run(
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
    return run_s4_rcr<float, baracuda::Activation::BiasSilu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

// -------- i32 bias --------

int32_t baracuda_kernels_gemm_s4_rcr_sm89_bias_i32_run(
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
    return run_s4_rcr<int32_t, baracuda::Activation::Bias>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s4_rcr_sm89_bias_relu_i32_run(
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
    return run_s4_rcr<int32_t, baracuda::Activation::BiasRelu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s4_rcr_sm89_bias_gelu_i32_run(
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
    return run_s4_rcr<int32_t, baracuda::Activation::BiasGelu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

int32_t baracuda_kernels_gemm_s4_rcr_sm89_bias_silu_i32_run(
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
    return run_s4_rcr<int32_t, baracuda::Activation::BiasSilu>(
        m, n, k, a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        bias, alpha, beta, stream
    );
}

} // extern "C"
