// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_fp8_e4m3_rcr_sm89.cu — FP8 E4M3 GEMM with RCR layout, Identity
// epilogue, for sm_89 (Ada Lovelace; first arch with FP8 tensor cores).
//
// Computes D = sat_cast_e4m3(alpha * (A @ B) + beta * C).
//
// Thin instantiation of `baracuda_fp8_rcr_sm89.cuh` with
// `<Fp8Encoding::E4M3, float, Activation::None>`. The kernel body, the
// MMA wrappers, and the saturating-cast machinery all live in shared
// headers so the E5M2 SKU, the bias / activation family, and the RRR
// layout can reuse them.
//
// This was the Phase 2 trailblazer; lifted into the shared header when
// the rest of the 20-SKU matrix landed.

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_fp8_rcr_sm89.cuh"

extern "C" {

// Status codes (shared across all baracuda-kernels-sys entry points):
//   0 = success
//   1 = misaligned operand            (not enforced today)
//   2 = invalid problem               (M, N, or K non-positive)
//   3 = unsupported                   (this kernel doesn't implement that combo)
//   4 = workspace too small or null   (this SKU has zero workspace)
//   5 = internal kernel error         (kernel launch failure)
int32_t baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda,
    const void *b, int64_t ldb,
    const void *c, int64_t ldc,
    void *d, int64_t ldd,
    float alpha, float beta,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void *stream
) {
    return baracuda::launch_gemm_fp8_rcr_sm89<
        baracuda::Fp8Encoding::E4M3, float, baracuda::Activation::None
    >(
        m, n, k,
        a, lda, b, ldb, c, ldc, d, ldd,
        /*bias=*/nullptr,
        alpha, beta,
        stream
    );
}

size_t baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_workspace_size(
    int32_t /*m*/, int32_t /*n*/, int32_t /*k*/
) {
    return 0;
}

int32_t baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_can_implement(
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
