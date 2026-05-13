// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_s4_rcr_sm89.cu — int4 (signed) GEMM with RCR layout, Identity
// epilogue, for sm_89 (Ada Lovelace; first arch with int4 tensor cores
// via `mma.sync.m16n8k64.s4.s4.s32`).
//
// Computes D = sat_cast_s4(alpha * (A @ B) + beta * dequant_s4(C)).
//
// Thin instantiation of `baracuda_int4_rcr_sm89.cuh` with
// `<Int4Encoding::S4, float, Activation::None>`. The kernel body, the
// MMA wrappers, and the saturating-cast machinery all live in the
// shared header so the U4 / RRR / bias-family variants can reuse them
// when the rest of the Phase 2 int4 fanout lands.
//
// This is the Phase 2 int4 trailblazer — first kernel to prove the
// packed-storage path (two int4 per byte, low-nibble = even index,
// high-nibble = odd index) and the m16n8k64.s4 PTX intrinsic against
// the f32 / int32 dequant-style reference.

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_int4_rcr_sm89.cuh"

extern "C" {

// Status codes (shared across all baracuda-kernels-sys entry points):
//   0 = success
//   1 = misaligned operand            (not enforced today)
//   2 = invalid problem               (M, N, or K non-positive)
//   3 = unsupported                   (K odd / N odd / shape not implemented)
//   4 = workspace too small or null   (this SKU has zero workspace)
//   5 = internal kernel error         (kernel launch failure)
int32_t baracuda_kernels_gemm_s4_rcr_sm89_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    const void *c, int64_t ldc_bytes,
    void *d, int64_t ldd_bytes,
    float alpha, float beta,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void *stream
) {
    return baracuda::launch_gemm_int4_rcr_sm89<
        baracuda::Int4Encoding::S4, float, baracuda::Activation::None
    >(
        m, n, k,
        a, lda_bytes, b, ldb_bytes, c, ldc_bytes, d, ldd_bytes,
        /*bias=*/nullptr,
        alpha, beta,
        stream
    );
}

size_t baracuda_kernels_gemm_s4_rcr_sm89_workspace_size(
    int32_t /*m*/, int32_t /*n*/, int32_t /*k*/
) {
    return 0;
}

int32_t baracuda_kernels_gemm_s4_rcr_sm89_can_implement(
    int32_t m, int32_t n, int32_t k,
    const void * /*a*/, int64_t /*lda_bytes*/,
    const void * /*b*/, int64_t /*ldb_bytes*/,
    const void * /*c*/, int64_t /*ldc_bytes*/,
    const void * /*d*/, int64_t /*ldd_bytes*/
) {
    if (m <= 0 || n <= 0 || k <= 0) return 2;
    if ((k & 1) != 0) return 3;   // packing requires K even
    if ((n & 1) != 0) return 3;   // packed output requires N even
    return 0;
}

} // extern "C"
