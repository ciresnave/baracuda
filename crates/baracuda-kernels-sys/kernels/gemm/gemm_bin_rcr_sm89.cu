// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_bin_rcr_sm89.cu — binary (B1) GEMM with RCR layout, Identity
// epilogue, for sm_89 (works on sm_80+; gated to sm_89 here for
// consistency with the FP8 / int4 build set).
//
// Computes D = sum_k popcount( A[i, k_byte] XOR B[k_byte, j] ) as int32,
// using `mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc`.
// No α / β / bias / activation chain.

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_bin_rcr_sm89.cuh"

extern "C" {

// Status codes (shared across all baracuda-kernels-sys entry points):
//   0 = success
//   1 = misaligned operand            (not enforced today)
//   2 = invalid problem               (M, N, or K non-positive)
//   3 = unsupported                   (K not divisible by 8)
//   4 = workspace too small or null   (this SKU has zero workspace)
//   5 = internal kernel error         (kernel launch failure)
int32_t baracuda_kernels_gemm_bin_rcr_sm89_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    void *d, int64_t ldd_elements,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void *stream
) {
    return baracuda::launch_gemm_bin_rcr_sm89(
        m, n, k,
        a, lda_bytes, b, ldb_bytes, d, ldd_elements,
        stream
    );
}

size_t baracuda_kernels_gemm_bin_rcr_sm89_workspace_size(
    int32_t /*m*/, int32_t /*n*/, int32_t /*k*/
) {
    return 0;
}

int32_t baracuda_kernels_gemm_bin_rcr_sm89_can_implement(
    int32_t m, int32_t n, int32_t k,
    const void * /*a*/, int64_t /*lda_bytes*/,
    const void * /*b*/, int64_t /*ldb_bytes*/,
    const void * /*d*/, int64_t /*ldd_elements*/
) {
    if (m <= 0 || n <= 0 || k <= 0) return 2;
    if ((k & 7) != 0) return 3;   // K must be divisible by 8
    return 0;
}

} // extern "C"
