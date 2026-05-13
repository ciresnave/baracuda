// SPDX-License-Identifier: MIT OR Apache-2.0
//
// gemm_bin_rrr_sm89.cu — binary (B1) GEMM with RRR layout, Identity
// epilogue, for sm_89. Computes
//   D = sum_k popcount( A[i, k_byte] XOR B[k_byte, j] )
// where B is gmem-bit-packed along N (RRR layout); the kernel re-
// packs into K-bit-packed smem with a bit-gather load before the MMA
// (see `baracuda_bin_rrr_sm89.cuh`).

#include <cstdint>
#include <cuda_runtime.h>

#include "baracuda_bin_rrr_sm89.cuh"

extern "C" {

int32_t baracuda_kernels_gemm_bin_rrr_sm89_run(
    int32_t m, int32_t n, int32_t k,
    const void *a, int64_t lda_bytes,
    const void *b, int64_t ldb_bytes,
    void *d, int64_t ldd_elements,
    void * /*workspace*/, size_t /*workspace_bytes*/,
    void *stream
) {
    return baracuda::launch_gemm_bin_rrr_sm89(
        m, n, k,
        a, lda_bytes, b, ldb_bytes, d, ldd_elements,
        stream
    );
}

size_t baracuda_kernels_gemm_bin_rrr_sm89_workspace_size(
    int32_t /*m*/, int32_t /*n*/, int32_t /*k*/
) {
    return 0;
}

int32_t baracuda_kernels_gemm_bin_rrr_sm89_can_implement(
    int32_t m, int32_t n, int32_t k,
    const void * /*a*/, int64_t /*lda_bytes*/,
    const void * /*b*/, int64_t /*ldb_bytes*/,
    const void * /*d*/, int64_t /*ldd_elements*/
) {
    if (m <= 0 || n <= 0 || k <= 0) return 2;
    if ((k & 7) != 0) return 3;   // K must be divisible by 8
    if ((n & 7) != 0) return 3;   // N must be divisible by 8 (B is N-bit-packed in gmem)
    return 0;
}

} // extern "C"
