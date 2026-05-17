// baracuda-kernels Phase 6 Category K — ALiBi FW for FP types.
//
// Adds per-(head, q, k) linear bias to attention scores:
//   y[b, h, i, j] = A[b, h, i, j] + slope[h] · (j - i)

#include "../include/baracuda_attention.cuh"

BARACUDA_KERNELS_ALIBI_INSTANTIATE(alibi_f32, float)
BARACUDA_KERNELS_ALIBI_INSTANTIATE(alibi_f16, __half)
BARACUDA_KERNELS_ALIBI_INSTANTIATE(alibi_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ALIBI_INSTANTIATE(alibi_f64, double)
