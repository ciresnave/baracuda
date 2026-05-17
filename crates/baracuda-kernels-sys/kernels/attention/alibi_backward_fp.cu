// baracuda-kernels Phase 6 Category K — ALiBi BW for FP types.
//
// Two-kernel launch (when both grads requested):
//   1. da = dy pass-through copy (one thread per cell).
//   2. dslope[h] = Σ_{b, i, j} dy[b, h, i, j] · (j - i) — one block per
//      head, warp-shuffle + smem reduction, deterministic (no atomicAdd).

#include "../include/baracuda_attention.cuh"

BARACUDA_KERNELS_ALIBI_BACKWARD_INSTANTIATE(alibi_backward_f32, float)
BARACUDA_KERNELS_ALIBI_BACKWARD_INSTANTIATE(alibi_backward_f16, __half)
BARACUDA_KERNELS_ALIBI_BACKWARD_INSTANTIATE(alibi_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ALIBI_BACKWARD_INSTANTIATE(alibi_backward_f64, double)
