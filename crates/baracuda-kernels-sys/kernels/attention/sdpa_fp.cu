// baracuda-kernels Phase 6 Category K — naive SDPA FW for FP types.
//
// Three-kernel pipeline (scores = Q@K^T*scale, row-softmax, y = attn @ V),
// bundled behind a single `_run` symbol per dtype.

#include "../include/baracuda_sdpa.cuh"

BARACUDA_KERNELS_SDPA_INSTANTIATE(sdpa_f32, float)
BARACUDA_KERNELS_SDPA_INSTANTIATE(sdpa_f16, __half)
BARACUDA_KERNELS_SDPA_INSTANTIATE(sdpa_bf16, __nv_bfloat16)
BARACUDA_KERNELS_SDPA_INSTANTIATE(sdpa_f64, double)
