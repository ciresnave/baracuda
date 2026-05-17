// baracuda-kernels Phase 6 Category K — naive SDPA BW for FP types.
//
// Five-kernel pipeline (dV, dattn, dscores=softmax_bw, dQ, dK), bundled
// behind a single `_run` symbol per dtype. Caller supplies the
// `dscores_ws` workspace ([B, H, Q, K]) which is reused as dattn → dscores
// in-place.

#include "../include/baracuda_sdpa.cuh"

BARACUDA_KERNELS_SDPA_BACKWARD_INSTANTIATE(sdpa_backward_f32, float)
BARACUDA_KERNELS_SDPA_BACKWARD_INSTANTIATE(sdpa_backward_f16, __half)
BARACUDA_KERNELS_SDPA_BACKWARD_INSTANTIATE(sdpa_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_SDPA_BACKWARD_INSTANTIATE(sdpa_backward_f64, double)
