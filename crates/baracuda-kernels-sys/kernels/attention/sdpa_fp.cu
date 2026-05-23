// baracuda-kernels Phase 6 Category K — naive SDPA FW for FP types.
//
// Three-kernel pipeline (scores = Q@K^T*scale, row-softmax, y = attn @ V),
// bundled behind a single `_run` symbol per dtype.

#include "../include/baracuda_sdpa.cuh"

BARACUDA_KERNELS_SDPA_INSTANTIATE(sdpa_f32, float)
BARACUDA_KERNELS_SDPA_INSTANTIATE(sdpa_f16, __half)
BARACUDA_KERNELS_SDPA_INSTANTIATE(sdpa_bf16, __nv_bfloat16)
BARACUDA_KERNELS_SDPA_INSTANTIATE(sdpa_f64, double)

// Phase 14.4 strided siblings — outer-dim strides on Q/K/V/y. The
// innermost head_dim axis must be stride=1 (enforced by Rust plan).
// GQA broadcast is supported by passing zero stride on stride_k_h /
// stride_v_h.
BARACUDA_KERNELS_SDPA_STRIDED_INSTANTIATE(sdpa_f32, float)
BARACUDA_KERNELS_SDPA_STRIDED_INSTANTIATE(sdpa_f16, __half)
BARACUDA_KERNELS_SDPA_STRIDED_INSTANTIATE(sdpa_bf16, __nv_bfloat16)
BARACUDA_KERNELS_SDPA_STRIDED_INSTANTIATE(sdpa_f64, double)
