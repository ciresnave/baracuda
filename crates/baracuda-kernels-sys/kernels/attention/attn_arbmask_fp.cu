// Phase 51 — arbitrary additive-mask attention forward (FW only).
//
// Mirrors `flash_sdpa_fp.cu` for the Tier-1 dtype set
// {f32, f16, bf16, f64}. Adds an f32 `[B, H, Q, K]` additive bias
// applied to S = QK^T·scale before the row max/softmax. Unlocks
// spec-decode tree masks, MoE expert masking, prefix-LM, and
// sliding-window-with-sinks at the SDPA layer without bespoke per-use
// kernels.

#include "../include/baracuda_attn_arbmask.cuh"

BARACUDA_KERNELS_ATTN_ARBMASK_INSTANTIATE(sdpa_f32,  float)
BARACUDA_KERNELS_ATTN_ARBMASK_INSTANTIATE(sdpa_f16,  __half)
BARACUDA_KERNELS_ATTN_ARBMASK_INSTANTIATE(sdpa_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ATTN_ARBMASK_INSTANTIATE(sdpa_f64,  double)
