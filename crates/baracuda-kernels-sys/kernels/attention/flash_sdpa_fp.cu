// baracuda-kernels Phase 6 Category K — Flash Attention SDPA FW (Milestone 6.6).
//
// Tiled, fused online-softmax kernel that avoids materializing the
// `[B, H, Q, K]` attention matrix. Algorithm from Tri Dao 2022
// (https://arxiv.org/abs/2205.14135). Trailblazer constraints:
// Br = Bc = 64, d_k = d_v ≤ 128. One CUDA block per (batch, head,
// q_block). Saved `lse` (log-sum-exp, `[B, H, Q]`) feeds the BW pass.

#include "../include/baracuda_flash_sdpa.cuh"

BARACUDA_KERNELS_FLASH_SDPA_INSTANTIATE(flash_sdpa_f32, float)
BARACUDA_KERNELS_FLASH_SDPA_INSTANTIATE(flash_sdpa_f16, __half)
BARACUDA_KERNELS_FLASH_SDPA_INSTANTIATE(flash_sdpa_bf16, __nv_bfloat16)
BARACUDA_KERNELS_FLASH_SDPA_INSTANTIATE(flash_sdpa_f64, double)
