// baracuda-kernels Phase 6 Category K — Flash Attention SDPA BW (Milestone 6.6).
//
// Three-kernel deterministic pipeline:
//   K1: D = rowsum(y ⊙ dy)                         [B, H, Q]
//   K2: dQ  per (b, h, q_block) — owns its q-rows  [B, H, Q, D_k]
//   K3: dK, dV per (b, h, k_block) — owns its k-cols
// No atomicAdd: each output cell is written by exactly one block.

#include "../include/baracuda_flash_sdpa.cuh"

BARACUDA_KERNELS_FLASH_SDPA_BACKWARD_INSTANTIATE(flash_sdpa_backward_f32, float)
BARACUDA_KERNELS_FLASH_SDPA_BACKWARD_INSTANTIATE(flash_sdpa_backward_f16, __half)
BARACUDA_KERNELS_FLASH_SDPA_BACKWARD_INSTANTIATE(flash_sdpa_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_FLASH_SDPA_BACKWARD_INSTANTIATE(flash_sdpa_backward_f64, double)
