// SPDX-License-Identifier: MIT OR Apache-2.0
//
// baracuda-kernels Phase 10 Milestone 10.3 — Flash Attention SDPA FW,
// sm_89 (Ada Lovelace) specialization. Sibling of `flash_sdpa_fp.cu` —
// same math, but with `cp.async`-based double-buffered K/V loads and a
// wider (256-thread) block to exploit Ada's larger per-SM register file.
//
// Dtype scope: f16 + bf16 only. f32 / f64 stay on the sm_80 baseline.

#include "../include/baracuda_flash_sdpa_sm89.cuh"

BARACUDA_KERNELS_FLASH_SDPA_SM89_INSTANTIATE(flash_sdpa_sm89_f16,  __half)
BARACUDA_KERNELS_FLASH_SDPA_SM89_INSTANTIATE(flash_sdpa_sm89_bf16, __nv_bfloat16)

// Phase 17.1 — strided FW siblings (transposed / GQA-broadcast inputs).
BARACUDA_KERNELS_FLASH_SDPA_SM89_STRIDED_INSTANTIATE(flash_sdpa_sm89_f16,  __half)
BARACUDA_KERNELS_FLASH_SDPA_SM89_STRIDED_INSTANTIATE(flash_sdpa_sm89_bf16, __nv_bfloat16)
