// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// FlashDecoding instantiations — Phase 73 follow-up.
//
// Split-K parallel attention decode for seq_q = 1. See header for the
// algorithm narrative.

#include "../include/baracuda_flash_decoding.cuh"

BARACUDA_KERNELS_FLASH_DECODING_INSTANTIATE(flash_decoding_f16, __half)
BARACUDA_KERNELS_FLASH_DECODING_INSTANTIATE(flash_decoding_bf16, __nv_bfloat16)
