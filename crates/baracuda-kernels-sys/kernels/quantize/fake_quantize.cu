// baracuda-kernels Phase 8 Milestone 8.1 — fake_quantize FW + BW.
//
// Round-trip quantize-then-dequantize in FP space. Output dtype == input
// dtype (no int storage involved). Per-tensor only for the trailblazer;
// per-channel fake-quantize follows in 8.2.
//
// Backward via STE: dx = dy * in_range_mask. No `1/scale` factor — the
// FW's dequant-side multiply by scale cancels the STE's 1/scale (see
// `baracuda_quantize.cuh` head-comment for the algebra).

#include "../include/baracuda_quantize.cuh"

// ----- fake_quantize FW -----
BARACUDA_KERNELS_FAKE_QUANTIZE_INSTANTIATE(fake_quantize_f32, float)
BARACUDA_KERNELS_FAKE_QUANTIZE_INSTANTIATE(fake_quantize_f16, __half)
BARACUDA_KERNELS_FAKE_QUANTIZE_INSTANTIATE(fake_quantize_bf16, __nv_bfloat16)
BARACUDA_KERNELS_FAKE_QUANTIZE_F64_INSTANTIATE(fake_quantize_f64)

// ----- fake_quantize BW -----
BARACUDA_KERNELS_FAKE_QUANTIZE_BW_INSTANTIATE(fake_quantize_backward_f32, float)
BARACUDA_KERNELS_FAKE_QUANTIZE_BW_INSTANTIATE(fake_quantize_backward_f16, __half)
BARACUDA_KERNELS_FAKE_QUANTIZE_BW_INSTANTIATE(fake_quantize_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_FAKE_QUANTIZE_BW_F64_INSTANTIATE(fake_quantize_backward_f64)
