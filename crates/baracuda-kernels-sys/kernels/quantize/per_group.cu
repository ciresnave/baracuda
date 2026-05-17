// baracuda-kernels Phase 8 Milestone 8.2 — per-group quantize / dequantize
// (Category P, INT4-LLM-style weight quantization; g=128 typical).
//
// Trailblazer scope: quant axis MUST be the rightmost (last) axis so the
// layout is naturally group-contiguous. Higher-rank inputs are flattened
// to `[outer, axis_size]` at the safe-plan layer.
//
// 8 SKUs per quantize/dequantize FW; 4 per BW direction (TIn only).
//
// Sibling 8.1 owns the per-tensor / per-channel / fake_quantize .cu
// files; no symbol overlap.

#include "../include/baracuda_quantize_per_token_group.cuh"

// ---------- quantize_per_group forward ----------
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_INSTANTIATE(
    quantize_per_group_f32_s8,  float,         int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_INSTANTIATE(
    quantize_per_group_f32_u8,  float,         uint8_t)
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_INSTANTIATE(
    quantize_per_group_f64_s8,  double,        int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_INSTANTIATE(
    quantize_per_group_f64_u8,  double,        uint8_t)
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_INSTANTIATE(
    quantize_per_group_f16_s8,  __half,        int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_INSTANTIATE(
    quantize_per_group_f16_u8,  __half,        uint8_t)
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_INSTANTIATE(
    quantize_per_group_bf16_s8, __nv_bfloat16, int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_INSTANTIATE(
    quantize_per_group_bf16_u8, __nv_bfloat16, uint8_t)

// ---------- quantize_per_group backward (STE — TIn only) ----------
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_BACKWARD_INSTANTIATE(
    quantize_per_group_backward_f32,  float)
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_BACKWARD_INSTANTIATE(
    quantize_per_group_backward_f64,  double)
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_BACKWARD_INSTANTIATE(
    quantize_per_group_backward_f16,  __half)
BARACUDA_KERNELS_QUANTIZE_PER_GROUP_BACKWARD_INSTANTIATE(
    quantize_per_group_backward_bf16, __nv_bfloat16)

// ---------- dequantize_per_group forward ----------
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_INSTANTIATE(
    dequantize_per_group_f32_s8,  float,         int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_INSTANTIATE(
    dequantize_per_group_f32_u8,  float,         uint8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_INSTANTIATE(
    dequantize_per_group_f64_s8,  double,        int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_INSTANTIATE(
    dequantize_per_group_f64_u8,  double,        uint8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_INSTANTIATE(
    dequantize_per_group_f16_s8,  __half,        int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_INSTANTIATE(
    dequantize_per_group_f16_u8,  __half,        uint8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_INSTANTIATE(
    dequantize_per_group_bf16_s8, __nv_bfloat16, int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_INSTANTIATE(
    dequantize_per_group_bf16_u8, __nv_bfloat16, uint8_t)

// ---------- dequantize_per_group backward (straight-through) ----------
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_BACKWARD_INSTANTIATE(
    dequantize_per_group_backward_f32,  float)
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_BACKWARD_INSTANTIATE(
    dequantize_per_group_backward_f64,  double)
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_BACKWARD_INSTANTIATE(
    dequantize_per_group_backward_f16,  __half)
BARACUDA_KERNELS_DEQUANTIZE_PER_GROUP_BACKWARD_INSTANTIATE(
    dequantize_per_group_backward_bf16, __nv_bfloat16)
