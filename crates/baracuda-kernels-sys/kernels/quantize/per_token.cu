// baracuda-kernels Phase 8 Milestone 8.2 — per-token quantize / dequantize
// (Category P, LLM/GPTQ-style activation quantization).
//
// 8 SKUs per direction:
//   FW  (TIn x TOut): {f32, f64, f16, bf16} x {s8, u8} = 8
//   BW  (TIn only):   {f32, f64, f16, bf16} = 4
//   Dequant FW:       {f32, f64, f16, bf16} x {s8, u8} = 8
//   Dequant BW:       {f32, f64, f16, bf16} = 4
//
// Sibling 8.1 (per-tensor / per-channel / fake_quantize) ships in
// `kernels/quantize/per_tensor.cu`, `per_channel.cu`, `fake_quantize.cu`.
// No symbol overlap — every name here is scoped under
// `baracuda_kernels_quantize_per_token_*` /
// `baracuda_kernels_dequantize_per_token_*`.

#include "../include/baracuda_quantize_per_token_group.cuh"

// ---------- quantize_per_token forward ----------
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_INSTANTIATE(
    quantize_per_token_f32_s8,  float,         int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_INSTANTIATE(
    quantize_per_token_f32_u8,  float,         uint8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_INSTANTIATE(
    quantize_per_token_f64_s8,  double,        int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_INSTANTIATE(
    quantize_per_token_f64_u8,  double,        uint8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_INSTANTIATE(
    quantize_per_token_f16_s8,  __half,        int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_INSTANTIATE(
    quantize_per_token_f16_u8,  __half,        uint8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_INSTANTIATE(
    quantize_per_token_bf16_s8, __nv_bfloat16, int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_INSTANTIATE(
    quantize_per_token_bf16_u8, __nv_bfloat16, uint8_t)

// ---------- quantize_per_token backward (STE — TIn only) ----------
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_BACKWARD_INSTANTIATE(
    quantize_per_token_backward_f32,  float)
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_BACKWARD_INSTANTIATE(
    quantize_per_token_backward_f64,  double)
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_BACKWARD_INSTANTIATE(
    quantize_per_token_backward_f16,  __half)
BARACUDA_KERNELS_QUANTIZE_PER_TOKEN_BACKWARD_INSTANTIATE(
    quantize_per_token_backward_bf16, __nv_bfloat16)

// ---------- dequantize_per_token forward ----------
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_INSTANTIATE(
    dequantize_per_token_f32_s8,  float,         int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_INSTANTIATE(
    dequantize_per_token_f32_u8,  float,         uint8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_INSTANTIATE(
    dequantize_per_token_f64_s8,  double,        int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_INSTANTIATE(
    dequantize_per_token_f64_u8,  double,        uint8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_INSTANTIATE(
    dequantize_per_token_f16_s8,  __half,        int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_INSTANTIATE(
    dequantize_per_token_f16_u8,  __half,        uint8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_INSTANTIATE(
    dequantize_per_token_bf16_s8, __nv_bfloat16, int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_INSTANTIATE(
    dequantize_per_token_bf16_u8, __nv_bfloat16, uint8_t)

// ---------- dequantize_per_token backward (straight-through) ----------
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_BACKWARD_INSTANTIATE(
    dequantize_per_token_backward_f32,  float)
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_BACKWARD_INSTANTIATE(
    dequantize_per_token_backward_f64,  double)
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_BACKWARD_INSTANTIATE(
    dequantize_per_token_backward_f16,  __half)
BARACUDA_KERNELS_DEQUANTIZE_PER_TOKEN_BACKWARD_INSTANTIATE(
    dequantize_per_token_backward_bf16, __nv_bfloat16)
