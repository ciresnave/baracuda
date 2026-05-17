// baracuda-kernels Phase 8 Milestone 8.1 — per-channel quantize / dequantize.
//
// Same dtype matrix as per_tensor.cu (f32/f64/f16/bf16 × s8/u8). The
// per-channel kernel works on a rank-4 padded tensor (caller pads with
// extents of 1) and uses a runtime `axis` to select which dim indexes
// the scale[] / zp[] vectors (both length C).

#include "../include/baracuda_quantize.cuh"

// ----- quantize_per_channel FW -----
BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_INSTANTIATE(quantize_per_channel_f32_s8, float, int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_INSTANTIATE(quantize_per_channel_f32_u8, float, uint8_t)
BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_INSTANTIATE(quantize_per_channel_f16_s8, __half, int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_INSTANTIATE(quantize_per_channel_f16_u8, __half, uint8_t)
BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_INSTANTIATE(quantize_per_channel_bf16_s8, __nv_bfloat16, int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_INSTANTIATE(quantize_per_channel_bf16_u8, __nv_bfloat16, uint8_t)

BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_F64_INSTANTIATE(quantize_per_channel_f64_s8, int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_F64_INSTANTIATE(quantize_per_channel_f64_u8, uint8_t)

// ----- quantize_per_channel BW (STE) -----
BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_BW_INSTANTIATE(quantize_per_channel_backward_f32, float)
BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_BW_INSTANTIATE(quantize_per_channel_backward_f16, __half)
BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_BW_INSTANTIATE(quantize_per_channel_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_QUANTIZE_PER_CHANNEL_BW_F64_INSTANTIATE(quantize_per_channel_backward_f64)

// ----- dequantize_per_channel FW -----
BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_INSTANTIATE(dequantize_per_channel_f32_s8, float, int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_INSTANTIATE(dequantize_per_channel_f32_u8, float, uint8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_INSTANTIATE(dequantize_per_channel_f16_s8, __half, int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_INSTANTIATE(dequantize_per_channel_f16_u8, __half, uint8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_INSTANTIATE(dequantize_per_channel_bf16_s8, __nv_bfloat16, int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_INSTANTIATE(dequantize_per_channel_bf16_u8, __nv_bfloat16, uint8_t)

BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_F64_INSTANTIATE(dequantize_per_channel_f64_s8, int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_F64_INSTANTIATE(dequantize_per_channel_f64_u8, uint8_t)

// ----- dequantize_per_channel BW -----
BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_BW_INSTANTIATE(dequantize_per_channel_backward_f32, float)
BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_BW_INSTANTIATE(dequantize_per_channel_backward_f16, __half)
BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_BW_INSTANTIATE(dequantize_per_channel_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_DEQUANTIZE_PER_CHANNEL_BW_F64_INSTANTIATE(dequantize_per_channel_backward_f64)
