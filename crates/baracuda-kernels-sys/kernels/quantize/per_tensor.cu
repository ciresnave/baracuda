// baracuda-kernels Phase 8 Milestone 8.1 — per-tensor quantize / dequantize.
//
// Trailblazer dtype matrix:
//   Input FP  : f32, f64, f16, bf16
//   Output Q  : s8 (int8_t), u8 (uint8_t)
//
// FW kernels: 8 SKUs per direction (quantize × 4 FP × 2 Q + dequantize × same).
// BW kernels: 4 SKUs per direction (input-FP only — gradient is FP-typed).
//
// `scale` is carried at FP-input precision: f32 for the f32 / f16 / bf16
// inputs (since the math happens after promoting to f32), and f64 for the
// f64 input (preserves double precision through the divide / round). The
// macros split into `_f32` and `_f64` flavors at the FFI boundary so the
// Rust plan layer can call the right signature without runtime branching.

#include "../include/baracuda_quantize.cuh"

// ----- quantize_per_tensor FW (f32-scale) -----
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_INSTANTIATE(quantize_per_tensor_f32_s8, float, int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_INSTANTIATE(quantize_per_tensor_f32_u8, float, uint8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_INSTANTIATE(quantize_per_tensor_f16_s8, __half, int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_INSTANTIATE(quantize_per_tensor_f16_u8, __half, uint8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_INSTANTIATE(quantize_per_tensor_bf16_s8, __nv_bfloat16, int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_INSTANTIATE(quantize_per_tensor_bf16_u8, __nv_bfloat16, uint8_t)

// ----- quantize_per_tensor FW (f64-scale) -----
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_F64_INSTANTIATE(quantize_per_tensor_f64_s8, int8_t)
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_F64_INSTANTIATE(quantize_per_tensor_f64_u8, uint8_t)

// ----- quantize_per_tensor BW (STE, in-range mask recomputed from x) -----
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_BW_INSTANTIATE(quantize_per_tensor_backward_f32, float)
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_BW_INSTANTIATE(quantize_per_tensor_backward_f16, __half)
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_BW_INSTANTIATE(quantize_per_tensor_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_QUANTIZE_PER_TENSOR_BW_F64_INSTANTIATE(quantize_per_tensor_backward_f64)

// ----- dequantize_per_tensor FW -----
BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_INSTANTIATE(dequantize_per_tensor_f32_s8, float, int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_INSTANTIATE(dequantize_per_tensor_f32_u8, float, uint8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_INSTANTIATE(dequantize_per_tensor_f16_s8, __half, int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_INSTANTIATE(dequantize_per_tensor_f16_u8, __half, uint8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_INSTANTIATE(dequantize_per_tensor_bf16_s8, __nv_bfloat16, int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_INSTANTIATE(dequantize_per_tensor_bf16_u8, __nv_bfloat16, uint8_t)

BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_F64_INSTANTIATE(dequantize_per_tensor_f64_s8, int8_t)
BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_F64_INSTANTIATE(dequantize_per_tensor_f64_u8, uint8_t)

// ----- dequantize_per_tensor BW (dq = dy * scale; FP-typed) -----
BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_BW_INSTANTIATE(dequantize_per_tensor_backward_f32, float)
BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_BW_INSTANTIATE(dequantize_per_tensor_backward_f16, __half)
BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_BW_INSTANTIATE(dequantize_per_tensor_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_DEQUANTIZE_PER_TENSOR_BW_F64_INSTANTIATE(dequantize_per_tensor_backward_f64)
