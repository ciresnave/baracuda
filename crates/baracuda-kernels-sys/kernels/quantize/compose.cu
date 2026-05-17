// baracuda-kernels Phase 8 Milestone 8.3 — composing quantization ops
// (DynamicRangeQuantize + QuantizedLinear).
//
// SKUs:
//   dynamic_range_quantize_per_token_sym × {f32, f64} × {s8}  = 2
//   quantized_linear_w8a8                  × {f32, f64}        = 2
//
// Sibling 8.1 / 8.2 ships the primitive per-tensor / per-channel /
// per-token / per-group quantize + dequantize kernels; this milestone
// composes on top of them. We deliberately re-implement the
// per-token-quantize loop inside the dynamic-range kernel rather than
// calling the standalone one, because the row-wise max-abs reduction
// produces the divisor that the per-row quantize step needs — running
// them as one launch avoids a separate scale buffer round-trip.

#include "../include/baracuda_quantize_compose.cuh"

// ---------- dynamic_range_quantize_per_token, symmetric ----------
BARACUDA_KERNELS_DYNAMIC_RANGE_QUANTIZE_PER_TOKEN_SYM_INSTANTIATE(
    dynamic_range_quantize_per_token_sym_f32_s8, float,  int8_t)
BARACUDA_KERNELS_DYNAMIC_RANGE_QUANTIZE_PER_TOKEN_SYM_INSTANTIATE(
    dynamic_range_quantize_per_token_sym_f64_s8, double, int8_t)

// ---------- quantized_linear (W8A8 naive trailblazer) ----------
BARACUDA_KERNELS_QUANTIZED_LINEAR_W8A8_INSTANTIATE(
    quantized_linear_w8a8_f32, float)
BARACUDA_KERNELS_QUANTIZED_LINEAR_W8A8_INSTANTIATE(
    quantized_linear_w8a8_f64, double)
