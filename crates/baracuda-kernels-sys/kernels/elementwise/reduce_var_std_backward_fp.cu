// baracuda-kernels Phase 4 reduce backward: variance / std-dev backward
// (Welford BW).
//
// Forward: `y = var(x, axis=k, correction=c)` or `y = std = sqrt(var)`.
// Backward (with denom `m = max(n - correction, 1)`):
//   Var BW: `dx[c] = dy[c_reduced] * 2 * (x[c] - mean[c_reduced]) / m`
//   Std BW: `dx[c] = dy[c_reduced] * (x[c] - mean[c_reduced]) / (m * y[c_reduced])`
//
// Requires saved `x` (forward input, full shape). Std BW additionally
// requires saved `y` (forward output, keepdim shape). Var BW accepts a
// `y` pointer but ignores it (ABI uniformity with Std).
//
// `mean` is recomputed inline (one extra pass over the reduce axis on
// `x` per output cell) — keeps the dual-save ABI tidy. Cost: `n` extra
// reads per output cell; acceptable for n in 16–1024.
//
// All four FP dtypes wired (Phase 4 deferral 4.2 close-out):
//   {var_bw, std_bw} × {f32, f16, bf16, f64} = 8 FFI symbols.
// Internal accumulation: f32 for f32/f16/bf16, f64 for f64.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_REDUCE_WELFORD_BACKWARD_INSTANTIATE(reduce_var_backward_f32,  float,         false)
BARACUDA_KERNELS_REDUCE_WELFORD_BACKWARD_INSTANTIATE(reduce_std_backward_f32,  float,         true)
BARACUDA_KERNELS_REDUCE_WELFORD_BACKWARD_INSTANTIATE(reduce_var_backward_f16,  __half,        false)
BARACUDA_KERNELS_REDUCE_WELFORD_BACKWARD_INSTANTIATE(reduce_std_backward_f16,  __half,        true)
BARACUDA_KERNELS_REDUCE_WELFORD_BACKWARD_INSTANTIATE(reduce_var_backward_bf16, __nv_bfloat16, false)
BARACUDA_KERNELS_REDUCE_WELFORD_BACKWARD_INSTANTIATE(reduce_std_backward_bf16, __nv_bfloat16, true)
BARACUDA_KERNELS_REDUCE_WELFORD_BACKWARD_INSTANTIATE(reduce_var_backward_f64,  double,        false)
BARACUDA_KERNELS_REDUCE_WELFORD_BACKWARD_INSTANTIATE(reduce_std_backward_f64,  double,        true)
