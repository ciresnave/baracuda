// baracuda-kernels Phase 4: variance + std-dev axis reductions
// via Welford's one-pass online algorithm.
//
// `correction` parameter:
//   correction = 1 → sample variance (Bessel-corrected, PyTorch default)
//   correction = 0 → population variance
//
// Internal Welford state runs in `WelfordAcc<T>` — `float` for
// f32/f16/bf16 and `double` for f64. f16 / bf16 detour through f32 at
// load/store time; the math (delta / mean update / M2 accumulation) is
// done at f32. f64 stays at double end-to-end. This is critical for
// f16 — its 11-bit mantissa would lose all precision during the
// Welford update if we accumulated in T.
//
// All four FP dtypes wired (Phase 4 deferral 4.2 close-out):
//   {var, std} × {f32, f16, bf16, f64} = 8 FFI symbols.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_REDUCE_WELFORD_INSTANTIATE(reduce_var_f32,  float,         false)
BARACUDA_KERNELS_REDUCE_WELFORD_INSTANTIATE(reduce_std_f32,  float,         true)
BARACUDA_KERNELS_REDUCE_WELFORD_INSTANTIATE(reduce_var_f16,  __half,        false)
BARACUDA_KERNELS_REDUCE_WELFORD_INSTANTIATE(reduce_std_f16,  __half,        true)
BARACUDA_KERNELS_REDUCE_WELFORD_INSTANTIATE(reduce_var_bf16, __nv_bfloat16, false)
BARACUDA_KERNELS_REDUCE_WELFORD_INSTANTIATE(reduce_std_bf16, __nv_bfloat16, true)
BARACUDA_KERNELS_REDUCE_WELFORD_INSTANTIATE(reduce_var_f64,  double,        false)
BARACUDA_KERNELS_REDUCE_WELFORD_INSTANTIATE(reduce_std_f64,  double,        true)
