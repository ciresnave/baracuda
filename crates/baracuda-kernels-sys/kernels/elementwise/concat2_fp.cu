// baracuda-kernels Phase 3 Category N: 2-input `concat` for FP types.
//
// `y = cat(a, b, dim=k)`. Output shape per-axis: same as a / b except
// `output[k] = a.shape[k] + b.shape[k]`. Trailblazer scope:
// - 2 inputs only (variable-arity Concat-N is a future plan shape)
// - f32 only (other dtypes are single-INSTANTIATE fanout)
// - Contig inputs and contig output
//
// Kernel template lives in `include/baracuda_elementwise.cuh`; this
// file supplies INSTANTIATE invocations.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_CONCAT2_INSTANTIATE(concat2_f32, float)
BARACUDA_KERNELS_CONCAT2_INSTANTIATE(concat2_f16, __half)
BARACUDA_KERNELS_CONCAT2_INSTANTIATE(concat2_bf16, __nv_bfloat16)
BARACUDA_KERNELS_CONCAT2_INSTANTIATE(concat2_f64, double)
