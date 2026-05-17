// baracuda-kernels Phase 3 shape/layout trailblazer: elementwise
// `pad` across all four modes.
//
// `y = pad(x, pad_low, pad_high, mode[, value])` — output shape
// per-axis is `input[d] + pad_low[d] + pad_high[d]`. Output cells
// inside the shifted input region copy from x; cells in the pad region
// are filled per `mode`:
//
//   Constant  — fixed scalar `value`.
//   Reflect   — mirror across the boundary (no edge dup).
//   Replicate — clamp to the boundary value.
//   Circular  — cyclic wrap from the opposite end.
//
// Each (mode, dtype) cell is one INSTANTIATE on the matching kernel
// template in `baracuda_elementwise.cuh`. Modes other than Constant
// don't take a `value` parameter — the pad region is read from the
// input itself.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_PAD_CONSTANT_INSTANTIATE(pad_constant_f32, float)
BARACUDA_KERNELS_PAD_CONSTANT_INSTANTIATE(pad_constant_f16, __half)
BARACUDA_KERNELS_PAD_CONSTANT_INSTANTIATE(pad_constant_bf16, __nv_bfloat16)
BARACUDA_KERNELS_PAD_CONSTANT_INSTANTIATE(pad_constant_f64, double)

// Reflect — mirror across boundary, no edge duplication.
BARACUDA_KERNELS_PAD_REFLECT_INSTANTIATE(pad_reflect_f32, float)
BARACUDA_KERNELS_PAD_REFLECT_INSTANTIATE(pad_reflect_f16, __half)
BARACUDA_KERNELS_PAD_REFLECT_INSTANTIATE(pad_reflect_bf16, __nv_bfloat16)
BARACUDA_KERNELS_PAD_REFLECT_INSTANTIATE(pad_reflect_f64, double)

// Replicate — clamp to edge.
BARACUDA_KERNELS_PAD_REPLICATE_INSTANTIATE(pad_replicate_f32, float)
BARACUDA_KERNELS_PAD_REPLICATE_INSTANTIATE(pad_replicate_f16, __half)
BARACUDA_KERNELS_PAD_REPLICATE_INSTANTIATE(pad_replicate_bf16, __nv_bfloat16)
BARACUDA_KERNELS_PAD_REPLICATE_INSTANTIATE(pad_replicate_f64, double)

// Circular — cyclic wrap.
BARACUDA_KERNELS_PAD_CIRCULAR_INSTANTIATE(pad_circular_f32, float)
BARACUDA_KERNELS_PAD_CIRCULAR_INSTANTIATE(pad_circular_f16, __half)
BARACUDA_KERNELS_PAD_CIRCULAR_INSTANTIATE(pad_circular_bf16, __nv_bfloat16)
BARACUDA_KERNELS_PAD_CIRCULAR_INSTANTIATE(pad_circular_f64, double)
