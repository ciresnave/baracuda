// baracuda-kernels Phase 3 heterogeneous-dtype ternary BW:
// elementwise `where_backward(cond, dy)`.
//
// Forward: `y = cond ? a : b`. Backward (cond is non-differentiable):
//   da[i] = cond[i] ? dy[i] : 0
//   db[i] = cond[i] ? 0     : dy[i]
//
// Pure mask + copy — no arithmetic — so output is bit-exact against
// host reference at every dtype. Trailblazer is contig-only: caller
// materializes broadcasted dy / da / db before launching. The FW Where
// supports broadcast on every operand; BW doesn't because materializing
// cond-shaped gradients into full-shape `da` / `db` is what the
// autograd reduction step does upstream of this kernel anyway.
//
// All 4 FP value dtypes wired: {f32, f16, bf16, f64} = 4 launcher
// symbols. The kernel template is fully generic in `T`.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_WHERE_BACKWARD_INSTANTIATE(where_backward_f32, float)
BARACUDA_KERNELS_WHERE_BACKWARD_INSTANTIATE(where_backward_f16, __half)
BARACUDA_KERNELS_WHERE_BACKWARD_INSTANTIATE(where_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_WHERE_BACKWARD_INSTANTIATE(where_backward_f64, double)
