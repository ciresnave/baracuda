// baracuda-kernels Phase 4 reduce backward: L2 norm (norm2) backward.
//
// Forward: `y = sqrt(sum(x², axis=k))` (keepdim). Backward:
//   `dx[c] = dy[c_reduced] * x[c] / y[c_reduced]`
// Requires BOTH saved `x` (full shape) and saved `y` (keepdim shape).
// Caller must ensure `y[c_reduced] != 0` (only happens if every x in
// the reduced group is exactly zero).

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_REDUCE_NORM2_BACKWARD_INSTANTIATE(reduce_norm2_backward_f32, float)
BARACUDA_KERNELS_REDUCE_NORM2_BACKWARD_INSTANTIATE(reduce_norm2_backward_f16, __half)
BARACUDA_KERNELS_REDUCE_NORM2_BACKWARD_INSTANTIATE(reduce_norm2_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_REDUCE_NORM2_BACKWARD_INSTANTIATE(reduce_norm2_backward_f64, double)
