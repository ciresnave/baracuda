// baracuda-kernels Phase 4 reduce backward: prod backward.
//
// Forward: `y = prod(x, axis=k)` (keepdim). Backward:
//   `dx[c] = dy[c_reduced] * y[c_reduced] / x[c]`
// where `c_reduced` collapses the reduce axis to 0. Requires BOTH saved
// `x` (forward input, full shape) and saved `y` (forward output,
// keepdim shape). Caller must ensure `x[c] != 0` per cell — division
// produces NaN/inf otherwise (matches PyTorch convention).

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_REDUCE_PROD_BACKWARD_INSTANTIATE(reduce_prod_backward_f32, float)
BARACUDA_KERNELS_REDUCE_PROD_BACKWARD_INSTANTIATE(reduce_prod_backward_f16, __half)
BARACUDA_KERNELS_REDUCE_PROD_BACKWARD_INSTANTIATE(reduce_prod_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_REDUCE_PROD_BACKWARD_INSTANTIATE(reduce_prod_backward_f64, double)
