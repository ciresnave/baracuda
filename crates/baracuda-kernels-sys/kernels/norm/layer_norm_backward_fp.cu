// baracuda-kernels Phase 5 Category G: LayerNorm BW for FP types.
//
// Standard layer-norm gradient (biased variance):
//   x_hat[i] = (x[i] - mean) * inv_std
//   dx_hat[i] = dy[i] · gamma[i]   (or dy[i] if no gamma)
//   dx[i] = inv_std · (dx_hat[i] - sum_dxh / N - x_hat[i] · sum_dxhxh / N)
//   dgamma[i] = Σ over non-norm-axis cells dy[..., i] · x_hat[..., i]
//   dbeta[i]  = Σ over non-norm-axis cells dy[..., i]
//
// Single launcher fires the dx kernel and (when dgamma/dbeta non-null)
// a one-block-per-feature reduction kernel for the affine grads.

#include "../include/baracuda_norm.cuh"

BARACUDA_KERNELS_LAYER_NORM_BACKWARD_INSTANTIATE(layer_norm_backward_f32, float)
BARACUDA_KERNELS_LAYER_NORM_BACKWARD_INSTANTIATE(layer_norm_backward_f16, __half)
BARACUDA_KERNELS_LAYER_NORM_BACKWARD_INSTANTIATE(layer_norm_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LAYER_NORM_BACKWARD_INSTANTIATE(layer_norm_backward_f64, double)

// Phase 72 strided-sibling FFI exports — same underlying launcher.
BARACUDA_KERNELS_LAYER_NORM_BACKWARD_INSTANTIATE_STRIDED(layer_norm_backward_f32, float)
BARACUDA_KERNELS_LAYER_NORM_BACKWARD_INSTANTIATE_STRIDED(layer_norm_backward_f16, __half)
BARACUDA_KERNELS_LAYER_NORM_BACKWARD_INSTANTIATE_STRIDED(layer_norm_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LAYER_NORM_BACKWARD_INSTANTIATE_STRIDED(layer_norm_backward_f64, double)
