// baracuda-kernels Phase 5 trailblazer: softmax BW for FP types.
//
// `dx[k] = y[k] · (dy[k] - Σ_j y[j] · dy[j])` along the softmax axis,
// where `y` is the saved forward output. Per thread: walks the softmax
// axis once to compute the row's `dot` = Σ_j y[j] · dy[j], then
// applies the per-cell BW formula. f16 / bf16 accumulate in f32.

#include "../include/baracuda_softmax.cuh"

BARACUDA_KERNELS_SOFTMAX_BACKWARD_INSTANTIATE(softmax_backward_f32, float)
BARACUDA_KERNELS_SOFTMAX_BACKWARD_INSTANTIATE(softmax_backward_f16, __half)
BARACUDA_KERNELS_SOFTMAX_BACKWARD_INSTANTIATE(softmax_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_SOFTMAX_BACKWARD_INSTANTIATE(softmax_backward_f64, double)

// Phase 72 strided-sibling FFI exports — same underlying launcher.
BARACUDA_KERNELS_SOFTMAX_BACKWARD_INSTANTIATE_STRIDED(softmax_backward_f32, float)
BARACUDA_KERNELS_SOFTMAX_BACKWARD_INSTANTIATE_STRIDED(softmax_backward_f16, __half)
BARACUDA_KERNELS_SOFTMAX_BACKWARD_INSTANTIATE_STRIDED(softmax_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_SOFTMAX_BACKWARD_INSTANTIATE_STRIDED(softmax_backward_f64, double)
