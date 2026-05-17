// baracuda-kernels Milestone 5.4: GumbelSoftmax FW for FP types.
//
// `y = softmax((x + g) / τ)` where `g = -log(-log(u))` with `u` a
// caller-supplied cuRAND uniform-rand buffer. Optional `hard` mode
// emits a one-hot at the row's noisy argmax; BW pipes through the
// existing softmax_backward_fp kernel using the saved soft output.

#include "../include/baracuda_softmax.cuh"

BARACUDA_KERNELS_GUMBEL_SOFTMAX_INSTANTIATE(gumbel_softmax_f32, float)
BARACUDA_KERNELS_GUMBEL_SOFTMAX_INSTANTIATE(gumbel_softmax_f16, __half)
BARACUDA_KERNELS_GUMBEL_SOFTMAX_INSTANTIATE(gumbel_softmax_bf16, __nv_bfloat16)
BARACUDA_KERNELS_GUMBEL_SOFTMAX_INSTANTIATE(gumbel_softmax_f64, double)
