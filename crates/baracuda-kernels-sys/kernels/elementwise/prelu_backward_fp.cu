// baracuda-kernels Milestone 5.3 — PReLU BW for FP types.
#include "../include/baracuda_prelu.cuh"
BARACUDA_KERNELS_PRELU_BW_INSTANTIATE(prelu_backward_f32, float)
BARACUDA_KERNELS_PRELU_BW_INSTANTIATE(prelu_backward_f16, __half)
BARACUDA_KERNELS_PRELU_BW_INSTANTIATE(prelu_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_PRELU_BW_INSTANTIATE(prelu_backward_f64, double)
