// baracuda-kernels Milestone 5.3 — PReLU FW for FP types.
#include "../include/baracuda_prelu.cuh"
BARACUDA_KERNELS_PRELU_FW_INSTANTIATE(prelu_f32, float)
BARACUDA_KERNELS_PRELU_FW_INSTANTIATE(prelu_f16, __half)
BARACUDA_KERNELS_PRELU_FW_INSTANTIATE(prelu_bf16, __nv_bfloat16)
BARACUDA_KERNELS_PRELU_FW_INSTANTIATE(prelu_f64, double)
