// baracuda-kernels Milestone 5.4: Sparsemax BW for FP types.
//
// `dx[i] = dy[i] - sum_dy_active / n_active` for active positions
// (`y > 0`), else `0`. Needs saved forward output `y`.

#include "../include/baracuda_softmax.cuh"

BARACUDA_KERNELS_SPARSEMAX_BACKWARD_INSTANTIATE(sparsemax_backward_f32, float)
BARACUDA_KERNELS_SPARSEMAX_BACKWARD_INSTANTIATE(sparsemax_backward_f16, __half)
BARACUDA_KERNELS_SPARSEMAX_BACKWARD_INSTANTIATE(sparsemax_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_SPARSEMAX_BACKWARD_INSTANTIATE(sparsemax_backward_f64, double)
