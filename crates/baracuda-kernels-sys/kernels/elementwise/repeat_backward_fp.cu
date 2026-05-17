// baracuda-kernels Phase 3 BW for Category N: repeat backward
// (gather-adjoint sum). `dx[c_in] = sum_{k} dy[c_in + k * input_shape]`
// per axis — every dy cell whose `c_out[d] % input_shape[d] == c_in[d]`
// for all d contributes. One thread per dx cell loops the per-axis
// repeats grid and accumulates. f16 / bf16 accumulate in float for
// numerical stability; f32 / f64 accumulate in their native dtype.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_REPEAT_BACKWARD_INSTANTIATE(repeat_backward_f32, float)
BARACUDA_KERNELS_REPEAT_BACKWARD_INSTANTIATE(repeat_backward_f16, __half)
BARACUDA_KERNELS_REPEAT_BACKWARD_INSTANTIATE(repeat_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_REPEAT_BACKWARD_INSTANTIATE(repeat_backward_f64, double)
