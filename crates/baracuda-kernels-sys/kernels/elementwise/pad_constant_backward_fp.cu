// baracuda-kernels Phase 3 BW for Category N: pad-constant backward
// (slice). `dx = dy[pad_low : pad_low + input_shape]` per axis. The
// pad-region cells of `dy` are discarded — their gradient w.r.t. the
// forward input is identically zero (the forward write was a constant).

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_PAD_CONSTANT_BACKWARD_INSTANTIATE(pad_constant_backward_f32, float)
BARACUDA_KERNELS_PAD_CONSTANT_BACKWARD_INSTANTIATE(pad_constant_backward_f16, __half)
BARACUDA_KERNELS_PAD_CONSTANT_BACKWARD_INSTANTIATE(pad_constant_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_PAD_CONSTANT_BACKWARD_INSTANTIATE(pad_constant_backward_f64, double)
