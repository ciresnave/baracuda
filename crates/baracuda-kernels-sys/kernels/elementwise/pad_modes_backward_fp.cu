// baracuda-kernels SA-30: pad backward for Reflect / Replicate / Circular.
// Deterministic gather-adjoint of the non-constant pad forward modes (the
// mode is a runtime argument: 0 = Reflect, 1 = Replicate, 2 = Circular).

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_PAD_MODES_BACKWARD_INSTANTIATE(pad_modes_backward_f32, float)
BARACUDA_KERNELS_PAD_MODES_BACKWARD_INSTANTIATE(pad_modes_backward_f16, __half)
BARACUDA_KERNELS_PAD_MODES_BACKWARD_INSTANTIATE(pad_modes_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_PAD_MODES_BACKWARD_INSTANTIATE(pad_modes_backward_f64, double)
