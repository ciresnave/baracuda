// baracuda-kernels Phase 6 Category K — RoPE BW for FP types.
//
// Rotation matrix is orthogonal, so BW is rotation by -θ:
//   dx[2i]   = dy[2i]   · cos(θ) + dy[2i+1] · sin(θ)
//   dx[2i+1] = dy[2i+1] · cos(θ) - dy[2i]   · sin(θ)

#include "../include/baracuda_attention.cuh"

BARACUDA_KERNELS_ROPE_BACKWARD_INSTANTIATE(rope_backward_f32, float)
BARACUDA_KERNELS_ROPE_BACKWARD_INSTANTIATE(rope_backward_f16, __half)
BARACUDA_KERNELS_ROPE_BACKWARD_INSTANTIATE(rope_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ROPE_BACKWARD_INSTANTIATE(rope_backward_f64, double)
