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

// Phase 14.4 strided siblings.
BARACUDA_KERNELS_ROPE_BACKWARD_STRIDED_INSTANTIATE(rope_backward_f32, float)
BARACUDA_KERNELS_ROPE_BACKWARD_STRIDED_INSTANTIATE(rope_backward_f16, __half)
BARACUDA_KERNELS_ROPE_BACKWARD_STRIDED_INSTANTIATE(rope_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ROPE_BACKWARD_STRIDED_INSTANTIATE(rope_backward_f64, double)

// Phase 36 (Fuel ask Gap 2) — RoPE apply BW siblings. Caller-supplied
// cos/sin tables; orthogonal-rotation backward.
BARACUDA_KERNELS_ROPE_APPLY_BACKWARD_INSTANTIATE(rope_apply_backward_f32,  float)
BARACUDA_KERNELS_ROPE_APPLY_BACKWARD_INSTANTIATE(rope_apply_backward_f16,  __half)
BARACUDA_KERNELS_ROPE_APPLY_BACKWARD_INSTANTIATE(rope_apply_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ROPE_APPLY_BACKWARD_INSTANTIATE(rope_apply_backward_f64,  double)

// Phase 41 (Fuel Phase 6c.4 Gap 7) — RoPE apply interleaved BW. Thin
// re-export of `launch_rope_apply_backward_fp<T>` under the Fuel-expected
// name. Pair convention `(2k, 2k+1)`.
BARACUDA_KERNELS_ROPE_APPLY_INTERLEAVED_BACKWARD_INSTANTIATE(rope_apply_interleaved_backward_f32,  float)
BARACUDA_KERNELS_ROPE_APPLY_INTERLEAVED_BACKWARD_INSTANTIATE(rope_apply_interleaved_backward_f16,  __half)
BARACUDA_KERNELS_ROPE_APPLY_INTERLEAVED_BACKWARD_INSTANTIATE(rope_apply_interleaved_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ROPE_APPLY_INTERLEAVED_BACKWARD_INSTANTIATE(rope_apply_interleaved_backward_f64,  double)

// Phase 41 (Fuel Phase 6c.4 Gap 8) — RoPE apply THD BW. Orthogonal-rotation
// reverse over `[T, H, D]` layout.
BARACUDA_KERNELS_ROPE_APPLY_THD_BACKWARD_INSTANTIATE(rope_apply_thd_backward_f32,  float)
BARACUDA_KERNELS_ROPE_APPLY_THD_BACKWARD_INSTANTIATE(rope_apply_thd_backward_f16,  __half)
BARACUDA_KERNELS_ROPE_APPLY_THD_BACKWARD_INSTANTIATE(rope_apply_thd_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ROPE_APPLY_THD_BACKWARD_INSTANTIATE(rope_apply_thd_backward_f64,  double)
