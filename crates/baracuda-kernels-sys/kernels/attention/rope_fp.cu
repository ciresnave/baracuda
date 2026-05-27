// baracuda-kernels Phase 6 Category K — RoPE FW for FP types.
//
// Rotary position embedding: rotates pairs (2i, 2i+1) of a [B, H, S, D]
// Q/K tensor by per-position angles θ_i = pos · base^(-2i/D).
// f16 / bf16 detour through f32 for trig and arithmetic.

#include "../include/baracuda_attention.cuh"

BARACUDA_KERNELS_ROPE_INSTANTIATE(rope_f32, float)
BARACUDA_KERNELS_ROPE_INSTANTIATE(rope_f16, __half)
BARACUDA_KERNELS_ROPE_INSTANTIATE(rope_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ROPE_INSTANTIATE(rope_f64, double)

// Phase 14.4 strided siblings — outer-dim strides only (head_dim
// must remain stride=1, enforced by the Rust plan layer).
BARACUDA_KERNELS_ROPE_STRIDED_INSTANTIATE(rope_f32, float)
BARACUDA_KERNELS_ROPE_STRIDED_INSTANTIATE(rope_f16, __half)
BARACUDA_KERNELS_ROPE_STRIDED_INSTANTIATE(rope_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ROPE_STRIDED_INSTANTIATE(rope_f64, double)

// Phase 36 (Fuel ask Gap 2) — RoPE apply variant with caller-supplied
// precomputed cos/sin tables. Coexists with the existing θ-from-scratch
// `rope_<dt>_run` family. Cos/sin are always f32 over the FFI (Fuel
// convention: bake-time trig in f32, f16/bf16 detour through f32 in
// the kernel).
BARACUDA_KERNELS_ROPE_APPLY_INSTANTIATE(rope_apply_f32,  float)
BARACUDA_KERNELS_ROPE_APPLY_INSTANTIATE(rope_apply_f16,  __half)
BARACUDA_KERNELS_ROPE_APPLY_INSTANTIATE(rope_apply_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ROPE_APPLY_INSTANTIATE(rope_apply_f64,  double)
