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

// Phase 41 (Fuel Phase 6c.4 Gap 7) — RoPE apply interleaved variant.
// Pair convention `(2k, 2k+1)`. The existing `rope_apply_fp_kernel`
// already implements exactly this pairing — these symbols are thin
// re-exports under the Fuel-expected name so `RotaryEmbI` callers can
// retire the `Id::Reduce` PTX module and the `fuel-cuda-kernels`
// workspace member.
BARACUDA_KERNELS_ROPE_APPLY_INTERLEAVED_INSTANTIATE(rope_apply_interleaved_f32,  float)
BARACUDA_KERNELS_ROPE_APPLY_INTERLEAVED_INSTANTIATE(rope_apply_interleaved_f16,  __half)
BARACUDA_KERNELS_ROPE_APPLY_INTERLEAVED_INSTANTIATE(rope_apply_interleaved_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ROPE_APPLY_INTERLEAVED_INSTANTIATE(rope_apply_interleaved_f64,  double)

// Phase 41 (Fuel Phase 6c.4 Gap 8) — RoPE apply THD-layout variant.
// Operand layout `[T, H, D]` (packed batch * seq into T) instead of
// canonical `[B, H, T, D]`. cos/sin layout `cs[t * stride_b + pair]`
// with `stride_b == D/2` per-t tables or `0` shared.
BARACUDA_KERNELS_ROPE_APPLY_THD_INSTANTIATE(rope_apply_thd_f32,  float)
BARACUDA_KERNELS_ROPE_APPLY_THD_INSTANTIATE(rope_apply_thd_f16,  __half)
BARACUDA_KERNELS_ROPE_APPLY_THD_INSTANTIATE(rope_apply_thd_bf16, __nv_bfloat16)
BARACUDA_KERNELS_ROPE_APPLY_THD_INSTANTIATE(rope_apply_thd_f64,  double)
