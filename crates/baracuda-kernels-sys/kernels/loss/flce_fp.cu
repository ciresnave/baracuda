// baracuda-kernels Phase 47: Fused Linear Cross-Entropy bespoke kernels.
//
// Math/algorithm credit: LinkedIn Liger-Kernel (BSD-2-Clause).
//   https://github.com/linkedin/Liger-Kernel
// Clean-room CUDA reimplementation; no Liger source vendored.
//
// Today wired: f32, f16, bf16, f64.

#include "../include/baracuda_flce.cuh"

// Per-row fused step (logits -> grad_logits in place, per-row f32 loss).
BARACUDA_KERNELS_FLCE_PER_ROW_INSTANTIATE(loss_flce_per_row_f32, float)
BARACUDA_KERNELS_FLCE_PER_ROW_INSTANTIATE(loss_flce_per_row_f16, __half)
BARACUDA_KERNELS_FLCE_PER_ROW_INSTANTIATE(loss_flce_per_row_bf16, __nv_bfloat16)
BARACUDA_KERNELS_FLCE_PER_ROW_INSTANTIATE(loss_flce_per_row_f64, double)

// Per-row loss cast (None mode finalizer): f32 loss_1d -> T per-cell.
BARACUDA_KERNELS_FLCE_PER_ROW_CAST_INSTANTIATE(loss_flce_per_row_cast_f32, float)
BARACUDA_KERNELS_FLCE_PER_ROW_CAST_INSTANTIATE(loss_flce_per_row_cast_f16, __half)
BARACUDA_KERNELS_FLCE_PER_ROW_CAST_INSTANTIATE(loss_flce_per_row_cast_bf16, __nv_bfloat16)
BARACUDA_KERNELS_FLCE_PER_ROW_CAST_INSTANTIATE(loss_flce_per_row_cast_f64, double)

// Scalar finalize (Mean / Sum mode finalizer): f32 loss_1d -> scalar T.
BARACUDA_KERNELS_FLCE_SCALAR_FINALIZE_INSTANTIATE(loss_flce_scalar_finalize_f32, float)
BARACUDA_KERNELS_FLCE_SCALAR_FINALIZE_INSTANTIATE(loss_flce_scalar_finalize_f16, __half)
BARACUDA_KERNELS_FLCE_SCALAR_FINALIZE_INSTANTIATE(loss_flce_scalar_finalize_bf16, __nv_bfloat16)
BARACUDA_KERNELS_FLCE_SCALAR_FINALIZE_INSTANTIATE(loss_flce_scalar_finalize_f64, double)

// In-place scale (BW: saved grad_* *= grad_output_scalar).
BARACUDA_KERNELS_FLCE_INPLACE_SCALE_INSTANTIATE(loss_flce_inplace_scale_f32, float)
BARACUDA_KERNELS_FLCE_INPLACE_SCALE_INSTANTIATE(loss_flce_inplace_scale_f16, __half)
BARACUDA_KERNELS_FLCE_INPLACE_SCALE_INSTANTIATE(loss_flce_inplace_scale_bf16, __nv_bfloat16)
BARACUDA_KERNELS_FLCE_INPLACE_SCALE_INSTANTIATE(loss_flce_inplace_scale_f64, double)

// Count non-ignore (single dtype: i64 target -> i64 count).
BARACUDA_KERNELS_FLCE_COUNT_NON_IGNORE_INSTANTIATE_BODY()
