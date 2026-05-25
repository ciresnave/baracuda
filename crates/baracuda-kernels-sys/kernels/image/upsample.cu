// baracuda-kernels Phase 19.2 — upsample nearest-2D FW + BW.
//
// 4-fp-dtype instantiation set. Bilinear-2D variants intentionally NOT
// re-instantiated here — the existing `interpolate_bilinear_2d_*`
// symbols (kernels/image/interpolate.cu) remain authoritative; the
// upsample-named bilinear FFI is exposed as Rust aliases in
// crates/baracuda-kernels-sys/src/lib.rs.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../include/baracuda_upsample.cuh"

// ---------- Forward (4 fp dtypes) ----------

BARACUDA_KERNELS_UPSAMPLE_NEAREST_2D_FW_INSTANTIATE(upsample_nearest_2d_fw_f32, float)
BARACUDA_KERNELS_UPSAMPLE_NEAREST_2D_FW_INSTANTIATE(upsample_nearest_2d_fw_f64, double)
BARACUDA_KERNELS_UPSAMPLE_NEAREST_2D_FW_INSTANTIATE(upsample_nearest_2d_fw_f16, __half)
BARACUDA_KERNELS_UPSAMPLE_NEAREST_2D_FW_INSTANTIATE(upsample_nearest_2d_fw_bf16, __nv_bfloat16)

// ---------- Backward (4 fp dtypes; uses atomic::add<T> for f16/bf16) ----------

BARACUDA_KERNELS_UPSAMPLE_NEAREST_2D_BW_INSTANTIATE(upsample_nearest_2d_bw_f32, float)
BARACUDA_KERNELS_UPSAMPLE_NEAREST_2D_BW_INSTANTIATE(upsample_nearest_2d_bw_f64, double)
BARACUDA_KERNELS_UPSAMPLE_NEAREST_2D_BW_INSTANTIATE(upsample_nearest_2d_bw_f16, __half)
BARACUDA_KERNELS_UPSAMPLE_NEAREST_2D_BW_INSTANTIATE(upsample_nearest_2d_bw_bf16, __nv_bfloat16)
