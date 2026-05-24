// Phase 16.3 — FractionalMaxPool 2-D + 3-D (FW + BW × {f16, bf16, f32, f64}).
//
// Replaces the Phase 11.8 stubs. See
// `../include/baracuda_fractional_max_pool.cuh` for the algorithm /
// window-placement formula divergence vs PyTorch and the random-samples
// ABI.

#include "../include/baracuda_fractional_max_pool.cuh"

// 2-D forward — 4 FP dtypes.
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_2D_FW_INSTANTIATE(
    fractional_max_pool_2d_fw_f32, float)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_2D_FW_INSTANTIATE(
    fractional_max_pool_2d_fw_f64, double)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_2D_FW_INSTANTIATE(
    fractional_max_pool_2d_fw_f16, __half)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_2D_FW_INSTANTIATE(
    fractional_max_pool_2d_fw_bf16, __nv_bfloat16)

// 2-D backward — 4 FP dtypes (half / bf16 atomicAdd routes through CAS).
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_2D_BW_INSTANTIATE(
    fractional_max_pool_2d_bw_f32, float)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_2D_BW_INSTANTIATE(
    fractional_max_pool_2d_bw_f64, double)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_2D_BW_INSTANTIATE(
    fractional_max_pool_2d_bw_f16, __half)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_2D_BW_INSTANTIATE(
    fractional_max_pool_2d_bw_bf16, __nv_bfloat16)

// 3-D forward — 4 FP dtypes.
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_3D_FW_INSTANTIATE(
    fractional_max_pool_3d_fw_f32, float)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_3D_FW_INSTANTIATE(
    fractional_max_pool_3d_fw_f64, double)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_3D_FW_INSTANTIATE(
    fractional_max_pool_3d_fw_f16, __half)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_3D_FW_INSTANTIATE(
    fractional_max_pool_3d_fw_bf16, __nv_bfloat16)

// 3-D backward — 4 FP dtypes.
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_3D_BW_INSTANTIATE(
    fractional_max_pool_3d_bw_f32, float)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_3D_BW_INSTANTIATE(
    fractional_max_pool_3d_bw_f64, double)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_3D_BW_INSTANTIATE(
    fractional_max_pool_3d_bw_f16, __half)
BARACUDA_KERNELS_FRACTIONAL_MAX_POOL_3D_BW_INSTANTIATE(
    fractional_max_pool_3d_bw_bf16, __nv_bfloat16)
