// baracuda-kernels Phase 4 scan fanout: cumprod backward for FP types.
//
// `dx[j] = Σ_{i in suffix} dy[i] * y[i] / x[j]`. Suffix = `{i ≥ j}` for
// forward FW; `{i ≤ j}` for reverse FW. Caller must ensure x has no
// zeros along the scan axis. f32-detour accumulator for f16 / bf16.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_SCAN_CUMPROD_BACKWARD_INSTANTIATE(
    scan_cumprod_backward_f32, float, float)

BARACUDA_KERNELS_SCAN_CUMPROD_BACKWARD_INSTANTIATE(
    scan_cumprod_backward_f16, __half, float)

BARACUDA_KERNELS_SCAN_CUMPROD_BACKWARD_INSTANTIATE(
    scan_cumprod_backward_bf16, __nv_bfloat16, float)

BARACUDA_KERNELS_SCAN_CUMPROD_BACKWARD_INSTANTIATE(
    scan_cumprod_backward_f64, double, double)
