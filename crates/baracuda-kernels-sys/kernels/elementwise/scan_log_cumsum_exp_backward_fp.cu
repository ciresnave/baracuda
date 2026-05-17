// baracuda-kernels Phase 4 scan fanout: log_cumsum_exp backward for FP
// types.
//
// `dx[k] = Σ_{i ∈ range(k)} dy[i] * exp(x[k] - y[i])`. Range is
// `[k, extent)` for forward FW (`reverse == 0`) and `[0, k]` for reverse
// FW. Needs both saved FW input `x` and saved FW output `y` — same
// shape (scans are length-preserving). Each `x[k] - y[i]` is ≤ 0 by
// construction (y[i] is the LSE of a window containing x[k]), so `exp`
// stays in `[0, 1]` and no extra max-tracking is needed in BW.
//
// f16 / bf16 use a float accumulator (every load / op in float; final
// store converts back). f64 stays in double.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_LOG_CUMSUM_EXP_BACKWARD_INSTANTIATE(
    scan_log_cumsum_exp_backward_f32, float)

BARACUDA_KERNELS_LOG_CUMSUM_EXP_BACKWARD_INSTANTIATE(
    scan_log_cumsum_exp_backward_f16, __half)

BARACUDA_KERNELS_LOG_CUMSUM_EXP_BACKWARD_INSTANTIATE(
    scan_log_cumsum_exp_backward_bf16, __nv_bfloat16)

BARACUDA_KERNELS_LOG_CUMSUM_EXP_BACKWARD_INSTANTIATE(
    scan_log_cumsum_exp_backward_f64, double)
