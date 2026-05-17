// baracuda-kernels Phase 4 scan fanout: log_cumsum_exp FW for FP types.
//
// `y[k] = log(Σ_{j ≤ k} exp(x[j]))` (inclusive prefix LSE), or the
// suffix-LSE when `reverse != 0`. Numerically stable via the online
// running-max algorithm — see the kernel template in
// `baracuda_elementwise.cuh` for the full derivation.
//
// f32 / f64 compute natively; f16 / bf16 use a float accumulator
// throughout the per-thread walk and convert only at the final store.
// The single-rounding-at-store property matches every other f16 / bf16
// reduction / scan kernel in this crate.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_LOG_CUMSUM_EXP_INSTANTIATE(scan_log_cumsum_exp_f32, float)
BARACUDA_KERNELS_LOG_CUMSUM_EXP_INSTANTIATE(scan_log_cumsum_exp_f16, __half)
BARACUDA_KERNELS_LOG_CUMSUM_EXP_INSTANTIATE(scan_log_cumsum_exp_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOG_CUMSUM_EXP_INSTANTIATE(scan_log_cumsum_exp_f64, double)
