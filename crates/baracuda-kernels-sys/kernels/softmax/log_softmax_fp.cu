// baracuda-kernels Phase 5 fanout: log-softmax FW for FP types.
//
// `y[k] = (x[k] - max(x)) - log(Σ_j exp(x[j] - max(x)))` along a single
// axis. Numerically stable max-subtraction (same technique as Softmax).
// f16 / bf16 accumulate in f32; f32 / f64 use native precision.

#include "../include/baracuda_softmax.cuh"

BARACUDA_KERNELS_LOG_SOFTMAX_INSTANTIATE(log_softmax_f32, float)
BARACUDA_KERNELS_LOG_SOFTMAX_INSTANTIATE(log_softmax_f16, __half)
BARACUDA_KERNELS_LOG_SOFTMAX_INSTANTIATE(log_softmax_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOG_SOFTMAX_INSTANTIATE(log_softmax_f64, double)

// Phase 72 strided-sibling FFI exports — same underlying launcher.
BARACUDA_KERNELS_LOG_SOFTMAX_INSTANTIATE_STRIDED(log_softmax_f32, float)
BARACUDA_KERNELS_LOG_SOFTMAX_INSTANTIATE_STRIDED(log_softmax_f16, __half)
BARACUDA_KERNELS_LOG_SOFTMAX_INSTANTIATE_STRIDED(log_softmax_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOG_SOFTMAX_INSTANTIATE_STRIDED(log_softmax_f64, double)
