// baracuda-kernels Phase 5 fanout: log-softmax BW for FP types.
//
// `dx[k] = dy[k] - exp(y[k]) · Σ_j dy[j]` along the softmax axis,
// where `y` is the saved forward log-softmax output (so `exp(y)`
// recovers `softmax(x)` and lives in `[0, 1]`). Per thread: walks the
// softmax axis once to compute the row-sum `dy_sum`, then applies the
// per-cell BW formula. f16 / bf16 accumulate in f32.

#include "../include/baracuda_softmax.cuh"

BARACUDA_KERNELS_LOG_SOFTMAX_BACKWARD_INSTANTIATE(log_softmax_backward_f32, float)
BARACUDA_KERNELS_LOG_SOFTMAX_BACKWARD_INSTANTIATE(log_softmax_backward_f16, __half)
BARACUDA_KERNELS_LOG_SOFTMAX_BACKWARD_INSTANTIATE(log_softmax_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOG_SOFTMAX_BACKWARD_INSTANTIATE(log_softmax_backward_f64, double)

// Phase 72 strided-sibling FFI exports — same underlying launcher.
BARACUDA_KERNELS_LOG_SOFTMAX_BACKWARD_INSTANTIATE_STRIDED(log_softmax_backward_f32, float)
BARACUDA_KERNELS_LOG_SOFTMAX_BACKWARD_INSTANTIATE_STRIDED(log_softmax_backward_f16, __half)
BARACUDA_KERNELS_LOG_SOFTMAX_BACKWARD_INSTANTIATE_STRIDED(log_softmax_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_LOG_SOFTMAX_BACKWARD_INSTANTIATE_STRIDED(log_softmax_backward_f64, double)
