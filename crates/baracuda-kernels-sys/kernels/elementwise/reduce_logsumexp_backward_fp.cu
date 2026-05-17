// baracuda-kernels Phase 4 reduce backward: LogSumExp backward.
//
// Forward: `y = log(sum(exp(x - max), axis=k)) + max` (keepdim).
// Backward (softmax identity):
//   `dx[c] = dy[c_reduced] * exp(x[c] - y[c_reduced])`
// Numerically safe at all dtypes: `y = lse(x) ≥ max(x) ≥ x[c]`, so
// `x - y ∈ (-∞, 0]` and `exp(x - y) ∈ (0, 1]`. No overflow, only an
// underflow toward zero for far-from-max coordinates — which is the
// correct gradient.
//
// Requires BOTH saved `x` (full shape) and saved `y` (keepdim shape).
// f16 / bf16 detour through f32 for the exp; f32 / f64 use libdevice
// `expf` / `exp` directly.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_REDUCE_LOGSUMEXP_BACKWARD_INSTANTIATE(reduce_logsumexp_backward_f32, float)
BARACUDA_KERNELS_REDUCE_LOGSUMEXP_BACKWARD_INSTANTIATE(reduce_logsumexp_backward_f16, __half)
BARACUDA_KERNELS_REDUCE_LOGSUMEXP_BACKWARD_INSTANTIATE(reduce_logsumexp_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_REDUCE_LOGSUMEXP_BACKWARD_INSTANTIATE(reduce_logsumexp_backward_f64, double)
