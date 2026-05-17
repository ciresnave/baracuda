// baracuda-kernels Phase 4 reduce backward: max / min backward.
//
// Single kernel serves both Max BW and Min BW — the routing logic
// is identical, just compare `x[c]` to the saved forward output
// `y[c_reduced]` and pass `dy` through to matching positions.
//
// Tie semantic: every tied position receives the FULL upstream
// gradient (split-across-ties / JAX convention). PyTorch's first-
// index convention would need a saved argmax/argmin tensor — that's
// a future wave.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_REDUCE_MAX_MIN_BACKWARD_INSTANTIATE(reduce_max_min_backward_f32, float)

BARACUDA_KERNELS_REDUCE_MAX_MIN_BACKWARD_INSTANTIATE(reduce_max_min_backward_f16, __half)

BARACUDA_KERNELS_REDUCE_MAX_MIN_BACKWARD_INSTANTIATE(reduce_max_min_backward_bf16, __nv_bfloat16)

BARACUDA_KERNELS_REDUCE_MAX_MIN_BACKWARD_INSTANTIATE(reduce_max_min_backward_f64, double)
