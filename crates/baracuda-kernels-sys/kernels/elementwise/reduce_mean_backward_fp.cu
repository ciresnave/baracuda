// baracuda-kernels Phase 4 reduce backward: mean backward.
//
// Forward: `y = mean(x, dim=k) = sum(x, dim=k) / k_extent`. Backward:
// `dx[c] = dy[c with c[k] = 0] * (1 / k_extent)` — same broadcast as
// Sum BW with an extra `1/k` scale factor. The Rust dispatcher
// computes `1/k_extent` once on the host (in f64) and passes it
// through; the kernel casts to T at use.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_REDUCE_MEAN_BACKWARD_INSTANTIATE(reduce_mean_backward_f32, float)

BARACUDA_KERNELS_REDUCE_MEAN_BACKWARD_INSTANTIATE(reduce_mean_backward_f16, __half)

BARACUDA_KERNELS_REDUCE_MEAN_BACKWARD_INSTANTIATE(reduce_mean_backward_bf16, __nv_bfloat16)

BARACUDA_KERNELS_REDUCE_MEAN_BACKWARD_INSTANTIATE(reduce_mean_backward_f64, double)
