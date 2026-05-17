// baracuda-kernels Phase 4 reduce backward trailblazer: sum backward.
//
// Forward: `y = sum(x, dim=k)` (keepdim=true convention; y.shape[k] = 1).
// Backward: `dx[c] = dy[c with c[k] = 0]` — broadcast dy across the
// reduced axis. The Rust dispatcher sets `stride_dy[reduce_axis] = 0`
// before calling, so the kernel just does a strided copy.
//
// Trivial generic-on-T kernel — no transcendentals, no math beyond
// addressing. Covers all four FP dtypes via straight instantiation.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_REDUCE_SUM_BACKWARD_INSTANTIATE(reduce_sum_backward_f32, float)

BARACUDA_KERNELS_REDUCE_SUM_BACKWARD_INSTANTIATE(reduce_sum_backward_f16, __half)

BARACUDA_KERNELS_REDUCE_SUM_BACKWARD_INSTANTIATE(reduce_sum_backward_bf16, __nv_bfloat16)

BARACUDA_KERNELS_REDUCE_SUM_BACKWARD_INSTANTIATE(reduce_sum_backward_f64, double)
