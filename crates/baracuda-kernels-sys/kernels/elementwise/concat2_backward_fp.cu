// baracuda-kernels Phase 3 BW for Category N: concat2 backward
// (pure slice-split). `da = dy[..., :split_offset, ...]` and
// `db = dy[..., split_offset:, ...]` along `concat_dim`. Every dy cell
// maps to exactly one of `da` or `db` — bit-exact, no arithmetic.

#include "../include/baracuda_elementwise.cuh"

BARACUDA_KERNELS_CONCAT2_BACKWARD_INSTANTIATE(concat2_backward_f32, float)
BARACUDA_KERNELS_CONCAT2_BACKWARD_INSTANTIATE(concat2_backward_f16, __half)
BARACUDA_KERNELS_CONCAT2_BACKWARD_INSTANTIATE(concat2_backward_bf16, __nv_bfloat16)
BARACUDA_KERNELS_CONCAT2_BACKWARD_INSTANTIATE(concat2_backward_f64, double)
