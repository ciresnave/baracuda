// baracuda-kernels Milestone 5.4: Sparsemax FW for FP types.
//
// `y = ProjSimplex(x)` via the standard sort-then-threshold algorithm.
// Per-thread serial insertion sort in local memory; row extent limited
// to BARACUDA_SPARSEMAX_MAX_EXTENT (64) for the trailblazer. Future
// tuning can use a cooperative block-wide bitonic sort to lift the cap.

#include "../include/baracuda_softmax.cuh"

BARACUDA_KERNELS_SPARSEMAX_INSTANTIATE(sparsemax_f32, float)
BARACUDA_KERNELS_SPARSEMAX_INSTANTIATE(sparsemax_f16, __half)
BARACUDA_KERNELS_SPARSEMAX_INSTANTIATE(sparsemax_bf16, __nv_bfloat16)
BARACUDA_KERNELS_SPARSEMAX_INSTANTIATE(sparsemax_f64, double)
