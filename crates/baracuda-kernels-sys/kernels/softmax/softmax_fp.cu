// baracuda-kernels Phase 5 trailblazer: softmax FW for FP types.
//
// `y[k] = exp(x[k] - max(x)) / Σ_j exp(x[j] - max(x))` along a single
// axis. Numerically stable max-subtraction. f16 / bf16 accumulate in
// f32; f32 / f64 use native precision throughout.

#include "../include/baracuda_softmax.cuh"

BARACUDA_KERNELS_SOFTMAX_INSTANTIATE(softmax_f32, float)
BARACUDA_KERNELS_SOFTMAX_INSTANTIATE(softmax_f16, __half)
BARACUDA_KERNELS_SOFTMAX_INSTANTIATE(softmax_bf16, __nv_bfloat16)
BARACUDA_KERNELS_SOFTMAX_INSTANTIATE(softmax_f64, double)
