// baracuda-kernels Phase 9 Category O — topk FW + BW.
//
// Trailblazer dtype coverage: f32, f64 (FP only — int top-k deferred,
// not a common ML inference dependency).
// Trailblazer limits: row_len ≤ 1024, k ≤ 64 (LLM-inference range).

#include "../include/baracuda_topk.cuh"

BARACUDA_KERNELS_TOPK_INSTANTIATE(topk_f32, float)
BARACUDA_KERNELS_TOPK_INSTANTIATE(topk_f64, double)

BARACUDA_KERNELS_TOPK_BACKWARD_INSTANTIATE(topk_backward_f32, float)
BARACUDA_KERNELS_TOPK_BACKWARD_INSTANTIATE(topk_backward_f64, double)
