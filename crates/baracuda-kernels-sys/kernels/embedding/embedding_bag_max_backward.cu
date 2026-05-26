// baracuda-kernels Phase 25 — `embedding_bag` Max mode BW.
//
// Sparse scatter: for each (b, d) with `out_index[b, d] >= 0`,
// `atomicAdd(dweight[out_index[b, d], d], dout[b, d])`. Index dtype
// in the contributing-row buffer is fixed at i32 (the FW always
// writes i32) — generic over the **value** dtype (T) only.
//
// Dtype coverage: f32, f64 (atomicAdd is native-FP).

#include "../include/baracuda_embedding.cuh"

BARACUDA_KERNELS_EMBEDDING_BAG_MAX_BACKWARD_INSTANTIATE(embedding_bag_max_backward_f32, float)
BARACUDA_KERNELS_EMBEDDING_BAG_MAX_BACKWARD_INSTANTIATE(embedding_bag_max_backward_f64, double)
