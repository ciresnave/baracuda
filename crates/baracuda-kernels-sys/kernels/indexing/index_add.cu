// baracuda-kernels Phase 39 (Fuel 6c.4 Gap 5) — `index_add` FW.
//
// `dst[idx[i], ...] += src[i, ...]` along `add_dim` (atomicAdd-Σ
// accumulation, dup-safe). 1-D `idx` tensor; algorithm matches
// `index_select_backward` but exposed under a non-autograd-flavored
// name + with f16 / bf16 dtype fanout that `index_select_backward`
// lacks. f32 / f64 are deliberately re-exposed here so callers can
// use one consistent name across the whole dtype matrix.
//
// FFI surface: 8 symbols = {f32, f64, f16, bf16} × {i32 idx, i64 idx}.

#include "../include/baracuda_indexing.cuh"

BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_f32,  float,          int32_t)
BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_f64,  double,         int32_t)
BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_f16,  __half,         int32_t)
BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_bf16, __nv_bfloat16,  int32_t)

BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_i64idx_f32,  float,          int64_t)
BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_i64idx_f64,  double,         int64_t)
BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_i64idx_f16,  __half,         int64_t)
BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_i64idx_bf16, __nv_bfloat16,  int64_t)
