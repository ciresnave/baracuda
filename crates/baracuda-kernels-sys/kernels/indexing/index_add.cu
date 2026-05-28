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

// Phase 40 (Fuel 6c.4 Gap 6b spillover) — integer value-dtype fanout
// for `index_add` (atomicAdd-Σ accumulation).
//
// Coverage limited to value dtypes with native CUDA `atomicAdd`
// support on baracuda's sm_80+ baseline:
//   * `int32_t` (native `atomicAdd(int*, int)`)
//   * `uint32_t` (native `atomicAdd(unsigned int*, unsigned int)`)
//   * `int64_t` (reinterpret as `unsigned long long*`; see new spec
//     in `baracuda_atomic.cuh`)
// = 3 values × 2 idx = 6 symbols.
//
// u8 / i8 / u16 / i16 would need a 32-bit CAS-loop similar to the
// half / bf16 path; deferred until a clear caller materializes.
BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_i32, int32_t,  int32_t)
BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_u32, uint32_t, int32_t)
BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_i64, int64_t,  int32_t)

BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_i64idx_i32, int32_t,  int64_t)
BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_i64idx_u32, uint32_t, int64_t)
BARACUDA_KERNELS_INDEX_ADD_INSTANTIATE(index_add_i64idx_i64, int64_t,  int64_t)
