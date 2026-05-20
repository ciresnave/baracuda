// Phase 13.4 — Triu / Tril (upper / lower triangular matrix masks).
//
// `torch.triu(input, diagonal)` keeps elements at output[..., i, j]
// where `j >= i + diagonal`, zeroing the rest. `torch.tril` is the
// mirror with `j <= i + diagonal`. Both ops operate on the last two
// dimensions; the batch prefix is masked independently.
//
// One templated kernel body covers both ops via a Predicate functor;
// per-dtype instantiations land here. Coverage matches the rest of the
// shape-layout family that touches numeric arithmetic-free copies:
// {f16, bf16, f32, f64, i32, i64, Bool}.
//
// Driven by Fuel team's CPU-only triu/tril gap.

#include "../include/baracuda_triu_tril.cuh"

// Triu fanout — 7 dtypes.
BARACUDA_KERNELS_TRIU_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_TRIU_INSTANTIATE(bf16, __nv_bfloat16)
BARACUDA_KERNELS_TRIU_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_TRIU_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_TRIU_INSTANTIATE(i32,  int32_t)
BARACUDA_KERNELS_TRIU_INSTANTIATE(i64,  int64_t)
BARACUDA_KERNELS_TRIU_INSTANTIATE(bool, uint8_t)

// Tril fanout — 7 dtypes.
BARACUDA_KERNELS_TRIL_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_TRIL_INSTANTIATE(bf16, __nv_bfloat16)
BARACUDA_KERNELS_TRIL_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_TRIL_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_TRIL_INSTANTIATE(i32,  int32_t)
BARACUDA_KERNELS_TRIL_INSTANTIATE(i64,  int64_t)
BARACUDA_KERNELS_TRIL_INSTANTIATE(bool, uint8_t)
