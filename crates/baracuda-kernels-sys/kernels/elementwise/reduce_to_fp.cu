// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Phase 31 — instantiations for `reduce_sum_to` and `reduce_max_to`
// (broadcast-REVERSE reductions; the autograd primitive that undoes a
// forward `BroadcastTo`).
//
// Phase 37 — fanout extended to `reduce_min_to` and `reduce_prod_to`
// across the same 4 fp dtypes. Min's identity is `+FLT_MAX` (narrows
// to `+inf` in half-precision dst on empty broadcast sets, matching
// the per-axis `MinReduce`'s `+INFINITY` initial); Prod's identity is
// `1`.
//
// Coverage: f32 / f64 / f16 / bf16. f16 / bf16 dst is written with the
// same accumulator-in-float convention the rest of the family uses.

#include "../include/baracuda_reduce_to.cuh"

BARACUDA_KERNELS_REDUCE_SUM_TO_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_REDUCE_SUM_TO_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_REDUCE_SUM_TO_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_REDUCE_SUM_TO_INSTANTIATE(bf16, __nv_bfloat16)

BARACUDA_KERNELS_REDUCE_MAX_TO_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_REDUCE_MAX_TO_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_REDUCE_MAX_TO_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_REDUCE_MAX_TO_INSTANTIATE(bf16, __nv_bfloat16)

BARACUDA_KERNELS_REDUCE_MIN_TO_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_REDUCE_MIN_TO_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_REDUCE_MIN_TO_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_REDUCE_MIN_TO_INSTANTIATE(bf16, __nv_bfloat16)

BARACUDA_KERNELS_REDUCE_PROD_TO_INSTANTIATE(f32,  float)
BARACUDA_KERNELS_REDUCE_PROD_TO_INSTANTIATE(f64,  double)
BARACUDA_KERNELS_REDUCE_PROD_TO_INSTANTIATE(f16,  __half)
BARACUDA_KERNELS_REDUCE_PROD_TO_INSTANTIATE(bf16, __nv_bfloat16)
