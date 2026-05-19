// baracuda-kernels Phase 4: argmax / argmin axis reductions.
//
// `y = argmax(x, axis=k)` — returns the index of the max along axis k.
// Output shape == input shape with `[reduce_axis]` collapsed to 1.
//
// Ties broken by FIRST occurrence (smallest index wins) — matches
// PyTorch. The kernel walks the reduce axis from k=0 upward,
// replacing best only on a strict prefer outcome.
//
// Phase 12.2 (Fuel team feedback): output dtype generalized from a
// hard-coded `int64_t` to one of `{int64_t, uint32_t, int32_t}`.
// PyTorch defaults to `int64_t`; the legacy `_f32 / _f16 / _bf16 / _f64`
// symbols stay at `int64_t` for back-compat. New `_u32 / _i32` suffix
// variants narrow the output store. Internal best-index tracking stays
// `int64_t` (the reduce-axis extent is i64-safe); only the final store
// narrows.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Argmax policy: prefer new value if STRICTLY greater than current best.
// Equal values keep the earlier (smaller) index — matches PyTorch.
template <typename T>
struct ArgmaxPolicy {
    __device__ __forceinline__ bool prefer(
        T new_v, int64_t /*new_i*/, T best_v, int64_t /*best_i*/) const
    {
        return new_v > best_v;
    }
};

// Argmin policy.
template <typename T>
struct ArgminPolicy {
    __device__ __forceinline__ bool prefer(
        T new_v, int64_t /*new_i*/, T best_v, int64_t /*best_i*/) const
    {
        return new_v < best_v;
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations — 4 value dtypes × 3 output dtypes × 2 ops = 24 total.
// Legacy i64-output symbols (the 8 from Phase 4) keep their original
// names to preserve back-compat; new u32 / i32 outputs use the
// `_u32 / _i32` suffix.
// =============================================================================

// -----------------------------------------------------------------------------
// i64 output (legacy / default — PyTorch convention).
// -----------------------------------------------------------------------------
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f32, float, baracuda::elementwise::ArgmaxPolicy<float>, int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f32, float, baracuda::elementwise::ArgminPolicy<float>, int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f16, __half, baracuda::elementwise::ArgmaxPolicy<__half>, int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f16, __half, baracuda::elementwise::ArgminPolicy<__half>, int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_bf16, __nv_bfloat16, baracuda::elementwise::ArgmaxPolicy<__nv_bfloat16>, int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_bf16, __nv_bfloat16, baracuda::elementwise::ArgminPolicy<__nv_bfloat16>, int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f64, double, baracuda::elementwise::ArgmaxPolicy<double>, int64_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f64, double, baracuda::elementwise::ArgminPolicy<double>, int64_t)

// -----------------------------------------------------------------------------
// u32 output (Phase 12.2 — Fuel team preferred).
// -----------------------------------------------------------------------------
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f32_u32, float, baracuda::elementwise::ArgmaxPolicy<float>, uint32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f32_u32, float, baracuda::elementwise::ArgminPolicy<float>, uint32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f16_u32, __half, baracuda::elementwise::ArgmaxPolicy<__half>, uint32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f16_u32, __half, baracuda::elementwise::ArgminPolicy<__half>, uint32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_bf16_u32, __nv_bfloat16, baracuda::elementwise::ArgmaxPolicy<__nv_bfloat16>, uint32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_bf16_u32, __nv_bfloat16, baracuda::elementwise::ArgminPolicy<__nv_bfloat16>, uint32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f64_u32, double, baracuda::elementwise::ArgmaxPolicy<double>, uint32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f64_u32, double, baracuda::elementwise::ArgminPolicy<double>, uint32_t)

// -----------------------------------------------------------------------------
// i32 output (Phase 12.2).
// -----------------------------------------------------------------------------
BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f32_i32, float, baracuda::elementwise::ArgmaxPolicy<float>, int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f32_i32, float, baracuda::elementwise::ArgminPolicy<float>, int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f16_i32, __half, baracuda::elementwise::ArgmaxPolicy<__half>, int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f16_i32, __half, baracuda::elementwise::ArgminPolicy<__half>, int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_bf16_i32, __nv_bfloat16, baracuda::elementwise::ArgmaxPolicy<__nv_bfloat16>, int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_bf16_i32, __nv_bfloat16, baracuda::elementwise::ArgminPolicy<__nv_bfloat16>, int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f64_i32, double, baracuda::elementwise::ArgmaxPolicy<double>, int32_t)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f64_i32, double, baracuda::elementwise::ArgminPolicy<double>, int32_t)
