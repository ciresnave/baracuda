// baracuda-kernels Phase 4: argmax / argmin axis reductions.
//
// `y = argmax(x, axis=k)` — returns the i64 index of the max along
// axis k. Output dtype is i64 (PyTorch convention). Output shape ==
// input shape with `[reduce_axis]` collapsed to 1.
//
// Ties broken by FIRST occurrence (smallest index wins) — matches
// PyTorch. The kernel walks the reduce axis from k=0 upward,
// replacing best only on a strict prefer outcome.
//
// Today only f32 is wired (the trailblazer); other dtypes follow as
// single-INSTANTIATE fanout. The kernel template is fully generic in T.

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
// Instantiations
// =============================================================================

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f32, float, baracuda::elementwise::ArgmaxPolicy<float>)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f32, float, baracuda::elementwise::ArgminPolicy<float>)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f16, __half, baracuda::elementwise::ArgmaxPolicy<__half>)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f16, __half, baracuda::elementwise::ArgminPolicy<__half>)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_bf16, __nv_bfloat16, baracuda::elementwise::ArgmaxPolicy<__nv_bfloat16>)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_bf16, __nv_bfloat16, baracuda::elementwise::ArgminPolicy<__nv_bfloat16>)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmax_f64, double, baracuda::elementwise::ArgmaxPolicy<double>)

BARACUDA_KERNELS_ARG_REDUCE_AXIS_INSTANTIATE(
    arg_reduce_argmin_f64, double, baracuda::elementwise::ArgminPolicy<double>)
