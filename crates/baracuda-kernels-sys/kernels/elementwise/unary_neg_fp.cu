// baracuda-kernels Phase 3 unary trailblazer: elementwise negation for
// FP types.
//
// Implements `y = -x` over both contiguous tensors (fast path) and
// arbitrary strided views (transposed / sliced — the same strided
// kernel template that the binary path uses, specialized for unary
// arity). The kernel templates and INSTANTIATE macros live in
// `include/baracuda_elementwise.cuh`; this file supplies the
// `NegFunctor<T>` and the per-dtype instantiations.
//
// All four FP dtypes are wired — f32 was the trailblazer; f16 / bf16 /
// f64 followed in the unary-fanout session. The functor is templated
// on T so all four dtypes share one definition (`-x` is a single PTX
// neg.f* instruction per element on every FP dtype).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Unary negation. `-x` is well-defined for every FP type; the GPU
// hardware emits a single neg instruction (PTX `neg.f32` / `neg.f16x2`
// / etc.) so this is a one-cycle op per element on top of the gmem
// bandwidth.
template <typename T>
struct NegFunctor {
    __device__ __forceinline__ T operator()(T x) const { return -x; }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_neg_f32,
    float,
    baracuda::elementwise::NegFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_neg_f32,
    float,
    baracuda::elementwise::NegFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_neg_f16,
    __half,
    baracuda::elementwise::NegFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_neg_f16,
    __half,
    baracuda::elementwise::NegFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_neg_bf16,
    __nv_bfloat16,
    baracuda::elementwise::NegFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_neg_bf16,
    __nv_bfloat16,
    baracuda::elementwise::NegFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_neg_f64,
    double,
    baracuda::elementwise::NegFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_neg_f64,
    double,
    baracuda::elementwise::NegFunctor<double>)
