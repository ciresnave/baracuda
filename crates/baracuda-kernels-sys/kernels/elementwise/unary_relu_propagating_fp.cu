// baracuda-kernels: elementwise NaN-PROPAGATING ReLU for FP types.
//
// The sibling of `unary_relu_fp.cu` (which stays as the Fmax-family,
// NaN-scrubbing `fmaxf(x, 0)` semantics). This family implements
// `y = (x < 0 ? 0 : x)` — the torch.relu convention where a NaN input PASSES
// THROUGH (because `NaN < 0` is false), and a `-0.0` input stays `-0.0`.
//
// Fuel rebinds `OpKind::ReluElementwise` to THIS family (their 2026-07-08
// consolidated NaN-convention decision: ReluElementwise = NaN-propagating,
// torch parity). It is BIT-IDENTICAL to the baracuda-kernelgen generated relu
// (the semantics oracle — `crate::cuda` `UnaryOp::Relu` lowers to
// `x < 0.0f ? 0.0f : x` for f32/f64, and the f32-detour form for f16/bf16), so
// a JIT adopt that swaps the generated cell for this bespoke kernel is
// behaviorally identical on the advertised backend.
//
// f32 / f64 evaluate natively; f16 / bf16 use the SAME f32-detour the generated
// half path uses (`widen -> compare-select -> narrow`) so the bespoke matches
// the generated form bit-for-bit over the full 16-bit sweep (every NaN payload,
// +/-Inf, +/-0, every subnormal).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ReluPropagatingFunctor {
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct ReluPropagatingFunctor<float> {
    // `x < 0 ? 0 : x`: NaN passes (NaN < 0 is false), -0.0 stays -0.0. NOT
    // `fmaxf(x, 0)` (which would scrub NaN to 0 and normalize -0.0 to +0.0).
    __device__ __forceinline__ float operator()(float x) const {
        return x < 0.0f ? 0.0f : x;
    }
};

template <>
struct ReluPropagatingFunctor<double> {
    __device__ __forceinline__ double operator()(double x) const {
        return x < 0.0 ? 0.0 : x;
    }
};

template <>
struct ReluPropagatingFunctor<__half> {
    // f32-detour, matching the generated half relu bit-for-bit: widen to f32,
    // compare-select on the widened value, narrow. When x >= 0 or NaN the
    // returned value is the round-tripped `__float2half(__half2float(x))` —
    // identical to what the generated kernel produces.
    __device__ __forceinline__ __half operator()(__half x) const {
        float xf = __half2float(x);
        return __float2half(xf < 0.0f ? 0.0f : xf);
    }
};

template <>
struct ReluPropagatingFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float xf = __bfloat162float(x);
        return __float2bfloat16(xf < 0.0f ? 0.0f : xf);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations — contig + strided, f32 / f16 / bf16 / f64.
// =============================================================================

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu_propagating_f32,
    float,
    baracuda::elementwise::ReluPropagatingFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu_propagating_f32,
    float,
    baracuda::elementwise::ReluPropagatingFunctor<float>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu_propagating_f16,
    __half,
    baracuda::elementwise::ReluPropagatingFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu_propagating_f16,
    __half,
    baracuda::elementwise::ReluPropagatingFunctor<__half>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu_propagating_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ReluPropagatingFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu_propagating_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ReluPropagatingFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE(
    unary_relu_propagating_f64,
    double,
    baracuda::elementwise::ReluPropagatingFunctor<double>)

BARACUDA_KERNELS_UNARY_POINTWISE_INSTANTIATE_STRIDED(
    unary_relu_propagating_f64,
    double,
    baracuda::elementwise::ReluPropagatingFunctor<double>)
