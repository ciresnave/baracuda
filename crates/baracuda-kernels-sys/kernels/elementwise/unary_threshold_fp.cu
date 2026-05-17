// baracuda-kernels Phase 3 deferral: elementwise threshold for FP types.
//
// Implements `y = (x > t) ? x : v` where `t` (threshold) and `v`
// (replacement value) are runtime scalar parameters threaded through the
// kernel ABI via the new `UNARY_PARAM_INSTANTIATE` macro. Pure
// compare-and-select — no arithmetic, so the result is bit-exact for the
// matched branch (`x` returned unchanged) and an exact round-trip of the
// f32 param `v` to T for the unmatched branch.
//
// f16 / bf16 do the compare in f32 (the param itself is f32), but return
// the original `x` bit-identically on the matched branch — no conversion
// hop. On the unmatched branch we convert the f32 param `v` to T once.
// f32 / f64 compare natively (f64 widens the f32 param `t` losslessly).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ThresholdFunctor {
    __device__ __forceinline__ T operator()(T x, float t, float v) const {
        // Default-template specialization unused — see explicit specs below.
        return (x > T(t)) ? x : T(v);
    }
};

template <>
struct ThresholdFunctor<float> {
    __device__ __forceinline__ float operator()(float x, float t, float v) const {
        return (x > t) ? x : v;
    }
};

template <>
struct ThresholdFunctor<double> {
    __device__ __forceinline__ double operator()(double x, float t, float v) const {
        // f64 input compares against the f32 param widened to f64
        // losslessly; replacement value also widens to f64.
        return (x > (double)t) ? x : (double)v;
    }
};

template <>
struct ThresholdFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half x, float t, float v) const {
        float fx = __half2float(x);
        // Compare in f32; on the matched branch return the original
        // `x` bits unchanged — no conversion hop.
        return (fx > t) ? x : __float2half(v);
    }
};

template <>
struct ThresholdFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x, float t, float v) const {
        float fx = __bfloat162float(x);
        return (fx > t) ? x : __float2bfloat16(v);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE(
    unary_threshold_f32,
    float,
    baracuda::elementwise::ThresholdFunctor<float>)

BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE(
    unary_threshold_f16,
    __half,
    baracuda::elementwise::ThresholdFunctor<__half>)

BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE(
    unary_threshold_bf16,
    __nv_bfloat16,
    baracuda::elementwise::ThresholdFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_PARAM_INSTANTIATE(
    unary_threshold_f64,
    double,
    baracuda::elementwise::ThresholdFunctor<double>)
