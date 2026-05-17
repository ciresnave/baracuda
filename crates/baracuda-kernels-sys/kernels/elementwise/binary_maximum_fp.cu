// baracuda-kernels Phase 3 binary fanout: elementwise maximum
// `y = max(a, b)` — IEEE 754 NaN-PROPAGATING semantics. Any NaN input
// produces a NaN output, matching `torch.maximum`. Distinct from
// `BinaryKind::Fmax`, which uses the NaN-aware (NaN-ignored) libdevice
// `fmaxf` / `fmax`.
//
// The implementation explicitly checks `isnan` on each operand and
// returns it, so the bit-pattern of the NaN that propagates is the
// operand's own (no quieting through the `>` compare-and-select).
//
// f32 → bare compare-and-select with NaN guards; f64 → same. f16 /
// bf16 → f32-detour pattern matching the unary transcendental family.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct MaximumFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct MaximumFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        // NaN-propagating: if either input is NaN, return that NaN.
        // `x != x` is true iff x is NaN (no `isnan` device header dep).
        if (a != a) return a;
        if (b != b) return b;
        return (a > b) ? a : b;
    }
};

template <>
struct MaximumFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        if (a != a) return a;
        if (b != b) return b;
        return (a > b) ? a : b;
    }
};

template <>
struct MaximumFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        float af = __half2float(a);
        float bf = __half2float(b);
        if (af != af) return a;
        if (bf != bf) return b;
        return __float2half((af > bf) ? af : bf);
    }
};

template <>
struct MaximumFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        float af = __bfloat162float(a);
        float bf = __bfloat162float(b);
        if (af != af) return a;
        if (bf != bf) return b;
        return __float2bfloat16((af > bf) ? af : bf);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_maximum_f32, float, baracuda::elementwise::MaximumFunctor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_maximum_f32, float, baracuda::elementwise::MaximumFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_maximum_f16, __half, baracuda::elementwise::MaximumFunctor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_maximum_f16, __half, baracuda::elementwise::MaximumFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_maximum_bf16, __nv_bfloat16, baracuda::elementwise::MaximumFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_maximum_bf16, __nv_bfloat16, baracuda::elementwise::MaximumFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_maximum_f64, double, baracuda::elementwise::MaximumFunctor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_maximum_f64, double, baracuda::elementwise::MaximumFunctor<double>)
