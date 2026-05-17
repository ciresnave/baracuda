// baracuda-kernels Phase 3 binary fanout: elementwise minimum
// `y = min(a, b)` — IEEE 754 NaN-PROPAGATING semantics. Any NaN input
// produces a NaN output, matching `torch.minimum`. Distinct from
// `BinaryKind::Fmin`, which uses the NaN-aware (NaN-ignored) libdevice
// `fminf` / `fmin`.
//
// The implementation explicitly checks `isnan` on each operand and
// returns it, so the bit-pattern of the NaN that propagates is the
// operand's own (no quieting through the `<` compare-and-select).
//
// f32 → bare compare-and-select with NaN guards; f64 → same. f16 /
// bf16 → f32-detour pattern matching the unary transcendental family.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct MinimumFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct MinimumFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        if (a != a) return a;
        if (b != b) return b;
        return (a < b) ? a : b;
    }
};

template <>
struct MinimumFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        if (a != a) return a;
        if (b != b) return b;
        return (a < b) ? a : b;
    }
};

template <>
struct MinimumFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        float af = __half2float(a);
        float bf = __half2float(b);
        if (af != af) return a;
        if (bf != bf) return b;
        return __float2half((af < bf) ? af : bf);
    }
};

template <>
struct MinimumFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        float af = __bfloat162float(a);
        float bf = __bfloat162float(b);
        if (af != af) return a;
        if (bf != bf) return b;
        return __float2bfloat16((af < bf) ? af : bf);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_minimum_f32, float, baracuda::elementwise::MinimumFunctor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_minimum_f32, float, baracuda::elementwise::MinimumFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_minimum_f16, __half, baracuda::elementwise::MinimumFunctor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_minimum_f16, __half, baracuda::elementwise::MinimumFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_minimum_bf16, __nv_bfloat16, baracuda::elementwise::MinimumFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_minimum_bf16, __nv_bfloat16, baracuda::elementwise::MinimumFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_minimum_f64, double, baracuda::elementwise::MinimumFunctor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_minimum_f64, double, baracuda::elementwise::MinimumFunctor<double>)
