// baracuda-kernels Phase 3 binary fanout: elementwise hypot `y = sqrt(a² + b²)`.
//
// f32 → `hypotf`, f64 → `hypot`, f16 / bf16 → f32-detour pattern matching
// the unary transcendental family. libdevice `hypotf` / `hypot` are
// overflow-/underflow-safe (they internally scale by max(|a|, |b|))
// which matches PyTorch's `torch.hypot` semantics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct HypotFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct HypotFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return hypotf(a, b);
    }
};

template <>
struct HypotFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        return hypot(a, b);
    }
};

template <>
struct HypotFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __float2half(hypotf(__half2float(a), __half2float(b)));
    }
};

template <>
struct HypotFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __float2bfloat16(hypotf(__bfloat162float(a), __bfloat162float(b)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_hypot_f32, float, baracuda::elementwise::HypotFunctor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_hypot_f32, float, baracuda::elementwise::HypotFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_hypot_f16, __half, baracuda::elementwise::HypotFunctor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_hypot_f16, __half, baracuda::elementwise::HypotFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_hypot_bf16, __nv_bfloat16, baracuda::elementwise::HypotFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_hypot_bf16, __nv_bfloat16, baracuda::elementwise::HypotFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_hypot_f64, double, baracuda::elementwise::HypotFunctor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_hypot_f64, double, baracuda::elementwise::HypotFunctor<double>)
