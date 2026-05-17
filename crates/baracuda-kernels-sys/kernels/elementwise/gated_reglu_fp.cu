// baracuda-kernels Phase 3 Category C′ — ReGLU forward.
//
// Plan shape: split input `x` along `split_dim` into `(a, b)`; output
// `y = a · relu(b) = a · max(b, 0)`. Bit-exact at f32 / f64 (pure
// mul-or-zero, no transcendental). f16 / bf16 use the f32 detour for
// uniform dispatch (1 ULP from final round).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct RegluFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
};

template <>
struct RegluFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return (b > 0.0f) ? (a * b) : 0.0f;
    }
};

template <>
struct RegluFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        return (b > 0.0) ? (a * b) : 0.0;
    }
};

template <>
struct RegluFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        float fa = __half2float(a);
        float fb = __half2float(b);
        float y = (fb > 0.0f) ? (fa * fb) : 0.0f;
        return __float2half(y);
    }
};

template <>
struct RegluFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        float fa = __bfloat162float(a);
        float fb = __bfloat162float(b);
        float y = (fb > 0.0f) ? (fa * fb) : 0.0f;
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_reglu_f32, float,
    baracuda::elementwise::RegluFunctor<float>)

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_reglu_f16, __half,
    baracuda::elementwise::RegluFunctor<__half>)

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_reglu_bf16, __nv_bfloat16,
    baracuda::elementwise::RegluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_reglu_f64, double,
    baracuda::elementwise::RegluFunctor<double>)
