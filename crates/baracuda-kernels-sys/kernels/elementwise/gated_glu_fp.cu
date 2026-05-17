// baracuda-kernels Phase 3 Category C′ — GLU forward.
//
// Plan shape: split input `x` along `split_dim` into `(a, b)`; output
// `y = a · sigmoid(b)`. PyTorch `torch.nn.functional.glu`.
//
// f32 / f64 use direct intrinsic math; f16 / bf16 use the f32 detour.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct GluFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
};

template <>
struct GluFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        float s;
        if (b >= 0.0f) {
            s = 1.0f / (1.0f + expf(-b));
        } else {
            float e = expf(b);
            s = e / (1.0f + e);
        }
        return a * s;
    }
};

template <>
struct GluFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        double s;
        if (b >= 0.0) {
            s = 1.0 / (1.0 + exp(-b));
        } else {
            double e = exp(b);
            s = e / (1.0 + e);
        }
        return a * s;
    }
};

template <>
struct GluFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        float fa = __half2float(a);
        float fb = __half2float(b);
        float s;
        if (fb >= 0.0f) {
            s = 1.0f / (1.0f + expf(-fb));
        } else {
            float e = expf(fb);
            s = e / (1.0f + e);
        }
        return __float2half(fa * s);
    }
};

template <>
struct GluFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        float fa = __bfloat162float(a);
        float fb = __bfloat162float(b);
        float s;
        if (fb >= 0.0f) {
            s = 1.0f / (1.0f + expf(-fb));
        } else {
            float e = expf(fb);
            s = e / (1.0f + e);
        }
        return __float2bfloat16(fa * s);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_glu_f32, float,
    baracuda::elementwise::GluFunctor<float>)

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_glu_f16, __half,
    baracuda::elementwise::GluFunctor<__half>)

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_glu_bf16, __nv_bfloat16,
    baracuda::elementwise::GluFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_GATED_ACTIVATION_INSTANTIATE(
    gated_glu_f64, double,
    baracuda::elementwise::GluFunctor<double>)
