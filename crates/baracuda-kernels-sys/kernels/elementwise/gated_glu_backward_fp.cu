// baracuda-kernels Phase 3 Category C′ — GLU backward.
//
// Forward: `y = a · sigmoid(b)`. Backward (saved `x`):
//   da = dy · sigmoid(b)
//   db = dy · a · sigmoid(b) · (1 - sigmoid(b))

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct GluBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T a, T b, T& da_out, T& db_out) const {
        da_out = dy * b;
        db_out = dy * a;
    }
};

template <>
struct GluBackwardFunctor<float> {
    __device__ __forceinline__ void operator()(float dy, float a, float b,
                                                float& da_out, float& db_out) const {
        float s;
        if (b >= 0.0f) {
            s = 1.0f / (1.0f + expf(-b));
        } else {
            float e = expf(b);
            s = e / (1.0f + e);
        }
        da_out = dy * s;
        db_out = dy * a * s * (1.0f - s);
    }
};

template <>
struct GluBackwardFunctor<double> {
    __device__ __forceinline__ void operator()(double dy, double a, double b,
                                                double& da_out, double& db_out) const {
        double s;
        if (b >= 0.0) {
            s = 1.0 / (1.0 + exp(-b));
        } else {
            double e = exp(b);
            s = e / (1.0 + e);
        }
        da_out = dy * s;
        db_out = dy * a * s * (1.0 - s);
    }
};

template <>
struct GluBackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(__half dy, __half a, __half b,
                                                __half& da_out, __half& db_out) const {
        float fdy = __half2float(dy);
        float fa  = __half2float(a);
        float fb  = __half2float(b);
        float s;
        if (fb >= 0.0f) {
            s = 1.0f / (1.0f + expf(-fb));
        } else {
            float e = expf(fb);
            s = e / (1.0f + e);
        }
        da_out = __float2half(fdy * s);
        db_out = __float2half(fdy * fa * s * (1.0f - s));
    }
};

template <>
struct GluBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(__nv_bfloat16 dy, __nv_bfloat16 a, __nv_bfloat16 b,
                                                __nv_bfloat16& da_out, __nv_bfloat16& db_out) const {
        float fdy = __bfloat162float(dy);
        float fa  = __bfloat162float(a);
        float fb  = __bfloat162float(b);
        float s;
        if (fb >= 0.0f) {
            s = 1.0f / (1.0f + expf(-fb));
        } else {
            float e = expf(fb);
            s = e / (1.0f + e);
        }
        da_out = __float2bfloat16(fdy * s);
        db_out = __float2bfloat16(fdy * fa * s * (1.0f - s));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_glu_backward_f32, float,
    baracuda::elementwise::GluBackwardFunctor<float>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_glu_backward_f16, __half,
    baracuda::elementwise::GluBackwardFunctor<__half>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_glu_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::GluBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_glu_backward_f64, double,
    baracuda::elementwise::GluBackwardFunctor<double>)
