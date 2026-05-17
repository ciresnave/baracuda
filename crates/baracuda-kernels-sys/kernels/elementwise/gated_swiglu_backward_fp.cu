// baracuda-kernels Phase 3 Category C′ — SwiGLU backward.
//
// Forward: `y = a · silu(b) = a · b · sigmoid(b)`. Backward (saved `x`):
//   da = dy · silu(b) = dy · b · sigmoid(b)
//   db = dy · a · silu'(b) = dy · a · sigmoid(b) · (1 + b·(1 - sigmoid(b)))
//
// f32 / f64 use direct intrinsic math; f16 / bf16 use the f32 detour
// once (read a, b, dy as f32; compute sigmoid in f32; round each output
// half once on store).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct SwigluBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T a, T b, T& da_out, T& db_out) const {
        da_out = dy * b;
        db_out = dy * a;
    }
};

template <>
struct SwigluBackwardFunctor<float> {
    __device__ __forceinline__ void operator()(float dy, float a, float b,
                                                float& da_out, float& db_out) const {
        float s;
        if (b >= 0.0f) {
            s = 1.0f / (1.0f + expf(-b));
        } else {
            float e = expf(b);
            s = e / (1.0f + e);
        }
        float silu_b = b * s;
        float silu_prime_b = s * (1.0f + b * (1.0f - s));
        da_out = dy * silu_b;
        db_out = dy * a * silu_prime_b;
    }
};

template <>
struct SwigluBackwardFunctor<double> {
    __device__ __forceinline__ void operator()(double dy, double a, double b,
                                                double& da_out, double& db_out) const {
        double s;
        if (b >= 0.0) {
            s = 1.0 / (1.0 + exp(-b));
        } else {
            double e = exp(b);
            s = e / (1.0 + e);
        }
        double silu_b = b * s;
        double silu_prime_b = s * (1.0 + b * (1.0 - s));
        da_out = dy * silu_b;
        db_out = dy * a * silu_prime_b;
    }
};

template <>
struct SwigluBackwardFunctor<__half> {
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
        float silu_b = fb * s;
        float silu_prime_b = s * (1.0f + fb * (1.0f - s));
        da_out = __float2half(fdy * silu_b);
        db_out = __float2half(fdy * fa * silu_prime_b);
    }
};

template <>
struct SwigluBackwardFunctor<__nv_bfloat16> {
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
        float silu_b = fb * s;
        float silu_prime_b = s * (1.0f + fb * (1.0f - s));
        da_out = __float2bfloat16(fdy * silu_b);
        db_out = __float2bfloat16(fdy * fa * silu_prime_b);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_swiglu_backward_f32, float,
    baracuda::elementwise::SwigluBackwardFunctor<float>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_swiglu_backward_f16, __half,
    baracuda::elementwise::SwigluBackwardFunctor<__half>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_swiglu_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::SwigluBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_swiglu_backward_f64, double,
    baracuda::elementwise::SwigluBackwardFunctor<double>)
