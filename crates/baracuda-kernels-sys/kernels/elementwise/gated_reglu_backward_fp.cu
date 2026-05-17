// baracuda-kernels Phase 3 Category C′ — ReGLU backward.
//
// Forward: `y = a · relu(b)`. Backward (saved `x`):
//   da = dy · relu(b)               = (b > 0) ? dy·b : 0
//   db = dy · a · relu'(b)          = (b > 0) ? dy·a : 0
// Bit-exact at f32 / f64.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct RegluBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T a, T b, T& da_out, T& db_out) const {
        da_out = dy * b;
        db_out = dy * a;
    }
};

template <>
struct RegluBackwardFunctor<float> {
    __device__ __forceinline__ void operator()(float dy, float a, float b,
                                                float& da_out, float& db_out) const {
        if (b > 0.0f) {
            da_out = dy * b;
            db_out = dy * a;
        } else {
            da_out = 0.0f;
            db_out = 0.0f;
        }
    }
};

template <>
struct RegluBackwardFunctor<double> {
    __device__ __forceinline__ void operator()(double dy, double a, double b,
                                                double& da_out, double& db_out) const {
        if (b > 0.0) {
            da_out = dy * b;
            db_out = dy * a;
        } else {
            da_out = 0.0;
            db_out = 0.0;
        }
    }
};

template <>
struct RegluBackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(__half dy, __half a, __half b,
                                                __half& da_out, __half& db_out) const {
        float fdy = __half2float(dy);
        float fa  = __half2float(a);
        float fb  = __half2float(b);
        if (fb > 0.0f) {
            da_out = __float2half(fdy * fb);
            db_out = __float2half(fdy * fa);
        } else {
            da_out = __float2half(0.0f);
            db_out = __float2half(0.0f);
        }
    }
};

template <>
struct RegluBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(__nv_bfloat16 dy, __nv_bfloat16 a, __nv_bfloat16 b,
                                                __nv_bfloat16& da_out, __nv_bfloat16& db_out) const {
        float fdy = __bfloat162float(dy);
        float fa  = __bfloat162float(a);
        float fb  = __bfloat162float(b);
        if (fb > 0.0f) {
            da_out = __float2bfloat16(fdy * fb);
            db_out = __float2bfloat16(fdy * fa);
        } else {
            da_out = __float2bfloat16(0.0f);
            db_out = __float2bfloat16(0.0f);
        }
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_reglu_backward_f32, float,
    baracuda::elementwise::RegluBackwardFunctor<float>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_reglu_backward_f16, __half,
    baracuda::elementwise::RegluBackwardFunctor<__half>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_reglu_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::RegluBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_GATED_ACTIVATION_BACKWARD_INSTANTIATE(
    gated_reglu_backward_f64, double,
    baracuda::elementwise::RegluBackwardFunctor<double>)
