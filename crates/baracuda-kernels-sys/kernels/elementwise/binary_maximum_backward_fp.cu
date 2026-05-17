// baracuda-kernels Phase 3 backward fanout: elementwise maximum backward.
//
// Forward: `y = maximum(a, b)`. Backward needs both saved forward inputs
// (`a` and `b`) — not as multipliers, but as references for the comparison
// that decides which operand received the gradient. This is the first
// binary BW where the saves are used purely as references.
//
// Tie-break: PyTorch parity (derivatives.yaml `maximum`):
//     da = at::where(a == b, dy / 2, dy).masked_fill_(a < b, 0)
//     db = at::where(a == b, dy / 2, dy).masked_fill_(b < a, 0)
// Resolving the table:
//     a > b : da = dy,    db = 0
//     a < b : da = 0,     db = dy
//     a == b: da = dy/2,  db = dy/2     (tie — split)
//     NaN   : da = dy,    db = dy       (all comparisons false → both keep dy)
//
// f16 / bf16 use the f32-detour pattern (compare and scale in f32, round
// once on store), mirroring the rest of the elementwise family.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct MaximumBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T a, T b, T& da, T& db) const {
        // Generic path used for float / double. Half-precision specializations
        // below detour through f32.
        if (a == b) {
            T half_dy = dy * T(0.5);
            da = half_dy;
            db = half_dy;
        } else {
            da = (a < b) ? T(0) : dy;
            db = (b < a) ? T(0) : dy;
        }
    }
};

template <>
struct MaximumBackwardFunctor<__half> {
    __device__ __forceinline__ void operator()(__half dy, __half a, __half b,
                                               __half& da, __half& db) const {
        float fa  = __half2float(a);
        float fb  = __half2float(b);
        float fdy = __half2float(dy);
        float fda, fdb;
        if (fa == fb) {
            float half_dy = fdy * 0.5f;
            fda = half_dy;
            fdb = half_dy;
        } else {
            fda = (fa < fb) ? 0.0f : fdy;
            fdb = (fb < fa) ? 0.0f : fdy;
        }
        da = __float2half(fda);
        db = __float2half(fdb);
    }
};

template <>
struct MaximumBackwardFunctor<__nv_bfloat16> {
    __device__ __forceinline__ void operator()(__nv_bfloat16 dy, __nv_bfloat16 a,
                                               __nv_bfloat16 b, __nv_bfloat16& da,
                                               __nv_bfloat16& db) const {
        float fa  = __bfloat162float(a);
        float fb  = __bfloat162float(b);
        float fdy = __bfloat162float(dy);
        float fda, fdb;
        if (fa == fb) {
            float half_dy = fdy * 0.5f;
            fda = half_dy;
            fdb = half_dy;
        } else {
            fda = (fa < fb) ? 0.0f : fdy;
            fdb = (fb < fa) ? 0.0f : fdy;
        }
        da = __float2bfloat16(fda);
        db = __float2bfloat16(fdb);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_maximum_backward_f32, float,
    baracuda::elementwise::MaximumBackwardFunctor<float>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_maximum_backward_f16, __half,
    baracuda::elementwise::MaximumBackwardFunctor<__half>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_maximum_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::MaximumBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_maximum_backward_f64, double,
    baracuda::elementwise::MaximumBackwardFunctor<double>)
