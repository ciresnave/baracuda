// baracuda-kernels Phase 3 unary backward fanout: expm1 backward.
//
// Forward: `y = exp(x) - 1`. Backward: `dx = dy * exp(x) = dy * (y + 1)`
// — saved-y. We add `T(1)` then multiply; one fused-mul-add at worst.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct Expm1BackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T y) const {
        return dy * (y + T(1));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_expm1_backward_f32, float,
    baracuda::elementwise::Expm1BackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_expm1_backward_f16, __half,
    baracuda::elementwise::Expm1BackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_expm1_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::Expm1BackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_expm1_backward_f64, double,
    baracuda::elementwise::Expm1BackwardFunctor<double>)
