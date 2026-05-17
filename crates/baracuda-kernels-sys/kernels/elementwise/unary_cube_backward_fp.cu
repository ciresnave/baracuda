// baracuda-kernels Phase 3 unary backward fanout: cube backward.
//
// Forward: `y = x³`. Backward: `dx = dy * 3 * x²` — saved-x. Pure
// arithmetic; generic-on-T covers all four FP dtypes.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct CubeBackwardFunctor {
    __device__ __forceinline__ T operator()(T dy, T x) const {
        return dy * T(3) * x * x;
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cube_backward_f32, float,
    baracuda::elementwise::CubeBackwardFunctor<float>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cube_backward_f16, __half,
    baracuda::elementwise::CubeBackwardFunctor<__half>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cube_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::CubeBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_UNARY_BACKWARD_INSTANTIATE(
    unary_cube_backward_f64, double,
    baracuda::elementwise::CubeBackwardFunctor<double>)
