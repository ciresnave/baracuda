// baracuda-kernels Phase 3 backward fanout: elementwise div backward.
//
// Forward: `y = a / b`. Backward: `(da, db) = (dy / b, -dy * a / b²)`.
// Needs the saved forward inputs `a` and `b` from the caller. Callers
// must ensure `b` is non-zero on every cell — the kernel does not guard
// against divide-by-zero (matching the forward div kernel's contract).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct DivBackwardFunctor {
    __device__ __forceinline__ void operator()(T dy, T a, T b, T& da, T& db) const {
        // da = dy / b
        da = dy / b;
        // db = -(dy * a) / (b * b). We negate via `T(0) - x` so the
        // expression compiles uniformly across {float, double, __half,
        // __nv_bfloat16} regardless of whether unary minus is provided.
        T prod_ab = dy * a;
        T denom   = b * b;
        db = (T(0) - prod_ab) / denom;
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_div_backward_f32, float,
    baracuda::elementwise::DivBackwardFunctor<float>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_div_backward_f16, __half,
    baracuda::elementwise::DivBackwardFunctor<__half>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_div_backward_bf16, __nv_bfloat16,
    baracuda::elementwise::DivBackwardFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_BACKWARD_SAVES_INSTANTIATE(
    binary_div_backward_f64, double,
    baracuda::elementwise::DivBackwardFunctor<double>)
