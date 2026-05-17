// baracuda-kernels Phase 3 binary fanout: elementwise floor_divide
// `y = floor(a / b)` — elementwise floored division. Matches PyTorch's
// `torch.div(a, b, rounding_mode='floor')` and `torch.floor_divide`.
//
// f32 → `floorf(a / b)`, f64 → `floor(a / b)`, f16 / bf16 → f32-detour
// pattern matching the unary transcendental family. Behavior on
// `b == 0` is `+/-inf` or NaN per IEEE 754 div followed by floor, which
// preserves IEEE inf/NaN as IEEE inf/NaN (`floorf(inf) == inf`,
// `floorf(NaN) == NaN`). Callers are responsible for masking `b == 0`
// inputs if they need integer-division-style trap semantics.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct FloorDivideFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct FloorDivideFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return floorf(a / b);
    }
};

template <>
struct FloorDivideFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        return floor(a / b);
    }
};

template <>
struct FloorDivideFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __float2half(floorf(__half2float(a) / __half2float(b)));
    }
};

template <>
struct FloorDivideFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        return __float2bfloat16(floorf(__bfloat162float(a) / __bfloat162float(b)));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_floor_divide_f32, float, baracuda::elementwise::FloorDivideFunctor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_floor_divide_f32, float, baracuda::elementwise::FloorDivideFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_floor_divide_f16, __half, baracuda::elementwise::FloorDivideFunctor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_floor_divide_f16, __half, baracuda::elementwise::FloorDivideFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_floor_divide_bf16, __nv_bfloat16, baracuda::elementwise::FloorDivideFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_floor_divide_bf16, __nv_bfloat16, baracuda::elementwise::FloorDivideFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_floor_divide_f64, double, baracuda::elementwise::FloorDivideFunctor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_floor_divide_f64, double, baracuda::elementwise::FloorDivideFunctor<double>)
