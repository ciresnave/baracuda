// baracuda-kernels Phase 3: elementwise `addcdiv` for FP types.
//
// PyTorch `torch.addcdiv(input=a, tensor1=b, tensor2=c, value=scale)`:
//   y = a + scale * (b / c)
//
// All 4 FP dtypes wired × {contig, strided}. Same unfused-rounding
// strategy as `ternary_addcmul_fp.cu` — explicit `__fdiv_rn` /
// `__ddiv_rn` for the divide step, `__fmul_rn` / `__dmul_rn` for the
// scale, `__fadd_rn` / `__dadd_rn` for the final accumulate. nvcc
// won't fuse div with mul/add, but we use the explicit intrinsics
// for consistency with the addcmul / fma family.
//
// Divide-by-zero: produces ±inf on f32 / f64; the kernel doesn't
// special-case it. Callers must avoid zero `c` if they don't want
// inf propagation. For f16 / bf16 the f32-detour propagates inf
// through the round-to-half conversion as expected.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AddcdivFunctor {
    __device__ __forceinline__ T operator()(T a, T b, T c, float scale) const;
};

template <>
struct AddcdivFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b, float c, float scale) const {
        // PyTorch convention: input + value * (tensor1 / tensor2)
        float t = __fdiv_rn(b, c);
        t = __fmul_rn(scale, t);
        return __fadd_rn(a, t);
    }
};

template <>
struct AddcdivFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b, double c, float scale) const {
        double t = __ddiv_rn(b, c);
        t = __dmul_rn(static_cast<double>(scale), t);
        return __dadd_rn(a, t);
    }
};

template <>
struct AddcdivFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b, __half c, float scale) const {
        float af = __half2float(a);
        float bf = __half2float(b);
        float cf = __half2float(c);
        float t = __fdiv_rn(bf, cf);
        t = __fmul_rn(scale, t);
        return __float2half(__fadd_rn(af, t));
    }
};

template <>
struct AddcdivFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16
    operator()(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, float scale) const {
        float af = __bfloat162float(a);
        float bf = __bfloat162float(b);
        float cf = __bfloat162float(c);
        float t = __fdiv_rn(bf, cf);
        t = __fmul_rn(scale, t);
        return __float2bfloat16(__fadd_rn(af, t));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE(
    ternary_addcdiv_f32, float, baracuda::elementwise::AddcdivFunctor<float>)
BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE_STRIDED(
    ternary_addcdiv_f32, float, baracuda::elementwise::AddcdivFunctor<float>)

BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE(
    ternary_addcdiv_f16, __half, baracuda::elementwise::AddcdivFunctor<__half>)
BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE_STRIDED(
    ternary_addcdiv_f16, __half, baracuda::elementwise::AddcdivFunctor<__half>)

BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE(
    ternary_addcdiv_bf16, __nv_bfloat16, baracuda::elementwise::AddcdivFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE_STRIDED(
    ternary_addcdiv_bf16, __nv_bfloat16, baracuda::elementwise::AddcdivFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE(
    ternary_addcdiv_f64, double, baracuda::elementwise::AddcdivFunctor<double>)
BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE_STRIDED(
    ternary_addcdiv_f64, double, baracuda::elementwise::AddcdivFunctor<double>)
