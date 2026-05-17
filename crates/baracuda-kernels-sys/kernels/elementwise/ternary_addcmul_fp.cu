// baracuda-kernels Phase 3: elementwise `addcmul` for FP types.
//
// PyTorch `torch.addcmul(input=a, tensor1=b, tensor2=c, value=scale)`:
//   y = a + scale * b * c
//
// All 4 FP dtypes wired × {contig, strided}. Per-dtype specialization
// for f32 / f64 uses unfused round-to-nearest intrinsics
// (`__fmul_rn` / `__fadd_rn` / `__dmul_rn` / `__dadd_rn`) so that nvcc
// doesn't auto-fuse `scale * b * c + a` into an IEEE fma (which would
// produce one fewer rounding step than PyTorch's plain mul+add
// convention). f16 / bf16 use the f32 detour with the same unfused
// intrinsics.
//
// See `ternary_fma_fp.cu` for the prior occurrence of the fma-fusion
// workaround.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct AddcmulFunctor {
    __device__ __forceinline__ T operator()(T a, T b, T c, float scale) const;
};

template <>
struct AddcmulFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b, float c, float scale) const {
        // PyTorch convention: a + value * tensor1 * tensor2, plain
        // mul+add (no IEEE fma fusion).
        float t = __fmul_rn(scale, b);
        t = __fmul_rn(t, c);
        return __fadd_rn(a, t);
    }
};

template <>
struct AddcmulFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b, double c, float scale) const {
        double t = __dmul_rn(static_cast<double>(scale), b);
        t = __dmul_rn(t, c);
        return __dadd_rn(a, t);
    }
};

template <>
struct AddcmulFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b, __half c, float scale) const {
        float af = __half2float(a);
        float bf = __half2float(b);
        float cf = __half2float(c);
        float t = __fmul_rn(scale, bf);
        t = __fmul_rn(t, cf);
        return __float2half(__fadd_rn(af, t));
    }
};

template <>
struct AddcmulFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16
    operator()(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, float scale) const {
        float af = __bfloat162float(a);
        float bf = __bfloat162float(b);
        float cf = __bfloat162float(c);
        float t = __fmul_rn(scale, bf);
        t = __fmul_rn(t, cf);
        return __float2bfloat16(__fadd_rn(af, t));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE(
    ternary_addcmul_f32, float, baracuda::elementwise::AddcmulFunctor<float>)
BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE_STRIDED(
    ternary_addcmul_f32, float, baracuda::elementwise::AddcmulFunctor<float>)

BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE(
    ternary_addcmul_f16, __half, baracuda::elementwise::AddcmulFunctor<__half>)
BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE_STRIDED(
    ternary_addcmul_f16, __half, baracuda::elementwise::AddcmulFunctor<__half>)

BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE(
    ternary_addcmul_bf16, __nv_bfloat16, baracuda::elementwise::AddcmulFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE_STRIDED(
    ternary_addcmul_bf16, __nv_bfloat16, baracuda::elementwise::AddcmulFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE(
    ternary_addcmul_f64, double, baracuda::elementwise::AddcmulFunctor<double>)
BARACUDA_KERNELS_TERNARY_SCALED_POINTWISE_INSTANTIATE_STRIDED(
    ternary_addcmul_f64, double, baracuda::elementwise::AddcmulFunctor<double>)
