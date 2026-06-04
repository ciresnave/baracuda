// baracuda-kernels Phase 4 reduction: axis-Norm2 for FP types.
//
// `y = ||x||_2 along dim=k = sqrt(sum(x^2))` with keepdim=true
// convention (output[k] = 1). Same naive trailblazer kernel template
// as Sum/Mean/Max/Min/Prod — one thread per output cell, loops over
// the reduce axis accumulating x*x, then a `finalize()` sqrt step.
//
// All four FP dtypes are wired. The functor is templated on T;
// f16 / bf16 specialize so accumulation and sqrt detour through f32
// (consistent with the existing SumReduce f16 / bf16 specializations).

#include "../include/baracuda_elementwise.cuh"
#include <math.h>

namespace baracuda { namespace elementwise {

// Norm2 reduction functor. `init()` returns 0; `op(acc, x) = acc + x*x`;
// `finalize(acc, extent) = sqrt(acc)`. The functor is templated on T;
// generic body works for f32 / f64 (single-line `sqrtf` / `sqrt`
// intrinsic at finalize); f16 / bf16 specialize to detour through f32
// for both the square-accumulate step AND the final sqrt.
// Note: `merge` is intentionally `a + b` (plain sum), NOT `op(a, x)`.
// The op embeds a square (`acc + x*x`) for the fold; once two threads
// hold partial sums-of-squares, merging is plain addition.
template <typename T>
struct Norm2Reduce {
    static __device__ __forceinline__ T init() { return T(0); }
    static __device__ __forceinline__ T finalize(T acc, int32_t /*extent*/) {
        // Default-template branch — chosen by f64 specialization only
        // (f32 has its own specialization below).
        return (T)sqrt((double)acc);
    }
    __device__ __forceinline__ T operator()(T acc, T x) const { return acc + x * x; }
    static __device__ __forceinline__ T merge(T a, T b) { return a + b; }
};

template <>
struct Norm2Reduce<float> {
    static __device__ __forceinline__ float init() { return 0.0f; }
    static __device__ __forceinline__ float finalize(float acc, int32_t /*extent*/) {
        return sqrtf(acc);
    }
    __device__ __forceinline__ float operator()(float acc, float x) const {
        return acc + x * x;
    }
    static __device__ __forceinline__ float merge(float a, float b) { return a + b; }
};

template <>
struct Norm2Reduce<__half> {
    static __device__ __forceinline__ __half init() { return __float2half(0.0f); }
    static __device__ __forceinline__ __half finalize(__half acc, int32_t /*extent*/) {
        return __float2half(sqrtf(__half2float(acc)));
    }
    __device__ __forceinline__ __half operator()(__half acc, __half x) const {
        float a  = __half2float(acc);
        float xf = __half2float(x);
        return __float2half(a + xf * xf);
    }
    static __device__ __forceinline__ __half merge(__half a, __half b) {
        return __float2half(__half2float(a) + __half2float(b));
    }
};

template <>
struct Norm2Reduce<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 init() {
        return __float2bfloat16(0.0f);
    }
    static __device__ __forceinline__ __nv_bfloat16 finalize(
        __nv_bfloat16 acc, int32_t /*extent*/)
    {
        return __float2bfloat16(sqrtf(__bfloat162float(acc)));
    }
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 acc, __nv_bfloat16 x) const
    {
        float a  = __bfloat162float(acc);
        float xf = __bfloat162float(x);
        return __float2bfloat16(a + xf * xf);
    }
    static __device__ __forceinline__ __nv_bfloat16 merge(
        __nv_bfloat16 a, __nv_bfloat16 b)
    {
        return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_norm2_f32, float, baracuda::elementwise::Norm2Reduce<float>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_norm2_f16, __half, baracuda::elementwise::Norm2Reduce<__half>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_norm2_bf16, __nv_bfloat16, baracuda::elementwise::Norm2Reduce<__nv_bfloat16>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_norm2_f64, double, baracuda::elementwise::Norm2Reduce<double>)
