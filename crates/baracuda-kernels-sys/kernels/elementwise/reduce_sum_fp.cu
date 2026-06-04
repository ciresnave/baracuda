// baracuda-kernels Phase 4 reduction trailblazer: axis-sum for FP types.
//
// `y = sum(x, dim=k)` with keepdim=true convention: output shape ==
// input shape but with `output[k] = 1`. Trailblazer kernel is naive —
// one thread per output cell, loops over the reduced axis. Warp /
// block reduction tile optimizations are Phase 4 follow-up.
//
// All four FP dtypes are wired (f32 trailblazer + f16 / bf16 / f64
// fanout). The functor is templated on T; f16 / bf16 specialize the
// op to detour through f32 (the standard `__half2float` / `+` /
// `__float2half` pattern used throughout the unary fanout).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Sum reduction functor. Templated on T; the `init()` static returns
// the additive identity (0 in each dtype's representation). `finalize`
// is pass-through for Sum (no extent divide — that's Mean's job).
// Generic body works for f32 / f64; f16 / bf16 specialize so the init
// constant + op both go through the dtype's explicit converters.
template <typename T>
struct SumReduce {
    static __device__ __forceinline__ T init() { return T(0); }
    static __device__ __forceinline__ T finalize(T acc, int32_t /*extent*/) {
        return acc;
    }
    __device__ __forceinline__ T operator()(T acc, T x) const { return acc + x; }
    static __device__ __forceinline__ T merge(T a, T b) { return a + b; }
};

template <>
struct SumReduce<__half> {
    static __device__ __forceinline__ __half init() { return __float2half(0.0f); }
    static __device__ __forceinline__ __half finalize(__half acc, int32_t /*extent*/) {
        return acc;
    }
    __device__ __forceinline__ __half operator()(__half acc, __half x) const {
        return __float2half(__half2float(acc) + __half2float(x));
    }
    static __device__ __forceinline__ __half merge(__half a, __half b) {
        return __float2half(__half2float(a) + __half2float(b));
    }
};

template <>
struct SumReduce<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 init() {
        return __float2bfloat16(0.0f);
    }
    static __device__ __forceinline__ __nv_bfloat16 finalize(
        __nv_bfloat16 acc, int32_t /*extent*/)
    {
        return acc;
    }
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 acc, __nv_bfloat16 x) const
    {
        return __float2bfloat16(__bfloat162float(acc) + __bfloat162float(x));
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
    reduce_sum_f32, float, baracuda::elementwise::SumReduce<float>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_sum_f16, __half, baracuda::elementwise::SumReduce<__half>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_sum_bf16, __nv_bfloat16, baracuda::elementwise::SumReduce<__nv_bfloat16>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_sum_f64, double, baracuda::elementwise::SumReduce<double>)
