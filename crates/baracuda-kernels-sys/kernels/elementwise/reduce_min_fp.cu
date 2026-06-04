// baracuda-kernels Phase 4 reduction fanout: axis-min for FP types.
//
// `y = min(x, dim=k)` with keepdim=true. `init = +INFINITY`; op uses
// `fmin` / `fminf`; `finalize` is pass-through. For f16 / bf16 the
// compare detours through f32.

#include "../include/baracuda_elementwise.cuh"

#include <cmath>

namespace baracuda { namespace elementwise {

template <typename T>
struct MinReduce {
    static __device__ __forceinline__ T init();
    static __device__ __forceinline__ T finalize(T acc, int32_t /*extent*/) { return acc; }
    __device__ __forceinline__ T operator()(T acc, T x) const;
};

template <>
struct MinReduce<float> {
    static __device__ __forceinline__ float init() { return INFINITY; }
    static __device__ __forceinline__ float finalize(float acc, int32_t /*extent*/) {
        return acc;
    }
    __device__ __forceinline__ float operator()(float acc, float x) const {
        return fminf(acc, x);
    }
    static __device__ __forceinline__ float merge(float a, float b) {
        return fminf(a, b);
    }
};

template <>
struct MinReduce<double> {
    static __device__ __forceinline__ double init() {
        return static_cast<double>(INFINITY);
    }
    static __device__ __forceinline__ double finalize(double acc, int32_t /*extent*/) {
        return acc;
    }
    __device__ __forceinline__ double operator()(double acc, double x) const {
        return fmin(acc, x);
    }
    static __device__ __forceinline__ double merge(double a, double b) {
        return fmin(a, b);
    }
};

template <>
struct MinReduce<__half> {
    static __device__ __forceinline__ __half init() { return __float2half(INFINITY); }
    static __device__ __forceinline__ __half finalize(__half acc, int32_t /*extent*/) {
        return acc;
    }
    __device__ __forceinline__ __half operator()(__half acc, __half x) const {
        return __float2half(fminf(__half2float(acc), __half2float(x)));
    }
    static __device__ __forceinline__ __half merge(__half a, __half b) {
        return __float2half(fminf(__half2float(a), __half2float(b)));
    }
};

template <>
struct MinReduce<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 init() {
        return __float2bfloat16(INFINITY);
    }
    static __device__ __forceinline__ __nv_bfloat16 finalize(
        __nv_bfloat16 acc, int32_t /*extent*/)
    {
        return acc;
    }
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 acc, __nv_bfloat16 x) const
    {
        return __float2bfloat16(fminf(__bfloat162float(acc), __bfloat162float(x)));
    }
    static __device__ __forceinline__ __nv_bfloat16 merge(
        __nv_bfloat16 a, __nv_bfloat16 b)
    {
        return __float2bfloat16(fminf(__bfloat162float(a), __bfloat162float(b)));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_min_f32, float, baracuda::elementwise::MinReduce<float>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_min_f16, __half, baracuda::elementwise::MinReduce<__half>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_min_bf16, __nv_bfloat16, baracuda::elementwise::MinReduce<__nv_bfloat16>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_min_f64, double, baracuda::elementwise::MinReduce<double>)
