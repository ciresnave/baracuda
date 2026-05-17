// baracuda-kernels Phase 4 scan fanout: cummax (inclusive prefix
// running max) for FP types.
//
// `y[i] = max(x[0..=i])` along the scan axis (or `max(x[i..extent])`
// when `reverse != 0`). Functor: `init = -INF`, `op(a, x) = max(a, x)`,
// `finalize = pass-through`. f16 / bf16 detour through f32 for the
// comparison.

#include "../include/baracuda_elementwise.cuh"
#include <math_constants.h>

namespace baracuda { namespace elementwise {

template <typename T>
struct CummaxScan;

template <>
struct CummaxScan<float> {
    static __device__ __forceinline__ float init() { return -CUDART_INF_F; }
    static __device__ __forceinline__ float finalize(float acc, int32_t /*ext*/) { return acc; }
    __device__ __forceinline__ float operator()(float acc, float x) const {
        return (x > acc) ? x : acc;
    }
};

template <>
struct CummaxScan<double> {
    static __device__ __forceinline__ double init() { return -CUDART_INF; }
    static __device__ __forceinline__ double finalize(double acc, int32_t /*ext*/) { return acc; }
    __device__ __forceinline__ double operator()(double acc, double x) const {
        return (x > acc) ? x : acc;
    }
};

template <>
struct CummaxScan<__half> {
    static __device__ __forceinline__ __half init() {
        return __float2half(-CUDART_INF_F);
    }
    static __device__ __forceinline__ __half finalize(__half acc, int32_t /*ext*/) { return acc; }
    __device__ __forceinline__ __half operator()(__half acc, __half x) const {
        float a = __half2float(acc);
        float b = __half2float(x);
        return __float2half((b > a) ? b : a);
    }
};

template <>
struct CummaxScan<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 init() {
        return __float2bfloat16(-CUDART_INF_F);
    }
    static __device__ __forceinline__ __nv_bfloat16 finalize(
        __nv_bfloat16 acc, int32_t /*ext*/) { return acc; }
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 acc, __nv_bfloat16 x) const {
        float a = __bfloat162float(acc);
        float b = __bfloat162float(x);
        return __float2bfloat16((b > a) ? b : a);
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cummax_f32, float, baracuda::elementwise::CummaxScan<float>)

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cummax_f16, __half, baracuda::elementwise::CummaxScan<__half>)

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cummax_bf16, __nv_bfloat16, baracuda::elementwise::CummaxScan<__nv_bfloat16>)

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cummax_f64, double, baracuda::elementwise::CummaxScan<double>)
