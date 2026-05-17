// baracuda-kernels Phase 4 scan fanout: cumprod (inclusive prefix
// product) for FP types.
//
// `y[i] = ∏_{j≤i} x[j]` along the scan axis (or `∏_{j≥i} x[j]` when
// `reverse != 0`). Functor mirrors CumsumScan shape — `init = T(1)`,
// `op(a, x) = a * x`, `finalize = pass-through`. f16 / bf16 detour
// through f32 for the accumulator.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct CumprodScan {
    static __device__ __forceinline__ T init() { return T(1); }
    static __device__ __forceinline__ T finalize(T acc, int32_t /*ext*/) { return acc; }
    __device__ __forceinline__ T operator()(T acc, T x) const { return acc * x; }
};

template <>
struct CumprodScan<__half> {
    static __device__ __forceinline__ __half init() { return __float2half(1.0f); }
    static __device__ __forceinline__ __half finalize(__half acc, int32_t /*ext*/) { return acc; }
    __device__ __forceinline__ __half operator()(__half acc, __half x) const {
        return __float2half(__half2float(acc) * __half2float(x));
    }
};

template <>
struct CumprodScan<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 init() { return __float2bfloat16(1.0f); }
    static __device__ __forceinline__ __nv_bfloat16 finalize(
        __nv_bfloat16 acc, int32_t /*ext*/) { return acc; }
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 acc, __nv_bfloat16 x) const {
        return __float2bfloat16(__bfloat162float(acc) * __bfloat162float(x));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cumprod_f32, float, baracuda::elementwise::CumprodScan<float>)

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cumprod_f16, __half, baracuda::elementwise::CumprodScan<__half>)

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cumprod_bf16, __nv_bfloat16, baracuda::elementwise::CumprodScan<__nv_bfloat16>)

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cumprod_f64, double, baracuda::elementwise::CumprodScan<double>)
