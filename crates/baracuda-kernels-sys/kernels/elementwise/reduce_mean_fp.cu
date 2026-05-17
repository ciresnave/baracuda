// baracuda-kernels Phase 4 reduction fanout: axis-mean for FP types.
//
// `y = mean(x, dim=k)` with keepdim=true. Accumulate as a sum, then
// divide by `reduce_extent` in `finalize`. The kernel template handles
// the dispatch; this file supplies the `MeanReduce<T>` functor + four
// per-dtype instantiations.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Mean reduction functor. `init = 0`; op accumulates a running sum;
// `finalize(acc, extent) = acc / extent`. For f16 / bf16 the divide
// detours through f32 (the same pattern used throughout the unary
// transcendental fanout).
template <typename T>
struct MeanReduce {
    static __device__ __forceinline__ T init() { return T(0); }
    static __device__ __forceinline__ T finalize(T acc, int32_t extent) {
        return acc / static_cast<T>(extent);
    }
    __device__ __forceinline__ T operator()(T acc, T x) const { return acc + x; }
};

template <>
struct MeanReduce<__half> {
    static __device__ __forceinline__ __half init() { return __float2half(0.0f); }
    static __device__ __forceinline__ __half finalize(__half acc, int32_t extent) {
        return __float2half(__half2float(acc) / static_cast<float>(extent));
    }
    __device__ __forceinline__ __half operator()(__half acc, __half x) const {
        return __float2half(__half2float(acc) + __half2float(x));
    }
};

template <>
struct MeanReduce<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 init() {
        return __float2bfloat16(0.0f);
    }
    static __device__ __forceinline__ __nv_bfloat16 finalize(
        __nv_bfloat16 acc, int32_t extent)
    {
        return __float2bfloat16(__bfloat162float(acc) / static_cast<float>(extent));
    }
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 acc, __nv_bfloat16 x) const
    {
        return __float2bfloat16(__bfloat162float(acc) + __bfloat162float(x));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_mean_f32, float, baracuda::elementwise::MeanReduce<float>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_mean_f16, __half, baracuda::elementwise::MeanReduce<__half>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_mean_bf16, __nv_bfloat16, baracuda::elementwise::MeanReduce<__nv_bfloat16>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_mean_f64, double, baracuda::elementwise::MeanReduce<double>)
