// baracuda-kernels Phase 4 reduction fanout: axis-product for FP types.
//
// `y = prod(x, dim=k)` with keepdim=true. `init = 1`; op multiplies;
// `finalize` is pass-through. For f16 / bf16 the multiply detours
// through f32 (same pattern as Sum/Mean fanout). Callers must keep
// values close to 1 in low-precision dtypes since cumulative product
// overflows fast.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct ProdReduce {
    static __device__ __forceinline__ T init() { return T(1); }
    static __device__ __forceinline__ T finalize(T acc, int32_t /*extent*/) {
        return acc;
    }
    __device__ __forceinline__ T operator()(T acc, T x) const { return acc * x; }
};

template <>
struct ProdReduce<__half> {
    static __device__ __forceinline__ __half init() { return __float2half(1.0f); }
    static __device__ __forceinline__ __half finalize(__half acc, int32_t /*extent*/) {
        return acc;
    }
    __device__ __forceinline__ __half operator()(__half acc, __half x) const {
        return __float2half(__half2float(acc) * __half2float(x));
    }
};

template <>
struct ProdReduce<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 init() {
        return __float2bfloat16(1.0f);
    }
    static __device__ __forceinline__ __nv_bfloat16 finalize(
        __nv_bfloat16 acc, int32_t /*extent*/)
    {
        return acc;
    }
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 acc, __nv_bfloat16 x) const
    {
        return __float2bfloat16(__bfloat162float(acc) * __bfloat162float(x));
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_prod_f32, float, baracuda::elementwise::ProdReduce<float>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_prod_f16, __half, baracuda::elementwise::ProdReduce<__half>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_prod_bf16, __nv_bfloat16, baracuda::elementwise::ProdReduce<__nv_bfloat16>)

BARACUDA_KERNELS_REDUCE_AXIS_INSTANTIATE(
    reduce_prod_f64, double, baracuda::elementwise::ProdReduce<double>)
