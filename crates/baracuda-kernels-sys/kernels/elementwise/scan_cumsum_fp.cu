// baracuda-kernels Phase 4 scan trailblazer: cumsum (inclusive prefix
// sum) for FP types.
//
// `y[i] = Σ_{j≤i} x[j]` along the scan axis (or `Σ_{j≥i} x[j]` when
// `reverse != 0`). Trailblazer kernel is naive — one thread per output
// cell, each loops O(extent) along the scan axis to accumulate its
// prefix. Parallel-scan (Blelloch / Hillis-Steele) optimization is
// future work for large extents.
//
// Functor reuses the same `init / op / finalize` shape as the reduce
// family (init = 0, op = `a + x`, finalize = pass-through), so the
// same `SumReduce` semantic that `reduce_sum_fp.cu` ships could be
// reused — but to avoid cross-file template coupling, we define a
// local `CumsumScan` mirror here. f16 / bf16 detour through f32 for
// the accumulator (preserves single-rounding behavior at the f16/bf16
// store).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct CumsumScan {
    static __device__ __forceinline__ T init() { return T(0); }
    static __device__ __forceinline__ T finalize(T acc, int32_t /*ext*/) { return acc; }
    __device__ __forceinline__ T operator()(T acc, T x) const { return acc + x; }
};

template <>
struct CumsumScan<__half> {
    static __device__ __forceinline__ __half init() { return __float2half(0.0f); }
    static __device__ __forceinline__ __half finalize(__half acc, int32_t /*ext*/) { return acc; }
    __device__ __forceinline__ __half operator()(__half acc, __half x) const {
        return __float2half(__half2float(acc) + __half2float(x));
    }
};

template <>
struct CumsumScan<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 init() { return __float2bfloat16(0.0f); }
    static __device__ __forceinline__ __nv_bfloat16 finalize(
        __nv_bfloat16 acc, int32_t /*ext*/) { return acc; }
    __device__ __forceinline__ __nv_bfloat16 operator()(
        __nv_bfloat16 acc, __nv_bfloat16 x) const {
        return __float2bfloat16(__bfloat162float(acc) + __bfloat162float(x));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cumsum_f32, float, baracuda::elementwise::CumsumScan<float>)

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cumsum_f16, __half, baracuda::elementwise::CumsumScan<__half>)

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cumsum_bf16, __nv_bfloat16, baracuda::elementwise::CumsumScan<__nv_bfloat16>)

BARACUDA_KERNELS_SCAN_AXIS_INSTANTIATE(
    scan_cumsum_f64, double, baracuda::elementwise::CumsumScan<double>)
