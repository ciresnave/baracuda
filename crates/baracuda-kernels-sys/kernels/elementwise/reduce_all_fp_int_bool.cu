// baracuda-kernels Phase 4 deferral 4.4: `all` along one axis.
//
// `y[c_reduced] = AND_j (x[..., j, ...] != 0)`. Output dtype is
// `uint8_t` (Bool — PyTorch convention: 0 = false, 1 = true).
//
// Mirror of `Any`. The functor starts at `init = 1` (true) and ANDs
// each `(x != 0)` predicate. NaN is truthy (matches PyTorch / IEEE 754).
//
// Wired matrix: `All × {f32, f16, bf16, f64, i32, i64, Bool}` — 7 SKUs.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// `All` functor for FP / int / bool input. `acc` is the bool
// accumulator (`uint8_t`, 0 or 1); init = 1 (true), op ANDs in
// `(x != 0)`. Once acc hits 0, subsequent values can't flip it back.
template <typename T_in>
struct AllReduce {
    static __device__ __forceinline__ uint8_t init() { return 1; }
    __device__ __forceinline__ uint8_t operator()(uint8_t acc, T_in x) const {
        return (acc != 0 && x != T_in(0)) ? uint8_t(1) : uint8_t(0);
    }
};

template <>
struct AllReduce<__half> {
    static __device__ __forceinline__ uint8_t init() { return 1; }
    __device__ __forceinline__ uint8_t operator()(uint8_t acc, __half x) const {
        float xf = __half2float(x);
        return (acc != 0 && xf != 0.0f) ? uint8_t(1) : uint8_t(0);
    }
};

template <>
struct AllReduce<__nv_bfloat16> {
    static __device__ __forceinline__ uint8_t init() { return 1; }
    __device__ __forceinline__ uint8_t operator()(uint8_t acc, __nv_bfloat16 x) const {
        float xf = __bfloat162float(x);
        return (acc != 0 && xf != 0.0f) ? uint8_t(1) : uint8_t(0);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_REDUCE_ALL_INSTANTIATE(
    reduce_all_f32, float, baracuda::elementwise::AllReduce<float>)

BARACUDA_KERNELS_REDUCE_ALL_INSTANTIATE(
    reduce_all_f16, __half, baracuda::elementwise::AllReduce<__half>)

BARACUDA_KERNELS_REDUCE_ALL_INSTANTIATE(
    reduce_all_bf16, __nv_bfloat16, baracuda::elementwise::AllReduce<__nv_bfloat16>)

BARACUDA_KERNELS_REDUCE_ALL_INSTANTIATE(
    reduce_all_f64, double, baracuda::elementwise::AllReduce<double>)

BARACUDA_KERNELS_REDUCE_ALL_INSTANTIATE(
    reduce_all_i32, int32_t, baracuda::elementwise::AllReduce<int32_t>)

BARACUDA_KERNELS_REDUCE_ALL_INSTANTIATE(
    reduce_all_i64, int64_t, baracuda::elementwise::AllReduce<int64_t>)

BARACUDA_KERNELS_REDUCE_ALL_INSTANTIATE(
    reduce_all_bool, uint8_t, baracuda::elementwise::AllReduce<uint8_t>)
