// baracuda-kernels Phase 4 deferral 4.4: `any` along one axis.
//
// `y[c_reduced] = OR_j (x[..., j, ...] != 0)`. Output dtype is
// `uint8_t` (Bool — PyTorch convention: 0 = false, 1 = true).
//
// Heterogeneous output dtype, so dispatch goes through the
// `reduce_axis_hetero_kernel<T_in, T_out, F>` template (and not
// `reduce_axis_kernel<T, F>` which is same-dtype-in-out).
//
// Wired matrix: `Any × {f32, f16, bf16, f64, i32, i64, Bool}` — 7 SKUs.
//
// NaN semantics: `NaN != 0` is true for FP inputs (PyTorch convention).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// `Any` functor for FP input. `acc` is the bool accumulator (`uint8_t`,
// 0 or 1); on each new value, OR in `(x != 0)`. We treat NaN as truthy
// (matches PyTorch / IEEE 754: `NaN != 0` is true).
template <typename T_in>
struct AnyReduce {
    static __device__ __forceinline__ uint8_t init() { return 0; }
    __device__ __forceinline__ uint8_t operator()(uint8_t acc, T_in x) const {
        return (acc != 0 || x != T_in(0)) ? uint8_t(1) : uint8_t(0);
    }
};

// f16: detour through float for the comparison (`__half` doesn't have
// a direct `!= __half(0)` operator on the host-visible path that's
// portable; the f32 detour matches the rest of the f16 reduction
// kernels).
template <>
struct AnyReduce<__half> {
    static __device__ __forceinline__ uint8_t init() { return 0; }
    __device__ __forceinline__ uint8_t operator()(uint8_t acc, __half x) const {
        float xf = __half2float(x);
        // NaN != 0.0f is true, so this preserves PyTorch's
        // NaN-is-truthy convention.
        return (acc != 0 || xf != 0.0f) ? uint8_t(1) : uint8_t(0);
    }
};

template <>
struct AnyReduce<__nv_bfloat16> {
    static __device__ __forceinline__ uint8_t init() { return 0; }
    __device__ __forceinline__ uint8_t operator()(uint8_t acc, __nv_bfloat16 x) const {
        float xf = __bfloat162float(x);
        return (acc != 0 || xf != 0.0f) ? uint8_t(1) : uint8_t(0);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_REDUCE_ANY_INSTANTIATE(
    reduce_any_f32, float, baracuda::elementwise::AnyReduce<float>)

BARACUDA_KERNELS_REDUCE_ANY_INSTANTIATE(
    reduce_any_f16, __half, baracuda::elementwise::AnyReduce<__half>)

BARACUDA_KERNELS_REDUCE_ANY_INSTANTIATE(
    reduce_any_bf16, __nv_bfloat16, baracuda::elementwise::AnyReduce<__nv_bfloat16>)

BARACUDA_KERNELS_REDUCE_ANY_INSTANTIATE(
    reduce_any_f64, double, baracuda::elementwise::AnyReduce<double>)

BARACUDA_KERNELS_REDUCE_ANY_INSTANTIATE(
    reduce_any_i32, int32_t, baracuda::elementwise::AnyReduce<int32_t>)

BARACUDA_KERNELS_REDUCE_ANY_INSTANTIATE(
    reduce_any_i64, int64_t, baracuda::elementwise::AnyReduce<int64_t>)

BARACUDA_KERNELS_REDUCE_ANY_INSTANTIATE(
    reduce_any_bool, uint8_t, baracuda::elementwise::AnyReduce<uint8_t>)
