// baracuda-kernels Phase 4 deferral 4.4: `count_nonzero` along one axis.
//
// `y[c_reduced] = Σ_j (x[..., j, ...] != 0 ? 1 : 0)`. Output dtype is
// `int64_t` (PyTorch `torch.count_nonzero` returns int64).
//
// Heterogeneous output dtype, dispatched through
// `reduce_axis_hetero_kernel<T_in, int64_t, F>`.
//
// Wired matrix: `CountNonzero × {f32, f16, bf16, f64, i32, i64, Bool}`
// — 7 SKUs.
//
// NaN semantics: `NaN != 0` is true for FP inputs (matches PyTorch).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// `CountNonzero` functor: init = 0, op accumulates `+ (x != 0 ? 1 : 0)`
// into an int64 running count.
template <typename T_in>
struct CountNonzeroReduce {
    static __device__ __forceinline__ int64_t init() { return 0; }
    __device__ __forceinline__ int64_t operator()(int64_t acc, T_in x) const {
        return acc + (int64_t)((x != T_in(0)) ? 1 : 0);
    }
};

template <>
struct CountNonzeroReduce<__half> {
    static __device__ __forceinline__ int64_t init() { return 0; }
    __device__ __forceinline__ int64_t operator()(int64_t acc, __half x) const {
        float xf = __half2float(x);
        // NaN != 0.0f is true → counted as non-zero (PyTorch convention).
        return acc + (int64_t)((xf != 0.0f) ? 1 : 0);
    }
};

template <>
struct CountNonzeroReduce<__nv_bfloat16> {
    static __device__ __forceinline__ int64_t init() { return 0; }
    __device__ __forceinline__ int64_t operator()(int64_t acc, __nv_bfloat16 x) const {
        float xf = __bfloat162float(x);
        return acc + (int64_t)((xf != 0.0f) ? 1 : 0);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Instantiations
// =============================================================================

BARACUDA_KERNELS_REDUCE_COUNT_NONZERO_INSTANTIATE(
    reduce_count_nonzero_f32, float, baracuda::elementwise::CountNonzeroReduce<float>)

BARACUDA_KERNELS_REDUCE_COUNT_NONZERO_INSTANTIATE(
    reduce_count_nonzero_f16, __half, baracuda::elementwise::CountNonzeroReduce<__half>)

BARACUDA_KERNELS_REDUCE_COUNT_NONZERO_INSTANTIATE(
    reduce_count_nonzero_bf16, __nv_bfloat16, baracuda::elementwise::CountNonzeroReduce<__nv_bfloat16>)

BARACUDA_KERNELS_REDUCE_COUNT_NONZERO_INSTANTIATE(
    reduce_count_nonzero_f64, double, baracuda::elementwise::CountNonzeroReduce<double>)

BARACUDA_KERNELS_REDUCE_COUNT_NONZERO_INSTANTIATE(
    reduce_count_nonzero_i32, int32_t, baracuda::elementwise::CountNonzeroReduce<int32_t>)

BARACUDA_KERNELS_REDUCE_COUNT_NONZERO_INSTANTIATE(
    reduce_count_nonzero_i64, int64_t, baracuda::elementwise::CountNonzeroReduce<int64_t>)

BARACUDA_KERNELS_REDUCE_COUNT_NONZERO_INSTANTIATE(
    reduce_count_nonzero_bool, uint8_t, baracuda::elementwise::CountNonzeroReduce<uint8_t>)
