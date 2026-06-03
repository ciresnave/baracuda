// baracuda-kernels Phase 3 unary fanout: elementwise ELU for FP types.
//
// Phase 31 (Fuel Phase 6c.2 storage.rs unblock): α is now a runtime
// parameter, no longer hardcoded to 1.0. The functor stores α at
// construction time so the per-thread `operator()(T x)` can stay
// `__device__ T(T)`-shaped — the contig and strided launchers each
// build a stateful `EluFunctor` from the `alpha: float` ABI arg and
// pass it through the standard unary pointwise launch templates.
//
// Implements `y = x if x > 0 else α·(exp(x) - 1)`. f32 uses `expf`;
// f64 uses `exp`. f16 / bf16 use the f32-detour with `expf` (same
// pattern as `unary_exp_fp.cu`). For f64, α is widened from the f32
// ABI to double at the functor's site of use.
//
// BREAKING CHANGE (Phase 31): `baracuda_kernels_unary_elu_<dtype>_run`
// and `..._strided_run` symbols gained an `alpha: float` parameter
// between the tensor pointers and the workspace pointers. Fuel is
// the sole external caller and has committed to migrating.

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

// Stateful α-bearing ELU functor. Default specialization is unused
// (we explicit-spec every FP dtype below) but keeps the template
// well-formed if a hypothetical TInt instantiation slipped in.
template <typename T>
struct EluFunctor {
    float alpha;
    __host__ __device__ __forceinline__ EluFunctor() : alpha(1.0f) {}
    __host__ __device__ __forceinline__ explicit EluFunctor(float a) : alpha(a) {}
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct EluFunctor<float> {
    float alpha;
    __host__ __device__ __forceinline__ EluFunctor() : alpha(1.0f) {}
    __host__ __device__ __forceinline__ explicit EluFunctor(float a) : alpha(a) {}
    __device__ __forceinline__ float operator()(float x) const {
        return (x > 0.0f) ? x : (alpha * (expf(x) - 1.0f));
    }
};

template <>
struct EluFunctor<double> {
    // Stored as f32 to match the ABI; widened at the use site.
    float alpha;
    __host__ __device__ __forceinline__ EluFunctor() : alpha(1.0f) {}
    __host__ __device__ __forceinline__ explicit EluFunctor(float a) : alpha(a) {}
    __device__ __forceinline__ double operator()(double x) const {
        return (x > 0.0) ? x : (static_cast<double>(alpha) * (exp(x) - 1.0));
    }
};

template <>
struct EluFunctor<__half> {
    float alpha;
    __host__ __device__ __forceinline__ EluFunctor() : alpha(1.0f) {}
    __host__ __device__ __forceinline__ explicit EluFunctor(float a) : alpha(a) {}
    __device__ __forceinline__ __half operator()(__half x) const {
        float f = __half2float(x);
        float y = (f > 0.0f) ? f : (alpha * (expf(f) - 1.0f));
        return __float2half(y);
    }
};

template <>
struct EluFunctor<__nv_bfloat16> {
    float alpha;
    __host__ __device__ __forceinline__ EluFunctor() : alpha(1.0f) {}
    __host__ __device__ __forceinline__ explicit EluFunctor(float a) : alpha(a) {}
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float f = __bfloat162float(x);
        float y = (f > 0.0f) ? f : (alpha * (expf(f) - 1.0f));
        return __float2bfloat16(y);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Launchers — bespoke (NOT via UNARY_POINTWISE_INSTANTIATE) because we
// need to thread α through, and the macro hardcodes a default-
// constructed functor. Same shape as the macro otherwise.
// =============================================================================

#define BARACUDA_KERNELS_ELU_INSTANTIATE_CONTIG(SUFFIX, T)                                            \
    extern "C" int32_t baracuda_kernels_unary_elu_##SUFFIX##_run(                                    \
        int64_t numel,                                                                                \
        const void* x, void* y,                                                                       \
        float alpha,                                                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                              \
        void* stream_ptr)                                                                             \
    {                                                                                                 \
        if (numel < 0) return 2;                                                                     \
        if (numel == 0) return 0;                                                                    \
        if (x == nullptr || y == nullptr) return 2;                                                   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::elementwise::launch_unary_pointwise_contig<T, baracuda::elementwise::EluFunctor<T>>( \
            static_cast<const T*>(x), static_cast<T*>(y), numel, stream,                              \
            baracuda::elementwise::EluFunctor<T>(alpha));                                             \
    }                                                                                                 \
    extern "C" int32_t baracuda_kernels_unary_elu_##SUFFIX##_can_implement(                          \
        int64_t numel,                                                                                \
        const void* /*x*/, const void* /*y*/)                                                         \
    {                                                                                                 \
        if (numel < 0) return 2;                                                                     \
        return 0;                                                                                     \
    }

#define BARACUDA_KERNELS_ELU_INSTANTIATE_STRIDED(SUFFIX, T)                                           \
    extern "C" int32_t baracuda_kernels_unary_elu_##SUFFIX##_strided_run(                            \
        int64_t numel,                                                                                \
        int32_t rank,                                                                                 \
        const int32_t* shape,                                                                         \
        const int64_t* stride_x,                                                                      \
        const int64_t* stride_y,                                                                      \
        const void* x, void* y,                                                                       \
        float alpha,                                                                                  \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                              \
        void* stream_ptr)                                                                             \
    {                                                                                                 \
        if (numel < 0) return 2;                                                                     \
        if (numel == 0) return 0;                                                                    \
        if (x == nullptr || y == nullptr) return 2;                                                   \
        if (shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::elementwise::launch_unary_pointwise_strided<T, baracuda::elementwise::EluFunctor<T>>( \
            static_cast<const T*>(x), static_cast<T*>(y),                                             \
            numel, rank, shape, stride_x, stride_y, stream,                                           \
            baracuda::elementwise::EluFunctor<T>(alpha));                                             \
    }                                                                                                 \
    extern "C" int32_t baracuda_kernels_unary_elu_##SUFFIX##_strided_can_implement(                  \
        int64_t numel,                                                                                \
        int32_t rank,                                                                                 \
        const int32_t* shape,                                                                         \
        const int64_t* stride_x,                                                                      \
        const int64_t* stride_y,                                                                      \
        const void* /*x*/, const void* /*y*/,                                                         \
        float /*alpha*/)                                                                              \
    {                                                                                                 \
        if (numel < 0) return 2;                                                                     \
        if (rank < 0) return 2;                                                                      \
        if (numel > 0 && (shape == nullptr || stride_x == nullptr ||                                  \
                           stride_y == nullptr)) return 2;                                            \
        return 0;                                                                                     \
    }

BARACUDA_KERNELS_ELU_INSTANTIATE_CONTIG(f32,  float)
BARACUDA_KERNELS_ELU_INSTANTIATE_CONTIG(f16,  __half)
BARACUDA_KERNELS_ELU_INSTANTIATE_CONTIG(bf16, __nv_bfloat16)
BARACUDA_KERNELS_ELU_INSTANTIATE_CONTIG(f64,  double)

BARACUDA_KERNELS_ELU_INSTANTIATE_STRIDED(f32,  float)
BARACUDA_KERNELS_ELU_INSTANTIATE_STRIDED(f16,  __half)
BARACUDA_KERNELS_ELU_INSTANTIATE_STRIDED(bf16, __nv_bfloat16)
BARACUDA_KERNELS_ELU_INSTANTIATE_STRIDED(f64,  double)
