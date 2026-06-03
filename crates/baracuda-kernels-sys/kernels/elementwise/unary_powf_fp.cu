// baracuda-kernels Phase 31 — elementwise float-exponent power (PowF).
//
// Forward: `y = pow(x, exponent)` where `exponent` is a runtime f32
// threaded as a single ABI parameter. Distinct from `unary_powi`
// because the integer-exponent path uses power-by-squaring; this
// variant calls the libdevice transcendental.
//
// Semantics:
//   * f32  → `__powf(x, exponent)`. Single-precision, fast (≤4 ULP for
//            most ranges) — same convention Fuel uses.
//   * f64  → `pow(x, (double)exponent)`. The exponent itself round-
//            trips exactly through f32 for any value PyTorch normally
//            ships.
//   * f16  → f32 detour, exactly as the rest of the unary family.
//   * bf16 → f32 detour.
//
// NaN policy: `pow` natively returns NaN for `pow(-x, non_integer)`,
// matches Fuel + PyTorch.
//
// ABI: `(numel, x, y, exponent, ws, ws_bytes, stream) -> int32` —
// single-parameter FFI (Fuel ask, distinct from the unary-param family
// which carries two f32 slots).

#include "../include/baracuda_elementwise.cuh"

namespace baracuda { namespace elementwise {

template <typename T>
struct PowFFunctor {
    float exponent;
    __host__ __device__ __forceinline__ PowFFunctor() : exponent(1.0f) {}
    __host__ __device__ __forceinline__ explicit PowFFunctor(float e) : exponent(e) {}
    __device__ __forceinline__ T operator()(T x) const { return x; }
};

template <>
struct PowFFunctor<float> {
    float exponent;
    __host__ __device__ __forceinline__ PowFFunctor() : exponent(1.0f) {}
    __host__ __device__ __forceinline__ explicit PowFFunctor(float e) : exponent(e) {}
    __device__ __forceinline__ float operator()(float x) const {
        return __powf(x, exponent);
    }
};

template <>
struct PowFFunctor<double> {
    float exponent;
    __host__ __device__ __forceinline__ PowFFunctor() : exponent(1.0f) {}
    __host__ __device__ __forceinline__ explicit PowFFunctor(float e) : exponent(e) {}
    __device__ __forceinline__ double operator()(double x) const {
        return pow(x, static_cast<double>(exponent));
    }
};

template <>
struct PowFFunctor<__half> {
    float exponent;
    __host__ __device__ __forceinline__ PowFFunctor() : exponent(1.0f) {}
    __host__ __device__ __forceinline__ explicit PowFFunctor(float e) : exponent(e) {}
    __device__ __forceinline__ __half operator()(__half x) const {
        float fx = __half2float(x);
        float fy = __powf(fx, exponent);
        return __float2half(fy);
    }
};

template <>
struct PowFFunctor<__nv_bfloat16> {
    float exponent;
    __host__ __device__ __forceinline__ PowFFunctor() : exponent(1.0f) {}
    __host__ __device__ __forceinline__ explicit PowFFunctor(float e) : exponent(e) {}
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 x) const {
        float fx = __bfloat162float(x);
        float fy = __powf(fx, exponent);
        return __float2bfloat16(fy);
    }
};

} } // namespace baracuda::elementwise

// =============================================================================
// Launchers — bespoke (NOT via UNARY_POINTWISE_INSTANTIATE) because the
// FFI surface carries `exponent` not via (p0, p1). Same shape as the
// ELU launchers (Phase 31).
// =============================================================================

#define BARACUDA_KERNELS_POWF_INSTANTIATE_CONTIG(SUFFIX, T)                                           \
    extern "C" int32_t baracuda_kernels_unary_powf_##SUFFIX##_run(                                   \
        int64_t numel,                                                                                \
        const void* x, void* y,                                                                       \
        float exponent,                                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                              \
        void* stream_ptr)                                                                             \
    {                                                                                                 \
        if (numel < 0) return 2;                                                                     \
        if (numel == 0) return 0;                                                                    \
        if (x == nullptr || y == nullptr) return 2;                                                   \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::elementwise::launch_unary_pointwise_contig<T, baracuda::elementwise::PowFFunctor<T>>( \
            static_cast<const T*>(x), static_cast<T*>(y), numel, stream,                              \
            baracuda::elementwise::PowFFunctor<T>(exponent));                                         \
    }                                                                                                 \
    extern "C" int32_t baracuda_kernels_unary_powf_##SUFFIX##_can_implement(                         \
        int64_t numel,                                                                                \
        const void* /*x*/, const void* /*y*/)                                                         \
    {                                                                                                 \
        if (numel < 0) return 2;                                                                     \
        return 0;                                                                                     \
    }

#define BARACUDA_KERNELS_POWF_INSTANTIATE_STRIDED(SUFFIX, T)                                          \
    extern "C" int32_t baracuda_kernels_unary_powf_##SUFFIX##_strided_run(                           \
        int64_t numel,                                                                                \
        int32_t rank,                                                                                 \
        const int32_t* shape,                                                                         \
        const int64_t* stride_x,                                                                      \
        const int64_t* stride_y,                                                                      \
        const void* x, void* y,                                                                       \
        float exponent,                                                                               \
        void* /*workspace*/, size_t /*workspace_bytes*/,                                              \
        void* stream_ptr)                                                                             \
    {                                                                                                 \
        if (numel < 0) return 2;                                                                     \
        if (numel == 0) return 0;                                                                    \
        if (x == nullptr || y == nullptr) return 2;                                                   \
        if (shape == nullptr || stride_x == nullptr || stride_y == nullptr) return 2;                 \
        cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);                                  \
        return baracuda::elementwise::launch_unary_pointwise_strided<T, baracuda::elementwise::PowFFunctor<T>>( \
            static_cast<const T*>(x), static_cast<T*>(y),                                             \
            numel, rank, shape, stride_x, stride_y, stream,                                           \
            baracuda::elementwise::PowFFunctor<T>(exponent));                                         \
    }                                                                                                 \
    extern "C" int32_t baracuda_kernels_unary_powf_##SUFFIX##_strided_can_implement(                 \
        int64_t numel,                                                                                \
        int32_t rank,                                                                                 \
        const int32_t* shape,                                                                         \
        const int64_t* stride_x,                                                                      \
        const int64_t* stride_y,                                                                      \
        const void* /*x*/, const void* /*y*/,                                                         \
        float /*exponent*/)                                                                           \
    {                                                                                                 \
        if (numel < 0) return 2;                                                                     \
        if (rank < 0) return 2;                                                                      \
        if (numel > 0 && (shape == nullptr || stride_x == nullptr ||                                  \
                           stride_y == nullptr)) return 2;                                            \
        return 0;                                                                                     \
    }

BARACUDA_KERNELS_POWF_INSTANTIATE_CONTIG(f32,  float)
BARACUDA_KERNELS_POWF_INSTANTIATE_CONTIG(f16,  __half)
BARACUDA_KERNELS_POWF_INSTANTIATE_CONTIG(bf16, __nv_bfloat16)
BARACUDA_KERNELS_POWF_INSTANTIATE_CONTIG(f64,  double)

BARACUDA_KERNELS_POWF_INSTANTIATE_STRIDED(f32,  float)
BARACUDA_KERNELS_POWF_INSTANTIATE_STRIDED(f16,  __half)
BARACUDA_KERNELS_POWF_INSTANTIATE_STRIDED(bf16, __nv_bfloat16)
BARACUDA_KERNELS_POWF_INSTANTIATE_STRIDED(f64,  double)
