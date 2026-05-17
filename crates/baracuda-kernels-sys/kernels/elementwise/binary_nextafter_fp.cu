// baracuda-kernels Phase 3 binary fanout: elementwise nextafter
// `y = nextafter(a, b)` — next representable value from `a` toward `b`.
//
// f32 → `nextafterf`, f64 → `nextafter`. f16 / bf16 do NOT use the
// f32-detour pattern — an f32 has many representable values between
// any two adjacent f16 / bf16 cells, so detouring through f32 would
// return a value that is itself the same f16 / bf16 after round-back.
// Instead the f16 / bf16 specializations implement IEEE 754 nextafter
// via direct bit-pattern manipulation on the half-width int.
//
// Algorithm (mirrors IEEE 754 nextafter semantics for both half forms):
//   - NaN in either operand          → NaN
//   - a == b                          → b
//   - a == ±0                         → smallest subnormal with sign(b)
//   - (a > 0) == (b > a), i.e. moving away from zero → bits++
//   - else (moving toward zero)      → bits--
//   - signed-zero cross-over handled via the explicit a == 0 branch.

#include "../include/baracuda_elementwise.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace baracuda { namespace elementwise {

template <typename T>
struct NextafterFunctor {
    __device__ __forceinline__ T operator()(T a, T b) const { return a; }
};

template <>
struct NextafterFunctor<float> {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return nextafterf(a, b);
    }
};

template <>
struct NextafterFunctor<double> {
    __device__ __forceinline__ double operator()(double a, double b) const {
        return nextafter(a, b);
    }
};

// f16 / bf16 bit-pattern nextafter. The half-width int reinterpretation
// uses the standard `__half_as_short` / `__bfloat16_as_short` intrinsics
// (which are 16-bit bitcasts despite the name). On Ampere+ these compile
// to a single `mov` — no actual memory traffic.

template <>
struct NextafterFunctor<__half> {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        // NaN propagation: any NaN input → NaN output.
        if (__hisnan(a) || __hisnan(b)) {
            return __float2half(__int_as_float(0x7fc00000)); // canonical f16 NaN via f32 quiet NaN
        }
        if (__heq(a, b)) {
            return b;
        }
        // f16 sign bit is bit 15; positive zero = 0x0000, negative zero = 0x8000.
        // After the equality check above, a == 0 covers both ±0 (both compare equal).
        const float af = __half2float(a);
        const float bf = __half2float(b);
        if (af == 0.0f) {
            // Smallest positive subnormal in f16 has bit pattern 0x0001
            // (value ≈ 5.96e-8). Apply sign of b.
            unsigned short bits = (bf > 0.0f) ? 0x0001u : 0x8001u;
            return __short_as_half(static_cast<short>(bits));
        }
        unsigned short bits = static_cast<unsigned short>(__half_as_short(a));
        // (af > 0) == (bf > af) ⇒ move away from zero (magnitude up) ⇒ bits++
        // else                  ⇒ move toward zero (magnitude down) ⇒ bits--
        const bool away = (af > 0.0f) == (bf > af);
        if (away) {
            bits = static_cast<unsigned short>(bits + 1u);
        } else {
            bits = static_cast<unsigned short>(bits - 1u);
        }
        return __short_as_half(static_cast<short>(bits));
    }
};

template <>
struct NextafterFunctor<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(__nv_bfloat16 a, __nv_bfloat16 b) const {
        if (__hisnan(a) || __hisnan(b)) {
            return __float2bfloat16(__int_as_float(0x7fc00000));
        }
        if (__heq(a, b)) {
            return b;
        }
        const float af = __bfloat162float(a);
        const float bf = __bfloat162float(b);
        if (af == 0.0f) {
            // Smallest positive subnormal in bf16 has bit pattern 0x0001
            // (value ≈ 9.18e-41).
            unsigned short bits = (bf > 0.0f) ? 0x0001u : 0x8001u;
            return __short_as_bfloat16(static_cast<short>(bits));
        }
        unsigned short bits = static_cast<unsigned short>(__bfloat16_as_short(a));
        const bool away = (af > 0.0f) == (bf > af);
        if (away) {
            bits = static_cast<unsigned short>(bits + 1u);
        } else {
            bits = static_cast<unsigned short>(bits - 1u);
        }
        return __short_as_bfloat16(static_cast<short>(bits));
    }
};

} } // namespace baracuda::elementwise

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_nextafter_f32, float, baracuda::elementwise::NextafterFunctor<float>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_nextafter_f32, float, baracuda::elementwise::NextafterFunctor<float>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_nextafter_f16, __half, baracuda::elementwise::NextafterFunctor<__half>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_nextafter_f16, __half, baracuda::elementwise::NextafterFunctor<__half>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_nextafter_bf16, __nv_bfloat16, baracuda::elementwise::NextafterFunctor<__nv_bfloat16>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_nextafter_bf16, __nv_bfloat16, baracuda::elementwise::NextafterFunctor<__nv_bfloat16>)

BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE(
    binary_nextafter_f64, double, baracuda::elementwise::NextafterFunctor<double>)
BARACUDA_KERNELS_BINARY_POINTWISE_INSTANTIATE_STRIDED(
    binary_nextafter_f64, double, baracuda::elementwise::NextafterFunctor<double>)
