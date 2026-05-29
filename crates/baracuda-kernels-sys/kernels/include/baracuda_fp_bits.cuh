// baracuda_fp_bits.cuh
//
// Phase 44b — FP bit-twiddling helpers consolidated for baracuda's
// internal use. Provides templated access to the bit-layout of `half`
// / `float` / `double` (and the symmetric uint types) plus a generic
// "truncate the mantissa to N bits with a configurable rounding mode"
// utility and the TF32 (10-bit-mantissa) convenience wrapper. The
// 128-bit unsigned integer used by the Ozaki-scheme splitter
// (`split.cu`) also lives here so MSVC can compile that path without
// the GCC/Clang-only `__uint128_t` extension.
//
// **Provenance.** The bit-layout / mantissa-truncation / TF32 helpers
// are derived from Hiroyuki Ootomo's `cutf` utility headers
// (`include/cutf/experimental/{fp,exponent,mantissa,tf32}.hpp`),
// which baracuda used to consume via the `cutf` git submodule
// pulled in alongside the vendored ozIMMU library. `cutf` upstream
// (https://gitlab.momo86.net/mutsuki/cutf) went offline during the
// Phase 44 → 44b transition and we don't expect it to return, so the
// ~360 lines we still cared about have been folded into baracuda
// proper. The original code is MIT-licensed; we have re-styled it to
// match baracuda conventions and added doc-comments per the workspace
// style guide.
//
//   - Original repo: https://gitlab.momo86.net/mutsuki/cutf  (offline)
//   - Author: Hiroyuki Ootomo
//   - License: MIT
//   - Algorithmic reference: Ootomo / Ozaki / Yokota,
//     "DGEMM on Integer Matrix Multiplication Unit",
//     IJHPCA 2024 (arXiv:2306.11975).
//
// The portable Uint128 struct is a new baracuda contribution — `cutf`
// upstream used the GCC/Clang `__uint128_t` extension directly; we
// fall through to it on Linux (preserving bit-for-bit behavior with
// the prior code path) and use a struct emulator on MSVC.
//
// **Scope.** The helpers here only do bit-shuffling. No transcendentals,
// no atomics, no library calls. The Ozaki splitter is the primary
// consumer; future quantization / cast kernels can reuse the mantissa
// truncator instead of rolling their own.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace baracuda {
namespace fp {
namespace detail {

/// Type-punning helper used by `reinterpret_as_uint` / `reinterpret_as_fp`.
/// A `union` is the only standards-blessed way to do this in C++14 and
/// the only way that the NVCC optimizer reliably recognizes as a
/// no-op move-to-register on every arch.
template <class FpT, class BsT>
union reinterpret_medium {
    FpT fp;
    BsT bs;
};

}  // namespace detail

/// Same-size unsigned integer companion type — for `half` it's
/// `uint16_t`, for `float` it's `uint32_t`, for `double` it's
/// `uint64_t`. The default specialization picks `uint32_t` (matches
/// the `float` case) and exists only so the generic template
/// compiles cleanly when probed with non-FP types in dead code paths.
template <class T> struct same_size_uint { using type = uint32_t; };
template <> struct same_size_uint<half>   { using type = uint16_t; };
template <> struct same_size_uint<float>  { using type = uint32_t; };
template <> struct same_size_uint<double> { using type = uint64_t; };

/// Symmetric inverse of `same_size_uint`: given an unsigned-int width,
/// pick the FP type with the matching binary layout.
template <class T> struct same_size_fp { using type = float; };
template <> struct same_size_fp<uint16_t> { using type = half;   };
template <> struct same_size_fp<uint32_t> { using type = float;  };
template <> struct same_size_fp<uint64_t> { using type = double; };

/// `sizeof(T)` lifted into a template trait so the Ozaki splitter can
/// pick the right shift width for a mixed-precision mantissa unpack.
/// Default specialization returns `0` so unintended uses fail at
/// compile time on a `static_assert(value != 0)` check downstream.
template <class T> struct size_of { static const unsigned value = 0; };
template <> struct size_of<half>     { static const unsigned value = 2; };
template <> struct size_of<float>    { static const unsigned value = 4; };
template <> struct size_of<double>   { static const unsigned value = 8; };
template <> struct size_of<uint8_t>  { static const unsigned value = 1; };
template <> struct size_of<uint16_t> { static const unsigned value = 2; };
template <> struct size_of<uint32_t> { static const unsigned value = 4; };
template <> struct size_of<uint64_t> { static const unsigned value = 8; };
template <> struct size_of<int8_t>   { static const unsigned value = 1; };
template <> struct size_of<int16_t>  { static const unsigned value = 2; };
template <> struct size_of<int32_t>  { static const unsigned value = 4; };
template <> struct size_of<int64_t>  { static const unsigned value = 8; };
// NOTE: `__uint128_t` / `__int128_t` specializations are intentionally
// omitted — neither is supported on MSVC and no code path references
// them after the Phase 44b Uint128 portability work. The Ozaki
// splitter uses `sizeof(Uint128)` directly (which is 16 by struct
// layout).

/// Width of the exponent field, in bits, for the given FP type.
template <class T> __device__ __host__ inline unsigned get_exponent_size();
template <> __device__ __host__ inline unsigned get_exponent_size<half>()   { return 5;  }
template <> __device__ __host__ inline unsigned get_exponent_size<float>()  { return 8;  }
template <> __device__ __host__ inline unsigned get_exponent_size<double>() { return 11; }

/// Width of the mantissa field, in bits, for the given FP type.
/// (Excludes the implicit leading 1 — IEEE-754 normal numbers.)
template <class T> __device__ __host__ inline unsigned get_mantissa_size();
template <> __device__ __host__ inline unsigned get_mantissa_size<half>()   { return 10; }
template <> __device__ __host__ inline unsigned get_mantissa_size<float>()  { return 23; }
template <> __device__ __host__ inline unsigned get_mantissa_size<double>() { return 52; }

/// Exponent bias for the given FP type. (`half` = 15, `float` = 127,
/// `double` = 1023 — the constant subtracted from the stored exponent
/// to get the unbiased exponent.)
template <class T> __device__ __host__ inline unsigned get_bias();
template <> __device__ __host__ inline unsigned get_bias<half>()   { return 0xf;   }
template <> __device__ __host__ inline unsigned get_bias<float>()  { return 0x7f;  }
template <> __device__ __host__ inline unsigned get_bias<double>() { return 0x3ff; }

/// Reinterpret the bits of `fp` as an unsigned integer of matching
/// width. Cheaper than `memcpy` on every arch we target; relies on
/// the `union` punning idiom which NVCC's optimizer collapses to a
/// register move.
template <class T>
__device__ __host__ inline typename same_size_uint<T>::type
reinterpret_as_uint(const T fp) {
    return detail::reinterpret_medium<T, typename same_size_uint<T>::type>{ .fp = fp }.bs;
}

/// Inverse of `reinterpret_as_uint`: take an unsigned-int bit pattern
/// and reinterpret as the same-width FP type.
template <class T>
__device__ __host__ inline typename same_size_fp<T>::type
reinterpret_as_fp(const T bs) {
    return detail::reinterpret_medium<typename same_size_fp<T>::type, T>{ .bs = bs }.fp;
}

/// `uint16_t -> half` specialization. The default template would
/// generate the right code on every device compiler we've seen, but
/// some older host compilers stumble on the implicit-`half`-init in
/// the union, so we route through a raw pointer reinterpret here. The
/// behaviour is identical at the bit level.
template <>
__device__ __host__ inline typename same_size_fp<uint16_t>::type
reinterpret_as_fp<uint16_t>(const uint16_t bs) {
    return *reinterpret_cast<const half *>(&bs);
}

/// Extract the mantissa bits (low `get_mantissa_size<T>()` bits) of
/// `fp` as an unsigned integer. Excludes the implicit leading 1.
template <class T>
__device__ __host__ inline typename same_size_uint<T>::type
mask_mantissa(const T fp) {
    const auto u = reinterpret_as_uint(fp);
    const auto mask = (decltype(u)(1) << get_mantissa_size<T>()) - 1;
    return u & mask;
}

/// Extract the exponent bits of `fp` as an unsigned integer — the
/// result is already shifted into the exponent position (i.e. it's
/// the raw IEEE-754 biased exponent left-shifted by the mantissa
/// width). Useful for re-assembling FP values after sign + exponent
/// manipulation.
template <class T>
__device__ __host__ inline typename same_size_uint<T>::type
mask_exponent(const T fp) {
    const auto u = reinterpret_as_uint(fp);
    const auto mask = ((decltype(u)(1) << get_exponent_size<T>()) - 1)
                        << get_mantissa_size<T>();
    return u & mask;
}

/// Extract the sign bit of `fp` — left-shifted into the high bit of
/// the same-width unsigned integer (so `mask_sign(x) | mask_exponent(x) | mask_mantissa(x)`
/// reconstructs `x`).
template <class T>
__device__ __host__ inline typename same_size_uint<T>::type
mask_sign(const T fp) {
    const auto u = reinterpret_as_uint(fp);
    const auto mask = decltype(u)(1) << (sizeof(decltype(u)) * 8 - 1);
    return u & mask;
}

}  // namespace fp
}  // namespace baracuda

// =============================================================================
// Rounding-mode tags (used by `baracuda::mantissa::cut_mantissa<N, R>`)
// =============================================================================

namespace baracuda {
namespace rounding {

/// Round to nearest, ties away from zero (a.k.a. "rr" in the cutf
/// vocabulary — "round-up-on-tie").
struct rr;
/// Round down (toward -inf).
struct rd;
/// Round to nearest, ties to even (IEEE-754 default).
struct rn;
/// Round up (toward +inf).
struct ru;
/// Round toward zero (truncate).
struct rz;
/// Bias-corrected force-1 rounding (rare; only used by exotic
/// quantization paths).
struct rb;

}  // namespace rounding
}  // namespace baracuda

// =============================================================================
// Exponent clamping (`baracuda::exponent::min_exponent`)
// =============================================================================

namespace baracuda {
namespace exponent {

/// If the unbiased exponent of `v` is below `min_e`, return a signed
/// zero matching `v`'s sign; otherwise return `v` untouched. Used by
/// quantization paths that need to flush subnormal-ish values without
/// losing the sign for downstream cancellation accounting.
template <class T>
__device__ __host__ inline T min_exponent(const T v, const int min_e) {
    const auto bitstring = baracuda::fp::reinterpret_as_uint(v);
    const auto exponent = ((bitstring << 1) >> (1 + baracuda::fp::get_mantissa_size<T>()));
    const auto sp_exponent = static_cast<int>(exponent)
                           - static_cast<int>(baracuda::fp::get_bias<T>());
    if (sp_exponent < min_e) {
        // Multiply by zero of matching type to keep the sign bit.
        return v * static_cast<T>(0);
    }
    return v;
}

}  // namespace exponent
}  // namespace baracuda

// =============================================================================
// Mantissa truncation (`baracuda::mantissa::cut_mantissa<N, R>`)
// =============================================================================

namespace baracuda {
namespace mantissa {
namespace detail {

template <class T>
__device__ __host__ inline T
adjust_mantissa(const T mantissa,
                const T mantissa_mask,
                const uint32_t carry_bit,
                T &move_up) {
    move_up = (mantissa >> carry_bit) & 0x1;
    return mantissa & mantissa_mask;
}

template <class Rounding>
__device__ __host__ inline uint32_t
rounding_mantissa(const uint32_t fp_bitstring, const uint32_t cut_length, uint32_t &move_up);

template <>
__device__ __host__ inline uint32_t
rounding_mantissa<baracuda::rounding::rz>(const uint32_t fp_bitstring,
                                          const uint32_t cut_length,
                                          uint32_t &move_up) {
    move_up = 0;
    return (fp_bitstring &
            (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)));
}

template <>
__device__ __host__ inline uint32_t
rounding_mantissa<baracuda::rounding::rr>(const uint32_t fp_bitstring,
                                          const uint32_t cut_length,
                                          uint32_t &move_up) {
    const uint32_t m0 = (fp_bitstring &
            (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)));
    const uint32_t c0 = (fp_bitstring & (1u << (cut_length - 1)));
    const uint32_t m1 = m0 + (c0 << 1);
    return adjust_mantissa(m1,
            (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)),
            23, move_up);
}

template <>
__device__ __host__ inline uint32_t
rounding_mantissa<baracuda::rounding::rn>(const uint32_t fp_bitstring,
                                          const uint32_t cut_length,
                                          uint32_t &move_up) {
    const uint32_t m0 = (fp_bitstring &
            (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)));
    const uint32_t c0 = (fp_bitstring & (1u << cut_length));
    const uint32_t m1 = m0 + c0;
    return adjust_mantissa(m1,
            (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)),
            23, move_up);
}

template <>
__device__ __host__ inline uint32_t
rounding_mantissa<baracuda::rounding::rb>(const uint32_t fp_bitstring,
                                          const uint32_t cut_length,
                                          uint32_t &move_up) {
    const uint32_t m0 = (fp_bitstring &
            (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)));
    const uint32_t m0_res = (fp_bitstring & ((1u << cut_length) - 1));
    const uint32_t c0 = (m0_res != 0) ? (1u << cut_length) : 0;
    const uint32_t m1 = m0 + c0;
    return adjust_mantissa(m1,
            (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)),
            23, move_up);
}

template <class Rounding>
__device__ __host__ inline uint64_t
rounding_mantissa(const uint64_t fp_bitstring, const uint64_t cut_length, uint64_t &move_up);

template <>
__device__ __host__ inline uint64_t
rounding_mantissa<baracuda::rounding::rz>(const uint64_t fp_bitstring,
                                          const uint64_t cut_length,
                                          uint64_t &move_up) {
    move_up = 0;
    return (fp_bitstring & (0x000fffffffffffffllu - ((1llu << cut_length) - 1)));
}

template <>
__device__ __host__ inline uint64_t
rounding_mantissa<baracuda::rounding::rr>(const uint64_t fp_bitstring,
                                          const uint64_t cut_length,
                                          uint64_t &move_up) {
    const uint64_t m0 = (fp_bitstring & (0x000fffffffffffffllu - ((1llu << cut_length) - 1)));
    const uint64_t c0 = (fp_bitstring & (1ull << (cut_length - 1)));
    const uint64_t m1 = m0 + (c0 << 1);
    return adjust_mantissa(m1, (0x000fffffffffffffllu - ((1llu << cut_length) - 1)),
                           53, move_up);
}

template <>
__device__ __host__ inline uint64_t
rounding_mantissa<baracuda::rounding::rn>(const uint64_t fp_bitstring,
                                          const uint64_t cut_length,
                                          uint64_t &move_up) {
    const uint64_t m0 = (fp_bitstring & (0x000fffffffffffffllu - ((1llu << cut_length) - 1)));
    const uint64_t c0 = (fp_bitstring & (1ull << cut_length));
    const uint64_t m1 = m0 + c0;
    return adjust_mantissa(m1, (0x000fffffffffffffllu - ((1llu << cut_length) - 1)),
                           53, move_up);
}

template <>
__device__ __host__ inline uint64_t
rounding_mantissa<baracuda::rounding::rb>(const uint64_t fp_bitstring,
                                          const uint64_t cut_length,
                                          uint64_t &move_up) {
    const uint64_t m0 = (fp_bitstring & (0x000fffffffffffffllu - ((1llu << cut_length) - 1)));
    const uint64_t m0_res = (fp_bitstring & ((1ull << cut_length) - 1));
    const uint64_t c0 = (m0_res != 0) ? (1ull << cut_length) : 0;
    const uint64_t m1 = m0 + c0;
    return adjust_mantissa(m1, (0x000fffffffffffffllu - ((1llu << cut_length) - 1)),
                           53, move_up);
}

}  // namespace detail

/// Truncate a `float`'s mantissa down to `MantissaLength` bits using
/// the rounding mode `Rounding` (defaults to `rr` = round-half-away-
/// from-zero). The exponent and sign are preserved. `MantissaLength`
/// must be `0 < N < 23`.
///
/// Used by `to_tf32` (MantissaLength = 10) and by various low-bit
/// quantization paths.
template <unsigned MantissaLength, class Rounding = baracuda::rounding::rr>
__device__ __host__ inline float cut_mantissa(const float v) {
    static_assert(MantissaLength > 0,  "MantissaLength must be greater than 0");
    static_assert(MantissaLength < 23, "MantissaLength must be smaller than 23");

    constexpr unsigned cut_length = 23u - MantissaLength;
    const uint32_t in = baracuda::fp::reinterpret_as_uint(v);
    const uint32_t e  = (in & 0b0'11111111'00000000000000000000000u);
    const uint32_t s  = (in & 0b1'00000000'00000000000000000000000u);

    uint32_t c1;
    const uint32_t m_pre = detail::rounding_mantissa<Rounding>(in, cut_length, c1);
    const uint32_t e_pre = e + (c1 << 23);

    return baracuda::fp::reinterpret_as_fp(s | m_pre | e_pre);
}

/// Truncate a `double`'s mantissa down to `MantissaLength` bits using
/// the rounding mode `Rounding`. `MantissaLength` must be `0 < N < 52`.
template <unsigned MantissaLength, class Rounding = baracuda::rounding::rr>
__device__ __host__ inline double cut_mantissa(const double v) {
    static_assert(MantissaLength > 0,  "MantissaLength must be greater than 0");
    static_assert(MantissaLength < 52, "MantissaLength must be smaller than 52");

    constexpr unsigned cut_length = 52u - MantissaLength;
    const uint64_t in = baracuda::fp::reinterpret_as_uint(v);
    const uint64_t e  = (in & 0xfff0000000000000llu);
    const uint64_t s  = (in & 0x8000000000000000llu);

    uint64_t c1;
    const uint64_t m_pre = detail::rounding_mantissa<Rounding>(in, cut_length, c1);
    const uint64_t e_pre = e + (c1 << 52);

    return baracuda::fp::reinterpret_as_fp(s | m_pre | e_pre);
}

}  // namespace mantissa
}  // namespace baracuda

// =============================================================================
// TF32 (`baracuda::tf32`)
// =============================================================================

namespace baracuda {
namespace tf32 {

/// TF32 ("tensor float 32") — a `float` with the mantissa truncated to
/// 10 bits. Same exponent range as `float` (so Inf/NaN/subnormal
/// behaviour matches), but the reduced mantissa lets Ampere's TF32
/// tensor-core path consume it as if it were `half` precision.
/// Storage type is `float`.
using tf32_t = float;

/// Cast a `float` to TF32, truncating the mantissa.
__device__ __host__ inline tf32_t to_tf32(const float v) {
    return baracuda::mantissa::cut_mantissa<10>(v);
}

}  // namespace tf32
}  // namespace baracuda

// =============================================================================
// Portable 128-bit unsigned integer (Phase 44b — Windows port)
// =============================================================================
//
// The Ozaki splitter (`split.cu`) needs a 128-bit holding register so
// it can stage a full `double` mantissa (52 bits) plus 8-bit slack
// before unpacking it into `int8_t` lanes. Upstream uses GCC's
// `__uint128_t` builtin; MSVC has no equivalent type. The struct
// below covers exactly the operations the splitter exercises: cast
// from `uint64_t`, left/right shift by a variable amount, truncate
// to `int8_t`, and `sizeof`.
//
// Determinism note: the GCC `__uint128_t` path is preserved verbatim
// on Linux (typedef alias) so the Ozaki output is bit-for-bit
// unchanged from alpha.56. The struct path on MSVC matches the same
// operations modulo the platform's wider-int codegen, which is
// trivial since the operations involved are pure bit-shuffle.

#if defined(__GNUC__) || defined(__clang__)

namespace baracuda {
/// 128-bit unsigned integer alias — bit-identical to the prior
/// `__uint128_t` path on Linux. Use `baracuda::Uint128` instead of
/// the raw `__uint128_t` extension so the Windows fall-back kicks in
/// automatically when building under MSVC.
using Uint128 = __uint128_t;
}  // namespace baracuda

#else  // !__GNUC__ && !__clang__ (MSVC)

#include <cstring>

namespace baracuda {

/// Portable 128-bit unsigned integer for the Ozaki splitter. Supports
/// exactly the operations the splitter uses (construction from
/// `uint64_t`, left / right shift by a variable amount,
/// truncating cast to `int8_t`, and `sizeof`). Not a full `uint128`
/// implementation — no multiply, divide, add, or compare.
///
/// Layout: little-endian limbs (`lo` is the low 64 bits, `hi` is the
/// high 64 bits). Matches what the GCC `__uint128_t` extension
/// produces on x86-64 Linux, so swapping out the typedef alias does
/// not change the splitter's per-thread arithmetic.
struct Uint128 {
    uint64_t lo;
    uint64_t hi;

    __device__ __host__ Uint128() : lo(0), hi(0) {}
    __device__ __host__ Uint128(const uint64_t v) : lo(v), hi(0) {}

    /// Truncating cast to `int8_t` — keeps the low 8 bits of `lo`,
    /// re-interpreted as a signed byte. The splitter always shifts
    /// the desired byte down to position 0 before casting, so the
    /// "drop the rest" behaviour matches the GCC builtin.
    __device__ __host__ explicit operator int8_t() const {
        return static_cast<int8_t>(static_cast<uint8_t>(lo & 0xff));
    }

    /// Left shift by `s` bits. Splits at the 64-bit boundary.
    /// `s == 0` is a no-op; `s >= 128` produces zero (matches the
    /// GCC builtin's unsigned shift semantics).
    __device__ __host__ Uint128 operator<<(unsigned s) const {
        Uint128 r;
        if (s == 0) {
            r = *this;
        } else if (s >= 128) {
            r.lo = 0;
            r.hi = 0;
        } else if (s >= 64) {
            r.lo = 0;
            r.hi = lo << (s - 64);
        } else {
            r.lo = lo << s;
            r.hi = (hi << s) | (lo >> (64 - s));
        }
        return r;
    }
    __device__ __host__ Uint128 &operator<<=(unsigned s) {
        *this = *this << s;
        return *this;
    }

    /// Right shift by `s` bits (logical — zero-fill from the top).
    __device__ __host__ Uint128 operator>>(unsigned s) const {
        Uint128 r;
        if (s == 0) {
            r = *this;
        } else if (s >= 128) {
            r.lo = 0;
            r.hi = 0;
        } else if (s >= 64) {
            r.lo = hi >> (s - 64);
            r.hi = 0;
        } else {
            r.lo = (lo >> s) | (hi << (64 - s));
            r.hi = hi >> s;
        }
        return r;
    }
    __device__ __host__ Uint128 &operator>>=(unsigned s) {
        *this = *this >> s;
        return *this;
    }
};

}  // namespace baracuda

// `sizeof(baracuda::Uint128)` is 16 by struct layout. Verify so the
// splitter's shift-width math (`sizeof(MANTISSA_T) * 8`) matches the
// `__uint128_t` path on Linux.
static_assert(sizeof(baracuda::Uint128) == 16,
              "baracuda::Uint128 must be exactly 16 bytes (matches __uint128_t)");

#endif  // __GNUC__ || __clang__
