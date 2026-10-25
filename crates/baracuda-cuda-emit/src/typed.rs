//! The CUDA spellings of [`ScalarExpr::Fma`], [`ScalarExpr::Cast`] and
//! [`ScalarExpr::Bitcast`] (ROADMAP SA-43 / SA-44; unpopped 0.24.0).
//!
//! Each seam spells **exactly** the requested semantics or declines; none falls back to a
//! plain C cast or to `a * b + c`, which have a different rounding / saturation contract.
//!
//! - **Fma**: `__fmaf_rn` / `__fma_rn`, one rounding. f16 / bf16 decline (a half FMA
//!   through the float detour rounds twice).
//! - **Cast**: the CUDA conversion intrinsics, which take the rounding direction in
//!   their name (`_rn`/`_rz`/`_rd`/`_ru`). Float to integer saturates and maps NaN to 0
//!   (the PTX `cvt` contract, which is also the IR's), narrow integer targets clamp the
//!   32-bit result, and integer `Clamp` goes through an explicit `min`/`max`.
//! - **Bitcast**: the `__*_as_*` reinterpret intrinsics (no pointer cast, no aliasing).
//!
//! The semantics being matched are `unpopped::cast_ref`'s; the end-to-end check is the
//! device test `tests/typed_exprs.rs`, which compares against that oracle.
//!
//! [`ScalarExpr::Fma`]: unpopped::ir::ScalarExpr::Fma
//! [`ScalarExpr::Cast`]: unpopped::ir::ScalarExpr::Cast
//! [`ScalarExpr::Bitcast`]: unpopped::ir::ScalarExpr::Bitcast

use unpopped::backend::{Decline, DeclinedOp, LowerError, Spelling};
use unpopped::ir::{CastSpec, RoundMode, Saturation};
use unpopped::types::{int_range, is_cast_float, typed_bits};
use unpopped_vocab::ElementKind;

/// The CUDA C type of a value in the typed table, or `None` outside it.
pub(crate) fn type_name(t: ElementKind) -> Option<String> {
    Some(
        match t {
            ElementKind::F32 | ElementKind::F32Strict => "float",
            ElementKind::F64 => "double",
            ElementKind::I8 => "signed char",
            ElementKind::I16 => "short",
            ElementKind::I32 => "int",
            ElementKind::I64 => "long long",
            ElementKind::U8 => "unsigned char",
            ElementKind::U16 => "unsigned short",
            ElementKind::U32 => "unsigned int",
            _ => return None,
        }
        .to_string(),
    )
}

fn decline(op: DeclinedOp, why: String) -> Result<Spelling, LowerError> {
    Ok(Spelling::Declined(Decline::UnsupportedOp { op, why }))
}

/// `a * b + c` with one rounding, f32.
pub(crate) fn fma_f32(a: String, b: String, c: String) -> Result<Spelling, LowerError> {
    Ok(Spelling::Spelled(format!("__fmaf_rn({a}, {b}, {c})")))
}

/// `a * b + c` with one rounding, f64.
pub(crate) fn fma_f64(a: String, b: String, c: String) -> Result<Spelling, LowerError> {
    Ok(Spelling::Spelled(format!("__fma_rn({a}, {b}, {c})")))
}

/// Every other compute dtype declines: spelling a half FMA as float math and narrowing
/// rounds twice, and an integer compute dtype has no float FMA at all.
pub(crate) fn fma_decline(_a: String, _b: String, _c: String) -> Result<Spelling, LowerError> {
    decline(
        DeclinedOp::Fma,
        "cuda backend: Fma is spelled only at f32 / f32s (__fmaf_rn) and f64 (__fma_rn); a \
         half FMA through the float detour would round twice, so f16 / bf16 decline"
            .to_string(),
    )
}

/// The rounding suffix of a CUDA conversion intrinsic.
fn rsuf(r: RoundMode) -> Option<&'static str> {
    Some(match r {
        RoundMode::NearestEven => "rn",
        RoundMode::TowardZero => "rz",
        RoundMode::TowardNegInf => "rd",
        RoundMode::TowardPosInf => "ru",
        _ => return None,
    })
}

/// A signed 64-bit literal that is also valid for `i64::MIN` in CUDA C.
fn ll(v: i128) -> String {
    if v == i128::from(i64::MIN) {
        "(-9223372036854775807LL - 1)".to_string()
    } else {
        format!("{v}LL")
    }
}

/// The largest finite f32 as a double literal.
const F32_MAX: &str = "3.4028234663852886e+38";

/// Spell `Cast(x)` from `from` per `spec`, or decline.
pub(crate) fn cast(x: String, from: ElementKind, spec: CastSpec) -> Result<Spelling, LowerError> {
    let to = spec.to;
    let (Some(_), Some(to_c)) = (type_name(from), type_name(to)) else {
        return decline(
            DeclinedOp::Cast,
            format!("cuda backend: no CUDA type for the Cast {from:?} -> {to:?}"),
        );
    };
    let Some(r) = rsuf(spec.round) else {
        return decline(
            DeclinedOp::Cast,
            format!(
                "cuda backend: no CUDA spelling for the rounding mode {:?}",
                spec.round
            ),
        );
    };
    let spelled = match (is_cast_float(from), is_cast_float(to)) {
        (true, true) => float_to_float(&x, from, to, spec.sat, r),
        (true, false) => float_to_int(&x, from, to, r),
        (false, true) => int_to_float(&x, from, to, r),
        (false, false) => int_to_int(&x, to, &to_c, spec.sat),
    };
    Ok(Spelling::Spelled(spelled))
}

fn float_to_float(x: &str, from: ElementKind, to: ElementKind, sat: Saturation, r: &str) -> String {
    let narrowing = typed_bits(from) > typed_bits(to);
    if narrowing {
        // f64 -> f32. ClampFinite clamps in double BEFORE narrowing, so an overflowing
        // or infinite input lands on the largest finite f32 and a directed round-up
        // cannot reach infinity; NaN passes through.
        let arg = if sat == Saturation::ClampFinite {
            format!("(({x}) != ({x}) ? ({x}) : fmin(fmax(({x}), -{F32_MAX}), {F32_MAX}))")
        } else {
            x.to_string()
        };
        format!("__double2float_{r}({arg})")
    } else if typed_bits(from) < typed_bits(to) {
        // f32 -> f64 is exact: every rounding mode agrees.
        format!("((double)({x}))")
    } else {
        // f32 <-> f32s: same bits.
        format!("({x})")
    }
}

fn float_to_int(x: &str, from: ElementKind, to: ElementKind, r: &str) -> String {
    let f = if from == ElementKind::F64 {
        "__double2"
    } else {
        "__float2"
    };
    let (lo, hi) = int_range(to).expect("table checked");
    let conv = match to {
        ElementKind::I64 => format!("{f}ll_{r}({x})"),
        ElementKind::U32 => format!("{f}uint_{r}({x})"),
        ElementKind::I32 => format!("{f}int_{r}({x})"),
        // 8 / 16-bit targets: the 32-bit conversion saturates, then the result is clamped
        // into the narrow range.
        _ => {
            let to_c = type_name(to).expect("table checked");
            return nan_to_zero(
                x,
                from,
                to,
                format!("(({to_c})min({hi}, max({lo}, {f}int_{r}({x}))))"),
            );
        }
    };
    nan_to_zero(x, from, to, conv)
}

/// The IR defines float-to-integer as saturating with NaN mapping to 0. The f32 -> 32-bit-or-
/// narrower intrinsics already return 0 for NaN; the 64-bit and double-source ones return
/// the type's minimum (measured on sm_89 against the oracle), so those get an explicit guard.
fn nan_to_zero(x: &str, from: ElementKind, to: ElementKind, conv: String) -> String {
    if from == ElementKind::F64 || to == ElementKind::I64 {
        format!("(({x}) != ({x}) ? 0 : {conv})")
    } else {
        conv
    }
}

fn int_to_float(x: &str, from: ElementKind, to: ElementKind, r: &str) -> String {
    let f64_out = to == ElementKind::F64;
    match from {
        ElementKind::I64 if f64_out => format!("__ll2double_{r}({x})"),
        ElementKind::I64 => format!("__ll2float_{r}({x})"),
        ElementKind::U32 if f64_out => format!("__uint2double_rn((unsigned int)({x}))"),
        ElementKind::U32 => format!("__uint2float_{r}((unsigned int)({x}))"),
        // i8 / i16 / i32 / u8 / u16 all fit an int exactly.
        _ if f64_out => format!("__int2double_rn((int)({x}))"),
        _ => format!("__int2float_{r}((int)({x}))"),
    }
}

fn int_to_int(x: &str, to: ElementKind, to_c: &str, sat: Saturation) -> String {
    match sat {
        Saturation::Clamp => {
            let (lo, hi) = int_range(to).expect("table checked");
            format!(
                "(({to_c})min((long long){}, max((long long){}, (long long)({x}))))",
                ll(hi),
                ll(lo)
            )
        }
        // Two's-complement truncation: nvcc / NVRTC wrap an out-of-range conversion.
        _ => format!("(({to_c})({x}))"),
    }
}

/// Spell `Bitcast(x)` from `from` to `to`, or decline.
pub(crate) fn bitcast(
    x: String,
    from: ElementKind,
    to: ElementKind,
) -> Result<Spelling, LowerError> {
    use ElementKind as E;
    let spelled = match (from, to) {
        (a, b) if a == b => format!("({x})"),
        (E::F32 | E::F32Strict, E::F32 | E::F32Strict) => format!("({x})"),
        (E::F32 | E::F32Strict, E::I32) => format!("__float_as_int({x})"),
        (E::F32 | E::F32Strict, E::U32) => format!("__float_as_uint({x})"),
        (E::I32, E::F32 | E::F32Strict) => format!("__int_as_float({x})"),
        (E::U32, E::F32 | E::F32Strict) => format!("__uint_as_float({x})"),
        (E::F64, E::I64) => format!("__double_as_longlong({x})"),
        (E::I64, E::F64) => format!("__longlong_as_double({x})"),
        // Same-width integers: a conversion that preserves the bit pattern.
        (E::I8, E::U8)
        | (E::U8, E::I8)
        | (E::I16, E::U16)
        | (E::U16, E::I16)
        | (E::I32, E::U32)
        | (E::U32, E::I32) => {
            let to_c = type_name(to).expect("table checked");
            format!("(({to_c})({x}))")
        }
        _ => {
            return decline(
                DeclinedOp::Bitcast,
                format!("cuda backend: no CUDA spelling for the Bitcast {from:?} -> {to:?}"),
            );
        }
    };
    Ok(Spelling::Spelled(spelled))
}
