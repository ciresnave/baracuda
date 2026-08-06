//! Shared **C-family scalar-op speller vocabulary** — the neutral scalar
//! syntax (f32/f64/int math, the C ternary `select`, ctype names, and
//! runtime-param spelling) that every C-family backend emits verbatim. These
//! spellers are the single source of truth consumed by the CUDA backend
//! (`crate::cuda`), the portable-C CpuC reference backend (`crate::cpu_c`), and
//! the Slang backend (`crate::slang`), so a per-op spelling can never drift
//! between the three.
//!
//! The vocabulary is deliberately backend-neutral: it depends only on the
//! neutral IR (`crate::ir`), the plan (`crate::plan::KernelPlan`), and the
//! op/dtype vocab (`baracuda_kernel_vocab`) — never on any CUDA launch
//! harness. It is the module the standalone kernel generator (Unpopped) keeps
//! when the CUDA-specific emitter is later carved into its own crate.
//!
//! # KNOWN WART / follow-up: the f16/bf16 arms are NOT neutral
//!
//! `scalar_ctype`'s `F16`/`Bf16` arms return the NVIDIA type spellings `__half`
//! / `__nv_bfloat16`, and the half load/store tail (`half_load_intrinsic`,
//! `half_store_intrinsic`, `promote_load_f32`, `demote_store_f32`, and
//! `cast_scalar`'s half arms) emits `__half2float`-class CUDA intrinsics. Once
//! the CUDA backend is carved out, these arms are reachable ONLY from the CUDA
//! backend — CpuC declines f16/bf16 (`supports_dtype`) and Slang never calls
//! them — so no neutral backend exercises or tests them. That makes them a
//! **silent-wrong-output hazard**: the first non-CUDA backend that supports f16
//! (e.g. a Vulkan/SPIR-V backend) would call a neutral-looking API and silently
//! get `__half` spelled into its output, with no test able to catch it.
//!
//! **FOLLOW-UP (deliberately deferred out of the byte-identity-critical
//! extraction move): abstract the f16/bf16 ctype and the half load/store
//! intrinsics behind the `Backend` trait**, so a non-CUDA f16 backend supplies
//! its own spelling and cfamily's neutral core declines f16 rather than
//! mis-spelling it.
//!

use crate::ir::{BinaryOp, ScalarExpr, UnaryOp, is_admissible_int_reduction_operand};
use crate::plan::KernelPlan;
use baracuda_kernel_vocab::ElementKind;

/// CUDA scalar type for a dtype, or `None` if the backend can't lower it yet.
/// `U8` (increment 0b) is the comparison-predicate mask dtype — `unsigned char`
/// per the FKC §5 Bool→U8 pinning — and, since increment 0c, an audited
/// COMPUTE dtype (wrapping mod-256 C semantics), same class as the i32/i64
/// arms. `S8` (FKC `I8`, increment 0c) is `signed char` — two's-complement
/// wrapping via integer promotion + store truncation (see the ir.rs table).
pub(crate) fn scalar_ctype(dt: ElementKind) -> Option<&'static str> {
    Some(match dt {
        ElementKind::F32 | ElementKind::F32Strict => "float",
        ElementKind::F64 => "double",
        ElementKind::F16 => "__half",
        ElementKind::Bf16 => "__nv_bfloat16",
        ElementKind::I32 => "int",
        ElementKind::I64 => "long long",
        ElementKind::S8 => "signed char",
        ElementKind::U8 => "unsigned char",
        // U32 is the gather/scatter INDEX-operand ctype (`unsigned int`) — a
        // 4-byte address dtype used ONLY for the index-load pointer type (the
        // Model-A u32-index path), never a compute operand. It has no `Element`
        // impl and no vector/packed path; a compute op never keys `plan.dtype =
        // U32` (no constructor builds one), so this arm serves the index load.
        ElementKind::U32 => "unsigned int",
        _ => return None,
    })
}

/// Short dtype tag for generated symbol names. Only called for dtypes that pass
/// [`scalar_ctype`].
pub(crate) fn dtype_tag(dt: ElementKind) -> &'static str {
    match dt {
        ElementKind::F32 => "f32",
        ElementKind::F32Strict => "f32s",
        ElementKind::F64 => "f64",
        ElementKind::F16 => "f16",
        ElementKind::Bf16 => "bf16",
        ElementKind::I32 => "i32",
        ElementKind::I64 => "i64",
        ElementKind::S8 => "i8",
        ElementKind::U8 => "u8",
        // U32 index-dtype infix: `gather_f32_u32` (the Fuel-facing u32-index
        // variant's entry_point symbol).
        ElementKind::U32 => "u32",
        _ => "x",
    }
}

/// Output `j`'s pointer scalar C type — the input `ctype` for a uniform output
/// (`out_dtype_of(j) == plan.dtype`), `unsigned char` for a u8-predicate/keep-mask
/// output (increment 0b single-output; the hetero multi-output / dropout-class
/// increment per-output). `out_ctype_of(plan, 0, ctype)` is byte-identical to the
/// pre-generalization `out_ctype` (output 0's dtype is `plan.out_dtype`), so every
/// single-output + uniform-multi emitter that passes `j = 0`/a uniform `j` is
/// unchanged.
pub(crate) fn out_ctype_of<'c>(plan: &KernelPlan<'_>, j: usize, ctype: &'c str) -> &'c str {
    let d = plan.out_dtype_of(j);
    if d == plan.dtype {
        ctype
    } else {
        scalar_ctype(d).expect("validated out dtype has a scalar ctype")
    }
}

/// The store expression for output `j`'s lowered body root. A uniform output
/// (`out_dtype_of(j) == plan.dtype`) stores the root unchanged (byte-identical to
/// pre-0b output). A u8 keep-mask output converts the exact 0.0/1.0 predicate
/// (lowered in the COMPUTE dtype `plan.dtype`) to `unsigned char` — exact by
/// construction (the G1 plan gate + G5 backstop pin the body root to a `Cmp*`).
/// The conversion is applied HERE at the store site, per output, **never baked
/// into the shared DAG node**: for dropout the same compute-dtype `Cmp*` temp is
/// consumed by output 0 inside a `Select` (tested `!= 0.0f`) AND stored as
/// `(unsigned char)` by output 1, so a cast on the shared node would corrupt
/// output 0's value (mutation M9). f16/bf16 re-promote first: the root lowered in
/// the house promote-demote convention is the demoted `__float2half(<pred_f32>)`,
/// and 1.0/0.0 round-trip f32→half→f32 bit-exactly, so the conversion pair is
/// value-exact (and folded by ptxas). `store_expr_of(plan, 0, root)` is
/// byte-identical to the pre-generalization `store_expr`.
pub(crate) fn store_expr_of(plan: &KernelPlan<'_>, j: usize, root: String) -> String {
    let d = plan.out_dtype_of(j);
    if d == plan.dtype {
        return root;
    }
    // The hetero elementwise store is exactly the U8 keep-mask (a `Cmp*`
    // predicate, pinned by `assert_valid_out_dtype`; the bincount-I32 scatter
    // narrows itself via `scatter_combine_store`, never through here). The
    // per-element narrowing is the shared [`cast_scalar`] routine — one source of
    // truth with the generated cast helper ([`emit_cast_helper`]). BYTE-IDENTICAL
    // to the prior hand-inlined forms: `cast_scalar(F16, U8, r)` =
    // `(unsigned char)__half2float(r)`, `(Bf16, U8)` =
    // `(unsigned char)__bfloat162float(r)`, every other `(_, U8)` =
    // `(unsigned char)r`.
    cast_scalar(plan.dtype, d, &root)
}

/// The f16/bf16 → f32 widening intrinsic (`__half2float` / `__bfloat162float`),
/// or `None` for a dtype loaded without one. The ONE place the generator names
/// the half→float promotion — shared by the inline load sites and the generated
/// dtype-promote helper (`emit_dtype_promote_helper`).
pub(crate) fn half_load_intrinsic(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::F16 => Some("__half2float"),
        ElementKind::Bf16 => Some("__bfloat162float"),
        _ => None,
    }
}

/// The f32 → f16/bf16 narrowing intrinsic (`__float2half` / `__float2bfloat16`),
/// or `None` for a dtype stored without one. Counterpart of
/// [`half_load_intrinsic`].
pub(crate) fn half_store_intrinsic(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::F16 => Some("__float2half"),
        ElementKind::Bf16 => Some("__float2bfloat16"),
        _ => None,
    }
}

/// Widen a loaded `inner` expression to `float`: the half/bf16 intrinsic, else
/// the value unchanged (already ≥ f32, or an integer loaded natively).
pub(crate) fn promote_load_f32(kind: ElementKind, inner: &str) -> String {
    match half_load_intrinsic(kind) {
        Some(f) => format!("{f}({inner})"),
        None => inner.to_string(),
    }
}

/// Narrow a `float`-valued `inner` expression to the storage dtype: the
/// half/bf16 intrinsic, else the value unchanged (the caller adds any cast).
pub(crate) fn demote_store_f32(kind: ElementKind, inner: &str) -> String {
    match half_store_intrinsic(kind) {
        Some(f) => format!("{f}({inner})"),
        None => inner.to_string(),
    }
}

/// A single element-wise dtype-cast expression — the value of `expr` (of dtype
/// `from`) converted to dtype `to`, with the house f16/bf16 float-detour
/// convention. The ONE place the generator spells a per-element conversion
/// between two scalar dtypes, shared by the inline hetero store
/// ([`store_expr_of`]) and the generated cast helper (`emit_cast_helper`), so
/// the two can never drift.
///
/// Mirrors `baracuda_cast.cuh`'s `cast_value<TIn, TOut>` (value-identical, not
/// necessarily text-identical — the generated form uses C-style casts and the
/// shared [`promote_load_f32`] / [`demote_store_f32`] intrinsic picks):
///   * `from == to` → identity (no cast).
///   * f16/bf16 → f16/bf16 (cross) → widen to `float`, then narrow.
///   * f16/bf16 → arithmetic → widen to `float`, then a C-style cast to the target.
///   * arithmetic → f16/bf16 → cast to `float` (unless already `float`), then narrow.
///   * arithmetic → arithmetic → a plain C-style cast.
pub(crate) fn cast_scalar(from: ElementKind, to: ElementKind, expr: &str) -> String {
    if from == to {
        return expr.to_string();
    }
    match (
        half_load_intrinsic(from).is_some(),
        half_store_intrinsic(to).is_some(),
    ) {
        // f16/bf16 -> f16/bf16 (cross): widen to f32, then narrow.
        (true, true) => demote_store_f32(to, &promote_load_f32(from, expr)),
        // f16/bf16 -> arithmetic: widen to f32, C-cast to the target.
        (true, false) => {
            let oct = scalar_ctype(to).expect("cast target dtype has a scalar ctype");
            format!("({oct}){}", promote_load_f32(from, expr))
        }
        // arithmetic -> f16/bf16: cast to float (unless already float), then narrow.
        (false, true) => {
            let widened = if matches!(from, ElementKind::F32 | ElementKind::F32Strict) {
                expr.to_string()
            } else {
                format!("(float){expr}")
            };
            demote_store_f32(to, &widened)
        }
        // arithmetic -> arithmetic: plain C-style cast.
        (false, false) => {
            let oct = scalar_ctype(to).expect("cast target dtype has a scalar ctype");
            format!("({oct}){expr}")
        }
    }
}

/// Spell a [`UnaryOp`] applied to an already-lowered f32 inner expression.
/// Inner strings are atomic or parenthesized, so the function-call forms need no
/// extra wrapping; the operator forms wrap themselves. (`Sigmoid`/`Gelu`/`Silu`
/// reference the inner twice — fine for an atomic load; a temp-binding pass to
/// avoid recompute on compound inners is a follow-up.)
pub(crate) fn unary_f32(op: UnaryOp, x: String) -> String {
    match op {
        UnaryOp::Neg => format!("(-{x})"),
        UnaryOp::Abs => format!("fabsf({x})"),
        UnaryOp::Sqr => format!("({x}*{x})"),
        UnaryOp::Sqrt => format!("sqrtf({x})"),
        UnaryOp::Rsqrt => format!("rsqrtf({x})"),
        UnaryOp::Recip => format!("(1.0f/{x})"),
        UnaryOp::Exp => format!("expf({x})"),
        UnaryOp::Log => format!("logf({x})"),
        UnaryOp::Tanh => format!("tanhf({x})"),
        UnaryOp::Sigmoid => format!("(1.0f/(1.0f+expf(-{x})))"),
        // NaN-propagating: `NaN < 0` is false, so NaN passes through (matches
        // PyTorch). `fmaxf(x,0)` would scrub NaN to 0. (Inner duplicated — the
        // temp-binding pass that fixes recompute is a follow-up.)
        UnaryOp::Relu => format!("({x} < 0.0f ? 0.0f : {x})"),
        UnaryOp::Erf => format!("erff({x})"),
        UnaryOp::Gelu => format!("(0.5f*{x}*(1.0f+erff({x}*0.70710678f)))"),
        UnaryOp::Silu => format!("({x}*(1.0f/(1.0f+expf(-{x}))))"),
        UnaryOp::Sin => format!("sinf({x})"),
        UnaryOp::Cos => format!("cosf({x})"),
        UnaryOp::Floor => format!("floorf({x})"),
        UnaryOp::Ceil => format!("ceilf({x})"),
        UnaryOp::Round => format!("rintf({x})"), // ties to even
        UnaryOp::Sign => format!("({x} > 0.0f ? 1.0f : ({x} < 0.0f ? -1.0f : 0.0f))"),
        UnaryOp::Step => format!("({x} > 0.0f ? 1.0f : 0.0f)"), // heaviside(x, 0): step(0)=0
        // increment-0a scalar fns — all implicit CUDA device math (headerless
        // under nvrtc, same class as expf; no includes).
        UnaryOp::Erfc => format!("erfcf({x})"),
        UnaryOp::Trunc => format!("truncf({x})"),
        UnaryOp::Exp2 => format!("exp2f({x})"),
        UnaryOp::Expm1 => format!("expm1f({x})"),
        UnaryOp::Log2 => format!("log2f({x})"),
        UnaryOp::Log10 => format!("log10f({x})"),
        UnaryOp::Log1p => format!("log1pf({x})"),
        UnaryOp::Sinh => format!("sinhf({x})"),
        UnaryOp::Cosh => format!("coshf({x})"),
        UnaryOp::Tan => format!("tanf({x})"),
        UnaryOp::Asin => format!("asinf({x})"),
        UnaryOp::Acos => format!("acosf({x})"),
        UnaryOp::Atan => format!("atanf({x})"),
        UnaryOp::Asinh => format!("asinhf({x})"),
        UnaryOp::Acosh => format!("acoshf({x})"),
        UnaryOp::Atanh => format!("atanhf({x})"),
        UnaryOp::Cbrt => format!("cbrtf({x})"),
        UnaryOp::Lgamma => format!("lgammaf({x})"),
    }
}

/// Same as [`unary_f32`] but with f64 math-function names and double literals.
pub(crate) fn unary_f64(op: UnaryOp, x: String) -> String {
    match op {
        UnaryOp::Neg => format!("(-{x})"),
        UnaryOp::Abs => format!("fabs({x})"),
        UnaryOp::Sqr => format!("({x}*{x})"),
        UnaryOp::Sqrt => format!("sqrt({x})"),
        UnaryOp::Rsqrt => format!("rsqrt({x})"),
        UnaryOp::Recip => format!("(1.0/{x})"),
        UnaryOp::Exp => format!("exp({x})"),
        UnaryOp::Log => format!("log({x})"),
        UnaryOp::Tanh => format!("tanh({x})"),
        UnaryOp::Sigmoid => format!("(1.0/(1.0+exp(-{x})))"),
        UnaryOp::Relu => format!("({x} < 0.0 ? 0.0 : {x})"),
        UnaryOp::Erf => format!("erf({x})"),
        UnaryOp::Gelu => format!("(0.5*{x}*(1.0+erf({x}*0.7071067811865476)))"),
        UnaryOp::Silu => format!("({x}*(1.0/(1.0+exp(-{x}))))"),
        UnaryOp::Sin => format!("sin({x})"),
        UnaryOp::Cos => format!("cos({x})"),
        UnaryOp::Floor => format!("floor({x})"),
        UnaryOp::Ceil => format!("ceil({x})"),
        UnaryOp::Round => format!("rint({x})"), // ties to even
        UnaryOp::Sign => format!("({x} > 0.0 ? 1.0 : ({x} < 0.0 ? -1.0 : 0.0))"),
        UnaryOp::Step => format!("({x} > 0.0 ? 1.0 : 0.0)"), // heaviside(x, 0): step(0)=0
        // increment-0a scalar fns — the double variants of the f32 spellings.
        UnaryOp::Erfc => format!("erfc({x})"),
        UnaryOp::Trunc => format!("trunc({x})"),
        UnaryOp::Exp2 => format!("exp2({x})"),
        UnaryOp::Expm1 => format!("expm1({x})"),
        UnaryOp::Log2 => format!("log2({x})"),
        UnaryOp::Log10 => format!("log10({x})"),
        UnaryOp::Log1p => format!("log1p({x})"),
        UnaryOp::Sinh => format!("sinh({x})"),
        UnaryOp::Cosh => format!("cosh({x})"),
        UnaryOp::Tan => format!("tan({x})"),
        UnaryOp::Asin => format!("asin({x})"),
        UnaryOp::Acos => format!("acos({x})"),
        UnaryOp::Atan => format!("atan({x})"),
        UnaryOp::Asinh => format!("asinh({x})"),
        UnaryOp::Acosh => format!("acosh({x})"),
        UnaryOp::Atanh => format!("atanh({x})"),
        UnaryOp::Cbrt => format!("cbrt({x})"),
        UnaryOp::Lgamma => format!("lgamma({x})"),
    }
}

/// Non-infix binary op in f32 math.
///
/// `Maximum`/`Minimum` are **NaN-propagating** (a NaN operand ⇒ NaN out) —
/// matching `torch.maximum`/`minimum` and the house reference kernel
/// `binary_maximum_fp.cu`, which deliberately reserves `fmaxf`/`fminf` (IEEE
/// `maxNum`, NaN-*suppressing*) for a *separate* op. That separate op now exists
/// as [`BinaryOp::FmaxIeee`]/[`BinaryOp::FminIeee`] below — so `Max`/`Min` emit
/// the compare-select, never `fmaxf`. (Operands appear 3× — the deferred
/// temp-binding pass, cf. relu/sigmoid, removes the recompute on compound inners.)
pub(crate) fn binary_f32(op: BinaryOp, a: String, b: String) -> String {
    match op {
        // A ON TIES (`>=`/`<=`): the KISS-Ops `max_prop`/`min_prop` normative
        // decomposition (`cmp_ge`/`cmp_le` select a) and numpy/torch
        // `where(a >= b, a, b)`. Bit-visible only on signed-zero ties
        // (`max_prop(-0.0, +0.0) = -0.0`); a `>`-spelled tie would return b.
        BinaryOp::Max => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} >= {b} ? {a} : {b})))")
        }
        BinaryOp::Min => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} <= {b} ? {a} : {b})))")
        }
        BinaryOp::Pow => format!("powf({a}, {b})"),
        // Floored remainder (torch.remainder, sign-of-divisor — Fuel's Op::Rem),
        // not C fmodf (sign-of-dividend). Operands appear twice (temp-binding TODO).
        BinaryOp::Rem => format!("({a} - floorf({a} / {b}) * {b})"),
        // increment-0a scalar fns. FmaxIeee/FminIeee are the deliberate
        // NaN-SUPPRESSING fmaxf/fminf — the separate op the house reserves them
        // for; Max/Min above stay the NaN-propagating compare-selects. RemTrunc
        // is C fmodf (sign-of-dividend) — the truncated sibling of Rem above.
        BinaryOp::Atan2 => format!("atan2f({a}, {b})"),
        BinaryOp::Copysign => format!("copysignf({a}, {b})"),
        BinaryOp::Nextafter => format!("nextafterf({a}, {b})"),
        BinaryOp::FmaxIeee => format!("fmaxf({a}, {b})"),
        BinaryOp::FminIeee => format!("fminf({a}, {b})"),
        BinaryOp::RemTrunc => format!("fmodf({a}, {b})"),
        // increment-0b comparison predicates: the C operators, with BOTH
        // operands cast to float so the compare is decided IN THE COMPUTE
        // DTYPE. Without the casts, a `Const` operand (spelled as a
        // suffix-less double literal) promotes the float side to double and
        // the compare is decided against the UNROUNDED constant — e.g.
        // `in0[i] == 0.1` is false at every x including 0.1f, while the
        // compute-dtype compare (and torch scalar promotion, and this
        // emitter's own f16 path, which rounds the constant to half first)
        // says true. The cast is a no-op for already-float operands;
        // arithmetic ops keep the double-then-round-once convention
        // (correctly rounded THROUGH the store — compares have no rounding
        // step, so they must round operands first instead). NaN semantics
        // are the C operators' (any comparison with NaN is false EXCEPT
        // `!=`, which is true); the value is EXACTLY 1.0f or 0.0f.
        BinaryOp::CmpEq => format!("((float){a} == (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpNe => format!("((float){a} != (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpLt => format!("((float){a} < (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpLe => format!("((float){a} <= (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpGt => format!("((float){a} > (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpGe => format!("((float){a} >= (float){b} ? 1.0f : 0.0f)"),
        // increment-0c INT-ONLY ops: an independent emitter backstop behind
        // the plan gate (assert_int_op_admissibility) — a bitwise/logical op
        // must never reach a float speller, including the f16/bf16 promote
        // path and the reduction-class accumulator lowerings, which all route
        // through here.
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => {
            panic!("cuda backend: {op:?} is int-only (I32/I64/S8/U8) — it has no f32 lowering")
        }
    }
}

/// Same as [`binary_f32`] but with f64 math-function names.
pub(crate) fn binary_f64(op: BinaryOp, a: String, b: String) -> String {
    match op {
        // A ON TIES (`>=`/`<=`) — see [`binary_f32`]'s Max/Min note.
        BinaryOp::Max => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} >= {b} ? {a} : {b})))")
        }
        BinaryOp::Min => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} <= {b} ? {a} : {b})))")
        }
        BinaryOp::Pow => format!("pow({a}, {b})"),
        BinaryOp::Rem => format!("({a} - floor({a} / {b}) * {b})"),
        BinaryOp::Atan2 => format!("atan2({a}, {b})"),
        BinaryOp::Copysign => format!("copysign({a}, {b})"),
        BinaryOp::Nextafter => format!("nextafter({a}, {b})"),
        BinaryOp::FmaxIeee => format!("fmax({a}, {b})"),
        BinaryOp::FminIeee => format!("fmin({a}, {b})"),
        BinaryOp::RemTrunc => format!("fmod({a}, {b})"),
        // increment-0b comparison predicates — double literals, same C-operator
        // NaN semantics as the f32 arms.
        BinaryOp::CmpEq => format!("({a} == {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpNe => format!("({a} != {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpLt => format!("({a} < {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpLe => format!("({a} <= {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpGt => format!("({a} > {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpGe => format!("({a} >= {b} ? 1.0 : 0.0)"),
        // increment-0c INT-ONLY ops — same backstop as the f32 speller.
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => {
            panic!("cuda backend: {op:?} is int-only (I32/I64/S8/U8) — it has no f64 lowering")
        }
    }
}

/// Spell an increment-0c INT-ONLY binary op (bitwise/shift/logical) over two
/// already-lowered integer operand strings — the RAW C operators, matching the
/// bespoke functors **exactly** (the 0c charter: express the bespoke
/// functionality, never "improve" it):
///
/// - `BitAnd`/`BitOr`/`BitXor`: `binary_bitwise_{and,or,xor}_int.cu`'s
///   `return a OP b;` — no rounding, no overflow concerns.
/// - `Shl`/`Shr`: `binary_bitwise_{left,right}_shift_int.cu`'s `return a << b;`
///   / `return a >> b;` — NO masking or clamping. Out-of-range amounts
///   (`b < 0` or `b >= 8*sizeof(promoted T)`) inherit the architecture's
///   behavior (the bespoke caller contract, carried verbatim); signed `>>` is
///   arithmetic on every CUDA compiler (PTX `shr.s32`/`shr.s64` — the bespoke
///   kernel's documented reliance), unsigned is logical.
/// - `LogicalAnd`/`LogicalOr`/`LogicalXor`: `binary_logical_*_bool.cu`'s
///   normalize-then-op — `(a != 0 OP b != 0) ? 1 : 0`, so the output is
///   strictly 0/1 even for unnormalized bytes (`2 && 4 == 1`). U8-only (the
///   bespoke Bool surface); the plan gate enforces it and the assert here is
///   the independent emitter backstop.
///
/// **Integer-promotion note (S8/U8):** the operand strings are `signed char`/
/// `unsigned char` loads — GUARANTEED, not assumed: the plan gate's 8-bit
/// composition pin (`plan::assert_int_op_admissibility` rule 3) requires every
/// int-op operand at `S8`/`U8` to be a leaf `Input`, so a composed operand
/// (whose inlined un-truncated value would diverge from its hoisted 8-bit-tmp
/// value under DAG sharing) can never reach this speller. The loads promote to
/// `int` (sign-/zero-extended) before any operator. NO defeating casts are
/// emitted, deliberately:
/// - and/or/xor: promote → op → store-truncate is bit-identical to a native
///   8-bit op (extension bits AND/OR/XOR among themselves and truncate away);
/// - `Shl`: the 32-bit shift result store-truncates mod 2⁸ — equal to a
///   native wrapping 8-bit shift for in-range amounts, and amounts 8..31 take
///   the promoted (well-defined-in-practice) semantics rather than native-8-bit
///   UB. This matches how the bespoke i32/i64 kernels compose with C — there
///   is no bespoke 8-bit shift to defer to, so the promotion semantics ARE the
///   documented contract (see `BinaryOp::Shl`);
/// - `Shr`: the promoted value's high bits are the extension of the 8-bit
///   value, so the shifted result always fits 8 bits — truncation is exact,
///   arithmetic for `signed char`, logical for `unsigned char`;
/// - logical ops: the `!= 0` tests and the 0/1 result are promotion-invariant.
///
/// The final `other` arm is the second half of the emitter backstop: a float
/// fn / cmp op that reaches the int speller (i.e. bypassed the plan gate at an
/// int dtype) panics rather than emitting C that happens to compile.
pub(crate) fn binary_int(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> String {
    if op.is_logical() {
        assert!(
            dtype == ElementKind::U8,
            "cuda backend: {op:?} is U8 (Bool)-only — the bespoke logical surface \
             instantiates exactly uint8_t; got {dtype:?}"
        );
    }
    match op {
        BinaryOp::BitAnd => format!("({a} & {b})"),
        BinaryOp::BitOr => format!("({a} | {b})"),
        BinaryOp::BitXor => format!("({a} ^ {b})"),
        BinaryOp::Shl => format!("({a} << {b})"),
        BinaryOp::Shr => format!("({a} >> {b})"),
        BinaryOp::LogicalAnd => format!("(({a} != 0 && {b} != 0) ? 1 : 0)"),
        BinaryOp::LogicalOr => format!("(({a} != 0 || {b} != 0) ? 1 : 0)"),
        BinaryOp::LogicalXor => format!("((({a} != 0) != ({b} != 0)) ? 1 : 0)"),
        other => panic!(
            "cuda backend: {other:?} has no integer lowering — the bespoke \
             elementwise surface instantiates it for float dtypes only \
             (int dtype {dtype:?} must miss honestly at the plan gate)"
        ),
    }
}

/// Ternary select in f32 math: `cond != 0.0f` picks arm `a`, else `b`
/// ([`ScalarExpr::Select`] — nonzero-true, `-0.0` false, NaN true).
///
/// The `(float)` casts are MANDATORY, not cosmetic (the 0b double-promotion
/// lesson): they are identity no-ops on already-float operands but pin the
/// compare AND the ternary's type against a suffix-less double `Const`
/// literal from `const_lit` — without the arm casts, a double-literal arm
/// (`select(c, x, 0.0)`) promotes the whole ternary to double and the
/// double→float round-trip at the store QUIETS an f32 sNaN arm payload (a
/// bit diff vs the bespoke select). The cond cast makes the `!= 0` decision
/// happen in the compute dtype (the cmp-operand precedent, `binary_f32`).
/// No arithmetic ever touches an arm: the ternary is data movement
/// (setp+selp), so ±0 signs and NaN payloads (quiet and signaling) move
/// intact — byte-for-byte the bespoke `keep ? input[k] : zero_of<T>()`.
pub(crate) fn select_f32(c: String, a: String, b: String) -> String {
    format!("(((float)({c})) != 0.0f ? (float)({a}) : (float)({b}))")
}

/// [`select_f32`] with double literals/casts (the `binary_f64` cmp precedent).
pub(crate) fn select_f64(c: String, a: String, b: String) -> String {
    format!("(((double)({c})) != 0.0 ? (double)({a}) : (double)({b}))")
}

/// Runtime scalar-param indices used by `e`, ascending + unique.
pub(crate) fn params_used(e: &ScalarExpr) -> Vec<u8> {
    fn rec(e: &ScalarExpr, out: &mut std::collections::BTreeSet<u8>) {
        match e {
            ScalarExpr::Param(i) => {
                out.insert(*i);
            }
            ScalarExpr::Unary(_, x) => rec(x, out),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                rec(a, out);
                rec(b, out);
            }
            ScalarExpr::Select(c, a, b) => {
                rec(c, out);
                rec(a, out);
                rec(b, out);
            }
            ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Reduced(_)
            | ScalarExpr::Coord(_) => {}
        }
    }
    let mut set = std::collections::BTreeSet::new();
    rec(e, &mut set);
    set.into_iter().collect()
}

/// Emitter backstop for the two dtype-blind spellings (increment 0c): panic if
/// `e` contains an infix [`ScalarExpr::Div`] node or a [`ScalarExpr::Const`]
/// leaf while `Cuda::lower` is lowering an INTEGER dtype. Both are spelled by
/// shared backend code with no dtype context (`lower_expr` emits C `/` and an
/// f64 C literal for every dtype), so unlike the unary/binary-fn/int-only ops
/// they have no per-op speller panic to catch a plan-gate bypass — and they are
/// exactly the device-dangerous pair: integer `/0` is device-UB, and an
/// f64-spelled Const injects double math into an int kernel (f64 cannot even
/// represent all i64). Called from `Cuda::lower` over the body and every
/// reduction-class stage/epilogue, independent of `assert_int_op_admissibility`.
///
/// `in_reduction` mirrors `plan::assert_int_op_admissibility`'s rule 4 (the
/// any/all/count fused-predicate lift, ba325509/Task 3b): `true` only when the
/// expression is this plan's `Access::Reduction` body/post — CpuC/Slang (v1,
/// Elementwise-only) always pass `false`, so their coverage is unchanged.
/// `at_reduction_root` mirrors `plan::assert_int_op_admissibility`'s
/// `at_reduction_root` (whole-branch-review fix, closing the composed-
/// predicate leak): `true` ONLY for the initial call on the reduction
/// body/post root, `false` for every recursive descent — CpuC/Slang pass
/// `false` for both parameters (inert, since `in_reduction` is already
/// `false` there). Within that scope (`in_reduction && at_reduction_root`), a
/// `Cmp*` node's operands (leaf `Input`/`Reduced` or an exact 0/1 `Const`)
/// are exempted from the blanket `Const` panic below — that 0/1 `Const` is
/// safe (it lowers as an INTEGER literal in `cuda::emit_reduction`'s
/// `int_reduction_predicate`, which — like this gate — only inspects the
/// body/post ROOT node, never a nested one), never the f64 C literal this
/// backstop exists to catch); anything else in that position still recurses
/// into the general check and panics as before. A `Cmp*` reached as a
/// sub-node of Add/Sub/Mul (not the root) falls to the general
/// `ScalarExpr::Binary(_, a, b)` arm below like any other binary node, so its
/// own `Const`/`Div` operands are still policed by the blanket rules — it is
/// never itself the leaf-or-{0,1} exemption target.
pub(crate) fn assert_no_int_div_or_const(
    e: &ScalarExpr,
    dtype: ElementKind,
    in_reduction: bool,
    at_reduction_root: bool,
) {
    match e {
        ScalarExpr::Input(_) | ScalarExpr::Param(_) | ScalarExpr::Reduced(_) => {}
        // A Coord at an int dtype is the SAME hazard class as Const (its
        // spelling is a float cast) — but it has its own dedicated backstop,
        // `assert_coord_lowerable`, which runs beside this walk in
        // `Cuda::lower` for every dtype (not just int) and carries the
        // targeted message; no second assert here (one message per layer).
        ScalarExpr::Coord(_) => {}
        ScalarExpr::Const(_) => panic!(
            "cuda backend: Const at an integer dtype ({dtype:?}) — a Const is \
             spelled as an f64 C literal, which would silently run double math \
             in an integer kernel; the plan gate rejects this (an int-literal \
             speller is a follow-up)"
        ),
        ScalarExpr::Div(_, _) => panic!(
            "cuda backend: infix Div has no integer lowering ({dtype:?}) — the \
             bespoke elementwise surface has no int div and C `/` by zero is \
             device-UB; the plan gate rejects this"
        ),
        // Select at an integer dtype is validate-rejected at the plan gate
        // (v1 select is float-only); this walk — which only runs for int
        // dtypes — is its independent emitter backstop (G5), beside the
        // per-speller panic in `cuda_select` (which only the elementwise
        // paths route through; the reduction-class accumulator closures
        // don't, so the walk is the layer that covers them).
        ScalarExpr::Select(_, _, _) => panic!(
            "cuda backend: Select at an integer dtype ({dtype:?}) — v1 select is \
             float-only (the 0c U8/I8 cond-observer question is unresolved); the \
             plan gate rejects this"
        ),
        ScalarExpr::Unary(_, x) => {
            assert_no_int_div_or_const(x, dtype, in_reduction, false);
        }
        ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) | ScalarExpr::Mul(a, b) => {
            assert_no_int_div_or_const(a, dtype, in_reduction, false);
            assert_no_int_div_or_const(b, dtype, in_reduction, false);
        }
        // The exemption test is `ir::is_admissible_int_reduction_operand` —
        // the SAME helper `plan::assert_int_op_admissibility` (rule 4) and
        // `cuda::emit_reduction`'s `int_reduction_predicate`/`int_cmp_operand`
        // call, so this shape cannot drift from the gate/emitter again. An
        // operand this helper doesn't admit still recurses into the general
        // walk below (this backstop's job is narrower than the plan gate's —
        // it only polices the Const/Div/Select double-math hazards, not
        // composition — so a non-admitted operand isn't rejected here
        // outright, just walked normally).
        ScalarExpr::Binary(bop, a, b) if in_reduction && at_reduction_root && bop.is_cmp() => {
            for operand in [&**a, &**b] {
                if !is_admissible_int_reduction_operand(operand) {
                    assert_no_int_div_or_const(operand, dtype, in_reduction, false);
                }
            }
        }
        ScalarExpr::Binary(_, a, b) => {
            assert_no_int_div_or_const(a, dtype, in_reduction, false);
            assert_no_int_div_or_const(b, dtype, in_reduction, false);
        }
    }
}

#[cfg(test)]
mod int_div_or_const_root_gate_validate {
    //! Direct unit coverage for `assert_no_int_div_or_const`'s
    //! `at_reduction_root` restriction (whole-branch-review fix, mirroring
    //! `plan::int_reduction_predicate_gate_validate`). This backstop normally
    //! only runs AFTER `plan::assert_int_op_admissibility` has already
    //! validated the op at `build_plan` time, so a composed-predicate body
    //! never reaches it via the `generate()`/`Cuda::lower` path — the plan
    //! gate rejects it first. These tests call the function directly (it is
    //! `pub(crate)`) to exercise it as an independent layer in its own right
    //! (the "gate every layer" principle the surrounding code comments name
    //! throughout this file), the same way the plan-gate tests bypass
    //! `build_plan` to isolate `assert_int_op_admissibility`.
    use super::assert_no_int_div_or_const;
    use crate::ir::{BinaryOp, ScalarExpr, input, konst};
    use baracuda_kernel_vocab::ElementKind;

    // Root-Cmp positive case (mirrors the shipped `count` shape): a bare
    // `Cmp*` IS the reduction body root — admitted (0/1 Const operand
    // exempted from the blanket Const panic), must not panic.
    #[test]
    fn root_cmp_with_01_const_admitted() {
        let body = input(0).binary(BinaryOp::CmpNe, konst(0.0)).0;
        assert_no_int_div_or_const(&body, ElementKind::S8, true, true);
    }

    // Root-only guard: a COMPOSED predicate — `Add(Cmp*, Cmp*)` — reached as
    // the reduction body root. The root itself is `Add`, not `Cmp`, so the
    // exemption arm never matches at this level; the walk recurses into each
    // `Cmp*` child with `at_reduction_root: false` (this fix), lands the
    // nested Cmp on the general `Binary(_, a, b)` arm instead of the
    // exemption arm, and its `Const(0.0)` operand then hits the ordinary
    // blanket `Const` panic — closing the same composed-predicate leak this
    // gate mirrors from `plan::assert_int_op_admissibility`. Before this fix
    // the nested Cmp still matched `in_reduction && bop.is_cmp()`
    // (unconditionally, no root check) and its 0/1 Const was wrongly
    // exempted, so this call did NOT panic.
    #[test]
    fn composed_predicate_not_at_root_panics() {
        let cmp = || input(0).binary(BinaryOp::CmpNe, konst(0.0)).0;
        let body = ScalarExpr::Add(Box::new(cmp()), Box::new(cmp()));
        let r = std::panic::catch_unwind(|| {
            assert_no_int_div_or_const(&body, ElementKind::S8, true, true)
        });
        assert!(
            r.is_err(),
            "a Cmp* reached as a sub-node of Add (not the reduction body/post \
             root) must still panic on its Const operand — admitting it here \
             would mirror the plan-gate leak this fix closes"
        );
    }
}

/// The SCALAR COMPUTE ctype for an op's runtime launch params — `scalar_ctype(
/// plan.dtype)`, i.e. `"float"` for F32/F32Strict (byte-identical to the pre-F64
/// hardcode) and `"double"` for F64. A launch param is ALWAYS a scalar arg, never
/// vectorized, so this is the declaration ctype at EVERY emitter (including the
/// vectorized ones, whose OPERANDS are `float4`/`double2` but whose param stays
/// `double p0`) — the F64-param increment's load-bearing distinction: pass THIS,
/// never `vty`/`octype`. The `Cuda::lower` param assert (see the `matches!` on
/// `plan.dtype` above) guarantees a param-bearing plan has a spellable scalar
/// ctype, so the `expect` is unreachable for any op that actually declares a param.
pub(crate) fn param_ctype(plan: &KernelPlan<'_>) -> &'static str {
    scalar_ctype(plan.dtype).expect("param dtype checked by the Cuda::lower param assert")
}

/// The trailing `, <ctype> p0, <ctype> p1, …` kernel-signature suffix for the
/// op's runtime scalar params (empty when the op has none). `param_ctype` is the
/// SCALAR COMPUTE ctype `scalar_ctype(plan.dtype)` — `"float"` for F32/F32Strict
/// (byte-identical to the pre-F64 hardcode by construction), `"double"` for F64.
/// A launch param is ALWAYS scalar: even on the vectorized emitter (operands are
/// `float4`/`double2`) the declaration stays `double p0`, NOT `double2 p0` — so
/// callers pass the SCALAR compute ctype, never `vty`/`octype` (the F64-param
/// increment's load-bearing distinction).
pub(crate) fn param_args(e: &ScalarExpr, param_ctype: &str) -> String {
    params_used(e)
        .iter()
        .map(|i| format!(", {param_ctype} p{i}"))
        .collect()
}
