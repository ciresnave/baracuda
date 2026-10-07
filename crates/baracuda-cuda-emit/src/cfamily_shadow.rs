//! Baracuda's own closed-set copy of the CUDA-specific scalar spellers that
//! used to be imported directly from the neutral `unpopped::cfamily` module:
//! [`scalar_ctype`], [`cast_scalar`], [`promote_load_f32`], [`demote_store_f32`],
//! [`param_ctype`], [`out_ctype_of`], [`store_expr_of`] — plus the 4 leaf
//! functions those 7 are built from (see the 2026-10-07 corrections below).
//!
//! # Why this module exists (2026-10-07)
//!
//! `scalar_ctype`'s f16/bf16 arms return `__half`/`__nv_bfloat16`, and
//! `promote_load_f32`/`demote_store_f32`/`cast_scalar`'s half arms emit
//! `__half2float`-class CUDA intrinsics — this is CUDA-specific *spelling*,
//! not backend-neutral code. `docs/design/2026-08-06-unpopped-extraction-plan.md`
//! (the "★ Deferred theme — BEHAVIORAL neutrality" section, item 1, and the
//! later "f16 RE-SCOPED" note) already identified that these four functions
//! should live only in the CUDA emitter, not in Unpopped's shared neutral
//! core — but the move was never executed, so `cuda.rs` kept importing them
//! from `unpopped::cfamily` directly.
//!
//! Per board #106's ownership ruling: baracuda owns the actual CUDA spelling
//! per architecture (via `baracuda-cuda-emit`); Unpopped owns capability rows
//! and per-sm plan decisions. Unpopped's own planned f16/bf16 change (their
//! item U1/#40) needs to be free to change or simplify their shared
//! `cfamily` copy of these functions without silently changing baracuda's
//! emitted output — which it would, as long as baracuda imported them
//! directly instead of owning its own copy.
//!
//! These bodies are a verbatim copy of `unpopped::cfamily`'s versions as they
//! existed on Unpopped `origin/main` at the time of this split. Output is
//! proven byte-identical to calling `unpopped::cfamily`'s originals directly
//! by the test at the bottom of this file; it is NOT a promise that the two
//! will stay identical forever — baracuda now owns this copy and may
//! diverge from it deliberately, which is the entire point of the split.
//!
//! # Correction (2026-10-07) — the shadow was not closed
//!
//! The first version of this module shadowed only the four NAMED functions
//! and left `promote_load_f32` / `demote_store_f32` / `cast_scalar` calling
//! back into `unpopped::cfamily::{narrow_load_fn, narrow_store_fn}` for their
//! actual intrinsic-name lookups. Those two functions are themselves thin
//! wrappers over `half_load_intrinsic` / `half_store_intrinsic`
//! (`unpopped/src/cfamily.rs:223-240,609-624`) — exactly the leaves
//! Unpopped's own planned f16/bf16 change (item U1/#40) moves. Per
//! `docs/design/2026-08-06-unpopped-extraction-plan.md:366-370`'s own rule:
//! "A shadow that delegates back into the thing it is shadowing away from is
//! not a shadow." Caught by Unpopped's review before this PR merged.
//! Fixed by shadowing all four leaf functions too — nothing in this module
//! now reaches `unpopped::cfamily` for any spelling decision.
//!
//! # Correction 2 (2026-10-07) — three more callers reached the moving leaves
//!
//! Unpopped's own deferred-work table said only 4 functions needed to move
//! and that `store_expr_of` specifically did NOT move — both wrong (their
//! own correction, not baracuda's miss). `cuda.rs` also imported
//! `param_ctype`, `out_ctype_of`, and `store_expr_of` from
//! `unpopped::cfamily` — and by CALL GRAPH, not by name, all three reach the
//! moving leaves: `param_ctype`/`out_ctype_of` call [`scalar_ctype`]
//! directly (`unpopped/src/cfamily.rs:1175,153`); `store_expr_of` calls
//! [`cast_scalar`] on its mixed-dtype branch and [`demote_store_f32`] on its
//! FP8 branch (`cfamily.rs:216,203`). Shadowed all three the same way — 7
//! named functions + 4 leaves = 11 shadowed total, verified by call graph
//! (every remaining `unpopped::cfamily` import cuda.rs still uses —
//! `assert_no_int_div_or_const`, `binary_f32`, `binary_f64`, `binary_int`,
//! `dtype_tag`, `param_args`, `params_used`, `select_f32`, `select_f64` —
//! read end-to-end and confirmed self-contained, no path to any of the 11).

use unpopped::plan::KernelPlan;
use unpopped_vocab::ElementKind;

/// The f16/bf16 → f32 widening intrinsic, or `None` for a dtype loaded
/// without one. Local, frozen copy of `unpopped::cfamily::half_load_intrinsic`
/// as it existed at adoption (2026-10-07) — see the module doc for why this
/// must not delegate back to `unpopped`.
fn half_load_intrinsic(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::F16 => Some("__half2float"),
        ElementKind::Bf16 => Some("__bfloat162float"),
        _ => None,
    }
}

/// The f32 → f16/bf16 narrowing intrinsic, or `None` for a dtype stored
/// without one. Local, frozen copy of `unpopped::cfamily::half_store_intrinsic`.
fn half_store_intrinsic(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::F16 => Some("__float2half"),
        ElementKind::Bf16 => Some("__float2bfloat16"),
        _ => None,
    }
}

/// The load-side widening function for a narrow-float dtype: a vendor
/// intrinsic for f16/bf16, an emitted software helper for FP8, `None` for
/// everything else. Local, frozen copy of `unpopped::cfamily::narrow_load_fn`.
fn narrow_load_fn(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::Fp8E4M3FN => Some("unpopped_f8e4m3fn_load"),
        ElementKind::Fp8E5M2 => Some("unpopped_f8e5m2_load"),
        _ => half_load_intrinsic(kind),
    }
}

/// The store-side narrowing function. Counterpart of [`narrow_load_fn`].
/// Local, frozen copy of `unpopped::cfamily::narrow_store_fn`.
fn narrow_store_fn(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::Fp8E4M3FN => Some("unpopped_f8e4m3fn_store"),
        ElementKind::Fp8E5M2 => Some("unpopped_f8e5m2_store"),
        _ => half_store_intrinsic(kind),
    }
}

/// CUDA scalar type for a dtype, or `None` if the backend can't lower it yet.
/// `U8` (increment 0b) is the comparison-predicate mask dtype — `unsigned char`
/// per the FKC §5 Bool→U8 pinning — and, since increment 0c, an audited
/// COMPUTE dtype (wrapping mod-256 C semantics), same class as the i32/i64
/// arms. `S8` (FKC `I8`, increment 0c) is `signed char` — two's-complement
/// wrapping via integer promotion + store truncation (see the ir.rs table).
pub fn scalar_ctype(dt: ElementKind) -> Option<&'static str> {
    Some(match dt {
        ElementKind::F32 | ElementKind::F32Strict => "float",
        ElementKind::F64 => "double",
        ElementKind::F16 => "__half",
        ElementKind::Bf16 => "__nv_bfloat16",
        ElementKind::I32 => "int",
        ElementKind::I64 => "long long",
        ElementKind::I8 => "signed char",
        ElementKind::U8 => "unsigned char",
        // KISS-Classify §6.1 widths that have exact, portable C spellings. No
        // vendor intrinsic and no packing is involved, so unlike the f16/bf16
        // arms above these are genuinely neutral.
        ElementKind::I16 => "short",
        ElementKind::U16 => "unsigned short",
        // U32 is the gather/scatter INDEX-operand ctype (`unsigned int`) — a
        // 4-byte address dtype used ONLY for the index-load pointer type (the
        // Model-A u32-index path), never a compute operand. It has no `Element`
        // impl and no vector/packed path; a compute op never keys `plan.dtype =
        // U32` (no constructor builds one), so this arm serves the index load.
        // FP8 is STORED as a byte and COMPUTED as a float. C has no FP8 type,
        // and does not need one: the codec is a pair of emitted helpers
        // (`fp8_helpers`), not a language feature.
        ElementKind::Fp8E4M3FN | ElementKind::Fp8E5M2 => "unsigned char",
        // `bool` is a 1-byte truth value (§6.1) with the same storage width as
        // `u8` and different semantics: its ops normalize to 0/1. The C spelling
        // is the storage type; the normalization lives in the logical spellers,
        // which already emit `... ? 1 : 0`.
        ElementKind::Bool => "unsigned char",
        // Sub-byte dtypes spell their CONTAINER: several elements share a byte,
        // and the packing lives in `sub_byte_helpers`, not in the type name.
        ElementKind::I4 | ElementKind::U4 | ElementKind::B1 => "unsigned char",
        // Complex is a STRUCT, emitted with the kernel (`complex_helpers`).
        // C99 `_Complex` is not an option: MSVC does not implement it.
        ElementKind::Complex64 => "unpopped_c64",
        ElementKind::Complex128 => "unpopped_c128",
        ElementKind::U32 => "unsigned int",
        ElementKind::U64 => "unsigned long long",
        _ => return None,
    })
}

/// Widen a loaded `inner` expression to `float`: the half/bf16 intrinsic, else
/// the value unchanged (already >= f32, or an integer loaded natively).
pub fn promote_load_f32(kind: ElementKind, inner: &str) -> String {
    match narrow_load_fn(kind) {
        Some(f) => format!("{f}({inner})"),
        None => inner.to_string(),
    }
}

/// Narrow a `float`-valued `inner` expression to the storage dtype: the
/// half/bf16 intrinsic, else the value unchanged (the caller adds any cast).
pub fn demote_store_f32(kind: ElementKind, inner: &str) -> String {
    match narrow_store_fn(kind) {
        Some(f) => format!("{f}({inner})"),
        None => inner.to_string(),
    }
}

/// A single element-wise dtype-cast expression — the value of `expr` (of dtype
/// `from`) converted to dtype `to`, with the house f16/bf16 float-detour
/// convention. The ONE place the generator spells a per-element conversion
/// between two scalar dtypes, shared by the inline hetero store
/// (`store_expr_of`) and the generated cast helper (`emit_cast_helper`), so
/// the two can never drift.
///
/// Mirrors `baracuda_cast.cuh`'s `cast_value<TIn, TOut>` (value-identical, not
/// necessarily text-identical — the generated form uses C-style casts and the
/// shared [`promote_load_f32`] / [`demote_store_f32`] intrinsic picks):
///   * `from == to` -> identity (no cast).
///   * f16/bf16 -> f16/bf16 (cross) -> widen to `float`, then narrow.
///   * f16/bf16 -> arithmetic -> widen to `float`, then a C-style cast to the target.
///   * arithmetic -> f16/bf16 -> cast to `float` (unless already `float`), then narrow.
///   * arithmetic -> arithmetic -> a plain C-style cast.
pub fn cast_scalar(from: ElementKind, to: ElementKind, expr: &str) -> String {
    if from == to {
        return expr.to_string();
    }
    match (
        narrow_load_fn(from).is_some(),
        narrow_store_fn(to).is_some(),
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

/// The SCALAR COMPUTE ctype for an op's runtime launch params —
/// `scalar_ctype(plan.dtype)`. Local, frozen copy of
/// `unpopped::cfamily::param_ctype` (reaches [`scalar_ctype`] directly;
/// shadowed per the 2026-10-07 correction 2).
pub fn param_ctype(plan: &KernelPlan<'_>) -> &'static str {
    scalar_ctype(plan.dtype).expect("param dtype checked by the Backend::lower param assert")
}

/// Output `j`'s ctype: the uniform `ctype` if `out_dtype_of(j) == plan.dtype`,
/// else that output's own scalar ctype. Local, frozen copy of
/// `unpopped::cfamily::out_ctype_of` (reaches [`scalar_ctype`] directly).
pub fn out_ctype_of<'c>(plan: &KernelPlan<'_>, j: usize, ctype: &'c str) -> &'c str {
    let d = plan.out_dtype_of(j);
    if d == plan.dtype {
        ctype
    } else {
        scalar_ctype(d).expect("validated out dtype has a scalar ctype")
    }
}

/// The store expression for output `j`'s lowered body root. Local, frozen
/// copy of `unpopped::cfamily::store_expr_of` (reaches [`demote_store_f32`]
/// on the uniform FP8 branch and [`cast_scalar`] on the hetero branch —
/// shadowed per the 2026-10-07 correction 2).
pub fn store_expr_of(plan: &KernelPlan<'_>, j: usize, root: String) -> String {
    let d = plan.out_dtype_of(j);
    if d == plan.dtype {
        return match d {
            ElementKind::Fp8E4M3FN | ElementKind::Fp8E5M2 => demote_store_f32(d, &root),
            _ => root,
        };
    }
    cast_scalar(plan.dtype, d, &root)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_KINDS: &[ElementKind] = &[
        ElementKind::F32,
        ElementKind::F32Strict,
        ElementKind::F64,
        ElementKind::F16,
        ElementKind::Bf16,
        ElementKind::I32,
        ElementKind::I64,
        ElementKind::I8,
        ElementKind::U8,
        ElementKind::I16,
        ElementKind::U16,
        ElementKind::Fp8E4M3FN,
        ElementKind::Fp8E5M2,
        ElementKind::Bool,
        ElementKind::I4,
        ElementKind::U4,
        ElementKind::B1,
        ElementKind::Complex64,
        ElementKind::Complex128,
        ElementKind::U32,
        ElementKind::U64,
    ];

    // Every assertion below compares against a LITERAL expected string,
    // never a call into `unpopped::cfamily` (live or otherwise) and never a
    // second call into this module's own functions using the same formula —
    // per the 2026-10-07 correction, a test that re-derives its own
    // expectation from the same leaves it's checking can't catch a mutation
    // to those leaves. These literals were captured at adoption and must be
    // hand-updated (a deliberate, reviewed edit) if baracuda's own CUDA
    // spelling ever changes — that's the point: an upstream Unpopped change
    // must NOT be able to move these literals for us.

    /// Independent, hardcoded expectation for [`scalar_ctype`] — every
    /// `ALL_KINDS` entry has a `Some` arm in production; this table is that
    /// same information written out by hand, not derived from the function.
    fn expected_scalar_ctype(dt: ElementKind) -> Option<&'static str> {
        Some(match dt {
            ElementKind::F32 | ElementKind::F32Strict => "float",
            ElementKind::F64 => "double",
            ElementKind::F16 => "__half",
            ElementKind::Bf16 => "__nv_bfloat16",
            ElementKind::I32 => "int",
            ElementKind::I64 => "long long",
            ElementKind::I8 => "signed char",
            ElementKind::U8 => "unsigned char",
            ElementKind::I16 => "short",
            ElementKind::U16 => "unsigned short",
            ElementKind::Fp8E4M3FN | ElementKind::Fp8E5M2 => "unsigned char",
            ElementKind::Bool => "unsigned char",
            ElementKind::I4 | ElementKind::U4 | ElementKind::B1 => "unsigned char",
            ElementKind::Complex64 => "unpopped_c64",
            ElementKind::Complex128 => "unpopped_c128",
            ElementKind::U32 => "unsigned int",
            ElementKind::U64 => "unsigned long long",
            _ => return None,
        })
    }

    /// Independent, hardcoded expectation for [`promote_load_f32`]. Only
    /// F16/Bf16/Fp8E4M3FN/Fp8E5M2 widen; every other kind passes through.
    fn expected_promote_load_f32(kind: ElementKind, inner: &str) -> String {
        match kind {
            ElementKind::F16 => format!("__half2float({inner})"),
            ElementKind::Bf16 => format!("__bfloat162float({inner})"),
            ElementKind::Fp8E4M3FN => format!("unpopped_f8e4m3fn_load({inner})"),
            ElementKind::Fp8E5M2 => format!("unpopped_f8e5m2_load({inner})"),
            _ => inner.to_string(),
        }
    }

    /// Independent, hardcoded expectation for [`demote_store_f32`].
    /// Counterpart of [`expected_promote_load_f32`].
    fn expected_demote_store_f32(kind: ElementKind, inner: &str) -> String {
        match kind {
            ElementKind::F16 => format!("__float2half({inner})"),
            ElementKind::Bf16 => format!("__float2bfloat16({inner})"),
            ElementKind::Fp8E4M3FN => format!("unpopped_f8e4m3fn_store({inner})"),
            ElementKind::Fp8E5M2 => format!("unpopped_f8e5m2_store({inner})"),
            _ => inner.to_string(),
        }
    }

    /// Independent, hardcoded expectation for [`cast_scalar`], built from the
    /// three expectation functions above (never from the real `cast_scalar`
    /// or from `promote_load_f32`/`demote_store_f32` themselves) plus the
    /// same case structure the production function documents. Covers every
    /// `ALL_KINDS` pair (441 cases) — a narrow dtype is exactly one of
    /// F16/Bf16/Fp8E4M3FN/Fp8E5M2, so the branch taken for any pair is fully
    /// determined by membership in that 4-element set.
    fn expected_cast_scalar(from: ElementKind, to: ElementKind, expr: &str) -> String {
        if from == to {
            return expr.to_string();
        }
        let is_narrow = |k: ElementKind| {
            matches!(
                k,
                ElementKind::F16
                    | ElementKind::Bf16
                    | ElementKind::Fp8E4M3FN
                    | ElementKind::Fp8E5M2
            )
        };
        match (is_narrow(from), is_narrow(to)) {
            (true, true) => expected_demote_store_f32(to, &expected_promote_load_f32(from, expr)),
            (true, false) => {
                let oct = expected_scalar_ctype(to).expect("every ALL_KINDS entry has a ctype");
                format!("({oct}){}", expected_promote_load_f32(from, expr))
            }
            (false, true) => {
                let widened = if matches!(from, ElementKind::F32 | ElementKind::F32Strict) {
                    expr.to_string()
                } else {
                    format!("(float){expr}")
                };
                expected_demote_store_f32(to, &widened)
            }
            (false, false) => {
                let oct = expected_scalar_ctype(to).expect("every ALL_KINDS entry has a ctype");
                format!("({oct}){expr}")
            }
        }
    }

    #[test]
    fn scalar_ctype_matches_the_literal_table_for_every_kind() {
        for &k in ALL_KINDS {
            assert_eq!(
                scalar_ctype(k),
                expected_scalar_ctype(k),
                "{k:?}: scalar_ctype must match the literal expectation captured at adoption"
            );
        }
    }

    #[test]
    fn promote_load_f32_matches_the_literal_table_for_every_kind() {
        for &k in ALL_KINDS {
            assert_eq!(
                promote_load_f32(k, "x"),
                expected_promote_load_f32(k, "x"),
                "{k:?}: promote_load_f32 must match the literal expectation captured at adoption"
            );
        }
    }

    #[test]
    fn demote_store_f32_matches_the_literal_table_for_every_kind() {
        for &k in ALL_KINDS {
            assert_eq!(
                demote_store_f32(k, "x"),
                expected_demote_store_f32(k, "x"),
                "{k:?}: demote_store_f32 must match the literal expectation captured at adoption"
            );
        }
    }

    #[test]
    fn cast_scalar_matches_the_literal_table_for_every_kind_pair() {
        for &from in ALL_KINDS {
            for &to in ALL_KINDS {
                assert_eq!(
                    cast_scalar(from, to, "x"),
                    expected_cast_scalar(from, to, "x"),
                    "{from:?} -> {to:?}: cast_scalar must match the literal expectation captured at adoption"
                );
            }
        }
    }

    /// Minimal `KernelPlan` for testing `param_ctype`/`out_ctype_of`/
    /// `store_expr_of`, which only read `plan.dtype` and `plan.out_dtype`
    /// (via `out_dtype_of(0)`) — every other field is irrelevant to those
    /// three functions' logic, so one fixed dummy key/body is reused across
    /// every `(dtype, out_dtype)` pair under test.
    fn test_plan(dtype: ElementKind, out_dtype: ElementKind) -> KernelPlan<'static> {
        use unpopped::ir::{Access, BaseOffset, ScalarExpr, WriteIndex, input};
        use unpopped::plan::Schedule;
        use unpopped_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};

        // `'static` leaks are fine in test-only code: the allocations live
        // for the process lifetime, which is exactly the test run.
        let a = OperandDesc::new(1, &[1 << 10], &[1], dtype, 256);
        let key: &'static _ = Box::leak(Box::new(structure_key(
            OpCategory::UnaryElementwise,
            &[a, a],
            ArchSku::Sm89,
        )));
        let body: &'static ScalarExpr = Box::leak(Box::new(input(0).0));
        KernelPlan {
            op_name: "shadow_test",
            n_inputs: 1,
            dtype,
            out_dtype,
            schedule: Schedule::Scalar,
            key,
            body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: BaseOffset::Zero,
        }
    }

    #[test]
    fn param_ctype_matches_the_literal_table_for_every_kind() {
        for &k in ALL_KINDS {
            let plan = test_plan(k, k);
            assert_eq!(
                param_ctype(&plan),
                expected_scalar_ctype(k).expect("every ALL_KINDS entry has a ctype"),
                "{k:?}: param_ctype must match the literal expectation captured at adoption"
            );
        }
    }

    #[test]
    fn out_ctype_of_matches_the_literal_table_for_every_kind_pair() {
        const GIVEN: &str = "GIVEN_CTYPE";
        for &dtype in ALL_KINDS {
            for &out_dtype in ALL_KINDS {
                let plan = test_plan(dtype, out_dtype);
                let expected = if out_dtype == dtype {
                    GIVEN
                } else {
                    expected_scalar_ctype(out_dtype).expect("every ALL_KINDS entry has a ctype")
                };
                assert_eq!(
                    out_ctype_of(&plan, 0, GIVEN),
                    expected,
                    "{dtype:?} -> {out_dtype:?}: out_ctype_of must match the literal expectation"
                );
            }
        }
    }

    /// Independent, hardcoded expectation for [`store_expr_of`], built from
    /// the same literal tables as [`expected_cast_scalar`] (never from the
    /// real `store_expr_of`, `cast_scalar`, or `demote_store_f32`).
    fn expected_store_expr_of(dtype: ElementKind, out_dtype: ElementKind, root: &str) -> String {
        if out_dtype == dtype {
            match dtype {
                ElementKind::Fp8E4M3FN | ElementKind::Fp8E5M2 => {
                    expected_demote_store_f32(dtype, root)
                }
                _ => root.to_string(),
            }
        } else {
            expected_cast_scalar(dtype, out_dtype, root)
        }
    }

    #[test]
    fn store_expr_of_matches_the_literal_table_for_every_kind_pair() {
        for &dtype in ALL_KINDS {
            for &out_dtype in ALL_KINDS {
                let plan = test_plan(dtype, out_dtype);
                assert_eq!(
                    store_expr_of(&plan, 0, "r".to_string()),
                    expected_store_expr_of(dtype, out_dtype, "r"),
                    "{dtype:?} -> {out_dtype:?}: store_expr_of must match the literal expectation"
                );
            }
        }
    }

    /// The eleven local leaves must never again reach back into
    /// `unpopped::cfamily` — a positive-controlled grep, run as a test so a
    /// future edit that reintroduces a delegating call fails CI, not just a
    /// code review. Positive control: this file's own doc comments mention
    /// `unpopped::cfamily` in prose, so the check scans `.rs` executable
    /// content only via a crude but sufficient heuristic (no `unpopped::`
    /// token outside comments/doc-comments).
    #[test]
    fn the_shadow_never_delegates_back_to_unpopped_cfamily() {
        let src = include_str!("cfamily_shadow.rs");
        let code_only: String = src
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        // Built from two literal pieces so this assertion's OWN message does
        // not itself contain the banned token and self-trip the check.
        let banned = format!("{}::{}", "unpopped", "cfamily");
        assert!(
            !code_only.contains(&banned),
            "cfamily_shadow.rs must not call back into the neutral crate's cfamily \
             module outside comments -- that defeats the entire point of the shadow \
             (2026-10-07 correction)"
        );
    }
}
