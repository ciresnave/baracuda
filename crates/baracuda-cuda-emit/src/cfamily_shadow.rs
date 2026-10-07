//! Baracuda's own closed-set copy of four CUDA-specific scalar spellers that
//! used to be imported directly from the neutral `unpopped::cfamily` module:
//! [`scalar_ctype`], [`cast_scalar`], [`promote_load_f32`], [`demote_store_f32`].
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
//! These four bodies are a verbatim copy of `unpopped::cfamily`'s versions as
//! they existed on Unpopped `origin/main` at the time of this split (bodies
//! byte-identical; the two helper calls inside `promote_load_f32` /
//! `demote_store_f32` are qualified as `unpopped::cfamily::{narrow_load_fn,
//! narrow_store_fn}` since those two helpers are NOT part of this shadow —
//! only the four named functions are). Output is proven byte-identical to
//! calling `unpopped::cfamily`'s originals directly by the test at the
//! bottom of this file; it is NOT a promise that the two will stay identical
//! forever — baracuda now owns this copy and may diverge from it
//! deliberately, which is the entire point of the split.

use unpopped::cfamily::{narrow_load_fn, narrow_store_fn};
use unpopped_vocab::ElementKind;

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

    #[test]
    fn scalar_ctype_matches_unpopped_cfamily_for_every_kind() {
        for &k in ALL_KINDS {
            assert_eq!(
                scalar_ctype(k),
                unpopped::cfamily::scalar_ctype(k),
                "{k:?}: shadow must match unpopped::cfamily::scalar_ctype"
            );
        }
    }

    #[test]
    fn promote_load_f32_matches_unpopped_cfamily_for_every_kind() {
        for &k in ALL_KINDS {
            assert_eq!(
                promote_load_f32(k, "x"),
                unpopped::cfamily::promote_load_f32(k, "x"),
                "{k:?}: shadow must match unpopped::cfamily::promote_load_f32"
            );
        }
    }

    #[test]
    fn demote_store_f32_matches_unpopped_cfamily_for_every_kind() {
        for &k in ALL_KINDS {
            assert_eq!(
                demote_store_f32(k, "x"),
                unpopped::cfamily::demote_store_f32(k, "x"),
                "{k:?}: shadow must match unpopped::cfamily::demote_store_f32"
            );
        }
    }

    #[test]
    fn cast_scalar_matches_unpopped_cfamily_for_every_kind_pair() {
        for &from in ALL_KINDS {
            for &to in ALL_KINDS {
                // Only pairs where both dtypes have a scalar ctype are
                // meaningful (cast_scalar's arithmetic<->arithmetic arm calls
                // `scalar_ctype(to).expect(..)`, which panics otherwise on
                // some dtypes with no plain ctype arm for one of the two
                // float-detour branches). Skip pairs that would panic in
                // *either* implementation identically, rather than papering
                // over a real divergence with a narrower input space.
                if scalar_ctype(from).is_none() || scalar_ctype(to).is_none() {
                    continue;
                }
                assert_eq!(
                    cast_scalar(from, to, "x"),
                    unpopped::cfamily::cast_scalar(from, to, "x"),
                    "{from:?} -> {to:?}: shadow must match unpopped::cfamily::cast_scalar"
                );
            }
        }
    }
}
