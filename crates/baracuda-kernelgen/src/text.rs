//! Human-readable / -writable **text form** for an [`OpDef`] (IR-hub Phase 3).
//!
//! The neutral IR ([`crate::ir`]) is authored via the Rust EDSL
//! (`OpDef::elementwise(name, n_inputs, dtypes, body)` etc.). This module adds a
//! *round-tripping* text serialization so an `OpDef` can also be written in a
//! file, inspected, and diffed — the frontend (`lift.rs`) emits `OpDef`s and a
//! text form lets a developer read/author/review them without the EDSL.
//!
//! **Approach: serde.** Every IR type reachable from an `OpDef` derives
//! `serde::Serialize + serde::Deserialize`, so the whole IR — not just the
//! elementwise subset — round-trips through [`op_to_text`] / [`op_from_text`]
//! (pretty JSON). serde's *externally tagged* enum encoding is the readable
//! Lisp-ish shape the task calls for: an fma body
//! `Add(Mul(Input(0), Input(1)), Input(2))` prints as nested
//! `{"Add": [{"Mul": [{"Input": 0}, {"Input": 1}]}, {"Input": 2}]}`.
//!
//! Two IR-referenced types live in `baracuda-kernels-types` and do **not** derive
//! serde ([`baracuda_kernel_vocab::ElementKind`], a C-like dtype tag, and
//! [`baracuda_kernel_vocab::AxisMask`], a `u8` bitmask newtype). Rather than
//! modify that crate, each is bridged by a small `#[serde(with = "...")]` shim
//! below: a dtype serializes as its readable name string (`"F32"`), an axis mask
//! as its raw `u8`. A [`ScalarExpr::Const`](crate::ir::ScalarExpr::Const) `f64`
//! rides [`f64_repr`], which keeps finite values as plain JSON numbers (readable)
//! while encoding `NaN`/`±inf` as tagged strings so they survive the round trip
//! instead of collapsing to JSON `null`.

use crate::ir::OpDef;

/// Serialize an [`OpDef`] to the human-readable, round-trippable text form
/// (pretty-printed JSON). Inverse of [`op_from_text`]:
/// `op_from_text(&op_to_text(op)) == *op` for every well-formed `OpDef`.
///
/// Serialization of these IR types is infallible for the JSON backend (no
/// non-string map keys, no failing `Serialize` impls), so this returns a bare
/// `String`.
#[must_use]
pub fn op_to_text(op: &OpDef) -> String {
    serde_json::to_string_pretty(op).expect("OpDef -> JSON is infallible for serde_json")
}

/// Parse an [`OpDef`] from the text produced by [`op_to_text`]. Returns the
/// parse/shape error as a `String` on malformed input (unknown dtype name,
/// bad JSON, missing field, …).
///
/// # Errors
/// If `s` is not valid IR text (JSON parse error, unknown `ElementKind` name,
/// or a structural mismatch with the [`OpDef`] schema).
pub fn op_from_text(s: &str) -> Result<OpDef, String> {
    serde_json::from_str(s).map_err(|e| e.to_string())
}

/// Readable name string for an [`ElementKind`] — the Rust variant spelling, so
/// the text form is stable and self-documenting. Exhaustive: a new dtype variant
/// forces an update here (compile error), which is the intended coupling.
fn ek_to_str(k: baracuda_kernel_vocab::ElementKind) -> &'static str {
    use baracuda_kernel_vocab::ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, F16, F32, F32Strict, F64, Fp8E4M3, Fp8E5M2, I32,
        I64, S4, S8, U4, U8, U32,
    };
    match k {
        F16 => "F16",
        Bf16 => "Bf16",
        F32 => "F32",
        F32Strict => "F32Strict",
        F64 => "F64",
        S8 => "S8",
        U8 => "U8",
        I32 => "I32",
        I64 => "I64",
        Bool => "Bool",
        Fp8E4M3 => "Fp8E4M3",
        Fp8E5M2 => "Fp8E5M2",
        S4 => "S4",
        U4 => "U4",
        Bin => "Bin",
        Complex32 => "Complex32",
        Complex64 => "Complex64",
        U32 => "U32",
    }
}

/// Inverse of [`ek_to_str`].
fn ek_from_str(s: &str) -> Result<baracuda_kernel_vocab::ElementKind, String> {
    use baracuda_kernel_vocab::ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, F16, F32, F32Strict, F64, Fp8E4M3, Fp8E5M2, I32,
        I64, S4, S8, U4, U8, U32,
    };
    Ok(match s {
        "F16" => F16,
        "Bf16" => Bf16,
        "F32" => F32,
        "F32Strict" => F32Strict,
        "F64" => F64,
        "S8" => S8,
        "U8" => U8,
        "I32" => I32,
        "I64" => I64,
        "Bool" => Bool,
        "Fp8E4M3" => Fp8E4M3,
        "Fp8E5M2" => Fp8E5M2,
        "S4" => S4,
        "U4" => U4,
        "Bin" => Bin,
        "Complex32" => Complex32,
        "Complex64" => Complex64,
        "U32" => U32,
        other => return Err(format!("unknown ElementKind `{other}`")),
    })
}

/// `#[serde(with)]` bridge for a bare [`ElementKind`] field (dtype name string).
pub(crate) mod ek {
    use baracuda_kernel_vocab::ElementKind;
    use serde::{Deserialize, Deserializer, Serializer};

    pub(crate) fn serialize<S: Serializer>(v: &ElementKind, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(super::ek_to_str(*v))
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<ElementKind, D::Error> {
        let name = String::deserialize(d)?;
        super::ek_from_str(&name).map_err(serde::de::Error::custom)
    }
}

/// `#[serde(with)]` bridge for a `Vec<ElementKind>` field (a list of dtype names).
pub(crate) mod ek_vec {
    use baracuda_kernel_vocab::ElementKind;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(crate) fn serialize<S: Serializer>(v: &[ElementKind], s: S) -> Result<S::Ok, S::Error> {
        let names: Vec<&'static str> = v.iter().map(|&k| super::ek_to_str(k)).collect();
        names.serialize(s)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Vec<ElementKind>, D::Error> {
        let names = Vec::<String>::deserialize(d)?;
        names
            .iter()
            .map(|n| super::ek_from_str(n).map_err(serde::de::Error::custom))
            .collect()
    }
}

/// `#[serde(with)]` bridge for an `Option<ElementKind>` field.
pub(crate) mod ek_opt {
    use baracuda_kernel_vocab::ElementKind;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(crate) fn serialize<S: Serializer>(
        v: &Option<ElementKind>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        (*v).map(super::ek_to_str).serialize(s)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<ElementKind>, D::Error> {
        Option::<String>::deserialize(d)?
            .map(|n| super::ek_from_str(&n))
            .transpose()
            .map_err(serde::de::Error::custom)
    }
}

/// `#[serde(with)]` bridge for a `Vec<Option<ElementKind>>` field (per-extra-output
/// dtypes, each optionally uniform).
pub(crate) mod ek_opt_vec {
    use baracuda_kernel_vocab::ElementKind;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub(crate) fn serialize<S: Serializer>(
        v: &[Option<ElementKind>],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let names: Vec<Option<&'static str>> =
            v.iter().map(|o| (*o).map(super::ek_to_str)).collect();
        names.serialize(s)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Vec<Option<ElementKind>>, D::Error> {
        Vec::<Option<String>>::deserialize(d)?
            .into_iter()
            .map(|o| {
                o.map(|n| super::ek_from_str(&n))
                    .transpose()
                    .map_err(serde::de::Error::custom)
            })
            .collect()
    }
}

/// `#[serde(with)]` bridge for an [`AxisMask`](baracuda_kernel_vocab::AxisMask)
/// field — serialized as its raw `u8` bitmask.
pub(crate) mod axis {
    use baracuda_kernel_vocab::AxisMask;
    use serde::{Deserialize, Deserializer, Serializer};

    pub(crate) fn serialize<S: Serializer>(v: &AxisMask, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u8(v.0)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<AxisMask, D::Error> {
        Ok(AxisMask(u8::deserialize(d)?))
    }
}

/// `#[serde(with)]` bridge for a [`ScalarExpr::Const`](crate::ir::ScalarExpr::Const)
/// `f64`. Finite values serialize as plain JSON numbers (readable); the
/// non-finite `f64`s (`NaN`, `±inf`) serialize as tagged strings so they survive
/// the round trip rather than collapsing to JSON `null` (serde_json's default for
/// non-finite floats).
pub(crate) mod f64_repr {
    use serde::{Deserialize, Deserializer, Serializer};

    /// Untagged reader: a JSON number is a finite float; a JSON string is a
    /// non-finite tag.
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Repr {
        Num(f64),
        Tag(String),
    }

    pub(crate) fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_finite() {
            s.serialize_f64(*v)
        } else if v.is_nan() {
            s.serialize_str("NaN")
        } else if *v > 0.0 {
            s.serialize_str("inf")
        } else {
            s.serialize_str("-inf")
        }
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        match Repr::deserialize(d)? {
            Repr::Num(n) => Ok(n),
            Repr::Tag(t) => match t.as_str() {
                "NaN" | "nan" => Ok(f64::NAN),
                "inf" | "+inf" | "Infinity" => Ok(f64::INFINITY),
                "-inf" | "-Infinity" => Ok(f64::NEG_INFINITY),
                other => Err(serde::de::Error::custom(format!("bad f64 tag `{other}`"))),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{op_from_text, op_to_text};
    use crate::ir::{
        Access, BinaryOp, OobPolicy, OpDef, ReduceOp, ScalarExpr, UnaryOp, input, konst,
    };
    use baracuda_kernel_vocab::{AxisMask, ElementKind};

    /// A 3-input fused-multiply-add built via `OpDef::elementwise`, the canonical
    /// example: text is human-readable AND round-trips bit-for-bit.
    #[test]
    fn fma_round_trips_and_is_readable() {
        let op = OpDef::elementwise(
            "fma",
            3,
            &[ElementKind::F32],
            input(0) * input(1) + input(2),
        );

        let text = op_to_text(&op);
        // Human-readable: the structural tokens are all present as plain names.
        for tok in [
            "\"fma\"",
            "\"n_inputs\"",
            "Add",
            "Mul",
            "Input",
            "\"F32\"",
            "Elementwise",
        ] {
            assert!(text.contains(tok), "text form missing `{tok}`:\n{text}");
        }

        // Round-trips to an equal OpDef.
        let back = op_from_text(&text).expect("fma text parses");
        assert_eq!(back, op, "fma did not round-trip");

        // Explicit field + body checks (the task's assertions).
        assert_eq!(back.name, "fma");
        assert_eq!(back.n_inputs, 3);
        assert_eq!(back.dtypes, vec![ElementKind::F32]);
        assert_eq!(
            back.body,
            ScalarExpr::Add(
                Box::new(ScalarExpr::Mul(
                    Box::new(ScalarExpr::Input(0)),
                    Box::new(ScalarExpr::Input(1)),
                )),
                Box::new(ScalarExpr::Input(2)),
            ),
        );
    }

    /// A unary op (`relu`) round-trips; text carries the `Unary`/`Relu` tokens.
    #[test]
    fn unary_relu_round_trips() {
        let op = OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu());
        let text = op_to_text(&op);
        assert!(text.contains("Unary"), "missing Unary:\n{text}");
        assert!(text.contains("Relu"), "missing Relu:\n{text}");

        let back = op_from_text(&text).expect("relu parses");
        assert_eq!(back, op);
        assert_eq!(
            back.body,
            ScalarExpr::Unary(UnaryOp::Relu, Box::new(ScalarExpr::Input(0))),
        );
    }

    /// A body with a `Const` leaf and a non-infix `Binary` (max), plus two accepted
    /// dtypes — exercises the f64 `Const` codec and the `ElementKind`-list shim.
    #[test]
    fn const_and_binary_round_trips() {
        let op = OpDef::elementwise(
            "scaled_max",
            2,
            &[ElementKind::F32, ElementKind::F64],
            input(0).binary(BinaryOp::Max, input(1)) * konst(0.5),
        );
        let text = op_to_text(&op);
        for tok in ["Const", "0.5", "Binary", "Max", "\"F64\""] {
            assert!(text.contains(tok), "missing `{tok}`:\n{text}");
        }

        let back = op_from_text(&text).expect("scaled_max parses");
        assert_eq!(back, op);
        assert_eq!(back.dtypes, vec![ElementKind::F32, ElementKind::F64]);
    }

    /// A non-finite `Const` (`+inf`) survives the round trip as a tagged string
    /// instead of collapsing to JSON `null`.
    #[test]
    fn non_finite_const_round_trips() {
        let op = OpDef::elementwise(
            "inf_add",
            1,
            &[ElementKind::F32],
            input(0) + konst(f64::INFINITY),
        );
        let text = op_to_text(&op);
        assert!(text.contains("inf"), "missing inf tag:\n{text}");

        let back = op_from_text(&text).expect("inf parses");
        match &back.body {
            ScalarExpr::Add(_, rhs) => {
                assert!(
                    matches!(**rhs, ScalarExpr::Const(c) if c.is_infinite() && c > 0.0),
                    "rhs was not +inf: {rhs:?}",
                );
            }
            other => panic!("expected Add body, got {other:?}"),
        }
    }

    /// A reduction over an explicit `AxisMask` exercises the `AxisMask` serde shim
    /// on `Access::Reduction`, proving the text form covers non-elementwise ops.
    #[test]
    fn reduction_axismask_round_trips() {
        let mut mask = AxisMask::EMPTY;
        mask.set(1);
        let op = OpDef::reduction_axes(
            "sum_axis1",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            mask,
            true,
        );
        let text = op_to_text(&op);
        assert!(text.contains("Reduction"), "missing Reduction:\n{text}");

        let back = op_from_text(&text).expect("reduction parses");
        assert_eq!(back, op);
        // The mask survived (it rode the `u8` shim).
        match back.access {
            Access::Reduction { axes, keepdim, .. } => {
                assert!(axes.is_set(1) && keepdim);
            }
            other => panic!("expected Reduction access, got {other:?}"),
        }
    }

    /// A gather exercises the `ElementKind` shim nested inside `ReadIndex::Indexed`
    /// (`index_dtype`).
    #[test]
    fn gather_index_dtype_shim_round_trips() {
        let op = OpDef::gather(
            "gather_f32_i64",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::I64,
        );
        let text = op_to_text(&op);
        assert!(text.contains("Indexed"), "missing Indexed:\n{text}");
        assert!(text.contains("\"I64\""), "missing I64:\n{text}");

        let back = op_from_text(&text).expect("gather parses");
        assert_eq!(back, op);
    }
}
