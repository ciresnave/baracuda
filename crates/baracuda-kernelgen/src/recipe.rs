//! The **recipe** — a kernel's neutral KISS-Ops Semantics op-DAG (KISS-Contract
//! §2.3): the mandatory spine a recipe-verify importer runs to validate + register
//! an op it does not already know, and the decomposition that resolves an op to the
//! primitive floor.
//!
//! This re-bases the internal IR op names ([`crate::ir::ScalarExpr`] /
//! [`UnaryOp`] / [`BinaryOp`]) onto the single **KISS-Ops op-token set** — the same
//! op-vocab re-basing that lets the emitted contract carry a neutral op DAG instead
//! of a Fuel-private name. An op whose node has no confirmed KISS-Ops name yet is an
//! **honest miss**: [`semantics_dag`] returns `None` rather than fabricate a token.
//!
//! # Format — PROVISIONAL
//!
//! The Semantics text is a compact functional op-DAG — `op(arg, arg, …)`, leaves
//! `in<i>` (kernel input) and `const(<v>)` — e.g. a fused `relu_add` is
//! `add(relu(in0), in1)`. KISS §2.3 shows the DAG shape (`{ op: add }`, nested for a
//! fusion) but does not pin the exact literal grammar, so this spelling is a strawman
//! to co-pin with Fuel (the recipe format the importer parses), isolated here.
//!
//! # Scope (increment 1)
//!
//! Elementwise bodies over `Input`/`Const` leaves. Reductions, scans,
//! gather/scatter, scalar `Param`/`Coord`/`Reduced` leaves, and the OpAttrs channel
//! are follow-ups — an honest miss (`None`) until covered.

use crate::ir::{Access, BinaryOp, OpDef, ScalarExpr, UnaryOp};

/// The KISS-Ops op-DAG (recipe) for `op`, or `None` if `op` is not yet expressible
/// as a neutral recipe (non-elementwise access, or a node with no confirmed
/// KISS-Ops name).
#[must_use]
pub fn semantics_dag(op: &OpDef) -> Option<String> {
    // Increment 1: elementwise bodies only. Reductions, scans, gather/scatter,
    // pooling, and contractions carry structure (`Access`) the body alone doesn't
    // express, so their recipe is a follow-up — an honest miss for now.
    if !matches!(op.access, Access::Elementwise) {
        return None;
    }
    expr_to_recipe(&op.body)
}

/// Serialize a scalar expression as a KISS-Ops recipe sub-DAG, or `None` if any
/// node has no confirmed KISS-Ops re-basing (never fabricates a token).
fn expr_to_recipe(e: &ScalarExpr) -> Option<String> {
    use ScalarExpr as E;
    Some(match e {
        E::Input(i) => format!("in{i}"),
        E::Const(c) => format!("const({})", const_repr(*c)),
        E::Add(a, b) => format!("add({}, {})", expr_to_recipe(a)?, expr_to_recipe(b)?),
        E::Sub(a, b) => format!("sub({}, {})", expr_to_recipe(a)?, expr_to_recipe(b)?),
        E::Mul(a, b) => format!("mul({}, {})", expr_to_recipe(a)?, expr_to_recipe(b)?),
        E::Div(a, b) => format!("div({}, {})", expr_to_recipe(a)?, expr_to_recipe(b)?),
        E::Unary(u, x) => format!("{}({})", unary_kiss_name(*u)?, expr_to_recipe(x)?),
        E::Binary(op, x, y) => format!(
            "{}({}, {})",
            binary_kiss_name(*op)?,
            expr_to_recipe(x)?,
            expr_to_recipe(y)?
        ),
        E::Select(c, a, b) => format!(
            "select({}, {}, {})",
            expr_to_recipe(c)?,
            expr_to_recipe(a)?,
            expr_to_recipe(b)?
        ),
        // Param / Reduced / Coord leaves: not yet a recipe leaf (need the OpAttrs /
        // Access channels) — honest miss.
        E::Param(_) | E::Reduced(_) | E::Coord(_) => return None,
    })
}

/// Readable KISS-Ops `const` value: the finite number, or a non-finite tag.
fn const_repr(c: f64) -> String {
    if c.is_finite() {
        format!("{c}")
    } else if c.is_nan() {
        "nan".to_string()
    } else if c > 0.0 {
        "inf".to_string()
    } else {
        "-inf".to_string()
    }
}

/// The confirmed KISS-Ops op token for a [`UnaryOp`], or `None` if not yet re-based.
/// Only tokens verified present in the KISS-Ops op set are mapped; anything else
/// (e.g. `Round`, or a variant added after this arm set) is an honest miss.
fn unary_kiss_name(op: UnaryOp) -> Option<&'static str> {
    use UnaryOp as U;
    Some(match op {
        U::Neg => "neg",
        U::Abs => "abs",
        U::Sqr => "sqr",
        U::Sqrt => "sqrt",
        U::Rsqrt => "rsqrt",
        U::Recip => "recip",
        U::Exp => "exp",
        U::Log => "log",
        U::Tanh => "tanh",
        U::Sigmoid => "sigmoid",
        U::Relu => "relu",
        U::Erf => "erf",
        U::Gelu => "gelu",
        U::Silu => "silu",
        U::Sin => "sin",
        U::Cos => "cos",
        U::Floor => "floor",
        U::Ceil => "ceil",
        U::Sign => "sign",
        U::Step => "step",
        U::Erfc => "erfc",
        U::Trunc => "trunc",
        U::Exp2 => "exp2",
        U::Expm1 => "expm1",
        U::Log2 => "log2",
        U::Log10 => "log10",
        U::Log1p => "log1p",
        U::Sinh => "sinh",
        U::Cosh => "cosh",
        U::Tan => "tan",
        U::Asin => "asin",
        U::Acos => "acos",
        U::Atan => "atan",
        _ => return None,
    })
}

/// The confirmed KISS-Ops op token for a [`BinaryOp`], or `None` if not yet re-based.
fn binary_kiss_name(op: BinaryOp) -> Option<&'static str> {
    use BinaryOp as B;
    Some(match op {
        B::Max => "max",
        B::Min => "min",
        B::Pow => "pow",
        B::Atan2 => "atan2",
        B::Copysign => "copysign",
        B::FmaxIeee => "fmax_ieee",
        B::FminIeee => "fmin_ieee",
        B::RemTrunc => "rem_trunc",
        B::CmpEq => "cmp_eq",
        B::CmpNe => "cmp_ne",
        B::CmpLt => "cmp_lt",
        B::CmpLe => "cmp_le",
        B::CmpGt => "cmp_gt",
        B::CmpGe => "cmp_ge",
        B::BitAnd => "bit_and",
        B::BitOr => "bit_or",
        B::BitXor => "bit_xor",
        B::Shl => "shl",
        B::Shr => "shr",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Expr, OpDef, ScalarExpr, UnaryOp, input, konst};
    use baracuda_kernel_vocab::ElementKind::F32;

    #[test]
    fn primitive_add_recipe() {
        let op = OpDef::elementwise("add", 2, &[F32], input(0) + input(1));
        assert_eq!(semantics_dag(&op).as_deref(), Some("add(in0, in1)"));
    }

    #[test]
    fn fused_relu_add_recipe() {
        let op = OpDef::elementwise("relu_add", 2, &[F32], input(0).relu() + input(1));
        assert_eq!(semantics_dag(&op).as_deref(), Some("add(relu(in0), in1)"));
    }

    #[test]
    fn fma_recipe_nests() {
        let op = OpDef::elementwise("fma", 3, &[F32], input(0) * input(1) + input(2));
        assert_eq!(
            semantics_dag(&op).as_deref(),
            Some("add(mul(in0, in1), in2)")
        );
    }

    #[test]
    fn const_leaf_recipe() {
        let op = OpDef::elementwise("scale", 1, &[F32], input(0) * konst(0.5));
        assert_eq!(semantics_dag(&op).as_deref(), Some("mul(in0, const(0.5))"));
    }

    #[test]
    fn unmapped_node_is_an_honest_miss() {
        // A Coord leaf isn't expressible as a recipe leaf yet → None, never a wrong
        // token. Guards against fabricating a KISS-Ops name.
        let op = OpDef::elementwise("coordy", 1, &[F32], Expr(ScalarExpr::Coord(0)));
        assert_eq!(semantics_dag(&op), None);
    }

    #[test]
    fn round_has_no_confirmed_kiss_name_yet() {
        // `Round` is not (yet) a confirmed KISS-Ops token, so its recipe is withheld
        // rather than guessed.
        let op = OpDef::elementwise(
            "roundy",
            1,
            &[F32],
            Expr(ScalarExpr::Unary(
                UnaryOp::Round,
                Box::new(ScalarExpr::Input(0)),
            )),
        );
        assert_eq!(semantics_dag(&op), None);
    }
}
