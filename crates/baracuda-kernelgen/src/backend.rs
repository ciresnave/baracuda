//! Backend abstraction — the one language-specific seam.
//!
//! Everything else in the crate (the [`crate::ir`] op IR and the schedule
//! decision in [`crate::plan`]) is language-agnostic. A [`Backend`] lowers a
//! neutral [`crate::plan::KernelPlan`] to concrete kernel source. CUDA is the
//! first impl ([`crate::cuda::Cuda`]); Slang / SPIR-V / Metal / CPU backends
//! slot in as additional impls without touching the core — which is what lets
//! this generator eventually target backends beyond CUDA (and move out of
//! Baracuda) without a rewrite.

use crate::ir::{ScalarExpr, UnaryOp};

/// A generated kernel: its exported symbol name and source text.
#[derive(Clone, Debug)]
pub struct GeneratedKernel {
    /// The exported (`extern "C"` or backend-equivalent) symbol name.
    pub name: String,
    /// The kernel source text, in the backend's language.
    pub source: String,
}

/// Lowers a neutral [`crate::plan::KernelPlan`] to concrete kernel source.
pub trait Backend {
    /// Short backend identifier (e.g. `"cuda"`).
    fn name(&self) -> &str;
    /// Lower a kernel plan to source.
    fn lower(&self, plan: &crate::plan::KernelPlan<'_>) -> GeneratedKernel;
}

/// Lower a [`ScalarExpr`] DAG to an infix expression string.
///
/// Two backend seams, because the math half splits cleanly: `+ - * /` and
/// parenthesization are **universal** across CUDA/Slang/HLSL/Metal/GLSL, so
/// only `leaf` (how input operand `i`'s value is named — `in0[i]` scalar, `v0.x`
/// for a vector lane) is backend-specific for those. Transcendentals are **not**
/// universal (`expf` is CUDA-specific), so `unary` is a second backend-injected
/// seam: it spells a [`UnaryOp`] applied to an already-lowered inner string.
#[must_use]
pub fn lower_expr(
    e: &ScalarExpr,
    leaf: &dyn Fn(u8) -> String,
    unary: &dyn Fn(UnaryOp, String) -> String,
) -> String {
    match e {
        ScalarExpr::Input(i) => leaf(*i),
        ScalarExpr::Const(v) => format!("{v:?}"),
        ScalarExpr::Unary(op, x) => unary(*op, lower_expr(x, leaf, unary)),
        ScalarExpr::Add(a, b) => {
            format!("({} + {})", lower_expr(a, leaf, unary), lower_expr(b, leaf, unary))
        }
        ScalarExpr::Sub(a, b) => {
            format!("({} - {})", lower_expr(a, leaf, unary), lower_expr(b, leaf, unary))
        }
        ScalarExpr::Mul(a, b) => {
            format!("({} * {})", lower_expr(a, leaf, unary), lower_expr(b, leaf, unary))
        }
        ScalarExpr::Div(a, b) => {
            format!("({} / {})", lower_expr(a, leaf, unary), lower_expr(b, leaf, unary))
        }
    }
}
