//! Backend abstraction — the one language-specific seam.
//!
//! Everything else in the crate (the [`crate::ir`] op IR and the schedule
//! decision in [`crate::plan`]) is language-agnostic. A [`Backend`] lowers a
//! neutral [`crate::plan::KernelPlan`] to concrete kernel source. CUDA is the
//! first impl ([`crate::cuda::Cuda`]); Slang / SPIR-V / Metal / CPU backends
//! slot in as additional impls without touching the core — which is what lets
//! this generator eventually target backends beyond CUDA (and move out of
//! Baracuda) without a rewrite.

use crate::ir::ScalarExpr;

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
/// **Language-neutral**: `+ - * /` and parenthesization are common to CUDA,
/// Slang, HLSL, Metal, and GLSL, so the only backend-specific input is `acc` —
/// how input operand `i`'s value is named at the current position (e.g.
/// `in0[i]` scalar, `v0.x` for a vector lane). That single seam is why the math
/// half of codegen is portable while the memory/launch half is not.
#[must_use]
pub fn lower_expr(e: &ScalarExpr, acc: &dyn Fn(u8) -> String) -> String {
    match e {
        ScalarExpr::Input(i) => acc(*i),
        ScalarExpr::Const(v) => format!("{v:?}"),
        ScalarExpr::Add(a, b) => format!("({} + {})", lower_expr(a, acc), lower_expr(b, acc)),
        ScalarExpr::Sub(a, b) => format!("({} - {})", lower_expr(a, acc), lower_expr(b, acc)),
        ScalarExpr::Mul(a, b) => format!("({} * {})", lower_expr(a, acc), lower_expr(b, acc)),
        ScalarExpr::Div(a, b) => format!("({} / {})", lower_expr(a, acc), lower_expr(b, acc)),
    }
}
