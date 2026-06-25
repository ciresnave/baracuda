//! Backend abstraction — the one language-specific seam.
//!
//! Everything else in the crate (the [`crate::ir`] op IR and the schedule
//! decision in [`crate::plan`]) is language-agnostic. A [`Backend`] lowers a
//! neutral [`crate::plan::KernelPlan`] to concrete kernel source. CUDA is the
//! first impl ([`crate::cuda::Cuda`]); Slang / SPIR-V / Metal / CPU backends
//! slot in as additional impls without touching the core — which is what lets
//! this generator eventually target backends beyond CUDA (and move out of
//! Baracuda) without a rewrite.

use crate::ir::{BinaryOp, ScalarExpr, UnaryOp};
use baracuda_kernels_types::ElementKind;

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
    /// Whether the backend can lower `dtype` to a scalar type at all. The JIT
    /// trust boundary checks this *before* [`Backend::lower`] so an unlowerable
    /// dtype is a typed decline, not a lowering panic. (AOT op authoring is
    /// trusted, so `lower` itself may still panic on a dtype it can't spell.)
    fn supports_dtype(&self, dtype: ElementKind) -> bool;
}

/// Backend-injected lowering closures for the **non-universal** parts of the
/// math. Infix `+ - * /` and parenthesization are universal across
/// CUDA/Slang/HLSL/Metal/GLSL and inlined directly; everything else is a seam:
///
/// - `leaf` — how input operand `i`'s value is named (`in0[i]` scalar, `v0.x`
///   for a vector lane);
/// - `unary` — spells a [`UnaryOp`] over an already-lowered inner string
///   (`expf(...)` is CUDA-specific);
/// - `binary` — spells a non-infix [`BinaryOp`] over two operand strings
///   (`fmaxf(a, b)`, `powf(a, b)`).
pub struct Lowering<'a> {
    /// Operand-access spelling.
    pub leaf: &'a dyn Fn(u8) -> String,
    /// Per-row reduced-scalar spelling ([`ScalarExpr::Reduced`]). Only the
    /// `RowReduce` emitter produces a body containing a `Reduced` leaf; every other
    /// emitter passes a closure that panics (its bodies never contain one).
    pub reduced: &'a dyn Fn(u8) -> String,
    /// Unary-op spelling.
    pub unary: &'a dyn Fn(UnaryOp, String) -> String,
    /// Binary-function-op spelling.
    pub binary: &'a dyn Fn(BinaryOp, String, String) -> String,
}

impl std::fmt::Debug for Lowering<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lowering").finish_non_exhaustive()
    }
}

/// Lower a [`ScalarExpr`] DAG to a backend expression string via `lo`'s seams.
#[must_use]
pub fn lower_expr(e: &ScalarExpr, lo: &Lowering<'_>) -> String {
    match e {
        ScalarExpr::Input(i) => (lo.leaf)(*i),
        ScalarExpr::Reduced(i) => (lo.reduced)(*i),
        ScalarExpr::Param(i) => format!("p{i}"),
        // `{v:?}` emits `inf`/`NaN`, which aren't valid C literals; map to the
        // standard macros. (The f32 `f`-suffix vs double-promotion is dtype-
        // dependent and tracked as a follow-up — it's a perf, not correctness,
        // concern since the result narrows back.)
        ScalarExpr::Const(v) => {
            if v.is_nan() {
                "NAN".to_string()
            } else if v.is_infinite() {
                if *v > 0.0 {
                    "INFINITY".to_string()
                } else {
                    "-INFINITY".to_string()
                }
            } else {
                format!("{v:?}")
            }
        }
        ScalarExpr::Unary(op, x) => (lo.unary)(*op, lower_expr(x, lo)),
        ScalarExpr::Binary(op, a, b) => (lo.binary)(*op, lower_expr(a, lo), lower_expr(b, lo)),
        ScalarExpr::Add(a, b) => format!("({} + {})", lower_expr(a, lo), lower_expr(b, lo)),
        ScalarExpr::Sub(a, b) => format!("({} - {})", lower_expr(a, lo), lower_expr(b, lo)),
        ScalarExpr::Mul(a, b) => format!("({} * {})", lower_expr(a, lo), lower_expr(b, lo)),
        ScalarExpr::Div(a, b) => format!("({} / {})", lower_expr(a, lo), lower_expr(b, lo)),
    }
}
