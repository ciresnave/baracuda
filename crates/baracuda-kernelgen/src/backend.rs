//! Backend abstraction — the one language-specific seam.
//!
//! Everything else in the crate (the [`crate::ir`] op IR and the schedule
//! decision in [`crate::plan`]) is language-agnostic. A [`Backend`] lowers a
//! neutral [`crate::plan::KernelPlan`] to concrete kernel source. CUDA is the
//! first impl ([`crate::cuda::Cuda`]); Slang / SPIR-V / Metal / CPU backends
//! slot in as additional impls without touching the core — which is what lets
//! this generator eventually target backends beyond CUDA (and move out of
//! Baracuda) without a rewrite.

use crate::ir::{BinaryOp, DagNode, ExprDag, NodeId, ScalarExpr, UnaryOp};
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

/// Spell an `f64` constant as a valid C literal. `{v:?}` emits `inf`/`NaN`, which
/// aren't valid C literals; map the non-finite cases to the standard macros.
/// (The f32 `f`-suffix vs double-promotion is dtype-dependent and tracked as a
/// follow-up — a perf, not correctness, concern since the result narrows back.)
#[must_use]
pub fn const_lit(v: f64) -> String {
    if v.is_nan() {
        "NAN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 {
            "INFINITY".to_string()
        } else {
            "-INFINITY".to_string()
        }
    } else {
        format!("{v:?}")
    }
}

/// Lower a [`ScalarExpr`] tree to a backend expression string via `lo`'s seams.
///
/// Structural: a subtree reachable by two paths is re-rendered once per path. For
/// shared-interior dedup (emit a value once as a `tmp`), lower an [`ExprDag`] via
/// [`lower_dag`] instead — this remains the inlining primitive both paths share
/// (single-use nodes lower identically through either).
#[must_use]
pub fn lower_expr(e: &ScalarExpr, lo: &Lowering<'_>) -> String {
    match e {
        ScalarExpr::Input(i) => (lo.leaf)(*i),
        ScalarExpr::Reduced(i) => (lo.reduced)(*i),
        ScalarExpr::Param(i) => format!("p{i}"),
        ScalarExpr::Const(v) => const_lit(*v),
        ScalarExpr::Unary(op, x) => (lo.unary)(*op, lower_expr(x, lo)),
        ScalarExpr::Binary(op, a, b) => (lo.binary)(*op, lower_expr(a, lo), lower_expr(b, lo)),
        ScalarExpr::Add(a, b) => format!("({} + {})", lower_expr(a, lo), lower_expr(b, lo)),
        ScalarExpr::Sub(a, b) => format!("({} - {})", lower_expr(a, lo), lower_expr(b, lo)),
        ScalarExpr::Mul(a, b) => format!("({} * {})", lower_expr(a, lo), lower_expr(b, lo)),
        ScalarExpr::Div(a, b) => format!("({} / {})", lower_expr(a, lo), lower_expr(b, lo)),
    }
}

/// Lower an [`ExprDag`] to `(prelude, root_ref)`.
///
/// `prelude` is the block of `<ctype> tmpN = <expr>;` statements — one per shared
/// non-leaf node, in topological order (a `tmp`'s RHS references only earlier
/// `tmp`s / inlined leaves) — that the caller emits before the use site.
/// `root_ref` names the DAG's output value.
///
/// A node with `consumers <= 1`, or any leaf, is **inlined** at its use site;
/// only a shared *interior* (`consumers > 1`, non-leaf) is hoisted. So for a body
/// with no shared interior the prelude is empty and `root_ref` is byte-identical
/// to [`lower_expr`] — the DAG is transparent for every single-use body, which is
/// the no-regression guarantee for existing goldens.
#[must_use]
pub fn lower_dag(dag: &ExprDag, ctype: &str, lo: &Lowering<'_>) -> (Vec<String>, String) {
    let mut refs: Vec<Option<String>> = vec![None; dag.len()];
    let mut prelude: Vec<String> = Vec::new();
    let root_ref = lower_node(dag, dag.root(), ctype, lo, &mut refs, &mut prelude, false);
    (prelude, root_ref)
}

/// [`lower_dag`], but hoisting **every** non-leaf node to a named `tmp` (not
/// just shared ones). For lowerings whose op spellings reference an operand
/// string more than once (e.g. the packed f16/bf16 pair-scalarization, which
/// splits one operand into `__low2half(x)` / `__high2half(x)`), inlining would
/// duplicate whole subexpression *text* per reference — exponential in depth.
/// Hoist-all makes every operand a `tmp` name, so a duplicate is a name, never
/// an expression, and the emitted source stays linear. Values are unchanged.
#[must_use]
pub fn lower_dag_all(dag: &ExprDag, ctype: &str, lo: &Lowering<'_>) -> (Vec<String>, String) {
    let mut refs: Vec<Option<String>> = vec![None; dag.len()];
    let mut prelude: Vec<String> = Vec::new();
    let root_ref = lower_node(dag, dag.root(), ctype, lo, &mut refs, &mut prelude, true);
    (prelude, root_ref)
}

/// Post-order, memoized lowering of one DAG node. Emits a shared interior once
/// (into `prelude`) and returns the string every use site references (a `tmpN`
/// name for a hoisted node, the inlined expression otherwise).
fn lower_node(
    dag: &ExprDag,
    id: NodeId,
    ctype: &str,
    lo: &Lowering<'_>,
    refs: &mut Vec<Option<String>>,
    prelude: &mut Vec<String>,
    hoist_all: bool,
) -> String {
    if let Some(r) = &refs[id as usize] {
        return r.clone();
    }
    // Copy the node out (all fields are `Copy`) so the immutable borrow of `dag`
    // is released before the `&mut refs`/`&mut prelude` recursion.
    let node = dag.node(id).clone();
    let rhs = match node {
        DagNode::Input(i) => (lo.leaf)(i),
        DagNode::Reduced(i) => (lo.reduced)(i),
        DagNode::Param(i) => format!("p{i}"),
        DagNode::Const(v) => const_lit(v),
        DagNode::Unary(op, x) => {
            (lo.unary)(op, lower_node(dag, x, ctype, lo, refs, prelude, hoist_all))
        }
        DagNode::Binary(op, a, b) => {
            let a = lower_node(dag, a, ctype, lo, refs, prelude, hoist_all);
            let b = lower_node(dag, b, ctype, lo, refs, prelude, hoist_all);
            (lo.binary)(op, a, b)
        }
        DagNode::Add(a, b) => {
            let a = lower_node(dag, a, ctype, lo, refs, prelude, hoist_all);
            let b = lower_node(dag, b, ctype, lo, refs, prelude, hoist_all);
            format!("({a} + {b})")
        }
        DagNode::Sub(a, b) => {
            let a = lower_node(dag, a, ctype, lo, refs, prelude, hoist_all);
            let b = lower_node(dag, b, ctype, lo, refs, prelude, hoist_all);
            format!("({a} - {b})")
        }
        DagNode::Mul(a, b) => {
            let a = lower_node(dag, a, ctype, lo, refs, prelude, hoist_all);
            let b = lower_node(dag, b, ctype, lo, refs, prelude, hoist_all);
            format!("({a} * {b})")
        }
        DagNode::Div(a, b) => {
            let a = lower_node(dag, a, ctype, lo, refs, prelude, hoist_all);
            let b = lower_node(dag, b, ctype, lo, refs, prelude, hoist_all);
            format!("({a} / {b})")
        }
    };
    // Hoist a shared interior (edge count > 1, non-leaf) to a named tmp so it is
    // computed once; inline everything else (byte-identical to `lower_expr`). The
    // root has consumers == 0 (nothing references it), so it is never hoisted —
    // unless `hoist_all`, which hoists every non-leaf including the root.
    let r = if !node.is_leaf() && (hoist_all || dag.consumers(id) > 1) {
        let name = format!("tmp{}", prelude.len());
        prelude.push(format!("{ctype} {name} = {rhs};"));
        name
    } else {
        rhs
    };
    refs[id as usize] = Some(r.clone());
    r
}
