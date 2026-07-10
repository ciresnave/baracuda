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

/// How a schedule variant's computed bits relate to the cell's default lowering.
/// Drives the selection policy (variants backlog doc): a [`VariantFidelity::BitIdentical`]
/// variant may be selected silently; anything else is selectable only through an
/// honest FKC contract (the caller's precision policy decides), never silently.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum VariantFidelity {
    /// Same result bits as the default lowering for every input.
    BitIdentical,
    /// Deterministic (fixed order for a fixed launch configuration), but a
    /// different operation *association* than the default — e.g. a split-K
    /// partial-sum tree vs the sequential fold.
    ReassociatedDeterministic,
    /// **Run-to-run non-deterministic** (increment 5, SCATTER): the result bits
    /// vary *between launches of the same configuration* because the schedule
    /// accumulates through order-varying floating-point atomics (`atomicAdd` on
    /// an FP cell whose completion order the hardware does not fix), and FP add
    /// is non-associative. This is strictly weaker than
    /// [`Self::ReassociatedDeterministic`] (which is at least stable for a fixed
    /// launch): a `Nondeterministic` variant can differ from ITSELF run to run.
    ///
    /// Per the house variant-selection rule this may **never** be selected
    /// silently — only through an honest FKC contract whose determinism block
    /// flips to Fuel's `nondeterministic` spelling (and, per Fuel's precision
    /// coherence rule `fuel-dispatch fkc/validate.rs`, that contract must also
    /// carry `bit_stable_on_same_hardware: false` + `audited: true`). The
    /// deterministic default (a gather-sum or sorted-segment sweep — the bespoke
    /// `segment_sorted_kernel` precedent) stays the base route.
    Nondeterministic,
    /// **Strictly more accurate** than the default lowering, and deterministic.
    /// The default f32 reduction accumulates in `float` (error growing with the
    /// reduced length); this variant forces a `double` accumulator and a
    /// no-reassociation serial fold, yielding ~0.5 ULP(f32) of the correctly-
    /// rounded reduction — a *directed* "closer to the true reduction" guarantee.
    /// That directedness is the whole selection signal, and it is why this is
    /// neither [`Self::BitIdentical`] (it differs from the default bits, so it
    /// must never be chosen silently) nor [`Self::ReassociatedDeterministic`]
    /// (which is same-accuracy-different-rounding, undirected).
    ///
    /// The serial double fold is bitwise-reproducible on any IEEE-754 double
    /// hardware (fixed order, per-op-deterministic), so its determinism spelling
    /// is the strongest — `bitwise`, not `same_hardware_bitwise`. Selectable only
    /// through an honest FKC contract whose precision block advertises the tighter
    /// bound; the caller's precision policy decides.
    MorePrecise,
}

impl VariantFidelity {
    /// The Fuel FKC `determinism:` block spelling for this fidelity — the exact
    /// string Fuel's contract schema accepts (`fuel-dispatch fkc/schema.rs`:
    /// `bitwise` | `same_hardware_bitwise` | `nondeterministic`). Used when a
    /// variant's contract is emitted so the determinism block flips **honestly**
    /// with the schedule's numeric class (never a hardcoded `bitwise`).
    ///
    /// `Nondeterministic` maps to `nondeterministic`, which — per Fuel's
    /// precision coherence rule (`fkc/validate.rs` Rule 9) — additionally
    /// obligates the emitted precision block to declare
    /// `bit_stable_on_same_hardware: false` + `audited: true`.
    #[must_use]
    pub fn determinism_str(self) -> &'static str {
        match self {
            VariantFidelity::BitIdentical => "bitwise",
            VariantFidelity::ReassociatedDeterministic => "same_hardware_bitwise",
            VariantFidelity::Nondeterministic => "nondeterministic",
            // A serial double fold is reproducible across IEEE-754 hardware.
            VariantFidelity::MorePrecise => "bitwise",
        }
    }
}

/// One alternative schedule for a cell: a tagged set of kernels (most variants
/// are a single kernel; a split-K pair is two, in launch order) plus the launch
/// protocol the contract must carry. The ship-top-K policy: every validated
/// variant ships with its own contract under the same `accept.structure_key`;
/// the dispatch table records Baracuda's measured default, and Fuel remains the
/// runtime selector.
#[derive(Clone, Debug)]
pub struct Variant {
    /// Short stable tag (`"base"`, `"splitk"`, `"unroll4"`, …). Rides in the
    /// generated symbol names and, eventually, contract front-matter (opaque on
    /// the wire — the entry point stays the true identity).
    pub tag: &'static str,
    /// The kernels implementing this variant, in launch order.
    pub kernels: Vec<GeneratedKernel>,
    /// Bit relationship to the default lowering.
    pub fidelity: VariantFidelity,
    /// Launch protocol (grids, workspace sizing, chunking) — contract-facing.
    pub launch_note: String,
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
    /// Alternative schedule variants for this plan's cell, beyond the default
    /// [`Backend::lower`] kernel. Default: none. Every returned variant must
    /// pass the same validation gate as the default (nvrtc/nvcc compile +
    /// numeric oracle + sanitizer where the schedule warrants) before it is
    /// shipped or ranked by the bench gate.
    fn lower_variants(&self, _plan: &crate::plan::KernelPlan<'_>) -> Vec<Variant> {
        Vec::new()
    }
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
    /// Output-coordinate spelling ([`ScalarExpr::Coord`], increment 0d): the
    /// per-axis coordinate of the output element, cast to the compute dtype
    /// (`(float)c{d}` in the CUDA strided emitter). Only the strided
    /// elementwise emitter materializes coordinates; every other emitter
    /// passes a panicking closure — the plan gate routes Coord bodies to
    /// `Schedule::Strided`, and those closures are the per-emitter backstop.
    pub coord: &'a dyn Fn(u8) -> String,
    /// Unary-op spelling.
    pub unary: &'a dyn Fn(UnaryOp, String) -> String,
    /// Binary-function-op spelling.
    pub binary: &'a dyn Fn(BinaryOp, String, String) -> String,
    /// Ternary select spelling ([`ScalarExpr::Select`]) over the three
    /// already-lowered operand strings `(cond, a, b)`. Its own seam — the
    /// 2-operand `binary` closure cannot carry three operands, and the select
    /// spelling has its own bitwise contract (the arms must move raw bits, so
    /// it must NEVER route through a promote-demote wrapper; see
    /// `cuda::cuda_select`). Emitters whose bodies can never contain a Select
    /// (the packed f16/bf16 pair path — `body_packs` excludes it) pass a
    /// panicking closure, the `coord` precedent.
    pub select: &'a dyn Fn(String, String, String) -> String,
}

impl std::fmt::Debug for Lowering<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lowering").finish_non_exhaustive()
    }
}

/// Spell an `f64` constant as a valid C literal. `{v:?}` emits `inf`/`NaN`, which
/// aren't valid C literals; map the non-finite cases to the standard macros.
///
/// The f32 `f`-suffix vs double-promotion question is dtype-dependent and tracked
/// as a follow-up — but it is **not** purely a perf concern: the optimizer's
/// bit-preservation contract (e.g. the `x/2^k -> x*2^-k` rule) and the packed
/// path's const gate are proven against the current double-promoted,
/// correctly-rounded semantics. Changing the const spelling (or compiling with
/// `--use_fast_math`) invalidates those proofs; re-verify the rule set first.
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
        ScalarExpr::Coord(d) => (lo.coord)(*d),
        ScalarExpr::Param(i) => format!("p{i}"),
        ScalarExpr::Const(v) => const_lit(*v),
        ScalarExpr::Unary(op, x) => (lo.unary)(*op, lower_expr(x, lo)),
        ScalarExpr::Binary(op, a, b) => (lo.binary)(*op, lower_expr(a, lo), lower_expr(b, lo)),
        ScalarExpr::Select(c, a, b) => {
            (lo.select)(lower_expr(c, lo), lower_expr(a, lo), lower_expr(b, lo))
        }
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
    let policy = HoistPolicy {
        hoist_all: false,
        hoist_shared_leaves: false,
        extra_uses: &[],
    };
    let root_ref = lower_node(dag, dag.root(), ctype, lo, &mut refs, &mut prelude, &policy);
    (prelude, root_ref)
}

/// Hoisting policy for [`lower_node`] — how aggressively a value is bound to a
/// named `tmp` vs inlined at its use site.
struct HoistPolicy<'u> {
    /// Hoist EVERY non-leaf (the packed pair-split path — see [`lower_dag_all`]).
    hoist_all: bool,
    /// Also hoist a **shared `Input` leaf** (a memory load referenced by more
    /// than one use) — the multi-output path, so the shared `dy` load appears
    /// once. Single-output paths keep leaves inlined (a leaf ref is free).
    hoist_shared_leaves: bool,
    /// Per-node "extra use" count beyond the intra-DAG `consumers` edges — the
    /// multi-output root multiplicity (a node that is the output root of `k`
    /// bodies has `k` extra uses). Empty ⇒ all zero (single-output). Combined
    /// with `consumers` this is the node's total use count, which decides
    /// sharing.
    extra_uses: &'u [u32],
}

impl HoistPolicy<'_> {
    /// Total uses of `id` = intra-DAG consumer edges + multi-output root uses.
    fn total_uses(&self, dag: &ExprDag, id: NodeId) -> u32 {
        dag.consumers(id) + self.extra_uses.get(id as usize).copied().unwrap_or(0)
    }
}

/// Lower a **multi-root** DAG (one root per output body, from
/// [`ExprDag::from_exprs`]) to `(prelude, root_refs)` — the cross-body-CSE core
/// of the multi-output emitter. All roots are lowered against ONE shared `refs`
/// memo and ONE shared `prelude`, so a subexpression shared between outputs is
/// emitted once and referenced by each store. Beyond the intra-body sharing
/// [`lower_dag`] already hoists, this additionally hoists:
///
/// - a value used by more than one output body (a shared interior, or a node
///   that is one body's root and another body's interior), via the
///   root-multiplicity `extra_uses`;
/// - a **shared `Input` leaf** — the shared `dy` load — so it appears in the
///   source exactly once (the "strictly fewer global loads" win).
///
/// `hoist_all` mirrors [`lower_dag_all`] for the packed pair-split path (every
/// non-leaf a `tmp`). `root_refs[j]` names output body `j`'s value; a node that
/// is the sole use of its value inlines exactly as the single-output path does.
#[must_use]
pub fn lower_dag_multi(
    dag: &ExprDag,
    ctype: &str,
    lo: &Lowering<'_>,
    hoist_all: bool,
) -> (Vec<String>, Vec<String>) {
    // Root multiplicity: how many output bodies name each node as their root.
    // A node that is a body root AND referenced by another node (or the root of
    // two bodies) has total_uses > 1, so it hoists once rather than re-emitting.
    let mut extra_uses = vec![0u32; dag.len()];
    for &r in dag.roots() {
        extra_uses[r as usize] += 1;
    }
    let policy = HoistPolicy {
        hoist_all,
        hoist_shared_leaves: true,
        extra_uses: &extra_uses,
    };
    let mut refs: Vec<Option<String>> = vec![None; dag.len()];
    let mut prelude: Vec<String> = Vec::new();
    let root_refs = dag
        .roots()
        .iter()
        .map(|&r| lower_node(dag, r, ctype, lo, &mut refs, &mut prelude, &policy))
        .collect();
    (prelude, root_refs)
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
    let policy = HoistPolicy {
        hoist_all: true,
        hoist_shared_leaves: false,
        extra_uses: &[],
    };
    let root_ref = lower_node(dag, dag.root(), ctype, lo, &mut refs, &mut prelude, &policy);
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
    policy: &HoistPolicy<'_>,
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
        DagNode::Coord(d) => (lo.coord)(d),
        DagNode::Param(i) => format!("p{i}"),
        DagNode::Const(v) => const_lit(v),
        DagNode::Unary(op, x) => {
            (lo.unary)(op, lower_node(dag, x, ctype, lo, refs, prelude, policy))
        }
        DagNode::Binary(op, a, b) => {
            let a = lower_node(dag, a, ctype, lo, refs, prelude, policy);
            let b = lower_node(dag, b, ctype, lo, refs, prelude, policy);
            (lo.binary)(op, a, b)
        }
        DagNode::Select(c, a, b) => {
            let c = lower_node(dag, c, ctype, lo, refs, prelude, policy);
            let a = lower_node(dag, a, ctype, lo, refs, prelude, policy);
            let b = lower_node(dag, b, ctype, lo, refs, prelude, policy);
            (lo.select)(c, a, b)
        }
        DagNode::Add(a, b) => {
            let a = lower_node(dag, a, ctype, lo, refs, prelude, policy);
            let b = lower_node(dag, b, ctype, lo, refs, prelude, policy);
            format!("({a} + {b})")
        }
        DagNode::Sub(a, b) => {
            let a = lower_node(dag, a, ctype, lo, refs, prelude, policy);
            let b = lower_node(dag, b, ctype, lo, refs, prelude, policy);
            format!("({a} - {b})")
        }
        DagNode::Mul(a, b) => {
            let a = lower_node(dag, a, ctype, lo, refs, prelude, policy);
            let b = lower_node(dag, b, ctype, lo, refs, prelude, policy);
            format!("({a} * {b})")
        }
        DagNode::Div(a, b) => {
            let a = lower_node(dag, a, ctype, lo, refs, prelude, policy);
            let b = lower_node(dag, b, ctype, lo, refs, prelude, policy);
            format!("({a} / {b})")
        }
    };
    // Hoisting decision:
    // - a non-leaf hoists when `hoist_all` (packed pair-split), or when it is
    //   shared — total uses > 1 (intra-body consumer edges + multi-output root
    //   multiplicity). A single-use root has total_uses <= 1, so it inlines
    //   into its store, byte-identical to `lower_expr`.
    // - a leaf normally inlines (a leaf ref is free); under `hoist_shared_leaves`
    //   (multi-output) a shared `Input` LEAF — a memory load referenced by more
    //   than one output — hoists so the load appears once. Const/Param/Coord/
    //   Reduced leaves stay inlined (no load to dedup; multi-output bodies carry
    //   no Coord/Reduced anyway).
    let shared = policy.total_uses(dag, id) > 1;
    let hoist = if node.is_leaf() {
        policy.hoist_shared_leaves && matches!(node, DagNode::Input(_)) && shared
    } else {
        policy.hoist_all || shared
    };
    let r = if hoist {
        let name = format!("tmp{}", prelude.len());
        prelude.push(format!("{ctype} {name} = {rhs};"));
        name
    } else {
        rhs
    };
    refs[id as usize] = Some(r.clone());
    r
}
