//! JIT-on-request synthesis — Baracuda as the **synthesizer** (Kernel-Seam §5).
//!
//! The division of labor is fixed by the constitution (§5.1): **Fuel is the
//! strategist** — it chooses *which* primitive-subgraph region to fuse, *when*,
//! and whether to *adopt* the result (cost-gated); **Baracuda is the synthesizer**
//! — it builds the best kernel for the **Fuel-chosen** region and returns it. No
//! backend-side opportunity-finding: we never scan a graph to pick regions, we
//! only synthesize the one we're handed.
//!
//! A [`JitRequest`] carries that region (a graph-`Op` subgraph, the same shape
//! [`derive_pattern`] emits — read in reverse), the operand projection that keys
//! the schedule, and a target. [`synthesize`] turns it into a [`JitResponse`] =
//! `(kernel + FKC contract + recipe + link row)`, the §5 shape. The heavy lifting
//! reuses the AOT generator ([`generate`], [`contract`], [`derive_pattern`]); the
//! only new step is [`region_to_op`] (region → op IR) and the on-demand
//! [`Compiler`] seam.
//!
//! # Scope (increment 1)
//!
//! The elementwise-epilogue vocabulary the IR already covers ([`ScalarExpr`]):
//! `Add`/`Sub`/`Mul`/`Div`, the scalar-param ops `AddScalar`/`MulScalar`, and the
//! unary math/activations, **single (uniform) dtype**. The on-demand compiler is
//! behind a trait with a stub impl; the real nvrtc backend, the FFI wire surface
//! (reconciling these Rust types with Fuel's `JitRequest`/`JitResponse`), an
//! inward e-graph optimizer (§5.1 permits it), per-operand dtypes, and the
//! telemetry trigger are the growth path.

use crate::contract::contract;
use crate::ir::{Access, BinaryOp, OpDef, ScalarExpr, UnaryOp};
use crate::link::{LinkEntry, link_entry};
use crate::optimize::optimize;
use crate::pattern::{PatternError, PatternNode, derive_pattern, to_fkc};
use crate::{Backend, generate};
use baracuda_kernel_vocab::{
    ArchSku, ElementKind, MAX_OPERANDS, OpCategory, OperandDesc, structure_key,
};

/// A JIT synthesis request from Fuel (the strategist).
#[derive(Clone, Debug)]
pub struct JitRequest {
    /// The primitive subgraph to fuse — a graph-`Op` tree rooted at the sink,
    /// with `bind` leaves for the region's inputs (the §4.1 vocabulary; the same
    /// node shape [`derive_pattern`] produces). Per-node `consumers`/`extract`
    /// are ignored — [`region_to_op`] regenerates them (see its docs).
    pub region: PatternNode,
    /// Region input count; `bind` indices must be exactly `[0, n_inputs)`, and
    /// [`Self::operands`] must hold exactly `n_inputs + 1` entries.
    pub n_inputs: u8,
    /// Op taxonomy for the structure key (drives schedule legality). Fuel's to
    /// choose (strategist); the synthesizer does not second-guess it.
    pub op_category: OpCategory,
    /// Operand descriptors (inputs then output) — Fuel's `FdxOperandDesc`
    /// projection, the input to [`structure_key`]. Increment 1 requires a single
    /// shared dtype across all operands.
    pub operands: Vec<OperandDesc>,
    /// Target compute capability — keys the schedule. The finer device identity
    /// (ordinal / exact SM / driver) that §5.2's `target.device` carries is
    /// folded into `arch` here; the real on-demand compiler (increment 2) will
    /// refine it where the artifact must be SM-specific.
    pub arch: ArchSku,
    /// Stable identity to register the synthesized fused op under.
    pub fused_op_id: String,
    /// Compile/resource budget (Fuel sets it). Threaded into [`Compiler::compile`].
    pub budget: JitBudget,
}

/// Compile-time / resource budget for a synthesis request.
#[derive(Copy, Clone, Debug)]
pub struct JitBudget {
    /// Ceiling on on-demand compilation time. Must be `> 0`.
    pub max_compile_ms: u32,
}

/// The synthesizer's response — `(kernel + contract + recipe + link)`, the §5 shape.
#[derive(Clone, Debug)]
pub struct JitResponse {
    /// The synthesized kernel: entry-point symbol, source, compiled artifact, and
    /// the artifact's provenance.
    pub kernel: SynthKernel,
    /// The full FKC contract for the kernel (the per-kernel block).
    pub contract: String,
    /// The declarative recipe — `pattern:` (recognize the region) + `decompose:`
    /// (expand back to it). Both halves, per the rev-4 recipe principle.
    pub recipe: Recipe,
    /// The `link_registry` row that resolves the kernel's `entry_point` to a
    /// `KernelRef` at load (FKC §12.6) — without it an adopted kernel is unbindable.
    pub link: LinkEntry,
}

/// A synthesized kernel.
#[derive(Clone, Debug)]
pub struct SynthKernel {
    /// Linkable entry-point symbol (matches the contract's `entry_point`).
    pub entry_point: String,
    /// The generated backend source (`.cu`).
    pub source: String,
    /// The compiled artifact (PTX / cubin / stub) from the on-demand [`Compiler`].
    pub artifact: Vec<u8>,
    /// What kind of artifact this is — a loader **must** refuse [`ArtifactKind::Stub`].
    pub kind: ArtifactKind,
}

/// Provenance of a [`SynthKernel::artifact`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArtifactKind {
    /// Compiled PTX (driver-JIT-linked; portable across SMs of the arch).
    Ptx,
    /// Compiled cubin (SM-specific machine code).
    Cubin,
    /// A stand-in artifact ([`StubCompiler`]) — **not loadable**; a device-module
    /// loader must refuse it rather than feed it to the driver.
    Stub,
}

/// The two-directional recipe (rev-4 §1): both mandatory for a fused op.
#[derive(Clone, Debug)]
pub struct Recipe {
    /// The `pattern:` block — recognize the primitive subgraph.
    pub pattern: String,
    /// The `decompose:` block — expand the fused op back to that subgraph. For a
    /// JIT'd op this is, by construction, the region itself. Derived from the same
    /// canonical pattern node as [`Recipe::pattern`], so the two halves are
    /// structurally identical and the scalar `extract:` routing is preserved. The
    /// declarative decompose *format* is §9-deferred by Fuel (provisional header).
    pub decompose: String,
}

/// Why a region can't be synthesized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JitError {
    /// A region op name outside the increment-1 IR vocabulary.
    UnsupportedOp(String),
    /// Wrong tensor-operand arity for a region op.
    Arity {
        /// The op.
        op: String,
        /// Expected operand count.
        expected: usize,
        /// Actual operand count.
        got: usize,
    },
    /// `operands.len()` isn't `n_inputs + 1`, or exceeds [`MAX_OPERANDS`] — the
    /// kernel signature and the `accept` predicate would describe different arities.
    OperandArity {
        /// Declared region input count.
        n_inputs: u8,
        /// Operand-projection length supplied.
        operands: usize,
    },
    /// Region operands don't all share one dtype (increment-1 is uniform-dtype) —
    /// rejected as an honest miss rather than mistyped.
    MixedDtype,
    /// The budget is meaningless (e.g. `max_compile_ms == 0`).
    Budget(String),
    /// The region's bind set isn't `[0, n_inputs)` (rejected by [`derive_pattern`]).
    Pattern(PatternError),
    /// The target dtype has no FKC §5 base-dtype spelling — no contract.
    UnsupportedDtype,
    /// On-demand compilation failed.
    Compile(String),
}

impl From<PatternError> for JitError {
    fn from(e: PatternError) -> Self {
        JitError::Pattern(e)
    }
}

/// A KISS typed decline code (KISS-Announce §6.4 / KISS-Synth decline table) —
/// the machine-actionable `u32` a provider returns on the provision /
/// contract-query path, in place of a free-text error string.
///
/// The current KISS code set is coarse: every [`JitError`] is a "cell understood
/// but not buildable by this provider" and maps to [`DeclineCode::CannotProvision`].
/// The finer per-reason distinctions (unknown-op vs attrs-channel-gap vs
/// operand-keying-gap vs determinism-gap) await the KISS typed-decline taxonomy
/// proposed in ThinkersJournal/KISS#17.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DeclineCode {
    /// No such cell, and none can be provisioned.
    UnknownStructureKey = 0x1,
    /// Cell understood, but this provider cannot build it (unsupported op/dtype,
    /// inexpressible operand tuple, or a failed on-demand compile).
    CannotProvision = 0x2,
    /// The provider does not advertise the contract-query capability.
    QueryNotSupported = 0x4,
    /// `structure_key` known but the requested `revision_hash` is not held.
    UnknownRevision = 0x6,
}

impl JitError {
    /// Map this error to its KISS typed decline code for the seam. The free-text
    /// rendering (`Debug`) stays available as an optional human detail; the code
    /// is the machine-actionable currency a non-Rust consumer can act on.
    #[must_use]
    pub fn decline_code(&self) -> DeclineCode {
        // Every synthesis failure means "understood the request, cannot build it"
        // — CANNOT_PROVISION under the current KISS code set (see the type docs).
        DeclineCode::CannotProvision
    }
}

/// The on-demand compilation seam: source → artifact (PTX/cubin). The production
/// impl drives nvrtc; tests use [`StubCompiler`]. Kept a trait so synthesis is
/// testable without a CUDA toolchain and so a pre-built-variant cache can slot in.
pub trait Compiler {
    /// Compile `source` (exposing `entry`) to a device artifact, within
    /// `max_compile_ms` (the request budget; an impl may cap optimization or
    /// abort on overrun).
    ///
    /// # Errors
    /// Returns the compiler diagnostic string on failure.
    fn compile(&self, source: &str, entry: &str, max_compile_ms: u32) -> Result<Vec<u8>, String>;

    /// Provenance of the artifacts this compiler emits. Defaults to
    /// [`ArtifactKind::Ptx`]; [`StubCompiler`] overrides to [`ArtifactKind::Stub`].
    fn artifact_kind(&self) -> ArtifactKind {
        ArtifactKind::Ptx
    }
}

/// A no-toolchain stand-in compiler for tests / dry-runs. Its artifact is tagged
/// [`ArtifactKind::Stub`] so it can never be mistaken for loadable code.
#[derive(Copy, Clone, Debug, Default)]
pub struct StubCompiler;

impl Compiler for StubCompiler {
    fn compile(&self, source: &str, entry: &str, _max_compile_ms: u32) -> Result<Vec<u8>, String> {
        Ok(format!("// stub-ptx: {entry} from {}B source", source.len()).into_bytes())
    }
    fn artifact_kind(&self) -> ArtifactKind {
        ArtifactKind::Stub
    }
}

/// Synthesize a [`JitResponse`] for a Fuel-chosen region. The synthesizer core:
/// region → op IR → specialized kernel → on-demand compile → FKC contract +
/// recipe + link row. The optimizer that §5.1 permits (an inward e-graph) would
/// sit between `region_to_op` and `generate`; increment 1 lowers directly.
///
/// `backend` selects the lowering target (§5.2 `target.backend`); `compiler` is
/// the matching on-demand toolchain. Both are injected so the engine is
/// backend-agnostic and testable.
///
/// # Errors
/// See [`JitError`] — a malformed request (arity / mixed dtype / zero budget), an
/// unsupported op/dtype, or a compile failure.
pub fn synthesize(
    req: &JitRequest,
    backend: &dyn Backend,
    compiler: &dyn Compiler,
) -> Result<JitResponse, JitError> {
    // --- Trust-boundary validation of the Fuel-supplied request -------------
    let expected = usize::from(req.n_inputs) + 1;
    if req.operands.len() != expected || req.operands.len() > MAX_OPERANDS {
        return Err(JitError::OperandArity {
            n_inputs: req.n_inputs,
            operands: req.operands.len(),
        });
    }
    if req.budget.max_compile_ms == 0 {
        return Err(JitError::Budget("max_compile_ms must be > 0".to_string()));
    }
    let dtype = req.operands[0].dtype; // operands non-empty: len == n_inputs + 1 >= 1
    if req.operands.iter().any(|o| o.dtype != dtype) {
        // Increment 1 is uniform-dtype (StructureKey carries one dtype slot); a
        // mixed region would be mistyped + misdescribed, so miss honestly.
        return Err(JitError::MixedDtype);
    }

    // --- Synthesis ----------------------------------------------------------
    let (op, derived) = region_to_op(&req.region, req.n_inputs, &req.fused_op_id, dtype)?;
    synthesize_op(
        op,
        derived,
        &req.operands,
        req.op_category,
        req.arch,
        req.budget.max_compile_ms,
        backend,
        compiler,
    )
}

/// Core synthesis shared by [`synthesize`] (our [`PatternNode`] region) and the
/// `seam` front-end (Fuel's `fuel_kernel_seam_types::PatternNode` region): op IR +
/// its canonical recipe pattern → optimized kernel → on-demand compile → FKC
/// contract + recipe + link row. The §5.1 inward optimizer runs on the *kernel*
/// body; `derived` (the original region) carries the recipe.
fn synthesize_op(
    op: OpDef,
    derived: PatternNode,
    operands: &[OperandDesc],
    op_category: OpCategory,
    arch: ArchSku,
    max_compile_ms: u32,
    backend: &dyn Backend,
    compiler: &dyn Compiler,
) -> Result<JitResponse, JitError> {
    let dtype = operands.first().map_or(ElementKind::F32, |o| o.dtype);
    // Trust boundary, gate 1: the backend must be able to spell this dtype as a
    // scalar type at all. `dtype_compatible` (gate 2) only checks unary/binary-fn
    // float-ness and f32-only params, so it lets a PURE-INFIX body (Add/Sub/Mul/Div
    // over binds, no fn/param) through for ANY dtype — a Bool/S8/Complex Add region
    // would then panic in `scalar_ctype` during `generate`. Decline it here instead
    // (the Synthesizer trait must never unwind across the boundary).
    if !backend.supports_dtype(dtype) {
        return Err(JitError::UnsupportedDtype);
    }
    // CUDA backend dtype limits: a unary / binary-fn node needs a float dtype, and
    // scalar params are f32-only — honest miss rather than a lowering panic.
    if !dtype_compatible(&op.body, dtype) {
        return Err(JitError::UnsupportedDtype);
    }

    // The schedule cell is keyed from Fuel's operand projection — never re-derived.
    let key = structure_key(op_category, operands, arch);
    let kernel_op = OpDef {
        body: optimize(&op.body),
        ..op.clone()
    };
    let kernel = generate(&kernel_op, &key, backend);

    let artifact = compiler
        .compile(&kernel.source, &kernel.name, max_compile_ms)
        .map_err(JitError::Compile)?;
    let contract = contract(&op, &key, &kernel, backend).ok_or(JitError::UnsupportedDtype)?;

    // Both recipe halves come from the SINGLE canonical pattern node, so they are
    // structurally identical and decompose carries the scalar `extract:` routing.
    let pattern = to_fkc(&derived);
    let decompose = to_fkc(&derived).replacen("pattern:", "decompose:", 1);

    let link = link_entry(&op, &key, &kernel);
    let kind = compiler.artifact_kind();

    Ok(JitResponse {
        kernel: SynthKernel {
            entry_point: kernel.name,
            source: kernel.source,
            artifact,
            kind,
        },
        contract,
        recipe: Recipe { pattern, decompose },
        link,
    })
}

/// Translate a region (graph-`Op` subgraph) into the op IR and its canonical
/// pattern — the inverse of [`crate::pattern::derive_pattern`]'s walk. `bind: i`
/// → `Input(i)`; a scalar-param op (`AddScalar`/`MulScalar`) → the arithmetic op
/// with a runtime `Param` (the scalar is a launch arg, exactly as the AOT path
/// treats it). The returned [`PatternNode`] is [`derive_pattern`]'s canonical
/// form (which also performs the bind-set / elementwise validation).
///
/// The region's per-node `consumers`/`extract` fields are ignored and regenerated
/// by `derive_pattern` under the sole-consumer rule — sound because the IR is a
/// pure tree (no shared interiors), so the only fusable shape (sole-consumer
/// interiors) is the only representable one.
fn region_to_op(
    region: &PatternNode,
    n_inputs: u8,
    name: &str,
    dtype: ElementKind,
) -> Result<(OpDef, PatternNode), JitError> {
    let mut next_param = 0u8;
    let body = node_to_expr(region, &mut next_param)?;
    // A region ROOTED at a comparison produces a U8 mask (§4.1 "Comparison
    // (→ U8 mask)"), i.e. an output dtype differing from the inputs' — which
    // the increment-1 uniform-dtype keying cannot express (an honest Fuel
    // projection of such a region also carries a U8 output OperandDesc and is
    // already declined as MixedDtype upstream; this gate closes the
    // uniform-dtype projection too, so a synthesized "cmp" can never bind as a
    // key-dtype-store kernel where Fuel expects a 1-byte mask). Typed decline,
    // never a panic; NESTED comparisons in a float region synthesize normally.
    if matches!(&body, ScalarExpr::Binary(b, _, _) if b.is_cmp()) {
        return Err(JitError::UnsupportedOp(
            "comparison at region root (U8-mask output; uniform-dtype JIT keying \
             cannot express it)"
                .to_string(),
        ));
    }
    // INTERIOR comparisons decline too (review-tightened): the response
    // contract's pattern block would encode the Cmp* as a direct operand of a
    // float op — an edge no constructible Fuel graph has (Fuel's compare
    // builders pin U8 output and its binary ops assert dtype equality, so
    // real mask-multiply graphs interpose Cast(U8→float), which is outside
    // the §4.1 pattern grammar and the see-through set). Fuel's matcher can
    // therefore never produce this region from a real graph either — the
    // decline costs zero live coverage and avoids advertising an unmatchable
    // pattern. Revisit when Cast joins the pattern vocabulary.
    //
    // WHERE/SELECT carve-out: a cmp IS permitted iff it is the COND CHILD of
    // a Select (only that position — see `expr_contains_cmp_outside_select_cond`).
    // The decline above exists because real Fuel graphs interpose
    // Cast(U8→float) before a FLOAT op consumes a mask — but `Where` consumes
    // the mask DIRECTLY (its cond edge is the compare's U8 output, no Cast),
    // so a region like `[Gt, Where]` with all-float binds is a constructible
    // Fuel shape: the interior U8 edge projects to Baracuda's interior
    // 1.0/0.0 cond tested `!= 0` — value-equivalent, arms bit-untouched.
    // Without the carve-out every Where region would keep declining for the
    // WRONG (Cast) reason; with it, the region reaches `derive_pattern`,
    // whose v1 `SelectUnsupported` typed miss names the REAL blocker (the
    // withheld Where advert — see `PatternError::SelectUnsupported`).
    if expr_contains_cmp_outside_select_cond(&body) {
        return Err(JitError::UnsupportedOp(
            "interior comparison under float ops (the fused pattern needs Cast \
             in the §4.1 vocabulary — awaiting the mixed-dtype/Cast follow-up)"
                .to_string(),
        ));
    }
    let op = OpDef {
        name: name.to_string(),
        n_inputs,
        body,
        dtypes: vec![dtype],
        access: Access::Elementwise,
        // Seam regions carry no layout facts (Fuel `OpAttrs` lacks perm/shape —
        // ask F1), so a synthesized op is always view-free until F1 lands.
        views: Vec::new(),
        // Uniform-dtype (increment 1): a JIT'd op never stores hetero output —
        // the root-cmp gate above is what keeps this unconditional.
        out_dtype: None,
        // Single-output: a seam region is a single-rooted PatternNode
        // (fuel-kernel-seam-types), so `region_to_op` builds exactly one output
        // body. Multi-output (increment 1) is AOT-only until Fuel's seam grows a
        // multi-output region envelope — see the module notes / synthesize.
        extra_out_bodies: Vec::new(),
        // Uniform per-output dtype (the hetero multi-output / dropout-class
        // increment): a synthesized op is single-output (above), so it never
        // carries per-output hetero dtypes. Empty ⇒ no hetero store.
        extra_out_dtypes: Vec::new(),
        // Index-free: a gather (increment 4) is AOT-only — its index-operand dtype
        // is unkeyable in the single-dtype token (see `PatternError::GatherUnsupported`),
        // so a seam region never synthesizes one.
        read_index: Vec::new(),
        // Write-Direct: a scatter (increment 5) is AOT-only for the same
        // unkeyable-index-dtype reason (see `PatternError::ScatterUnsupported`) plus
        // the determinism flip — a seam region never synthesizes one.
        write_index: crate::ir::WriteIndex::Direct,
        // Offset-free: a BASE_OFFSET SLICE (runtime pointer bump) is AOT-only — the
        // frozen JIT envelope has no slot to transport the runtime `off` scalar at
        // dispatch (see `PatternError::OffsetUnsupported`), so a seam region never
        // synthesizes one.
        base_offsets: Vec::new(),
        out_base_offset: crate::ir::BaseOffset::Zero,
    };
    // Reuse the AOT bind-set / elementwise validation, and keep the canonical
    // pattern (so synthesize derives it exactly once).
    let pattern = derive_pattern(&op)?;
    Ok((op, pattern))
}

fn node_to_expr(n: &PatternNode, next_param: &mut u8) -> Result<ScalarExpr, JitError> {
    match n {
        PatternNode::Bind(i) => Ok(ScalarExpr::Input(*i)),
        PatternNode::Op { op, operands, .. } => synth_op(op, operands, next_param),
    }
}

/// [`crate::contract::expr_contains_cmp`] with the WHERE/SELECT carve-out: a
/// `Cmp*` node does NOT count when it is the direct **cond child of a
/// Select** — only that position (Fuel's `Where` consumes the compare's U8
/// mask edge directly, no interposed `Cast`, so `[Gt, Where]` is a
/// constructible Fuel region). Everywhere else — a cmp under a float op, a
/// cmp inside a select ARM, a cmp nested deeper inside a composed cond, or
/// the cond-root cmp's own operands — the interior-cmp decline keeps its full
/// reach (those edges still need `Cast` in the §4.1 vocabulary). JIT-only:
/// the contract-side `expr_contains_cmp` walk deliberately has NO carve-out
/// (select bodies are withheld wholesale there).
fn expr_contains_cmp_outside_select_cond(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => false,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b) => {
            expr_contains_cmp_outside_select_cond(a) || expr_contains_cmp_outside_select_cond(b)
        }
        ScalarExpr::Unary(_, a) => expr_contains_cmp_outside_select_cond(a),
        ScalarExpr::Binary(bop, a, b) => {
            bop.is_cmp()
                || expr_contains_cmp_outside_select_cond(a)
                || expr_contains_cmp_outside_select_cond(b)
        }
        ScalarExpr::Select(c, a, b) => {
            let cond_interior = match &**c {
                // The permitted position: a cmp ROOTING the cond child. Its own
                // operands must still be cmp-free.
                ScalarExpr::Binary(bop, x, y) if bop.is_cmp() => {
                    expr_contains_cmp_outside_select_cond(x)
                        || expr_contains_cmp_outside_select_cond(y)
                }
                other => expr_contains_cmp_outside_select_cond(other),
            };
            cond_interior
                || expr_contains_cmp_outside_select_cond(a)
                || expr_contains_cmp_outside_select_cond(b)
        }
    }
}

fn synth_op(op: &str, operands: &[PatternNode], np: &mut u8) -> Result<ScalarExpr, JitError> {
    // Scalar-param ops: one tensor operand; the scalar becomes a runtime Param
    // (the AOT emitter's `extract:` pulls it back out — round-trip stable).
    if op == "AddScalar" || op == "MulScalar" {
        let t = unary_operand(op, operands, np)?;
        let p = ScalarExpr::Param(*np);
        *np += 1;
        return Ok(if op == "AddScalar" {
            ScalarExpr::Add(Box::new(t), Box::new(p))
        } else {
            ScalarExpr::Mul(Box::new(t), Box::new(p))
        });
    }
    // `Where` (OpTag::Where, dispatch spelling bare "Where"): the ternary
    // select, operand order (cond, a, b) — the first arity-3 region op.
    if op == "Where" {
        // BOUND-COND decline (typed, before recursion): Fuel's `Where` cond is
        // strictly a U8 TENSOR, so a cond that is a bare region input (`bind`)
        // means the honest operand tuple is `[U8, T, T, T]` — inexpressible
        // under uniform-dtype keying (the honest Fuel projection also carries
        // a U8 cond OperandDesc and already declines MixedDtype upstream; this
        // gate closes the uniform all-T projection too, which would otherwise
        // synthesize a key-dtype `!= 0` kernel misdescribing Fuel's U8-cond
        // op). A cond that is an INTERIOR cmp is the carve-out shape and
        // recurses normally. Mirrors the root-cmp decline's rationale.
        if matches!(operands.first(), Some(PatternNode::Bind(_))) {
            return Err(JitError::UnsupportedOp(
                "bound cond operand on Where (a U8 cond tensor — [U8,T,T] operand \
                 tuple; uniform-dtype JIT keying cannot express it)"
                    .to_string(),
            ));
        }
        let (c, a, b) = ternary_operands(op, operands, np)?;
        return Ok(ScalarExpr::Select(Box::new(c), Box::new(a), Box::new(b)));
    }
    if let Some(u) = region_unary(op) {
        let x = unary_operand(op, operands, np)?;
        return Ok(ScalarExpr::Unary(u, Box::new(x)));
    }
    // Non-infix binary fns (Maximum/Minimum/Pow/Rem) — two tensor operands.
    if let Some(bop) = region_binary(op) {
        let (a, b) = binary_operands(op, operands, np)?;
        return Ok(ScalarExpr::Binary(bop, Box::new(a), Box::new(b)));
    }
    // Infix binary tensor ops.
    let ctor: fn(Box<ScalarExpr>, Box<ScalarExpr>) -> ScalarExpr = match op {
        "Add" => ScalarExpr::Add,
        "Sub" => ScalarExpr::Sub,
        "Mul" => ScalarExpr::Mul,
        "Div" => ScalarExpr::Div,
        _ => return Err(JitError::UnsupportedOp(op.to_string())),
    };
    let (a, b) = binary_operands(op, operands, np)?;
    Ok(ctor(Box::new(a), Box::new(b)))
}

/// Resolve the exactly-three operands of a ternary op (`Where`), recursing
/// into each — beside [`binary_operands`], same arity discipline.
fn ternary_operands(
    op: &str,
    operands: &[PatternNode],
    np: &mut u8,
) -> Result<(ScalarExpr, ScalarExpr, ScalarExpr), JitError> {
    if operands.len() != 3 {
        return Err(JitError::Arity {
            op: op.to_string(),
            expected: 3,
            got: operands.len(),
        });
    }
    let c = node_to_expr(&operands[0], np)?;
    let a = node_to_expr(&operands[1], np)?;
    let b = node_to_expr(&operands[2], np)?;
    Ok((c, a, b))
}

/// Resolve the exactly-two operands of a binary op, recursing into each.
fn binary_operands(
    op: &str,
    operands: &[PatternNode],
    np: &mut u8,
) -> Result<(ScalarExpr, ScalarExpr), JitError> {
    if operands.len() != 2 {
        return Err(JitError::Arity {
            op: op.to_string(),
            expected: 2,
            got: operands.len(),
        });
    }
    let a = node_to_expr(&operands[0], np)?;
    let b = node_to_expr(&operands[1], np)?;
    Ok((a, b))
}

/// Whether the CUDA backend can lower `body` at `dtype` — the JIT's op×dtype
/// legality gate (gate 2 behind `supports_dtype`; the same table the AOT plan
/// gate `assert_int_op_admissibility` enforces — see the `ir::BinaryOp` docs):
///
/// - unary / float-binary-fn nodes require a float dtype (no integer math);
///   `Nextafter` additionally excludes f16/bf16 (the half path computes
///   promoted-to-f32, which would step the f32 lattice — the wrong neighbor;
///   see `cuda_binary`);
/// - the increment-0c INT-ONLY ops (bitwise/shift/logical) require an int
///   dtype (`I32`/`I64`/`S8`/`U8`), the logical ops exactly `U8` — today
///   defensive-only, since no `OpTag` names them (no region can request one),
///   but the gate stands ahead of the vocabulary like the Nextafter arm does.
///   NOTE: the AOT plan gate additionally pins int-op operands at `S8`/`U8`
///   to leaf `Input`s (rule 3 — composed operands diverge under DAG sharing);
///   moot here while no OpTag names these ops, but a future vocabulary
///   extension must mirror that rule HERE before regions can compose them,
///   or `build_plan` would panic across the JIT trust boundary;
/// - a runtime scalar `Param` is f32-only; a `Const` is f64-spelled, so it
///   rejects at int dtypes (double math in an int kernel — see the plan gate);
/// - infix `Add`/`Sub`/`Mul` work at any supported dtype (int = wrapping);
///   infix `Div` is FLOAT-ONLY — the bespoke surface has no int elementwise
///   div and C `/0` is device-UB, so a uniform-int Div region declines even
///   though the dtype itself is supported (the audited replacement for the 0b
///   supports_dtype(U8) hold).
fn dtype_compatible(body: &ScalarExpr, dtype: ElementKind) -> bool {
    let is_float = matches!(
        dtype,
        ElementKind::F16
            | ElementKind::Bf16
            | ElementKind::F32
            | ElementKind::F32Strict
            | ElementKind::F64
    );
    let is_int = crate::plan::is_int_dtype(dtype);
    let f32_only = matches!(dtype, ElementKind::F32 | ElementKind::F32Strict);
    let is_half = matches!(dtype, ElementKind::F16 | ElementKind::Bf16);
    struct Ctx {
        is_float: bool,
        is_int: bool,
        f32_only: bool,
        is_half: bool,
        dtype: ElementKind,
    }
    fn walk(e: &ScalarExpr, c: &Ctx) -> bool {
        match e {
            // Reduced only appears in a RowReduce epilogue, which never reaches the
            // JIT path (region_to_op builds Elementwise only) — treat as a benign
            // float scalar leaf for exhaustiveness.
            ScalarExpr::Input(_) | ScalarExpr::Reduced(_) => true,
            // Coord (increment 0d) mirrors the AOT plan gate: f32/f64 only
            // (halves round past 2048; ints would take the float-cast
            // coordinate). Defensive today — `region_to_op` never constructs
            // a Coord (OpTag::Iota is declined typed at `optag_name`, see the
            // seam module) — but the gate stands ahead of the vocabulary like
            // the Nextafter arm does, so a future Iota bridge cannot panic
            // `build_plan` across the trust boundary.
            ScalarExpr::Coord(_) => matches!(
                c.dtype,
                ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64
            ),
            ScalarExpr::Const(_) => !c.is_int,
            ScalarExpr::Param(_) => c.f32_only,
            ScalarExpr::Unary(_, x) => c.is_float && walk(x, c),
            ScalarExpr::Binary(op, a, b) => {
                let op_ok = if op.is_int_only() {
                    c.is_int && (!op.is_logical() || c.dtype == ElementKind::U8)
                } else {
                    c.is_float && !(c.is_half && matches!(op, BinaryOp::Nextafter))
                };
                op_ok && walk(a, c) && walk(b, c)
            }
            ScalarExpr::Div(a, b) => !c.is_int && walk(a, c) && walk(b, c),
            ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) | ScalarExpr::Mul(a, b) => {
                walk(a, c) && walk(b, c)
            }
            // Select is float-only in v1 (f32/f32s/f64/f16/bf16) — mirrors the
            // AOT plan gate (`assert_int_op_admissibility`'s Select arm): an
            // int select would raise the 0c cond-observer question, so it
            // declines typed here rather than panicking `build_plan` across
            // the JIT trust boundary.
            ScalarExpr::Select(cond, a, b) => {
                c.is_float && walk(cond, c) && walk(a, c) && walk(b, c)
            }
        }
    }
    walk(
        body,
        &Ctx {
            is_float,
            is_int,
            f32_only,
            is_half,
            dtype,
        },
    )
}

/// Inverse of [`crate::pattern`]'s `binary_name`. The increment-0a binaries
/// (`Atan2`/`Copysign`/`Nextafter`/`FmaxIeee`/`FminIeee`/`RemTrunc`) have NO
/// name here on purpose — §4.1/`OpTag` doesn't name them yet, so no region can
/// request them (an honest `UnsupportedOp`, never invented vocabulary). The
/// comparisons ARE §4.1/`OpTag` vocabulary (`Equal`…`Ge`) and map — legal
/// **nested** in a float region (an inline 0.0/1.0 mask, e.g. relu-backward's
/// `Mul(dy, Gt(x, zeros))`); a region whose ROOT is a comparison is declined
/// typed in [`region_to_op`] (its output is a U8 mask the uniform-dtype
/// increment-1 keying cannot express).
#[doc(hidden)]
pub fn region_binary(op: &str) -> Option<BinaryOp> {
    Some(match op {
        "Maximum" => BinaryOp::Max,
        "Minimum" => BinaryOp::Min,
        "Pow" => BinaryOp::Pow,
        "Rem" => BinaryOp::Rem,
        "Equal" => BinaryOp::CmpEq,
        "Ne" => BinaryOp::CmpNe,
        "Lt" => BinaryOp::CmpLt,
        "Le" => BinaryOp::CmpLe,
        "Gt" => BinaryOp::CmpGt,
        "Ge" => BinaryOp::CmpGe,
        _ => return None,
    })
}

fn unary_operand(op: &str, operands: &[PatternNode], np: &mut u8) -> Result<ScalarExpr, JitError> {
    if operands.len() != 1 {
        return Err(JitError::Arity {
            op: op.to_string(),
            expected: 1,
            got: operands.len(),
        });
    }
    node_to_expr(&operands[0], np)
}

/// Inverse of [`crate::pattern`]'s `unary_name`. `GeluErf` → [`UnaryOp::Gelu`]
/// (our exact-erf flavor); bare `Gelu` (tanh approx) is unsupported. The
/// increment-0a unaries (`Erfc`…`Lgamma`) have NO name here on purpose —
/// §4.1/`OpTag` doesn't name them yet (see [`region_binary`]).
#[doc(hidden)]
pub fn region_unary(op: &str) -> Option<UnaryOp> {
    Some(match op {
        "Neg" => UnaryOp::Neg,
        "Abs" => UnaryOp::Abs,
        "Sqr" => UnaryOp::Sqr,
        "Sqrt" => UnaryOp::Sqrt,
        "Rsqrt" => UnaryOp::Rsqrt,
        "Recip" => UnaryOp::Recip,
        "Exp" => UnaryOp::Exp,
        "Log" => UnaryOp::Log,
        "Tanh" => UnaryOp::Tanh,
        "Sigmoid" => UnaryOp::Sigmoid,
        "Relu" => UnaryOp::Relu,
        "Erf" => UnaryOp::Erf,
        "GeluErf" => UnaryOp::Gelu,
        "Silu" => UnaryOp::Silu,
        "Sin" => UnaryOp::Sin,
        "Cos" => UnaryOp::Cos,
        "Floor" => UnaryOp::Floor,
        "Ceil" => UnaryOp::Ceil,
        "Round" => UnaryOp::Round,
        "Sign" => UnaryOp::Sign,
        "Step" => UnaryOp::Step,
        _ => return None,
    })
}

/// The direct-Rust §5 seam (`--features seam`): synthesize for a region in Fuel's
/// frozen grammar (`fuel_kernel_seam_types`). Fuel owns the region grammar
/// (`PatternNode`/`OpTag`); Baracuda owns the classifier input (`OperandDesc`).
/// We convert Fuel's node to our internal node form and reuse the exact native
/// `region_to_op` + core synthesis — no duplicated op logic.
#[cfg(feature = "seam")]
pub mod seam {
    use super::*;
    use fuel_kernel_seam_types::{OpTag, PatternNode as SeamNode};

    /// Synthesize a kernel for a Fuel-chosen `region`. `operands` is the
    /// inputs-then-output `OperandDesc` projection; `n_inputs = operands.len() - 1`.
    ///
    /// # Errors
    /// See [`JitError`] — a malformed request, an op/dtype outside the
    /// synthesizer's coverage (honest miss), or a compile failure.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize(
        region: &SeamNode,
        operands: &[OperandDesc],
        op_category: OpCategory,
        arch: ArchSku,
        fused_op_id: &str,
        max_compile_ms: u32,
        backend: &dyn Backend,
        compiler: &dyn Compiler,
    ) -> Result<JitResponse, JitError> {
        if operands.is_empty() || operands.len() > MAX_OPERANDS {
            return Err(JitError::OperandArity {
                n_inputs: 0,
                operands: operands.len(),
            });
        }
        if max_compile_ms == 0 {
            return Err(JitError::Budget("max_compile_ms must be > 0".to_string()));
        }
        let dtype = operands[0].dtype;
        if operands.iter().any(|o| o.dtype != dtype) {
            return Err(JitError::MixedDtype);
        }
        let n_inputs = (operands.len() - 1) as u8;

        let internal = to_internal(region)?;
        let (op, derived) = region_to_op(&internal, n_inputs, fused_op_id, dtype)?;
        synthesize_op(
            op,
            derived,
            operands,
            op_category,
            arch,
            max_compile_ms,
            backend,
            compiler,
        )
    }

    /// Max region nesting the seam will convert — a trust-boundary guard so a
    /// pathologically deep region from Fuel can't overflow the stack (an
    /// uncatchable abort, not a catchable panic) during the recursive conversion.
    /// Elementwise fusion regions are shallow; 64 is far above any real subgraph.
    const MAX_REGION_DEPTH: u32 = 64;

    /// Convert a Fuel `PatternNode` (region direction) to Baracuda's internal node
    /// (op vocabulary mapped by name). An `OpTag` the synthesizer doesn't cover and
    /// the matcher-only `SeeThrough`/`Any` are honest `UnsupportedOp` misses; a
    /// region nested past [`MAX_REGION_DEPTH`] is declined before it can overflow.
    fn to_internal(n: &SeamNode) -> Result<PatternNode, JitError> {
        to_internal_at(n, 0)
    }

    fn to_internal_at(n: &SeamNode, depth: u32) -> Result<PatternNode, JitError> {
        if depth > MAX_REGION_DEPTH {
            return Err(JitError::UnsupportedOp(
                "region nested past MAX_REGION_DEPTH".to_string(),
            ));
        }
        match n {
            SeamNode::Bind { index } => Ok(PatternNode::Bind(*index)),
            SeamNode::Op { op, operands, .. } => {
                let name =
                    optag_name(*op).ok_or_else(|| JitError::UnsupportedOp(format!("{op:?}")))?;
                let ops = operands
                    .iter()
                    .map(|o| to_internal_at(o, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(PatternNode::Op {
                    op: name.to_string(),
                    operands: ops,
                    consumers: None,
                    extract: Vec::new(),
                })
            }
            SeamNode::SeeThrough { .. } => Err(JitError::UnsupportedOp("SeeThrough".to_string())),
            SeamNode::Any => Err(JitError::UnsupportedOp("Any".to_string())),
        }
    }

    /// `OpTag` → Baracuda's emitter op-name (what `region_to_op` parses). `None`
    /// for any tag outside the increment-1 synthesizer coverage.
    fn optag_name(op: OpTag) -> Option<&'static str> {
        Some(match op {
            OpTag::Add => "Add",
            OpTag::Sub => "Sub",
            OpTag::Mul => "Mul",
            OpTag::Div => "Div",
            OpTag::Maximum => "Maximum",
            OpTag::Minimum => "Minimum",
            OpTag::Pow => "Pow",
            OpTag::Rem => "Rem",
            OpTag::Neg => "Neg",
            OpTag::Abs => "Abs",
            OpTag::Sqr => "Sqr",
            OpTag::Sqrt => "Sqrt",
            OpTag::Rsqrt => "Rsqrt",
            OpTag::Recip => "Recip",
            OpTag::Exp => "Exp",
            OpTag::Log => "Log",
            OpTag::Sin => "Sin",
            OpTag::Cos => "Cos",
            OpTag::Tanh => "Tanh",
            OpTag::Sigmoid => "Sigmoid",
            OpTag::Silu => "Silu",
            OpTag::GeluErf => "GeluErf",
            OpTag::Relu => "Relu",
            OpTag::Erf => "Erf",
            OpTag::Step => "Step",
            OpTag::Floor => "Floor",
            OpTag::Ceil => "Ceil",
            OpTag::Round => "Round",
            OpTag::Sign => "Sign",
            OpTag::AddScalar => "AddScalar",
            OpTag::MulScalar => "MulScalar",
            // Comparisons (→ U8 mask): mapped so a comparison NESTED in a float
            // region synthesizes (inline 0.0/1.0 mask — the relu-backward
            // `Mul(dy, Gt(x, z))` shape); a region ROOTED at one is declined
            // typed by `region_to_op` (hetero U8 output — see its docs).
            OpTag::Equal => "Equal",
            OpTag::Ne => "Ne",
            OpTag::Lt => "Lt",
            OpTag::Le => "Le",
            OpTag::Gt => "Gt",
            OpTag::Ge => "Ge",
            // Where (select/mask; dispatch spelling is bare "Where", NOT
            // Elementwise-suffixed): maps to the ternary Select — operand
            // order (cond, a, b) matches Fuel's. A cmp-cond region ([Gt,
            // Where]) passes the interior-cmp carve-out and reaches
            // `derive_pattern`, whose v1 SelectUnsupported typed miss is the
            // decline (the Where advert is withheld — see pattern.rs); a
            // bound-cond region declines typed in `synth_op` under BOTH
            // projections (U8 cond → MixedDtype upstream; uniform all-T →
            // the bound-cond gate).
            OpTag::Where => "Where",
            // Op::Gelu (tanh), PowI/Clamp, MaskedFill, reductions,
            // MatMul, shape/layout, indexing, LogSoftmaxLastDim — not
            // synthesized. OpTag::Iota (0.10.2 "value source") is ALSO
            // declined here even though the IR now has `ScalarExpr::Coord`
            // (increment 0d): a Fuel Iota is a graph node whose axis rides
            // `OpAttrs.axis`, and this converter drops attrs — mapping it
            // axis-less would synthesize the wrong coordinate. Typed decline
            // (UnsupportedOp("Iota")), never a panic — pinned by
            // `iota_region_declines_typed`; the attrs-aware Coord bridge is
            // the follow-up.
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op_node(op: &str, operands: Vec<PatternNode>) -> PatternNode {
        PatternNode::Op {
            op: op.to_string(),
            operands,
            consumers: None,
            extract: Vec::new(),
        }
    }

    #[test]
    fn jit_errors_map_to_cannot_provision_decline() {
        for e in [
            JitError::UnsupportedDtype,
            JitError::MixedDtype,
            JitError::UnsupportedOp("Foo".into()),
            JitError::Compile("nvrtc failed".into()),
            JitError::Budget("max_compile_ms must be > 0".into()),
            JitError::Arity {
                op: "add".into(),
                expected: 2,
                got: 3,
            },
            JitError::OperandArity {
                n_inputs: 1,
                operands: 5,
            },
        ] {
            assert_eq!(
                e.decline_code(),
                DeclineCode::CannotProvision,
                "every synthesis failure is a CANNOT_PROVISION decline: {e:?}"
            );
        }
    }

    #[test]
    fn decline_codes_match_kiss_wire_values() {
        // The u32 discriminants are the KISS-Announce §6.4 / KISS-Synth codes.
        assert_eq!(DeclineCode::UnknownStructureKey as u32, 0x1);
        assert_eq!(DeclineCode::CannotProvision as u32, 0x2);
        assert_eq!(DeclineCode::QueryNotSupported as u32, 0x4);
        assert_eq!(DeclineCode::UnknownRevision as u32, 0x6);
    }

    #[test]
    fn int_op_dtype_compatible_pins_both_directions() {
        use crate::ir::input;
        // The op×dtype table, pinned at the JIT gate (the composition story:
        // supports_dtype says "the dtype has a scalar C type", THIS says
        // "this body is legal at that dtype").
        let band = input(0).binary(BinaryOp::BitAnd, input(1)).0;
        for dt in [
            ElementKind::I32,
            ElementKind::I64,
            ElementKind::S8,
            ElementKind::U8,
        ] {
            assert!(dtype_compatible(&band, dt), "BitAnd legal at {dt:?}");
        }
        for dt in [
            ElementKind::F32,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::F64,
        ] {
            assert!(!dtype_compatible(&band, dt), "BitAnd illegal at {dt:?}");
        }
        let shl = input(0).binary(BinaryOp::Shl, input(1)).0;
        assert!(dtype_compatible(&shl, ElementKind::U8));
        assert!(!dtype_compatible(&shl, ElementKind::F32));
        // Logical ops: U8 (Bool) ONLY — the bespoke surface instantiates
        // exactly uint8_t, so wider ints reject too.
        let land = input(0).binary(BinaryOp::LogicalAnd, input(1)).0;
        assert!(dtype_compatible(&land, ElementKind::U8));
        for dt in [
            ElementKind::I32,
            ElementKind::I64,
            ElementKind::S8,
            ElementKind::F32,
        ] {
            assert!(!dtype_compatible(&land, dt), "LogicalAnd illegal at {dt:?}");
        }
        // Div: float-only (int div rejected); Add: legal at ints; Const: f64-
        // spelled, rejected at ints; cmp: float-only.
        let div = (input(0) / input(1)).0;
        assert!(dtype_compatible(&div, ElementKind::F32));
        for dt in [
            ElementKind::I32,
            ElementKind::I64,
            ElementKind::S8,
            ElementKind::U8,
        ] {
            assert!(!dtype_compatible(&div, dt), "Div illegal at {dt:?}");
        }
        let add = (input(0) + input(1)).0;
        assert!(dtype_compatible(&add, ElementKind::U8));
        assert!(dtype_compatible(&add, ElementKind::S8));
        let addk = (input(0) + crate::ir::konst(2.0)).0;
        assert!(dtype_compatible(&addk, ElementKind::F32));
        assert!(
            !dtype_compatible(&addk, ElementKind::I64),
            "f64 Const at int rejects"
        );
        let cmp = input(0).binary(BinaryOp::CmpLt, input(1)).0;
        assert!(
            !dtype_compatible(&cmp, ElementKind::I32),
            "int cmp rejects (bespoke is fp-only)"
        );
    }

    #[test]
    fn where_synth_preserves_cond_a_b_arm_order() {
        // The brief-mandated (cond, a, b) order at the synth site (jit.rs synth_op
        // Where arm). In v1 EVERY Where region declines downstream (bound-cond /
        // interior-cmp / Pattern(SelectUnsupported)), and all those decline reasons
        // are symmetric in the two arms — so an a<->b swap at the Select
        // construction is otherwise a surviving mutant that would silently flip
        // Fuel's `Where` (pick `b` where Fuel means `a`) the moment a follow-up
        // lifts the SelectUnsupported withhold. Observe the synthesized expr
        // DIRECTLY at `node_to_expr` (before any decline): cond = the interior
        // cmp, arm a = Bind(2) -> Input(2), arm b = Bind(3) -> Input(3).
        let region = op_node(
            "Where",
            vec![
                op_node("Gt", vec![PatternNode::Bind(0), PatternNode::Bind(1)]),
                PatternNode::Bind(2),
                PatternNode::Bind(3),
            ],
        );
        let mut np = 0u8;
        let expr = node_to_expr(&region, &mut np).expect("carve-out Where synthesizes a Select");
        match expr {
            ScalarExpr::Select(c, a, b) => {
                assert!(
                    matches!(*c, ScalarExpr::Binary(BinaryOp::CmpGt, _, _)),
                    "cond must be the interior cmp (position 0), got {c:?}"
                );
                assert_eq!(
                    *a,
                    ScalarExpr::Input(2),
                    "arm a must be Bind(2) -> Input(2)"
                );
                assert_eq!(
                    *b,
                    ScalarExpr::Input(3),
                    "arm b must be Bind(3) -> Input(3)"
                );
            }
            other => panic!("expected Select(cond, a, b), got {other:?}"),
        }
    }

    #[test]
    fn select_dtype_compatible_is_float_only() {
        use crate::ir::input;
        // Mirrors the AOT plan gate: select is legal at every float dtype and
        // declines typed at every int dtype (never a build_plan panic across
        // the trust boundary).
        let body = input(0).select(input(1), input(2)).0;
        for dt in [
            ElementKind::F32,
            ElementKind::F32Strict,
            ElementKind::F64,
            ElementKind::F16,
            ElementKind::Bf16,
        ] {
            assert!(dtype_compatible(&body, dt), "select legal at {dt:?}");
        }
        for dt in [
            ElementKind::I32,
            ElementKind::I64,
            ElementKind::S8,
            ElementKind::U8,
        ] {
            assert!(!dtype_compatible(&body, dt), "select illegal at {dt:?}");
        }
        // …and buried under other ops it still gates.
        let nested = input(0).select(input(1), input(2)).relu().0;
        assert!(!dtype_compatible(&nested, ElementKind::I32));
    }

    #[test]
    fn nextafter_half_dtypes_are_gated_as_honest_misses() {
        use crate::ir::input;
        // Nextafter's half path would step the f32 lattice (wrong neighbor) —
        // dtype_compatible must refuse f16/bf16 while keeping f32/f64 lowerable.
        let body = input(0).binary(BinaryOp::Nextafter, input(1)).0;
        assert!(dtype_compatible(&body, ElementKind::F32));
        assert!(dtype_compatible(&body, ElementKind::F32Strict));
        assert!(dtype_compatible(&body, ElementKind::F64));
        assert!(!dtype_compatible(&body, ElementKind::F16));
        assert!(!dtype_compatible(&body, ElementKind::Bf16));
        // …and buried under other ops it still gates.
        let nested = input(0).binary(BinaryOp::Nextafter, input(1)).relu().0;
        assert!(!dtype_compatible(&nested, ElementKind::F16));
        // The other new binaries keep the normal float rule (halves promote
        // value-correctly — see cuda_binary's Copysign note) and ints still miss.
        let cs = input(0).binary(BinaryOp::Copysign, input(1)).0;
        assert!(dtype_compatible(&cs, ElementKind::F16));
        assert!(!dtype_compatible(&cs, ElementKind::I32));
        let erfc = input(0).unary(crate::ir::UnaryOp::Erfc).0;
        assert!(!dtype_compatible(&erfc, ElementKind::I64));
        assert!(dtype_compatible(&erfc, ElementKind::Bf16));
    }
}
