//! Emit a complete Fuel **FKC kernel contract** from an op + its structure cell.
//!
//! Where [`crate::pattern`] derives only the `pattern:` block (for a fused op),
//! this assembles the *whole* importable contract Fuel reads to bind a kernel:
//! the bundle front-matter plus the per-kernel `accept` / `return` / `op_params`
//! / `caps` / `cost` / `precision` / `determinism` blocks (and `pattern:` when
//! the op is a recognized fusion).
//!
//! The block field set is pinned to the Profile-v1 conformance matrix
//! (*Kernel-Seam Interop* §4.3) and the `ImplId` five-field tuple (FKC §4.11:
//! `backend`, op, `dtypes`, `kernel_source`, `kernel_revision_hash` — five
//! separable wire fields, never a hash). The admissibility predicate **is** the
//! structure key (per `docs/design/kernel-specialization.md`), so each contract
//! carries its [`StructureKey::to_token`] verbatim under `accept` — the planner's
//! miss signal stays honest by construction.
//!
//! # Scope (v1) / reconciliation
//!
//! Elementwise ops. The block *field names* follow §4.3 and the `ImplId` tuple;
//! a few leaf *spellings* (the dtype tokens — review item E5 — and the precision
//! / layout enums) are reconstructed from the FKC annex and reconciled when the
//! full `kernel-contract-format.md` is wired. They are isolated in the small
//! helpers below so reconciliation is a localized change.

use crate::backend::GeneratedKernel;
use crate::ir::{BinaryOp, ExprDag, NodeId, OpDef, ScalarExpr, UnaryOp};
use crate::pattern::{derive_pattern, to_fkc, PatternNode};
use baracuda_kernels_types::{Contiguity, ElementKind, StructureKey, VecWidth};

/// Provider-wide FKC bundle front-matter (FKC §0/§3.1) — emitted once per bundle
/// file, above the per-kernel [`contract`] blocks. `revision_base` is the
/// provider source-tree revision the kernels were built from (the
/// `ImplId.kernel_revision_hash` base). Carries `seam_profiles: [1]` so an
/// importer can reject a contract outside the negotiated seam profile (§3.5).
#[must_use]
pub fn front_matter(backend_name: &str, revision_base: &str) -> String {
    format!(
        "---\n\
         fkc_version: 1\n\
         provider:\n  \
         name: baracuda\n  \
         backend: {backend_name}\n  \
         kernel_source: baracuda\n  \
         link_registry: baracuda_link_registry\n  \
         revision_base: \"{revision_base}\"\n\
         seam_profiles: [1]\n\
         ---\n"
    )
}

/// Emit the per-kernel FKC contract block (a ```` ```fkc ```` fenced section) for
/// `kernel`, generated for `op` at structure cell `key` and lowered by
/// `backend_name`.
///
/// A single graph-`Op` body advertises as a primitive (`op_kind`); a multi-op
/// body advertises as a recognized fusion (`fused_op` + a `pattern:` block).
#[must_use]
pub fn contract(
    op: &OpDef,
    key: &StructureKey,
    kernel: &GeneratedKernel,
    backend_name: &str,
) -> Option<String> {
    // Skip a cell whose dtype has no FKC §5 base-dtype spelling — an unbindable
    // contract would corrupt the planner's honest miss signal (§4.3).
    let dtype = fkc_dtype(key.dtype)?;

    // Increment-1 MULTI-OUTPUT honest miss (typed, no contract; the kernel still
    // generates + works AOT — the cmp/bitwise/Coord precedent). Read against
    // Fuel's actual sources, a multi-output elementwise op has no advertisable
    // identity in the current seam envelope:
    //   1. `op_kind`: no Fuel `OpKind` names an elementwise multi-output dual
    //      (fuel-dispatch fkc/lower.rs `lower_op_kind` has no MulElementwiseBackward
    //      etc.; Fuel splits multi-output backward into per-output OpKinds — e.g.
    //      FlashAttnBackwardQ/K/V — one single-output kernel each).
    //   2. `fused_op`: needs a `pattern:` block, but a `PatternNode`
    //      (fuel-kernel-seam-types) is single-ROOTED — a single-output subgraph;
    //      a forest of N distinct output roots is not expressible, and
    //      `derive_pattern` reads only `op.body` (output 0), so emitting a
    //      contract here would advertise a WRONG single-output pattern that
    //      ignores the other outputs and mis-describes the N-operand key.
    //   3. Fuel's ONLY multi-output ABI (kernel-contract-format.md §5.5, "Option
    //      C") is a `return.bundle` — ONE packed output buffer mapped to a
    //      `FusedOp.output_views` — semantically different from our N DISTINCT
    //      output buffers, and it still needs the fused-op pattern identity (2)
    //      that we cannot provide.
    // So: no contract, until Fuel's seam grows a multi-output region envelope
    // (a forest pattern, or a per-output split). Guarded up front so the wrong
    // single-output pattern is never emitted.
    if op.n_outputs() > 1 {
        return None;
    }

    // Item-01 LAYOUT-VIEW honest miss (typed, no contract; the kernel still
    // generates + runs AOT — the Coord/multi-output/nested-cmp precedent). An
    // address-affecting view (a `Permute`/transpose or `Broadcast` read-through)
    // means the kernel computes `body(transpose(input))`, but our EMITTED pattern
    // grammar cannot say so — verified against Fuel's actual sources:
    //   1. Baracuda's `pattern::PatternNode` is `Op` + `Bind` ONLY — no layout
    //      node and no `OpAttrs` channel. `derive_pattern` walks `op.body`
    //      alone (views live on `OpDef::views`, outside the value walk), so the
    //      pattern it derives describes reading `Input(i)` AT the iteration
    //      coordinate — silently dropping the transpose. Advertising it (a
    //      primitive `op_kind: Relu`, or a `fused_op`) would bind where Fuel's
    //      graph has `relu(transpose(x))`, not `relu(x)`: a wrong bind held off
    //      only by the structure key.
    //   2. Fuel CAN express it honestly, but only in a shape our grammar lacks:
    //      fuel-kernel-seam-types `PatternNode::Op { op: OpTag::Permute, attrs:
    //      OpAttrs { perm } }` with a `perm` guard. Its §4.3 spec
    //      (fkc-fusion-patterns.md) is explicit that a layout op whose ATTRIBUTES
    //      are load-bearing (a transpose's `perm` IS the correctness check) MUST
    //      be matched with `op:` + a `guard:`, NOT via `see_through` — and Fuel's
    //      `see_through` skip is a no-op STUB today anyway (fuel-graph
    //      jit.rs `match_node`'s `SeeThrough` arm). We have no `OpTag`/attrs
    //      vocabulary to author that guard.
    //   3. The concrete-region (decompose) direction rejects layout re-emit
    //      outright (fuel-graph runtime_fused.rs: `Transpose`/`Permute`/`Reshape`
    //      are `UnRepresentable`), so there is no bidirectional identity to bind.
    // So: no contract until Baracuda's pattern grammar grows a layout node + an
    // attrs channel (the `perm` guard). A same-rank `Reshape` / `Identity` view
    // is NOT address-affecting (identity linear map) and still advertises
    // normally — the derived `body`-over-inputs pattern is exactly correct there.
    if crate::plan::op_has_addressing_view(op) {
        return None;
    }

    // Increment-4 GATHER honest miss (typed, no contract; the kernel still
    // generates + runs AOT — the Contraction-node precedent). A data-dependent
    // `ReadIndex::Indexed` read cannot be advertised honestly against Fuel's
    // ACTUAL sources:
    //   1. The index operand's dtype is UNKEYABLE. Baracuda's `StructureKey` has
    //      NO per-operand dtype FIELD — a single operand-0 dtype (structure_key.rs:
    //      "v1 assumes a uniform operand dtype"), so nothing in the token names the
    //      index operand as i32 vs i64. (The dtype's byte size DOES leak
    //      incidentally into that operand's `vec_width` — for a full-shape index an
    //      i32 vectorizes wider than an i64 — but that side-channel is unreliable:
    //      it collapses to equal for the 1-D/broadcast index of index_select /
    //      embedding, where both dtypes are `Scalar`. So the token neither reliably
    //      distinguishes nor is meant to distinguish index dtype.) Fuel's gather
    //      admissibility is instead an explicit per-operand dtype TUPLE —
    //      fuel-dispatch `fkc/cpu_link.rs` keys gather/index_select as `[T, U32, T]`
    //      (the `indices` operand is a FIXED U32 slot; `out: passthrough(source)`).
    //      A Baracuda contract keyed only on T would advertise a kernel Fuel could
    //      bind to the wrong index dtype (i32-kernel ↔ i64/U32-call) — no keyed
    //      field guards it.
    //   2. Even setting dtype aside, the emitted `PatternNode` grammar (`Op` +
    //      `Bind`, no `OpAttrs` channel) cannot carry the gather `axis` or the OOB
    //      policy. Fuel DOES name `OpTag::Gather`/`IndexSelect`
    //      (fuel-kernel-seam-types) — but their identity rides `OpAttrs.axis` and,
    //      for residency gathers, a `fdx.gather.kind` admissibility enum
    //      (fuel-dispatch `fkc/validate.rs`) we have no vocabulary to author.
    //   3. Bespoke gather's OOB semantics (silently skip; embedding zero-fills)
    //      also differ from torch/Fuel gather's in-bounds contract — a third
    //      reason the advertised op would mis-describe the kernel.
    // KEYING RESOLVED (Fuel reply 2026-07-04, `docs/fuel-reply-mixed-dtype-key`):
    // reason 1 is answered — Fuel keys off the FKC per-operand dtype TUPLE
    // assembled from `accept.inputs[i].dtype`, NOT a coarse token, so filling the
    // accept block honestly (data `T`, index `U32`) makes wrong-bind structurally
    // impossible with **no `STRUCTURE_KEY_VERSION` bump** (a Baracuda-only change;
    // our own `StructureKey` stays uniform-dtype). The keyed `u32`-index gather
    // contract is therefore a QUEUED Baracuda-only follow-up (emit the u32 variant
    // + fill accept honestly + advertise an `oob_policy` field), sequenced after
    // ramp #5. Until that follow-up lands, still no contract (honest miss) — reason
    // 2 (the `Op`+`Bind` grammar can't carry the gather axis/OOB) also holds today.
    // Guarded up front so no wrong-dtype gather identity is ever emitted meanwhile.
    if crate::plan::op_has_gather(op) {
        return None;
    }

    // Increment-5 SCATTER honest miss (typed, no contract; the kernel still
    // generates + runs AOT). A `WriteIndex::ScatterIndexed` write is unadvertisable
    // against Fuel's ACTUAL sources for BOTH the gather reasons AND a third:
    //   1. **Index dtype — keyable via Model A, wiring queued** (same as gather;
    //      Fuel reply 2026-07-04). Fuel keys scatter_add/index_add off the
    //      per-operand dtype TUPLE `[T, U32, T, T]` (fuel-dispatch `fkc/cpu_link.rs`:
    //      `base`, FIXED U32 `indices`, `src`, `passthrough(base)` out) assembled
    //      from the accept block — so a `u32`-index scatter contract with honest
    //      per-input dtypes binds correctly, NO `STRUCTURE_KEY_VERSION` bump. That's
    //      the queued Baracuda-only follow-up; NOT the blocker. The blockers below
    //      are.
    //   2. The `Op`+`Bind` `PatternNode` grammar can't carry the scatter `axis`, the
    //      OOB (Skip) policy, or the combine mode. Fuel names `OpKind::ScatterAdd`/
    //      `IndexAdd` (fuel-ir `dispatch.rs`) but their identity rides `OpAttrs.axis`;
    //      it has NO bare `Scatter`, NO `Bincount`/`Histogram` op-kind and NO
    //      scatter-reduce (amax/amin/prod) mode enum at all — so `scatter`
    //      (pure-assign), `bincount`, and any AtomicMax/Min scatter are a DOUBLE
    //      honest miss (net-new vocabulary to negotiate).
    //   3. **Determinism (the increment-5 discipline).** An FP-`atomicAdd` scatter
    //      is run-to-run non-deterministic; an honest contract would have to set
    //      `determinism: nondeterministic` (fuel-dispatch `fkc/schema.rs`), which by
    //      Fuel's precision-coherence rule (`fkc/validate.rs` Rule 9) ALSO obligates
    //      `precision.bit_stable_on_same_hardware: false` + `audited: true`. Baracuda
    //      does not yet author that coupled precision block, so emitting a scatter
    //      contract could ship an INCOHERENT (silently-deterministic-looking) advert
    //      of a nondeterministic kernel — the exact failure the honest-miss rule
    //      forbids. The determinism flip is spelled + ready
    //      (`VariantFidelity::determinism_str` → `nondeterministic`) for when the
    //      queued Model-A scatter contract wiring lands AND the missing op-kind
    //      vocabulary (reason 2) is negotiated; until then, no contract for ANY
    //      scatter. Guarded up front.
    if crate::plan::op_has_scatter(op) {
        return None;
    }

    // Increment 0b honesty gate (tightened by the adversarial review): a body
    // containing a Cmp* ANYWHERE emits a contract only as the u8-out
    // single-op primitive.
    //
    // - Float-mask top-level cmp (out_dtype None): a key-dtype-store kernel is
    //   not Fuel's "comparison → U8 mask" op — advertising `op_kind: Lt` would
    //   bind where Fuel expects a 1-byte mask. No contract.
    // - NESTED cmp (e.g. relu-bw `dy * (x > z)`): the emitted pattern would
    //   encode the interior Cmp* as a DIRECT operand of a float op, but no
    //   constructible Fuel graph has that edge — Fuel's compare builders pin
    //   the output to U8 and its binary ops assert operand-dtype equality, so
    //   every real mask-multiply graph interposes `Cast(U8→float)`, which is
    //   outside the §4.1 pattern grammar and NOT in the §4.3 see-through set.
    //   The advertised matcher could therefore never fire on the graphs it
    //   means (silent coverage loss), while the only structurally-matching
    //   graphs would be all-U8 mask arithmetic — a wrong bind held off only by
    //   the structure key. Withheld until `Cast` joins the pattern
    //   vocabulary; the kernel itself still generates (AOT + seam lowering
    //   are unaffected).
    let out_u8 = op.out_dtype == Some(ElementKind::U8);
    if expr_contains_cmp(&op.body) && !out_u8 {
        return None;
    }

    let pattern = derive_pattern(op).ok();
    let n_ops = pattern.as_ref().map_or(0, count_ops);
    // A u8-out predicate advertises ONLY as the single-op primitive; a FUSED
    // u8-out body would hit the same missing-Cast pattern problem above.
    if out_u8 && n_ops != 1 {
        return None;
    }
    let is_fusion = n_ops > 1;
    let op_line = match &pattern {
        // exactly one graph op → a primitive identity (e.g. `Add`, `AddScalar`).
        // Comparisons use the DISPATCH OpKind spellings (`LessElementwise`, …):
        // Fuel's FKC importer (fuel-dispatch fkc/lower.rs `lower_op_kind`) is an
        // exhaustive string table that typed-rejects unknown names — and a
        // single bad section fails the whole bundle. (The importer expects
        // `AddElementwise`-style names for the arithmetic primitives too; that
        // pre-existing spelling reconciliation is tracked in the module
        // header — comparisons land importable from day one.)
        Some(_) if n_ops == 1 && out_u8 => {
            format!("op_kind: {}", cmp_dispatch_op_kind(&op.body))
        }
        Some(p) if n_ops == 1 => format!("op_kind: {}", root_op_name(p)),
        // ≥2 graph ops → a fused identity carried by the op's stable name.
        Some(_) => format!("fused_op: {}", op.name),
        // body not expressible as a pattern (Const / non-elementwise / bind
        // mismatch) → not advertisable; skip rather than fake an op_kind from
        // the op's free-form name (which is not an OpKind dispatch key).
        None => return None,
    };

    let out_idx = key.n_operands.saturating_sub(1) as usize;
    let params = params_used(&op.body);
    let (prec_mode, prec_ulp) = precision_of(&op.body);

    let mut s = String::from("```fkc\n");
    s.push_str(&format!("kernel: {}_{}\n", op.name, cell_suffix(key)));
    s.push_str(&op_line);
    s.push('\n');
    s.push_str(&format!("blurb: \"{}\"\n", blurb(op, key, dtype, is_fusion)));
    // ImplId tuple (FKC §4.11), kept as five separable fields.
    s.push_str(&format!("backend: {backend_name}\n"));
    s.push_str("kernel_source: baracuda\n");
    s.push_str(&format!("dtypes: [{dtype}]\n"));
    s.push_str(&format!("entry_point: {}\n", kernel.name));
    s.push_str(&format!(
        "kernel_revision_hash: \"{:016x}\"\n",
        revision_hash(&kernel.source)
    ));

    // accept — the admissibility predicate IS the structure key (the honesty
    // invariant); the per-input dtype/layout lines are a human-readable gloss.
    s.push_str("accept:\n");
    s.push_str(&format!("  structure_key: \"{}\"\n", key.to_token()));
    s.push_str("  inputs:\n");
    for i in 0..op.n_inputs as usize {
        s.push_str(&format!(
            "    - dtype: {dtype}\n      layout: {}\n",
            layout_token(key, i)
        ));
    }

    if !params.is_empty() {
        s.push_str("op_params:\n");
        for p in &params {
            // v1 scalar params are f32 launch arguments (the `extract:` carrier).
            s.push_str(&format!("  - name: param{p}\n    dtype: F32\n"));
        }
    }

    s.push_str("return:\n  outputs:\n");
    // A u8-predicate op returns the mask dtype, not the input dtype:
    // `fixed(U8)` per FKC §5.1 ("a constant … comparisons → U8"), the exact
    // rule Fuel's own CPU compare contracts carry. Everything else keeps the
    // uniform-dtype passthrough.
    let dtype_rule = if out_u8 { "fixed(U8)" } else { "same_as_input(0)" };
    s.push_str(&format!(
        "    - dtype_rule: {dtype_rule}\n      \
         shape_rule: same_as_input(0)\n      \
         layout: {}\n      \
         aliasing: none\n",
        layout_token(key, out_idx)
    ));

    s.push_str("caps:\n");
    // A u8 output can never alias a wider input buffer (a 1-byte store into a
    // 4-byte element corrupts neighbors under the grid-stride order), so the
    // predicate cells forbid in-place; uniform-dtype cells keep the existing
    // declaration.
    s.push_str(if out_u8 { "  in_place: forbidden\n" } else { "  in_place: allowed\n" });
    s.push_str(&format!("  alignment_bytes: {}\n", required_align(key)));
    s.push_str(&format!("  awkward_layout: {}\n", awkward_layout(key)));
    // The unit of the kernel's `n` launch argument: a vectorized/packed cell
    // counts w-element VECTORS (n/width), everything else elements. Pinned on
    // the wire per the 2026-07-03 Fuel exchange (documentation-only for Fuel
    // today — launches are provider-internal — but load-bearing for their
    // declared-cost trampoline, and an 8x launch/cost hazard if ever assumed).
    let cw = crate::cuda::effective_count_width(&crate::plan::build_plan(op, key));
    if cw > 1 {
        s.push_str(&format!("  count_unit: vectors_x{cw}\n"));
    } else {
        s.push_str("  count_unit: elements\n");
    }

    s.push_str("cost:\n");
    s.push_str("  provenance: declared\n");
    s.push_str("  class: elementwise\n");
    s.push_str(&format!("  flops_per_elem: {}\n", count_flops(&op.body)));
    s.push_str(&format!("  bytes_per_elem: {}\n", bytes_per_elem(op, key)));

    s.push_str("precision:\n");
    s.push_str(&format!("  mode: {prec_mode}\n"));
    if let Some(u) = prec_ulp {
        s.push_str(&format!("  max_ulp: {u}\n"));
    }
    s.push_str("determinism: bitwise\n");

    if let (Some(p), true) = (&pattern, is_fusion) {
        s.push_str(&to_fkc(p));
    }

    s.push_str("```\n");
    Some(s)
}

// ---------------------------------------------------------------------------
// Pattern shape helpers
// ---------------------------------------------------------------------------

fn count_ops(node: &PatternNode) -> usize {
    match node {
        PatternNode::Bind(_) => 0,
        PatternNode::Op { operands, .. } => 1 + operands.iter().map(count_ops).sum::<usize>(),
    }
}

fn root_op_name(node: &PatternNode) -> String {
    match node {
        PatternNode::Op { op, .. } => op.clone(),
        PatternNode::Bind(_) => "Identity".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Body scans
// ---------------------------------------------------------------------------

/// Sorted, unique runtime-parameter indices referenced by the body.
fn params_used(body: &ScalarExpr) -> Vec<u8> {
    let mut out = Vec::new();
    scan_params(body, &mut out);
    out.sort_unstable();
    out.dedup();
    out
}

fn scan_params(e: &ScalarExpr, out: &mut Vec<u8>) {
    match e {
        ScalarExpr::Param(i) => out.push(*i),
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => {}
        ScalarExpr::Unary(_, x) => scan_params(x, out),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => {
            scan_params(a, out);
            scan_params(b, out);
        }
    }
}

/// Declared flop count per output element: **one per distinct arithmetic / unary
/// node in the value DAG**.
///
/// Counting the [`ExprDag`] (not the authored tree) charges a shared interior
/// *once* — matching the item-02 emitter, which hoists it to a `tmp` computed
/// once. A body with no sharing counts identically to the old tree walk, so this
/// is a pure fix: `flops_per_elem` for a body with a duplicated subtree **drops**
/// (to the honest, actually-computed count) and never rises.
fn count_flops(e: &ScalarExpr) -> u32 {
    let dag = ExprDag::from_expr(e);
    (0..dag.len() as NodeId)
        .filter(|&id| !dag.node(id).is_leaf())
        .count() as u32
}

/// Conservative max-ULP bound over the lowered body — a *declared upper bound*,
/// the sum of the **vendor-approximate** ops' errors (transcendentals ~2 ulp,
/// `powf` 4, `logf` 1, the hand-rolled `Sigmoid`/`Silu`/`Gelu` composites a bit
/// higher). Correctly-rounded ops (the infix arithmetic, `Neg`/`Abs`/`Sqr`/`Sqrt`/
/// `Recip`/`Relu`/`Floor`/`Ceil`/`Round`/`Sign`/`Step`, `Max`/`Min`/`Rem`)
/// contribute 0, so a body with none is `correctly_rounded`. Over-stating is safe
/// (the planner won't admit into a too-tight slot); under-stating is not.
fn ulp_bound(e: &ScalarExpr) -> f64 {
    match e {
        // Coord rates 0 like the other leaves: the long-long → float/double
        // cast is exact under the documented caller precondition (axis extent
        // within the dtype's exact-integer range — see `ScalarExpr::Coord`).
        // Defensive today: no contract is ever emitted for a Coord body
        // (derive_pattern rejects CoordUnsupported first), but the table
        // stays exhaustive on purpose so the rating is decided here, not
        // silently defaulted, when the Iota bridge lands.
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => 0.0,
        ScalarExpr::Unary(op, x) => ulp_bound(x) + unary_ulp(*op),
        ScalarExpr::Binary(op, a, b) => ulp_bound(a) + ulp_bound(b) + binary_ulp(*op),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b) => ulp_bound(a) + ulp_bound(b),
    }
}

/// Per-op CUDA f32 ULP error. Correctly-rounded / exact ops (`Neg`/`Abs`/`Sqr`/
/// `Sqrt`/`Recip`/`Relu`/`Floor`/`Ceil`/`Round`/`Sign`/`Step`/`Trunc`) are 0.
/// Deliberately **exhaustive** (no wildcard): a future op must be rated here on
/// purpose, or the compiler objects — a silent 0.0 default would under-state,
/// and under-stating is the one unsafe direction.
fn unary_ulp(op: UnaryOp) -> f64 {
    match op {
        // exact / correctly rounded
        UnaryOp::Neg
        | UnaryOp::Abs
        | UnaryOp::Sqr
        | UnaryOp::Sqrt
        | UnaryOp::Recip
        | UnaryOp::Relu
        | UnaryOp::Floor
        | UnaryOp::Ceil
        | UnaryOp::Round
        | UnaryOp::Sign
        | UnaryOp::Step
        | UnaryOp::Trunc => 0.0,
        UnaryOp::Log | UnaryOp::Expm1 | UnaryOp::Log1p | UnaryOp::Cbrt | UnaryOp::Log2 => 1.0,
        UnaryOp::Exp
        | UnaryOp::Tanh
        | UnaryOp::Erf
        | UnaryOp::Sin
        | UnaryOp::Cos
        | UnaryOp::Rsqrt
        | UnaryOp::Exp2
        | UnaryOp::Log10
        | UnaryOp::Cosh
        | UnaryOp::Atan => 2.0,
        // composites of expf/erff + rounding — conservatively a bit higher —
        // and the vendor-3-ulp inverse/hyperbolic tier.
        UnaryOp::Sigmoid
        | UnaryOp::Silu
        | UnaryOp::Gelu
        | UnaryOp::Sinh
        | UnaryOp::Acos
        | UnaryOp::Asinh
        | UnaryOp::Atanh => 3.0,
        // domain-edge-sensitive fns at the loosest shared tier (vendor: tanf 4,
        // asinf 4, acoshf 4, erfcf 4).
        UnaryOp::Tan | UnaryOp::Asin | UnaryOp::Acosh | UnaryOp::Erfc => 4.0,
        // lgammaf: vendor declares 6 ulp OUTSIDE the interval (-10.001, -2.264)
        // and larger inside it (near the negative-real poles). 6 is the honest
        // vendor headline bound — rating it 4 (the previous loosest tier) would
        // under-state, so this op introduces the 6 tier; the near-pole caveat is
        // inherent to lgamma at any finite rating.
        UnaryOp::Lgamma => 6.0,
    }
}

/// Per-op CUDA f32 ULP error for the binary fns, mirroring [`unary_ulp`]
/// (exhaustive, no wildcard). `Max`/`Min` (compare-selects) and the bit-level
/// increment-0a ops (`copysignf`/`nextafterf`/`fmaxf`/`fminf`/`fmodf` — all
/// vendor 0-ulp) contribute 0. Floored `Rem` is rated UNBOUNDED (infinity ⇒
/// the contract emits `approximate` with no `max_ulp` claim): it lowers as
/// the composite `a - floor(a/b)*b`, and a 0.5-ulp quotient perturbation that
/// crosses an integer boundary flips the floor — the result is then off by
/// |b|, which no finite result-ULP number bounds. (Inherent to the floored-mod
/// formula; torch.remainder behaves identically. `RemTrunc` = `fmodf` is a
/// genuinely exact vendor primitive and keeps 0.)
fn binary_ulp(op: BinaryOp) -> f64 {
    match op {
        BinaryOp::Max
        | BinaryOp::Min
        | BinaryOp::Copysign
        | BinaryOp::Nextafter
        | BinaryOp::FmaxIeee
        | BinaryOp::FminIeee
        | BinaryOp::RemTrunc => 0.0,
        // The Cmp* predicates are exact C-operator compares producing exactly
        // 1.0/0.0 — 0 ulp. A compare of *approximate* subexpressions is still
        // an exact compare: this table rates each op's own rounding, and the
        // subexpressions' tiers are summed by `ulp_bound` as usual (the
        // decision-boundary sensitivity of a predicate is not a result-ULP
        // quantity — same modeling call as rating Sign/Step 0 in `unary_ulp`).
        BinaryOp::CmpEq
        | BinaryOp::CmpNe
        | BinaryOp::CmpLt
        | BinaryOp::CmpLe
        | BinaryOp::CmpGt
        | BinaryOp::CmpGe => 0.0,
        BinaryOp::Rem => f64::INFINITY,
        BinaryOp::Atan2 => 3.0, // vendor atan2f: 3 ulp
        BinaryOp::Pow => 4.0,
        // The increment-0c integer ops are BIT-EXACT: "exact" here means exact
        // WRAPPING two's-complement semantics (and, for the logical ops, the
        // exact 0/1 normalization) — there is no rounding step at all, so 0 is
        // the honest rating, not an under-statement. (Today these ops emit NO
        // contract — no OpTag/lower_op_kind name exists, so derive_pattern
        // rejects first — but this table stays exhaustive on purpose: when
        // Fuel names them, the rating is already decided here, not silently
        // defaulted.)
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => 0.0,
    }
}

/// Precision contract: `correctly_rounded` (0 ulp, bit-reproducible) when the
/// body is all correctly-rounded primitives, else `approximate` with the
/// conservative [`ulp_bound`]. (`F32Strict` would force `correctly_rounded` — it
/// is a precision mode, not a wire dtype.)
/// Whether `e` contains any comparison predicate anywhere (not just the root)
/// — the contract-withholding walk for the missing-`Cast` pattern gap. Shared
/// with the JIT's interior-cmp decline (`crate::jit`).
pub(crate) fn expr_contains_cmp(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => false,
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b) => expr_contains_cmp(a) || expr_contains_cmp(b),
        ScalarExpr::Unary(_, a) => expr_contains_cmp(a),
        ScalarExpr::Binary(bop, a, b) => {
            bop.is_cmp() || expr_contains_cmp(a) || expr_contains_cmp(b)
        }
    }
}

/// The dispatch `OpKind` spelling for a validated u8-out predicate body (root
/// IS a Cmp* — `assert_valid_out_dtype` guarantees it before any contract is
/// emitted). These are the exact strings Fuel's `lower_op_kind` table accepts.
fn cmp_dispatch_op_kind(body: &ScalarExpr) -> &'static str {
    let ScalarExpr::Binary(bop, _, _) = body else {
        unreachable!("u8-out body root is a validated Cmp*");
    };
    match bop {
        BinaryOp::CmpEq => "EqualElementwise",
        BinaryOp::CmpNe => "NotEqualElementwise",
        BinaryOp::CmpLt => "LessElementwise",
        BinaryOp::CmpLe => "LessEqualElementwise",
        BinaryOp::CmpGt => "GreaterElementwise",
        BinaryOp::CmpGe => "GreaterEqualElementwise",
        _ => unreachable!("u8-out body root is a validated Cmp*"),
    }
}

fn precision_of(body: &ScalarExpr) -> (&'static str, Option<u32>) {
    let u = ulp_bound(body);
    if u <= 0.0 {
        ("correctly_rounded", Some(0))
    } else if u.is_infinite() {
        // A body containing an op with no finite result-ULP bound (floored
        // Rem's quotient-boundary flip): honest contract = approximate with
        // NO max_ulp claim, never an under-stated finite number.
        ("approximate", None)
    } else {
        ("approximate", Some(u.ceil() as u32))
    }
}

// ---------------------------------------------------------------------------
// Structure-cell projections
// ---------------------------------------------------------------------------

fn layout_token(key: &StructureKey, i: usize) -> &'static str {
    match key.operands[i].contig {
        Contiguity::Contig => "contiguous",
        Contiguity::InnerContig => "inner_contiguous",
        Contiguity::Strided => "strided",
        Contiguity::Broadcast => "broadcast",
    }
}

/// A kernel specialized for a strided/broadcast cell handles awkward layouts;
/// a contiguous/vectorized cell requires the packed layout it was built for.
fn awkward_layout(key: &StructureKey) -> &'static str {
    match key.operands[0].contig {
        Contiguity::Strided | Contiguity::Broadcast => "handles_strided",
        Contiguity::Contig | Contiguity::InnerContig => "requires_contiguous",
    }
}

/// Required base-pointer alignment (bytes): a vectorized cell needs its vector
/// width; a scalar cell needs the dtype's natural alignment.
///
/// SEAM NOTE (H2, adversarial pass 2026-07-02 — RESOLVED 2026-07-03): the
/// kernel's `n` argument is a VECTOR count (`n / width`) on vectorized/packed
/// cells and an ELEMENT count otherwise — an 8x hazard for any consumer that
/// assumed elements. The caps block now states it explicitly (`count_unit:`,
/// via `effective_count_width`, which mirrors the emitter's dispatch including
/// its scalar fallbacks); Fuel confirmed they never derive launches from
/// contracts and will consume the field when their declared-cost trampoline
/// compiles cost expressions over `n`.
fn required_align(key: &StructureKey) -> u32 {
    let dsz = dtype_size(key.dtype);
    match key.operands[0].vec_width {
        VecWidth::V8 => (8 * dsz).min(16),
        VecWidth::V4 => (4 * dsz).min(16),
        VecWidth::V2 => 2 * dsz,
        VecWidth::Scalar => dsz,
    }
}

fn bytes_per_elem(op: &OpDef, key: &StructureKey) -> u32 {
    // inputs at the key dtype + one output at the (possibly narrower, u8-mask)
    // output dtype (broadcast operands touch fewer; this is a declared upper
    // estimate). Uniform ops reduce to the old (n_inputs + 1) * size exactly.
    u32::from(op.n_inputs) * dtype_size(key.dtype) + dtype_size(op.out_dtype.unwrap_or(key.dtype))
}

/// `<op>_<dtype>_<contig0>_<vec0>` cell discriminator for the readable `kernel`
/// name (the linkable symbol is `entry_point`).
fn cell_suffix(key: &StructureKey) -> String {
    let o = &key.operands[0];
    format!("{}_{}_{}", dtype_short(key.dtype), contig_short(o.contig), vec_short(o.vec_width))
}

fn blurb(op: &OpDef, key: &StructureKey, dtype: &str, is_fusion: bool) -> String {
    let kind = if is_fusion { "fused" } else { "elementwise" };
    format!("{} {} ({}, {} layout).", kind, op.name, dtype, layout_token(key, 0))
}

// ---------------------------------------------------------------------------
// Leaf spellings (reconciled against the FKC annex; review item E5 for dtypes)
// ---------------------------------------------------------------------------

/// FKC §5 logical-DType token, or `None` for a dtype with no §5 *base-dtype*
/// slot (so the caller skips the cell rather than emit an unbindable contract).
///
/// Reconciled to FKC rev-4 §5 (review item E5): `Bool` → `U8` (Fuel has no Bool
/// dtype — masks are U8), signed-8 → `I8`, `F32Strict` rides as `F32` (a
/// precision mode, not a wire dtype). Packed sub-byte / quant payloads
/// (`S4`/`U4`/`Bin`) ride the **FDX sidecar**, not a base dtype, so carry no
/// token here; `Fp8E5M2` and complex have no §5 slot yet — all return `None`.
fn fkc_dtype(dt: ElementKind) -> Option<&'static str> {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, Fp8E4M3, Fp8E5M2, F16, F32, F32Strict, F64, I32,
        I64, S4, S8, U4, U8,
    };
    Some(match dt {
        F32 | F32Strict => "F32",
        F16 => "F16",
        Bf16 => "BF16",
        F64 => "F64",
        I32 => "I32",
        I64 => "I64",
        S8 => "I8",        // §5: signed-8 spells I8
        U8 | Bool => "U8", // §5 (B5/E5): Fuel has no Bool — masks are U8
        Fp8E4M3 => "F8E4M3",
        // No §5 base-dtype slot: FDX-sidecar payloads + unlisted fp8 / complex.
        Fp8E5M2 | S4 | U4 | Bin | Complex32 | Complex64 => return None,
    })
}

fn dtype_short(dt: ElementKind) -> &'static str {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, Fp8E4M3, Fp8E5M2, F16, F32, F32Strict, F64, I32,
        I64, S4, S8, U4, U8,
    };
    match dt {
        F32 | F32Strict => "f32",
        F16 => "f16",
        Bf16 => "bf16",
        F64 => "f64",
        I32 => "i32",
        I64 => "i64",
        S8 => "s8",
        U8 => "u8",
        Bool => "bool",
        Fp8E4M3 => "e4m3",
        Fp8E5M2 => "e5m2",
        S4 => "s4",
        U4 => "u4",
        Bin => "b1",
        Complex32 => "c32",
        Complex64 => "c64",
    }
}

fn contig_short(c: Contiguity) -> &'static str {
    match c {
        Contiguity::Contig => "co",
        Contiguity::InnerContig => "ic",
        Contiguity::Strided => "st",
        Contiguity::Broadcast => "br",
    }
}

fn vec_short(v: VecWidth) -> &'static str {
    match v {
        VecWidth::Scalar => "v1",
        VecWidth::V2 => "v2",
        VecWidth::V4 => "v4",
        VecWidth::V8 => "v8",
    }
}

fn dtype_size(dt: ElementKind) -> u32 {
    use ElementKind::{
        Bf16, Bin, Bool, Complex32, Complex64, Fp8E4M3, Fp8E5M2, F16, F32, F32Strict, F64, I32,
        I64, S4, S8, U4, U8,
    };
    match dt {
        S4 | U4 | Bin => 1, // sub-byte: round up to a byte for the declared estimate
        S8 | U8 | Bool | Fp8E4M3 | Fp8E5M2 => 1,
        F16 | Bf16 => 2,
        F32 | F32Strict | I32 => 4,
        F64 | I64 | Complex32 => 8,
        Complex64 => 16,
    }
}

/// 64-bit FNV-1a over the kernel source — the `ImplId.kernel_revision_hash` base.
/// Stable and dependency-free (unlike `DefaultHasher`, which is unspecified).
pub(crate) fn revision_hash(src: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in src.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{input, param, OpDef};
    use crate::{generate, Cuda};
    use baracuda_kernels_types::{
        structure_key, ArchSku, ElementKind, OpCategory, OperandDesc,
    };

    fn key_for(n_operands: usize, op_cat: OpCategory) -> StructureKey {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let operands: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(op_cat, &operands, ArchSku::Sm89)
    }

    #[test]
    fn contraction_is_an_honest_miss_no_contract() {
        use crate::ir::{reduced, ContractionAxes};
        use crate::pattern::PatternError;
        // The generated contraction node must NEVER leak a bindable elementwise
        // contract — the honest-miss wall holds until the Fuel region grammar
        // for contractions lands (item-10 spike §10).
        let mm = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        );
        let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let kernel = generate(&mm, &key, &Cuda);
        assert!(contract(&mm, &key, &kernel, "cuda").is_none());
        assert!(matches!(
            crate::derive_pattern(&mm),
            Err(PatternError::NotElementwise)
        ));
    }

    #[test]
    fn viewed_op_is_an_honest_miss_no_contract() {
        use crate::ir::View;
        use crate::pattern::PatternError;
        // A fused transpose-elementwise (relu(x^T)) computes body(transpose(x)),
        // but the Op+Bind pattern grammar (no layout node, no attrs channel) can't
        // express the transpose — advertising `op_kind: Relu` would bind where
        // Fuel's graph has relu(transpose(x)). Honest miss (kernel still AOT-runs).
        let op = OpDef::elementwise("relu_t", 1, &[ElementKind::F32], input(0).relu())
            .with_views(vec![View::Permute { perm: vec![1, 0] }]);
        let key = key_for(2, OpCategory::UnaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        assert!(
            contract(&op, &key, &kernel, "cuda").is_none(),
            "a viewed op must emit NO contract (the transpose is inexpressible)"
        );
        assert!(matches!(
            crate::derive_pattern(&op),
            Err(PatternError::ViewUnsupported)
        ));
    }

    #[test]
    fn gathered_op_is_an_honest_miss_no_contract() {
        use crate::ir::OobPolicy;
        use crate::pattern::PatternError;
        // A gather's index-operand dtype (i32/i64) is unkeyable in Baracuda's
        // single-dtype token, while Fuel's gather admissibility is a per-operand
        // dtype tuple `[T, U32, T]` — advertising `op_kind: Gather` would let Fuel
        // bind the wrong index dtype. Honest miss, AOT-only (kernel still runs).
        let op = OpDef::gather("gather", &[ElementKind::F32], 0, OobPolicy::Skip, ElementKind::I32);
        let key = key_for(3, OpCategory::BinaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        assert!(
            contract(&op, &key, &kernel, "cuda").is_none(),
            "a gathered op must emit NO contract (the index dtype is unkeyable)"
        );
        assert!(matches!(
            crate::derive_pattern(&op),
            Err(PatternError::GatherUnsupported)
        ));
    }

    #[test]
    fn scattered_op_is_an_honest_miss_no_contract() {
        use crate::pattern::PatternError;
        // A scatter's index dtype is unkeyable (like gather; Fuel keys scatter_add
        // as `[T, U32, T, T]`), the Op+Bind grammar can't carry axis/OOB/combine
        // (Fuel has no bare Scatter/Bincount op-kind), AND an FP-atomic scatter's
        // determinism flip (nondeterministic + coupled precision) is unauthored.
        // Honest miss, AOT-only for BOTH the pure-assign scatter and scatter_add.
        let key = key_for(3, OpCategory::BinaryElementwise);
        for op in [
            OpDef::scatter("scatter", &[ElementKind::F32], 0, ElementKind::I32),
            OpDef::scatter_add("scatter_add", &[ElementKind::F32], 0, ElementKind::I32),
        ] {
            let kernel = generate(&op, &key, &Cuda);
            assert!(
                contract(&op, &key, &kernel, "cuda").is_none(),
                "a scattered op must emit NO contract"
            );
            assert!(matches!(
                crate::derive_pattern(&op),
                Err(PatternError::ScatterUnsupported)
            ));
        }
    }

    #[test]
    fn identity_view_still_advertises_a_contract() {
        use crate::ir::View;
        // The view guard is PRECISE to address-affecting views: an all-Identity
        // view (an identity linear map) leaves the body-over-inputs pattern exactly
        // correct, so the op still advertises — same as view-free.
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1))
            .with_views(vec![View::Identity, View::Identity]);
        let key = key_for(3, OpCategory::BinaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        assert!(
            contract(&op, &key, &kernel, "cuda").is_some(),
            "an all-Identity view must not suppress the contract"
        );
    }

    #[test]
    fn count_unit_matches_the_emitted_abi() {
        use crate::{generate, Cuda};
        let c = |op: &OpDef, key: &StructureKey| {
            let k = generate(op, key, &Cuda);
            contract(op, key, &k, "cuda").unwrap()
        };
        let add = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        // f32 contiguous/aligned → float4 kernel: n counts 4-element vectors.
        let kf = key_for(3, OpCategory::BinaryElementwise);
        assert!(c(&add, &kf).contains("count_unit: vectors_x4"));
        // f16 contiguous/aligned → packed half2 V8 kernel: 8-element vectors.
        let h = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F16, 256);
        let addh = OpDef::elementwise("add", 2, &[ElementKind::F16], input(0) + input(1));
        let kh = structure_key(OpCategory::BinaryElementwise, &[h, h, h], ArchSku::Sm89);
        assert!(c(&addh, &kh).contains("count_unit: vectors_x8"));
        // i32 keys V4 but has no int vector/packed path → the SCALAR fallback:
        // the contract must say elements, mirroring the emitted ABI, not the key.
        let gi = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::I32, 256);
        let addi = OpDef::elementwise("add", 2, &[ElementKind::I32], input(0) + input(1));
        let ki = structure_key(OpCategory::BinaryElementwise, &[gi, gi, gi], ArchSku::Sm89);
        assert!(c(&addi, &ki).contains("count_unit: elements"));
        // Strided cell → elements.
        let t = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::F32, 256);
        let kt = structure_key(OpCategory::BinaryElementwise, &[t, t, t], ArchSku::Sm89);
        assert!(c(&add, &kt).contains("count_unit: elements"));
    }

    #[test]
    fn flops_count_dedups_shared_subtree() {
        use crate::ir::konst;
        // out = g / (g + 1), g = a*b: the product appears on both sides but is
        // computed once by the emitter, so it is charged once. Distinct DAG
        // non-leaf nodes: Mul, Add, Div = 3 (a tree walk would count Mul twice → 4).
        let g = input(0) * input(1);
        assert_eq!(count_flops(&(g.clone() / (g + konst(1.0))).0), 3);
        // Structurally-identical products authored separately still hash-cons to one.
        let (m1, m2) = (input(0) * input(1), input(0) * input(1));
        assert_eq!(count_flops(&(m1 / (m2 + konst(1.0))).0), 3);
        // A genuinely different second product does NOT merge → 4 ops.
        let distinct = (input(0) * input(1)) / ((input(0) * input(2)) + konst(1.0));
        assert_eq!(count_flops(&distinct.0), 4);
    }

    #[test]
    fn reduction_is_an_honest_miss_no_contract() {
        use crate::ir::ReduceOp;
        use crate::pattern::PatternError;
        use baracuda_kernels_types::AxisMask;
        // A general-path reduction (explicit axis set) must NEVER leak a bindable
        // elementwise contract — the honest-miss wall (§5f/§6). Pins that item 03's
        // axes/keepdim did not open a path around the `NotElementwise` reject.
        let a = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let op = OpDef::reduction_axes(
            "sum",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b01),
            false,
        );
        let kernel = generate(&op, &key, &Cuda);
        assert!(contract(&op, &key, &kernel, "cuda").is_none());
        assert!(matches!(derive_pattern(&op), Err(PatternError::NotElementwise)));
    }

    #[test]
    fn prod_and_hetero_out_reductions_are_honest_misses_no_contract() {
        use crate::ir::{konst, reduced, BinaryOp, ReduceOp};
        use crate::pattern::PatternError;
        // The 0e reductions (Prod combiner; boolean/count hetero-out via a Cmp*
        // post) emit NO contract — reductions are not in the elementwise pattern
        // vocabulary, AND Fuel has no ProdReduce/Any/All/CountNonzero OpKind yet
        // (fuel-cuda-backend/src/baracuda/reduce.rs), so a contract would be
        // un-importable. Honest miss, typed (NotElementwise), same wall as Sum.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);

        // (a) Prod.
        let prod_out = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
        let pk = structure_key(OpCategory::Reduction, &[a, prod_out], ArchSku::Sm89);
        let prod = OpDef::reduction("p", 1, &[ElementKind::F32], input(0), ReduceOp::Prod);
        assert!(contract(&prod, &pk, &generate(&prod, &pk, &Cuda), "cuda").is_none());
        assert!(matches!(derive_pattern(&prod), Err(PatternError::NotElementwise)));

        // (b) hetero-out any (Sum(x!=0) → u8 via a Cmp* post).
        let any_out = OperandDesc::new(1, &[256], &[1], ElementKind::U8, 256);
        let ak = structure_key(OpCategory::Reduction, &[a, any_out], ArchSku::Sm89);
        let mut any = OpDef::reduction_post(
            "any",
            1,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Sum,
            reduced(0).binary(BinaryOp::CmpGt, konst(0.0)),
        );
        any.out_dtype = Some(ElementKind::U8);
        assert!(contract(&any, &ak, &generate(&any, &ak, &Cuda), "cuda").is_none());
        assert!(matches!(derive_pattern(&any), Err(PatternError::NotElementwise)));
    }

    #[test]
    fn front_matter_has_provider_and_seam_profiles() {
        let fm = front_matter("cuda", "abc123");
        assert!(fm.contains("fkc_version: 1"));
        assert!(fm.contains("name: baracuda"));
        assert!(fm.contains("link_registry: baracuda_link_registry"));
        assert!(fm.contains("seam_profiles: [1]"));
        assert!(fm.contains("revision_base: \"abc123\""));
    }

    #[test]
    fn primitive_add_uses_op_kind_and_carries_required_blocks() {
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        let key = key_for(3, OpCategory::BinaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        let c = contract(&op, &key, &kernel, "cuda").unwrap();

        // primitive → op_kind, no fused_op, no pattern block.
        assert!(c.contains("op_kind: Add"));
        assert!(!c.contains("fused_op:"));
        assert!(!c.contains("pattern:"));
        // ImplId five fields all present + separable.
        assert!(c.contains("backend: cuda"));
        assert!(c.contains("kernel_source: baracuda"));
        assert!(c.contains("dtypes: [F32]"));
        assert!(c.contains("entry_point: "));
        assert!(c.contains("kernel_revision_hash: \""));
        // required §4.3 blocks.
        for block in [
            "accept:", "structure_key: \"sk1|", "return:", "caps:", "cost:", "precision:",
            "determinism: bitwise",
        ] {
            assert!(c.contains(block), "missing block: {block}");
        }
        // correctly-rounded arithmetic.
        assert!(c.contains("mode: correctly_rounded"));
    }

    #[test]
    fn fused_activation_uses_fused_op_with_pattern() {
        // relu(a + b) — two graph ops → a fused identity + a pattern block.
        let op = OpDef::elementwise(
            "relu_add",
            2,
            &[ElementKind::F32],
            (input(0) + input(1)).relu(),
        );
        let key = key_for(3, OpCategory::BinaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        let c = contract(&op, &key, &kernel, "cuda").unwrap();
        assert!(c.contains("fused_op: relu_add"));
        assert!(!c.contains("op_kind:"));
        assert!(c.contains("pattern:"));
        assert!(c.contains("op: Relu"));
    }

    #[test]
    fn scalar_param_emits_op_params_and_transcendental_relaxes_precision() {
        // silu(x * p0 + p1): a transcendental (approximate) with two scalar params.
        let op = OpDef::elementwise(
            "affine_silu",
            1,
            &[ElementKind::F32],
            (input(0) * param(0) + param(1)).silu(),
        );
        let key = key_for(2, OpCategory::UnaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        let c = contract(&op, &key, &kernel, "cuda").unwrap();
        assert!(c.contains("op_params:"));
        assert!(c.contains("name: param0"));
        assert!(c.contains("name: param1"));
        assert!(c.contains("mode: approximate"));
        // silu(x*p0 + p1): the silu composite (~3 ulp); arithmetic is exact.
        assert!(c.contains("max_ulp: 3"));
    }

    #[test]
    fn revision_hash_is_source_sensitive() {
        assert_ne!(revision_hash("kernel a"), revision_hash("kernel b"));
        assert_eq!(revision_hash("stable"), revision_hash("stable"));
    }

    fn key_dtype(dt: ElementKind, n_operands: usize) -> StructureKey {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], dt, 256);
        let operands: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89)
    }

    // The dtype-classification tests don't exercise CUDA codegen (which rightly
    // rejects Bool/Complex), only the contract's dtype channel — a stand-in kernel.
    fn stub_kernel() -> GeneratedKernel {
        GeneratedKernel { name: "k".into(), source: "s".into() }
    }

    #[test]
    fn bool_dtype_maps_to_u8_not_bool() {
        // §5 (B5/E5): Fuel has no Bool dtype — a provider's Bool rides as U8.
        let op = OpDef::elementwise("eq", 2, &[ElementKind::Bool], input(0) + input(1));
        let key = key_dtype(ElementKind::Bool, 3);
        let c = contract(&op, &key, &stub_kernel(), "cuda").unwrap();
        assert!(c.contains("dtypes: [U8]"));
        assert!(!c.contains("Bool"));
    }

    #[test]
    fn unsupported_dtype_yields_no_contract() {
        // Complex has no FKC §5 base-dtype slot — skip the cell (honest miss),
        // never emit an unbindable `dtypes: [C64]` contract.
        let op = OpDef::elementwise("add", 2, &[ElementKind::Complex64], input(0) + input(1));
        let key = key_dtype(ElementKind::Complex64, 3);
        assert!(contract(&op, &key, &stub_kernel(), "cuda").is_none());
    }

    #[test]
    fn vocab_ulp_tiers_rate_the_new_fns() {
        use crate::ir::{BinaryOp, UnaryOp};
        // Op-sensitive precision for the increment-0a vocabulary: the declared
        // max_ulp is the fn's vendor tier, and the exact/bit-level ops stay
        // correctly_rounded. (Under-stating is the unsafe direction — these pins
        // hold the table honest.)
        let u = |op: UnaryOp| precision_of(&input(0).unary(op).0);
        assert_eq!(u(UnaryOp::Trunc), ("correctly_rounded", Some(0)));
        assert_eq!(u(UnaryOp::Log1p), ("approximate", Some(1)));
        assert_eq!(u(UnaryOp::Expm1), ("approximate", Some(1)));
        assert_eq!(u(UnaryOp::Cbrt), ("approximate", Some(1)));
        assert_eq!(u(UnaryOp::Exp2), ("approximate", Some(2)));
        assert_eq!(u(UnaryOp::Sinh), ("approximate", Some(3)));
        // the domain-edge-sensitive fns carry the loose tiers.
        assert_eq!(u(UnaryOp::Tan), ("approximate", Some(4)));
        assert_eq!(u(UnaryOp::Asin), ("approximate", Some(4)));
        assert_eq!(u(UnaryOp::Acosh), ("approximate", Some(4)));
        assert_eq!(u(UnaryOp::Erfc), ("approximate", Some(4)));
        assert_eq!(u(UnaryOp::Lgamma), ("approximate", Some(6)));
        let b = |op: BinaryOp| precision_of(&input(0).binary(op, input(1)).0);
        assert_eq!(b(BinaryOp::Atan2), ("approximate", Some(3)));
        // bit-level / exact binaries are correctly rounded.
        assert_eq!(b(BinaryOp::Copysign), ("correctly_rounded", Some(0)));
        assert_eq!(b(BinaryOp::Nextafter), ("correctly_rounded", Some(0)));
        assert_eq!(b(BinaryOp::FmaxIeee), ("correctly_rounded", Some(0)));
        assert_eq!(b(BinaryOp::FminIeee), ("correctly_rounded", Some(0)));
        assert_eq!(b(BinaryOp::RemTrunc), ("correctly_rounded", Some(0)));
    }

    #[test]
    fn int_ops_rate_zero_ulp_and_emit_no_contract() {
        use crate::ir::BinaryOp;
        const INT_OPS: [BinaryOp; 8] = [
            BinaryOp::BitAnd,
            BinaryOp::BitOr,
            BinaryOp::BitXor,
            BinaryOp::Shl,
            BinaryOp::Shr,
            BinaryOp::LogicalAnd,
            BinaryOp::LogicalOr,
            BinaryOp::LogicalXor,
        ];
        // Precision table: all eight rated 0 EXHAUSTIVELY (bit-exact wrapping
        // int semantics — no rounding step exists), pinned per op so a future
        // arm shuffle can't silently re-rate one.
        for op in INT_OPS {
            assert_eq!(
                precision_of(&input(0).binary(op, input(1)).0),
                ("correctly_rounded", Some(0)),
                "{op:?}"
            );
        }
        // Contract: HONEST MISS — neither OpTag 0.10.2 nor lower_op_kind names
        // the bitwise/logical ops, so no pattern derives and no contract is
        // emitted; the kernel itself still generates (bitwise at i32, logical
        // at u8 — the audited-legal cells).
        use crate::pattern::PatternError;
        let band = OpDef::elementwise(
            "band",
            2,
            &[ElementKind::I32],
            input(0).binary(BinaryOp::BitAnd, input(1)),
        );
        let ki = key_dtype(ElementKind::I32, 3);
        let k = generate(&band, &ki, &Cuda);
        assert!(k.source.contains("(in0[i] & in1[i])"), "the kernel still lowers");
        assert!(contract(&band, &ki, &k, "cuda").is_none(), "but no contract");
        assert!(matches!(
            derive_pattern(&band),
            Err(PatternError::NoFkcName { ref op }) if op == "BitAnd"
        ));
        let land = OpDef::elementwise(
            "land",
            2,
            &[ElementKind::U8],
            input(0).binary(BinaryOp::LogicalAnd, input(1)),
        );
        let ku = key_dtype(ElementKind::U8, 3);
        let kl = generate(&land, &ku, &Cuda);
        assert!(kl.source.contains("!= 0 &&"), "the kernel still lowers");
        assert!(contract(&land, &ku, &kl, "cuda").is_none(), "but no contract");
    }

    #[test]
    fn uniform_int_add_contracts_carry_the_audited_dtype() {
        // Increment 0c: uniform-U8/S8 COMPUTE is audited, so an infix Add at
        // U8/S8 emits a real contract — dtypes carry the FKC §5 spellings
        // (U8; S8 spells I8), correctly_rounded means exact WRAPPING
        // semantics, and count_unit stays elements (int cells never
        // vectorize — no int vector/packed path exists).
        let addu = OpDef::elementwise("add", 2, &[ElementKind::U8], input(0) + input(1));
        let ku = key_dtype(ElementKind::U8, 3);
        let k = generate(&addu, &ku, &Cuda);
        let c = contract(&addu, &ku, &k, "cuda").unwrap();
        assert!(c.contains("op_kind: Add"));
        assert!(c.contains("dtypes: [U8]"));
        assert!(c.contains("mode: correctly_rounded"));
        assert!(c.contains("count_unit: elements"));
        // 2 u8 reads + 1 u8 write.
        assert!(c.contains("bytes_per_elem: 3"));
        let adds = OpDef::elementwise("add", 2, &[ElementKind::S8], input(0) + input(1));
        let ks = key_dtype(ElementKind::S8, 3);
        let k8 = generate(&adds, &ks, &Cuda);
        let c8 = contract(&adds, &ks, &k8, "cuda").unwrap();
        assert!(c8.contains("dtypes: [I8]"), "S8 spells I8 on the FKC wire");
        assert!(c8.contains("count_unit: elements"));
    }

    #[test]
    fn cmp_predicates_rate_zero_ulp() {
        use crate::ir::BinaryOp;
        // The compare itself is exact (0 ulp): a pure-cmp body is
        // correctly_rounded…
        for op in [
            BinaryOp::CmpEq,
            BinaryOp::CmpNe,
            BinaryOp::CmpLt,
            BinaryOp::CmpLe,
            BinaryOp::CmpGt,
            BinaryOp::CmpGe,
        ] {
            assert_eq!(
                precision_of(&input(0).binary(op, input(1)).0),
                ("correctly_rounded", Some(0)),
                "{op:?}"
            );
        }
        // …and a compare of an APPROXIMATE subexpression adds 0 of its own —
        // the body ulp is exactly the subexpression's tier (exp -> 2), the
        // pinned increment-0b modeling decision.
        let e = input(0).exp().binary(BinaryOp::CmpGt, input(1));
        assert_eq!(precision_of(&e.0), ("approximate", Some(2)));
    }

    fn pred_key() -> StructureKey {
        // f32 inputs, u8 mask output — the elementwise_pred caller key shape.
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::U8, 256);
        structure_key(OpCategory::BinaryElementwise, &[a, a, o], ArchSku::Sm89)
    }

    #[test]
    fn cmp_u8_contract_returns_fixed_u8_and_forbids_in_place() {
        use crate::ir::BinaryOp;
        let op = OpDef::elementwise_pred(
            "cmp_lt",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::CmpLt, input(1)),
        );
        let key = pred_key();
        let kernel = generate(&op, &key, &Cuda);
        let c = contract(&op, &key, &kernel, "cuda").unwrap();
        // Primitive identity: the DISPATCH OpKind spelling — the exact string
        // Fuel's lower_op_kind table accepts (`op_kind: Lt` would typed-reject
        // and fail the whole bundle import).
        assert!(c.contains("op_kind: LessElementwise"), "{c}");
        assert!(!c.contains("op_kind: Lt"), "{c}");
        // The return dtype is HONEST — the §5.1 constant rule Fuel's own
        // compare contracts use, never the input passthrough.
        assert!(c.contains("dtype_rule: fixed(U8)"));
        assert!(!c.contains("dtype_rule: same_as_input(0)"));
        // The ImplId dtype channel stays the key (input) dtype.
        assert!(c.contains("dtypes: [F32]"));
        // A 1-byte store can't alias a 4-byte input buffer.
        assert!(c.contains("in_place: forbidden"));
        // Scalar path (no packed u8 store) => n counts elements…
        assert!(c.contains("count_unit: elements"));
        // …and the traffic estimate is 2 f32 reads + 1 u8 write = 9 B/elem.
        assert!(c.contains("bytes_per_elem: 9"));
        // The predicate is exact.
        assert!(c.contains("mode: correctly_rounded"));
        assert!(c.contains("determinism: bitwise"));
    }

    #[test]
    fn float_mask_toplevel_cmp_has_no_contract() {
        use crate::ir::BinaryOp;
        // A top-level cmp with out_dtype = None stores 1.0f/0.0f in the KEY
        // dtype — not Fuel's "comparison → U8 mask" op. The kernel generates,
        // the pattern even derives (the vocabulary exists), but the contract is
        // withheld: advertising `op_kind: Gt` for a 4-byte-store kernel would
        // bind where Fuel expects a 1-byte mask.
        let op = OpDef::elementwise(
            "gt_mask",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::CmpGt, input(1)),
        );
        let key = key_for(3, OpCategory::BinaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        assert!(kernel.source.contains("? 1.0f : 0.0f"), "the kernel still lowers");
        assert!(contract(&op, &key, &kernel, "cuda").is_none(), "but no contract");
        assert!(derive_pattern(&op).is_ok(), "vocabulary exists; the gate is honesty");
    }

    #[test]
    fn nested_cmp_fusion_contract_is_withheld() {
        use crate::ir::BinaryOp;
        // relu-backward mask-multiply: dy * (x > z). The kernel is correct and
        // still generates — but the fused PATTERN would encode Gt as a direct
        // operand of Mul, an edge no constructible Fuel graph has (Fuel's
        // compare builders pin U8 output and its binary ops assert dtype
        // equality, so real graphs interpose Cast(U8→float); Cast is outside
        // the §4.1 pattern grammar and the §4.3 see-through set). Advertising
        // it would register a matcher that can never fire on the graphs it
        // means. Withheld until Cast joins the pattern vocabulary.
        let op = OpDef::elementwise(
            "relu_bw",
            3,
            &[ElementKind::F32],
            input(0) * input(1).binary(BinaryOp::CmpGt, input(2)),
        );
        let key = key_for(4, OpCategory::TernaryElementwise);
        let kernel = generate(&op, &key, &Cuda);
        assert!(kernel.source.contains("? 1.0f : 0.0f"), "the kernel still lowers");
        assert!(
            contract(&op, &key, &kernel, "cuda").is_none(),
            "nested-cmp fused contract is withheld (missing-Cast pattern gap)"
        );
        assert!(derive_pattern(&op).is_ok(), "vocabulary exists; the gate is honesty");
    }

    #[test]
    fn coord_bodies_have_no_contract_until_the_iota_bridge_lands() {
        use crate::ir::{coord, BinaryOp};
        use crate::pattern::PatternError;
        // OpTag::Iota exists (0.10.2), but the emitted pattern grammar cannot
        // carry its axis attribute and the Iota↔Coord correspondence is
        // unreconciled — so a Coord body derives NO pattern
        // (CoordUnsupported) and therefore NO contract (importable-honest:
        // an axis-less `op: Iota` would advertise a wrong-bind matcher). The
        // kernel itself still generates via the strided schedule.
        let triu = OpDef::elementwise(
            "triu_mask",
            1,
            &[ElementKind::F32],
            input(0) * coord(1).binary(BinaryOp::CmpGe, coord(0) + crate::ir::konst(0.0)),
        );
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a], ArchSku::Sm89);
        let k = generate(&triu, &key, &Cuda);
        assert!(k.source.contains("(float)c1"), "the kernel still lowers");
        assert!(contract(&triu, &key, &k, "cuda").is_none(), "but no contract");
        assert!(matches!(
            derive_pattern(&triu),
            Err(PatternError::CoordUnsupported { .. })
        ));
    }

    #[test]
    fn vocab_ops_have_no_contract_until_fuel_names_them() {
        use crate::ir::{BinaryOp, UnaryOp};
        use crate::pattern::PatternError;
        // Fuel's §4.1/OpTag vocabulary doesn't name the increment-0a fns: no
        // pattern derives (NoFkcName), so no contract is emitted — the honest
        // miss. The kernel itself still generates (lowering is unaffected).
        let erfc = OpDef::elementwise("erfc", 1, &[ElementKind::F32], input(0).unary(UnaryOp::Erfc));
        let ukey = key_for(2, OpCategory::UnaryElementwise);
        let uk = generate(&erfc, &ukey, &Cuda);
        assert!(uk.source.contains("erfcf("), "the kernel still lowers");
        assert!(contract(&erfc, &ukey, &uk, "cuda").is_none(), "but no contract");
        assert!(matches!(
            crate::derive_pattern(&erfc),
            Err(PatternError::NoFkcName { ref op }) if op == "Erfc"
        ));
        let at2 = OpDef::elementwise(
            "atan2",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::Atan2, input(1)),
        );
        let bkey = key_for(3, OpCategory::BinaryElementwise);
        let bk = generate(&at2, &bkey, &Cuda);
        assert!(contract(&at2, &bkey, &bk, "cuda").is_none());
        assert!(matches!(
            crate::derive_pattern(&at2),
            Err(PatternError::NoFkcName { ref op }) if op == "Atan2"
        ));
    }
}
