//! Language-agnostic kernel plan — the schedule decision.
//!
//! [`build_plan`] turns an [`OpDef`] + a [`StructureKey`] cell into a neutral
//! [`KernelPlan`]: *what* to compute (the op body + dtype) and the *schedule*
//! (vectorized vs scalar) to compute it with. A [`crate::backend::Backend`]
//! lowers the plan to a concrete language. Choosing the schedule here, not in
//! the backend, keeps the decision shared across every backend.

use crate::ir::{Access, OpDef, ReduceOp, ReduceStage, ScalarExpr};
use baracuda_kernels_types::{
    AxisMask, Contiguity, ElementKind, OperandKey, StructureKey, VecWidth, MAX_OPERANDS,
};

/// How the kernel iterates the data — the backend-neutral schedule.
///
/// `#[non_exhaustive]`: strided / broadcast / reduction schedules are the
/// growth path; backends match what they support and reject the rest.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Schedule {
    /// Linear access, `width` elements at a time (e.g. `float4` for width 4).
    Vectorized {
        /// Vector width in elements.
        width: u32,
    },
    /// Linear access, one element at a time. Contiguous operands only.
    Scalar,
    /// Per-element coordinate unravel over the cell's iteration rank — for
    /// non-contiguous operands (strided / broadcast). The emitter specializes
    /// it per cell: the rank is unrolled, broadcast axes drop their offset
    /// terms, and a fully-broadcast operand is hoisted to a loop-invariant load.
    Strided,
    /// One thread per output element; sequential fold over the contiguous trailing
    /// axis. The v1 reduction schedule — block/warp-parallel reduction is the perf
    /// follow-up.
    Reduction {
        /// The associative combine to apply along the axis.
        op: ReduceOp,
        /// Reduce-axis geometry (design-doc predicate #9). All classes lower to
        /// the sequential fold in v2; the class reserves the dispatch token for a
        /// future block-parallel outer-axis kernel. The full axis mask rides on
        /// [`KernelPlan::key`]'s `reduce_axes` (this enum is `Copy`).
        class: ReduceAxisClass,
        /// Keep reduced axes as size-1 (broadcast-back) vs. collapse them.
        keepdim: bool,
    },
    /// Fused reduce → broadcast → elementwise, one block per output row (warp-
    /// shuffle + shared-memory tree reduce): `n_stages` reductions then a full-width
    /// epilogue. The stages + epilogue ride on [`KernelPlan::access`] (this enum is
    /// `Copy`, so a `Vec` can't live here). `block` selects the block-parallel tree
    /// (v1 always `true`) over a sequential fallback.
    RowReduce {
        /// Number of reduction stages (each produces a `Reduced(i)`).
        n_stages: u8,
        /// Block-parallel tree reduce (`true`, v1) vs the sequential fallback.
        block: bool,
    },
    /// Batched contraction (`out[m,n] = epi(Σ_k lhs[m,k]·rhs[k,n])`) — the
    /// terminal ORDER-3 schedule. v1: the skinny SIMT kernel (thread per output
    /// column, M-row register accumulators, coalesced K-streaming of the rhs) —
    /// the decode / flat-GEMM long-tail cell; tiled/MMA schedules join as
    /// bench-gated variants. Axes/accum/epilogue ride on [`KernelPlan::access`].
    Contraction,
}

/// Reduce-axis geometry (design-doc predicate #9). All classes lower to the same
/// sequential fold in v2; the class reserves the dispatch token so a later
/// block-parallel outer-axis kernel is an additive drop-in, not a re-key.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReduceAxisClass {
    /// Empty mask (legacy last-axis sentinel) or a single contiguous trailing
    /// axis — today's sequential fast path.
    InnerContig,
    /// A single outermost (axis 0) reduced axis.
    Outer,
    /// A single interior reduced axis (not axis 0, not a contiguous trailing axis).
    Middle,
    /// Two or more reduced axes.
    Multi,
}

/// A language-agnostic description of the kernel to emit.
#[derive(Clone, Debug)]
pub struct KernelPlan<'a> {
    /// Op name (used to form the generated symbol).
    pub op_name: &'a str,
    /// Number of input operands.
    pub n_inputs: u8,
    /// Element dtype.
    pub dtype: ElementKind,
    /// Output element dtype — the *resolved* [`crate::ir::OpDef::out_dtype`]
    /// (`out_dtype.unwrap_or(key.dtype)`), so `out_dtype == dtype` for every
    /// uniform-dtype op and `U8` only for a validated predicate op. Backends
    /// read this for the output pointer type + store conversion.
    pub out_dtype: ElementKind,
    /// The chosen schedule.
    pub schedule: Schedule,
    /// The structure cell this plan targets. Backends read structural detail
    /// from it (rank, per-operand broadcast mask, flip) for strided lowering,
    /// and its token for traceability.
    pub key: &'a StructureKey,
    /// Output `= body`, evaluated per coordinate. For [`Schedule::RowReduce`] this
    /// is the epilogue (`OpDef::row_reduce` sets `body = epilogue`). For a
    /// multi-output op this is output 0; the further outputs are in
    /// [`Self::extra_out_bodies`].
    pub body: &'a ScalarExpr,
    /// Number of outputs the kernel writes (`1` for every single-output op —
    /// `OpDef::n_outputs`). `> 1` only for a validated multi-output
    /// `Access::Elementwise` op (increment 1); the emitter then writes the last
    /// `n_outputs` operands of the key from one shared body-DAG.
    pub n_outputs: u8,
    /// Additional output bodies (`OpDef::extra_out_bodies`) — **empty for every
    /// single-output op**, so the single-output emitters are byte-identical. The
    /// multi-output emitter interns `[body] ++ extra_out_bodies` into one
    /// [`crate::ir::ExprDag`] for cross-body CSE.
    pub extra_out_bodies: &'a [ScalarExpr],
    /// The op's access pattern — the [`Schedule::RowReduce`] emitter reads its
    /// `stages` (and epilogue) off here, since `Schedule` is `Copy` and can't carry
    /// the stage `Vec`.
    pub access: &'a Access,
}

impl KernelPlan<'_> {
    /// All output bodies in order — `body` (output 0) then `extra_out_bodies`.
    /// One element for a single-output plan; the multi-output emitter interns
    /// these together for cross-body CSE, and the backstop walks gate every one.
    #[must_use]
    pub fn output_bodies(&self) -> Vec<&ScalarExpr> {
        std::iter::once(self.body)
            .chain(self.extra_out_bodies.iter())
            .collect()
    }
}

/// Choose the schedule for `op` at structure cell `key` and return a neutral
/// [`KernelPlan`].
///
/// Elementwise ops vectorize when every operand is `Contig` + `V4`, scalar/strided
/// otherwise. A reduction op maps straight to [`Schedule::Reduction`] (the fold is
/// the schedule). (Whether a backend can lower the chosen dtype is the backend's
/// call, not this function's.)
#[must_use]
pub fn build_plan<'a>(op: &'a OpDef, key: &'a StructureKey) -> KernelPlan<'a> {
    assert_valid_out_dtype(op);
    assert_valid_multi_output(op, key);
    assert_no_half_nextafter(op, key.dtype);
    assert_int_op_admissibility(op, key.dtype);
    assert_coord_admissibility(op, key);
    assert_valid_reduction_post(op);
    let schedule = match op.access {
        Access::Reduction {
            op: rop,
            axes,
            keepdim,
            post: _,
        } => {
            // `class`/`keepdim` are consumed by the emitter in step 3; today all
            // classes lower to the same sequential fold, so the legacy last-axis
            // path (empty mask ⇒ `InnerContig`) stays byte-identical.
            let input0_contig =
                key.n_operands > 0 && key.operands[0].contig == Contiguity::Contig;
            Schedule::Reduction {
                op: rop,
                class: classify_reduce_axes(axes, key.rank, input0_contig),
                keepdim,
            }
        }
        // `ref` borrows (the Vec/expr can't move out of the borrowed `op.access`);
        // v1 always routes RowReduce to the block-parallel tree reduce.
        Access::RowReduce {
            ref stages,
            ref epilogue,
        } => {
            validate_row_reduce(stages, epilogue, op.n_inputs, key);
            Schedule::RowReduce {
                n_stages: stages.len() as u8,
                block: true,
            }
        }
        Access::Contraction { ref axes, ref epilogue, .. } => {
            // v1 admissibility: the canonical rank-2 dense matmul cell, keyed
            // with contraction facts, and an epilogue over the K-sum only.
            assert_eq!(
                (axes.lhs.as_slice(), axes.rhs.as_slice()),
                (
                    crate::ir::ContractionAxes::matmul().lhs.as_slice(),
                    crate::ir::ContractionAxes::matmul().rhs.as_slice()
                ),
                "contraction v1: canonical rank-2 matmul axis roles only"
            );
            assert!(
                key.contraction.is_some(),
                "contraction cell must carry ContractionKey facts (rank-2 dense \
                 row-major [M,K]x[K,N]->[M,N]); got token {}",
                key.to_token()
            );
            assert!(
                epilogue_reads_only_reduced0(epilogue),
                "contraction v1: epilogue over Reduced(0) only (fused bias inputs \
                 are a follow-up)"
            );
            Schedule::Contraction
        }
        Access::Elementwise => {
            let n = key.n_operands as usize;
            let all_contig =
                n > 0 && (0..n).all(|k| key.operands[k].contig == Contiguity::Contig);
            // The kernel vectorizes at the *narrowest* width every operand supports.
            let min_width = (0..n)
                .map(|k| vec_width_elems(key.operands[k].vec_width))
                .min()
                .unwrap_or(1);
            if expr_contains_coord(&op.body) {
                // A Coord body always takes the STRIDED schedule (increment
                // 0d): the strided emitter is the only one that materializes
                // the per-axis output coordinates `c{d}` a Coord leaf reads —
                // the Vectorized/Scalar emitters iterate a bare linear index.
                // Contiguous cells are still CORRECT under strided emission
                // (the unravel + stride dot-product reproduces the linear
                // offset exactly), just unoptimized — a coordinate-aware
                // vectorized variant is a follow-up. Pinned by the
                // vectorize-never test in `cuda`.
                Schedule::Strided
            } else if !all_contig {
                Schedule::Strided
            } else if min_width >= 2 && op.out_dtype.is_none() {
                // A hetero-output (u8 predicate) kernel takes the SCALAR path
                // in v1 — never Vectorized: the vector/packed emitters load and
                // STORE one vector type, and a u8-mask output has no packed
                // store (a contiguous u8 output even keys V8, which would
                // otherwise widen `min_width` past the inputs'). Pinned by the
                // packed-fallback golden in `cuda`.
                Schedule::Vectorized { width: min_width }
            } else {
                Schedule::Scalar
            }
        }
    };
    KernelPlan {
        op_name: &op.name,
        n_inputs: op.n_inputs,
        dtype: key.dtype,
        out_dtype: op.out_dtype.unwrap_or(key.dtype),
        schedule,
        key,
        body: &op.body,
        n_outputs: op.n_outputs(),
        extra_out_bodies: &op.extra_out_bodies,
        access: &op.access,
    }
}

/// Classify a reduction's axis geometry (design-doc predicate #9) from its
/// reduced-axis mask + the input's contiguity. An empty mask is the legacy
/// last-axis sentinel; a single trailing axis over a contiguous input is the
/// existing fast path. All classes lower to the same sequential fold in v2 — the
/// class only reserves the dispatch token for a future block-parallel kernel.
fn classify_reduce_axes(axes: AxisMask, rank: u8, input0_contig: bool) -> ReduceAxisClass {
    match axes.0.count_ones() {
        0 => ReduceAxisClass::InnerContig, // empty ⇒ legacy last-axis default
        1 => {
            let d = axes.0.trailing_zeros() as u8;
            if rank > 0 && d == rank - 1 && input0_contig {
                ReduceAxisClass::InnerContig // contiguous trailing axis = fast path
            } else if d == 0 {
                ReduceAxisClass::Outer
            } else {
                ReduceAxisClass::Middle
            }
        }
        _ => ReduceAxisClass::Multi,
    }
}

#[cfg(test)]
mod reduce_class_tests {
    use super::*;

    #[test]
    fn classify_axis_geometry() {
        // empty mask ⇒ legacy last-axis fast path
        assert_eq!(
            classify_reduce_axes(AxisMask::EMPTY, 3, true),
            ReduceAxisClass::InnerContig
        );
        // trailing axis (rank-1) over a contiguous input ⇒ fast path
        assert_eq!(
            classify_reduce_axes(AxisMask(0b100), 3, true),
            ReduceAxisClass::InnerContig
        );
        // outermost axis 0
        assert_eq!(
            classify_reduce_axes(AxisMask(0b001), 3, true),
            ReduceAxisClass::Outer
        );
        // interior axis 1
        assert_eq!(
            classify_reduce_axes(AxisMask(0b010), 3, true),
            ReduceAxisClass::Middle
        );
        // two reduced axes
        assert_eq!(
            classify_reduce_axes(AxisMask(0b011), 3, true),
            ReduceAxisClass::Multi
        );
        // trailing axis but a STRIDED input ⇒ no longer the contiguous fast path
        assert_eq!(
            classify_reduce_axes(AxisMask(0b100), 3, false),
            ReduceAxisClass::Middle
        );
    }
}

/// `true` if `e` references no leaf other than `Reduced(0)` and constants — the
/// contraction-v1 epilogue admissibility (no `Input`/`Param`, no other stage).
fn epilogue_reads_only_reduced0(e: &crate::ir::ScalarExpr) -> bool {
    use crate::ir::ScalarExpr as E;
    match e {
        E::Reduced(0) | E::Const(_) => true,
        // Coord rejects here too: a contraction epilogue iterates the (m, n)
        // output space, not an elementwise cell's — Coord's v1 semantics are
        // Elementwise-only (`assert_coord_admissibility` fires first with the
        // targeted message; this arm keeps the predicate honest regardless).
        E::Input(_) | E::Param(_) | E::Reduced(_) | E::Coord(_) => false,
        E::Unary(_, x) => epilogue_reads_only_reduced0(x),
        E::Add(a, b) | E::Sub(a, b) | E::Mul(a, b) | E::Div(a, b) | E::Binary(_, a, b) => {
            epilogue_reads_only_reduced0(a) && epilogue_reads_only_reduced0(b)
        }
    }
}

/// The access role of a [`Access::RowReduce`] input operand, from its layout.
///
/// The three roles are the three broadcast geometries a row-reduce operand can
/// take, and they map one-to-one onto the emitter's load index (`last` = the
/// feature/reduced axis, `rank-1`):
///
/// | role | broadcast mask | varies along | index |
/// |---|---|---|---|
/// | [`RowStreamed`](RrRole::RowStreamed) | empty | row **and** feature | `in_i[base+j]` |
/// | [`ColBroadcast`](RrRole::ColBroadcast) | every outer axis, **not** `last` | feature only | `in_i[j]` |
/// | [`RowScalar`](RrRole::RowScalar) | `last` set, **no** outer axis | row only | `in_i[row]` (hoisted) |
///
/// `RowScalar` is the exact inverse of `ColBroadcast`: a `ColBroadcast` weight is
/// constant *across rows* and varies *along the feature axis*; a `RowScalar`
/// (a saved per-row statistic — μ, rstd, lse) is constant *along the feature
/// axis* and varies *across rows*.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum RrRole {
    /// A reduced/streamed tensor ([n_out, k], full / empty bcast) — `in_i[base+j]`.
    /// Input 0 (`x`) is always this; a second streamed input (softmax-bw's `dy`
    /// beside `y`) is now also legal (the increment-2 lift).
    RowStreamed,
    /// A per-column `[k]` weight/bias, broadcast over the row axis — `in_i[j]`.
    ColBroadcast,
    /// A per-row scalar (saved stat), broadcast over the feature axis —
    /// `in_i[row]`, loaded **once** per row (hoisted outside the feature loop).
    RowScalar,
}

/// Classify a RowReduce input by its broadcast mask, given the feature axis
/// `last` (`rank-1`). **Total / non-panicking** — the emitter calls this for the
/// load index and must never crash; all *rejection* of malformed shapes lives in
/// [`validate_row_reduce`] (one source of truth, no drift). The three-way split:
///
/// - empty bcast ⇒ [`RrRole::RowStreamed`] (the reduced/streamed tensor);
/// - `last` axis broadcast ⇒ [`RrRole::RowScalar`] (constant along the feature
///   axis ⇒ one value per row). An all-broadcast operand also lands here (last is
///   set) and is then *rejected* by validate's outer-axis-clear check — a true
///   scalar is a `Const`, not an operand;
/// - otherwise (some axis broadcast, but not `last`) ⇒ [`RrRole::ColBroadcast`]
///   (varies along the feature axis ⇒ a per-column weight/bias).
///
/// Which *specific* broadcast masks are legal for each role (every outer axis for
/// a column, no outer axis for a row-scalar, contiguity for a streamed input) is
/// validate's job, so the classification here is deliberately coarse and total.
pub(crate) fn rr_role(o: OperandKey, last: u8) -> RrRole {
    if o.bcast.is_empty() {
        RrRole::RowStreamed
    } else if o.bcast.is_set(last) {
        RrRole::RowScalar
    } else {
        RrRole::ColBroadcast
    }
}

/// Validate a [`Access::RowReduce`] op at build time (AOT — RowReduce never crosses
/// the JIT trust boundary, so a panic here is an author-error backstop, like
/// `emit_reduction`'s asserts). Catches expression errors (a `Reduced(s)` not yet
/// produced, out-of-range `Input`, a `Param`, a non-finite `Const`, a column input
/// inside a reduction stage) **and** operand-layout errors that would mis-index or
/// read out of bounds. Input 0 (`x`) must be row-streamed + contiguous. Each other
/// input takes one of three [`RrRole`]s, classified by broadcast mask ([`rr_role`])
/// and validated for the load index the emitter uses:
///
/// - **`RowStreamed`** (empty bcast, `in_i[base+j]`) — a second reduced/streamed
///   tensor (softmax-bw's `dy` beside `y`); must be contiguous. This is the
///   increment-2 lift of the former "inputs>0 must be column-broadcast" guard.
/// - **`ColBroadcast`** (every outer axis bcast, `last` not, `in_i[j]`) — a
///   per-column `[k]` weight/bias; not reversed; rank ≥ 2.
/// - **`RowScalar`** (`last` bcast, no outer axis, `in_i[row]`) — a saved per-row
///   scalar (μ, rstd, lse), the inverse of `ColBroadcast`; not reversed; rank ≥ 2.
///   An all-broadcast operand (a true scalar → `Const`) is rejected here.
///
/// The output is full-width contiguous.
///
/// v1 assumes a **uniform operand dtype** (the structure key carries one dtype) — a
/// mixed-dtype LayerNorm (fp16 `x` + fp32 weight) is unrepresentable here and must
/// be refused upstream by the caller.
///
/// **Caller pre-conditions this cannot check** (the structure key carries broadcast
/// masks but **no numeric extents** — specialize on structure, not extents), each at
/// the same trust level as the `n_out`/`k` launch args, asserted by the layer still
/// holding the `OperandDesc` extents (an AOT op author, or the live seam caller once
/// `region_to_op` wires RowReduce):
/// - a `ColBroadcast` weight's feature-axis extent must equal `x`'s `k` (else the
///   emitter reads `in_i[j]` past its buffer — a confirmed on-device OOB);
/// - a second `RowStreamed` input must be full `[n_out,k]` dense (else `in_i[base+j]`
///   over-reads — the identical trust as input 0; a bare rank-1 `[k]` has the same
///   key as a full `[n_out,k]`, so this can no longer be a key-visible rejection);
/// - a `RowScalar` must be `[n_out]`-shaped with a dense outer layout so its linear
///   offset equals `row` (else `in_i[row]` mis-indexes).
///
/// Nextafter is declared f32/f64-only at the IR level (its half lowering via
/// promote-to-f32 would step the f32 lattice — ~2^13 steps inside one half
/// step, so the demote rounds straight back: a silently wrong no-op). The
/// CUDA emitter's `cuda_binary` panic only guards the elementwise path; the
/// reduction pre-body, RowReduce stages/epilogue, and contraction epilogues
/// lower through accumulator-width helpers that never pass through it. This
/// plan-level walk covers EVERY Access arm, so no lowering path — present or
/// future backend — can bypass the honest miss.
/// Increment-1 **multi-output** admissibility gate — runs at the top of
/// [`build_plan`] (the house pattern), with an independent emitter backstop in
/// [`crate::cuda::Cuda::lower`]. A single-output op (`extra_out_bodies` empty,
/// every pre-increment-1 op) returns immediately, so nothing about the
/// established path changes and emission stays byte-identical. For a multi-output
/// op (built only by [`OpDef::elementwise_multi`]) the v1 rules, all honest
/// AOT panics:
///
/// 1. **Access**: `Access::Elementwise` only. Multi-output is meaningful only for
///    an elementwise map; the reduction-class arms reject `extra_out_bodies`
///    (a fused reduction/contraction stores one accumulator, not N bodies).
/// 2. **Uniform dtype**: `out_dtype == None`. All outputs share the key dtype;
///    hetero multi-output (a u8 mask beside a float grad — dropout fw) is the
///    follow-up.
/// 3. **Operand budget**: `1 ≤ n_outputs` and `n_inputs + n_outputs ≤
///    MAX_OPERANDS`, and the key must carry exactly `n_inputs + n_outputs`
///    operands (inputs then outputs) — the caller's `OperandDesc` list.
/// 4. **Body legality** (every output body): `Input(i) < n_inputs`; NO `Reduced`
///    (there is no reduction here); NO `Coord` (Elementwise-map only in v1 — a
///    multi-output coordinate kernel is deferred, same rejection the Coord gate
///    would give); `Const` finite. `Param` f32-only is enforced by the emitter's
///    param assert over all output bodies (same rule as the single-output path).
///    (This gate rejecting `Coord` here is also what lets the downstream
///    `assert_coord_admissibility` keep its `Access::Elementwise => {}` arm — a
///    multi-output `Coord` never reaches it.)
/// 5. **Output operands**: each of the last `n_outputs` operands must be
///    **non-broadcast** (a stride-0 output would alias its own writes across
///    iteration coordinates — a write race, and not the full output shape) and
///    **not flipped**. This is the key-visible slice of "outputs must not alias
///    and must match the output shape".
///
/// **Caller preconditions the key cannot see** (documented honestly, the same
/// trust level as the RowReduce `n_out`/`k` and `Coord` extent preconditions):
/// true buffer aliasing — an output buffer pointer equal to an input's (in-place)
/// — and exact per-output extent agreement are abstracted away by the structure
/// key (buffer identity and numeric extents are not keyed). The AOT op author (or
/// a future seam caller, once a multi-output region envelope exists) must ensure
/// distinct, correctly-shaped output buffers; v1 defers in-place entirely.
fn assert_valid_multi_output(op: &OpDef, key: &StructureKey) {
    if op.extra_out_bodies.is_empty() {
        return; // single-output — the established path, unchanged.
    }
    let name = &op.name;
    assert!(
        matches!(op.access, Access::Elementwise),
        "OpDef '{name}': multi-output is Access::Elementwise-only in v1 — a fused \
         reduction/contraction stores a single accumulator, not N bodies; \
         extra_out_bodies is rejected on a {}-class op",
        access_tag(&op.access)
    );
    assert!(
        op.out_dtype.is_none(),
        "OpDef '{name}': multi-output requires a uniform output dtype (out_dtype \
         None) in v1 — a hetero multi-output (u8 mask beside a float grad) is the \
         follow-up"
    );
    let n_inputs = op.n_inputs as usize;
    let n_outputs = op.n_outputs() as usize;
    assert!(
        n_inputs + n_outputs <= MAX_OPERANDS,
        "OpDef '{name}': n_inputs ({n_inputs}) + n_outputs ({n_outputs}) exceeds \
         MAX_OPERANDS ({MAX_OPERANDS})"
    );
    assert!(
        key.n_operands as usize == n_inputs + n_outputs,
        "OpDef '{name}': multi-output key must carry n_inputs+n_outputs operands \
         (inputs then outputs) = {}, got {} — the caller's OperandDesc list is a \
         shape mismatch",
        n_inputs + n_outputs,
        key.n_operands
    );

    // Body legality — walk every output body.
    fn check_body(e: &ScalarExpr, n_inputs: u8, name: &str) {
        match e {
            ScalarExpr::Input(i) => assert!(
                *i < n_inputs,
                "OpDef '{name}': multi-output body Input({i}) >= n_inputs {n_inputs}"
            ),
            ScalarExpr::Const(v) => assert!(
                v.is_finite(),
                "OpDef '{name}': multi-output body Const must be finite, got {v}"
            ),
            ScalarExpr::Param(_) => {}
            ScalarExpr::Reduced(s) => panic!(
                "OpDef '{name}': multi-output body must not read Reduced({s}) — there \
                 is no reduction in an Elementwise multi-output op"
            ),
            ScalarExpr::Coord(d) => panic!(
                "OpDef '{name}': multi-output body must not read Coord({d}) — v1 is \
                 elementwise-map only (a multi-output coordinate kernel is deferred)"
            ),
            ScalarExpr::Unary(_, x) => check_body(x, n_inputs, name),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check_body(a, n_inputs, name);
                check_body(b, n_inputs, name);
            }
        }
    }
    for e in op.output_bodies() {
        check_body(e, op.n_inputs, name);
    }

    // Output operands: the last n_outputs entries. A writable output must not be
    // broadcast (stride-0 → aliased writes) or flipped.
    for j in 0..n_outputs {
        let o = key.operands[n_inputs + j];
        assert!(
            o.bcast.is_empty(),
            "OpDef '{name}': multi-output output {j} is broadcast (mask {:#04x}) — a \
             stride-0 output aliases its own writes across iteration coordinates \
             (a write race) and is not the full output shape",
            o.bcast.0
        );
        assert!(
            !o.flipped,
            "OpDef '{name}': multi-output output {j} is flipped (negative stride) — \
             a reversed output view is deferred"
        );
    }
}

/// Short tag for an [`Access`] variant, for the multi-output rejection message.
fn access_tag(a: &Access) -> &'static str {
    match a {
        Access::Elementwise => "Elementwise",
        Access::Reduction { .. } => "Reduction",
        Access::RowReduce { .. } => "RowReduce",
        Access::Contraction { .. } => "Contraction",
    }
}

fn assert_no_half_nextafter(op: &OpDef, dtype: ElementKind) {
    use crate::ir::BinaryOp;
    if !matches!(dtype, ElementKind::F16 | ElementKind::Bf16) {
        return;
    }
    fn walk(e: &ScalarExpr) -> bool {
        match e {
            ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Param(_)
            | ScalarExpr::Reduced(_)
            | ScalarExpr::Coord(_) => false,
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b) => walk(a) || walk(b),
            ScalarExpr::Unary(_, a) => walk(a),
            ScalarExpr::Binary(bop, a, b) => {
                matches!(bop, BinaryOp::Nextafter) || walk(a) || walk(b)
            }
        }
    }
    let mut exprs: Vec<&ScalarExpr> = vec![&op.body];
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            exprs.extend(stages.iter().map(|s| &s.pre));
            exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => exprs.push(epilogue),
        // The reduction post-expr (0e) lowers through the accumulator-width
        // spellers too, so the honest-miss walk must cover it (body is already in).
        Access::Reduction { post, .. } => exprs.push(post),
        // Review-caught gate asymmetry (increment 1): a multi-output op's EXTRA
        // output bodies must be walked too — else a half `Nextafter` hidden in an
        // extra body bypasses this honest-miss gate. Non-elementwise multi-output
        // is already rejected by `assert_valid_multi_output` (runs first), so this
        // arm is the ONLY place `extra_out_bodies` is legal; empty for every
        // single-output op (byte-identical).
        Access::Elementwise => exprs.extend(op.extra_out_bodies.iter()),
    }
    for e in exprs {
        assert!(
            !walk(e),
            "Nextafter has no half-precision lowering (IR contract: f32/f64 only; \
             the promote-to-f32 path silently no-ops after the demote) — op '{}' \
             at {dtype:?} must miss honestly",
            op.name
        );
    }
}

/// The integer compute dtypes of increment 0c: `I32`/`I64` (already lowering
/// pre-0c) plus the newly-promoted `S8` (FKC `I8`) and `U8`.
pub(crate) fn is_int_dtype(dt: ElementKind) -> bool {
    matches!(
        dt,
        ElementKind::I32 | ElementKind::I64 | ElementKind::S8 | ElementKind::U8
    )
}

/// Increment-0c op × dtype admissibility gate — the plan-level enforcement of
/// the table in [`crate::ir::BinaryOp`]'s docs. Runs at the TOP of
/// [`build_plan`] and walks the body of EVERY Access arm (the 0a lesson: the
/// emitter backstops alone were bypassed by the reduction-class lowering
/// paths), so no lowering path — present or future backend — can bypass it.
///
/// Two directions, both validate-reject (honest miss, never silent):
///
/// 1. **Int-only ops** (`BitAnd`/`BitOr`/`BitXor`/`Shl`/`Shr`/`Logical*`) may
///    appear ONLY in an [`Access::Elementwise`] body at an int dtype (the
///    reduction/RowReduce/contraction paths lower through the FLOAT
///    accumulator spellers, which have no int arms), and the logical ops
///    narrow further to `U8` — the bespoke Bool surface (`uint8_t` only).
/// 2. **At an int dtype**, only the audited op set lowers: infix
///    `Add`/`Sub`/`Mul` (wrapping) and the int-only ops. Everything else —
///    every `UnaryOp`, infix `Div` (no bespoke int div; `/0` is device-UB),
///    the float binary fns, and the `Cmp*` predicates (bespoke cmp is
///    `_fp`-only) — rejects. `Const` rejects too: it is spelled as an f64 C
///    literal, so an int body would silently run double math (and f64 cannot
///    even represent all i64); an int-literal speller is a follow-up. `Param`
///    rejects at int for the same f32-only reason the emitter asserts.
/// 3. **8-bit composition pin (v1):** at `U8`/`S8`, EVERY operand of an
///    int-only op must be a leaf [`ScalarExpr::Input`]. Why: `Add`/`Sub`/
///    `Mul`/`BitAnd`/`BitOr`/`BitXor` compositions are congruent under
///    deferred truncation (the wrapping ring ops and the bit-local ops
///    commute with the final 8-bit store truncate), but `Shr`, shift
///    AMOUNTS, and the logical `!= 0` tests OBSERVE the un-truncated
///    promoted-`int` value — and the DAG emitter truncates a composed
///    interior only when sharing hoists it to an 8-bit tmp, so one body
///    could compute two different results depending on DAG sharing
///    (`(in0+in1)>>in2` at u8 with `(200,100,1)`: inlined `300>>1 = 150`,
///    hoisted `44>>1 = 22`). Rather than a per-position observer analysis,
///    v1 pins ALL int-op operands at 8-bit to leaves — the bespoke surface
///    has no 8-bit bitwise at all, so zero parity is lost; the dtype-aware
///    truncating speller is the follow-up that lifts this. At `I32`/`I64`
///    compositions stay legal: integer promotion never widens past the
///    compute width there, so no un-truncated wider value exists to observe.
fn assert_int_op_admissibility(op: &OpDef, dtype: ElementKind) {
    let int_dt = is_int_dtype(dtype);
    let elementwise = matches!(op.access, Access::Elementwise);
    fn walk(
        e: &ScalarExpr,
        op_name: &str,
        dtype: ElementKind,
        int_dt: bool,
        elementwise: bool,
    ) {
        match e {
            ScalarExpr::Input(_) | ScalarExpr::Reduced(_) => {}
            // Coord's own gate (`assert_coord_admissibility`, which also runs
            // at the top of build_plan) rejects EVERY int dtype — a Coord is
            // spelled as a float cast, the same double-math hazard this walk
            // polices for Const/Param — so this arm carries no second assert
            // (one source of truth for the message). It is also structurally
            // moot for rule 3: an int-only op's operands are pinned to leaf
            // Inputs at 8-bit before Coord could ever appear there.
            ScalarExpr::Coord(_) => {}
            ScalarExpr::Const(_) => assert!(
                !int_dt,
                "op '{op_name}': Const at int dtype {dtype:?} is rejected — a Const \
                 is spelled as an f64 C literal, which would silently run double \
                 math in an integer kernel (and f64 cannot represent all i64); \
                 int-literal Const spelling is a follow-up"
            ),
            ScalarExpr::Param(_) => assert!(
                !int_dt,
                "op '{op_name}': scalar params are f32-only (int dtype {dtype:?})"
            ),
            ScalarExpr::Unary(uop, x) => {
                assert!(
                    !int_dt,
                    "op '{op_name}': {uop:?} has no integer lowering — the bespoke \
                     unary elementwise surface is float-only, so int dtype {dtype:?} \
                     must miss honestly"
                );
                walk(x, op_name, dtype, int_dt, elementwise);
            }
            ScalarExpr::Div(a, b) => {
                assert!(
                    !int_dt,
                    "op '{op_name}': integer division is rejected at {dtype:?} — the \
                     bespoke elementwise surface has no int div (binary_div_fp.cu is \
                     float-only) and C `/` division by zero is device-undefined; \
                     miss honestly"
                );
                walk(a, op_name, dtype, int_dt, elementwise);
                walk(b, op_name, dtype, int_dt, elementwise);
            }
            ScalarExpr::Binary(bop, a, b) => {
                if bop.is_int_only() {
                    assert!(
                        elementwise,
                        "op '{op_name}': {bop:?} is Elementwise-only in 0c — the \
                         reduction-class paths lower through the float accumulator \
                         spellers, which have no integer arms"
                    );
                    assert!(
                        int_dt,
                        "op '{op_name}': {bop:?} is int-only (I32/I64/S8/U8) — float \
                         dtype {dtype:?} must miss honestly (the bespoke bitwise/\
                         logical kernels have no float instantiation)"
                    );
                    assert!(
                        !bop.is_logical() || dtype == ElementKind::U8,
                        "op '{op_name}': {bop:?} is U8 (Bool)-only — the bespoke \
                         binary_logical_*_bool.cu surface instantiates exactly \
                         uint8_t, so {dtype:?} must miss honestly"
                    );
                    // Rule 3 (8-bit composition pin, v1): every operand of an
                    // int-only op at U8/S8 must be a LEAF Input — a composed
                    // operand's value differs between the inlined (un-truncated
                    // promoted-int) and hoisted (8-bit tmp, truncated) spellings,
                    // so admitting it would make the result depend on DAG
                    // sharing. See the doc comment above for the full rationale.
                    if matches!(dtype, ElementKind::U8 | ElementKind::S8) {
                        for (side, operand) in [("lhs", &**a), ("rhs", &**b)] {
                            assert!(
                                matches!(operand, ScalarExpr::Input(_)),
                                "op '{op_name}': {bop:?} at {dtype:?} requires LEAF \
                                 Input operands ({side} is a composed expression) — \
                                 at 8-bit dtypes a composed operand observes the \
                                 un-truncated promoted-int value when inlined but \
                                 the truncated 8-bit value when hoisted to a shared \
                                 tmp (one body, two results); v1 pins all int-op \
                                 operands at U8/S8 to leaves. Compose at I32/I64, \
                                 or wait for the dtype-aware truncating speller"
                            );
                        }
                    }
                } else {
                    assert!(
                        !int_dt,
                        "op '{op_name}': {bop:?} has no integer lowering — the \
                         bespoke elementwise surface instantiates it for float \
                         dtypes only, so int dtype {dtype:?} must miss honestly"
                    );
                }
                walk(a, op_name, dtype, int_dt, elementwise);
                walk(b, op_name, dtype, int_dt, elementwise);
            }
            ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) | ScalarExpr::Mul(a, b) => {
                // Wrapping two's-complement at int dtypes — the audited-legal set.
                walk(a, op_name, dtype, int_dt, elementwise);
                walk(b, op_name, dtype, int_dt, elementwise);
            }
        }
    }
    let mut exprs: Vec<&ScalarExpr> = vec![&op.body];
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            exprs.extend(stages.iter().map(|s| &s.pre));
            exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => exprs.push(epilogue),
        // The reduction post-expr (0e) lowers at the accumulator dtype — a
        // Const/Param/Div/unary there hits the same int-dtype hazards, so gate it.
        Access::Reduction { post, .. } => exprs.push(post),
        // Review-caught gate asymmetry (increment 1): walk the EXTRA output bodies
        // too. Without this, an int-only op with a COMPOSED operand hides in a
        // multi-output extra body at U8/S8 and bypasses the 8-bit leaf-operand pin
        // (rule 3) — cross-body CSE then hoists it into a truncated 8-bit tmp, the
        // exact 0c value-divergence ((200+100)>>1 = 22 hoisted vs 150 inlined).
        // Non-elementwise multi-output is already rejected by
        // `assert_valid_multi_output` (runs first), so this is the only place
        // `extra_out_bodies` is legal; empty for every single-output op.
        Access::Elementwise => exprs.extend(op.extra_out_bodies.iter()),
    }
    for e in exprs {
        walk(e, &op.name, dtype, int_dt, elementwise);
    }
}

/// Whether `e` contains a [`ScalarExpr::Coord`] leaf anywhere — drives the
/// increment-0d Strided schedule routing in [`build_plan`] (a Coord body must
/// reach the one emitter that materializes per-axis coordinates) and mirrors
/// `contract::expr_contains_cmp` in shape.
pub(crate) fn expr_contains_coord(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Coord(_) => true,
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_) => false,
        ScalarExpr::Unary(_, x) => expr_contains_coord(x),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => expr_contains_coord(a) || expr_contains_coord(b),
    }
}

/// Increment-0d [`ScalarExpr::Coord`] admissibility gate — runs at the TOP of
/// [`build_plan`] and walks the expressions of EVERY Access arm (the 0a
/// lesson: emitter backstops alone are bypassed by the reduction-class
/// lowering paths), with independent emitter backstops in `cuda`. Three
/// validate-reject rules, all honest misses:
///
/// 1. **Access**: Coord is legal ONLY in an [`Access::Elementwise`] body (v1).
///    A coordinate along a reduced/folded axis is ambiguous (which fold
///    iteration produced the output element?), and the RowReduce/Contraction
///    epilogues iterate their own coordinate spaces ((row, j) and (m, n)) —
///    lifting Coord into them needs explicit per-arm semantics, deferred.
/// 2. **Dtype**: `F32`/`F32Strict`/`F64` ONLY. f16/bf16 reject — the max
///    exactly-representable integer is 2048 (bf16: 256), which real axis
///    extents exceed, so a half coordinate would silently round. Int dtypes
///    reject — the coordinate lowers as a float cast (`(float)c{d}`), the
///    same double-math hazard as `Const`/`Param` at int dtypes; the
///    int-literal coordinate spelling is the queued follow-up.
/// 3. **Axis**: `axis < key.rank` — an out-of-range axis has no `c{d}` to
///    read (the emitter would spell an undefined identifier).
///
/// The exactness bound (f32 coordinates exact to 2²⁴, f64 to 2⁵³) is a CALLER
/// precondition — the key abstracts extents away, the same trust level as the
/// RowReduce column-weight extent precondition (see [`ScalarExpr::Coord`]).
fn assert_coord_admissibility(op: &OpDef, key: &StructureKey) {
    let elementwise = matches!(op.access, Access::Elementwise);
    fn walk(e: &ScalarExpr, op_name: &str, dtype: ElementKind, rank: u8, elementwise: bool) {
        match e {
            ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Param(_)
            | ScalarExpr::Reduced(_) => {}
            ScalarExpr::Coord(d) => {
                assert!(
                    elementwise,
                    "op '{op_name}': Coord({d}) is Elementwise-only in 0d — a coordinate \
                     along a reduced/folded axis is ambiguous (which fold iteration?), and \
                     the RowReduce/Contraction stages/epilogues iterate their own \
                     coordinate spaces; miss honestly"
                );
                assert!(
                    matches!(
                        dtype,
                        ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64
                    ),
                    "op '{op_name}': Coord({d}) requires an f32/f64 compute dtype, got \
                     {dtype:?} — f16/bf16 coordinates round past extent 2048 (bf16: 256) \
                     and int dtypes would inject the float-cast coordinate into integer \
                     math (int-literal coordinate spelling is a follow-up); miss honestly"
                );
                assert!(
                    *d < rank,
                    "op '{op_name}': Coord({d}) axis out of range for rank {rank} — the \
                     iteration space has no such coordinate"
                );
            }
            ScalarExpr::Unary(_, x) => walk(x, op_name, dtype, rank, elementwise),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                walk(a, op_name, dtype, rank, elementwise);
                walk(b, op_name, dtype, rank, elementwise);
            }
        }
    }
    let mut exprs: Vec<&ScalarExpr> = vec![&op.body];
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            exprs.extend(stages.iter().map(|s| &s.pre));
            exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => exprs.push(epilogue),
        // A Coord in a reduction post-expr is doubly rejected (here, non-
        // elementwise → the Coord arm fires; and by assert_valid_reduction_post).
        Access::Reduction { post, .. } => exprs.push(post),
        Access::Elementwise => {}
    }
    for e in exprs {
        walk(e, &op.name, key.dtype, key.rank, elementwise);
    }
}

/// Validate [`crate::ir::OpDef::out_dtype`] at plan time (AOT — like
/// `assert_no_half_nextafter`, this runs at the top of [`build_plan`] so EVERY
/// Access arm and every lowering path is covered; a panic here is an
/// author-error backstop, and the JIT never constructs a `Some` out_dtype).
///
/// Two admitted hetero-output shapes, both with an EXACT store conversion:
///
/// 1. **[`Access::Elementwise`] predicate → `Some(U8)`** (increment 0b): the
///    body ROOT must be a `Cmp*` — the value is exactly 0.0/1.0 and
///    `(unsigned char)` of that is exactly 1/0 (`OpDef::elementwise_pred`).
///
/// 2. **[`Access::Reduction`] hetero-out** (increment 0e — the roadmap "any/all
///    → U8, count → I64" reduction):
///    - `Some(U8)`: the POST-expr ROOT must be a `Cmp*`, so the stored value is
///      exactly 0.0/1.0 regardless of the fold magnitude — the honest
///      boolean-reduce (`any` = `Sum(x≠0)` with post `Reduced(0) > 0`; `all` =
///      `Sum(x=0)` with post `Reduced(0) = 0`; or `Max`/`Min` of a predicate
///      wrapped in a redundant cmp post). A non-cmp post would truncate the raw
///      accumulator silently (a count of 300 → `44` at u8), so it rejects.
///    - `Some(I64)`: the combine must be `Sum` and the post the identity
///      `Reduced(0)` — a **count** (`Sum(x≠0)`) or a sum-widening. The store is
///      `(long long)` of the accumulator: exact for an int input (i32→i64
///      widening) and exact for a float accumulator while the count ≤ 2²⁴ (a
///      documented CALLER precondition, the same trust level as `Coord`'s
///      exact-integer bound — the key abstracts extents away). `Mean`/`Max`/
///      `Min` → I64 reject (fractional / not the count shape).
///
/// Everything else panics honestly: `Some(non-U8/I64)` has no store conversion;
/// a `RowReduce`/`Contraction` stores its accumulator, not a predicate, so a
/// hetero store there would truncate silently.
///
/// A `Cmp*` NESTED inside a float body (mask-multiply `dy * (x > 0)`) is legal
/// with `out_dtype = None` — it is an inline 0.0/1.0 float, no u8 store — and a
/// top-level cmp with `out_dtype = None` (a float mask) is likewise legal.
fn assert_valid_out_dtype(op: &OpDef) {
    let Some(od) = op.out_dtype else { return };
    match &op.access {
        Access::Elementwise => {
            assert!(
                od == ElementKind::U8,
                "OpDef '{}': out_dtype Some({od:?}) is unsupported for an \
                 Elementwise op — the only hetero output dtype there is U8 (the \
                 comparison-predicate mask; use OpDef::elementwise_pred)",
                op.name
            );
            assert!(
                matches!(&op.body, ScalarExpr::Binary(bop, _, _) if bop.is_cmp()),
                "OpDef '{}': out_dtype = Some(U8) requires the body ROOT to be a \
                 comparison (ScalarExpr::Binary with a Cmp* op) — only a predicate \
                 yields exactly 0.0/1.0, so any other body would truncate silently \
                 under the u8 store; nested comparisons in a float body take \
                 out_dtype = None instead",
                op.name
            );
        }
        Access::Reduction { op: rop, post, .. } => match od {
            ElementKind::U8 => assert!(
                matches!(post, ScalarExpr::Binary(bop, _, _) if bop.is_cmp()),
                "OpDef '{}': a U8-output reduction requires the POST-expr ROOT to \
                 be a comparison (Cmp*) — only then is the stored value exactly \
                 0.0/1.0 (the honest any/all boolean-reduce); a non-cmp post would \
                 truncate the raw accumulator silently under the u8 store",
                op.name
            ),
            ElementKind::I64 => assert!(
                matches!(rop, ReduceOp::Sum) && matches!(post, ScalarExpr::Reduced(0)),
                "OpDef '{}': an I64-output reduction is the count/sum-widening shape \
                 — it requires op = Sum and the identity post (Reduced(0)); \
                 Mean/Max/Min → I64 or a non-identity post is out of scope (would \
                 not be an exact integer store)",
                op.name
            ),
            other => panic!(
                "OpDef '{}': out_dtype Some({other:?}) is unsupported for a \
                 reduction — v1 admits U8 (boolean any/all, via a Cmp* post) and \
                 I64 (count, via Sum + identity post)",
                op.name
            ),
        },
        Access::RowReduce { .. } | Access::Contraction { .. } => panic!(
            "OpDef '{}': out_dtype = Some({od:?}) is rejected under \
             RowReduce/Contraction — a fused reduction/contraction stores its \
             accumulator, not a 0/1 predicate, so a hetero store would truncate \
             silently; only Access::Elementwise (predicate → U8) and \
             Access::Reduction (any/all → U8, count → I64) carry a hetero output",
            op.name
        ),
    }
}

/// Validate the increment-0e reduction **post-expression** at plan time (AOT).
/// Runs at the top of [`build_plan`] (like the other honest-miss gates), with an
/// independent emitter backstop in `cuda::emit_reduction` (the post's `leaf`
/// closure panics if an `Input` ever reaches it). The post references the fold
/// result as `Reduced(0)` and MAY read `Const`/`Param`; it must NOT read:
///
/// - `Input(_)` — the reduced axis is gone, so an input at the output coordinate
///   is a different, ambiguous tensor (this mirrors the contraction epilogue's
///   `epilogue_reads_only_reduced0`, generalized to also admit `Param`);
/// - `Coord(_)` — reduction-class, Elementwise-only (also caught upstream);
/// - `Reduced(s)` for `s ≥ 1` — a single-fold reduction produces only
///   `Reduced(0)`.
fn assert_valid_reduction_post(op: &OpDef) {
    let Access::Reduction { post, .. } = &op.access else {
        return;
    };
    fn walk(e: &ScalarExpr, name: &str) {
        match e {
            ScalarExpr::Reduced(0) | ScalarExpr::Const(_) | ScalarExpr::Param(_) => {}
            ScalarExpr::Reduced(s) => panic!(
                "OpDef '{name}': reduction post-expr Reduced({s}) — a single-fold \
                 reduction produces only Reduced(0)"
            ),
            ScalarExpr::Input(i) => panic!(
                "OpDef '{name}': reduction post-expr must not read Input({i}) — the \
                 reduced axis is gone, so an input at the output coordinate is a \
                 different, ambiguous tensor; the post reads Reduced(0)/Const/Param"
            ),
            ScalarExpr::Coord(d) => panic!(
                "OpDef '{name}': reduction post-expr must not read Coord({d}) — Coord \
                 is Elementwise-only (a coordinate along a folded axis is ambiguous)"
            ),
            ScalarExpr::Unary(_, x) => walk(x, name),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                walk(a, name);
                walk(b, name);
            }
        }
    }
    walk(post, &op.name);
}

fn validate_row_reduce(stages: &[ReduceStage], epilogue: &ScalarExpr, n_inputs: u8, key: &StructureKey) {
    let dtype = key.dtype;
    assert!(
        matches!(
            dtype,
            ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
        ),
        "RowReduce requires a float dtype, got {dtype:?}"
    );
    let n = n_inputs as usize;
    assert!(
        (1..MAX_OPERANDS).contains(&n),
        "RowReduce n_inputs {n_inputs} out of [1, MAX_OPERANDS)"
    );
    assert!(
        key.n_operands as usize == n + 1,
        "RowReduce expects n_inputs+1 operands (inputs then output); got {}",
        key.n_operands
    );
    let rank = key.rank as usize;
    assert!(rank >= 1, "RowReduce needs a last (reduced) axis");
    let last = (rank - 1) as u8;

    // Operand roles + layout legality (the OOB / mis-index guards). Parallel index
    // over key.operands, so a range loop is the natural form. `is_col` feeds the
    // epilogue-only check below (reducing a per-column weight is rejected); a
    // RowScalar is legal in BOTH a stage `pre` and the epilogue (it is constant
    // along the reduced axis — layer-norm-bw's x_hat reads μ/rstd inside a fold),
    // so it is deliberately NOT tracked as a column.
    let mut is_col = [false; MAX_OPERANDS];
    let mut input0_streamed = false;
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let o = key.operands[i];
        match rr_role(o, last) {
            RrRole::RowStreamed => {
                assert!(
                    o.contig == Contiguity::Contig,
                    "RowReduce row-streamed input {i} must be contiguous (base = row*k assumes a dense last axis)"
                );
                // `Contiguity::Contig` is |stride|-based (classify_contiguity uses
                // strides[d].abs()), so a dense-but-REVERSED view passes the contig
                // check while carrying flipped=true — and the emitter walks memory
                // FORWARD (idx = row*k + j), reading the tensor mirrored / off the
                // end. `flipped` is a key-visible axis (the 'r'/'f' token field), so
                // reject it here as the ColBroadcast/RowScalar branches do. (Review
                // #2: the pre-lift inputs>0-must-be-column guard flip-checked every
                // extra input; lifting it to allow a 2nd row-streamed input newly
                // exposed this path.)
                assert!(
                    !o.flipped,
                    "RowReduce row-streamed input {i} must not be reversed along an axis (base = row*k reads forward-dense; a flipped view would read mirrored/OOB)"
                );
                input0_streamed |= i == 0;
            }
            RrRole::ColBroadcast => {
                assert!(
                    !o.bcast.is_set(last),
                    "RowReduce input {i}: the feature (last) axis is broadcast (mask {:#04x}) — a column weight/bias must vary along it; bake a true scalar as Const",
                    o.bcast.0
                );
                // Must broadcast EVERY outer axis (a per-column [k] vector), else
                // in_i[j] silently drops an outer-axis dependence.
                assert!(
                    (0..last).all(|d| o.bcast.is_set(d)),
                    "RowReduce column input {i} must broadcast every outer (row) axis — a per-column [k] weight/bias"
                );
                assert!(
                    !o.flipped,
                    "RowReduce column input {i} must not be reversed along the feature axis"
                );
                is_col[i] = true;
            }
            RrRole::RowScalar => {
                // Per-row scalar (a saved stat: μ, rstd, lse). The feature (last)
                // axis is broadcast (guaranteed by rr_role); NO outer (row) axis may
                // be — else it is either all-broadcast (a true scalar → bake as
                // Const) or drops a row dependence. Indexed `in_i[row]`, so it needs
                // an outer axis (rank >= 2) laid out dense (offset == row), the
                // latter a caller precondition at the same trust level as `x`'s
                // base = row*k (see the module note).
                assert!(
                    (0..last).all(|d| !o.bcast.is_set(d)),
                    "RowReduce row-scalar input {i}: an outer (row) axis is broadcast (mask {:#04x}) — a per-row scalar varies across rows and is constant only along the feature axis; an all-broadcast operand is a true scalar (bake as Const)",
                    o.bcast.0
                );
                assert!(
                    !o.flipped,
                    "RowReduce row-scalar input {i} must not be reversed"
                );
                assert!(
                    rank >= 2,
                    "RowReduce row-scalar input {i} needs rank >= 2 (an outer row axis to index by `row`)"
                );
            }
        }
    }
    assert!(
        input0_streamed,
        "RowReduce Input0 (x) must be the row-streamed reduced tensor, not a column-broadcast weight or a per-row scalar"
    );
    // Inputs 1.. may now be a second **row-streamed** tensor (softmax-bw's `dy`
    // beside `y`), a per-column weight/bias, OR a per-row scalar (layer-norm-bw's
    // μ/rstd). The former "inputs>0 must be column-broadcast" guard is LIFTED
    // (increment 2): a second row-streamed input full [n_out,k] is the point. A
    // bare rank-1 [k] passed as input>0 has an empty bcast and so is now accepted
    // as row-streamed — its full extent [n_out,k] is a caller precondition (the key
    // cannot see n_out/k), the identical trust level as input 0. See the module note.
    if n > 1 {
        assert!(rank >= 2, "RowReduce with a multi-operand epilogue needs rank >= 2");
    }
    let out = key.operands[n];
    assert!(
        out.bcast.is_empty() && out.contig == Contiguity::Contig,
        "RowReduce output must be full-width contiguous (empty bcast)"
    );

    // Expression legality. `max_reduced` = stages already produced (stage `i` may
    // read `Reduced(0..i)`; the epilogue may read all). `in_stage` forbids a column
    // input inside a reduction `pre` (reducing a per-column operand is nonsense).
    fn check(e: &ScalarExpr, n_inputs: u8, max_reduced: u8, in_stage: bool, is_col: &[bool]) {
        match e {
            ScalarExpr::Input(i) => {
                assert!(*i < n_inputs, "RowReduce Input({i}) >= n_inputs {n_inputs}");
                if in_stage {
                    assert!(
                        !is_col[*i as usize],
                        "RowReduce column input {i} used inside a reduction stage.pre — column weight/bias are epilogue-only"
                    );
                }
            }
            ScalarExpr::Reduced(s) => assert!(
                *s < max_reduced,
                "RowReduce Reduced({s}) references a stage not yet produced (have {max_reduced})"
            ),
            ScalarExpr::Param(i) => {
                panic!("RowReduce v1 forbids Param({i}) — bake scalars (eps) as Const")
            }
            ScalarExpr::Coord(d) => {
                panic!(
                    "RowReduce forbids Coord({d}) — the RowReduce stages/epilogue iterate \
                     the (row, j) space, not an elementwise output coordinate space; \
                     Coord is Elementwise-only in 0d"
                )
            }
            ScalarExpr::Const(v) => assert!(v.is_finite(), "RowReduce Const must be finite, got {v}"),
            ScalarExpr::Unary(_, x) => check(x, n_inputs, max_reduced, in_stage, is_col),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check(a, n_inputs, max_reduced, in_stage, is_col);
                check(b, n_inputs, max_reduced, in_stage, is_col);
            }
        }
    }
    for (i, st) in stages.iter().enumerate() {
        // Prod stages (0e added the combiner to Access::Reduction, not to the
        // fused RowReduce cooperative reducer) are an honest miss here — the
        // emitter has no block_prod in the row path. Gate + emitter backstop.
        assert!(
            !matches!(st.op, ReduceOp::Prod),
            "RowReduce stage {i}: the Prod combiner is not supported in the fused \
             row-reduce path (0e adds Prod to Access::Reduction only); miss honestly"
        );
        check(&st.pre, n_inputs, i as u8, true, &is_col);
    }
    check(epilogue, n_inputs, stages.len() as u8, false, &is_col);
}

/// Vector width in elements for a [`VecWidth`] bucket.
fn vec_width_elems(v: VecWidth) -> u32 {
    match v {
        VecWidth::V8 => 8,
        VecWidth::V4 => 4,
        VecWidth::V2 => 2,
        VecWidth::Scalar => 1,
    }
}

#[cfg(test)]
mod multi_output_validate {
    //! Increment-1 multi-output gate-rejection tests. Per the house rule these
    //! call `build_plan` DIRECTLY (an emitter panic would mask a gate mutation).
    use super::build_plan;
    use crate::ir::{input, konst, Access, OpDef, ReduceOp, ScalarExpr};
    use baracuda_kernels_types::{
        structure_key, ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey,
    };

    // n_operands contiguous 1D operands of `dtype`.
    fn key_dt(dtype: ElementKind, n_operands: usize) -> StructureKey {
        let a = OperandDesc::new(1, &[1024], &[1], dtype, 256);
        let ops: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89)
    }
    // F32 shorthand for the layout-shape rejection tests.
    fn key(n_operands: usize) -> StructureKey {
        key_dt(ElementKind::F32, n_operands)
    }

    fn mul_backward() -> OpDef {
        OpDef::elementwise_multi(
            "mul_backward",
            3,
            &[ElementKind::F32],
            vec![input(0) * input(2), input(0) * input(1)],
        )
    }

    #[test]
    fn valid_multi_output_builds() {
        // The happy path: 3 inputs + 2 outputs = 5 operands.
        let _ = build_plan(&mul_backward(), &key(5));
    }

    #[test]
    #[should_panic(expected = "broadcast")]
    fn output_broadcast_aliases_its_writes_rejected() {
        // A broadcast (stride-0) OUTPUT aliases its own writes across iteration
        // coordinates (a write race) and is not the full output shape.
        let a = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::F32, 256);
        // Output operand 4 (the last): broadcast inner axis (stride 0 → bcast).
        let bcast_out = OperandDesc::new(2, &[8, 4], &[1, 0], ElementKind::F32, 256);
        let ops = vec![a, a, a, a, bcast_out];
        let k = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
        let _ = build_plan(&mul_backward(), &k);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn operand_count_mismatch_rejected() {
        // The key must carry exactly n_inputs+n_outputs = 5 operands; a 4-operand
        // key is a declared shape/operand mismatch.
        let _ = build_plan(&mul_backward(), &key(4));
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_OPERANDS")]
    fn n_outputs_overflow_max_operands_rejected() {
        // 6 inputs + 3 outputs = 9 > MAX_OPERANDS(8).
        let op = OpDef::elementwise_multi(
            "over",
            6,
            &[ElementKind::F32],
            vec![input(0), input(1), input(2)],
        );
        let _ = build_plan(&op, &key(8));
    }

    #[test]
    #[should_panic(expected = "Reduced")]
    fn reduced_in_a_body_rejected() {
        // A multi-output body must not read Reduced (no reduction in an
        // elementwise map). Built directly (no constructor produces this).
        let mut op = OpDef::elementwise_multi(
            "bad",
            2,
            &[ElementKind::F32],
            vec![input(0) * input(1), input(0)],
        );
        op.extra_out_bodies[0] = ScalarExpr::Reduced(0);
        let _ = build_plan(&op, &key(4));
    }

    #[test]
    #[should_panic(expected = "Coord")]
    fn coord_in_a_body_rejected() {
        // A multi-output body must not read Coord (elementwise-map only in v1).
        let mut op = OpDef::elementwise_multi(
            "bad",
            2,
            &[ElementKind::F32],
            vec![input(0) * input(1), input(0)],
        );
        op.extra_out_bodies[0] = ScalarExpr::Coord(0);
        let _ = build_plan(&op, &key(4));
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn multi_output_on_a_reduction_rejected() {
        // extra_out_bodies on a non-Elementwise op — a fused reduction stores one
        // accumulator, not N bodies. Built directly by pushing onto the field.
        let mut op = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        op.extra_out_bodies.push(ScalarExpr::Input(0));
        assert!(matches!(op.access, Access::Reduction { .. }));
        // A reduction key = [input, output]; the Elementwise check fires first.
        let _ = build_plan(&op, &key(2));
    }

    #[test]
    #[should_panic(expected = "uniform output dtype")]
    fn hetero_multi_output_rejected() {
        // out_dtype = Some on a multi-output op — hetero multi-out is deferred.
        // Body 0's root is a valid Cmp so the pre-existing assert_valid_out_dtype
        // predicate gate PASSES; the rejection is the multi-output uniform-dtype
        // rule specifically.
        use crate::ir::BinaryOp;
        let mut op = OpDef::elementwise_multi(
            "bad",
            2,
            &[ElementKind::F32],
            vec![input(0).binary(BinaryOp::CmpLt, input(1)), input(0)],
        );
        op.out_dtype = Some(ElementKind::U8);
        let _ = build_plan(&op, &key(4));
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn non_finite_const_in_a_body_rejected() {
        let op = OpDef::elementwise_multi(
            "bad",
            2,
            &[ElementKind::F32],
            vec![input(0) * input(1), input(0) + konst(f64::INFINITY)],
        );
        let _ = build_plan(&op, &key(4));
    }

    // ---- Review-caught gate asymmetry: EXTRA output bodies must be walked by the
    // half-Nextafter and int-op-admissibility gates too (they seeded only op.body).

    #[test]
    #[should_panic(expected = "requires LEAF")]
    fn composed_int_op_operand_in_an_extra_body_rejected_at_the_plan_gate() {
        use crate::ir::BinaryOp;
        let op = OpDef::elementwise_multi(
            "bad",
            3,
            &[ElementKind::U8],
            vec![
                input(0) + input(1),
                (input(0) + input(1)).binary(BinaryOp::Shr, input(2)),
            ],
        );
        let _ = build_plan(&op, &key_dt(ElementKind::U8, 5));
    }

    #[test]
    #[should_panic(expected = "must miss honestly")]
    fn half_nextafter_in_an_extra_body_rejected_at_the_plan_gate() {
        use crate::ir::BinaryOp;
        let op = OpDef::elementwise_multi(
            "bad",
            1,
            &[ElementKind::F16],
            vec![
                input(0) * input(0),
                input(0).binary(BinaryOp::Nextafter, input(0)),
            ],
        );
        let _ = build_plan(&op, &key_dt(ElementKind::F16, 3));
    }
}

#[cfg(test)]
mod rowreduce_role_validate {
    //! Increment-2 RowReduce role tests: `rr_role` classification units + gate
    //! tests that call `build_plan` DIRECTLY (the house rule — an emitter panic
    //! would mask a gate mutation). Covers the new `RowScalar` role, the lifted
    //! "inputs>0 must be column-broadcast" restriction, and the rejected-ambiguous
    //! cases.
    use super::{build_plan, rr_role, RrRole};
    use crate::ir::{input, reduced, OpDef, ReduceOp, ReduceStage};
    use baracuda_kernels_types::{
        structure_key, ArchSku, AxisMask, Contiguity, DivBucket, ElementKind, OpCategory,
        OperandDesc, OperandKey, VecWidth,
    };

    // A minimal OperandKey carrying only the broadcast mask + flip (all rr_role /
    // validate read for classification); contig is irrelevant to the role.
    fn opkey(bcast: u8, flipped: bool) -> OperandKey {
        OperandKey {
            contig: Contiguity::Broadcast,
            bcast: AxisMask(bcast),
            vec_width: VecWidth::Scalar,
            inner_div: DivBucket::Any,
            flipped,
        }
    }

    #[test]
    fn rr_role_classifies_the_three_geometries_and_the_ambiguous_case() {
        // rank 2 ⇒ feature (last) axis = 1.
        assert_eq!(rr_role(opkey(0b00, false), 1), RrRole::RowStreamed); // nothing bcast
        assert_eq!(rr_role(opkey(0b10, false), 1), RrRole::RowScalar); // feature bcast ⇒ per-row
        assert_eq!(rr_role(opkey(0b01, false), 1), RrRole::ColBroadcast); // outer bcast ⇒ per-col
        // A varying feature axis is NEVER a RowScalar (that is exactly ColBroadcast).
        assert_ne!(rr_role(opkey(0b01, false), 1), RrRole::RowScalar);
        // All-broadcast is ambiguous: classified RowScalar (last is set), then REJECTED
        // by validate's outer-axis-clear check (a true scalar is a Const, not an operand).
        assert_eq!(rr_role(opkey(0b11, false), 1), RrRole::RowScalar);
    }

    // A full-width row-streamed operand [256,128].
    fn stream() -> OperandDesc {
        OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256)
    }
    // A per-row scalar: [n_out,k]-presented, strides [1,0] (feature-axis broadcast,
    // outer varies dense).
    fn rowscalar() -> OperandDesc {
        OperandDesc::new(2, &[256, 128], &[1, 0], ElementKind::F32, 256)
    }

    #[test]
    fn second_row_streamed_input_builds() {
        // softmax bw: y, dy both row-streamed [n,k] + dx. The former guard rejected
        // input>0 unless column-broadcast; the lift makes this the point — it PASSES.
        let op = OpDef::row_reduce(
            "softmax_bw",
            2,
            &[ElementKind::F32],
            vec![ReduceStage { pre: (input(0) * input(1)).0, op: ReduceOp::Sum }],
            input(0) * (input(1) - reduced(0)),
        );
        let s = stream();
        let key = structure_key(OpCategory::Softmax, &[s, s, s], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    fn row_scalar_inputs_build() {
        // layer_norm bw: x, dy row-streamed; mean, rstd per-row scalars (used INSIDE a
        // stage pre — x_hat — and the epilogue).
        let x_hat = (input(0) - input(2)) * input(3);
        let op = OpDef::row_reduce(
            "layer_norm_bw",
            4,
            &[ElementKind::F32],
            vec![
                ReduceStage { pre: input(1).0, op: ReduceOp::Mean },
                ReduceStage { pre: (input(1) * x_hat.clone()).0, op: ReduceOp::Mean },
            ],
            input(3) * (input(1) - reduced(0) - x_hat * reduced(1)),
        );
        let s = stream();
        let rs = rowscalar();
        let key = structure_key(OpCategory::Normalization, &[s, s, rs, rs, s], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    // A 2-input op whose input 1 is the offending operand under test.
    fn probe_op() -> OpDef {
        OpDef::row_reduce(
            "probe",
            2,
            &[ElementKind::F32],
            vec![ReduceStage { pre: input(0).0, op: ReduceOp::Sum }],
            input(0) + input(1),
        )
    }

    #[test]
    #[should_panic(expected = "true scalar")]
    fn all_broadcast_input_rejected_as_a_true_scalar() {
        // strides [0,0] ⇒ both axes broadcast ⇒ classified RowScalar (last set) then
        // rejected by the outer-axis-clear check — a genuinely ambiguous mask.
        let s = stream();
        let allb = OperandDesc::new(2, &[256, 128], &[0, 0], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[s, allb, s], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "reversed")]
    fn row_scalar_flipped_rejected() {
        // feature-axis broadcast (RowScalar) but a NEGATIVE outer stride ⇒ flipped.
        let s = stream();
        let flipped_rs = OperandDesc::new(2, &[256, 128], &[-1, 0], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[s, flipped_rs, s], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "reversed")]
    fn flipped_row_streamed_input_rejected() {
        // Review #2 CRITICAL: a dense-but-REVERSED second row-streamed input
        // (empty bcast ⇒ RowStreamed; strides [128,-1] ⇒ |stride|-contig=Contig
        // but flipped=true). Pre-fix the RowStreamed branch checked only Contig
        // and accepted it, then the emitter read it forward (mirrored/OOB). Now
        // rejected, matching the ColBroadcast/RowScalar branches. Via build_plan
        // DIRECTLY so only the plan gate can fire.
        let s = stream();
        let flipped = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[s, flipped, s], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "every outer")]
    fn rank3_partial_middle_broadcast_rejected() {
        // rank 3, only the MIDDLE axis broadcast: feature (last) varies ⇒ ColBroadcast,
        // but it fails "must broadcast every outer axis" (axis 0 is not broadcast) — an
        // ambiguous partial broadcast, neither a clean column nor a row-scalar.
        let x = OperandDesc::new(3, &[4, 8, 16], &[128, 16, 1], ElementKind::F32, 256);
        let mid = OperandDesc::new(3, &[4, 8, 16], &[16, 0, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(3, &[4, 8, 16], &[128, 16, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[x, mid, out], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "Input0")]
    fn input0_as_row_scalar_rejected() {
        // Input 0 (the reduced tensor) presented feature-broadcast (a row-scalar) is
        // illegal — it must be the row-streamed x.
        let op = OpDef::row_reduce(
            "t",
            1,
            &[ElementKind::F32],
            vec![ReduceStage { pre: input(0).0, op: ReduceOp::Sum }],
            reduced(0),
        );
        let key = structure_key(OpCategory::Softmax, &[rowscalar(), stream()], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }
}
