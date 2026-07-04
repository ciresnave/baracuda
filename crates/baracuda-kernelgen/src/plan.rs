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
    /// is the epilogue (`OpDef::row_reduce` sets `body = epilogue`).
    pub body: &'a ScalarExpr,
    /// The op's access pattern — the [`Schedule::RowReduce`] emitter reads its
    /// `stages` (and epilogue) off here, since `Schedule` is `Copy` and can't carry
    /// the stage `Vec`.
    pub access: &'a Access,
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
    assert_no_half_nextafter(op, key.dtype);
    assert_int_op_admissibility(op, key.dtype);
    assert_coord_admissibility(op, key);
    let schedule = match op.access {
        Access::Reduction {
            op: rop,
            axes,
            keepdim,
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
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum RrRole {
    /// The reduced tensor `x` ([n_out, k], full / empty bcast) — `in_i[base+j]`.
    RowStreamed,
    /// A per-column `[k]` weight/bias, broadcast over the row axis — `in_i[j]`.
    ColBroadcast,
}

/// Classify a RowReduce input by its broadcast mask. **Total / non-panicking** —
/// the emitter calls this for the load index and must never crash; all *rejection*
/// of malformed shapes lives in [`validate_row_reduce`] (one source of truth, no
/// drift). An empty bcast is the row-streamed `x`; any broadcast is a column
/// operand (the legality of *which* broadcast is validate's job).
pub(crate) fn rr_role(o: OperandKey) -> RrRole {
    if o.bcast.is_empty() {
        RrRole::RowStreamed
    } else {
        RrRole::ColBroadcast
    }
}

/// Validate a [`Access::RowReduce`] op at build time (AOT — RowReduce never crosses
/// the JIT trust boundary, so a panic here is an author-error backstop, like
/// `emit_reduction`'s asserts). Catches expression errors (a `Reduced(s)` not yet
/// produced, out-of-range `Input`, a `Param`, a non-finite `Const`, a column input
/// inside a reduction stage) **and** operand-layout errors that would mis-index or
/// read out of bounds: `x` (input 0) must be row-streamed + contiguous; every other
/// input must be a per-column `[k]` weight/bias broadcast over **all** outer axes
/// (rank ≥ 2), not reversed, and never a bare rank-1 `[k]` tensor (whose empty bcast
/// would misclassify as row-streamed and read `in_i[row*k+j]` past the buffer); the
/// output is full-width contiguous.
///
/// v1 assumes a **uniform operand dtype** (the structure key carries one dtype) — a
/// mixed-dtype LayerNorm (fp16 `x` + fp32 weight) is unrepresentable here and must
/// be refused upstream by the caller.
///
/// **Caller pre-condition this cannot check:** a column operand's feature-axis extent
/// must equal `x`'s `k`. The structure key carries broadcast masks but **no numeric
/// extents** (specialize on structure, not extents), so a too-short weight has the
/// same key as a correct one — it's accepted here and the emitter reads `in_i[j]`
/// past its buffer (a confirmed on-device OOB). This is the same trust level as the
/// `n_out`/`k` launch args; the layer that still holds the `OperandDesc` extents (an
/// AOT op author, or the live seam caller once `region_to_op` wires RowReduce) must
/// assert it — the key has already abstracted the extents away by the time we run.
/// Nextafter is declared f32/f64-only at the IR level (its half lowering via
/// promote-to-f32 would step the f32 lattice — ~2^13 steps inside one half
/// step, so the demote rounds straight back: a silently wrong no-op). The
/// CUDA emitter's `cuda_binary` panic only guards the elementwise path; the
/// reduction pre-body, RowReduce stages/epilogue, and contraction epilogues
/// lower through accumulator-width helpers that never pass through it. This
/// plan-level walk covers EVERY Access arm, so no lowering path — present or
/// future backend — can bypass the honest miss.
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
        Access::Elementwise | Access::Reduction { .. } => {}
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
        Access::Elementwise | Access::Reduction { .. } => {}
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
        Access::Elementwise | Access::Reduction { .. } => {}
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
/// The v1 rule: `Some(U8)` is legal **only** for an [`Access::Elementwise`] op
/// whose body ROOT is a `Cmp*` predicate — the one shape whose store conversion
/// is exact (the predicate is exactly 0.0/1.0, and `(unsigned char)` of that is
/// exactly 1/0). Everything else panics with an honest-miss message:
/// - a non-cmp body with a u8 output would truncate real float values silently;
/// - a cmp under `Reduction`/`RowReduce`/`Contraction` stores the *accumulator*,
///   not a predicate (a predicate-reduce — any/all/count — is the roadmap's
///   "hetero output dtype" reduction follow-up, not this increment);
/// - `Some(non-U8)` is unimplemented (no store conversion exists for it).
///
/// A `Cmp*` NESTED inside a float body (mask-multiply `dy * (x > 0)`) is legal
/// with `out_dtype = None` — it is an inline 0.0/1.0 float, no u8 store — and a
/// top-level cmp with `out_dtype = None` (a float mask) is likewise legal.
fn assert_valid_out_dtype(op: &OpDef) {
    let Some(od) = op.out_dtype else { return };
    assert!(
        od == ElementKind::U8,
        "OpDef '{}': out_dtype Some({od:?}) is unsupported — the only hetero \
         output dtype in v1 is U8 (the comparison-predicate mask; use \
         OpDef::elementwise_pred)",
        op.name
    );
    assert!(
        matches!(op.access, Access::Elementwise),
        "OpDef '{}': out_dtype = Some(U8) is legal only for Access::Elementwise \
         — a Reduction/RowReduce/Contraction stores its accumulator, not a \
         0/1 predicate, so a u8 store there would truncate silently; \
         predicate-reductions (any/all/count) are a separate follow-up",
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
    // over key.operands + is_col, so a range loop is the natural form.
    let mut is_col = [false; MAX_OPERANDS];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let o = key.operands[i];
        match rr_role(o) {
            RrRole::RowStreamed => assert!(
                o.contig == Contiguity::Contig,
                "RowReduce row-streamed input {i} must be contiguous (base = row*k assumes a dense last axis)"
            ),
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
        }
    }
    assert!(
        !is_col[0],
        "RowReduce Input0 (x) must be the row-streamed reduced tensor, not column-broadcast"
    );
    // Inputs 1.. must be column-broadcast — closes the silent OOB where a bare
    // rank-1 [k] weight (empty bcast) misclassifies as a second row-streamed input
    // and reads in_i[row*k+j] past its k elements. (A second row-streamed input —
    // residual fusion — is a deliberate follow-up.)
    #[allow(clippy::needless_range_loop)]
    for i in 1..n {
        assert!(
            is_col[i],
            "RowReduce input {i} must be a per-column [k] weight/bias (rank-aligned [n_out,k] with outer stride 0), not a bare row-streamed tensor"
        );
    }
    if n > 1 {
        assert!(rank >= 2, "RowReduce with weight/bias needs rank >= 2");
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
