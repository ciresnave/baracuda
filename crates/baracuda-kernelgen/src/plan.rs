//! Language-agnostic kernel plan — the schedule decision.
//!
//! [`build_plan`] turns an [`OpDef`] + a [`StructureKey`] cell into a neutral
//! [`KernelPlan`]: *what* to compute (the op body + dtype) and the *schedule*
//! (vectorized vs scalar) to compute it with. A [`crate::backend::Backend`]
//! lowers the plan to a concrete language. Choosing the schedule here, not in
//! the backend, keeps the decision shared across every backend.

use crate::ir::{Access, OpDef, ReduceOp, ReduceStage, ScalarExpr};
use baracuda_kernels_types::{
    Contiguity, ElementKind, OperandKey, StructureKey, VecWidth, MAX_OPERANDS,
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
    let schedule = match op.access {
        Access::Reduction { op: rop } => Schedule::Reduction { op: rop },
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
        Access::Elementwise => {
            let n = key.n_operands as usize;
            let all_contig =
                n > 0 && (0..n).all(|k| key.operands[k].contig == Contiguity::Contig);
            // The kernel vectorizes at the *narrowest* width every operand supports.
            let min_width = (0..n)
                .map(|k| vec_width_elems(key.operands[k].vec_width))
                .min()
                .unwrap_or(1);
            if !all_contig {
                Schedule::Strided
            } else if min_width >= 2 {
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
        schedule,
        key,
        body: &op.body,
        access: &op.access,
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
