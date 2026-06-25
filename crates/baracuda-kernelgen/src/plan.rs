//! Language-agnostic kernel plan — the schedule decision.
//!
//! [`build_plan`] turns an [`OpDef`] + a [`StructureKey`] cell into a neutral
//! [`KernelPlan`]: *what* to compute (the op body + dtype) and the *schedule*
//! (vectorized vs scalar) to compute it with. A [`crate::backend::Backend`]
//! lowers the plan to a concrete language. Choosing the schedule here, not in
//! the backend, keeps the decision shared across every backend.

use crate::ir::{Access, OpDef, ReduceOp, ReduceStage, ScalarExpr};
use baracuda_kernels_types::{Contiguity, ElementKind, StructureKey, VecWidth};

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
            validate_row_reduce(stages, epilogue, op.n_inputs, key.dtype);
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

/// Validate a [`Access::RowReduce`] op at build time (AOT — RowReduce never crosses
/// the JIT trust boundary, so a panic here is an author-error backstop, like
/// `emit_reduction`'s asserts). Catches: a stage `pre` referencing a `Reduced(s)`
/// not yet produced (`s >= i`); an out-of-range `Input`; a `Param` (v1 scalars are
/// `Const`, which also sidesteps the f32-only-param lowering assert); a non-finite
/// `Const` (would emit the headerless-illegal `INFINITY`/`NAN`); or a non-float dtype.
fn validate_row_reduce(
    stages: &[ReduceStage],
    epilogue: &ScalarExpr,
    n_inputs: u8,
    dtype: ElementKind,
) {
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
    // `max_reduced` = the number of stages already produced (stage `i` may read
    // `Reduced(0..i)`; the epilogue may read `Reduced(0..n_stages)`).
    fn check(e: &ScalarExpr, n_inputs: u8, max_reduced: u8) {
        match e {
            ScalarExpr::Input(i) => {
                assert!(*i < n_inputs, "RowReduce Input({i}) >= n_inputs {n_inputs}");
            }
            ScalarExpr::Reduced(s) => assert!(
                *s < max_reduced,
                "RowReduce Reduced({s}) references a stage not yet produced (have {max_reduced})"
            ),
            ScalarExpr::Param(i) => {
                panic!("RowReduce v1 forbids Param({i}) — bake scalars (eps) as Const")
            }
            ScalarExpr::Const(v) => assert!(v.is_finite(), "RowReduce Const must be finite, got {v}"),
            ScalarExpr::Unary(_, x) => check(x, n_inputs, max_reduced),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check(a, n_inputs, max_reduced);
                check(b, n_inputs, max_reduced);
            }
        }
    }
    for (i, st) in stages.iter().enumerate() {
        check(&st.pre, n_inputs, i as u8);
    }
    check(epilogue, n_inputs, stages.len() as u8);
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
