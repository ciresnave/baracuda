//! Language-agnostic kernel plan — the schedule decision.
//!
//! [`build_plan`] turns an [`OpDef`] + a [`StructureKey`] cell into a neutral
//! [`KernelPlan`]: *what* to compute (the op body + dtype) and the *schedule*
//! (vectorized vs scalar) to compute it with. A [`crate::backend::Backend`]
//! lowers the plan to a concrete language. Choosing the schedule here, not in
//! the backend, keeps the decision shared across every backend.

use crate::ir::{Access, OpDef, ScalarExpr};
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
    /// Linear access, one element at a time.
    Scalar,
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
    /// The [`StructureKey`] token of the cell this plan targets (traceability).
    pub cell: String,
    /// Output `= body`, evaluated per coordinate.
    pub body: &'a ScalarExpr,
}

/// Choose the schedule for `op` at structure cell `key` and return a neutral
/// [`KernelPlan`].
///
/// v1: elementwise — vectorized when every operand is `Contig` + `V4`, scalar
/// otherwise. (Whether a backend can lower the chosen dtype is the backend's
/// call, not this function's.)
///
/// # Panics
/// Panics if the op is not elementwise (the only access pattern v1 schedules).
#[must_use]
pub fn build_plan<'a>(op: &'a OpDef, key: &StructureKey) -> KernelPlan<'a> {
    assert!(
        matches!(op.access, Access::Elementwise),
        "v1 schedules elementwise ops only"
    );
    let n = key.n_operands as usize;
    let all_contig_v4 = n > 0
        && (0..n).all(|k| {
            let o = key.operands[k];
            o.contig == Contiguity::Contig && o.vec_width == VecWidth::V4
        });
    let schedule = if all_contig_v4 {
        Schedule::Vectorized { width: 4 }
    } else {
        Schedule::Scalar
    };
    KernelPlan {
        op_name: &op.name,
        n_inputs: op.n_inputs,
        dtype: key.dtype,
        schedule,
        cell: key.to_token(),
        body: &op.body,
    }
}
