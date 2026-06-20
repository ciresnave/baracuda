//! # baracuda-kernelgen
//!
//! Build-time generator that turns an op's **abstract IR** (the algorithm) plus
//! a [`baracuda_kernels_types::StructureKey`] cell (the schedule) into a
//! specialized kernel — and, next, its FKC contract.
//!
//! The crate is **language-agnostic except for the lowering backend**:
//!
//! - [`ir`] — the op IR (a [`ScalarExpr`] DAG). Backend-neutral.
//! - [`plan`] — the schedule decision (`StructureKey` → [`KernelPlan`]). Neutral.
//! - [`backend`] — the [`Backend`] trait + the neutral [`backend::lower_expr`].
//! - [`cuda`] — the **only** backend-specific module today. Slang / SPIR-V /
//!   Metal / CPU backends are additional [`Backend`] impls, no core changes.
//!
//! Op logic is described as IR rather than opaque CUDA precisely so the emitter
//! can *see the dataflow* and transform it (vectorize, hoist, fuse) — and so the
//! same op can be lowered to any backend. A hand-written escape hatch is
//! reserved for bespoke ops the IR can't yet express.
//!
//! This is a dev/build tool (`publish = false`); the artifacts it emits are
//! committed and ship inside `baracuda-kernels-sys`.
//!
//! ## Status (v1)
//!
//! Pilot scope: f32 elementwise ops, contiguous operands, emitted `float4`-
//! vectorized when the cell says V4 and scalar otherwise, lowered to CUDA.
//! Other dtypes, strided / broadcast / reduction schedules, additional
//! backends, FKC emission, and the algebraic optimizer are the growth path.

pub mod backend;
pub mod cuda;
pub mod ir;
pub mod pattern;
pub mod plan;

pub use backend::{Backend, GeneratedKernel};
pub use cuda::Cuda;
pub use ir::{input, konst, Access, Expr, OpDef, ScalarExpr, UnaryOp};
pub use pattern::{derive_pattern, to_fkc, PatternError, PatternNode};
pub use plan::{build_plan, KernelPlan, Schedule};

use baracuda_kernels_types::StructureKey;

/// Generate a specialized kernel for `op` at structure cell `key`, lowered by
/// `backend`. Convenience over [`build_plan`] followed by [`Backend::lower`].
#[must_use]
pub fn generate(op: &OpDef, key: &StructureKey, backend: &dyn Backend) -> GeneratedKernel {
    backend.lower(&build_plan(op, key))
}
