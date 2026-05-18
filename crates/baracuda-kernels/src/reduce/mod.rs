//! Reduction op family — Phase 4 (Category E).
//!
//! Output shape differs from input: the reduced axis collapses to size
//! 1 (keepdim convention). Single-axis reductions today; full-tensor
//! and multi-axis reductions reduce to repeated single-axis ops.
//!
//! Today wired:
//!
//! - **[`ReducePlan`]** — `{Sum, Mean, Max, Min, Prod, Norm2, LogSumExp,
//!   Var, Std} × {f32, f16, bf16, f64}` (36 cells). Var / Std use a
//!   one-pass Welford kernel; LogSumExp uses a dedicated two-pass kernel
//!   (max-then-sum-exp) for numerical stability. [`ReduceBackwardPlan`]
//!   covers the same matrix.
//!
//! - **[`ArgReducePlan`]** — `{Argmax, Argmin}` returning `i64` indices.
//!   Separate plan because the output dtype differs from the input
//!   (index, not value); no BW (non-differentiable through `argmax`).
//!
//! - **[`BoolReducePlan`]** — `{Any, All}` returning `Bool`. Distinct
//!   reduction algebra (logical-or / logical-and short-circuit). No BW.
//!
//! - **[`CountReducePlan`]** — `CountNonzero`, value count → `i64`.
//!
//! - **[`TracePlan`]** — scalar `trace(M)` for rank-2 matrices (both
//!   axes reduced along the diagonal).
//!
//! All reduction FW + BW are deterministic and bit-stable on the same
//! hardware (no atomic-add; one-block-per-output-cell or one-thread-per-
//! output-cell). f16 / bf16 accumulate in f32 (FP detour); f64 keeps
//! everything in double.

pub mod arg_axis;
pub mod axis;
pub mod axis_backward;
pub mod bool_axis;
pub mod count_axis;
pub mod trace;

pub use arg_axis::{ArgReduceArgs, ArgReduceDescriptor, ArgReducePlan};
pub use axis::{ReduceArgs, ReduceDescriptor, ReducePlan};
pub use axis_backward::{ReduceBackwardArgs, ReduceBackwardDescriptor, ReduceBackwardPlan};
pub use bool_axis::{BoolReduceArgs, BoolReduceDescriptor, BoolReducePlan};
pub use count_axis::{CountReduceArgs, CountReduceDescriptor, CountReducePlan};
pub use trace::{TraceArgs, TraceDescriptor, TracePlan};
