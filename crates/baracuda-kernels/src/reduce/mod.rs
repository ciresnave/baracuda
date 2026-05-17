//! Reduction op family — Phase 4 trailblazer surface (Category E).
//!
//! Output shape differs from input: the reduced axis collapses to size
//! 1 (keepdim convention). Single-axis reductions today; full-tensor
//! and multi-axis reductions follow in fanout sessions (each becomes
//! repeated single-axis reductions internally, or fused multi-axis
//! kernels for hot paths).
//!
//! Today only [`ReducePlan`] (Sum) on f32 is wired — the Phase 4
//! trailblazer. The other reduction kinds (Mean, Max, Min, Prod, Var,
//! Std, Norm2, LogSumExp, …) are reserved discriminants in
//! [`baracuda_kernels_types::ReduceKind`]. Argmax / Argmin have a
//! different output dtype (index, not value) and will need a separate
//! plan shape.

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
