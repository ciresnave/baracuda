//! Shape / layout op family — Category N from the comprehensive plan.
//!
//! Each op has its own plan type because the descriptor / args shapes
//! differ enough that one unified `ShapeLayoutPlan<T, N>` doesn't fit.
//! Forward plans for all six wired ops (Pad, Concat, Permute, Repeat,
//! Flip, Roll) live alongside their backward siblings:
//!
//! - Flip / Roll / Permute BWs are pure Rust wrappers that dispatch to
//!   the forward CUDA kernel with mutated params (involution / negated
//!   shifts / inverse permutation respectively).
//! - Pad BW (constant mode) is a slice — own kernel.
//! - Repeat / Concat BWs land in fanout.

pub mod concat;
pub mod concat_backward;
pub mod contiguize;
pub mod fill;
pub mod flip;
pub mod flip_backward;
pub mod pad;
pub mod pad_backward;
pub mod permute;
pub mod permute_backward;
pub mod repeat;
pub mod repeat_backward;
pub mod roll;
pub mod roll_backward;
pub mod tril;
pub mod tril_backward;
pub mod triu;
pub mod triu_backward;
pub mod write_slice;

pub use concat::{ConcatArgs, ConcatDescriptor, ConcatPlan};
pub use contiguize::{ContiguizeArgs, ContiguizeDescriptor, ContiguizePlan};
pub use fill::{FillArgs, FillDescriptor, FillPlan};
pub use concat_backward::{
    ConcatBackwardArgs, ConcatBackwardDescriptor, ConcatBackwardPlan,
};
pub use flip::{FlipArgs, FlipDescriptor, FlipPlan};
pub use flip_backward::{FlipBackwardArgs, FlipBackwardDescriptor, FlipBackwardPlan};
pub use pad::{PadArgs, PadDescriptor, PadPlan};
pub use pad_backward::{PadBackwardArgs, PadBackwardDescriptor, PadBackwardPlan};
pub use permute::{PermuteArgs, PermuteDescriptor, PermutePlan};
pub use permute_backward::{
    PermuteBackwardArgs, PermuteBackwardDescriptor, PermuteBackwardPlan,
};
pub use repeat::{RepeatArgs, RepeatDescriptor, RepeatPlan};
pub use repeat_backward::{RepeatBackwardArgs, RepeatBackwardDescriptor, RepeatBackwardPlan};
pub use roll::{RollArgs, RollDescriptor, RollPlan};
pub use roll_backward::{RollBackwardArgs, RollBackwardDescriptor, RollBackwardPlan};
pub use tril::{TrilArgs, TrilDescriptor, TrilPlan};
pub use tril_backward::{TrilBackwardArgs, TrilBackwardDescriptor, TrilBackwardPlan};
pub use triu::{TriuArgs, TriuDescriptor, TriuPlan};
pub use triu_backward::{TriuBackwardArgs, TriuBackwardDescriptor, TriuBackwardPlan};
pub use write_slice::{WriteSliceArgs, WriteSliceDescriptor, WriteSlicePlan};
