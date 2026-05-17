//! Random / sampling op family — Phase 4.5 (Category Q).
//!
//! Plan-per-op-family because the random ops have heterogeneous arg
//! shapes:
//!
//! - [`RandomPlan`] — pure-generator ops with no input tensor. Today
//!   wires `Uniform`, `Normal` (f32 / f64 via cuRAND) and `Bernoulli`
//!   (Bool output via cuRAND uniform + custom threshold kernel).
//!
//! - [`DropoutPlan`] / [`DropoutBackwardPlan`] — dropout takes an input
//!   tensor and returns both the output and the saved mask; backward
//!   replays the mask against `dy`. f32 / f64 only.
//!
//! Multinomial / Randint / exponential / gamma / quasi-random / stateful
//! RNG replay are reserved for future milestones (see the
//! `~/.claude/plans/warm-prancing-comet.md` Phase 4 deferral list).
//!
//! ## Generator lifetime
//!
//! cuRAND generators are stateful objects that bind to a CUDA stream.
//! Each `*Plan` creates one lazily on first `run` (the generator API
//! requires a live CUDA context, which `select()` cannot rely on having
//! at construction time) and rebinds it via `curandSetStream` on every
//! call so the plan is reusable across streams. The handle is destroyed
//! on `Drop`. cuRAND generators are *not* thread-safe — the plan is
//! `!Sync` by virtue of the `Cell<curandGenerator_t>` it holds.

pub mod dropout;
pub mod plan;

pub use dropout::{
    DropoutArgs, DropoutBackwardArgs, DropoutBackwardDescriptor, DropoutBackwardPlan,
    DropoutDescriptor, DropoutPlan,
};
pub use plan::{RandomArgs, RandomBoolArgs, RandomDescriptor, RandomPlan};
