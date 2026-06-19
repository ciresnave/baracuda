//! # baracuda-kernelgen
//!
//! Build-time generator that turns an op's **abstract IR** (the algorithm) plus
//! a [`baracuda_kernels_types::StructureKey`] cell (the schedule) into a
//! specialized CUDA kernel — and, next, its FKC contract.
//!
//! Op logic is described as a backend-agnostic [`ir::ScalarExpr`] DAG rather
//! than written as opaque CUDA, precisely so the emitter can *see the dataflow*
//! and transform it: vectorize (splat the expression across lanes), hoist
//! broadcast loads, fuse, and later retarget to other backends. A hand-written
//! CUDA escape hatch is reserved for bespoke ops the IR can't yet express.
//!
//! This is a dev/build tool (`publish = false`); the `.cu` / `.fkc` artifacts it
//! emits are committed and ship inside `baracuda-kernels-sys`.
//!
//! ## Status (v1)
//!
//! Pilot scope: **f32 elementwise** ops, contiguous operands, emitted vectorized
//! (`float4`) when the structure cell says V4 and scalar otherwise. Other
//! dtypes, strided / broadcast schedules, reductions, and FKC-contract emission
//! are the immediate next expansions — the IR and emitter are shaped to grow
//! into them.

pub mod emit;
pub mod ir;

pub use emit::{generate, GeneratedKernel};
pub use ir::{input, Access, Expr, OpDef, ScalarExpr};
