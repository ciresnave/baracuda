//! GEMM family — unified plan-based API.
//!
//! Today this module hosts the integer GEMM dispatcher [`IntGemmPlan`],
//! which routes:
//!
//! - [`LayoutSku::Rcr`] → `baracuda-cutlass`'s CUTLASS-based int8
//!   kernels (`IntGemmPlan<T, BT>` over RCR with the five-epilogue
//!   bias family).
//! - [`LayoutSku::Rrr`] → bespoke `mma.sync.m16n8k32` kernels in
//!   `baracuda-kernels-sys`. RRR coverage starts with `S8 × Identity`
//!   (this commit) and grows out the rest of the 18-SKU matrix in
//!   subsequent commits.
//!
//! Callers see a single `IntGemmPlan` type with one `select` / `run`
//! contract; the per-layout backend is observable via [`IntGemmPlan::sku`]
//! for telemetry but doesn't leak into the call signature.

pub mod int_gemm;

pub use int_gemm::{IntGemmArgs, IntGemmDescriptor, IntGemmPlan};
