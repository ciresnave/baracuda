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

pub mod bin_gemm;
pub mod fp8_gemm;
pub mod int4_gemm;
pub mod int_gemm;

// Phase 54 — 2:4 Structured Sparsity GEMM (xFormers algorithmic-
// reference hand-port). Plan file always compiles; FFI calls inside
// `run()` are `#[cfg(feature = "xformers_sparse24")]`-gated so the
// public API surface exists even without the feature.
pub mod sparse24;

pub use bin_gemm::{BinGemmArgs, BinGemmDescriptor, BinGemmPlan};
pub use fp8_gemm::{Fp8GemmArgs, Fp8GemmDescriptor, Fp8GemmPlan};
pub use int4_gemm::{Int4GemmArgs, Int4GemmDescriptor, Int4GemmPlan};
pub use int_gemm::{IntGemmArgs, IntGemmDescriptor, IntGemmPlan};
pub use sparse24::{GemmSparse24Args, GemmSparse24Descriptor, GemmSparse24Plan};
