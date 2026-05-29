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

// Phase 48 — Marlin + AWQ 4-bit GEMM (vendored). Both plan files
// always compile so the public API surface is stable; FFI calls in
// `run()` are `#[cfg(feature = "marlin")]` / `#[cfg(feature = "awq")]`
// gated. `gptq_to_marlin` is a pure-Rust host-side repack utility
// (no GPU dependency) and compiles unconditionally under the
// `marlin` feature.
pub mod int4_marlin;
pub mod int4_awq;
pub mod gptq_to_marlin;

pub use bin_gemm::{BinGemmArgs, BinGemmDescriptor, BinGemmPlan};
pub use fp8_gemm::{Fp8GemmArgs, Fp8GemmDescriptor, Fp8GemmPlan};
pub use int4_gemm::{Int4GemmArgs, Int4GemmDescriptor, Int4GemmPlan};
pub use int_gemm::{IntGemmArgs, IntGemmDescriptor, IntGemmPlan};
pub use sparse24::{GemmSparse24Args, GemmSparse24Descriptor, GemmSparse24Plan};

// Phase 48 re-exports.
pub use int4_marlin::{
    Int4MarlinGemmArgs, Int4MarlinGemmDescriptor, Int4MarlinGemmPlan, MarlinActivation,
};
pub use int4_awq::{
    AwqActivation, Int4AwqGemmArgs, Int4AwqGemmDescriptor, Int4AwqGemmPlan,
};
pub use gptq_to_marlin::{
    repack as gptq_to_marlin_repack, GptqWeights, MarlinWeights, MARLIN_PERM_LEN,
    MARLIN_SCALE_PERM_LEN,
};
