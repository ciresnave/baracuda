//! # baracuda-kernels-sys
//!
//! Raw `extern "C"` entry points for compiled bespoke kernels.
//! **You almost certainly want [`baracuda-kernels`] instead** — that
//! crate wraps these unsafe calls with typed plans, lifetime-checked
//! device buffers, and a proper Rust API.
//!
//! Functions in this crate take raw `void*` pointers, integer
//! dimensions, and a `cudaStream_t` cast as `*mut c_void`. They are
//! unsafe because:
//!
//! - They dereference the pointer arguments without bounds-checking.
//! - They assume the pointers are valid device addresses.
//! - They assume the workspace pointer (when non-null) points to at
//!   least `workspace_bytes` of writable device memory.
//! - They assume the stream is a valid CUDA stream owned by the calling
//!   thread's current context.
//!
//! ## Status codes
//!
//! All `*_run` and `*_can_implement` functions return an [`i32`] status:
//! - `0`: success.
//! - `1`: misaligned operand.
//! - `2`: invalid problem (e.g. M, N, or K is non-positive).
//! - `3`: not supported (this kernel doesn't implement the requested shape).
//! - `4`: workspace too small or null when required.
//! - `5`: internal kernel error (typically a launch failure).
//!
//! ## Status
//!
//! Phase 0 scaffolding: this crate currently contains zero kernel
//! entry points. The first kernels (int8 GEMM RRR, Phase 1) land in
//! workspace alpha.16.
//!
//! [`baracuda-kernels`]: https://docs.rs/baracuda-kernels

#![no_std]

// Phase 1 kernel signatures will be declared inside this module as
// individual `unsafe extern "C" { ... }` blocks gated on
// `feature = "sm80"` (etc.). Today the module is empty — the build
// system (kernels/, build.rs, include/) is already in place so the
// first kernel can land in a single follow-up commit.
