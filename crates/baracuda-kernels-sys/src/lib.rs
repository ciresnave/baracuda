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
//! [`baracuda-kernels`]: https://docs.rs/baracuda-kernels

#![no_std]

use core::ffi::c_void;

// ============================================================================
// int8 GEMM — RRR layout, sm_80 (Phase 1)
// ============================================================================
//
// Layout convention `RRR`:
//   A: row-major  [M, K]  leading dimension `lda` along K
//   B: row-major  [K, N]  leading dimension `ldb` along N
//   C: row-major  [M, N]  leading dimension `ldc` along N (optional;
//                                                    pass null + beta = 0 to skip)
//   D: row-major  [M, N]  leading dimension `ldd` along N (always written)
//
// Accumulator: int32 via `mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32`.
// Epilogue: f32 alpha/beta on the int32 accum → saturating cast to s8 on
// store. Identity epilogue only in this SKU; bias variants follow in
// `gemm_s8_rrr_sm80_bias.cu` (later session).

#[cfg(feature = "sm80")]
unsafe extern "C" {
    /// `S8` GEMM, RRR layout, Identity epilogue, sm_80.
    ///
    /// # Safety
    /// All pointer args must be device-resident (or null where allowed) and
    /// remain valid for the duration of the launch. `stream` must be a live
    /// CUDA stream in the current context.
    pub fn baracuda_kernels_gemm_s8_rrr_sm80_run(
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *mut c_void,
        ldd: i64,
        alpha: f32,
        beta: f32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace size in bytes for the `S8` RRR sm_80 Identity SKU at
    /// the given problem size. Always returns zero today; reserved for
    /// future SKUs that need scratch.
    pub fn baracuda_kernels_gemm_s8_rrr_sm80_workspace_size(
        m: i32,
        n: i32,
        k: i32,
    ) -> usize;

    /// Pre-launch implementability check for the `S8` RRR sm_80
    /// Identity SKU.
    ///
    /// Returns `0` when the kernel can launch with the given shape and
    /// leading dimensions; non-zero with the standard status-code
    /// mapping otherwise. Does not launch a kernel and does not require
    /// a stream.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `*_run`
    /// function, but no device dereferences occur — only host-side
    /// shape checks.
    pub fn baracuda_kernels_gemm_s8_rrr_sm80_can_implement(
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        b: *const c_void,
        ldb: i64,
        c: *const c_void,
        ldc: i64,
        d: *const c_void,
        ldd: i64,
    ) -> i32;
}
