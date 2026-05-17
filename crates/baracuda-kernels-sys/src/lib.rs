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

    /// `U8` GEMM, RRR layout, Identity epilogue, sm_80.
    ///
    /// Identical shape to the S8 variant; differs only in the
    /// MMA operand encoding (`.u8.u8`) and the saturating cast back
    /// to `u8` on store.
    ///
    /// # Safety
    /// Same pointer-validity contract as the S8 entry point.
    pub fn baracuda_kernels_gemm_u8_rrr_sm80_run(
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

    pub fn baracuda_kernels_gemm_u8_rrr_sm80_workspace_size(
        m: i32,
        n: i32,
        k: i32,
    ) -> usize;

    pub fn baracuda_kernels_gemm_u8_rrr_sm80_can_implement(
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

// ============================================================================
// int8 GEMM — RRR layout, sm_80, bias + activation epilogue family
// ============================================================================
//
// Eight launchers per element type: `{Bias, BiasRelu, BiasGelu, BiasSilu}`
// × `{f32, i32}` bias. The `bias` argument is an `[N]` device pointer of
// the indicated element type; it is broadcast across rows of D.
//
// All other arguments match the Identity launcher above.

#[cfg(feature = "sm80")]
unsafe extern "C" {
    // -------- S8, f32 bias --------
    pub fn baracuda_kernels_gemm_s8_rrr_sm80_bias_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s8_rrr_sm80_bias_relu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s8_rrr_sm80_bias_gelu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s8_rrr_sm80_bias_silu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // -------- S8, i32 bias --------
    pub fn baracuda_kernels_gemm_s8_rrr_sm80_bias_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s8_rrr_sm80_bias_relu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s8_rrr_sm80_bias_gelu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s8_rrr_sm80_bias_silu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // -------- U8, f32 bias --------
    pub fn baracuda_kernels_gemm_u8_rrr_sm80_bias_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u8_rrr_sm80_bias_relu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u8_rrr_sm80_bias_gelu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u8_rrr_sm80_bias_silu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // -------- U8, i32 bias --------
    pub fn baracuda_kernels_gemm_u8_rrr_sm80_bias_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u8_rrr_sm80_bias_relu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u8_rrr_sm80_bias_gelu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u8_rrr_sm80_bias_silu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// FP8 GEMM — sm_89, full 20-SKU matrix
// ============================================================================
//
// SKU matrix: {E4M3, E5M2} × {RCR, RRR} × {Identity, Bias, BiasRelu,
// BiasGelu, BiasSilu} = 20 SKUs.
//
// Layout conventions:
//
//   `RCR`:
//     A: row-major  [M, K]  leading dimension `lda` along K
//     B: col-major  [K, N]  leading dimension `ldb` along K
//     C: row-major  [M, N]  leading dimension `ldc` along N (optional)
//     D: row-major  [M, N]  leading dimension `ldd` along N
//
//   `RRR`:
//     A: row-major  [M, K]  leading dimension `lda` along K
//     B: row-major  [K, N]  leading dimension `ldb` along N
//     C: row-major  [M, N]  leading dimension `ldc` along N (optional)
//     D: row-major  [M, N]  leading dimension `ldd` along N
//
// Tensor-core path: `mma.sync.aligned.m16n8k32.row.col.f32.{e4m3|e5m2}.
// {e4m3|e5m2}.f32`. Accumulator is F32; the epilogue casts to the
// output FP8 encoding with NVIDIA's `__NV_SATFINITE` semantics
// (round-half-to-even, clamp |x| to E4M3 max-finite 448.0 / E5M2
// max-finite 57344.0).
//
// Identity SKUs ship 3 fns each (`_run`, `_workspace_size`,
// `_can_implement`); bias-family SKUs share the Identity SKU's
// workspace_size + can_implement (their kernel shape is identical),
// so they ship only the `_run` fn and take an extra `bias` argument.
//
// Status codes are shared with the int-GEMM entry points (see
// crate-level doc).

#[cfg(feature = "sm89")]
unsafe extern "C" {
    // -------- Identity: E4M3 × RCR (Phase 2 trailblazer) --------

    /// FP8 E4M3 GEMM, RCR layout, Identity epilogue, sm_89.
    ///
    /// # Safety
    /// All pointer args must be device-resident (or null where allowed) and
    /// remain valid for the duration of the launch. `stream` must be a live
    /// CUDA stream in the current context. Operand bytes are interpreted
    /// as E4M3 storage (`__nv_fp8_storage_t`); no host-side validation is
    /// performed.
    pub fn baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;
    pub fn baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *const c_void, ldd: i64,
    ) -> i32;

    // -------- Identity: E4M3 × RRR --------

    /// FP8 E4M3 GEMM, RRR layout, Identity epilogue, sm_89.
    pub fn baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;
    pub fn baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *const c_void, ldd: i64,
    ) -> i32;

    // -------- Identity: E5M2 × RCR --------

    /// FP8 E5M2 GEMM, RCR layout, Identity epilogue, sm_89.
    pub fn baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;
    pub fn baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *const c_void, ldd: i64,
    ) -> i32;

    // -------- Identity: E5M2 × RRR --------

    /// FP8 E5M2 GEMM, RRR layout, Identity epilogue, sm_89.
    pub fn baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;
    pub fn baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *const c_void, ldd: i64,
    ) -> i32;

    // -------- Bias family: E4M3 × RCR --------

    pub fn baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_bias_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_bias_relu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_bias_gelu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e4m3_rcr_sm89_bias_silu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // -------- Bias family: E4M3 × RRR --------

    pub fn baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_relu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_gelu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e4m3_rrr_sm89_bias_silu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // -------- Bias family: E5M2 × RCR --------

    pub fn baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_bias_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_bias_relu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_bias_gelu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e5m2_rcr_sm89_bias_silu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // -------- Bias family: E5M2 × RRR --------

    pub fn baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_bias_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_bias_relu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_bias_gelu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_fp8_e5m2_rrr_sm89_bias_silu_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void,   ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// int4 GEMM — sm_89, S4 RCR Identity trailblazer
// ============================================================================
//
// Phase 2 int4 trailblazer (alpha.17). The S4 RCR Identity SKU proves
// the packed-storage path (two int4 per byte, low-nibble = even index,
// high-nibble = odd index along the K axis for A/B and along the N
// axis for D output) and the `mma.sync.aligned.m16n8k64.row.col.
// satfinite.s32.s4.s4.s32` PTX intrinsic. The U4 / RRR / bias-family
// variants follow in subsequent fanout commits.
//
// Layout convention `RCR`:
//
//   A: row-major  [M, K], leading dimension `lda_bytes` along K (= K/2
//                         storage bytes per row when there's no padding)
//   B: col-major  [K, N], leading dimension `ldb_bytes` along K (= K/2
//                         storage bytes per column when there's no padding)
//   C: row-major  [M, N], leading dimension `ldc_bytes` along N (= N/2
//                         storage bytes per row; optional — pass null
//                         + beta = 0 to skip)
//   D: row-major  [M, N], leading dimension `ldd_bytes` along N (= N/2
//                         storage bytes per row; always written)
//
// `M`, `N`, `K` are **element** counts; `lda_bytes` / `ldb_bytes` /
// `ldc_bytes` / `ldd_bytes` are **byte** counts (= storage-slot counts;
// the kernel walks byte arithmetic internally). Both `K` and `N` must
// be even (the packing is byte-aligned at K for A/B and at N for D);
// odd `K` or `N` returns status code 3.
//
// Tensor-core path: `mma.sync.aligned.m16n8k64.row.col.satfinite.s32.
// s4.s4.s32`. Accumulator is S32; the epilogue applies `f32 * alpha + f32
// * beta * dequant(C)` then saturating-casts back to S4 with round-
// half-to-even and clamp to `[-8, +7]`.

#[cfg(feature = "sm89")]
unsafe extern "C" {
    /// S4 GEMM, RCR layout, Identity epilogue, sm_89.
    ///
    /// `lda_bytes` / `ldb_bytes` / `ldc_bytes` / `ldd_bytes` are in
    /// **bytes** (= packed-pair storage slot counts).
    ///
    /// # Safety
    /// All pointer args must be device-resident (or null where allowed)
    /// and remain valid for the duration of the launch. `stream` must
    /// be a live CUDA stream in the current context. Operand bytes are
    /// interpreted as packed-pair int4 storage (low nibble = even index,
    /// high nibble = odd index along the K axis for A/B and along the
    /// N axis for C/D); no host-side validation is performed.
    pub fn baracuda_kernels_gemm_s4_rcr_sm89_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rcr_sm89_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;
    pub fn baracuda_kernels_gemm_s4_rcr_sm89_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *const c_void, ldd_bytes: i64,
    ) -> i32;

    /// U4 GEMM, RCR layout, Identity epilogue, sm_89.
    ///
    /// Identical shape to the S4 variant; differs only in the MMA
    /// operand encoding (`.u4.u4`) and the saturating cast back to u4
    /// (clamp `[0, 15]`).
    ///
    /// # Safety
    /// Same pointer-validity contract as the S4 entry point. Operand
    /// bytes are interpreted as packed-pair u4 storage (low nibble +
    /// high nibble); no host-side validation is performed.
    pub fn baracuda_kernels_gemm_u4_rcr_sm89_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rcr_sm89_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;
    pub fn baracuda_kernels_gemm_u4_rcr_sm89_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *const c_void, ldd_bytes: i64,
    ) -> i32;

    /// S4 GEMM, RRR layout, Identity epilogue, sm_89.
    ///
    /// `B` is row-major `[K, N]` pair-packed along N. The kernel
    /// gathers two nibbles from two gmem K-rows to assemble one
    /// packed-pair smem byte per output column (see header comment in
    /// `baracuda_int4_rrr_sm89.cuh`).
    ///
    /// # Safety
    /// Same pointer-validity contract as the S4 RCR entry point.
    pub fn baracuda_kernels_gemm_s4_rrr_sm89_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rrr_sm89_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;
    pub fn baracuda_kernels_gemm_s4_rrr_sm89_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *const c_void, ldd_bytes: i64,
    ) -> i32;

    /// U4 GEMM, RRR layout, Identity epilogue, sm_89.
    pub fn baracuda_kernels_gemm_u4_rrr_sm89_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rrr_sm89_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;
    pub fn baracuda_kernels_gemm_u4_rrr_sm89_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *const c_void, ldd_bytes: i64,
    ) -> i32;
}

// ============================================================================
// int4 GEMM — sm_89, bias + activation epilogue family
// ============================================================================
//
// 32 launchers covering `{S4, U4} × {RCR, RRR} × {Bias, BiasRelu,
// BiasGelu, BiasSilu} × {f32 bias, i32 bias}`. The `bias` argument is
// an `[N]` device pointer of the indicated element type; it is
// broadcast across rows of D. All other arguments match the Identity
// int4 launchers above.
//
// The kernel body is identical to the Identity case — only the
// epilogue chain (bias-add → optional scalar activation → saturating
// cast back to int4) varies. `_workspace_size` and `_can_implement`
// are shared with the Identity SKU of the same `(element, layout)`
// pair (call the Identity entry points for those).

#[cfg(feature = "sm89")]
unsafe extern "C" {
    // -------- S4 × RCR --------
    pub fn baracuda_kernels_gemm_s4_rcr_sm89_bias_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rcr_sm89_bias_relu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rcr_sm89_bias_gelu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rcr_sm89_bias_silu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rcr_sm89_bias_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rcr_sm89_bias_relu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rcr_sm89_bias_gelu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rcr_sm89_bias_silu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // -------- U4 × RCR --------
    pub fn baracuda_kernels_gemm_u4_rcr_sm89_bias_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rcr_sm89_bias_relu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rcr_sm89_bias_gelu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rcr_sm89_bias_silu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rcr_sm89_bias_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rcr_sm89_bias_relu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rcr_sm89_bias_gelu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rcr_sm89_bias_silu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // -------- S4 × RRR --------
    pub fn baracuda_kernels_gemm_s4_rrr_sm89_bias_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rrr_sm89_bias_relu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rrr_sm89_bias_gelu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rrr_sm89_bias_silu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rrr_sm89_bias_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rrr_sm89_bias_relu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rrr_sm89_bias_gelu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_s4_rrr_sm89_bias_silu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // -------- U4 × RRR --------
    pub fn baracuda_kernels_gemm_u4_rrr_sm89_bias_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rrr_sm89_bias_relu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rrr_sm89_bias_gelu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rrr_sm89_bias_silu_f32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rrr_sm89_bias_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rrr_sm89_bias_relu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rrr_sm89_bias_gelu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_u4_rrr_sm89_bias_silu_i32_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        c: *const c_void, ldc_bytes: i64,
        d: *mut c_void,   ldd_bytes: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Binary (B1) GEMM — sm_89 (Identity-only, RCR layout)
// ============================================================================
//
// Distinct programming model: `D[i,j] = sum_k popcount(A[i, k_byte] XOR
// B[k_byte, j])` (raw int32 accumulator, no re-quantization back to b1
// and no α/β/bias/activation chain).
//
// Layout convention `RCR`:
//
//   A: row-major  [M, K bits], leading dimension `lda_bytes` along K
//                              (= K/8 storage bytes per row)
//   B: col-major  [K, N bits], leading dimension `ldb_bytes` along K
//                              (= K/8 storage bytes per column)
//   D: row-major  [M, N i32],  leading dimension `ldd_elements` along N
//                              (int32 element count, NOT bytes — D is a
//                              plain int32 matrix with no packing)
//
// `M`, `N`, `K` are **element** counts; `lda_bytes` / `ldb_bytes` are
// **byte** counts; `ldd_elements` is in **i32 element** count. `K` must
// be divisible by 8 (packing is byte-aligned). No constraint on N
// (output is plain int32).
//
// Tensor-core path:
// `mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.xor.popc`.

#[cfg(feature = "sm89")]
unsafe extern "C" {
    /// Binary (B1) GEMM, RCR layout, Identity epilogue, sm_89.
    ///
    /// `ldd_elements` is in **i32 element count**, not bytes — the D
    /// output is a plain `int32_t[M, N]` matrix with no packing. A/B
    /// `ld` values are in bytes (= packed-bit storage slots).
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. Operand bytes are interpreted as packed-bit
    /// B1 storage (LSB = lowest K index within each byte); no host-side
    /// validation is performed. The `d` buffer must hold at least
    /// `M * ldd_elements` `int32_t` elements.
    pub fn baracuda_kernels_gemm_bin_rcr_sm89_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        d: *mut c_void,   ldd_elements: i64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_bin_rcr_sm89_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;
    pub fn baracuda_kernels_gemm_bin_rcr_sm89_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        d: *const c_void, ldd_elements: i64,
    ) -> i32;

    /// Binary (B1) GEMM, RRR layout, Identity epilogue, sm_89.
    ///
    /// Distinct from the RCR variant in that `B` is row-major and
    /// bit-packed along N in gmem (the kernel re-packs into K-bit-
    /// packed smem via a bit-gather load). Same int32 output
    /// convention as RCR — `ldd_elements` is in i32 element count.
    ///
    /// Requires both `K` and `N` to be divisible by 8.
    ///
    /// # Safety
    /// Same pointer-validity contract as the RCR entry point.
    pub fn baracuda_kernels_gemm_bin_rrr_sm89_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        d: *mut c_void,   ldd_elements: i64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    pub fn baracuda_kernels_gemm_bin_rrr_sm89_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;
    pub fn baracuda_kernels_gemm_bin_rrr_sm89_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda_bytes: i64,
        b: *const c_void, ldb_bytes: i64,
        d: *const c_void, ldd_elements: i64,
    ) -> i32;
}

// ============================================================================
// Elementwise — Phase 3 trailblazer (binary add, FP family)
// ============================================================================
//
// Contiguous pointwise binary kernels. Inputs / output are arbitrary-
// rank tensors flattened to a single `numel` element count on the FFI
// boundary (the Rust plan layer collapses contiguous shapes for the
// "all-contig 1D sweep" fast path).
//
// ABI:
//   numel               — i64 element count (product of shape).
//   a / b               — input device pointers (T const*).
//   y                   — output device pointer (T*). Aliasing with
//                         either input is fine for the all-contig
//                         fast path (the kernel reads each i once
//                         before writing each i once).
//   workspace / bytes   — unused for elementwise; pass null + 0 from
//                         Rust. Carried in the signature for ABI
//                         parity with the GEMM family.
//   stream              — cudaStream_t cast to `*mut c_void`.
//
// Status codes are shared with the GEMM family (see crate-level doc).

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Binary elementwise `add`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `float`s of device memory.
    pub fn baracuda_kernels_binary_add_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_add_f32`. Validates
    /// the problem size without launching a kernel. Returns the standard
    /// status code mapping.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_add_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `sub`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `float`s of device memory.
    pub fn baracuda_kernels_binary_sub_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_sub_f32`. Validates
    /// the problem size without launching a kernel. Returns the standard
    /// status code mapping.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_sub_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `mul`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `float`s of device memory.
    pub fn baracuda_kernels_binary_mul_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_mul_f32`. Validates
    /// the problem size without launching a kernel. Returns the standard
    /// status code mapping.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_mul_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `div`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `float`s of device memory.
    pub fn baracuda_kernels_binary_div_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_div_f32`. Validates
    /// the problem size without launching a kernel. Returns the standard
    /// status code mapping.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_div_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ------------------------------------------------------------------
    // dtype fanout — f16 / bf16 / f64 siblings of the four f32 launchers
    // above. ABI is identical to the f32 variants (the dtype is encoded
    // only in the symbol name); see those decls for ABI contract docs.
    // ------------------------------------------------------------------

    /// Binary elementwise `add`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `__half`s of device memory.
    pub fn baracuda_kernels_binary_add_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_add_f16`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_add_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `add`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `__nv_bfloat16`s of device memory.
    pub fn baracuda_kernels_binary_add_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_add_bf16`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_add_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `add`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `double`s of device memory.
    pub fn baracuda_kernels_binary_add_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_add_f64`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_add_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `sub`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `__half`s of device memory.
    pub fn baracuda_kernels_binary_sub_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_sub_f16`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_sub_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `sub`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `__nv_bfloat16`s of device memory.
    pub fn baracuda_kernels_binary_sub_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_sub_bf16`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_sub_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `sub`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `double`s of device memory.
    pub fn baracuda_kernels_binary_sub_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_sub_f64`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_sub_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `mul`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `__half`s of device memory.
    pub fn baracuda_kernels_binary_mul_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_mul_f16`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_mul_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `mul`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `__nv_bfloat16`s of device memory.
    pub fn baracuda_kernels_binary_mul_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_mul_bf16`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_mul_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `mul`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `double`s of device memory.
    pub fn baracuda_kernels_binary_mul_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_mul_f64`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_mul_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `div`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `__half`s of device memory.
    pub fn baracuda_kernels_binary_div_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_div_f16`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_div_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `div`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `__nv_bfloat16`s of device memory.
    pub fn baracuda_kernels_binary_div_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_div_bf16`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_div_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `div`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `double`s of device memory.
    pub fn baracuda_kernels_binary_div_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_div_f64`.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_binary_div_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary pow (`y = a^b`), contig fast path ------------------
    // f16 / bf16 transcend through an f32 detour inside the kernel; f32
    // uses `powf`, f64 uses `pow`. All four dtypes share the same ABI as
    // the other binary contig launchers (numel, a, b, y, ws, ws_bytes,
    // stream).

    /// Binary `pow`, f32, contig.
    pub fn baracuda_kernels_binary_pow_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `pow`, f32, can-implement.
    pub fn baracuda_kernels_binary_pow_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `pow`, f16, contig.
    pub fn baracuda_kernels_binary_pow_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `pow`, f16, can-implement.
    pub fn baracuda_kernels_binary_pow_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `pow`, bf16, contig.
    pub fn baracuda_kernels_binary_pow_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `pow`, bf16, can-implement.
    pub fn baracuda_kernels_binary_pow_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `pow`, f64, contig.
    pub fn baracuda_kernels_binary_pow_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `pow`, f64, can-implement.
    pub fn baracuda_kernels_binary_pow_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary atan2 (`y = atan2(a, b)`), contig fast path --------
    // f16 / bf16 transcend through an f32 detour inside the kernel; f32
    // uses `atan2f`, f64 uses `atan2`.

    /// Binary `atan2`, f32, contig.
    pub fn baracuda_kernels_binary_atan2_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `atan2`, f32, can-implement.
    pub fn baracuda_kernels_binary_atan2_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `atan2`, f16, contig.
    pub fn baracuda_kernels_binary_atan2_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `atan2`, f16, can-implement.
    pub fn baracuda_kernels_binary_atan2_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `atan2`, bf16, contig.
    pub fn baracuda_kernels_binary_atan2_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `atan2`, bf16, can-implement.
    pub fn baracuda_kernels_binary_atan2_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `atan2`, f64, contig.
    pub fn baracuda_kernels_binary_atan2_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `atan2`, f64, can-implement.
    pub fn baracuda_kernels_binary_atan2_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary hypot (`y = sqrt(a² + b²)`), contig fast path ------
    // f16 / bf16 transcend through an f32 detour inside the kernel; f32
    // uses `hypotf`, f64 uses `hypot`. Both libdevice intrinsics are
    // overflow-/underflow-safe (internally rescale by max(|a|, |b|)).

    /// Binary `hypot`, f32, contig.
    pub fn baracuda_kernels_binary_hypot_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `hypot`, f32, can-implement.
    pub fn baracuda_kernels_binary_hypot_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `hypot`, f16, contig.
    pub fn baracuda_kernels_binary_hypot_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `hypot`, f16, can-implement.
    pub fn baracuda_kernels_binary_hypot_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `hypot`, bf16, contig.
    pub fn baracuda_kernels_binary_hypot_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `hypot`, bf16, can-implement.
    pub fn baracuda_kernels_binary_hypot_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `hypot`, f64, contig.
    pub fn baracuda_kernels_binary_hypot_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `hypot`, f64, can-implement.
    pub fn baracuda_kernels_binary_hypot_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary copysign (`y = copysign(a, b) = |a| · sign(b)`), contig
    // f16 / bf16 transcend through an f32 detour inside the kernel; f32
    // uses `copysignf`, f64 uses `copysign`. Pure sign-bit manipulation —
    // well-defined for every IEEE input including NaN.

    /// Binary `copysign`, f32, contig.
    pub fn baracuda_kernels_binary_copysign_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `copysign`, f32, can-implement.
    pub fn baracuda_kernels_binary_copysign_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `copysign`, f16, contig.
    pub fn baracuda_kernels_binary_copysign_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `copysign`, f16, can-implement.
    pub fn baracuda_kernels_binary_copysign_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `copysign`, bf16, contig.
    pub fn baracuda_kernels_binary_copysign_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `copysign`, bf16, can-implement.
    pub fn baracuda_kernels_binary_copysign_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `copysign`, f64, contig.
    pub fn baracuda_kernels_binary_copysign_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `copysign`, f64, can-implement.
    pub fn baracuda_kernels_binary_copysign_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary nextafter (`y = nextafter(a, b)`), contig fast path -
    // f32 → `nextafterf`, f64 → `nextafter`. f16 / bf16 use direct
    // bit-pattern manipulation (no f32 detour — adjacent half cells
    // round-trip through f32 to themselves, so a naive f32 detour
    // returns `a`, not its neighbor).

    /// Binary `nextafter`, f32, contig.
    pub fn baracuda_kernels_binary_nextafter_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `nextafter`, f32, can-implement.
    pub fn baracuda_kernels_binary_nextafter_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `nextafter`, f16, contig.
    pub fn baracuda_kernels_binary_nextafter_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `nextafter`, f16, can-implement.
    pub fn baracuda_kernels_binary_nextafter_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `nextafter`, bf16, contig.
    pub fn baracuda_kernels_binary_nextafter_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `nextafter`, bf16, can-implement.
    pub fn baracuda_kernels_binary_nextafter_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `nextafter`, f64, contig.
    pub fn baracuda_kernels_binary_nextafter_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `nextafter`, f64, can-implement.
    pub fn baracuda_kernels_binary_nextafter_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary fmin (`y = fmin(a, b)` IEEE 754, NaN-aware), contig --
    // Distinct from `BinaryKind::Minimum` which propagates NaN. f32 →
    // `fminf`, f64 → `fmin`, f16 / bf16 → f32-detour.

    /// Binary `fmin`, f32, contig.
    pub fn baracuda_kernels_binary_fmin_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmin`, f32, can-implement.
    pub fn baracuda_kernels_binary_fmin_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `fmin`, f16, contig.
    pub fn baracuda_kernels_binary_fmin_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmin`, f16, can-implement.
    pub fn baracuda_kernels_binary_fmin_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `fmin`, bf16, contig.
    pub fn baracuda_kernels_binary_fmin_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmin`, bf16, can-implement.
    pub fn baracuda_kernels_binary_fmin_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `fmin`, f64, contig.
    pub fn baracuda_kernels_binary_fmin_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmin`, f64, can-implement.
    pub fn baracuda_kernels_binary_fmin_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary fmax (`y = fmax(a, b)` IEEE 754, NaN-aware), contig --
    // Distinct from `BinaryKind::Maximum` which propagates NaN. f32 →
    // `fmaxf`, f64 → `fmax`, f16 / bf16 → f32-detour.

    /// Binary `fmax`, f32, contig.
    pub fn baracuda_kernels_binary_fmax_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmax`, f32, can-implement.
    pub fn baracuda_kernels_binary_fmax_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `fmax`, f16, contig.
    pub fn baracuda_kernels_binary_fmax_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmax`, f16, can-implement.
    pub fn baracuda_kernels_binary_fmax_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `fmax`, bf16, contig.
    pub fn baracuda_kernels_binary_fmax_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmax`, bf16, can-implement.
    pub fn baracuda_kernels_binary_fmax_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `fmax`, f64, contig.
    pub fn baracuda_kernels_binary_fmax_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmax`, f64, can-implement.
    pub fn baracuda_kernels_binary_fmax_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary maximum (`y = max(a, b)` NaN-PROPAGATING), contig ----
    // Distinct from `BinaryKind::Fmax` which is NaN-aware (NaN-ignored).
    // Any NaN input produces a NaN output, matching `torch.maximum`.
    // f32 / f64 → compare-and-select with explicit NaN guards;
    // f16 / bf16 → f32-detour with same NaN guard.

    /// Binary `maximum`, f32, contig.
    pub fn baracuda_kernels_binary_maximum_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `maximum`, f32, can-implement.
    pub fn baracuda_kernels_binary_maximum_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `maximum`, f16, contig.
    pub fn baracuda_kernels_binary_maximum_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `maximum`, f16, can-implement.
    pub fn baracuda_kernels_binary_maximum_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `maximum`, bf16, contig.
    pub fn baracuda_kernels_binary_maximum_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `maximum`, bf16, can-implement.
    pub fn baracuda_kernels_binary_maximum_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `maximum`, f64, contig.
    pub fn baracuda_kernels_binary_maximum_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `maximum`, f64, can-implement.
    pub fn baracuda_kernels_binary_maximum_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary minimum (`y = min(a, b)` NaN-PROPAGATING), contig ----
    // Distinct from `BinaryKind::Fmin` which is NaN-aware (NaN-ignored).
    // Any NaN input produces a NaN output, matching `torch.minimum`.

    /// Binary `minimum`, f32, contig.
    pub fn baracuda_kernels_binary_minimum_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `minimum`, f32, can-implement.
    pub fn baracuda_kernels_binary_minimum_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `minimum`, f16, contig.
    pub fn baracuda_kernels_binary_minimum_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `minimum`, f16, can-implement.
    pub fn baracuda_kernels_binary_minimum_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `minimum`, bf16, contig.
    pub fn baracuda_kernels_binary_minimum_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `minimum`, bf16, can-implement.
    pub fn baracuda_kernels_binary_minimum_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `minimum`, f64, contig.
    pub fn baracuda_kernels_binary_minimum_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `minimum`, f64, can-implement.
    pub fn baracuda_kernels_binary_minimum_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary floor_divide (`y = floor(a / b)`), contig ------------

    /// Binary `floor_divide`, f32, contig.
    pub fn baracuda_kernels_binary_floor_divide_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `floor_divide`, f32, can-implement.
    pub fn baracuda_kernels_binary_floor_divide_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `floor_divide`, f16, contig.
    pub fn baracuda_kernels_binary_floor_divide_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `floor_divide`, f16, can-implement.
    pub fn baracuda_kernels_binary_floor_divide_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `floor_divide`, bf16, contig.
    pub fn baracuda_kernels_binary_floor_divide_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `floor_divide`, bf16, can-implement.
    pub fn baracuda_kernels_binary_floor_divide_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `floor_divide`, f64, contig.
    pub fn baracuda_kernels_binary_floor_divide_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `floor_divide`, f64, can-implement.
    pub fn baracuda_kernels_binary_floor_divide_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary mod (`y = a - floor(a/b)*b`, sign of b), contig -------
    // Python-style modulo. Distinct from `BinaryKind::Remainder`, which
    // is C-style (sign of a).

    /// Binary `mod`, f32, contig.
    pub fn baracuda_kernels_binary_mod_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `mod`, f32, can-implement.
    pub fn baracuda_kernels_binary_mod_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `mod`, f16, contig.
    pub fn baracuda_kernels_binary_mod_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `mod`, f16, can-implement.
    pub fn baracuda_kernels_binary_mod_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `mod`, bf16, contig.
    pub fn baracuda_kernels_binary_mod_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `mod`, bf16, can-implement.
    pub fn baracuda_kernels_binary_mod_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `mod`, f64, contig.
    pub fn baracuda_kernels_binary_mod_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `mod`, f64, can-implement.
    pub fn baracuda_kernels_binary_mod_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // ----- Binary remainder (`y = fmod(a, b)`, sign of a), contig ------
    // C-style remainder via libdevice `fmodf` / `fmod`. Distinct from
    // `BinaryKind::Mod`, which is Python-style (sign of b).

    /// Binary `remainder`, f32, contig.
    pub fn baracuda_kernels_binary_remainder_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `remainder`, f32, can-implement.
    pub fn baracuda_kernels_binary_remainder_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `remainder`, f16, contig.
    pub fn baracuda_kernels_binary_remainder_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `remainder`, f16, can-implement.
    pub fn baracuda_kernels_binary_remainder_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `remainder`, bf16, contig.
    pub fn baracuda_kernels_binary_remainder_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `remainder`, bf16, can-implement.
    pub fn baracuda_kernels_binary_remainder_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary `remainder`, f64, contig.
    pub fn baracuda_kernels_binary_remainder_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `remainder`, f64, can-implement.
    pub fn baracuda_kernels_binary_remainder_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;
}

// ============================================================================
// Elementwise — integer / bool binary ops (contig only)
// ============================================================================
//
// Phase 3.3 integer + bool fanout. Five bitwise ops (`and` / `or` /
// `xor` / `left_shift` / `right_shift`) over `{i32, i64}` plus three
// logical ops (`and` / `or` / `xor`) over `Bool` (1-byte storage).
//
// **Contig only.** Strided / broadcast variants are deferred to a
// later milestone — the caller is expected to materialize a contiguous
// operand if it needs broadcast semantics for these op families. The
// Rust dispatcher therefore routes any non-contig launch through
// `Error::Unsupported` for these (kind, dtype) cells.
//
// Right-shift on signed integers is **arithmetic** (sign-extending),
// matching PyTorch's contract — see
// `kernels/elementwise/binary_bitwise_right_shift_int.cu` for the
// portability rationale.
//
// Logical ops normalize their inputs to canonical 0 / 1 before
// applying the boolean op so the output is always strictly 0 or 1
// even when the inputs are unnormalized byte storage.
//
// ABI mirrors the FP contig binary launchers (numel + 3 pointers +
// workspace + stream). Status codes are shared with the GEMM family
// (see crate-level doc).

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    // -------- bitwise_and --------

    /// Binary bitwise `and`, i32 dtype, contig.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for
    /// the duration of the launch. `stream` must be a live CUDA stream
    /// in the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` `int32_t`s of device memory.
    pub fn baracuda_kernels_binary_bitwise_and_i32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary bitwise `and`, i32 dtype, can-implement.
    pub fn baracuda_kernels_binary_bitwise_and_i32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary bitwise `and`, i64 dtype, contig.
    ///
    /// # Safety
    /// Same contract as
    /// `baracuda_kernels_binary_bitwise_and_i32_run`, but each tensor
    /// covers at least `numel` `int64_t`s.
    pub fn baracuda_kernels_binary_bitwise_and_i64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary bitwise `and`, i64 dtype, can-implement.
    pub fn baracuda_kernels_binary_bitwise_and_i64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // -------- bitwise_or --------

    /// Binary bitwise `or`, i32 dtype, contig.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_bitwise_and_i32_run`.
    pub fn baracuda_kernels_binary_bitwise_or_i32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary bitwise `or`, i32 dtype, can-implement.
    pub fn baracuda_kernels_binary_bitwise_or_i32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary bitwise `or`, i64 dtype, contig.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_bitwise_and_i64_run`.
    pub fn baracuda_kernels_binary_bitwise_or_i64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary bitwise `or`, i64 dtype, can-implement.
    pub fn baracuda_kernels_binary_bitwise_or_i64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // -------- bitwise_xor --------

    /// Binary bitwise `xor`, i32 dtype, contig.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_bitwise_and_i32_run`.
    pub fn baracuda_kernels_binary_bitwise_xor_i32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary bitwise `xor`, i32 dtype, can-implement.
    pub fn baracuda_kernels_binary_bitwise_xor_i32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary bitwise `xor`, i64 dtype, contig.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_bitwise_and_i64_run`.
    pub fn baracuda_kernels_binary_bitwise_xor_i64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary bitwise `xor`, i64 dtype, can-implement.
    pub fn baracuda_kernels_binary_bitwise_xor_i64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // -------- bitwise_left_shift --------

    /// Binary bitwise `left_shift`, i32 dtype, contig.
    ///
    /// `y = a << b`. Out-of-range shift amounts inherit the host
    /// architecture's behavior — callers requiring defined behavior
    /// should clamp `b` to `[0, 31]` themselves.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_bitwise_and_i32_run`.
    pub fn baracuda_kernels_binary_bitwise_left_shift_i32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary bitwise `left_shift`, i32 dtype, can-implement.
    pub fn baracuda_kernels_binary_bitwise_left_shift_i32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary bitwise `left_shift`, i64 dtype, contig.
    ///
    /// `y = a << b`. Out-of-range shift amounts inherit the host
    /// architecture's behavior — callers requiring defined behavior
    /// should clamp `b` to `[0, 63]` themselves.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_bitwise_and_i64_run`.
    pub fn baracuda_kernels_binary_bitwise_left_shift_i64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary bitwise `left_shift`, i64 dtype, can-implement.
    pub fn baracuda_kernels_binary_bitwise_left_shift_i64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // -------- bitwise_right_shift --------

    /// Binary bitwise `right_shift`, i32 dtype, contig. **Arithmetic**
    /// shift (sign-extending), matching PyTorch.
    ///
    /// `y = a >> b`. Out-of-range shift amounts inherit the host
    /// architecture's behavior.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_bitwise_and_i32_run`.
    pub fn baracuda_kernels_binary_bitwise_right_shift_i32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary bitwise `right_shift`, i32 dtype, can-implement.
    pub fn baracuda_kernels_binary_bitwise_right_shift_i32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary bitwise `right_shift`, i64 dtype, contig. **Arithmetic**
    /// shift (sign-extending), matching PyTorch.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_bitwise_and_i64_run`.
    pub fn baracuda_kernels_binary_bitwise_right_shift_i64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary bitwise `right_shift`, i64 dtype, can-implement.
    pub fn baracuda_kernels_binary_bitwise_right_shift_i64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // -------- logical_and --------

    /// Binary logical `and`, Bool dtype (1-byte storage), contig.
    ///
    /// Truthiness convention: 0 = false, any non-zero = true. The
    /// kernel normalizes each input before applying `&&`, so the output
    /// is always strictly 0 or 1.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for
    /// the duration of the launch. `stream` must be a live CUDA stream
    /// in the current context. `a`, `b`, and `y` must each point to at
    /// least `numel` bytes of device memory.
    pub fn baracuda_kernels_binary_logical_and_bool_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary logical `and`, Bool dtype, can-implement.
    pub fn baracuda_kernels_binary_logical_and_bool_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // -------- logical_or --------

    /// Binary logical `or`, Bool dtype, contig.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_logical_and_bool_run`.
    pub fn baracuda_kernels_binary_logical_or_bool_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary logical `or`, Bool dtype, can-implement.
    pub fn baracuda_kernels_binary_logical_or_bool_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    // -------- logical_xor --------

    /// Binary logical `xor`, Bool dtype, contig.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_logical_and_bool_run`.
    pub fn baracuda_kernels_binary_logical_xor_bool_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Binary logical `xor`, Bool dtype, can-implement.
    pub fn baracuda_kernels_binary_logical_xor_bool_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;
}

// ============================================================================
// Elementwise — strided / broadcast variants
// ============================================================================
//
// Companion launchers to the contig fast path above. The Rust dispatcher
// picks contig vs strided at launch time based on whether all three
// operands are fully contiguous. The strided kernel handles every
// non-contig case: broadcast (stride 0 along an axis), transposed
// views, arbitrary strided slices.
//
// ABI:
//   numel               — i64 element count of the OUTPUT tensor
//                         (product of `shape`).
//   rank                — i32, number of valid axes in [0, 8].
//   shape               — points to `[i32; rank]` on the host stack.
//                         The OUTPUT shape; operands `a` and `b` are
//                         read at the same coords via their own
//                         strides (broadcast = stride 0).
//   stride_a / b / y    — points to `[i64; rank]` on the host stack,
//                         the per-axis element stride for each tensor.
//                         A stride of 0 along axis d marks a broadcast
//                         operand. Output stride is typically
//                         contiguous but the kernel accepts arbitrary
//                         strides.
//   a / b               — input device pointers (T const*).
//   y                   — output device pointer (T*). Aliasing is
//                         safe in the contig case (i ≤ N); in the
//                         strided / broadcast case it's caller-
//                         responsibility — the kernel reads each
//                         (off_a, off_b) once before writing each
//                         off_y once, but stride-0 broadcast means
//                         many writes to the same off_y if the output
//                         is also broadcast, which is undefined.
//   workspace / bytes   — unused; pass null + 0 from Rust.
//   stream              — cudaStream_t cast to `*mut c_void`.
//
// Status codes mirror the GEMM family (see crate-level doc).

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Binary elementwise `add`, f32 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// All device pointer args must be device-resident and remain valid
    /// for the duration of the launch. `shape` / `stride_*` are
    /// host-side pointers to arrays of at least `rank` elements that
    /// remain valid for the duration of the host-side launch call
    /// (the launcher copies them into the kernel parameter block
    /// before returning — they may be freed after the host call
    /// completes, before the kernel completes on device).
    pub fn baracuda_kernels_binary_add_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `add`, f16 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_add_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `add`, bf16 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_add_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `add`, f64 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_add_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `sub`, f32 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_sub_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `sub`, f16 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_sub_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `sub`, bf16 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_sub_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `sub`, f64 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_sub_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `mul`, f32 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_mul_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `mul`, f16 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_mul_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `mul`, bf16 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_mul_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `mul`, f64 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_mul_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `div`, f32 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_div_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `div`, f16 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_div_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `div`, bf16 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_div_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `div`, f64 dtype, strided / broadcast path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_binary_add_f32_strided_run`.
    pub fn baracuda_kernels_binary_div_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `pow`, f32, strided.
    pub fn baracuda_kernels_binary_pow_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `pow`, f16, strided.
    pub fn baracuda_kernels_binary_pow_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `pow`, bf16, strided.
    pub fn baracuda_kernels_binary_pow_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `pow`, f64, strided.
    pub fn baracuda_kernels_binary_pow_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `atan2`, f32, strided.
    pub fn baracuda_kernels_binary_atan2_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `atan2`, f16, strided.
    pub fn baracuda_kernels_binary_atan2_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `atan2`, bf16, strided.
    pub fn baracuda_kernels_binary_atan2_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `atan2`, f64, strided.
    pub fn baracuda_kernels_binary_atan2_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `hypot`, f32, strided.
    pub fn baracuda_kernels_binary_hypot_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `hypot`, f16, strided.
    pub fn baracuda_kernels_binary_hypot_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `hypot`, bf16, strided.
    pub fn baracuda_kernels_binary_hypot_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `hypot`, f64, strided.
    pub fn baracuda_kernels_binary_hypot_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `copysign`, f32, strided.
    pub fn baracuda_kernels_binary_copysign_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `copysign`, f16, strided.
    pub fn baracuda_kernels_binary_copysign_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `copysign`, bf16, strided.
    pub fn baracuda_kernels_binary_copysign_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `copysign`, f64, strided.
    pub fn baracuda_kernels_binary_copysign_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `nextafter`, f32, strided.
    pub fn baracuda_kernels_binary_nextafter_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `nextafter`, f16, strided.
    pub fn baracuda_kernels_binary_nextafter_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `nextafter`, bf16, strided.
    pub fn baracuda_kernels_binary_nextafter_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `nextafter`, f64, strided.
    pub fn baracuda_kernels_binary_nextafter_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmin`, f32, strided.
    pub fn baracuda_kernels_binary_fmin_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmin`, f16, strided.
    pub fn baracuda_kernels_binary_fmin_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmin`, bf16, strided.
    pub fn baracuda_kernels_binary_fmin_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmin`, f64, strided.
    pub fn baracuda_kernels_binary_fmin_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmax`, f32, strided.
    pub fn baracuda_kernels_binary_fmax_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmax`, f16, strided.
    pub fn baracuda_kernels_binary_fmax_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmax`, bf16, strided.
    pub fn baracuda_kernels_binary_fmax_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `fmax`, f64, strided.
    pub fn baracuda_kernels_binary_fmax_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ----- Binary maximum (NaN-PROPAGATING), strided -------------------

    /// Binary `maximum`, f32, strided.
    pub fn baracuda_kernels_binary_maximum_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `maximum`, f16, strided.
    pub fn baracuda_kernels_binary_maximum_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `maximum`, bf16, strided.
    pub fn baracuda_kernels_binary_maximum_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `maximum`, f64, strided.
    pub fn baracuda_kernels_binary_maximum_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ----- Binary minimum (NaN-PROPAGATING), strided -------------------

    /// Binary `minimum`, f32, strided.
    pub fn baracuda_kernels_binary_minimum_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `minimum`, f16, strided.
    pub fn baracuda_kernels_binary_minimum_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `minimum`, bf16, strided.
    pub fn baracuda_kernels_binary_minimum_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `minimum`, f64, strided.
    pub fn baracuda_kernels_binary_minimum_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ----- Binary floor_divide, strided --------------------------------

    /// Binary `floor_divide`, f32, strided.
    pub fn baracuda_kernels_binary_floor_divide_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `floor_divide`, f16, strided.
    pub fn baracuda_kernels_binary_floor_divide_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `floor_divide`, bf16, strided.
    pub fn baracuda_kernels_binary_floor_divide_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `floor_divide`, f64, strided.
    pub fn baracuda_kernels_binary_floor_divide_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ----- Binary mod (Python-style, sign of b), strided ---------------

    /// Binary `mod`, f32, strided.
    pub fn baracuda_kernels_binary_mod_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `mod`, f16, strided.
    pub fn baracuda_kernels_binary_mod_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `mod`, bf16, strided.
    pub fn baracuda_kernels_binary_mod_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `mod`, f64, strided.
    pub fn baracuda_kernels_binary_mod_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ----- Binary remainder (C-style, sign of a), strided --------------

    /// Binary `remainder`, f32, strided.
    pub fn baracuda_kernels_binary_remainder_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `remainder`, f16, strided.
    pub fn baracuda_kernels_binary_remainder_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `remainder`, bf16, strided.
    pub fn baracuda_kernels_binary_remainder_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary `remainder`, f64, strided.
    pub fn baracuda_kernels_binary_remainder_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Elementwise — ternary (3→1) ops
// ============================================================================
//
// 3-input, 1-output pointwise ops with same-dtype operands. Same
// INSTANTIATE-driven kernel family as binary, with one extra input
// (`c`) and one extra stride array (`stride_c`) for the strided path.
//
// Wired matrix: {Clamp, Fma} × {f32, f16, bf16, f64} = 8 cells, each
// with contig + strided launchers (3 symbols per cell). {Addcmul,
// Addcdiv} are reserved-but-deferred — they take a scalar runtime
// parameter not yet representable in the ternary plan shape.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Ternary elementwise `clamp`, f32, contig fast path.
    ///
    /// `y = min(max(a, b), c)` where `a` is the input, `b` is the lower
    /// bound, `c` is the upper bound — matches PyTorch's
    /// `torch.clamp(x, min=lo, max=hi)` semantics with `a = x`, `b = lo`,
    /// `c = hi`. The caller is responsible for `lo <= hi`; if not, the
    /// output is `hi` (PyTorch's convention: max wins).
    ///
    /// # Safety
    /// All device pointers must remain valid for the duration of the
    /// launch. `a`, `b`, `c`, `y` must each point to at least `numel`
    /// `float`s. Aliasing `y` with any input is safe — the kernel reads
    /// each input cell before writing each output cell per thread.
    pub fn baracuda_kernels_ternary_clamp_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `ternary_clamp_f32`.
    pub fn baracuda_kernels_ternary_clamp_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Ternary elementwise `clamp`, f32, strided / broadcast path.
    ///
    /// Handles non-contig views and broadcast — each input's
    /// per-axis stride may be 0 (broadcast along that axis) or any
    /// integer (transposed / sliced view). The PyTorch convention
    /// `clamp(x, min=lo, max=hi)` typically has `lo` / `hi` as scalars
    /// — represent them as rank-N tensors with `shape[d] = 1` and
    /// `stride[d] = 0` on every axis.
    pub fn baracuda_kernels_ternary_clamp_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_c: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Ternary elementwise `clamp`, f16, contig fast path.
    ///
    /// See `baracuda_kernels_ternary_clamp_f32_run`. Inputs and output
    /// are `__half`; the kernel applies a single f32-detour
    /// `fminf(fmaxf(...))` per cell — matches host
    /// `half::f16` round-to-f32-min/max-round-back-to-f16 semantics.
    pub fn baracuda_kernels_ternary_clamp_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `ternary_clamp_f16`.
    pub fn baracuda_kernels_ternary_clamp_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Ternary elementwise `clamp`, f16, strided / broadcast path.
    pub fn baracuda_kernels_ternary_clamp_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_c: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Ternary elementwise `clamp`, bf16, contig fast path.
    ///
    /// See `baracuda_kernels_ternary_clamp_f32_run`. Same f32-detour
    /// pipeline as the f16 variant but with `__nv_bfloat16` storage.
    pub fn baracuda_kernels_ternary_clamp_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `ternary_clamp_bf16`.
    pub fn baracuda_kernels_ternary_clamp_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Ternary elementwise `clamp`, bf16, strided / broadcast path.
    pub fn baracuda_kernels_ternary_clamp_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_c: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Ternary elementwise `clamp`, f64, contig fast path.
    ///
    /// See `baracuda_kernels_ternary_clamp_f32_run`. Inputs and output
    /// are `double`; uses `fmin(fmax(...))` directly (no detour).
    pub fn baracuda_kernels_ternary_clamp_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `ternary_clamp_f64`.
    pub fn baracuda_kernels_ternary_clamp_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Ternary elementwise `clamp`, f64, strided / broadcast path.
    pub fn baracuda_kernels_ternary_clamp_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_c: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// --- Fma --------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Ternary elementwise `fma`, f32, contig fast path.
    ///
    /// `y = a * b + c` — computed as two separate rounding steps
    /// (multiply then add), NOT the IEEE single-rounding fma. This
    /// matches PyTorch's `torch.addcmul(c, a, b)` with implicit
    /// `value=1` and gives bit-exact compare with the host reference's
    /// `a * b + c` on f32 / f64. The f16 / bf16 variants follow the
    /// usual f32-detour pattern (each scalar op promotes to f32, runs
    /// once, rounds back).
    ///
    /// # Safety
    /// All device pointers must remain valid for the duration of the
    /// launch. `a`, `b`, `c`, `y` must each point to at least `numel`
    /// `float`s. Aliasing `y` with any input is safe — each thread
    /// reads each input cell before writing the output cell.
    pub fn baracuda_kernels_ternary_fma_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `ternary_fma_f32`.
    pub fn baracuda_kernels_ternary_fma_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Ternary elementwise `fma`, f32, strided / broadcast path.
    pub fn baracuda_kernels_ternary_fma_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_c: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Ternary elementwise `fma`, f16, contig fast path.
    pub fn baracuda_kernels_ternary_fma_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `ternary_fma_f16`.
    pub fn baracuda_kernels_ternary_fma_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Ternary elementwise `fma`, f16, strided / broadcast path.
    pub fn baracuda_kernels_ternary_fma_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_c: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Ternary elementwise `fma`, bf16, contig fast path.
    pub fn baracuda_kernels_ternary_fma_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `ternary_fma_bf16`.
    pub fn baracuda_kernels_ternary_fma_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Ternary elementwise `fma`, bf16, strided / broadcast path.
    pub fn baracuda_kernels_ternary_fma_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_c: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Ternary elementwise `fma`, f64, contig fast path.
    pub fn baracuda_kernels_ternary_fma_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `ternary_fma_f64`.
    pub fn baracuda_kernels_ternary_fma_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Ternary elementwise `fma`, f64, strided / broadcast path.
    pub fn baracuda_kernels_ternary_fma_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_c: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        c: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Reductions — Phase 4 trailblazer (axis reduction)
// ============================================================================
//
// `y = reduce(x, axis=k)` with keepdim=true (output shape == input
// shape but the reduced axis collapses to size 1). Single-axis only
// today — multi-axis / full-tensor reductions are fanout. Naive
// implementation: one thread per output cell.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Sum reduction along one axis, f32, naive thread-per-output-cell.
    ///
    /// `output_shape` matches input shape with `[reduce_axis]` set to 1.
    /// `reduce_extent` is the input's extent along the reduced axis.
    /// `reduce_stride_x` is the input stride along the reduced axis
    /// (in elements).
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_sum_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sum reduction along one axis, f16.
    ///
    /// Same parameter shape as the f32 variant; functor specializes the
    /// accumulator op through the standard f32-detour pattern
    /// (`__half2float` / `+` / `__float2half`).
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_sum_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sum reduction along one axis, bf16 (f32-detour functor).
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_sum_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sum reduction along one axis, f64.
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_sum_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Mean reduction along one axis, f32. Sum then divide by extent.
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_mean_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Mean reduction along one axis, f16 (f32-detour for sum + divide).
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_mean_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Mean reduction along one axis, bf16 (f32-detour for sum + divide).
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_mean_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Mean reduction along one axis, f64.
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_mean_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Max reduction along one axis, f32. `init = -INFINITY`, `fmaxf`.
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_max_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Max reduction along one axis, f16 (f32-detour fmaxf).
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_max_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Max reduction along one axis, bf16 (f32-detour fmaxf).
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_max_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Max reduction along one axis, f64.
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_max_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Min reduction along one axis, f32. `init = +INFINITY`, `fminf`.
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_min_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Min reduction along one axis, f16 (f32-detour fminf).
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_min_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Min reduction along one axis, bf16 (f32-detour fminf).
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_min_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Min reduction along one axis, f64.
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_min_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Product reduction along one axis, f32. `init = 1`, op = `*`.
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_prod_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Product reduction along one axis, f16 (f32-detour multiply).
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_prod_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Product reduction along one axis, bf16 (f32-detour multiply).
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_prod_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Product reduction along one axis, f64.
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_reduce_prod_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Norm2 reduction along one axis, f32. `y = sqrt(sum(x*x))` —
    /// shares the simple-reduce parameter shape.
    pub fn baracuda_kernels_reduce_norm2_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Norm2 reduction along one axis, f16 (f32-detour functor + sqrt).
    pub fn baracuda_kernels_reduce_norm2_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Norm2 reduction along one axis, bf16 (f32-detour functor + sqrt).
    pub fn baracuda_kernels_reduce_norm2_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Norm2 reduction along one axis, f64.
    pub fn baracuda_kernels_reduce_norm2_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSumExp reduction along one axis, f32 — numerically stable
    /// two-pass max-then-sum-exp. Shares the simple-reduce parameter
    /// shape so the Rust dispatcher can reach it through the same FFI
    /// signature; the kernel internally performs two passes over the
    /// reduce axis.
    pub fn baracuda_kernels_reduce_logsumexp_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSumExp reduction along one axis, f16 (f32-detour throughout).
    pub fn baracuda_kernels_reduce_logsumexp_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSumExp reduction along one axis, bf16 (f32-detour throughout).
    pub fn baracuda_kernels_reduce_logsumexp_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSumExp reduction along one axis, f64.
    pub fn baracuda_kernels_reduce_logsumexp_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Scans (Category F) — length-preserving prefix operators along a
// single axis. ABI mirrors reduce-axis but adds a `reverse` flag and
// uses the full input shape (no axis collapse in the output).
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Inclusive prefix sum (`cumsum`) along a single axis, f32.
    /// `reverse != 0` flips the scan direction.
    pub fn baracuda_kernels_scan_cumsum_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cumsum, f16. f32-detour accumulator inside the kernel.
    pub fn baracuda_kernels_scan_cumsum_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cumsum, bf16.
    pub fn baracuda_kernels_scan_cumsum_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cumsum, f64.
    pub fn baracuda_kernels_scan_cumsum_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cumprod (inclusive prefix product), f32. Same ABI as cumsum.
    pub fn baracuda_kernels_scan_cumprod_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cumprod, f16. f32-detour accumulator.
    pub fn baracuda_kernels_scan_cumprod_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cumprod, bf16.
    pub fn baracuda_kernels_scan_cumprod_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cumprod, f64.
    pub fn baracuda_kernels_scan_cumprod_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummax (inclusive prefix running max), f32.
    pub fn baracuda_kernels_scan_cummax_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummax, f16.
    pub fn baracuda_kernels_scan_cummax_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummax, bf16.
    pub fn baracuda_kernels_scan_cummax_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummax, f64.
    pub fn baracuda_kernels_scan_cummax_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummin (inclusive prefix running min), f32.
    pub fn baracuda_kernels_scan_cummin_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummin, f16.
    pub fn baracuda_kernels_scan_cummin_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummin, bf16.
    pub fn baracuda_kernels_scan_cummin_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummin, f64.
    pub fn baracuda_kernels_scan_cummin_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cumprod backward, f32. Per-cell suffix accumulator of
    /// `dy[i] * y[i] / x[j]`. Caller must ensure x has no zeros along
    /// the scan axis.
    pub fn baracuda_kernels_scan_cumprod_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cumprod backward, f16. f32-detour accumulator.
    pub fn baracuda_kernels_scan_cumprod_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cumprod backward, bf16.
    pub fn baracuda_kernels_scan_cumprod_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cumprod backward, f64.
    pub fn baracuda_kernels_scan_cumprod_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummax backward, f32. Walks the forward scan tracking
    /// first-occurrence argmax; gradient flows to the source position.
    pub fn baracuda_kernels_scan_cummax_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummax backward, f16.
    pub fn baracuda_kernels_scan_cummax_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummax backward, bf16.
    pub fn baracuda_kernels_scan_cummax_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummax backward, f64.
    pub fn baracuda_kernels_scan_cummax_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummin backward, f32. Same kernel shape as Cummax BW with
    /// `<` instead of `>` for the tie-tracking comparison.
    pub fn baracuda_kernels_scan_cummin_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummin backward, f16.
    pub fn baracuda_kernels_scan_cummin_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummin backward, bf16.
    pub fn baracuda_kernels_scan_cummin_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Cummin backward, f64.
    pub fn baracuda_kernels_scan_cummin_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogCumsumExp FW, f32. `y[k] = log(Σ_{j ≤ k} exp(x[j]))`
    /// (or suffix-LSE when `reverse != 0`). Numerically stable via
    /// the online running-max algorithm. Same ABI as cumsum.
    pub fn baracuda_kernels_scan_log_cumsum_exp_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogCumsumExp FW, f16. f32-detour accumulator inside the kernel.
    pub fn baracuda_kernels_scan_log_cumsum_exp_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogCumsumExp FW, bf16.
    pub fn baracuda_kernels_scan_log_cumsum_exp_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogCumsumExp FW, f64.
    pub fn baracuda_kernels_scan_log_cumsum_exp_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        scan_stride_x: i64,
        reverse: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogCumsumExp BW, f32. Per-cell accumulator of
    /// `Σ dy[i] * exp(x[k] - y[i])` over the FW-direction-dependent
    /// `i` range. Needs both saved `x` and saved `y` (same shape since
    /// scans are length-preserving). Stable by construction:
    /// `x[k] - y[i] ≤ 0` so `exp(.) ∈ [0, 1]`.
    pub fn baracuda_kernels_scan_log_cumsum_exp_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogCumsumExp BW, f16. f32-detour accumulator.
    pub fn baracuda_kernels_scan_log_cumsum_exp_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogCumsumExp BW, bf16.
    pub fn baracuda_kernels_scan_log_cumsum_exp_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogCumsumExp BW, f64.
    pub fn baracuda_kernels_scan_log_cumsum_exp_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        scan_axis: i32,
        scan_extent: i32,
        reverse: i32,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Softmax family (Category H) — length-preserving stable softmax along
// a single axis. Output shape == input shape. FW + BW per dtype.
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Softmax FW, f32. `y[k] = exp(x[k] - max) / Σ exp(x[j] - max)`
    /// along `softmax_axis`. Numerically stable.
    pub fn baracuda_kernels_softmax_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Softmax FW, f16. f32 accumulator inside the kernel.
    pub fn baracuda_kernels_softmax_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Softmax FW, bf16.
    pub fn baracuda_kernels_softmax_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Softmax FW, f64.
    pub fn baracuda_kernels_softmax_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Softmax BW, f32. `dx[k] = y[k] * (dy[k] - Σ_j y[j] * dy[j])`.
    /// Caller passes the saved forward output `y`.
    pub fn baracuda_kernels_softmax_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Softmax BW, f16.
    pub fn baracuda_kernels_softmax_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Softmax BW, bf16.
    pub fn baracuda_kernels_softmax_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Softmax BW, f64.
    pub fn baracuda_kernels_softmax_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSoftmax FW, f32. `y[k] = (x[k] - max) - log(Σ exp(x[j] - max))`
    /// along `softmax_axis`. Numerically stable.
    pub fn baracuda_kernels_log_softmax_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSoftmax FW, f16. f32 accumulator inside the kernel.
    pub fn baracuda_kernels_log_softmax_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSoftmax FW, bf16.
    pub fn baracuda_kernels_log_softmax_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSoftmax FW, f64.
    pub fn baracuda_kernels_log_softmax_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSoftmax BW, f32. `dx[k] = dy[k] - exp(y[k]) * Σ_j dy[j]`.
    /// Caller passes the saved forward output `y` (log-softmax values).
    pub fn baracuda_kernels_log_softmax_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSoftmax BW, f16.
    pub fn baracuda_kernels_log_softmax_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSoftmax BW, bf16.
    pub fn baracuda_kernels_log_softmax_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSoftmax BW, f64.
    pub fn baracuda_kernels_log_softmax_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GumbelSoftmax FW, f32. `y = softmax((x + g) / τ)` where
    /// `g = -log(-log(u))` and `u` is a caller-supplied cuRAND uniform
    /// buffer (one f32 per output cell, dense / contiguous layout).
    /// `inv_tau = 1/τ`. `hard != 0` → one-hot at the noisy argmax.
    pub fn baracuda_kernels_gumbel_softmax_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        inv_tau: f32,
        hard: i32,
        x: *const c_void,
        u_rand: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GumbelSoftmax FW, f16.
    pub fn baracuda_kernels_gumbel_softmax_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        inv_tau: f32,
        hard: i32,
        x: *const c_void,
        u_rand: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GumbelSoftmax FW, bf16.
    pub fn baracuda_kernels_gumbel_softmax_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        inv_tau: f32,
        hard: i32,
        x: *const c_void,
        u_rand: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GumbelSoftmax FW, f64.
    pub fn baracuda_kernels_gumbel_softmax_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        inv_tau: f32,
        hard: i32,
        x: *const c_void,
        u_rand: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sparsemax FW, f32. `y = ProjSimplex(x)` via threshold τ found
    /// after sorting the row descending. Row extent limited to 64.
    pub fn baracuda_kernels_sparsemax_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sparsemax FW, f16.
    pub fn baracuda_kernels_sparsemax_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sparsemax FW, bf16.
    pub fn baracuda_kernels_sparsemax_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sparsemax FW, f64.
    pub fn baracuda_kernels_sparsemax_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_x: i64,
        softmax_stride_y: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sparsemax BW, f32. `dx[i] = dy[i] - sum_dy_active / n_active` for
    /// active positions (`y > 0`), `0` elsewhere.
    pub fn baracuda_kernels_sparsemax_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sparsemax BW, f16.
    pub fn baracuda_kernels_sparsemax_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sparsemax BW, bf16.
    pub fn baracuda_kernels_sparsemax_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sparsemax BW, f64.
    pub fn baracuda_kernels_sparsemax_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        softmax_axis: i32,
        softmax_extent: i32,
        softmax_stride_dy: i64,
        softmax_stride_y: i64,
        dy: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Loss family (Category R) — MSE / NLL / CrossEntropy / BCE / KLDiv
// (FW + BW × 4 FP dtypes). Per-cell or per-row kernel emits to a
// workspace buffer; a single-block deterministic tree reduction collapses
// to the final scalar for Mean / Sum modes. For None mode the per-cell
// kernel writes directly to out (no reduction). Reduction modes: 0=None,
// 1=Mean, 2=Sum.
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// MSE FW, f32. `(pred - target)²` per-cell; mean/sum reduce to scalar.
    /// Workspace: `numel * sizeof(T)` bytes for Mean/Sum; unused for None.
    pub fn baracuda_kernels_loss_mse_f32_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// MSE FW, f16.
    pub fn baracuda_kernels_loss_mse_f16_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// MSE FW, bf16.
    pub fn baracuda_kernels_loss_mse_bf16_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// MSE FW, f64.
    pub fn baracuda_kernels_loss_mse_f64_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// MSE BW, f32. `dpred = 2·(pred - target) · scale`.
    pub fn baracuda_kernels_loss_mse_backward_f32_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// MSE BW, f16.
    pub fn baracuda_kernels_loss_mse_backward_f16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// MSE BW, bf16.
    pub fn baracuda_kernels_loss_mse_backward_bf16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// MSE BW, f64.
    pub fn baracuda_kernels_loss_mse_backward_f64_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// BCE FW, f32. `-(t·log(p) + (1-t)·log(1-p))` per-cell, then reduce.
    /// Caller ensures pred ∈ (0, 1).
    pub fn baracuda_kernels_loss_bce_f32_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCE FW, f16.
    pub fn baracuda_kernels_loss_bce_f16_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCE FW, bf16.
    pub fn baracuda_kernels_loss_bce_bf16_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCE FW, f64.
    pub fn baracuda_kernels_loss_bce_f64_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// BCE BW, f32. `dpred = (pred - target) / (pred·(1-pred)) · scale`.
    pub fn baracuda_kernels_loss_bce_backward_f32_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCE BW, f16.
    pub fn baracuda_kernels_loss_bce_backward_f16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCE BW, bf16.
    pub fn baracuda_kernels_loss_bce_backward_bf16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCE BW, f64.
    pub fn baracuda_kernels_loss_bce_backward_f64_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// KLDiv FW, f32. `target·(log(target) - input)` per-cell. PyTorch
    /// convention: input is already log-prob.
    pub fn baracuda_kernels_loss_kl_div_f32_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// KLDiv FW, f16.
    pub fn baracuda_kernels_loss_kl_div_f16_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// KLDiv FW, bf16.
    pub fn baracuda_kernels_loss_kl_div_bf16_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// KLDiv FW, f64.
    pub fn baracuda_kernels_loss_kl_div_f64_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// KLDiv BW, f32. `dinput = -target · scale`.
    pub fn baracuda_kernels_loss_kl_div_backward_f32_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// KLDiv BW, f16.
    pub fn baracuda_kernels_loss_kl_div_backward_f16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// KLDiv BW, bf16.
    pub fn baracuda_kernels_loss_kl_div_backward_bf16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// KLDiv BW, f64.
    pub fn baracuda_kernels_loss_kl_div_backward_f64_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// NLL FW, f32. `-input[i, target[i]]` per row. Heterogeneous-dtype:
    /// input is `T`, target is `i64`. `row_stride_input` is the i64 stride
    /// between adjacent rows of `input` (must equal `class_extent` for
    /// contiguous input).
    pub fn baracuda_kernels_loss_nll_f32_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// NLL FW, f16.
    pub fn baracuda_kernels_loss_nll_f16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// NLL FW, bf16.
    pub fn baracuda_kernels_loss_nll_bf16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// NLL FW, f64.
    pub fn baracuda_kernels_loss_nll_f64_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// NLL BW, f32. Pre-zeros `dinput` (size `dinput_numel · sizeof(T)`),
    /// then writes `dinput[i, target[i]] = -dy_or_scale`.
    pub fn baracuda_kernels_loss_nll_backward_f32_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        dinput_numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        dy: *const c_void,
        target: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// NLL BW, f16.
    pub fn baracuda_kernels_loss_nll_backward_f16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        dinput_numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        dy: *const c_void,
        target: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// NLL BW, bf16.
    pub fn baracuda_kernels_loss_nll_backward_bf16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        dinput_numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        dy: *const c_void,
        target: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// NLL BW, f64.
    pub fn baracuda_kernels_loss_nll_backward_f64_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        dinput_numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        dy: *const c_void,
        target: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CrossEntropy FW, f32. Fused LogSoftmax + NLL. Numerically stable
    /// per-row two-pass max subtraction.
    pub fn baracuda_kernels_loss_cross_entropy_f32_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CrossEntropy FW, f16.
    pub fn baracuda_kernels_loss_cross_entropy_f16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CrossEntropy FW, bf16.
    pub fn baracuda_kernels_loss_cross_entropy_bf16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CrossEntropy FW, f64.
    pub fn baracuda_kernels_loss_cross_entropy_f64_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CrossEntropy BW, f32. `dinput[i, c] = (softmax(input)[i, c] - 1{c==t[i]}) · scale`.
    pub fn baracuda_kernels_loss_cross_entropy_backward_f32_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CrossEntropy BW, f16.
    pub fn baracuda_kernels_loss_cross_entropy_backward_f16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CrossEntropy BW, bf16.
    pub fn baracuda_kernels_loss_cross_entropy_backward_bf16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CrossEntropy BW, f64.
    pub fn baracuda_kernels_loss_cross_entropy_backward_f64_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Milestone 5.2 — Tier-1 losses (L1 / SmoothL1 / Huber / BCEWithLogits /
// PoissonNLL / GaussianNLL / soft-target CrossEntropy). All follow the same
// per-cell-kernel + tree-reduction-finalizer pattern as the original loss
// family; the SmoothL1 / Huber / PoissonNLL / GaussianNLL launchers thread
// the corresponding scalar parameter (β / δ / log_input_flag / eps) through.
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// L1 FW, f32. `y = |pred - target|` per-cell; mean/sum reduce to scalar.
    pub fn baracuda_kernels_loss_l1_f32_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// L1 FW, f16.
    pub fn baracuda_kernels_loss_l1_f16_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// L1 FW, bf16.
    pub fn baracuda_kernels_loss_l1_bf16_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// L1 FW, f64.
    pub fn baracuda_kernels_loss_l1_f64_run(
        numel: i64,
        reduction_mode: i32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// L1 BW, f32. `dpred = sign(pred - target) · scale`.
    pub fn baracuda_kernels_loss_l1_backward_f32_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// L1 BW, f16.
    pub fn baracuda_kernels_loss_l1_backward_f16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// L1 BW, bf16.
    pub fn baracuda_kernels_loss_l1_backward_bf16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// L1 BW, f64.
    pub fn baracuda_kernels_loss_l1_backward_f64_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// SmoothL1 FW, f32. `param = β`.
    pub fn baracuda_kernels_loss_smooth_l1_f32_run(
        numel: i64,
        reduction_mode: i32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SmoothL1 FW, f16.
    pub fn baracuda_kernels_loss_smooth_l1_f16_run(
        numel: i64,
        reduction_mode: i32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SmoothL1 FW, bf16.
    pub fn baracuda_kernels_loss_smooth_l1_bf16_run(
        numel: i64,
        reduction_mode: i32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SmoothL1 FW, f64.
    pub fn baracuda_kernels_loss_smooth_l1_f64_run(
        numel: i64,
        reduction_mode: i32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// SmoothL1 BW, f32.
    pub fn baracuda_kernels_loss_smooth_l1_backward_f32_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SmoothL1 BW, f16.
    pub fn baracuda_kernels_loss_smooth_l1_backward_f16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SmoothL1 BW, bf16.
    pub fn baracuda_kernels_loss_smooth_l1_backward_bf16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SmoothL1 BW, f64.
    pub fn baracuda_kernels_loss_smooth_l1_backward_f64_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Huber FW, f32. `param = δ`.
    pub fn baracuda_kernels_loss_huber_f32_run(
        numel: i64,
        reduction_mode: i32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Huber FW, f16.
    pub fn baracuda_kernels_loss_huber_f16_run(
        numel: i64,
        reduction_mode: i32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Huber FW, bf16.
    pub fn baracuda_kernels_loss_huber_bf16_run(
        numel: i64,
        reduction_mode: i32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Huber FW, f64.
    pub fn baracuda_kernels_loss_huber_f64_run(
        numel: i64,
        reduction_mode: i32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Huber BW, f32.
    pub fn baracuda_kernels_loss_huber_backward_f32_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Huber BW, f16.
    pub fn baracuda_kernels_loss_huber_backward_f16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Huber BW, bf16.
    pub fn baracuda_kernels_loss_huber_backward_bf16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Huber BW, f64.
    pub fn baracuda_kernels_loss_huber_backward_f64_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        param: f32,
        pred: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// BCEWithLogits FW, f32. Stable BCE for raw logits.
    pub fn baracuda_kernels_loss_bce_with_logits_f32_run(
        numel: i64,
        reduction_mode: i32,
        logits: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCEWithLogits FW, f16.
    pub fn baracuda_kernels_loss_bce_with_logits_f16_run(
        numel: i64,
        reduction_mode: i32,
        logits: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCEWithLogits FW, bf16.
    pub fn baracuda_kernels_loss_bce_with_logits_bf16_run(
        numel: i64,
        reduction_mode: i32,
        logits: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCEWithLogits FW, f64.
    pub fn baracuda_kernels_loss_bce_with_logits_f64_run(
        numel: i64,
        reduction_mode: i32,
        logits: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// BCEWithLogits BW, f32. `dlogits = (sigmoid(x) - target) · scale`.
    pub fn baracuda_kernels_loss_bce_with_logits_backward_f32_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        logits: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCEWithLogits BW, f16.
    pub fn baracuda_kernels_loss_bce_with_logits_backward_f16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        logits: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCEWithLogits BW, bf16.
    pub fn baracuda_kernels_loss_bce_with_logits_backward_bf16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        logits: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// BCEWithLogits BW, f64.
    pub fn baracuda_kernels_loss_bce_with_logits_backward_f64_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        logits: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dpred: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// PoissonNLL FW, f32. `log_input_flag` 0/1.
    pub fn baracuda_kernels_loss_poisson_nll_f32_run(
        numel: i64,
        reduction_mode: i32,
        log_input_flag: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// PoissonNLL FW, f16.
    pub fn baracuda_kernels_loss_poisson_nll_f16_run(
        numel: i64,
        reduction_mode: i32,
        log_input_flag: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// PoissonNLL FW, bf16.
    pub fn baracuda_kernels_loss_poisson_nll_bf16_run(
        numel: i64,
        reduction_mode: i32,
        log_input_flag: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// PoissonNLL FW, f64.
    pub fn baracuda_kernels_loss_poisson_nll_f64_run(
        numel: i64,
        reduction_mode: i32,
        log_input_flag: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// PoissonNLL BW, f32.
    pub fn baracuda_kernels_loss_poisson_nll_backward_f32_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        log_input_flag: i32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// PoissonNLL BW, f16.
    pub fn baracuda_kernels_loss_poisson_nll_backward_f16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        log_input_flag: i32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// PoissonNLL BW, bf16.
    pub fn baracuda_kernels_loss_poisson_nll_backward_bf16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        log_input_flag: i32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// PoissonNLL BW, f64.
    pub fn baracuda_kernels_loss_poisson_nll_backward_f64_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        log_input_flag: i32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GaussianNLL FW, f32. 3-tensor input (input, target, var).
    pub fn baracuda_kernels_loss_gaussian_nll_f32_run(
        numel: i64,
        reduction_mode: i32,
        eps: f32,
        input: *const c_void,
        target: *const c_void,
        var: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// GaussianNLL FW, f16.
    pub fn baracuda_kernels_loss_gaussian_nll_f16_run(
        numel: i64,
        reduction_mode: i32,
        eps: f32,
        input: *const c_void,
        target: *const c_void,
        var: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// GaussianNLL FW, bf16.
    pub fn baracuda_kernels_loss_gaussian_nll_bf16_run(
        numel: i64,
        reduction_mode: i32,
        eps: f32,
        input: *const c_void,
        target: *const c_void,
        var: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// GaussianNLL FW, f64.
    pub fn baracuda_kernels_loss_gaussian_nll_f64_run(
        numel: i64,
        reduction_mode: i32,
        eps: f32,
        input: *const c_void,
        target: *const c_void,
        var: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GaussianNLL BW, f32.
    pub fn baracuda_kernels_loss_gaussian_nll_backward_f32_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        eps: f32,
        input: *const c_void,
        target: *const c_void,
        var: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// GaussianNLL BW, f16.
    pub fn baracuda_kernels_loss_gaussian_nll_backward_f16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        eps: f32,
        input: *const c_void,
        target: *const c_void,
        var: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// GaussianNLL BW, bf16.
    pub fn baracuda_kernels_loss_gaussian_nll_backward_bf16_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        eps: f32,
        input: *const c_void,
        target: *const c_void,
        var: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// GaussianNLL BW, f64.
    pub fn baracuda_kernels_loss_gaussian_nll_backward_f64_run(
        numel: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        eps: f32,
        input: *const c_void,
        target: *const c_void,
        var: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Soft-target CrossEntropy FW, f32. Target is `T`-typed prob tensor.
    pub fn baracuda_kernels_loss_cross_entropy_soft_f32_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        row_stride_target: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Soft-target CrossEntropy FW, f16.
    pub fn baracuda_kernels_loss_cross_entropy_soft_f16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        row_stride_target: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Soft-target CrossEntropy FW, bf16.
    pub fn baracuda_kernels_loss_cross_entropy_soft_bf16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        row_stride_target: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Soft-target CrossEntropy FW, f64.
    pub fn baracuda_kernels_loss_cross_entropy_soft_f64_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        row_stride_target: i64,
        reduction_mode: i32,
        input: *const c_void,
        target: *const c_void,
        out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Soft-target CrossEntropy BW, f32.
    pub fn baracuda_kernels_loss_cross_entropy_soft_backward_f32_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        row_stride_target: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// MarginRanking FW, f32. ABI: `(numel, reduction_mode, margin,
    /// x1, x2, t, out, workspace, workspace_bytes, stream)`.
    pub fn baracuda_kernels_loss_margin_ranking_f32_run(
        numel: i64, reduction_mode: i32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MarginRanking FW, f16.
    pub fn baracuda_kernels_loss_margin_ranking_f16_run(
        numel: i64, reduction_mode: i32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MarginRanking FW, bf16.
    pub fn baracuda_kernels_loss_margin_ranking_bf16_run(
        numel: i64, reduction_mode: i32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MarginRanking FW, f64.
    pub fn baracuda_kernels_loss_margin_ranking_f64_run(
        numel: i64, reduction_mode: i32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MarginRanking BW, f32. ABI: `(numel, reduction_mode, scale, margin,
    /// x1, x2, t, dy, dx1, dx2, workspace, workspace_bytes, stream)`.
    pub fn baracuda_kernels_loss_margin_ranking_backward_f32_run(
        numel: i64, reduction_mode: i32, scale_scalar: f32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, dy: *const c_void,
        dx1: *mut c_void, dx2: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MarginRanking BW, f16.
    pub fn baracuda_kernels_loss_margin_ranking_backward_f16_run(
        numel: i64, reduction_mode: i32, scale_scalar: f32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, dy: *const c_void,
        dx1: *mut c_void, dx2: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MarginRanking BW, bf16.
    pub fn baracuda_kernels_loss_margin_ranking_backward_bf16_run(
        numel: i64, reduction_mode: i32, scale_scalar: f32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, dy: *const c_void,
        dx1: *mut c_void, dx2: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MarginRanking BW, f64.
    pub fn baracuda_kernels_loss_margin_ranking_backward_f64_run(
        numel: i64, reduction_mode: i32, scale_scalar: f32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, dy: *const c_void,
        dx1: *mut c_void, dx2: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// HingeEmbedding FW, f32. ABI: `(numel, reduction_mode, margin,
    /// input, target_i64, out, workspace, workspace_bytes, stream)`.
    pub fn baracuda_kernels_loss_hinge_embedding_f32_run(
        numel: i64, reduction_mode: i32, margin: f32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// HingeEmbedding FW, f16.
    pub fn baracuda_kernels_loss_hinge_embedding_f16_run(
        numel: i64, reduction_mode: i32, margin: f32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// HingeEmbedding FW, bf16.
    pub fn baracuda_kernels_loss_hinge_embedding_bf16_run(
        numel: i64, reduction_mode: i32, margin: f32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// HingeEmbedding FW, f64.
    pub fn baracuda_kernels_loss_hinge_embedding_f64_run(
        numel: i64, reduction_mode: i32, margin: f32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// HingeEmbedding BW, f32.
    pub fn baracuda_kernels_loss_hinge_embedding_backward_f32_run(
        numel: i64, reduction_mode: i32, scale_scalar: f32, margin: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// HingeEmbedding BW, f16.
    pub fn baracuda_kernels_loss_hinge_embedding_backward_f16_run(
        numel: i64, reduction_mode: i32, scale_scalar: f32, margin: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// HingeEmbedding BW, bf16.
    pub fn baracuda_kernels_loss_hinge_embedding_backward_bf16_run(
        numel: i64, reduction_mode: i32, scale_scalar: f32, margin: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// HingeEmbedding BW, f64.
    pub fn baracuda_kernels_loss_hinge_embedding_backward_f64_run(
        numel: i64, reduction_mode: i32, scale_scalar: f32, margin: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// CosineEmbedding FW (per-row). ABI: `(n_rows, d_extent, row_stride_x,
    /// reduction_mode, margin, x1, x2, t, out, workspace, workspace_bytes, stream)`.
    pub fn baracuda_kernels_loss_cosine_embedding_f32_run(
        n_rows: i64, d_extent: i32, row_stride_x: i64,
        reduction_mode: i32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CosineEmbedding FW, f16.
    pub fn baracuda_kernels_loss_cosine_embedding_f16_run(
        n_rows: i64, d_extent: i32, row_stride_x: i64,
        reduction_mode: i32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CosineEmbedding FW, bf16.
    pub fn baracuda_kernels_loss_cosine_embedding_bf16_run(
        n_rows: i64, d_extent: i32, row_stride_x: i64,
        reduction_mode: i32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CosineEmbedding FW, f64.
    pub fn baracuda_kernels_loss_cosine_embedding_f64_run(
        n_rows: i64, d_extent: i32, row_stride_x: i64,
        reduction_mode: i32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CosineEmbedding BW.
    pub fn baracuda_kernels_loss_cosine_embedding_backward_f32_run(
        n_rows: i64, d_extent: i32, row_stride_x: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, dy: *const c_void,
        dx1: *mut c_void, dx2: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CosineEmbedding BW, f16.
    pub fn baracuda_kernels_loss_cosine_embedding_backward_f16_run(
        n_rows: i64, d_extent: i32, row_stride_x: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, dy: *const c_void,
        dx1: *mut c_void, dx2: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CosineEmbedding BW, bf16.
    pub fn baracuda_kernels_loss_cosine_embedding_backward_bf16_run(
        n_rows: i64, d_extent: i32, row_stride_x: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, dy: *const c_void,
        dx1: *mut c_void, dx2: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CosineEmbedding BW, f64.
    pub fn baracuda_kernels_loss_cosine_embedding_backward_f64_run(
        n_rows: i64, d_extent: i32, row_stride_x: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32,
        x1: *const c_void, x2: *const c_void, t: *const c_void, dy: *const c_void,
        dx1: *mut c_void, dx2: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// TripletMargin FW (per-row). ABI: `(n_rows, d_extent, row_stride,
    /// reduction_mode, margin, p_norm, a, p, n, out, workspace, workspace_bytes, stream)`.
    pub fn baracuda_kernels_loss_triplet_margin_f32_run(
        n_rows: i64, d_extent: i32, row_stride: i64,
        reduction_mode: i32, margin: f32, p_norm: f32,
        a: *const c_void, p_tensor: *const c_void, n_tensor: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// TripletMargin FW, f16.
    pub fn baracuda_kernels_loss_triplet_margin_f16_run(
        n_rows: i64, d_extent: i32, row_stride: i64,
        reduction_mode: i32, margin: f32, p_norm: f32,
        a: *const c_void, p_tensor: *const c_void, n_tensor: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// TripletMargin FW, bf16.
    pub fn baracuda_kernels_loss_triplet_margin_bf16_run(
        n_rows: i64, d_extent: i32, row_stride: i64,
        reduction_mode: i32, margin: f32, p_norm: f32,
        a: *const c_void, p_tensor: *const c_void, n_tensor: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// TripletMargin FW, f64.
    pub fn baracuda_kernels_loss_triplet_margin_f64_run(
        n_rows: i64, d_extent: i32, row_stride: i64,
        reduction_mode: i32, margin: f32, p_norm: f32,
        a: *const c_void, p_tensor: *const c_void, n_tensor: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// TripletMargin BW.
    pub fn baracuda_kernels_loss_triplet_margin_backward_f32_run(
        n_rows: i64, d_extent: i32, row_stride: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32, p_norm: f32,
        a: *const c_void, p_tensor: *const c_void, n_tensor: *const c_void, dy: *const c_void,
        da: *mut c_void, dp: *mut c_void, dn: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// TripletMargin BW, f16.
    pub fn baracuda_kernels_loss_triplet_margin_backward_f16_run(
        n_rows: i64, d_extent: i32, row_stride: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32, p_norm: f32,
        a: *const c_void, p_tensor: *const c_void, n_tensor: *const c_void, dy: *const c_void,
        da: *mut c_void, dp: *mut c_void, dn: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// TripletMargin BW, bf16.
    pub fn baracuda_kernels_loss_triplet_margin_backward_bf16_run(
        n_rows: i64, d_extent: i32, row_stride: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32, p_norm: f32,
        a: *const c_void, p_tensor: *const c_void, n_tensor: *const c_void, dy: *const c_void,
        da: *mut c_void, dp: *mut c_void, dn: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// TripletMargin BW, f64.
    pub fn baracuda_kernels_loss_triplet_margin_backward_f64_run(
        n_rows: i64, d_extent: i32, row_stride: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32, p_norm: f32,
        a: *const c_void, p_tensor: *const c_void, n_tensor: *const c_void, dy: *const c_void,
        da: *mut c_void, dp: *mut c_void, dn: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// MultiMargin FW (per-row). ABI: `(n_rows, class_extent, row_stride,
    /// reduction_mode, margin, p_norm, input, target_i64, out, workspace, workspace_bytes, stream)`.
    pub fn baracuda_kernels_loss_multi_margin_f32_run(
        n_rows: i64, class_extent: i32, row_stride: i64,
        reduction_mode: i32, margin: f32, p_norm: f32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultiMargin FW, f16.
    pub fn baracuda_kernels_loss_multi_margin_f16_run(
        n_rows: i64, class_extent: i32, row_stride: i64,
        reduction_mode: i32, margin: f32, p_norm: f32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultiMargin FW, bf16.
    pub fn baracuda_kernels_loss_multi_margin_bf16_run(
        n_rows: i64, class_extent: i32, row_stride: i64,
        reduction_mode: i32, margin: f32, p_norm: f32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultiMargin FW, f64.
    pub fn baracuda_kernels_loss_multi_margin_f64_run(
        n_rows: i64, class_extent: i32, row_stride: i64,
        reduction_mode: i32, margin: f32, p_norm: f32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultiMargin BW.
    pub fn baracuda_kernels_loss_multi_margin_backward_f32_run(
        n_rows: i64, class_extent: i32, row_stride: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32, p_norm: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultiMargin BW, f16.
    pub fn baracuda_kernels_loss_multi_margin_backward_f16_run(
        n_rows: i64, class_extent: i32, row_stride: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32, p_norm: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultiMargin BW, bf16.
    pub fn baracuda_kernels_loss_multi_margin_backward_bf16_run(
        n_rows: i64, class_extent: i32, row_stride: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32, p_norm: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultiMargin BW, f64.
    pub fn baracuda_kernels_loss_multi_margin_backward_f64_run(
        n_rows: i64, class_extent: i32, row_stride: i64,
        reduction_mode: i32, scale_scalar: f32, margin: f32, p_norm: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// MultilabelMargin FW (per-row). ABI: `(n_rows, class_extent,
    /// row_stride_in, row_stride_tgt, reduction_mode, input, target_i64,
    /// out, workspace, workspace_bytes, stream)`.
    pub fn baracuda_kernels_loss_multilabel_margin_f32_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelMargin FW, f16.
    pub fn baracuda_kernels_loss_multilabel_margin_f16_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelMargin FW, bf16.
    pub fn baracuda_kernels_loss_multilabel_margin_bf16_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelMargin FW, f64.
    pub fn baracuda_kernels_loss_multilabel_margin_f64_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelMargin BW.
    pub fn baracuda_kernels_loss_multilabel_margin_backward_f32_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32, scale_scalar: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelMargin BW, f16.
    pub fn baracuda_kernels_loss_multilabel_margin_backward_f16_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32, scale_scalar: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelMargin BW, bf16.
    pub fn baracuda_kernels_loss_multilabel_margin_backward_bf16_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32, scale_scalar: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelMargin BW, f64.
    pub fn baracuda_kernels_loss_multilabel_margin_backward_f64_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32, scale_scalar: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// MultilabelSoftMargin FW.
    pub fn baracuda_kernels_loss_multilabel_soft_margin_f32_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelSoftMargin FW, f16.
    pub fn baracuda_kernels_loss_multilabel_soft_margin_f16_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelSoftMargin FW, bf16.
    pub fn baracuda_kernels_loss_multilabel_soft_margin_bf16_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelSoftMargin FW, f64.
    pub fn baracuda_kernels_loss_multilabel_soft_margin_f64_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32,
        input: *const c_void, target: *const c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelSoftMargin BW.
    pub fn baracuda_kernels_loss_multilabel_soft_margin_backward_f32_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32, scale_scalar: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelSoftMargin BW, f16.
    pub fn baracuda_kernels_loss_multilabel_soft_margin_backward_f16_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32, scale_scalar: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelSoftMargin BW, bf16.
    pub fn baracuda_kernels_loss_multilabel_soft_margin_backward_bf16_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32, scale_scalar: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// MultilabelSoftMargin BW, f64.
    pub fn baracuda_kernels_loss_multilabel_soft_margin_backward_f64_run(
        n_rows: i64, class_extent: i32,
        row_stride_in: i64, row_stride_tgt: i64,
        reduction_mode: i32, scale_scalar: f32,
        input: *const c_void, target: *const c_void, dy: *const c_void, dinput: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // -------------------------------------------------------------------------
    // CTCLoss — Milestone 5.5 (Phase 5 final deferral).
    //
    // log_probs is `T[T, N, C]` row-major. targets is `i64[N, S]`.
    // input_lengths / target_lengths are `i64[N]`. The kernel runs
    // forward DP on the lattice once per batch sample (one CUDA block
    // per sample). `alpha_ws` is a workspace of accumulator type
    // (f32 for {f32, f16, bf16}; f64 for f64) shaped `[T, N, 2·S_max+1]`.
    // `workspace` carries the per-sample loss buffer ([N] floats/doubles).
    // -------------------------------------------------------------------------

    /// CTCLoss FW, f32.
    pub fn baracuda_kernels_loss_ctc_f32_run(
        max_time: i32, batch_size: i32, num_classes: i32, max_target_len: i32,
        blank: i32, reduction_mode: i32, zero_infinity: i32,
        log_probs: *const c_void, targets: *const c_void,
        input_lengths: *const c_void, target_lengths: *const c_void,
        alpha_ws: *mut c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CTCLoss FW, f16.
    pub fn baracuda_kernels_loss_ctc_f16_run(
        max_time: i32, batch_size: i32, num_classes: i32, max_target_len: i32,
        blank: i32, reduction_mode: i32, zero_infinity: i32,
        log_probs: *const c_void, targets: *const c_void,
        input_lengths: *const c_void, target_lengths: *const c_void,
        alpha_ws: *mut c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CTCLoss FW, bf16.
    pub fn baracuda_kernels_loss_ctc_bf16_run(
        max_time: i32, batch_size: i32, num_classes: i32, max_target_len: i32,
        blank: i32, reduction_mode: i32, zero_infinity: i32,
        log_probs: *const c_void, targets: *const c_void,
        input_lengths: *const c_void, target_lengths: *const c_void,
        alpha_ws: *mut c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CTCLoss FW, f64.
    pub fn baracuda_kernels_loss_ctc_f64_run(
        max_time: i32, batch_size: i32, num_classes: i32, max_target_len: i32,
        blank: i32, reduction_mode: i32, zero_infinity: i32,
        log_probs: *const c_void, targets: *const c_void,
        input_lengths: *const c_void, target_lengths: *const c_void,
        alpha_ws: *mut c_void, out: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// CTCLoss BW, f32.
    pub fn baracuda_kernels_loss_ctc_backward_f32_run(
        max_time: i32, batch_size: i32, num_classes: i32, max_target_len: i32,
        blank: i32, reduction_mode: i32, zero_infinity: i32, inv_denom: f32,
        log_probs: *const c_void, targets: *const c_void,
        input_lengths: *const c_void, target_lengths: *const c_void,
        alpha_ws: *const c_void, per_sample_loss: *const c_void,
        dloss: *const c_void, dlog_probs: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CTCLoss BW, f16.
    pub fn baracuda_kernels_loss_ctc_backward_f16_run(
        max_time: i32, batch_size: i32, num_classes: i32, max_target_len: i32,
        blank: i32, reduction_mode: i32, zero_infinity: i32, inv_denom: f32,
        log_probs: *const c_void, targets: *const c_void,
        input_lengths: *const c_void, target_lengths: *const c_void,
        alpha_ws: *const c_void, per_sample_loss: *const c_void,
        dloss: *const c_void, dlog_probs: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CTCLoss BW, bf16.
    pub fn baracuda_kernels_loss_ctc_backward_bf16_run(
        max_time: i32, batch_size: i32, num_classes: i32, max_target_len: i32,
        blank: i32, reduction_mode: i32, zero_infinity: i32, inv_denom: f32,
        log_probs: *const c_void, targets: *const c_void,
        input_lengths: *const c_void, target_lengths: *const c_void,
        alpha_ws: *const c_void, per_sample_loss: *const c_void,
        dloss: *const c_void, dlog_probs: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// CTCLoss BW, f64.
    pub fn baracuda_kernels_loss_ctc_backward_f64_run(
        max_time: i32, batch_size: i32, num_classes: i32, max_target_len: i32,
        blank: i32, reduction_mode: i32, zero_infinity: i32, inv_denom: f32,
        log_probs: *const c_void, targets: *const c_void,
        input_lengths: *const c_void, target_lengths: *const c_void,
        alpha_ws: *const c_void, per_sample_loss: *const c_void,
        dloss: *const c_void, dlog_probs: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// PReLU FW, f32. ABI: `(numel, channel_stride, channel_extent,
    /// scalar_weight, x, weight, y, workspace, workspace_bytes, stream)`.
    pub fn baracuda_kernels_prelu_f32_run(
        numel: i64, channel_stride: i64,
        channel_extent: i32, scalar_weight: i32,
        x: *const c_void, weight: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// PReLU FW, f16.
    pub fn baracuda_kernels_prelu_f16_run(
        numel: i64, channel_stride: i64,
        channel_extent: i32, scalar_weight: i32,
        x: *const c_void, weight: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// PReLU FW, bf16.
    pub fn baracuda_kernels_prelu_bf16_run(
        numel: i64, channel_stride: i64,
        channel_extent: i32, scalar_weight: i32,
        x: *const c_void, weight: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// PReLU FW, f64.
    pub fn baracuda_kernels_prelu_f64_run(
        numel: i64, channel_stride: i64,
        channel_extent: i32, scalar_weight: i32,
        x: *const c_void, weight: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// PReLU BW, f32. ABI: `(numel, channel_stride, channel_extent,
    /// scalar_weight, dy, x, weight, dx, dweight, workspace, workspace_bytes, stream)`.
    pub fn baracuda_kernels_prelu_backward_f32_run(
        numel: i64, channel_stride: i64,
        channel_extent: i32, scalar_weight: i32,
        dy: *const c_void, x: *const c_void, weight: *const c_void,
        dx: *mut c_void, dweight: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// PReLU BW, f16.
    pub fn baracuda_kernels_prelu_backward_f16_run(
        numel: i64, channel_stride: i64,
        channel_extent: i32, scalar_weight: i32,
        dy: *const c_void, x: *const c_void, weight: *const c_void,
        dx: *mut c_void, dweight: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// PReLU BW, bf16.
    pub fn baracuda_kernels_prelu_backward_bf16_run(
        numel: i64, channel_stride: i64,
        channel_extent: i32, scalar_weight: i32,
        dy: *const c_void, x: *const c_void, weight: *const c_void,
        dx: *mut c_void, dweight: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// PReLU BW, f64.
    pub fn baracuda_kernels_prelu_backward_f64_run(
        numel: i64, channel_stride: i64,
        channel_extent: i32, scalar_weight: i32,
        dy: *const c_void, x: *const c_void, weight: *const c_void,
        dx: *mut c_void, dweight: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Soft-target CrossEntropy BW, f16.
    pub fn baracuda_kernels_loss_cross_entropy_soft_backward_f16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        row_stride_target: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Soft-target CrossEntropy BW, bf16.
    pub fn baracuda_kernels_loss_cross_entropy_soft_backward_bf16_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        row_stride_target: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Soft-target CrossEntropy BW, f64.
    pub fn baracuda_kernels_loss_cross_entropy_soft_backward_f64_run(
        n_rows: i64,
        class_extent: i32,
        row_stride_input: i64,
        row_stride_target: i64,
        reduction_mode: i32,
        scale_scalar: f32,
        input: *const c_void,
        target: *const c_void,
        dy: *const c_void,
        dinput: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Normalization family (Category G) — RMSNorm + LayerNorm + BatchNorm
// + GroupNorm (FW + BW).
//
// **RMSNorm / LayerNorm** use a multi-axis bitmask scheme: `norm_axes_mask`
// is an int32 bitmask (bit `d` set ⇒ axis `d` is normalized). The mask
// must be a suffix of `[0, rank)` (axes contiguous from the right —
// PyTorch's `normalized_shape` convention; validated in `can_implement`
// on the Rust side). `norm_total_extent` is the product of all axes in
// the mask. Per-output-cell two-pass row-stat scheme.
//
// **BatchNorm / GroupNorm** pre-collapse the input to logical
// `[N, C, S]` (S = product of spatial dims) and use a three-stage
// scheme: stage-1 per-group stat reduction, stage-2 per-cell normalize,
// stage-3 per-channel affine grads. group_kind selects BN (0) or GN/IN
// (1).
//
// BW launchers internally fire multiple kernels but all reductions are
// done via warp shuffles + smem — fully deterministic, no atomic-adds.
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// RMSNorm FW, f32. `y = x / sqrt(mean(x², over norm_axes) + eps) * gamma`.
    /// `norm_axes_mask` is a bitmask over input axes (suffix of `[0,
    /// rank)`); `norm_total_extent` is the product of those axes'
    /// extents. `gamma` may be null (treated as 1). `rms_out` shape
    /// equals input shape with norm axes collapsed to 1; only the
    /// slot at inner_lin == 0 within each row is written.
    pub fn baracuda_kernels_rms_norm_f32_run(
        eps: f32,
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_rms: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        x: *const c_void,
        gamma: *const c_void,
        y: *mut c_void,
        rms_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// RMSNorm FW, f16. f32 accumulator inside the kernel.
    pub fn baracuda_kernels_rms_norm_f16_run(
        eps: f32,
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_rms: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        x: *const c_void,
        gamma: *const c_void,
        y: *mut c_void,
        rms_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// RMSNorm FW, bf16. f32 accumulator inside the kernel.
    pub fn baracuda_kernels_rms_norm_bf16_run(
        eps: f32,
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_rms: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        x: *const c_void,
        gamma: *const c_void,
        y: *mut c_void,
        rms_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// RMSNorm FW, f64.
    pub fn baracuda_kernels_rms_norm_f64_run(
        eps: f32,
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_rms: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        x: *const c_void,
        gamma: *const c_void,
        y: *mut c_void,
        rms_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// RMSNorm BW, f32. Computes `dx` and (when `dgamma != null`)
    /// `dgamma[i] = Σ over outer cells dy[..., i] · (x[..., i] / rms[..., 0])`
    /// where `i` ranges over the joint normalized region of length
    /// `norm_total_extent`.
    pub fn baracuda_kernels_rms_norm_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_rms: *const i64,
        stride_dx: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        rms: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// RMSNorm BW, f16.
    pub fn baracuda_kernels_rms_norm_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_rms: *const i64,
        stride_dx: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        rms: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// RMSNorm BW, bf16.
    pub fn baracuda_kernels_rms_norm_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_rms: *const i64,
        stride_dx: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        rms: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// RMSNorm BW, f64.
    pub fn baracuda_kernels_rms_norm_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_rms: *const i64,
        stride_dx: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        rms: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LayerNorm FW, f32. `y = (x - mean) / sqrt(var + eps) * gamma + beta`.
    /// `gamma` / `beta` independently optional. Biased ("population")
    /// variance. Save buffers `mean_out` / `inv_std_out` share
    /// `stride_save`, each shape == input with norm axes collapsed to 1.
    pub fn baracuda_kernels_layer_norm_f32_run(
        eps: f32,
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_save: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        mean_out: *mut c_void,
        inv_std_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LayerNorm FW, f16. f32 accumulator inside the kernel.
    pub fn baracuda_kernels_layer_norm_f16_run(
        eps: f32,
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_save: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        mean_out: *mut c_void,
        inv_std_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LayerNorm FW, bf16. f32 accumulator inside the kernel.
    pub fn baracuda_kernels_layer_norm_bf16_run(
        eps: f32,
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_save: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        mean_out: *mut c_void,
        inv_std_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LayerNorm FW, f64.
    pub fn baracuda_kernels_layer_norm_f64_run(
        eps: f32,
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_save: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        mean_out: *mut c_void,
        inv_std_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LayerNorm BW, f32. Computes `dx` and (when non-null) `dgamma` /
    /// `dbeta` reductions. Caller passes saved `mean` + `inv_std` from FW.
    pub fn baracuda_kernels_layer_norm_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_save: *const i64,
        stride_dx: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        mean_in: *const c_void,
        inv_std_in: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LayerNorm BW, f16.
    pub fn baracuda_kernels_layer_norm_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_save: *const i64,
        stride_dx: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        mean_in: *const c_void,
        inv_std_in: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LayerNorm BW, bf16.
    pub fn baracuda_kernels_layer_norm_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_save: *const i64,
        stride_dx: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        mean_in: *const c_void,
        inv_std_in: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LayerNorm BW, f64.
    pub fn baracuda_kernels_layer_norm_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_save: *const i64,
        stride_dx: *const i64,
        norm_axes_mask: i32,
        norm_total_extent: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        mean_in: *const c_void,
        inv_std_in: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// BatchNorm + GroupNorm — Phase 5.1 Norm family completion.
//
// Caller pre-collapses input to logical [N, C, S] (S = product of
// spatial dims). Channel axis is axis 1 of the original tensor
// (PyTorch convention). All BN/GN kernels share the same launcher ABI;
// `group_kind` selects BN (0, one group per channel) vs GN/IN
// (1, num_groups caller-supplied; InstanceNorm = num_groups == c_extent).
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// BatchNorm FW, f32. Training mode: computes per-channel
    /// `(mean, inv_std)` from the batch + spatial cells, writes them to
    /// `saved_mean` / `saved_rstd` for BW. `gamma` / `beta` optional
    /// (both supplied together per PyTorch convention).
    pub fn baracuda_kernels_batch_norm_f32_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        eps: f32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        saved_mean: *mut c_void,
        saved_rstd: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// BatchNorm FW, f16.
    pub fn baracuda_kernels_batch_norm_f16_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        eps: f32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        saved_mean: *mut c_void,
        saved_rstd: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// BatchNorm FW, bf16.
    pub fn baracuda_kernels_batch_norm_bf16_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        eps: f32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        saved_mean: *mut c_void,
        saved_rstd: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// BatchNorm FW, f64.
    pub fn baracuda_kernels_batch_norm_f64_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        eps: f32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        saved_mean: *mut c_void,
        saved_rstd: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// BatchNorm BW, f32. Three-stage: per-group sum_dxh / sum_dxhxh,
    /// per-cell dx, per-channel dgamma / dbeta. Requires workspace of
    /// `2 * group_count * sizeof(float)` bytes for the stage-1 partial
    /// sums (group_count = c_extent for BN).
    pub fn baracuda_kernels_batch_norm_backward_f32_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        saved_mean: *const c_void,
        saved_rstd: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// BatchNorm BW, f16.
    pub fn baracuda_kernels_batch_norm_backward_f16_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        saved_mean: *const c_void,
        saved_rstd: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// BatchNorm BW, bf16.
    pub fn baracuda_kernels_batch_norm_backward_bf16_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        saved_mean: *const c_void,
        saved_rstd: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// BatchNorm BW, f64.
    pub fn baracuda_kernels_batch_norm_backward_f64_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        saved_mean: *const c_void,
        saved_rstd: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GroupNorm FW, f32. Per `(sample, group)` mean / inv_std,
    /// per-channel affine. `num_groups` must divide `c_extent`.
    /// `group_kind = 1` selects the GN dispatch (also used by
    /// InstanceNorm with `num_groups == c_extent`).
    pub fn baracuda_kernels_group_norm_f32_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        eps: f32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        saved_mean: *mut c_void,
        saved_rstd: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GroupNorm FW, f16.
    pub fn baracuda_kernels_group_norm_f16_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        eps: f32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        saved_mean: *mut c_void,
        saved_rstd: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GroupNorm FW, bf16.
    pub fn baracuda_kernels_group_norm_bf16_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        eps: f32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        saved_mean: *mut c_void,
        saved_rstd: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GroupNorm FW, f64.
    pub fn baracuda_kernels_group_norm_f64_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        eps: f32,
        x: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        y: *mut c_void,
        saved_mean: *mut c_void,
        saved_rstd: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GroupNorm BW, f32. Workspace size: `2 * (n_extent * num_groups) *
    /// sizeof(float)` bytes for the stage-1 partial sums.
    pub fn baracuda_kernels_group_norm_backward_f32_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        saved_mean: *const c_void,
        saved_rstd: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GroupNorm BW, f16.
    pub fn baracuda_kernels_group_norm_backward_f16_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        saved_mean: *const c_void,
        saved_rstd: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GroupNorm BW, bf16.
    pub fn baracuda_kernels_group_norm_backward_bf16_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        saved_mean: *const c_void,
        saved_rstd: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// GroupNorm BW, f64.
    pub fn baracuda_kernels_group_norm_backward_f64_run(
        n_extent: i32,
        c_extent: i32,
        spatial_extent: i32,
        num_groups: i32,
        group_kind: i32,
        dy: *const c_void,
        x: *const c_void,
        gamma: *const c_void,
        saved_mean: *const c_void,
        saved_rstd: *const c_void,
        dx: *mut c_void,
        dgamma: *mut c_void,
        dbeta: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Trace — sum of the diagonal of a 2-D square matrix. Dispatched
// through `TracePlan<T>` (not `ReducePlan`) because trace reduces both
// axes via the i==i constraint rather than a single reduce_axis.
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Trace of a 2-D square matrix, f32. `y[0] = Σ x[i * stride_row + i * stride_col]`
    /// for `i` in `0..rows`. Output is a single scalar.
    pub fn baracuda_kernels_trace_f32_run(
        rows: i32,
        stride_row: i64,
        stride_col: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Trace, f16 (f32-detour accumulator).
    pub fn baracuda_kernels_trace_f16_run(
        rows: i32,
        stride_row: i64,
        stride_col: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Trace, bf16 (f32-detour accumulator).
    pub fn baracuda_kernels_trace_bf16_run(
        rows: i32,
        stride_row: i64,
        stride_col: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Trace, f64.
    pub fn baracuda_kernels_trace_f64_run(
        rows: i32,
        stride_row: i64,
        stride_col: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Shape / layout — Flip (Category N)
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Flip (reverse along selected axes), f32. `flip_axes[d]` is
    /// 1 = reverse axis d, 0 = no-op.
    pub fn baracuda_kernels_flip_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        flip_axes: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Flip, f16. Pure element copy — no math.
    pub fn baracuda_kernels_flip_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        flip_axes: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Flip, bf16. Pure element copy — no math.
    pub fn baracuda_kernels_flip_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        flip_axes: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Flip, f64. Pure element copy — no math.
    pub fn baracuda_kernels_flip_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        flip_axes: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Shape / layout — Roll (Category N)
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Roll (cyclic shift along axes), f32. `shifts[d]` is the shift
    /// amount on axis d (positive or negative, mod shape[d]).
    pub fn baracuda_kernels_roll_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        shifts: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Roll, f16. Pure element copy — no math.
    pub fn baracuda_kernels_roll_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        shifts: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Roll, bf16. Pure element copy — no math.
    pub fn baracuda_kernels_roll_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        shifts: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Roll, f64. Pure element copy — no math.
    pub fn baracuda_kernels_roll_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        shifts: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Shape / layout — Repeat (Category N, per-axis tile)
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Repeat (per-axis tile), f32. `output.shape[d] =
    /// input.shape[d] * repeats[d]`. Kernel computes
    /// `input_coord[d] = output_coord[d] % input.shape[d]`.
    pub fn baracuda_kernels_repeat_f32_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Repeat (per-axis tile), f16. Same parameter shape as the f32
    /// variant — pure copy, no arithmetic.
    pub fn baracuda_kernels_repeat_f16_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Repeat (per-axis tile), bf16.
    pub fn baracuda_kernels_repeat_bf16_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Repeat (per-axis tile), f64.
    pub fn baracuda_kernels_repeat_f64_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Reductions — variance / std-dev (Phase 4 — Welford one-pass)
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Variance reduction along one axis, f32, Welford one-pass.
    /// `correction = 1` for Bessel-corrected sample variance, 0 for
    /// population variance.
    pub fn baracuda_kernels_reduce_var_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Std-dev along one axis, f32, Welford + sqrt.
    pub fn baracuda_kernels_reduce_std_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ---- Var / Std FW dtype fanout (Phase 4 deferral 4.2 close-out) ----
    // Welford state runs at the `WelfordAcc<T>` precision: f32 for
    // f16/bf16/f32 inputs (the f16/bf16 detour through f32 at load /
    // store time), f64 for f64 inputs. ABI identical to the f32 variant.

    /// Variance reduction along one axis, f16.
    pub fn baracuda_kernels_reduce_var_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Std-dev along one axis, f16.
    pub fn baracuda_kernels_reduce_std_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Variance reduction along one axis, bf16.
    pub fn baracuda_kernels_reduce_var_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Std-dev along one axis, bf16.
    pub fn baracuda_kernels_reduce_std_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Variance reduction along one axis, f64 (Welford in f64).
    pub fn baracuda_kernels_reduce_var_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Std-dev along one axis, f64 (Welford in f64 + sqrt).
    pub fn baracuda_kernels_reduce_std_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Reductions — argmax / argmin (Phase 4 — i64 index output)
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// `argmax(x, axis=k)`, f32 input, i64 output. Ties broken by
    /// first occurrence (smallest index wins).
    pub fn baracuda_kernels_arg_reduce_argmax_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `argmin(x, axis=k)`, f32 input, i64 output.
    pub fn baracuda_kernels_arg_reduce_argmin_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `argmax(x, axis=k)`, f16 input, i64 output.
    pub fn baracuda_kernels_arg_reduce_argmax_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `argmin(x, axis=k)`, f16 input, i64 output.
    pub fn baracuda_kernels_arg_reduce_argmin_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `argmax(x, axis=k)`, bf16 input, i64 output.
    pub fn baracuda_kernels_arg_reduce_argmax_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `argmin(x, axis=k)`, bf16 input, i64 output.
    pub fn baracuda_kernels_arg_reduce_argmin_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `argmax(x, axis=k)`, f64 input, i64 output.
    pub fn baracuda_kernels_arg_reduce_argmax_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `argmin(x, axis=k)`, f64 input, i64 output.
    pub fn baracuda_kernels_arg_reduce_argmin_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Reductions — any / all / count_nonzero (Phase 4 deferral 4.4 — heterogeneous
// output dtype: Any / All → uint8_t Bool output; CountNonzero → int64_t output)
// ============================================================================
//
// Parameter shape mirrors the simple-reduce family (same ABI as
// `baracuda_kernels_reduce_sum_f32_run`); only the output dtype is
// fixed by the symbol. Wired matrix per op:
//   {Any, All, CountNonzero} × {f32, f16, bf16, f64, i32, i64, Bool}
// = 21 SKUs total.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// `any(x, axis=k)` with f32 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_any_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `any(x, axis=k)` with f16 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_any_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `any(x, axis=k)` with bf16 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_any_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `any(x, axis=k)` with f64 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_any_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `any(x, axis=k)` with i32 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_any_i32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `any(x, axis=k)` with i64 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_any_i64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `any(x, axis=k)` with Bool (uint8_t) input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_any_bool_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `all(x, axis=k)` with f32 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_all_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `all(x, axis=k)` with f16 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_all_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `all(x, axis=k)` with bf16 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_all_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `all(x, axis=k)` with f64 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_all_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `all(x, axis=k)` with i32 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_all_i32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `all(x, axis=k)` with i64 input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_all_i64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `all(x, axis=k)` with Bool (uint8_t) input, uint8_t Bool output.
    pub fn baracuda_kernels_reduce_all_bool_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `count_nonzero(x, axis=k)` with f32 input, i64 output.
    pub fn baracuda_kernels_reduce_count_nonzero_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `count_nonzero(x, axis=k)` with f16 input, i64 output.
    pub fn baracuda_kernels_reduce_count_nonzero_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `count_nonzero(x, axis=k)` with bf16 input, i64 output.
    pub fn baracuda_kernels_reduce_count_nonzero_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `count_nonzero(x, axis=k)` with f64 input, i64 output.
    pub fn baracuda_kernels_reduce_count_nonzero_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `count_nonzero(x, axis=k)` with i32 input, i64 output.
    pub fn baracuda_kernels_reduce_count_nonzero_i32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `count_nonzero(x, axis=k)` with i64 input, i64 output.
    pub fn baracuda_kernels_reduce_count_nonzero_i64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `count_nonzero(x, axis=k)` with Bool (uint8_t) input, i64 output.
    pub fn baracuda_kernels_reduce_count_nonzero_bool_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Shape / layout — Permute (Category N)
// ============================================================================
//
// `y = x.permute(dims)` — output axis d is input axis `dims[d]`. The
// kernel walks input cells and writes to permuted output positions.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Materialized permute, f32.
    pub fn baracuda_kernels_permute_f32_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        dims: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Materialized permute, f16. Pure element copy — no math.
    pub fn baracuda_kernels_permute_f16_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        dims: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Materialized permute, bf16. Pure element copy — no math.
    pub fn baracuda_kernels_permute_bf16_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        dims: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Materialized permute, f64. Pure element copy — no math.
    pub fn baracuda_kernels_permute_f64_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        dims: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Shape / layout — Concat (2-input variant of Category N)
// ============================================================================
//
// `y = cat(a, b, dim=k)` with 2-input arity. Output shape per-axis
// matches a / b except `output[k] = a.shape[k] + b.shape[k]`. Variable-
// arity (N inputs) is a future plan shape (would need device-side
// packing of N pointers + N stride arrays through kernel param block).
// Today only f32 is wired; f16/bf16/f64 are single-INSTANTIATE fanout.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// `cat(a, b, dim)`, f32, contig output.
    ///
    /// `output_shape` matches a / b shape except `[concat_dim]` =
    /// `a.shape[concat_dim] + b.shape[concat_dim]`. `split_offset` is
    /// `a.shape[concat_dim]` — the kernel uses it to branch between
    /// reading from a or b.
    ///
    /// # Safety
    /// All device pointers must remain valid for the launch. Host
    /// arrays must remain valid for the host-side launch call.
    pub fn baracuda_kernels_concat2_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        concat_dim: i32,
        split_offset: i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `cat(a, b, dim)`, f16, contig output. See f32 variant.
    pub fn baracuda_kernels_concat2_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        concat_dim: i32,
        split_offset: i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `cat(a, b, dim)`, bf16, contig output. See f32 variant.
    pub fn baracuda_kernels_concat2_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        concat_dim: i32,
        split_offset: i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `cat(a, b, dim)`, f64, contig output. See f32 variant.
    pub fn baracuda_kernels_concat2_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        concat_dim: i32,
        split_offset: i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Shape / layout — Pad (Category N trailblazer)
// ============================================================================
//
// `y = pad(x, pad_low, pad_high, value)` over arbitrary-rank tensors.
// Output shape per-axis is `input[d] + pad_low[d] + pad_high[d]`.
// Today only f32 + constant mode is wired; future fanout adds the
// remaining dtypes and pad modes (Reflect, Replicate, Circular).

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Pad with a constant value, f32, contig output.
    ///
    /// `input_shape` / `output_shape` / `pad_low` point to host arrays
    /// of at least `rank` elements (i32). `stride_x` / `stride_y` are
    /// element-stride arrays (i64). The output is conventionally
    /// contiguous but the FFI accepts any stride pattern.
    ///
    /// # Safety
    /// All device pointers must remain valid for the duration of the
    /// launch. Host pointers must remain valid for the duration of the
    /// host-side launch call (the launcher copies them into kernel
    /// param-block structs).
    pub fn baracuda_kernels_pad_constant_f32_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        value: f32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad with a constant value, f16, contig output. The `value`
    /// argument carries the `__half` bit pattern as `u16` — Rust callers
    /// can produce it via `half::f16::to_bits()`. ABI-compatible because
    /// `__half` is a 2-byte `__CUDA_ALIGN__(2)` POD struct passed in the
    /// same register slot as `unsigned short`.
    pub fn baracuda_kernels_pad_constant_f16_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        value: u16,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad with a constant value, bf16, contig output. The `value`
    /// argument carries the `__nv_bfloat16` bit pattern as `u16` — Rust
    /// callers can produce it via `half::bf16::to_bits()`.
    pub fn baracuda_kernels_pad_constant_bf16_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        value: u16,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad with a constant value, f64, contig output.
    pub fn baracuda_kernels_pad_constant_f64_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        value: f64,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// Pad — Reflect / Replicate / Circular modes. None of these take a
// `value` parameter; the pad-region values come from the input itself
// (mirror, clamp, or cyclic wrap respectively). Parameter shape is
// otherwise identical to the constant-mode launchers.
#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Pad reflect, f32. Mirror input across the boundary (no edge
    /// duplication).
    pub fn baracuda_kernels_pad_reflect_f32_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad reflect, f16.
    pub fn baracuda_kernels_pad_reflect_f16_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad reflect, bf16.
    pub fn baracuda_kernels_pad_reflect_bf16_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad reflect, f64.
    pub fn baracuda_kernels_pad_reflect_f64_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad replicate, f32. Clamp to the edge value of the input.
    pub fn baracuda_kernels_pad_replicate_f32_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad replicate, f16.
    pub fn baracuda_kernels_pad_replicate_f16_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad replicate, bf16.
    pub fn baracuda_kernels_pad_replicate_bf16_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad replicate, f64.
    pub fn baracuda_kernels_pad_replicate_f64_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad circular, f32. Cyclic wrap from the opposite end of each
    /// axis.
    pub fn baracuda_kernels_pad_circular_f32_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad circular, f16.
    pub fn baracuda_kernels_pad_circular_f16_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad circular, bf16.
    pub fn baracuda_kernels_pad_circular_bf16_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad circular, f64.
    pub fn baracuda_kernels_pad_circular_f64_run(
        output_numel: i64,
        rank: i32,
        input_shape: *const i32,
        output_shape: *const i32,
        pad_low: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Shape / layout — Pad constant BW (slice)
// ============================================================================
//
// Backward of `y = pad(x, pad_low, pad_high, mode=Constant, value=v)`:
// `dx = dy[pad_low : pad_low + input_shape]` — pure slice. The
// gradient at pad-region cells is identically zero (the forward wrote
// a constant there) and is discarded. Iterates `input_numel` (dx-coord
// space). One launcher per fp dtype.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Pad-constant backward (slice), f32.
    pub fn baracuda_kernels_pad_constant_backward_f32_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        pad_low: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad-constant backward (slice), f16.
    pub fn baracuda_kernels_pad_constant_backward_f16_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        pad_low: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad-constant backward (slice), bf16.
    pub fn baracuda_kernels_pad_constant_backward_bf16_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        pad_low: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pad-constant backward (slice), f64.
    pub fn baracuda_kernels_pad_constant_backward_f64_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        pad_low: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Shape / layout — Repeat backward (Category N, gather-adjoint sum)
// ============================================================================
//
// Backward of `y = repeat(x, repeats)`: `dx[c_in] = sum_{k}
// dy[c_in + k * input_shape]` per axis — every dy cell whose
// `c_out[d] mod input_shape[d] == c_in[d]` for all d contributes. One
// thread per dx cell loops the per-axis repeats grid and accumulates;
// f16 / bf16 accumulate in float for numerical stability. Iterates
// `input_numel` (dx-coord space). One launcher per fp dtype.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Repeat backward (gather-adjoint sum), f32.
    pub fn baracuda_kernels_repeat_backward_f32_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        repeats: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Repeat backward (gather-adjoint sum), f16. Accumulates in float.
    pub fn baracuda_kernels_repeat_backward_f16_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        repeats: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Repeat backward (gather-adjoint sum), bf16. Accumulates in float.
    pub fn baracuda_kernels_repeat_backward_bf16_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        repeats: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Repeat backward (gather-adjoint sum), f64.
    pub fn baracuda_kernels_repeat_backward_f64_run(
        input_numel: i64,
        rank: i32,
        input_shape: *const i32,
        repeats: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Shape / layout — Concat2 backward (Category N, pure slice-split)
// ============================================================================
//
// Backward of `y = cat(a, b, dim=k)`: every dy cell maps to exactly one
// of `da` or `db`. Pure inverse routing — bit-exact across every wired
// dtype, no arithmetic. `da` collects `dy[..., :split_offset, ...]` and
// `db` collects `dy[..., split_offset:, ...]` along `concat_dim`. One
// thread per dy cell. Iterates `output_numel` (= dy.numel()).
// `split_offset` is `a.shape[concat_dim]` from the forward.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Concat2 backward (slice-split), f32. Bit-exact, no arithmetic.
    pub fn baracuda_kernels_concat2_backward_f32_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        concat_dim: i32,
        split_offset: i32,
        stride_dy: *const i64,
        stride_da: *const i64,
        stride_db: *const i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Concat2 backward (slice-split), f16. See f32 variant.
    pub fn baracuda_kernels_concat2_backward_f16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        concat_dim: i32,
        split_offset: i32,
        stride_dy: *const i64,
        stride_da: *const i64,
        stride_db: *const i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Concat2 backward (slice-split), bf16. See f32 variant.
    pub fn baracuda_kernels_concat2_backward_bf16_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        concat_dim: i32,
        split_offset: i32,
        stride_dy: *const i64,
        stride_da: *const i64,
        stride_db: *const i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Concat2 backward (slice-split), f64. See f32 variant.
    pub fn baracuda_kernels_concat2_backward_f64_run(
        output_numel: i64,
        rank: i32,
        output_shape: *const i32,
        concat_dim: i32,
        split_offset: i32,
        stride_dy: *const i64,
        stride_da: *const i64,
        stride_db: *const i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Elementwise — scaled ternary (3→1 ops with a scalar parameter)
// ============================================================================
//
// `Addcmul` and `Addcdiv` follow PyTorch's `torch.addcmul(c, a, b, value=k)`
// / `torch.addcdiv(c, a, b, value=k)` semantics:
//   addcmul: y = a + scale * b * c
//   addcdiv: y = a + scale * (b / c)
//
// FFI signature mirrors the unparameterized ternary launchers but
// inserts a `float scale` parameter between the y pointer and the
// workspace pointer. The Rust dispatcher reads `desc.scale` and
// forwards it; unparameterized ternary ops (Clamp, Fma) take a separate
// FFI without the scale arg (above).

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    // ---- addcmul ----
    /// `y = a + scale * b * c`, f32, contig fast path.
    pub fn baracuda_kernels_ternary_addcmul_f32_run(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `addcmul_f32`.
    pub fn baracuda_kernels_ternary_addcmul_f32_can_implement(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *const c_void,
    ) -> i32;
    /// `y = a + scale * b * c`, f32, strided / broadcast path.
    pub fn baracuda_kernels_ternary_addcmul_f32_strided_run(
        numel: i64, rank: i32,
        shape: *const i32,
        stride_a: *const i64, stride_b: *const i64, stride_c: *const i64, stride_y: *const i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `addcmul`, f16, contig.
    pub fn baracuda_kernels_ternary_addcmul_f16_run(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Pre-launch check for `addcmul_f16`.
    pub fn baracuda_kernels_ternary_addcmul_f16_can_implement(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *const c_void,
    ) -> i32;
    /// `addcmul`, f16, strided.
    pub fn baracuda_kernels_ternary_addcmul_f16_strided_run(
        numel: i64, rank: i32,
        shape: *const i32,
        stride_a: *const i64, stride_b: *const i64, stride_c: *const i64, stride_y: *const i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `addcmul`, bf16, contig.
    pub fn baracuda_kernels_ternary_addcmul_bf16_run(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Pre-launch check for `addcmul_bf16`.
    pub fn baracuda_kernels_ternary_addcmul_bf16_can_implement(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *const c_void,
    ) -> i32;
    /// `addcmul`, bf16, strided.
    pub fn baracuda_kernels_ternary_addcmul_bf16_strided_run(
        numel: i64, rank: i32,
        shape: *const i32,
        stride_a: *const i64, stride_b: *const i64, stride_c: *const i64, stride_y: *const i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `addcmul`, f64, contig.
    pub fn baracuda_kernels_ternary_addcmul_f64_run(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Pre-launch check for `addcmul_f64`.
    pub fn baracuda_kernels_ternary_addcmul_f64_can_implement(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *const c_void,
    ) -> i32;
    /// `addcmul`, f64, strided.
    pub fn baracuda_kernels_ternary_addcmul_f64_strided_run(
        numel: i64, rank: i32,
        shape: *const i32,
        stride_a: *const i64, stride_b: *const i64, stride_c: *const i64, stride_y: *const i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ---- addcdiv ----
    /// `y = a + scale * (b / c)`, f32, contig.
    pub fn baracuda_kernels_ternary_addcdiv_f32_run(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Pre-launch check for `addcdiv_f32`.
    pub fn baracuda_kernels_ternary_addcdiv_f32_can_implement(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *const c_void,
    ) -> i32;
    /// `addcdiv`, f32, strided.
    pub fn baracuda_kernels_ternary_addcdiv_f32_strided_run(
        numel: i64, rank: i32,
        shape: *const i32,
        stride_a: *const i64, stride_b: *const i64, stride_c: *const i64, stride_y: *const i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `addcdiv`, f16, contig.
    pub fn baracuda_kernels_ternary_addcdiv_f16_run(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Pre-launch check for `addcdiv_f16`.
    pub fn baracuda_kernels_ternary_addcdiv_f16_can_implement(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *const c_void,
    ) -> i32;
    /// `addcdiv`, f16, strided.
    pub fn baracuda_kernels_ternary_addcdiv_f16_strided_run(
        numel: i64, rank: i32,
        shape: *const i32,
        stride_a: *const i64, stride_b: *const i64, stride_c: *const i64, stride_y: *const i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `addcdiv`, bf16, contig.
    pub fn baracuda_kernels_ternary_addcdiv_bf16_run(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Pre-launch check for `addcdiv_bf16`.
    pub fn baracuda_kernels_ternary_addcdiv_bf16_can_implement(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *const c_void,
    ) -> i32;
    /// `addcdiv`, bf16, strided.
    pub fn baracuda_kernels_ternary_addcdiv_bf16_strided_run(
        numel: i64, rank: i32,
        shape: *const i32,
        stride_a: *const i64, stride_b: *const i64, stride_c: *const i64, stride_y: *const i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `addcdiv`, f64, contig.
    pub fn baracuda_kernels_ternary_addcdiv_f64_run(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Pre-launch check for `addcdiv_f64`.
    pub fn baracuda_kernels_ternary_addcdiv_f64_can_implement(
        numel: i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *const c_void,
    ) -> i32;
    /// `addcdiv`, f64, strided.
    pub fn baracuda_kernels_ternary_addcdiv_f64_strided_run(
        numel: i64, rank: i32,
        shape: *const i32,
        stride_a: *const i64, stride_b: *const i64, stride_c: *const i64, stride_y: *const i64,
        a: *const c_void, b: *const c_void, c: *const c_void, y: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Elementwise — where (heterogeneous-dtype ternary, u8 cond + T → T)
// ============================================================================
//
// `y = cond ? a : b` with `cond: TensorRef<u8, N>` (PyTorch / NumPy
// bool storage convention: 0 = false, non-zero = true) and same-dtype
// a / b / y. Distinct family from the homogeneous-dtype ternary path
// above — the cond input has a different dtype than the value inputs,
// so the FFI takes an extra `stride_cond` array on the strided path.
//
// All 4 FP value dtypes wired: {f32, f16, bf16, f64} × {contig,
// strided}.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// `where(cond, a, b)`, f32 values + u8 cond, contig fast path.
    ///
    /// `y = cond ? a : b` elementwise. `cond` is interpreted as bool
    /// per PyTorch convention (0 → b, non-zero → a).
    ///
    /// # Safety
    /// All device pointers must remain valid for the duration of the
    /// launch. `cond` must point to at least `numel` `u8`s; `a`, `b`,
    /// `y` to at least `numel` `f32`s.
    pub fn baracuda_kernels_where_f32_run(
        numel: i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch check for `where_f32`.
    pub fn baracuda_kernels_where_f32_can_implement(
        numel: i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// `where(cond, a, b)`, f32 values, strided / broadcast path.
    ///
    /// Each operand has its own stride array — cond can be broadcast
    /// independently from a and b (typical use: per-row mask
    /// `[M, 1] + [M, N] + [M, N]`).
    pub fn baracuda_kernels_where_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_cond: *const i64,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// `where(cond, a, b)`, f16 values + u8 cond, contig fast path.
    ///
    /// `y = cond ? a : b` elementwise. `cond` is interpreted as bool
    /// per PyTorch convention (0 → b, non-zero → a).
    ///
    /// # Safety
    /// All device pointers must remain valid for the duration of the
    /// launch. `cond` must point to at least `numel` `u8`s; `a`, `b`,
    /// `y` to at least `numel` `f16`s.
    pub fn baracuda_kernels_where_f16_run(
        numel: i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch check for `where_f16`.
    pub fn baracuda_kernels_where_f16_can_implement(
        numel: i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// `where(cond, a, b)`, f16 values, strided / broadcast path.
    ///
    /// Each operand has its own stride array — cond can be broadcast
    /// independently from a and b (typical use: per-row mask
    /// `[M, 1] + [M, N] + [M, N]`).
    pub fn baracuda_kernels_where_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_cond: *const i64,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// `where(cond, a, b)`, bf16 values + u8 cond, contig fast path.
    ///
    /// `y = cond ? a : b` elementwise. `cond` is interpreted as bool
    /// per PyTorch convention (0 → b, non-zero → a).
    ///
    /// # Safety
    /// All device pointers must remain valid for the duration of the
    /// launch. `cond` must point to at least `numel` `u8`s; `a`, `b`,
    /// `y` to at least `numel` `bf16`s.
    pub fn baracuda_kernels_where_bf16_run(
        numel: i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch check for `where_bf16`.
    pub fn baracuda_kernels_where_bf16_can_implement(
        numel: i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// `where(cond, a, b)`, bf16 values, strided / broadcast path.
    ///
    /// Each operand has its own stride array — cond can be broadcast
    /// independently from a and b (typical use: per-row mask
    /// `[M, 1] + [M, N] + [M, N]`).
    pub fn baracuda_kernels_where_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_cond: *const i64,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// `where(cond, a, b)`, f64 values + u8 cond, contig fast path.
    ///
    /// `y = cond ? a : b` elementwise. `cond` is interpreted as bool
    /// per PyTorch convention (0 → b, non-zero → a).
    ///
    /// # Safety
    /// All device pointers must remain valid for the duration of the
    /// launch. `cond` must point to at least `numel` `u8`s; `a`, `b`,
    /// `y` to at least `numel` `f64`s.
    pub fn baracuda_kernels_where_f64_run(
        numel: i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch check for `where_f64`.
    pub fn baracuda_kernels_where_f64_can_implement(
        numel: i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// `where(cond, a, b)`, f64 values, strided / broadcast path.
    ///
    /// Each operand has its own stride array — cond can be broadcast
    /// independently from a and b (typical use: per-row mask
    /// `[M, 1] + [M, N] + [M, N]`).
    pub fn baracuda_kernels_where_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_cond: *const i64,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        cond: *const c_void,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Elementwise — where backward (heterogeneous-dtype ternary BW, u8 cond + T → T,T)
// ============================================================================
//
// Forward: `y = cond ? a : b`. Backward (cond is non-differentiable):
//   da[i] = cond[i] ? dy[i] : 0
//   db[i] = cond[i] ? 0     : dy[i]
//
// Pure mask + copy: bit-exact at every dtype. Trailblazer is contig-only
// — broadcasting on dy / da / db is the caller's responsibility (it's
// what the autograd reduction step does upstream of this kernel anyway).
//
// All 4 FP value dtypes wired: {f32, f16, bf16, f64}.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// `where` backward, f32. Writes `da = cond ? dy : 0` and
    /// `db = cond ? 0 : dy`.
    ///
    /// # Safety
    /// All device pointers must remain valid for the duration of the
    /// launch. `cond` must point to at least `numel` `u8`s; `dy`, `da`,
    /// `db` to at least `numel` `f32`s.
    pub fn baracuda_kernels_where_backward_f32_run(
        numel: i64,
        cond: *const c_void,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `where` backward, f16.
    pub fn baracuda_kernels_where_backward_f16_run(
        numel: i64,
        cond: *const c_void,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `where` backward, bf16.
    pub fn baracuda_kernels_where_backward_bf16_run(
        numel: i64,
        cond: *const c_void,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `where` backward, f64.
    pub fn baracuda_kernels_where_backward_f64_run(
        numel: i64,
        cond: *const c_void,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Elementwise — binary backward ops (Phase 3 backward family)
// ============================================================================
//
// `(da, db) = backward(dy, [saved tensors per op])`. Two ABI shapes:
//
//   * **No-save backward** (Add, Sub) — gradient depends only on `dy`.
//     ABI: `(numel, dy, da, db, workspace, workspace_bytes, stream)`.
//   * **Saves-using backward** (Mul, Div) — gradient references the
//     saved forward inputs `a` and `b`.
//     ABI: `(numel, dy, a, b, da, db, workspace, workspace_bytes, stream)`.
//
// All four ops are wired across `{f32, f16, bf16, f64}`.
//
// Status codes match the elementwise forward family: 0 success, 2 invalid
// problem (e.g. negative `numel`, null pointer), 5 internal kernel error.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Add backward, f32. Writes `da = dy` and `db = dy`.
    pub fn baracuda_kernels_binary_add_backward_f32_run(
        numel: i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Add backward, f16.
    pub fn baracuda_kernels_binary_add_backward_f16_run(
        numel: i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Add backward, bf16.
    pub fn baracuda_kernels_binary_add_backward_bf16_run(
        numel: i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Add backward, f64.
    pub fn baracuda_kernels_binary_add_backward_f64_run(
        numel: i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sub backward, f32. Writes `da = dy` and `db = -dy`.
    pub fn baracuda_kernels_binary_sub_backward_f32_run(
        numel: i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sub backward, f16.
    pub fn baracuda_kernels_binary_sub_backward_f16_run(
        numel: i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sub backward, bf16.
    pub fn baracuda_kernels_binary_sub_backward_bf16_run(
        numel: i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Sub backward, f64.
    pub fn baracuda_kernels_binary_sub_backward_f64_run(
        numel: i64,
        dy: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Mul backward, f32. Writes `da = dy * b` and `db = dy * a`.
    /// Both saved tensors `a` and `b` must be non-null.
    pub fn baracuda_kernels_binary_mul_backward_f32_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Mul backward, f16.
    pub fn baracuda_kernels_binary_mul_backward_f16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Mul backward, bf16.
    pub fn baracuda_kernels_binary_mul_backward_bf16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Mul backward, f64.
    pub fn baracuda_kernels_binary_mul_backward_f64_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Div backward, f32. Writes `da = dy / b` and `db = -dy * a / b²`.
    /// Both saved tensors `a` and `b` must be non-null; callers must
    /// also ensure `b[i] != 0` for every cell.
    pub fn baracuda_kernels_binary_div_backward_f32_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Div backward, f16.
    pub fn baracuda_kernels_binary_div_backward_f16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Div backward, bf16.
    pub fn baracuda_kernels_binary_div_backward_bf16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Div backward, f64.
    pub fn baracuda_kernels_binary_div_backward_f64_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pow backward, f32. `da = dy * b * a^(b-1)`, `db = dy * a^b * ln(a)`.
    /// Caller responsible for guarding against undefined regions
    /// (`a < 0` non-integer `b`, or `a == 0` with `b < 1`).
    pub fn baracuda_kernels_binary_pow_backward_f32_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pow backward, f16.
    pub fn baracuda_kernels_binary_pow_backward_f16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pow backward, bf16.
    pub fn baracuda_kernels_binary_pow_backward_bf16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pow backward, f64.
    pub fn baracuda_kernels_binary_pow_backward_f64_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Atan2 backward, f32. `denom = a²+b²`, `da = dy*b/denom`,
    /// `db = -dy*a/denom`. Caller responsible for guarding against
    /// `a == 0 && b == 0` (denom == 0).
    pub fn baracuda_kernels_binary_atan2_backward_f32_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Atan2 backward, f16.
    pub fn baracuda_kernels_binary_atan2_backward_f16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Atan2 backward, bf16.
    pub fn baracuda_kernels_binary_atan2_backward_bf16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Atan2 backward, f64.
    pub fn baracuda_kernels_binary_atan2_backward_f64_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Hypot backward, f32. `y = sqrt(a²+b²)` is reconstructed inside
    /// the kernel from saved `a` and `b` (no saved-y slot in
    /// `BinaryBackwardArgs`); `da = dy*a/y`, `db = dy*b/y`. Caller
    /// responsible for guarding against `a == 0 && b == 0` (y == 0).
    pub fn baracuda_kernels_binary_hypot_backward_f32_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Hypot backward, f16.
    pub fn baracuda_kernels_binary_hypot_backward_f16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Hypot backward, bf16.
    pub fn baracuda_kernels_binary_hypot_backward_bf16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Hypot backward, f64.
    pub fn baracuda_kernels_binary_hypot_backward_f64_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Maximum backward, f32. Tie-break: split `dy` evenly on `a == b`;
    /// NaN inputs propagate `dy` to both. Saved `a` and `b` are used purely
    /// as references for the comparison.
    pub fn baracuda_kernels_binary_maximum_backward_f32_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Maximum backward, f16.
    pub fn baracuda_kernels_binary_maximum_backward_f16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Maximum backward, bf16.
    pub fn baracuda_kernels_binary_maximum_backward_bf16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Maximum backward, f64.
    pub fn baracuda_kernels_binary_maximum_backward_f64_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Minimum backward, f32. Tie-break: split `dy` evenly on `a == b`;
    /// NaN inputs propagate `dy` to both. Saved `a` and `b` are used purely
    /// as references for the comparison.
    pub fn baracuda_kernels_binary_minimum_backward_f32_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Minimum backward, f16.
    pub fn baracuda_kernels_binary_minimum_backward_f16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Minimum backward, bf16.
    pub fn baracuda_kernels_binary_minimum_backward_bf16_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Minimum backward, f64.
    pub fn baracuda_kernels_binary_minimum_backward_f64_run(
        numel: i64,
        dy: *const c_void,
        a: *const c_void,
        b: *const c_void,
        da: *mut c_void,
        db: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Elementwise — unary backward ops (Phase 3 unary-backward trailblazer)
// ============================================================================
//
// `dx = f'(saved) * dy` for the unary op family. The kernel ABI is
// uniform — one saved tensor of dtype `T` and one gradient input `dy`,
// producing one gradient output `dx`. Which save (`x` or `y`) the
// caller must pass depends on the op's BW formula:
//
//   * Saved-x ops (Sin, Cos, Log, ...): caller passes `x` as `saved`.
//     Example: Sin BW: `dx = dy * cos(x)`.
//   * Saved-y ops (Exp, Sigmoid, Tanh, Sqrt, ...): caller passes `y`
//     as `saved`. Example: Exp BW: `dx = dy * y`.
//
// Trailblazer scope: Sin BW × f32 (saved-x) and Exp BW × f32 (saved-y).
// Other ops / dtypes land in fanout.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Sin backward, f32. `dx = dy * cos(x)`. Caller must pass the
    /// forward input `x` as `saved`.
    pub fn baracuda_kernels_unary_sin_backward_f32_run(
        numel: i64,
        dy: *const c_void,
        saved: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Exp backward, f32. `dx = dy * y`. Caller must pass the forward
    /// output `y` as `saved`.
    pub fn baracuda_kernels_unary_exp_backward_f32_run(
        numel: i64,
        dy: *const c_void,
        saved: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Exp backward, f16.
    pub fn baracuda_kernels_unary_exp_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Exp backward, bf16.
    pub fn baracuda_kernels_unary_exp_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Exp backward, f64.
    pub fn baracuda_kernels_unary_exp_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Expm1 backward, f32. `dx = dy * (y + 1)`. Saved-y.
    pub fn baracuda_kernels_unary_expm1_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Expm1 backward, f16.
    pub fn baracuda_kernels_unary_expm1_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Expm1 backward, bf16.
    pub fn baracuda_kernels_unary_expm1_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Expm1 backward, f64.
    pub fn baracuda_kernels_unary_expm1_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Tanh backward, f32. `dx = dy * (1 - y²)`. Saved-y.
    pub fn baracuda_kernels_unary_tanh_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Tanh backward, f16.
    pub fn baracuda_kernels_unary_tanh_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Tanh backward, bf16.
    pub fn baracuda_kernels_unary_tanh_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Tanh backward, f64.
    pub fn baracuda_kernels_unary_tanh_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Sigmoid backward, f32. `dx = dy * y * (1 - y)`. Saved-y.
    pub fn baracuda_kernels_unary_sigmoid_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Sigmoid backward, f16.
    pub fn baracuda_kernels_unary_sigmoid_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Sigmoid backward, bf16.
    pub fn baracuda_kernels_unary_sigmoid_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Sigmoid backward, f64.
    pub fn baracuda_kernels_unary_sigmoid_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Sqrt backward, f32. `dx = dy / (2 * y)`. Saved-y. Callers must
    /// ensure `y[i] != 0`.
    pub fn baracuda_kernels_unary_sqrt_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Sqrt backward, f16.
    pub fn baracuda_kernels_unary_sqrt_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Sqrt backward, bf16.
    pub fn baracuda_kernels_unary_sqrt_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Sqrt backward, f64.
    pub fn baracuda_kernels_unary_sqrt_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Rsqrt backward, f32. `dx = -0.5 * dy * y³`. Saved-y.
    pub fn baracuda_kernels_unary_rsqrt_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Rsqrt backward, f16.
    pub fn baracuda_kernels_unary_rsqrt_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Rsqrt backward, bf16.
    pub fn baracuda_kernels_unary_rsqrt_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Rsqrt backward, f64.
    pub fn baracuda_kernels_unary_rsqrt_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Sin backward fanout (saves-x, transcendental) ----
    /// Sin backward, f16.
    pub fn baracuda_kernels_unary_sin_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Sin backward, bf16.
    pub fn baracuda_kernels_unary_sin_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Sin backward, f64.
    pub fn baracuda_kernels_unary_sin_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Log backward (saves-x, no transcendental) ----
    /// Log backward, f32. `dx = dy / x`.
    pub fn baracuda_kernels_unary_log_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log backward, f16.
    pub fn baracuda_kernels_unary_log_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log backward, bf16.
    pub fn baracuda_kernels_unary_log_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log backward, f64.
    pub fn baracuda_kernels_unary_log_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Log1p backward (saves-x, no transcendental) ----
    /// Log1p backward, f32. `dx = dy / (1 + x)`.
    pub fn baracuda_kernels_unary_log1p_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log1p backward, f16.
    pub fn baracuda_kernels_unary_log1p_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log1p backward, bf16.
    pub fn baracuda_kernels_unary_log1p_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log1p backward, f64.
    pub fn baracuda_kernels_unary_log1p_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Log2 backward (saves-x, constant ln(2)) ----
    /// Log2 backward, f32. `dx = dy / (x * ln(2))`.
    pub fn baracuda_kernels_unary_log2_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log2 backward, f16.
    pub fn baracuda_kernels_unary_log2_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log2 backward, bf16.
    pub fn baracuda_kernels_unary_log2_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log2 backward, f64.
    pub fn baracuda_kernels_unary_log2_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Log10 backward (saves-x, constant ln(10)) ----
    /// Log10 backward, f32. `dx = dy / (x * ln(10))`.
    pub fn baracuda_kernels_unary_log10_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log10 backward, f16.
    pub fn baracuda_kernels_unary_log10_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log10 backward, bf16.
    pub fn baracuda_kernels_unary_log10_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Log10 backward, f64.
    pub fn baracuda_kernels_unary_log10_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Atan backward (saves-x, no transcendental) ----
    /// Atan backward, f32. `dx = dy / (1 + x²)`.
    pub fn baracuda_kernels_unary_atan_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Atan backward, f16.
    pub fn baracuda_kernels_unary_atan_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Atan backward, bf16.
    pub fn baracuda_kernels_unary_atan_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Atan backward, f64.
    pub fn baracuda_kernels_unary_atan_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Cos backward (saves-x, transcendental) ----
    /// Cos backward, f32. `dx = -dy * sin(x)`. Saved-x.
    pub fn baracuda_kernels_unary_cos_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Cos backward, f16.
    pub fn baracuda_kernels_unary_cos_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Cos backward, bf16.
    pub fn baracuda_kernels_unary_cos_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Cos backward, f64.
    pub fn baracuda_kernels_unary_cos_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Tan backward (saves-x, transcendental) ----
    /// Tan backward, f32. `dx = dy * (1 + tan(x)²)`. Saved-x.
    pub fn baracuda_kernels_unary_tan_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Tan backward, f16.
    pub fn baracuda_kernels_unary_tan_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Tan backward, bf16.
    pub fn baracuda_kernels_unary_tan_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Tan backward, f64.
    pub fn baracuda_kernels_unary_tan_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Sinh backward (saves-x, transcendental) ----
    /// Sinh backward, f32. `dx = dy * cosh(x)`. Saved-x.
    pub fn baracuda_kernels_unary_sinh_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Sinh backward, f16.
    pub fn baracuda_kernels_unary_sinh_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Sinh backward, bf16.
    pub fn baracuda_kernels_unary_sinh_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Sinh backward, f64.
    pub fn baracuda_kernels_unary_sinh_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Cosh backward (saves-x, transcendental) ----
    /// Cosh backward, f32. `dx = dy * sinh(x)`. Saved-x.
    pub fn baracuda_kernels_unary_cosh_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Cosh backward, f16.
    pub fn baracuda_kernels_unary_cosh_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Cosh backward, bf16.
    pub fn baracuda_kernels_unary_cosh_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Cosh backward, f64.
    pub fn baracuda_kernels_unary_cosh_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Asin backward (saves-x, sqrt) ----
    /// Asin backward, f32. `dx = dy / sqrt(1 - x²)`. Saved-x. Domain: `|x| < 1`.
    pub fn baracuda_kernels_unary_asin_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Asin backward, f16.
    pub fn baracuda_kernels_unary_asin_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Asin backward, bf16.
    pub fn baracuda_kernels_unary_asin_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Asin backward, f64.
    pub fn baracuda_kernels_unary_asin_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Acos backward (saves-x, sqrt) ----
    /// Acos backward, f32. `dx = -dy / sqrt(1 - x²)`. Saved-x. Domain: `|x| < 1`.
    pub fn baracuda_kernels_unary_acos_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Acos backward, f16.
    pub fn baracuda_kernels_unary_acos_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Acos backward, bf16.
    pub fn baracuda_kernels_unary_acos_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Acos backward, f64.
    pub fn baracuda_kernels_unary_acos_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Asinh backward (saves-x, sqrt) ----
    /// Asinh backward, f32. `dx = dy / sqrt(1 + x²)`. Saved-x.
    pub fn baracuda_kernels_unary_asinh_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Asinh backward, f16.
    pub fn baracuda_kernels_unary_asinh_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Asinh backward, bf16.
    pub fn baracuda_kernels_unary_asinh_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Asinh backward, f64.
    pub fn baracuda_kernels_unary_asinh_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Acosh backward (saves-x, sqrt) ----
    /// Acosh backward, f32. `dx = dy / sqrt(x² - 1)`. Saved-x. Domain: `x > 1`.
    pub fn baracuda_kernels_unary_acosh_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Acosh backward, f16.
    pub fn baracuda_kernels_unary_acosh_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Acosh backward, bf16.
    pub fn baracuda_kernels_unary_acosh_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Acosh backward, f64.
    pub fn baracuda_kernels_unary_acosh_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Atanh backward (saves-x, no transcendental) ----
    /// Atanh backward, f32. `dx = dy / (1 - x²)`. Saved-x. Domain: `|x| < 1`.
    pub fn baracuda_kernels_unary_atanh_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Atanh backward, f16.
    pub fn baracuda_kernels_unary_atanh_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Atanh backward, bf16.
    pub fn baracuda_kernels_unary_atanh_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Atanh backward, f64.
    pub fn baracuda_kernels_unary_atanh_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Square backward (saves-x, dy * 2 * x) ----
    /// Square backward, f32. `dx = dy * 2 * x`.
    pub fn baracuda_kernels_unary_square_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Square backward, f16.
    pub fn baracuda_kernels_unary_square_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Square backward, bf16.
    pub fn baracuda_kernels_unary_square_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Square backward, f64.
    pub fn baracuda_kernels_unary_square_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Cube backward (saves-x, dy * 3 * x²) ----
    /// Cube backward, f32. `dx = dy * 3 * x²`.
    pub fn baracuda_kernels_unary_cube_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Cube backward, f16.
    pub fn baracuda_kernels_unary_cube_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Cube backward, bf16.
    pub fn baracuda_kernels_unary_cube_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Cube backward, f64.
    pub fn baracuda_kernels_unary_cube_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Exp2 backward (saves-y, dy * y * ln(2)) ----
    /// Exp2 backward, f32. `dx = dy * y * ln(2)`.
    pub fn baracuda_kernels_unary_exp2_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Exp2 backward, f16.
    pub fn baracuda_kernels_unary_exp2_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Exp2 backward, bf16.
    pub fn baracuda_kernels_unary_exp2_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Exp2 backward, f64.
    pub fn baracuda_kernels_unary_exp2_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Tanhshrink backward (saves-x, dy * tanh(x)²) ----
    /// Tanhshrink backward, f32. `dx = dy * tanh(x)²`.
    pub fn baracuda_kernels_unary_tanhshrink_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Tanhshrink backward, f16.
    pub fn baracuda_kernels_unary_tanhshrink_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Tanhshrink backward, bf16.
    pub fn baracuda_kernels_unary_tanhshrink_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Tanhshrink backward, f64.
    pub fn baracuda_kernels_unary_tanhshrink_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Logit backward (saves-x, dy / (x * (1 - x))) ----
    /// Logit backward, f32. `dx = dy / (x * (1 - x))`. Domain `0 < x < 1`.
    pub fn baracuda_kernels_unary_logit_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Logit backward, f16.
    pub fn baracuda_kernels_unary_logit_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Logit backward, bf16.
    pub fn baracuda_kernels_unary_logit_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Logit backward, f64.
    pub fn baracuda_kernels_unary_logit_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Reciprocal backward (saves-x, -dy / x²) ----
    /// Reciprocal backward, f32. `dx = -dy / x²`. Domain `x != 0`.
    pub fn baracuda_kernels_unary_reciprocal_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Reciprocal backward, f16.
    pub fn baracuda_kernels_unary_reciprocal_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Reciprocal backward, bf16.
    pub fn baracuda_kernels_unary_reciprocal_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Reciprocal backward, f64.
    pub fn baracuda_kernels_unary_reciprocal_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Erf backward (saves-x, transcendental, 2/√π * exp(-x²)) ----
    /// Erf backward, f32. `dx = dy * (2/√π) * exp(-x²)`.
    pub fn baracuda_kernels_unary_erf_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Erf backward, f16.
    pub fn baracuda_kernels_unary_erf_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Erf backward, bf16.
    pub fn baracuda_kernels_unary_erf_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Erf backward, f64.
    pub fn baracuda_kernels_unary_erf_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Erfc backward (saves-x, transcendental, -2/√π * exp(-x²)) ----
    /// Erfc backward, f32. `dx = -dy * (2/√π) * exp(-x²)`.
    pub fn baracuda_kernels_unary_erfc_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Erfc backward, f16.
    pub fn baracuda_kernels_unary_erfc_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Erfc backward, bf16.
    pub fn baracuda_kernels_unary_erfc_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Erfc backward, f64.
    pub fn baracuda_kernels_unary_erfc_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Activation BW (saved-x, piecewise — Category B' trailblazer + fanout) ----

    /// ReLU backward, f32. `dx = (x > 0) ? dy : 0`. Saved-x.
    pub fn baracuda_kernels_unary_relu_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReLU backward, f16.
    pub fn baracuda_kernels_unary_relu_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReLU backward, bf16.
    pub fn baracuda_kernels_unary_relu_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReLU backward, f64.
    pub fn baracuda_kernels_unary_relu_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Hardtanh backward, f32. `dx = (-1 < x < 1) ? dy : 0`. Saved-x.
    pub fn baracuda_kernels_unary_hardtanh_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardtanh backward, f16.
    pub fn baracuda_kernels_unary_hardtanh_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardtanh backward, bf16.
    pub fn baracuda_kernels_unary_hardtanh_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardtanh backward, f64.
    pub fn baracuda_kernels_unary_hardtanh_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// ReLU6 backward, f32. `dx = (0 < x < 6) ? dy : 0`. Saved-x.
    pub fn baracuda_kernels_unary_relu6_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReLU6 backward, f16.
    pub fn baracuda_kernels_unary_relu6_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReLU6 backward, bf16.
    pub fn baracuda_kernels_unary_relu6_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReLU6 backward, f64.
    pub fn baracuda_kernels_unary_relu6_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Hardsigmoid backward, f32. `dx = (-3 < x < 3) ? dy / 6 : 0`. Saved-x.
    pub fn baracuda_kernels_unary_hardsigmoid_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardsigmoid backward, f16.
    pub fn baracuda_kernels_unary_hardsigmoid_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardsigmoid backward, bf16.
    pub fn baracuda_kernels_unary_hardsigmoid_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardsigmoid backward, f64.
    pub fn baracuda_kernels_unary_hardsigmoid_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Hardswish backward, f32. Three-region piecewise + `(2x+3)/6` middle. Saved-x.
    pub fn baracuda_kernels_unary_hardswish_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardswish backward, f16.
    pub fn baracuda_kernels_unary_hardswish_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardswish backward, bf16.
    pub fn baracuda_kernels_unary_hardswish_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardswish backward, f64.
    pub fn baracuda_kernels_unary_hardswish_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Softplus backward, f32. `dx = dy / (1 + exp(-x))`. Saved-x.
    pub fn baracuda_kernels_unary_softplus_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Softplus backward, f16.
    pub fn baracuda_kernels_unary_softplus_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Softplus backward, bf16.
    pub fn baracuda_kernels_unary_softplus_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Softplus backward, f64.
    pub fn baracuda_kernels_unary_softplus_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// SiLU (Swish) backward, f32. `dx = dy * s * (1 + x*(1-s))` with `s = sigmoid(x)`. Saved-x.
    pub fn baracuda_kernels_unary_silu_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SiLU backward, f16.
    pub fn baracuda_kernels_unary_silu_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SiLU backward, bf16.
    pub fn baracuda_kernels_unary_silu_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SiLU backward, f64.
    pub fn baracuda_kernels_unary_silu_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Mish backward, f32. `dx = dy * (tanh(sp) + x*s*(1 - tanh(sp)^2))`, `sp = softplus(x)`. Saved-x.
    pub fn baracuda_kernels_unary_mish_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Mish backward, f16.
    pub fn baracuda_kernels_unary_mish_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Mish backward, bf16.
    pub fn baracuda_kernels_unary_mish_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Mish backward, f64.
    pub fn baracuda_kernels_unary_mish_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// GELU (exact / erf-based) backward, f32. `dx = dy * (Φ(x) + x*φ(x))`. Saved-x.
    pub fn baracuda_kernels_unary_gelu_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GELU (erf-based) backward, f16.
    pub fn baracuda_kernels_unary_gelu_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GELU (erf-based) backward, bf16.
    pub fn baracuda_kernels_unary_gelu_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GELU (erf-based) backward, f64.
    pub fn baracuda_kernels_unary_gelu_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// GELU (tanh approximation) backward, f32. Saved-x.
    pub fn baracuda_kernels_unary_gelu_tanh_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GELU (tanh approximation) backward, f16.
    pub fn baracuda_kernels_unary_gelu_tanh_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GELU (tanh approximation) backward, bf16.
    pub fn baracuda_kernels_unary_gelu_tanh_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GELU (tanh approximation) backward, f64.
    pub fn baracuda_kernels_unary_gelu_tanh_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// SELU backward, f32. `x>0 → dy*scale`; `x<=0 → dy*scale*alpha*exp(x)`. Saved-x.
    pub fn baracuda_kernels_unary_selu_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SELU backward, f16.
    pub fn baracuda_kernels_unary_selu_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SELU backward, bf16.
    pub fn baracuda_kernels_unary_selu_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SELU backward, f64.
    pub fn baracuda_kernels_unary_selu_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Parameterized-activation BW (hardcoded defaults — LeakyRelu
    // α=0.01, ELU α=1.0, Hardshrink λ=0.5, Softshrink λ=0.5). All
    // saved-x, no strided BW path (matches the existing activation BW
    // pattern). When the parameterized-unary plan ships these get
    // re-emitted with the parameter as a runtime arg. ----

    /// LeakyReLU backward, f32. `dx = (x > 0) ? dy : dy·α` with α=0.01. Saved-x.
    pub fn baracuda_kernels_unary_leaky_relu_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// LeakyReLU backward, f16.
    pub fn baracuda_kernels_unary_leaky_relu_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// LeakyReLU backward, bf16.
    pub fn baracuda_kernels_unary_leaky_relu_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// LeakyReLU backward, f64.
    pub fn baracuda_kernels_unary_leaky_relu_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// ELU backward, f32. `dx = (x > 0) ? dy : dy·α·exp(x)` with α=1.0. Saved-x.
    pub fn baracuda_kernels_unary_elu_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ELU backward, f16.
    pub fn baracuda_kernels_unary_elu_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ELU backward, bf16.
    pub fn baracuda_kernels_unary_elu_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ELU backward, f64.
    pub fn baracuda_kernels_unary_elu_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Hardshrink backward, f32. `dx = (|x| > λ) ? dy : 0` with λ=0.5. Saved-x.
    pub fn baracuda_kernels_unary_hardshrink_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardshrink backward, f16.
    pub fn baracuda_kernels_unary_hardshrink_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardshrink backward, bf16.
    pub fn baracuda_kernels_unary_hardshrink_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Hardshrink backward, f64.
    pub fn baracuda_kernels_unary_hardshrink_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Softshrink backward, f32. `dx = (|x| > λ) ? dy : 0` with λ=0.5. Saved-x.
    pub fn baracuda_kernels_unary_softshrink_backward_f32_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Softshrink backward, f16.
    pub fn baracuda_kernels_unary_softshrink_backward_f16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Softshrink backward, bf16.
    pub fn baracuda_kernels_unary_softshrink_backward_bf16_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Softshrink backward, f64.
    pub fn baracuda_kernels_unary_softshrink_backward_f64_run(
        numel: i64, dy: *const c_void, saved: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Reduction backward (Phase 4 BW trailblazer)
// ============================================================================
//
// `dx[c] = dy[c with reduce_axis collapsed]` — Sum BW is a pure
// broadcast-copy of dy across the reduced axis. The Rust dispatcher
// constructs the dy strides with `stride[reduce_axis] = 0` so the
// kernel just walks the dx coord space and reads dy via strides.
//
// ABI mirrors the binary strided launcher: `(numel, rank, shape,
// stride_dy, stride_dx, dy, dx, ws, ws_bytes, stream)`. `shape` is the
// full dx shape.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Sum reduction backward, f32. `dx[c] = dy[c_with_reduce_axis_0]`
    /// realized via stride-0 broadcast on the reduce axis.
    pub fn baracuda_kernels_reduce_sum_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Sum reduction backward, f16.
    pub fn baracuda_kernels_reduce_sum_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Sum reduction backward, bf16.
    pub fn baracuda_kernels_reduce_sum_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Sum reduction backward, f64.
    pub fn baracuda_kernels_reduce_sum_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Mean reduction backward, f32. Same as Sum BW with extra `1/k`
    /// scale (`inv_extent` is `1.0 / reduced_extent` computed in f64
    /// on the host).
    pub fn baracuda_kernels_reduce_mean_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        inv_extent: f64,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Mean reduction backward, f16.
    pub fn baracuda_kernels_reduce_mean_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        inv_extent: f64,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Mean reduction backward, bf16.
    pub fn baracuda_kernels_reduce_mean_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        inv_extent: f64,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Mean reduction backward, f64.
    pub fn baracuda_kernels_reduce_mean_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        dx: *mut c_void,
        inv_extent: f64,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ---- Max / Min reduction backward ----
    //
    // Single kernel serves BOTH Max BW and Min BW. Compares `x[c]` to
    // saved forward output `y[c_reduced]`; matching positions receive
    // `dy[c_reduced]`, others get 0. Tie semantic: every tied position
    // gets the full gradient (split-across-ties / JAX convention).

    /// Max/Min reduction backward, f32.
    pub fn baracuda_kernels_reduce_max_min_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Max/Min reduction backward, f16.
    pub fn baracuda_kernels_reduce_max_min_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Max/Min reduction backward, bf16.
    pub fn baracuda_kernels_reduce_max_min_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Max/Min reduction backward, f64.
    pub fn baracuda_kernels_reduce_max_min_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ---- Prod backward — dual-save (saved x and saved y) ------------
    // `dx[c] = dy[c_reduced] * y[c_reduced] / x[c]`.

    /// Prod reduction backward, f32.
    pub fn baracuda_kernels_reduce_prod_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Prod reduction backward, f16.
    pub fn baracuda_kernels_reduce_prod_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Prod reduction backward, bf16.
    pub fn baracuda_kernels_reduce_prod_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Prod reduction backward, f64.
    pub fn baracuda_kernels_reduce_prod_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ---- Norm2 backward — dual-save (saved x and saved y) -----------
    // `dx[c] = dy[c_reduced] * x[c] / y[c_reduced]` where
    // `y = sqrt(sum(x², axis=k))`.

    /// Norm2 reduction backward, f32.
    pub fn baracuda_kernels_reduce_norm2_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Norm2 reduction backward, f16.
    pub fn baracuda_kernels_reduce_norm2_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Norm2 reduction backward, bf16.
    pub fn baracuda_kernels_reduce_norm2_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Norm2 reduction backward, f64.
    pub fn baracuda_kernels_reduce_norm2_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ---- LogSumExp backward — `dy * exp(x - y)`, dual-save ----------
    // `dx[c] = dy[c_reduced] * exp(x[c] - y[c_reduced])` where
    // `y = log(sum(exp(x), axis=k)) + max`. Always numerically safe:
    // `x - y ≤ 0`, so `exp(x - y) ∈ (0, 1]`. f16 / bf16 do the exp in
    // f32; f32 / f64 use libdevice `expf` / `exp`.

    /// LogSumExp reduction backward, f32.
    pub fn baracuda_kernels_reduce_logsumexp_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSumExp reduction backward, f16.
    pub fn baracuda_kernels_reduce_logsumexp_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSumExp reduction backward, bf16.
    pub fn baracuda_kernels_reduce_logsumexp_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// LogSumExp reduction backward, f64.
    pub fn baracuda_kernels_reduce_logsumexp_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ---- Var / Std backward — Welford BW, f32-only ------------------
    // Var BW: `dx[c] = dy[c_reduced] * 2 * (x[c] - mean[c_reduced]) / m`
    // Std BW: `dx[c] = dy[c_reduced] * (x[c] - mean[c_reduced]) / (m * y[c_reduced])`
    // where `m = max(n - correction, 1)` and `n = reduce_extent`. Mean
    // is recomputed inside the kernel (single-pass sum/n over the
    // reduce axis on `x`). Saved-x required; saved-y required for Std
    // BW and ignored by Var BW (pass null or any valid pointer).
    // f32-only — matches the FW Welford scope.

    /// Variance reduction backward, f32 (Welford BW).
    pub fn baracuda_kernels_reduce_var_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Std-dev reduction backward, f32 (Welford BW + sqrt term).
    pub fn baracuda_kernels_reduce_std_backward_f32_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // ---- Var / Std BW dtype fanout (Phase 4 deferral 4.2 close-out) ----
    // Internal accumulation runs at `WelfordAcc<T>`: f32 for
    // f16/bf16/f32, f64 for f64. ABI identical to the f32 variants.

    /// Variance reduction backward, f16.
    pub fn baracuda_kernels_reduce_var_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Std-dev reduction backward, f16.
    pub fn baracuda_kernels_reduce_std_backward_f16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Variance reduction backward, bf16.
    pub fn baracuda_kernels_reduce_var_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Std-dev reduction backward, bf16.
    pub fn baracuda_kernels_reduce_std_backward_bf16_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Variance reduction backward, f64 (Welford BW in f64).
    pub fn baracuda_kernels_reduce_var_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Std-dev reduction backward, f64 (Welford BW in f64 + sqrt term).
    pub fn baracuda_kernels_reduce_std_backward_f64_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_dy: *const i64,
        stride_x: *const i64,
        stride_y: *const i64,
        stride_dx: *const i64,
        dy: *const c_void,
        x: *const c_void,
        y: *const c_void,
        dx: *mut c_void,
        reduce_axis: i32,
        reduce_extent: i32,
        reduce_stride_x: i64,
        correction: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Elementwise — binary comparison ops (T → bool)
// ============================================================================
//
// Same shape as the binary contig + strided launchers above, but the
// output is `uint8_t` (0 / 1) rather than `T`. The kernel returns the
// comparison result as a bool stored in one byte — PyTorch / NumPy
// convention. The C ABI uses `void*` for the output pointer; the
// kernel wrapper casts it to `uint8_t*` internally.
//
// Full matrix wired: {Eq, Ne, Gt, Ge, Lt, Le} ops × {f32, f16, bf16,
// f64} dtypes × {contig, strided} = 48 launchers (3 symbols per cell:
// `_run`, `_can_implement`, `_strided_run`). NaN handling follows IEEE
// 754: `Eq` / ordered comparisons return 0 when either operand is NaN;
// `Ne` returns 1 (since `NaN != anything`).

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    // --- Eq -----------------------------------------------------------------

    /// Binary elementwise `eq`, f32 inputs, u8 output, contig fast path.
    ///
    /// # Safety
    /// All device pointers must remain valid for the duration of the
    /// launch. `y` must point to at least `numel` `u8`s. The kernel
    /// writes only `0u8` and `1u8` to `y`.
    pub fn baracuda_kernels_binary_cmp_eq_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_eq_f32`.
    pub fn baracuda_kernels_binary_cmp_eq_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `eq`, f32 inputs, u8 output, strided path.
    ///
    /// Handles non-contig views (broadcast / transposed / sliced). The
    /// output's stride is in u8 elements (one element per byte).
    pub fn baracuda_kernels_binary_cmp_eq_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `eq`, f16 inputs, u8 output, contig fast path.
    ///
    /// # Safety
    /// See `baracuda_kernels_binary_cmp_eq_f32_run`. Inputs are
    /// `__half` (one rounding step when storing — but `==` on bit
    /// patterns is exact, so the GPU result matches host
    /// `half::f16 == half::f16`).
    pub fn baracuda_kernels_binary_cmp_eq_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_eq_f16`.
    pub fn baracuda_kernels_binary_cmp_eq_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `eq`, f16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_eq_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `eq`, bf16 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_eq_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_eq_bf16`.
    pub fn baracuda_kernels_binary_cmp_eq_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `eq`, bf16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_eq_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `eq`, f64 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_eq_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_eq_f64`.
    pub fn baracuda_kernels_binary_cmp_eq_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `eq`, f64 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_eq_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// --- Ne -------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Binary elementwise `ne`, f32 inputs, u8 output, contig fast path.
    ///
    /// `NaN != anything` returns 1 per IEEE 754.
    ///
    /// # Safety
    /// See `baracuda_kernels_binary_cmp_eq_f32_run`.
    pub fn baracuda_kernels_binary_cmp_ne_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_ne_f32`.
    pub fn baracuda_kernels_binary_cmp_ne_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `ne`, f32 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_ne_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `ne`, f16 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_ne_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_ne_f16`.
    pub fn baracuda_kernels_binary_cmp_ne_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `ne`, f16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_ne_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `ne`, bf16 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_ne_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_ne_bf16`.
    pub fn baracuda_kernels_binary_cmp_ne_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `ne`, bf16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_ne_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `ne`, f64 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_ne_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_ne_f64`.
    pub fn baracuda_kernels_binary_cmp_ne_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `ne`, f64 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_ne_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// --- Gt -------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Binary elementwise `gt` (`a > b`), f32 inputs, u8 output, contig fast path.
    ///
    /// Any comparison involving NaN returns 0 per IEEE 754.
    ///
    /// # Safety
    /// See `baracuda_kernels_binary_cmp_eq_f32_run`.
    pub fn baracuda_kernels_binary_cmp_gt_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_gt_f32`.
    pub fn baracuda_kernels_binary_cmp_gt_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `gt`, f32 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_gt_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `gt`, f16 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_gt_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_gt_f16`.
    pub fn baracuda_kernels_binary_cmp_gt_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `gt`, f16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_gt_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `gt`, bf16 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_gt_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_gt_bf16`.
    pub fn baracuda_kernels_binary_cmp_gt_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `gt`, bf16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_gt_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `gt`, f64 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_gt_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_gt_f64`.
    pub fn baracuda_kernels_binary_cmp_gt_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `gt`, f64 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_gt_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// --- Ge -------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Binary elementwise `ge` (`a >= b`), f32 inputs, u8 output, contig fast path.
    ///
    /// Any comparison involving NaN returns 0 per IEEE 754.
    ///
    /// # Safety
    /// See `baracuda_kernels_binary_cmp_eq_f32_run`.
    pub fn baracuda_kernels_binary_cmp_ge_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_ge_f32`.
    pub fn baracuda_kernels_binary_cmp_ge_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `ge`, f32 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_ge_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `ge`, f16 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_ge_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_ge_f16`.
    pub fn baracuda_kernels_binary_cmp_ge_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `ge`, f16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_ge_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `ge`, bf16 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_ge_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_ge_bf16`.
    pub fn baracuda_kernels_binary_cmp_ge_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `ge`, bf16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_ge_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `ge`, f64 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_ge_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_ge_f64`.
    pub fn baracuda_kernels_binary_cmp_ge_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `ge`, f64 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_ge_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// --- Lt -------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Binary elementwise `lt` (`a < b`), f32 inputs, u8 output, contig fast path.
    ///
    /// Any comparison involving NaN returns 0 per IEEE 754.
    ///
    /// # Safety
    /// See `baracuda_kernels_binary_cmp_eq_f32_run`.
    pub fn baracuda_kernels_binary_cmp_lt_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_lt_f32`.
    pub fn baracuda_kernels_binary_cmp_lt_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `lt`, f32 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_lt_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `lt`, f16 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_lt_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_lt_f16`.
    pub fn baracuda_kernels_binary_cmp_lt_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `lt`, f16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_lt_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `lt`, bf16 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_lt_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_lt_bf16`.
    pub fn baracuda_kernels_binary_cmp_lt_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `lt`, bf16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_lt_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `lt`, f64 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_lt_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_lt_f64`.
    pub fn baracuda_kernels_binary_cmp_lt_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `lt`, f64 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_lt_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// --- Le -------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Binary elementwise `le` (`a <= b`), f32 inputs, u8 output, contig fast path.
    ///
    /// Any comparison involving NaN returns 0 per IEEE 754.
    ///
    /// # Safety
    /// See `baracuda_kernels_binary_cmp_eq_f32_run`.
    pub fn baracuda_kernels_binary_cmp_le_f32_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_le_f32`.
    pub fn baracuda_kernels_binary_cmp_le_f32_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `le`, f32 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_le_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `le`, f16 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_le_f16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_le_f16`.
    pub fn baracuda_kernels_binary_cmp_le_f16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `le`, f16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_le_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `le`, bf16 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_le_bf16_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_le_bf16`.
    pub fn baracuda_kernels_binary_cmp_le_bf16_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `le`, bf16 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_le_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Binary elementwise `le`, f64 inputs, u8 output, contig fast path.
    pub fn baracuda_kernels_binary_cmp_le_f64_run(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `binary_cmp_le_f64`.
    pub fn baracuda_kernels_binary_cmp_le_f64_can_implement(
        numel: i64,
        a: *const c_void,
        b: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Binary elementwise `le`, f64 inputs, u8 output, strided path.
    pub fn baracuda_kernels_binary_cmp_le_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_a: *const i64,
        stride_b: *const i64,
        stride_y: *const i64,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Elementwise — unary (1→1) ops
// ============================================================================
//
// Same INSTANTIATE-driven kernel-family pattern as the binary path
// above, but for 1→1 ops (`y = f(x)`). Both contig and strided
// variants ship per (op, dtype) cell. The Rust dispatcher picks the
// fast contig path when input + output are both contiguous, else
// strided.
//
// ABI shape mirrors the binary launchers minus the second operand;
// strided variants drop the `stride_b` array too.
//
// Status codes mirror the GEMM family (see crate-level doc).

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `neg`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// All pointer args must be device-resident and remain valid for the
    /// duration of the launch. `stream` must be a live CUDA stream in
    /// the current context. `x` and `y` must each point to at least
    /// `numel` `float`s of device memory. Aliasing `y` with `x` is
    /// safe — each thread reads x[i] before writing y[i].
    pub fn baracuda_kernels_unary_neg_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_neg_f32`. Validates
    /// the problem size without launching a kernel. Returns the standard
    /// status code mapping.
    ///
    /// # Safety
    /// Same pointer-validity contract as the corresponding `_run` fn,
    /// but no device dereferences occur — only host-side checks.
    pub fn baracuda_kernels_unary_neg_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `neg`, f32 dtype, strided path.
    ///
    /// Handles non-contig views (transposed, sliced). Input shape must
    /// equal output shape — broadcast is not a meaningful unary
    /// semantic and is rejected by the Rust dispatcher upstream.
    ///
    /// # Safety
    /// Same device-pointer contract as the contig launcher. `shape`,
    /// `stride_x`, `stride_y` are host-side pointers to arrays of at
    /// least `rank` elements that must remain valid for the duration
    /// of the host-side launch call (the launcher copies them into
    /// the kernel parameter block before returning).
    pub fn baracuda_kernels_unary_neg_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `neg`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as `baracuda_kernels_unary_neg_f32_run`. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_neg_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_neg_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_neg_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `neg`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_unary_neg_f32_strided_run`. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_neg_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `neg`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as `baracuda_kernels_unary_neg_f32_run`. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_neg_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_neg_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_neg_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `neg`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_unary_neg_f32_strided_run`. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_neg_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `neg`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as `baracuda_kernels_unary_neg_f32_run`. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_neg_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_neg_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_neg_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `neg`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_unary_neg_f32_strided_run`. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_neg_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `abs` — `y = |x|` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `abs`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as `baracuda_kernels_unary_neg_f32_run`.
    pub fn baracuda_kernels_unary_abs_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_abs_f32`.
    ///
    /// # Safety
    /// Host-side checks only — same pointer-validity contract as the `_run` fn.
    pub fn baracuda_kernels_unary_abs_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `abs`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as `baracuda_kernels_unary_neg_f32_strided_run`.
    pub fn baracuda_kernels_unary_abs_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `abs`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_abs_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_abs_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_abs_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `abs`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_abs_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `abs`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_abs_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_abs_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_abs_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `abs`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_abs_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `abs`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_abs_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_abs_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_abs_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `abs`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_abs_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `sign` — `y = sign(x) ∈ {-1, 0, +1}` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `sign`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_sign_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sign_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sign_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sign`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_sign_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sign`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_sign_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sign_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sign_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sign`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_sign_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sign`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_sign_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sign_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sign_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sign`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_sign_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sign`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_sign_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sign_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sign_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sign`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_sign_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `reciprocal` — `y = 1 / x` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `reciprocal`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_reciprocal_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_reciprocal_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_reciprocal_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `reciprocal`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_reciprocal_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `reciprocal`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_reciprocal_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_reciprocal_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_reciprocal_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `reciprocal`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_reciprocal_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `reciprocal`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_reciprocal_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_reciprocal_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_reciprocal_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `reciprocal`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_reciprocal_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `reciprocal`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_reciprocal_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_reciprocal_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_reciprocal_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `reciprocal`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_reciprocal_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `square` — `y = x * x` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `square`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_square_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_square_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_square_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `square`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_square_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `square`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_square_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_square_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_square_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `square`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_square_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `square`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_square_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_square_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_square_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `square`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_square_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `square`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_square_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_square_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_square_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `square`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_square_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `cube` — `y = x * x * x` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `cube`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_cube_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cube_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cube_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cube`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_cube_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cube`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_cube_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cube_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cube_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cube`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_cube_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cube`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_cube_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cube_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cube_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cube`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_cube_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cube`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_cube_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cube_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cube_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cube`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same as the f32 variant; `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_cube_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `sqrt` — `y = sqrt(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `sqrt`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_sqrt_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sqrt_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sqrt_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sqrt`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_sqrt_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sqrt`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_sqrt_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sqrt_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sqrt_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sqrt`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_sqrt_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sqrt`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_sqrt_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sqrt_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sqrt_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sqrt`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_sqrt_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sqrt`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_sqrt_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sqrt_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sqrt_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sqrt`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_sqrt_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `rsqrt` — `y = 1 / sqrt(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `rsqrt`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_rsqrt_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_rsqrt_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_rsqrt_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `rsqrt`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_rsqrt_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `rsqrt`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_rsqrt_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_rsqrt_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_rsqrt_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `rsqrt`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_rsqrt_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `rsqrt`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_rsqrt_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_rsqrt_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_rsqrt_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `rsqrt`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_rsqrt_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `rsqrt`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_rsqrt_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_rsqrt_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_rsqrt_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `rsqrt`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_rsqrt_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `exp` — `y = exp(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `exp`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_exp_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_exp_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_exp_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `exp`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_exp_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `exp`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_exp_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_exp_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_exp_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `exp`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_exp_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `exp`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_exp_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_exp_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_exp_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `exp`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_exp_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `exp`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_exp_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_exp_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_exp_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `exp`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_exp_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `expm1` — `y = exp(x) - 1` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `expm1`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_expm1_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_expm1_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_expm1_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `expm1`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_expm1_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `expm1`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_expm1_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_expm1_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_expm1_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `expm1`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_expm1_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `expm1`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_expm1_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_expm1_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_expm1_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `expm1`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_expm1_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `expm1`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_expm1_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_expm1_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_expm1_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `expm1`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_expm1_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `log` — `y = ln(x)` (natural log) across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `log`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_log_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_log_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_log_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_log_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_log_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_log_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_log_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_log_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `log1p` — `y = ln(1 + x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `log1p`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_log1p_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log1p_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log1p_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log1p`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_log1p_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log1p`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_log1p_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log1p_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log1p_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log1p`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_log1p_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log1p`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_log1p_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log1p_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log1p_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log1p`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_log1p_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log1p`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_log1p_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log1p_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log1p_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log1p`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_log1p_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `sin` — `y = sin(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `sin`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_sin_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sin_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sin_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sin`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_sin_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sin`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_sin_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sin_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sin_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sin`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_sin_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sin`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_sin_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sin_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sin_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sin`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_sin_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sin`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_sin_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sin_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sin_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sin`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_sin_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `cos` — `y = cos(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `cos`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_cos_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cos_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cos_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cos`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_cos_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cos`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_cos_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cos_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cos_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cos`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_cos_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cos`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_cos_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cos_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cos_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cos`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_cos_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cos`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_cos_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cos_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cos_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cos`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_cos_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `tan` — `y = tan(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `tan`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_tan_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tan_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tan_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tan`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_tan_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `tan`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_tan_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tan_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tan_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tan`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_tan_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `tan`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_tan_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tan_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tan_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tan`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_tan_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `tan`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_tan_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tan_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tan_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tan`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_tan_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `sinh` — `y = sinh(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `sinh`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_sinh_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sinh_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sinh_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sinh`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_sinh_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sinh`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_sinh_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sinh_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sinh_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sinh`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_sinh_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sinh`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_sinh_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sinh_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sinh_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sinh`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_sinh_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sinh`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_sinh_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sinh_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sinh_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sinh`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_sinh_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `cosh` — `y = cosh(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `cosh`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_cosh_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cosh_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cosh_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cosh`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_cosh_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cosh`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_cosh_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cosh_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cosh_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cosh`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_cosh_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cosh`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_cosh_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cosh_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cosh_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cosh`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_cosh_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cosh`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_cosh_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cosh_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cosh_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cosh`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_cosh_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `tanh` — `y = tanh(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `tanh`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_tanh_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tanh_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tanh_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tanh`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_tanh_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `tanh`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_tanh_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tanh_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tanh_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tanh`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_tanh_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `tanh`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_tanh_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tanh_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tanh_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tanh`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_tanh_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `tanh`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_tanh_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tanh_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tanh_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tanh`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_tanh_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `relu` — `y = relu(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `relu`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_relu_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_relu_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_relu_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `relu`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_relu_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `relu`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_relu_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_relu_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_relu_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `relu`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_relu_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `relu`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_relu_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_relu_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_relu_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `relu`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_relu_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `relu`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_relu_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_relu_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_relu_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `relu`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_relu_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `gelu` — `y = gelu(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `gelu`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_gelu_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_gelu_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_gelu_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `gelu`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_gelu_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `gelu`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_gelu_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_gelu_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_gelu_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `gelu`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_gelu_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `gelu`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_gelu_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_gelu_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_gelu_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `gelu`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_gelu_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `gelu`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_gelu_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_gelu_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_gelu_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `gelu`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_gelu_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `gelu_tanh` — `y = gelu_tanh(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `gelu_tanh`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_gelu_tanh_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_gelu_tanh_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_gelu_tanh_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `gelu_tanh`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_gelu_tanh_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `gelu_tanh`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_gelu_tanh_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_gelu_tanh_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_gelu_tanh_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `gelu_tanh`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_gelu_tanh_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `gelu_tanh`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_gelu_tanh_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_gelu_tanh_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_gelu_tanh_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `gelu_tanh`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_gelu_tanh_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `gelu_tanh`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_gelu_tanh_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_gelu_tanh_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_gelu_tanh_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `gelu_tanh`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_gelu_tanh_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `silu` — `y = silu(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `silu`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_silu_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_silu_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_silu_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `silu`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_silu_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `silu`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_silu_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_silu_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_silu_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `silu`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_silu_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `silu`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_silu_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_silu_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_silu_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `silu`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_silu_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `silu`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_silu_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_silu_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_silu_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `silu`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_silu_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `mish` — `y = mish(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `mish`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_mish_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_mish_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_mish_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `mish`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_mish_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `mish`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_mish_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_mish_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_mish_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `mish`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_mish_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `mish`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_mish_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_mish_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_mish_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `mish`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_mish_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `mish`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_mish_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_mish_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_mish_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `mish`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_mish_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `sigmoid` — `y = sigmoid(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `sigmoid`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_sigmoid_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sigmoid_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sigmoid_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sigmoid`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_sigmoid_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sigmoid`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_sigmoid_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sigmoid_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sigmoid_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sigmoid`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_sigmoid_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sigmoid`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_sigmoid_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sigmoid_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sigmoid_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sigmoid`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_sigmoid_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `sigmoid`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_sigmoid_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_sigmoid_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_sigmoid_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `sigmoid`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_sigmoid_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `softplus` — `y = softplus(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `softplus`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_softplus_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_softplus_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softplus_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `softplus`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_softplus_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `softplus`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_softplus_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_softplus_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softplus_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `softplus`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_softplus_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `softplus`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_softplus_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_softplus_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softplus_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `softplus`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_softplus_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `softplus`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_softplus_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_softplus_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softplus_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `softplus`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_softplus_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `hardswish` — `y = hardswish(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `hardswish`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_hardswish_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardswish_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardswish_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardswish`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_hardswish_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardswish`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_hardswish_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardswish_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardswish_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardswish`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_hardswish_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardswish`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_hardswish_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardswish_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardswish_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardswish`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_hardswish_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardswish`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_hardswish_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardswish_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardswish_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardswish`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_hardswish_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `hardsigmoid` — `y = hardsigmoid(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `hardsigmoid`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_hardsigmoid_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardsigmoid_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardsigmoid_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardsigmoid`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_hardsigmoid_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardsigmoid`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_hardsigmoid_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardsigmoid_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardsigmoid_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardsigmoid`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_hardsigmoid_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardsigmoid`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_hardsigmoid_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardsigmoid_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardsigmoid_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardsigmoid`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_hardsigmoid_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardsigmoid`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_hardsigmoid_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardsigmoid_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardsigmoid_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardsigmoid`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_hardsigmoid_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `hardtanh` — `y = hardtanh(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `hardtanh`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_hardtanh_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardtanh_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardtanh_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardtanh`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_hardtanh_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardtanh`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_hardtanh_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardtanh_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardtanh_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardtanh`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_hardtanh_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardtanh`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_hardtanh_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardtanh_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardtanh_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardtanh`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_hardtanh_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardtanh`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_hardtanh_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_hardtanh_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardtanh_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `hardtanh`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_hardtanh_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `cbrt` — cube root across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `cbrt`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_cbrt_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cbrt_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cbrt_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cbrt`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_cbrt_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cbrt`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_cbrt_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cbrt_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cbrt_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cbrt`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_cbrt_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cbrt`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_cbrt_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cbrt_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cbrt_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cbrt`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_cbrt_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `cbrt`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_cbrt_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_cbrt_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_cbrt_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `cbrt`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_cbrt_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `exp2` — `y = 2^x` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `exp2`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_exp2_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_exp2_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_exp2_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `exp2`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_exp2_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `exp2`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_exp2_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_exp2_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_exp2_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `exp2`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_exp2_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `exp2`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_exp2_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_exp2_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_exp2_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `exp2`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_exp2_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `exp2`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_exp2_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_exp2_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_exp2_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `exp2`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_exp2_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `log2` — base-2 log across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `log2`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_log2_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log2_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log2_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log2`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_log2_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log2`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_log2_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log2_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log2_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log2`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_log2_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log2`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_log2_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log2_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log2_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log2`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_log2_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log2`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_log2_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log2_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log2_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log2`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_log2_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `log10` — base-10 log across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `log10`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_log10_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log10_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log10_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log10`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_log10_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log10`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_log10_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log10_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log10_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log10`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_log10_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log10`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_log10_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log10_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log10_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log10`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_log10_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `log10`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_log10_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_log10_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_log10_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `log10`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_log10_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `asin` — inverse sine across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `asin`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_asin_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_asin_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_asin_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `asin`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_asin_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `asin`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_asin_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_asin_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_asin_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `asin`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_asin_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `asin`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_asin_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_asin_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_asin_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `asin`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_asin_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `asin`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_asin_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_asin_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_asin_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `asin`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_asin_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `acos` — inverse cosine across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `acos`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_acos_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_acos_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_acos_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `acos`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_acos_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `acos`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_acos_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_acos_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_acos_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `acos`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_acos_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `acos`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_acos_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_acos_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_acos_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `acos`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_acos_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `acos`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_acos_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_acos_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_acos_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `acos`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_acos_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `atan` — inverse tangent across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `atan`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_atan_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_atan_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_atan_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `atan`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_atan_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `atan`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_atan_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_atan_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_atan_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `atan`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_atan_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `atan`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_atan_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_atan_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_atan_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `atan`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_atan_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `atan`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_atan_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_atan_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_atan_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `atan`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_atan_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `asinh` — inverse hyperbolic sine across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `asinh`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_asinh_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_asinh_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_asinh_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `asinh`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_asinh_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `asinh`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_asinh_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_asinh_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_asinh_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `asinh`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_asinh_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `asinh`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_asinh_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_asinh_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_asinh_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `asinh`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_asinh_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `asinh`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_asinh_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_asinh_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_asinh_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `asinh`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_asinh_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `acosh` — inverse hyperbolic cosine across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `acosh`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_acosh_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_acosh_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_acosh_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `acosh`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_acosh_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `acosh`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_acosh_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_acosh_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_acosh_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `acosh`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_acosh_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `acosh`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_acosh_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_acosh_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_acosh_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `acosh`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_acosh_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `acosh`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_acosh_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_acosh_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_acosh_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `acosh`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_acosh_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `atanh` — inverse hyperbolic tangent across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `atanh`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_atanh_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_atanh_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_atanh_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `atanh`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_atanh_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `atanh`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_atanh_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_atanh_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_atanh_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `atanh`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_atanh_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `atanh`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_atanh_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_atanh_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_atanh_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `atanh`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_atanh_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `atanh`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_atanh_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_atanh_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_atanh_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `atanh`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_atanh_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `floor` — round toward -infinity across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `floor`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_floor_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_floor_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_floor_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `floor`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_floor_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `floor`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_floor_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_floor_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_floor_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `floor`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_floor_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `floor`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_floor_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_floor_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_floor_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `floor`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_floor_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `floor`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_floor_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_floor_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_floor_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `floor`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_floor_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `ceil` — round toward +infinity across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `ceil`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_ceil_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_ceil_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_ceil_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `ceil`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_ceil_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `ceil`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_ceil_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_ceil_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_ceil_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `ceil`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_ceil_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `ceil`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_ceil_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_ceil_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_ceil_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `ceil`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_ceil_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `ceil`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_ceil_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_ceil_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_ceil_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `ceil`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_ceil_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `round` — round-half-to-even (PyTorch convention) across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `round`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_round_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_round_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_round_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `round`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_round_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `round`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_round_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_round_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_round_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `round`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_round_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `round`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_round_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_round_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_round_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `round`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_round_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `round`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_round_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_round_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_round_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `round`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_round_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `trunc` — round toward zero across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `trunc`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_trunc_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_trunc_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_trunc_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `trunc`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_trunc_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `trunc`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_trunc_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_trunc_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_trunc_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `trunc`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_trunc_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `trunc`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_trunc_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_trunc_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_trunc_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `trunc`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_trunc_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `trunc`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_trunc_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_trunc_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_trunc_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `trunc`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_trunc_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `frac` — fractional part (sign of x) across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `frac`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_frac_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_frac_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_frac_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `frac`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_frac_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `frac`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_frac_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_frac_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_frac_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `frac`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_frac_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `frac`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_frac_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_frac_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_frac_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `frac`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_frac_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `frac`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_frac_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_frac_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_frac_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `frac`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_frac_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Unary `erf` — `y = erf(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `erf`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_erf_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_erf_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_erf_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `erf`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_erf_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `erf`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_erf_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_erf_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_erf_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `erf`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_erf_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `erf`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_erf_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_erf_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_erf_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `erf`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_erf_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `erf`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_erf_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_erf_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_erf_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `erf`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_erf_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `erfc` — `y = erfc(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `erfc`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_erfc_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_erfc_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_erfc_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `erfc`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_erfc_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `erfc`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_erfc_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_erfc_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_erfc_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `erfc`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_erfc_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `erfc`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_erfc_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_erfc_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_erfc_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `erfc`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_erfc_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `erfc`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_erfc_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_erfc_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_erfc_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `erfc`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_erfc_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `lgamma` — `y = lgamma(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `lgamma`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_lgamma_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_lgamma_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_lgamma_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `lgamma`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_lgamma_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `lgamma`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_lgamma_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_lgamma_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_lgamma_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `lgamma`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_lgamma_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `lgamma`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_lgamma_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_lgamma_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_lgamma_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `lgamma`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_lgamma_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `lgamma`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_lgamma_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_lgamma_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_lgamma_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `lgamma`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_lgamma_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `logit` — `y = logit(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `logit`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_logit_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_logit_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_logit_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `logit`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_logit_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `logit`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_logit_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_logit_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_logit_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `logit`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_logit_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `logit`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_logit_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_logit_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_logit_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `logit`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_logit_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `logit`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_logit_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_logit_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_logit_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `logit`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_logit_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `softsign` — `y = softsign(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `softsign`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_softsign_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_softsign_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softsign_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `softsign`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_softsign_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `softsign`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_softsign_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_softsign_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softsign_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `softsign`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_softsign_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `softsign`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_softsign_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_softsign_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softsign_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `softsign`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_softsign_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `softsign`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_softsign_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_softsign_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softsign_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `softsign`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_softsign_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `tanhshrink` — `y = tanhshrink(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `tanhshrink`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_tanhshrink_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tanhshrink_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tanhshrink_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tanhshrink`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_tanhshrink_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `tanhshrink`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_tanhshrink_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tanhshrink_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tanhshrink_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tanhshrink`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_tanhshrink_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `tanhshrink`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_tanhshrink_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tanhshrink_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tanhshrink_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tanhshrink`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_tanhshrink_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `tanhshrink`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_tanhshrink_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_tanhshrink_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_tanhshrink_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `tanhshrink`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_tanhshrink_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `relu6` — `y = relu6(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `relu6`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_relu6_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_relu6_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_relu6_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `relu6`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_relu6_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `relu6`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_relu6_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_relu6_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_relu6_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `relu6`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_relu6_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `relu6`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_relu6_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_relu6_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_relu6_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `relu6`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_relu6_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `relu6`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_relu6_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_relu6_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_relu6_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `relu6`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_relu6_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Unary `selu` — `y = selu(x)` across f32 / f16 / bf16 / f64.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// Unary elementwise `selu`, f32 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer.
    pub fn baracuda_kernels_unary_selu_f32_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_selu_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_selu_f32_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `selu`, f32 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher.
    pub fn baracuda_kernels_unary_selu_f32_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `selu`, f16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_selu_f16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_selu_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_selu_f16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `selu`, f16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_selu_f16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `selu`, bf16 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_selu_bf16_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_selu_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_selu_bf16_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `selu`, bf16 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_selu_bf16_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `selu`, f64 dtype, contiguous fast path.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-neg trailblazer. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_selu_f64_run(
        numel: i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Pre-launch implementability check for `unary_selu_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_selu_f64_can_implement(
        numel: i64,
        x: *const c_void,
        y: *const c_void,
    ) -> i32;

    /// Unary elementwise `selu`, f64 dtype, strided path.
    ///
    /// # Safety
    /// Same contract as the unary-neg strided launcher. `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_selu_f64_strided_run(
        numel: i64,
        rank: i32,
        shape: *const i32,
        stride_x: *const i64,
        stride_y: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

}

// ----------------------------------------------------------------------------
// Parameterized-activation FW fanout — LeakyRelu / ELU / Hardshrink /
// Softshrink across f32 / f16 / bf16 / f64. Parameters are hardcoded
// (LeakyRelu α=0.01, ELU α=1.0, Hardshrink λ=0.5, Softshrink λ=0.5) to
// match PyTorch defaults. When the parameterized-unary plan ships these
// re-emit with the parameter as a runtime arg — same dispatch shape, no
// extern signature change.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    // ---- LeakyReLU (α=0.01) ----

    /// Unary elementwise `leaky_relu` (α=0.01), f32, contig.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-relu trailblazer.
    pub fn baracuda_kernels_unary_leaky_relu_f32_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_leaky_relu_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_leaky_relu_f32_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `leaky_relu` (α=0.01), f32, strided.
    ///
    /// # Safety
    /// Same contract as the unary-relu strided launcher.
    pub fn baracuda_kernels_unary_leaky_relu_f32_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `leaky_relu` (α=0.01), f16, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `__half` storage; same contract as the unary-relu trailblazer.
    pub fn baracuda_kernels_unary_leaky_relu_f16_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_leaky_relu_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_leaky_relu_f16_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `leaky_relu` (α=0.01), f16, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_leaky_relu_f16_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `leaky_relu` (α=0.01), bf16, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_leaky_relu_bf16_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_leaky_relu_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_leaky_relu_bf16_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `leaky_relu` (α=0.01), bf16, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_leaky_relu_bf16_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `leaky_relu` (α=0.01), f64, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_leaky_relu_f64_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_leaky_relu_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_leaky_relu_f64_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `leaky_relu` (α=0.01), f64, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_leaky_relu_f64_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- ELU (α=1.0) ----

    /// Unary elementwise `elu` (α=1.0), f32, contig.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-relu trailblazer.
    pub fn baracuda_kernels_unary_elu_f32_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_elu_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_elu_f32_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `elu` (α=1.0), f32, strided.
    ///
    /// # Safety
    /// Same contract as the unary-relu strided launcher.
    pub fn baracuda_kernels_unary_elu_f32_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `elu` (α=1.0), f16, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_elu_f16_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_elu_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_elu_f16_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `elu` (α=1.0), f16, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_elu_f16_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `elu` (α=1.0), bf16, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_elu_bf16_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_elu_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_elu_bf16_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `elu` (α=1.0), bf16, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_elu_bf16_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `elu` (α=1.0), f64, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_elu_f64_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_elu_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_elu_f64_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `elu` (α=1.0), f64, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_elu_f64_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Hardshrink (λ=0.5) ----

    /// Unary elementwise `hardshrink` (λ=0.5), f32, contig.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-relu trailblazer.
    pub fn baracuda_kernels_unary_hardshrink_f32_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_hardshrink_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardshrink_f32_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `hardshrink` (λ=0.5), f32, strided.
    ///
    /// # Safety
    /// Same contract as the unary-relu strided launcher.
    pub fn baracuda_kernels_unary_hardshrink_f32_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardshrink` (λ=0.5), f16, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_hardshrink_f16_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_hardshrink_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardshrink_f16_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `hardshrink` (λ=0.5), f16, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_hardshrink_f16_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardshrink` (λ=0.5), bf16, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_hardshrink_bf16_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_hardshrink_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardshrink_bf16_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `hardshrink` (λ=0.5), bf16, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_hardshrink_bf16_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `hardshrink` (λ=0.5), f64, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_hardshrink_f64_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_hardshrink_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_hardshrink_f64_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `hardshrink` (λ=0.5), f64, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_hardshrink_f64_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Softshrink (λ=0.5) ----

    /// Unary elementwise `softshrink` (λ=0.5), f32, contig.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the unary-relu trailblazer.
    pub fn baracuda_kernels_unary_softshrink_f32_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_softshrink_f32`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softshrink_f32_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `softshrink` (λ=0.5), f32, strided.
    ///
    /// # Safety
    /// Same contract as the unary-relu strided launcher.
    pub fn baracuda_kernels_unary_softshrink_f32_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `softshrink` (λ=0.5), f16, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_softshrink_f16_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_softshrink_f16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softshrink_f16_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `softshrink` (λ=0.5), f16, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_softshrink_f16_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `softshrink` (λ=0.5), bf16, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_softshrink_bf16_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_softshrink_bf16`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softshrink_bf16_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `softshrink` (λ=0.5), bf16, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_softshrink_bf16_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// Unary elementwise `softshrink` (λ=0.5), f64, contig.
    ///
    /// # Safety
    /// `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_softshrink_f64_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Pre-launch implementability check for `unary_softshrink_f64`.
    ///
    /// # Safety
    /// Host-side checks only.
    pub fn baracuda_kernels_unary_softshrink_f64_can_implement(
        numel: i64, x: *const c_void, y: *const c_void,
    ) -> i32;
    /// Unary elementwise `softshrink` (λ=0.5), f64, strided.
    ///
    /// # Safety
    /// `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_softshrink_f64_strided_run(
        numel: i64, rank: i32, shape: *const i32, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // =========================================================================
    // Phase 3 Category C′ — gated activations (forward + backward).
    //
    // ABI shape: one thread per OUTPUT cell. The input is split along
    // `split_dim` into two halves `(a, b)`; output `y = a · gate(b)`
    // has shape `input_shape` with `input_shape[split_dim]` halved.
    // `x_half_offset` is `(input_shape[split_dim] / 2) · stride_x[split_dim]`
    // — the element-offset between the a-half cell and the b-half cell
    // for a given output coord. `dx_half_offset` is the same for `dx`
    // (which is contig over `input_shape`).
    // =========================================================================

    /// SwiGLU forward, f32. `y = a · b · sigmoid(b)`.
    ///
    /// # Safety
    /// `x` / `y` point to `float` storage.
    pub fn baracuda_kernels_gated_swiglu_f32_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SwiGLU forward, f16.
    pub fn baracuda_kernels_gated_swiglu_f16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SwiGLU forward, bf16.
    pub fn baracuda_kernels_gated_swiglu_bf16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SwiGLU forward, f64.
    pub fn baracuda_kernels_gated_swiglu_f64_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// SwiGLU backward, f32. `da = dy·silu(b)`, `db = dy·a·silu'(b)`.
    ///
    /// # Safety
    /// `x` / `dy` / `dx` point to `float` storage.
    pub fn baracuda_kernels_gated_swiglu_backward_f32_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SwiGLU backward, f16.
    pub fn baracuda_kernels_gated_swiglu_backward_f16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SwiGLU backward, bf16.
    pub fn baracuda_kernels_gated_swiglu_backward_bf16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// SwiGLU backward, f64.
    pub fn baracuda_kernels_gated_swiglu_backward_f64_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// GLU forward, f32. `y = a · sigmoid(b)`.
    ///
    /// # Safety
    /// `x` / `y` point to `float` storage.
    pub fn baracuda_kernels_gated_glu_f32_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GLU forward, f16.
    pub fn baracuda_kernels_gated_glu_f16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GLU forward, bf16.
    pub fn baracuda_kernels_gated_glu_bf16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GLU forward, f64.
    pub fn baracuda_kernels_gated_glu_f64_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// GLU backward, f32. `da = dy·sigmoid(b)`, `db = dy·a·sigmoid(b)·(1-sigmoid(b))`.
    ///
    /// # Safety
    /// `x` / `dy` / `dx` point to `float` storage.
    pub fn baracuda_kernels_gated_glu_backward_f32_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GLU backward, f16.
    pub fn baracuda_kernels_gated_glu_backward_f16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GLU backward, bf16.
    pub fn baracuda_kernels_gated_glu_backward_bf16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GLU backward, f64.
    pub fn baracuda_kernels_gated_glu_backward_f64_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// ReGLU forward, f32. `y = a · relu(b) = a · max(b, 0)`.
    ///
    /// # Safety
    /// `x` / `y` point to `float` storage.
    pub fn baracuda_kernels_gated_reglu_f32_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReGLU forward, f16.
    pub fn baracuda_kernels_gated_reglu_f16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReGLU forward, bf16.
    pub fn baracuda_kernels_gated_reglu_bf16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReGLU forward, f64.
    pub fn baracuda_kernels_gated_reglu_f64_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// ReGLU backward, f32. `da = (b>0)?dy·b:0`, `db = (b>0)?dy·a:0`.
    ///
    /// # Safety
    /// `x` / `dy` / `dx` point to `float` storage.
    pub fn baracuda_kernels_gated_reglu_backward_f32_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReGLU backward, f16.
    pub fn baracuda_kernels_gated_reglu_backward_f16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReGLU backward, bf16.
    pub fn baracuda_kernels_gated_reglu_backward_bf16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// ReGLU backward, f64.
    pub fn baracuda_kernels_gated_reglu_backward_f64_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// GeGLU forward, f32. `y = a · gelu(b)`, exact erf-based.
    ///
    /// # Safety
    /// `x` / `y` point to `float` storage.
    pub fn baracuda_kernels_gated_geglu_f32_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GeGLU forward, f16.
    pub fn baracuda_kernels_gated_geglu_f16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GeGLU forward, bf16.
    pub fn baracuda_kernels_gated_geglu_bf16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GeGLU forward, f64.
    pub fn baracuda_kernels_gated_geglu_f64_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, stride_x: *const i64, stride_y: *const i64,
        x: *const c_void, y: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    /// GeGLU backward, f32. `da = dy·gelu(b)`, `db = dy·a·gelu'(b)`.
    ///
    /// # Safety
    /// `x` / `dy` / `dx` point to `float` storage.
    pub fn baracuda_kernels_gated_geglu_backward_f32_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GeGLU backward, f16.
    pub fn baracuda_kernels_gated_geglu_backward_f16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GeGLU backward, bf16.
    pub fn baracuda_kernels_gated_geglu_backward_bf16_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// GeGLU backward, f64.
    pub fn baracuda_kernels_gated_geglu_backward_f64_run(
        output_numel: i64, rank: i32, output_shape: *const i32, split_dim: i32,
        x_half_offset: i64, dx_half_offset: i64,
        stride_x: *const i64, stride_dy: *const i64, stride_dx: *const i64,
        x: *const c_void, dy: *const c_void, dx: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
}

// =============================================================================
// Ternary backward family — Phase 3 backward fanout (Milestone F)
// =============================================================================
//
// 4 ops × 4 FP dtypes = 16 launchers.
//
// Unscaled (Fma, Clamp) — 7-pointer ABI: dy, a, b, c, da, db, dc.
// Scaled (Addcmul, Addcdiv) — same 7 pointers + `float scale` between
// `dc` and the workspace pointer, mirroring the FW scaled-ternary ABI.
//
// All four saved inputs are read every cell regardless of whether the
// op's gradient references them — see the .cu file comments for why
// (uniform ABI across the family; one extra coalesced load is cheap).

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    // ---- Fma backward (unscaled) ----
    /// Fma backward, f32. Writes `da = dy*b`, `db = dy*a`, `dc = dy`.
    pub fn baracuda_kernels_ternary_fma_backward_f32_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Fma backward, f16.
    pub fn baracuda_kernels_ternary_fma_backward_f16_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Fma backward, bf16.
    pub fn baracuda_kernels_ternary_fma_backward_bf16_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Fma backward, f64.
    pub fn baracuda_kernels_ternary_fma_backward_f64_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Clamp backward (unscaled, mask × dy) ----
    /// Clamp backward, f32. Writes mask × dy per axis (a/b/c).
    pub fn baracuda_kernels_ternary_clamp_backward_f32_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Clamp backward, f16.
    pub fn baracuda_kernels_ternary_clamp_backward_f16_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Clamp backward, bf16.
    pub fn baracuda_kernels_ternary_clamp_backward_bf16_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Clamp backward, f64.
    pub fn baracuda_kernels_ternary_clamp_backward_f64_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Addcmul backward (scaled) ----
    /// Addcmul backward, f32. Reads `desc.scale`.
    /// Writes `da = dy`, `db = dy*scale*c`, `dc = dy*scale*b`.
    pub fn baracuda_kernels_ternary_addcmul_backward_f32_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Addcmul backward, f16.
    pub fn baracuda_kernels_ternary_addcmul_backward_f16_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Addcmul backward, bf16.
    pub fn baracuda_kernels_ternary_addcmul_backward_bf16_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Addcmul backward, f64.
    pub fn baracuda_kernels_ternary_addcmul_backward_f64_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Addcdiv backward (scaled) ----
    /// Addcdiv backward, f32. Reads `desc.scale`.
    /// Writes `da = dy`, `db = dy*scale/c`, `dc = -dy*scale*b/c²`.
    pub fn baracuda_kernels_ternary_addcdiv_backward_f32_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Addcdiv backward, f16.
    pub fn baracuda_kernels_ternary_addcdiv_backward_f16_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Addcdiv backward, bf16.
    pub fn baracuda_kernels_ternary_addcdiv_backward_bf16_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// Addcdiv backward, f64.
    pub fn baracuda_kernels_ternary_addcdiv_backward_f64_run(
        numel: i64,
        dy: *const c_void, a: *const c_void, b: *const c_void, c: *const c_void,
        da: *mut c_void, db: *mut c_void, dc: *mut c_void,
        scale: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
}

// ----------------------------------------------------------------------------
// Parameterized unary / binary plan families — Phase 3 deferred ops.
//
// New ABI shape vs the plain unary / binary launchers: f32 scalar
// parameters threaded by value through the launcher signature.
//   Unary param FW : `(numel, x, y, p0, p1, ws, ws_bytes, stream)`
//   Unary param BW : `(numel, dy, x, dx, p0, p1, ws, ws_bytes, stream)`
//   Binary param FW: `(numel, a, b, y, p, ws, ws_bytes, stream)`
//   Binary param BW: `(numel, dy, da, db, p, ws, ws_bytes, stream)`
//
// Today's wired ops:
//   Threshold (2 params: t = p0, v = p1) — FW + BW × {f32, f16, bf16, f64}.
//   Lerp      (1 param : weight = p)     — FW + BW × {f32, f16, bf16, f64}.
//
// Contig-only — no strided variant for the trailblazer.
// ----------------------------------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    // ---- Threshold FW (params: p0 = t, p1 = v) ----

    /// Unary elementwise `threshold(x; t, v) = (x > t) ? x : v`, f32, contig.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the plain unary launchers.
    /// `p0` carries the threshold `t`; `p1` carries the replacement value `v`.
    pub fn baracuda_kernels_unary_threshold_f32_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        p0: f32, p1: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `threshold` FW, f16.
    ///
    /// # Safety
    /// `x` / `y` point to `__half` storage.
    pub fn baracuda_kernels_unary_threshold_f16_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        p0: f32, p1: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `threshold` FW, bf16.
    ///
    /// # Safety
    /// `x` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_threshold_bf16_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        p0: f32, p1: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `threshold` FW, f64. The f32 params widen to f64 losslessly.
    ///
    /// # Safety
    /// `x` / `y` point to `double` storage.
    pub fn baracuda_kernels_unary_threshold_f64_run(
        numel: i64, x: *const c_void, y: *mut c_void,
        p0: f32, p1: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Threshold BW (saved-x; params: p0 = t, p1 = v unused) ----

    /// `threshold` backward: `dx = (x > t) ? dy : 0`, f32. Saved-x.
    ///
    /// # Safety
    /// `dy`, `x`, `dx` device pointers; `p1` ignored by the kernel (kept on the
    /// ABI for shape parity with FW).
    pub fn baracuda_kernels_unary_threshold_backward_f32_run(
        numel: i64, dy: *const c_void, x: *const c_void, dx: *mut c_void,
        p0: f32, p1: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `threshold` BW, f16.
    ///
    /// # Safety
    /// All tensor pointers reference `__half` storage.
    pub fn baracuda_kernels_unary_threshold_backward_f16_run(
        numel: i64, dy: *const c_void, x: *const c_void, dx: *mut c_void,
        p0: f32, p1: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `threshold` BW, bf16.
    ///
    /// # Safety
    /// All tensor pointers reference `__nv_bfloat16` storage.
    pub fn baracuda_kernels_unary_threshold_backward_bf16_run(
        numel: i64, dy: *const c_void, x: *const c_void, dx: *mut c_void,
        p0: f32, p1: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `threshold` BW, f64.
    ///
    /// # Safety
    /// All tensor pointers reference `double` storage.
    pub fn baracuda_kernels_unary_threshold_backward_f64_run(
        numel: i64, dy: *const c_void, x: *const c_void, dx: *mut c_void,
        p0: f32, p1: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Lerp FW (param: weight) ----

    /// Binary elementwise `lerp(a, b; weight) = a + weight·(b - a)`, f32, contig.
    ///
    /// # Safety
    /// Same device-pointer / stream contract as the plain binary launchers.
    /// `p` carries the broadcast scalar `weight`.
    pub fn baracuda_kernels_binary_lerp_f32_run(
        numel: i64, a: *const c_void, b: *const c_void, y: *mut c_void,
        p: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `lerp` FW, f16.
    ///
    /// # Safety
    /// `a` / `b` / `y` point to `__half` storage.
    pub fn baracuda_kernels_binary_lerp_f16_run(
        numel: i64, a: *const c_void, b: *const c_void, y: *mut c_void,
        p: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `lerp` FW, bf16.
    ///
    /// # Safety
    /// `a` / `b` / `y` point to `__nv_bfloat16` storage.
    pub fn baracuda_kernels_binary_lerp_bf16_run(
        numel: i64, a: *const c_void, b: *const c_void, y: *mut c_void,
        p: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `lerp` FW, f64. The f32 weight widens to f64 losslessly.
    ///
    /// # Safety
    /// `a` / `b` / `y` point to `double` storage.
    pub fn baracuda_kernels_binary_lerp_f64_run(
        numel: i64, a: *const c_void, b: *const c_void, y: *mut c_void,
        p: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;

    // ---- Lerp BW (no saves; param: weight) ----

    /// `lerp` backward: `da = (1 - weight)·dy`, `db = weight·dy`, f32. No saves.
    ///
    /// # Safety
    /// `dy`, `da`, `db` device pointers.
    pub fn baracuda_kernels_binary_lerp_backward_f32_run(
        numel: i64, dy: *const c_void, da: *mut c_void, db: *mut c_void,
        p: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `lerp` BW, f16.
    ///
    /// # Safety
    /// All tensor pointers reference `__half` storage.
    pub fn baracuda_kernels_binary_lerp_backward_f16_run(
        numel: i64, dy: *const c_void, da: *mut c_void, db: *mut c_void,
        p: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `lerp` BW, bf16.
    ///
    /// # Safety
    /// All tensor pointers reference `__nv_bfloat16` storage.
    pub fn baracuda_kernels_binary_lerp_backward_bf16_run(
        numel: i64, dy: *const c_void, da: *mut c_void, db: *mut c_void,
        p: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
    /// `lerp` BW, f64.
    ///
    /// # Safety
    /// All tensor pointers reference `double` storage.
    pub fn baracuda_kernels_binary_lerp_backward_f64_run(
        numel: i64, dy: *const c_void, da: *mut c_void, db: *mut c_void,
        p: f32,
        workspace: *mut c_void, workspace_bytes: usize, stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Random / cuRAND — Phase 4.5
// ============================================================================
//
// Host-API cuRAND bindings + custom bespoke kernels for Bernoulli /
// Dropout. cuRAND covers `Uniform` and `Normal` directly. Bernoulli is
// custom (uniform + threshold → Bool), Dropout is custom (uniform +
// threshold + scale, writes both `y` and `mask`).
//
// Linkage: `cargo:rustc-link-lib=dylib=curand` (added in build.rs). The
// system resolves `libcurand.so` on Linux and `curand64_*.dll` on Windows
// from the CUDA installation that ships them alongside cudart.

/// Opaque cuRAND generator handle. Treated as a stateful object owned by
/// safe Rust at the plan layer — never inspect its internals here.
#[allow(non_camel_case_types)]
pub type curandGenerator_t = *mut c_void;

/// `CURAND_RNG_PSEUDO_DEFAULT` — XORWOW pseudo-random generator. Adequate
/// for the dropout / sampling use cases this milestone targets; future
/// QRNG / Philox / MT19937 work can extend the descriptor surface.
pub const CURAND_RNG_PSEUDO_DEFAULT: i32 = 100;

/// `CURAND_STATUS_SUCCESS` — only success code. Any non-zero return from
/// the cuRAND host API is mapped to status `5` ("internal kernel error")
/// at the safe-plan layer.
pub const CURAND_STATUS_SUCCESS: i32 = 0;

unsafe extern "C" {
    /// `curandCreateGenerator(generator, rng_type)`. Returns 0 on success.
    ///
    /// # Safety
    /// `generator` must point to writable storage for one `curandGenerator_t`.
    pub fn curandCreateGenerator(
        generator: *mut curandGenerator_t,
        rng_type: i32,
    ) -> i32;

    /// `curandSetPseudoRandomGeneratorSeed(generator, seed)`. Returns 0 on success.
    ///
    /// # Safety
    /// `generator` must be a valid handle returned by `curandCreateGenerator`.
    pub fn curandSetPseudoRandomGeneratorSeed(
        generator: curandGenerator_t,
        seed: u64,
    ) -> i32;

    /// `curandSetStream(generator, stream)`. Binds subsequent generator calls
    /// to the given CUDA stream. Returns 0 on success.
    ///
    /// # Safety
    /// `generator` must be a valid handle; `stream` must be a valid CUDA stream
    /// in the current context, or null for the default stream.
    pub fn curandSetStream(generator: curandGenerator_t, stream: *mut c_void) -> i32;

    /// `curandGenerateUniform(generator, ptr, n)` — writes `n` `float` samples
    /// in `(0, 1]` to `ptr`. Returns 0 on success.
    ///
    /// # Safety
    /// `ptr` must point to at least `n * sizeof(f32)` writable device bytes.
    pub fn curandGenerateUniform(
        generator: curandGenerator_t,
        ptr: *mut f32,
        n: usize,
    ) -> i32;

    /// `curandGenerateUniformDouble(generator, ptr, n)` — writes `n` `double`
    /// samples in `(0, 1]` to `ptr`. Returns 0 on success.
    ///
    /// # Safety
    /// `ptr` must point to at least `n * sizeof(f64)` writable device bytes.
    pub fn curandGenerateUniformDouble(
        generator: curandGenerator_t,
        ptr: *mut f64,
        n: usize,
    ) -> i32;

    /// `curandGenerateNormal(generator, ptr, n, mean, stddev)` — writes `n`
    /// normally-distributed `float` samples to `ptr`. Note: cuRAND
    /// requires `n` be even for the Box-Muller pair generator. Returns 0
    /// on success.
    ///
    /// # Safety
    /// `ptr` must point to at least `n * sizeof(f32)` writable device bytes.
    pub fn curandGenerateNormal(
        generator: curandGenerator_t,
        ptr: *mut f32,
        n: usize,
        mean: f32,
        stddev: f32,
    ) -> i32;

    /// `curandGenerateNormalDouble(generator, ptr, n, mean, stddev)`.
    /// Same parity contract as `curandGenerateNormal`. Returns 0 on success.
    ///
    /// # Safety
    /// `ptr` must point to at least `n * sizeof(f64)` writable device bytes.
    pub fn curandGenerateNormalDouble(
        generator: curandGenerator_t,
        ptr: *mut f64,
        n: usize,
        mean: f64,
        stddev: f64,
    ) -> i32;

    /// `curandDestroyGenerator(generator)`. Returns 0 on success.
    ///
    /// # Safety
    /// `generator` must be a valid handle returned by `curandCreateGenerator`
    /// that has not been previously destroyed.
    pub fn curandDestroyGenerator(generator: curandGenerator_t) -> i32;
}

// ----------------------------------------------------------------------------
// Bespoke random kernels — Bernoulli + Dropout
// ----------------------------------------------------------------------------
//
// Two custom kernels per dtype because cuRAND only generates uniform /
// normal directly:
//
// * `bernoulli_<dtype>` — reads a `float` uniform-rand buffer and a
//   probability `p`; writes Bool output (`1` if rand < p else `0`).
// * `dropout_<dtype>` — reads input `x` + `float` uniform-rand buffer +
//   `p` (drop probability); writes `y = mask · x / (1 - p)` and `mask`
//   (`1` kept, `0` dropped). Caller saves `mask` for backward.
// * `dropout_backward_<dtype>` — reads `dy` + saved `mask` + `p`; writes
//   `dx = mask · dy / (1 - p)`.

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// `bernoulli` over a `float` uniform-rand buffer.
    ///
    /// Writes one `Bool` (encoded as `uint8_t` 0/1) per output cell:
    /// `y[i] = (rand[i] < p) ? 1 : 0`.
    ///
    /// # Safety
    /// `rand` points to `numel` `float` samples (caller-generated via
    /// cuRAND); `y` points to `numel` `uint8_t` cells.
    pub fn baracuda_kernels_bernoulli_run(
        numel: i64,
        p: f32,
        rand: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Dropout forward (f32). Writes:
    /// - `y[i] = (rand[i] < (1 - p)) ? x[i] * scale : 0`
    /// - `mask[i] = (rand[i] < (1 - p)) ? 1 : 0` (encoded as `uint8_t`)
    /// where `scale = 1 / (1 - p)`. Caller computes `scale` to keep the
    /// kernel branch-free of the `p == 1` edge case.
    ///
    /// # Safety
    /// All tensor pointers reference device memory. `x` / `rand` / `y`
    /// hold `f32`; `mask` is a packed Bool (`uint8_t`).
    pub fn baracuda_kernels_dropout_f32_run(
        numel: i64,
        p: f32,
        scale: f32,
        x: *const c_void,
        rand: *const c_void,
        y: *mut c_void,
        mask: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Dropout forward (f64). Same shape as the f32 variant.
    ///
    /// # Safety
    /// `x` / `y` reference `double`; `rand` reads `float` samples (one
    /// per output cell); `mask` is a packed Bool (`uint8_t`).
    pub fn baracuda_kernels_dropout_f64_run(
        numel: i64,
        p: f32,
        scale: f64,
        x: *const c_void,
        rand: *const c_void,
        y: *mut c_void,
        mask: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Dropout backward (f32). Writes `dx[i] = dy[i] * mask[i] * scale`
    /// where `scale = 1 / (1 - p)`.
    ///
    /// # Safety
    /// `dy` / `dx` reference `float`; `mask` is a packed Bool (`uint8_t`).
    pub fn baracuda_kernels_dropout_backward_f32_run(
        numel: i64,
        scale: f32,
        dy: *const c_void,
        mask: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Dropout backward (f64).
    ///
    /// # Safety
    /// `dy` / `dx` reference `double`; `mask` is a packed Bool (`uint8_t`).
    pub fn baracuda_kernels_dropout_backward_f64_run(
        numel: i64,
        scale: f64,
        dy: *const c_void,
        mask: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// In-place affine `y = scale * y + offset` (f32). Used by the
    /// safe-plan layer to remap a cuRAND uniform-(0, 1] buffer into
    /// `Uniform(low, high]`.
    ///
    /// # Safety
    /// `y` points to `numel` `float` device cells.
    pub fn baracuda_kernels_affine_inplace_f32_run(
        numel: i64,
        scale: f32,
        offset: f32,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// In-place affine `y = scale * y + offset` (f64).
    ///
    /// # Safety
    /// `y` points to `numel` `double` device cells.
    pub fn baracuda_kernels_affine_inplace_f64_run(
        numel: i64,
        scale: f64,
        offset: f64,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// Phase 6.1 — Attention positional encodings (Category K).
//
// RoPE (rotary position embedding) + ALiBi (attention-with-linear-biases),
// FW + BW × 4 FP dtypes. RoPE rotates pairs (2i, 2i+1) of a [B, H, S, D]
// Q/K tensor by per-position angles θ = pos · base^(-2i/D); BW reverses
// the trig sign (rotation by -θ). ALiBi adds slope[h]·(j-i) to score
// cell (b, h, i, j); BW is pass-through dA copy + per-head deterministic
// warp-shuffle reduction for dslope.
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]
unsafe extern "C" {
    /// RoPE FW, f32. Input/output are [B, H, S, D] contiguous row-major;
    /// `head_dim` (D) must be even. When `pos_default_flag != 0`, the
    /// kernel ignores `positions` and uses position index = sequence
    /// index; otherwise `positions` is `int64_t[seq]`.
    pub fn baracuda_kernels_rope_f32_run(
        batch: i32,
        heads: i32,
        seq: i32,
        head_dim: i32,
        base: f32,
        pos_default_flag: i32,
        x: *const c_void,
        positions: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// RoPE FW, f16 (f32 trig detour internally).
    pub fn baracuda_kernels_rope_f16_run(
        batch: i32,
        heads: i32,
        seq: i32,
        head_dim: i32,
        base: f32,
        pos_default_flag: i32,
        x: *const c_void,
        positions: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// RoPE FW, bf16.
    pub fn baracuda_kernels_rope_bf16_run(
        batch: i32,
        heads: i32,
        seq: i32,
        head_dim: i32,
        base: f32,
        pos_default_flag: i32,
        x: *const c_void,
        positions: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// RoPE FW, f64.
    pub fn baracuda_kernels_rope_f64_run(
        batch: i32,
        heads: i32,
        seq: i32,
        head_dim: i32,
        base: f32,
        pos_default_flag: i32,
        x: *const c_void,
        positions: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// RoPE BW, f32. Same shape as FW; computes `dx` from `dy` by
    /// rotation through `-θ`.
    pub fn baracuda_kernels_rope_backward_f32_run(
        batch: i32,
        heads: i32,
        seq: i32,
        head_dim: i32,
        base: f32,
        pos_default_flag: i32,
        dy: *const c_void,
        positions: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// RoPE BW, f16.
    pub fn baracuda_kernels_rope_backward_f16_run(
        batch: i32,
        heads: i32,
        seq: i32,
        head_dim: i32,
        base: f32,
        pos_default_flag: i32,
        dy: *const c_void,
        positions: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// RoPE BW, bf16.
    pub fn baracuda_kernels_rope_backward_bf16_run(
        batch: i32,
        heads: i32,
        seq: i32,
        head_dim: i32,
        base: f32,
        pos_default_flag: i32,
        dy: *const c_void,
        positions: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// RoPE BW, f64.
    pub fn baracuda_kernels_rope_backward_f64_run(
        batch: i32,
        heads: i32,
        seq: i32,
        head_dim: i32,
        base: f32,
        pos_default_flag: i32,
        dy: *const c_void,
        positions: *const c_void,
        dx: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// ALiBi FW, f32. `y[b, h, i, j] = scores[b, h, i, j] + slopes[h] · (j - i)`.
    pub fn baracuda_kernels_alibi_f32_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        scores: *const c_void,
        slopes: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// ALiBi FW, f16.
    pub fn baracuda_kernels_alibi_f16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        scores: *const c_void,
        slopes: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// ALiBi FW, bf16.
    pub fn baracuda_kernels_alibi_bf16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        scores: *const c_void,
        slopes: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// ALiBi FW, f64.
    pub fn baracuda_kernels_alibi_f64_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        scores: *const c_void,
        slopes: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// ALiBi BW, f32. `da[b, h, i, j] = dy[b, h, i, j]` (pass-through);
    /// `dslope[h] = Σ_{b, i, j} dy[b, h, i, j] · (j - i)`. Either `da`
    /// or `dslope` may be null to skip; both null is rejected.
    pub fn baracuda_kernels_alibi_backward_f32_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        dy: *const c_void,
        da: *mut c_void,
        dslope: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// ALiBi BW, f16.
    pub fn baracuda_kernels_alibi_backward_f16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        dy: *const c_void,
        da: *mut c_void,
        dslope: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// ALiBi BW, bf16.
    pub fn baracuda_kernels_alibi_backward_bf16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        dy: *const c_void,
        da: *mut c_void,
        dslope: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// ALiBi BW, f64.
    pub fn baracuda_kernels_alibi_backward_f64_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        dy: *const c_void,
        da: *mut c_void,
        dslope: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // =========================================================================
    // Milestone 6.2 — naive Scaled Dot-Product Attention (SDPA). One
    // `_run` symbol per dtype per direction; each runs the full
    // 3-kernel (FW) / 5-kernel (BW) pipeline internally.
    // =========================================================================

    /// SDPA FW, f32. Computes `y = softmax(Q·K^T·scale + mask) · V`. The
    /// `attn` buffer ([B, H, Q, K]) doubles as the scores intermediate
    /// and is overwritten in place with the softmax output (saved for
    /// BW). Pass `has_mask = 0` and `mask = nullptr` to skip the mask
    /// add. `is_causal = 1` applies an upper-triangular -inf mask
    /// inside the scores kernel.
    pub fn baracuda_kernels_sdpa_f32_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        has_mask: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        mask: *const c_void,
        attn: *mut c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SDPA FW, f16 (f32 accumulators).
    pub fn baracuda_kernels_sdpa_f16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        has_mask: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        mask: *const c_void,
        attn: *mut c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SDPA FW, bf16 (f32 accumulators).
    pub fn baracuda_kernels_sdpa_bf16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        has_mask: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        mask: *const c_void,
        attn: *mut c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SDPA FW, f64.
    pub fn baracuda_kernels_sdpa_f64_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        has_mask: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        mask: *const c_void,
        attn: *mut c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// SDPA BW, f32. Given the FW-saved `attn` ([B, H, Q, K]), `Q`, `K`,
    /// `V`, and upstream `dy`, computes `dQ`, `dK`, `dV`. The
    /// `dscores_ws` argument is a caller-allocated [B, H, Q, K] scratch
    /// buffer reused as the dattn → dscores intermediate; size matches
    /// the FW `attn` tensor.
    pub fn baracuda_kernels_sdpa_backward_f32_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        attn: *const c_void,
        dy: *const c_void,
        dscores_ws: *mut c_void,
        dQ: *mut c_void,
        dK: *mut c_void,
        dV: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SDPA BW, f16.
    pub fn baracuda_kernels_sdpa_backward_f16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        attn: *const c_void,
        dy: *const c_void,
        dscores_ws: *mut c_void,
        dQ: *mut c_void,
        dK: *mut c_void,
        dV: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SDPA BW, bf16.
    pub fn baracuda_kernels_sdpa_backward_bf16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        attn: *const c_void,
        dy: *const c_void,
        dscores_ws: *mut c_void,
        dQ: *mut c_void,
        dK: *mut c_void,
        dV: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// SDPA BW, f64.
    pub fn baracuda_kernels_sdpa_backward_f64_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        attn: *const c_void,
        dy: *const c_void,
        dscores_ws: *mut c_void,
        dQ: *mut c_void,
        dK: *mut c_void,
        dV: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // =========================================================================
    // Milestone 6.5 — KV-cache append (decoder-inference helper). Each
    // launcher fires two device-side copy kernels (K + V) on the same
    // stream. Pure copy → bit-exact at every dtype.
    //
    // Inputs:
    //   k_new          : T[B, H, L_new, D_k]
    //   v_new          : T[B, H, L_new, D_v]
    //   cache_offsets  : i64[B] — per-sample insert offset
    // Outputs (modified in place):
    //   k_cache        : T[B, H, L_max, D_k]
    //   v_cache        : T[B, H, L_max, D_v]
    // Cells where `cache_offsets[b] + l_new >= L_max` are silently skipped.
    // =========================================================================

    /// KV-cache append, f32.
    pub fn baracuda_kernels_kv_cache_append_f32_run(
        batch: i32,
        heads: i32,
        new_len: i32,
        max_cache_len: i32,
        d_k: i32,
        d_v: i32,
        k_new: *const c_void,
        v_new: *const c_void,
        cache_offsets: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// KV-cache append, f16.
    pub fn baracuda_kernels_kv_cache_append_f16_run(
        batch: i32,
        heads: i32,
        new_len: i32,
        max_cache_len: i32,
        d_k: i32,
        d_v: i32,
        k_new: *const c_void,
        v_new: *const c_void,
        cache_offsets: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// KV-cache append, bf16.
    pub fn baracuda_kernels_kv_cache_append_bf16_run(
        batch: i32,
        heads: i32,
        new_len: i32,
        max_cache_len: i32,
        d_k: i32,
        d_v: i32,
        k_new: *const c_void,
        v_new: *const c_void,
        cache_offsets: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// KV-cache append, f64.
    pub fn baracuda_kernels_kv_cache_append_f64_run(
        batch: i32,
        heads: i32,
        new_len: i32,
        max_cache_len: i32,
        d_k: i32,
        d_v: i32,
        k_new: *const c_void,
        v_new: *const c_void,
        cache_offsets: *const c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    // =========================================================================
    // Milestone 6.6 — Flash Attention SDPA (Tri Dao 2022). Tiled fused
    // online-softmax kernel that avoids materializing the full
    // `[B, H, Q, K]` attention matrix; saves a small `lse: [B, H, Q]`
    // log-sum-exp tensor for the BW pass instead. BW is a deterministic
    // 3-kernel pipeline (D = rowsum(y ⊙ dy), then dQ per q-block, then
    // dK/dV per k-block — each output cell written by exactly one block,
    // no atomicAdd). Trailblazer constraints: Br = Bc = 64,
    // d_k = d_v ≤ 128.
    // =========================================================================

    /// Flash SDPA FW, f32. Computes `y = softmax(Q·K^T·scale) · V` via
    /// tiled fused online softmax. Optional upper-triangular causal mask
    /// (`is_causal = 1`); explicit additive mask is not supported in the
    /// trailblazer. Writes `y: [B, H, Q, D_v]` and the saved
    /// `lse: [B, H, Q]` log-sum-exp tensor that BW consumes.
    pub fn baracuda_kernels_flash_sdpa_f32_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        y: *mut c_void,
        lse: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Flash SDPA FW, f16 (f32 accumulators).
    pub fn baracuda_kernels_flash_sdpa_f16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        y: *mut c_void,
        lse: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Flash SDPA FW, bf16 (f32 accumulators).
    pub fn baracuda_kernels_flash_sdpa_bf16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        y: *mut c_void,
        lse: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Flash SDPA FW, f64.
    pub fn baracuda_kernels_flash_sdpa_f64_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        y: *mut c_void,
        lse: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Flash SDPA BW, f32. Given the FW-saved `y`, `lse`, plus upstream
    /// `dy`, computes `dQ`, `dK`, `dV`. The `d_ws` argument is a
    /// caller-allocated `[B, H, Q]` scratch buffer (overwritten with the
    /// per-row `D = rowsum(y ⊙ dy)` intermediate; element type matches T).
    pub fn baracuda_kernels_flash_sdpa_backward_f32_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        y: *const c_void,
        lse: *const c_void,
        dy: *const c_void,
        d_ws: *mut c_void,
        dQ: *mut c_void,
        dK: *mut c_void,
        dV: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Flash SDPA BW, f16.
    pub fn baracuda_kernels_flash_sdpa_backward_f16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        y: *const c_void,
        lse: *const c_void,
        dy: *const c_void,
        d_ws: *mut c_void,
        dQ: *mut c_void,
        dK: *mut c_void,
        dV: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Flash SDPA BW, bf16.
    pub fn baracuda_kernels_flash_sdpa_backward_bf16_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        y: *const c_void,
        lse: *const c_void,
        dy: *const c_void,
        d_ws: *mut c_void,
        dQ: *mut c_void,
        dK: *mut c_void,
        dV: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Flash SDPA BW, f64.
    pub fn baracuda_kernels_flash_sdpa_backward_f64_run(
        batch: i32,
        heads: i32,
        q_len: i32,
        k_len: i32,
        d_k: i32,
        d_v: i32,
        scale: f32,
        is_causal: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        y: *const c_void,
        lse: *const c_void,
        dy: *const c_void,
        d_ws: *mut c_void,
        dQ: *mut c_void,
        dK: *mut c_void,
        dV: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// cuSOLVER — Milestone 6.3 dense linalg
// ============================================================================
//
// Host-API cuSOLVER bindings for the four canonical dense factorizations:
// Cholesky (`potrf`), LU (`getrf`), QR (`geqrf` + `ormqr`), SVD (`gesvd`).
// f32 + f64 only — cuSOLVER's dense API does not expose f16 / bf16 for
// these operations. Cholesky and LU also expose batched variants
// (`*Batched`) that operate on an array of `[N, N]` matrices in a
// single launch; QR / SVD have no batched dense variant on cuSOLVER and
// are 2-D-only at the plan layer.
//
// Linkage: `cargo:rustc-link-lib=dylib=cusolver` + `=cublas` (both
// added in build.rs — cuSOLVER's dense API depends on cuBLAS). On Linux
// these resolve to `libcusolver.so` / `libcublas.so`; on Windows to
// `cusolver64_*.dll` / `cublas64_*.dll` (loaded from `CUDA_PATH\bin`).
//
// All cuSOLVER routines are column-major (matching LAPACK convention).
// The safe-plan layer in `baracuda-kernels` handles the row-major →
// column-major adapter — for symmetric ops (Cholesky) this is a uplo-flip
// (row-major lower-L over storage `S` is bit-identical to column-major
// upper-U over the same `S`); for non-symmetric ops (LU / QR / SVD) the
// plan documents that the input/output is interpreted as the transpose.

/// Opaque cuSOLVER dense handle. Stateful object; the plan layer creates
/// one lazily on first `run` and reuses across launches.
#[allow(non_camel_case_types)]
pub type cusolverDnHandle_t = *mut c_void;

/// Opaque cuBLAS handle. Used by `cublas*geqrfBatched` (which lives in
/// cuBLAS, not cuSOLVER) and any future cuBLAS-routed linalg paths.
#[allow(non_camel_case_types)]
pub type cublasHandle_t = *mut c_void;

/// Opaque cuSOLVER Jacobi-SVD parameter object. Stateful; created
/// once per plan, reused across launches, destroyed on plan drop.
/// Used by `cusolverDn*gesvdjBatched` for the batched-SVD path.
#[allow(non_camel_case_types)]
pub type gesvdjInfo_t = *mut c_void;

/// cuBLAS fill-mode tag re-used by cuSOLVER for triangular factorizations.
/// `CUBLAS_FILL_MODE_LOWER = 0`, `CUBLAS_FILL_MODE_UPPER = 1`.
#[allow(non_camel_case_types)]
pub type cublasFillMode_t = i32;

/// `CUBLAS_FILL_MODE_LOWER` — pass to `potrf` to request the lower-
/// triangular Cholesky factor.
pub const CUBLAS_FILL_MODE_LOWER: i32 = 0;

/// `CUBLAS_FILL_MODE_UPPER` — pass to `potrf` to request the upper-
/// triangular Cholesky factor.
pub const CUBLAS_FILL_MODE_UPPER: i32 = 1;

/// `CUBLAS_OP_N` — no transpose. Used by `ormqr` to control whether to
/// apply `Q` or `Q^T`.
pub const CUBLAS_OP_N: i32 = 0;

/// `CUBLAS_OP_T` — transpose.
pub const CUBLAS_OP_T: i32 = 1;

/// `CUBLAS_OP_C` — conjugate transpose (only meaningful for complex
/// dtypes). Used by `cusolverDn{C,Z}unmqr` to apply `Q^H`.
pub const CUBLAS_OP_C: i32 = 2;

/// `CUBLAS_SIDE_LEFT` — `Q` is applied from the left in `ormqr`
/// (`C := Q · C` or `C := Q^T · C`).
pub const CUBLAS_SIDE_LEFT: i32 = 0;

/// `CUBLAS_SIDE_RIGHT` — `Q` is applied from the right.
pub const CUBLAS_SIDE_RIGHT: i32 = 1;

/// cuBLAS diag-type tag for triangular solves (`trsm`).
/// `CUBLAS_DIAG_NON_UNIT = 0`, `CUBLAS_DIAG_UNIT = 1`.
#[allow(non_camel_case_types)]
pub type cublasDiagType_t = i32;

/// `CUBLAS_DIAG_NON_UNIT` — `trsm` reads the actual diagonal of `A`.
/// Used by the LstSq QR-fallback path for the back-substitution
/// `R · X = Q^T · B`, where `R`'s diagonal is the meaningful pivots.
pub const CUBLAS_DIAG_NON_UNIT: i32 = 0;

/// `CUBLAS_DIAG_UNIT` — `trsm` treats the diagonal as all-1s
/// (unit-triangular). Not used by the current plan layer; surfaced
/// for completeness.
pub const CUBLAS_DIAG_UNIT: i32 = 1;

/// `CUSOLVER_STATUS_SUCCESS` — the only success code. Any non-zero
/// return from a cuSOLVER routine is mapped to a negative status at the
/// safe-plan layer for distinct error reporting.
pub const CUSOLVER_STATUS_SUCCESS: i32 = 0;

/// cuSOLVER eig-mode enum tag (used by `syevd` / `heevd` / `Xgeev`).
/// `0 = NOVECTOR` (compute eigenvalues only), `1 = VECTOR` (eigenvalues +
/// eigenvectors). Routed through as an `i32` for the legacy syevd /
/// heevd APIs. The `CUSOLVER_EIG_MODE_NOVECTOR` / `_VECTOR` constants
/// live further down (originally introduced for `gesvdjBatched`'s
/// `jobz` argument; reused verbatim here for the eig family).
#[allow(non_camel_case_types)]
pub type cusolverEigMode_t = i32;

/// `cudaDataType` tag used by the 64-bit cuSOLVER APIs (`Xgeev`,
/// `Xgesvd`, …) to identify tensor element types. These constants
/// originate in `<library_types.h>` and are stable across CUDA versions.
#[allow(non_camel_case_types)]
pub type cudaDataType = i32;

/// `CUDA_R_32F` — real `f32`.
pub const CUDA_R_32F: i32 = 0;
/// `CUDA_R_64F` — real `f64`.
pub const CUDA_R_64F: i32 = 1;
/// `CUDA_R_16F` — real `f16`.
pub const CUDA_R_16F: i32 = 2;
/// `CUDA_C_32F` — complex `f32` (interleaved real/imag).
pub const CUDA_C_32F: i32 = 4;
/// `CUDA_C_64F` — complex `f64` (interleaved real/imag).
pub const CUDA_C_64F: i32 = 5;

/// Opaque parameter struct used by the 64-bit cuSOLVER APIs (`Xgeev`,
/// `Xpotrf`, …). The struct holds advanced configuration (algorithm
/// choice, precision modes) — for the trailblazer the plan layer leaves
/// it at defaults. Created via `cusolverDnCreateParams` and destroyed via
/// `cusolverDnDestroyParams`.
#[allow(non_camel_case_types)]
pub type cusolverDnParams_t = *mut c_void;

/// ABI-compatible single-precision complex struct, matching `cuComplex`
/// from `<cuComplex.h>` (interleaved real/imag `f32`). Identical layout
/// to [`crate::cufftComplex`] and to the safe-side [`Complex32`] from
/// `baracuda-kernels-types` — a `DeviceBuffer<Complex32>` can be cast
/// to a `*mut cuComplex` for the cuSOLVER complex APIs without copy.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[allow(non_camel_case_types)]
pub struct cuComplex {
    /// Real component.
    pub x: f32,
    /// Imaginary component.
    pub y: f32,
}

/// ABI-compatible double-precision complex struct, matching
/// `cuDoubleComplex` from `<cuComplex.h>`. Sibling to [`cuComplex`].
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[allow(non_camel_case_types)]
pub struct cuDoubleComplex {
    /// Real component.
    pub x: f64,
    /// Imaginary component.
    pub y: f64,
}

/// `cuFloatComplex` is the canonical CUDA name for the single-precision
/// complex struct — an alias for [`cuComplex`]. Surfaced so cuSOLVER's
/// complex APIs (`cusolverDn{C,Z}unmqr`, …) can spell their signatures
/// in the same vocabulary as the NVIDIA headers.
#[allow(non_camel_case_types)]
pub type cuFloatComplex = cuComplex;

unsafe extern "C" {
    // ----- handle lifecycle ----------------------------------------------

    /// `cusolverDnCreate(handle)`. Returns 0 on success.
    ///
    /// # Safety
    /// `handle` must point to writable storage for one `cusolverDnHandle_t`.
    pub fn cusolverDnCreate(handle: *mut cusolverDnHandle_t) -> i32;

    /// `cusolverDnDestroy(handle)`. Returns 0 on success.
    ///
    /// # Safety
    /// `handle` must be a valid handle returned by `cusolverDnCreate` that
    /// has not been previously destroyed.
    pub fn cusolverDnDestroy(handle: cusolverDnHandle_t) -> i32;

    /// `cusolverDnSetStream(handle, stream)`. Binds subsequent cuSOLVER
    /// calls to the given CUDA stream. Returns 0 on success.
    ///
    /// # Safety
    /// `handle` must be a live cuSOLVER handle; `stream` must be a valid
    /// CUDA stream in the current context (or null for the default stream).
    pub fn cusolverDnSetStream(handle: cusolverDnHandle_t, stream: *mut c_void) -> i32;

    // ----- Cholesky: potrf (f32 / f64) -----------------------------------

    /// `cusolverDnSpotrf_bufferSize` — query workspace bytes (as element
    /// count, must be multiplied by `sizeof(T)` for `cudaMalloc`).
    ///
    /// # Safety
    /// `handle` live; `A` device pointer to `n*n` `float` cells with leading
    /// dimension `lda`; `lwork` writable storage for one `int`.
    pub fn cusolverDnSpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *mut f32,
        lda: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnSpotrf` — Cholesky factorization in-place (`A := L`
    /// or `A := U`). Writes the unused triangle untouched. `dev_info`
    /// returns 0 on success, `k > 0` if the leading `k`-minor is not
    /// positive definite (factorization halted at step `k`).
    ///
    /// # Safety
    /// All pointers reference device memory; `workspace` has at least
    /// `lwork * sizeof(float)` bytes; `dev_info` references one `int`.
    pub fn cusolverDnSpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *mut f32,
        lda: i32,
        workspace: *mut f32,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnDpotrf_bufferSize`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *mut f64,
        lda: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnDpotrf`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *mut f64,
        lda: i32,
        workspace: *mut f64,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    // ----- Cholesky batched ----------------------------------------------

    /// `cusolverDnSpotrfBatched(handle, uplo, n, Aarray, lda, infoArray,
    /// batchSize)`. Each matrix in `Aarray[batch_size]` is factored
    /// independently in-place. Returns 0 on success; per-matrix factor
    /// info lands in `infoArray[i]`.
    ///
    /// # Safety
    /// `Aarray` is a device-resident array of `batch_size` pointers, each
    /// pointing to an `n × n` `float` matrix with leading dimension `lda`.
    /// `infoArray` is a device-resident `int[batch_size]` written by the
    /// kernel. Note: cuSOLVER's batched API does **not** take a workspace
    /// argument — the library allocates internally.
    pub fn cusolverDnSpotrfBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: i32,
        a_array: *mut *mut f32,
        lda: i32,
        info_array: *mut i32,
        batch_size: i32,
    ) -> i32;

    /// `cusolverDnDpotrfBatched`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDpotrfBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: i32,
        a_array: *mut *mut f64,
        lda: i32,
        info_array: *mut i32,
        batch_size: i32,
    ) -> i32;

    // ----- LU: getrf (f32 / f64) -----------------------------------------

    /// `cusolverDnSgetrf_bufferSize` — query workspace element count.
    ///
    /// # Safety
    /// `handle` live; `A` device pointer to `m*n` `float` cells with
    /// leading dimension `lda`; `lwork` writable `int`.
    pub fn cusolverDnSgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnSgetrf` — LU factorization with partial pivoting in
    /// place. `A := L · U` (with `L` unit-diagonal, stored in the strict
    /// lower triangle; `U` in the upper triangle). `ipiv[i]` is the row
    /// swap performed at step `i` (1-based per LAPACK convention).
    ///
    /// # Safety
    /// All pointers reference device memory; `workspace` ≥
    /// `lwork * sizeof(float)` bytes; `ipiv` ≥ `min(m, n) * sizeof(int)`
    /// bytes; `dev_info` is one `int`.
    pub fn cusolverDnSgetrf(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        workspace: *mut f32,
        ipiv: *mut i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnDgetrf_bufferSize`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut f64,
        lda: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnDgetrf`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDgetrf(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut f64,
        lda: i32,
        workspace: *mut f64,
        ipiv: *mut i32,
        dev_info: *mut i32,
    ) -> i32;

    // ----- LU solve: getrs (f32 / f64) -----------------------------------
    //
    // `getrs` consumes the packed `LU` factors + pivot vector produced
    // by `getrf` and solves `op(A) · X = B` in place over `B`. cuSOLVER
    // does not expose a `_bufferSize` query for `getrs` — the routine
    // is workspace-free.

    /// `cusolverDnSgetrs` — solve `op(A) · X = B` using the packed `LU`
    /// + pivot produced by `cusolverDnSgetrf`. `B` is overwritten in
    /// place with the solution `X`. `trans` selects `op(A)`:
    /// `CUBLAS_OP_N` for `A`, `CUBLAS_OP_T` for `A^T`.
    ///
    /// # Safety
    /// `handle` live + stream-bound; `A` is the packed `getrf` output
    /// `n × n` `float` (lda ≥ n); `ipiv` is the 1-based pivot vector of
    /// length `n` returned by `getrf`; `B` is `n × nrhs` `float` (ldb ≥
    /// n); `dev_info` is one writable `int`.
    pub fn cusolverDnSgetrs(
        handle: cusolverDnHandle_t,
        trans: i32,
        n: i32,
        nrhs: i32,
        a: *const f32,
        lda: i32,
        ipiv: *const i32,
        b: *mut f32,
        ldb: i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnDgetrs`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDgetrs(
        handle: cusolverDnHandle_t,
        trans: i32,
        n: i32,
        nrhs: i32,
        a: *const f64,
        lda: i32,
        ipiv: *const i32,
        b: *mut f64,
        ldb: i32,
        dev_info: *mut i32,
    ) -> i32;

    // ----- QR: geqrf + ormqr (f32 / f64) ---------------------------------
    //
    // Note: cuSOLVER's dense API does not expose a batched LU
    // (`cublasSgetrfBatched` lives in cuBLAS — wiring batched LU
    // through cuBLAS is deferred to a future milestone). Batched
    // Cholesky stays in cuSOLVER (`*potrfBatched` above).

    /// `cusolverDnSgeqrf_bufferSize`.
    ///
    /// # Safety
    /// `handle` live; `A` device `m × n` `float` with leading dimension
    /// `lda`; `lwork` writable `int`.
    pub fn cusolverDnSgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnSgeqrf` — QR factorization in place. `A` is overwritten:
    /// upper triangle = `R`, strict lower triangle + `tau` = Householder
    /// reflectors that encode `Q`. To materialize `Q` as a dense matrix,
    /// follow with `ormqr` against an identity.
    ///
    /// # Safety
    /// All pointers reference device memory; `tau ≥ min(m, n) * sizeof(T)`;
    /// `workspace ≥ lwork * sizeof(T)`; `dev_info` is one `int`.
    pub fn cusolverDnSgeqrf(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        tau: *mut f32,
        workspace: *mut f32,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnDgeqrf_bufferSize`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut f64,
        lda: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnDgeqrf`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDgeqrf(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut f64,
        lda: i32,
        tau: *mut f64,
        workspace: *mut f64,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnSormqr_bufferSize`. `trans` selects `Q` vs `Q^T`;
    /// `side` selects left vs right multiply.
    ///
    /// # Safety
    /// `handle` live; `A` / `C` are the `geqrf`-output matrix and the
    /// target matrix respectively; `tau` is the Householder scalars from
    /// `geqrf`. `lwork` writable `int`.
    pub fn cusolverDnSormqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: i32,
        trans: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const f32,
        lda: i32,
        tau: *const f32,
        c: *const f32,
        ldc: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnSormqr` — apply `Q` (or `Q^T`) from `geqrf` output to
    /// a matrix `C` in place. With `C = I` this materializes `Q` as a
    /// dense matrix for the "thin" or "full" QR.
    ///
    /// # Safety
    /// All pointers reference device memory; `workspace ≥
    /// lwork * sizeof(T)`; `dev_info` is one `int`.
    pub fn cusolverDnSormqr(
        handle: cusolverDnHandle_t,
        side: i32,
        trans: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const f32,
        lda: i32,
        tau: *const f32,
        c: *mut f32,
        ldc: i32,
        workspace: *mut f32,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnDormqr_bufferSize`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDormqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: i32,
        trans: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const f64,
        lda: i32,
        tau: *const f64,
        c: *const f64,
        ldc: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnDormqr`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDormqr(
        handle: cusolverDnHandle_t,
        side: i32,
        trans: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const f64,
        lda: i32,
        tau: *const f64,
        c: *mut f64,
        ldc: i32,
        workspace: *mut f64,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    // ----- QR factorization (complex): geqrf (Complex32 / Complex64) ------

    /// `cusolverDnCgeqrf_bufferSize` — workspace query for single-precision
    /// complex QR factorization. Mirrors `cusolverDnSgeqrf_bufferSize`.
    ///
    /// # Safety
    /// `handle` live; `A` device `m × n` `cuFloatComplex` with leading
    /// dimension `lda`; `lwork` writable `int`.
    pub fn cusolverDnCgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut cuFloatComplex,
        lda: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnCgeqrf` — single-precision complex QR factorization,
    /// in place. The packed output uses the same convention as the real
    /// variant: strict lower triangle + `tau` encode the Householder
    /// reflectors; the upper triangle holds `R`.
    ///
    /// # Safety
    /// All pointers reference device memory; `tau ≥ min(m, n)` cells;
    /// `workspace ≥ lwork * sizeof(cuFloatComplex)`; `dev_info` one `int`.
    pub fn cusolverDnCgeqrf(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut cuFloatComplex,
        lda: i32,
        tau: *mut cuFloatComplex,
        workspace: *mut cuFloatComplex,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnZgeqrf_bufferSize`. f64-complex analogue of the C variant.
    ///
    /// # Safety
    /// Same as the C variant with `cuDoubleComplex` storage.
    pub fn cusolverDnZgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut cuDoubleComplex,
        lda: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnZgeqrf` — double-precision complex QR factorization.
    ///
    /// # Safety
    /// Same as the C variant with `cuDoubleComplex` storage.
    pub fn cusolverDnZgeqrf(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        a: *mut cuDoubleComplex,
        lda: i32,
        tau: *mut cuDoubleComplex,
        workspace: *mut cuDoubleComplex,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    // ----- Apply Q from QR (complex): unmqr (Complex32 / Complex64) -------
    //
    // cuSOLVER spells the complex apply-Q routine `unmqr` ("unitary mqr")
    // — the same API surface as `ormqr` but with `cuComplex` /
    // `cuDoubleComplex` storage. `trans = CUBLAS_OP_C` selects `Q^H`
    // (conjugate transpose).

    /// `cusolverDnCunmqr_bufferSize`.
    ///
    /// # Safety
    /// All pointers device-resident; `lwork` writable `int`.
    pub fn cusolverDnCunmqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: i32,
        trans: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const cuFloatComplex,
        lda: i32,
        tau: *const cuFloatComplex,
        c: *const cuFloatComplex,
        ldc: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnCunmqr` — apply `Q`, `Q^T`, or `Q^H` from a complex
    /// `geqrf` factorization to a complex `C` in place.
    ///
    /// # Safety
    /// All pointers reference device memory; `workspace ≥
    /// lwork * sizeof(cuFloatComplex)`; `dev_info` is one `int`.
    pub fn cusolverDnCunmqr(
        handle: cusolverDnHandle_t,
        side: i32,
        trans: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const cuFloatComplex,
        lda: i32,
        tau: *const cuFloatComplex,
        c: *mut cuFloatComplex,
        ldc: i32,
        workspace: *mut cuFloatComplex,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnZunmqr_bufferSize`. f64-complex analogue.
    ///
    /// # Safety
    /// Same as the C variant with `cuDoubleComplex` storage.
    pub fn cusolverDnZunmqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: i32,
        trans: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const cuDoubleComplex,
        lda: i32,
        tau: *const cuDoubleComplex,
        c: *const cuDoubleComplex,
        ldc: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnZunmqr`. f64-complex analogue.
    ///
    /// # Safety
    /// Same as the C variant with `cuDoubleComplex` storage.
    pub fn cusolverDnZunmqr(
        handle: cusolverDnHandle_t,
        side: i32,
        trans: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const cuDoubleComplex,
        lda: i32,
        tau: *const cuDoubleComplex,
        c: *mut cuDoubleComplex,
        ldc: i32,
        workspace: *mut cuDoubleComplex,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    // ----- SVD: gesvd (f32 / f64) ----------------------------------------

    /// `cusolverDnSgesvd_bufferSize`.
    ///
    /// # Safety
    /// `handle` live; `lwork` writable `int`. cuSOLVER's `gesvd_bufferSize`
    /// signature does not take a matrix pointer (m and n suffice).
    pub fn cusolverDnSgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnSgesvd` — SVD: `A = U · diag(S) · V^T`. The `jobu` /
    /// `jobv` characters are ASCII bytes: `'A'` (full U/V^T), `'S'` (thin
    /// U/V^T), `'O'` (overwrite A — disallowed at plan layer), `'N'`
    /// (skip).
    ///
    /// # Safety
    /// All pointers reference device memory; `S ≥ min(m, n) * sizeof(T)`;
    /// `U ≥ m*m * sizeof(T)` (full) or `m * min(m,n) * sizeof(T)` (thin);
    /// `VT ≥ n*n * sizeof(T)` (full) or `min(m,n) * n * sizeof(T)` (thin);
    /// `workspace ≥ lwork * sizeof(T)`; `rwork` may be null for real
    /// dtypes; `dev_info` is one `int`. Important: cuSOLVER's `gesvd`
    /// **requires** `m ≥ n` — callers that need `m < n` must transpose
    /// the input first.
    pub fn cusolverDnSgesvd(
        handle: cusolverDnHandle_t,
        jobu: u8,
        jobv: u8,
        m: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        s: *mut f32,
        u: *mut f32,
        ldu: i32,
        vt: *mut f32,
        ldvt: i32,
        workspace: *mut f32,
        lwork: i32,
        rwork: *mut f32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnDgesvd_bufferSize`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant.
    pub fn cusolverDnDgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnDgesvd`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDgesvd(
        handle: cusolverDnHandle_t,
        jobu: u8,
        jobv: u8,
        m: i32,
        n: i32,
        a: *mut f64,
        lda: i32,
        s: *mut f64,
        u: *mut f64,
        ldu: i32,
        vt: *mut f64,
        ldvt: i32,
        workspace: *mut f64,
        lwork: i32,
        rwork: *mut f64,
        dev_info: *mut i32,
    ) -> i32;

    // ----- Symmetric / Hermitian eigendecomposition: syevd / heevd ------
    //
    // `syevd` / `heevd` compute the eigenvalues + eigenvectors of a real
    // symmetric (`syevd`) or complex Hermitian (`heevd`) matrix using the
    // divide-and-conquer algorithm. The input matrix is overwritten in
    // place with the eigenvectors (column-major); a separate `W` vector
    // receives the (always-real) eigenvalues.
    //
    // `jobz` is `CUSOLVER_EIG_MODE_VECTOR` (compute eigenvectors) or
    // `CUSOLVER_EIG_MODE_NOVECTOR` (eigenvalues only). `uplo` selects
    // which triangle of the input to read (`CUBLAS_FILL_MODE_LOWER` /
    // `_UPPER`).

    /// `cusolverDnSsyevd_bufferSize` — query workspace element count for
    /// real-symmetric divide-and-conquer eigh, f32.
    ///
    /// # Safety
    /// `handle` live; `A` device `n × n` `float` (lda ≥ n); `W` device
    /// `float` of length `n`; `lwork` writable `int`.
    pub fn cusolverDnSsyevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *const f32,
        lda: i32,
        w: *const f32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnSsyevd` — real-symmetric eigh, f32. `A` is overwritten
    /// in place with the eigenvectors (column-major) when `jobz ==
    /// VECTOR`. `W` receives the `n` eigenvalues sorted ascending.
    ///
    /// # Safety
    /// All pointers reference device memory; `workspace ≥ lwork *
    /// sizeof(float)`; `dev_info` is one writable `int`.
    pub fn cusolverDnSsyevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *mut f32,
        lda: i32,
        w: *mut f32,
        workspace: *mut f32,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnDsyevd_bufferSize`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDsyevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *const f64,
        lda: i32,
        w: *const f64,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnDsyevd`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDsyevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *mut f64,
        lda: i32,
        w: *mut f64,
        workspace: *mut f64,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnCheevd_bufferSize` — complex-Hermitian divide-and-conquer
    /// eigh, single precision (`Complex32`). Eigenvalues are real-valued
    /// `float`.
    ///
    /// # Safety
    /// `handle` live; `A` device `n × n` `cuComplex` (lda ≥ n); `W`
    /// device `float` of length `n`; `lwork` writable `int`.
    pub fn cusolverDnCheevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *const cuComplex,
        lda: i32,
        w: *const f32,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnCheevd` — complex-Hermitian eigh (`Complex32`). `A` is
    /// overwritten in place with the eigenvectors (column-major); `W`
    /// receives the `n` real eigenvalues sorted ascending.
    ///
    /// # Safety
    /// All pointers reference device memory; `workspace ≥ lwork *
    /// sizeof(cuComplex)`; `dev_info` is one writable `int`.
    pub fn cusolverDnCheevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *mut cuComplex,
        lda: i32,
        w: *mut f32,
        workspace: *mut cuComplex,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnZheevd_bufferSize`. `Complex64` analogue.
    ///
    /// # Safety
    /// Same as the `Cheevd` variant with `cuDoubleComplex` / `f64`
    /// storage.
    pub fn cusolverDnZheevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *const cuDoubleComplex,
        lda: i32,
        w: *const f64,
        lwork: *mut i32,
    ) -> i32;

    /// `cusolverDnZheevd`. `Complex64` analogue.
    ///
    /// # Safety
    /// Same as the `Cheevd` variant with `cuDoubleComplex` / `f64`
    /// storage.
    pub fn cusolverDnZheevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i32,
        a: *mut cuDoubleComplex,
        lda: i32,
        w: *mut f64,
        workspace: *mut cuDoubleComplex,
        lwork: i32,
        dev_info: *mut i32,
    ) -> i32;

    // ----- Generic eigendecomposition: Xgeev (64-bit API) ---------------
    //
    // `Xgeev` is the cuSOLVER 11+ 64-bit-index API for the general (non-
    // symmetric) eigendecomposition. Differences from the legacy `Sgeev`
    // / `Dgeev` family:
    //
    //   - Takes a `cusolverDnParams_t` opaque settings struct (created
    //     via `cusolverDnCreateParams`, destroyed via
    //     `cusolverDnDestroyParams`).
    //   - Indices are `int64_t`, not `int`.
    //   - Workspace sizes are `size_t` byte counts, NOT element counts;
    //     the buffer-size query returns BOTH a host-side and a device-side
    //     byte count, and `Xgeev` itself takes both buffers.
    //   - Tensor element types are passed as `cudaDataType` tags (CUDA_R_32F
    //     / CUDA_R_64F / CUDA_C_32F / CUDA_C_64F). The same routine handles
    //     all four input dtypes.
    //   - Eigenvalues `W` are **always complex** (`cudaDataType` must be
    //     CUDA_C_32F or CUDA_C_64F) — for real input the complex-conjugate
    //     pairs are stored explicitly rather than packed into a wr/wi
    //     LAPACK-style split.
    //
    // `jobvl` / `jobvr` are `CUSOLVER_EIG_MODE_VECTOR` (compute) or
    // `CUSOLVER_EIG_MODE_NOVECTOR` (skip — pass null for the corresponding
    // VL / VR pointers in that case).

    /// `cusolverDnCreateParams` — allocate the opaque params struct used
    /// by all 64-bit cuSOLVER APIs. Plan layer creates one lazily on
    /// first `run` (mirroring the handle lifecycle).
    ///
    /// # Safety
    /// `params` must point to writable storage for one `cusolverDnParams_t`.
    pub fn cusolverDnCreateParams(params: *mut cusolverDnParams_t) -> i32;

    /// `cusolverDnDestroyParams`. Returns 0 on success.
    ///
    /// # Safety
    /// `params` must be a live params struct returned by
    /// `cusolverDnCreateParams` that has not already been destroyed.
    pub fn cusolverDnDestroyParams(params: cusolverDnParams_t) -> i32;

    /// `cusolverDnXgeev_bufferSize` — query the host + device byte
    /// counts for `cusolverDnXgeev` at the given problem size and
    /// element types. The two output pointers receive byte counts (NOT
    /// element counts — different from the legacy `_bufferSize` APIs).
    ///
    /// # Safety
    /// `handle` / `params` live; pointer args reference device memory of
    /// the indicated `cudaDataType`; `workspace_in_bytes_on_device` and
    /// `workspace_in_bytes_on_host` point to writable `size_t`.
    pub fn cusolverDnXgeev_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobvl: cusolverEigMode_t,
        jobvr: cusolverEigMode_t,
        n: i64,
        data_type_a: cudaDataType,
        a: *const c_void,
        lda: i64,
        data_type_w: cudaDataType,
        w: *const c_void,
        data_type_vl: cudaDataType,
        vl: *const c_void,
        ldvl: i64,
        data_type_vr: cudaDataType,
        vr: *const c_void,
        ldvr: i64,
        compute_type: cudaDataType,
        workspace_in_bytes_on_device: *mut usize,
        workspace_in_bytes_on_host: *mut usize,
    ) -> i32;

    /// `cusolverDnXgeev` — general (non-symmetric) eigendecomposition.
    /// `A` is **destroyed in place** (used as scratch by the LAPACK-
    /// equivalent algorithm). `W` receives the `n` complex eigenvalues;
    /// `VL` / `VR` (when requested) receive the column-major left /
    /// right complex eigenvectors. For non-Hermitian input the
    /// eigenvalues can be complex even when the input is real, hence
    /// the always-complex `W` storage.
    ///
    /// # Safety
    /// All tensor pointer args reference device memory of the indicated
    /// `cudaDataType`; `workspace_on_device` ≥ `workspace_in_bytes_on_device`
    /// device bytes; `workspace_on_host` ≥ `workspace_in_bytes_on_host`
    /// host bytes (or null if `workspace_in_bytes_on_host == 0`); `info`
    /// is one writable device `int`. Pass null for `VL` / `VR` when the
    /// corresponding `jobv*` is `NOVECTOR`.
    pub fn cusolverDnXgeev(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobvl: cusolverEigMode_t,
        jobvr: cusolverEigMode_t,
        n: i64,
        data_type_a: cudaDataType,
        a: *mut c_void,
        lda: i64,
        data_type_w: cudaDataType,
        w: *mut c_void,
        data_type_vl: cudaDataType,
        vl: *mut c_void,
        ldvl: i64,
        data_type_vr: cudaDataType,
        vr: *mut c_void,
        ldvr: i64,
        compute_type: cudaDataType,
        workspace_on_device: *mut c_void,
        workspace_in_bytes_on_device: usize,
        workspace_on_host: *mut c_void,
        workspace_in_bytes_on_host: usize,
        info: *mut i32,
    ) -> i32;

    // ----- Batched QR: cublas*geqrfBatched (f32 / f64) -------------------
    //
    // NOTE: Despite belonging to the "linalg" family conceptually, the
    // batched-QR factorization is implemented in **cuBLAS**, not cuSOLVER.
    // cuSOLVER-Dn has no batched-geqrf entry point (only the non-batched
    // `cusolverDn<t>geqrf`). cuBLAS's variant is workspace-free (cuBLAS
    // allocates internally); it takes a *device-resident array of device
    // pointers* (`Aarray[]`, `TauArray[]`) — the plan layer builds this
    // array per-launch in caller-provided workspace.

    /// `cublasSgeqrfBatched` — batched QR factorization (single precision).
    /// Each `Aarray[b]` is overwritten in place with the `geqrf`-packed
    /// `R` (upper) + Householder reflectors (strict lower);
    /// `TauArray[b]` receives the Householder scalars.
    ///
    /// # Safety
    /// All pointers are device-resident. `Aarray` / `TauArray` are device
    /// arrays of device pointers (length `batch_size`). `info` is a single
    /// host `i32` indicating non-batched argument-validity (cuBLAS-batched
    /// QR contract differs from cuSOLVER: it returns a single info, not
    /// a per-slot array).
    pub fn cublasSgeqrfBatched(
        handle: cublasHandle_t,
        m: i32,
        n: i32,
        a_array: *mut *mut f32,
        lda: i32,
        tau_array: *mut *mut f32,
        info: *mut i32,
        batch_size: i32,
    ) -> i32;

    /// `cublasDgeqrfBatched`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cublasDgeqrfBatched(
        handle: cublasHandle_t,
        m: i32,
        n: i32,
        a_array: *mut *mut f64,
        lda: i32,
        tau_array: *mut *mut f64,
        info: *mut i32,
        batch_size: i32,
    ) -> i32;

    /// `cublasCgeqrfBatched`. Complex32 analogue. `tau_array[b]` is
    /// `cuComplex` (NOT real-typed even though tau is real-magnitude for
    /// real Householder — cuBLAS uses complex tau across the complex
    /// family so the same `apply` routines can dispatch uniformly).
    ///
    /// # Safety
    /// Same as the f32 variant with `cuComplex` storage.
    pub fn cublasCgeqrfBatched(
        handle: cublasHandle_t,
        m: i32,
        n: i32,
        a_array: *mut *mut cuComplex,
        lda: i32,
        tau_array: *mut *mut cuComplex,
        info: *mut i32,
        batch_size: i32,
    ) -> i32;

    /// `cublasZgeqrfBatched`. Complex64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `cuDoubleComplex` storage.
    pub fn cublasZgeqrfBatched(
        handle: cublasHandle_t,
        m: i32,
        n: i32,
        a_array: *mut *mut cuDoubleComplex,
        lda: i32,
        tau_array: *mut *mut cuDoubleComplex,
        info: *mut i32,
        batch_size: i32,
    ) -> i32;

    // ----- cuBLAS handle lifecycle ---------------------------------------

    /// `cublasCreate_v2` — create a cuBLAS handle.
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> i32;

    /// `cublasDestroy_v2` — destroy a cuBLAS handle.
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> i32;

    /// `cublasSetStream_v2` — bind a CUDA stream to the cuBLAS handle.
    pub fn cublasSetStream_v2(handle: cublasHandle_t, stream: *mut c_void) -> i32;

    // ----- Strided-batched GEMM: cublas{S,D}gemmStridedBatched (f32 / f64) -----
    //
    // Single-launch batched GEMM where each batch slot has identical
    // shape `(m, n, k)` but its operand pointers are reached by adding
    // a fixed `stride{A,B,C}` (in *element* counts) to the base pointer.
    // Used by the WY-blocked batched-`ormqr` plan (Milestone 6.17) to
    // apply each block reflector via three GEMMs per block: V^T·C, T·W,
    // and the rank-`nb` update C -= V·W.
    //
    // `alpha` / `beta` are host pointers (cuBLAS default pointer-mode).

    /// `cublasSgemmStridedBatched` — single-precision strided-batched
    /// matrix-matrix multiply. Each slot computes
    /// `C[i] := α · op(A[i]) · op(B[i]) + β · C[i]` where `A[i]`,
    /// `B[i]`, `C[i]` are reached by stepping `stride{A,B,C}` element
    /// counts from the respective base pointers.
    ///
    /// # Safety
    /// `handle` is a live cuBLAS handle bound to the desired stream.
    /// `alpha` / `beta` are host pointers to one `f32`. `a`, `b`, `c`
    /// are device pointers. Strides are in `f32` element counts (not
    /// bytes).
    pub fn cublasSgemmStridedBatched(
        handle: cublasHandle_t,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        stride_a: i64,
        b: *const f32,
        ldb: i32,
        stride_b: i64,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> i32;

    /// `cublasDgemmStridedBatched` — double-precision strided-batched
    /// matrix-matrix multiply. f64 analogue of [`cublasSgemmStridedBatched`].
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage; strides are in `f64`
    /// element counts.
    pub fn cublasDgemmStridedBatched(
        handle: cublasHandle_t,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f64,
        a: *const f64,
        lda: i32,
        stride_a: i64,
        b: *const f64,
        ldb: i32,
        stride_b: i64,
        beta: *const f64,
        c: *mut f64,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> i32;

    // ----- Triangular solve: cublas{S,D}trsm (f32 / f64) -----------------
    //
    // `trsm` solves one of the matrix equations
    //   op(A) · X = α · B   (side = LEFT)
    //   X · op(A) = α · B   (side = RIGHT)
    // for `X`, overwriting `B` in place. `A` is triangular (upper or
    // lower per `uplo`); `op(A)` is `A`, `A^T`, or `A^H` per `trans`.
    // `alpha` is a host pointer (cuBLAS default pointer-mode).
    //
    // The LstSq QR-fallback path uses
    //   side=LEFT, uplo=UPPER, trans=N, diag=NON_UNIT, α=1
    // to back-substitute `R · X = Q^T · B` (with `R` the top-left
    // `N × N` upper triangle of the post-`geqrf` packed `A`).

    /// `cublasStrsm` — single-precision triangular solve.
    ///
    /// # Safety
    /// `handle` is a live cuBLAS handle bound to the desired stream.
    /// `alpha` is a host pointer to one `f32`. `a` is a device pointer
    /// to at least `lda · k` floats (`k = m` for LEFT, `n` for RIGHT)
    /// of which only the requested triangle is read. `b` is device-
    /// resident `[ldb · n]` and is overwritten with the solution `X`.
    pub fn cublasStrsm(
        handle: cublasHandle_t,
        side: i32,
        uplo: cublasFillMode_t,
        trans: i32,
        diag: cublasDiagType_t,
        m: i32,
        n: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        b: *mut f32,
        ldb: i32,
    ) -> i32;

    /// `cublasDtrsm` — double-precision triangular solve. f64 analogue
    /// of [`cublasStrsm`].
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cublasDtrsm(
        handle: cublasHandle_t,
        side: i32,
        uplo: cublasFillMode_t,
        trans: i32,
        diag: cublasDiagType_t,
        m: i32,
        n: i32,
        alpha: *const f64,
        a: *const f64,
        lda: i32,
        b: *mut f64,
        ldb: i32,
    ) -> i32;

    // ----- Batched SVD (Jacobi): gesvdjBatched (f32 / f64) ---------------
    //
    // Jacobi-method batched SVD. Requires a `gesvdjInfo_t` parameter
    // object created via `cusolverDnCreateGesvdjInfo` / destroyed via
    // `cusolverDnDestroyGesvdjInfo`. The plan layer creates one on first
    // run (same lifetime pattern as the cuSOLVER handle itself).

    /// `cusolverDnCreateGesvdjInfo` — allocate a Jacobi-SVD params object
    /// with cuSOLVER's defaults (`tol = 1e-7` for f32 / `1e-12` for f64,
    /// `max_sweeps = 100`, `sort_eig = 1`).
    ///
    /// # Safety
    /// `info` must point to writable storage for one `gesvdjInfo_t`.
    pub fn cusolverDnCreateGesvdjInfo(info: *mut gesvdjInfo_t) -> i32;

    /// `cusolverDnDestroyGesvdjInfo`. Returns 0 on success.
    ///
    /// # Safety
    /// `info` must be a valid `gesvdjInfo_t` returned by
    /// `cusolverDnCreateGesvdjInfo` that has not been previously destroyed.
    pub fn cusolverDnDestroyGesvdjInfo(info: gesvdjInfo_t) -> i32;

    /// `cusolverDnSgesvdjBatched_bufferSize`. `jobz` is `0` (no vectors)
    /// or `1` (compute U / V). For batched, each matrix in `A` is
    /// independently SVD'd; outputs are packed `[batch * m * m]` etc.
    ///
    /// # Safety
    /// `handle` live; `params` a valid `gesvdjInfo_t`; `lwork` writable
    /// `int`.
    pub fn cusolverDnSgesvdjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: i32,
        m: i32,
        n: i32,
        a: *const f32,
        lda: i32,
        s: *const f32,
        u: *const f32,
        ldu: i32,
        v: *const f32,
        ldv: i32,
        lwork: *mut i32,
        params: gesvdjInfo_t,
        batch_size: i32,
    ) -> i32;

    /// `cusolverDnSgesvdjBatched` — batched Jacobi SVD `A = U · diag(S) · V^T`
    /// (single precision). Each matrix is square `[m, m]` (cuSOLVER's
    /// Jacobi-batched API requires square input; thin rectangular is
    /// achievable via `gesvdaStridedBatched` — deferred). The plan
    /// surfaces `V` (not `V^T`); callers apply the transpose if needed.
    ///
    /// # Safety
    /// All pointers are device-resident; `S ≥ batch*min(m,n) * sizeof(T)`;
    /// `U ≥ batch*m*m * sizeof(T)`; `V ≥ batch*n*n * sizeof(T)`;
    /// `workspace ≥ lwork * sizeof(T)`; `dev_info ≥ batch * sizeof(int)`.
    pub fn cusolverDnSgesvdjBatched(
        handle: cusolverDnHandle_t,
        jobz: i32,
        m: i32,
        n: i32,
        a: *mut f32,
        lda: i32,
        s: *mut f32,
        u: *mut f32,
        ldu: i32,
        v: *mut f32,
        ldv: i32,
        workspace: *mut f32,
        lwork: i32,
        info: *mut i32,
        params: gesvdjInfo_t,
        batch_size: i32,
    ) -> i32;

    /// `cusolverDnDgesvdjBatched_bufferSize`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDgesvdjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: i32,
        m: i32,
        n: i32,
        a: *const f64,
        lda: i32,
        s: *const f64,
        u: *const f64,
        ldu: i32,
        v: *const f64,
        ldv: i32,
        lwork: *mut i32,
        params: gesvdjInfo_t,
        batch_size: i32,
    ) -> i32;

    /// `cusolverDnDgesvdjBatched`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDgesvdjBatched(
        handle: cusolverDnHandle_t,
        jobz: i32,
        m: i32,
        n: i32,
        a: *mut f64,
        lda: i32,
        s: *mut f64,
        u: *mut f64,
        ldu: i32,
        v: *mut f64,
        ldv: i32,
        workspace: *mut f64,
        lwork: i32,
        info: *mut i32,
        params: gesvdjInfo_t,
        batch_size: i32,
    ) -> i32;

    // ----- Least-squares: gels (f32 / f64) -------------------------------
    //
    // Mixed-precision iterative-refinement `_gels` routine. The single-
    // precision entry is `cusolverDnSSgels` (S-input, S-compute) and the
    // double-precision entry is `cusolverDnDDgels` (D-input, D-compute).
    // Other letter combinations (SH, SB, DS, DH, DB) exist for mixed-
    // precision strategies but the plan layer surfaces only the
    // same-precision variants today. The routine returns the iteration
    // count via `niters`; if it failed to converge the safe-plan layer
    // reports a non-convergence error (QR-fallback is deferred).

    /// `cusolverDnSSgels_bufferSize` — query bytes (the routine's
    /// workspace is supplied as a raw byte buffer, not a typed element
    /// count, distinct from the `*_bufferSize` entries above).
    ///
    /// # Safety
    /// `handle` live; `lwork_bytes` writable `size_t`.
    pub fn cusolverDnSSgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        nrhs: i32,
        a: *mut f32,
        lda: i32,
        b: *mut f32,
        ldb: i32,
        x: *mut f32,
        ldx: i32,
        workspace: *mut c_void,
        lwork_bytes: *mut usize,
    ) -> i32;

    /// `cusolverDnSSgels` — least-squares solve `min ||A·x - b||²` for
    /// `m ≥ n` full-rank `A`. Iterative refinement; returns `niters` ≥ 0
    /// on convergence, `-N` on fallback-needed. Single precision.
    ///
    /// # Safety
    /// All pointers reference device memory; `x` is the device-resident
    /// solution buffer of length `n * nrhs * sizeof(T)`. `workspace_bytes`
    /// from the matching `_bufferSize`.
    pub fn cusolverDnSSgels(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        nrhs: i32,
        a: *mut f32,
        lda: i32,
        b: *mut f32,
        ldb: i32,
        x: *mut f32,
        ldx: i32,
        workspace: *mut c_void,
        lwork_bytes: usize,
        niters: *mut i32,
        dev_info: *mut i32,
    ) -> i32;

    /// `cusolverDnDDgels_bufferSize`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDDgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        nrhs: i32,
        a: *mut f64,
        lda: i32,
        b: *mut f64,
        ldb: i32,
        x: *mut f64,
        ldx: i32,
        workspace: *mut c_void,
        lwork_bytes: *mut usize,
    ) -> i32;

    /// `cusolverDnDDgels`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDDgels(
        handle: cusolverDnHandle_t,
        m: i32,
        n: i32,
        nrhs: i32,
        a: *mut f64,
        lda: i32,
        b: *mut f64,
        ldb: i32,
        x: *mut f64,
        ldx: i32,
        workspace: *mut c_void,
        lwork_bytes: usize,
        niters: *mut i32,
        dev_info: *mut i32,
    ) -> i32;

    // ----- Rectangular batched SVD: gesvdaStridedBatched (f32 / f64) ------
    //
    // Approximate-SVD batched API that, unlike `gesvdjBatched`, accepts
    // **rectangular** `[m, n]` matrices and uses **element-strides** between
    // batch slots (not pointer arrays). Per-slot residual Frobenius norms
    // are written to a **host** array `h_R_nrmF`.
    //
    // Gotcha: the `lwork` returned by `_bufferSize` (and accepted by the
    // exec call) is measured in **elements**, not bytes — multiply by
    // `sizeof(T)` to get the byte count for the `Workspace` buffer.
    //
    // The `rank` parameter (≤ `min(m, n)`) selects the number of singular
    // triplets to compute; pass `min(m, n)` for the full thin SVD. Outputs
    // are `S: [batch, rank]`, `U: [batch, m, rank]`, `V: [batch, n, rank]`
    // (column-major per slot). cuSOLVER returns `V` directly (not `V^T`).

    /// `cusolverDnSgesvdaStridedBatched_bufferSize` — query the device
    /// workspace size (in **elements**, multiply by `sizeof(f32)` for
    /// bytes) for the f32 rectangular-batched approximate-SVD.
    ///
    /// # Safety
    /// `handle` live; `lwork` writable `int`. Pointer args may be null
    /// (they're only inspected for shape inference).
    pub fn cusolverDnSgesvdaStridedBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: i32,
        rank: i32,
        m: i32,
        n: i32,
        a: *const f32,
        lda: i32,
        stride_a: i64,
        s: *const f32,
        stride_s: i64,
        u: *const f32,
        ldu: i32,
        stride_u: i64,
        v: *const f32,
        ldv: i32,
        stride_v: i64,
        lwork: *mut i32,
        batch_size: i32,
    ) -> i32;

    /// `cusolverDnSgesvdaStridedBatched` — f32 rectangular-batched
    /// approximate-SVD. Each batch slot factors a `[m, n]` matrix into
    /// `U: [m, rank]`, `S: [rank]`, `V: [n, rank]` (column-major;
    /// cuSOLVER returns `V`, not `V^T`). The host array `h_R_nrmF` (size
    /// `batch_size`) receives per-slot residual Frobenius norms.
    ///
    /// # Safety
    /// Device pointers `a`, `s`, `u` (when `jobz == VECTOR`), `v` (when
    /// `jobz == VECTOR`), `work`, `info` must be valid; `h_R_nrmF` is a
    /// **host** buffer of `batch_size` `f64`s; `lwork` from the matching
    /// `_bufferSize`.
    pub fn cusolverDnSgesvdaStridedBatched(
        handle: cusolverDnHandle_t,
        jobz: i32,
        rank: i32,
        m: i32,
        n: i32,
        a: *const f32,
        lda: i32,
        stride_a: i64,
        s: *mut f32,
        stride_s: i64,
        u: *mut f32,
        ldu: i32,
        stride_u: i64,
        v: *mut f32,
        ldv: i32,
        stride_v: i64,
        work: *mut f32,
        lwork: i32,
        info: *mut i32,
        h_r_nrm_f: *mut f64,
        batch_size: i32,
    ) -> i32;

    /// `cusolverDnDgesvdaStridedBatched_bufferSize`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDgesvdaStridedBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: i32,
        rank: i32,
        m: i32,
        n: i32,
        a: *const f64,
        lda: i32,
        stride_a: i64,
        s: *const f64,
        stride_s: i64,
        u: *const f64,
        ldu: i32,
        stride_u: i64,
        v: *const f64,
        ldv: i32,
        stride_v: i64,
        lwork: *mut i32,
        batch_size: i32,
    ) -> i32;

    /// `cusolverDnDgesvdaStridedBatched`. f64 analogue.
    ///
    /// # Safety
    /// Same as the f32 variant with `f64` storage.
    pub fn cusolverDnDgesvdaStridedBatched(
        handle: cusolverDnHandle_t,
        jobz: i32,
        rank: i32,
        m: i32,
        n: i32,
        a: *const f64,
        lda: i32,
        stride_a: i64,
        s: *mut f64,
        stride_s: i64,
        u: *mut f64,
        ldu: i32,
        stride_u: i64,
        v: *mut f64,
        ldv: i32,
        stride_v: i64,
        work: *mut f64,
        lwork: i32,
        info: *mut i32,
        h_r_nrm_f: *mut f64,
        batch_size: i32,
    ) -> i32;
}

/// `CUSOLVER_EIG_MODE_NOVECTOR` — `gesvdjBatched` `jobz` value for
/// computing singular values only (skip U / V).
pub const CUSOLVER_EIG_MODE_NOVECTOR: i32 = 0;

/// `CUSOLVER_EIG_MODE_VECTOR` — `gesvdjBatched` `jobz` value for
/// computing both singular values and singular vectors.
pub const CUSOLVER_EIG_MODE_VECTOR: i32 = 1;

// ============================================================================
// cuFFT — Milestone 6.4 Fast Fourier Transforms
// ============================================================================
//
// Host-API cuFFT bindings for the four canonical PyTorch / JAX 1-D FFT
// ops: FFT (C2C forward), IFFT (C2C inverse), RFFT (R2C forward), IRFFT
// (C2R inverse). Plus single/double precision sibling entry points.
//
// f32 (single) + f64 (double) only — cuFFT's main API does not expose
// f16 / bf16 for native FFTs. Callers needing reduced precision must
// cast on either side. Inverse transforms are *unnormalized* — cuFFT
// returns N · IFFT(FFT(x)); the safe-plan layer multiplies by 1/N after
// each inverse exec to match PyTorch's `norm="backward"` default.
//
// Linkage: `cargo:rustc-link-lib=dylib=cufft` (added in build.rs). On
// Linux this resolves to `libcufft.so`; on Windows to `cufft64_*.dll`
// (loaded from `CUDA_PATH\bin`).
//
// Note: cuFFT handles are **integer IDs**, not pointers. This is unusual
// among CUDA libraries (cuSOLVER / cuBLAS / cuRAND all use opaque
// pointer handles) — we represent the handle as `i32` to match the
// upstream C ABI exactly. A sentinel value of `-1` marks "not yet
// created" at the plan layer.

/// Opaque cuFFT plan handle. Unusually for CUDA libraries this is an
/// **integer ID** (`int`), not a pointer. A value of `-1` is reserved
/// at the safe-plan layer as the "not yet created" sentinel — cuFFT
/// itself returns small non-negative integers for live handles.
#[allow(non_camel_case_types)]
pub type cufftHandle = i32;

/// cuFFT result code type. `CUFFT_SUCCESS = 0`. Any non-zero return is
/// mapped to a negative status at the safe-plan layer for distinct
/// error reporting.
#[allow(non_camel_case_types)]
pub type cufftResult = i32;

/// `CUFFT_SUCCESS` — the only success code.
pub const CUFFT_SUCCESS: i32 = 0;

/// cuFFT plan type: real-to-complex (single precision). Output buffer
/// size is `N/2 + 1` complex cells for an `N`-long real input
/// (Hermitian symmetry).
pub const CUFFT_R2C: i32 = 0x2a;

/// cuFFT plan type: complex-to-real (single precision). Input is
/// `N/2 + 1` complex cells (Hermitian-half), output is `N` real cells.
pub const CUFFT_C2R: i32 = 0x2c;

/// cuFFT plan type: complex-to-complex (single precision). Direction is
/// supplied to `cufftExecC2C`.
pub const CUFFT_C2C: i32 = 0x29;

/// cuFFT plan type: double-precision real-to-complex.
pub const CUFFT_D2Z: i32 = 0x6a;

/// cuFFT plan type: double-precision complex-to-real.
pub const CUFFT_Z2D: i32 = 0x6c;

/// cuFFT plan type: double-precision complex-to-complex.
pub const CUFFT_Z2Z: i32 = 0x69;

/// Forward FFT direction tag for `cufftExecC2C` / `cufftExecZ2Z`.
/// cuFFT's forward transform is unnormalized.
pub const CUFFT_FORWARD: i32 = -1;

/// Inverse FFT direction tag for `cufftExecC2C` / `cufftExecZ2Z`.
/// cuFFT's inverse transform is **also unnormalized** — the safe-plan
/// layer multiplies the output by `1/N` after exec to match PyTorch's
/// `norm="backward"` (forward unnormalized, inverse normalized by N)
/// convention.
pub const CUFFT_INVERSE: i32 = 1;

/// Single-precision complex element layout. Interleaved real/imag
/// pairs — `#[repr(C)]` matches NVIDIA's `cufftComplex` struct exactly
/// (which is itself an alias for `float2` in `<vector_types.h>`). The
/// plan layer pairs this with the [`crate`]-level `Complex32` newtype.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[allow(non_camel_case_types)]
pub struct cufftComplex {
    /// Real component.
    pub x: f32,
    /// Imaginary component.
    pub y: f32,
}

/// Double-precision complex element layout. ABI-compatible with cuFFT's
/// `cufftDoubleComplex` (alias for `double2`).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[allow(non_camel_case_types)]
pub struct cufftDoubleComplex {
    /// Real component.
    pub x: f64,
    /// Imaginary component.
    pub y: f64,
}

unsafe extern "C" {
    // ----- plan lifecycle ------------------------------------------------

    /// `cufftPlan1d(plan, nx, type, batch)`. Allocates a 1-D plan
    /// (single FFT of length `nx`, or `batch` independent FFTs each of
    /// length `nx` laid out contiguously). cuFFT's plan struct owns its
    /// own workspace internally — no caller-supplied workspace is
    /// required for the basic 1-D APIs.
    ///
    /// # Safety
    /// `plan` must point to writable storage for one `cufftHandle`. The
    /// underlying CUDA context must be live.
    pub fn cufftPlan1d(plan: *mut cufftHandle, nx: i32, fft_type: i32, batch: i32) -> i32;

    /// `cufftDestroy(plan)`. Frees the plan's internal workspace.
    ///
    /// # Safety
    /// `plan` must be a valid handle returned by `cufftPlan1d` (or any
    /// other plan-creation entry) that has not been destroyed.
    pub fn cufftDestroy(plan: cufftHandle) -> i32;

    /// `cufftSetStream(plan, stream)`. Binds subsequent exec calls on
    /// this plan to the given CUDA stream. Returns 0 on success.
    ///
    /// # Safety
    /// `plan` must be a live cuFFT handle; `stream` must be a valid
    /// CUDA stream in the current context (or null for the default).
    pub fn cufftSetStream(plan: cufftHandle, stream: *mut c_void) -> i32;

    // ----- exec entry points (single precision) --------------------------

    /// `cufftExecC2C(plan, idata, odata, direction)` — complex-to-
    /// complex single-precision exec. `direction` is `CUFFT_FORWARD`
    /// or `CUFFT_INVERSE`. Inverse is unnormalized.
    ///
    /// # Safety
    /// `plan` live, `idata` / `odata` device pointers to at least
    /// `nx * batch` `cufftComplex` cells each (in-place exec when
    /// `idata == odata` is allowed by cuFFT).
    pub fn cufftExecC2C(
        plan: cufftHandle,
        idata: *mut cufftComplex,
        odata: *mut cufftComplex,
        direction: i32,
    ) -> i32;

    /// `cufftExecR2C(plan, idata, odata)` — real-to-complex single
    /// precision. Input length is `nx`, output length is `nx/2 + 1`
    /// (Hermitian-half).
    ///
    /// # Safety
    /// `plan` live, `idata` to `nx * batch` `float` cells, `odata` to
    /// `(nx/2 + 1) * batch` `cufftComplex` cells.
    pub fn cufftExecR2C(plan: cufftHandle, idata: *mut f32, odata: *mut cufftComplex) -> i32;

    /// `cufftExecC2R(plan, idata, odata)` — complex-to-real single
    /// precision. Input length is `nx/2 + 1`, output length is `nx`.
    /// Unnormalized — caller must scale by `1/nx`.
    ///
    /// # Safety
    /// `plan` live, `idata` to `(nx/2 + 1) * batch` `cufftComplex`,
    /// `odata` to `nx * batch` `float`.
    pub fn cufftExecC2R(plan: cufftHandle, idata: *mut cufftComplex, odata: *mut f32) -> i32;

    // ----- exec entry points (double precision) --------------------------

    /// `cufftExecZ2Z(plan, idata, odata, direction)` — complex-to-
    /// complex double precision. Same semantics as `cufftExecC2C`.
    ///
    /// # Safety
    /// Same as `cufftExecC2C` with `cufftDoubleComplex` cells.
    pub fn cufftExecZ2Z(
        plan: cufftHandle,
        idata: *mut cufftDoubleComplex,
        odata: *mut cufftDoubleComplex,
        direction: i32,
    ) -> i32;

    /// `cufftExecD2Z(plan, idata, odata)` — real-to-complex double
    /// precision. Same semantics as `cufftExecR2C`.
    ///
    /// # Safety
    /// `plan` live, `idata` to `nx * batch` `double` cells, `odata` to
    /// `(nx/2 + 1) * batch` `cufftDoubleComplex` cells.
    pub fn cufftExecD2Z(plan: cufftHandle, idata: *mut f64, odata: *mut cufftDoubleComplex)
        -> i32;

    /// `cufftExecZ2D(plan, idata, odata)` — complex-to-real double
    /// precision. Unnormalized.
    ///
    /// # Safety
    /// `plan` live, `idata` to `(nx/2 + 1) * batch` `cufftDoubleComplex`,
    /// `odata` to `nx * batch` `double`.
    pub fn cufftExecZ2D(plan: cufftHandle, idata: *mut cufftDoubleComplex, odata: *mut f64)
        -> i32;

    // ----- plan lifecycle (multi-dimensional / advanced layout) ----------

    /// `cufftPlanMany(plan, rank, n, inembed, istride, idist,
    ///                onembed, ostride, odist, type, batch)`.
    ///
    /// Allocates a `rank`-D plan covering `batch` independent transforms.
    /// `n` points to a `rank`-element array of per-axis lengths
    /// (`n[0]` is the slowest-varying transform axis, `n[rank-1]` the
    /// fastest). `inembed` / `onembed` describe the stride layout in
    /// memory; passing `core::ptr::null_mut()` for both selects cuFFT's
    /// "tight default layout" — each batched transform occupies a
    /// contiguous block of `n[0] * n[1] * ... * n[rank-1]` elements
    /// (the case the ND wrappers in `baracuda-kernels` use).
    ///
    /// `istride` / `ostride` are element strides between consecutive
    /// elements within a single transform (use `1` for the default
    /// layout). `idist` / `odist` are batch strides — the element
    /// offset from one transform's first element to the next. For the
    /// default layout pass `idist = odist = product(n)` (R2C / C2R
    /// follow cuFFT's Hermitian-half rules — the last-axis extent
    /// halves on the complex side).
    ///
    /// Returns 0 (`CUFFT_SUCCESS`) on success.
    ///
    /// # Safety
    /// `plan` must point to writable storage for one `cufftHandle`.
    /// `n` must point to `rank` `i32` cells; `inembed` / `onembed` may
    /// be null (default layout) or point to `rank` `i32` cells each.
    /// The CUDA context must be live.
    pub fn cufftPlanMany(
        plan: *mut cufftHandle,
        rank: i32,
        n: *mut i32,
        inembed: *mut i32,
        istride: i32,
        idist: i32,
        onembed: *mut i32,
        ostride: i32,
        odist: i32,
        fft_type: i32,
        batch: i32,
    ) -> i32;
}

// ============================================================================
// cuFFT bespoke kernels — fftshift / ifftshift + in-place scale-by-1/N.
// ============================================================================
//
// Two bespoke kernel families used by the cuFFT wrap:
//
// 1. **fftshift / ifftshift** — index permutation along the last axis
//    of a `[batch, n]` tensor. Templated on element width (4 bytes for
//    f32, 8 bytes for f64 / Complex32, 16 bytes for Complex64). cuFFT
//    has no native fftshift — these complete the `torch.fft` family.
//
// 2. **scale_inplace_{c32,c64,f32,f64}** — multiply an in-place buffer
//    by a scalar. Used to apply the `1/N` normalization to inverse
//    transforms (cuFFT returns N · IFFT(x); PyTorch's `norm="backward"`
//    convention wants IFFT(x)).
//
// ABI mirrors the elementwise / random kernel families:
//   0 success, 1 misaligned, 2 invalid problem, 3 unsupported,
//   4 workspace too small, 5 internal launch failure.

unsafe extern "C" {
    /// `fftshift` along the last axis of a `[batch, n]` tensor:
    /// `y[b, i] = x[b, (i + n/2) % n]`. Element-width specialization
    /// (4 bytes per element) — used for `Bool` / `f32` / packed-Bool
    /// shifts; the same kernel re-instantiated at 8 / 16 bytes covers
    /// `f64` / `Complex32` and `Complex64`.
    ///
    /// ABI: `(batch, n, x, y, ws, ws_bytes, stream) -> i32`.
    ///
    /// # Safety
    /// `x` / `y` must each point to at least `batch * n` cells of the
    /// kernel's element width. `stream` must be a live CUDA stream in
    /// the current context.
    pub fn baracuda_kernels_fftshift_4_run(
        batch: i64,
        n: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// 8-byte-element `fftshift` (covers `f64` and `Complex32`).
    ///
    /// # Safety
    /// Same as `baracuda_kernels_fftshift_4_run` with 8-byte cells.
    pub fn baracuda_kernels_fftshift_8_run(
        batch: i64,
        n: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// 16-byte-element `fftshift` (covers `Complex64`).
    ///
    /// # Safety
    /// Same as `baracuda_kernels_fftshift_4_run` with 16-byte cells.
    pub fn baracuda_kernels_fftshift_16_run(
        batch: i64,
        n: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Inverse `fftshift` along the last axis of a `[batch, n]` tensor:
    /// `y[b, i] = x[b, (i + (n + 1) / 2) % n]`. Differs from `fftshift`
    /// only for odd `n`; for even `n` the two are identical (each
    /// permutation is self-inverse). 4-byte cells.
    ///
    /// # Safety
    /// Same as `baracuda_kernels_fftshift_4_run`.
    pub fn baracuda_kernels_ifftshift_4_run(
        batch: i64,
        n: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// 8-byte-element inverse `fftshift`.
    ///
    /// # Safety
    /// Same as `baracuda_kernels_ifftshift_4_run` with 8-byte cells.
    pub fn baracuda_kernels_ifftshift_8_run(
        batch: i64,
        n: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// 16-byte-element inverse `fftshift`.
    ///
    /// # Safety
    /// Same as `baracuda_kernels_ifftshift_4_run` with 16-byte cells.
    pub fn baracuda_kernels_ifftshift_16_run(
        batch: i64,
        n: i32,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// In-place scale of a `cufftComplex` buffer by a real scalar:
    /// `y[i].x *= scale; y[i].y *= scale;`. Applied after `cufftExecC2C`
    /// in the inverse direction to bake in the 1/N normalization
    /// PyTorch expects.
    ///
    /// # Safety
    /// `y` must point to `numel` `cufftComplex` cells; `stream` live.
    pub fn baracuda_kernels_scale_inplace_c32_run(
        numel: i64,
        scale: f32,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// In-place scale of a `cufftDoubleComplex` buffer by a real
    /// scalar. f64 analogue of `baracuda_kernels_scale_inplace_c32_run`.
    ///
    /// # Safety
    /// `y` must point to `numel` `cufftDoubleComplex` cells.
    pub fn baracuda_kernels_scale_inplace_c64_run(
        numel: i64,
        scale: f64,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// In-place scale of a real `f32` buffer. Used to bake the `1/N`
    /// normalization into the output of `cufftExecC2R` (IRFFT).
    ///
    /// # Safety
    /// `y` must point to `numel` `f32` cells.
    pub fn baracuda_kernels_scale_inplace_real_f32_run(
        numel: i64,
        scale: f32,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// In-place scale of a real `f64` buffer. f64 analogue.
    ///
    /// # Safety
    /// `y` must point to `numel` `f64` cells.
    pub fn baracuda_kernels_scale_inplace_real_f64_run(
        numel: i64,
        scale: f64,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// N-D `fftshift` / `ifftshift` — single-pass general-permutation
    /// kernel covering up to rank-8 tensors. The caller passes a per-
    /// axis `shape`, per-axis `shift_amt` (0 for pass-through axes;
    /// `n/2` for fftshift / `n - n/2` for ifftshift on shifted axes),
    /// and per-axis contiguous `stride` (in elements). The same kernel
    /// covers both directions — the direction lives entirely in the
    /// `shift_amt` array.
    ///
    /// 4-byte cell width (covers `f32`).
    ///
    /// ABI: `(total, rank, shape, shift_amt, stride, x, y, ws,
    /// ws_bytes, stream) -> i32`.
    ///
    /// # Safety
    /// `x` / `y` must each point to at least `total` cells of the
    /// kernel's element width. `shape` / `shift_amt` / `stride` must
    /// each point to at least `rank` valid entries (host memory).
    /// `rank <= 8`. `stream` must be a live CUDA stream in the current
    /// context.
    pub fn baracuda_kernels_fftshift_nd_4_run(
        total: i64,
        rank: i32,
        shape: *const i32,
        shift_amt: *const i32,
        stride: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// 8-byte-cell N-D fftshift (covers `f64` and `Complex32`).
    ///
    /// # Safety
    /// Same as `baracuda_kernels_fftshift_nd_4_run` with 8-byte cells.
    pub fn baracuda_kernels_fftshift_nd_8_run(
        total: i64,
        rank: i32,
        shape: *const i32,
        shift_amt: *const i32,
        stride: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// 16-byte-cell N-D fftshift (covers `Complex64`).
    ///
    /// # Safety
    /// Same as `baracuda_kernels_fftshift_nd_4_run` with 16-byte cells.
    pub fn baracuda_kernels_fftshift_nd_16_run(
        total: i64,
        rank: i32,
        shape: *const i32,
        shift_amt: *const i32,
        stride: *const i64,
        x: *const c_void,
        y: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// =============================================================================
// Milestone 6.14 — bespoke batched-`ormqr` + batched-QR dense Q/R
// materialization helpers (linalg family). cuSOLVER's `ormqr` is
// non-batched; this kernel fuses all batch slots into one launch so the
// small-matrix regime (where batched-QR is most useful) is not
// latency-bound. Scope: Side = Left, op ∈ {N, T}, dtype ∈ {f32, f64};
// Right-side + complex variants deferred.
// =============================================================================

unsafe extern "C" {
    /// Batched-`ormqr`, `f32`. Applies the implicit `Q` (or `Q^T`) from a
    /// `BatchedQrPlan` packed output (`A_packed [B, M, K]` column-major
    /// + `tau [B, K]`) to a stack of right-hand-side matrices
    /// `C [B, M, N]` in place. One CUDA block per batch slot. `side` is
    /// fixed to `0` (Left) in the trailblazer; `op` is `0` (N — apply Q)
    /// or `1` (T — apply Q^T). Status: 0 success, 2 invalid problem,
    /// 3 unsupported (e.g. side = Right), 5 internal launch failure.
    ///
    /// # Safety
    /// `a_packed` must point to at least `batch * M * K` `f32` cells
    /// (column-major); `tau` to at least `batch * K`; `c` to at least
    /// `batch * M * N`. `stream` must be live.
    pub fn baracuda_kernels_batched_ormqr_f32_run(
        batch: i32,
        m: i32,
        n: i32,
        k: i32,
        side: i32,
        op: i32,
        a_packed: *const c_void,
        tau: *const c_void,
        c: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Batched-`ormqr`, `f64`. Same contract as the `f32` variant.
    ///
    /// # Safety
    /// Same as the `f32` variant with `f64` storage.
    pub fn baracuda_kernels_batched_ormqr_f64_run(
        batch: i32,
        m: i32,
        n: i32,
        k: i32,
        side: i32,
        op: i32,
        a_packed: *const c_void,
        tau: *const c_void,
        c: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Batched-`unmqr`, `Complex32`. Same shape/contract as the `f32`
    /// variant but with `cuFloatComplex` storage. `op = 2` (C —
    /// conjugate transpose) is supported; `op = 1` (T — plain transpose)
    /// is rejected by the Rust safe layer for complex (mathematically
    /// unusual for Householder).
    ///
    /// # Safety
    /// Pointer sizes counted in `cuFloatComplex` cells; layout otherwise
    /// identical to the `f32` runner.
    pub fn baracuda_kernels_batched_ormqr_complex32_run(
        batch: i32,
        m: i32,
        n: i32,
        k: i32,
        side: i32,
        op: i32,
        a_packed: *const c_void,
        tau: *const c_void,
        c: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Batched-`unmqr`, `Complex64`. Same as the `complex32` variant
    /// with `cuDoubleComplex` storage.
    ///
    /// # Safety
    /// Same as the `complex32` variant with `cuDoubleComplex` storage.
    pub fn baracuda_kernels_batched_ormqr_complex64_run(
        batch: i32,
        m: i32,
        n: i32,
        k: i32,
        side: i32,
        op: i32,
        a_packed: *const c_void,
        tau: *const c_void,
        c: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Materialize dense `R [B, K, N]` from a `geqrf`-packed
    /// `A [B, M, N]` (column-major). `K = min(M, N)`. Cell `R[b, i, j]`
    /// = `A[b, i, j]` if `i ≤ j`, else `0`. One CUDA block per
    /// `(batch_slot, column)`. `f32`.
    ///
    /// # Safety
    /// `a_packed` ≥ `batch * M * N` `f32` cells; `r` ≥ `batch * K * N`.
    pub fn baracuda_kernels_batched_qr_materialize_r_f32_run(
        batch: i32,
        m: i32,
        n: i32,
        k: i32,
        a_packed: *const c_void,
        r: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Materialize dense `R`, `f64` analogue.
    ///
    /// # Safety
    /// Same as the `f32` variant with `f64` storage.
    pub fn baracuda_kernels_batched_qr_materialize_r_f64_run(
        batch: i32,
        m: i32,
        n: i32,
        k: i32,
        a_packed: *const c_void,
        r: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Stage a column-major identity `Q [B, M, M]` (one identity per
    /// batch slot) into a freshly allocated buffer. Caller then chains
    /// `baracuda_kernels_batched_ormqr_*_run` with `op = 0` (N) to
    /// overwrite `Q` in place with the dense Q matrix from the
    /// `geqrf`-packed input. `f32`.
    ///
    /// # Safety
    /// `q` must point to at least `batch * M * M` `f32` cells.
    pub fn baracuda_kernels_batched_qr_materialize_identity_f32_run(
        batch: i32,
        m: i32,
        q: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// Stage identity, `f64` analogue.
    ///
    /// # Safety
    /// Same as the `f32` variant with `f64` storage.
    pub fn baracuda_kernels_batched_qr_materialize_identity_f64_run(
        batch: i32,
        m: i32,
        q: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// =============================================================================
// Milestone 6.17 — WY-blocked batched-`ormqr`. Companion to the GEMV-rates
// kernel above (`baracuda_kernels_batched_ormqr_*_run`). Two bespoke
// kernels (T-build + V-extract) pair with cuBLAS strided-batched GEMM at
// the safe-plan layer to lift the apply step from GEMV-rates to GEMM-
// rates. Scope: Side = Left, op ∈ {N, T}, dtype ∈ {f32, f64}.
// =============================================================================

unsafe extern "C" {
    /// WY block T-build, `f32`. For each `(batch_slot, block_index)`,
    /// builds the `[nb, nb]` upper-triangular block-reflector matrix `T`
    /// such that `H_0 · ... · H_{nb-1} = I - V·T·V^T`. One CUDA block
    /// per `(batch, num_blocks)` cell. Status codes: 0 success,
    /// 2 invalid problem, 5 launch failure.
    ///
    /// # Safety
    /// `a_packed` ≥ `batch * M * K` `f32` cells (column-major); `tau` ≥
    /// `batch * K`; `t_out` ≥ `batch * num_blocks * nb * nb`.
    /// `num_blocks` must satisfy `(K + nb - 1) / nb`.
    pub fn baracuda_kernels_batched_ormqr_wy_build_t_f32_run(
        batch: i32,
        m: i32,
        k: i32,
        nb: i32,
        num_blocks: i32,
        a_packed: *const c_void,
        tau: *const c_void,
        t_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// WY block T-build, `f64` analogue.
    ///
    /// # Safety
    /// Same as the `f32` variant with `f64` storage.
    pub fn baracuda_kernels_batched_ormqr_wy_build_t_f64_run(
        batch: i32,
        m: i32,
        k: i32,
        nb: i32,
        num_blocks: i32,
        a_packed: *const c_void,
        tau: *const c_void,
        t_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// WY V-extraction, `f32`. Materializes the dense `V [B, M, nb]`
    /// panel for one block of reflectors (block_start = `block_start`,
    /// `block_k = min(nb, K - block_start)`) into a contiguous workspace
    /// buffer. Sets the implicit-1 at each reflector's diagonal, copies
    /// the packed-A strict lower below, zeros above the diagonal, and
    /// zeros entire columns past `block_k` (handles the partial-last-
    /// block case).
    ///
    /// # Safety
    /// `a_packed` ≥ `batch * M * K` `f32` cells; `v_out` ≥ `batch * M * nb`.
    pub fn baracuda_kernels_batched_ormqr_wy_extract_v_f32_run(
        batch: i32,
        m: i32,
        k: i32,
        nb: i32,
        block_start: i32,
        block_k: i32,
        a_packed: *const c_void,
        v_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// WY V-extraction, `f64` analogue.
    ///
    /// # Safety
    /// Same as the `f32` variant with `f64` storage.
    pub fn baracuda_kernels_batched_ormqr_wy_extract_v_f64_run(
        batch: i32,
        m: i32,
        k: i32,
        nb: i32,
        block_start: i32,
        block_k: i32,
        a_packed: *const c_void,
        v_out: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}
