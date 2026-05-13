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
