//! # baracuda-cutlass-kernels-sys
//!
//! Raw `extern "C"` entry points for compiled CUTLASS template
//! instantiations. **You almost certainly want [`baracuda-cutlass`]
//! instead** — that crate wraps these unsafe calls with typed plans,
//! lifetime-checked device buffers, and a proper Rust API.
//!
//! Functions in this crate take raw `void*` pointers, integer dimensions,
//! and a `cudaStream_t` cast as `*mut c_void`. They are unsafe because:
//!
//! - They dereference the pointer arguments without bounds-checking.
//! - They assume the pointers are valid device addresses.
//! - They assume the workspace pointer (when non-null) points to at least
//!   `workspace_bytes` of writable device memory.
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
//! - `5`: internal CUTLASS error (typically a kernel launch failure).
//!
//! [`baracuda-cutlass`]: https://docs.rs/baracuda-cutlass

#![no_std]

use core::ffi::c_void;

// ============================================================================
// GEMM — RCR layout, sm_80 instantiation
// ============================================================================
//
// Layout convention `RCR`:
//   A: row-major    [M, K], leading dimension `lda`
//   B: column-major [K, N], leading dimension `ldb`
//   C: row-major    [M, N], leading dimension `ldc` (optional; pass null
//                                                    + beta = 0 to skip)
//   D: row-major    [M, N], leading dimension `ldd` (always written)
//
// Accumulator and alpha/beta scalars are FP32. Identity epilogue only
// (`D = alpha * AB + beta * C`). The Bias epilogue lands in a follow-up
// once a `LinearCombinationBias` template instantiation is added; until
// then there is no `bias` argument and the safe layer's `EpilogueKind`
// enum has no `Bias` variant.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    /// `f16` GEMM, RCR layout, sm_80.
    ///
    /// # Safety
    /// All pointer args must be device-resident (or null where allowed) and
    /// remain valid for the duration of the launch. `stream` must be a live
    /// CUDA stream in the current context.
    pub fn baracuda_cutlass_gemm_f16_rcr_sm80_run(
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

    /// Workspace size in bytes for `f16` RCR sm_80 GEMM at the given problem size.
    pub fn baracuda_cutlass_gemm_f16_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// Pre-launch implementability check for `f16` RCR sm_80.
    ///
    /// Returns `0` when the kernel can launch with the given shape, leading
    /// dimensions, and pointer alignments; non-zero with the standard
    /// status-code mapping otherwise. Does not launch a kernel and does
    /// not require a stream.
    ///
    /// # Safety
    /// Same pointer-validity contract as [`baracuda_cutlass_gemm_f16_rcr_sm80_run`],
    /// but no device dereferences occur — only host-side checks of pointer
    /// alignment and the leading-dimension fields.
    pub fn baracuda_cutlass_gemm_f16_rcr_sm80_can_implement(
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
    ) -> i32;

    /// `bf16` GEMM, RCR layout, sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_run`].
    pub fn baracuda_cutlass_gemm_bf16_rcr_sm80_run(
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

    /// Workspace size in bytes for `bf16` RCR sm_80 GEMM at the given problem size.
    pub fn baracuda_cutlass_gemm_bf16_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// Pre-launch implementability check for `bf16` RCR sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bf16_rcr_sm80_can_implement(
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
    ) -> i32;
}

// ============================================================================
// GEMM — bias-fused (with optional activation), RCR layout, sm_80
// ============================================================================
//
// Computes `D = activation(alpha*AB + beta*C + bias_broadcast(N))` in a
// single kernel pass via `cutlass::gemm::device::GemmUniversalWithBroadcast`
// + `LinearCombinationBiasElementwise`. The bias vector has length `N`
// (one element per output column) and is broadcast across rows. Layout
// matches the standard RCR variant (A row-major, B column-major, C/D
// row-major).
//
// Symbol naming: `..._gemm_<flavor>_<dtype>_rcr_sm80_<op>` where
//   flavor ∈ {bias, bias_relu, bias_gelu, bias_silu}
//   dtype  ∈ {f16, bf16}
//   op     ∈ {run, workspace_size, can_implement}
// = 24 entry points. The `bias` flavor uses Identity activation; the
// others fuse the named CUTLASS activation functor into the same
// epilogue pass (no extra memory traffic vs plain bias).

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    /// `f16` bias-fused GEMM, RCR layout, sm_80.
    ///
    /// # Safety
    /// All pointer args must be device-resident. `bias` must be a
    /// device-resident length-`n` vector. See
    /// [`baracuda_cutlass_gemm_f16_rcr_sm80_run`] for the rest.
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_f16_rcr_sm80_run(
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
        bias: *const c_void,
        alpha: f32,
        beta: f32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace bytes needed by the `f16` bias-fused RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_bias_f16_rcr_sm80_workspace_size(
        m: i32,
        n: i32,
        k: i32,
    ) -> usize;

    /// Pre-launch implementability check for `f16` bias RCR sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_f16_rcr_sm80_can_implement(
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
        bias: *const c_void,
    ) -> i32;

    /// `bf16` bias-fused GEMM, RCR layout, sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_bf16_rcr_sm80_run(
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
        bias: *const c_void,
        alpha: f32,
        beta: f32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace bytes needed by the `bf16` bias-fused RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_bias_bf16_rcr_sm80_workspace_size(
        m: i32,
        n: i32,
        k: i32,
    ) -> usize;

    /// Pre-launch implementability check for `bf16` bias RCR sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_bf16_rcr_sm80_can_implement(
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
        bias: *const c_void,
    ) -> i32;

    // ---- bias + ReLU activation ---------------------------------------

    /// `f16` bias + ReLU activation GEMM, RCR layout, sm_80.
    /// Computes `D = max(alpha*AB + beta*C + bias_broadcast(N), 0)`.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_relu_f16_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace bytes for `f16` bias+ReLU RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_bias_relu_f16_rcr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// Pre-launch check for `f16` bias+ReLU RCR sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_relu_f16_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// `bf16` bias + ReLU activation GEMM, RCR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_relu_bf16_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace bytes for `bf16` bias+ReLU RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_bias_relu_bf16_rcr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// Pre-launch check for `bf16` bias+ReLU RCR sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_relu_bf16_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- bias + GELU activation (exact, erf-based) ---------------------

    /// `f16` bias + GELU activation GEMM, RCR layout, sm_80.
    /// Computes `D = gelu(alpha*AB + beta*C + bias_broadcast(N))` using
    /// the exact (erf-based) GELU formulation, matching PyTorch's
    /// default `nn.GELU()`.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_gelu_f16_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace bytes for `f16` bias+GELU RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_bias_gelu_f16_rcr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// Pre-launch check for `f16` bias+GELU RCR sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_gelu_f16_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// `bf16` bias + GELU activation GEMM, RCR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_gelu_bf16_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace bytes for `bf16` bias+GELU RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_bias_gelu_bf16_rcr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// Pre-launch check for `bf16` bias+GELU RCR sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_gelu_bf16_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- bias + SiLU activation (x * sigmoid(x)) -----------------------

    /// `f16` bias + SiLU activation GEMM, RCR layout, sm_80.
    /// Computes `D = silu(alpha*AB + beta*C + bias_broadcast(N))` where
    /// `silu(x) = x * sigmoid(x)`. Also known as Swish.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_silu_f16_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace bytes for `f16` bias+SiLU RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_bias_silu_f16_rcr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// Pre-launch check for `f16` bias+SiLU RCR sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_silu_f16_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// `bf16` bias + SiLU activation GEMM, RCR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_silu_bf16_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace bytes for `bf16` bias+SiLU RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_bias_silu_bf16_rcr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// Pre-launch check for `bf16` bias+SiLU RCR sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_silu_bf16_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;
}

// ============================================================================
// GEMM — RRR layout, sm_80 instantiation
// ============================================================================
//
// Layout convention `RRR`:
//   A: row-major [M, K], leading dimension `lda`
//   B: row-major [K, N], leading dimension `ldb`
//   C: row-major [M, N], leading dimension `ldc` (optional; null + beta = 0)
//   D: row-major [M, N], leading dimension `ldd`
//
// Same accumulator (FP32), epilogue (Identity), and status-code mapping as
// the RCR variant. This is the natural shape for activations stored
// row-major and weights stored row-major (no transpose copy).

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    /// `f16` GEMM, RRR layout, sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_run`].
    pub fn baracuda_cutlass_gemm_f16_rrr_sm80_run(
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

    /// Workspace size in bytes for `f16` RRR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_f16_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// Pre-launch implementability check for `f16` RRR sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_f16_rrr_sm80_can_implement(
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
    ) -> i32;

    /// `bf16` GEMM, RRR layout, sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_run`].
    pub fn baracuda_cutlass_gemm_bf16_rrr_sm80_run(
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

    /// Workspace size in bytes for `bf16` RRR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_bf16_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// Pre-launch implementability check for `bf16` RRR sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bf16_rrr_sm80_can_implement(
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
    ) -> i32;
}

// ============================================================================
// GEMM — bias-fused (with optional activation), RRR layout, sm_80
// ============================================================================
//
// Mirror of the RCR bias family but with `B` row-major rather than
// column-major. Computes
// `D = activation(alpha*AB + beta*C + bias_broadcast(N))` in a single
// fused kernel pass. Symbol naming mirrors the RCR set, with `_rrr_`
// in place of `_rcr_`. 24 entry points total (4 flavors × 2 dtypes ×
// 3 ops).

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    // ---- plain bias (Identity activation) -------------------------------

    /// `f16` bias-fused GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_f16_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_f16_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_f16_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_f16_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// `bf16` bias-fused GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_bf16_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_bf16_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_bf16_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_bf16_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- bias + ReLU activation ---------------------------------------

    /// `f16` bias+ReLU GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_relu_f16_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_relu_f16_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_f16_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_relu_f16_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// `bf16` bias+ReLU GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_relu_bf16_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_relu_bf16_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_bf16_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_relu_bf16_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- bias + GELU activation (exact, erf-based) --------------------

    /// `f16` bias+GELU GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_gelu_f16_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_gelu_f16_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_f16_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_gelu_f16_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// `bf16` bias+GELU GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_gelu_bf16_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_gelu_bf16_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_bf16_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_gelu_bf16_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- bias + SiLU activation ---------------------------------------

    /// `f16` bias+SiLU GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_silu_f16_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_silu_f16_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_f16_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_silu_f16_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// `bf16` bias+SiLU GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_silu_bf16_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_silu_bf16_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_bf16_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_silu_bf16_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;
}

// ============================================================================
// GEMM — TF32 (f32 input via TF32 tensor cores), RCR layout, sm_80
// ============================================================================
//
// Inputs are IEEE 754 binary32 stored in device memory. The math
// instruction reduces inputs to TF32 (10-bit mantissa, 8-bit exponent)
// and accumulates into FP32. Faster than full-F32 SIMT GEMM at the cost
// of ~10-bit math precision — analogous to cuBLAS's
// `CUBLAS_COMPUTE_32F_FAST_TF32`.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    /// `f32` GEMM via TF32 tensor cores, RCR layout, sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_run`].
    pub fn baracuda_cutlass_gemm_tf32_rcr_sm80_run(
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

    /// Workspace size in bytes for the `tf32` RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_tf32_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// Pre-launch implementability check for `tf32` RCR sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_tf32_rcr_sm80_can_implement(
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
    ) -> i32;

    /// `f32` GEMM via TF32 tensor cores, RRR layout, sm_80.
    ///
    /// Same numerical behavior as the RCR TF32 kernel but with `B` row-major.
    /// The natural shape for f32 activations × f32 weights when both tensors
    /// are stored row-major (no transpose pass before launch).
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_run`].
    pub fn baracuda_cutlass_gemm_tf32_rrr_sm80_run(
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

    /// Workspace size in bytes for the `tf32` RRR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_tf32_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// Pre-launch implementability check for `tf32` RRR sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_tf32_rrr_sm80_can_implement(
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
    ) -> i32;
}

// ============================================================================
// GEMM — bias-fused (with optional activation), TF32 path, RCR layout, sm_80
// ============================================================================
//
// f32 inputs reduced through Ampere TF32 tensor cores, with bias and
// optional activation fused into the epilogue. Mirrors the f16/bf16
// bias family but uses the TF32 tile shape (4 elements per access).
// 12 entry points total (4 flavors × 3 ops; single element type since
// TF32 implies f32 storage).

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    // ---- plain bias (Identity activation) ----

    /// `f32` (TF32) bias-fused GEMM, RCR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_tf32_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_tf32_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_tf32_rcr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_tf32_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- bias + ReLU activation ----

    /// `f32` (TF32) bias+ReLU GEMM, RCR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_relu_tf32_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_relu_tf32_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_tf32_rcr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_relu_tf32_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- bias + GELU activation (exact, erf-based) ----

    /// `f32` (TF32) bias+GELU GEMM, RCR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_gelu_tf32_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_gelu_tf32_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_tf32_rcr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_gelu_tf32_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- bias + SiLU activation ----

    /// `f32` (TF32) bias+SiLU GEMM, RCR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_silu_tf32_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_silu_tf32_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_tf32_rcr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_silu_tf32_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;
}

// ============================================================================
// GEMM — bias-fused (with optional activation), TF32 path, RRR layout, sm_80
// ============================================================================
//
// Mirror of the TF32 RCR bias family with `B` row-major rather than
// column-major. Same TF32 tile shape, same epilogue family. 12 entry
// points total (4 flavors × 3 ops; single element type since TF32
// implies f32).

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    // ---- plain bias (Identity activation) ----

    /// `f32` (TF32) bias-fused GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_tf32_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_tf32_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_tf32_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_tf32_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- bias + ReLU activation ----

    /// `f32` (TF32) bias+ReLU GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_relu_tf32_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_relu_tf32_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_tf32_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_relu_tf32_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- bias + GELU activation (exact, erf-based) ----

    /// `f32` (TF32) bias+GELU GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_gelu_tf32_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_gelu_tf32_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_tf32_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_gelu_tf32_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- bias + SiLU activation ----

    /// `f32` (TF32) bias+SiLU GEMM, RRR layout, sm_80.
    /// # Safety
    /// See [`baracuda_cutlass_gemm_bias_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_silu_tf32_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_silu_tf32_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_tf32_rrr_sm80_workspace_size(
        m: i32, n: i32, k: i32,
    ) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_bias_silu_tf32_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;
}

// ============================================================================
// GEMM — f32-SIMT path (CUDA cores, no tensor cores), RCR + RRR, sm_80
// ============================================================================
//
// Strict-precision counterpart to the TF32 family. f32 inputs are
// multiplied through the SIMT mainloop (full IEEE 754 binary32 FMA, no
// tensor-core warp-reduction nondeterminism) and accumulated into f32.
// Identical layout conventions to the f16/bf16 kernels.
//
// Bias variants use a vendored partial specialization of
// `cutlass::gemm::kernel::DefaultGemmWithBroadcast` for `OpClassSimt`
// (see `kernels/include/baracuda_simt_broadcast_epilogue.h`) so that
// `GemmUniversalWithBroadcast` can route through the SIMT broadcast
// epilogue path that CUTLASS ships but doesn't wire by default.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    /// `f32` GEMM via SIMT (CUDA cores), RCR layout, sm_80.
    /// Full-precision counterpart to the TF32 RCR kernel.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_run`].
    pub fn baracuda_cutlass_gemm_f32_simt_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace size in bytes for `f32_simt` RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_f32_simt_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_f32_simt_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
    ) -> i32;

    /// `f32` GEMM via SIMT (CUDA cores), RRR layout, sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_run`].
    pub fn baracuda_cutlass_gemm_f32_simt_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace size in bytes for `f32_simt` RRR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_f32_simt_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_f32_simt_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
    ) -> i32;
}

// ============================================================================
// GEMM — bias-fused (with optional activation), f32-SIMT path, sm_80
// ============================================================================
//
// Routes through the vendored `OpClassSimt` partial specialization of
// `DefaultGemmWithBroadcast`. 24 entry points total (4 flavors × 2 layouts ×
// 3 ops). All scalars and the bias vector are `float`.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    // ---- RCR layout ----

    /// CUTLASS GEMM trampoline (launch gemm_bias_f32_simt_rcr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_f32_simt_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_f32_simt_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_f32_simt_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_f32_simt_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_f32_simt_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_relu_f32_simt_rcr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_relu_f32_simt_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_relu_f32_simt_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_f32_simt_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_relu_f32_simt_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_f32_simt_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_gelu_f32_simt_rcr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_gelu_f32_simt_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_gelu_f32_simt_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_f32_simt_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_gelu_f32_simt_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_f32_simt_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_silu_f32_simt_rcr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_silu_f32_simt_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_silu_f32_simt_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_f32_simt_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_silu_f32_simt_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_f32_simt_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- RRR layout ----

    /// CUTLASS GEMM trampoline (launch gemm_bias_f32_simt_rrr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_f32_simt_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_f32_simt_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_f32_simt_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_f32_simt_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_f32_simt_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_relu_f32_simt_rrr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_relu_f32_simt_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_relu_f32_simt_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_f32_simt_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_relu_f32_simt_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_f32_simt_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_gelu_f32_simt_rrr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_gelu_f32_simt_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_gelu_f32_simt_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_f32_simt_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_gelu_f32_simt_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_f32_simt_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_silu_f32_simt_rrr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_silu_f32_simt_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f32, beta: f32,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_silu_f32_simt_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_f32_simt_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_silu_f32_simt_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_f32_simt_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;
}

// ============================================================================
// GEMM — f64 (DGEMM via Ampere FP64 tensor cores), RCR + RRR, sm_80
// ============================================================================
//
// Full IEEE 754 binary64 throughout: inputs, accumulator, alpha/beta, and
// output. Routes through the Ampere DGEMM mma instruction (`m8n8k4` in
// double). Analogous to cuBLAS's `CUBLAS_COMPUTE_64F`.
//
// Note the FFI signature difference: `alpha` and `beta` are `f64` (vs
// `f32` for every other shipped element type). Bias-family kernels
// follow the same f64-scalar convention. The plan layer routes through
// these symbols when `T::Scalar::IS_F64` is true.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    /// `f64` GEMM via Ampere FP64 tensor cores, RCR layout, sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_run`].
    pub fn baracuda_cutlass_gemm_f64_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        alpha: f64, beta: f64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_f64_rcr_sm80).
    pub fn baracuda_cutlass_gemm_f64_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_f64_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
    ) -> i32;

    /// `f64` GEMM via Ampere FP64 tensor cores, RRR layout, sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_run`].
    pub fn baracuda_cutlass_gemm_f64_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        alpha: f64, beta: f64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_f64_rrr_sm80).
    pub fn baracuda_cutlass_gemm_f64_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_f64_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
    ) -> i32;
}

// ============================================================================
// GEMM — bias-fused (with optional activation), f64 (DGEMM), sm_80
// ============================================================================
//
// 24 entry points (4 flavors × 2 layouts × 3 ops). All scalars and the
// bias vector are `double` / `f64`.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    // ---- RCR layout ----

    /// CUTLASS GEMM trampoline (launch gemm_bias_f64_rcr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_f64_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f64, beta: f64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_f64_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_f64_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_f64_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_f64_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_relu_f64_rcr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_relu_f64_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f64, beta: f64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_relu_f64_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_f64_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_relu_f64_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_f64_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_gelu_f64_rcr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_gelu_f64_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f64, beta: f64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_gelu_f64_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_f64_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_gelu_f64_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_f64_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_silu_f64_rcr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_silu_f64_rcr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f64, beta: f64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_silu_f64_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_f64_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_silu_f64_rcr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_f64_rcr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    // ---- RRR layout ----

    /// CUTLASS GEMM trampoline (launch gemm_bias_f64_rrr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_f64_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f64, beta: f64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_f64_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_f64_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_f64_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_f64_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_relu_f64_rrr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_relu_f64_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f64, beta: f64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_relu_f64_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_f64_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_relu_f64_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_relu_f64_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_gelu_f64_rrr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_gelu_f64_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f64, beta: f64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_gelu_f64_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_f64_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_gelu_f64_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_gelu_f64_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;

    /// CUTLASS GEMM trampoline (launch gemm_bias_silu_f64_rrr_sm80).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_bias_silu_f64_rrr_sm80_run(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
        alpha: f64, beta: f64,
        workspace: *mut c_void, workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    /// CUTLASS GEMM trampoline (workspace-bytes query for gemm_bias_silu_f64_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_f64_rrr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;
    /// CUTLASS GEMM trampoline (implementability check for gemm_bias_silu_f64_rrr_sm80).
    pub fn baracuda_cutlass_gemm_bias_silu_f64_rrr_sm80_can_implement(
        m: i32, n: i32, k: i32,
        a: *const c_void, lda: i64,
        b: *const c_void, ldb: i64,
        c: *const c_void, ldc: i64,
        d: *mut c_void, ldd: i64,
        bias: *const c_void,
    ) -> i32;
}

// ============================================================================
// Batched GEMM — uniform-shape, RCR layout, sm_80 instantiation
// ============================================================================
//
// All batches share the same (M, N, K). Per-tensor `stride_*` (in
// elements, not bytes) gives the offset between batch slabs. Layout
// matches the single-GEMM RCR case. This is the natural fit for
// equal-batch attention / repeated linear layers; for variable-shape
// per-group problems use the grouped-GEMM API.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    /// `f16` batched GEMM, RCR layout, sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_run`]. Each batch's
    /// operand pointers are derived from base + `i * stride_*`; all
    /// derived addresses must be device-resident in the current context.
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_batched_f16_rcr_sm80_run(
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        stride_a: i64,
        b: *const c_void,
        ldb: i64,
        stride_b: i64,
        c: *const c_void,
        ldc: i64,
        stride_c: i64,
        d: *mut c_void,
        ldd: i64,
        stride_d: i64,
        alpha: f32,
        beta: f32,
        batch_count: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace bytes needed by the `f16` batched RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_batched_f16_rcr_sm80_workspace_size(
        m: i32,
        n: i32,
        k: i32,
        batch_count: i32,
    ) -> usize;

    /// Pre-launch implementability check for `f16` batched RCR sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_batched_f16_rcr_sm80_can_implement(
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        stride_a: i64,
        b: *const c_void,
        ldb: i64,
        stride_b: i64,
        c: *const c_void,
        ldc: i64,
        stride_c: i64,
        d: *mut c_void,
        ldd: i64,
        stride_d: i64,
        batch_count: i32,
    ) -> i32;

    /// `bf16` batched GEMM, RCR layout, sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_batched_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_batched_bf16_rcr_sm80_run(
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        stride_a: i64,
        b: *const c_void,
        ldb: i64,
        stride_b: i64,
        c: *const c_void,
        ldc: i64,
        stride_c: i64,
        d: *mut c_void,
        ldd: i64,
        stride_d: i64,
        alpha: f32,
        beta: f32,
        batch_count: i32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// Workspace bytes needed by the `bf16` batched RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_batched_bf16_rcr_sm80_workspace_size(
        m: i32,
        n: i32,
        k: i32,
        batch_count: i32,
    ) -> usize;

    /// Pre-launch implementability check for `bf16` batched RCR sm_80.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_f16_rcr_sm80_can_implement`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_gemm_batched_bf16_rcr_sm80_can_implement(
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        lda: i64,
        stride_a: i64,
        b: *const c_void,
        ldb: i64,
        stride_b: i64,
        c: *const c_void,
        ldc: i64,
        stride_c: i64,
        d: *mut c_void,
        ldd: i64,
        stride_d: i64,
        batch_count: i32,
    ) -> i32;
}

// ============================================================================
// Grouped GEMM — RCR layout, sm_80 instantiation
// ============================================================================
//
// Per-group layout matches the single-GEMM `RCR` case. The safe Rust layer
// (`baracuda-cutlass`) packs per-group `problem_sizes`, pointer arrays,
// and leading-dimension arrays into a caller-supplied workspace, then
// hands us pointers to those packed regions. The CUTLASS internal scratch
// (size from `*_scratch_bytes`) lives at the tail of the same workspace.
//
// `h_problem_sizes` is a HOST pointer to the same `[GemmCoord; G]` data
// that's also packed into device memory at `d_problem_sizes` — CUTLASS
// uses the host copy for `sufficient` / tile-count math.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    /// Compute the number of threadblocks to launch for an `f16` grouped
    /// GEMM with the given per-group `(M, N, K)` shapes. CUTLASS chooses
    /// based on device SM count vs total tile count.
    ///
    /// # Safety
    /// `h_m`, `h_n`, `h_k` must each be valid pointers to at least
    /// `group_count` `i32`s of host memory.
    pub fn baracuda_cutlass_grouped_gemm_f16_rcr_sm80_sufficient(
        h_m: *const i32,
        h_n: *const i32,
        h_k: *const i32,
        group_count: i32,
    ) -> i32;

    /// CUTLASS-internal scratch bytes needed for the launch.
    ///
    /// # Safety
    /// Same as [`baracuda_cutlass_grouped_gemm_f16_rcr_sm80_sufficient`].
    pub fn baracuda_cutlass_grouped_gemm_f16_rcr_sm80_scratch_bytes(
        h_m: *const i32,
        h_n: *const i32,
        h_k: *const i32,
        group_count: i32,
        threadblock_count: i32,
    ) -> usize;

    /// Pre-launch implementability check (host-only, no CUDA traffic).
    ///
    /// # Safety
    /// Same as [`baracuda_cutlass_grouped_gemm_f16_rcr_sm80_sufficient`].
    pub fn baracuda_cutlass_grouped_gemm_f16_rcr_sm80_can_implement(
        h_m: *const i32,
        h_n: *const i32,
        h_k: *const i32,
        group_count: i32,
    ) -> i32;

    /// Launch the grouped GEMM.
    ///
    /// # Safety
    /// All `d_*` pointers must be device-resident, in the current context,
    /// and remain valid for the duration of the launch. `h_problem_sizes`
    /// must be a host pointer to a `[GemmCoord; group_count]` array (same
    /// data as `d_problem_sizes`). `scratch` must be at least
    /// `scratch_bytes` bytes of writable device memory. `stream` must be a
    /// live CUDA stream.
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_grouped_gemm_f16_rcr_sm80_run(
        group_count: i32,
        threadblock_count: i32,
        d_problem_sizes: *const c_void,
        d_ptr_a: *const c_void,
        d_ptr_b: *const c_void,
        d_ptr_c: *const c_void,
        d_ptr_d: *mut c_void,
        d_lda: *const c_void,
        d_ldb: *const c_void,
        d_ldc: *const c_void,
        d_ldd: *const c_void,
        h_problem_sizes: *const c_void,
        alpha: f32,
        beta: f32,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    /// `bf16` grouped GEMM — see f16 counterpart for documentation.
    ///
    /// # Safety
    /// Same contract as [`baracuda_cutlass_grouped_gemm_f16_rcr_sm80_sufficient`].
    pub fn baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_sufficient(
        h_m: *const i32,
        h_n: *const i32,
        h_k: *const i32,
        group_count: i32,
    ) -> i32;

    /// # Safety
    /// Same as [`baracuda_cutlass_grouped_gemm_f16_rcr_sm80_scratch_bytes`].
    pub fn baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_scratch_bytes(
        h_m: *const i32,
        h_n: *const i32,
        h_k: *const i32,
        group_count: i32,
        threadblock_count: i32,
    ) -> usize;

    /// # Safety
    /// Same as [`baracuda_cutlass_grouped_gemm_f16_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_can_implement(
        h_m: *const i32,
        h_n: *const i32,
        h_k: *const i32,
        group_count: i32,
    ) -> i32;

    /// # Safety
    /// Same as [`baracuda_cutlass_grouped_gemm_f16_rcr_sm80_run`].
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_cutlass_grouped_gemm_bf16_rcr_sm80_run(
        group_count: i32,
        threadblock_count: i32,
        d_problem_sizes: *const c_void,
        d_ptr_a: *const c_void,
        d_ptr_b: *const c_void,
        d_ptr_c: *const c_void,
        d_ptr_d: *mut c_void,
        d_lda: *const c_void,
        d_ldb: *const c_void,
        d_ldc: *const c_void,
        d_ldd: *const c_void,
        h_problem_sizes: *const c_void,
        alpha: f32,
        beta: f32,
        scratch: *mut c_void,
        scratch_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

// ============================================================================
// int8 GEMM — RCR layout, sm_80 instantiations (Phase 2)
// ============================================================================
//
// Layout convention `RCR`:
//   A: row-major    [M, K], leading dimension `lda`  (int8 or uint8)
//   B: column-major [K, N], leading dimension `ldb`  (matches A signedness)
//   C: row-major    [M, N], leading dimension `ldc`  (matches A signedness; optional)
//   D: row-major    [M, N], leading dimension `ldd`  (matches A signedness)
//
// Accumulator is int32; alpha/beta are f32 (the standard CUTLASS dequant-
// in-epilogue convention for integer GEMM). The final saturating cast
// from float compute back to int8/uint8 uses the `cvt.rni.sat.{s8,u8}.f32`
// PTX instruction. Operator = `OpMultiplyAddSaturate` — the accumulator
// clamps on overflow rather than wrapping.
//
// Bias-family symbols carry both an activation suffix (`bias`, `bias_relu`,
// `bias_gelu`, `bias_silu`) and a bias-element suffix (`f32bias` or
// `i32bias`) that picks the broadcast-vector dtype. The activation runs
// in float (after int32→float dequant), so GELU/SiLU compose without a
// custom epilogue.
//
// `RRR` (row-major × row-major) is **not** present at int8 — CUTLASS 4.2.0
// has no warp-level `MmaTensorOpMultiplicandTileIterator` specialization
// for `TensorOpMultiplicandCongruous<8, ...>`, so `RowMajor × RowMajor ×
// OpClassTensorOp` cannot be instantiated for 8-bit operands. The safe
// layer reports `Error::Unsupported` for an int8 RRR descriptor.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
unsafe extern "C" {
    // ---------- s8 Identity, RCR, sm_80 ----------

    /// Signed-int8 GEMM, RCR layout, sm_80.
    ///
    /// Computes `D = saturating_cast<i8>(alpha * (A * B) + beta * C)`
    /// with int8 inputs, int32 accumulator, and f32 alpha/beta. `c` may
    /// be null with `beta = 0.0` to skip the source-tensor read.
    ///
    /// # Safety
    /// All pointer args must be device-resident (or null where allowed)
    /// and remain valid for the duration of the launch. `stream` must be
    /// a live CUDA stream in the current context.
    pub fn baracuda_cutlass_gemm_s8_rcr_sm80_run(
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

    /// Workspace size in bytes for the `s8` RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_s8_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// Pre-launch implementability check for the `s8` RCR sm_80 GEMM.
    ///
    /// # Safety
    /// Same pointer-validity contract as the matching `_run`, but only
    /// host-side alignment and leading-dimension checks occur.
    pub fn baracuda_cutlass_gemm_s8_rcr_sm80_can_implement(
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
    ) -> i32;

    // ---------- u8 Identity, RCR, sm_80 ----------

    /// Unsigned-uint8 GEMM, RCR layout, sm_80.
    ///
    /// Same template family as the signed variant; output clamps to
    /// `[0, 255]` via `cvt.rni.sat.u8.f32`.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_s8_rcr_sm80_run`].
    pub fn baracuda_cutlass_gemm_u8_rcr_sm80_run(
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

    /// Workspace size for `u8` RCR sm_80 GEMM.
    pub fn baracuda_cutlass_gemm_u8_rcr_sm80_workspace_size(m: i32, n: i32, k: i32) -> usize;

    /// Pre-launch check for `u8` RCR sm_80 GEMM.
    ///
    /// # Safety
    /// See [`baracuda_cutlass_gemm_s8_rcr_sm80_can_implement`].
    pub fn baracuda_cutlass_gemm_u8_rcr_sm80_can_implement(
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
    ) -> i32;
}

// ---- int8 bias-family FFI decls (Bias + 3 activations × 2 sgn × 2 bias-types) ----
//
// 16 kernel families × 3 ops each = 48 extern decls. All share the same
// three signatures (run / workspace_size / can_implement); only the
// function name differs across families. A local macro_rules!
// generates each triplet to keep this section readable.
//
// Naming: `baracuda_cutlass_gemm_<epi>_<bias-elem>_<sgn>_rcr_sm80_<op>`
//   epi       ∈ {bias, bias_relu, bias_gelu, bias_silu}
//   bias-elem ∈ {f32bias, i32bias}
//   sgn       ∈ {s8, u8}
//   op        ∈ {run, workspace_size, can_implement}

/// Internal: stamps out the (run, workspace_size, can_implement) extern
/// decl triple for one int8 bias-kernel family. Each `$run` / `$ws` /
/// `$ck` is the fully-qualified C symbol name as a Rust identifier.
macro_rules! int8_bias_ffi {
    ($run:ident, $ws:ident, $ck:ident) => {
        unsafe extern "C" {
            #[doc = concat!(
                "int8 bias-fused GEMM with optional fused activation.\n\n",
                "Computes `D = saturating_cast(activation(alpha * (A * B) ",
                "+ beta * C + bias_broadcast(N)))`. See the section header for ",
                "the layout / accumulator / clamp contract.\n\n",
                "# Safety\nSame contract as ",
                "[`baracuda_cutlass_gemm_s8_rcr_sm80_run`]."
            )]
            pub fn $run(
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
                bias: *const c_void,
                alpha: f32,
                beta: f32,
                workspace: *mut c_void,
                workspace_bytes: usize,
                stream: *mut c_void,
            ) -> i32;

            #[doc = "Workspace size in bytes for the corresponding `_run` entry point."]
            pub fn $ws(m: i32, n: i32, k: i32) -> usize;

            #[doc = concat!(
                "Pre-launch implementability check for the corresponding ",
                "`_run` entry point.\n\n# Safety\nSame pointer-validity ",
                "contract as the matching `_run`, but only host-side ",
                "alignment and leading-dimension checks occur."
            )]
            pub fn $ck(
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
                bias: *const c_void,
            ) -> i32;
        }
    };
}

#[cfg(any(feature = "sm80", feature = "sm90a"))]
mod int8_bias_decls {
    use super::c_void;

    // ===== s8 × f32 bias =====
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_f32bias_s8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_f32bias_s8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_f32bias_s8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_relu_f32bias_s8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_relu_f32bias_s8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_relu_f32bias_s8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_gelu_f32bias_s8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_gelu_f32bias_s8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_gelu_f32bias_s8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_silu_f32bias_s8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_silu_f32bias_s8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_silu_f32bias_s8_rcr_sm80_can_implement
    );

    // ===== s8 × i32 bias =====
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_i32bias_s8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_i32bias_s8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_i32bias_s8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_relu_i32bias_s8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_relu_i32bias_s8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_relu_i32bias_s8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_gelu_i32bias_s8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_gelu_i32bias_s8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_gelu_i32bias_s8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_silu_i32bias_s8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_silu_i32bias_s8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_silu_i32bias_s8_rcr_sm80_can_implement
    );

    // ===== u8 × f32 bias =====
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_f32bias_u8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_f32bias_u8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_f32bias_u8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_relu_f32bias_u8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_relu_f32bias_u8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_relu_f32bias_u8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_gelu_f32bias_u8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_gelu_f32bias_u8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_gelu_f32bias_u8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_silu_f32bias_u8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_silu_f32bias_u8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_silu_f32bias_u8_rcr_sm80_can_implement
    );

    // ===== u8 × i32 bias =====
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_i32bias_u8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_i32bias_u8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_i32bias_u8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_relu_i32bias_u8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_relu_i32bias_u8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_relu_i32bias_u8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_gelu_i32bias_u8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_gelu_i32bias_u8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_gelu_i32bias_u8_rcr_sm80_can_implement
    );
    int8_bias_ffi!(
        baracuda_cutlass_gemm_bias_silu_i32bias_u8_rcr_sm80_run,
        baracuda_cutlass_gemm_bias_silu_i32bias_u8_rcr_sm80_workspace_size,
        baracuda_cutlass_gemm_bias_silu_i32bias_u8_rcr_sm80_can_implement
    );
}

#[cfg(any(feature = "sm80", feature = "sm90a"))]
pub use int8_bias_decls::*;

