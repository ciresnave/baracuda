//! Phase 24 — `baracuda-kernels-sys` C-ABI FFI re-export surface for the
//! CUTLASS-backed GEMM family.
//!
//! Background: closes the Phase 19 design correction (every library-backed
//! Rust plan must also ship a flat C-ABI `baracuda-kernels-sys` FFI
//! symbol) for the Cutlass slice. Phases 22 (cuSOLVER) + 23 (cuFFT /
//! cuRAND) covered the library-backed cuSOLVER/cuFFT/cuRAND plans;
//! Phase 24 closes Cutlass.
//!
//! ## Why a re-export and not a fresh facade
//!
//! Unlike cuSOLVER / cuFFT / cuRAND (which are NVIDIA shared libraries
//! we dlopen at runtime), CUTLASS is a header-only template library — its
//! kernels are pre-compiled into the `baracuda-cutlass-kernels-sys`
//! static archive and already exposed as `extern "C"` entry points named
//! `baracuda_cutlass_gemm_*_run` / `_workspace_size` / `_can_implement`.
//!
//! Phase 24 wraps each of those 162 symbols with a one-line trampoline
//! function under the unified `baracuda_kernels_gemm_*` naming convention,
//! so non-Rust callers (Fuel et al.) can drive every Cutlass GEMM SKU
//! through the same `baracuda_kernels_*` namespace as the bespoke,
//! cuSOLVER, cuFFT, cuRAND, and cuDNN kernels — without needing a
//! separate `baracuda-cutlass-kernels-sys` link-line dep.
//!
//! ## Coverage
//!
//! 54 plan families × 3 entry points = **162 trampoline symbols**:
//!
//! - **Non-bias GEMM, f32 alpha/beta** (44 families): every layout × dtype
//!   combination shipped by Cutlass except f64 (which has f64 scalars).
//!   - `gemm_{f16,bf16,tf32,f32_simt}_{rcr,rrr}_sm80` — 8 families.
//!   - `gemm_s8_rcr_sm80`, `gemm_u8_rcr_sm80` — 2 families (int8 RCR).
//!   - `gemm_bias_{f16,bf16,tf32,f32_simt}_{rcr,rrr}_sm80` — 8 families
//!     (no-activation bias-add variant).
//!   - `gemm_bias_{relu,gelu,silu}_{f16,bf16,tf32,f32_simt}_{rcr,rrr}_sm80`
//!     — 24 families (bias + activation).
//!   - `gemm_bias_{,relu,gelu,silu}_{f32bias,i32bias}_{s8,u8}_rcr_sm80` —
//!     16 families (int8/u8 with f32 or i32 bias broadcast). f32 alpha/beta.
//! - **Non-bias GEMM, f64 alpha/beta** (2 families):
//!   - `gemm_f64_{rcr,rrr}_sm80` — DGEMM via Ampere fp64 tensor cores.
//! - **Bias-fused GEMM, f64 alpha/beta** (8 families):
//!   - `gemm_bias_{,relu,gelu,silu}_f64_{rcr,rrr}_sm80`.
//! - **Batched GEMM, f32 alpha/beta** (2 families):
//!   - `gemm_batched_{f16,bf16}_rcr_sm80` — equal-batch stride-strided.
//!
//! Each family ships three FFI siblings:
//! - `*_run` — launch the GEMM on the given stream.
//! - `*_workspace_size` — host-side query (no stream).
//! - `*_can_implement` — pre-launch validation (no stream, no exec).
//!
//! ## Layout convention recap
//!
//! - `rcr` — A row-major `[M, K]`, B column-major `[K, N]`, C/D row-major.
//! - `rrr` — A row-major `[M, K]`, B row-major `[K, N]`, C/D row-major.
//!
//! Single accumulator/scalar convention: f32 alpha/beta unless the
//! family name carries `f64`, in which case f64 alpha/beta. Bias is
//! always `*const c_void` whose pointee type is implied by the family
//! name (`*_f32bias_*` / `*_i32bias_*` for int8; matches kernel-element
//! type for fp variants).
//!
//! ## Status codes
//!
//! Identical to the upstream CUTLASS FFI — the trampoline simply
//! forwards the return value. Standard convention:
//! - `0` — success.
//! - `1` — misaligned operand.
//! - `2` — invalid problem (non-positive M/N/K, etc.).
//! - `3` — not supported (kernel can't run with this shape/alignment).
//! - `4` — workspace too small or null when required.
//! - `5` — internal CUTLASS / launch failure.
//!
//! ## Workspace contract
//!
//! Same as the upstream — query `*_workspace_size(m, n, k)` (or with
//! `batch_count` for batched), allocate that many bytes, pass to `*_run`.
//! Some Cutlass kernels need zero workspace; the query returns 0 and the
//! `_run` ignores the (`workspace`, `workspace_bytes`) pair. The
//! trampoline does no workspace check of its own — the underlying
//! kernel validates internally.
//!
//! ## Stream / pointer residency
//!
//! Identical to the rest of `baracuda-kernels-sys`: all `a / b / c / d /
//! bias / workspace` are **device** pointers; `stream` is a CUDA
//! stream cast to `*mut c_void`. The `_can_implement` and
//! `_workspace_size` queries do no device work — pointers can be null
//! (validation only checks shape).
//!
//! ## Skip notes (other Phase 24 libraries)
//!
//! Per the Phase 23 cuSPARSE precedent (no Rust plans → no facade):
//! - **cuTENSOR** — `baracuda-cutensor` exists but no plan in
//!   `baracuda-kernels` wraps it (einsum / permute are bespoke or
//!   delegated to other kernels). Skip until a cuTENSOR-backed plan
//!   ships in `baracuda-kernels`.
//! - **NPP** — `baracuda-npp` exists but no plan in `baracuda-kernels`
//!   wraps NPP (the image family is all bespoke or cuDNN-pool-backed).
//!   Skip until a plan lands.
//! - **CV-CUDA** — `baracuda-cvcuda` exists but no plan in
//!   `baracuda-kernels` wraps it. Skip until a plan lands.

#![allow(non_camel_case_types)]
#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;

use baracuda_cutlass_kernels_sys as kk;

// ============================================================================
// Trampoline macro families
// ============================================================================
//
// Each macro emits the three FFI siblings (`_run`, `_workspace_size`,
// `_can_implement`) for one Cutlass GEMM family. The macros take the
// downstream cutlass symbol names as identifiers and the rebranded
// `baracuda_kernels_*` symbol names as the first three identifiers.

/// Trampoline trio for a standard non-bias single GEMM with `f32`
/// alpha/beta. Covers f16/bf16/tf32/f32_simt and s8/u8 (RCR layouts).
macro_rules! gemm_nobias_f32 {
    ($run:ident, $ws:ident, $ci:ident,
     $kk_run:ident, $kk_ws:ident, $kk_ci:ident) => {
        /// Workspace bytes required by this Cutlass GEMM SKU.
        ///
        /// # Safety
        /// Pure host-side query; no device pointers dereferenced.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws(m: i32, n: i32, k: i32) -> usize {
            unsafe { kk::$kk_ws(m, n, k) }
        }

        /// Pre-launch implementability check for this Cutlass GEMM SKU.
        ///
        /// # Safety
        /// No device dereferences — only host-side checks of pointer
        /// alignment and leading dimensions.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ci(
            m: i32, n: i32, k: i32,
            a: *const c_void, lda: i64,
            b: *const c_void, ldb: i64,
            c: *const c_void, ldc: i64,
            d: *mut c_void, ldd: i64,
        ) -> i32 {
            unsafe {
                kk::$kk_ci(m, n, k, a, lda, b, ldb, c, ldc, d, ldd)
            }
        }

        /// Launch this Cutlass GEMM SKU on `stream`.
        ///
        /// # Safety
        /// All `a / b / c / d / workspace` are device pointers (or null
        /// where allowed); `stream` is a live CUDA stream in the current
        /// context. Workspace must be at least `_workspace_size(m, n, k)`
        /// bytes if non-zero.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            m: i32, n: i32, k: i32,
            a: *const c_void, lda: i64,
            b: *const c_void, ldb: i64,
            c: *const c_void, ldc: i64,
            d: *mut c_void, ldd: i64,
            alpha: f32, beta: f32,
            workspace: *mut c_void, workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                kk::$kk_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            }
        }
    };
}

/// Trampoline trio for a non-bias single GEMM with `f64` alpha/beta
/// (DGEMM via Ampere fp64 tensor cores).
macro_rules! gemm_nobias_f64 {
    ($run:ident, $ws:ident, $ci:ident,
     $kk_run:ident, $kk_ws:ident, $kk_ci:ident) => {
        /// Workspace bytes required.
        ///
        /// # Safety
        /// Pure host-side query.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws(m: i32, n: i32, k: i32) -> usize {
            unsafe { kk::$kk_ws(m, n, k) }
        }

        /// Pre-launch implementability check.
        ///
        /// # Safety
        /// No device dereferences.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ci(
            m: i32, n: i32, k: i32,
            a: *const c_void, lda: i64,
            b: *const c_void, ldb: i64,
            c: *const c_void, ldc: i64,
            d: *mut c_void, ldd: i64,
        ) -> i32 {
            unsafe {
                kk::$kk_ci(m, n, k, a, lda, b, ldb, c, ldc, d, ldd)
            }
        }

        /// Launch DGEMM. f64 alpha/beta.
        ///
        /// # Safety
        /// See the `gemm_nobias_f32` trampoline family.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            m: i32, n: i32, k: i32,
            a: *const c_void, lda: i64,
            b: *const c_void, ldb: i64,
            c: *const c_void, ldc: i64,
            d: *mut c_void, ldd: i64,
            alpha: f64, beta: f64,
            workspace: *mut c_void, workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                kk::$kk_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    alpha, beta, workspace, workspace_bytes, stream,
                )
            }
        }
    };
}

/// Trampoline trio for a bias-fused single GEMM with `f32` alpha/beta.
/// Covers Bias / BiasRelu / BiasGelu / BiasSilu for f16/bf16/tf32/f32_simt
/// and the int8 SKUs with f32-or-i32 bias broadcast.
macro_rules! gemm_bias_f32 {
    ($run:ident, $ws:ident, $ci:ident,
     $kk_run:ident, $kk_ws:ident, $kk_ci:ident) => {
        /// Workspace bytes required.
        ///
        /// # Safety
        /// Pure host-side query.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws(m: i32, n: i32, k: i32) -> usize {
            unsafe { kk::$kk_ws(m, n, k) }
        }

        /// Pre-launch implementability check (bias variant).
        ///
        /// # Safety
        /// No device dereferences.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ci(
            m: i32, n: i32, k: i32,
            a: *const c_void, lda: i64,
            b: *const c_void, ldb: i64,
            c: *const c_void, ldc: i64,
            d: *mut c_void, ldd: i64,
            bias: *const c_void,
        ) -> i32 {
            unsafe {
                kk::$kk_ci(m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias)
            }
        }

        /// Launch bias-fused Cutlass GEMM on `stream`. `bias` is an
        /// `[N]` device vector broadcast across rows of D.
        ///
        /// # Safety
        /// See the `gemm_nobias_f32` trampoline family. `bias` is non-null and
        /// device-resident.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            m: i32, n: i32, k: i32,
            a: *const c_void, lda: i64,
            b: *const c_void, ldb: i64,
            c: *const c_void, ldc: i64,
            d: *mut c_void, ldd: i64,
            bias: *const c_void,
            alpha: f32, beta: f32,
            workspace: *mut c_void, workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                kk::$kk_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            }
        }
    };
}

/// Trampoline trio for a bias-fused single GEMM with `f64` alpha/beta
/// (DGEMM bias family).
macro_rules! gemm_bias_f64 {
    ($run:ident, $ws:ident, $ci:ident,
     $kk_run:ident, $kk_ws:ident, $kk_ci:ident) => {
        /// Workspace bytes required.
        ///
        /// # Safety
        /// Pure host-side query.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws(m: i32, n: i32, k: i32) -> usize {
            unsafe { kk::$kk_ws(m, n, k) }
        }

        /// Pre-launch implementability check (DGEMM bias variant).
        ///
        /// # Safety
        /// No device dereferences.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ci(
            m: i32, n: i32, k: i32,
            a: *const c_void, lda: i64,
            b: *const c_void, ldb: i64,
            c: *const c_void, ldc: i64,
            d: *mut c_void, ldd: i64,
            bias: *const c_void,
        ) -> i32 {
            unsafe {
                kk::$kk_ci(m, n, k, a, lda, b, ldb, c, ldc, d, ldd, bias)
            }
        }

        /// Launch bias-fused DGEMM. f64 alpha/beta + f64 bias vector.
        ///
        /// # Safety
        /// See the `gemm_bias_f32` trampoline family.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            m: i32, n: i32, k: i32,
            a: *const c_void, lda: i64,
            b: *const c_void, ldb: i64,
            c: *const c_void, ldc: i64,
            d: *mut c_void, ldd: i64,
            bias: *const c_void,
            alpha: f64, beta: f64,
            workspace: *mut c_void, workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                kk::$kk_run(
                    m, n, k, a, lda, b, ldb, c, ldc, d, ldd,
                    bias, alpha, beta, workspace, workspace_bytes, stream,
                )
            }
        }
    };
}

/// Trampoline trio for a strided-batched GEMM with `f32` alpha/beta.
/// Adds `stride_{a,b,c,d}: i64` and `batch_count: i32` versus the
/// single-GEMM signature.
macro_rules! gemm_batched_f32 {
    ($run:ident, $ws:ident, $ci:ident,
     $kk_run:ident, $kk_ws:ident, $kk_ci:ident) => {
        /// Workspace bytes required for a `batch_count`-deep batched
        /// launch.
        ///
        /// # Safety
        /// Pure host-side query.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ws(
            m: i32, n: i32, k: i32, batch_count: i32,
        ) -> usize {
            unsafe { kk::$kk_ws(m, n, k, batch_count) }
        }

        /// Pre-launch implementability check (batched).
        ///
        /// # Safety
        /// No device dereferences.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $ci(
            m: i32, n: i32, k: i32,
            a: *const c_void, lda: i64, stride_a: i64,
            b: *const c_void, ldb: i64, stride_b: i64,
            c: *const c_void, ldc: i64, stride_c: i64,
            d: *mut c_void, ldd: i64, stride_d: i64,
            batch_count: i32,
        ) -> i32 {
            unsafe {
                kk::$kk_ci(
                    m, n, k,
                    a, lda, stride_a, b, ldb, stride_b,
                    c, ldc, stride_c, d, ldd, stride_d,
                    batch_count,
                )
            }
        }

        /// Launch strided-batched Cutlass GEMM. Batch `i` operates on
        /// `A + i * stride_a`, `B + i * stride_b`, etc. (strides in
        /// **elements**, not bytes).
        ///
        /// # Safety
        /// See the `gemm_nobias_f32` trampoline family. Additionally
        /// every derived per-batch pointer must remain device-resident.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $run(
            m: i32, n: i32, k: i32,
            a: *const c_void, lda: i64, stride_a: i64,
            b: *const c_void, ldb: i64, stride_b: i64,
            c: *const c_void, ldc: i64, stride_c: i64,
            d: *mut c_void, ldd: i64, stride_d: i64,
            alpha: f32, beta: f32,
            batch_count: i32,
            workspace: *mut c_void, workspace_bytes: usize,
            stream: *mut c_void,
        ) -> i32 {
            unsafe {
                kk::$kk_run(
                    m, n, k,
                    a, lda, stride_a, b, ldb, stride_b,
                    c, ldc, stride_c, d, ldd, stride_d,
                    alpha, beta, batch_count,
                    workspace, workspace_bytes, stream,
                )
            }
        }
    };
}

// ============================================================================
// Non-bias single GEMM, f32 alpha/beta
// ============================================================================
//
// 10 families × {f16, bf16, tf32, f32_simt} × {rcr, rrr} + {s8, u8} × rcr.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f32!(
    baracuda_kernels_gemm_f16_rcr_sm80_run,
    baracuda_kernels_gemm_f16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_f16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_f16_rcr_sm80_run,
    baracuda_cutlass_gemm_f16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_f16_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f32!(
    baracuda_kernels_gemm_f16_rrr_sm80_run,
    baracuda_kernels_gemm_f16_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_f16_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_f16_rrr_sm80_run,
    baracuda_cutlass_gemm_f16_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_f16_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f32!(
    baracuda_kernels_gemm_bf16_rcr_sm80_run,
    baracuda_kernels_gemm_bf16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bf16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bf16_rcr_sm80_run,
    baracuda_cutlass_gemm_bf16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bf16_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f32!(
    baracuda_kernels_gemm_bf16_rrr_sm80_run,
    baracuda_kernels_gemm_bf16_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bf16_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bf16_rrr_sm80_run,
    baracuda_cutlass_gemm_bf16_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bf16_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f32!(
    baracuda_kernels_gemm_tf32_rcr_sm80_run,
    baracuda_kernels_gemm_tf32_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_tf32_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_tf32_rcr_sm80_run,
    baracuda_cutlass_gemm_tf32_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_tf32_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f32!(
    baracuda_kernels_gemm_tf32_rrr_sm80_run,
    baracuda_kernels_gemm_tf32_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_tf32_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_tf32_rrr_sm80_run,
    baracuda_cutlass_gemm_tf32_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_tf32_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f32!(
    baracuda_kernels_gemm_f32_simt_rcr_sm80_run,
    baracuda_kernels_gemm_f32_simt_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_f32_simt_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_f32_simt_rcr_sm80_run,
    baracuda_cutlass_gemm_f32_simt_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_f32_simt_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f32!(
    baracuda_kernels_gemm_f32_simt_rrr_sm80_run,
    baracuda_kernels_gemm_f32_simt_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_f32_simt_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_f32_simt_rrr_sm80_run,
    baracuda_cutlass_gemm_f32_simt_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_f32_simt_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f32!(
    baracuda_kernels_gemm_s8_rcr_sm80_run,
    baracuda_kernels_gemm_s8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_s8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_s8_rcr_sm80_run,
    baracuda_cutlass_gemm_s8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_s8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f32!(
    baracuda_kernels_gemm_u8_rcr_sm80_run,
    baracuda_kernels_gemm_u8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_u8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_u8_rcr_sm80_run,
    baracuda_cutlass_gemm_u8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_u8_rcr_sm80_can_implement
);

// ============================================================================
// Non-bias single GEMM, f64 alpha/beta (DGEMM)
// ============================================================================

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f64!(
    baracuda_kernels_gemm_f64_rcr_sm80_run,
    baracuda_kernels_gemm_f64_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_f64_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_f64_rcr_sm80_run,
    baracuda_cutlass_gemm_f64_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_f64_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_nobias_f64!(
    baracuda_kernels_gemm_f64_rrr_sm80_run,
    baracuda_kernels_gemm_f64_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_f64_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_f64_rrr_sm80_run,
    baracuda_cutlass_gemm_f64_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_f64_rrr_sm80_can_implement
);

// ============================================================================
// Bias-fused single GEMM, f32 alpha/beta — 4 epilogues × 4 dtypes × 2 layouts
// ============================================================================
//
// Epilogues: Bias / BiasRelu / BiasGelu / BiasSilu.
// Dtypes:    f16 / bf16 / tf32 / f32_simt.
// Layouts:   rcr / rrr.
// Total:     32 families.

// We stay zero-dep (no `paste` crate) and write each macro invocation
// explicitly. The cost is verbosity; the benefit is no extra crate in
// the no_std FFI shim.

// ---------- Bias (no activation) × {f16, bf16, tf32, f32_simt} × {rcr, rrr} -

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_f16_rcr_sm80_run,
    baracuda_kernels_gemm_bias_f16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_f16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_f16_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_f16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_f16_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_f16_rrr_sm80_run,
    baracuda_kernels_gemm_bias_f16_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_f16_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_f16_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_f16_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_f16_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_bf16_rcr_sm80_run,
    baracuda_kernels_gemm_bias_bf16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_bf16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_bf16_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_bf16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_bf16_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_bf16_rrr_sm80_run,
    baracuda_kernels_gemm_bias_bf16_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_bf16_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_bf16_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_bf16_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_bf16_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_tf32_rcr_sm80_run,
    baracuda_kernels_gemm_bias_tf32_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_tf32_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_tf32_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_tf32_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_tf32_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_tf32_rrr_sm80_run,
    baracuda_kernels_gemm_bias_tf32_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_tf32_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_tf32_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_tf32_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_tf32_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_f32_simt_rcr_sm80_run,
    baracuda_kernels_gemm_bias_f32_simt_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_f32_simt_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_f32_simt_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_f32_simt_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_f32_simt_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_f32_simt_rrr_sm80_run,
    baracuda_kernels_gemm_bias_f32_simt_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_f32_simt_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_f32_simt_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_f32_simt_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_f32_simt_rrr_sm80_can_implement
);

// ---------- BiasRelu × {f16, bf16, tf32, f32_simt} × {rcr, rrr} ------------

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_f16_rcr_sm80_run,
    baracuda_kernels_gemm_bias_relu_f16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_f16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_f16_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_f16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_f16_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_f16_rrr_sm80_run,
    baracuda_kernels_gemm_bias_relu_f16_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_f16_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_f16_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_f16_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_f16_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_bf16_rcr_sm80_run,
    baracuda_kernels_gemm_bias_relu_bf16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_bf16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_bf16_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_bf16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_bf16_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_bf16_rrr_sm80_run,
    baracuda_kernels_gemm_bias_relu_bf16_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_bf16_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_bf16_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_bf16_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_bf16_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_tf32_rcr_sm80_run,
    baracuda_kernels_gemm_bias_relu_tf32_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_tf32_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_tf32_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_tf32_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_tf32_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_tf32_rrr_sm80_run,
    baracuda_kernels_gemm_bias_relu_tf32_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_tf32_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_tf32_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_tf32_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_tf32_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_f32_simt_rcr_sm80_run,
    baracuda_kernels_gemm_bias_relu_f32_simt_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_f32_simt_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_f32_simt_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_f32_simt_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_f32_simt_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_f32_simt_rrr_sm80_run,
    baracuda_kernels_gemm_bias_relu_f32_simt_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_f32_simt_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_f32_simt_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_f32_simt_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_f32_simt_rrr_sm80_can_implement
);

// ---------- BiasGelu × {f16, bf16, tf32, f32_simt} × {rcr, rrr} ------------

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_f16_rcr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_f16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_f16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_f16_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_f16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_f16_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_f16_rrr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_f16_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_f16_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_f16_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_f16_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_f16_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_bf16_rcr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_bf16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_bf16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_bf16_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_bf16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_bf16_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_bf16_rrr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_bf16_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_bf16_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_bf16_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_bf16_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_bf16_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_tf32_rcr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_tf32_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_tf32_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_tf32_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_tf32_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_tf32_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_tf32_rrr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_tf32_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_tf32_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_tf32_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_tf32_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_tf32_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_f32_simt_rcr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_f32_simt_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_f32_simt_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_f32_simt_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_f32_simt_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_f32_simt_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_f32_simt_rrr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_f32_simt_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_f32_simt_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_f32_simt_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_f32_simt_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_f32_simt_rrr_sm80_can_implement
);

// ---------- BiasSilu × {f16, bf16, tf32, f32_simt} × {rcr, rrr} ------------

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_f16_rcr_sm80_run,
    baracuda_kernels_gemm_bias_silu_f16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_f16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_f16_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_f16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_f16_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_f16_rrr_sm80_run,
    baracuda_kernels_gemm_bias_silu_f16_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_f16_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_f16_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_f16_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_f16_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_bf16_rcr_sm80_run,
    baracuda_kernels_gemm_bias_silu_bf16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_bf16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_bf16_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_bf16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_bf16_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_bf16_rrr_sm80_run,
    baracuda_kernels_gemm_bias_silu_bf16_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_bf16_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_bf16_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_bf16_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_bf16_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_tf32_rcr_sm80_run,
    baracuda_kernels_gemm_bias_silu_tf32_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_tf32_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_tf32_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_tf32_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_tf32_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_tf32_rrr_sm80_run,
    baracuda_kernels_gemm_bias_silu_tf32_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_tf32_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_tf32_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_tf32_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_tf32_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_f32_simt_rcr_sm80_run,
    baracuda_kernels_gemm_bias_silu_f32_simt_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_f32_simt_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_f32_simt_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_f32_simt_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_f32_simt_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_f32_simt_rrr_sm80_run,
    baracuda_kernels_gemm_bias_silu_f32_simt_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_f32_simt_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_f32_simt_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_f32_simt_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_f32_simt_rrr_sm80_can_implement
);

// ============================================================================
// Bias-fused int8 GEMM, f32 alpha/beta with f32/i32 bias broadcast
// ============================================================================
//
// 16 families: {Bias, BiasRelu, BiasGelu, BiasSilu} × {f32bias, i32bias}
// × {s8, u8} × rcr.

// ---------- f32bias / s8 ---------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_f32bias_s8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_f32bias_s8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_f32bias_s8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_f32bias_s8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_f32bias_s8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_f32bias_s8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_f32bias_s8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_relu_f32bias_s8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_f32bias_s8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_f32bias_s8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_f32bias_s8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_f32bias_s8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_f32bias_s8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_f32bias_s8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_f32bias_s8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_f32bias_s8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_f32bias_s8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_f32bias_s8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_f32bias_s8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_silu_f32bias_s8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_f32bias_s8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_f32bias_s8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_f32bias_s8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_f32bias_s8_rcr_sm80_can_implement
);

// ---------- f32bias / u8 ---------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_f32bias_u8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_f32bias_u8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_f32bias_u8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_f32bias_u8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_f32bias_u8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_f32bias_u8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_f32bias_u8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_relu_f32bias_u8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_f32bias_u8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_f32bias_u8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_f32bias_u8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_f32bias_u8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_f32bias_u8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_f32bias_u8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_f32bias_u8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_f32bias_u8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_f32bias_u8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_f32bias_u8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_f32bias_u8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_silu_f32bias_u8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_f32bias_u8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_f32bias_u8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_f32bias_u8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_f32bias_u8_rcr_sm80_can_implement
);

// ---------- i32bias / s8 ---------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_i32bias_s8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_i32bias_s8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_i32bias_s8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_i32bias_s8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_i32bias_s8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_i32bias_s8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_i32bias_s8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_relu_i32bias_s8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_i32bias_s8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_i32bias_s8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_i32bias_s8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_i32bias_s8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_i32bias_s8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_i32bias_s8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_i32bias_s8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_i32bias_s8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_i32bias_s8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_i32bias_s8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_i32bias_s8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_silu_i32bias_s8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_i32bias_s8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_i32bias_s8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_i32bias_s8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_i32bias_s8_rcr_sm80_can_implement
);

// ---------- i32bias / u8 ---------------------------------------------------

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_i32bias_u8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_i32bias_u8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_i32bias_u8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_i32bias_u8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_i32bias_u8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_i32bias_u8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_relu_i32bias_u8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_relu_i32bias_u8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_i32bias_u8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_i32bias_u8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_i32bias_u8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_i32bias_u8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_gelu_i32bias_u8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_i32bias_u8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_i32bias_u8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_i32bias_u8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_i32bias_u8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_i32bias_u8_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f32!(
    baracuda_kernels_gemm_bias_silu_i32bias_u8_rcr_sm80_run,
    baracuda_kernels_gemm_bias_silu_i32bias_u8_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_i32bias_u8_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_i32bias_u8_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_i32bias_u8_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_i32bias_u8_rcr_sm80_can_implement
);

// ============================================================================
// Bias-fused single GEMM, f64 alpha/beta (DGEMM bias family)
// ============================================================================
//
// 8 families: {Bias, BiasRelu, BiasGelu, BiasSilu} × {rcr, rrr}.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f64!(
    baracuda_kernels_gemm_bias_f64_rcr_sm80_run,
    baracuda_kernels_gemm_bias_f64_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_f64_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_f64_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_f64_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_f64_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f64!(
    baracuda_kernels_gemm_bias_f64_rrr_sm80_run,
    baracuda_kernels_gemm_bias_f64_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_f64_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_f64_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_f64_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_f64_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f64!(
    baracuda_kernels_gemm_bias_relu_f64_rcr_sm80_run,
    baracuda_kernels_gemm_bias_relu_f64_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_f64_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_f64_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_f64_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_f64_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f64!(
    baracuda_kernels_gemm_bias_relu_f64_rrr_sm80_run,
    baracuda_kernels_gemm_bias_relu_f64_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_relu_f64_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_relu_f64_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_relu_f64_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_relu_f64_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f64!(
    baracuda_kernels_gemm_bias_gelu_f64_rcr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_f64_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_f64_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_f64_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_f64_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_f64_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f64!(
    baracuda_kernels_gemm_bias_gelu_f64_rrr_sm80_run,
    baracuda_kernels_gemm_bias_gelu_f64_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_gelu_f64_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_gelu_f64_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_gelu_f64_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_gelu_f64_rrr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f64!(
    baracuda_kernels_gemm_bias_silu_f64_rcr_sm80_run,
    baracuda_kernels_gemm_bias_silu_f64_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_f64_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_f64_rcr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_f64_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_f64_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_bias_f64!(
    baracuda_kernels_gemm_bias_silu_f64_rrr_sm80_run,
    baracuda_kernels_gemm_bias_silu_f64_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_silu_f64_rrr_sm80_can_implement,
    baracuda_cutlass_gemm_bias_silu_f64_rrr_sm80_run,
    baracuda_cutlass_gemm_bias_silu_f64_rrr_sm80_workspace_size,
    baracuda_cutlass_gemm_bias_silu_f64_rrr_sm80_can_implement
);

// ============================================================================
// Strided-batched single GEMM, f32 alpha/beta
// ============================================================================
//
// 2 families: {f16, bf16} × rcr.

#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_batched_f32!(
    baracuda_kernels_gemm_batched_f16_rcr_sm80_run,
    baracuda_kernels_gemm_batched_f16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_batched_f16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_batched_f16_rcr_sm80_run,
    baracuda_cutlass_gemm_batched_f16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_batched_f16_rcr_sm80_can_implement
);
#[cfg(any(feature = "sm80", feature = "sm90a"))]
gemm_batched_f32!(
    baracuda_kernels_gemm_batched_bf16_rcr_sm80_run,
    baracuda_kernels_gemm_batched_bf16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_batched_bf16_rcr_sm80_can_implement,
    baracuda_cutlass_gemm_batched_bf16_rcr_sm80_run,
    baracuda_cutlass_gemm_batched_bf16_rcr_sm80_workspace_size,
    baracuda_cutlass_gemm_batched_bf16_rcr_sm80_can_implement
);
