//! # baracuda-transformer-engine-sys
//!
//! Raw FFI bindings + static-link wiring for the
//! [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)
//! (Apache-2.0) FP8 cast/transpose + delayed-scaling recipe primitives.
//!
//! This crate is the FFI shim — the safe wrapper lives in
//! [`baracuda-transformer-engine`]. Most baracuda callers will not
//! touch this crate directly; they go through the safe wrapper's
//! `Fp8Recipe` / `Fp8CastPlan` / `Fp8DequantPlan` types, which are
//! re-exported from `baracuda_kernels::transformer_engine` when the
//! `tensor_engine` cargo feature on `baracuda-kernels` is enabled.
//!
//! ## Scope
//!
//! Phase 55 lifts only the **cast/recipe** subset of TE upstream:
//!
//! - **`cast/`** — FP8 cast/transpose with delayed-scaling recipe
//! - **`recipe/`** — `DelayedScaling` recipe + amax history management
//!
//! Out of scope (every one of them overlaps an existing baracuda
//! phase, would need cuDNN, or targets Hopper-only hardware
//! baracuda's current test target doesn't have):
//!
//! - `normalization` (baracuda Phase 5 RMSNorm/LayerNorm)
//! - `fused_rope` (baracuda Phase 14/36/41)
//! - `fused_attn` (baracuda Phase 17/42 — also the one piece of TE
//!   that *would* need cuDNN 9.3+, hence "no cuDNN" is achievable
//!   in this phase by skipping it)
//! - `fused_softmax` (baracuda Phase 5)
//! - `activation` (baracuda Phase 3/31)
//! - `gemm` (baracuda Phase 1+24+30)
//! - `comm_gemm_overlap` / `nvshmem_api` (Hopper-only)
//! - `fused_router` (baracuda Phase 8 + 20 MoE)
//! - `hadamard_transform`, `newton_schulz`, `swizzle`, `permutation`
//! - `multi_tensor` (baracuda Phase 49 Apex optimizer)
//! - `dropout` (caller can compose)
//! - All Python bindings (`transformer_engine/pytorch/`,
//!   `transformer_engine/jax/`) — we expose a raw C ABI, not pybind11.
//!
//! ## sm_89 reality check (RTX 4070)
//!
//! Ada Lovelace (sm_89) has FP8 storage + cast intrinsics, but the
//! tensor-core FP8 MMA throughput is roughly equivalent to BF16. So
//! on this hardware:
//!
//! - The cast intrinsics work — `__nv_cvt_float_to_fp8` etc. compile
//!   and execute via the in-toolkit `<cuda_fp8.h>` API.
//! - You get a **2× bandwidth saving** vs BF16 on the cast endpoints
//!   (KV cache, weight storage, gradient passes).
//! - You do **not** get an MMA throughput win on the GEMM that
//!   consumes the FP8 tensor — that's the same speed as a BF16 GEMM.
//!
//! On Hopper (sm_90a) and Blackwell (sm_100), the MMA throughput
//! win actually materializes. The recipe machinery in this crate
//! is forward-compatible with those generations — once you have
//! the hardware, swap `BackendKind::TransformerEngineFp8` for an
//! MMA-aware GEMM kernel and the same recipe drives both.
//!
//! ## What this crate exposes
//!
//! A small flat C ABI defined by `csrc/baracuda_te_shim.cu`:
//!
//! - [`baracuda_te_fused_cast_amax_run`] — fused FP8 cast +
//!   `max(|x|)` amax reduction (one kernel launch).
//! - [`baracuda_te_dequant_run`] — FP8 → {f32, f16, bf16} dequantize.
//! - [`baracuda_te_recipe_update_run`] — TE delayed-scaling recipe
//!   update: reduces the amax history ring with `fmax`, computes
//!   `scale = max_repr / max_amax`, publishes `scale` + `scale_inv`.
//! - [`baracuda_te_recipe_init_run`] — set recipe defaults
//!   (`scale=1, scale_inv=1, amax_history=0`).
//!
//! The shim implements the published TE algorithm directly — same
//! `max_representable / max_amax_in_history` formula as TE's
//! `transformer_engine/common/recipe/delayed_scaling.cu`, `fmax`
//! reduction (the TE default), wrap-around index ring.
//!
//! Bypasses pybind11 — Rust talks raw C ABI. Bypasses cuDNN — the
//! cast/recipe paths don't need it (cuDNN is needed only for
//! `fused_attn`, which we skip).
//!
//! ## Provenance
//!
//! See `ATTRIBUTION.md` at the crate root for full license + scope
//! provenance.

#![no_std]
#![deny(missing_docs)]

use core::ffi::c_void;

unsafe extern "C" {
    /// Format id helper — returns the integer that the rest of the
    /// FFI expects for the E4M3 format (currently 0).
    pub fn baracuda_te_fp8_format_e4m3() -> i32;
    /// Format id helper — returns the integer that the rest of the
    /// FFI expects for the E5M2 format (currently 1).
    pub fn baracuda_te_fp8_format_e5m2() -> i32;
    /// `max_representable` for the given format. E4M3 = 448.0,
    /// E5M2 = 57344.0; other inputs return 1.0.
    pub fn baracuda_te_fp8_max_representable(fmt: i32) -> f32;

    /// Fused FP8 cast + amax reduction.
    ///
    /// Routes input through `scale` and saturates into FP8 (`SATFINITE`
    /// semantics — `|x|` clamps to the format's max-finite instead of
    /// producing infinities). In the same kernel, reduces `max(|x_in|)`
    /// (un-scaled — the amax tracks the raw tensor's dynamic range,
    /// not the post-scale value) and `atomicMax`-publishes it into
    /// `amax_history[write_pos]`.
    ///
    /// - `x_in`: device pointer to TIn elements (`in_dtype` selects
    ///   TIn: 0=f32, 1=f16, 2=bf16).
    /// - `x_out`: device pointer to `numel` bytes of FP8 output.
    /// - `scale`: device-resident f32 scalar.
    /// - `amax_history`: device-resident f32 array (length checked
    ///   only via `write_pos` in this call).
    /// - `write_pos`: index into `amax_history` that this call's amax
    ///   reduction lands in. The recipe-update kernel resets this slot
    ///   on each round, so the caller is responsible for not racing
    ///   two cast calls into the same slot.
    /// - `numel`: number of elements to cast.
    /// - `fmt`: `0` = E4M3, `1` = E5M2.
    /// - `in_dtype`: `0` = f32, `1` = f16, `2` = bf16.
    /// - `stream`: opaque `cudaStream_t` pointer.
    ///
    /// Returns `0` on success, `1` on validation failure, `5` on
    /// launch failure (`cudaGetLastError` returned non-zero).
    #[allow(clippy::too_many_arguments)]
    pub fn baracuda_te_fused_cast_amax_run(
        x_in: *const c_void,
        x_out: *mut c_void,
        scale: *const f32,
        amax_history: *mut f32,
        write_pos: i32,
        numel: i64,
        fmt: i32,
        in_dtype: i32,
        stream: *mut c_void,
    ) -> i32;

    /// FP8 → {f32, f16, bf16} dequantize.
    ///
    /// - `x_in`: device pointer to `numel` bytes of FP8 input.
    /// - `y_out`: device pointer to TOut elements (`out_dtype`
    ///   selects TOut: 0=f32, 1=f16, 2=bf16).
    /// - `scale_inv`: device-resident f32 scalar (reciprocal of
    ///   the scale used at cast time).
    /// - `numel`: number of elements.
    /// - `fmt`: `0` = E4M3, `1` = E5M2.
    /// - `out_dtype`: `0` = f32, `1` = f16, `2` = bf16.
    /// - `stream`: opaque `cudaStream_t` pointer.
    ///
    /// Returns 0/1/5 per the standard convention.
    pub fn baracuda_te_dequant_run(
        x_in: *const c_void,
        y_out: *mut c_void,
        scale_inv: *const f32,
        numel: i64,
        fmt: i32,
        out_dtype: i32,
        stream: *mut c_void,
    ) -> i32;

    /// TE delayed-scaling recipe update.
    ///
    /// Reduces `amax_history[0 .. hist_len]` with `fmax`, computes
    /// `scale = max_repr / max_amax` (clamped to identity if the
    /// history is all-zero), publishes `scale` + `scale_inv` to
    /// their device-resident scalars, and resets
    /// `amax_history[write_pos]` to 0.0f so the next FW pass's
    /// fused-cast `atomicMax` starts from a clean slate.
    ///
    /// - `amax_history`, `scale`, `scale_inv`: device-resident
    ///   buffers (length 1 for scalars; `hist_len` for the ring).
    /// - `hist_len`: ring length (1..=8192). Typical: 1024.
    /// - `write_pos`: slot the caller wants reset (the just-finished
    ///   FW pass wrote to it).
    /// - `fmt`: `0` = E4M3, `1` = E5M2.
    /// - `stream`: opaque `cudaStream_t` pointer.
    ///
    /// Returns 0/1/5 per the standard convention.
    pub fn baracuda_te_recipe_update_run(
        amax_history: *mut f32,
        scale: *mut f32,
        scale_inv: *mut f32,
        hist_len: i32,
        write_pos: i32,
        fmt: i32,
        stream: *mut c_void,
    ) -> i32;

    /// Initialize a recipe's device-resident state:
    /// `scale = 1.0`, `scale_inv = 1.0`, `amax_history[..] = 0.0`.
    ///
    /// Async on the caller's stream.
    pub fn baracuda_te_recipe_init_run(
        amax_history: *mut f32,
        scale: *mut f32,
        scale_inv: *mut f32,
        hist_len: i32,
        stream: *mut c_void,
    ) -> i32;
}

/// Status code returned by the shim on success.
pub const STATUS_OK: i32 = 0;
/// Status code returned by the shim on argument validation failure.
pub const STATUS_INVALID_ARGUMENT: i32 = 1;
/// Status code returned by the shim on `cudaGetLastError` failure.
pub const STATUS_LAUNCH_FAILED: i32 = 5;

/// Integer-cast format ids. Match the values returned by
/// [`baracuda_te_fp8_format_e4m3`] / [`baracuda_te_fp8_format_e5m2`].
pub const FP8_FORMAT_E4M3: i32 = 0;
/// See [`FP8_FORMAT_E4M3`].
pub const FP8_FORMAT_E5M2: i32 = 1;

/// Integer-cast dtype ids for the cast / dequant `in_dtype` /
/// `out_dtype` arguments.
pub const DTYPE_F32: i32 = 0;
/// See [`DTYPE_F32`].
pub const DTYPE_F16: i32 = 1;
/// See [`DTYPE_F32`].
pub const DTYPE_BF16: i32 = 2;
