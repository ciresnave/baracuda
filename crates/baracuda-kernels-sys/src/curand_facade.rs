//! Phase 23 — `baracuda-kernels-sys` C-ABI FFI wrappers for the
//! cuRAND-backed random-sampling family.
//!
//! Background: continues the Phase 22 design correction — every Rust
//! plan that wraps an NVIDIA library MUST also expose a flat C-ABI
//! entry. Phase 23's cuRAND slice covers the two pure-cuRAND-routed
//! samplers (Uniform / Normal). Bernoulli + Dropout are *composite*
//! ops (cuRAND-uniform + a bespoke kernel) and ALREADY ship with their
//! own bespoke FFI symbols (`baracuda_kernels_bernoulli_run`,
//! `baracuda_kernels_dropout_*_run`) plus the cuRAND lifecycle entry
//! points (`curandCreateGenerator`, etc.) re-exported from `lib.rs` —
//! external callers can compose them directly with no facade needed.
//!
//! ## Coverage
//!
//! 2 cuRAND-backed plan families, 8 FFI symbols total
//! (4 `_run` + 4 `_workspace_size`):
//!
//! - `curand_uniform` — `Uniform(low, high]` sampling × {f32, f64}.
//!   Wraps `curandGenerateUniform` (`f32`) / `curandGenerateUniformDouble`
//!   (`f64`) plus a follow-up in-place affine remap when
//!   `(low, high) != (0, 1)`.
//! - `curand_normal` — `Normal(mean, stddev)` sampling × {f32, f64}.
//!   Wraps `curandGenerateNormal` / `curandGenerateNormalDouble`.
//!
//! ## Generator lifecycle
//!
//! Each FFI call creates a transient `curandGenerator_t` via
//! `curandCreateGenerator(CURAND_RNG_PSEUDO_DEFAULT)`, seeds it with
//! the caller's `seed`, binds it to the caller's stream via
//! `curandSetStream`, generates the samples, applies the in-place
//! affine remap (Uniform only when `(low, high) != (0, 1)`), then
//! destroys the generator. No cross-call caching at the FFI layer —
//! callers that repeat a launch should drive the matching
//! `baracuda-kernels` Rust plan (which caches the generator across
//! launches for the lifetime of the plan).
//!
//! **Determinism trade-off**: because the generator is created + seeded
//! per call, repeated invocations with the same `seed` produce the
//! same sequence — this matches the safe-plan layer's contract. If a
//! caller drives the FFI in a loop expecting *advancing* state across
//! calls, they should mutate the seed externally or drive the
//! lower-level cuRAND symbols directly.
//!
//! ## Status codes
//!
//! Same convention as the rest of `baracuda-kernels-sys`:
//! - `0` — success.
//! - `2` — invalid problem (null pointers, non-positive numel,
//!   out-of-range distribution params).
//! - `5` — internal cuRAND error (non-zero status from the library).
//!
//! ## Workspace contract
//!
//! Pure-Uniform / pure-Normal sampling needs no caller-supplied
//! workspace — cuRAND writes directly into the output buffer. The
//! `*_workspace_size` queries always return `0`; the matching `*_run`
//! ignores the `workspace` / `workspace_bytes` pair.

#![allow(non_camel_case_types)]
#![allow(clippy::too_many_arguments)]

use core::ffi::c_void;
use core::ptr;

use super::{
    baracuda_kernels_affine_inplace_f32_run, baracuda_kernels_affine_inplace_f64_run,
    curandCreateGenerator, curandDestroyGenerator, curandGenerateNormal,
    curandGenerateNormalDouble, curandGenerateUniform, curandGenerateUniformDouble,
    curandGenerator_t, curandSetPseudoRandomGeneratorSeed, curandSetStream,
    CURAND_RNG_PSEUDO_DEFAULT,
};

// =============================================================================
// Status codes
// =============================================================================

const OK: i32 = 0;
const INVALID: i32 = 2;
const INTERNAL: i32 = 5;

#[inline]
fn map_curand(status: i32) -> i32 {
    if status == 0 { OK } else { INTERNAL }
}

// =============================================================================
// Internal RAII helpers
// =============================================================================

/// RAII guard for a cuRAND generator. Destroys on `Drop`, idempotent on
/// null.
struct Generator {
    g: curandGenerator_t,
}

impl Generator {
    #[inline]
    fn new() -> Self {
        Self { g: ptr::null_mut() }
    }
}

impl Drop for Generator {
    fn drop(&mut self) {
        if !self.g.is_null() {
            unsafe {
                let _ = curandDestroyGenerator(self.g);
            }
        }
    }
}

/// Create + seed a transient generator and bind it to the caller's
/// stream. Returns a status code (`OK` / `INTERNAL`).
#[inline]
unsafe fn setup_generator(g: &mut Generator, seed: u64, stream: *mut c_void) -> i32 {
    let s = unsafe { curandCreateGenerator(&mut g.g as *mut _, CURAND_RNG_PSEUDO_DEFAULT) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe { curandSetPseudoRandomGeneratorSeed(g.g, seed) };
    if s != 0 {
        return INTERNAL;
    }
    let s = unsafe { curandSetStream(g.g, stream) };
    if s != 0 {
        return INTERNAL;
    }
    OK
}

// =============================================================================
// curand_uniform — Uniform(low, high] × {f32, f64}
// =============================================================================
//
// Signature: (numel, low, high, seed, y, workspace, workspace_bytes,
//             stream).
// cuRAND generates samples in `(0, 1]`; if `(low, high) != (0, 1)`,
// the wrapper chains the in-place affine kernel
// `baracuda_kernels_affine_inplace_*_run` (`y = (high - low) * y + low`)
// on the same stream. `seed` is the per-call generator seed.

/// Uniform-sampler workspace size in bytes for `f32` — always `0`.
///
/// # Safety
/// `out_bytes` must point to a writable `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_curand_uniform_f32_workspace_size(
    _numel: i64,
    out_bytes: *mut usize,
) -> i32 {
    if out_bytes.is_null() {
        return INVALID;
    }
    unsafe { *out_bytes = 0 };
    OK
}

/// Uniform-sampler workspace size in bytes for `f64` — always `0`.
///
/// # Safety
/// `out_bytes` must point to a writable `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_curand_uniform_f64_workspace_size(
    _numel: i64,
    out_bytes: *mut usize,
) -> i32 {
    if out_bytes.is_null() {
        return INVALID;
    }
    unsafe { *out_bytes = 0 };
    OK
}

/// Sample `numel` `f32` cells from `Uniform(low, high]`.
///
/// Implementation: cuRAND writes `Uniform(0, 1]` into `y`, then (when
/// `(low, high) != (0, 1)`) an in-place affine kernel remaps to
/// `(low, high]`.
///
/// # Safety
/// `y` must point to at least `numel * sizeof(f32)` writable device
/// bytes. `stream` must be a live CUDA stream in the current context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_curand_uniform_f32_run(
    numel: i64,
    low: f32,
    high: f32,
    seed: u64,
    y: *mut c_void,
    _workspace: *mut c_void,
    _workspace_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    if numel < 0 || y.is_null() {
        return INVALID;
    }
    if !(high > low) && numel > 0 {
        return INVALID;
    }
    if numel == 0 {
        return OK;
    }
    let mut g = Generator::new();
    let s = unsafe { setup_generator(&mut g, seed, stream) };
    if s != OK {
        return s;
    }
    let st = unsafe { curandGenerateUniform(g.g, y as *mut f32, numel as usize) };
    if st != 0 {
        return INTERNAL;
    }
    if low != 0.0 || high != 1.0 {
        let scale = high - low;
        let s = unsafe {
            baracuda_kernels_affine_inplace_f32_run(
                numel,
                scale,
                low,
                y,
                ptr::null_mut(),
                0,
                stream,
            )
        };
        if s != OK {
            return s;
        }
    }
    OK
}

/// Sample `numel` `f64` cells from `Uniform(low, high]`.
///
/// # Safety
/// `y` must point to at least `numel * sizeof(f64)` writable device
/// bytes. `stream` must be a live CUDA stream in the current context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_curand_uniform_f64_run(
    numel: i64,
    low: f64,
    high: f64,
    seed: u64,
    y: *mut c_void,
    _workspace: *mut c_void,
    _workspace_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    if numel < 0 || y.is_null() {
        return INVALID;
    }
    if !(high > low) && numel > 0 {
        return INVALID;
    }
    if numel == 0 {
        return OK;
    }
    let mut g = Generator::new();
    let s = unsafe { setup_generator(&mut g, seed, stream) };
    if s != OK {
        return s;
    }
    let st = unsafe { curandGenerateUniformDouble(g.g, y as *mut f64, numel as usize) };
    if st != 0 {
        return INTERNAL;
    }
    if low != 0.0 || high != 1.0 {
        let scale = high - low;
        let s = unsafe {
            baracuda_kernels_affine_inplace_f64_run(
                numel,
                scale,
                low,
                y,
                ptr::null_mut(),
                0,
                stream,
            )
        };
        if s != OK {
            return s;
        }
    }
    OK
}

// =============================================================================
// curand_normal — Normal(mean, stddev) × {f32, f64}
// =============================================================================
//
// Signature: (numel, mean, stddev, seed, y, workspace, workspace_bytes,
//             stream).
// Wraps `curandGenerateNormal` / `curandGenerateNormalDouble` directly.
//
// **Even-numel note**: older cuRAND versions required `numel` to be
// even (Box-Muller paired output). Modern cuRAND (12.x) accepts any
// `numel`; if the runtime cuRAND rejects an odd `numel` the wrapper
// surfaces the non-zero status as `INTERNAL`.

/// Normal-sampler workspace size in bytes for `f32` — always `0`.
///
/// # Safety
/// `out_bytes` must point to a writable `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_curand_normal_f32_workspace_size(
    _numel: i64,
    out_bytes: *mut usize,
) -> i32 {
    if out_bytes.is_null() {
        return INVALID;
    }
    unsafe { *out_bytes = 0 };
    OK
}

/// Normal-sampler workspace size in bytes for `f64` — always `0`.
///
/// # Safety
/// `out_bytes` must point to a writable `usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_curand_normal_f64_workspace_size(
    _numel: i64,
    out_bytes: *mut usize,
) -> i32 {
    if out_bytes.is_null() {
        return INVALID;
    }
    unsafe { *out_bytes = 0 };
    OK
}

/// Sample `numel` `f32` cells from `Normal(mean, stddev)`.
///
/// # Safety
/// `y` must point to at least `numel * sizeof(f32)` writable device
/// bytes. `stream` must be a live CUDA stream in the current context.
/// `stddev` must be > 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_curand_normal_f32_run(
    numel: i64,
    mean: f32,
    stddev: f32,
    seed: u64,
    y: *mut c_void,
    _workspace: *mut c_void,
    _workspace_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    if numel < 0 || y.is_null() || !(stddev > 0.0) {
        return INVALID;
    }
    if numel == 0 {
        return OK;
    }
    let mut g = Generator::new();
    let s = unsafe { setup_generator(&mut g, seed, stream) };
    if s != OK {
        return s;
    }
    let st = unsafe {
        curandGenerateNormal(g.g, y as *mut f32, numel as usize, mean, stddev)
    };
    map_curand(st)
}

/// Sample `numel` `f64` cells from `Normal(mean, stddev)`.
///
/// # Safety
/// `y` must point to at least `numel * sizeof(f64)` writable device
/// bytes. `stream` must be a live CUDA stream in the current context.
/// `stddev` must be > 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn baracuda_kernels_curand_normal_f64_run(
    numel: i64,
    mean: f64,
    stddev: f64,
    seed: u64,
    y: *mut c_void,
    _workspace: *mut c_void,
    _workspace_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    if numel < 0 || y.is_null() || !(stddev > 0.0) {
        return INVALID;
    }
    if numel == 0 {
        return OK;
    }
    let mut g = Generator::new();
    let s = unsafe { setup_generator(&mut g, seed, stream) };
    if s != OK {
        return s;
    }
    let st = unsafe {
        curandGenerateNormalDouble(g.g, y as *mut f64, numel as usize, mean, stddev)
    };
    map_curand(st)
}
