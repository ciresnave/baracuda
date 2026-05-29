//! # baracuda-transformer-engine
//!
//! Safe Rust wrapper for baracuda's port of NVIDIA
//! [TransformerEngine](https://github.com/NVIDIA/TransformerEngine)'s
//! FP8 cast/transpose + delayed-scaling recipe primitives.
//!
//! Provides:
//!
//! - [`Fp8Recipe`] — RAII handle for a per-tensor delayed-scaling
//!   recipe (amax history ring buffer + current scale +
//!   reciprocal scale).
//! - [`Fp8CastPlan`] — fused cast `{f32, f16, bf16}` → FP8 with
//!   per-launch `max(|x|)` amax reduction into the recipe.
//! - [`Fp8DequantPlan`] — FP8 → `{f32, f16, bf16}` dequant using
//!   the recipe's `scale_inv` scalar.
//!
//! ## Sm_89 reality check
//!
//! On Ada Lovelace (RTX 4070, RTX 4090, L40, L4 — sm_89), the FP8
//! storage and cast intrinsics work natively, but the tensor-core
//! FP8 MMA throughput is roughly equivalent to BF16. So on this
//! hardware, the FP8 wins are **bandwidth-saving only**:
//!
//! - **Real win**: 2× memory savings on KV cache, weight storage,
//!   activation memory.
//! - **No win**: FP8 GEMM consumed downstream runs at BF16-equivalent
//!   throughput on the tensor cores.
//!
//! On Hopper (sm_90a) and Blackwell (sm_100), the MMA throughput
//! win also materializes. The recipe machinery in this crate is
//! forward-compatible — once you have that hardware, the same
//! [`Fp8Recipe`] drives whatever MMA-aware GEMM kernel you wire up.
//!
//! ## Algorithm
//!
//! Mirrors TE's published `transformer_engine/common/recipe/delayed_scaling.cu`:
//!
//! 1. **During each forward pass**: the [`Fp8CastPlan`] fuses cast
//!    + `max(|x|)` reduction in one kernel. The reduced amax is
//!    `atomicMax`-published into `amax_history[write_pos]`.
//! 2. **After the forward pass**: [`Fp8Recipe::update_after_pass`]
//!    reduces the amax history ring with `fmax`, computes
//!    `new_scale = max_representable / max_amax`, publishes
//!    `scale` + `scale_inv`, and resets the just-written slot.
//! 3. The ring write pointer advances; the next forward pass
//!    writes into the new slot, etc.
//!
//! ## Scope
//!
//! Phase 55 lifts only the cast/recipe subset of TE upstream.
//! Everything else (normalization, fused attention, fused RoPE,
//! activations, GEMM, comm overlap, multi-tensor, dropout, Python
//! bindings) is deliberately skipped — see the
//! `baracuda-transformer-engine-sys` crate's `ATTRIBUTION.md` for
//! the full scope discussion.
//!
//! ## Example
//!
//! ```no_run
//! use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
//! use baracuda_transformer_engine::{Fp8Format, Fp8Recipe, Fp8CastPlan};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! baracuda_driver::init()?;
//! let ctx = Context::new(&Device::get(0)?)?;
//! let stream = Stream::new(&ctx)?;
//!
//! // 1. Build a recipe (amax history len 1024 is TE's typical default).
//! let mut recipe = Fp8Recipe::new(&ctx, &stream, Fp8Format::E4M3, 1024)?;
//!
//! // 2. Build a cast plan for f16 -> E4M3.
//! let plan: Fp8CastPlan<half::f16> = Fp8CastPlan::select()?;
//!
//! // 3. Per forward pass: cast inputs through the plan.
//! let x: DeviceBuffer<half::f16> = DeviceBuffer::zeros(&ctx, 4096)?;
//! let mut y: DeviceBuffer<u8>    = DeviceBuffer::zeros(&ctx, 4096)?;
//! plan.run(&x, &mut y, &mut recipe, &stream)?;
//!
//! // 4. Periodically (e.g. once per training step) advance the recipe.
//! recipe.update_after_pass(&stream)?;
//! # Ok(()) }
//! ```

#![deny(missing_docs)]

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_driver::{Context, DeviceBuffer, Stream};
use baracuda_transformer_engine_sys as sys;
use baracuda_types::DeviceRepr;
use thiserror::Error;

// ============================================================================
// Error / Result
// ============================================================================

/// Error category surfaced by the FP8 plans + recipe.
///
/// `#[non_exhaustive]` per the baracuda Phase 28 audit — additional
/// variants may land as the surface grows (e.g. transpose-specific
/// errors when the Phase 55b transpose kernel ships).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Caller-side validation failed before launch (length, alignment,
    /// dtype mismatch, etc.).
    #[error("transformer-engine argument invalid: {0}")]
    InvalidArgument(&'static str),
    /// The shim's argument-validation path returned `STATUS_INVALID_ARGUMENT`.
    #[error("shim reported invalid argument (status 1)")]
    ShimInvalidArgument,
    /// The shim returned `STATUS_LAUNCH_FAILED` — `cudaGetLastError`
    /// was non-zero after a kernel launch.
    #[error("shim kernel launch failed (status 5)")]
    LaunchFailed,
    /// The shim returned a non-zero status code we don't have a
    /// dedicated variant for.
    #[error("shim returned unknown status {0}")]
    UnknownStatus(i32),
    /// `baracuda_driver` propagation.
    #[error("driver error: {0}")]
    Driver(#[from] baracuda_driver::Error),
}

/// `Result` alias used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;

fn status_to_result(status: i32) -> Result<()> {
    match status {
        sys::STATUS_OK => Ok(()),
        sys::STATUS_INVALID_ARGUMENT => Err(Error::ShimInvalidArgument),
        sys::STATUS_LAUNCH_FAILED => Err(Error::LaunchFailed),
        other => Err(Error::UnknownStatus(other)),
    }
}

// ============================================================================
// FP8 format + input/output dtype tags
// ============================================================================

/// The FP8 format the recipe + cast plan target.
///
/// - [`Fp8Format::E4M3`] — 4-bit exponent, 3-bit mantissa, finite max
///   448.0. Higher precision, narrower dynamic range. TE's default
///   for the forward pass and weights.
/// - [`Fp8Format::E5M2`] — 5-bit exponent, 2-bit mantissa, finite max
///   57344.0. Wider dynamic range, lower precision. TE's default
///   for the backward pass (gradients).
///
/// `#[non_exhaustive]` per the baracuda Phase 28 audit — the OFP8
/// spec is fixed but future TE versions may add additional
/// per-tensor format flavors (e.g. the in-development MXFP8 with
/// per-block scales).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Fp8Format {
    /// 4-bit exponent, 3-bit mantissa. Finite max = 448.0.
    E4M3,
    /// 5-bit exponent, 2-bit mantissa. Finite max = 57344.0.
    E5M2,
}

impl Fp8Format {
    /// Integer cast for the FFI shim.
    pub fn to_ffi(self) -> i32 {
        match self {
            Fp8Format::E4M3 => sys::FP8_FORMAT_E4M3,
            Fp8Format::E5M2 => sys::FP8_FORMAT_E5M2,
        }
    }

    /// Maximum representable finite value in this format.
    /// E4M3 = 448.0, E5M2 = 57344.0.
    pub fn max_representable(self) -> f32 {
        match self {
            Fp8Format::E4M3 => 448.0,
            Fp8Format::E5M2 => 57344.0,
        }
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
    impl Sealed for half::bf16 {}
}

/// Sealed trait — input / output dtypes supported by the FP8
/// cast / dequant plans.
///
/// Implemented for `f32`, `half::f16`, and `half::bf16`. The
/// associated `FFI_TAG` is the integer the shim's `in_dtype` /
/// `out_dtype` argument expects.
///
/// `DeviceRepr` supertrait is required by `DeviceBuffer<T>`'s own
/// bound (so the plan can hold typed buffers of `T`).
pub trait Fp8WideDtype: Copy + 'static + DeviceRepr + sealed::Sealed {
    /// Integer dtype tag the FFI shim consumes.
    const FFI_TAG: i32;
}

impl Fp8WideDtype for f32 {
    const FFI_TAG: i32 = sys::DTYPE_F32;
}
impl Fp8WideDtype for half::f16 {
    const FFI_TAG: i32 = sys::DTYPE_F16;
}
impl Fp8WideDtype for half::bf16 {
    const FFI_TAG: i32 = sys::DTYPE_BF16;
}

// ============================================================================
// Fp8Recipe
// ============================================================================

/// Delayed-scaling FP8 recipe state.
///
/// Owns:
///
/// - `amax_history`: device-resident f32 ring buffer (length
///   chosen at construction; TE's typical default is 1024).
/// - `scale`: device-resident f32 scalar — the multiplier applied
///   to wide-dtype values during the cast.
/// - `scale_inv`: device-resident f32 scalar — `1.0 / scale`,
///   precomputed for the dequant hot path.
///
/// Workflow per training step:
///
/// 1. The forward pass [`Fp8CastPlan::run`] calls land amax samples
///    into `amax_history[write_pos]` via the fused-cast atomicMax.
/// 2. Call [`Fp8Recipe::update_after_pass`] to reduce the history,
///    compute the new scale, and advance the write pointer.
///
/// The first forward pass runs with the default identity scale
/// (`scale = scale_inv = 1.0`), so the very first cast emits
/// saturated values for large inputs. This matches TE's published
/// behavior — the recipe stabilizes after `amax_history_len`
/// passes (typically a fraction of a training step).
pub struct Fp8Recipe {
    format: Fp8Format,
    amax_history: DeviceBuffer<f32>,
    scale: DeviceBuffer<f32>,
    scale_inv: DeviceBuffer<f32>,
    /// Length of the amax ring (matches `amax_history.len()`).
    history_len: i32,
    /// Next slot the cast plan's atomicMax will land in.
    write_pos: i32,
}

impl Fp8Recipe {
    /// Construct a new recipe with `history_len` slots, initial
    /// `scale = scale_inv = 1.0`, all-zero amax history.
    ///
    /// `history_len` must be in `1..=8192`. TE's typical default
    /// is `1024` — large enough that transient outliers don't
    /// permanently distort the scale, small enough that the recipe
    /// adapts to phase changes in the training run.
    pub fn new(
        ctx: &Context,
        stream: &Stream,
        format: Fp8Format,
        history_len: usize,
    ) -> Result<Self> {
        if history_len == 0 || history_len > 8192 {
            return Err(Error::InvalidArgument(
                "amax history length must be in 1..=8192",
            ));
        }
        let history_len_i32 = history_len as i32;

        let amax_history: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, history_len)?;
        let scale: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, 1)?;
        let scale_inv: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, 1)?;

        // SAFETY: pointers are valid for the duration of the call;
        // stream is a valid CUstream; lengths match the buffer sizes
        // we just allocated.
        let status = unsafe {
            sys::baracuda_te_recipe_init_run(
                device_ptr_mut(&amax_history) as *mut f32,
                device_ptr_mut(&scale) as *mut f32,
                device_ptr_mut(&scale_inv) as *mut f32,
                history_len_i32,
                stream_as_raw(stream),
            )
        };
        status_to_result(status)?;

        Ok(Self {
            format,
            amax_history,
            scale,
            scale_inv,
            history_len: history_len_i32,
            write_pos: 0,
        })
    }

    /// FP8 format this recipe targets.
    pub fn format(&self) -> Fp8Format {
        self.format
    }

    /// Length of the amax history ring.
    pub fn history_len(&self) -> usize {
        self.history_len as usize
    }

    /// Slot index the next [`Fp8CastPlan::run`] call will land in.
    /// Advances inside [`Fp8Recipe::update_after_pass`].
    pub fn write_pos(&self) -> usize {
        self.write_pos as usize
    }

    /// Recipe update — call once per training step, after the
    /// forward pass(es) that wrote amax samples into the current
    /// slot have completed (in stream order).
    ///
    /// Reduces `amax_history` with `fmax`, computes
    /// `new_scale = max_representable / max_amax`, publishes
    /// `scale` + `scale_inv` to their device-resident scalars,
    /// resets the just-written slot to 0.0, and advances the
    /// write pointer with wrap-around.
    pub fn update_after_pass(&mut self, stream: &Stream) -> Result<()> {
        // SAFETY: all pointers belong to this recipe; stream is a
        // valid CUstream; lengths are bounded by construction.
        let status = unsafe {
            sys::baracuda_te_recipe_update_run(
                device_ptr_mut(&self.amax_history) as *mut f32,
                device_ptr_mut(&self.scale) as *mut f32,
                device_ptr_mut(&self.scale_inv) as *mut f32,
                self.history_len,
                self.write_pos,
                self.format.to_ffi(),
                stream_as_raw(stream),
            )
        };
        status_to_result(status)?;

        // Advance the write pointer with wrap-around.
        self.write_pos = (self.write_pos + 1) % self.history_len;
        Ok(())
    }

    /// Borrow the device-resident amax history. Useful for
    /// inspection / logging via `copy_to_host`.
    pub fn amax_history(&self) -> &DeviceBuffer<f32> {
        &self.amax_history
    }

    /// Borrow the device-resident `scale` scalar.
    pub fn scale(&self) -> &DeviceBuffer<f32> {
        &self.scale
    }

    /// Borrow the device-resident `scale_inv` scalar.
    pub fn scale_inv(&self) -> &DeviceBuffer<f32> {
        &self.scale_inv
    }

    /// Read the current scale to the host. Synchronizes the stream.
    pub fn scale_host(&self, stream: &Stream) -> Result<f32> {
        stream.synchronize()?;
        let mut out = [0.0f32; 1];
        self.scale.copy_to_host(&mut out)?;
        Ok(out[0])
    }

    /// Read the current scale_inv to the host. Synchronizes the
    /// stream.
    pub fn scale_inv_host(&self, stream: &Stream) -> Result<f32> {
        stream.synchronize()?;
        let mut out = [0.0f32; 1];
        self.scale_inv.copy_to_host(&mut out)?;
        Ok(out[0])
    }
}

// ============================================================================
// Fp8CastPlan
// ============================================================================

/// Fused FP8 cast plan for `TIn` → FP8 with running amax reduction
/// into the recipe.
///
/// The plan is zero-state (the kernel selection is purely a
/// function of dtype); a single instance is reusable across many
/// calls and across many recipe instances. The [`Fp8CastPlan::run`]
/// method takes the recipe by `&mut Self` because each launch
/// mutates the recipe's amax history slot via atomicMax.
pub struct Fp8CastPlan<TIn: Fp8WideDtype> {
    _marker: PhantomData<TIn>,
}

impl<TIn: Fp8WideDtype> Fp8CastPlan<TIn> {
    /// Construct a plan instance. Cheap; no GPU activity.
    pub fn select() -> Result<Self> {
        Ok(Self {
            _marker: PhantomData,
        })
    }

    /// Run the cast.
    ///
    /// - `input`: wide-dtype source (`TIn` elements).
    /// - `output`: FP8 destination (one `u8` per element).
    /// - `recipe`: delayed-scaling state; the kernel reads `recipe.scale`
    ///   and atomicMaxes into `recipe.amax_history[recipe.write_pos]`.
    ///
    /// Async on `stream`. The caller is responsible for ensuring
    /// `input` and `output` outlive the launch (standard CUDA
    /// async contract — see `Stream::synchronize`).
    pub fn run(
        &self,
        input: &DeviceBuffer<TIn>,
        output: &mut DeviceBuffer<u8>,
        recipe: &mut Fp8Recipe,
        stream: &Stream,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::InvalidArgument(
                "Fp8CastPlan::run: input.len() != output.len()",
            ));
        }
        if input.is_empty() {
            return Err(Error::InvalidArgument(
                "Fp8CastPlan::run: empty input",
            ));
        }

        // SAFETY: pointers are valid + correctly typed; recipe's
        // amax_history was allocated with the same length we record
        // in `history_len`; write_pos is bounded `< history_len` by
        // construction (see Fp8Recipe::update_after_pass).
        let status = unsafe {
            sys::baracuda_te_fused_cast_amax_run(
                device_ptr(input),
                device_ptr_mut(output),
                device_ptr(&recipe.scale) as *const f32,
                device_ptr_mut(&recipe.amax_history) as *mut f32,
                recipe.write_pos,
                input.len() as i64,
                recipe.format.to_ffi(),
                TIn::FFI_TAG,
                stream_as_raw(stream),
            )
        };
        status_to_result(status)
    }
}

// ============================================================================
// Fp8DequantPlan
// ============================================================================

/// FP8 → `TOut` dequantize plan.
///
/// Symmetric to [`Fp8CastPlan`]; uses `recipe.scale_inv` to map
/// FP8 values back to the wide dtype. No amax bookkeeping (dequant
/// doesn't influence the recipe state — it's a pure read).
pub struct Fp8DequantPlan<TOut: Fp8WideDtype> {
    _marker: PhantomData<TOut>,
}

impl<TOut: Fp8WideDtype> Fp8DequantPlan<TOut> {
    /// Construct a plan instance. Cheap; no GPU activity.
    pub fn select() -> Result<Self> {
        Ok(Self {
            _marker: PhantomData,
        })
    }

    /// Run the dequant.
    ///
    /// - `input`: FP8 source (one `u8` per element).
    /// - `output`: wide-dtype destination.
    /// - `recipe`: provides `scale_inv` (immutably read).
    pub fn run(
        &self,
        input: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<TOut>,
        recipe: &Fp8Recipe,
        stream: &Stream,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(Error::InvalidArgument(
                "Fp8DequantPlan::run: input.len() != output.len()",
            ));
        }
        if input.is_empty() {
            return Err(Error::InvalidArgument(
                "Fp8DequantPlan::run: empty input",
            ));
        }

        // SAFETY: pointers are valid + correctly typed; scale_inv
        // is a single-element scalar of f32.
        let status = unsafe {
            sys::baracuda_te_dequant_run(
                device_ptr(input),
                device_ptr_mut(output),
                device_ptr(&recipe.scale_inv) as *const f32,
                input.len() as i64,
                recipe.format.to_ffi(),
                TOut::FFI_TAG,
                stream_as_raw(stream),
            )
        };
        status_to_result(status)
    }
}

// ============================================================================
// Internal helpers — translate baracuda-driver handles to raw FFI ptrs
// ============================================================================

fn device_ptr<T: DeviceRepr>(buf: &DeviceBuffer<T>) -> *const c_void {
    // CUdeviceptr is a tuple-struct wrapper around u64; same
    // pattern as in baracuda-cudnn / baracuda-optim.
    buf.as_raw().0 as *const c_void
}

fn device_ptr_mut<T: DeviceRepr>(buf: &DeviceBuffer<T>) -> *mut c_void {
    buf.as_raw().0 as *mut c_void
}

fn stream_as_raw(stream: &Stream) -> *mut c_void {
    stream.as_raw() as *mut c_void
}
