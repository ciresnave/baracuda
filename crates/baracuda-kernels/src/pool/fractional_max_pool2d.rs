//! FractionalMaxPool2d — bespoke kernel (Phase 16.3).
//!
//! `nn.FractionalMaxPool2d` pools an `[N, C, H_in, W_in]` input with a
//! fixed `(kh, kw)` window into an `[N, C, H_out, W_out]` output where
//! `H_out` is a *fractional* multiple of `H_in` (not necessarily a
//! divisor). Each output cell pulls from a kernel-shaped input window
//! whose start position is chosen pseudorandomly per-axis, per-(n, c).
//!
//! cuDNN has no fractional-pool primitive — this plan ships a bespoke
//! CUDA kernel (FW + BW × `{f16, bf16, f32, f64}`).
//!
//! # Window-placement formula
//!
//! baracuda uses an **evenly-spaced base position + per-output-cell α
//! perturbation**, not PyTorch's exact `start_index`/`end_index`
//! sequence derivation. The formula for each axis:
//!
//! ```text
//! base[i]  = i * (in - k) / (out - 1)      (out > 1)
//! start[i] = clamp(floor(base[i] + α[n, c, axis]), 0, in - k)
//! ```
//!
//! where `α[n, c, axis] ∈ [0, 1)` is the caller-provided sample. For
//! `out == 1` we collapse to `start[0] = floor(α * (in - k))`.
//!
//! This produces a valid fractional-max-pool — each output cell sees a
//! unique `kh × kw` window — but **does not bit-match PyTorch's
//! `nn.FractionalMaxPool2d`**. Outputs are deterministic in α; pick a
//! reproducible cuRAND seed to lock the schedule across runs.
//!
//! # Random-samples ABI
//!
//! Callers supply `random_samples: TensorRef<f32, 3>` of shape
//! `[N, C, 2]` — one α per (batch, channel, {h-axis, w-axis}). f32 is
//! used regardless of `T` (input dtype) because uniform[0, 1) precision
//! past ~24 bits is meaningless.
//!
//! Generate the samples via [`baracuda_curand`] (or `RandomPlan` with
//! `RandomKind::Uniform`) before each forward call — internal RNG state
//! is **not** managed by the plan.
//!
//! # Saved indices (FW → BW)
//!
//! The FW kernel writes both `y` (per-window max value) and `indices`
//! (per-window argmax as an `i64` linear index into the input tensor).
//! The BW kernel consumes `indices` to scatter `dy[i]` into
//! `dx[indices[i]]` via `atomicAdd` (half / bf16 route through the
//! 32-bit `atomicCAS` helper per Phase 11.3 / Fuel feedback #6).
//!
//! Callers must retain `indices` between FW and BW. The plan does not
//! allocate it internally — it's a caller-supplied `[N, C, H_out, W_out]`
//! `i64` tensor.
//!
//! # Workspace
//!
//! Zero — the kernel is fully scratchless. Pass [`Workspace::None`].
//!
//! # Stub history
//!
//! Phase 11.8 shipped this plan as `Error::Unsupported` stub; Phase 16.3
//! ships the bespoke kernel.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    baracuda_kernels_fractional_max_pool_2d_bw_bf16_run,
    baracuda_kernels_fractional_max_pool_2d_bw_f16_run,
    baracuda_kernels_fractional_max_pool_2d_bw_f32_run,
    baracuda_kernels_fractional_max_pool_2d_bw_f64_run,
    baracuda_kernels_fractional_max_pool_2d_fw_bf16_run,
    baracuda_kernels_fractional_max_pool_2d_fw_f16_run,
    baracuda_kernels_fractional_max_pool_2d_fw_f32_run,
    baracuda_kernels_fractional_max_pool_2d_fw_f64_run,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PoolKind, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for `FractionalMaxPool2d`.
///
/// Input shape: `[batch, channels, h_in, w_in]`. Output shape:
/// `[batch, channels, h_out, w_out]`. `(window_h, window_w)` must each
/// be `>= 1` and `<= h_in / w_in`.
///
/// `h_out` / `w_out` are arbitrary positive integers — they need NOT
/// divide `h_in` / `w_in`. The whole point of fractional pooling is the
/// "non-divisor" output extent.
#[derive(Copy, Clone, Debug)]
pub struct FractionalMaxPool2dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input height.
    pub h_in: i32,
    /// Input width.
    pub w_in: i32,
    /// Window height.
    pub window_h: i32,
    /// Window width.
    pub window_w: i32,
    /// Desired output height.
    pub h_out: i32,
    /// Desired output width.
    pub w_out: i32,
    /// PRNG seed for the per-cell sampling.
    ///
    /// **Unused** in Phase 16.3 — random samples are supplied by the
    /// caller via the `random_samples` tensor in
    /// [`FractionalMaxPool2dFwArgs`]. Retained for ABI back-compat and
    /// for a future internal-RNG variant.
    pub seed: u64,
    /// Element dtype. Must match `T::KIND` and be one of
    /// `{F32, F64, F16, Bf16}`.
    pub element: ElementKind,
}

/// Args bundle for the forward launch.
///
/// `indices` is **caller-allocated** and written by the kernel — retain
/// it for the matching backward call.
pub struct FractionalMaxPool2dFwArgs<'a, T: Element> {
    /// Input `[N, C, H_in, W_in]` NCHW contiguous.
    pub x: TensorRef<'a, T, 4>,
    /// Output `[N, C, H_out, W_out]` NCHW contiguous.
    pub y: TensorMut<'a, T, 4>,
    /// Per-window argmax linear-index output `[N, C, H_out, W_out]`,
    /// `i64`. Consumed by [`FractionalMaxPool2dBwArgs::indices`].
    pub indices: TensorMut<'a, i64, 4>,
    /// Per-(batch, channel, axis) uniform[0, 1) samples
    /// `[N, C, 2]` f32. The caller fills this (typically via
    /// `RandomPlan + RandomKind::Uniform`).
    pub random_samples: TensorRef<'a, f32, 3>,
}

/// Args bundle for the backward launch.
///
/// **Caller must zero `dx` before calling.** The kernel uses
/// `atomicAdd` scatter; pre-existing values in `dx` would be summed
/// with the gradient.
pub struct FractionalMaxPool2dBwArgs<'a, T: Element> {
    /// Upstream gradient `[N, C, H_out, W_out]`.
    pub dy: TensorRef<'a, T, 4>,
    /// Saved forward argmax indices `[N, C, H_out, W_out]` i64
    /// (linear into the input tensor).
    pub indices: TensorRef<'a, i64, 4>,
    /// Output gradient w.r.t. input `[N, C, H_in, W_in]`. Must be
    /// pre-zeroed by the caller.
    pub dx: TensorMut<'a, T, 4>,
}

/// 2-D fractional max-pool plan (bespoke kernel).
///
/// See module docs for the window-placement formula and the
/// random-samples ABI. Saved-`indices` BW pattern shared with cuDNN
/// max-pool.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16`.
///
/// **Workspace**: zero (Workspace::None).
///
/// **Precision guarantee**: BW is non-deterministic across launches
/// (atomicAdd order races); FW is deterministic given the same input +
/// samples. half / bf16 BW routes through atomicCAS for per-thread
/// bit-stability.
pub struct FractionalMaxPool2dPlan<T: Element> {
    desc: FractionalMaxPool2dDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> FractionalMaxPool2dPlan<T> {
    /// Validate the descriptor and pick a kernel SKU.
    pub fn select(
        _stream: &Stream,
        desc: &FractionalMaxPool2dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        let sku = build_sku::<T>(PoolKind::FractionalMaxPool2d, true);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Kernel SKU identity.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Workspace size in bytes. Always `0`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Run the forward pass.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FractionalMaxPool2dFwArgs<'_, T>,
    ) -> Result<()> {
        check_fw_args(&self.desc, &args)?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x = args.x.data.as_raw().0 as *const c_void;
        let y = args.y.data.as_raw().0 as *mut c_void;
        let indices = args.indices.data.as_raw().0 as *mut c_void;
        let rs = args.random_samples.data.as_raw().0 as *const f32;
        let status = unsafe {
            match T::KIND {
                ElementKind::F32 => baracuda_kernels_fractional_max_pool_2d_fw_f32_run(
                    x, y, indices, rs,
                    self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.h_out, self.desc.w_out,
                    self.desc.window_h, self.desc.window_w,
                    stream_ptr,
                ),
                ElementKind::F64 => baracuda_kernels_fractional_max_pool_2d_fw_f64_run(
                    x, y, indices, rs,
                    self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.h_out, self.desc.w_out,
                    self.desc.window_h, self.desc.window_w,
                    stream_ptr,
                ),
                ElementKind::F16 => baracuda_kernels_fractional_max_pool_2d_fw_f16_run(
                    x, y, indices, rs,
                    self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.h_out, self.desc.w_out,
                    self.desc.window_h, self.desc.window_w,
                    stream_ptr,
                ),
                ElementKind::Bf16 => baracuda_kernels_fractional_max_pool_2d_fw_bf16_run(
                    x, y, indices, rs,
                    self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.h_out, self.desc.w_out,
                    self.desc.window_h, self.desc.window_w,
                    stream_ptr,
                ),
                _ => return Err(Error::Unsupported(
                    "baracuda-kernels::FractionalMaxPool2dPlan: dtype not in {f16, bf16, f32, f64}",
                )),
            }
        };
        ffi_status(status)
    }

    /// Run the backward pass. **Caller must zero `dx` before this call.**
    pub fn run_bw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FractionalMaxPool2dBwArgs<'_, T>,
    ) -> Result<()> {
        check_bw_args(&self.desc, &args)?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let dy = args.dy.data.as_raw().0 as *const c_void;
        let indices = args.indices.data.as_raw().0 as *const c_void;
        let dx = args.dx.data.as_raw().0 as *mut c_void;
        let status = unsafe {
            match T::KIND {
                ElementKind::F32 => baracuda_kernels_fractional_max_pool_2d_bw_f32_run(
                    dy, indices, dx,
                    self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.h_out, self.desc.w_out,
                    stream_ptr,
                ),
                ElementKind::F64 => baracuda_kernels_fractional_max_pool_2d_bw_f64_run(
                    dy, indices, dx,
                    self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.h_out, self.desc.w_out,
                    stream_ptr,
                ),
                ElementKind::F16 => baracuda_kernels_fractional_max_pool_2d_bw_f16_run(
                    dy, indices, dx,
                    self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.h_out, self.desc.w_out,
                    stream_ptr,
                ),
                ElementKind::Bf16 => baracuda_kernels_fractional_max_pool_2d_bw_bf16_run(
                    dy, indices, dx,
                    self.desc.batch, self.desc.channels,
                    self.desc.h_in, self.desc.w_in,
                    self.desc.h_out, self.desc.w_out,
                    stream_ptr,
                ),
                _ => return Err(Error::Unsupported(
                    "baracuda-kernels::FractionalMaxPool2dPlan: dtype not in {f16, bf16, f32, f64}",
                )),
            }
        };
        ffi_status(status)
    }
}

// ============================================================================
// Shared helpers (used by the 3-D sibling too).
// ============================================================================

pub(crate) fn validate_descriptor<T: Element>(
    desc: &FractionalMaxPool2dDescriptor,
) -> Result<()> {
    if desc.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::FractionalMaxPool2dPlan: descriptor.element != T::KIND",
        ));
    }
    if !matches!(
        T::KIND,
        ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
    ) {
        return Err(Error::Unsupported(
            "baracuda-kernels::FractionalMaxPool2dPlan: dtype not in {f16, bf16, f32, f64}",
        ));
    }
    if desc.batch <= 0 || desc.channels <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: batch / channels must be > 0",
        ));
    }
    if desc.h_in <= 0 || desc.w_in <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: h_in / w_in must be > 0",
        ));
    }
    if desc.h_out <= 0 || desc.w_out <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: h_out / w_out must be > 0",
        ));
    }
    if desc.window_h <= 0 || desc.window_w <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: window extents must be > 0",
        ));
    }
    if desc.window_h > desc.h_in || desc.window_w > desc.w_in {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: window must fit within input",
        ));
    }
    Ok(())
}

pub(crate) fn build_sku<T: Element>(op: PoolKind, deterministic_fw_only: bool) -> KernelSku {
    let math_precision = match T::KIND {
        ElementKind::F64 => MathPrecision::F64,
        ElementKind::F16 => MathPrecision::F16,
        ElementKind::Bf16 => MathPrecision::Bf16,
        _ => MathPrecision::F32,
    };
    let accumulator = match T::KIND {
        ElementKind::F64 => ElementKind::F64,
        _ => ElementKind::F32,
    };
    let precision_guarantee = PrecisionGuarantee {
        math_precision,
        accumulator,
        // FW is bit-stable given identical input + samples; BW uses
        // atomicAdd so racing across launches.
        bit_stable_on_same_hardware: false,
        // FW is deterministic; BW is not. The KernelSku field is
        // op-agnostic so we set it to FW's guarantee (callers using BW
        // should treat it as non-deterministic).
        deterministic: deterministic_fw_only,
    };
    KernelSku {
        category: OpCategory::Pooling,
        op: op as u16,
        element: T::KIND,
        aux_element: None,
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee,
    }
}

fn check_fw_args<T: Element>(
    desc: &FractionalMaxPool2dDescriptor,
    args: &FractionalMaxPool2dFwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, desc.h_out, desc.w_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: x shape != [N, C, H_in, W_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: y shape != [N, C, H_out, W_out]",
        ));
    }
    if args.indices.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: indices shape != [N, C, H_out, W_out]",
        ));
    }
    if args.random_samples.shape != [desc.batch, desc.channels, 2] {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: random_samples shape != [N, C, 2]",
        ));
    }
    Ok(())
}

fn check_bw_args<T: Element>(
    desc: &FractionalMaxPool2dDescriptor,
    args: &FractionalMaxPool2dBwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, desc.h_out, desc.w_out];
    if args.dy.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: dy shape != [N, C, H_out, W_out]",
        ));
    }
    if args.indices.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: indices shape != [N, C, H_out, W_out]",
        ));
    }
    if args.dx.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool2dPlan: dx shape != [N, C, H_in, W_in]",
        ));
    }
    Ok(())
}

pub(crate) fn ffi_status(status: i32) -> Result<()> {
    match status {
        0 => Ok(()),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels::FractionalMaxPool*dPlan: invalid problem (ffi status 2)",
        )),
        _ => Err(Error::CutlassInternal(-status)),
    }
}
