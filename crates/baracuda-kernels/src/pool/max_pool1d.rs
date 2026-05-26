//! MaxPool1d — NCL 1-D max-pool via cuDNN's Nd-pooling descriptor API.
//!
//! Sibling of [`super::MaxPool2dPlan`]. Drives the same cuDNN exec
//! entry points but configures a rank-1 pooling descriptor (the rank-3
//! `[N, C, L]` tensor descriptor is internally padded to rank-4 with
//! `W = 1` because cuDNN's `cudnnSetTensorNdDescriptor` requires
//! `nb_dims >= 4`).
//!
//! Dtype coverage: `f32` / `f64` / `f16` / `bf16` — same as the 2-D
//! plan. Workspace-free. Gated under `feature = "cudnn"`.

use core::cell::Cell;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{cudnnHandle_t, cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t};
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PoolKind, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

use super::max_pool2d::{build_sku, PoolMode};
use super::pool_nd::{
    bind_stream, drop_descriptors_nd, ensure_descriptors_nd, ensure_handle, out_dim, run_bw_nd,
    run_fw_nd, validate_dtype,
};

/// Descriptor for a 1-D pooling op over NCL tensors.
///
/// Input shape: `[batch, channels, l_in]`. Output shape:
/// `[batch, channels, l_out]` where
/// `l_out = floor((l_in + 2·pad - window) / stride) + 1`.
///
/// Shared between [`super::MaxPool1dPlan`] and [`super::AvgPool1dPlan`];
/// the [`Self::mode`] field selects max vs. one of the two average-pool
/// flavors.
///
/// `#[non_exhaustive]` (Phase 32) — see [`super::Pool2dDescriptor`]
/// for the builder rationale. Use [`Self::new`] + the `with_*` setters
/// from downstream code.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct Pool1dDescriptor {
    /// Batch size `N`.
    pub batch: i32,
    /// Channel count `C`.
    pub channels: i32,
    /// Input length `L_in`.
    pub l_in: i32,
    /// Pooling window length.
    pub window: i32,
    /// Zero-padding on each side of the input length axis.
    pub pad: i32,
    /// Stride along the length axis.
    pub stride: i32,
    /// Pooling flavor — selects the cuDNN exec path.
    pub mode: PoolMode,
    /// Element dtype. Must be `F32`, `F64`, `F16`, or `Bf16`.
    pub element: ElementKind,
}

impl Pool1dDescriptor {
    /// Build a descriptor with `pad` defaulted to `0` and `stride`
    /// defaulted to the window extent (PyTorch's default). Chain with
    /// [`Self::with_padding`] / [`Self::with_stride`] to override.
    pub fn new(
        batch: i32,
        channels: i32,
        l_in: i32,
        window: i32,
        mode: PoolMode,
        element: ElementKind,
    ) -> Self {
        Self {
            batch,
            channels,
            l_in,
            window,
            pad: 0,
            stride: window,
            mode,
            element,
        }
    }

    /// Override the padding. Default `0`.
    #[inline]
    pub fn with_padding(mut self, pad: i32) -> Self {
        self.pad = pad;
        self
    }

    /// Override the stride. Default `window` (PyTorch's pooling
    /// default).
    #[inline]
    pub fn with_stride(mut self, stride: i32) -> Self {
        self.stride = stride;
        self
    }
}

/// Args bundle for a 1-D pooling forward launch.
pub struct Pool1dFwArgs<'a, T: Element> {
    /// Input activations `[N, C, L_in]` row-major contiguous.
    pub x: TensorRef<'a, T, 3>,
    /// Output activations `[N, C, L_out]` row-major contiguous.
    pub y: TensorMut<'a, T, 3>,
}

/// Args bundle for a 1-D pooling backward launch.
///
/// As with the 2-D path, **both** `y` (saved FW output) and `x` (saved
/// FW input) must be retained — cuDNN needs both to recover the
/// per-window argmax for max-pool.
pub struct Pool1dBwArgs<'a, T: Element> {
    /// Saved forward output `[N, C, L_out]`.
    pub y: TensorRef<'a, T, 3>,
    /// Upstream gradient `[N, C, L_out]`.
    pub dy: TensorRef<'a, T, 3>,
    /// Saved forward input `[N, C, L_in]`.
    pub x: TensorRef<'a, T, 3>,
    /// Output gradient w.r.t. input `[N, C, L_in]`.
    pub dx: TensorMut<'a, T, 3>,
}

/// 1-D max-pool plan (cuDNN-backed). FW + BW over NCL.
///
/// `select` requires `descriptor.mode == PoolMode::Max`. Use
/// [`super::AvgPool1dPlan`] for average-pool.
pub struct MaxPool1dPlan<T: Element> {
    desc: Pool1dDescriptor,
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    x_desc: Cell<cudnnTensorDescriptor_t>,
    y_desc: Cell<cudnnTensorDescriptor_t>,
    pool_desc: Cell<cudnnPoolingDescriptor_t>,
    _marker: PhantomData<T>,
}

impl<T: Element> MaxPool1dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &Pool1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        if !matches!(desc.mode, PoolMode::Max) {
            return Err(Error::Unsupported(
                "baracuda-kernels::MaxPool1dPlan: descriptor.mode must be PoolMode::Max",
            ));
        }
        let sku = build_sku::<T>(PoolKind::MaxPool1d);
        Ok(Self {
            desc: *desc,
            sku,
            handle: Cell::new(core::ptr::null_mut()),
            x_desc: Cell::new(core::ptr::null_mut()),
            y_desc: Cell::new(core::ptr::null_mut()),
            pool_desc: Cell::new(core::ptr::null_mut()),
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

    /// `L_out` under the configured window / pad / stride.
    #[inline]
    pub fn output_dim(&self) -> i32 {
        out_dim(self.desc.l_in, self.desc.pad, self.desc.window, self.desc.stride)
    }

    /// Run the forward pass. Computes `y := max_pool(x)`.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: Pool1dFwArgs<'_, T>,
    ) -> Result<()> {
        check_fw_args(&self.desc, &args)?;
        let h = ensure_handle(&self.handle)?;
        bind_stream(h, stream)?;
        self.ensure_descs()?;
        run_fw_nd::<T>(
            h,
            self.pool_desc.get(),
            self.x_desc.get(),
            self.y_desc.get(),
            args.x.data.as_raw().0,
            args.y.data.as_raw().0,
        )
    }

    /// Run the backward pass.
    pub fn run_bw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: Pool1dBwArgs<'_, T>,
    ) -> Result<()> {
        check_bw_args(&self.desc, &args)?;
        let h = ensure_handle(&self.handle)?;
        bind_stream(h, stream)?;
        self.ensure_descs()?;
        run_bw_nd::<T>(
            h,
            self.pool_desc.get(),
            self.x_desc.get(),
            self.y_desc.get(),
            args.y.data.as_raw().0,
            args.dy.data.as_raw().0,
            args.x.data.as_raw().0,
            args.dx.data.as_raw().0,
        )
    }

    fn ensure_descs(&self) -> Result<()> {
        let l_out = self.output_dim();
        let x_dims = [self.desc.batch, self.desc.channels, self.desc.l_in];
        let y_dims = [self.desc.batch, self.desc.channels, l_out];
        let window = [self.desc.window];
        let padding = [self.desc.pad];
        let stride = [self.desc.stride];
        ensure_descriptors_nd::<T>(
            &x_dims,
            &y_dims,
            &window,
            &padding,
            &stride,
            self.desc.mode,
            &self.x_desc,
            &self.y_desc,
            &self.pool_desc,
        )
    }
}

impl<T: Element> Drop for MaxPool1dPlan<T> {
    fn drop(&mut self) {
        drop_descriptors_nd(&self.x_desc, &self.y_desc, &self.pool_desc, &self.handle);
    }
}

// =============================================================================
// Shared 1-D pooling helpers (used by sibling avg_pool1d)
// =============================================================================

pub(crate) fn validate_descriptor<T: Element>(desc: &Pool1dDescriptor) -> Result<()> {
    validate_dtype::<T>()?;
    if desc.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::Pool1dPlan: descriptor.element != T::KIND",
        ));
    }
    if desc.batch <= 0 || desc.channels <= 0 || desc.l_in <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: input shape extents must be > 0",
        ));
    }
    if desc.window <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: window must be > 0",
        ));
    }
    if desc.stride <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: stride must be > 0",
        ));
    }
    if desc.pad < 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: padding must be >= 0",
        ));
    }
    if desc.pad * 2 > desc.window {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: padding must be <= window / 2",
        ));
    }
    let l_out = out_dim(desc.l_in, desc.pad, desc.window, desc.stride);
    if l_out <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: computed output extent <= 0",
        ));
    }
    Ok(())
}

pub(crate) fn check_fw_args<T: Element>(
    desc: &Pool1dDescriptor,
    args: &Pool1dFwArgs<'_, T>,
) -> Result<()> {
    let l_out = out_dim(desc.l_in, desc.pad, desc.window, desc.stride);
    let x_shape = [desc.batch, desc.channels, desc.l_in];
    let y_shape = [desc.batch, desc.channels, l_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: x shape != [N, C, L_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: y shape != [N, C, L_out]",
        ));
    }
    Ok(())
}

pub(crate) fn check_bw_args<T: Element>(
    desc: &Pool1dDescriptor,
    args: &Pool1dBwArgs<'_, T>,
) -> Result<()> {
    let l_out = out_dim(desc.l_in, desc.pad, desc.window, desc.stride);
    let x_shape = [desc.batch, desc.channels, desc.l_in];
    let y_shape = [desc.batch, desc.channels, l_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: x shape != [N, C, L_in]",
        ));
    }
    if args.dx.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: dx shape != [N, C, L_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: y shape != [N, C, L_out]",
        ));
    }
    if args.dy.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool1dPlan: dy shape != [N, C, L_out]",
        ));
    }
    Ok(())
}
