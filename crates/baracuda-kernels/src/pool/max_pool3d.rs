//! MaxPool3d — NCDHW 3-D max-pool via cuDNN's Nd-pooling descriptor API.
//!
//! Sibling of [`super::MaxPool2dPlan`]. Drives the same cuDNN exec
//! entry points but configures a rank-3 (spatial-axes) pooling
//! descriptor and a rank-5 `[N, C, D, H, W]` tensor descriptor.
//!
//! Dtype coverage: `f32` / `f64` / `f16` / `bf16`. Workspace-free.
//! Gated under `feature = "cudnn"`.

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

/// Descriptor for a 3-D pooling op over NCDHW tensors.
///
/// Input shape: `[batch, channels, d_in, h_in, w_in]`. Output shape:
/// `[batch, channels, d_out, h_out, w_out]` computed per-axis under the
/// standard floor formula. Shared by [`super::MaxPool3dPlan`] and
/// [`super::AvgPool3dPlan`].
///
/// `#[non_exhaustive]` (Phase 32) — see [`super::Pool2dDescriptor`]
/// for the builder rationale. Use [`Self::new`] + the `with_*` setters
/// from downstream code.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct Pool3dDescriptor {
    /// Batch size `N`.
    pub batch: i32,
    /// Channel count `C`.
    pub channels: i32,
    /// Input depth `D_in`.
    pub d_in: i32,
    /// Input height `H_in`.
    pub h_in: i32,
    /// Input width `W_in`.
    pub w_in: i32,
    /// Window depth.
    pub window_d: i32,
    /// Window height.
    pub window_h: i32,
    /// Window width.
    pub window_w: i32,
    /// Padding along the depth axis.
    pub pad_d: i32,
    /// Padding along the height axis.
    pub pad_h: i32,
    /// Padding along the width axis.
    pub pad_w: i32,
    /// Stride along the depth axis.
    pub stride_d: i32,
    /// Stride along the height axis.
    pub stride_h: i32,
    /// Stride along the width axis.
    pub stride_w: i32,
    /// Pooling flavor.
    pub mode: PoolMode,
    /// Element dtype.
    pub element: ElementKind,
}

impl Pool3dDescriptor {
    /// Build a descriptor with `pad` defaulted to `(0, 0, 0)` and
    /// `stride` defaulted to the per-axis window extent (PyTorch's
    /// default). Chain with [`Self::with_padding`] /
    /// [`Self::with_stride`] to override.
    pub fn new(
        batch: i32,
        channels: i32,
        d_in: i32,
        h_in: i32,
        w_in: i32,
        window_d: i32,
        window_h: i32,
        window_w: i32,
        mode: PoolMode,
        element: ElementKind,
    ) -> Self {
        Self {
            batch,
            channels,
            d_in,
            h_in,
            w_in,
            window_d,
            window_h,
            window_w,
            pad_d: 0,
            pad_h: 0,
            pad_w: 0,
            stride_d: window_d,
            stride_h: window_h,
            stride_w: window_w,
            mode,
            element,
        }
    }

    /// Override `(pad_d, pad_h, pad_w)`. Default `(0, 0, 0)`.
    #[inline]
    pub fn with_padding(mut self, pad_d: i32, pad_h: i32, pad_w: i32) -> Self {
        self.pad_d = pad_d;
        self.pad_h = pad_h;
        self.pad_w = pad_w;
        self
    }

    /// Override `(stride_d, stride_h, stride_w)`. Default `(window_d,
    /// window_h, window_w)` (PyTorch's pooling default).
    #[inline]
    pub fn with_stride(mut self, stride_d: i32, stride_h: i32, stride_w: i32) -> Self {
        self.stride_d = stride_d;
        self.stride_h = stride_h;
        self.stride_w = stride_w;
        self
    }
}

/// Args bundle for a 3-D pooling forward launch.
pub struct Pool3dFwArgs<'a, T: Element> {
    /// Input `[N, C, D_in, H_in, W_in]` NCDHW contiguous.
    pub x: TensorRef<'a, T, 5>,
    /// Output `[N, C, D_out, H_out, W_out]` NCDHW contiguous.
    pub y: TensorMut<'a, T, 5>,
}

/// Args bundle for a 3-D pooling backward launch.
pub struct Pool3dBwArgs<'a, T: Element> {
    /// Saved forward output `[N, C, D_out, H_out, W_out]`.
    pub y: TensorRef<'a, T, 5>,
    /// Upstream gradient `[N, C, D_out, H_out, W_out]`.
    pub dy: TensorRef<'a, T, 5>,
    /// Saved forward input `[N, C, D_in, H_in, W_in]`.
    pub x: TensorRef<'a, T, 5>,
    /// Output gradient `[N, C, D_in, H_in, W_in]`.
    pub dx: TensorMut<'a, T, 5>,
}

/// 3-D max-pool plan (cuDNN-backed). FW + BW over NCDHW.
pub struct MaxPool3dPlan<T: Element> {
    desc: Pool3dDescriptor,
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    x_desc: Cell<cudnnTensorDescriptor_t>,
    y_desc: Cell<cudnnTensorDescriptor_t>,
    pool_desc: Cell<cudnnPoolingDescriptor_t>,
    _marker: PhantomData<T>,
}

impl<T: Element> MaxPool3dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &Pool3dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        if !matches!(desc.mode, PoolMode::Max) {
            return Err(Error::Unsupported(
                "baracuda-kernels::MaxPool3dPlan: descriptor.mode must be PoolMode::Max",
            ));
        }
        let sku = build_sku::<T>(PoolKind::MaxPool3d);
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

    /// `(D_out, H_out, W_out)` under the configured window / pad / stride.
    #[inline]
    pub fn output_dims(&self) -> (i32, i32, i32) {
        compute_output_dims(&self.desc)
    }

    /// Run the forward pass.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: Pool3dFwArgs<'_, T>,
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
        args: Pool3dBwArgs<'_, T>,
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
        let (d_out, h_out, w_out) = compute_output_dims(&self.desc);
        let x_dims = [
            self.desc.batch,
            self.desc.channels,
            self.desc.d_in,
            self.desc.h_in,
            self.desc.w_in,
        ];
        let y_dims = [self.desc.batch, self.desc.channels, d_out, h_out, w_out];
        let window = [self.desc.window_d, self.desc.window_h, self.desc.window_w];
        let padding = [self.desc.pad_d, self.desc.pad_h, self.desc.pad_w];
        let stride = [self.desc.stride_d, self.desc.stride_h, self.desc.stride_w];
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

impl<T: Element> Drop for MaxPool3dPlan<T> {
    fn drop(&mut self) {
        drop_descriptors_nd(&self.x_desc, &self.y_desc, &self.pool_desc, &self.handle);
    }
}

// =============================================================================
// Shared 3-D helpers (used by sibling avg_pool3d)
// =============================================================================

#[inline]
pub(crate) fn compute_output_dims(d: &Pool3dDescriptor) -> (i32, i32, i32) {
    (
        out_dim(d.d_in, d.pad_d, d.window_d, d.stride_d),
        out_dim(d.h_in, d.pad_h, d.window_h, d.stride_h),
        out_dim(d.w_in, d.pad_w, d.window_w, d.stride_w),
    )
}

pub(crate) fn validate_descriptor<T: Element>(desc: &Pool3dDescriptor) -> Result<()> {
    validate_dtype::<T>()?;
    if desc.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::Pool3dPlan: descriptor.element != T::KIND",
        ));
    }
    if desc.batch <= 0
        || desc.channels <= 0
        || desc.d_in <= 0
        || desc.h_in <= 0
        || desc.w_in <= 0
    {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: input shape extents must be > 0",
        ));
    }
    if desc.window_d <= 0 || desc.window_h <= 0 || desc.window_w <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: window extents must be > 0",
        ));
    }
    if desc.stride_d <= 0 || desc.stride_h <= 0 || desc.stride_w <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: stride extents must be > 0",
        ));
    }
    if desc.pad_d < 0 || desc.pad_h < 0 || desc.pad_w < 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: padding must be >= 0",
        ));
    }
    if desc.pad_d * 2 > desc.window_d
        || desc.pad_h * 2 > desc.window_h
        || desc.pad_w * 2 > desc.window_w
    {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: padding must be <= window / 2",
        ));
    }
    let (d_out, h_out, w_out) = compute_output_dims(desc);
    if d_out <= 0 || h_out <= 0 || w_out <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: computed output extent <= 0",
        ));
    }
    Ok(())
}

pub(crate) fn check_fw_args<T: Element>(
    desc: &Pool3dDescriptor,
    args: &Pool3dFwArgs<'_, T>,
) -> Result<()> {
    let (d_out, h_out, w_out) = compute_output_dims(desc);
    let x_shape = [desc.batch, desc.channels, desc.d_in, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, d_out, h_out, w_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: x shape != [N, C, D_in, H_in, W_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: y shape != [N, C, D_out, H_out, W_out]",
        ));
    }
    Ok(())
}

pub(crate) fn check_bw_args<T: Element>(
    desc: &Pool3dDescriptor,
    args: &Pool3dBwArgs<'_, T>,
) -> Result<()> {
    let (d_out, h_out, w_out) = compute_output_dims(desc);
    let x_shape = [desc.batch, desc.channels, desc.d_in, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, d_out, h_out, w_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: x shape != [N, C, D_in, H_in, W_in]",
        ));
    }
    if args.dx.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: dx shape != [N, C, D_in, H_in, W_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: y shape != [N, C, D_out, H_out, W_out]",
        ));
    }
    if args.dy.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool3dPlan: dy shape != [N, C, D_out, H_out, W_out]",
        ));
    }
    Ok(())
}
