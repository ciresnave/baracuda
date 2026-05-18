//! AdaptiveAvgPool3d — NCDHW adaptive average-pool via cuDNN.
//!
//! See [`super::adaptive_avg_pool1d`] for the cuDNN-approximation
//! caveat. Per-axis kernel/stride derivation:
//!
//! ```text
//! kernel_i = ceil(in_i / out_i); stride_i = floor(in_i / out_i)
//! ```
//!
//! for `i ∈ {D, H, W}`, `padding = 0`. Bit-exact PyTorch only when each
//! axis is divisible.

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
    adaptive_kernel_stride, bind_stream, drop_descriptors_nd, ensure_descriptors_nd, ensure_handle,
    run_bw_nd, run_fw_nd, validate_dtype,
};

/// Descriptor for an adaptive 3-D pooling op over NCDHW tensors.
#[derive(Copy, Clone, Debug)]
pub struct AdaptivePool3dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input depth `D_in`.
    pub d_in: i32,
    /// Input height `H_in`.
    pub h_in: i32,
    /// Input width `W_in`.
    pub w_in: i32,
    /// Desired output depth.
    pub d_out: i32,
    /// Desired output height.
    pub h_out: i32,
    /// Desired output width.
    pub w_out: i32,
    /// Element dtype.
    pub element: ElementKind,
}

/// FW args.
pub struct AdaptivePool3dFwArgs<'a, T: Element> {
    /// Input `[N, C, D_in, H_in, W_in]`.
    pub x: TensorRef<'a, T, 5>,
    /// Output `[N, C, D_out, H_out, W_out]`.
    pub y: TensorMut<'a, T, 5>,
}

/// BW args.
pub struct AdaptivePool3dBwArgs<'a, T: Element> {
    /// Saved FW output.
    pub y: TensorRef<'a, T, 5>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, 5>,
    /// Saved FW input.
    pub x: TensorRef<'a, T, 5>,
    /// Output gradient.
    pub dx: TensorMut<'a, T, 5>,
}

/// Adaptive 3-D average-pool plan (cuDNN approximation).
pub struct AdaptiveAvgPool3dPlan<T: Element> {
    desc: AdaptivePool3dDescriptor,
    /// `((kernel_d, stride_d), (kernel_h, stride_h), (kernel_w, stride_w))`.
    derived: ((i32, i32), (i32, i32), (i32, i32)),
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    x_desc: Cell<cudnnTensorDescriptor_t>,
    y_desc: Cell<cudnnTensorDescriptor_t>,
    pool_desc: Cell<cudnnPoolingDescriptor_t>,
    _marker: PhantomData<T>,
}

impl<T: Element> AdaptiveAvgPool3dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &AdaptivePool3dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        let d = adaptive_kernel_stride(desc.d_in, desc.d_out);
        let h = adaptive_kernel_stride(desc.h_in, desc.h_out);
        let w = adaptive_kernel_stride(desc.w_in, desc.w_out);
        let sku = build_sku::<T>(PoolKind::AdaptiveAvgPool3d);
        Ok(Self {
            desc: *desc,
            derived: (d, h, w),
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

    /// `((kernel_d, stride_d), (kernel_h, stride_h), (kernel_w, stride_w))`
    /// derived per axis.
    #[inline]
    pub fn derived_kernel_stride(&self) -> ((i32, i32), (i32, i32), (i32, i32)) {
        self.derived
    }

    /// Run the forward pass.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AdaptivePool3dFwArgs<'_, T>,
    ) -> Result<()> {
        check_fw_args(&self.desc, &args)?;
        let h = ensure_handle(&self.handle)?;
        bind_stream(h, stream)?;
        self.ensure_descs(PoolMode::AvgExcludePad)?;
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
        args: AdaptivePool3dBwArgs<'_, T>,
    ) -> Result<()> {
        check_bw_args(&self.desc, &args)?;
        let h = ensure_handle(&self.handle)?;
        bind_stream(h, stream)?;
        self.ensure_descs(PoolMode::AvgExcludePad)?;
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

    fn ensure_descs(&self, mode: PoolMode) -> Result<()> {
        let x_dims = [
            self.desc.batch,
            self.desc.channels,
            self.desc.d_in,
            self.desc.h_in,
            self.desc.w_in,
        ];
        let y_dims = [
            self.desc.batch,
            self.desc.channels,
            self.desc.d_out,
            self.desc.h_out,
            self.desc.w_out,
        ];
        let window = [self.derived.0 .0, self.derived.1 .0, self.derived.2 .0];
        let padding = [0i32, 0i32, 0i32];
        let stride = [self.derived.0 .1, self.derived.1 .1, self.derived.2 .1];
        ensure_descriptors_nd::<T>(
            &x_dims,
            &y_dims,
            &window,
            &padding,
            &stride,
            mode,
            &self.x_desc,
            &self.y_desc,
            &self.pool_desc,
        )
    }
}

impl<T: Element> Drop for AdaptiveAvgPool3dPlan<T> {
    fn drop(&mut self) {
        drop_descriptors_nd(&self.x_desc, &self.y_desc, &self.pool_desc, &self.handle);
    }
}

// =============================================================================
// Shared adaptive-3d helpers (used by sibling adaptive_max_pool3d)
// =============================================================================

pub(crate) fn validate_descriptor<T: Element>(
    desc: &AdaptivePool3dDescriptor,
) -> Result<()> {
    validate_dtype::<T>()?;
    if desc.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::AdaptivePool3dPlan: descriptor.element != T::KIND",
        ));
    }
    if desc.batch <= 0
        || desc.channels <= 0
        || desc.d_in <= 0
        || desc.h_in <= 0
        || desc.w_in <= 0
        || desc.d_out <= 0
        || desc.h_out <= 0
        || desc.w_out <= 0
    {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool3dPlan: extents must be > 0",
        ));
    }
    if desc.d_out > desc.d_in || desc.h_out > desc.h_in || desc.w_out > desc.w_in {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool3dPlan: output extents must be <= input \
             extents (upsampling is not a pooling op)",
        ));
    }
    Ok(())
}

pub(crate) fn check_fw_args<T: Element>(
    desc: &AdaptivePool3dDescriptor,
    args: &AdaptivePool3dFwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.d_in, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, desc.d_out, desc.h_out, desc.w_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool3dPlan: x shape != [N, C, D_in, H_in, W_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool3dPlan: y shape != [N, C, D_out, H_out, W_out]",
        ));
    }
    Ok(())
}

pub(crate) fn check_bw_args<T: Element>(
    desc: &AdaptivePool3dDescriptor,
    args: &AdaptivePool3dBwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.d_in, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, desc.d_out, desc.h_out, desc.w_out];
    if args.x.shape != x_shape || args.dx.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool3dPlan: x/dx shape != [N, C, D_in, H_in, W_in]",
        ));
    }
    if args.y.shape != y_shape || args.dy.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool3dPlan: y/dy shape != [N, C, D_out, H_out, W_out]",
        ));
    }
    Ok(())
}
