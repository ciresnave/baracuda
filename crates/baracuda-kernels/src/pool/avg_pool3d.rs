//! AvgPool3d — NCDHW 3-D average-pool via cuDNN's Nd-pooling API.
//!
//! Sibling of [`super::AvgPool2dPlan`] for the rank-5 case.

use core::cell::Cell;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{cudnnHandle_t, cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t};
use baracuda_kernels_types::{
    Element, KernelSku, PlanPreference, PoolKind, PrecisionGuarantee, Workspace,
};

use super::max_pool2d::{build_sku, PoolMode};
use super::max_pool3d::{
    check_bw_args, check_fw_args, compute_output_dims, validate_descriptor, Pool3dBwArgs,
    Pool3dDescriptor, Pool3dFwArgs,
};
use super::pool_nd::{
    bind_stream, drop_descriptors_nd, ensure_descriptors_nd, ensure_handle, run_bw_nd, run_fw_nd,
};

/// 3-D average-pool plan (cuDNN-backed).
pub struct AvgPool3dPlan<T: Element> {
    desc: Pool3dDescriptor,
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    x_desc: Cell<cudnnTensorDescriptor_t>,
    y_desc: Cell<cudnnTensorDescriptor_t>,
    pool_desc: Cell<cudnnPoolingDescriptor_t>,
    _marker: PhantomData<T>,
}

impl<T: Element> AvgPool3dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &Pool3dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        let op = match desc.mode {
            PoolMode::AvgIncludePad => PoolKind::AvgPool3dIncludePad,
            PoolMode::AvgExcludePad => PoolKind::AvgPool3dExcludePad,
            PoolMode::Max => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::AvgPool3dPlan: descriptor.mode must be one of \
                     PoolMode::AvgIncludePad | AvgExcludePad — use MaxPool3dPlan for max",
                ));
            }
        };
        let sku = build_sku::<T>(op);
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

impl<T: Element> Drop for AvgPool3dPlan<T> {
    fn drop(&mut self) {
        drop_descriptors_nd(&self.x_desc, &self.y_desc, &self.pool_desc, &self.handle);
    }
}
