//! AdaptiveMaxPool2d — NCHW adaptive max-pool via cuDNN.
//!
//! See [`super::adaptive_avg_pool2d`] for the cuDNN-approximation caveat.

use core::cell::Cell;
use core::marker::PhantomData;

use baracuda_cutlass::Result;
use baracuda_driver::Stream;
use baracuda_kernels_sys::{cudnnHandle_t, cudnnPoolingDescriptor_t, cudnnTensorDescriptor_t};
use baracuda_kernels_types::{
    Element, KernelSku, PlanPreference, PoolKind, PrecisionGuarantee, Workspace,
};

use super::adaptive_avg_pool2d::{
    check_bw_args, check_fw_args, validate_descriptor, AdaptivePool2dBwArgs,
    AdaptivePool2dDescriptor, AdaptivePool2dFwArgs,
};
use super::max_pool2d::{build_sku, PoolMode};
use super::pool_nd::{
    adaptive_kernel_stride, bind_stream, drop_descriptors_nd, ensure_descriptors_nd, ensure_handle,
    run_bw_nd, run_fw_nd,
};

/// Adaptive 2-D max-pool plan (cuDNN approximation).
pub struct AdaptiveMaxPool2dPlan<T: Element> {
    desc: AdaptivePool2dDescriptor,
    derived: ((i32, i32), (i32, i32)),
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    x_desc: Cell<cudnnTensorDescriptor_t>,
    y_desc: Cell<cudnnTensorDescriptor_t>,
    pool_desc: Cell<cudnnPoolingDescriptor_t>,
    _marker: PhantomData<T>,
}

impl<T: Element> AdaptiveMaxPool2dPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &AdaptivePool2dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        let h = adaptive_kernel_stride(desc.h_in, desc.h_out);
        let w = adaptive_kernel_stride(desc.w_in, desc.w_out);
        let sku = build_sku::<T>(PoolKind::AdaptiveMaxPool2d);
        Ok(Self {
            desc: *desc,
            derived: (h, w),
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

    /// `((kernel_h, stride_h), (kernel_w, stride_w))` derived per axis.
    #[inline]
    pub fn derived_kernel_stride(&self) -> ((i32, i32), (i32, i32)) {
        self.derived
    }

    /// Run the forward pass.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AdaptivePool2dFwArgs<'_, T>,
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
        args: AdaptivePool2dBwArgs<'_, T>,
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
        let x_dims = [self.desc.batch, self.desc.channels, self.desc.h_in, self.desc.w_in];
        let y_dims = [self.desc.batch, self.desc.channels, self.desc.h_out, self.desc.w_out];
        let window = [self.derived.0 .0, self.derived.1 .0];
        let padding = [0i32, 0i32];
        let stride = [self.derived.0 .1, self.derived.1 .1];
        ensure_descriptors_nd::<T>(
            &x_dims,
            &y_dims,
            &window,
            &padding,
            &stride,
            PoolMode::Max,
            &self.x_desc,
            &self.y_desc,
            &self.pool_desc,
        )
    }
}

impl<T: Element> Drop for AdaptiveMaxPool2dPlan<T> {
    fn drop(&mut self) {
        drop_descriptors_nd(&self.x_desc, &self.y_desc, &self.pool_desc, &self.handle);
    }
}
