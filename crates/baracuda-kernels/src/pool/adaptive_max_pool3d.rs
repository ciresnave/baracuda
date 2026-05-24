//! AdaptiveMaxPool3d — NCDHW adaptive max-pool, bit-exact PyTorch
//! (Phase 16.1 bespoke kernel).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::Result;
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, KernelSku, PlanPreference, PoolKind, PrecisionGuarantee, Workspace,
};

use super::adaptive_avg_pool1d::{
    build_sku, dispatch_max_bw, dispatch_max_fw, map_status,
};
use super::adaptive_avg_pool3d::{
    check_bw_args, check_fw_args, validate_descriptor, AdaptivePool3dBwArgs,
    AdaptivePool3dDescriptor, AdaptivePool3dFwArgs,
};

/// Adaptive 3-D max-pool plan (bit-exact PyTorch, bespoke kernel).
pub struct AdaptiveMaxPool3dPlan<T: Element> {
    desc: AdaptivePool3dDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> AdaptiveMaxPool3dPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &AdaptivePool3dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        let sku = build_sku::<T>(PoolKind::AdaptiveMaxPool3d);
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

    /// Deprecated. Always returns three `(0, 0)` pairs.
    #[inline]
    #[deprecated(
        since = "0.0.1-alpha.33",
        note = "Phase 16.1 uses bit-exact per-output-cell windows; no single (kernel, stride) pair applies."
    )]
    pub fn derived_kernel_stride(&self) -> ((i32, i32), (i32, i32), (i32, i32)) {
        ((0, 0), (0, 0), (0, 0))
    }

    /// Run the forward pass.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AdaptivePool3dFwArgs<'_, T>,
    ) -> Result<()> {
        check_fw_args(&self.desc, &args)?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let nc = self.desc.batch * self.desc.channels;
        let status = dispatch_max_fw::<T>(
            x_ptr,
            y_ptr,
            nc,
            3,
            self.desc.d_in, self.desc.h_in, self.desc.w_in,
            self.desc.d_out, self.desc.h_out, self.desc.w_out,
            stream_ptr,
        );
        map_status(status)
    }

    /// Run the backward pass.
    pub fn run_bw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AdaptivePool3dBwArgs<'_, T>,
    ) -> Result<()> {
        check_bw_args(&self.desc, &args)?;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let nc = self.desc.batch * self.desc.channels;
        let status = dispatch_max_bw::<T>(
            x_ptr,
            dy_ptr,
            dx_ptr,
            nc,
            3,
            self.desc.d_in, self.desc.h_in, self.desc.w_in,
            self.desc.d_out, self.desc.h_out, self.desc.w_out,
            stream_ptr,
        );
        map_status(status)
    }
}
