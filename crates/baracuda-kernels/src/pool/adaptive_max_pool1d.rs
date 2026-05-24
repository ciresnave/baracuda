//! AdaptiveMaxPool1d — NCL adaptive max-pool, bit-exact PyTorch
//! (Phase 16.1 bespoke kernel).
//!
//! See [`super::adaptive_avg_pool1d`] for the FW window formula and the
//! cuDNN-approximation history. MaxPool BW recomputes the per-window
//! argmax from saved `x` (no separate `indices` tensor — keeps the
//! Phase 11.8 args shape intact). half / bf16 BW atomics route through
//! `baracuda_atomic.cuh`'s CAS helper.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::Result;
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, KernelSku, PlanPreference, PoolKind, PrecisionGuarantee, Workspace,
};

use super::adaptive_avg_pool1d::{
    build_sku, check_bw_args, check_fw_args, dispatch_max_bw, dispatch_max_fw, map_status,
    validate_descriptor, AdaptivePool1dBwArgs, AdaptivePool1dDescriptor, AdaptivePool1dFwArgs,
};

/// Adaptive 1-D max-pool plan (bit-exact PyTorch, bespoke kernel).
pub struct AdaptiveMaxPool1dPlan<T: Element> {
    desc: AdaptivePool1dDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> AdaptiveMaxPool1dPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &AdaptivePool1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        let sku = build_sku::<T>(PoolKind::AdaptiveMaxPool1d);
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

    /// Deprecated. Always returns `(0, 0)` — see
    /// [`super::AdaptiveAvgPool1dPlan::derived_kernel_stride`].
    #[inline]
    #[deprecated(
        since = "0.0.1-alpha.33",
        note = "Phase 16.1 uses bit-exact per-output-cell windows; no single (kernel, stride) pair applies."
    )]
    pub fn derived_kernel_stride(&self) -> (i32, i32) {
        (0, 0)
    }

    /// Run the forward pass.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AdaptivePool1dFwArgs<'_, T>,
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
            1, // spatial_rank
            1, 1, self.desc.l_in,
            1, 1, self.desc.l_out,
            stream_ptr,
        );
        map_status(status)
    }

    /// Run the backward pass. Recomputes argmax from saved `x`; zeros
    /// `dx` internally before atomic-scatter.
    pub fn run_bw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AdaptivePool1dBwArgs<'_, T>,
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
            1,
            1, 1, self.desc.l_in,
            1, 1, self.desc.l_out,
            stream_ptr,
        );
        map_status(status)
    }
}
