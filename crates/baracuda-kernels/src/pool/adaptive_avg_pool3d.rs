//! AdaptiveAvgPool3d — NCDHW adaptive average-pool, bit-exact PyTorch
//! (Phase 16.1 bespoke kernel).
//!
//! See [`super::adaptive_avg_pool1d`] for the FW window formula.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    Element, ElementKind, KernelSku, PlanPreference, PoolKind, PrecisionGuarantee, TensorMut,
    TensorRef, Workspace,
};

use super::adaptive_avg_pool1d::{
    build_sku, dispatch_avg_bw, dispatch_avg_fw, map_status, validate_dtype,
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
    /// Saved FW output (unused by AvgPool BW; retained for API symmetry).
    pub y: TensorRef<'a, T, 5>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, 5>,
    /// Saved FW input (unused by AvgPool BW; retained for API symmetry).
    pub x: TensorRef<'a, T, 5>,
    /// Output gradient. Zeroed internally before atomic-scatter.
    pub dx: TensorMut<'a, T, 5>,
}

/// Adaptive 3-D average-pool plan (bit-exact PyTorch, bespoke kernel).
pub struct AdaptiveAvgPool3dPlan<T: Element> {
    desc: AdaptivePool3dDescriptor,
    sku: KernelSku,
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
        let sku = build_sku::<T>(PoolKind::AdaptiveAvgPool3d);
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
        let status = dispatch_avg_fw::<T>(
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
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let nc = self.desc.batch * self.desc.channels;
        let status = dispatch_avg_bw::<T>(
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
