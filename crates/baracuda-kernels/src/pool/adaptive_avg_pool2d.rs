//! AdaptiveAvgPool2d — NCHW adaptive average-pool, bit-exact PyTorch
//! (Phase 16.1 bespoke kernel).
//!
//! Replaces the Phase 11.8 cuDNN-approximation path. PyTorch's
//! non-uniform per-output-cell window formula is applied independently
//! per spatial axis:
//!
//! ```text
//! start_i = floor(i * in / out)
//! end_i   = ceil((i + 1) * in / out)
//! ```
//!
//! See [`super::adaptive_avg_pool1d`] for the full algorithmic / BW
//! / determinism story.

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

/// Descriptor for an adaptive 2-D pooling op over NCHW tensors.
#[derive(Copy, Clone, Debug)]
pub struct AdaptivePool2dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input height `H_in`.
    pub h_in: i32,
    /// Input width `W_in`.
    pub w_in: i32,
    /// Desired output height.
    pub h_out: i32,
    /// Desired output width.
    pub w_out: i32,
    /// Element dtype.
    pub element: ElementKind,
}

/// FW args.
pub struct AdaptivePool2dFwArgs<'a, T: Element> {
    /// Input `[N, C, H_in, W_in]`.
    pub x: TensorRef<'a, T, 4>,
    /// Output `[N, C, H_out, W_out]`.
    pub y: TensorMut<'a, T, 4>,
}

/// BW args. `y` retained for API symmetry with [`super::AdaptiveMaxPool2dPlan`]
/// (MaxPool BW uses `x`; AvgPool BW uses neither — only `dy` and `dx`).
pub struct AdaptivePool2dBwArgs<'a, T: Element> {
    /// Saved FW output (unused by AvgPool BW; retained for API symmetry).
    pub y: TensorRef<'a, T, 4>,
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, 4>,
    /// Saved FW input (unused by AvgPool BW; retained for API symmetry).
    pub x: TensorRef<'a, T, 4>,
    /// Output gradient. Fully overwritten by the launch (kernel zeros
    /// internally before atomic-scattering into it).
    pub dx: TensorMut<'a, T, 4>,
}

/// Adaptive 2-D average-pool plan (bit-exact PyTorch, bespoke kernel).
pub struct AdaptiveAvgPool2dPlan<T: Element> {
    desc: AdaptivePool2dDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> AdaptiveAvgPool2dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &AdaptivePool2dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        let sku = build_sku::<T>(PoolKind::AdaptiveAvgPool2d);
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

    /// Deprecated. Always returns `((0, 0), (0, 0))` — see
    /// [`super::AdaptiveAvgPool1dPlan::derived_kernel_stride`].
    #[inline]
    #[deprecated(
        since = "0.0.1-alpha.33",
        note = "Phase 16.1 uses bit-exact per-output-cell windows; no single (kernel, stride) pair applies."
    )]
    pub fn derived_kernel_stride(&self) -> ((i32, i32), (i32, i32)) {
        ((0, 0), (0, 0))
    }

    /// Run the forward pass.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AdaptivePool2dFwArgs<'_, T>,
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
            2, // spatial_rank
            1, self.desc.h_in, self.desc.w_in,
            1, self.desc.h_out, self.desc.w_out,
            stream_ptr,
        );
        map_status(status)
    }

    /// Run the backward pass.
    pub fn run_bw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AdaptivePool2dBwArgs<'_, T>,
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
            2,
            1, self.desc.h_in, self.desc.w_in,
            1, self.desc.h_out, self.desc.w_out,
            stream_ptr,
        );
        map_status(status)
    }
}

// =============================================================================
// Shared adaptive-2d helpers (used by sibling adaptive_max_pool2d)
// =============================================================================

pub(crate) fn validate_descriptor<T: Element>(
    desc: &AdaptivePool2dDescriptor,
) -> Result<()> {
    validate_dtype::<T>()?;
    if desc.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::AdaptivePool2dPlan: descriptor.element != T::KIND",
        ));
    }
    if desc.batch <= 0
        || desc.channels <= 0
        || desc.h_in <= 0
        || desc.w_in <= 0
        || desc.h_out <= 0
        || desc.w_out <= 0
    {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool2dPlan: extents must be > 0",
        ));
    }
    if desc.h_out > desc.h_in || desc.w_out > desc.w_in {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool2dPlan: output extents must be <= input \
             extents (upsampling is not a pooling op)",
        ));
    }
    Ok(())
}

pub(crate) fn check_fw_args<T: Element>(
    desc: &AdaptivePool2dDescriptor,
    args: &AdaptivePool2dFwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, desc.h_out, desc.w_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool2dPlan: x shape != [N, C, H_in, W_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool2dPlan: y shape != [N, C, H_out, W_out]",
        ));
    }
    Ok(())
}

pub(crate) fn check_bw_args<T: Element>(
    desc: &AdaptivePool2dDescriptor,
    args: &AdaptivePool2dBwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, desc.h_out, desc.w_out];
    if args.x.shape != x_shape || args.dx.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool2dPlan: x/dx shape != [N, C, H_in, W_in]",
        ));
    }
    if args.y.shape != y_shape || args.dy.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool2dPlan: y/dy shape != [N, C, H_out, W_out]",
        ));
    }
    Ok(())
}
