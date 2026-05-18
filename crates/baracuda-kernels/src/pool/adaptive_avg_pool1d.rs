//! AdaptiveAvgPool1d — NCL adaptive average-pool via cuDNN.
//!
//! **APPROXIMATION**: cuDNN does not have a direct adaptive-pool entry
//! point. This plan derives `(kernel, stride)` from `(input_size,
//! output_size)` using PyTorch's common-case formula:
//!
//! ```text
//! kernel = ceil(input_size / output_size)
//! stride = floor(input_size / output_size)
//! padding = 0
//! ```
//!
//! and then drives a regular average-pool (count-exclude-padding) over
//! the implied uniform window. **This is NOT bit-exact PyTorch
//! `nn.AdaptiveAvgPool1d`** when `input_size % output_size != 0` —
//! PyTorch's reference implementation uses *non-uniform* per-output-cell
//! kernel sizes on the boundary so that the entire input range is
//! tiled exactly. The uniform-kernel approximation here matches PyTorch
//! exactly when `input_size % output_size == 0` and degrades gracefully
//! (within ±1 input cell of the true window) otherwise.
//!
//! For a true bit-exact adaptive pool, a bespoke kernel is needed —
//! deferred to a future milestone. File a feature request if needed.

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

/// Descriptor for an adaptive 1-D pooling op over NCL tensors.
///
/// Input `[N, C, L_in]` → output `[N, C, L_out]` for caller-supplied
/// `L_out`. The plan internally computes the cuDNN-uniform kernel /
/// stride (see module docs).
#[derive(Copy, Clone, Debug)]
pub struct AdaptivePool1dDescriptor {
    /// Batch `N`.
    pub batch: i32,
    /// Channels `C`.
    pub channels: i32,
    /// Input length `L_in`.
    pub l_in: i32,
    /// Desired output length `L_out`.
    pub l_out: i32,
    /// Element dtype.
    pub element: ElementKind,
}

/// FW args (shape `[N, C, L_in]` → `[N, C, L_out]`).
pub struct AdaptivePool1dFwArgs<'a, T: Element> {
    /// Input `[N, C, L_in]`.
    pub x: TensorRef<'a, T, 3>,
    /// Output `[N, C, L_out]`.
    pub y: TensorMut<'a, T, 3>,
}

/// BW args (cuDNN's pool-bw requires both `y` and `x`).
pub struct AdaptivePool1dBwArgs<'a, T: Element> {
    /// Saved FW output `[N, C, L_out]`.
    pub y: TensorRef<'a, T, 3>,
    /// Upstream gradient `[N, C, L_out]`.
    pub dy: TensorRef<'a, T, 3>,
    /// Saved FW input `[N, C, L_in]`.
    pub x: TensorRef<'a, T, 3>,
    /// Output gradient `[N, C, L_in]`.
    pub dx: TensorMut<'a, T, 3>,
}

/// Adaptive 1-D average-pool plan (cuDNN approximation; see module
/// docs for the bit-exact-PyTorch caveat).
pub struct AdaptiveAvgPool1dPlan<T: Element> {
    desc: AdaptivePool1dDescriptor,
    /// Cached `(kernel, stride)` derived from the descriptor at
    /// `select` time.
    derived: (i32, i32),
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    x_desc: Cell<cudnnTensorDescriptor_t>,
    y_desc: Cell<cudnnTensorDescriptor_t>,
    pool_desc: Cell<cudnnPoolingDescriptor_t>,
    _marker: PhantomData<T>,
}

impl<T: Element> AdaptiveAvgPool1dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    ///
    /// Returns `Error::Unsupported` if the dtype isn't `f32` / `f64` /
    /// `f16` / `bf16`, or `Error::InvalidProblem` if `l_in <= 0` /
    /// `l_out <= 0` / `l_out > l_in`.
    pub fn select(
        _stream: &Stream,
        desc: &AdaptivePool1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        let derived = adaptive_kernel_stride(desc.l_in, desc.l_out);
        let sku = build_sku::<T>(PoolKind::AdaptiveAvgPool1d);
        Ok(Self {
            desc: *desc,
            derived,
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

    /// Internal `(kernel, stride)` derived from the descriptor. Exposed
    /// for visibility into the cuDNN-approximation behavior — not part
    /// of any stable contract.
    #[inline]
    pub fn derived_kernel_stride(&self) -> (i32, i32) {
        self.derived
    }

    /// Run the forward pass.
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: AdaptivePool1dFwArgs<'_, T>,
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
        args: AdaptivePool1dBwArgs<'_, T>,
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
        let x_dims = [self.desc.batch, self.desc.channels, self.desc.l_in];
        let y_dims = [self.desc.batch, self.desc.channels, self.desc.l_out];
        let window = [self.derived.0];
        let padding = [0i32];
        let stride = [self.derived.1];
        ensure_descriptors_nd::<T>(
            &x_dims,
            &y_dims,
            &window,
            &padding,
            &stride,
            PoolMode::AvgExcludePad,
            &self.x_desc,
            &self.y_desc,
            &self.pool_desc,
        )
    }
}

impl<T: Element> Drop for AdaptiveAvgPool1dPlan<T> {
    fn drop(&mut self) {
        drop_descriptors_nd(&self.x_desc, &self.y_desc, &self.pool_desc, &self.handle);
    }
}

// =============================================================================
// Shared adaptive-1d helpers (used by sibling adaptive_max_pool1d)
// =============================================================================

pub(crate) fn validate_descriptor<T: Element>(
    desc: &AdaptivePool1dDescriptor,
) -> Result<()> {
    validate_dtype::<T>()?;
    if desc.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::AdaptivePool1dPlan: descriptor.element != T::KIND",
        ));
    }
    if desc.batch <= 0 || desc.channels <= 0 || desc.l_in <= 0 || desc.l_out <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: extents must be > 0",
        ));
    }
    if desc.l_out > desc.l_in {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: l_out must be <= l_in (upsampling \
             is not a pooling op)",
        ));
    }
    Ok(())
}

pub(crate) fn check_fw_args<T: Element>(
    desc: &AdaptivePool1dDescriptor,
    args: &AdaptivePool1dFwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.l_in];
    let y_shape = [desc.batch, desc.channels, desc.l_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: x shape != [N, C, L_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: y shape != [N, C, L_out]",
        ));
    }
    Ok(())
}

pub(crate) fn check_bw_args<T: Element>(
    desc: &AdaptivePool1dDescriptor,
    args: &AdaptivePool1dBwArgs<'_, T>,
) -> Result<()> {
    let x_shape = [desc.batch, desc.channels, desc.l_in];
    let y_shape = [desc.batch, desc.channels, desc.l_out];
    if args.x.shape != x_shape || args.dx.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: x/dx shape != [N, C, L_in]",
        ));
    }
    if args.y.shape != y_shape || args.dy.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::AdaptivePool1dPlan: y/dy shape != [N, C, L_out]",
        ));
    }
    Ok(())
}
