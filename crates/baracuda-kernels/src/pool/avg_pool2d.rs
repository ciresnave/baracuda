//! AvgPool2d — NCHW 2-D average-pool via cuDNN's legacy descriptor API.
//!
//! Implements forward + backward over the four floating-point dtypes
//! (`f32`, `f64`, `f16`, `bf16`). Drives the same cuDNN exec functions
//! as the sibling [`super::MaxPool2dPlan`] but with one of the two
//! average-pool modes — selected via [`super::PoolMode`] on the
//! descriptor:
//!
//! - [`super::PoolMode::AvgExcludePad`] — PyTorch `nn.AvgPool2d` default
//!   (`count_include_pad=False`). Divides each output cell by the count
//!   of *valid* (non-padded) cells in the corresponding window.
//! - [`super::PoolMode::AvgIncludePad`] — TensorFlow-style. Divides by
//!   the full `window_h * window_w` (zero-padded cells included).
//!
//! Pooling is workspace-free in cuDNN's legacy API. See [`super`] for
//! the plan-level docs (handle ownership, descriptor lifecycle, dtype
//! coverage).

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cudnnCreate, cudnnHandle_t, cudnnPoolingDescriptor_t, cudnnSetStream,
    cudnnTensorDescriptor_t,
};
use baracuda_kernels_types::{
    Element, KernelSku, PlanPreference, PoolKind, PrecisionGuarantee, Workspace,
};

use super::max_pool2d::{
    build_sku, check_bw_args, check_fw_args, drop_pool_descriptors, ensure_pool_descriptors,
    run_bw_inner, run_fw_inner, validate_descriptor, Pool2dBwArgs, Pool2dDescriptor, Pool2dFwArgs,
    PoolMode,
};

/// 2-D average-pool plan (cuDNN-backed) — forward + backward over NCHW.
///
/// `select` requires `descriptor.mode == PoolMode::AvgIncludePad` or
/// `AvgExcludePad`. Defaults to PyTorch's `count_include_pad=False`
/// semantics: set [`Pool2dDescriptor::mode`] to
/// [`PoolMode::AvgExcludePad`] for PyTorch parity, or
/// [`PoolMode::AvgIncludePad`] for TensorFlow-style.
///
/// **When to use**: average-pool layer in a CNN. Sibling plan to
/// [`super::MaxPool2dPlan`]; they share descriptor + args types.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16`.
///
/// **Shape**: `[N, C, H_in, W_in]` → `[N, C, H_out, W_out]` with the
/// standard floor formula. NCHW only.
///
/// **Workspace**: zero (cuDNN pooling is workspace-free).
///
/// **Backward semantics**: callers must retain `y` (saved FW output)
/// and `x` (saved FW input) — cuDNN demands both for API uniformity,
/// though avg-pool's gradient depends only on `x` mathematically.
///
/// **Precision guarantee**: deterministic; not bit-stable across runs.
///
/// Owns one `cudnnHandle_t` + three lazy descriptors (`!Sync` /
/// `!Send`); released on `Drop`. Gated under `feature = "cudnn"`.
pub struct AvgPool2dPlan<T: Element> {
    desc: Pool2dDescriptor,
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    x_desc: Cell<cudnnTensorDescriptor_t>,
    y_desc: Cell<cudnnTensorDescriptor_t>,
    pool_desc: Cell<cudnnPoolingDescriptor_t>,
    _marker: PhantomData<T>,
}

impl<T: Element> AvgPool2dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    ///
    /// Requires `desc.mode` to be one of [`PoolMode::AvgIncludePad`] or
    /// [`PoolMode::AvgExcludePad`]. For max-pool use
    /// [`super::MaxPool2dPlan`] instead.
    pub fn select(
        _stream: &Stream,
        desc: &Pool2dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        let op = match desc.mode {
            PoolMode::AvgIncludePad => PoolKind::AvgPool2dIncludePad,
            PoolMode::AvgExcludePad => PoolKind::AvgPool2dExcludePad,
            PoolMode::Max => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::AvgPool2dPlan: descriptor.mode must be one of \
                     PoolMode::AvgIncludePad | AvgExcludePad — use MaxPool2dPlan for max",
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

    /// `(H_out, W_out)` output spatial extents.
    #[inline]
    pub fn output_dims(&self) -> (i32, i32) {
        super::max_pool2d::compute_output_dims(&self.desc)
    }

    /// Run the forward pass. Computes `y := avg_pool(x)` (alpha = 1,
    /// beta = 0).
    pub fn run_fw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: Pool2dFwArgs<'_, T>,
    ) -> Result<()> {
        check_fw_args(&self.desc, &args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;
        run_fw_inner::<T>(
            h,
            self.pool_desc.get(),
            self.x_desc.get(),
            self.y_desc.get(),
            args.x.data.as_raw().0,
            args.y.data.as_raw().0,
        )
    }

    /// Run the backward pass. Computes `dx := avg_pool_grad(y, dy, x)`.
    pub fn run_bw(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: Pool2dBwArgs<'_, T>,
    ) -> Result<()> {
        check_bw_args(&self.desc, &args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;
        run_bw_inner::<T>(
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

    // ------------------------------------------------------------------
    // Internal: lazy handle + descriptors
    // ------------------------------------------------------------------

    fn ensure_handle(&self) -> Result<cudnnHandle_t> {
        let h = self.handle.get();
        if !h.is_null() {
            return Ok(h);
        }
        let mut handle: cudnnHandle_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreate(&mut handle as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.handle.set(handle);
        Ok(handle)
    }

    fn bind_stream(&self, h: cudnnHandle_t, stream: &Stream) -> Result<()> {
        let status = unsafe { cudnnSetStream(h, stream.as_raw() as *mut c_void) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    fn ensure_descriptors(&self) -> Result<()> {
        ensure_pool_descriptors::<T>(
            &self.desc,
            &self.x_desc,
            &self.y_desc,
            &self.pool_desc,
        )
    }
}

impl<T: Element> Drop for AvgPool2dPlan<T> {
    fn drop(&mut self) {
        drop_pool_descriptors(&self.x_desc, &self.y_desc, &self.pool_desc, &self.handle);
    }
}
