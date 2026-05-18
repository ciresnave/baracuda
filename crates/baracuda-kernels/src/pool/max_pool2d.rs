//! MaxPool2d — NCHW 2-D max-pool via cuDNN's legacy descriptor API.
//!
//! Implements forward + backward over the four floating-point dtypes
//! (`f32`, `f64`, `f16`, `bf16`). The plan is workspace-free: cuDNN's
//! pooling kernel allocates its tiny internal scratch itself.
//!
//! This file also hosts the shared [`Pool2dDescriptor`] / [`Pool2dFwArgs`]
//! / [`Pool2dBwArgs`] / [`PoolMode`] types used by the sibling
//! [`super::avg_pool2d::AvgPool2dPlan`] — the descriptor / args shapes
//! are identical between max and average pooling.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cudnnCreate, cudnnCreatePoolingDescriptor, cudnnCreateTensorDescriptor, cudnnDestroy,
    cudnnDestroyPoolingDescriptor, cudnnDestroyTensorDescriptor, cudnnHandle_t,
    cudnnPoolingBackward, cudnnPoolingDescriptor_t, cudnnPoolingForward,
    cudnnSetPooling2dDescriptor, cudnnSetStream, cudnnSetTensor4dDescriptor,
    cudnnTensorDescriptor_t, CUDNN_DATA_BFLOAT16, CUDNN_DATA_DOUBLE, CUDNN_DATA_FLOAT,
    CUDNN_DATA_HALF, CUDNN_NOT_PROPAGATE_NAN, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_MAX, CUDNN_TENSOR_NCHW,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PoolKind, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Pooling mode — picks between max-pool and the two flavors of
/// average-pool that cuDNN exposes.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PoolMode {
    /// Max-pool: `y[..., i, j] = max over window`. PyTorch
    /// `nn.MaxPool2d`. Carried on a [`Pool2dDescriptor`] consumed by
    /// [`super::MaxPool2dPlan`].
    Max,
    /// Average-pool with **count-include-padding** denominator (divide
    /// by full `window_h * window_w`). Matches cuDNN's
    /// `*_COUNT_INCLUDE_PADDING` mode and TensorFlow's default.
    AvgIncludePad,
    /// Average-pool with **count-exclude-padding** denominator (divide
    /// only by the number of valid, non-padded cells in each window).
    /// PyTorch `nn.AvgPool2d` default (`count_include_pad=False`).
    AvgExcludePad,
}

/// Descriptor for a 2-D pooling op over NCHW tensors.
///
/// Input shape: `[batch, channels, h_in, w_in]`. Output shape:
/// `[batch, channels, h_out, w_out]` where
/// `h_out = floor((h_in + 2·pad_h - window_h) / stride_h) + 1` and
/// similarly for `w_out`.
///
/// Used by both [`super::MaxPool2dPlan`] and
/// [`super::AvgPool2dPlan`]; the [`Self::mode`] field selects which
/// cuDNN pooling exec path to drive. [`super::MaxPool2dPlan::select`]
/// requires `mode == PoolMode::Max`; [`super::AvgPool2dPlan::select`]
/// requires `mode == PoolMode::AvgIncludePad` or `AvgExcludePad`.
#[derive(Copy, Clone, Debug)]
pub struct Pool2dDescriptor {
    /// Batch size `N`.
    pub batch: i32,
    /// Channel count `C`. Pooling is per-channel — no cross-channel
    /// reductions.
    pub channels: i32,
    /// Input height `H_in`.
    pub h_in: i32,
    /// Input width `W_in`.
    pub w_in: i32,
    /// Pooling window height.
    pub window_h: i32,
    /// Pooling window width.
    pub window_w: i32,
    /// Zero-padding rows on each side of the input height axis.
    pub pad_h: i32,
    /// Zero-padding columns on each side of the input width axis.
    pub pad_w: i32,
    /// Stride along the height axis.
    pub stride_h: i32,
    /// Stride along the width axis.
    pub stride_w: i32,
    /// Pooling flavor — selects the cuDNN exec path.
    pub mode: PoolMode,
    /// Element dtype. Must be `F32`, `F64`, `F16`, or `Bf16`.
    pub element: ElementKind,
}

/// Args bundle for a 2-D pooling forward launch (shared between max and
/// average pooling).
pub struct Pool2dFwArgs<'a, T: Element> {
    /// Input activations `[N, C, H_in, W_in]` NCHW contiguous.
    pub x: TensorRef<'a, T, 4>,
    /// Output activations `[N, C, H_out, W_out]` NCHW contiguous.
    pub y: TensorMut<'a, T, 4>,
}

/// Args bundle for a 2-D pooling backward launch.
///
/// **Both `y` (saved forward output) and `x` (saved forward input) must
/// be retained from the FW launch.** cuDNN uses both to recover the
/// per-window argmax for max-pool; avg-pool uses only `x` mathematically
/// but cuDNN's API still demands `y` for uniformity.
pub struct Pool2dBwArgs<'a, T: Element> {
    /// Saved forward output `[N, C, H_out, W_out]`.
    pub y: TensorRef<'a, T, 4>,
    /// Upstream gradient `[N, C, H_out, W_out]` (matches `y` shape).
    pub dy: TensorRef<'a, T, 4>,
    /// Saved forward input `[N, C, H_in, W_in]`.
    pub x: TensorRef<'a, T, 4>,
    /// Output gradient w.r.t. input `[N, C, H_in, W_in]`. Fully
    /// overwritten by the launch (alpha = 1, beta = 0).
    pub dx: TensorMut<'a, T, 4>,
}

/// 2-D max-pool plan (cuDNN-backed) — forward + backward over NCHW.
///
/// `select` requires `descriptor.mode == PoolMode::Max`.
///
/// **When to use**: standard max-pool layer in a CNN. Use
/// [`super::AvgPool2dPlan`] for average-pool. Pair `run_fw` + `run_bw`
/// over the same plan instance for autograd.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16` — cuDNN's full FP coverage
/// for pooling.
///
/// **Shape**: `[N, C, H_in, W_in]` → `[N, C, H_out, W_out]` with
/// `H_out = floor((H_in + 2·pad_h - window_h) / stride_h) + 1`. NCHW
/// only (NHWC is a fanout milestone). No `ceil_mode` knob.
///
/// **Workspace**: zero — cuDNN's pooling kernel is workspace-free.
/// Pass [`Workspace::None`] (the plan accepts any `Workspace<'_>` for
/// API uniformity but never reads from it).
///
/// **Backward semantics**: callers must retain both `y` (saved FW
/// output) and `x` (saved FW input) — cuDNN's pooling-BW API requires
/// both to recover the per-window argmax internally.
///
/// **Precision guarantee**: deterministic; cuDNN may re-order
/// reductions across runs so not bit-stable.
///
/// Owns one `cudnnHandle_t` + three lazy descriptors (`!Sync` /
/// `!Send`); released on `Drop`. Gated under `feature = "cudnn"`.
pub struct MaxPool2dPlan<T: Element> {
    desc: Pool2dDescriptor,
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    x_desc: Cell<cudnnTensorDescriptor_t>,
    y_desc: Cell<cudnnTensorDescriptor_t>,
    pool_desc: Cell<cudnnPoolingDescriptor_t>,
    _marker: PhantomData<T>,
}

impl<T: Element> MaxPool2dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &Pool2dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        validate_descriptor::<T>(desc)?;
        if !matches!(desc.mode, PoolMode::Max) {
            return Err(Error::Unsupported(
                "baracuda-kernels::MaxPool2dPlan: descriptor.mode must be PoolMode::Max",
            ));
        }
        let sku = build_sku::<T>(PoolKind::MaxPool2d);
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

    /// Workspace size in bytes. Always `0` — cuDNN's pooling kernel
    /// allocates its tiny internal scratch itself.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// `(H_out, W_out)` output spatial extents under the configured
    /// window / pad / stride.
    #[inline]
    pub fn output_dims(&self) -> (i32, i32) {
        compute_output_dims(&self.desc)
    }

    /// Run the forward pass. Computes `y := max_pool(x)` (alpha = 1,
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

    /// Run the backward pass. Computes `dx := max_pool_grad(y, dy, x)`.
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

impl<T: Element> Drop for MaxPool2dPlan<T> {
    fn drop(&mut self) {
        drop_pool_descriptors(&self.x_desc, &self.y_desc, &self.pool_desc, &self.handle);
    }
}

// =============================================================================
// Shared helpers — also used by the sibling avg_pool2d module
// =============================================================================

/// Common descriptor validation: dtype gating + non-negative shape /
/// padding / window / stride. Used by both `MaxPool2dPlan::select` and
/// `AvgPool2dPlan::select`.
pub(crate) fn validate_descriptor<T: Element>(desc: &Pool2dDescriptor) -> Result<()> {
    if desc.element != T::KIND {
        return Err(Error::Unsupported(
            "baracuda-kernels::Pool2dPlan: descriptor.element != T::KIND",
        ));
    }
    if !matches!(
        T::KIND,
        ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
    ) {
        return Err(Error::Unsupported(
            "baracuda-kernels::Pool2dPlan: cuDNN pooling supports f32 / f64 / f16 / bf16",
        ));
    }
    if desc.batch <= 0 || desc.channels <= 0 || desc.h_in <= 0 || desc.w_in <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: input shape extents must be > 0",
        ));
    }
    if desc.window_h <= 0 || desc.window_w <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: window extents must be > 0",
        ));
    }
    if desc.stride_h <= 0 || desc.stride_w <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: stride must be > 0",
        ));
    }
    if desc.pad_h < 0 || desc.pad_w < 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: padding must be >= 0",
        ));
    }
    // cuDNN rejects `pad > window/2`; mirror that check at the safe
    // layer for an earlier / clearer error.
    if desc.pad_h * 2 > desc.window_h || desc.pad_w * 2 > desc.window_w {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: padding must be <= window / 2",
        ));
    }
    let (h_out, w_out) = compute_output_dims(desc);
    if h_out <= 0 || w_out <= 0 {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: computed output dims <= 0 — \
             window / stride / pad combo produces an empty output",
        ));
    }
    Ok(())
}

/// Build the `KernelSku` for a pooling plan with the given op
/// discriminant.
pub(crate) fn build_sku<T: Element>(op: PoolKind) -> KernelSku {
    let math_precision = match T::KIND {
        ElementKind::F64 => MathPrecision::F64,
        ElementKind::F16 => MathPrecision::F16,
        ElementKind::Bf16 => MathPrecision::Bf16,
        _ => MathPrecision::F32,
    };
    let accumulator = match T::KIND {
        ElementKind::F64 => ElementKind::F64,
        _ => ElementKind::F32,
    };
    let precision_guarantee = PrecisionGuarantee {
        math_precision,
        accumulator,
        // cuDNN does not contractually guarantee bit-stable pooling
        // across runs (the non-deterministic max-pool kernel can use
        // racing argmax; avg-pool uses a parallel reduction). The
        // `MaxPool2dPlan` could be tightened by switching to
        // `CUDNN_POOLING_MAX_DETERMINISTIC` — deferred.
        bit_stable_on_same_hardware: false,
        deterministic: true,
    };
    KernelSku {
        category: OpCategory::Pooling,
        op: op as u16,
        element: T::KIND,
        aux_element: None,
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Cudnn,
        precision_guarantee,
    }
}

/// `H_out = floor((H_in + 2·pad_h - window_h) / stride_h) + 1`; same
/// for `W_out`.
#[inline]
pub(crate) fn compute_output_dims(d: &Pool2dDescriptor) -> (i32, i32) {
    let h_out = (d.h_in + 2 * d.pad_h - d.window_h) / d.stride_h + 1;
    let w_out = (d.w_in + 2 * d.pad_w - d.window_w) / d.stride_w + 1;
    (h_out, w_out)
}

#[inline]
pub(crate) fn cudnn_dtype<T: Element>() -> i32 {
    match T::KIND {
        ElementKind::F32 => CUDNN_DATA_FLOAT,
        ElementKind::F64 => CUDNN_DATA_DOUBLE,
        ElementKind::F16 => CUDNN_DATA_HALF,
        ElementKind::Bf16 => CUDNN_DATA_BFLOAT16,
        _ => unreachable!("Pool2dPlan::select gates on F32/F64/F16/Bf16"),
    }
}

#[inline]
pub(crate) fn is_double_compute<T: Element>() -> bool {
    matches!(T::KIND, ElementKind::F64)
}

#[inline]
pub(crate) fn cudnn_pool_mode(mode: PoolMode) -> i32 {
    match mode {
        PoolMode::Max => CUDNN_POOLING_MAX,
        PoolMode::AvgIncludePad => CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        PoolMode::AvgExcludePad => CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
    }
}

pub(crate) fn check_fw_args<T: Element>(
    desc: &Pool2dDescriptor,
    args: &Pool2dFwArgs<'_, T>,
) -> Result<()> {
    let (h_out, w_out) = compute_output_dims(desc);
    let x_shape = [desc.batch, desc.channels, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, h_out, w_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: x shape != [N, C, H_in, W_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: y shape != [N, C, H_out, W_out]",
        ));
    }
    Ok(())
}

pub(crate) fn check_bw_args<T: Element>(
    desc: &Pool2dDescriptor,
    args: &Pool2dBwArgs<'_, T>,
) -> Result<()> {
    let (h_out, w_out) = compute_output_dims(desc);
    let x_shape = [desc.batch, desc.channels, desc.h_in, desc.w_in];
    let y_shape = [desc.batch, desc.channels, h_out, w_out];
    if args.x.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: x shape != [N, C, H_in, W_in]",
        ));
    }
    if args.dx.shape != x_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: dx shape != [N, C, H_in, W_in]",
        ));
    }
    if args.y.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: y shape != [N, C, H_out, W_out]",
        ));
    }
    if args.dy.shape != y_shape {
        return Err(Error::InvalidProblem(
            "baracuda-kernels::Pool2dPlan: dy shape != [N, C, H_out, W_out]",
        ));
    }
    Ok(())
}

/// Allocate (once) the three cuDNN descriptors `(x_desc, y_desc,
/// pool_desc)`. Idempotent — subsequent calls are a no-op.
pub(crate) fn ensure_pool_descriptors<T: Element>(
    desc: &Pool2dDescriptor,
    x_desc: &Cell<cudnnTensorDescriptor_t>,
    y_desc: &Cell<cudnnTensorDescriptor_t>,
    pool_desc: &Cell<cudnnPoolingDescriptor_t>,
) -> Result<()> {
    if !x_desc.get().is_null() {
        return Ok(());
    }
    let dt = cudnn_dtype::<T>();
    let (h_out, w_out) = compute_output_dims(desc);

    // x descriptor.
    let mut xd: cudnnTensorDescriptor_t = core::ptr::null_mut();
    let status = unsafe { cudnnCreateTensorDescriptor(&mut xd as *mut _) };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    let status = unsafe {
        cudnnSetTensor4dDescriptor(
            xd,
            CUDNN_TENSOR_NCHW,
            dt,
            desc.batch,
            desc.channels,
            desc.h_in,
            desc.w_in,
        )
    };
    if status != 0 {
        unsafe {
            let _ = cudnnDestroyTensorDescriptor(xd);
        }
        return Err(Error::CutlassInternal(-status));
    }
    x_desc.set(xd);

    // y descriptor.
    let mut yd: cudnnTensorDescriptor_t = core::ptr::null_mut();
    let status = unsafe { cudnnCreateTensorDescriptor(&mut yd as *mut _) };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    let status = unsafe {
        cudnnSetTensor4dDescriptor(
            yd,
            CUDNN_TENSOR_NCHW,
            dt,
            desc.batch,
            desc.channels,
            h_out,
            w_out,
        )
    };
    if status != 0 {
        unsafe {
            let _ = cudnnDestroyTensorDescriptor(yd);
        }
        return Err(Error::CutlassInternal(-status));
    }
    y_desc.set(yd);

    // Pooling descriptor.
    let mut pd: cudnnPoolingDescriptor_t = core::ptr::null_mut();
    let status = unsafe { cudnnCreatePoolingDescriptor(&mut pd as *mut _) };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    let status = unsafe {
        cudnnSetPooling2dDescriptor(
            pd,
            cudnn_pool_mode(desc.mode),
            CUDNN_NOT_PROPAGATE_NAN,
            desc.window_h,
            desc.window_w,
            desc.pad_h,
            desc.pad_w,
            desc.stride_h,
            desc.stride_w,
        )
    };
    if status != 0 {
        unsafe {
            let _ = cudnnDestroyPoolingDescriptor(pd);
        }
        return Err(Error::CutlassInternal(-status));
    }
    pool_desc.set(pd);
    Ok(())
}

pub(crate) fn drop_pool_descriptors(
    x_desc: &Cell<cudnnTensorDescriptor_t>,
    y_desc: &Cell<cudnnTensorDescriptor_t>,
    pool_desc: &Cell<cudnnPoolingDescriptor_t>,
    handle: &Cell<cudnnHandle_t>,
) {
    let pd = pool_desc.get();
    if !pd.is_null() {
        unsafe {
            let _ = cudnnDestroyPoolingDescriptor(pd);
        }
        pool_desc.set(core::ptr::null_mut());
    }
    let yd = y_desc.get();
    if !yd.is_null() {
        unsafe {
            let _ = cudnnDestroyTensorDescriptor(yd);
        }
        y_desc.set(core::ptr::null_mut());
    }
    let xd = x_desc.get();
    if !xd.is_null() {
        unsafe {
            let _ = cudnnDestroyTensorDescriptor(xd);
        }
        x_desc.set(core::ptr::null_mut());
    }
    let h = handle.get();
    if !h.is_null() {
        unsafe {
            let _ = cudnnDestroy(h);
        }
        handle.set(core::ptr::null_mut());
    }
}

/// Drive `cudnnPoolingForward` with appropriate alpha/beta scalar dtype.
pub(crate) fn run_fw_inner<T: Element>(
    h: cudnnHandle_t,
    pool_desc: cudnnPoolingDescriptor_t,
    x_desc: cudnnTensorDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    x_ptr: u64,
    y_ptr: u64,
) -> Result<()> {
    let status = if is_double_compute::<T>() {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        unsafe {
            cudnnPoolingForward(
                h,
                pool_desc,
                &alpha as *const f64 as *const c_void,
                x_desc,
                x_ptr as *const c_void,
                &beta as *const f64 as *const c_void,
                y_desc,
                y_ptr as *mut c_void,
            )
        }
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cudnnPoolingForward(
                h,
                pool_desc,
                &alpha as *const f32 as *const c_void,
                x_desc,
                x_ptr as *const c_void,
                &beta as *const f32 as *const c_void,
                y_desc,
                y_ptr as *mut c_void,
            )
        }
    };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    Ok(())
}

/// Drive `cudnnPoolingBackward` with appropriate alpha/beta scalar dtype.
pub(crate) fn run_bw_inner<T: Element>(
    h: cudnnHandle_t,
    pool_desc: cudnnPoolingDescriptor_t,
    x_desc: cudnnTensorDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    y_ptr: u64,
    dy_ptr: u64,
    x_ptr: u64,
    dx_ptr: u64,
) -> Result<()> {
    let status = if is_double_compute::<T>() {
        let alpha: f64 = 1.0;
        let beta: f64 = 0.0;
        unsafe {
            cudnnPoolingBackward(
                h,
                pool_desc,
                &alpha as *const f64 as *const c_void,
                y_desc,
                y_ptr as *const c_void,
                y_desc,
                dy_ptr as *const c_void,
                x_desc,
                x_ptr as *const c_void,
                &beta as *const f64 as *const c_void,
                x_desc,
                dx_ptr as *mut c_void,
            )
        }
    } else {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        unsafe {
            cudnnPoolingBackward(
                h,
                pool_desc,
                &alpha as *const f32 as *const c_void,
                y_desc,
                y_ptr as *const c_void,
                y_desc,
                dy_ptr as *const c_void,
                x_desc,
                x_ptr as *const c_void,
                &beta as *const f32 as *const c_void,
                x_desc,
                dx_ptr as *mut c_void,
            )
        }
    };
    if status != 0 {
        return Err(Error::CutlassInternal(-status));
    }
    Ok(())
}
