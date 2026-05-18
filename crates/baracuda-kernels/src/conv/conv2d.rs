//! Conv2d — NCHW 2-D convolution via cuDNN's legacy descriptor API.
//!
//! Implements forward + both backward passes for 2-D convolution over
//! the four floating-point dtypes (`f32`, `f64`, `f16`, `bf16`). The
//! mathematical convention is **cross-correlation** (kernel applied
//! directly, not flipped) to match PyTorch / JAX. See [`super`] for the
//! plan-level docs (handle / descriptor / workspace ownership,
//! algorithm pinning, dtype coverage).

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter, cudnnConvolutionDescriptor_t,
    cudnnConvolutionForward, cudnnCreate, cudnnCreateConvolutionDescriptor,
    cudnnCreateFilterDescriptor, cudnnCreateTensorDescriptor, cudnnDestroy,
    cudnnDestroyConvolutionDescriptor, cudnnDestroyFilterDescriptor,
    cudnnDestroyTensorDescriptor, cudnnFilterDescriptor_t,
    cudnnGetConvolutionBackwardDataWorkspaceSize, cudnnGetConvolutionBackwardFilterWorkspaceSize,
    cudnnGetConvolutionForwardWorkspaceSize, cudnnHandle_t, cudnnSetConvolution2dDescriptor,
    cudnnSetFilter4dDescriptor, cudnnSetStream, cudnnSetTensor4dDescriptor,
    cudnnTensorDescriptor_t, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_BFLOAT16, CUDNN_DATA_DOUBLE, CUDNN_DATA_FLOAT,
    CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, ConvKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a 2-D convolution over NCHW tensors.
///
/// Input shape: `[batch, c_in, h_in, w_in]`. Filter shape: `[c_out,
/// c_in, h_filt, w_filt]`. The padding / stride / dilation extents
/// follow PyTorch's `nn.Conv2d` semantics (kernel applied directly —
/// **cross-correlation**, not mathematical convolution).
#[derive(Copy, Clone, Debug)]
pub struct Conv2dDescriptor {
    /// Batch size `N`.
    pub batch: i32,
    /// Input channel count `C_in`.
    pub c_in: i32,
    /// Input height `H_in`.
    pub h_in: i32,
    /// Input width `W_in`.
    pub w_in: i32,
    /// Output channel count `C_out`.
    pub c_out: i32,
    /// Filter height `H_filt`.
    pub h_filt: i32,
    /// Filter width `W_filt`.
    pub w_filt: i32,
    /// Zero-padding rows on each side of the input height axis.
    pub pad_h: i32,
    /// Zero-padding columns on each side of the input width axis.
    pub pad_w: i32,
    /// Stride along the height axis.
    pub stride_h: i32,
    /// Stride along the width axis.
    pub stride_w: i32,
    /// Dilation along the height axis (`1` for the "dense" default).
    pub dilation_h: i32,
    /// Dilation along the width axis.
    pub dilation_w: i32,
    /// Element dtype. Must be `F32`, `F64`, `F16`, or `Bf16`.
    pub element: ElementKind,
}

/// Args bundle for a Conv2d forward launch.
pub struct Conv2dArgs<'a, T: Element> {
    /// Input activations `[N, C_in, H_in, W_in]` NCHW contiguous.
    pub x: TensorRef<'a, T, 4>,
    /// Filter weights `[C_out, C_in, H_filt, W_filt]`.
    pub w: TensorRef<'a, T, 4>,
    /// Output activations `[N, C_out, H_out, W_out]` NCHW contiguous.
    pub y: TensorMut<'a, T, 4>,
}

/// Args bundle for a Conv2d data-gradient launch (BW w.r.t. the input
/// activations).
///
/// Computes `dx = conv_T(w, dy)`. Output `dx` matches the forward `x`
/// shape and is fully overwritten (alpha = 1, beta = 0).
pub struct Conv2dBwArgs<'a, T: Element> {
    /// Filter weights `[C_out, C_in, H_filt, W_filt]` — same tensor as
    /// in the forward pass.
    pub w: TensorRef<'a, T, 4>,
    /// Upstream gradient `[N, C_out, H_out, W_out]` (matches the forward
    /// `y` shape).
    pub dy: TensorRef<'a, T, 4>,
    /// Output gradient w.r.t. input `[N, C_in, H_in, W_in]`.
    pub dx: TensorMut<'a, T, 4>,
}

/// Args bundle for a Conv2d filter-gradient launch (BW w.r.t. the
/// filter weights).
///
/// Computes `dw = conv_grad(x, dy)`. Output `dw` matches the forward
/// `w` shape and is fully overwritten.
pub struct Conv2dDwArgs<'a, T: Element> {
    /// Input activations `[N, C_in, H_in, W_in]` — same tensor as in the
    /// forward pass.
    pub x: TensorRef<'a, T, 4>,
    /// Upstream gradient `[N, C_out, H_out, W_out]` (matches the forward
    /// `y` shape).
    pub dy: TensorRef<'a, T, 4>,
    /// Output gradient w.r.t. the filter `[C_out, C_in, H_filt, W_filt]`.
    pub dw: TensorMut<'a, T, 4>,
}

/// 2-D convolution plan (cuDNN-backed) — forward + both backward
/// passes for NCHW activations.
///
/// **When to use**: any 2-D convolution from a typical CNN
/// (Conv2d / Conv1d-as-Conv2d / dilated conv / strided conv).
/// Mathematical convention is cross-correlation to match PyTorch's
/// `nn.Conv2d` (kernel applied directly, not flipped). Run `run_fw`
/// for the forward, `run_bw_data` for `dx`, `run_bw_filter` for `dw`.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16` — the four cuDNN-supported
/// floating-point types. Accumulator is `f32` for `f32` / `f16` /
/// `bf16` and `f64` for `f64`, per cuDNN's mixed-precision
/// conventions.
///
/// **Shape**: input `[N, C_in, H_in, W_in]`, filter `[C_out, C_in,
/// H_filt, W_filt]`, output `[N, C_out, H_out, W_out]` with
/// `H_out`, `W_out` computed from pad / stride / dilation. NCHW only
/// (NHWC is a fanout milestone). Conv1d / Conv3d / transposed conv
/// are deferred.
///
/// **Workspace**: caller-provided (`Workspace::Borrowed`). Each
/// direction has its own size — call the relevant
/// `query_*_workspace_size(stream)` accessor before launching to
/// populate the cache. Running multiple directions over the same plan
/// should size the workspace at the max across directions.
///
/// **Algorithm pinning**: FW = `IMPLICIT_GEMM` (algo 0), BW-data /
/// BW-filter = `ALGO_1` (`IMPLICIT_PRECOMP_GEMM`). Heuristic search
/// via `cudnnGet*Algorithm_v7` is a perf-tuning follow-up.
///
/// **Precision guarantee**: deterministic per run, but **not**
/// bit-stable across runs — cuDNN's `IMPLICIT_GEMM` with tensor-core
/// paths can re-order reductions.
///
/// Owns one `cudnnHandle_t` + four lazy descriptors + per-direction
/// workspace-size caches (`!Sync` / `!Send`). All released on `Drop`.
///
/// Gated under `feature = "cudnn"` at the crate root.
pub struct Conv2dPlan<T: Element> {
    desc: Conv2dDescriptor,
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    x_desc: Cell<cudnnTensorDescriptor_t>,
    w_desc: Cell<cudnnFilterDescriptor_t>,
    y_desc: Cell<cudnnTensorDescriptor_t>,
    conv_desc: Cell<cudnnConvolutionDescriptor_t>,
    workspace_bytes_fw: Cell<usize>,
    workspace_bytes_bw_data: Cell<usize>,
    workspace_bytes_bw_filter: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> Conv2dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &Conv2dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::Conv2dPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::Conv2dPlan: cuDNN Conv2d supports f32 / f64 / f16 / bf16",
            ));
        }
        if desc.batch <= 0 || desc.c_in <= 0 || desc.h_in <= 0 || desc.w_in <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: input shape extents must be > 0",
            ));
        }
        if desc.c_out <= 0 || desc.h_filt <= 0 || desc.w_filt <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: filter shape extents must be > 0",
            ));
        }
        if desc.stride_h <= 0
            || desc.stride_w <= 0
            || desc.dilation_h <= 0
            || desc.dilation_w <= 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: stride / dilation must be > 0",
            ));
        }
        if desc.pad_h < 0 || desc.pad_w < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: padding must be >= 0",
            ));
        }
        let (h_out, w_out) = compute_output_dims(desc);
        if h_out <= 0 || w_out <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: computed output dims <= 0 — \
                 padding / stride / dilation combination produces an empty output",
            ));
        }

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
            // cuDNN does not contractually guarantee bit-stable results
            // across runs — IMPLICIT_GEMM with tensor-core paths can
            // re-order reductions.
            bit_stable_on_same_hardware: false,
            // The pinned IMPLICIT_GEMM forward + ALGO_1 backward
            // selections are deterministic in the "no internal atomics"
            // sense (single-run-from-one-thread).
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Convolution,
            op: ConvKind::Conv2d as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Cudnn,
            precision_guarantee,
        };

        Ok(Self {
            desc: *desc,
            sku,
            handle: Cell::new(core::ptr::null_mut()),
            x_desc: Cell::new(core::ptr::null_mut()),
            w_desc: Cell::new(core::ptr::null_mut()),
            y_desc: Cell::new(core::ptr::null_mut()),
            conv_desc: Cell::new(core::ptr::null_mut()),
            workspace_bytes_fw: Cell::new(0),
            workspace_bytes_bw_data: Cell::new(0),
            workspace_bytes_bw_filter: Cell::new(0),
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

    /// Computed `(H_out, W_out)` output spatial extents under the
    /// configured pad / stride / dilation.
    #[inline]
    pub fn output_dims(&self) -> (i32, i32) {
        compute_output_dims(&self.desc)
    }

    /// Cached forward workspace size in bytes (`0` before the first
    /// query / run). See [`Self::query_fw_workspace_size`] to force the
    /// query.
    #[inline]
    pub fn workspace_size_fw(&self) -> usize {
        self.workspace_bytes_fw.get()
    }

    /// Cached BW-data workspace size in bytes.
    #[inline]
    pub fn workspace_size_bw_data(&self) -> usize {
        self.workspace_bytes_bw_data.get()
    }

    /// Cached BW-filter workspace size in bytes.
    #[inline]
    pub fn workspace_size_bw_filter(&self) -> usize {
        self.workspace_bytes_bw_filter.get()
    }

    /// Materialize the cuDNN handle / descriptors if needed and query
    /// the FW workspace size. Caches the result. Returns the same value
    /// as [`Self::workspace_size_fw`] would after a `run_fw`.
    pub fn query_fw_workspace_size(&self, stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;
        let mut bytes: usize = 0;
        let status = unsafe {
            cudnnGetConvolutionForwardWorkspaceSize(
                h,
                self.x_desc.get(),
                self.w_desc.get(),
                self.conv_desc.get(),
                self.y_desc.get(),
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                &mut bytes as *mut usize,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.workspace_bytes_fw.set(bytes);
        Ok(bytes)
    }

    /// Materialize handle / descriptors and query the BW-data workspace
    /// size. Caches the result.
    pub fn query_bw_data_workspace_size(&self, stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;
        let mut bytes: usize = 0;
        let status = unsafe {
            cudnnGetConvolutionBackwardDataWorkspaceSize(
                h,
                self.w_desc.get(),
                self.y_desc.get(),
                self.conv_desc.get(),
                self.x_desc.get(),
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                &mut bytes as *mut usize,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.workspace_bytes_bw_data.set(bytes);
        Ok(bytes)
    }

    /// Materialize handle / descriptors and query the BW-filter
    /// workspace size. Caches the result.
    pub fn query_bw_filter_workspace_size(&self, stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;
        let mut bytes: usize = 0;
        let status = unsafe {
            cudnnGetConvolutionBackwardFilterWorkspaceSize(
                h,
                self.x_desc.get(),
                self.y_desc.get(),
                self.conv_desc.get(),
                self.w_desc.get(),
                CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                &mut bytes as *mut usize,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.workspace_bytes_bw_filter.set(bytes);
        Ok(bytes)
    }

    /// Run the forward pass. Computes `y := conv(x, w)` (alpha = 1,
    /// beta = 0).
    pub fn run_fw(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: Conv2dArgs<'_, T>,
    ) -> Result<()> {
        self.check_fw_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;

        let needed = if self.workspace_bytes_fw.get() == 0 {
            self.query_fw_workspace_size(stream)?
        } else {
            self.workspace_bytes_fw.get()
        };
        let (ws_ptr, _ws_bytes) = unpack_workspace(workspace, needed)?;

        // Host-side alpha / beta. The scalar dtype is f32 for
        // f16/bf16/f32 inputs and f64 for f64 inputs (matching the
        // cuDNN compute_type set on the convolution descriptor).
        let status = if is_double_compute::<T>() {
            let alpha: f64 = 1.0;
            let beta: f64 = 0.0;
            unsafe {
                cudnnConvolutionForward(
                    h,
                    &alpha as *const f64 as *const c_void,
                    self.x_desc.get(),
                    args.x.data.as_raw().0 as *const c_void,
                    self.w_desc.get(),
                    args.w.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                    ws_ptr,
                    needed,
                    &beta as *const f64 as *const c_void,
                    self.y_desc.get(),
                    args.y.data.as_raw().0 as *mut c_void,
                )
            }
        } else {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                cudnnConvolutionForward(
                    h,
                    &alpha as *const f32 as *const c_void,
                    self.x_desc.get(),
                    args.x.data.as_raw().0 as *const c_void,
                    self.w_desc.get(),
                    args.w.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                    ws_ptr,
                    needed,
                    &beta as *const f32 as *const c_void,
                    self.y_desc.get(),
                    args.y.data.as_raw().0 as *mut c_void,
                )
            }
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    /// Run the data-gradient pass. Computes `dx := conv_T(w, dy)`.
    pub fn run_bw_data(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: Conv2dBwArgs<'_, T>,
    ) -> Result<()> {
        self.check_bw_data_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;

        let needed = if self.workspace_bytes_bw_data.get() == 0 {
            self.query_bw_data_workspace_size(stream)?
        } else {
            self.workspace_bytes_bw_data.get()
        };
        let (ws_ptr, _ws_bytes) = unpack_workspace(workspace, needed)?;

        let status = if is_double_compute::<T>() {
            let alpha: f64 = 1.0;
            let beta: f64 = 0.0;
            unsafe {
                cudnnConvolutionBackwardData(
                    h,
                    &alpha as *const f64 as *const c_void,
                    self.w_desc.get(),
                    args.w.data.as_raw().0 as *const c_void,
                    self.y_desc.get(),
                    args.dy.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                    ws_ptr,
                    needed,
                    &beta as *const f64 as *const c_void,
                    self.x_desc.get(),
                    args.dx.data.as_raw().0 as *mut c_void,
                )
            }
        } else {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                cudnnConvolutionBackwardData(
                    h,
                    &alpha as *const f32 as *const c_void,
                    self.w_desc.get(),
                    args.w.data.as_raw().0 as *const c_void,
                    self.y_desc.get(),
                    args.dy.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                    ws_ptr,
                    needed,
                    &beta as *const f32 as *const c_void,
                    self.x_desc.get(),
                    args.dx.data.as_raw().0 as *mut c_void,
                )
            }
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    /// Run the filter-gradient pass. Computes `dw := conv_grad(x, dy)`.
    pub fn run_dw(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: Conv2dDwArgs<'_, T>,
    ) -> Result<()> {
        self.check_dw_args(&args)?;
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;

        let needed = if self.workspace_bytes_bw_filter.get() == 0 {
            self.query_bw_filter_workspace_size(stream)?
        } else {
            self.workspace_bytes_bw_filter.get()
        };
        let (ws_ptr, _ws_bytes) = unpack_workspace(workspace, needed)?;

        let status = if is_double_compute::<T>() {
            let alpha: f64 = 1.0;
            let beta: f64 = 0.0;
            unsafe {
                cudnnConvolutionBackwardFilter(
                    h,
                    &alpha as *const f64 as *const c_void,
                    self.x_desc.get(),
                    args.x.data.as_raw().0 as *const c_void,
                    self.y_desc.get(),
                    args.dy.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                    ws_ptr,
                    needed,
                    &beta as *const f64 as *const c_void,
                    self.w_desc.get(),
                    args.dw.data.as_raw().0 as *mut c_void,
                )
            }
        } else {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                cudnnConvolutionBackwardFilter(
                    h,
                    &alpha as *const f32 as *const c_void,
                    self.x_desc.get(),
                    args.x.data.as_raw().0 as *const c_void,
                    self.y_desc.get(),
                    args.dy.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                    ws_ptr,
                    needed,
                    &beta as *const f32 as *const c_void,
                    self.w_desc.get(),
                    args.dw.data.as_raw().0 as *mut c_void,
                )
            }
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
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

    /// Allocate (once) and populate the four cuDNN descriptors from the
    /// stored [`Conv2dDescriptor`]. Idempotent — subsequent calls are a
    /// no-op once the descriptors are live.
    fn ensure_descriptors(&self) -> Result<()> {
        if !self.x_desc.get().is_null() {
            return Ok(());
        }
        let dt = cudnn_dtype::<T>();
        let compute_dt = if is_double_compute::<T>() {
            CUDNN_DATA_DOUBLE
        } else {
            CUDNN_DATA_FLOAT
        };
        let (h_out, w_out) = compute_output_dims(&self.desc);

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
                self.desc.batch,
                self.desc.c_in,
                self.desc.h_in,
                self.desc.w_in,
            )
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(xd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.x_desc.set(xd);

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
                self.desc.batch,
                self.desc.c_out,
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
        self.y_desc.set(yd);

        // w (filter) descriptor.
        let mut wd: cudnnFilterDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateFilterDescriptor(&mut wd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        // NB: cudnnSetFilter4dDescriptor argument order is (desc, dtype,
        // format, k, c, h, w) — dtype precedes format, opposite of
        // cudnnSetTensor4dDescriptor.
        let status = unsafe {
            cudnnSetFilter4dDescriptor(
                wd,
                dt,
                CUDNN_TENSOR_NCHW,
                self.desc.c_out,
                self.desc.c_in,
                self.desc.h_filt,
                self.desc.w_filt,
            )
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyFilterDescriptor(wd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.w_desc.set(wd);

        // Convolution descriptor.
        let mut cd: cudnnConvolutionDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateConvolutionDescriptor(&mut cd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let status = unsafe {
            cudnnSetConvolution2dDescriptor(
                cd,
                self.desc.pad_h,
                self.desc.pad_w,
                self.desc.stride_h,
                self.desc.stride_w,
                self.desc.dilation_h,
                self.desc.dilation_w,
                CUDNN_CROSS_CORRELATION,
                compute_dt,
            )
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyConvolutionDescriptor(cd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.conv_desc.set(cd);

        Ok(())
    }

    // ------------------------------------------------------------------
    // Internal: arg validation
    // ------------------------------------------------------------------

    fn check_fw_args(&self, args: &Conv2dArgs<'_, T>) -> Result<()> {
        let (h_out, w_out) = compute_output_dims(&self.desc);
        let x_shape = [self.desc.batch, self.desc.c_in, self.desc.h_in, self.desc.w_in];
        let w_shape = [
            self.desc.c_out,
            self.desc.c_in,
            self.desc.h_filt,
            self.desc.w_filt,
        ];
        let y_shape = [self.desc.batch, self.desc.c_out, h_out, w_out];
        if args.x.shape != x_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: x shape != [N, C_in, H_in, W_in]",
            ));
        }
        if args.w.shape != w_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: w shape != [C_out, C_in, H_filt, W_filt]",
            ));
        }
        if args.y.shape != y_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: y shape != [N, C_out, H_out, W_out]",
            ));
        }
        Ok(())
    }

    fn check_bw_data_args(&self, args: &Conv2dBwArgs<'_, T>) -> Result<()> {
        let (h_out, w_out) = compute_output_dims(&self.desc);
        let w_shape = [
            self.desc.c_out,
            self.desc.c_in,
            self.desc.h_filt,
            self.desc.w_filt,
        ];
        let dy_shape = [self.desc.batch, self.desc.c_out, h_out, w_out];
        let dx_shape = [self.desc.batch, self.desc.c_in, self.desc.h_in, self.desc.w_in];
        if args.w.shape != w_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: w shape != [C_out, C_in, H_filt, W_filt]",
            ));
        }
        if args.dy.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: dy shape != [N, C_out, H_out, W_out]",
            ));
        }
        if args.dx.shape != dx_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: dx shape != [N, C_in, H_in, W_in]",
            ));
        }
        Ok(())
    }

    fn check_dw_args(&self, args: &Conv2dDwArgs<'_, T>) -> Result<()> {
        let (h_out, w_out) = compute_output_dims(&self.desc);
        let x_shape = [self.desc.batch, self.desc.c_in, self.desc.h_in, self.desc.w_in];
        let dy_shape = [self.desc.batch, self.desc.c_out, h_out, w_out];
        let dw_shape = [
            self.desc.c_out,
            self.desc.c_in,
            self.desc.h_filt,
            self.desc.w_filt,
        ];
        if args.x.shape != x_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: x shape != [N, C_in, H_in, W_in]",
            ));
        }
        if args.dy.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: dy shape != [N, C_out, H_out, W_out]",
            ));
        }
        if args.dw.shape != dw_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv2dPlan: dw shape != [C_out, C_in, H_filt, W_filt]",
            ));
        }
        Ok(())
    }
}

impl<T: Element> Drop for Conv2dPlan<T> {
    fn drop(&mut self) {
        let cd = self.conv_desc.get();
        if !cd.is_null() {
            unsafe {
                let _ = cudnnDestroyConvolutionDescriptor(cd);
            }
            self.conv_desc.set(core::ptr::null_mut());
        }
        let wd = self.w_desc.get();
        if !wd.is_null() {
            unsafe {
                let _ = cudnnDestroyFilterDescriptor(wd);
            }
            self.w_desc.set(core::ptr::null_mut());
        }
        let yd = self.y_desc.get();
        if !yd.is_null() {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(yd);
            }
            self.y_desc.set(core::ptr::null_mut());
        }
        let xd = self.x_desc.get();
        if !xd.is_null() {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(xd);
            }
            self.x_desc.set(core::ptr::null_mut());
        }
        let h = self.handle.get();
        if !h.is_null() {
            unsafe {
                let _ = cudnnDestroy(h);
            }
            self.handle.set(core::ptr::null_mut());
        }
    }
}

// ----- helpers --------------------------------------------------------

/// `(H_out, W_out) = floor((dim_in + 2·pad - dilation·(filt - 1) - 1) /
/// stride) + 1` — the standard PyTorch / cuDNN formula.
#[inline]
fn compute_output_dims(d: &Conv2dDescriptor) -> (i32, i32) {
    let h_eff = d.dilation_h * (d.h_filt - 1) + 1;
    let w_eff = d.dilation_w * (d.w_filt - 1) + 1;
    let h_out = (d.h_in + 2 * d.pad_h - h_eff) / d.stride_h + 1;
    let w_out = (d.w_in + 2 * d.pad_w - w_eff) / d.stride_w + 1;
    (h_out, w_out)
}

/// Map a Rust [`Element`] to a cuDNN data-type tag.
#[inline]
fn cudnn_dtype<T: Element>() -> i32 {
    match T::KIND {
        ElementKind::F32 => CUDNN_DATA_FLOAT,
        ElementKind::F64 => CUDNN_DATA_DOUBLE,
        ElementKind::F16 => CUDNN_DATA_HALF,
        ElementKind::Bf16 => CUDNN_DATA_BFLOAT16,
        _ => unreachable!("Conv2dPlan::select gates on F32/F64/F16/Bf16"),
    }
}

/// `true` iff the cuDNN "compute type" / alpha-beta scalar type is f64
/// (i.e. the element is f64). All other supported dtypes route through
/// the f32 compute / scalar path.
#[inline]
fn is_double_compute<T: Element>() -> bool {
    matches!(T::KIND, ElementKind::F64)
}

/// Sibling to `super::super::linalg::cholesky::unpack_workspace` —
/// inlined here to keep the conv module independent of the linalg
/// module. Same contract: `None` is valid iff `needed == 0`, otherwise
/// `Borrowed(s)` must have `s.len() >= needed`.
fn unpack_workspace(workspace: Workspace<'_>, needed: usize) -> Result<(*mut c_void, usize)> {
    match workspace {
        Workspace::None => {
            if needed == 0 {
                Ok((core::ptr::null_mut(), 0))
            } else {
                Err(Error::WorkspaceTooSmall { needed, got: 0 })
            }
        }
        Workspace::Borrowed(slice) => {
            let got = slice.len();
            if got < needed {
                return Err(Error::WorkspaceTooSmall { needed, got });
            }
            Ok((slice.as_raw().0 as *mut c_void, got))
        }
    }
}
