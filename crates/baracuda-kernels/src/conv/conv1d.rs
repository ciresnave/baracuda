//! Conv1d — NCL 1-D convolution via cuDNN's NdDescriptor API.
//!
//! Implements forward + both backward passes for 1-D convolution over
//! the four floating-point dtypes (`f32`, `f64`, `f16`, `bf16`). The
//! mathematical convention is **cross-correlation** (kernel applied
//! directly, not flipped) to match PyTorch / JAX. Filter / activation
//! tensors carry the standard `[N, C, L]` (rank-3) layout from the
//! caller's perspective. Internally the plan pads the rank-3 NCL shape
//! to rank-4 NCLW with `W = 1` (singleton trailing spatial axis),
//! because cuDNN's `cudnnSetTensorNdDescriptor` /
//! `cudnnSetFilterNdDescriptor` reject `nb_dims < 4`. The trailing
//! singleton axis is zero-padded / unit-strided / unit-dilated in the
//! convolution descriptor (`array_length = 2`) so it has no semantic
//! effect — the dummy dim is transparent to callers.

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
    cudnnGetConvolutionForwardWorkspaceSize, cudnnHandle_t, cudnnSetConvolutionGroupCount,
    cudnnSetConvolutionNdDescriptor, cudnnSetFilterNdDescriptor, cudnnSetStream,
    cudnnSetTensorNdDescriptor, cudnnTensorDescriptor_t, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_BFLOAT16, CUDNN_DATA_DOUBLE, CUDNN_DATA_FLOAT,
    CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, ConvKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a 1-D convolution over NCL tensors.
///
/// Input shape: `[batch, c_in, l_in]`. Filter shape: `[c_out,
/// c_in / groups, l_filt]`. The padding / stride / dilation extents
/// follow PyTorch's `nn.Conv1d` semantics (kernel applied directly —
/// **cross-correlation**, not mathematical convolution).
#[derive(Copy, Clone, Debug)]
pub struct Conv1dDescriptor {
    /// Batch size `N`.
    pub batch: i32,
    /// Input channel count `C_in`.
    pub c_in: i32,
    /// Input length `L_in`.
    pub l_in: i32,
    /// Output channel count `C_out`.
    pub c_out: i32,
    /// Filter length `L_filt`.
    pub l_filt: i32,
    /// Zero-padding on each side of the length axis.
    pub pad_l: i32,
    /// Stride along the length axis.
    pub stride_l: i32,
    /// Dilation along the length axis (`1` for the "dense" default).
    pub dilation_l: i32,
    /// Group count (`1` for plain conv, `c_in` for depthwise).
    pub groups: i32,
    /// Element dtype. Must be `F32`, `F64`, `F16`, or `Bf16`.
    pub element: ElementKind,
}

/// Args bundle for a Conv1d forward launch.
pub struct Conv1dArgs<'a, T: Element> {
    /// Input activations `[N, C_in, L_in]` NCL contiguous.
    pub x: TensorRef<'a, T, 3>,
    /// Filter weights `[C_out, C_in / groups, L_filt]`.
    pub w: TensorRef<'a, T, 3>,
    /// Output activations `[N, C_out, L_out]` NCL contiguous.
    pub y: TensorMut<'a, T, 3>,
}

/// Args bundle for a Conv1d data-gradient launch (BW w.r.t. the input
/// activations). Computes `dx = conv_T(w, dy)`.
pub struct Conv1dBwArgs<'a, T: Element> {
    /// Filter weights `[C_out, C_in / groups, L_filt]`.
    pub w: TensorRef<'a, T, 3>,
    /// Upstream gradient `[N, C_out, L_out]`.
    pub dy: TensorRef<'a, T, 3>,
    /// Output gradient w.r.t. input `[N, C_in, L_in]`.
    pub dx: TensorMut<'a, T, 3>,
}

/// Args bundle for a Conv1d filter-gradient launch (BW w.r.t. the
/// filter weights). Computes `dw = conv_grad(x, dy)`.
pub struct Conv1dDwArgs<'a, T: Element> {
    /// Input activations `[N, C_in, L_in]`.
    pub x: TensorRef<'a, T, 3>,
    /// Upstream gradient `[N, C_out, L_out]`.
    pub dy: TensorRef<'a, T, 3>,
    /// Output gradient w.r.t. the filter `[C_out, C_in / groups, L_filt]`.
    pub dw: TensorMut<'a, T, 3>,
}

/// 1-D convolution plan (cuDNN-backed) — forward + both backward
/// passes for NCL activations.
///
/// **When to use**: any 1-D convolution from a typical text / audio
/// CNN (Conv1d / dilated Conv1d / strided Conv1d / depthwise Conv1d).
/// Mathematical convention is cross-correlation. Run `run_fw` for the
/// forward, `run_bw_data` for `dx`, `run_dw` for `dw`.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16`.
///
/// **Shape**: input `[N, C_in, L_in]`, filter `[C_out, C_in / groups,
/// L_filt]`, output `[N, C_out, L_out]` with `L_out` computed from
/// pad / stride / dilation.
///
/// **Algorithm pinning**: FW = `IMPLICIT_GEMM` (algo 0), BW-data /
/// BW-filter = `ALGO_1`.
///
/// **Workspace**: caller-provided (`Workspace::Borrowed`). Each
/// direction has its own size — call the relevant
/// `query_*_workspace_size(stream)` accessor before launching.
///
/// Gated under `feature = "cudnn"` at the crate root.
pub struct Conv1dPlan<T: Element> {
    desc: Conv1dDescriptor,
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

impl<T: Element> Conv1dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &Conv1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::Conv1dPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::Conv1dPlan: cuDNN Conv1d supports f32 / f64 / f16 / bf16",
            ));
        }
        if desc.batch <= 0 || desc.c_in <= 0 || desc.l_in <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: input shape extents must be > 0",
            ));
        }
        if desc.c_out <= 0 || desc.l_filt <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: filter shape extents must be > 0",
            ));
        }
        if desc.stride_l <= 0 || desc.dilation_l <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: stride / dilation must be > 0",
            ));
        }
        if desc.pad_l < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: padding must be >= 0",
            ));
        }
        if desc.groups <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: groups must be > 0",
            ));
        }
        if desc.c_in % desc.groups != 0 || desc.c_out % desc.groups != 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: groups must divide both c_in and c_out",
            ));
        }
        let l_out = compute_l_out(desc);
        if l_out <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: computed output dim <= 0",
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
            bit_stable_on_same_hardware: false,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Convolution,
            op: ConvKind::Conv1d as u16,
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

    /// Computed `L_out` output extent under the configured pad / stride
    /// / dilation.
    #[inline]
    pub fn output_dim(&self) -> i32 {
        compute_l_out(&self.desc)
    }

    /// Cached forward workspace size in bytes.
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

    /// Materialize handle / descriptors and query FW workspace size.
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
    /// size.
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
    /// workspace size.
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

    /// Run the forward pass. Computes `y := conv(x, w)`.
    pub fn run_fw(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: Conv1dArgs<'_, T>,
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
        args: Conv1dBwArgs<'_, T>,
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
        args: Conv1dDwArgs<'_, T>,
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
    /// stored [`Conv1dDescriptor`]. Idempotent.
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
        let l_out = compute_l_out(&self.desc);
        let c_in_per_group = self.desc.c_in / self.desc.groups;

        // x descriptor — rank-3 [N, C_in, L_in].
        let mut xd: cudnnTensorDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateTensorDescriptor(&mut xd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        // cuDNN's NdDescriptor APIs require `nb_dims >= 4`. Pad the
        // logical rank-3 NCL shape to rank-4 NCLW with W = 1 (singleton
        // trailing spatial dim). Convolution descriptor below is
        // `array_length = 2` with the trailing axis zero-padded /
        // unit-strided / unit-dilated. The dummy axis is transparent to
        // callers — output is still rank-3 NCL.
        let x_dims = [self.desc.batch, self.desc.c_in, self.desc.l_in, 1];
        let x_strides = [self.desc.c_in * self.desc.l_in, self.desc.l_in, 1, 1];
        let status = unsafe {
            cudnnSetTensorNdDescriptor(
                xd,
                dt,
                4,
                x_dims.as_ptr(),
                x_strides.as_ptr(),
            )
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(xd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.x_desc.set(xd);

        // y descriptor — rank-3 [N, C_out, L_out].
        let mut yd: cudnnTensorDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateTensorDescriptor(&mut yd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let y_dims = [self.desc.batch, self.desc.c_out, l_out, 1];
        let y_strides = [self.desc.c_out * l_out, l_out, 1, 1];
        let status = unsafe {
            cudnnSetTensorNdDescriptor(
                yd,
                dt,
                4,
                y_dims.as_ptr(),
                y_strides.as_ptr(),
            )
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(yd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.y_desc.set(yd);

        // w (filter) descriptor — rank-3 [C_out, C_in/groups, L_filt].
        let mut wd: cudnnFilterDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateFilterDescriptor(&mut wd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let w_dims = [self.desc.c_out, c_in_per_group, self.desc.l_filt, 1];
        let status = unsafe {
            cudnnSetFilterNdDescriptor(wd, dt, CUDNN_TENSOR_NCHW, 4, w_dims.as_ptr())
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyFilterDescriptor(wd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.w_desc.set(wd);

        // Convolution descriptor — array_length = 1 (spatial rank).
        let mut cd: cudnnConvolutionDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateConvolutionDescriptor(&mut cd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let pad_a = [self.desc.pad_l, 0];
        let stride_a = [self.desc.stride_l, 1];
        let dilation_a = [self.desc.dilation_l, 1];
        let status = unsafe {
            cudnnSetConvolutionNdDescriptor(
                cd,
                2,
                pad_a.as_ptr(),
                stride_a.as_ptr(),
                dilation_a.as_ptr(),
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
        let status = unsafe { cudnnSetConvolutionGroupCount(cd, self.desc.groups) };
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

    fn check_fw_args(&self, args: &Conv1dArgs<'_, T>) -> Result<()> {
        let l_out = compute_l_out(&self.desc);
        let c_in_per_group = self.desc.c_in / self.desc.groups;
        let x_shape = [self.desc.batch, self.desc.c_in, self.desc.l_in];
        let w_shape = [self.desc.c_out, c_in_per_group, self.desc.l_filt];
        let y_shape = [self.desc.batch, self.desc.c_out, l_out];
        if args.x.shape != x_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: x shape != [N, C_in, L_in]",
            ));
        }
        if args.w.shape != w_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: w shape != [C_out, C_in/groups, L_filt]",
            ));
        }
        if args.y.shape != y_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: y shape != [N, C_out, L_out]",
            ));
        }
        Ok(())
    }

    fn check_bw_data_args(&self, args: &Conv1dBwArgs<'_, T>) -> Result<()> {
        let l_out = compute_l_out(&self.desc);
        let c_in_per_group = self.desc.c_in / self.desc.groups;
        let w_shape = [self.desc.c_out, c_in_per_group, self.desc.l_filt];
        let dy_shape = [self.desc.batch, self.desc.c_out, l_out];
        let dx_shape = [self.desc.batch, self.desc.c_in, self.desc.l_in];
        if args.w.shape != w_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: w shape != [C_out, C_in/groups, L_filt]",
            ));
        }
        if args.dy.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: dy shape != [N, C_out, L_out]",
            ));
        }
        if args.dx.shape != dx_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: dx shape != [N, C_in, L_in]",
            ));
        }
        Ok(())
    }

    fn check_dw_args(&self, args: &Conv1dDwArgs<'_, T>) -> Result<()> {
        let l_out = compute_l_out(&self.desc);
        let c_in_per_group = self.desc.c_in / self.desc.groups;
        let x_shape = [self.desc.batch, self.desc.c_in, self.desc.l_in];
        let dy_shape = [self.desc.batch, self.desc.c_out, l_out];
        let dw_shape = [self.desc.c_out, c_in_per_group, self.desc.l_filt];
        if args.x.shape != x_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: x shape != [N, C_in, L_in]",
            ));
        }
        if args.dy.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: dy shape != [N, C_out, L_out]",
            ));
        }
        if args.dw.shape != dw_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv1dPlan: dw shape != [C_out, C_in/groups, L_filt]",
            ));
        }
        Ok(())
    }
}

impl<T: Element> Drop for Conv1dPlan<T> {
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

/// `L_out = floor((L_in + 2·pad - dilation·(L_filt - 1) - 1) / stride) + 1`.
#[inline]
fn compute_l_out(d: &Conv1dDescriptor) -> i32 {
    let l_eff = d.dilation_l * (d.l_filt - 1) + 1;
    (d.l_in + 2 * d.pad_l - l_eff) / d.stride_l + 1
}

#[inline]
fn cudnn_dtype<T: Element>() -> i32 {
    match T::KIND {
        ElementKind::F32 => CUDNN_DATA_FLOAT,
        ElementKind::F64 => CUDNN_DATA_DOUBLE,
        ElementKind::F16 => CUDNN_DATA_HALF,
        ElementKind::Bf16 => CUDNN_DATA_BFLOAT16,
        _ => unreachable!("Conv1dPlan::select gates on F32/F64/F16/Bf16"),
    }
}

#[inline]
fn is_double_compute<T: Element>() -> bool {
    matches!(T::KIND, ElementKind::F64)
}

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
