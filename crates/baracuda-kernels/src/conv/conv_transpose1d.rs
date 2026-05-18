//! ConvTranspose1d — 1-D transposed convolution via cuDNN.
//!
//! Implementation strategy mirrors [`super::conv_transpose2d`]: cuDNN
//! has no direct "transpose forward" entry point, so we configure
//! descriptors as if for a synthetic dense Conv1d that maps
//! `[N, c_out, L_out] → [N, c_in, L_in]`, then call
//! `cudnnConvolutionBackwardData` (with roles swapped) for the
//! ConvTranspose forward, and `cudnnConvolutionForward` for the
//! data-backward direction. The filter-gradient direction maps
//! straight to `cudnnConvolutionBackwardFilter` with the same role
//! swap. Filter shape follows PyTorch: `[C_in, C_out / groups, L_filt]`.
//!
//! Cross-correlation only. Dtypes: `f32`, `f64`, `f16`, `bf16`.

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

/// Descriptor for a 1-D transposed convolution over NCL tensors.
#[derive(Copy, Clone, Debug)]
pub struct ConvTranspose1dDescriptor {
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
    /// Implicit zero-padding on each side of the output length axis.
    pub pad_l: i32,
    /// Stride.
    pub stride_l: i32,
    /// Dilation.
    pub dilation_l: i32,
    /// Output-side padding (`< max(stride, dilation)`).
    pub output_pad_l: i32,
    /// Group count.
    pub groups: i32,
    /// Element dtype. Must be `F32`, `F64`, `F16`, or `Bf16`.
    pub element: ElementKind,
}

/// Args bundle for a ConvTranspose1d forward launch.
pub struct ConvTranspose1dArgs<'a, T: Element> {
    /// Input activations `[N, C_in, L_in]`.
    pub x: TensorRef<'a, T, 3>,
    /// Filter weights `[C_in, C_out / groups, L_filt]` (PyTorch order).
    pub w: TensorRef<'a, T, 3>,
    /// Output activations `[N, C_out, L_out]`.
    pub y: TensorMut<'a, T, 3>,
}

/// Args bundle for a ConvTranspose1d data-gradient launch.
pub struct ConvTranspose1dBwArgs<'a, T: Element> {
    /// Filter weights `[C_in, C_out / groups, L_filt]`.
    pub w: TensorRef<'a, T, 3>,
    /// Upstream gradient `[N, C_out, L_out]`.
    pub dy: TensorRef<'a, T, 3>,
    /// Output gradient w.r.t. input `[N, C_in, L_in]`.
    pub dx: TensorMut<'a, T, 3>,
}

/// Args bundle for a ConvTranspose1d filter-gradient launch.
pub struct ConvTranspose1dDwArgs<'a, T: Element> {
    /// Input activations `[N, C_in, L_in]`.
    pub x: TensorRef<'a, T, 3>,
    /// Upstream gradient `[N, C_out, L_out]`.
    pub dy: TensorRef<'a, T, 3>,
    /// Output gradient w.r.t. the filter `[C_in, C_out / groups, L_filt]`.
    pub dw: TensorMut<'a, T, 3>,
}

/// 1-D transposed-convolution plan (cuDNN-backed). See the module
/// docs for the role-swap strategy.
pub struct ConvTranspose1dPlan<T: Element> {
    desc: ConvTranspose1dDescriptor,
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    synth_x_desc: Cell<cudnnTensorDescriptor_t>,
    synth_w_desc: Cell<cudnnFilterDescriptor_t>,
    synth_y_desc: Cell<cudnnTensorDescriptor_t>,
    conv_desc: Cell<cudnnConvolutionDescriptor_t>,
    workspace_bytes_fw: Cell<usize>,
    workspace_bytes_bw_data: Cell<usize>,
    workspace_bytes_bw_filter: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> ConvTranspose1dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &ConvTranspose1dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ConvTranspose1dPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::ConvTranspose1dPlan: dtype must be f32 / f64 / f16 / bf16",
            ));
        }
        if desc.batch <= 0 || desc.c_in <= 0 || desc.l_in <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: input shape extents must be > 0",
            ));
        }
        if desc.c_out <= 0 || desc.l_filt <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: filter shape extents must be > 0",
            ));
        }
        if desc.stride_l <= 0 || desc.dilation_l <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: stride / dilation must be > 0",
            ));
        }
        if desc.pad_l < 0 || desc.output_pad_l < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: padding / output_padding must be >= 0",
            ));
        }
        if desc.groups <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: groups must be > 0",
            ));
        }
        if desc.c_in % desc.groups != 0 || desc.c_out % desc.groups != 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: groups must divide both c_in and c_out",
            ));
        }
        if desc.output_pad_l >= desc.stride_l.max(desc.dilation_l) {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: output_padding must be < max(stride, dilation)",
            ));
        }
        let l_out = compute_l_out(desc);
        if l_out <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: computed output dim <= 0",
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
            op: ConvKind::ConvTranspose1d as u16,
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
            synth_x_desc: Cell::new(core::ptr::null_mut()),
            synth_w_desc: Cell::new(core::ptr::null_mut()),
            synth_y_desc: Cell::new(core::ptr::null_mut()),
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

    /// Computed `L_out`.
    #[inline]
    pub fn output_dim(&self) -> i32 {
        compute_l_out(&self.desc)
    }

    /// Cached FW workspace size in bytes.
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

    /// Query FW workspace size (synthetic BW-data workspace).
    pub fn query_fw_workspace_size(&self, stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;
        let mut bytes: usize = 0;
        let status = unsafe {
            cudnnGetConvolutionBackwardDataWorkspaceSize(
                h,
                self.synth_w_desc.get(),
                self.synth_y_desc.get(),
                self.conv_desc.get(),
                self.synth_x_desc.get(),
                CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                &mut bytes as *mut usize,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.workspace_bytes_fw.set(bytes);
        Ok(bytes)
    }

    /// Query BW-data workspace size (synthetic FW workspace).
    pub fn query_bw_data_workspace_size(&self, stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;
        let mut bytes: usize = 0;
        let status = unsafe {
            cudnnGetConvolutionForwardWorkspaceSize(
                h,
                self.synth_x_desc.get(),
                self.synth_w_desc.get(),
                self.conv_desc.get(),
                self.synth_y_desc.get(),
                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                &mut bytes as *mut usize,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        self.workspace_bytes_bw_data.set(bytes);
        Ok(bytes)
    }

    /// Query BW-filter workspace size.
    pub fn query_bw_filter_workspace_size(&self, stream: &Stream) -> Result<usize> {
        let h = self.ensure_handle()?;
        self.bind_stream(h, stream)?;
        self.ensure_descriptors()?;
        let mut bytes: usize = 0;
        let status = unsafe {
            cudnnGetConvolutionBackwardFilterWorkspaceSize(
                h,
                self.synth_x_desc.get(),
                self.synth_y_desc.get(),
                self.conv_desc.get(),
                self.synth_w_desc.get(),
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

    /// Run the forward pass — under the hood
    /// `cudnnConvolutionBackwardData` with the role swap.
    pub fn run_fw(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: ConvTranspose1dArgs<'_, T>,
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
                cudnnConvolutionBackwardData(
                    h,
                    &alpha as *const f64 as *const c_void,
                    self.synth_w_desc.get(),
                    args.w.data.as_raw().0 as *const c_void,
                    self.synth_y_desc.get(),
                    args.x.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                    ws_ptr,
                    needed,
                    &beta as *const f64 as *const c_void,
                    self.synth_x_desc.get(),
                    args.y.data.as_raw().0 as *mut c_void,
                )
            }
        } else {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                cudnnConvolutionBackwardData(
                    h,
                    &alpha as *const f32 as *const c_void,
                    self.synth_w_desc.get(),
                    args.w.data.as_raw().0 as *const c_void,
                    self.synth_y_desc.get(),
                    args.x.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                    ws_ptr,
                    needed,
                    &beta as *const f32 as *const c_void,
                    self.synth_x_desc.get(),
                    args.y.data.as_raw().0 as *mut c_void,
                )
            }
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    /// Run the data-gradient pass — under the hood
    /// `cudnnConvolutionForward` with the role swap.
    pub fn run_bw_data(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: ConvTranspose1dBwArgs<'_, T>,
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
                cudnnConvolutionForward(
                    h,
                    &alpha as *const f64 as *const c_void,
                    self.synth_x_desc.get(),
                    args.dy.data.as_raw().0 as *const c_void,
                    self.synth_w_desc.get(),
                    args.w.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                    ws_ptr,
                    needed,
                    &beta as *const f64 as *const c_void,
                    self.synth_y_desc.get(),
                    args.dx.data.as_raw().0 as *mut c_void,
                )
            }
        } else {
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                cudnnConvolutionForward(
                    h,
                    &alpha as *const f32 as *const c_void,
                    self.synth_x_desc.get(),
                    args.dy.data.as_raw().0 as *const c_void,
                    self.synth_w_desc.get(),
                    args.w.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                    ws_ptr,
                    needed,
                    &beta as *const f32 as *const c_void,
                    self.synth_y_desc.get(),
                    args.dx.data.as_raw().0 as *mut c_void,
                )
            }
        };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        Ok(())
    }

    /// Run the filter-gradient pass.
    pub fn run_dw(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: ConvTranspose1dDwArgs<'_, T>,
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
                    self.synth_x_desc.get(),
                    args.dy.data.as_raw().0 as *const c_void,
                    self.synth_y_desc.get(),
                    args.x.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                    ws_ptr,
                    needed,
                    &beta as *const f64 as *const c_void,
                    self.synth_w_desc.get(),
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
                    self.synth_x_desc.get(),
                    args.dy.data.as_raw().0 as *const c_void,
                    self.synth_y_desc.get(),
                    args.x.data.as_raw().0 as *const c_void,
                    self.conv_desc.get(),
                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                    ws_ptr,
                    needed,
                    &beta as *const f32 as *const c_void,
                    self.synth_w_desc.get(),
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

    fn ensure_descriptors(&self) -> Result<()> {
        if !self.synth_x_desc.get().is_null() {
            return Ok(());
        }
        let dt = cudnn_dtype::<T>();
        let compute_dt = if is_double_compute::<T>() {
            CUDNN_DATA_DOUBLE
        } else {
            CUDNN_DATA_FLOAT
        };
        let l_out = compute_l_out(&self.desc);
        let c_out_per_group = self.desc.c_out / self.desc.groups;

        // cuDNN's NdDescriptor APIs require `nb_dims >= 4`. Pad the
        // logical rank-3 shapes to rank-4 with a singleton trailing
        // spatial dim (W = 1); convolution descriptor below uses
        // `array_length = 2` with the trailing axis unit-strided /
        // zero-padded / unit-dilated. Transparent to callers.
        //
        // synth_x: [N, C_out, L_out, 1].
        let mut xd: cudnnTensorDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateTensorDescriptor(&mut xd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let x_dims = [self.desc.batch, self.desc.c_out, l_out, 1];
        let x_strides = [self.desc.c_out * l_out, l_out, 1, 1];
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
        self.synth_x_desc.set(xd);

        // synth_y: [N, C_in, L_in, 1].
        let mut yd: cudnnTensorDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateTensorDescriptor(&mut yd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let y_dims = [self.desc.batch, self.desc.c_in, self.desc.l_in, 1];
        let y_strides = [self.desc.c_in * self.desc.l_in, self.desc.l_in, 1, 1];
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
        self.synth_y_desc.set(yd);

        // synth_w: [C_in, C_out/groups, L_filt].
        let mut wd: cudnnFilterDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateFilterDescriptor(&mut wd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let w_dims = [self.desc.c_in, c_out_per_group, self.desc.l_filt, 1];
        let status = unsafe {
            cudnnSetFilterNdDescriptor(wd, dt, CUDNN_TENSOR_NCHW, 4, w_dims.as_ptr())
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyFilterDescriptor(wd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.synth_w_desc.set(wd);

        // Convolution descriptor — array_length = 1.
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

    fn check_fw_args(&self, args: &ConvTranspose1dArgs<'_, T>) -> Result<()> {
        let l_out = compute_l_out(&self.desc);
        let c_out_per_group = self.desc.c_out / self.desc.groups;
        let x_shape = [self.desc.batch, self.desc.c_in, self.desc.l_in];
        let w_shape = [self.desc.c_in, c_out_per_group, self.desc.l_filt];
        let y_shape = [self.desc.batch, self.desc.c_out, l_out];
        if args.x.shape != x_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: x shape != [N, C_in, L_in]",
            ));
        }
        if args.w.shape != w_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: w shape != [C_in, C_out/groups, L_filt]",
            ));
        }
        if args.y.shape != y_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: y shape != [N, C_out, L_out]",
            ));
        }
        Ok(())
    }

    fn check_bw_data_args(&self, args: &ConvTranspose1dBwArgs<'_, T>) -> Result<()> {
        let l_out = compute_l_out(&self.desc);
        let c_out_per_group = self.desc.c_out / self.desc.groups;
        let w_shape = [self.desc.c_in, c_out_per_group, self.desc.l_filt];
        let dy_shape = [self.desc.batch, self.desc.c_out, l_out];
        let dx_shape = [self.desc.batch, self.desc.c_in, self.desc.l_in];
        if args.w.shape != w_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: w shape != [C_in, C_out/groups, L_filt]",
            ));
        }
        if args.dy.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: dy shape != [N, C_out, L_out]",
            ));
        }
        if args.dx.shape != dx_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: dx shape != [N, C_in, L_in]",
            ));
        }
        Ok(())
    }

    fn check_dw_args(&self, args: &ConvTranspose1dDwArgs<'_, T>) -> Result<()> {
        let l_out = compute_l_out(&self.desc);
        let c_out_per_group = self.desc.c_out / self.desc.groups;
        let x_shape = [self.desc.batch, self.desc.c_in, self.desc.l_in];
        let dy_shape = [self.desc.batch, self.desc.c_out, l_out];
        let dw_shape = [self.desc.c_in, c_out_per_group, self.desc.l_filt];
        if args.x.shape != x_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: x shape != [N, C_in, L_in]",
            ));
        }
        if args.dy.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: dy shape != [N, C_out, L_out]",
            ));
        }
        if args.dw.shape != dw_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose1dPlan: dw shape != [C_in, C_out/groups, L_filt]",
            ));
        }
        Ok(())
    }
}

impl<T: Element> Drop for ConvTranspose1dPlan<T> {
    fn drop(&mut self) {
        let cd = self.conv_desc.get();
        if !cd.is_null() {
            unsafe {
                let _ = cudnnDestroyConvolutionDescriptor(cd);
            }
            self.conv_desc.set(core::ptr::null_mut());
        }
        let wd = self.synth_w_desc.get();
        if !wd.is_null() {
            unsafe {
                let _ = cudnnDestroyFilterDescriptor(wd);
            }
            self.synth_w_desc.set(core::ptr::null_mut());
        }
        let yd = self.synth_y_desc.get();
        if !yd.is_null() {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(yd);
            }
            self.synth_y_desc.set(core::ptr::null_mut());
        }
        let xd = self.synth_x_desc.get();
        if !xd.is_null() {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(xd);
            }
            self.synth_x_desc.set(core::ptr::null_mut());
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

/// `L_out = (L_in - 1)·stride - 2·pad + dilation·(L_filt - 1) +
/// output_padding + 1`.
#[inline]
fn compute_l_out(d: &ConvTranspose1dDescriptor) -> i32 {
    (d.l_in - 1) * d.stride_l - 2 * d.pad_l
        + d.dilation_l * (d.l_filt - 1)
        + d.output_pad_l
        + 1
}

#[inline]
fn cudnn_dtype<T: Element>() -> i32 {
    match T::KIND {
        ElementKind::F32 => CUDNN_DATA_FLOAT,
        ElementKind::F64 => CUDNN_DATA_DOUBLE,
        ElementKind::F16 => CUDNN_DATA_HALF,
        ElementKind::Bf16 => CUDNN_DATA_BFLOAT16,
        _ => unreachable!("ConvTranspose1dPlan::select gates on F32/F64/F16/Bf16"),
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
