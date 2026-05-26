//! Conv3d — NCDHW 3-D convolution via cuDNN's NdDescriptor API.
//!
//! Implements forward + both backward passes for 3-D convolution over
//! the four floating-point dtypes (`f32`, `f64`, `f16`, `bf16`). The
//! mathematical convention is **cross-correlation** (kernel applied
//! directly, not flipped) to match PyTorch / JAX.
//!
//! Activation tensors carry the standard `[N, C, D, H, W]` (rank-5)
//! layout. cuDNN's `cudnnSetTensorNdDescriptor` +
//! `cudnnSetFilterNdDescriptor` + `cudnnSetConvolutionNdDescriptor`
//! (with `array_length = 3`) handle 3-D natively.

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

/// Descriptor for a 3-D convolution over NCDHW tensors.
///
/// Input shape: `[batch, c_in, d_in, h_in, w_in]`. Filter shape:
/// `[c_out, c_in / groups, d_filt, h_filt, w_filt]`. Cross-correlation
/// only (PyTorch parity).
///
/// `#[non_exhaustive]` (Phase 32) — see [`super::Conv2dDescriptor`] for
/// the builder rationale. Use [`Self::new`] + the `with_*` setters
/// from downstream code.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct Conv3dDescriptor {
    /// Batch size `N`.
    pub batch: i32,
    /// Input channel count `C_in`.
    pub c_in: i32,
    /// Input depth `D_in`.
    pub d_in: i32,
    /// Input height `H_in`.
    pub h_in: i32,
    /// Input width `W_in`.
    pub w_in: i32,
    /// Output channel count `C_out`.
    pub c_out: i32,
    /// Filter depth `D_filt`.
    pub d_filt: i32,
    /// Filter height `H_filt`.
    pub h_filt: i32,
    /// Filter width `W_filt`.
    pub w_filt: i32,
    /// Padding along D / H / W.
    pub pad_d: i32,
    /// Padding along H.
    pub pad_h: i32,
    /// Padding along W.
    pub pad_w: i32,
    /// Stride along D.
    pub stride_d: i32,
    /// Stride along H.
    pub stride_h: i32,
    /// Stride along W.
    pub stride_w: i32,
    /// Dilation along D.
    pub dilation_d: i32,
    /// Dilation along H.
    pub dilation_h: i32,
    /// Dilation along W.
    pub dilation_w: i32,
    /// Group count (`1` for plain conv).
    pub groups: i32,
    /// Element dtype. Must be `F32`, `F64`, `F16`, or `Bf16`.
    pub element: ElementKind,
}

impl Conv3dDescriptor {
    /// Build a descriptor with `pad / stride / dilation / groups`
    /// defaulted to PyTorch's `nn.Conv3d` defaults (`0 / 1 / 1 / 1`).
    /// Chain with [`Self::with_padding`] / [`Self::with_stride`] /
    /// [`Self::with_dilation`] / [`Self::with_groups`] to override.
    pub fn new(
        batch: i32,
        c_in: i32,
        d_in: i32,
        h_in: i32,
        w_in: i32,
        c_out: i32,
        d_filt: i32,
        h_filt: i32,
        w_filt: i32,
        element: ElementKind,
    ) -> Self {
        Self {
            batch,
            c_in,
            d_in,
            h_in,
            w_in,
            c_out,
            d_filt,
            h_filt,
            w_filt,
            pad_d: 0,
            pad_h: 0,
            pad_w: 0,
            stride_d: 1,
            stride_h: 1,
            stride_w: 1,
            dilation_d: 1,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
            element,
        }
    }

    /// Override `(pad_d, pad_h, pad_w)`. Default `(0, 0, 0)`.
    #[inline]
    pub fn with_padding(mut self, pad_d: i32, pad_h: i32, pad_w: i32) -> Self {
        self.pad_d = pad_d;
        self.pad_h = pad_h;
        self.pad_w = pad_w;
        self
    }

    /// Override `(stride_d, stride_h, stride_w)`. Default `(1, 1, 1)`.
    #[inline]
    pub fn with_stride(mut self, stride_d: i32, stride_h: i32, stride_w: i32) -> Self {
        self.stride_d = stride_d;
        self.stride_h = stride_h;
        self.stride_w = stride_w;
        self
    }

    /// Override `(dilation_d, dilation_h, dilation_w)`. Default
    /// `(1, 1, 1)`.
    #[inline]
    pub fn with_dilation(mut self, dilation_d: i32, dilation_h: i32, dilation_w: i32) -> Self {
        self.dilation_d = dilation_d;
        self.dilation_h = dilation_h;
        self.dilation_w = dilation_w;
        self
    }

    /// Override the group count. Default `1`.
    #[inline]
    pub fn with_groups(mut self, groups: i32) -> Self {
        self.groups = groups;
        self
    }
}

/// Args bundle for a Conv3d forward launch.
pub struct Conv3dArgs<'a, T: Element> {
    /// Input activations `[N, C_in, D_in, H_in, W_in]`.
    pub x: TensorRef<'a, T, 5>,
    /// Filter weights `[C_out, C_in/groups, D_filt, H_filt, W_filt]`.
    pub w: TensorRef<'a, T, 5>,
    /// Output activations `[N, C_out, D_out, H_out, W_out]`.
    pub y: TensorMut<'a, T, 5>,
}

/// Args bundle for a Conv3d data-gradient launch.
pub struct Conv3dBwArgs<'a, T: Element> {
    /// Filter weights `[C_out, C_in/groups, D_filt, H_filt, W_filt]`.
    pub w: TensorRef<'a, T, 5>,
    /// Upstream gradient `[N, C_out, D_out, H_out, W_out]`.
    pub dy: TensorRef<'a, T, 5>,
    /// Output gradient w.r.t. input `[N, C_in, D_in, H_in, W_in]`.
    pub dx: TensorMut<'a, T, 5>,
}

/// Args bundle for a Conv3d filter-gradient launch.
pub struct Conv3dDwArgs<'a, T: Element> {
    /// Input activations `[N, C_in, D_in, H_in, W_in]`.
    pub x: TensorRef<'a, T, 5>,
    /// Upstream gradient `[N, C_out, D_out, H_out, W_out]`.
    pub dy: TensorRef<'a, T, 5>,
    /// Output gradient w.r.t. the filter
    /// `[C_out, C_in/groups, D_filt, H_filt, W_filt]`.
    pub dw: TensorMut<'a, T, 5>,
}

/// 3-D convolution plan (cuDNN-backed) — forward + both backward
/// passes for NCDHW activations.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16`. **Algorithm pinning**:
/// FW = `IMPLICIT_GEMM` (algo 0), BW-data / BW-filter = `ALGO_1`.
///
/// Gated under `feature = "cudnn"` at the crate root.
pub struct Conv3dPlan<T: Element> {
    desc: Conv3dDescriptor,
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

impl<T: Element> Conv3dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &Conv3dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::Conv3dPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::Conv3dPlan: cuDNN Conv3d supports f32 / f64 / f16 / bf16",
            ));
        }
        if desc.batch <= 0
            || desc.c_in <= 0
            || desc.d_in <= 0
            || desc.h_in <= 0
            || desc.w_in <= 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: input shape extents must be > 0",
            ));
        }
        if desc.c_out <= 0 || desc.d_filt <= 0 || desc.h_filt <= 0 || desc.w_filt <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: filter shape extents must be > 0",
            ));
        }
        if desc.stride_d <= 0
            || desc.stride_h <= 0
            || desc.stride_w <= 0
            || desc.dilation_d <= 0
            || desc.dilation_h <= 0
            || desc.dilation_w <= 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: stride / dilation must be > 0",
            ));
        }
        if desc.pad_d < 0 || desc.pad_h < 0 || desc.pad_w < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: padding must be >= 0",
            ));
        }
        if desc.groups <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: groups must be > 0",
            ));
        }
        if desc.c_in % desc.groups != 0 || desc.c_out % desc.groups != 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: groups must divide both c_in and c_out",
            ));
        }
        let (d_out, h_out, w_out) = compute_output_dims(desc);
        if d_out <= 0 || h_out <= 0 || w_out <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: computed output dims <= 0",
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
            op: ConvKind::Conv3d as u16,
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

    /// Computed `(D_out, H_out, W_out)` output spatial extents.
    #[inline]
    pub fn output_dims(&self) -> (i32, i32, i32) {
        compute_output_dims(&self.desc)
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

    /// Materialize handle / descriptors and query BW-data workspace
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

    /// Materialize handle / descriptors and query BW-filter workspace
    /// size.
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
        args: Conv3dArgs<'_, T>,
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

    /// Run the data-gradient pass.
    pub fn run_bw_data(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: Conv3dBwArgs<'_, T>,
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

    /// Run the filter-gradient pass.
    pub fn run_dw(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: Conv3dDwArgs<'_, T>,
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
        let (d_out, h_out, w_out) = compute_output_dims(&self.desc);
        let c_in_per_group = self.desc.c_in / self.desc.groups;

        // x descriptor — rank-5 [N, C_in, D_in, H_in, W_in].
        let mut xd: cudnnTensorDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateTensorDescriptor(&mut xd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let x_dims = [
            self.desc.batch,
            self.desc.c_in,
            self.desc.d_in,
            self.desc.h_in,
            self.desc.w_in,
        ];
        // NCDHW strides: w stride = 1, h = w_in, d = h_in*w_in,
        // c = d_in*h_in*w_in, n = c_in*d_in*h_in*w_in.
        let s_w = 1;
        let s_h = self.desc.w_in;
        let s_d = self.desc.h_in * self.desc.w_in;
        let s_c = self.desc.d_in * self.desc.h_in * self.desc.w_in;
        let s_n = self.desc.c_in * s_c;
        let x_strides = [s_n, s_c, s_d, s_h, s_w];
        let status = unsafe {
            cudnnSetTensorNdDescriptor(
                xd,
                dt,
                5,
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

        // y descriptor — rank-5 [N, C_out, D_out, H_out, W_out].
        let mut yd: cudnnTensorDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateTensorDescriptor(&mut yd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let y_dims = [self.desc.batch, self.desc.c_out, d_out, h_out, w_out];
        let y_s_w = 1;
        let y_s_h = w_out;
        let y_s_d = h_out * w_out;
        let y_s_c = d_out * h_out * w_out;
        let y_s_n = self.desc.c_out * y_s_c;
        let y_strides = [y_s_n, y_s_c, y_s_d, y_s_h, y_s_w];
        let status = unsafe {
            cudnnSetTensorNdDescriptor(
                yd,
                dt,
                5,
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

        // w (filter) descriptor — rank-5 [C_out, C_in/groups, D, H, W].
        let mut wd: cudnnFilterDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateFilterDescriptor(&mut wd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let w_dims = [
            self.desc.c_out,
            c_in_per_group,
            self.desc.d_filt,
            self.desc.h_filt,
            self.desc.w_filt,
        ];
        let status = unsafe {
            cudnnSetFilterNdDescriptor(wd, dt, CUDNN_TENSOR_NCHW, 5, w_dims.as_ptr())
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyFilterDescriptor(wd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.w_desc.set(wd);

        // Convolution descriptor — array_length = 3 (spatial rank).
        let mut cd: cudnnConvolutionDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateConvolutionDescriptor(&mut cd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let pad_a = [self.desc.pad_d, self.desc.pad_h, self.desc.pad_w];
        let stride_a = [self.desc.stride_d, self.desc.stride_h, self.desc.stride_w];
        let dilation_a = [
            self.desc.dilation_d,
            self.desc.dilation_h,
            self.desc.dilation_w,
        ];
        let status = unsafe {
            cudnnSetConvolutionNdDescriptor(
                cd,
                3,
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

    fn check_fw_args(&self, args: &Conv3dArgs<'_, T>) -> Result<()> {
        let (d_out, h_out, w_out) = compute_output_dims(&self.desc);
        let c_in_per_group = self.desc.c_in / self.desc.groups;
        let x_shape = [
            self.desc.batch,
            self.desc.c_in,
            self.desc.d_in,
            self.desc.h_in,
            self.desc.w_in,
        ];
        let w_shape = [
            self.desc.c_out,
            c_in_per_group,
            self.desc.d_filt,
            self.desc.h_filt,
            self.desc.w_filt,
        ];
        let y_shape = [self.desc.batch, self.desc.c_out, d_out, h_out, w_out];
        if args.x.shape != x_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: x shape != [N, C_in, D_in, H_in, W_in]",
            ));
        }
        if args.w.shape != w_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: w shape != [C_out, C_in/groups, D_filt, H_filt, W_filt]",
            ));
        }
        if args.y.shape != y_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: y shape != [N, C_out, D_out, H_out, W_out]",
            ));
        }
        Ok(())
    }

    fn check_bw_data_args(&self, args: &Conv3dBwArgs<'_, T>) -> Result<()> {
        let (d_out, h_out, w_out) = compute_output_dims(&self.desc);
        let c_in_per_group = self.desc.c_in / self.desc.groups;
        let w_shape = [
            self.desc.c_out,
            c_in_per_group,
            self.desc.d_filt,
            self.desc.h_filt,
            self.desc.w_filt,
        ];
        let dy_shape = [self.desc.batch, self.desc.c_out, d_out, h_out, w_out];
        let dx_shape = [
            self.desc.batch,
            self.desc.c_in,
            self.desc.d_in,
            self.desc.h_in,
            self.desc.w_in,
        ];
        if args.w.shape != w_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: w shape != [C_out, C_in/groups, D_filt, H_filt, W_filt]",
            ));
        }
        if args.dy.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: dy shape != [N, C_out, D_out, H_out, W_out]",
            ));
        }
        if args.dx.shape != dx_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: dx shape != [N, C_in, D_in, H_in, W_in]",
            ));
        }
        Ok(())
    }

    fn check_dw_args(&self, args: &Conv3dDwArgs<'_, T>) -> Result<()> {
        let (d_out, h_out, w_out) = compute_output_dims(&self.desc);
        let c_in_per_group = self.desc.c_in / self.desc.groups;
        let x_shape = [
            self.desc.batch,
            self.desc.c_in,
            self.desc.d_in,
            self.desc.h_in,
            self.desc.w_in,
        ];
        let dy_shape = [self.desc.batch, self.desc.c_out, d_out, h_out, w_out];
        let dw_shape = [
            self.desc.c_out,
            c_in_per_group,
            self.desc.d_filt,
            self.desc.h_filt,
            self.desc.w_filt,
        ];
        if args.x.shape != x_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: x shape != [N, C_in, D_in, H_in, W_in]",
            ));
        }
        if args.dy.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: dy shape != [N, C_out, D_out, H_out, W_out]",
            ));
        }
        if args.dw.shape != dw_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::Conv3dPlan: dw shape != [C_out, C_in/groups, D_filt, H_filt, W_filt]",
            ));
        }
        Ok(())
    }
}

impl<T: Element> Drop for Conv3dPlan<T> {
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

#[inline]
fn compute_output_dims(d: &Conv3dDescriptor) -> (i32, i32, i32) {
    let d_eff = d.dilation_d * (d.d_filt - 1) + 1;
    let h_eff = d.dilation_h * (d.h_filt - 1) + 1;
    let w_eff = d.dilation_w * (d.w_filt - 1) + 1;
    let d_out = (d.d_in + 2 * d.pad_d - d_eff) / d.stride_d + 1;
    let h_out = (d.h_in + 2 * d.pad_h - h_eff) / d.stride_h + 1;
    let w_out = (d.w_in + 2 * d.pad_w - w_eff) / d.stride_w + 1;
    (d_out, h_out, w_out)
}

#[inline]
fn cudnn_dtype<T: Element>() -> i32 {
    match T::KIND {
        ElementKind::F32 => CUDNN_DATA_FLOAT,
        ElementKind::F64 => CUDNN_DATA_DOUBLE,
        ElementKind::F16 => CUDNN_DATA_HALF,
        ElementKind::Bf16 => CUDNN_DATA_BFLOAT16,
        _ => unreachable!("Conv3dPlan::select gates on F32/F64/F16/Bf16"),
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
