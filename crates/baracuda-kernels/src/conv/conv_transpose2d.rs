//! ConvTranspose2d — 2-D transposed convolution via cuDNN.
//!
//! Transposed convolution (a.k.a. "fractionally-strided" or "deconv") is
//! the gradient of a Conv2d with respect to its input. cuDNN does
//! **not** expose a direct "transpose forward" entry point — instead,
//! we configure descriptors as if for a synthetic dense conv mapping
//! `[N, c_out, H_out, W_out] → [N, c_in, H_in, W_in]`, then dispatch
//! through the conv backward-data / forward exec entry points with the
//! roles swapped:
//!
//! - **ConvTranspose forward** (`y = x ★ᵀ w`) maps to
//!   `cudnnConvolutionBackwardData(w, x, y)`.
//! - **ConvTranspose backward-data** (`dx` from `dy`, `w`) maps to
//!   `cudnnConvolutionForward(dy, w, dx)`.
//! - **ConvTranspose backward-filter** (`dw` from `x`, `dy`) maps to
//!   `cudnnConvolutionBackwardFilter` with the same role swap as FW.
//!
//! Tensor shapes follow PyTorch's `nn.ConvTranspose2d`:
//!
//! - Input: `[N, C_in, H_in, W_in]`.
//! - Filter: `[C_in, C_out / groups, kH, kW]` — note the **c_in is
//!   first**, opposite of plain Conv2d.
//! - Output: `[N, C_out, H_out, W_out]` with
//!   `H_out = (H_in - 1)·stride_h - 2·pad_h + dilation_h·(kH - 1) +
//!   output_padding_h + 1`, analogous for `W_out`.
//!
//! Cross-correlation only (PyTorch parity). Dtypes: `f32`, `f64`,
//! `f16`, `bf16`. Algorithm pinning matches `Conv2dPlan`. Gated under
//! `feature = "cudnn"`.

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
    cudnnSetConvolutionGroupCount, cudnnSetFilter4dDescriptor, cudnnSetStream,
    cudnnSetTensor4dDescriptor, cudnnTensorDescriptor_t, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_BFLOAT16, CUDNN_DATA_DOUBLE, CUDNN_DATA_FLOAT,
    CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, ConvKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a 2-D transposed convolution over NCHW tensors.
///
/// Output spatial extents follow PyTorch's `nn.ConvTranspose2d`:
/// `H_out = (H_in - 1)·stride_h - 2·pad_h + dilation_h·(kH - 1) +
/// output_padding_h + 1` (and analogous for `W_out`).
#[derive(Copy, Clone, Debug)]
pub struct ConvTranspose2dDescriptor {
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
    /// Filter height `kH`.
    pub h_filt: i32,
    /// Filter width `kW`.
    pub w_filt: i32,
    /// Implicit zero-padding on each side of the **output** height
    /// axis (matches PyTorch's `padding`).
    pub pad_h: i32,
    /// Implicit zero-padding on each side of the **output** width
    /// axis.
    pub pad_w: i32,
    /// Stride along H (semantically the up-sampling factor).
    pub stride_h: i32,
    /// Stride along W.
    pub stride_w: i32,
    /// Dilation along H.
    pub dilation_h: i32,
    /// Dilation along W.
    pub dilation_w: i32,
    /// Output-side padding along H (PyTorch's `output_padding`). Used
    /// to disambiguate the output extent when `stride > 1` (multiple
    /// input extents can map to the same output formula).
    pub output_pad_h: i32,
    /// Output-side padding along W.
    pub output_pad_w: i32,
    /// Group count.
    pub groups: i32,
    /// Element dtype. Must be `F32`, `F64`, `F16`, or `Bf16`.
    pub element: ElementKind,
}

/// Args bundle for a ConvTranspose2d forward launch.
pub struct ConvTranspose2dArgs<'a, T: Element> {
    /// Input activations `[N, C_in, H_in, W_in]`.
    pub x: TensorRef<'a, T, 4>,
    /// Filter weights `[C_in, C_out / groups, kH, kW]` (PyTorch
    /// convention — c_in is first).
    pub w: TensorRef<'a, T, 4>,
    /// Output activations `[N, C_out, H_out, W_out]`.
    pub y: TensorMut<'a, T, 4>,
}

/// Args bundle for a ConvTranspose2d data-gradient launch.
pub struct ConvTranspose2dBwArgs<'a, T: Element> {
    /// Filter weights `[C_in, C_out / groups, kH, kW]`.
    pub w: TensorRef<'a, T, 4>,
    /// Upstream gradient `[N, C_out, H_out, W_out]`.
    pub dy: TensorRef<'a, T, 4>,
    /// Output gradient w.r.t. input `[N, C_in, H_in, W_in]`.
    pub dx: TensorMut<'a, T, 4>,
}

/// Args bundle for a ConvTranspose2d filter-gradient launch.
pub struct ConvTranspose2dDwArgs<'a, T: Element> {
    /// Input activations `[N, C_in, H_in, W_in]`.
    pub x: TensorRef<'a, T, 4>,
    /// Upstream gradient `[N, C_out, H_out, W_out]`.
    pub dy: TensorRef<'a, T, 4>,
    /// Output gradient w.r.t. the filter `[C_in, C_out / groups, kH, kW]`.
    pub dw: TensorMut<'a, T, 4>,
}

/// 2-D transposed-convolution plan (cuDNN-backed).
///
/// Implementation strategy: configure descriptors as if for a
/// **synthetic dense forward conv** that maps the ConvTranspose's
/// output to its input. Then:
///
/// - `run_fw` invokes `cudnnConvolutionBackwardData` (the synthetic
///   conv's backward-data direction recovers the ConvTranspose's
///   forward result).
/// - `run_bw_data` invokes `cudnnConvolutionForward` (the synthetic
///   conv's forward direction recovers the ConvTranspose's
///   backward-data result).
/// - `run_dw` invokes `cudnnConvolutionBackwardFilter` directly — the
///   filter-gradient role-swap only changes which tensor is the
///   "input" vs "output" of the synthetic forward conv; the filter
///   gradient computation is symmetric.
pub struct ConvTranspose2dPlan<T: Element> {
    desc: ConvTranspose2dDescriptor,
    sku: KernelSku,
    handle: Cell<cudnnHandle_t>,
    // Descriptors are named relative to the **synthetic forward conv**:
    // synth_x = ConvTranspose output (`[N, C_out, H_out, W_out]`).
    // synth_y = ConvTranspose input  (`[N, C_in,  H_in,  W_in]`).
    // synth_w = filter `[C_in, C_out/groups, kH, kW]` (K=c_in, C=c_out).
    synth_x_desc: Cell<cudnnTensorDescriptor_t>,
    synth_w_desc: Cell<cudnnFilterDescriptor_t>,
    synth_y_desc: Cell<cudnnTensorDescriptor_t>,
    conv_desc: Cell<cudnnConvolutionDescriptor_t>,
    workspace_bytes_fw: Cell<usize>,
    workspace_bytes_bw_data: Cell<usize>,
    workspace_bytes_bw_filter: Cell<usize>,
    _marker: PhantomData<T>,
}

impl<T: Element> ConvTranspose2dPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &ConvTranspose2dDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::ConvTranspose2dPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::F16 | ElementKind::Bf16
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::ConvTranspose2dPlan: dtype must be f32 / f64 / f16 / bf16",
            ));
        }
        if desc.batch <= 0 || desc.c_in <= 0 || desc.h_in <= 0 || desc.w_in <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: input shape extents must be > 0",
            ));
        }
        if desc.c_out <= 0 || desc.h_filt <= 0 || desc.w_filt <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: filter shape extents must be > 0",
            ));
        }
        if desc.stride_h <= 0
            || desc.stride_w <= 0
            || desc.dilation_h <= 0
            || desc.dilation_w <= 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: stride / dilation must be > 0",
            ));
        }
        if desc.pad_h < 0 || desc.pad_w < 0 || desc.output_pad_h < 0 || desc.output_pad_w < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: padding / output_padding must be >= 0",
            ));
        }
        if desc.groups <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: groups must be > 0",
            ));
        }
        if desc.c_in % desc.groups != 0 || desc.c_out % desc.groups != 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: groups must divide both c_in and c_out",
            ));
        }
        // PyTorch requires output_padding < max(stride, dilation).
        if desc.output_pad_h >= desc.stride_h.max(desc.dilation_h)
            || desc.output_pad_w >= desc.stride_w.max(desc.dilation_w)
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: output_padding must be < max(stride, dilation)",
            ));
        }
        let (h_out, w_out) = compute_output_dims(desc);
        if h_out <= 0 || w_out <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: computed output dims <= 0",
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
            op: ConvKind::ConvTranspose2d as u16,
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

    /// Computed `(H_out, W_out)` output spatial extents.
    #[inline]
    pub fn output_dims(&self) -> (i32, i32) {
        compute_output_dims(&self.desc)
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

    /// Query FW workspace size — under the hood asks cuDNN for the
    /// **backward-data** workspace size of the synthetic forward conv.
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

    /// Query BW-data workspace size — under the hood asks cuDNN for
    /// the **forward** workspace size of the synthetic forward conv.
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

    /// Run the forward pass (`y = x ★ᵀ w`). Maps to
    /// `cudnnConvolutionBackwardData` with the synthetic forward conv
    /// descriptors set up so that `dy ↦ dx` recovers the ConvTranspose
    /// output from the input.
    pub fn run_fw(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: ConvTranspose2dArgs<'_, T>,
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

        // cuDNN BackwardData: dx (= ConvT.y) ← w, dy (= ConvT.x).
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

    /// Run the data-gradient pass. Maps to `cudnnConvolutionForward`.
    pub fn run_bw_data(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: ConvTranspose2dBwArgs<'_, T>,
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

        // Synthetic forward: dy (= ConvT.dx) ← synth_x = ConvT.dy, w.
        // ConvT.dy has the synthetic forward's *input* role (= synth_x
        // shape [N, C_out, H_out, W_out]); ConvT.dx has the synthetic
        // forward's *output* role (= synth_y shape [N, C_in, H_in, W_in]).
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

    /// Run the filter-gradient pass. Maps to
    /// `cudnnConvolutionBackwardFilter` with synthetic input = ConvT.dy
    /// and synthetic dy = ConvT.x (the role swap mirrors the FW path).
    pub fn run_dw(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: ConvTranspose2dDwArgs<'_, T>,
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
        let (h_out, w_out) = compute_output_dims(&self.desc);
        // For the synthetic forward conv:
        //   synth_x = [N, c_out, H_out, W_out]  (ConvTranspose output role)
        //   synth_w = [c_in (K), c_out/groups (C/groups), kH, kW]
        //   synth_y = [N, c_in, H_in, W_in]    (ConvTranspose input role)
        // i.e. it maps c_out → c_in via the same pad/stride/dilation as
        // the original ConvTranspose. PyTorch's filter shape
        // [c_in, c_out/groups, ...] matches cuDNN's [K, C/groups, ...]
        // when K = c_in.
        let c_out_per_group = self.desc.c_out / self.desc.groups;

        // synth_x descriptor — [N, C_out, H_out, W_out].
        let mut xd: cudnnTensorDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateTensorDescriptor(&mut xd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let status = unsafe {
            cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, dt, self.desc.batch, self.desc.c_out, h_out, w_out)
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(xd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.synth_x_desc.set(xd);

        // synth_y descriptor — [N, C_in, H_in, W_in].
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
                self.desc.c_in,
                self.desc.h_in,
                self.desc.w_in,
            )
        };
        if status != 0 {
            unsafe {
                let _ = cudnnDestroyTensorDescriptor(yd);
            }
            return Err(Error::CutlassInternal(-status));
        }
        self.synth_y_desc.set(yd);

        // synth_w descriptor — [C_in, C_out/groups, kH, kW].
        let mut wd: cudnnFilterDescriptor_t = core::ptr::null_mut();
        let status = unsafe { cudnnCreateFilterDescriptor(&mut wd as *mut _) };
        if status != 0 {
            return Err(Error::CutlassInternal(-status));
        }
        let status = unsafe {
            cudnnSetFilter4dDescriptor(
                wd,
                dt,
                CUDNN_TENSOR_NCHW,
                self.desc.c_in,
                c_out_per_group,
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
        self.synth_w_desc.set(wd);

        // Convolution descriptor — same pad/stride/dilation as the
        // ConvTranspose. The synthetic forward maps the H_out → H_in
        // direction using cudnn's standard formula.
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

    fn check_fw_args(&self, args: &ConvTranspose2dArgs<'_, T>) -> Result<()> {
        let (h_out, w_out) = compute_output_dims(&self.desc);
        let c_out_per_group = self.desc.c_out / self.desc.groups;
        let x_shape = [self.desc.batch, self.desc.c_in, self.desc.h_in, self.desc.w_in];
        let w_shape = [
            self.desc.c_in,
            c_out_per_group,
            self.desc.h_filt,
            self.desc.w_filt,
        ];
        let y_shape = [self.desc.batch, self.desc.c_out, h_out, w_out];
        if args.x.shape != x_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: x shape != [N, C_in, H_in, W_in]",
            ));
        }
        if args.w.shape != w_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: w shape != [C_in, C_out/groups, kH, kW]",
            ));
        }
        if args.y.shape != y_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: y shape != [N, C_out, H_out, W_out]",
            ));
        }
        Ok(())
    }

    fn check_bw_data_args(&self, args: &ConvTranspose2dBwArgs<'_, T>) -> Result<()> {
        let (h_out, w_out) = compute_output_dims(&self.desc);
        let c_out_per_group = self.desc.c_out / self.desc.groups;
        let w_shape = [
            self.desc.c_in,
            c_out_per_group,
            self.desc.h_filt,
            self.desc.w_filt,
        ];
        let dy_shape = [self.desc.batch, self.desc.c_out, h_out, w_out];
        let dx_shape = [self.desc.batch, self.desc.c_in, self.desc.h_in, self.desc.w_in];
        if args.w.shape != w_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: w shape != [C_in, C_out/groups, kH, kW]",
            ));
        }
        if args.dy.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: dy shape != [N, C_out, H_out, W_out]",
            ));
        }
        if args.dx.shape != dx_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: dx shape != [N, C_in, H_in, W_in]",
            ));
        }
        Ok(())
    }

    fn check_dw_args(&self, args: &ConvTranspose2dDwArgs<'_, T>) -> Result<()> {
        let (h_out, w_out) = compute_output_dims(&self.desc);
        let c_out_per_group = self.desc.c_out / self.desc.groups;
        let x_shape = [self.desc.batch, self.desc.c_in, self.desc.h_in, self.desc.w_in];
        let dy_shape = [self.desc.batch, self.desc.c_out, h_out, w_out];
        let dw_shape = [
            self.desc.c_in,
            c_out_per_group,
            self.desc.h_filt,
            self.desc.w_filt,
        ];
        if args.x.shape != x_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: x shape != [N, C_in, H_in, W_in]",
            ));
        }
        if args.dy.shape != dy_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: dy shape != [N, C_out, H_out, W_out]",
            ));
        }
        if args.dw.shape != dw_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::ConvTranspose2dPlan: dw shape != [C_in, C_out/groups, kH, kW]",
            ));
        }
        Ok(())
    }
}

impl<T: Element> Drop for ConvTranspose2dPlan<T> {
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

/// `H_out = (H_in - 1)·stride_h - 2·pad_h + dilation_h·(kH - 1) +
/// output_padding_h + 1` (PyTorch convention).
#[inline]
fn compute_output_dims(d: &ConvTranspose2dDescriptor) -> (i32, i32) {
    let h_out = (d.h_in - 1) * d.stride_h - 2 * d.pad_h
        + d.dilation_h * (d.h_filt - 1)
        + d.output_pad_h
        + 1;
    let w_out = (d.w_in - 1) * d.stride_w - 2 * d.pad_w
        + d.dilation_w * (d.w_filt - 1)
        + d.output_pad_w
        + 1;
    (h_out, w_out)
}

#[inline]
fn cudnn_dtype<T: Element>() -> i32 {
    match T::KIND {
        ElementKind::F32 => CUDNN_DATA_FLOAT,
        ElementKind::F64 => CUDNN_DATA_DOUBLE,
        ElementKind::F16 => CUDNN_DATA_HALF,
        ElementKind::Bf16 => CUDNN_DATA_BFLOAT16,
        _ => unreachable!("ConvTranspose2dPlan::select gates on F32/F64/F16/Bf16"),
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
