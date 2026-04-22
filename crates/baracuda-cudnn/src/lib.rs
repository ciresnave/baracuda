//! Safe Rust wrappers for NVIDIA cuDNN.
//!
//! v0.1 covers the handle, 4-D tensor descriptors, activation descriptors,
//! and `cudnnActivationForward` — enough to demonstrate the loader works
//! and to run simple ML pointwise ops.
//!
//! Convolution, pooling, batch-norm, RNN, and the modern graph API all
//! land in follow-ups. The plan documents explicitly wrapping the cuDNN
//! **backend/graph API** rather than the C++ frontend; this crate takes
//! that path.

#![warn(missing_debug_implementations)]

use baracuda_cudnn_sys::{
    cudnn, cudnnActivationDescriptor_t, cudnnActivationMode_t, cudnnBackendAttributeName_t,
    cudnnBackendAttributeType_t, cudnnBackendDescriptorType_t, cudnnBackendDescriptor_t,
    cudnnBatchNormMode_t, cudnnConvolutionBwdDataAlgo_t, cudnnConvolutionBwdFilterAlgo_t,
    cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t,
    cudnnDataType_t, cudnnDropoutDescriptor_t, cudnnFilterDescriptor_t, cudnnHandle_t,
    cudnnIndicesType_t, cudnnLRNDescriptor_t, cudnnNanPropagation_t, cudnnOpTensorDescriptor_t,
    cudnnOpTensorOp_t, cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnReduceTensorDescriptor_t,
    cudnnReduceTensorIndices_t, cudnnReduceTensorOp_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t,
    cudnnStatus_t, cudnnTensorDescriptor_t, cudnnTensorFormat_t,
};
use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_types::DeviceRepr;

/// Error type for cuDNN operations.
pub type Error = baracuda_core::Error<cudnnStatus_t>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: cudnnStatus_t) -> Result<()> {
    Error::check(status)
}

/// cuDNN context handle.
pub struct Handle {
    handle: cudnnHandle_t,
}

unsafe impl Send for Handle {}

impl core::fmt::Debug for Handle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("cudnn::Handle")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Handle {
    /// Create a new cuDNN handle.
    pub fn new() -> Result<Self> {
        let c = cudnn()?;
        let cu = c.cudnn_create()?;
        let mut h: cudnnHandle_t = core::ptr::null_mut();
        check(unsafe { cu(&mut h) })?;
        Ok(Self { handle: h })
    }

    /// Bind operations on this handle to `stream`.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let c = cudnn()?;
        let cu = c.cudnn_set_stream()?;
        check(unsafe { cu(self.handle, stream.as_raw() as _) })
    }

    /// Raw handle.
    #[inline]
    pub fn as_raw(&self) -> cudnnHandle_t {
        self.handle
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// cuDNN library version as a packed integer (e.g. `9106` for 9.1.6).
///
/// Does **not** require an initialized handle.
pub fn version() -> Result<usize> {
    let c = cudnn()?;
    let cu = c.cudnn_get_version()?;
    // SAFETY: cudnnGetVersion has no error path.
    Ok(unsafe { cu() })
}

/// Element dtype for a tensor.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DType {
    /// Single-precision 32-bit floating point.
    F32,
    /// Double-precision 64-bit floating point.
    F64,
    /// IEEE 754 half-precision (16-bit) floating point.
    F16,
    /// Brain half-precision (16-bit) floating point.
    BF16,
    /// 8-bit signed integer (quantized inference).
    I8,
    /// 32-bit signed integer (integer accumulators).
    I32,
}

impl DType {
    fn raw(self) -> cudnnDataType_t {
        match self {
            DType::F32 => cudnnDataType_t::Float,
            DType::F64 => cudnnDataType_t::Double,
            DType::F16 => cudnnDataType_t::Half,
            DType::BF16 => cudnnDataType_t::BFloat16,
            DType::I8 => cudnnDataType_t::Int8,
            DType::I32 => cudnnDataType_t::Int32,
        }
    }
}

/// Trait mapping Rust element types to their cuDNN [`DType`] tag.
///
/// Lets generic code accept "a tensor of T" and recover the cuDNN dtype
/// with `T::DTYPE`, instead of threading a `DType` argument through every
/// call. Useful for tensor-descriptor builders:
///
/// ```no_run
/// use baracuda_cudnn::{CudnnDataType, DType, TensorDescriptor, TensorFormat};
///
/// fn make_nchw<T: CudnnDataType>(n: i32, c: i32, h: i32, w: i32)
///     -> baracuda_cudnn::Result<TensorDescriptor>
/// {
///     TensorDescriptor::new_4d(TensorFormat::Nchw, T::DTYPE, n, c, h, w)
/// }
///
/// let desc = make_nchw::<f32>(1, 3, 224, 224)?;
/// # Ok::<(), baracuda_cudnn::Error>(())
/// ```
///
/// Implementors: `f32`, `f64`, [`baracuda_types::Half`],
/// [`baracuda_types::BFloat16`], `i8`, `i32`.
pub trait CudnnDataType: DeviceRepr + Copy + 'static {
    /// The [`DType`] tag cuDNN uses for this scalar type.
    const DTYPE: DType;
}

impl CudnnDataType for f32 {
    const DTYPE: DType = DType::F32;
}
impl CudnnDataType for f64 {
    const DTYPE: DType = DType::F64;
}
impl CudnnDataType for baracuda_types::Half {
    const DTYPE: DType = DType::F16;
}
impl CudnnDataType for baracuda_types::BFloat16 {
    const DTYPE: DType = DType::BF16;
}
impl CudnnDataType for i8 {
    const DTYPE: DType = DType::I8;
}
impl CudnnDataType for i32 {
    const DTYPE: DType = DType::I32;
}

/// Memory layout for a 4-D tensor.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum TensorFormat {
    /// Batch × Channels × Height × Width (channels-first, the cuDNN default).
    #[default]
    Nchw,
    /// Batch × Height × Width × Channels (channels-last).
    Nhwc,
}

impl TensorFormat {
    fn raw(self) -> cudnnTensorFormat_t {
        match self {
            TensorFormat::Nchw => cudnnTensorFormat_t::Nchw,
            TensorFormat::Nhwc => cudnnTensorFormat_t::Nhwc,
        }
    }
}

/// A 4-D tensor descriptor.
pub struct TensorDescriptor {
    desc: cudnnTensorDescriptor_t,
}

unsafe impl Send for TensorDescriptor {}

impl core::fmt::Debug for TensorDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorDescriptor")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl TensorDescriptor {
    /// Describe an `N × C × H × W` tensor with the given format and dtype.
    pub fn new_4d(
        format: TensorFormat,
        dtype: DType,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> Result<Self> {
        let cu_crate = cudnn()?;
        let create = cu_crate.cudnn_create_tensor_descriptor()?;
        let set = cu_crate.cudnn_set_tensor_4d_descriptor()?;
        let mut desc: cudnnTensorDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe { set(this.desc, format.raw(), dtype.raw(), n, c, h, w) })?;
        Ok(this)
    }

    /// Describe an N-dimensional tensor. `dims` and `strides` must have the
    /// same length (≤8) and correspond to a valid, non-overlapping
    /// cuDNN-supported layout.
    pub fn new_nd(dtype: DType, dims: &[i32], strides: &[i32]) -> Result<Self> {
        assert_eq!(
            dims.len(),
            strides.len(),
            "dims/strides length mismatch for Nd tensor descriptor"
        );
        let cu_crate = cudnn()?;
        let create = cu_crate.cudnn_create_tensor_descriptor()?;
        let set = cu_crate.cudnn_set_tensor_nd_descriptor()?;
        let mut desc: cudnnTensorDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe {
            set(
                this.desc,
                dtype.raw(),
                dims.len() as core::ffi::c_int,
                dims.as_ptr(),
                strides.as_ptr(),
            )
        })?;
        Ok(this)
    }

    /// Raw descriptor. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cudnnTensorDescriptor_t {
        self.desc
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_tensor_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Activation function kind.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ActivationMode {
    Relu,
    Sigmoid,
    Tanh,
    /// Clipped ReLU: `min(max(0, x), ceiling)`.
    ClippedRelu,
    Elu,
    Identity,
    /// Swish / SiLU: `x · sigmoid(x)`.
    Swish,
}

impl ActivationMode {
    fn raw(self) -> cudnnActivationMode_t {
        match self {
            ActivationMode::Relu => cudnnActivationMode_t::Relu,
            ActivationMode::Sigmoid => cudnnActivationMode_t::Sigmoid,
            ActivationMode::Tanh => cudnnActivationMode_t::Tanh,
            ActivationMode::ClippedRelu => cudnnActivationMode_t::ClippedRelu,
            ActivationMode::Elu => cudnnActivationMode_t::Elu,
            ActivationMode::Identity => cudnnActivationMode_t::Identity,
            ActivationMode::Swish => cudnnActivationMode_t::Swish,
        }
    }
}

/// An activation descriptor.
pub struct ActivationDescriptor {
    desc: cudnnActivationDescriptor_t,
}

unsafe impl Send for ActivationDescriptor {}

impl core::fmt::Debug for ActivationDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ActivationDescriptor")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl ActivationDescriptor {
    /// Create a descriptor for `mode`. `coef` is only used by ClippedReLU
    /// (ceiling) and ELU (α); pass `0.0` when irrelevant.
    pub fn new(mode: ActivationMode, coef: f64) -> Result<Self> {
        let c = cudnn()?;
        let create = c.cudnn_create_activation_descriptor()?;
        let set = c.cudnn_set_activation_descriptor()?;
        let mut desc: cudnnActivationDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe {
            set(
                this.desc,
                mode.raw(),
                cudnnNanPropagation_t::PropagateNan,
                coef,
            )
        })?;
        Ok(this)
    }

    /// Raw descriptor.
    #[inline]
    pub fn as_raw(&self) -> cudnnActivationDescriptor_t {
        self.desc
    }
}

impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_activation_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Compute `y = alpha * activation(x) + beta * y` element-wise.
///
/// `x` and `y` may alias (in-place activation is legal).
#[allow(clippy::too_many_arguments)]
pub fn activation_forward<T: DeviceRepr>(
    handle: &Handle,
    activation: &ActivationDescriptor,
    alpha: f32,
    x_desc: &TensorDescriptor,
    x: &DeviceBuffer<T>,
    beta: f32,
    y_desc: &TensorDescriptor,
    y: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_activation_forward()?;
    check(unsafe {
        cu(
            handle.handle,
            activation.desc,
            &alpha as *const f32 as *const core::ffi::c_void,
            x_desc.desc,
            x.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            y_desc.desc,
            y.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

// ---- convolution ---------------------------------------------------------

/// `N × C × H × W` 4-D filter.
pub struct FilterDescriptor {
    desc: cudnnFilterDescriptor_t,
}

unsafe impl Send for FilterDescriptor {}

impl core::fmt::Debug for FilterDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FilterDescriptor")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl FilterDescriptor {
    /// Describe a 4-D filter (convolution weight): K output channels, C input
    /// channels, H filter height, W filter width.
    pub fn new_4d(
        format: TensorFormat,
        dtype: DType,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> Result<Self> {
        let cu = cudnn()?;
        let create = cu.cudnn_create_filter_descriptor()?;
        let set = cu.cudnn_set_filter_4d_descriptor()?;
        let mut desc: cudnnFilterDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe { set(this.desc, dtype.raw(), format.raw(), k, c, h, w) })?;
        Ok(this)
    }

    /// Raw descriptor.
    #[inline]
    pub fn as_raw(&self) -> cudnnFilterDescriptor_t {
        self.desc
    }
}

impl Drop for FilterDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_filter_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Convolution mathematical mode.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum ConvMode {
    /// True convolution (flips the filter).
    Convolution,
    /// Cross-correlation — what ML frameworks mean by "convolution". **Default.**
    #[default]
    CrossCorrelation,
}

impl ConvMode {
    fn raw(self) -> cudnnConvolutionMode_t {
        match self {
            ConvMode::Convolution => cudnnConvolutionMode_t::Convolution,
            ConvMode::CrossCorrelation => cudnnConvolutionMode_t::CrossCorrelation,
        }
    }
}

/// Forward-convolution algorithm selector. `Gemm` is the most broadly
/// supported; `ImplicitPrecompGemm` / `Winograd` are faster where
/// applicable.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum FwdAlgo {
    #[default]
    ImplicitGemm,
    ImplicitPrecompGemm,
    Gemm,
    Direct,
    Fft,
    FftTiling,
    Winograd,
    WinogradNonfused,
}

impl FwdAlgo {
    fn raw(self) -> cudnnConvolutionFwdAlgo_t {
        match self {
            FwdAlgo::ImplicitGemm => cudnnConvolutionFwdAlgo_t::ImplicitGemm,
            FwdAlgo::ImplicitPrecompGemm => cudnnConvolutionFwdAlgo_t::ImplicitPrecompGemm,
            FwdAlgo::Gemm => cudnnConvolutionFwdAlgo_t::Gemm,
            FwdAlgo::Direct => cudnnConvolutionFwdAlgo_t::Direct,
            FwdAlgo::Fft => cudnnConvolutionFwdAlgo_t::Fft,
            FwdAlgo::FftTiling => cudnnConvolutionFwdAlgo_t::FftTiling,
            FwdAlgo::Winograd => cudnnConvolutionFwdAlgo_t::Winograd,
            FwdAlgo::WinogradNonfused => cudnnConvolutionFwdAlgo_t::WinogradNonfused,
        }
    }
}

/// Convolution descriptor: padding, stride, dilation, and compute dtype.
pub struct ConvolutionDescriptor {
    desc: cudnnConvolutionDescriptor_t,
}

unsafe impl Send for ConvolutionDescriptor {}

impl core::fmt::Debug for ConvolutionDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ConvolutionDescriptor")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl ConvolutionDescriptor {
    /// 2-D convolution descriptor.
    /// - `pad_h`/`pad_w`: zero-padding on each side of the H/W axis.
    /// - `stride_h`/`stride_w`: per-axis stride.
    /// - `dilation_h`/`dilation_w`: per-axis dilation (1 = standard).
    /// - `mode`: [`ConvMode::CrossCorrelation`] for ML; [`ConvMode::Convolution`] for true math convolution.
    /// - `compute`: accumulation dtype (pass [`DType::F32`] even for mixed-precision FP16 input/output).
    #[allow(clippy::too_many_arguments)]
    pub fn new_2d(
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        mode: ConvMode,
        compute: DType,
    ) -> Result<Self> {
        let cu = cudnn()?;
        let create = cu.cudnn_create_convolution_descriptor()?;
        let set = cu.cudnn_set_convolution_2d_descriptor()?;
        let mut desc: cudnnConvolutionDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe {
            set(
                this.desc,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                mode.raw(),
                compute.raw(),
            )
        })?;
        Ok(this)
    }

    /// Compute the `N × C × H × W` shape this convolution would produce given
    /// the input + filter descriptors.
    pub fn output_dim_2d(
        &self,
        input: &TensorDescriptor,
        filter: &FilterDescriptor,
    ) -> Result<(i32, i32, i32, i32)> {
        let cu = cudnn()?;
        let q = cu.cudnn_get_convolution_2d_forward_output_dim()?;
        let mut n: core::ffi::c_int = 0;
        let mut c: core::ffi::c_int = 0;
        let mut h: core::ffi::c_int = 0;
        let mut w: core::ffi::c_int = 0;
        check(unsafe {
            q(
                self.desc,
                input.desc,
                filter.desc,
                &mut n,
                &mut c,
                &mut h,
                &mut w,
            )
        })?;
        Ok((n, c, h, w))
    }

    /// Raw descriptor.
    #[inline]
    pub fn as_raw(&self) -> cudnnConvolutionDescriptor_t {
        self.desc
    }
}

impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_convolution_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Query the minimum workspace (bytes) required to run `algo` with the given
/// tensor / filter / conv descriptors.
pub fn convolution_forward_workspace_size(
    handle: &Handle,
    x: &TensorDescriptor,
    w: &FilterDescriptor,
    conv: &ConvolutionDescriptor,
    y: &TensorDescriptor,
    algo: FwdAlgo,
) -> Result<usize> {
    let cu = cudnn()?;
    let q = cu.cudnn_get_convolution_forward_workspace_size()?;
    let mut size: usize = 0;
    check(unsafe {
        q(
            handle.handle,
            x.desc,
            w.desc,
            conv.desc,
            y.desc,
            algo.raw(),
            &mut size,
        )
    })?;
    Ok(size)
}

/// `Y = alpha * conv(X, W) + beta * Y` (forward pass).
///
/// `workspace` must be at least the size returned by
/// [`convolution_forward_workspace_size`].
#[allow(clippy::too_many_arguments)]
pub fn convolution_forward<T: DeviceRepr>(
    handle: &Handle,
    alpha: f32,
    x_desc: &TensorDescriptor,
    x: &DeviceBuffer<T>,
    w_desc: &FilterDescriptor,
    w: &DeviceBuffer<T>,
    conv: &ConvolutionDescriptor,
    algo: FwdAlgo,
    workspace: &mut DeviceBuffer<u8>,
    beta: f32,
    y_desc: &TensorDescriptor,
    y: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_convolution_forward()?;
    check(unsafe {
        cu(
            handle.handle,
            &alpha as *const f32 as *const core::ffi::c_void,
            x_desc.desc,
            x.as_raw().0 as *const core::ffi::c_void,
            w_desc.desc,
            w.as_raw().0 as *const core::ffi::c_void,
            conv.desc,
            algo.raw(),
            workspace.as_raw().0 as *mut core::ffi::c_void,
            workspace.byte_size(),
            &beta as *const f32 as *const core::ffi::c_void,
            y_desc.desc,
            y.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

// ---- convolution backward ------------------------------------------------

/// Backward-data convolution algorithm selector.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum BwdDataAlgo {
    #[default]
    Algo0,
    Algo1,
    Fft,
    FftTiling,
    Winograd,
    WinogradNonfused,
}

impl BwdDataAlgo {
    fn raw(self) -> cudnnConvolutionBwdDataAlgo_t {
        match self {
            Self::Algo0 => cudnnConvolutionBwdDataAlgo_t::Algo0,
            Self::Algo1 => cudnnConvolutionBwdDataAlgo_t::Algo1,
            Self::Fft => cudnnConvolutionBwdDataAlgo_t::Fft,
            Self::FftTiling => cudnnConvolutionBwdDataAlgo_t::FftTiling,
            Self::Winograd => cudnnConvolutionBwdDataAlgo_t::Winograd,
            Self::WinogradNonfused => cudnnConvolutionBwdDataAlgo_t::WinogradNonfused,
        }
    }
}

/// Backward-filter convolution algorithm selector.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum BwdFilterAlgo {
    #[default]
    Algo0,
    Algo1,
    Fft,
    Algo3,
    Winograd,
    WinogradNonfused,
    FftTiling,
}

impl BwdFilterAlgo {
    fn raw(self) -> cudnnConvolutionBwdFilterAlgo_t {
        match self {
            Self::Algo0 => cudnnConvolutionBwdFilterAlgo_t::Algo0,
            Self::Algo1 => cudnnConvolutionBwdFilterAlgo_t::Algo1,
            Self::Fft => cudnnConvolutionBwdFilterAlgo_t::Fft,
            Self::Algo3 => cudnnConvolutionBwdFilterAlgo_t::Algo3,
            Self::Winograd => cudnnConvolutionBwdFilterAlgo_t::Winograd,
            Self::WinogradNonfused => cudnnConvolutionBwdFilterAlgo_t::WinogradNonfused,
            Self::FftTiling => cudnnConvolutionBwdFilterAlgo_t::FftTiling,
        }
    }
}

pub fn convolution_backward_data_workspace_size(
    handle: &Handle,
    w: &FilterDescriptor,
    dy: &TensorDescriptor,
    conv: &ConvolutionDescriptor,
    dx: &TensorDescriptor,
    algo: BwdDataAlgo,
) -> Result<usize> {
    let cu = cudnn()?;
    let q = cu.cudnn_get_convolution_backward_data_workspace_size()?;
    let mut size = 0usize;
    check(unsafe {
        q(
            handle.handle,
            w.desc,
            dy.desc,
            conv.desc,
            dx.desc,
            algo.raw(),
            &mut size,
        )
    })?;
    Ok(size)
}

pub fn convolution_backward_filter_workspace_size(
    handle: &Handle,
    x: &TensorDescriptor,
    dy: &TensorDescriptor,
    conv: &ConvolutionDescriptor,
    dw: &FilterDescriptor,
    algo: BwdFilterAlgo,
) -> Result<usize> {
    let cu = cudnn()?;
    let q = cu.cudnn_get_convolution_backward_filter_workspace_size()?;
    let mut size = 0usize;
    check(unsafe {
        q(
            handle.handle,
            x.desc,
            dy.desc,
            conv.desc,
            dw.desc,
            algo.raw(),
            &mut size,
        )
    })?;
    Ok(size)
}

/// `dX = alpha * conv_bwd_data(W, dY) + beta * dX`.
#[allow(clippy::too_many_arguments)]
pub fn convolution_backward_data<T: DeviceRepr>(
    handle: &Handle,
    alpha: f32,
    w_desc: &FilterDescriptor,
    w: &DeviceBuffer<T>,
    dy_desc: &TensorDescriptor,
    dy: &DeviceBuffer<T>,
    conv: &ConvolutionDescriptor,
    algo: BwdDataAlgo,
    workspace: &mut DeviceBuffer<u8>,
    beta: f32,
    dx_desc: &TensorDescriptor,
    dx: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_convolution_backward_data()?;
    check(unsafe {
        cu(
            handle.handle,
            &alpha as *const f32 as *const core::ffi::c_void,
            w_desc.desc,
            w.as_raw().0 as *const core::ffi::c_void,
            dy_desc.desc,
            dy.as_raw().0 as *const core::ffi::c_void,
            conv.desc,
            algo.raw(),
            workspace.as_raw().0 as *mut core::ffi::c_void,
            workspace.byte_size(),
            &beta as *const f32 as *const core::ffi::c_void,
            dx_desc.desc,
            dx.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

/// `dW = alpha * conv_bwd_filter(X, dY) + beta * dW`.
#[allow(clippy::too_many_arguments)]
pub fn convolution_backward_filter<T: DeviceRepr>(
    handle: &Handle,
    alpha: f32,
    x_desc: &TensorDescriptor,
    x: &DeviceBuffer<T>,
    dy_desc: &TensorDescriptor,
    dy: &DeviceBuffer<T>,
    conv: &ConvolutionDescriptor,
    algo: BwdFilterAlgo,
    workspace: &mut DeviceBuffer<u8>,
    beta: f32,
    dw_desc: &FilterDescriptor,
    dw: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_convolution_backward_filter()?;
    check(unsafe {
        cu(
            handle.handle,
            &alpha as *const f32 as *const core::ffi::c_void,
            x_desc.desc,
            x.as_raw().0 as *const core::ffi::c_void,
            dy_desc.desc,
            dy.as_raw().0 as *const core::ffi::c_void,
            conv.desc,
            algo.raw(),
            workspace.as_raw().0 as *mut core::ffi::c_void,
            workspace.byte_size(),
            &beta as *const f32 as *const core::ffi::c_void,
            dw_desc.desc,
            dw.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

/// Add the bias gradient: sum over spatial dims of `dY`.
pub fn convolution_backward_bias<T: DeviceRepr>(
    handle: &Handle,
    alpha: f32,
    dy_desc: &TensorDescriptor,
    dy: &DeviceBuffer<T>,
    beta: f32,
    db_desc: &TensorDescriptor,
    db: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_convolution_backward_bias()?;
    check(unsafe {
        cu(
            handle.handle,
            &alpha as *const f32 as *const core::ffi::c_void,
            dy_desc.desc,
            dy.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            db_desc.desc,
            db.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

// ---- pooling --------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum PoolingMode {
    #[default]
    Max,
    AverageCountIncludePadding,
    AverageCountExcludePadding,
    MaxDeterministic,
}

impl PoolingMode {
    fn raw(self) -> cudnnPoolingMode_t {
        match self {
            Self::Max => cudnnPoolingMode_t::Max,
            Self::AverageCountIncludePadding => cudnnPoolingMode_t::AverageCountIncludePadding,
            Self::AverageCountExcludePadding => cudnnPoolingMode_t::AverageCountExcludePadding,
            Self::MaxDeterministic => cudnnPoolingMode_t::MaxDeterministic,
        }
    }
}

pub struct PoolingDescriptor {
    desc: cudnnPoolingDescriptor_t,
}

unsafe impl Send for PoolingDescriptor {}

impl core::fmt::Debug for PoolingDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PoolingDescriptor")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl PoolingDescriptor {
    #[allow(clippy::too_many_arguments)]
    pub fn new_2d(
        mode: PoolingMode,
        window_h: i32,
        window_w: i32,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
    ) -> Result<Self> {
        let cu = cudnn()?;
        let create = cu.cudnn_create_pooling_descriptor()?;
        let set = cu.cudnn_set_pooling_2d_descriptor()?;
        let mut desc: cudnnPoolingDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe {
            set(
                this.desc,
                mode.raw(),
                cudnnNanPropagation_t::PropagateNan,
                window_h,
                window_w,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
            )
        })?;
        Ok(this)
    }

    #[inline]
    pub fn as_raw(&self) -> cudnnPoolingDescriptor_t {
        self.desc
    }
}

impl Drop for PoolingDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_pooling_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn pooling_forward<T: DeviceRepr>(
    handle: &Handle,
    pool: &PoolingDescriptor,
    alpha: f32,
    x_desc: &TensorDescriptor,
    x: &DeviceBuffer<T>,
    beta: f32,
    y_desc: &TensorDescriptor,
    y: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_pooling_forward()?;
    check(unsafe {
        cu(
            handle.handle,
            pool.desc,
            &alpha as *const f32 as *const core::ffi::c_void,
            x_desc.desc,
            x.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            y_desc.desc,
            y.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn pooling_backward<T: DeviceRepr>(
    handle: &Handle,
    pool: &PoolingDescriptor,
    alpha: f32,
    y_desc: &TensorDescriptor,
    y: &DeviceBuffer<T>,
    dy_desc: &TensorDescriptor,
    dy: &DeviceBuffer<T>,
    x_desc: &TensorDescriptor,
    x: &DeviceBuffer<T>,
    beta: f32,
    dx_desc: &TensorDescriptor,
    dx: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_pooling_backward()?;
    check(unsafe {
        cu(
            handle.handle,
            pool.desc,
            &alpha as *const f32 as *const core::ffi::c_void,
            y_desc.desc,
            y.as_raw().0 as *const core::ffi::c_void,
            dy_desc.desc,
            dy.as_raw().0 as *const core::ffi::c_void,
            x_desc.desc,
            x.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            dx_desc.desc,
            dx.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

// ---- softmax --------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum SoftmaxAlgo {
    Fast,
    #[default]
    Accurate,
    Log,
}

impl SoftmaxAlgo {
    fn raw(self) -> cudnnSoftmaxAlgorithm_t {
        match self {
            Self::Fast => cudnnSoftmaxAlgorithm_t::Fast,
            Self::Accurate => cudnnSoftmaxAlgorithm_t::Accurate,
            Self::Log => cudnnSoftmaxAlgorithm_t::Log,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum SoftmaxMode {
    Instance,
    #[default]
    Channel,
}

impl SoftmaxMode {
    fn raw(self) -> cudnnSoftmaxMode_t {
        match self {
            Self::Instance => cudnnSoftmaxMode_t::Instance,
            Self::Channel => cudnnSoftmaxMode_t::Channel,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn softmax_forward<T: DeviceRepr>(
    handle: &Handle,
    algo: SoftmaxAlgo,
    mode: SoftmaxMode,
    alpha: f32,
    x_desc: &TensorDescriptor,
    x: &DeviceBuffer<T>,
    beta: f32,
    y_desc: &TensorDescriptor,
    y: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_softmax_forward()?;
    check(unsafe {
        cu(
            handle.handle,
            algo.raw(),
            mode.raw(),
            &alpha as *const f32 as *const core::ffi::c_void,
            x_desc.desc,
            x.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            y_desc.desc,
            y.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn softmax_backward<T: DeviceRepr>(
    handle: &Handle,
    algo: SoftmaxAlgo,
    mode: SoftmaxMode,
    alpha: f32,
    y_desc: &TensorDescriptor,
    y: &DeviceBuffer<T>,
    dy_desc: &TensorDescriptor,
    dy: &DeviceBuffer<T>,
    beta: f32,
    dx_desc: &TensorDescriptor,
    dx: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_softmax_backward()?;
    check(unsafe {
        cu(
            handle.handle,
            algo.raw(),
            mode.raw(),
            &alpha as *const f32 as *const core::ffi::c_void,
            y_desc.desc,
            y.as_raw().0 as *const core::ffi::c_void,
            dy_desc.desc,
            dy.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            dx_desc.desc,
            dx.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

// ---- batch normalization --------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum BatchNormMode {
    PerActivation,
    #[default]
    Spatial,
    SpatialPersistent,
}

impl BatchNormMode {
    fn raw(self) -> cudnnBatchNormMode_t {
        match self {
            Self::PerActivation => cudnnBatchNormMode_t::PerActivation,
            Self::Spatial => cudnnBatchNormMode_t::Spatial,
            Self::SpatialPersistent => cudnnBatchNormMode_t::SpatialPersistent,
        }
    }
}

/// Training-time BN forward: updates running statistics and returns saved
/// `mean` / `inv_variance` for use by [`batch_normalization_backward`].
#[allow(clippy::too_many_arguments)]
pub fn batch_normalization_forward_training<T: DeviceRepr>(
    handle: &Handle,
    mode: BatchNormMode,
    alpha: f32,
    beta: f32,
    x_desc: &TensorDescriptor,
    x: &DeviceBuffer<T>,
    y_desc: &TensorDescriptor,
    y: &mut DeviceBuffer<T>,
    bn_smbv_desc: &TensorDescriptor,
    bn_scale: &DeviceBuffer<T>,
    bn_bias: &DeviceBuffer<T>,
    exponential_avg_factor: f64,
    running_mean: &mut DeviceBuffer<T>,
    running_variance: &mut DeviceBuffer<T>,
    epsilon: f64,
    saved_mean: &mut DeviceBuffer<T>,
    saved_inv_variance: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_batch_normalization_forward_training()?;
    check(unsafe {
        cu(
            handle.handle,
            mode.raw(),
            &alpha as *const f32 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            x_desc.desc,
            x.as_raw().0 as *const core::ffi::c_void,
            y_desc.desc,
            y.as_raw().0 as *mut core::ffi::c_void,
            bn_smbv_desc.desc,
            bn_scale.as_raw().0 as *const core::ffi::c_void,
            bn_bias.as_raw().0 as *const core::ffi::c_void,
            exponential_avg_factor,
            running_mean.as_raw().0 as *mut core::ffi::c_void,
            running_variance.as_raw().0 as *mut core::ffi::c_void,
            epsilon,
            saved_mean.as_raw().0 as *mut core::ffi::c_void,
            saved_inv_variance.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

/// BN backward — matched with [`batch_normalization_forward_training`].
#[allow(clippy::too_many_arguments)]
pub fn batch_normalization_backward<T: DeviceRepr>(
    handle: &Handle,
    mode: BatchNormMode,
    alpha_data_diff: f32,
    beta_data_diff: f32,
    alpha_param_diff: f32,
    beta_param_diff: f32,
    x_desc: &TensorDescriptor,
    x: &DeviceBuffer<T>,
    dy_desc: &TensorDescriptor,
    dy: &DeviceBuffer<T>,
    dx_desc: &TensorDescriptor,
    dx: &mut DeviceBuffer<T>,
    bn_scale_bias_diff_desc: &TensorDescriptor,
    bn_scale: &DeviceBuffer<T>,
    d_bn_scale: &mut DeviceBuffer<T>,
    d_bn_bias: &mut DeviceBuffer<T>,
    epsilon: f64,
    saved_mean: &DeviceBuffer<T>,
    saved_inv_variance: &DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_batch_normalization_backward()?;
    check(unsafe {
        cu(
            handle.handle,
            mode.raw(),
            &alpha_data_diff as *const f32 as *const core::ffi::c_void,
            &beta_data_diff as *const f32 as *const core::ffi::c_void,
            &alpha_param_diff as *const f32 as *const core::ffi::c_void,
            &beta_param_diff as *const f32 as *const core::ffi::c_void,
            x_desc.desc,
            x.as_raw().0 as *const core::ffi::c_void,
            dy_desc.desc,
            dy.as_raw().0 as *const core::ffi::c_void,
            dx_desc.desc,
            dx.as_raw().0 as *mut core::ffi::c_void,
            bn_scale_bias_diff_desc.desc,
            bn_scale.as_raw().0 as *const core::ffi::c_void,
            d_bn_scale.as_raw().0 as *mut core::ffi::c_void,
            d_bn_bias.as_raw().0 as *mut core::ffi::c_void,
            epsilon,
            saved_mean.as_raw().0 as *const core::ffi::c_void,
            saved_inv_variance.as_raw().0 as *const core::ffi::c_void,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn batch_normalization_forward_inference<T: DeviceRepr>(
    handle: &Handle,
    mode: BatchNormMode,
    alpha: f32,
    beta: f32,
    x_desc: &TensorDescriptor,
    x: &DeviceBuffer<T>,
    y_desc: &TensorDescriptor,
    y: &mut DeviceBuffer<T>,
    bn_smbv_desc: &TensorDescriptor,
    bn_scale: &DeviceBuffer<T>,
    bn_bias: &DeviceBuffer<T>,
    estimated_mean: &DeviceBuffer<T>,
    estimated_var: &DeviceBuffer<T>,
    epsilon: f64,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_batch_normalization_forward_inference()?;
    check(unsafe {
        cu(
            handle.handle,
            mode.raw(),
            &alpha as *const f32 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            x_desc.desc,
            x.as_raw().0 as *const core::ffi::c_void,
            y_desc.desc,
            y.as_raw().0 as *mut core::ffi::c_void,
            bn_smbv_desc.desc,
            bn_scale.as_raw().0 as *const core::ffi::c_void,
            bn_bias.as_raw().0 as *const core::ffi::c_void,
            estimated_mean.as_raw().0 as *const core::ffi::c_void,
            estimated_var.as_raw().0 as *const core::ffi::c_void,
            epsilon,
        )
    })
}

// ---- dropout --------------------------------------------------------------

pub struct DropoutDescriptor {
    desc: cudnnDropoutDescriptor_t,
}

unsafe impl Send for DropoutDescriptor {}

impl core::fmt::Debug for DropoutDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DropoutDescriptor")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl DropoutDescriptor {
    /// Create a dropout descriptor with probability `dropout` ∈ \[0, 1\].
    ///
    /// `states` is a driver-owned buffer of at least
    /// [`dropout_states_size`] bytes, shared across many descriptors.
    pub fn new(
        handle: &Handle,
        dropout: f32,
        states: &mut DeviceBuffer<u8>,
        seed: u64,
    ) -> Result<Self> {
        let cu = cudnn()?;
        let create = cu.cudnn_create_dropout_descriptor()?;
        let set = cu.cudnn_set_dropout_descriptor()?;
        let mut desc: cudnnDropoutDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe {
            set(
                this.desc,
                handle.handle,
                dropout,
                states.as_raw().0 as *mut core::ffi::c_void,
                states.byte_size(),
                seed,
            )
        })?;
        Ok(this)
    }

    #[inline]
    pub fn as_raw(&self) -> cudnnDropoutDescriptor_t {
        self.desc
    }
}

impl Drop for DropoutDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_dropout_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Size in bytes of the state buffer required for a dropout RNG.
pub fn dropout_states_size(handle: &Handle) -> Result<usize> {
    let c = cudnn()?;
    let cu = c.cudnn_dropout_get_states_size()?;
    let mut size = 0usize;
    check(unsafe { cu(handle.handle, &mut size) })?;
    Ok(size)
}

/// Size in bytes of the reserve buffer required for dropout on `x`.
pub fn dropout_reserve_size(x: &TensorDescriptor) -> Result<usize> {
    let c = cudnn()?;
    let cu = c.cudnn_dropout_get_reserve_space_size()?;
    let mut size = 0usize;
    check(unsafe { cu(x.desc, &mut size) })?;
    Ok(size)
}

#[allow(clippy::too_many_arguments)]
pub fn dropout_forward<T: DeviceRepr>(
    handle: &Handle,
    dropout: &DropoutDescriptor,
    x_desc: &TensorDescriptor,
    x: &DeviceBuffer<T>,
    y_desc: &TensorDescriptor,
    y: &mut DeviceBuffer<T>,
    reserve: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_dropout_forward()?;
    check(unsafe {
        cu(
            handle.handle,
            dropout.desc,
            x_desc.desc,
            x.as_raw().0 as *const core::ffi::c_void,
            y_desc.desc,
            y.as_raw().0 as *mut core::ffi::c_void,
            reserve.as_raw().0 as *mut core::ffi::c_void,
            reserve.byte_size(),
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub fn dropout_backward<T: DeviceRepr>(
    handle: &Handle,
    dropout: &DropoutDescriptor,
    dy_desc: &TensorDescriptor,
    dy: &DeviceBuffer<T>,
    dx_desc: &TensorDescriptor,
    dx: &mut DeviceBuffer<T>,
    reserve: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_dropout_backward()?;
    check(unsafe {
        cu(
            handle.handle,
            dropout.desc,
            dy_desc.desc,
            dy.as_raw().0 as *const core::ffi::c_void,
            dx_desc.desc,
            dx.as_raw().0 as *mut core::ffi::c_void,
            reserve.as_raw().0 as *mut core::ffi::c_void,
            reserve.byte_size(),
        )
    })
}

// ---- LRN ------------------------------------------------------------------

pub struct LrnDescriptor {
    desc: cudnnLRNDescriptor_t,
}

unsafe impl Send for LrnDescriptor {}

impl core::fmt::Debug for LrnDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LrnDescriptor")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl LrnDescriptor {
    pub fn new(n: i32, alpha: f64, beta: f64, k: f64) -> Result<Self> {
        let cu = cudnn()?;
        let create = cu.cudnn_create_lrn_descriptor()?;
        let set = cu.cudnn_set_lrn_descriptor()?;
        let mut desc: cudnnLRNDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe { set(this.desc, n, alpha, beta, k) })?;
        Ok(this)
    }

    #[inline]
    pub fn as_raw(&self) -> cudnnLRNDescriptor_t {
        self.desc
    }
}

impl Drop for LrnDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_lrn_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

// ---- op-tensor / reduce / transform --------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum OpTensorOp {
    Add,
    Mul,
    Min,
    Max,
    Sqrt,
    Not,
}

impl OpTensorOp {
    fn raw(self) -> cudnnOpTensorOp_t {
        match self {
            Self::Add => cudnnOpTensorOp_t::Add,
            Self::Mul => cudnnOpTensorOp_t::Mul,
            Self::Min => cudnnOpTensorOp_t::Min,
            Self::Max => cudnnOpTensorOp_t::Max,
            Self::Sqrt => cudnnOpTensorOp_t::Sqrt,
            Self::Not => cudnnOpTensorOp_t::Not,
        }
    }
}

pub struct OpTensorDescriptor {
    desc: cudnnOpTensorDescriptor_t,
}

unsafe impl Send for OpTensorDescriptor {}

impl core::fmt::Debug for OpTensorDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OpTensorDescriptor")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl OpTensorDescriptor {
    pub fn new(op: OpTensorOp, compute: DType) -> Result<Self> {
        let cu = cudnn()?;
        let create = cu.cudnn_create_op_tensor_descriptor()?;
        let set = cu.cudnn_set_op_tensor_descriptor()?;
        let mut desc: cudnnOpTensorDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe {
            set(
                this.desc,
                op.raw(),
                compute.raw(),
                cudnnNanPropagation_t::PropagateNan,
            )
        })?;
        Ok(this)
    }

    #[inline]
    pub fn as_raw(&self) -> cudnnOpTensorDescriptor_t {
        self.desc
    }
}

impl Drop for OpTensorDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_op_tensor_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// `C = alpha1 * op(A) + alpha2 * op(B) + beta * C` element-wise.
#[allow(clippy::too_many_arguments)]
pub fn op_tensor<T: DeviceRepr>(
    handle: &Handle,
    op: &OpTensorDescriptor,
    alpha1: f32,
    a_desc: &TensorDescriptor,
    a: &DeviceBuffer<T>,
    alpha2: f32,
    b_desc: &TensorDescriptor,
    b: &DeviceBuffer<T>,
    beta: f32,
    c_desc: &TensorDescriptor,
    c: &mut DeviceBuffer<T>,
) -> Result<()> {
    let cu_crate = cudnn()?;
    let cu = cu_crate.cudnn_op_tensor()?;
    check(unsafe {
        cu(
            handle.handle,
            op.desc,
            &alpha1 as *const f32 as *const core::ffi::c_void,
            a_desc.desc,
            a.as_raw().0 as *const core::ffi::c_void,
            &alpha2 as *const f32 as *const core::ffi::c_void,
            b_desc.desc,
            b.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            c_desc.desc,
            c.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReduceOp {
    Add,
    Mul,
    Min,
    Max,
    AbsMax,
    Avg,
    Norm1,
    Norm2,
    MulNoZeros,
}

impl ReduceOp {
    fn raw(self) -> cudnnReduceTensorOp_t {
        match self {
            Self::Add => cudnnReduceTensorOp_t::Add,
            Self::Mul => cudnnReduceTensorOp_t::Mul,
            Self::Min => cudnnReduceTensorOp_t::Min,
            Self::Max => cudnnReduceTensorOp_t::Max,
            Self::AbsMax => cudnnReduceTensorOp_t::Amax,
            Self::Avg => cudnnReduceTensorOp_t::Avg,
            Self::Norm1 => cudnnReduceTensorOp_t::Norm1,
            Self::Norm2 => cudnnReduceTensorOp_t::Norm2,
            Self::MulNoZeros => cudnnReduceTensorOp_t::MulNoZeros,
        }
    }
}

pub struct ReduceTensorDescriptor {
    desc: cudnnReduceTensorDescriptor_t,
}

unsafe impl Send for ReduceTensorDescriptor {}

impl core::fmt::Debug for ReduceTensorDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ReduceTensorDescriptor")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl ReduceTensorDescriptor {
    pub fn new(op: ReduceOp, compute: DType) -> Result<Self> {
        let cu = cudnn()?;
        let create = cu.cudnn_create_reduce_tensor_descriptor()?;
        let set = cu.cudnn_set_reduce_tensor_descriptor()?;
        let mut desc: cudnnReduceTensorDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe {
            set(
                this.desc,
                op.raw(),
                compute.raw(),
                cudnnNanPropagation_t::PropagateNan,
                cudnnReduceTensorIndices_t::NoIndices,
                cudnnIndicesType_t::U32,
            )
        })?;
        Ok(this)
    }

    pub fn workspace_size(
        &self,
        handle: &Handle,
        a: &TensorDescriptor,
        c: &TensorDescriptor,
    ) -> Result<usize> {
        let cu = cudnn()?;
        let q = cu.cudnn_get_reduction_workspace_size()?;
        let mut size = 0usize;
        check(unsafe { q(handle.handle, self.desc, a.desc, c.desc, &mut size) })?;
        Ok(size)
    }

    #[inline]
    pub fn as_raw(&self) -> cudnnReduceTensorDescriptor_t {
        self.desc
    }
}

impl Drop for ReduceTensorDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_reduce_tensor_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// `C = alpha * reduce(A) + beta * C` over the axes where `A`'s extent is
/// preserved and `C`'s is 1.
#[allow(clippy::too_many_arguments)]
pub fn reduce_tensor<T: DeviceRepr>(
    handle: &Handle,
    reducer: &ReduceTensorDescriptor,
    workspace: &mut DeviceBuffer<u8>,
    alpha: f32,
    a_desc: &TensorDescriptor,
    a: &DeviceBuffer<T>,
    beta: f32,
    c_desc: &TensorDescriptor,
    c: &mut DeviceBuffer<T>,
) -> Result<()> {
    let cu_crate = cudnn()?;
    let cu = cu_crate.cudnn_reduce_tensor()?;
    check(unsafe {
        cu(
            handle.handle,
            reducer.desc,
            core::ptr::null_mut(),
            0,
            workspace.as_raw().0 as *mut core::ffi::c_void,
            workspace.byte_size(),
            &alpha as *const f32 as *const core::ffi::c_void,
            a_desc.desc,
            a.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            c_desc.desc,
            c.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

/// `C = alpha * A + beta * C` with broadcast. Useful for adding a per-channel
/// bias to a feature map.
pub fn add_tensor<T: DeviceRepr>(
    handle: &Handle,
    alpha: f32,
    a_desc: &TensorDescriptor,
    a: &DeviceBuffer<T>,
    beta: f32,
    c_desc: &TensorDescriptor,
    c: &mut DeviceBuffer<T>,
) -> Result<()> {
    let cu_crate = cudnn()?;
    let cu = cu_crate.cudnn_add_tensor()?;
    check(unsafe {
        cu(
            handle.handle,
            &alpha as *const f32 as *const core::ffi::c_void,
            a_desc.desc,
            a.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            c_desc.desc,
            c.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

// ---- backend (Graph) API --------------------------------------------------

/// Thin wrapper over a `cudnnBackendDescriptor_t`. Used to build Graph-API
/// operation graphs and execution plans. Callers set attributes with
/// [`BackendDescriptor::set_attribute_raw`] using the constants in
/// [`baracuda_cudnn_sys::cudnnBackendAttributeName_t`] /
/// [`baracuda_cudnn_sys::cudnnBackendAttributeType_t`].
pub struct BackendDescriptor {
    desc: cudnnBackendDescriptor_t,
    finalized: bool,
}

unsafe impl Send for BackendDescriptor {}

impl core::fmt::Debug for BackendDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BackendDescriptor")
            .field("desc", &self.desc)
            .field("finalized", &self.finalized)
            .finish()
    }
}

impl BackendDescriptor {
    pub fn new(kind: cudnnBackendDescriptorType_t) -> Result<Self> {
        let cu = cudnn()?;
        let create = cu.cudnn_backend_create_descriptor()?;
        let init = cu.cudnn_backend_initialize()?;
        let mut desc: cudnnBackendDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(kind, &mut desc) })?;
        let this = Self {
            desc,
            finalized: false,
        };
        check(unsafe { init(this.desc) })?;
        Ok(this)
    }

    /// Set an attribute by name/type. `element_count` is the number of
    /// elements in `array_of_elements` (not byte count).
    ///
    /// # Safety
    /// `array_of_elements` must point to valid data matching the attribute's
    /// expected type and count.
    pub unsafe fn set_attribute_raw(
        &self,
        name: cudnnBackendAttributeName_t,
        ty: cudnnBackendAttributeType_t,
        element_count: i64,
        array_of_elements: *const core::ffi::c_void,
    ) -> Result<()> {
        let cu = cudnn()?;
        let f = cu.cudnn_backend_set_attribute()?;
        check(f(self.desc, name, ty, element_count, array_of_elements))
    }

    pub fn finalize(&mut self) -> Result<()> {
        if self.finalized {
            return Ok(());
        }
        let cu = cudnn()?;
        let f = cu.cudnn_backend_finalize()?;
        check(unsafe { f(self.desc) })?;
        self.finalized = true;
        Ok(())
    }

    /// Execute an execution-plan descriptor. `self` should be the plan
    /// descriptor; `variant_pack` provides tensor addresses + workspace.
    pub fn execute(&self, handle: &Handle, variant_pack: &BackendDescriptor) -> Result<()> {
        let cu = cudnn()?;
        let f = cu.cudnn_backend_execute()?;
        check(unsafe { f(handle.handle, self.desc, variant_pack.desc) })
    }

    #[inline]
    pub fn as_raw(&self) -> cudnnBackendDescriptor_t {
        self.desc
    }
}

impl Drop for BackendDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_backend_destroy_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Re-export the backend attribute enums so callers don't have to reach
/// into the sys crate.
pub use baracuda_cudnn_sys::{
    cudnnBackendAttributeName_t as BackendAttrName,
    cudnnBackendAttributeType_t as BackendAttrType,
    cudnnBackendDescriptorType_t as BackendDescType,
};

// ---- CTC loss ------------------------------------------------------------

use baracuda_cudnn_sys::cudnnCTCLossDescriptor_t;

pub struct CtcLossDescriptor {
    desc: cudnnCTCLossDescriptor_t,
}

unsafe impl Send for CtcLossDescriptor {}

impl core::fmt::Debug for CtcLossDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CtcLossDescriptor")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl CtcLossDescriptor {
    pub fn new(compute: DType) -> Result<Self> {
        let cu = cudnn()?;
        let create = cu.cudnn_create_ctc_loss_descriptor()?;
        let set = cu.cudnn_set_ctc_loss_descriptor()?;
        let mut desc: cudnnCTCLossDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe { set(this.desc, compute.raw()) })?;
        Ok(this)
    }

    #[inline]
    pub fn as_raw(&self) -> cudnnCTCLossDescriptor_t {
        self.desc
    }
}

impl Drop for CtcLossDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_ctc_loss_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Bytes of scratch workspace needed for [`ctc_loss`].
#[allow(clippy::too_many_arguments)]
pub fn ctc_loss_workspace_size(
    handle: &Handle,
    probs: &TensorDescriptor,
    gradients: &TensorDescriptor,
    labels: &[i32],
    label_lengths: &[i32],
    input_lengths: &[i32],
    algo: i32,
    desc: &CtcLossDescriptor,
) -> Result<usize> {
    let cu = cudnn()?;
    let q = cu.cudnn_get_ctc_loss_workspace_size()?;
    let mut size = 0usize;
    check(unsafe {
        q(
            handle.handle,
            probs.desc,
            gradients.desc,
            labels.as_ptr(),
            label_lengths.as_ptr(),
            input_lengths.as_ptr(),
            algo,
            desc.desc,
            &mut size,
        )
    })?;
    Ok(size)
}

/// CTC (Connectionist Temporal Classification) loss.
#[allow(clippy::too_many_arguments)]
pub fn ctc_loss<T: DeviceRepr>(
    handle: &Handle,
    probs_desc: &TensorDescriptor,
    probs: &DeviceBuffer<T>,
    labels: &[i32],
    label_lengths: &[i32],
    input_lengths: &[i32],
    costs: &mut DeviceBuffer<T>,
    gradients_desc: &TensorDescriptor,
    gradients: &mut DeviceBuffer<T>,
    algo: i32,
    desc: &CtcLossDescriptor,
    workspace: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_ctc_loss()?;
    check(unsafe {
        cu(
            handle.handle,
            probs_desc.desc,
            probs.as_raw().0 as *const core::ffi::c_void,
            labels.as_ptr(),
            label_lengths.as_ptr(),
            input_lengths.as_ptr(),
            costs.as_raw().0 as *mut core::ffi::c_void,
            gradients_desc.desc,
            gradients.as_raw().0 as *mut core::ffi::c_void,
            algo,
            desc.desc,
            workspace.as_raw().0 as *mut core::ffi::c_void,
            workspace.byte_size(),
        )
    })
}

// ---- Spatial transformer ------------------------------------------------

use baracuda_cudnn_sys::cudnnSpatialTransformerDescriptor_t;

pub struct SpatialTransformerDescriptor {
    desc: cudnnSpatialTransformerDescriptor_t,
}

unsafe impl Send for SpatialTransformerDescriptor {}

impl core::fmt::Debug for SpatialTransformerDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SpatialTransformerDescriptor")
            .field("desc", &self.desc)
            .finish_non_exhaustive()
    }
}

impl SpatialTransformerDescriptor {
    /// `sampler_type` matches `CUDNN_SAMPLER_BILINEAR = 0`.
    pub fn new(sampler_type: i32, dtype: DType, dims: &[i32]) -> Result<Self> {
        let cu = cudnn()?;
        let create = cu.cudnn_create_spatial_transformer_descriptor()?;
        let set = cu.cudnn_set_spatial_transformer_nd_descriptor()?;
        let mut desc: cudnnSpatialTransformerDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe {
            set(
                this.desc,
                sampler_type,
                dtype.raw(),
                dims.len() as core::ffi::c_int,
                dims.as_ptr(),
            )
        })?;
        Ok(this)
    }

    #[inline]
    pub fn as_raw(&self) -> cudnnSpatialTransformerDescriptor_t {
        self.desc
    }
}

impl Drop for SpatialTransformerDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_spatial_transformer_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Compute the sampling grid from the affine transform `theta`.
pub fn spatial_tf_grid_generator<T: DeviceRepr>(
    handle: &Handle,
    st: &SpatialTransformerDescriptor,
    theta: &DeviceBuffer<T>,
    grid: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_spatial_tf_grid_generator_forward()?;
    check(unsafe {
        cu(
            handle.handle,
            st.desc,
            theta.as_raw().0 as *const core::ffi::c_void,
            grid.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

/// Bilinearly sample `x` at `grid` points to produce `y`.
#[allow(clippy::too_many_arguments)]
pub fn spatial_tf_sampler<T: DeviceRepr>(
    handle: &Handle,
    st: &SpatialTransformerDescriptor,
    alpha: f32,
    x_desc: &TensorDescriptor,
    x: &DeviceBuffer<T>,
    grid: &DeviceBuffer<T>,
    beta: f32,
    y_desc: &TensorDescriptor,
    y: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_spatial_tf_sampler_forward()?;
    check(unsafe {
        cu(
            handle.handle,
            st.desc,
            &alpha as *const f32 as *const core::ffi::c_void,
            x_desc.desc,
            x.as_raw().0 as *const core::ffi::c_void,
            grid.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            y_desc.desc,
            y.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}
