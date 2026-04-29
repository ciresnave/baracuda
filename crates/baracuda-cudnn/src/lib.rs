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
    cudnn, cudnnActivationDescriptor_t, cudnnActivationMode_t, cudnnAttnDescriptor_t,
    cudnnBackendAttributeName_t, cudnnBackendAttributeType_t, cudnnBackendDescriptorType_t,
    cudnnBackendDescriptor_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t,
    cudnnConvolutionBwdDataAlgo_t, cudnnConvolutionBwdFilterAlgo_t,
    cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, cudnnConvolutionMode_t,
    cudnnDataType_t, cudnnDropoutDescriptor_t, cudnnFilterDescriptor_t,
    cudnnHandle_t, cudnnIndicesType_t, cudnnLRNDescriptor_t, cudnnMathType_t, cudnnNanPropagation_t,
    cudnnNormAlgo_t, cudnnNormMode_t, cudnnNormOps_t, cudnnOpTensorDescriptor_t, cudnnOpTensorOp_t,
    cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnReduceTensorDescriptor_t,
    cudnnReduceTensorIndices_t, cudnnReduceTensorOp_t, cudnnReorderType_t,
    cudnnSeqDataDescriptor_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, cudnnStatus_t,
    cudnnTensorDescriptor_t, cudnnTensorFormat_t,
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

// Direct impls on the `half` crate's types so callers can use
// `half::f16` / `half::bf16` end-to-end without bridging through
// `baracuda_types::Half` / `BFloat16`. Both directions of `From` are
// already available in baracuda-types under the same feature.
#[cfg(feature = "half-crate")]
impl CudnnDataType for half::f16 {
    const DTYPE: DType = DType::F16;
}
#[cfg(feature = "half-crate")]
impl CudnnDataType for half::bf16 {
    const DTYPE: DType = DType::BF16;
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

    /// Set the group count for grouped convolution. The default is 1
    /// (regular convolution); pass `g > 1` for depthwise / grouped
    /// variants. Filter shape must match: input C divides g, filter C
    /// = input C / g.
    pub fn set_group_count(&self, group_count: i32) -> Result<()> {
        let cu = cudnn()?;
        let f = cu.cudnn_set_convolution_group_count()?;
        check(unsafe { f(self.desc, group_count) })
    }

    /// Read back the convolution group count.
    pub fn group_count(&self) -> Result<i32> {
        let cu = cudnn()?;
        let f = cu.cudnn_get_convolution_group_count()?;
        let mut g: core::ffi::c_int = 0;
        check(unsafe { f(self.desc, &mut g) })?;
        Ok(g)
    }

    /// Pick the math type cuDNN uses for this convolution — controls
    /// whether tensor cores are eligible.
    pub fn set_math_type(&self, math: MathType) -> Result<()> {
        let cu = cudnn()?;
        let f = cu.cudnn_set_convolution_math_type()?;
        check(unsafe { f(self.desc, math.raw()) })
    }

    /// Read back the convolution math type.
    pub fn math_type(&self) -> Result<MathType> {
        let cu = cudnn()?;
        let f = cu.cudnn_get_convolution_math_type()?;
        let mut m = cudnnMathType_t::DefaultMath;
        check(unsafe { f(self.desc, &mut m) })?;
        Ok(MathType::from_raw(m))
    }

    /// Set the filter / bias reorder type for INT8 quantized inference.
    pub fn set_reorder_type(&self, reorder: ReorderType) -> Result<()> {
        let cu = cudnn()?;
        let f = cu.cudnn_set_convolution_reorder_type()?;
        check(unsafe { f(self.desc, reorder.raw()) })
    }

    /// Read back the reorder type.
    pub fn reorder_type(&self) -> Result<ReorderType> {
        let cu = cudnn()?;
        let f = cu.cudnn_get_convolution_reorder_type()?;
        let mut r = cudnnReorderType_t::DefaultReorder;
        check(unsafe { f(self.desc, &mut r) })?;
        Ok(ReorderType::from_raw(r))
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
///
/// # Example
///
/// End-to-end "build descriptors → query workspace → run forward".
/// The example uses `f32` and a 3×3 convolution with padding 1,
/// stride 1, dilation 1, no groups.
///
/// ```no_run
/// use baracuda_cudnn::{
///     convolution_forward, convolution_forward_workspace_size,
///     ConvMode, ConvolutionDescriptor, DType, FilterDescriptor, FwdAlgo,
///     Handle, TensorDescriptor, TensorFormat,
/// };
/// use baracuda_driver::{Context, Device, DeviceBuffer};
///
/// # fn demo() -> Result<(), Box<dyn std::error::Error>> {
/// let ctx   = Context::new(&Device::get(0)?)?;
/// let cudnn = Handle::new()?;
///
/// // Shapes: NCHW 1×3×32×32 input, 16 output channels, 3×3 kernel, pad 1.
/// let (n, c, h, w)   = (1, 3, 32, 32);
/// let (k, kh, kw)    = (16, 3, 3);
/// let (pad_h, pad_w) = (1, 1);
/// let (str_h, str_w) = (1, 1);
/// let (dil_h, dil_w) = (1, 1);
/// let (out_h, out_w) = (h, w);   // same-size output for pad=1, k=3, str=1
///
/// // Note the argument order: TensorDescriptor::new_4d takes
/// // (format, dtype, n, c, h, w); FilterDescriptor::new_4d takes
/// // (format, dtype, k, c, kh, kw).
/// let x_desc = TensorDescriptor::new_4d(TensorFormat::Nchw, DType::F32, n, c, h, w)?;
/// let w_desc = FilterDescriptor::new_4d(TensorFormat::Nchw, DType::F32, k, c, kh, kw)?;
/// let y_desc = TensorDescriptor::new_4d(TensorFormat::Nchw, DType::F32, n, k, out_h, out_w)?;
/// let conv = ConvolutionDescriptor::new_2d(
///     pad_h, pad_w, str_h, str_w, dil_h, dil_w,
///     ConvMode::CrossCorrelation, DType::F32,
/// )?;
/// // For grouped conv, set the group count after creation:
/// // conv.set_group_count(groups)?;
///
/// // Pick an algorithm. ImplicitGemm is a safe default; for perf, use
/// // `find_convolution_forward_algorithm` to benchmark on your shapes.
/// let algo = FwdAlgo::ImplicitGemm;
///
/// // Workspace size depends on (descs, algo).
/// let ws_bytes = convolution_forward_workspace_size(
///     &cudnn, &x_desc, &w_desc, &conv, &y_desc, algo,
/// )?;
///
/// // Allocate input / weight / output / workspace on the device.
/// let x_buf:   DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (n*c*h*w) as usize)?;
/// let w_buf:   DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (k*c*kh*kw) as usize)?;
/// let mut y_buf: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, (n*k*out_h*out_w) as usize)?;
/// let mut ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes.max(1))?;
///
/// convolution_forward(
///     &cudnn,
///     1.0, &x_desc, &x_buf,
///          &w_desc, &w_buf,
///     &conv, algo,
///     &mut ws,
///     0.0, &y_desc, &mut y_buf,
/// )?;
/// # Ok(()) }
/// ```
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

// ============================================================================
// Tier 1 — math type / reorder type / fused conv+bias+act / activation back
// ============================================================================

/// Math-type selector for [`ConvolutionDescriptor::set_math_type`] —
/// controls tensor-core eligibility.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum MathType {
    /// Standard FMA-only math; tensor cores not used.
    #[default]
    Default,
    /// Allow tensor-core math (Volta+).
    TensorOp,
    /// Allow tensor-core math with implicit half-precision conversion.
    TensorOpAllowConversion,
    /// Strict FMA-only.
    FmaOnly,
}

impl MathType {
    pub(crate) fn raw(self) -> cudnnMathType_t {
        match self {
            MathType::Default => cudnnMathType_t::DefaultMath,
            MathType::TensorOp => cudnnMathType_t::TensorOpMath,
            MathType::TensorOpAllowConversion => cudnnMathType_t::TensorOpMathAllowConversion,
            MathType::FmaOnly => cudnnMathType_t::FmaMath,
        }
    }
    pub(crate) fn from_raw(raw: cudnnMathType_t) -> Self {
        match raw {
            cudnnMathType_t::DefaultMath => MathType::Default,
            cudnnMathType_t::TensorOpMath => MathType::TensorOp,
            cudnnMathType_t::TensorOpMathAllowConversion => MathType::TensorOpAllowConversion,
            cudnnMathType_t::FmaMath => MathType::FmaOnly,
        }
    }
}

/// Filter / bias reorder selector for INT8 quantized inference paths.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum ReorderType {
    #[default]
    Default,
    None,
}

impl ReorderType {
    pub(crate) fn raw(self) -> cudnnReorderType_t {
        match self {
            ReorderType::Default => cudnnReorderType_t::DefaultReorder,
            ReorderType::None => cudnnReorderType_t::NoReorder,
        }
    }
    pub(crate) fn from_raw(raw: cudnnReorderType_t) -> Self {
        match raw {
            cudnnReorderType_t::DefaultReorder => ReorderType::Default,
            cudnnReorderType_t::NoReorder => ReorderType::None,
        }
    }
}

/// Pre-process filter / bias buffers for INT8 inference.
///
/// # Safety
/// Output buffers must have at least the same byte size as the inputs.
/// `bias_data` / `reordered_bias` may be null when `reorder_bias` is false.
#[allow(clippy::too_many_arguments)]
pub unsafe fn reorder_filter_and_bias(
    handle: &Handle,
    filter_desc: &FilterDescriptor,
    reorder: ReorderType,
    filter_data: *const core::ffi::c_void,
    reordered_filter: *mut core::ffi::c_void,
    reorder_bias: bool,
    bias_data: *const core::ffi::c_void,
    reordered_bias: *mut core::ffi::c_void,
) -> Result<()> {
    let c = cudnn()?;
    let f = c.cudnn_reorder_filter_and_bias()?;
    check(f(
        handle.handle, filter_desc.desc, reorder.raw(),
        filter_data, reordered_filter,
        reorder_bias as core::ffi::c_int, bias_data, reordered_bias,
    ))
}

/// Fused convolution + bias + activation forward:
/// `Y = activation(alpha1 * conv(X, W) + alpha2 * Z + bias)`.
/// `Z` may alias `Y` for in-place residual add.
#[allow(clippy::too_many_arguments)]
pub fn convolution_bias_activation_forward<T: DeviceRepr>(
    handle: &Handle,
    alpha1: f32,
    x_desc: &TensorDescriptor, x: &DeviceBuffer<T>,
    w_desc: &FilterDescriptor, w: &DeviceBuffer<T>,
    conv: &ConvolutionDescriptor,
    algo: FwdAlgo,
    workspace: &mut DeviceBuffer<u8>,
    alpha2: f32,
    z_desc: &TensorDescriptor, z: &DeviceBuffer<T>,
    bias_desc: &TensorDescriptor, bias: &DeviceBuffer<T>,
    activation: &ActivationDescriptor,
    y_desc: &TensorDescriptor, y: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_convolution_bias_activation_forward()?;
    check(unsafe {
        cu(
            handle.handle,
            &alpha1 as *const f32 as *const core::ffi::c_void,
            x_desc.desc, x.as_raw().0 as *const core::ffi::c_void,
            w_desc.desc, w.as_raw().0 as *const core::ffi::c_void,
            conv.desc, algo.raw(),
            workspace.as_raw().0 as *mut core::ffi::c_void, workspace.byte_size(),
            &alpha2 as *const f32 as *const core::ffi::c_void,
            z_desc.desc, z.as_raw().0 as *const core::ffi::c_void,
            bias_desc.desc, bias.as_raw().0 as *const core::ffi::c_void,
            activation.desc,
            y_desc.desc, y.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

/// `dx = alpha * activation_backward(y, dy, x) + beta * dx`.
#[allow(clippy::too_many_arguments)]
pub fn activation_backward<T: DeviceRepr>(
    handle: &Handle,
    activation: &ActivationDescriptor,
    alpha: f32,
    y_desc: &TensorDescriptor, y: &DeviceBuffer<T>,
    dy_desc: &TensorDescriptor, dy: &DeviceBuffer<T>,
    x_desc: &TensorDescriptor, x: &DeviceBuffer<T>,
    beta: f32,
    dx_desc: &TensorDescriptor, dx: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_activation_backward()?;
    check(unsafe {
        cu(
            handle.handle, activation.desc,
            &alpha as *const f32 as *const core::ffi::c_void,
            y_desc.desc, y.as_raw().0 as *const core::ffi::c_void,
            dy_desc.desc, dy.as_raw().0 as *const core::ffi::c_void,
            x_desc.desc, x.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            dx_desc.desc, dx.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

/// Cross-channel LRN backward.
#[allow(clippy::too_many_arguments)]
pub fn lrn_cross_channel_backward<T: DeviceRepr>(
    handle: &Handle, lrn: &LrnDescriptor, mode: i32,
    alpha: f32,
    y_desc: &TensorDescriptor, y: &DeviceBuffer<T>,
    dy_desc: &TensorDescriptor, dy: &DeviceBuffer<T>,
    x_desc: &TensorDescriptor, x: &DeviceBuffer<T>,
    beta: f32,
    dx_desc: &TensorDescriptor, dx: &mut DeviceBuffer<T>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_lrn_cross_channel_backward()?;
    check(unsafe {
        cu(
            handle.handle, lrn.desc, mode,
            &alpha as *const f32 as *const core::ffi::c_void,
            y_desc.desc, y.as_raw().0 as *const core::ffi::c_void,
            dy_desc.desc, dy.as_raw().0 as *const core::ffi::c_void,
            x_desc.desc, x.as_raw().0 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            dx_desc.desc, dx.as_raw().0 as *mut core::ffi::c_void,
        )
    })
}

/// Bytes of indices buffer required for index-returning reductions.
pub fn reduction_indices_size(
    handle: &Handle,
    reducer: &ReduceTensorDescriptor,
    a: &TensorDescriptor,
    c: &TensorDescriptor,
) -> Result<usize> {
    let cu = cudnn()?;
    let q = cu.cudnn_get_reduction_indices_size()?;
    let mut size = 0usize;
    check(unsafe { q(handle.handle, reducer.desc, a.desc, c.desc, &mut size) })?;
    Ok(size)
}

impl ActivationDescriptor {
    /// Set the β parameter on a Swish activation.
    pub fn set_swish_beta(&self, beta: f64) -> Result<()> {
        let c = cudnn()?;
        let f = c.cudnn_set_activation_descriptor_swish_beta()?;
        check(unsafe { f(self.desc, beta) })
    }
    /// Read back the Swish β parameter.
    pub fn swish_beta(&self) -> Result<f64> {
        let c = cudnn()?;
        let f = c.cudnn_get_activation_descriptor_swish_beta()?;
        let mut b: f64 = 0.0;
        check(unsafe { f(self.desc, &mut b) })?;
        Ok(b)
    }
}

// ============================================================================
// Tier 2 — Algorithm finders / pickers
// ============================================================================

pub use baracuda_cudnn_sys::cudnnConvolutionFwdAlgoPerf_t as FwdAlgoPerf;
pub use baracuda_cudnn_sys::cudnnConvolutionBwdDataAlgoPerf_t as BwdDataAlgoPerf;
pub use baracuda_cudnn_sys::cudnnConvolutionBwdFilterAlgoPerf_t as BwdFilterAlgoPerf;

/// Heuristic-pick the top-N forward-convolution algorithms (cheap; doesn't run them).
pub fn get_convolution_forward_algorithm(
    handle: &Handle,
    src: &TensorDescriptor, filter: &FilterDescriptor,
    conv: &ConvolutionDescriptor, dst: &TensorDescriptor,
    requested: i32,
) -> Result<Vec<FwdAlgoPerf>> {
    let cu = cudnn()?;
    let f = cu.cudnn_get_convolution_forward_algorithm_v7()?;
    let mut returned: core::ffi::c_int = 0;
    let mut buf: Vec<FwdAlgoPerf> = Vec::with_capacity(requested as usize);
    let raw = unsafe {
        f(handle.handle, src.desc, filter.desc, conv.desc, dst.desc,
          requested, &mut returned, buf.as_mut_ptr())
    };
    check(raw)?;
    unsafe { buf.set_len(returned as usize); }
    Ok(buf)
}

/// Run all candidate forward-convolution algorithms and return measured runtimes.
pub fn find_convolution_forward_algorithm(
    handle: &Handle,
    src: &TensorDescriptor, filter: &FilterDescriptor,
    conv: &ConvolutionDescriptor, dst: &TensorDescriptor,
    requested: i32,
) -> Result<Vec<FwdAlgoPerf>> {
    let cu = cudnn()?;
    let f = cu.cudnn_find_convolution_forward_algorithm()?;
    let mut returned: core::ffi::c_int = 0;
    let mut buf: Vec<FwdAlgoPerf> = Vec::with_capacity(requested as usize);
    let raw = unsafe {
        f(handle.handle, src.desc, filter.desc, conv.desc, dst.desc,
          requested, &mut returned, buf.as_mut_ptr())
    };
    check(raw)?;
    unsafe { buf.set_len(returned as usize); }
    Ok(buf)
}

/// Heuristic-pick backward-data convolution algorithms.
pub fn get_convolution_backward_data_algorithm(
    handle: &Handle,
    filter: &FilterDescriptor, diff: &TensorDescriptor,
    conv: &ConvolutionDescriptor, grad: &TensorDescriptor,
    requested: i32,
) -> Result<Vec<BwdDataAlgoPerf>> {
    let cu = cudnn()?;
    let f = cu.cudnn_get_convolution_backward_data_algorithm_v7()?;
    let mut returned: core::ffi::c_int = 0;
    let mut buf: Vec<BwdDataAlgoPerf> = Vec::with_capacity(requested as usize);
    let raw = unsafe {
        f(handle.handle, filter.desc, diff.desc, conv.desc, grad.desc,
          requested, &mut returned, buf.as_mut_ptr())
    };
    check(raw)?;
    unsafe { buf.set_len(returned as usize); }
    Ok(buf)
}

/// Heuristic-pick backward-filter convolution algorithms.
pub fn get_convolution_backward_filter_algorithm(
    handle: &Handle,
    src: &TensorDescriptor, diff: &TensorDescriptor,
    conv: &ConvolutionDescriptor, grad: &FilterDescriptor,
    requested: i32,
) -> Result<Vec<BwdFilterAlgoPerf>> {
    let cu = cudnn()?;
    let f = cu.cudnn_get_convolution_backward_filter_algorithm_v7()?;
    let mut returned: core::ffi::c_int = 0;
    let mut buf: Vec<BwdFilterAlgoPerf> = Vec::with_capacity(requested as usize);
    let raw = unsafe {
        f(handle.handle, src.desc, diff.desc, conv.desc, grad.desc,
          requested, &mut returned, buf.as_mut_ptr())
    };
    check(raw)?;
    unsafe { buf.set_len(returned as usize); }
    Ok(buf)
}

// ============================================================================
// Tier 3 — Generic Normalization API enums (cuDNN 8+) + workspace queries
// ============================================================================

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum NormMode {
    PerActivation,
    #[default]
    PerChannel,
}
impl NormMode {
    fn raw(self) -> cudnnNormMode_t {
        match self {
            NormMode::PerActivation => cudnnNormMode_t::PerActivation,
            NormMode::PerChannel => cudnnNormMode_t::PerChannel,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum NormAlgo {
    #[default]
    Standard,
    Persist,
}
impl NormAlgo {
    fn raw(self) -> cudnnNormAlgo_t {
        match self {
            NormAlgo::Standard => cudnnNormAlgo_t::Standard,
            NormAlgo::Persist => cudnnNormAlgo_t::Persist,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum NormOp {
    #[default]
    Norm,
    NormActivation,
    NormAddActivation,
}
impl NormOp {
    fn raw(self) -> cudnnNormOps_t {
        match self {
            NormOp::Norm => cudnnNormOps_t::Norm,
            NormOp::NormActivation => cudnnNormOps_t::NormActivation,
            NormOp::NormAddActivation => cudnnNormOps_t::NormAddActivation,
        }
    }
}

/// Optional fused op for the `*Ex` BatchNorm variants.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum BnOp {
    #[default]
    Bn,
    BnActivation,
    BnAddActivation,
}
impl BnOp {
    fn raw(self) -> cudnnBatchNormOps_t {
        match self {
            BnOp::Bn => cudnnBatchNormOps_t::Bn,
            BnOp::BnActivation => cudnnBatchNormOps_t::BnActivation,
            BnOp::BnAddActivation => cudnnBatchNormOps_t::BnAddActivation,
        }
    }
}

/// Workspace bytes for [`batch_normalization_forward_training_ex`].
#[allow(clippy::too_many_arguments)]
pub fn batch_normalization_forward_training_ex_workspace_size(
    handle: &Handle,
    mode: BatchNormMode, bn_ops: BnOp,
    x: &TensorDescriptor, z: &TensorDescriptor, y: &TensorDescriptor,
    bn_smbv: &TensorDescriptor, activation: &ActivationDescriptor,
) -> Result<usize> {
    let cu = cudnn()?;
    let f = cu.cudnn_get_batch_normalization_forward_training_ex_workspace_size()?;
    let mut size = 0usize;
    check(unsafe {
        f(handle.handle, mode.raw(), bn_ops.raw(),
          x.desc, z.desc, y.desc, bn_smbv.desc, activation.desc, &mut size)
    })?;
    Ok(size)
}

/// Workspace bytes for [`batch_normalization_backward_ex`].
#[allow(clippy::too_many_arguments)]
pub fn batch_normalization_backward_ex_workspace_size(
    handle: &Handle,
    mode: BatchNormMode, bn_ops: BnOp,
    x: &TensorDescriptor, y: &TensorDescriptor, dy: &TensorDescriptor,
    dz: &TensorDescriptor, dx: &TensorDescriptor,
    d_bn_scale_bias: &TensorDescriptor, activation: &ActivationDescriptor,
) -> Result<usize> {
    let cu = cudnn()?;
    let f = cu.cudnn_get_batch_normalization_backward_ex_workspace_size()?;
    let mut size = 0usize;
    check(unsafe {
        f(handle.handle, mode.raw(), bn_ops.raw(),
          x.desc, y.desc, dy.desc, dz.desc, dx.desc,
          d_bn_scale_bias.desc, activation.desc, &mut size)
    })?;
    Ok(size)
}

/// Reserve-space bytes for the `*Ex` BatchNorm pair.
pub fn batch_normalization_training_ex_reserve_space_size(
    handle: &Handle,
    mode: BatchNormMode, bn_ops: BnOp,
    activation: &ActivationDescriptor, x: &TensorDescriptor,
) -> Result<usize> {
    let cu = cudnn()?;
    let f = cu.cudnn_get_batch_normalization_training_ex_reserve_space_size()?;
    let mut size = 0usize;
    check(unsafe {
        f(handle.handle, mode.raw(), bn_ops.raw(), activation.desc, x.desc, &mut size)
    })?;
    Ok(size)
}

// ============================================================================
// Tier 4 — RNN v8 + companion descriptors
// ============================================================================

/// Owned RNN descriptor.
pub struct RnnDescriptor {
    desc: baracuda_cudnn_sys::cudnnRNNDescriptor_t,
}
unsafe impl Send for RnnDescriptor {}
impl core::fmt::Debug for RnnDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RnnDescriptor").field("desc", &self.desc).finish_non_exhaustive()
    }
}
impl RnnDescriptor {
    pub fn new() -> Result<Self> {
        let c = cudnn()?;
        let create = c.cudnn_create_rnn_descriptor()?;
        let mut desc: baracuda_cudnn_sys::cudnnRNNDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        Ok(Self { desc })
    }

    /// Configure with the v8 setup. After this, call
    /// [`build_rnn_dynamic`] to bind a specific minibatch size.
    #[allow(clippy::too_many_arguments)]
    pub fn set_v8(
        &self,
        algo: i32, cell_mode: i32, bias_mode: i32,
        dir_mode: i32, input_mode: i32,
        data_type: DType, math_prec: DType, math_type: MathType,
        input_size: i32, hidden_size: i32, proj_size: i32, num_layers: i32,
        dropout: &DropoutDescriptor, aux_flags: u32,
    ) -> Result<()> {
        use baracuda_cudnn_sys::{cudnnDirectionMode_t, cudnnRNNAlgo_t, cudnnRNNInputMode_t, cudnnRNNMode_t};
        let c = cudnn()?;
        let f = c.cudnn_set_rnn_descriptor_v8()?;
        let algo = match algo {
            0 => cudnnRNNAlgo_t::Standard,
            1 => cudnnRNNAlgo_t::PersistStatic,
            2 => cudnnRNNAlgo_t::PersistDynamic,
            _ => cudnnRNNAlgo_t::PersistStaticSmallH,
        };
        let cell = match cell_mode {
            0 => cudnnRNNMode_t::ReluRnn,
            1 => cudnnRNNMode_t::TanhRnn,
            2 => cudnnRNNMode_t::Lstm,
            _ => cudnnRNNMode_t::Gru,
        };
        let dir = if dir_mode == 1 { cudnnDirectionMode_t::Bidirectional } else { cudnnDirectionMode_t::Unidirectional };
        let im = if input_mode == 1 { cudnnRNNInputMode_t::SkipInput } else { cudnnRNNInputMode_t::LinearInput };
        check(unsafe {
            f(self.desc, algo, cell, bias_mode, dir, im,
              data_type.raw(), math_prec.raw(), math_type.raw(),
              input_size, hidden_size, proj_size, num_layers,
              dropout.desc, aux_flags)
        })
    }

    #[inline]
    pub fn as_raw(&self) -> baracuda_cudnn_sys::cudnnRNNDescriptor_t { self.desc }
}
impl Drop for RnnDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_rnn_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Owned RNN-data descriptor used by the v8 RNN forward / backward path.
pub struct RnnDataDescriptor {
    desc: baracuda_cudnn_sys::cudnnRNNDataDescriptor_t,
}
unsafe impl Send for RnnDataDescriptor {}
impl core::fmt::Debug for RnnDataDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RnnDataDescriptor").field("desc", &self.desc).finish_non_exhaustive()
    }
}
impl RnnDataDescriptor {
    pub fn new() -> Result<Self> {
        let c = cudnn()?;
        let create = c.cudnn_create_rnn_data_descriptor()?;
        let mut desc: baracuda_cudnn_sys::cudnnRNNDataDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        Ok(Self { desc })
    }
    #[inline]
    pub fn as_raw(&self) -> baracuda_cudnn_sys::cudnnRNNDataDescriptor_t { self.desc }
}
impl Drop for RnnDataDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_rnn_data_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Finalize an RNN descriptor for a specific minibatch size.
pub fn build_rnn_dynamic(handle: &Handle, rnn: &RnnDescriptor, mini_batch: i32) -> Result<()> {
    let c = cudnn()?;
    let f = c.cudnn_build_rnn_dynamic()?;
    check(unsafe { f(handle.handle, rnn.desc, mini_batch) })
}

/// Returns `(work_space_size, reserve_space_size)`.
/// `fwd_mode = 0` for inference, `1` for training.
pub fn rnn_temp_space_sizes(
    handle: &Handle, rnn: &RnnDescriptor, fwd_mode: i32, x: &RnnDataDescriptor,
) -> Result<(usize, usize)> {
    let c = cudnn()?;
    let f = c.cudnn_get_rnn_temp_space_sizes()?;
    let (mut ws, mut rs) = (0usize, 0usize);
    check(unsafe { f(handle.handle, rnn.desc, fwd_mode, x.desc, &mut ws, &mut rs) })?;
    Ok((ws, rs))
}

/// Bytes the RNN's weight space needs.
pub fn rnn_weight_space_size(handle: &Handle, rnn: &RnnDescriptor) -> Result<usize> {
    let c = cudnn()?;
    let f = c.cudnn_get_rnn_weight_space_size()?;
    let mut size = 0usize;
    check(unsafe { f(handle.handle, rnn.desc, &mut size) })?;
    Ok(size)
}

// ============================================================================
// Tier 5 — Multi-head attention
// ============================================================================

/// Multi-head attention descriptor.
pub struct AttnDescriptor {
    desc: cudnnAttnDescriptor_t,
}
unsafe impl Send for AttnDescriptor {}
impl core::fmt::Debug for AttnDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AttnDescriptor").field("desc", &self.desc).finish_non_exhaustive()
    }
}
impl AttnDescriptor {
    pub fn new() -> Result<Self> {
        let c = cudnn()?;
        let cu = c.cudnn_create_attn_descriptor()?;
        let mut desc: cudnnAttnDescriptor_t = core::ptr::null_mut();
        check(unsafe { cu(&mut desc) })?;
        Ok(Self { desc })
    }

    /// Configure the descriptor. See `cudnnSetAttnDescriptor` in the
    /// cuDNN reference for each parameter.
    #[allow(clippy::too_many_arguments)]
    pub fn set(
        &self,
        attn_mode: u32, n_heads: i32, sm_scaler: f64,
        data_type: DType, compute_prec: DType, math_type: MathType,
        attn_dropout: &DropoutDescriptor, post_dropout: &DropoutDescriptor,
        q_size: i32, k_size: i32, v_size: i32,
        q_proj_size: i32, k_proj_size: i32, v_proj_size: i32, o_proj_size: i32,
        qo_max_seq_length: i32, kv_max_seq_length: i32,
        max_batch_size: i32, max_beam_size: i32,
    ) -> Result<()> {
        let c = cudnn()?;
        let f = c.cudnn_set_attn_descriptor()?;
        check(unsafe {
            f(self.desc, attn_mode, n_heads, sm_scaler,
              data_type.raw(), compute_prec.raw(), math_type.raw(),
              attn_dropout.desc, post_dropout.desc,
              q_size, k_size, v_size,
              q_proj_size, k_proj_size, v_proj_size, o_proj_size,
              qo_max_seq_length, kv_max_seq_length,
              max_batch_size, max_beam_size)
        })
    }

    #[inline]
    pub fn as_raw(&self) -> cudnnAttnDescriptor_t { self.desc }
}
impl Drop for AttnDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_attn_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Buffer requirements `(weights, work_space, reserve_space)`.
pub fn multi_head_attn_buffers(
    handle: &Handle, attn: &AttnDescriptor,
) -> Result<(usize, usize, usize)> {
    let c = cudnn()?;
    let f = c.cudnn_get_multi_head_attn_buffers()?;
    let (mut w, mut ws, mut rs) = (0usize, 0usize, 0usize);
    check(unsafe { f(handle.handle, attn.desc, &mut w, &mut ws, &mut rs) })?;
    Ok((w, ws, rs))
}

/// Sequence-data descriptor used by multi-head attention.
pub struct SeqDataDescriptor {
    desc: cudnnSeqDataDescriptor_t,
}
unsafe impl Send for SeqDataDescriptor {}
impl core::fmt::Debug for SeqDataDescriptor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SeqDataDescriptor").field("desc", &self.desc).finish_non_exhaustive()
    }
}
impl SeqDataDescriptor {
    pub fn new() -> Result<Self> {
        let c = cudnn()?;
        let cu = c.cudnn_create_seq_data_descriptor()?;
        let mut desc: cudnnSeqDataDescriptor_t = core::ptr::null_mut();
        check(unsafe { cu(&mut desc) })?;
        Ok(Self { desc })
    }

    /// # Safety
    /// `padding_fill` must point to a value of the descriptor's data type.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn set(
        &self,
        data_type: DType,
        dim_a: &[i32], axes: &[i32], seq_length_array: &[i32],
        padding_fill: *const core::ffi::c_void,
    ) -> Result<()> {
        let c = cudnn()?;
        let f = c.cudnn_set_seq_data_descriptor()?;
        check(f(
            self.desc, data_type.raw(),
            dim_a.len() as core::ffi::c_int,
            dim_a.as_ptr(), axes.as_ptr(),
            seq_length_array.len(), seq_length_array.as_ptr(),
            padding_fill,
        ))
    }

    #[inline]
    pub fn as_raw(&self) -> cudnnSeqDataDescriptor_t { self.desc }
}
impl Drop for SeqDataDescriptor {
    fn drop(&mut self) {
        if let Ok(c) = cudnn() {
            if let Ok(cu) = c.cudnn_destroy_seq_data_descriptor() {
                let _ = unsafe { cu(self.desc) };
            }
        }
    }
}

/// Re-exports for callers that want raw type access.
pub use baracuda_cudnn_sys::{cudnnMathType_t as RawMathType, cudnnReorderType_t as RawReorderType};

// ============================================================================
// Tier 1 leftovers — 4-D descriptor readback + DropoutDescriptor get/restore
// ============================================================================

impl TensorDescriptor {
    /// Strided 4-D constructor — per-axis strides instead of the
    /// row-major / channels-last layouts [`new_4d`](Self::new_4d) implies.
    #[allow(clippy::too_many_arguments)]
    pub fn new_4d_ex(
        dtype: DType,
        n: i32, c: i32, h: i32, w: i32,
        n_stride: i32, c_stride: i32, h_stride: i32, w_stride: i32,
    ) -> Result<Self> {
        let cu = cudnn()?;
        let create = cu.cudnn_create_tensor_descriptor()?;
        let set = cu.cudnn_set_tensor_4d_descriptor_ex()?;
        let mut desc: cudnnTensorDescriptor_t = core::ptr::null_mut();
        check(unsafe { create(&mut desc) })?;
        let this = Self { desc };
        check(unsafe {
            set(this.desc, dtype.raw(), n, c, h, w,
                n_stride, c_stride, h_stride, w_stride)
        })?;
        Ok(this)
    }

    /// Read the 4-D parameters back out: `(dtype, n, c, h, w, n_stride,
    /// c_stride, h_stride, w_stride)`.
    pub fn get_4d(&self) -> Result<(DType, i32, i32, i32, i32, i32, i32, i32, i32)> {
        let cu = cudnn()?;
        let f = cu.cudnn_get_tensor_4d_descriptor()?;
        let mut dt = cudnnDataType_t::Float;
        let (mut n, mut c, mut h, mut w) = (0i32, 0i32, 0i32, 0i32);
        let (mut ns, mut cs, mut hs, mut ws) = (0i32, 0i32, 0i32, 0i32);
        check(unsafe {
            f(self.desc, &mut dt, &mut n, &mut c, &mut h, &mut w,
              &mut ns, &mut cs, &mut hs, &mut ws)
        })?;
        let dtype = match dt {
            cudnnDataType_t::Float => DType::F32,
            cudnnDataType_t::Double => DType::F64,
            cudnnDataType_t::Half => DType::F16,
            cudnnDataType_t::BFloat16 => DType::BF16,
            cudnnDataType_t::Int8 => DType::I8,
            cudnnDataType_t::Int32 => DType::I32,
            _ => DType::F32,
        };
        Ok((dtype, n, c, h, w, ns, cs, hs, ws))
    }
}

impl FilterDescriptor {
    /// Read 4-D filter parameters: `(dtype, format, k, c, h, w)`.
    pub fn get_4d(&self) -> Result<(DType, TensorFormat, i32, i32, i32, i32)> {
        let cu = cudnn()?;
        let f = cu.cudnn_get_filter_4d_descriptor()?;
        let mut dt = cudnnDataType_t::Float;
        let mut fmt = cudnnTensorFormat_t::Nchw;
        let (mut k, mut c, mut h, mut w) = (0i32, 0i32, 0i32, 0i32);
        check(unsafe {
            f(self.desc, &mut dt, &mut fmt, &mut k, &mut c, &mut h, &mut w)
        })?;
        let dtype = match dt {
            cudnnDataType_t::Float => DType::F32,
            cudnnDataType_t::Double => DType::F64,
            cudnnDataType_t::Half => DType::F16,
            cudnnDataType_t::BFloat16 => DType::BF16,
            cudnnDataType_t::Int8 => DType::I8,
            cudnnDataType_t::Int32 => DType::I32,
            _ => DType::F32,
        };
        let format = match fmt {
            cudnnTensorFormat_t::Nchw => TensorFormat::Nchw,
            cudnnTensorFormat_t::Nhwc => TensorFormat::Nhwc,
            _ => TensorFormat::Nchw,
        };
        Ok((dtype, format, k, c, h, w))
    }
}

impl DropoutDescriptor {
    /// Read a dropout descriptor's current state. Returns `(dropout_p,
    /// states_ptr, seed)`. The states pointer is owned by cuDNN.
    pub fn get(&self, handle: &Handle) -> Result<(f32, *mut core::ffi::c_void, u64)> {
        let cu = cudnn()?;
        let f = cu.cudnn_get_dropout_descriptor()?;
        let mut dropout: f32 = 0.0;
        let mut states: *mut core::ffi::c_void = core::ptr::null_mut();
        let mut seed: u64 = 0;
        check(unsafe { f(self.desc, handle.handle, &mut dropout, &mut states, &mut seed) })?;
        Ok((dropout, states, seed))
    }

    /// Reattach a previously-saved RNG state buffer to this descriptor.
    /// Useful for reproducible eval / resume.
    ///
    /// # Safety
    /// `states` must be a buffer of at least [`dropout_states_size`] bytes
    /// from the same `handle`, valid for the descriptor's lifetime.
    pub unsafe fn restore(
        &self, handle: &Handle, dropout: f32,
        states: *mut core::ffi::c_void, state_size: usize, seed: u64,
    ) -> Result<()> {
        let cu = cudnn()?;
        let f = cu.cudnn_restore_dropout_descriptor()?;
        check(f(self.desc, handle.handle, dropout, states, state_size, seed))
    }
}

// ============================================================================
// BatchNormalization "Ex" — actual forward/backward (workspace queries are
// already above).
// ============================================================================

/// BN training forward with optional fused activation / residual add.
#[allow(clippy::too_many_arguments)]
pub fn batch_normalization_forward_training_ex<T: DeviceRepr>(
    handle: &Handle,
    mode: BatchNormMode, bn_ops: BnOp,
    alpha: f32, beta: f32,
    x_desc: &TensorDescriptor, x: &DeviceBuffer<T>,
    z_desc: &TensorDescriptor, z: &DeviceBuffer<T>,
    y_desc: &TensorDescriptor, y: &mut DeviceBuffer<T>,
    bn_smbv_desc: &TensorDescriptor,
    bn_scale: &DeviceBuffer<T>, bn_bias: &DeviceBuffer<T>,
    exponential_avg_factor: f64,
    running_mean: &mut DeviceBuffer<T>, running_var: &mut DeviceBuffer<T>,
    epsilon: f64,
    saved_mean: &mut DeviceBuffer<T>, saved_inv_var: &mut DeviceBuffer<T>,
    activation: &ActivationDescriptor,
    workspace: &mut DeviceBuffer<u8>, reserve: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_batch_normalization_forward_training_ex()?;
    check(unsafe {
        cu(
            handle.handle, mode.raw(), bn_ops.raw(),
            &alpha as *const f32 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            x_desc.desc, x.as_raw().0 as *const core::ffi::c_void,
            z_desc.desc, z.as_raw().0 as *const core::ffi::c_void,
            y_desc.desc, y.as_raw().0 as *mut core::ffi::c_void,
            bn_smbv_desc.desc,
            bn_scale.as_raw().0 as *const core::ffi::c_void,
            bn_bias.as_raw().0 as *const core::ffi::c_void,
            exponential_avg_factor,
            running_mean.as_raw().0 as *mut core::ffi::c_void,
            running_var.as_raw().0 as *mut core::ffi::c_void,
            epsilon,
            saved_mean.as_raw().0 as *mut core::ffi::c_void,
            saved_inv_var.as_raw().0 as *mut core::ffi::c_void,
            activation.desc,
            workspace.as_raw().0 as *mut core::ffi::c_void, workspace.byte_size(),
            reserve.as_raw().0 as *mut core::ffi::c_void, reserve.byte_size(),
        )
    })
}

/// BN backward matching [`batch_normalization_forward_training_ex`].
#[allow(clippy::too_many_arguments)]
pub fn batch_normalization_backward_ex<T: DeviceRepr>(
    handle: &Handle,
    mode: BatchNormMode, bn_ops: BnOp,
    alpha_data: f32, beta_data: f32,
    alpha_param: f32, beta_param: f32,
    x_desc: &TensorDescriptor, x: &DeviceBuffer<T>,
    y_desc: &TensorDescriptor, y: &DeviceBuffer<T>,
    dy_desc: &TensorDescriptor, dy: &DeviceBuffer<T>,
    dz_desc: &TensorDescriptor, dz: &mut DeviceBuffer<T>,
    dx_desc: &TensorDescriptor, dx: &mut DeviceBuffer<T>,
    d_bn_scale_bias_desc: &TensorDescriptor,
    bn_scale: &DeviceBuffer<T>, bn_bias: &DeviceBuffer<T>,
    d_bn_scale: &mut DeviceBuffer<T>, d_bn_bias: &mut DeviceBuffer<T>,
    epsilon: f64,
    saved_mean: &DeviceBuffer<T>, saved_inv_var: &DeviceBuffer<T>,
    activation: &ActivationDescriptor,
    workspace: &mut DeviceBuffer<u8>, reserve: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_batch_normalization_backward_ex()?;
    check(unsafe {
        cu(
            handle.handle, mode.raw(), bn_ops.raw(),
            &alpha_data as *const f32 as *const core::ffi::c_void,
            &beta_data as *const f32 as *const core::ffi::c_void,
            &alpha_param as *const f32 as *const core::ffi::c_void,
            &beta_param as *const f32 as *const core::ffi::c_void,
            x_desc.desc, x.as_raw().0 as *const core::ffi::c_void,
            y_desc.desc, y.as_raw().0 as *const core::ffi::c_void,
            dy_desc.desc, dy.as_raw().0 as *const core::ffi::c_void,
            dz_desc.desc, dz.as_raw().0 as *mut core::ffi::c_void,
            dx_desc.desc, dx.as_raw().0 as *mut core::ffi::c_void,
            d_bn_scale_bias_desc.desc,
            bn_scale.as_raw().0 as *const core::ffi::c_void,
            bn_bias.as_raw().0 as *const core::ffi::c_void,
            d_bn_scale.as_raw().0 as *mut core::ffi::c_void,
            d_bn_bias.as_raw().0 as *mut core::ffi::c_void,
            epsilon,
            saved_mean.as_raw().0 as *const core::ffi::c_void,
            saved_inv_var.as_raw().0 as *const core::ffi::c_void,
            activation.desc,
            workspace.as_raw().0 as *mut core::ffi::c_void, workspace.byte_size(),
            reserve.as_raw().0 as *mut core::ffi::c_void, reserve.byte_size(),
        )
    })
}

// ============================================================================
// Tier 3 - Generic Normalization API ops (cuDNN 8+)
// ============================================================================

/// Inference-time generic normalization.
#[allow(clippy::too_many_arguments)]
pub fn normalization_forward_inference<T: DeviceRepr>(
    handle: &Handle,
    mode: NormMode, ops: NormOp, algo: NormAlgo,
    alpha: f32, beta: f32,
    x_desc: &TensorDescriptor, x: &DeviceBuffer<T>,
    norm_scale_bias_desc: &TensorDescriptor,
    norm_scale: &DeviceBuffer<T>, norm_bias: &DeviceBuffer<T>,
    norm_mean_var_desc: &TensorDescriptor,
    estimated_mean: &DeviceBuffer<T>, estimated_var: &DeviceBuffer<T>,
    z_desc: &TensorDescriptor, z: &DeviceBuffer<T>,
    activation: &ActivationDescriptor,
    y_desc: &TensorDescriptor, y: &mut DeviceBuffer<T>,
    epsilon: f64, group_count: i32,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_normalization_forward_inference()?;
    check(unsafe {
        cu(
            handle.handle, mode.raw(), ops.raw(), algo.raw(),
            &alpha as *const f32 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            x_desc.desc, x.as_raw().0 as *const core::ffi::c_void,
            norm_scale_bias_desc.desc,
            norm_scale.as_raw().0 as *const core::ffi::c_void,
            norm_bias.as_raw().0 as *const core::ffi::c_void,
            norm_mean_var_desc.desc,
            estimated_mean.as_raw().0 as *const core::ffi::c_void,
            estimated_var.as_raw().0 as *const core::ffi::c_void,
            z_desc.desc, z.as_raw().0 as *const core::ffi::c_void,
            activation.desc,
            y_desc.desc, y.as_raw().0 as *mut core::ffi::c_void,
            epsilon, group_count,
        )
    })
}

/// Workspace bytes for normalization_forward_training.
#[allow(clippy::too_many_arguments)]
pub fn normalization_forward_training_workspace_size(
    handle: &Handle,
    mode: NormMode, ops: NormOp, algo: NormAlgo,
    x_desc: &TensorDescriptor, z_desc: &TensorDescriptor,
    y_desc: &TensorDescriptor, norm_scale_bias_desc: &TensorDescriptor,
    activation: &ActivationDescriptor, norm_mean_var_desc: &TensorDescriptor,
    group_count: i32,
) -> Result<usize> {
    let c = cudnn()?;
    let f = c.cudnn_get_normalization_forward_training_workspace_size()?;
    let mut size = 0usize;
    check(unsafe {
        f(handle.handle, mode.raw(), ops.raw(), algo.raw(),
          x_desc.desc, z_desc.desc, y_desc.desc, norm_scale_bias_desc.desc,
          activation.desc, norm_mean_var_desc.desc, &mut size, group_count)
    })?;
    Ok(size)
}

/// Workspace bytes for normalization_backward.
#[allow(clippy::too_many_arguments)]
pub fn normalization_backward_workspace_size(
    handle: &Handle,
    mode: NormMode, ops: NormOp, algo: NormAlgo,
    x_desc: &TensorDescriptor, y_desc: &TensorDescriptor,
    dy_desc: &TensorDescriptor, dz_desc: &TensorDescriptor,
    dx_desc: &TensorDescriptor, d_norm_scale_bias_desc: &TensorDescriptor,
    activation: &ActivationDescriptor, norm_mean_var_desc: &TensorDescriptor,
    group_count: i32,
) -> Result<usize> {
    let c = cudnn()?;
    let f = c.cudnn_get_normalization_backward_workspace_size()?;
    let mut size = 0usize;
    check(unsafe {
        f(handle.handle, mode.raw(), ops.raw(), algo.raw(),
          x_desc.desc, y_desc.desc, dy_desc.desc, dz_desc.desc,
          dx_desc.desc, d_norm_scale_bias_desc.desc,
          activation.desc, norm_mean_var_desc.desc, &mut size, group_count)
    })?;
    Ok(size)
}

/// Reserve-space bytes for the training fwd/bwd pair.
pub fn normalization_training_reserve_space_size(
    handle: &Handle,
    mode: NormMode, ops: NormOp, algo: NormAlgo,
    activation: &ActivationDescriptor, x_desc: &TensorDescriptor,
    group_count: i32,
) -> Result<usize> {
    let c = cudnn()?;
    let f = c.cudnn_get_normalization_training_reserve_space_size()?;
    let mut size = 0usize;
    check(unsafe {
        f(handle.handle, mode.raw(), ops.raw(), algo.raw(),
          activation.desc, x_desc.desc, &mut size, group_count)
    })?;
    Ok(size)
}

/// Training-time forward generic normalization.
#[allow(clippy::too_many_arguments)]
pub fn normalization_forward_training<T: DeviceRepr>(
    handle: &Handle,
    mode: NormMode, ops: NormOp, algo: NormAlgo,
    alpha: f32, beta: f32,
    x_desc: &TensorDescriptor, x: &DeviceBuffer<T>,
    norm_scale_bias_desc: &TensorDescriptor,
    norm_scale: &DeviceBuffer<T>, norm_bias: &DeviceBuffer<T>,
    exponential_avg_factor: f64,
    norm_mean_var_desc: &TensorDescriptor,
    running_mean: &mut DeviceBuffer<T>, running_var: &mut DeviceBuffer<T>,
    epsilon: f64,
    saved_mean: &mut DeviceBuffer<T>, saved_inv_var: &mut DeviceBuffer<T>,
    activation: &ActivationDescriptor,
    z_desc: &TensorDescriptor, z: &DeviceBuffer<T>,
    y_desc: &TensorDescriptor, y: &mut DeviceBuffer<T>,
    workspace: &mut DeviceBuffer<u8>, reserve: &mut DeviceBuffer<u8>,
    group_count: i32,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_normalization_forward_training()?;
    check(unsafe {
        cu(
            handle.handle, mode.raw(), ops.raw(), algo.raw(),
            &alpha as *const f32 as *const core::ffi::c_void,
            &beta as *const f32 as *const core::ffi::c_void,
            x_desc.desc, x.as_raw().0 as *const core::ffi::c_void,
            norm_scale_bias_desc.desc,
            norm_scale.as_raw().0 as *const core::ffi::c_void,
            norm_bias.as_raw().0 as *const core::ffi::c_void,
            exponential_avg_factor,
            norm_mean_var_desc.desc,
            running_mean.as_raw().0 as *mut core::ffi::c_void,
            running_var.as_raw().0 as *mut core::ffi::c_void,
            epsilon,
            saved_mean.as_raw().0 as *mut core::ffi::c_void,
            saved_inv_var.as_raw().0 as *mut core::ffi::c_void,
            activation.desc,
            z_desc.desc, z.as_raw().0 as *const core::ffi::c_void,
            y_desc.desc, y.as_raw().0 as *mut core::ffi::c_void,
            workspace.as_raw().0 as *mut core::ffi::c_void, workspace.byte_size(),
            reserve.as_raw().0 as *mut core::ffi::c_void, reserve.byte_size(),
            group_count,
        )
    })
}

/// Backward generic normalization.
#[allow(clippy::too_many_arguments)]
pub fn normalization_backward<T: DeviceRepr>(
    handle: &Handle,
    mode: NormMode, ops: NormOp, algo: NormAlgo,
    alpha_data: f32, beta_data: f32,
    alpha_param: f32, beta_param: f32,
    x_desc: &TensorDescriptor, x: &DeviceBuffer<T>,
    y_desc: &TensorDescriptor, y: &DeviceBuffer<T>,
    dy_desc: &TensorDescriptor, dy: &DeviceBuffer<T>,
    dz_desc: &TensorDescriptor, dz: &mut DeviceBuffer<T>,
    dx_desc: &TensorDescriptor, dx: &mut DeviceBuffer<T>,
    d_norm_scale_bias_desc: &TensorDescriptor,
    norm_scale: &DeviceBuffer<T>, norm_bias: &DeviceBuffer<T>,
    d_norm_scale: &mut DeviceBuffer<T>, d_norm_bias: &mut DeviceBuffer<T>,
    epsilon: f64,
    norm_mean_var_desc: &TensorDescriptor,
    saved_mean: &DeviceBuffer<T>, saved_inv_var: &DeviceBuffer<T>,
    activation: &ActivationDescriptor,
    workspace: &mut DeviceBuffer<u8>, reserve: &mut DeviceBuffer<u8>,
    group_count: i32,
) -> Result<()> {
    let c = cudnn()?;
    let cu = c.cudnn_normalization_backward()?;
    check(unsafe {
        cu(
            handle.handle, mode.raw(), ops.raw(), algo.raw(),
            &alpha_data as *const f32 as *const core::ffi::c_void,
            &beta_data as *const f32 as *const core::ffi::c_void,
            &alpha_param as *const f32 as *const core::ffi::c_void,
            &beta_param as *const f32 as *const core::ffi::c_void,
            x_desc.desc, x.as_raw().0 as *const core::ffi::c_void,
            y_desc.desc, y.as_raw().0 as *const core::ffi::c_void,
            dy_desc.desc, dy.as_raw().0 as *const core::ffi::c_void,
            dz_desc.desc, dz.as_raw().0 as *mut core::ffi::c_void,
            dx_desc.desc, dx.as_raw().0 as *mut core::ffi::c_void,
            d_norm_scale_bias_desc.desc,
            norm_scale.as_raw().0 as *const core::ffi::c_void,
            norm_bias.as_raw().0 as *const core::ffi::c_void,
            d_norm_scale.as_raw().0 as *mut core::ffi::c_void,
            d_norm_bias.as_raw().0 as *mut core::ffi::c_void,
            epsilon,
            norm_mean_var_desc.desc,
            saved_mean.as_raw().0 as *const core::ffi::c_void,
            saved_inv_var.as_raw().0 as *const core::ffi::c_void,
            activation.desc,
            workspace.as_raw().0 as *mut core::ffi::c_void, workspace.byte_size(),
            reserve.as_raw().0 as *mut core::ffi::c_void, reserve.byte_size(),
            group_count,
        )
    })
}

// ============================================================================
// Tier 5 - Multi-head attention ops (forward + backward + weights query)
// ============================================================================

/// Look up the descriptor of one of the (Q/K/V/O) weight matrices inside
/// the packed weights buffer.
///
/// `w_kind` matches cuDNN's `cudnnMultiHeadAttnWeightKind_t`:
///   0 = Q weights, 1 = K weights, 2 = V weights, 3 = O weights,
///   4 = Q bias, 5 = K bias, 6 = V bias, 7 = O bias.
///
/// # Safety
/// `weights` must point at the multi-head attention weight buffer
/// produced from `multi_head_attn_buffers`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn get_multi_head_attn_weights(
    handle: &Handle,
    attn: &AttnDescriptor,
    w_kind: i32,
    weight_size_in_bytes: usize,
    weights: *const core::ffi::c_void,
    w_desc: &TensorDescriptor,
) -> Result<*mut core::ffi::c_void> {
    let c = cudnn()?;
    let f = c.cudnn_get_multi_head_attn_weights()?;
    let mut addr: *mut core::ffi::c_void = core::ptr::null_mut();
    check(f(
        handle.handle, attn.desc, w_kind, weight_size_in_bytes, weights,
        w_desc.desc, &mut addr,
    ))?;
    Ok(addr)
}

/// Forward multi-head attention. The huge parameter list mirrors cuDNN's
/// `cudnnMultiHeadAttnForward` exactly; see the cuDNN reference for the
/// meaning of each window / sequence-length array.
///
/// # Safety
/// All device buffers must satisfy the size and alignment requirements
/// reported by [`multi_head_attn_buffers`]. `lo_win_idx`/`hi_win_idx`
/// must be host arrays of length `qo_max_seq_length`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn multi_head_attn_forward(
    handle: &Handle,
    attn: &AttnDescriptor,
    curr_idx: i32,
    lo_win_idx: &[i32],
    hi_win_idx: &[i32],
    dev_seq_lengths_qo: *const i32,
    dev_seq_lengths_kv: *const i32,
    q_desc: &SeqDataDescriptor, queries: *const core::ffi::c_void,
    residuals: *const core::ffi::c_void,
    k_desc: &SeqDataDescriptor, keys: *const core::ffi::c_void,
    v_desc: &SeqDataDescriptor, values: *const core::ffi::c_void,
    o_desc: &SeqDataDescriptor, out: *mut core::ffi::c_void,
    weights: &DeviceBuffer<u8>,
    work_space: &mut DeviceBuffer<u8>,
    reserve_space: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cudnn()?;
    let f = c.cudnn_multi_head_attn_forward()?;
    check(f(
        handle.handle, attn.desc,
        curr_idx, lo_win_idx.as_ptr(), hi_win_idx.as_ptr(),
        dev_seq_lengths_qo, dev_seq_lengths_kv,
        q_desc.desc, queries, residuals,
        k_desc.desc, keys,
        v_desc.desc, values,
        o_desc.desc, out,
        weights.byte_size(), weights.as_raw().0 as *const core::ffi::c_void,
        work_space.byte_size(), work_space.as_raw().0 as *mut core::ffi::c_void,
        reserve_space.byte_size(), reserve_space.as_raw().0 as *mut core::ffi::c_void,
    ))
}

/// Multi-head attention backward — data path (gradients w.r.t. Q/K/V).
///
/// # Safety
/// Same buffer-sizing rules as [`multi_head_attn_forward`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn multi_head_attn_backward_data(
    handle: &Handle,
    attn: &AttnDescriptor,
    lo_win_idx: &[i32],
    hi_win_idx: &[i32],
    dev_seq_lengths_dqdo: *const i32,
    dev_seq_lengths_dkdv: *const i32,
    do_desc: &SeqDataDescriptor, dout: *const core::ffi::c_void,
    dq_desc: &SeqDataDescriptor, dqueries: *mut core::ffi::c_void,
    queries: *const core::ffi::c_void,
    dk_desc: &SeqDataDescriptor, dkeys: *mut core::ffi::c_void,
    keys: *const core::ffi::c_void,
    dv_desc: &SeqDataDescriptor, dvalues: *mut core::ffi::c_void,
    values: *const core::ffi::c_void,
    weights: &DeviceBuffer<u8>,
    work_space: &mut DeviceBuffer<u8>,
    reserve_space: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cudnn()?;
    let f = c.cudnn_multi_head_attn_backward_data()?;
    check(f(
        handle.handle, attn.desc,
        lo_win_idx.as_ptr(), hi_win_idx.as_ptr(),
        dev_seq_lengths_dqdo, dev_seq_lengths_dkdv,
        do_desc.desc, dout,
        dq_desc.desc, dqueries, queries,
        dk_desc.desc, dkeys, keys,
        dv_desc.desc, dvalues, values,
        weights.byte_size(), weights.as_raw().0 as *const core::ffi::c_void,
        work_space.byte_size(), work_space.as_raw().0 as *mut core::ffi::c_void,
        reserve_space.byte_size(), reserve_space.as_raw().0 as *mut core::ffi::c_void,
    ))
}

/// Multi-head attention backward — weights path (gradient w.r.t. Q/K/V/O
/// projection weights). Pass `add_grad = true` to accumulate into
/// `dweights` (typical for multi-step training).
///
/// # Safety
/// Same as [`multi_head_attn_forward`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn multi_head_attn_backward_weights(
    handle: &Handle,
    attn: &AttnDescriptor,
    add_grad: bool,
    q_desc: &SeqDataDescriptor, queries: *const core::ffi::c_void,
    k_desc: &SeqDataDescriptor, keys: *const core::ffi::c_void,
    v_desc: &SeqDataDescriptor, values: *const core::ffi::c_void,
    do_desc: &SeqDataDescriptor, dout: *const core::ffi::c_void,
    weights: &DeviceBuffer<u8>,
    dweights: &mut DeviceBuffer<u8>,
    work_space: &mut DeviceBuffer<u8>,
    reserve_space: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cudnn()?;
    let f = c.cudnn_multi_head_attn_backward_weights()?;
    check(f(
        handle.handle, attn.desc, add_grad as core::ffi::c_int,
        q_desc.desc, queries,
        k_desc.desc, keys,
        v_desc.desc, values,
        do_desc.desc, dout,
        weights.byte_size(), weights.as_raw().0 as *const core::ffi::c_void,
        dweights.as_raw().0 as *mut core::ffi::c_void,
        work_space.byte_size(), work_space.as_raw().0 as *mut core::ffi::c_void,
        reserve_space.byte_size(), reserve_space.as_raw().0 as *mut core::ffi::c_void,
    ))
}

// ============================================================================
// Tier 4 (cont.) - RNN v8 forward + backward (data + weights)
// ============================================================================

/// Forward pass of an RNN built via [`RnnDescriptor::set_v8`] /
/// [`build_rnn_dynamic`]. Pass `fwd_mode = 0` for inference (no reserve
/// space writes), `1` for training.
///
/// `dev_seq_lengths` is a device array of length `batch_size` giving the
/// valid timestep count per sequence. `hx`/`cx` may be null for an
/// implicit zero initial state; `hy`/`cy` may be null if the caller does
/// not need the final state.
///
/// # Safety
/// Buffer sizes must match what [`rnn_temp_space_sizes`] /
/// [`rnn_weight_space_size`] reported.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rnn_forward(
    handle: &Handle,
    rnn: &RnnDescriptor,
    fwd_mode: i32,
    dev_seq_lengths: *const i32,
    x_desc: &RnnDataDescriptor, x: *const core::ffi::c_void,
    y_desc: &RnnDataDescriptor, y: *mut core::ffi::c_void,
    h_desc: &TensorDescriptor,
    hx: *const core::ffi::c_void,
    hy: *mut core::ffi::c_void,
    c_desc: &TensorDescriptor,
    cx: *const core::ffi::c_void,
    cy: *mut core::ffi::c_void,
    weight_space: &DeviceBuffer<u8>,
    work_space: &mut DeviceBuffer<u8>,
    reserve_space: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cudnn()?;
    let f = c.cudnn_rnn_forward()?;
    check(f(
        handle.handle, rnn.desc, fwd_mode, dev_seq_lengths,
        x_desc.desc, x,
        y_desc.desc, y,
        h_desc.desc, hx, hy,
        c_desc.desc, cx, cy,
        weight_space.byte_size(), weight_space.as_raw().0 as *const core::ffi::c_void,
        work_space.byte_size(),   work_space.as_raw().0 as *mut core::ffi::c_void,
        reserve_space.byte_size(), reserve_space.as_raw().0 as *mut core::ffi::c_void,
    ))
}

/// RNN backward — data path (gradients w.r.t. inputs and initial states).
///
/// # Safety
/// Same buffer-sizing rules as [`rnn_forward`]. The reserve space must
/// be the exact one populated during the matching training-mode
/// `rnn_forward`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rnn_backward_data_v8(
    handle: &Handle,
    rnn: &RnnDescriptor,
    dev_seq_lengths: *const i32,
    y_desc: &RnnDataDescriptor,
    y: *const core::ffi::c_void,
    dy: *const core::ffi::c_void,
    x_desc: &RnnDataDescriptor,
    dx: *mut core::ffi::c_void,
    h_desc: &TensorDescriptor,
    hx: *const core::ffi::c_void,
    dhy: *const core::ffi::c_void,
    dhx: *mut core::ffi::c_void,
    c_desc: &TensorDescriptor,
    cx: *const core::ffi::c_void,
    dcy: *const core::ffi::c_void,
    dcx: *mut core::ffi::c_void,
    weight_space: &DeviceBuffer<u8>,
    work_space: &mut DeviceBuffer<u8>,
    reserve_space: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cudnn()?;
    let f = c.cudnn_rnn_backward_data_v8()?;
    check(f(
        handle.handle, rnn.desc, dev_seq_lengths,
        y_desc.desc, y, dy,
        x_desc.desc, dx,
        h_desc.desc, hx, dhy, dhx,
        c_desc.desc, cx, dcy, dcx,
        weight_space.byte_size(), weight_space.as_raw().0 as *const core::ffi::c_void,
        work_space.byte_size(),   work_space.as_raw().0 as *mut core::ffi::c_void,
        reserve_space.byte_size(), reserve_space.as_raw().0 as *mut core::ffi::c_void,
    ))
}

/// RNN backward — weights path (gradients w.r.t. the weight space).
/// `add_grad = true` accumulates into `dweight_space` (typical for
/// multi-step training); `false` overwrites.
///
/// # Safety
/// Same as [`rnn_forward`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn rnn_backward_weights_v8(
    handle: &Handle,
    rnn: &RnnDescriptor,
    add_grad: bool,
    dev_seq_lengths: *const i32,
    x_desc: &RnnDataDescriptor, x: *const core::ffi::c_void,
    h_desc: &TensorDescriptor,  hx: *const core::ffi::c_void,
    y_desc: &RnnDataDescriptor, y: *const core::ffi::c_void,
    dweight_space: &mut DeviceBuffer<u8>,
    work_space: &mut DeviceBuffer<u8>,
    reserve_space: &mut DeviceBuffer<u8>,
) -> Result<()> {
    let c = cudnn()?;
    let f = c.cudnn_rnn_backward_weights_v8()?;
    check(f(
        handle.handle, rnn.desc, add_grad as core::ffi::c_int, dev_seq_lengths,
        x_desc.desc, x,
        h_desc.desc, hx,
        y_desc.desc, y,
        dweight_space.byte_size(), dweight_space.as_raw().0 as *mut core::ffi::c_void,
        work_space.byte_size(),    work_space.as_raw().0 as *mut core::ffi::c_void,
        reserve_space.byte_size(), reserve_space.as_raw().0 as *mut core::ffi::c_void,
    ))
}
