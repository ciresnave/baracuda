//! Raw FFI + dynamic loader for NVIDIA cuDNN (classic-API subset).
//!
//! Handles the non-standard cuDNN install location on Windows
//! (`C:\Program Files\NVIDIA\CUDNN\v<ver>\bin\<cuda-major>`) by probing
//! it in addition to the usual `baracuda-core::platform` search paths.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use core::ffi::{c_double, c_int, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError};
use baracuda_cuda_sys::runtime::cudaStream_t;
use baracuda_types::CudaStatus;

pub type cudnnHandle_t = *mut c_void;
pub type cudnnTensorDescriptor_t = *mut c_void;
pub type cudnnActivationDescriptor_t = *mut c_void;
pub type cudnnFilterDescriptor_t = *mut c_void;
pub type cudnnConvolutionDescriptor_t = *mut c_void;
pub type cudnnPoolingDescriptor_t = *mut c_void;
pub type cudnnLRNDescriptor_t = *mut c_void;
pub type cudnnOpTensorDescriptor_t = *mut c_void;
pub type cudnnReduceTensorDescriptor_t = *mut c_void;
pub type cudnnDropoutDescriptor_t = *mut c_void;
pub type cudnnCTCLossDescriptor_t = *mut c_void;
pub type cudnnRNNDescriptor_t = *mut c_void;
pub type cudnnRNNDataDescriptor_t = *mut c_void;
pub type cudnnBackendDescriptor_t = *mut c_void;

/// Forward-convolution algorithm selector.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnConvolutionFwdAlgo_t {
    ImplicitGemm = 0,
    ImplicitPrecompGemm = 1,
    Gemm = 2,
    Direct = 3,
    Fft = 4,
    FftTiling = 5,
    Winograd = 6,
    WinogradNonfused = 7,
}

/// Convolution cross-correlation vs true-convolution mode.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnConvolutionMode_t {
    Convolution = 0,
    CrossCorrelation = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnDataType_t {
    Float = 0,
    Double = 1,
    Half = 2,
    Int8 = 3,
    Int32 = 4,
    Int8x4 = 5,
    Uint8 = 6,
    Uint8x4 = 7,
    Int8x32 = 8,
    BFloat16 = 9,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnTensorFormat_t {
    Nchw = 0,
    Nhwc = 1,
    NchwVectC = 2,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnActivationMode_t {
    Sigmoid = 0,
    Relu = 1,
    Tanh = 2,
    ClippedRelu = 3,
    Elu = 4,
    Identity = 5,
    Swish = 6,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnNanPropagation_t {
    NotPropagateNan = 0,
    PropagateNan = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnPoolingMode_t {
    Max = 0,
    AverageCountIncludePadding = 1,
    AverageCountExcludePadding = 2,
    MaxDeterministic = 3,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnSoftmaxAlgorithm_t {
    Fast = 0,
    Accurate = 1,
    Log = 2,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnSoftmaxMode_t {
    Instance = 0,
    Channel = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnBatchNormMode_t {
    PerActivation = 0,
    Spatial = 1,
    SpatialPersistent = 2,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnOpTensorOp_t {
    Add = 0,
    Mul = 1,
    Min = 2,
    Max = 3,
    Sqrt = 4,
    Not = 5,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnReduceTensorOp_t {
    Add = 0,
    Mul = 1,
    Min = 2,
    Max = 3,
    Amax = 4,
    Avg = 5,
    Norm1 = 6,
    Norm2 = 7,
    MulNoZeros = 8,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnReduceTensorIndices_t {
    NoIndices = 0,
    FlattenedIndices = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnIndicesType_t {
    U32 = 0,
    U64 = 1,
    U16 = 2,
    U8 = 3,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnRNNMode_t {
    ReluRnn = 0,
    TanhRnn = 1,
    Lstm = 2,
    Gru = 3,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnDirectionMode_t {
    Unidirectional = 0,
    Bidirectional = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnRNNInputMode_t {
    LinearInput = 0,
    SkipInput = 1,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnRNNAlgo_t {
    Standard = 0,
    PersistStatic = 1,
    PersistDynamic = 2,
    PersistStaticSmallH = 3,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnConvolutionBwdDataAlgo_t {
    Algo0 = 0,
    Algo1 = 1,
    Fft = 2,
    FftTiling = 3,
    Winograd = 4,
    WinogradNonfused = 5,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnConvolutionBwdFilterAlgo_t {
    Algo0 = 0,
    Algo1 = 1,
    Fft = 2,
    Algo3 = 3,
    Winograd = 4,
    WinogradNonfused = 5,
    FftTiling = 6,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnBackendDescriptorType_t {
    PointwiseDescriptor = 0,
    ConvolutionDescriptor = 1,
    EngineDescriptor = 2,
    EngineCfgDescriptor = 3,
    ExecutionPlanDescriptor = 4,
    IntermediateInfoDescriptor = 5,
    KnobChoiceDescriptor = 6,
    KnobInfoDescriptor = 7,
    LayoutInfoDescriptor = 8,
    OperationConvolutionForwardDescriptor = 9,
    OperationConvolutionBackwardFilterDescriptor = 10,
    OperationConvolutionBackwardDataDescriptor = 11,
    OperationPointwiseDescriptor = 12,
    OperationGenStatsDescriptor = 13,
    OperationReductionDescriptor = 14,
    OperationBnFinalizeStatisticsDescriptor = 15,
    OperationGraphDescriptor = 16,
    VariantPackDescriptor = 17,
    TensorDescriptor = 18,
    MatmulDescriptor = 19,
    OperationMatmulDescriptor = 20,
    OperationBnBwdWeightsDescriptor = 21,
    ResampleDescriptor = 22,
    OperationResampleFwdDescriptor = 23,
    OperationResampleBwdDescriptor = 24,
    OperationConcatDescriptor = 25,
    OperationSignalDescriptor = 26,
    OperationNormForwardDescriptor = 27,
    OperationNormBackwardDescriptor = 28,
    OperationRngDescriptor = 30,
    RngDescriptor = 31,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnBackendAttributeName_t {
    // Just a representative subset — the real enum has ~200 entries.
    PointwiseMode = 0,
    PointwiseMathPrec = 1,
    PointwiseNanPropagation = 2,
    PointwiseReluLowerClip = 3,
    PointwiseReluUpperClip = 4,
    PointwiseEluAlpha = 5,
    // Tensor descriptor
    TensorUniqueId = 100,
    TensorDataType = 101,
    TensorByteAlignment = 102,
    TensorDimensions = 103,
    TensorStrides = 104,
    // Convolution descriptor
    ConvolutionCompType = 200,
    ConvolutionConvMode = 201,
    ConvolutionDilations = 202,
    ConvolutionFilterStrides = 203,
    ConvolutionPrePaddings = 204,
    ConvolutionPostPaddings = 205,
    ConvolutionSpatialDims = 206,
    // Operation graph
    OperationGraphHandle = 500,
    OperationGraphOps = 501,
    // Execution plan
    ExecutionPlanHandle = 600,
    ExecutionPlanEngineConfig = 601,
    ExecutionPlanWorkspaceSize = 602,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnBackendAttributeType_t {
    Handle = 0,
    DataType = 1,
    Boolean = 2,
    Int64 = 3,
    FloatValue = 4,
    DoubleValue = 5,
    PointwiseMode = 6,
    ConvolutionMode = 7,
    HeurMode = 8,
    KnobType = 9,
    NanPropagation = 10,
    NumericalNote = 11,
    LayoutType = 12,
    AttribName = 13,
    PointerT = 14,
    BackendDescriptor = 15,
    GenstatsMode = 16,
    BnFinalizeStatsMode = 17,
    ReductionOperatorType = 18,
    BehaviorNote = 19,
    TensorReorderingMode = 20,
    ResampleMode = 21,
    PaddingMode = 22,
    IntArray = 23,
    RngDistribution = 24,
    NormMode = 25,
    NormFwdPhase = 26,
    RngNormal = 27,
    RngUniform = 28,
}

// ---- status ---------------------------------------------------------------

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct cudnnStatus_t(pub i32);

impl cudnnStatus_t {
    pub const SUCCESS: Self = Self(0);
    pub const NOT_INITIALIZED: Self = Self(1);
    pub const ALLOC_FAILED: Self = Self(2);
    pub const BAD_PARAM: Self = Self(3);
    pub const INTERNAL_ERROR: Self = Self(4);
    pub const INVALID_VALUE: Self = Self(5);
    pub const ARCH_MISMATCH: Self = Self(6);
    pub const MAPPING_ERROR: Self = Self(7);
    pub const EXECUTION_FAILED: Self = Self(8);
    pub const NOT_SUPPORTED: Self = Self(9);
    pub const LICENSE_ERROR: Self = Self(10);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for cudnnStatus_t {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "CUDNN_STATUS_SUCCESS",
            1 => "CUDNN_STATUS_NOT_INITIALIZED",
            2 => "CUDNN_STATUS_ALLOC_FAILED",
            3 => "CUDNN_STATUS_BAD_PARAM",
            4 => "CUDNN_STATUS_INTERNAL_ERROR",
            5 => "CUDNN_STATUS_INVALID_VALUE",
            6 => "CUDNN_STATUS_ARCH_MISMATCH",
            8 => "CUDNN_STATUS_EXECUTION_FAILED",
            9 => "CUDNN_STATUS_NOT_SUPPORTED",
            _ => "CUDNN_STATUS_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            1 => "cuDNN not initialized",
            3 => "bad parameter",
            9 => "operation not supported on this device/version",
            _ => "unrecognized cuDNN status code",
        }
    }
    fn is_success(self) -> bool {
        cudnnStatus_t::is_success(self)
    }
    fn library(self) -> &'static str {
        "cudnn"
    }
}

// ---- function-pointer types ----------------------------------------------

pub type PFN_cudnnCreate = unsafe extern "C" fn(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroy = unsafe extern "C" fn(handle: cudnnHandle_t) -> cudnnStatus_t;
pub type PFN_cudnnSetStream =
    unsafe extern "C" fn(handle: cudnnHandle_t, stream: cudaStream_t) -> cudnnStatus_t;
pub type PFN_cudnnGetVersion = unsafe extern "C" fn() -> usize;

pub type PFN_cudnnCreateTensorDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyTensorDescriptor =
    unsafe extern "C" fn(desc: cudnnTensorDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnSetTensor4dDescriptor = unsafe extern "C" fn(
    desc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    data_type: cudnnDataType_t,
    n: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

pub type PFN_cudnnCreateActivationDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnActivationDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyActivationDescriptor =
    unsafe extern "C" fn(desc: cudnnActivationDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnSetActivationDescriptor = unsafe extern "C" fn(
    desc: cudnnActivationDescriptor_t,
    mode: cudnnActivationMode_t,
    nan_prop: cudnnNanPropagation_t,
    coef: c_double,
) -> cudnnStatus_t;

pub type PFN_cudnnActivationForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    activation_desc: cudnnActivationDescriptor_t,
    alpha: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    beta: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
) -> cudnnStatus_t;

// ---- convolution ----------------------------------------------------------

pub type PFN_cudnnCreateFilterDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyFilterDescriptor =
    unsafe extern "C" fn(desc: cudnnFilterDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnSetFilter4dDescriptor = unsafe extern "C" fn(
    desc: cudnnFilterDescriptor_t,
    data_type: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    k: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

pub type PFN_cudnnCreateConvolutionDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyConvolutionDescriptor =
    unsafe extern "C" fn(desc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnSetConvolution2dDescriptor = unsafe extern "C" fn(
    desc: cudnnConvolutionDescriptor_t,
    pad_h: c_int,
    pad_w: c_int,
    u: c_int,
    v: c_int,
    dilation_h: c_int,
    dilation_w: c_int,
    mode: cudnnConvolutionMode_t,
    compute_type: cudnnDataType_t,
) -> cudnnStatus_t;

pub type PFN_cudnnGetConvolution2dForwardOutputDim = unsafe extern "C" fn(
    conv_desc: cudnnConvolutionDescriptor_t,
    input_desc: cudnnTensorDescriptor_t,
    filter_desc: cudnnFilterDescriptor_t,
    n: *mut c_int,
    c: *mut c_int,
    h: *mut c_int,
    w: *mut c_int,
) -> cudnnStatus_t;

pub type PFN_cudnnGetConvolutionForwardWorkspaceSize = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    x_desc: cudnnTensorDescriptor_t,
    w_desc: cudnnFilterDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t;

pub type PFN_cudnnConvolutionForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    w_desc: cudnnFilterDescriptor_t,
    w: *const c_void,
    conv_desc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    workspace: *mut c_void,
    workspace_size: usize,
    beta: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnConvolutionBackwardData = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const c_void,
    w_desc: cudnnFilterDescriptor_t,
    w: *const c_void,
    dy_desc: cudnnTensorDescriptor_t,
    dy: *const c_void,
    conv_desc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    workspace: *mut c_void,
    workspace_size: usize,
    beta: *const c_void,
    dx_desc: cudnnTensorDescriptor_t,
    dx: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnConvolutionBackwardFilter = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    dy_desc: cudnnTensorDescriptor_t,
    dy: *const c_void,
    conv_desc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    workspace: *mut c_void,
    workspace_size: usize,
    beta: *const c_void,
    dw_desc: cudnnFilterDescriptor_t,
    dw: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnConvolutionBackwardBias = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const c_void,
    dy_desc: cudnnTensorDescriptor_t,
    dy: *const c_void,
    beta: *const c_void,
    db_desc: cudnnTensorDescriptor_t,
    db: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnGetConvolutionBackwardDataWorkspaceSize = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    w_desc: cudnnFilterDescriptor_t,
    dy_desc: cudnnTensorDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    dx_desc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t;

pub type PFN_cudnnGetConvolutionBackwardFilterWorkspaceSize = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    x_desc: cudnnTensorDescriptor_t,
    dy_desc: cudnnTensorDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    dw_desc: cudnnFilterDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t;

// ---- pooling --------------------------------------------------------------

pub type PFN_cudnnCreatePoolingDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnPoolingDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyPoolingDescriptor =
    unsafe extern "C" fn(desc: cudnnPoolingDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnSetPooling2dDescriptor = unsafe extern "C" fn(
    desc: cudnnPoolingDescriptor_t,
    mode: cudnnPoolingMode_t,
    nan_prop: cudnnNanPropagation_t,
    window_h: c_int,
    window_w: c_int,
    vertical_padding: c_int,
    horizontal_padding: c_int,
    vertical_stride: c_int,
    horizontal_stride: c_int,
) -> cudnnStatus_t;
pub type PFN_cudnnPoolingForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    pool_desc: cudnnPoolingDescriptor_t,
    alpha: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    beta: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
) -> cudnnStatus_t;
pub type PFN_cudnnPoolingBackward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    pool_desc: cudnnPoolingDescriptor_t,
    alpha: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *const c_void,
    dy_desc: cudnnTensorDescriptor_t,
    dy: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    beta: *const c_void,
    dx_desc: cudnnTensorDescriptor_t,
    dx: *mut c_void,
) -> cudnnStatus_t;

// ---- softmax --------------------------------------------------------------

pub type PFN_cudnnSoftmaxForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    algo: cudnnSoftmaxAlgorithm_t,
    mode: cudnnSoftmaxMode_t,
    alpha: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    beta: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnSoftmaxBackward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    algo: cudnnSoftmaxAlgorithm_t,
    mode: cudnnSoftmaxMode_t,
    alpha: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *const c_void,
    dy_desc: cudnnTensorDescriptor_t,
    dy: *const c_void,
    beta: *const c_void,
    dx_desc: cudnnTensorDescriptor_t,
    dx: *mut c_void,
) -> cudnnStatus_t;

// ---- batch normalization --------------------------------------------------

pub type PFN_cudnnBatchNormalizationForwardInference = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha: *const c_void,
    beta: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
    bn_scale_bias_mean_var_desc: cudnnTensorDescriptor_t,
    bn_scale: *const c_void,
    bn_bias: *const c_void,
    estimated_mean: *const c_void,
    estimated_variance: *const c_void,
    epsilon: c_double,
) -> cudnnStatus_t;

pub type PFN_cudnnBatchNormalizationForwardTraining = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha: *const c_void,
    beta: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
    bn_scale_bias_mean_var_desc: cudnnTensorDescriptor_t,
    bn_scale: *const c_void,
    bn_bias: *const c_void,
    exponential_average_factor: c_double,
    result_running_mean: *mut c_void,
    result_running_variance: *mut c_void,
    epsilon: c_double,
    result_save_mean: *mut c_void,
    result_save_inv_variance: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnBatchNormalizationBackward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    alpha_data_diff: *const c_void,
    beta_data_diff: *const c_void,
    alpha_param_diff: *const c_void,
    beta_param_diff: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    dy_desc: cudnnTensorDescriptor_t,
    dy: *const c_void,
    dx_desc: cudnnTensorDescriptor_t,
    dx: *mut c_void,
    bn_scale_bias_diff_desc: cudnnTensorDescriptor_t,
    bn_scale: *const c_void,
    bn_scale_result: *mut c_void,
    bn_bias_result: *mut c_void,
    epsilon: c_double,
    saved_mean: *const c_void,
    saved_inv_variance: *const c_void,
) -> cudnnStatus_t;

// ---- op-tensor / reduce / transform --------------------------------------

pub type PFN_cudnnCreateOpTensorDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnOpTensorDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyOpTensorDescriptor =
    unsafe extern "C" fn(desc: cudnnOpTensorDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnSetOpTensorDescriptor = unsafe extern "C" fn(
    desc: cudnnOpTensorDescriptor_t,
    op: cudnnOpTensorOp_t,
    compute_type: cudnnDataType_t,
    nan_prop: cudnnNanPropagation_t,
) -> cudnnStatus_t;
pub type PFN_cudnnOpTensor = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    desc: cudnnOpTensorDescriptor_t,
    alpha1: *const c_void,
    a_desc: cudnnTensorDescriptor_t,
    a: *const c_void,
    alpha2: *const c_void,
    b_desc: cudnnTensorDescriptor_t,
    b: *const c_void,
    beta: *const c_void,
    c_desc: cudnnTensorDescriptor_t,
    c: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnCreateReduceTensorDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyReduceTensorDescriptor =
    unsafe extern "C" fn(desc: cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnSetReduceTensorDescriptor = unsafe extern "C" fn(
    desc: cudnnReduceTensorDescriptor_t,
    op: cudnnReduceTensorOp_t,
    compute_type: cudnnDataType_t,
    nan_prop: cudnnNanPropagation_t,
    indices: cudnnReduceTensorIndices_t,
    indices_type: cudnnIndicesType_t,
) -> cudnnStatus_t;
pub type PFN_cudnnGetReductionWorkspaceSize = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    desc: cudnnReduceTensorDescriptor_t,
    a_desc: cudnnTensorDescriptor_t,
    c_desc: cudnnTensorDescriptor_t,
    workspace_size: *mut usize,
) -> cudnnStatus_t;
pub type PFN_cudnnReduceTensor = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    desc: cudnnReduceTensorDescriptor_t,
    indices: *mut c_void,
    indices_size: usize,
    workspace: *mut c_void,
    workspace_size: usize,
    alpha: *const c_void,
    a_desc: cudnnTensorDescriptor_t,
    a: *const c_void,
    beta: *const c_void,
    c_desc: cudnnTensorDescriptor_t,
    c: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnAddTensor = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const c_void,
    a_desc: cudnnTensorDescriptor_t,
    a: *const c_void,
    beta: *const c_void,
    c_desc: cudnnTensorDescriptor_t,
    c: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnTransformTensor = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    beta: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnScaleTensor = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
    alpha: *const c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnSetTensor = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
    value_ptr: *const c_void,
) -> cudnnStatus_t;

// ---- LRN ------------------------------------------------------------------

pub type PFN_cudnnCreateLRNDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnLRNDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyLRNDescriptor =
    unsafe extern "C" fn(desc: cudnnLRNDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnSetLRNDescriptor = unsafe extern "C" fn(
    desc: cudnnLRNDescriptor_t,
    lrn_n: c_int,
    lrn_alpha: c_double,
    lrn_beta: c_double,
    lrn_k: c_double,
) -> cudnnStatus_t;
pub type PFN_cudnnLRNCrossChannelForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    lrn_desc: cudnnLRNDescriptor_t,
    lrn_mode: c_int,
    alpha: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    beta: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
) -> cudnnStatus_t;

// ---- dropout --------------------------------------------------------------

pub type PFN_cudnnCreateDropoutDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnDropoutDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyDropoutDescriptor =
    unsafe extern "C" fn(desc: cudnnDropoutDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDropoutGetStatesSize =
    unsafe extern "C" fn(handle: cudnnHandle_t, size_in_bytes: *mut usize) -> cudnnStatus_t;
pub type PFN_cudnnDropoutGetReserveSpaceSize = unsafe extern "C" fn(
    x_desc: cudnnTensorDescriptor_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t;
pub type PFN_cudnnSetDropoutDescriptor = unsafe extern "C" fn(
    desc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: f32,
    states: *mut c_void,
    state_size: usize,
    seed: u64,
) -> cudnnStatus_t;
pub type PFN_cudnnDropoutForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    desc: cudnnDropoutDescriptor_t,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
    reserve_space: *mut c_void,
    reserve_space_size: usize,
) -> cudnnStatus_t;
pub type PFN_cudnnDropoutBackward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    desc: cudnnDropoutDescriptor_t,
    dy_desc: cudnnTensorDescriptor_t,
    dy: *const c_void,
    dx_desc: cudnnTensorDescriptor_t,
    dx: *mut c_void,
    reserve_space: *mut c_void,
    reserve_space_size: usize,
) -> cudnnStatus_t;

// ---- RNN ------------------------------------------------------------------

pub type PFN_cudnnCreateRNNDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnRNNDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyRNNDescriptor =
    unsafe extern "C" fn(desc: cudnnRNNDescriptor_t) -> cudnnStatus_t;

pub type PFN_cudnnCreateRNNDataDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnRNNDataDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyRNNDataDescriptor =
    unsafe extern "C" fn(desc: cudnnRNNDataDescriptor_t) -> cudnnStatus_t;

pub type PFN_cudnnRNNForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnn_desc: cudnnRNNDescriptor_t,
    fwd_mode: c_int,
    dev_seq_lengths: *const i32,
    x_desc: cudnnRNNDataDescriptor_t,
    x: *const c_void,
    y_desc: cudnnRNNDataDescriptor_t,
    y: *mut c_void,
    h_desc: cudnnTensorDescriptor_t,
    hx: *const c_void,
    hy: *mut c_void,
    c_desc: cudnnTensorDescriptor_t,
    cx: *const c_void,
    cy: *mut c_void,
    weight_space_size: usize,
    weight_space: *const c_void,
    work_space_size: usize,
    work_space: *mut c_void,
    reserve_space_size: usize,
    reserve_space: *mut c_void,
) -> cudnnStatus_t;

// ---- cuDNN backend (Graph) API -------------------------------------------

pub type PFN_cudnnBackendCreateDescriptor = unsafe extern "C" fn(
    descriptor_type: cudnnBackendDescriptorType_t,
    descriptor: *mut cudnnBackendDescriptor_t,
) -> cudnnStatus_t;
pub type PFN_cudnnBackendDestroyDescriptor =
    unsafe extern "C" fn(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnBackendInitialize =
    unsafe extern "C" fn(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnBackendFinalize =
    unsafe extern "C" fn(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnBackendSetAttribute = unsafe extern "C" fn(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    element_count: i64,
    array_of_elements: *const c_void,
) -> cudnnStatus_t;
pub type PFN_cudnnBackendGetAttribute = unsafe extern "C" fn(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    requested_element_count: i64,
    element_count: *mut i64,
    array_of_elements: *mut c_void,
) -> cudnnStatus_t;
pub type PFN_cudnnBackendExecute = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    execution_plan: cudnnBackendDescriptor_t,
    variant_pack: cudnnBackendDescriptor_t,
) -> cudnnStatus_t;

// ---- error-string helper -------------------------------------------------

pub type PFN_cudnnGetErrorString =
    unsafe extern "C" fn(status: cudnnStatus_t) -> *const core::ffi::c_char;

// ---- N-dimensional tensor / filter descriptors --------------------------

pub type PFN_cudnnSetTensorNdDescriptor = unsafe extern "C" fn(
    desc: cudnnTensorDescriptor_t,
    data_type: cudnnDataType_t,
    nb_dims: c_int,
    dim_a: *const c_int,
    stride_a: *const c_int,
) -> cudnnStatus_t;

pub type PFN_cudnnGetTensorNdDescriptor = unsafe extern "C" fn(
    desc: cudnnTensorDescriptor_t,
    nb_dims_requested: c_int,
    data_type: *mut cudnnDataType_t,
    nb_dims: *mut c_int,
    dim_a: *mut c_int,
    stride_a: *mut c_int,
) -> cudnnStatus_t;

pub type PFN_cudnnSetFilterNdDescriptor = unsafe extern "C" fn(
    desc: cudnnFilterDescriptor_t,
    data_type: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    nb_dims: c_int,
    filter_dim_a: *const c_int,
) -> cudnnStatus_t;

pub type PFN_cudnnSetConvolutionNdDescriptor = unsafe extern "C" fn(
    desc: cudnnConvolutionDescriptor_t,
    array_length: c_int,
    pad_a: *const c_int,
    filter_stride_a: *const c_int,
    dilation_a: *const c_int,
    mode: cudnnConvolutionMode_t,
    compute_type: cudnnDataType_t,
) -> cudnnStatus_t;

pub type PFN_cudnnSetPoolingNdDescriptor = unsafe extern "C" fn(
    desc: cudnnPoolingDescriptor_t,
    mode: cudnnPoolingMode_t,
    nan_prop: cudnnNanPropagation_t,
    nb_dims: c_int,
    window_dim_a: *const c_int,
    padding_a: *const c_int,
    stride_a: *const c_int,
) -> cudnnStatus_t;

// ---- CTC loss ------------------------------------------------------------

pub type PFN_cudnnCreateCTCLossDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnCTCLossDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroyCTCLossDescriptor =
    unsafe extern "C" fn(desc: cudnnCTCLossDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnSetCTCLossDescriptor = unsafe extern "C" fn(
    desc: cudnnCTCLossDescriptor_t,
    compute_type: cudnnDataType_t,
) -> cudnnStatus_t;

pub type PFN_cudnnGetCTCLossWorkspaceSize = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    probs_desc: cudnnTensorDescriptor_t,
    gradients_desc: cudnnTensorDescriptor_t,
    labels: *const c_int,
    label_lengths: *const c_int,
    input_lengths: *const c_int,
    ctc_loss_algo: c_int,
    ctc_loss_desc: cudnnCTCLossDescriptor_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t;

pub type PFN_cudnnCTCLoss = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    probs_desc: cudnnTensorDescriptor_t,
    probs: *const c_void,
    labels: *const c_int,
    label_lengths: *const c_int,
    input_lengths: *const c_int,
    costs: *mut c_void,
    gradients_desc: cudnnTensorDescriptor_t,
    gradients: *mut c_void,
    ctc_loss_algo: c_int,
    ctc_loss_desc: cudnnCTCLossDescriptor_t,
    workspace: *mut c_void,
    workspace_size: usize,
) -> cudnnStatus_t;

// ---- RNN backward --------------------------------------------------------

pub type PFN_cudnnRNNBackwardData_v8 = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnn_desc: cudnnRNNDescriptor_t,
    dev_seq_lengths: *const i32,
    y_desc: cudnnRNNDataDescriptor_t,
    y: *const c_void,
    dy: *const c_void,
    x_desc: cudnnRNNDataDescriptor_t,
    dx: *mut c_void,
    h_desc: cudnnTensorDescriptor_t,
    hx: *const c_void,
    dhy: *const c_void,
    dhx: *mut c_void,
    c_desc: cudnnTensorDescriptor_t,
    cx: *const c_void,
    dcy: *const c_void,
    dcx: *mut c_void,
    weight_space_size: usize,
    weight_space: *const c_void,
    work_space_size: usize,
    work_space: *mut c_void,
    reserve_space_size: usize,
    reserve_space: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnRNNBackwardWeights_v8 = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnn_desc: cudnnRNNDescriptor_t,
    add_grad: c_int,
    dev_seq_lengths: *const i32,
    x_desc: cudnnRNNDataDescriptor_t,
    x: *const c_void,
    h_desc: cudnnTensorDescriptor_t,
    hx: *const c_void,
    y_desc: cudnnRNNDataDescriptor_t,
    y: *const c_void,
    weight_space_size: usize,
    dweight_space: *mut c_void,
    work_space_size: usize,
    work_space: *mut c_void,
    reserve_space_size: usize,
    reserve_space: *mut c_void,
) -> cudnnStatus_t;

// ---- Spatial transformer --------------------------------------------------

pub type cudnnSpatialTransformerDescriptor_t = *mut c_void;

pub type PFN_cudnnCreateSpatialTransformerDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnSpatialTransformerDescriptor_t) -> cudnnStatus_t;
pub type PFN_cudnnDestroySpatialTransformerDescriptor =
    unsafe extern "C" fn(desc: cudnnSpatialTransformerDescriptor_t) -> cudnnStatus_t;

pub type PFN_cudnnSetSpatialTransformerNdDescriptor = unsafe extern "C" fn(
    desc: cudnnSpatialTransformerDescriptor_t,
    sampler_type: c_int,
    data_type: cudnnDataType_t,
    nb_dims: c_int,
    dim_a: *const c_int,
) -> cudnnStatus_t;

pub type PFN_cudnnSpatialTfGridGeneratorForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    st_desc: cudnnSpatialTransformerDescriptor_t,
    theta: *const c_void,
    grid: *mut c_void,
) -> cudnnStatus_t;

pub type PFN_cudnnSpatialTfSamplerForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    st_desc: cudnnSpatialTransformerDescriptor_t,
    alpha: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    grid: *const c_void,
    beta: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
) -> cudnnStatus_t;

// ---- loader --------------------------------------------------------------

/// cuDNN's install layout is non-standard — on Windows the DLLs live at
/// `C:\Program Files\NVIDIA\CUDNN\v<ver>\bin\<cuda_major>\` and are not on
/// the default DLL search path. Probe the common locations.
fn cudnn_candidates() -> Vec<String> {
    #[cfg(target_os = "linux")]
    {
        vec![
            "libcudnn.so.9".to_string(),
            "libcudnn.so.8".to_string(),
            "libcudnn.so".to_string(),
        ]
    }
    #[cfg(target_os = "windows")]
    {
        vec!["cudnn64_9.dll".to_string(), "cudnn64_8.dll".to_string()]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        Vec::new()
    }
}

fn cudnn_extra_search_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();

    if let Ok(p) = std::env::var("CUDNN_PATH") {
        let base = PathBuf::from(p);
        if cfg!(target_os = "windows") {
            out.push(base.join("bin"));
        } else {
            out.push(base.join("lib"));
            out.push(base.join("lib64"));
        }
    }

    if cfg!(target_os = "windows") {
        // Default Windows install: `C:\Program Files\NVIDIA\CUDNN\v*\bin\<cuda_major>`.
        if let Ok(pf) = std::env::var("ProgramFiles") {
            let cudnn_root = PathBuf::from(pf).join("NVIDIA").join("CUDNN");
            if let Ok(read_dir) = std::fs::read_dir(&cudnn_root) {
                for entry in read_dir.flatten() {
                    let p = entry.path();
                    if p.is_dir() {
                        let bin = p.join("bin");
                        if let Ok(sub) = std::fs::read_dir(&bin) {
                            for s in sub.flatten() {
                                let sp = s.path();
                                if sp.is_dir() {
                                    out.push(sp);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    out
}

/// cuDNN 9's main DLL (`cudnn64_9.dll`) is a facade that depends on several
/// companion DLLs in the same directory (`cudnn_ops64_9.dll`,
/// `cudnn_graph64_9.dll`, …). Windows resolves those dependencies via the
/// DLL search path, so we must ensure the cuDNN bin directory is on PATH
/// before `libloading` calls `LoadLibraryExW`.
#[cfg(target_os = "windows")]
fn ensure_cudnn_on_path(extra_dirs: &[PathBuf]) {
    use std::sync::OnceLock;
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        let existing = std::env::var("PATH").unwrap_or_default();
        let mut prefix = String::new();
        for dir in extra_dirs {
            if let Some(s) = dir.to_str() {
                if !existing.split(';').any(|p| p == s) {
                    if !prefix.is_empty() {
                        prefix.push(';');
                    }
                    prefix.push_str(s);
                }
            }
        }
        if !prefix.is_empty() {
            let new_path = if existing.is_empty() {
                prefix
            } else {
                format!("{prefix};{existing}")
            };
            std::env::set_var("PATH", new_path);
        }
    });
}

#[cfg(not(target_os = "windows"))]
fn ensure_cudnn_on_path(_extra_dirs: &[PathBuf]) {}

/// Open libcudnn across the usual baracuda search paths plus cuDNN-specific ones.
fn open_cudnn() -> Result<Library, LoaderError> {
    let candidates: Vec<&'static str> = cudnn_candidates()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    let candidates_leaked: &'static [&'static str] = Box::leak(candidates.into_boxed_slice());

    // Make sure cuDNN-specific directories are on PATH so the main DLL can
    // find its companion DLLs via Windows' dependency resolver.
    let extra = cudnn_extra_search_dirs();
    ensure_cudnn_on_path(&extra);

    // First try the standard baracuda search (now augmented on PATH).
    if let Ok(lib) = Library::open("cudnn", candidates_leaked) {
        return Ok(lib);
    }

    // Then try cuDNN-specific directories explicitly.
    for dir in &extra {
        for candidate in candidates_leaked {
            let full = dir.join(candidate);
            if let Ok(lib) = Library::open_at("cudnn", &full) {
                return Ok(lib);
            }
        }
    }

    Err(LoaderError::library_not_found_with_search(
        "cudnn",
        candidates_leaked,
        extra.len(),
    ))
}

macro_rules! cudnn_fns {
    ($($name:ident as $sym:literal : $pfn:ty);* $(;)?) => {
        pub struct Cudnn {
            lib: Library,
            $($name: OnceLock<$pfn>,)*
        }
        impl core::fmt::Debug for Cudnn {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Cudnn").field("lib", &self.lib).finish_non_exhaustive()
            }
        }
        impl Cudnn {
            $(
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
        }
    };
}

cudnn_fns! {
    // Handle + version
    cudnn_create as "cudnnCreate": PFN_cudnnCreate;
    cudnn_destroy as "cudnnDestroy": PFN_cudnnDestroy;
    cudnn_set_stream as "cudnnSetStream": PFN_cudnnSetStream;
    cudnn_get_version as "cudnnGetVersion": PFN_cudnnGetVersion;
    cudnn_get_error_string as "cudnnGetErrorString": PFN_cudnnGetErrorString;
    // Tensor descriptor
    cudnn_create_tensor_descriptor as "cudnnCreateTensorDescriptor": PFN_cudnnCreateTensorDescriptor;
    cudnn_destroy_tensor_descriptor as "cudnnDestroyTensorDescriptor": PFN_cudnnDestroyTensorDescriptor;
    cudnn_set_tensor_4d_descriptor as "cudnnSetTensor4dDescriptor": PFN_cudnnSetTensor4dDescriptor;
    // Activation
    cudnn_create_activation_descriptor as "cudnnCreateActivationDescriptor": PFN_cudnnCreateActivationDescriptor;
    cudnn_destroy_activation_descriptor as "cudnnDestroyActivationDescriptor": PFN_cudnnDestroyActivationDescriptor;
    cudnn_set_activation_descriptor as "cudnnSetActivationDescriptor": PFN_cudnnSetActivationDescriptor;
    cudnn_activation_forward as "cudnnActivationForward": PFN_cudnnActivationForward;
    // Convolution
    cudnn_create_filter_descriptor as "cudnnCreateFilterDescriptor": PFN_cudnnCreateFilterDescriptor;
    cudnn_destroy_filter_descriptor as "cudnnDestroyFilterDescriptor": PFN_cudnnDestroyFilterDescriptor;
    cudnn_set_filter_4d_descriptor as "cudnnSetFilter4dDescriptor": PFN_cudnnSetFilter4dDescriptor;
    cudnn_create_convolution_descriptor as "cudnnCreateConvolutionDescriptor": PFN_cudnnCreateConvolutionDescriptor;
    cudnn_destroy_convolution_descriptor as "cudnnDestroyConvolutionDescriptor": PFN_cudnnDestroyConvolutionDescriptor;
    cudnn_set_convolution_2d_descriptor as "cudnnSetConvolution2dDescriptor": PFN_cudnnSetConvolution2dDescriptor;
    cudnn_get_convolution_2d_forward_output_dim as "cudnnGetConvolution2dForwardOutputDim": PFN_cudnnGetConvolution2dForwardOutputDim;
    cudnn_get_convolution_forward_workspace_size as "cudnnGetConvolutionForwardWorkspaceSize": PFN_cudnnGetConvolutionForwardWorkspaceSize;
    cudnn_convolution_forward as "cudnnConvolutionForward": PFN_cudnnConvolutionForward;
    cudnn_convolution_backward_data as "cudnnConvolutionBackwardData": PFN_cudnnConvolutionBackwardData;
    cudnn_convolution_backward_filter as "cudnnConvolutionBackwardFilter": PFN_cudnnConvolutionBackwardFilter;
    cudnn_convolution_backward_bias as "cudnnConvolutionBackwardBias": PFN_cudnnConvolutionBackwardBias;
    cudnn_get_convolution_backward_data_workspace_size as "cudnnGetConvolutionBackwardDataWorkspaceSize": PFN_cudnnGetConvolutionBackwardDataWorkspaceSize;
    cudnn_get_convolution_backward_filter_workspace_size as "cudnnGetConvolutionBackwardFilterWorkspaceSize": PFN_cudnnGetConvolutionBackwardFilterWorkspaceSize;
    // Pooling
    cudnn_create_pooling_descriptor as "cudnnCreatePoolingDescriptor": PFN_cudnnCreatePoolingDescriptor;
    cudnn_destroy_pooling_descriptor as "cudnnDestroyPoolingDescriptor": PFN_cudnnDestroyPoolingDescriptor;
    cudnn_set_pooling_2d_descriptor as "cudnnSetPooling2dDescriptor": PFN_cudnnSetPooling2dDescriptor;
    cudnn_pooling_forward as "cudnnPoolingForward": PFN_cudnnPoolingForward;
    cudnn_pooling_backward as "cudnnPoolingBackward": PFN_cudnnPoolingBackward;
    // Softmax
    cudnn_softmax_forward as "cudnnSoftmaxForward": PFN_cudnnSoftmaxForward;
    cudnn_softmax_backward as "cudnnSoftmaxBackward": PFN_cudnnSoftmaxBackward;
    // BatchNorm
    cudnn_batch_normalization_forward_inference as "cudnnBatchNormalizationForwardInference": PFN_cudnnBatchNormalizationForwardInference;
    cudnn_batch_normalization_forward_training as "cudnnBatchNormalizationForwardTraining": PFN_cudnnBatchNormalizationForwardTraining;
    cudnn_batch_normalization_backward as "cudnnBatchNormalizationBackward": PFN_cudnnBatchNormalizationBackward;
    // Op-tensor / reduce / transform
    cudnn_create_op_tensor_descriptor as "cudnnCreateOpTensorDescriptor": PFN_cudnnCreateOpTensorDescriptor;
    cudnn_destroy_op_tensor_descriptor as "cudnnDestroyOpTensorDescriptor": PFN_cudnnDestroyOpTensorDescriptor;
    cudnn_set_op_tensor_descriptor as "cudnnSetOpTensorDescriptor": PFN_cudnnSetOpTensorDescriptor;
    cudnn_op_tensor as "cudnnOpTensor": PFN_cudnnOpTensor;
    cudnn_create_reduce_tensor_descriptor as "cudnnCreateReduceTensorDescriptor": PFN_cudnnCreateReduceTensorDescriptor;
    cudnn_destroy_reduce_tensor_descriptor as "cudnnDestroyReduceTensorDescriptor": PFN_cudnnDestroyReduceTensorDescriptor;
    cudnn_set_reduce_tensor_descriptor as "cudnnSetReduceTensorDescriptor": PFN_cudnnSetReduceTensorDescriptor;
    cudnn_get_reduction_workspace_size as "cudnnGetReductionWorkspaceSize": PFN_cudnnGetReductionWorkspaceSize;
    cudnn_reduce_tensor as "cudnnReduceTensor": PFN_cudnnReduceTensor;
    cudnn_add_tensor as "cudnnAddTensor": PFN_cudnnAddTensor;
    cudnn_transform_tensor as "cudnnTransformTensor": PFN_cudnnTransformTensor;
    cudnn_scale_tensor as "cudnnScaleTensor": PFN_cudnnScaleTensor;
    cudnn_set_tensor as "cudnnSetTensor": PFN_cudnnSetTensor;
    // LRN
    cudnn_create_lrn_descriptor as "cudnnCreateLRNDescriptor": PFN_cudnnCreateLRNDescriptor;
    cudnn_destroy_lrn_descriptor as "cudnnDestroyLRNDescriptor": PFN_cudnnDestroyLRNDescriptor;
    cudnn_set_lrn_descriptor as "cudnnSetLRNDescriptor": PFN_cudnnSetLRNDescriptor;
    cudnn_lrn_cross_channel_forward as "cudnnLRNCrossChannelForward": PFN_cudnnLRNCrossChannelForward;
    // Dropout
    cudnn_create_dropout_descriptor as "cudnnCreateDropoutDescriptor": PFN_cudnnCreateDropoutDescriptor;
    cudnn_destroy_dropout_descriptor as "cudnnDestroyDropoutDescriptor": PFN_cudnnDestroyDropoutDescriptor;
    cudnn_dropout_get_states_size as "cudnnDropoutGetStatesSize": PFN_cudnnDropoutGetStatesSize;
    cudnn_dropout_get_reserve_space_size as "cudnnDropoutGetReserveSpaceSize": PFN_cudnnDropoutGetReserveSpaceSize;
    cudnn_set_dropout_descriptor as "cudnnSetDropoutDescriptor": PFN_cudnnSetDropoutDescriptor;
    cudnn_dropout_forward as "cudnnDropoutForward": PFN_cudnnDropoutForward;
    cudnn_dropout_backward as "cudnnDropoutBackward": PFN_cudnnDropoutBackward;
    // RNN
    cudnn_create_rnn_descriptor as "cudnnCreateRNNDescriptor": PFN_cudnnCreateRNNDescriptor;
    cudnn_destroy_rnn_descriptor as "cudnnDestroyRNNDescriptor": PFN_cudnnDestroyRNNDescriptor;
    cudnn_create_rnn_data_descriptor as "cudnnCreateRNNDataDescriptor": PFN_cudnnCreateRNNDataDescriptor;
    cudnn_destroy_rnn_data_descriptor as "cudnnDestroyRNNDataDescriptor": PFN_cudnnDestroyRNNDataDescriptor;
    cudnn_rnn_forward as "cudnnRNNForward": PFN_cudnnRNNForward;
    // Graph / backend API
    cudnn_backend_create_descriptor as "cudnnBackendCreateDescriptor": PFN_cudnnBackendCreateDescriptor;
    cudnn_backend_destroy_descriptor as "cudnnBackendDestroyDescriptor": PFN_cudnnBackendDestroyDescriptor;
    cudnn_backend_initialize as "cudnnBackendInitialize": PFN_cudnnBackendInitialize;
    cudnn_backend_finalize as "cudnnBackendFinalize": PFN_cudnnBackendFinalize;
    cudnn_backend_set_attribute as "cudnnBackendSetAttribute": PFN_cudnnBackendSetAttribute;
    cudnn_backend_get_attribute as "cudnnBackendGetAttribute": PFN_cudnnBackendGetAttribute;
    cudnn_backend_execute as "cudnnBackendExecute": PFN_cudnnBackendExecute;
    // Nd descriptors
    cudnn_set_tensor_nd_descriptor as "cudnnSetTensorNdDescriptor": PFN_cudnnSetTensorNdDescriptor;
    cudnn_get_tensor_nd_descriptor as "cudnnGetTensorNdDescriptor": PFN_cudnnGetTensorNdDescriptor;
    cudnn_set_filter_nd_descriptor as "cudnnSetFilterNdDescriptor": PFN_cudnnSetFilterNdDescriptor;
    cudnn_set_convolution_nd_descriptor as "cudnnSetConvolutionNdDescriptor": PFN_cudnnSetConvolutionNdDescriptor;
    cudnn_set_pooling_nd_descriptor as "cudnnSetPoolingNdDescriptor": PFN_cudnnSetPoolingNdDescriptor;
    // CTC
    cudnn_create_ctc_loss_descriptor as "cudnnCreateCTCLossDescriptor": PFN_cudnnCreateCTCLossDescriptor;
    cudnn_destroy_ctc_loss_descriptor as "cudnnDestroyCTCLossDescriptor": PFN_cudnnDestroyCTCLossDescriptor;
    cudnn_set_ctc_loss_descriptor as "cudnnSetCTCLossDescriptor": PFN_cudnnSetCTCLossDescriptor;
    cudnn_get_ctc_loss_workspace_size as "cudnnGetCTCLossWorkspaceSize": PFN_cudnnGetCTCLossWorkspaceSize;
    cudnn_ctc_loss as "cudnnCTCLoss": PFN_cudnnCTCLoss;
    // RNN backward
    cudnn_rnn_backward_data_v8 as "cudnnRNNBackwardData_v8": PFN_cudnnRNNBackwardData_v8;
    cudnn_rnn_backward_weights_v8 as "cudnnRNNBackwardWeights_v8": PFN_cudnnRNNBackwardWeights_v8;
    // Spatial transformer
    cudnn_create_spatial_transformer_descriptor as "cudnnCreateSpatialTransformerDescriptor": PFN_cudnnCreateSpatialTransformerDescriptor;
    cudnn_destroy_spatial_transformer_descriptor as "cudnnDestroySpatialTransformerDescriptor": PFN_cudnnDestroySpatialTransformerDescriptor;
    cudnn_set_spatial_transformer_nd_descriptor as "cudnnSetSpatialTransformerNdDescriptor": PFN_cudnnSetSpatialTransformerNdDescriptor;
    cudnn_spatial_tf_grid_generator_forward as "cudnnSpatialTfGridGeneratorForward": PFN_cudnnSpatialTfGridGeneratorForward;
    cudnn_spatial_tf_sampler_forward as "cudnnSpatialTfSamplerForward": PFN_cudnnSpatialTfSamplerForward;
}

pub fn cudnn() -> Result<&'static Cudnn, LoaderError> {
    static CUDNN: OnceLock<Cudnn> = OnceLock::new();
    if let Some(c) = CUDNN.get() {
        return Ok(c);
    }
    let lib = open_cudnn()?;
    let c = Cudnn::empty(lib);
    let _ = CUDNN.set(c);
    Ok(CUDNN.get().expect("OnceLock set or lost race"))
}
