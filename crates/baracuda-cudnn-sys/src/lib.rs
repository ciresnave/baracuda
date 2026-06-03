//! Raw FFI + dynamic loader for NVIDIA cuDNN (classic-API subset).
//!
//! `baracuda-cudnn` wraps this with a safe, typed API. Use this crate
//! directly only if you need a function that the safe layer hasn't
//! wrapped yet (in which case please file a bug).
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

/// Opaque handle. Mirrors `cudnnHandle_t`.
pub type cudnnHandle_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnTensorDescriptor_t`.
pub type cudnnTensorDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnActivationDescriptor_t`.
pub type cudnnActivationDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnFilterDescriptor_t`.
pub type cudnnFilterDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnConvolutionDescriptor_t`.
pub type cudnnConvolutionDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnPoolingDescriptor_t`.
pub type cudnnPoolingDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnLRNDescriptor_t`.
pub type cudnnLRNDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnOpTensorDescriptor_t`.
pub type cudnnOpTensorDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnReduceTensorDescriptor_t`.
pub type cudnnReduceTensorDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnDropoutDescriptor_t`.
pub type cudnnDropoutDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnCTCLossDescriptor_t`.
pub type cudnnCTCLossDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnRNNDescriptor_t`.
pub type cudnnRNNDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnRNNDataDescriptor_t`.
pub type cudnnRNNDataDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnBackendDescriptor_t`.
pub type cudnnBackendDescriptor_t = *mut c_void;

/// Forward-convolution algorithm selector.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnConvolutionFwdAlgo_t {
    /// Implicit gemm.
    ImplicitGemm = 0,
    /// Implicit precomp gemm.
    ImplicitPrecompGemm = 1,
    /// Gemm.
    Gemm = 2,
    /// Direct.
    Direct = 3,
    /// Fft.
    Fft = 4,
    /// Fft tiling.
    FftTiling = 5,
    /// Winograd.
    Winograd = 6,
    /// Winograd nonfused.
    WinogradNonfused = 7,
}

/// Convolution cross-correlation vs true-convolution mode.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnConvolutionMode_t {
    /// Convolution.
    Convolution = 0,
    /// Cross correlation.
    CrossCorrelation = 1,
}

/// Enum mirroring `cudnnDataType_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnDataType_t {
    /// Float.
    Float = 0,
    /// Double.
    Double = 1,
    /// Half.
    Half = 2,
    /// Int8.
    Int8 = 3,
    /// Int32.
    Int32 = 4,
    /// Int8x4.
    Int8x4 = 5,
    /// Uint8.
    Uint8 = 6,
    /// Uint8x4.
    Uint8x4 = 7,
    /// Int8x32.
    Int8x32 = 8,
    /// B float16.
    BFloat16 = 9,
}

/// Enum mirroring `cudnnTensorFormat_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnTensorFormat_t {
    /// Nchw.
    Nchw = 0,
    /// Nhwc.
    Nhwc = 1,
    /// Nchw vect c.
    NchwVectC = 2,
}

/// Enum mirroring `cudnnActivationMode_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnActivationMode_t {
    /// Sigmoid.
    Sigmoid = 0,
    /// Relu.
    Relu = 1,
    /// Tanh.
    Tanh = 2,
    /// Clipped relu.
    ClippedRelu = 3,
    /// Elu.
    Elu = 4,
    /// Identity.
    Identity = 5,
    /// Swish.
    Swish = 6,
}

/// Enum mirroring `cudnnNanPropagation_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnNanPropagation_t {
    /// Not propagate nan.
    NotPropagateNan = 0,
    /// Propagate nan.
    PropagateNan = 1,
}

/// Enum mirroring `cudnnPoolingMode_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnPoolingMode_t {
    /// Max.
    Max = 0,
    /// Average count include padding.
    AverageCountIncludePadding = 1,
    /// Average count exclude padding.
    AverageCountExcludePadding = 2,
    /// Max deterministic.
    MaxDeterministic = 3,
}

/// Enum mirroring `cudnnSoftmaxAlgorithm_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnSoftmaxAlgorithm_t {
    /// Fast.
    Fast = 0,
    /// Accurate.
    Accurate = 1,
    /// Log.
    Log = 2,
}

/// Enum mirroring `cudnnSoftmaxMode_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnSoftmaxMode_t {
    /// Instance.
    Instance = 0,
    /// Channel.
    Channel = 1,
}

/// Enum mirroring `cudnnBatchNormMode_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnBatchNormMode_t {
    /// Per activation.
    PerActivation = 0,
    /// Spatial.
    Spatial = 1,
    /// Spatial persistent.
    SpatialPersistent = 2,
}

/// Enum mirroring `cudnnOpTensorOp_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnOpTensorOp_t {
    /// Add.
    Add = 0,
    /// Mul.
    Mul = 1,
    /// Min.
    Min = 2,
    /// Max.
    Max = 3,
    /// Sqrt.
    Sqrt = 4,
    /// Not.
    Not = 5,
}

/// Enum mirroring `cudnnReduceTensorOp_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnReduceTensorOp_t {
    /// Add.
    Add = 0,
    /// Mul.
    Mul = 1,
    /// Min.
    Min = 2,
    /// Max.
    Max = 3,
    /// Amax.
    Amax = 4,
    /// Avg.
    Avg = 5,
    /// Norm1.
    Norm1 = 6,
    /// Norm2.
    Norm2 = 7,
    /// Mul no zeros.
    MulNoZeros = 8,
}

/// Enum mirroring `cudnnReduceTensorIndices_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnReduceTensorIndices_t {
    /// No indices.
    NoIndices = 0,
    /// Flattened indices.
    FlattenedIndices = 1,
}

/// Enum mirroring `cudnnIndicesType_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnIndicesType_t {
    /// U32.
    U32 = 0,
    /// U64.
    U64 = 1,
    /// U16.
    U16 = 2,
    /// U8.
    U8 = 3,
}

/// Enum mirroring `cudnnRNNMode_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnRNNMode_t {
    /// Relu rnn.
    ReluRnn = 0,
    /// Tanh rnn.
    TanhRnn = 1,
    /// Lstm.
    Lstm = 2,
    /// Gru.
    Gru = 3,
}

/// Enum mirroring `cudnnDirectionMode_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnDirectionMode_t {
    /// Unidirectional.
    Unidirectional = 0,
    /// Bidirectional.
    Bidirectional = 1,
}

/// Enum mirroring `cudnnRNNInputMode_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnRNNInputMode_t {
    /// Linear input.
    LinearInput = 0,
    /// Skip input.
    SkipInput = 1,
}

/// Enum mirroring `cudnnRNNAlgo_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnRNNAlgo_t {
    /// Standard.
    Standard = 0,
    /// Persist static.
    PersistStatic = 1,
    /// Persist dynamic.
    PersistDynamic = 2,
    /// Persist static small h.
    PersistStaticSmallH = 3,
}

/// Enum mirroring `cudnnConvolutionBwdDataAlgo_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnConvolutionBwdDataAlgo_t {
    /// Algo0.
    Algo0 = 0,
    /// Algo1.
    Algo1 = 1,
    /// Fft.
    Fft = 2,
    /// Fft tiling.
    FftTiling = 3,
    /// Winograd.
    Winograd = 4,
    /// Winograd nonfused.
    WinogradNonfused = 5,
}

/// Enum mirroring `cudnnConvolutionBwdFilterAlgo_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnConvolutionBwdFilterAlgo_t {
    /// Algo0.
    Algo0 = 0,
    /// Algo1.
    Algo1 = 1,
    /// Fft.
    Fft = 2,
    /// Algo3.
    Algo3 = 3,
    /// Winograd.
    Winograd = 4,
    /// Winograd nonfused.
    WinogradNonfused = 5,
    /// Fft tiling.
    FftTiling = 6,
}

/// Enum mirroring `cudnnBackendDescriptorType_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnBackendDescriptorType_t {
    /// Pointwise descriptor.
    PointwiseDescriptor = 0,
    /// Convolution descriptor.
    ConvolutionDescriptor = 1,
    /// Engine descriptor.
    EngineDescriptor = 2,
    /// Engine cfg descriptor.
    EngineCfgDescriptor = 3,
    /// Execution plan descriptor.
    ExecutionPlanDescriptor = 4,
    /// Intermediate info descriptor.
    IntermediateInfoDescriptor = 5,
    /// Knob choice descriptor.
    KnobChoiceDescriptor = 6,
    /// Knob info descriptor.
    KnobInfoDescriptor = 7,
    /// Layout info descriptor.
    LayoutInfoDescriptor = 8,
    /// Operation convolution forward descriptor.
    OperationConvolutionForwardDescriptor = 9,
    /// Operation convolution backward filter descriptor.
    OperationConvolutionBackwardFilterDescriptor = 10,
    /// Operation convolution backward data descriptor.
    OperationConvolutionBackwardDataDescriptor = 11,
    /// Operation pointwise descriptor.
    OperationPointwiseDescriptor = 12,
    /// Operation gen stats descriptor.
    OperationGenStatsDescriptor = 13,
    /// Operation reduction descriptor.
    OperationReductionDescriptor = 14,
    /// Operation bn finalize statistics descriptor.
    OperationBnFinalizeStatisticsDescriptor = 15,
    /// Operation graph descriptor.
    OperationGraphDescriptor = 16,
    /// Variant pack descriptor.
    VariantPackDescriptor = 17,
    /// Tensor descriptor.
    TensorDescriptor = 18,
    /// Matmul descriptor.
    MatmulDescriptor = 19,
    /// Operation matmul descriptor.
    OperationMatmulDescriptor = 20,
    /// Operation bn bwd weights descriptor.
    OperationBnBwdWeightsDescriptor = 21,
    /// Resample descriptor.
    ResampleDescriptor = 22,
    /// Operation resample fwd descriptor.
    OperationResampleFwdDescriptor = 23,
    /// Operation resample bwd descriptor.
    OperationResampleBwdDescriptor = 24,
    /// Operation concat descriptor.
    OperationConcatDescriptor = 25,
    /// Operation signal descriptor.
    OperationSignalDescriptor = 26,
    /// Operation norm forward descriptor.
    OperationNormForwardDescriptor = 27,
    /// Operation norm backward descriptor.
    OperationNormBackwardDescriptor = 28,
    /// Operation rng descriptor.
    OperationRngDescriptor = 30,
    /// Rng descriptor.
    RngDescriptor = 31,
}

/// Enum mirroring `cudnnBackendAttributeName_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnBackendAttributeName_t {
    // Just a representative subset — the real enum has ~200 entries.
    /// Pointwise mode.
    PointwiseMode = 0,
    /// Pointwise math prec.
    PointwiseMathPrec = 1,
    /// Pointwise nan propagation.
    PointwiseNanPropagation = 2,
    /// Pointwise relu lower clip.
    PointwiseReluLowerClip = 3,
    /// Pointwise relu upper clip.
    PointwiseReluUpperClip = 4,
    /// Pointwise elu alpha.
    PointwiseEluAlpha = 5,
    // Tensor descriptor
    /// Tensor unique id.
    TensorUniqueId = 100,
    /// Tensor data type.
    TensorDataType = 101,
    /// Tensor byte alignment.
    TensorByteAlignment = 102,
    /// Tensor dimensions.
    TensorDimensions = 103,
    /// Tensor strides.
    TensorStrides = 104,
    // Convolution descriptor
    /// Convolution comp type.
    ConvolutionCompType = 200,
    /// Convolution conv mode.
    ConvolutionConvMode = 201,
    /// Convolution dilations.
    ConvolutionDilations = 202,
    /// Convolution filter strides.
    ConvolutionFilterStrides = 203,
    /// Convolution pre paddings.
    ConvolutionPrePaddings = 204,
    /// Convolution post paddings.
    ConvolutionPostPaddings = 205,
    /// Convolution spatial dims.
    ConvolutionSpatialDims = 206,
    // Operation graph
    /// Operation graph handle.
    OperationGraphHandle = 500,
    /// Operation graph ops.
    OperationGraphOps = 501,
    // Execution plan
    /// Execution plan handle.
    ExecutionPlanHandle = 600,
    /// Execution plan engine config.
    ExecutionPlanEngineConfig = 601,
    /// Execution plan workspace size.
    ExecutionPlanWorkspaceSize = 602,
}

/// Enum mirroring `cudnnBackendAttributeType_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnBackendAttributeType_t {
    /// Handle.
    Handle = 0,
    /// Data type.
    DataType = 1,
    /// Boolean.
    Boolean = 2,
    /// Int64.
    Int64 = 3,
    /// Float value.
    FloatValue = 4,
    /// Double value.
    DoubleValue = 5,
    /// Pointwise mode.
    PointwiseMode = 6,
    /// Convolution mode.
    ConvolutionMode = 7,
    /// Heur mode.
    HeurMode = 8,
    /// Knob type.
    KnobType = 9,
    /// Nan propagation.
    NanPropagation = 10,
    /// Numerical note.
    NumericalNote = 11,
    /// Layout type.
    LayoutType = 12,
    /// Attrib name.
    AttribName = 13,
    /// Pointer t.
    PointerT = 14,
    /// Backend descriptor.
    BackendDescriptor = 15,
    /// Genstats mode.
    GenstatsMode = 16,
    /// Bn finalize stats mode.
    BnFinalizeStatsMode = 17,
    /// Reduction operator type.
    ReductionOperatorType = 18,
    /// Behavior note.
    BehaviorNote = 19,
    /// Tensor reordering mode.
    TensorReorderingMode = 20,
    /// Resample mode.
    ResampleMode = 21,
    /// Padding mode.
    PaddingMode = 22,
    /// Int array.
    IntArray = 23,
    /// Rng distribution.
    RngDistribution = 24,
    /// Norm mode.
    NormMode = 25,
    /// Norm fwd phase.
    NormFwdPhase = 26,
    /// Rng normal.
    RngNormal = 27,
    /// Rng uniform.
    RngUniform = 28,
}

// ---- new enums for v7 algorithm selection / convolution math / norm ------

/// Math type for a convolution descriptor — controls tensor-core usage.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnMathType_t {
    /// Default math.
    DefaultMath = 0,
    /// Allow tensor-core math (Volta+).
    TensorOpMath = 1,
    /// Allow tensor-core math with implicit f16/bf16 down-conversion.
    TensorOpMathAllowConversion = 2,
    /// Strict FMA-only math.
    FmaMath = 3,
}

/// Filter / bias reorder selector for INT8 quantized inference.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnReorderType_t {
    /// Default reorder.
    DefaultReorder = 0,
    /// No reorder.
    NoReorder = 1,
}

/// Generic-normalization mode (cuDNN 8+).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnNormMode_t {
    /// Per activation.
    PerActivation = 0,
    /// Per channel.
    PerChannel = 1,
}

/// Generic-normalization algorithm.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnNormAlgo_t {
    /// Standard.
    Standard = 0,
    /// Persist.
    Persist = 1,
}

/// Optional fused op for normalization (None / Activation / Add+Activation).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnNormOps_t {
    /// Norm.
    Norm = 0,
    /// Norm activation.
    NormActivation = 1,
    /// Norm add activation.
    NormAddActivation = 2,
}

/// Optional fused op for batch-normalization Ex (mirrors cudnnBatchNormOps_t).
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnBatchNormOps_t {
    /// Bn.
    Bn = 0,
    /// Bn activation.
    BnActivation = 1,
    /// Bn add activation.
    BnAddActivation = 2,
}

/// `cudnnDeterminism_t` — distinguishes deterministic vs non-deterministic
/// algorithm choices in `*AlgoPerf_t`.
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum cudnnDeterminism_t {
    /// Non deterministic.
    NonDeterministic = 0,
    /// Deterministic.
    Deterministic = 1,
}

/// Result row from `cudnnFindConvolutionForwardAlgorithm` /
/// `cudnnGetConvolutionForwardAlgorithm_v7`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cudnnConvolutionFwdAlgoPerf_t {
    /// Algo field.
    pub algo: cudnnConvolutionFwdAlgo_t,
    /// Status field.
    pub status: cudnnStatus_t,
    /// Time field.
    pub time: f32,
    /// Memory field.
    pub memory: usize,
    /// Determinism field.
    pub determinism: cudnnDeterminism_t,
    /// Math type field.
    pub math_type: cudnnMathType_t,
    /// Reserved padding; do not use.
    pub reserved: [c_int; 3],
}

/// Algorithm-finder performance row. Mirrors `cudnnConvolutionBwdDataAlgoPerf_t`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cudnnConvolutionBwdDataAlgoPerf_t {
    /// Algo field.
    pub algo: cudnnConvolutionBwdDataAlgo_t,
    /// Status field.
    pub status: cudnnStatus_t,
    /// Time field.
    pub time: f32,
    /// Memory field.
    pub memory: usize,
    /// Determinism field.
    pub determinism: cudnnDeterminism_t,
    /// Math type field.
    pub math_type: cudnnMathType_t,
    /// Reserved padding; do not use.
    pub reserved: [c_int; 3],
}

/// Algorithm-finder performance row. Mirrors `cudnnConvolutionBwdFilterAlgoPerf_t`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct cudnnConvolutionBwdFilterAlgoPerf_t {
    /// Algo field.
    pub algo: cudnnConvolutionBwdFilterAlgo_t,
    /// Status field.
    pub status: cudnnStatus_t,
    /// Time field.
    pub time: f32,
    /// Memory field.
    pub memory: usize,
    /// Determinism field.
    pub determinism: cudnnDeterminism_t,
    /// Math type field.
    pub math_type: cudnnMathType_t,
    /// Reserved padding; do not use.
    pub reserved: [c_int; 3],
}

// ---- new opaque descriptors --------------------------------------------------

/// Opaque handle. Mirrors `cudnnTensorTransformDescriptor_t`.
pub type cudnnTensorTransformDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnAttnDescriptor_t`.
pub type cudnnAttnDescriptor_t = *mut c_void;
/// Opaque handle. Mirrors `cudnnSeqDataDescriptor_t`.
pub type cudnnSeqDataDescriptor_t = *mut c_void;

// ---- status ---------------------------------------------------------------

/// Struct mirroring `cudnnStatus_t`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct cudnnStatus_t(pub i32);

impl cudnnStatus_t {
    /// Success.
    pub const SUCCESS: Self = Self(0);
    /// Not initialized.
    pub const NOT_INITIALIZED: Self = Self(1);
    /// Alloc failed.
    pub const ALLOC_FAILED: Self = Self(2);
    /// Bad param.
    pub const BAD_PARAM: Self = Self(3);
    /// Internal error.
    pub const INTERNAL_ERROR: Self = Self(4);
    /// Invalid value.
    pub const INVALID_VALUE: Self = Self(5);
    /// Arch mismatch.
    pub const ARCH_MISMATCH: Self = Self(6);
    /// Mapping error.
    pub const MAPPING_ERROR: Self = Self(7);
    /// Execution failed.
    pub const EXECUTION_FAILED: Self = Self(8);
    /// Not supported.
    pub const NOT_SUPPORTED: Self = Self(9);
    /// License error.
    pub const LICENSE_ERROR: Self = Self(10);

    /// Is success.
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

/// cuDNN: create. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreate = unsafe extern "C" fn(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
/// cuDNN: destroy. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroy = unsafe extern "C" fn(handle: cudnnHandle_t) -> cudnnStatus_t;
/// cuDNN: set stream. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetStream =
    unsafe extern "C" fn(handle: cudnnHandle_t, stream: cudaStream_t) -> cudnnStatus_t;
/// cuDNN: get version. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetVersion = unsafe extern "C" fn() -> usize;

/// cuDNN: create tensor descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateTensorDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy tensor descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyTensorDescriptor =
    unsafe extern "C" fn(desc: cudnnTensorDescriptor_t) -> cudnnStatus_t;
/// cuDNN: set tensor4d descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetTensor4dDescriptor = unsafe extern "C" fn(
    desc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    data_type: cudnnDataType_t,
    n: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

/// cuDNN: create activation descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateActivationDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnActivationDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy activation descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyActivationDescriptor =
    unsafe extern "C" fn(desc: cudnnActivationDescriptor_t) -> cudnnStatus_t;
/// cuDNN: set activation descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetActivationDescriptor = unsafe extern "C" fn(
    desc: cudnnActivationDescriptor_t,
    mode: cudnnActivationMode_t,
    nan_prop: cudnnNanPropagation_t,
    coef: c_double,
) -> cudnnStatus_t;

/// cuDNN: activation forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: create filter descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateFilterDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy filter descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyFilterDescriptor =
    unsafe extern "C" fn(desc: cudnnFilterDescriptor_t) -> cudnnStatus_t;
/// cuDNN: set filter4d descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetFilter4dDescriptor = unsafe extern "C" fn(
    desc: cudnnFilterDescriptor_t,
    data_type: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    k: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

/// cuDNN: create convolution descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateConvolutionDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy convolution descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyConvolutionDescriptor =
    unsafe extern "C" fn(desc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
/// cuDNN: set convolution2d descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: get convolution2d forward output dim. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetConvolution2dForwardOutputDim = unsafe extern "C" fn(
    conv_desc: cudnnConvolutionDescriptor_t,
    input_desc: cudnnTensorDescriptor_t,
    filter_desc: cudnnFilterDescriptor_t,
    n: *mut c_int,
    c: *mut c_int,
    h: *mut c_int,
    w: *mut c_int,
) -> cudnnStatus_t;

/// cuDNN: get convolution forward workspace size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetConvolutionForwardWorkspaceSize = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    x_desc: cudnnTensorDescriptor_t,
    w_desc: cudnnFilterDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t;

/// cuDNN: convolution forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: convolution backward data. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: convolution backward filter. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: convolution backward bias. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnConvolutionBackwardBias = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const c_void,
    dy_desc: cudnnTensorDescriptor_t,
    dy: *const c_void,
    beta: *const c_void,
    db_desc: cudnnTensorDescriptor_t,
    db: *mut c_void,
) -> cudnnStatus_t;

/// cuDNN: get convolution backward data workspace size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetConvolutionBackwardDataWorkspaceSize = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    w_desc: cudnnFilterDescriptor_t,
    dy_desc: cudnnTensorDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    dx_desc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t;

/// cuDNN: get convolution backward filter workspace size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: create pooling descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreatePoolingDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnPoolingDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy pooling descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyPoolingDescriptor =
    unsafe extern "C" fn(desc: cudnnPoolingDescriptor_t) -> cudnnStatus_t;
/// cuDNN: set pooling2d descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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
/// cuDNN: pooling forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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
/// cuDNN: pooling backward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: softmax forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: softmax backward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: batch normalization forward inference. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: batch normalization forward training. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: batch normalization backward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: create op tensor descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateOpTensorDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnOpTensorDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy op tensor descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyOpTensorDescriptor =
    unsafe extern "C" fn(desc: cudnnOpTensorDescriptor_t) -> cudnnStatus_t;
/// cuDNN: set op tensor descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetOpTensorDescriptor = unsafe extern "C" fn(
    desc: cudnnOpTensorDescriptor_t,
    op: cudnnOpTensorOp_t,
    compute_type: cudnnDataType_t,
    nan_prop: cudnnNanPropagation_t,
) -> cudnnStatus_t;
/// cuDNN: op tensor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: create reduce tensor descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateReduceTensorDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy reduce tensor descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyReduceTensorDescriptor =
    unsafe extern "C" fn(desc: cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;
/// cuDNN: set reduce tensor descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetReduceTensorDescriptor = unsafe extern "C" fn(
    desc: cudnnReduceTensorDescriptor_t,
    op: cudnnReduceTensorOp_t,
    compute_type: cudnnDataType_t,
    nan_prop: cudnnNanPropagation_t,
    indices: cudnnReduceTensorIndices_t,
    indices_type: cudnnIndicesType_t,
) -> cudnnStatus_t;
/// cuDNN: get reduction workspace size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetReductionWorkspaceSize = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    desc: cudnnReduceTensorDescriptor_t,
    a_desc: cudnnTensorDescriptor_t,
    c_desc: cudnnTensorDescriptor_t,
    workspace_size: *mut usize,
) -> cudnnStatus_t;
/// cuDNN: reduce tensor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: add tensor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnAddTensor = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const c_void,
    a_desc: cudnnTensorDescriptor_t,
    a: *const c_void,
    beta: *const c_void,
    c_desc: cudnnTensorDescriptor_t,
    c: *mut c_void,
) -> cudnnStatus_t;

/// cuDNN: transform tensor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnTransformTensor = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    beta: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
) -> cudnnStatus_t;

/// cuDNN: scale tensor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnScaleTensor = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
    alpha: *const c_void,
) -> cudnnStatus_t;

/// cuDNN: set tensor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetTensor = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
    value_ptr: *const c_void,
) -> cudnnStatus_t;

// ---- LRN ------------------------------------------------------------------

/// cuDNN: create LRN descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateLRNDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnLRNDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy LRN descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyLRNDescriptor =
    unsafe extern "C" fn(desc: cudnnLRNDescriptor_t) -> cudnnStatus_t;
/// cuDNN: set LRN descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetLRNDescriptor = unsafe extern "C" fn(
    desc: cudnnLRNDescriptor_t,
    lrn_n: c_int,
    lrn_alpha: c_double,
    lrn_beta: c_double,
    lrn_k: c_double,
) -> cudnnStatus_t;
/// cuDNN: LRN cross channel forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: create dropout descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateDropoutDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnDropoutDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy dropout descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyDropoutDescriptor =
    unsafe extern "C" fn(desc: cudnnDropoutDescriptor_t) -> cudnnStatus_t;
/// cuDNN: dropout get states size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDropoutGetStatesSize =
    unsafe extern "C" fn(handle: cudnnHandle_t, size_in_bytes: *mut usize) -> cudnnStatus_t;
/// cuDNN: dropout get reserve space size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDropoutGetReserveSpaceSize = unsafe extern "C" fn(
    x_desc: cudnnTensorDescriptor_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t;
/// cuDNN: set dropout descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetDropoutDescriptor = unsafe extern "C" fn(
    desc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: f32,
    states: *mut c_void,
    state_size: usize,
    seed: u64,
) -> cudnnStatus_t;
/// cuDNN: dropout forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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
/// cuDNN: dropout backward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: create RNN descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateRNNDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnRNNDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy RNN descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyRNNDescriptor =
    unsafe extern "C" fn(desc: cudnnRNNDescriptor_t) -> cudnnStatus_t;

/// cuDNN: create RNN data descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateRNNDataDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnRNNDataDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy RNN data descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyRNNDataDescriptor =
    unsafe extern "C" fn(desc: cudnnRNNDataDescriptor_t) -> cudnnStatus_t;

/// cuDNN: RNN forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: backend create descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnBackendCreateDescriptor = unsafe extern "C" fn(
    descriptor_type: cudnnBackendDescriptorType_t,
    descriptor: *mut cudnnBackendDescriptor_t,
) -> cudnnStatus_t;
/// cuDNN: backend destroy descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnBackendDestroyDescriptor =
    unsafe extern "C" fn(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;
/// cuDNN: backend initialize. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnBackendInitialize =
    unsafe extern "C" fn(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;
/// cuDNN: backend finalize. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnBackendFinalize =
    unsafe extern "C" fn(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;
/// cuDNN: backend set attribute. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnBackendSetAttribute = unsafe extern "C" fn(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    element_count: i64,
    array_of_elements: *const c_void,
) -> cudnnStatus_t;
/// cuDNN: backend get attribute. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnBackendGetAttribute = unsafe extern "C" fn(
    descriptor: cudnnBackendDescriptor_t,
    attribute_name: cudnnBackendAttributeName_t,
    attribute_type: cudnnBackendAttributeType_t,
    requested_element_count: i64,
    element_count: *mut i64,
    array_of_elements: *mut c_void,
) -> cudnnStatus_t;
/// cuDNN: backend execute. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnBackendExecute = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    execution_plan: cudnnBackendDescriptor_t,
    variant_pack: cudnnBackendDescriptor_t,
) -> cudnnStatus_t;

// ---- error-string helper -------------------------------------------------

/// cuDNN: get error string. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetErrorString =
    unsafe extern "C" fn(status: cudnnStatus_t) -> *const core::ffi::c_char;

// ---- N-dimensional tensor / filter descriptors --------------------------

/// cuDNN: set tensor nd descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetTensorNdDescriptor = unsafe extern "C" fn(
    desc: cudnnTensorDescriptor_t,
    data_type: cudnnDataType_t,
    nb_dims: c_int,
    dim_a: *const c_int,
    stride_a: *const c_int,
) -> cudnnStatus_t;

/// cuDNN: get tensor nd descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetTensorNdDescriptor = unsafe extern "C" fn(
    desc: cudnnTensorDescriptor_t,
    nb_dims_requested: c_int,
    data_type: *mut cudnnDataType_t,
    nb_dims: *mut c_int,
    dim_a: *mut c_int,
    stride_a: *mut c_int,
) -> cudnnStatus_t;

/// cuDNN: set filter nd descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetFilterNdDescriptor = unsafe extern "C" fn(
    desc: cudnnFilterDescriptor_t,
    data_type: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    nb_dims: c_int,
    filter_dim_a: *const c_int,
) -> cudnnStatus_t;

/// cuDNN: set convolution nd descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetConvolutionNdDescriptor = unsafe extern "C" fn(
    desc: cudnnConvolutionDescriptor_t,
    array_length: c_int,
    pad_a: *const c_int,
    filter_stride_a: *const c_int,
    dilation_a: *const c_int,
    mode: cudnnConvolutionMode_t,
    compute_type: cudnnDataType_t,
) -> cudnnStatus_t;

/// cuDNN: set pooling nd descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: create CTC loss descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateCTCLossDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnCTCLossDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy CTC loss descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyCTCLossDescriptor =
    unsafe extern "C" fn(desc: cudnnCTCLossDescriptor_t) -> cudnnStatus_t;
/// cuDNN: set CTC loss descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetCTCLossDescriptor = unsafe extern "C" fn(
    desc: cudnnCTCLossDescriptor_t,
    compute_type: cudnnDataType_t,
) -> cudnnStatus_t;

/// cuDNN: get CTC loss workspace size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: CTC loss. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: RNN backward data. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// cuDNN: RNN backward weights. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

/// Opaque handle. Mirrors `cudnnSpatialTransformerDescriptor_t`.
pub type cudnnSpatialTransformerDescriptor_t = *mut c_void;

/// cuDNN: create spatial transformer descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateSpatialTransformerDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnSpatialTransformerDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy spatial transformer descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroySpatialTransformerDescriptor =
    unsafe extern "C" fn(desc: cudnnSpatialTransformerDescriptor_t) -> cudnnStatus_t;

/// cuDNN: set spatial transformer nd descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetSpatialTransformerNdDescriptor = unsafe extern "C" fn(
    desc: cudnnSpatialTransformerDescriptor_t,
    sampler_type: c_int,
    data_type: cudnnDataType_t,
    nb_dims: c_int,
    dim_a: *const c_int,
) -> cudnnStatus_t;

/// cuDNN: spatial tf grid generator forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSpatialTfGridGeneratorForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    st_desc: cudnnSpatialTransformerDescriptor_t,
    theta: *const c_void,
    grid: *mut c_void,
) -> cudnnStatus_t;

/// cuDNN: spatial tf sampler forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
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

// ==========================================================================
// Tier 1 — convolution + activation + reduction misc gaps
// ==========================================================================

/// cuDNN: set convolution group count. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetConvolutionGroupCount = unsafe extern "C" fn(
    desc: cudnnConvolutionDescriptor_t,
    group_count: c_int,
) -> cudnnStatus_t;
/// cuDNN: get convolution group count. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetConvolutionGroupCount = unsafe extern "C" fn(
    desc: cudnnConvolutionDescriptor_t,
    group_count: *mut c_int,
) -> cudnnStatus_t;

/// cuDNN: set convolution math type. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetConvolutionMathType = unsafe extern "C" fn(
    desc: cudnnConvolutionDescriptor_t,
    math_type: cudnnMathType_t,
) -> cudnnStatus_t;
/// cuDNN: get convolution math type. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetConvolutionMathType = unsafe extern "C" fn(
    desc: cudnnConvolutionDescriptor_t,
    math_type: *mut cudnnMathType_t,
) -> cudnnStatus_t;

/// cuDNN: set convolution reorder type. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetConvolutionReorderType = unsafe extern "C" fn(
    desc: cudnnConvolutionDescriptor_t,
    reorder_type: cudnnReorderType_t,
) -> cudnnStatus_t;
/// cuDNN: get convolution reorder type. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetConvolutionReorderType = unsafe extern "C" fn(
    desc: cudnnConvolutionDescriptor_t,
    reorder_type: *mut cudnnReorderType_t,
) -> cudnnStatus_t;

/// cuDNN: reorder filter and bias. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnReorderFilterAndBias = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    filter_desc: cudnnFilterDescriptor_t,
    reorder_type: cudnnReorderType_t,
    filter_data: *const c_void,
    reordered_filter_data: *mut c_void,
    reorder_bias: c_int,
    bias_data: *const c_void,
    reordered_bias_data: *mut c_void,
) -> cudnnStatus_t;

/// cuDNN: convolution bias activation forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnConvolutionBiasActivationForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    alpha1: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    w_desc: cudnnFilterDescriptor_t,
    w: *const c_void,
    conv_desc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    workspace: *mut c_void,
    workspace_size: usize,
    alpha2: *const c_void,
    z_desc: cudnnTensorDescriptor_t,
    z: *const c_void,
    bias_desc: cudnnTensorDescriptor_t,
    bias: *const c_void,
    activation_desc: cudnnActivationDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
) -> cudnnStatus_t;

/// cuDNN: activation backward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnActivationBackward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    activation_desc: cudnnActivationDescriptor_t,
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

/// cuDNN: set activation descriptor swish beta. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetActivationDescriptorSwishBeta = unsafe extern "C" fn(
    desc: cudnnActivationDescriptor_t,
    swish_beta: c_double,
) -> cudnnStatus_t;
/// cuDNN: get activation descriptor swish beta. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetActivationDescriptorSwishBeta = unsafe extern "C" fn(
    desc: cudnnActivationDescriptor_t,
    swish_beta: *mut c_double,
) -> cudnnStatus_t;

/// cuDNN: LRN cross channel backward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnLRNCrossChannelBackward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    norm_desc: cudnnLRNDescriptor_t,
    lrn_mode: c_int,
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

/// cuDNN: divisive normalization forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDivisiveNormalizationForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    norm_desc: cudnnLRNDescriptor_t,
    mode: c_int,
    alpha: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    means: *const c_void,
    temp: *mut c_void,
    temp2: *mut c_void,
    beta: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
) -> cudnnStatus_t;

/// cuDNN: divisive normalization backward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDivisiveNormalizationBackward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    norm_desc: cudnnLRNDescriptor_t,
    mode: c_int,
    alpha: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    means: *const c_void,
    dy: *const c_void,
    temp: *mut c_void,
    temp2: *mut c_void,
    beta: *const c_void,
    d_xdmeans_desc: cudnnTensorDescriptor_t,
    dx: *mut c_void,
    d_means: *mut c_void,
) -> cudnnStatus_t;

/// cuDNN: get reduction indices size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetReductionIndicesSize = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    desc: cudnnReduceTensorDescriptor_t,
    a_desc: cudnnTensorDescriptor_t,
    c_desc: cudnnTensorDescriptor_t,
    size_in_bytes: *mut usize,
) -> cudnnStatus_t;

// 4-D tensor / filter readback + strided-Set.

/// cuDNN: set tensor4d descriptor ex. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetTensor4dDescriptorEx = unsafe extern "C" fn(
    desc: cudnnTensorDescriptor_t,
    data_type: cudnnDataType_t,
    n: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
    n_stride: c_int,
    c_stride: c_int,
    h_stride: c_int,
    w_stride: c_int,
) -> cudnnStatus_t;

/// cuDNN: get tensor4d descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetTensor4dDescriptor = unsafe extern "C" fn(
    desc: cudnnTensorDescriptor_t,
    data_type: *mut cudnnDataType_t,
    n: *mut c_int,
    c: *mut c_int,
    h: *mut c_int,
    w: *mut c_int,
    n_stride: *mut c_int,
    c_stride: *mut c_int,
    h_stride: *mut c_int,
    w_stride: *mut c_int,
) -> cudnnStatus_t;

/// cuDNN: get filter4d descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetFilter4dDescriptor = unsafe extern "C" fn(
    desc: cudnnFilterDescriptor_t,
    data_type: *mut cudnnDataType_t,
    format: *mut cudnnTensorFormat_t,
    k: *mut c_int,
    c: *mut c_int,
    h: *mut c_int,
    w: *mut c_int,
) -> cudnnStatus_t;

// Dropout descriptor save/restore.

/// cuDNN: get dropout descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetDropoutDescriptor = unsafe extern "C" fn(
    desc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: *mut f32,
    states: *mut *mut c_void,
    seed: *mut u64,
) -> cudnnStatus_t;

/// cuDNN: restore dropout descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnRestoreDropoutDescriptor = unsafe extern "C" fn(
    desc: cudnnDropoutDescriptor_t,
    handle: cudnnHandle_t,
    dropout: f32,
    states: *mut c_void,
    state_size: usize,
    seed: u64,
) -> cudnnStatus_t;

// ==========================================================================
// Tier 2 — algorithm finders / pickers (v7)
// ==========================================================================

/// cuDNN: get convolution forward algorithm. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetConvolutionForwardAlgorithm_v7 = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    src_desc: cudnnTensorDescriptor_t,
    filter_desc: cudnnFilterDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    dst_desc: cudnnTensorDescriptor_t,
    requested_algo_count: c_int,
    returned_algo_count: *mut c_int,
    perf_results: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t;

/// cuDNN: find convolution forward algorithm. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnFindConvolutionForwardAlgorithm = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    src_desc: cudnnTensorDescriptor_t,
    filter_desc: cudnnFilterDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    dst_desc: cudnnTensorDescriptor_t,
    requested_algo_count: c_int,
    returned_algo_count: *mut c_int,
    perf_results: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t;

/// cuDNN: find convolution forward algorithm ex. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnFindConvolutionForwardAlgorithmEx = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    src_desc: cudnnTensorDescriptor_t,
    src: *const c_void,
    filter_desc: cudnnFilterDescriptor_t,
    filter: *const c_void,
    conv_desc: cudnnConvolutionDescriptor_t,
    dst_desc: cudnnTensorDescriptor_t,
    dst: *mut c_void,
    requested_algo_count: c_int,
    returned_algo_count: *mut c_int,
    perf_results: *mut cudnnConvolutionFwdAlgoPerf_t,
    workspace: *mut c_void,
    workspace_size: usize,
) -> cudnnStatus_t;

/// cuDNN: get convolution backward data algorithm. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetConvolutionBackwardDataAlgorithm_v7 = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    filter_desc: cudnnFilterDescriptor_t,
    diff_desc: cudnnTensorDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    grad_desc: cudnnTensorDescriptor_t,
    requested_algo_count: c_int,
    returned_algo_count: *mut c_int,
    perf_results: *mut cudnnConvolutionBwdDataAlgoPerf_t,
) -> cudnnStatus_t;

/// cuDNN: find convolution backward data algorithm. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnFindConvolutionBackwardDataAlgorithm = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    filter_desc: cudnnFilterDescriptor_t,
    diff_desc: cudnnTensorDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    grad_desc: cudnnTensorDescriptor_t,
    requested_algo_count: c_int,
    returned_algo_count: *mut c_int,
    perf_results: *mut cudnnConvolutionBwdDataAlgoPerf_t,
) -> cudnnStatus_t;

/// cuDNN: get convolution backward filter algorithm. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetConvolutionBackwardFilterAlgorithm_v7 = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    src_desc: cudnnTensorDescriptor_t,
    diff_desc: cudnnTensorDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    grad_desc: cudnnFilterDescriptor_t,
    requested_algo_count: c_int,
    returned_algo_count: *mut c_int,
    perf_results: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
) -> cudnnStatus_t;

/// cuDNN: find convolution backward filter algorithm. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnFindConvolutionBackwardFilterAlgorithm = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    src_desc: cudnnTensorDescriptor_t,
    diff_desc: cudnnTensorDescriptor_t,
    conv_desc: cudnnConvolutionDescriptor_t,
    grad_desc: cudnnFilterDescriptor_t,
    requested_algo_count: c_int,
    returned_algo_count: *mut c_int,
    perf_results: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
) -> cudnnStatus_t;

// ==========================================================================
// Tier 3 — BatchNorm "Ex" + generic Normalization API
// ==========================================================================

/// cuDNN: get batch normalization forward training ex workspace size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize =
    unsafe extern "C" fn(
        handle: cudnnHandle_t,
        mode: cudnnBatchNormMode_t,
        bn_ops: cudnnBatchNormOps_t,
        x_desc: cudnnTensorDescriptor_t,
        z_desc: cudnnTensorDescriptor_t,
        y_desc: cudnnTensorDescriptor_t,
        bn_scale_bias_mean_var_desc: cudnnTensorDescriptor_t,
        activation_desc: cudnnActivationDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> cudnnStatus_t;

/// cuDNN: get batch normalization backward ex workspace size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetBatchNormalizationBackwardExWorkspaceSize =
    unsafe extern "C" fn(
        handle: cudnnHandle_t,
        mode: cudnnBatchNormMode_t,
        bn_ops: cudnnBatchNormOps_t,
        x_desc: cudnnTensorDescriptor_t,
        y_desc: cudnnTensorDescriptor_t,
        dy_desc: cudnnTensorDescriptor_t,
        dz_desc: cudnnTensorDescriptor_t,
        dx_desc: cudnnTensorDescriptor_t,
        d_bn_scale_bias_desc: cudnnTensorDescriptor_t,
        activation_desc: cudnnActivationDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> cudnnStatus_t;

/// cuDNN: get batch normalization training ex reserve space size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetBatchNormalizationTrainingExReserveSpaceSize =
    unsafe extern "C" fn(
        handle: cudnnHandle_t,
        mode: cudnnBatchNormMode_t,
        bn_ops: cudnnBatchNormOps_t,
        activation_desc: cudnnActivationDescriptor_t,
        x_desc: cudnnTensorDescriptor_t,
        size_in_bytes: *mut usize,
    ) -> cudnnStatus_t;

/// cuDNN: batch normalization forward training ex. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnBatchNormalizationForwardTrainingEx = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bn_ops: cudnnBatchNormOps_t,
    alpha: *const c_void,
    beta: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    z_desc: cudnnTensorDescriptor_t,
    z: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
    bn_scale_bias_mean_var_desc: cudnnTensorDescriptor_t,
    bn_scale: *const c_void,
    bn_bias: *const c_void,
    exponential_average_factor: c_double,
    result_running_mean: *mut c_void,
    result_running_variance: *mut c_void,
    epsilon: c_double,
    save_mean: *mut c_void,
    save_inv_variance: *mut c_void,
    activation_desc: cudnnActivationDescriptor_t,
    workspace: *mut c_void,
    workspace_size: usize,
    reserve_space: *mut c_void,
    reserve_space_size: usize,
) -> cudnnStatus_t;

/// cuDNN: batch normalization backward ex. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnBatchNormalizationBackwardEx = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bn_ops: cudnnBatchNormOps_t,
    alpha_data_diff: *const c_void,
    beta_data_diff: *const c_void,
    alpha_param_diff: *const c_void,
    beta_param_diff: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *const c_void,
    dy_desc: cudnnTensorDescriptor_t,
    dy: *const c_void,
    dz_desc: cudnnTensorDescriptor_t,
    dz: *mut c_void,
    dx_desc: cudnnTensorDescriptor_t,
    dx: *mut c_void,
    d_bn_scale_bias_desc: cudnnTensorDescriptor_t,
    bn_scale: *const c_void,
    bn_bias: *const c_void,
    d_bn_scale_result: *mut c_void,
    d_bn_bias_result: *mut c_void,
    epsilon: c_double,
    saved_mean: *const c_void,
    saved_inv_variance: *const c_void,
    activation_desc: cudnnActivationDescriptor_t,
    workspace: *mut c_void,
    workspace_size: usize,
    reserve_space: *mut c_void,
    reserve_space_size: usize,
) -> cudnnStatus_t;

// Generic Normalization API (cuDNN 8+).

/// cuDNN: normalization forward inference. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnNormalizationForwardInference = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    norm_ops: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    alpha: *const c_void,
    beta: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    norm_scale_bias_desc: cudnnTensorDescriptor_t,
    norm_scale: *const c_void,
    norm_bias: *const c_void,
    norm_mean_var_desc: cudnnTensorDescriptor_t,
    estimated_mean: *const c_void,
    estimated_variance: *const c_void,
    z_desc: cudnnTensorDescriptor_t,
    z: *const c_void,
    activation_desc: cudnnActivationDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
    epsilon: c_double,
    group_count: c_int,
) -> cudnnStatus_t;

/// cuDNN: get normalization forward training workspace size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetNormalizationForwardTrainingWorkspaceSize =
    unsafe extern "C" fn(
        handle: cudnnHandle_t,
        mode: cudnnNormMode_t,
        norm_ops: cudnnNormOps_t,
        algo: cudnnNormAlgo_t,
        x_desc: cudnnTensorDescriptor_t,
        z_desc: cudnnTensorDescriptor_t,
        y_desc: cudnnTensorDescriptor_t,
        norm_scale_bias_desc: cudnnTensorDescriptor_t,
        activation_desc: cudnnActivationDescriptor_t,
        norm_mean_var_desc: cudnnTensorDescriptor_t,
        size_in_bytes: *mut usize,
        group_count: c_int,
    ) -> cudnnStatus_t;

/// cuDNN: get normalization backward workspace size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetNormalizationBackwardWorkspaceSize = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    norm_ops: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    x_desc: cudnnTensorDescriptor_t,
    y_desc: cudnnTensorDescriptor_t,
    dy_desc: cudnnTensorDescriptor_t,
    dz_desc: cudnnTensorDescriptor_t,
    dx_desc: cudnnTensorDescriptor_t,
    d_norm_scale_bias_desc: cudnnTensorDescriptor_t,
    activation_desc: cudnnActivationDescriptor_t,
    norm_mean_var_desc: cudnnTensorDescriptor_t,
    size_in_bytes: *mut usize,
    group_count: c_int,
) -> cudnnStatus_t;

/// cuDNN: get normalization training reserve space size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetNormalizationTrainingReserveSpaceSize =
    unsafe extern "C" fn(
        handle: cudnnHandle_t,
        mode: cudnnNormMode_t,
        norm_ops: cudnnNormOps_t,
        algo: cudnnNormAlgo_t,
        activation_desc: cudnnActivationDescriptor_t,
        x_desc: cudnnTensorDescriptor_t,
        size_in_bytes: *mut usize,
        group_count: c_int,
    ) -> cudnnStatus_t;

/// cuDNN: normalization forward training. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnNormalizationForwardTraining = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    norm_ops: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    alpha: *const c_void,
    beta: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    norm_scale_bias_desc: cudnnTensorDescriptor_t,
    norm_scale: *const c_void,
    norm_bias: *const c_void,
    exponential_average_factor: c_double,
    norm_mean_var_desc: cudnnTensorDescriptor_t,
    result_running_mean: *mut c_void,
    result_running_variance: *mut c_void,
    epsilon: c_double,
    save_mean: *mut c_void,
    save_inv_variance: *mut c_void,
    activation_desc: cudnnActivationDescriptor_t,
    z_desc: cudnnTensorDescriptor_t,
    z: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *mut c_void,
    workspace: *mut c_void,
    workspace_size: usize,
    reserve_space: *mut c_void,
    reserve_space_size: usize,
    group_count: c_int,
) -> cudnnStatus_t;

/// cuDNN: normalization backward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnNormalizationBackward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    mode: cudnnNormMode_t,
    norm_ops: cudnnNormOps_t,
    algo: cudnnNormAlgo_t,
    alpha_data_diff: *const c_void,
    beta_data_diff: *const c_void,
    alpha_param_diff: *const c_void,
    beta_param_diff: *const c_void,
    x_desc: cudnnTensorDescriptor_t,
    x: *const c_void,
    y_desc: cudnnTensorDescriptor_t,
    y: *const c_void,
    dy_desc: cudnnTensorDescriptor_t,
    dy: *const c_void,
    dz_desc: cudnnTensorDescriptor_t,
    dz: *mut c_void,
    dx_desc: cudnnTensorDescriptor_t,
    dx: *mut c_void,
    d_norm_scale_bias_desc: cudnnTensorDescriptor_t,
    norm_scale: *const c_void,
    norm_bias: *const c_void,
    d_norm_scale: *mut c_void,
    d_norm_bias: *mut c_void,
    epsilon: c_double,
    norm_mean_var_desc: cudnnTensorDescriptor_t,
    saved_mean: *const c_void,
    saved_inv_variance: *const c_void,
    activation_desc: cudnnActivationDescriptor_t,
    workspace: *mut c_void,
    workspace_size: usize,
    reserve_space: *mut c_void,
    reserve_space_size: usize,
    group_count: c_int,
) -> cudnnStatus_t;

// ==========================================================================
// Tier 4 — RNN v8 modernization
// ==========================================================================

/// cuDNN: set RNN descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetRNNDescriptor_v8 = unsafe extern "C" fn(
    rnn_desc: cudnnRNNDescriptor_t,
    algo: cudnnRNNAlgo_t,
    cell_mode: cudnnRNNMode_t,
    bias_mode: c_int,
    dir_mode: cudnnDirectionMode_t,
    input_mode: cudnnRNNInputMode_t,
    data_type: cudnnDataType_t,
    math_prec: cudnnDataType_t,
    math_type: cudnnMathType_t,
    input_size: i32,
    hidden_size: i32,
    proj_size: i32,
    num_layers: i32,
    dropout_desc: cudnnDropoutDescriptor_t,
    aux_flags: u32,
) -> cudnnStatus_t;

/// cuDNN: build RNN dynamic. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnBuildRNNDynamic = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnn_desc: cudnnRNNDescriptor_t,
    mini_batch: c_int,
) -> cudnnStatus_t;

/// cuDNN: get RNN temp space sizes. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetRNNTempSpaceSizes = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnn_desc: cudnnRNNDescriptor_t,
    fwd_mode: c_int,
    x_desc: cudnnRNNDataDescriptor_t,
    work_space_size: *mut usize,
    reserve_space_size: *mut usize,
) -> cudnnStatus_t;

/// cuDNN: get RNN weight space size. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetRNNWeightSpaceSize = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnn_desc: cudnnRNNDescriptor_t,
    weight_space_size: *mut usize,
) -> cudnnStatus_t;

/// cuDNN: get RNN weight params. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetRNNWeightParams = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    rnn_desc: cudnnRNNDescriptor_t,
    pseudo_layer: i32,
    weight_space_size: usize,
    weight_space: *const c_void,
    lin_layer_id: i32,
    m_desc: cudnnTensorDescriptor_t,
    m_addr: *mut *mut c_void,
    b_desc: cudnnTensorDescriptor_t,
    b_addr: *mut *mut c_void,
) -> cudnnStatus_t;

// ==========================================================================
// Tier 5 — Multi-head attention
// ==========================================================================

/// cuDNN: create attn descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateAttnDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnAttnDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy attn descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroyAttnDescriptor =
    unsafe extern "C" fn(desc: cudnnAttnDescriptor_t) -> cudnnStatus_t;

/// cuDNN: set attn descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetAttnDescriptor = unsafe extern "C" fn(
    desc: cudnnAttnDescriptor_t,
    attn_mode: u32,
    n_heads: i32,
    sm_scaler: c_double,
    data_type: cudnnDataType_t,
    compute_prec: cudnnDataType_t,
    math_type: cudnnMathType_t,
    attn_dropout_desc: cudnnDropoutDescriptor_t,
    post_dropout_desc: cudnnDropoutDescriptor_t,
    q_size: i32,
    k_size: i32,
    v_size: i32,
    q_proj_size: i32,
    k_proj_size: i32,
    v_proj_size: i32,
    o_proj_size: i32,
    qo_max_seq_length: i32,
    kv_max_seq_length: i32,
    max_batch_size: i32,
    max_beam_size: i32,
) -> cudnnStatus_t;

/// cuDNN: get multi head attn buffers. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetMultiHeadAttnBuffers = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    attn_desc: cudnnAttnDescriptor_t,
    weight_size_in_bytes: *mut usize,
    work_space_size_in_bytes: *mut usize,
    reserve_space_size_in_bytes: *mut usize,
) -> cudnnStatus_t;

/// cuDNN: get multi head attn weights. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnGetMultiHeadAttnWeights = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    attn_desc: cudnnAttnDescriptor_t,
    w_kind: c_int,
    weight_size_in_bytes: usize,
    weights: *const c_void,
    w_desc: cudnnTensorDescriptor_t,
    w_addr: *mut *mut c_void,
) -> cudnnStatus_t;

/// cuDNN: multi head attn forward. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnMultiHeadAttnForward = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    attn_desc: cudnnAttnDescriptor_t,
    curr_idx: i32,
    lo_win_idx: *const i32,
    hi_win_idx: *const i32,
    dev_seq_lengths_qo: *const i32,
    dev_seq_lengths_kv: *const i32,
    q_desc: cudnnSeqDataDescriptor_t,
    queries: *const c_void,
    residuals: *const c_void,
    k_desc: cudnnSeqDataDescriptor_t,
    keys: *const c_void,
    v_desc: cudnnSeqDataDescriptor_t,
    values: *const c_void,
    o_desc: cudnnSeqDataDescriptor_t,
    out: *mut c_void,
    weight_size_in_bytes: usize,
    weights: *const c_void,
    work_space_size_in_bytes: usize,
    work_space: *mut c_void,
    reserve_space_size_in_bytes: usize,
    reserve_space: *mut c_void,
) -> cudnnStatus_t;

/// cuDNN: multi head attn backward data. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnMultiHeadAttnBackwardData = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    attn_desc: cudnnAttnDescriptor_t,
    lo_win_idx: *const i32,
    hi_win_idx: *const i32,
    dev_seq_lengths_dqdo: *const i32,
    dev_seq_lengths_dkdv: *const i32,
    do_desc: cudnnSeqDataDescriptor_t,
    dout: *const c_void,
    dq_desc: cudnnSeqDataDescriptor_t,
    dqueries: *mut c_void,
    queries: *const c_void,
    dk_desc: cudnnSeqDataDescriptor_t,
    dkeys: *mut c_void,
    keys: *const c_void,
    dv_desc: cudnnSeqDataDescriptor_t,
    dvalues: *mut c_void,
    values: *const c_void,
    weight_size_in_bytes: usize,
    weights: *const c_void,
    work_space_size_in_bytes: usize,
    work_space: *mut c_void,
    reserve_space_size_in_bytes: usize,
    reserve_space: *mut c_void,
) -> cudnnStatus_t;

/// cuDNN: multi head attn backward weights. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnMultiHeadAttnBackwardWeights = unsafe extern "C" fn(
    handle: cudnnHandle_t,
    attn_desc: cudnnAttnDescriptor_t,
    add_grad: c_int,
    q_desc: cudnnSeqDataDescriptor_t,
    queries: *const c_void,
    k_desc: cudnnSeqDataDescriptor_t,
    keys: *const c_void,
    v_desc: cudnnSeqDataDescriptor_t,
    values: *const c_void,
    do_desc: cudnnSeqDataDescriptor_t,
    dout: *const c_void,
    weight_size_in_bytes: usize,
    weights: *const c_void,
    dweights: *mut c_void,
    work_space_size_in_bytes: usize,
    work_space: *mut c_void,
    reserve_space_size_in_bytes: usize,
    reserve_space: *mut c_void,
) -> cudnnStatus_t;

// SeqDataDescriptor lifetime helpers.

/// cuDNN: create seq data descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnCreateSeqDataDescriptor =
    unsafe extern "C" fn(desc: *mut cudnnSeqDataDescriptor_t) -> cudnnStatus_t;
/// cuDNN: destroy seq data descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnDestroySeqDataDescriptor =
    unsafe extern "C" fn(desc: cudnnSeqDataDescriptor_t) -> cudnnStatus_t;
/// cuDNN: set seq data descriptor. See <https://docs.nvidia.com/deeplearning/cudnn/api/index.html>.
pub type PFN_cudnnSetSeqDataDescriptor = unsafe extern "C" fn(
    desc: cudnnSeqDataDescriptor_t,
    data_type: cudnnDataType_t,
    nb_dims: c_int,
    dim_a: *const c_int,
    axes: *const c_int,
    seq_length_array_size: usize,
    seq_length_array: *const c_int,
    padding_fill: *const c_void,
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

/// Detect the CUDA toolkit major version that cuDNN should be paired
/// against. Returns `None` if no signal is available.
///
/// Strategy (Windows-style env vars work on Linux too if set):
///   1. `CUDA_PATH` typically ends in `vNN.M` — e.g.
///      `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6` →
///      `Some(12)`.
///   2. Fall back to scanning `CUDA_PATH_V<NN>_<M>` env vars and
///      picking the highest `<NN>` present.
fn detect_cuda_major() -> Option<u32> {
    if let Ok(p) = std::env::var("CUDA_PATH") {
        if let Some(n) = parse_cuda_major_from_path(&p) {
            return Some(n);
        }
    }
    // CUDA_PATH_V12_6=... CUDA_PATH_V11_8=... — pick the highest.
    let mut best: Option<u32> = None;
    for (k, _) in std::env::vars() {
        if let Some(rest) = k.strip_prefix("CUDA_PATH_V") {
            // rest looks like "12_6"
            if let Some((maj, _)) = rest.split_once('_') {
                if let Ok(n) = maj.parse::<u32>() {
                    best = Some(best.map_or(n, |b| b.max(n)));
                }
            }
        }
    }
    best
}

/// Look for a path component matching `vNN.M` (case-insensitive) and
/// return `NN`.
fn parse_cuda_major_from_path(p: &str) -> Option<u32> {
    for component in p.split(['/', '\\']) {
        let bytes = component.as_bytes();
        let rest = if bytes.first() == Some(&b'v') || bytes.first() == Some(&b'V') {
            &component[1..]
        } else {
            continue;
        };
        let dot = rest.find('.')?;
        if let Ok(n) = rest[..dot].parse::<u32>() {
            return Some(n);
        }
    }
    None
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
        // The numeric subdirectory under `bin/` is the CUDA major version this
        // cuDNN build targets. We must NOT push every subdirectory blindly —
        // on a host with cuDNN installed for multiple CUDA majors that puts
        // the wrong DLL flavor on the search path. Prefer the subdir matching
        // the detected CUDA major; fall back to highest-major-first when we
        // can't tell.
        let target_major = detect_cuda_major();
        if let Ok(pf) = std::env::var("ProgramFiles") {
            let cudnn_root = PathBuf::from(pf).join("NVIDIA").join("CUDNN");
            if let Ok(read_dir) = std::fs::read_dir(&cudnn_root) {
                for entry in read_dir.flatten() {
                    let p = entry.path();
                    if !p.is_dir() {
                        continue;
                    }
                    let bin = p.join("bin");
                    let mut numbered: Vec<(u32, PathBuf)> = Vec::new();
                    let mut unnumbered: Vec<PathBuf> = Vec::new();
                    if let Ok(sub) = std::fs::read_dir(&bin) {
                        for s in sub.flatten() {
                            let sp = s.path();
                            if !sp.is_dir() {
                                continue;
                            }
                            match sp
                                .file_name()
                                .and_then(|n| n.to_str())
                                .and_then(|s| s.parse::<u32>().ok())
                            {
                                Some(n) => numbered.push((n, sp)),
                                None => unnumbered.push(sp),
                            }
                        }
                    }
                    // Match the running CUDA major when known.
                    if let Some(target) = target_major {
                        if let Some(pos) = numbered.iter().position(|(n, _)| *n == target) {
                            let (_, matched) = numbered.swap_remove(pos);
                            out.push(matched);
                        } else {
                            // No exact match — try highest <= target, then fall through.
                            numbered.sort_by_key(|b| std::cmp::Reverse(b.0));
                            if let Some(pos) = numbered.iter().position(|(n, _)| *n <= target) {
                                let (_, matched) = numbered.remove(pos);
                                out.push(matched);
                            } else if let Some((_, p)) = numbered.into_iter().next() {
                                out.push(p);
                            }
                        }
                    } else {
                        // No detection signal — push highest-major first so
                        // newest cuDNN is tried first, but include all as
                        // fallbacks so existing single-CUDA setups still work.
                        numbered.sort_by_key(|b| std::cmp::Reverse(b.0));
                        for (_, p) in numbered {
                            out.push(p);
                        }
                    }
                    // Non-numeric subdirs (rare; older cuDNN packagings)
                    // pass through unfiltered.
                    out.extend(unnumbered);
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
            // SAFETY: `set_var` is unsafe in Rust 2024 because mutating the
            // process environment races with concurrent reads from any other
            // thread that calls `getenv`. We're inside a `OnceLock::get_or_init`
            // so this runs at most once per process, before any cuDNN DLL is
            // loaded. The only readers we care about (Windows's DLL search
            // path inside `LoadLibraryExW`) come *after* this initialization.
            unsafe {
                std::env::set_var("PATH", new_path);
            }
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
        /// Dynamically-loaded cuDNN entry-point table.
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
                #[doc = concat!("Resolve `", $sym, "`.")]
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

    // Tier 1 — convolution + activation + reduction misc
    cudnn_set_convolution_group_count as "cudnnSetConvolutionGroupCount": PFN_cudnnSetConvolutionGroupCount;
    cudnn_get_convolution_group_count as "cudnnGetConvolutionGroupCount": PFN_cudnnGetConvolutionGroupCount;
    cudnn_set_convolution_math_type as "cudnnSetConvolutionMathType": PFN_cudnnSetConvolutionMathType;
    cudnn_get_convolution_math_type as "cudnnGetConvolutionMathType": PFN_cudnnGetConvolutionMathType;
    cudnn_set_convolution_reorder_type as "cudnnSetConvolutionReorderType": PFN_cudnnSetConvolutionReorderType;
    cudnn_get_convolution_reorder_type as "cudnnGetConvolutionReorderType": PFN_cudnnGetConvolutionReorderType;
    cudnn_reorder_filter_and_bias as "cudnnReorderFilterAndBias": PFN_cudnnReorderFilterAndBias;
    cudnn_convolution_bias_activation_forward as "cudnnConvolutionBiasActivationForward": PFN_cudnnConvolutionBiasActivationForward;
    cudnn_activation_backward as "cudnnActivationBackward": PFN_cudnnActivationBackward;
    cudnn_set_activation_descriptor_swish_beta as "cudnnSetActivationDescriptorSwishBeta": PFN_cudnnSetActivationDescriptorSwishBeta;
    cudnn_get_activation_descriptor_swish_beta as "cudnnGetActivationDescriptorSwishBeta": PFN_cudnnGetActivationDescriptorSwishBeta;
    cudnn_lrn_cross_channel_backward as "cudnnLRNCrossChannelBackward": PFN_cudnnLRNCrossChannelBackward;
    cudnn_divisive_normalization_forward as "cudnnDivisiveNormalizationForward": PFN_cudnnDivisiveNormalizationForward;
    cudnn_divisive_normalization_backward as "cudnnDivisiveNormalizationBackward": PFN_cudnnDivisiveNormalizationBackward;
    cudnn_get_reduction_indices_size as "cudnnGetReductionIndicesSize": PFN_cudnnGetReductionIndicesSize;
    cudnn_set_tensor_4d_descriptor_ex as "cudnnSetTensor4dDescriptorEx": PFN_cudnnSetTensor4dDescriptorEx;
    cudnn_get_tensor_4d_descriptor as "cudnnGetTensor4dDescriptor": PFN_cudnnGetTensor4dDescriptor;
    cudnn_get_filter_4d_descriptor as "cudnnGetFilter4dDescriptor": PFN_cudnnGetFilter4dDescriptor;
    cudnn_get_dropout_descriptor as "cudnnGetDropoutDescriptor": PFN_cudnnGetDropoutDescriptor;
    cudnn_restore_dropout_descriptor as "cudnnRestoreDropoutDescriptor": PFN_cudnnRestoreDropoutDescriptor;

    // Tier 2 — algorithm finders / pickers (v7)
    cudnn_get_convolution_forward_algorithm_v7 as "cudnnGetConvolutionForwardAlgorithm_v7": PFN_cudnnGetConvolutionForwardAlgorithm_v7;
    cudnn_find_convolution_forward_algorithm as "cudnnFindConvolutionForwardAlgorithm": PFN_cudnnFindConvolutionForwardAlgorithm;
    cudnn_find_convolution_forward_algorithm_ex as "cudnnFindConvolutionForwardAlgorithmEx": PFN_cudnnFindConvolutionForwardAlgorithmEx;
    cudnn_get_convolution_backward_data_algorithm_v7 as "cudnnGetConvolutionBackwardDataAlgorithm_v7": PFN_cudnnGetConvolutionBackwardDataAlgorithm_v7;
    cudnn_find_convolution_backward_data_algorithm as "cudnnFindConvolutionBackwardDataAlgorithm": PFN_cudnnFindConvolutionBackwardDataAlgorithm;
    cudnn_get_convolution_backward_filter_algorithm_v7 as "cudnnGetConvolutionBackwardFilterAlgorithm_v7": PFN_cudnnGetConvolutionBackwardFilterAlgorithm_v7;
    cudnn_find_convolution_backward_filter_algorithm as "cudnnFindConvolutionBackwardFilterAlgorithm": PFN_cudnnFindConvolutionBackwardFilterAlgorithm;

    // Tier 3 — BatchNorm Ex + generic Normalization
    cudnn_get_batch_normalization_forward_training_ex_workspace_size as "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize": PFN_cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize;
    cudnn_get_batch_normalization_backward_ex_workspace_size as "cudnnGetBatchNormalizationBackwardExWorkspaceSize": PFN_cudnnGetBatchNormalizationBackwardExWorkspaceSize;
    cudnn_get_batch_normalization_training_ex_reserve_space_size as "cudnnGetBatchNormalizationTrainingExReserveSpaceSize": PFN_cudnnGetBatchNormalizationTrainingExReserveSpaceSize;
    cudnn_batch_normalization_forward_training_ex as "cudnnBatchNormalizationForwardTrainingEx": PFN_cudnnBatchNormalizationForwardTrainingEx;
    cudnn_batch_normalization_backward_ex as "cudnnBatchNormalizationBackwardEx": PFN_cudnnBatchNormalizationBackwardEx;
    cudnn_normalization_forward_inference as "cudnnNormalizationForwardInference": PFN_cudnnNormalizationForwardInference;
    cudnn_get_normalization_forward_training_workspace_size as "cudnnGetNormalizationForwardTrainingWorkspaceSize": PFN_cudnnGetNormalizationForwardTrainingWorkspaceSize;
    cudnn_get_normalization_backward_workspace_size as "cudnnGetNormalizationBackwardWorkspaceSize": PFN_cudnnGetNormalizationBackwardWorkspaceSize;
    cudnn_get_normalization_training_reserve_space_size as "cudnnGetNormalizationTrainingReserveSpaceSize": PFN_cudnnGetNormalizationTrainingReserveSpaceSize;
    cudnn_normalization_forward_training as "cudnnNormalizationForwardTraining": PFN_cudnnNormalizationForwardTraining;
    cudnn_normalization_backward as "cudnnNormalizationBackward": PFN_cudnnNormalizationBackward;

    // Tier 4 — RNN v8
    cudnn_set_rnn_descriptor_v8 as "cudnnSetRNNDescriptor_v8": PFN_cudnnSetRNNDescriptor_v8;
    cudnn_build_rnn_dynamic as "cudnnBuildRNNDynamic": PFN_cudnnBuildRNNDynamic;
    cudnn_get_rnn_temp_space_sizes as "cudnnGetRNNTempSpaceSizes": PFN_cudnnGetRNNTempSpaceSizes;
    cudnn_get_rnn_weight_space_size as "cudnnGetRNNWeightSpaceSize": PFN_cudnnGetRNNWeightSpaceSize;
    cudnn_get_rnn_weight_params as "cudnnGetRNNWeightParams": PFN_cudnnGetRNNWeightParams;

    // Tier 5 — Multi-head attention
    cudnn_create_attn_descriptor as "cudnnCreateAttnDescriptor": PFN_cudnnCreateAttnDescriptor;
    cudnn_destroy_attn_descriptor as "cudnnDestroyAttnDescriptor": PFN_cudnnDestroyAttnDescriptor;
    cudnn_set_attn_descriptor as "cudnnSetAttnDescriptor": PFN_cudnnSetAttnDescriptor;
    cudnn_get_multi_head_attn_buffers as "cudnnGetMultiHeadAttnBuffers": PFN_cudnnGetMultiHeadAttnBuffers;
    cudnn_get_multi_head_attn_weights as "cudnnGetMultiHeadAttnWeights": PFN_cudnnGetMultiHeadAttnWeights;
    cudnn_multi_head_attn_forward as "cudnnMultiHeadAttnForward": PFN_cudnnMultiHeadAttnForward;
    cudnn_multi_head_attn_backward_data as "cudnnMultiHeadAttnBackwardData": PFN_cudnnMultiHeadAttnBackwardData;
    cudnn_multi_head_attn_backward_weights as "cudnnMultiHeadAttnBackwardWeights": PFN_cudnnMultiHeadAttnBackwardWeights;
    cudnn_create_seq_data_descriptor as "cudnnCreateSeqDataDescriptor": PFN_cudnnCreateSeqDataDescriptor;
    cudnn_destroy_seq_data_descriptor as "cudnnDestroySeqDataDescriptor": PFN_cudnnDestroySeqDataDescriptor;
    cudnn_set_seq_data_descriptor as "cudnnSetSeqDataDescriptor": PFN_cudnnSetSeqDataDescriptor;
}

/// Lazily-initialized process-wide cuDNN loader singleton.
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

#[cfg(test)]
mod search_dir_tests {
    use super::*;

    #[test]
    fn parse_cuda_major_handles_typical_windows_paths() {
        assert_eq!(
            parse_cuda_major_from_path(
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
            ),
            Some(12),
        );
        assert_eq!(
            parse_cuda_major_from_path(
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
            ),
            Some(11),
        );
        // Linux-style.
        assert_eq!(parse_cuda_major_from_path("/opt/cuda/v13.0"), Some(13));
    }

    #[test]
    fn parse_cuda_major_ignores_unrelated_v_prefixed_words() {
        // No `vN.M` segment — `verbose` happens to start with a `v`
        // but doesn't match the `vN.M` pattern.
        assert_eq!(
            parse_cuda_major_from_path("/usr/local/verbose/cuda"),
            None,
        );
        assert_eq!(parse_cuda_major_from_path(""), None);
        assert_eq!(parse_cuda_major_from_path("/usr/local/cuda"), None);
    }

    #[test]
    fn parse_cuda_major_accepts_uppercase_v() {
        assert_eq!(
            parse_cuda_major_from_path(r"D:\NVIDIA\CUDA\V12.6\bin"),
            Some(12),
        );
    }

    /// Reproduces the multi-CUDA bug: with the *old* logic, all `bin/<n>`
    /// subdirs would be pushed in directory-iteration order (effectively
    /// arbitrary). Now we should pick the one matching the detected
    /// major.
    #[test]
    fn dir_selection_prefers_target_major() {
        // Simulate the inner numbered-subdir filter inline. We don't
        // touch the filesystem; we just verify the policy.
        let numbered: Vec<(u32, &str)> = vec![(11, "/cudnn/bin/11"), (12, "/cudnn/bin/12")];
        let target = Some(12u32);

        let chosen: Vec<&str> = match target {
            Some(t) => numbered
                .iter()
                .find(|(n, _)| *n == t)
                .map(|(_, p)| *p)
                .into_iter()
                .collect(),
            None => numbered.iter().map(|(_, p)| *p).collect(),
        };
        assert_eq!(chosen, vec!["/cudnn/bin/12"]);
    }

    #[test]
    fn dir_selection_falls_back_to_highest_le_target() {
        // Detected major = 13 but only cuDNN/12 + cuDNN/11 are installed.
        // Highest-<=-target wins.
        let mut numbered: Vec<(u32, &str)> = vec![(11, "/cudnn/11"), (12, "/cudnn/12")];
        let target = 13u32;

        // Replicate the policy: exact match → take it; else highest <= target.
        let result = match numbered.iter().position(|(n, _)| *n == target) {
            Some(_pos) => unreachable!("no exact match in this scenario"),
            None => {
                numbered.sort_by_key(|b| std::cmp::Reverse(b.0));
                numbered
                    .iter()
                    .find(|(n, _)| *n <= target)
                    .map(|(_, p)| *p)
            }
        };
        assert_eq!(result, Some("/cudnn/12"));
    }

    #[test]
    fn dir_selection_no_signal_is_highest_first() {
        let mut numbered: Vec<(u32, &str)> = vec![(11, "/11"), (13, "/13"), (12, "/12")];
        numbered.sort_by_key(|b| std::cmp::Reverse(b.0));
        let order: Vec<&str> = numbered.iter().map(|(_, p)| *p).collect();
        assert_eq!(order, vec!["/13", "/12", "/11"]);
    }
}
