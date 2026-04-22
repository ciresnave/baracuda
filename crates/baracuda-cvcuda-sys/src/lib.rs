//! Raw FFI + dynamic loader skeleton for NVIDIA CV-CUDA.
//!
//! CV-CUDA is a separate NVIDIA download (not bundled with the CUDA
//! Toolkit). v0.1 ships only the loader; the operator surface lands when
//! we can test against an installed CV-CUDA.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
#![warn(missing_debug_implementations)]

use std::sync::OnceLock;

use baracuda_core::{Library, LoaderError};
use baracuda_types::CudaStatus;

/// NVCV status code (CV-CUDA's error enum).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct NVCVStatus(pub i32);

impl NVCVStatus {
    pub const SUCCESS: Self = Self(0);
    pub const ERROR_NOT_IMPLEMENTED: Self = Self(1);
    pub const ERROR_INVALID_ARGUMENT: Self = Self(2);
    pub const ERROR_INVALID_IMAGE_FORMAT: Self = Self(3);
    pub const ERROR_INVALID_OPERATION: Self = Self(4);
    pub const ERROR_DEVICE: Self = Self(5);
    pub const ERROR_NOT_READY: Self = Self(6);
    pub const ERROR_OUT_OF_MEMORY: Self = Self(7);
    pub const ERROR_INTERNAL: Self = Self(8);
    pub const ERROR_NOT_COMPATIBLE: Self = Self(9);
    pub const ERROR_OVERFLOW: Self = Self(10);
    pub const ERROR_UNDERFLOW: Self = Self(11);

    pub const fn is_success(self) -> bool {
        self.0 == 0
    }
}

impl CudaStatus for NVCVStatus {
    fn code(self) -> i32 {
        self.0
    }
    fn name(self) -> &'static str {
        match self.0 {
            0 => "NVCV_SUCCESS",
            1 => "NVCV_ERROR_NOT_IMPLEMENTED",
            2 => "NVCV_ERROR_INVALID_ARGUMENT",
            3 => "NVCV_ERROR_INVALID_IMAGE_FORMAT",
            5 => "NVCV_ERROR_DEVICE",
            7 => "NVCV_ERROR_OUT_OF_MEMORY",
            8 => "NVCV_ERROR_INTERNAL",
            _ => "NVCV_ERROR_UNRECOGNIZED",
        }
    }
    fn description(self) -> &'static str {
        match self.0 {
            0 => "success",
            2 => "invalid argument",
            7 => "out of device memory",
            _ => "unrecognized CV-CUDA status code",
        }
    }
    fn is_success(self) -> bool {
        NVCVStatus::is_success(self)
    }
    fn library(self) -> &'static str {
        "cvcuda"
    }
}

use core::ffi::c_void;

// ---- NVCV handle typedefs ----

pub type NVCVTensorHandle = *mut c_void;
pub type NVCVImageHandle = *mut c_void;
pub type NVCVImageBatchHandle = *mut c_void;

// ---- CV-CUDA operator handles ----

pub type CVCUDA_OperatorHandle = *mut c_void;

// ---- Enums ----

/// `NVCVInterpolationType`.
#[allow(non_snake_case)]
pub mod NVCVInterpolationType {
    pub const NEAREST: i32 = 0;
    pub const LINEAR: i32 = 1;
    pub const CUBIC: i32 = 2;
    pub const AREA: i32 = 3;
    pub const LANCZOS: i32 = 4;
    pub const GAUSSIAN: i32 = 5;
}

/// `NVCVColorConversionCode` — opencv-style color conversion selector.
/// Subset — full enum has ~200 values.
#[allow(non_snake_case)]
pub mod NVCVColorConversionCode {
    pub const BGR2BGRA: i32 = 0;
    pub const BGRA2BGR: i32 = 1;
    pub const BGR2RGB: i32 = 4;
    pub const BGR2GRAY: i32 = 6;
    pub const RGB2GRAY: i32 = 7;
    pub const GRAY2BGR: i32 = 8;
    pub const BGR2YUV: i32 = 82;
    pub const YUV2BGR: i32 = 84;
    pub const BGR2HSV: i32 = 40;
    pub const HSV2BGR: i32 = 54;
}

// ---- PFN types ----

pub type PFN_cvcudaResizeCreate =
    unsafe extern "C" fn(handle_out: *mut CVCUDA_OperatorHandle) -> NVCVStatus;

pub type PFN_cvcudaResizeSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    interpolation: i32,
) -> NVCVStatus;

pub type PFN_cvcudaCvtColorCreate =
    unsafe extern "C" fn(handle_out: *mut CVCUDA_OperatorHandle) -> NVCVStatus;

pub type PFN_cvcudaCvtColorSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    code: i32,
) -> NVCVStatus;

pub type PFN_cvcudaConvertToCreate =
    unsafe extern "C" fn(handle_out: *mut CVCUDA_OperatorHandle) -> NVCVStatus;

pub type PFN_cvcudaConvertToSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    alpha: f64,
    beta: f64,
) -> NVCVStatus;

pub type PFN_cvcudaFlipCreate =
    unsafe extern "C" fn(handle_out: *mut CVCUDA_OperatorHandle) -> NVCVStatus;

pub type PFN_cvcudaFlipSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    flip_code: i32,
) -> NVCVStatus;

pub type PFN_cvcudaNormalizeCreate =
    unsafe extern "C" fn(handle_out: *mut CVCUDA_OperatorHandle) -> NVCVStatus;

pub type PFN_cvcudaNormalizeSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    base: NVCVTensorHandle,
    scale: NVCVTensorHandle,
    output: NVCVTensorHandle,
    global_scale: f32,
    shift: f32,
    epsilon: f32,
    flags: u32,
) -> NVCVStatus;

pub type PFN_cvcudaOperatorDestroy =
    unsafe extern "C" fn(handle: CVCUDA_OperatorHandle) -> NVCVStatus;

// NVCV Tensor create/destroy
pub type PFN_nvcvTensorConstruct = unsafe extern "C" fn(
    shape: *const i64,
    rank: i32,
    dtype: i32,
    layout: i32,
    handle_out: *mut NVCVTensorHandle,
) -> NVCVStatus;

pub type PFN_nvcvTensorDecRef =
    unsafe extern "C" fn(handle: NVCVTensorHandle, new_ref: *mut i32) -> NVCVStatus;

pub type PFN_nvcvTensorIncRef =
    unsafe extern "C" fn(handle: NVCVTensorHandle, new_ref: *mut i32) -> NVCVStatus;

pub type PFN_nvcvTensorWrapData = unsafe extern "C" fn(
    data: *const c_void,  // NVCVTensorData
    cleanup: *mut c_void, // NVCVTensorDataCleanupFunc
    ctx: *mut c_void,
    handle_out: *mut NVCVTensorHandle,
) -> NVCVStatus;

// ---- Uniform create / submit PFN shapes ----

/// Most operators have a parameterless create: `xxxCreate(handle_out)`.
pub type PFN_cvcudaOpCreate =
    unsafe extern "C" fn(handle_out: *mut CVCUDA_OperatorHandle) -> NVCVStatus;

/// Common `(op, stream, in, out)` submit signature.
pub type PFN_cvcudaSubmitInOut = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
) -> NVCVStatus;

// ---- Border / interpolation constants ----

#[allow(non_snake_case)]
pub mod NVCVBorderType {
    pub const CONSTANT: i32 = 0;
    pub const REPLICATE: i32 = 1;
    pub const REFLECT: i32 = 2;
    pub const WRAP: i32 = 3;
    pub const REFLECT101: i32 = 4;
    pub const TRANSPARENT: i32 = 5;
}

#[allow(non_snake_case)]
pub mod NVCVThresholdType {
    pub const BINARY: u32 = 1;
    pub const BINARY_INV: u32 = 2;
    pub const TRUNC: u32 = 4;
    pub const TOZERO: u32 = 8;
    pub const TOZERO_INV: u32 = 16;
    pub const OTSU: u32 = 32;
    pub const TRIANGLE: u32 = 64;
}

#[allow(non_snake_case)]
pub mod NVCVAdaptiveThresholdType {
    pub const MEAN_C: u32 = 0;
    pub const GAUSSIAN_C: u32 = 1;
}

#[allow(non_snake_case)]
pub mod NVCVMorphologyType {
    pub const ERODE: i32 = 0;
    pub const DILATE: i32 = 1;
    pub const OPEN: i32 = 2;
    pub const CLOSE: i32 = 3;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct NVCVSize2D {
    pub w: i32,
    pub h: i32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct NVCVFloat2 {
    pub x: f32,
    pub y: f32,
}

// ---- Per-operator Submit PFN signatures (unique-shape ops) ----

pub type PFN_cvcudaPillowResizeSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    interpolation: i32,
) -> NVCVStatus;

pub type PFN_cvcudaWarpAffineSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    xform: *const f32, // 2x3
    flags: i32,
    border_mode: i32,
    border_value: *const f32, // 4
) -> NVCVStatus;

pub type PFN_cvcudaWarpPerspectiveSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    xform: *const f32, // 3x3
    flags: i32,
    border_mode: i32,
    border_value: *const f32, // 4
) -> NVCVStatus;

pub type PFN_cvcudaRemapSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    map: NVCVTensorHandle,
    src_interp: i32,
    map_interp: i32,
    map_value_type: i32,
    align_corners: bool,
    border_mode: i32,
    border_value: *const f32,
) -> NVCVStatus;

pub type PFN_cvcudaRotateSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    angle_deg: f64,
    shift: *const f64, // 2
    interpolation: i32,
) -> NVCVStatus;

pub type PFN_cvcudaCenterCropSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    crop_size: NVCVSize2D,
) -> NVCVStatus;

pub type PFN_cvcudaCustomCropSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    crop: *const i32, // NVCVRectI: x,y,w,h
) -> NVCVStatus;

pub type PFN_cvcudaPadAndStackSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: *mut c_void, // NVCVImageBatchHandle (VarShape)
    output: NVCVTensorHandle,
    top: NVCVTensorHandle,
    left: NVCVTensorHandle,
    border_mode: i32,
    border_value: f32,
) -> NVCVStatus;

pub type PFN_cvcudaCopyMakeBorderSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    top: i32,
    left: i32,
    border_mode: i32,
    border_value: *const f32, // 4
) -> NVCVStatus;

pub type PFN_cvcudaReformatSubmit = PFN_cvcudaSubmitInOut;

// Filters

pub type PFN_cvcudaGaussianSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    kernel_size: NVCVSize2D,
    sigma: NVCVFloat2,
    border_mode: i32,
) -> NVCVStatus;

pub type PFN_cvcudaMedianBlurSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    kernel_size: NVCVSize2D,
) -> NVCVStatus;

pub type PFN_cvcudaAverageBlurSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    kernel_size: NVCVSize2D,
    kernel_anchor: NVCVSize2D,
    border_mode: i32,
) -> NVCVStatus;

pub type PFN_cvcudaLaplacianSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    ksize: i32,
    scale: f32,
    border_mode: i32,
) -> NVCVStatus;

pub type PFN_cvcudaBilateralFilterSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    diameter: i32,
    sigma_color: f32,
    sigma_space: f32,
    border_mode: i32,
) -> NVCVStatus;

pub type PFN_cvcudaMotionBlurSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    kernel_size: NVCVSize2D,
    angle: f32,
    border_mode: i32,
) -> NVCVStatus;

pub type PFN_cvcudaConv2DSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    kernel: NVCVTensorHandle,
    kernel_anchor: NVCVSize2D,
    border_mode: i32,
) -> NVCVStatus;

pub type PFN_cvcudaMorphologySubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    workspace: NVCVTensorHandle,
    morph_type: i32,
    mask_size: NVCVSize2D,
    anchor: NVCVSize2D,
    iteration: i32,
    border_mode: i32,
) -> NVCVStatus;

// Edge / Stat
pub type PFN_cvcudaCannySubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    threshold_low: f64,
    threshold_high: f64,
    aperture_size: i32,
    l2_gradient: bool,
) -> NVCVStatus;

pub type PFN_cvcudaHistogramSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    mask: NVCVTensorHandle,
    histogram: NVCVTensorHandle,
) -> NVCVStatus;

pub type PFN_cvcudaHistogramEqSubmit = PFN_cvcudaSubmitInOut;

pub type PFN_cvcudaMinMaxLocSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    min_val: NVCVTensorHandle,
    min_loc: NVCVTensorHandle,
    num_min: NVCVTensorHandle,
    max_val: NVCVTensorHandle,
    max_loc: NVCVTensorHandle,
    num_max: NVCVTensorHandle,
) -> NVCVStatus;

// Thresholds
pub type PFN_cvcudaThresholdCreate = unsafe extern "C" fn(
    handle_out: *mut CVCUDA_OperatorHandle,
    threshold_type: u32,
    max_batch_size: i32,
) -> NVCVStatus;

pub type PFN_cvcudaThresholdSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    thresh: NVCVTensorHandle,
    maxval: NVCVTensorHandle,
) -> NVCVStatus;

pub type PFN_cvcudaAdaptiveThresholdCreate = unsafe extern "C" fn(
    handle_out: *mut CVCUDA_OperatorHandle,
    max_block_size: i32,
    max_batch_size: i32,
) -> NVCVStatus;

pub type PFN_cvcudaAdaptiveThresholdSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    max_value: f64,
    adaptive_method: u32,
    threshold_type: u32,
    block_size: i32,
    c: f64,
) -> NVCVStatus;

// Color
pub type PFN_cvcudaColorTwistSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    twist: NVCVTensorHandle, // 3x4 or 4x4
) -> NVCVStatus;

pub type PFN_cvcudaBrightnessContrastSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    brightness: NVCVTensorHandle,
    contrast: NVCVTensorHandle,
    brightness_shift: NVCVTensorHandle,
    contrast_center: NVCVTensorHandle,
) -> NVCVStatus;

pub type PFN_cvcudaGammaContrastCreate = unsafe extern "C" fn(
    handle_out: *mut CVCUDA_OperatorHandle,
    max_varshape_batch_size: i32,
    max_varshape_channel_count: i32,
) -> NVCVStatus;

pub type PFN_cvcudaGammaContrastSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    gamma: NVCVTensorHandle,
) -> NVCVStatus;

// Composite / stack / channel
pub type PFN_cvcudaCompositeSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    foreground: NVCVTensorHandle,
    background: NVCVTensorHandle,
    fg_mask: NVCVTensorHandle,
    output: NVCVTensorHandle,
) -> NVCVStatus;

pub type PFN_cvcudaStackSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: *mut c_void, // ImageBatchVarShape
    output: NVCVTensorHandle,
) -> NVCVStatus;

pub type PFN_cvcudaChannelReorderSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    order: *const i32, // per-channel permutation indices
    num_channels: i32,
) -> NVCVStatus;

// Misc
pub type PFN_cvcudaEraseCreate = unsafe extern "C" fn(
    handle_out: *mut CVCUDA_OperatorHandle,
    max_num_erasing_area: i32,
) -> NVCVStatus;

pub type PFN_cvcudaEraseSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    anchor: NVCVTensorHandle,
    erasing: NVCVTensorHandle,
    values: NVCVTensorHandle,
    imgidx: NVCVTensorHandle,
    random: bool,
    seed: u32,
    inplace: bool,
) -> NVCVStatus;

pub type PFN_cvcudaInpaintCreate = unsafe extern "C" fn(
    handle_out: *mut CVCUDA_OperatorHandle,
    max_batch_size: i32,
    max_shape: NVCVSize2D,
) -> NVCVStatus;

pub type PFN_cvcudaInpaintSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    masks: NVCVTensorHandle,
    output: NVCVTensorHandle,
    inpaint_radius: f64,
) -> NVCVStatus;

pub type PFN_cvcudaBoxFilterSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    kernel_size: NVCVSize2D,
    kernel_anchor: NVCVSize2D,
    border_mode: i32,
) -> NVCVStatus;

pub type PFN_cvcudaAddWeightedSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input1: NVCVTensorHandle,
    alpha: f64,
    input2: NVCVTensorHandle,
    beta: f64,
    gamma: f64,
    output: NVCVTensorHandle,
) -> NVCVStatus;

pub type PFN_cvcudaNonMaxSuppressionSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    scores: NVCVTensorHandle,
    score_threshold: f32,
    iou_threshold: f32,
) -> NVCVStatus;

// ---- Additional operators (round 2) ----

pub type PFN_cvcudaPadSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    top: i32,
    left: i32,
    border_mode: i32,
    border_value: *const f32, // 4
) -> NVCVStatus;

pub type PFN_cvcudaJointBilateralFilterSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    input_color: NVCVTensorHandle,
    output: NVCVTensorHandle,
    diameter: i32,
    sigma_color: f32,
    sigma_space: f32,
    border_mode: i32,
) -> NVCVStatus;

// Label (connected-component labeling)
pub type PFN_cvcudaLabelCreate = unsafe extern "C" fn(
    handle_out: *mut CVCUDA_OperatorHandle,
    max_labels_per_batch: i32,
) -> NVCVStatus;

pub type PFN_cvcudaLabelSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    bg_label: NVCVTensorHandle,
    min_threshold: NVCVTensorHandle,
    max_threshold: NVCVTensorHandle,
    min_size: NVCVTensorHandle,
    count: NVCVTensorHandle,
    stats: NVCVTensorHandle,
    mask: NVCVTensorHandle,
    connectivity: i32,
    assign_labels: i32,
    mask_type: i32,
) -> NVCVStatus;

// FindContours
pub type PFN_cvcudaFindContoursCreate = unsafe extern "C" fn(
    handle_out: *mut CVCUDA_OperatorHandle,
    max_contour_size: NVCVSize2D,
    max_total_contour_count: i32,
) -> NVCVStatus;

pub type PFN_cvcudaFindContoursSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    points: NVCVTensorHandle,
    num_points: NVCVTensorHandle,
) -> NVCVStatus;

// MinAreaRect
pub type PFN_cvcudaMinAreaRectCreate = unsafe extern "C" fn(
    handle_out: *mut CVCUDA_OperatorHandle,
    max_contour_count: i32,
) -> NVCVStatus;

pub type PFN_cvcudaMinAreaRectSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    num_points_in_contour: NVCVTensorHandle,
    total_contours: i32,
) -> NVCVStatus;

// Bounding-box rendering
pub type PFN_cvcudaBndBoxSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    bboxes: *const c_void, // NVCVBndBoxesI (opaque)
) -> NVCVStatus;

// On-screen display
pub type PFN_cvcudaOSDSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    elements: *const c_void, // NVCVElements (opaque)
) -> NVCVStatus;

// RandomResizedCrop
pub type PFN_cvcudaRandomResizedCropCreate = unsafe extern "C" fn(
    handle_out: *mut CVCUDA_OperatorHandle,
    min_scale: f64,
    max_scale: f64,
    min_ratio: f64,
    max_ratio: f64,
    max_batch_size: i32,
    seed: u32,
) -> NVCVStatus;

pub type PFN_cvcudaRandomResizedCropSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    interpolation: i32,
) -> NVCVStatus;

// Gaussian noise
pub type PFN_cvcudaGaussianNoiseCreate =
    unsafe extern "C" fn(handle_out: *mut CVCUDA_OperatorHandle, max_batch_size: i32) -> NVCVStatus;

pub type PFN_cvcudaGaussianNoiseSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    mu: NVCVTensorHandle,
    sigma: NVCVTensorHandle,
    per_channel: bool,
    seed: u64,
) -> NVCVStatus;

// Rhomboid / Salt-and-pepper noise
pub type PFN_cvcudaRhomboidNoiseCreate =
    unsafe extern "C" fn(handle_out: *mut CVCUDA_OperatorHandle, max_batch_size: i32) -> NVCVStatus;

pub type PFN_cvcudaRhomboidNoiseSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    beta: NVCVTensorHandle,
    seed: u64,
) -> NVCVStatus;

// Advanced color conversion (AdvCvtColor)
pub type PFN_cvcudaAdvCvtColorSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    code: i32,
    spec: i32,
) -> NVCVStatus;

// SIFT
pub type PFN_cvcudaSIFTCreate = unsafe extern "C" fn(
    handle_out: *mut CVCUDA_OperatorHandle,
    max_shape: NVCVSize2D,
    max_num_features: i32,
    num_octave_layers: i32,
    contrast_threshold: f32,
    edge_threshold: f32,
    init_sigma: f32,
    flags: i32,
) -> NVCVStatus;

pub type PFN_cvcudaSIFTSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    feature_coords: NVCVTensorHandle,
    feature_metadata: NVCVTensorHandle,
    feature_descriptors: NVCVTensorHandle,
    num_features: NVCVTensorHandle,
    num_octave_layers: i32,
    contrast_threshold: f32,
    edge_threshold: f32,
    init_sigma: f32,
    flags: i32,
) -> NVCVStatus;

// Chroma-keying / alpha composite
pub type PFN_cvcudaSaltAndPepperNoiseCreate =
    unsafe extern "C" fn(handle_out: *mut CVCUDA_OperatorHandle, max_batch_size: i32) -> NVCVStatus;

pub type PFN_cvcudaSaltAndPepperNoiseSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    salt_prob: NVCVTensorHandle,
    pepper_prob: NVCVTensorHandle,
    per_channel: bool,
    seed: u64,
) -> NVCVStatus;

// HQResize (high-quality resize for ML preproc)
pub type PFN_cvcudaHQResizeSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    min_interpolation: i32,
    mag_interpolation: i32,
    antialias: bool,
    roi: *const c_void, // NVCVHQResizeRoisF (opaque)
) -> NVCVStatus;

// ---- Round 3: finish the operator surface ----

// CropFlipNormalizeReformat — the fused ML-preproc bundle.
pub type PFN_cvcudaCropFlipNormalizeReformatSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    crop_rect: NVCVTensorHandle,
    interpolation: i32,
    flip_code: NVCVTensorHandle,
    base: NVCVTensorHandle,
    scale: NVCVTensorHandle,
    global_scale: f32,
    shift: f32,
    epsilon: f32,
    flags: u32,
) -> NVCVStatus;

// GuidedFilter — edge-preserving filter that uses a guide image.
pub type PFN_cvcudaGuidedFilterSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    guide: NVCVTensorHandle,
    output: NVCVTensorHandle,
    radius: i32,
    eps: f32,
    border_mode: i32,
) -> NVCVStatus;

// PairwiseMatcher — feature-descriptor matcher (e.g. SIFT output).
pub type PFN_cvcudaPairwiseMatcherCreate =
    unsafe extern "C" fn(handle_out: *mut CVCUDA_OperatorHandle, algo_choice: i32) -> NVCVStatus;

pub type PFN_cvcudaPairwiseMatcherSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    set1: NVCVTensorHandle,
    set2: NVCVTensorHandle,
    num_set1: NVCVTensorHandle,
    num_set2: NVCVTensorHandle,
    matches: NVCVTensorHandle,
    num_matches: NVCVTensorHandle,
    distances: NVCVTensorHandle,
    cross_check: bool,
    match_per_point: i32,
    norm_type: i32,
) -> NVCVStatus;

// HausdorffDistance
pub type PFN_cvcudaHausdorffDistanceSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    set1: NVCVTensorHandle,
    set2: NVCVTensorHandle,
    output: NVCVTensorHandle,
) -> NVCVStatus;

// Resize for CropFlipNormalizeReformat (VarShape)
pub type PFN_cvcudaResizeCropConvertReformatSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    resize_size: NVCVSize2D,
    interpolation: i32,
    crop_pos: *const c_void, // NVCVPointI (x,y)
    manip: i32,
    scale: f32,
    offset: f32,
    cast_to_f32: bool,
    src_cast_to_f32: bool,
) -> NVCVStatus;

// ResizeCropConvertReformat alt API (direct signature per 0.14)
pub type PFN_cvcudaResizeCropConvertReformatWithParamsSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    resize_dim: NVCVSize2D,
    interpolation: i32,
    crop_start: *const c_void,
    manip: i32,
    scale: f32,
    offset: f32,
) -> NVCVStatus;

// RotateBatch
pub type PFN_cvcudaRotateBatchSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    angle_deg: NVCVTensorHandle,
    shift: NVCVTensorHandle,
    interpolation: i32,
) -> NVCVStatus;

// WarpAffine batched with per-frame transform
pub type PFN_cvcudaWarpAffineBatchSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: NVCVTensorHandle,
    output: NVCVTensorHandle,
    xform: NVCVTensorHandle, // batch × 2×3
    flags: i32,
    border_mode: i32,
    border_value: *const f32,
) -> NVCVStatus;

// HistogramEq (tensor-level, per-channel)
pub type PFN_cvcudaHistogramEqTensorSubmit = PFN_cvcudaSubmitInOut;

// ResizeVarShape
pub type PFN_cvcudaResizeVarShapeSubmit = unsafe extern "C" fn(
    handle: CVCUDA_OperatorHandle,
    stream: *mut c_void,
    input: *mut c_void, // ImageBatchVarShape
    output: *mut c_void,
    interpolation: i32,
) -> NVCVStatus;

// ---- Loader ----

macro_rules! cvcuda_fns {
    ($($(#[$attr:meta])* fn $name:ident as $sym:literal : $pfn:ty;)*) => {
        pub struct Cvcuda {
            pub lib: Library,
            $(
                $name: OnceLock<$pfn>,
            )*
        }

        impl core::fmt::Debug for Cvcuda {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct("Cvcuda").field("lib", &self.lib).finish_non_exhaustive()
            }
        }

        impl Cvcuda {
            fn empty(lib: Library) -> Self {
                Self { lib, $($name: OnceLock::new(),)* }
            }
            $(
                $(#[$attr])*
                #[doc = concat!("Resolve `", $sym, "`.")]
                pub fn $name(&self) -> Result<$pfn, LoaderError> {
                    if let Some(&p) = self.$name.get() { return Ok(p); }
                    let raw: *mut () = unsafe { self.lib.raw_symbol($sym)? };
                    let p: $pfn = unsafe { core::mem::transmute_copy::<*mut (), $pfn>(&raw) };
                    let _ = self.$name.set(p);
                    Ok(p)
                }
            )*
        }
    };
}

cvcuda_fns! {
    // Existing
    fn resize_create as "cvcudaResizeCreate": PFN_cvcudaResizeCreate;
    fn resize_submit as "cvcudaResizeSubmit": PFN_cvcudaResizeSubmit;
    fn cvt_color_create as "cvcudaCvtColorCreate": PFN_cvcudaCvtColorCreate;
    fn cvt_color_submit as "cvcudaCvtColorSubmit": PFN_cvcudaCvtColorSubmit;
    fn convert_to_create as "cvcudaConvertToCreate": PFN_cvcudaConvertToCreate;
    fn convert_to_submit as "cvcudaConvertToSubmit": PFN_cvcudaConvertToSubmit;
    fn flip_create as "cvcudaFlipCreate": PFN_cvcudaFlipCreate;
    fn flip_submit as "cvcudaFlipSubmit": PFN_cvcudaFlipSubmit;
    fn normalize_create as "cvcudaNormalizeCreate": PFN_cvcudaNormalizeCreate;
    fn normalize_submit as "cvcudaNormalizeSubmit": PFN_cvcudaNormalizeSubmit;
    fn operator_destroy as "cvcudaOperatorDestroy": PFN_cvcudaOperatorDestroy;
    fn nvcv_tensor_construct as "nvcvTensorConstruct": PFN_nvcvTensorConstruct;
    fn nvcv_tensor_dec_ref as "nvcvTensorDecRef": PFN_nvcvTensorDecRef;
    fn nvcv_tensor_inc_ref as "nvcvTensorIncRef": PFN_nvcvTensorIncRef;

    // Geometric
    fn pillow_resize_create as "cvcudaPillowResizeCreate": PFN_cvcudaOpCreate;
    fn pillow_resize_submit as "cvcudaPillowResizeSubmit": PFN_cvcudaPillowResizeSubmit;
    fn warp_affine_create as "cvcudaWarpAffineCreate": PFN_cvcudaOpCreate;
    fn warp_affine_submit as "cvcudaWarpAffineSubmit": PFN_cvcudaWarpAffineSubmit;
    fn warp_perspective_create as "cvcudaWarpPerspectiveCreate": PFN_cvcudaOpCreate;
    fn warp_perspective_submit as "cvcudaWarpPerspectiveSubmit": PFN_cvcudaWarpPerspectiveSubmit;
    fn remap_create as "cvcudaRemapCreate": PFN_cvcudaOpCreate;
    fn remap_submit as "cvcudaRemapSubmit": PFN_cvcudaRemapSubmit;
    fn rotate_create as "cvcudaRotateCreate": PFN_cvcudaOpCreate;
    fn rotate_submit as "cvcudaRotateSubmit": PFN_cvcudaRotateSubmit;
    fn center_crop_create as "cvcudaCenterCropCreate": PFN_cvcudaOpCreate;
    fn center_crop_submit as "cvcudaCenterCropSubmit": PFN_cvcudaCenterCropSubmit;
    fn custom_crop_create as "cvcudaCustomCropCreate": PFN_cvcudaOpCreate;
    fn custom_crop_submit as "cvcudaCustomCropSubmit": PFN_cvcudaCustomCropSubmit;
    fn pad_and_stack_create as "cvcudaPadAndStackCreate": PFN_cvcudaOpCreate;
    fn pad_and_stack_submit as "cvcudaPadAndStackSubmit": PFN_cvcudaPadAndStackSubmit;
    fn copy_make_border_create as "cvcudaCopyMakeBorderCreate": PFN_cvcudaOpCreate;
    fn copy_make_border_submit as "cvcudaCopyMakeBorderSubmit": PFN_cvcudaCopyMakeBorderSubmit;
    fn reformat_create as "cvcudaReformatCreate": PFN_cvcudaOpCreate;
    fn reformat_submit as "cvcudaReformatSubmit": PFN_cvcudaReformatSubmit;

    // Filters
    fn gaussian_create as "cvcudaGaussianCreate": PFN_cvcudaOpCreate;
    fn gaussian_submit as "cvcudaGaussianSubmit": PFN_cvcudaGaussianSubmit;
    fn median_blur_create as "cvcudaMedianBlurCreate": PFN_cvcudaOpCreate;
    fn median_blur_submit as "cvcudaMedianBlurSubmit": PFN_cvcudaMedianBlurSubmit;
    fn average_blur_create as "cvcudaAverageBlurCreate": PFN_cvcudaOpCreate;
    fn average_blur_submit as "cvcudaAverageBlurSubmit": PFN_cvcudaAverageBlurSubmit;
    fn laplacian_create as "cvcudaLaplacianCreate": PFN_cvcudaOpCreate;
    fn laplacian_submit as "cvcudaLaplacianSubmit": PFN_cvcudaLaplacianSubmit;
    fn bilateral_filter_create as "cvcudaBilateralFilterCreate": PFN_cvcudaOpCreate;
    fn bilateral_filter_submit as "cvcudaBilateralFilterSubmit": PFN_cvcudaBilateralFilterSubmit;
    fn motion_blur_create as "cvcudaMotionBlurCreate": PFN_cvcudaOpCreate;
    fn motion_blur_submit as "cvcudaMotionBlurSubmit": PFN_cvcudaMotionBlurSubmit;
    fn conv2d_create as "cvcudaConv2DCreate": PFN_cvcudaOpCreate;
    fn conv2d_submit as "cvcudaConv2DSubmit": PFN_cvcudaConv2DSubmit;
    fn box_filter_create as "cvcudaBoxFilterCreate": PFN_cvcudaOpCreate;
    fn box_filter_submit as "cvcudaBoxFilterSubmit": PFN_cvcudaBoxFilterSubmit;

    // Morph
    fn morphology_create as "cvcudaMorphologyCreate": PFN_cvcudaOpCreate;
    fn morphology_submit as "cvcudaMorphologySubmit": PFN_cvcudaMorphologySubmit;

    // Edge / stat
    fn canny_create as "cvcudaCannyCreate": PFN_cvcudaOpCreate;
    fn canny_submit as "cvcudaCannySubmit": PFN_cvcudaCannySubmit;
    fn histogram_create as "cvcudaHistogramCreate": PFN_cvcudaOpCreate;
    fn histogram_submit as "cvcudaHistogramSubmit": PFN_cvcudaHistogramSubmit;
    fn histogram_eq_create as "cvcudaHistogramEqCreate": PFN_cvcudaOpCreate;
    fn histogram_eq_submit as "cvcudaHistogramEqSubmit": PFN_cvcudaHistogramEqSubmit;
    fn min_max_loc_create as "cvcudaMinMaxLocCreate": PFN_cvcudaOpCreate;
    fn min_max_loc_submit as "cvcudaMinMaxLocSubmit": PFN_cvcudaMinMaxLocSubmit;

    // Thresholds
    fn threshold_create as "cvcudaThresholdCreate": PFN_cvcudaThresholdCreate;
    fn threshold_submit as "cvcudaThresholdSubmit": PFN_cvcudaThresholdSubmit;
    fn adaptive_threshold_create as "cvcudaAdaptiveThresholdCreate":
        PFN_cvcudaAdaptiveThresholdCreate;
    fn adaptive_threshold_submit as "cvcudaAdaptiveThresholdSubmit":
        PFN_cvcudaAdaptiveThresholdSubmit;

    // Color
    fn color_twist_create as "cvcudaColorTwistCreate": PFN_cvcudaOpCreate;
    fn color_twist_submit as "cvcudaColorTwistSubmit": PFN_cvcudaColorTwistSubmit;
    fn brightness_contrast_create as "cvcudaBrightnessContrastCreate": PFN_cvcudaOpCreate;
    fn brightness_contrast_submit as "cvcudaBrightnessContrastSubmit":
        PFN_cvcudaBrightnessContrastSubmit;
    fn gamma_contrast_create as "cvcudaGammaContrastCreate": PFN_cvcudaGammaContrastCreate;
    fn gamma_contrast_submit as "cvcudaGammaContrastSubmit": PFN_cvcudaGammaContrastSubmit;

    // Composite / stack / channel
    fn composite_create as "cvcudaCompositeCreate": PFN_cvcudaOpCreate;
    fn composite_submit as "cvcudaCompositeSubmit": PFN_cvcudaCompositeSubmit;
    fn stack_create as "cvcudaStackCreate": PFN_cvcudaOpCreate;
    fn stack_submit as "cvcudaStackSubmit": PFN_cvcudaStackSubmit;
    fn channel_reorder_create as "cvcudaChannelReorderCreate": PFN_cvcudaOpCreate;
    fn channel_reorder_submit as "cvcudaChannelReorderSubmit": PFN_cvcudaChannelReorderSubmit;

    // Misc
    fn erase_create as "cvcudaEraseCreate": PFN_cvcudaEraseCreate;
    fn erase_submit as "cvcudaEraseSubmit": PFN_cvcudaEraseSubmit;
    fn inpaint_create as "cvcudaInpaintCreate": PFN_cvcudaInpaintCreate;
    fn inpaint_submit as "cvcudaInpaintSubmit": PFN_cvcudaInpaintSubmit;
    fn add_weighted_create as "cvcudaAddWeightedCreate": PFN_cvcudaOpCreate;
    fn add_weighted_submit as "cvcudaAddWeightedSubmit": PFN_cvcudaAddWeightedSubmit;
    fn non_max_suppression_create as "cvcudaNonMaxSuppressionCreate": PFN_cvcudaOpCreate;
    fn non_max_suppression_submit as "cvcudaNonMaxSuppressionSubmit":
        PFN_cvcudaNonMaxSuppressionSubmit;

    // Extended operator set
    fn pad_create as "cvcudaPadCreate": PFN_cvcudaOpCreate;
    fn pad_submit as "cvcudaPadSubmit": PFN_cvcudaPadSubmit;
    fn joint_bilateral_filter_create as "cvcudaJointBilateralFilterCreate": PFN_cvcudaOpCreate;
    fn joint_bilateral_filter_submit as "cvcudaJointBilateralFilterSubmit":
        PFN_cvcudaJointBilateralFilterSubmit;
    fn label_create as "cvcudaLabelCreate": PFN_cvcudaLabelCreate;
    fn label_submit as "cvcudaLabelSubmit": PFN_cvcudaLabelSubmit;
    fn find_contours_create as "cvcudaFindContoursCreate": PFN_cvcudaFindContoursCreate;
    fn find_contours_submit as "cvcudaFindContoursSubmit": PFN_cvcudaFindContoursSubmit;
    fn min_area_rect_create as "cvcudaMinAreaRectCreate": PFN_cvcudaMinAreaRectCreate;
    fn min_area_rect_submit as "cvcudaMinAreaRectSubmit": PFN_cvcudaMinAreaRectSubmit;
    fn bndbox_create as "cvcudaBndBoxCreate": PFN_cvcudaOpCreate;
    fn bndbox_submit as "cvcudaBndBoxSubmit": PFN_cvcudaBndBoxSubmit;
    fn osd_create as "cvcudaOSDCreate": PFN_cvcudaOpCreate;
    fn osd_submit as "cvcudaOSDSubmit": PFN_cvcudaOSDSubmit;
    fn random_resized_crop_create as "cvcudaRandomResizedCropCreate":
        PFN_cvcudaRandomResizedCropCreate;
    fn random_resized_crop_submit as "cvcudaRandomResizedCropSubmit":
        PFN_cvcudaRandomResizedCropSubmit;
    fn gaussian_noise_create as "cvcudaGaussianNoiseCreate": PFN_cvcudaGaussianNoiseCreate;
    fn gaussian_noise_submit as "cvcudaGaussianNoiseSubmit": PFN_cvcudaGaussianNoiseSubmit;
    fn rhomboid_noise_create as "cvcudaRhomboidNoiseCreate": PFN_cvcudaRhomboidNoiseCreate;
    fn rhomboid_noise_submit as "cvcudaRhomboidNoiseSubmit": PFN_cvcudaRhomboidNoiseSubmit;
    fn salt_and_pepper_noise_create as "cvcudaSaltAndPepperNoiseCreate":
        PFN_cvcudaSaltAndPepperNoiseCreate;
    fn salt_and_pepper_noise_submit as "cvcudaSaltAndPepperNoiseSubmit":
        PFN_cvcudaSaltAndPepperNoiseSubmit;
    fn adv_cvt_color_create as "cvcudaAdvCvtColorCreate": PFN_cvcudaOpCreate;
    fn adv_cvt_color_submit as "cvcudaAdvCvtColorSubmit": PFN_cvcudaAdvCvtColorSubmit;
    fn sift_create as "cvcudaSIFTCreate": PFN_cvcudaSIFTCreate;
    fn sift_submit as "cvcudaSIFTSubmit": PFN_cvcudaSIFTSubmit;
    fn hq_resize_create as "cvcudaHQResizeCreate": PFN_cvcudaOpCreate;
    fn hq_resize_submit as "cvcudaHQResizeSubmit": PFN_cvcudaHQResizeSubmit;

    // Round-3 extensions
    fn crop_flip_normalize_reformat_create as "cvcudaCropFlipNormalizeReformatCreate":
        PFN_cvcudaOpCreate;
    fn crop_flip_normalize_reformat_submit as "cvcudaCropFlipNormalizeReformatSubmit":
        PFN_cvcudaCropFlipNormalizeReformatSubmit;
    fn guided_filter_create as "cvcudaGuidedFilterCreate": PFN_cvcudaOpCreate;
    fn guided_filter_submit as "cvcudaGuidedFilterSubmit": PFN_cvcudaGuidedFilterSubmit;
    fn pairwise_matcher_create as "cvcudaPairwiseMatcherCreate":
        PFN_cvcudaPairwiseMatcherCreate;
    fn pairwise_matcher_submit as "cvcudaPairwiseMatcherSubmit":
        PFN_cvcudaPairwiseMatcherSubmit;
    fn hausdorff_distance_create as "cvcudaHausdorffDistanceCreate": PFN_cvcudaOpCreate;
    fn hausdorff_distance_submit as "cvcudaHausdorffDistanceSubmit":
        PFN_cvcudaHausdorffDistanceSubmit;
    fn resize_crop_convert_reformat_create as "cvcudaResizeCropConvertReformatCreate":
        PFN_cvcudaOpCreate;
    fn resize_crop_convert_reformat_submit as "cvcudaResizeCropConvertReformatSubmit":
        PFN_cvcudaResizeCropConvertReformatSubmit;
    fn rotate_batch_create as "cvcudaRotateBatchCreate": PFN_cvcudaOpCreate;
    fn rotate_batch_submit as "cvcudaRotateBatchSubmit": PFN_cvcudaRotateBatchSubmit;
    fn warp_affine_batch_create as "cvcudaWarpAffineBatchCreate": PFN_cvcudaOpCreate;
    fn warp_affine_batch_submit as "cvcudaWarpAffineBatchSubmit":
        PFN_cvcudaWarpAffineBatchSubmit;
    fn resize_var_shape_create as "cvcudaResizeVarShapeCreate": PFN_cvcudaOpCreate;
    fn resize_var_shape_submit as "cvcudaResizeVarShapeSubmit":
        PFN_cvcudaResizeVarShapeSubmit;
}

fn cvcuda_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        &[
            "libcvcuda.so.0",
            "libcvcuda.so",
            "libnvcv_types.so.0",
            "libnvcv_types.so",
        ]
    }
    #[cfg(target_os = "windows")]
    {
        &["cvcuda.dll", "nvcv_types.dll"]
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        &[]
    }
}

pub fn cvcuda() -> Result<&'static Cvcuda, LoaderError> {
    static CVCUDA: OnceLock<Cvcuda> = OnceLock::new();
    if let Some(c) = CVCUDA.get() {
        return Ok(c);
    }
    let lib = Library::open("cvcuda", cvcuda_candidates())?;
    let _ = CVCUDA.set(Cvcuda::empty(lib));
    Ok(CVCUDA.get().expect("OnceLock set or lost race"))
}
