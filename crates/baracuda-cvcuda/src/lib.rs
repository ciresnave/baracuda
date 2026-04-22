//! Safe Rust wrappers for CV-CUDA (NVCV) — GPU computer-vision primitives.
//!
//! CV-CUDA is a separately-installed NVIDIA library; this crate wraps the
//! C API surface (not the C++ one). The workhorses are:
//!
//! - [`Resize`] — bilinear / cubic / area resampling.
//! - [`CvtColor`] — OpenCV-style colorspace conversion (BGR ↔ RGB ↔ YUV ↔ HSV ↔ grayscale).
//! - [`ConvertTo`] — dtype reinterpretation with `alpha * x + beta`.
//! - [`Flip`] — flip along one or both axes.
//! - [`Normalize`] — `(x - base) * scale + shift` with optional epsilon (ML normalization).
//!
//! NVCV tensors are constructed via [`Tensor::new`] and reference-counted
//! under the hood; Rust wrappers release their ref on drop.

#![warn(missing_debug_implementations)]

use core::ffi::c_void;

use baracuda_cvcuda_sys::{
    cvcuda, CVCUDA_OperatorHandle, NVCVInterpolationType, NVCVStatus, NVCVTensorHandle,
};

pub use baracuda_cvcuda_sys::NVCVColorConversionCode;

/// Error type for CV-CUDA operations.
pub type Error = baracuda_core::Error<NVCVStatus>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: NVCVStatus) -> Result<()> {
    Error::check(status)
}

/// Verify CV-CUDA is loadable on this host.
pub fn probe() -> Result<()> {
    cvcuda()?;
    Ok(())
}

/// Interpolation selector for resampling operators.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Interpolation {
    Nearest,
    Linear,
    Cubic,
    Area,
    Lanczos,
    Gaussian,
}

impl Interpolation {
    #[inline]
    fn raw(self) -> i32 {
        match self {
            Interpolation::Nearest => NVCVInterpolationType::NEAREST,
            Interpolation::Linear => NVCVInterpolationType::LINEAR,
            Interpolation::Cubic => NVCVInterpolationType::CUBIC,
            Interpolation::Area => NVCVInterpolationType::AREA,
            Interpolation::Lanczos => NVCVInterpolationType::LANCZOS,
            Interpolation::Gaussian => NVCVInterpolationType::GAUSSIAN,
        }
    }
}

/// An NVCV tensor handle. Drop releases the ref.
#[derive(Debug)]
pub struct Tensor {
    handle: NVCVTensorHandle,
}

impl Tensor {
    /// Construct a tensor from a shape, dtype, and NVCV layout code.
    /// Layout codes are opaque i32 enums in the CV-CUDA C API; see
    /// `NVCVTensorLayout.h` for the full list (NHWC=0, NCHW=1, ...).
    pub fn new(shape: &[i64], dtype: i32, layout: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.nvcv_tensor_construct()?;
        let mut h: NVCVTensorHandle = core::ptr::null_mut();
        check(unsafe { cu(shape.as_ptr(), shape.len() as i32, dtype, layout, &mut h) })?;
        Ok(Self { handle: h })
    }

    /// Wrap a pre-existing NVCV handle (does NOT increment ref).
    ///
    /// # Safety
    ///
    /// `handle` must own a ref that this wrapper will release on drop.
    pub unsafe fn from_raw(handle: NVCVTensorHandle) -> Self {
        Self { handle }
    }

    #[inline]
    pub fn as_raw(&self) -> NVCVTensorHandle {
        self.handle
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if let Ok(c) = cvcuda() {
            if let Ok(cu) = c.nvcv_tensor_dec_ref() {
                let mut _new_ref: i32 = 0;
                let _ = unsafe { cu(self.handle, &mut _new_ref) };
            }
        }
    }
}

/// Base operator RAII guard — destroys the operator on drop.
#[derive(Debug)]
struct OpHandle {
    handle: CVCUDA_OperatorHandle,
}

impl Drop for OpHandle {
    fn drop(&mut self) {
        if let Ok(c) = cvcuda() {
            if let Ok(cu) = c.operator_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// Resize operator. Reusable across submissions.
#[derive(Debug)]
pub struct Resize {
    op: OpHandle,
}

impl Resize {
    pub fn new() -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.resize_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// Submit a resize `input → output` on `stream`.
    ///
    /// # Safety
    ///
    /// `stream` must be a live `cudaStream_t`.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        interp: Interpolation,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.resize_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            interp.raw(),
        ))
    }
}

/// Color-conversion operator.
#[derive(Debug)]
pub struct CvtColor {
    op: OpHandle,
}

impl CvtColor {
    pub fn new() -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.cvt_color_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// Submit `output = CvtColor(input, code)`. `code` from
    /// [`NVCVColorConversionCode`] constants.
    ///
    /// # Safety
    ///
    /// `stream` must be a live `cudaStream_t`.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        code: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.cvt_color_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            code,
        ))
    }
}

/// Convert-to operator (`output = alpha * input + beta`, with dtype change).
#[derive(Debug)]
pub struct ConvertTo {
    op: OpHandle,
}

impl ConvertTo {
    pub fn new() -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.convert_to_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// Submit `output = alpha * input + beta`.
    ///
    /// # Safety
    ///
    /// `stream` must be a live `cudaStream_t`.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        alpha: f64,
        beta: f64,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.convert_to_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            alpha,
            beta,
        ))
    }
}

/// Flip operator. `flip_code`: 0 = vertical, 1 = horizontal, -1 = both.
#[derive(Debug)]
pub struct Flip {
    op: OpHandle,
}

impl Flip {
    pub fn new() -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.flip_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` must be a live `cudaStream_t`.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        flip_code: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.flip_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            flip_code,
        ))
    }
}

/// Normalize operator (ML pre-processing: `(x - base) * scale` + tweaks).
#[derive(Debug)]
pub struct Normalize {
    op: OpHandle,
}

impl Normalize {
    pub fn new() -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.normalize_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// Submit normalization.
    ///
    /// # Safety
    ///
    /// `stream` must be a live `cudaStream_t`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        base: &Tensor,
        scale: &Tensor,
        output: &Tensor,
        global_scale: f32,
        shift: f32,
        epsilon: f32,
        flags: u32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.normalize_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            base.as_raw(),
            scale.as_raw(),
            output.as_raw(),
            global_scale,
            shift,
            epsilon,
            flags,
        ))
    }
}

// ------------------------------------------------------------------
// The rest of the CV-CUDA operator surface.
//
// Each operator follows the same pattern:
//   struct OpName { op: OpHandle }
//   impl OpName { fn new() -> Result<Self>; unsafe fn submit(&self, ...) }
// Drop destroys the underlying operator handle via the shared
// `cvcudaOperatorDestroy`.
//
// All submit fns are `unsafe` because the NVCV tensor handles are C
// handles that the library doesn't bounds-check in the wrapper.
// ------------------------------------------------------------------

pub use baracuda_cvcuda_sys::{
    NVCVAdaptiveThresholdType, NVCVBorderType, NVCVFloat2, NVCVMorphologyType, NVCVSize2D,
    NVCVThresholdType,
};

/// Reusable helper: create an operator via the parameterless
/// `cvcuda<Name>Create(handle_out)` entry point.
fn create_op<F>(resolve: F) -> Result<OpHandle>
where
    F: FnOnce(
        &baracuda_cvcuda_sys::Cvcuda,
    ) -> std::result::Result<
        unsafe extern "C" fn(*mut CVCUDA_OperatorHandle) -> NVCVStatus,
        baracuda_core::LoaderError,
    >,
{
    let c = cvcuda()?;
    let cu = resolve(c)?;
    let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
    check(unsafe { cu(&mut h) })?;
    Ok(OpHandle { handle: h })
}

macro_rules! simple_op {
    (
        $(#[$m:meta])*
        $Name:ident, $create_fn:ident
    ) => {
        $(#[$m])*
        #[derive(Debug)]
        pub struct $Name {
            op: OpHandle,
        }
        impl $Name {
            pub fn new() -> Result<Self> {
                let op = create_op(|c| c.$create_fn())?;
                Ok(Self { op })
            }
            #[inline]
            pub(crate) fn raw(&self) -> CVCUDA_OperatorHandle {
                self.op.handle
            }
        }
    };
}

simple_op! {
    /// Pillow-style high-quality resize.
    PillowResize, pillow_resize_create
}

impl PillowResize {
    /// # Safety
    ///
    /// `stream` must be a live `cudaStream_t`.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        interp: Interpolation,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.pillow_resize_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            interp.raw(),
        ))
    }
}

simple_op! {
    /// 2×3 affine warp (`output = input @ M`).
    WarpAffine, warp_affine_create
}

impl WarpAffine {
    /// `xform` is the 2×3 affine matrix in row-major order.
    /// `border_value` is a 4-float RGBA constant (only used with
    /// `NVCVBorderType::CONSTANT`).
    ///
    /// # Safety
    ///
    /// `stream` live; `xform.len() == 6`; `border_value.len() == 4`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        xform: &[f32; 6],
        flags: i32,
        border_mode: i32,
        border_value: &[f32; 4],
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.warp_affine_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            xform.as_ptr(),
            flags,
            border_mode,
            border_value.as_ptr(),
        ))
    }
}

simple_op! {
    /// 3×3 perspective (projective) warp.
    WarpPerspective, warp_perspective_create
}

impl WarpPerspective {
    /// # Safety
    ///
    /// `xform.len() == 9`; `border_value.len() == 4`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        xform: &[f32; 9],
        flags: i32,
        border_mode: i32,
        border_value: &[f32; 4],
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.warp_perspective_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            xform.as_ptr(),
            flags,
            border_mode,
            border_value.as_ptr(),
        ))
    }
}

simple_op! {
    /// Remap by sampling a pixel-offset map (optical-flow, undistort).
    Remap, remap_create
}

impl Remap {
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        map: &Tensor,
        src_interp: Interpolation,
        map_interp: Interpolation,
        map_value_type: i32,
        align_corners: bool,
        border_mode: i32,
        border_value: &[f32; 4],
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.remap_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            map.as_raw(),
            src_interp.raw(),
            map_interp.raw(),
            map_value_type,
            align_corners,
            border_mode,
            border_value.as_ptr(),
        ))
    }
}

simple_op! {
    /// Rotate around image center by an angle in degrees.
    Rotate, rotate_create
}

impl Rotate {
    /// # Safety
    ///
    /// `stream` live; `shift.len() == 2`.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        angle_deg: f64,
        shift: &[f64; 2],
        interp: Interpolation,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.rotate_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            angle_deg,
            shift.as_ptr(),
            interp.raw(),
        ))
    }
}

simple_op! {
    /// Center-crop to a target size.
    CenterCrop, center_crop_create
}

impl CenterCrop {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        crop_size: NVCVSize2D,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.center_crop_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            crop_size,
        ))
    }
}

simple_op! {
    /// Arbitrary-rect crop.
    CustomCrop, custom_crop_create
}

impl CustomCrop {
    /// `rect = [x, y, width, height]`.
    ///
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        rect: &[i32; 4],
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.custom_crop_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            rect.as_ptr(),
        ))
    }
}

simple_op! {
    /// Constant / reflect / replicate border around `input`.
    CopyMakeBorder, copy_make_border_create
}

impl CopyMakeBorder {
    /// # Safety
    ///
    /// `stream` live; `border_value.len() == 4`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        top: i32,
        left: i32,
        border_mode: i32,
        border_value: &[f32; 4],
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.copy_make_border_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            top,
            left,
            border_mode,
            border_value.as_ptr(),
        ))
    }
}

simple_op! {
    /// Reformat layout (e.g. NHWC → NCHW).
    Reformat, reformat_create
}

impl Reformat {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.reformat_submit()?;
        check(cu(self.raw(), stream, input.as_raw(), output.as_raw()))
    }
}

// ---------------------- Filters ----------------------

simple_op! {
    /// Separable Gaussian blur.
    Gaussian, gaussian_create
}

impl Gaussian {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        kernel_size: NVCVSize2D,
        sigma: NVCVFloat2,
        border_mode: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.gaussian_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            kernel_size,
            sigma,
            border_mode,
        ))
    }
}

simple_op! {
    /// Median blur.
    MedianBlur, median_blur_create
}

impl MedianBlur {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        kernel_size: NVCVSize2D,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.median_blur_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            kernel_size,
        ))
    }
}

simple_op! {
    /// Box / average blur.
    AverageBlur, average_blur_create
}

impl AverageBlur {
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        kernel_size: NVCVSize2D,
        anchor: NVCVSize2D,
        border_mode: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.average_blur_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            kernel_size,
            anchor,
            border_mode,
        ))
    }
}

simple_op! {
    /// Laplacian edge filter.
    Laplacian, laplacian_create
}

impl Laplacian {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        ksize: i32,
        scale: f32,
        border_mode: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.laplacian_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            ksize,
            scale,
            border_mode,
        ))
    }
}

simple_op! {
    /// Bilateral (edge-preserving) filter.
    BilateralFilter, bilateral_filter_create
}

impl BilateralFilter {
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        diameter: i32,
        sigma_color: f32,
        sigma_space: f32,
        border_mode: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.bilateral_filter_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            diameter,
            sigma_color,
            sigma_space,
            border_mode,
        ))
    }
}

simple_op! {
    /// Directional motion blur.
    MotionBlur, motion_blur_create
}

impl MotionBlur {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        kernel_size: NVCVSize2D,
        angle: f32,
        border_mode: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.motion_blur_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            kernel_size,
            angle,
            border_mode,
        ))
    }
}

simple_op! {
    /// 2D convolution with a user-supplied kernel.
    Conv2D, conv2d_create
}

impl Conv2D {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        kernel: &Tensor,
        anchor: NVCVSize2D,
        border_mode: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.conv2d_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            kernel.as_raw(),
            anchor,
            border_mode,
        ))
    }
}

simple_op! {
    /// Unnormalized box filter.
    BoxFilter, box_filter_create
}

impl BoxFilter {
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        kernel_size: NVCVSize2D,
        anchor: NVCVSize2D,
        border_mode: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.box_filter_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            kernel_size,
            anchor,
            border_mode,
        ))
    }
}

// ---------------------- Morphology ----------------------

simple_op! {
    /// Erosion / dilation / open / close.
    Morphology, morphology_create
}

impl Morphology {
    /// # Safety
    ///
    /// `stream` live; `workspace` must be a correctly-shaped scratch tensor.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        workspace: &Tensor,
        morph_type: i32,
        mask_size: NVCVSize2D,
        anchor: NVCVSize2D,
        iteration: i32,
        border_mode: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.morphology_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            workspace.as_raw(),
            morph_type,
            mask_size,
            anchor,
            iteration,
            border_mode,
        ))
    }
}

// ---------------------- Edge / Stat ----------------------

simple_op! {
    /// Canny edge detector.
    Canny, canny_create
}

impl Canny {
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        threshold_low: f64,
        threshold_high: f64,
        aperture_size: i32,
        l2_gradient: bool,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.canny_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            threshold_low,
            threshold_high,
            aperture_size,
            l2_gradient,
        ))
    }
}

simple_op! {
    /// Compute per-channel histograms.
    Histogram, histogram_create
}

impl Histogram {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        mask: &Tensor,
        histogram: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.histogram_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            mask.as_raw(),
            histogram.as_raw(),
        ))
    }
}

simple_op! {
    /// Histogram equalization.
    HistogramEq, histogram_eq_create
}

impl HistogramEq {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.histogram_eq_submit()?;
        check(cu(self.raw(), stream, input.as_raw(), output.as_raw()))
    }
}

simple_op! {
    /// Find min/max values + locations in a tensor.
    MinMaxLoc, min_max_loc_create
}

impl MinMaxLoc {
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        min_val: &Tensor,
        min_loc: &Tensor,
        num_min: &Tensor,
        max_val: &Tensor,
        max_loc: &Tensor,
        num_max: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.min_max_loc_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            min_val.as_raw(),
            min_loc.as_raw(),
            num_min.as_raw(),
            max_val.as_raw(),
            max_loc.as_raw(),
            num_max.as_raw(),
        ))
    }
}

// ---------------------- Thresholds ----------------------

/// Threshold operator (binary / trunc / tozero / Otsu / triangle).
#[derive(Debug)]
pub struct Threshold {
    op: OpHandle,
}

impl Threshold {
    pub fn new(threshold_type: u32, max_batch_size: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.threshold_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, threshold_type, max_batch_size) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        thresh: &Tensor,
        maxval: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.threshold_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            thresh.as_raw(),
            maxval.as_raw(),
        ))
    }
}

/// Adaptive threshold operator.
#[derive(Debug)]
pub struct AdaptiveThreshold {
    op: OpHandle,
}

impl AdaptiveThreshold {
    pub fn new(max_block_size: i32, max_batch_size: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.adaptive_threshold_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, max_block_size, max_batch_size) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        max_value: f64,
        adaptive_method: u32,
        threshold_type: u32,
        block_size: i32,
        c_scalar: f64,
    ) -> Result<()> {
        let lib = cvcuda()?;
        let cu = lib.adaptive_threshold_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            max_value,
            adaptive_method,
            threshold_type,
            block_size,
            c_scalar,
        ))
    }
}

// ---------------------- Color ----------------------

simple_op! {
    /// Apply a 3×4 / 4×4 color-transform matrix.
    ColorTwist, color_twist_create
}

impl ColorTwist {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        twist: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.color_twist_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            twist.as_raw(),
        ))
    }
}

simple_op! {
    /// Per-channel brightness + contrast adjustment.
    BrightnessContrast, brightness_contrast_create
}

impl BrightnessContrast {
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        brightness: &Tensor,
        contrast: &Tensor,
        brightness_shift: &Tensor,
        contrast_center: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.brightness_contrast_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            brightness.as_raw(),
            contrast.as_raw(),
            brightness_shift.as_raw(),
            contrast_center.as_raw(),
        ))
    }
}

/// Gamma-correction operator.
#[derive(Debug)]
pub struct GammaContrast {
    op: OpHandle,
}

impl GammaContrast {
    pub fn new(max_varshape_batch_size: i32, max_varshape_channel_count: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.gamma_contrast_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, max_varshape_batch_size, max_varshape_channel_count) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        gamma: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.gamma_contrast_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            gamma.as_raw(),
        ))
    }
}

// ---------------------- Composite / channel ----------------------

simple_op! {
    /// Alpha-mask compositing (`out = fg * mask + bg * (1 - mask)`).
    Composite, composite_create
}

impl Composite {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        foreground: &Tensor,
        background: &Tensor,
        fg_mask: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.composite_submit()?;
        check(cu(
            self.raw(),
            stream,
            foreground.as_raw(),
            background.as_raw(),
            fg_mask.as_raw(),
            output.as_raw(),
        ))
    }
}

simple_op! {
    /// Stack a VarShape image batch into a single tensor.
    Stack, stack_create
}

impl Stack {
    /// # Safety
    ///
    /// `stream` live; `input_batch` must be a valid `NVCVImageBatchHandle`.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input_batch: *mut c_void,
        output: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.stack_submit()?;
        check(cu(self.raw(), stream, input_batch, output.as_raw()))
    }
}

simple_op! {
    /// Reorder output channels (e.g. BGR→RGB swap).
    ChannelReorder, channel_reorder_create
}

impl ChannelReorder {
    /// # Safety
    ///
    /// `stream` live; `order.len()` matches the channel count.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        order: &[i32],
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.channel_reorder_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            order.as_ptr(),
            order.len() as i32,
        ))
    }
}

// ---------------------- Misc ----------------------

/// Erase rectangular regions (random or fixed content).
#[derive(Debug)]
pub struct Erase {
    op: OpHandle,
}

impl Erase {
    pub fn new(max_num_erasing_area: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.erase_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, max_num_erasing_area) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        anchor: &Tensor,
        erasing: &Tensor,
        values: &Tensor,
        imgidx: &Tensor,
        random: bool,
        seed: u32,
        inplace: bool,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.erase_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            anchor.as_raw(),
            erasing.as_raw(),
            values.as_raw(),
            imgidx.as_raw(),
            random,
            seed,
            inplace,
        ))
    }
}

/// Mask-based inpainting.
#[derive(Debug)]
pub struct Inpaint {
    op: OpHandle,
}

impl Inpaint {
    pub fn new(max_batch_size: i32, max_shape: NVCVSize2D) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.inpaint_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, max_batch_size, max_shape) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        masks: &Tensor,
        output: &Tensor,
        inpaint_radius: f64,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.inpaint_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            masks.as_raw(),
            output.as_raw(),
            inpaint_radius,
        ))
    }
}

simple_op! {
    /// `out = α * in1 + β * in2 + γ` with saturation.
    AddWeighted, add_weighted_create
}

impl AddWeighted {
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input1: &Tensor,
        alpha: f64,
        input2: &Tensor,
        beta: f64,
        gamma: f64,
        output: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.add_weighted_submit()?;
        check(cu(
            self.raw(),
            stream,
            input1.as_raw(),
            alpha,
            input2.as_raw(),
            beta,
            gamma,
            output.as_raw(),
        ))
    }
}

simple_op! {
    /// Non-maximum suppression over detection boxes + scores.
    NonMaxSuppression, non_max_suppression_create
}

impl NonMaxSuppression {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        scores: &Tensor,
        score_threshold: f32,
        iou_threshold: f32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.non_max_suppression_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            scores.as_raw(),
            score_threshold,
            iou_threshold,
        ))
    }
}

/// PadAndStack — pad a VarShape batch into a fixed-shape tensor.
#[derive(Debug)]
pub struct PadAndStack {
    op: OpHandle,
}

impl PadAndStack {
    pub fn new() -> Result<Self> {
        let op = create_op(|c| c.pad_and_stack_create())?;
        Ok(Self { op })
    }

    /// # Safety
    ///
    /// `stream` live; `input_batch` is a live `NVCVImageBatchHandle`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input_batch: *mut c_void,
        output: &Tensor,
        top: &Tensor,
        left: &Tensor,
        border_mode: i32,
        border_value: f32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.pad_and_stack_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input_batch,
            output.as_raw(),
            top.as_raw(),
            left.as_raw(),
            border_mode,
            border_value,
        ))
    }
}

// ------------------------------------------------------------------
// Round-2 operators
// ------------------------------------------------------------------

simple_op! {
    /// Pad an image by `top` / `left` border amounts.
    Pad, pad_create
}

impl Pad {
    /// # Safety
    ///
    /// `stream` live; `border_value.len() == 4`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        top: i32,
        left: i32,
        border_mode: i32,
        border_value: &[f32; 4],
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.pad_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            top,
            left,
            border_mode,
            border_value.as_ptr(),
        ))
    }
}

simple_op! {
    /// Joint bilateral filter — uses a *guide* image's edges to preserve
    /// structure while smoothing the primary input.
    JointBilateralFilter, joint_bilateral_filter_create
}

impl JointBilateralFilter {
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        input_color: &Tensor,
        output: &Tensor,
        diameter: i32,
        sigma_color: f32,
        sigma_space: f32,
        border_mode: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.joint_bilateral_filter_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            input_color.as_raw(),
            output.as_raw(),
            diameter,
            sigma_color,
            sigma_space,
            border_mode,
        ))
    }
}

/// Connected-component labeling.
#[derive(Debug)]
pub struct Label {
    op: OpHandle,
}

impl Label {
    pub fn new(max_labels_per_batch: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.label_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, max_labels_per_batch) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// Any `NVCVTensorHandle` can be passed as `NULL` (via
    /// `Tensor::from_raw(core::ptr::null_mut())` wrapped in
    /// `core::mem::ManuallyDrop`) for unused parameters.
    ///
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        bg_label: &Tensor,
        min_threshold: &Tensor,
        max_threshold: &Tensor,
        min_size: &Tensor,
        count: &Tensor,
        stats: &Tensor,
        mask: &Tensor,
        connectivity: i32,
        assign_labels: i32,
        mask_type: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.label_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            bg_label.as_raw(),
            min_threshold.as_raw(),
            max_threshold.as_raw(),
            min_size.as_raw(),
            count.as_raw(),
            stats.as_raw(),
            mask.as_raw(),
            connectivity,
            assign_labels,
            mask_type,
        ))
    }
}

/// Find contours of binary regions.
#[derive(Debug)]
pub struct FindContours {
    op: OpHandle,
}

impl FindContours {
    pub fn new(max_contour_size: NVCVSize2D, max_total_contour_count: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.find_contours_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, max_contour_size, max_total_contour_count) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        points: &Tensor,
        num_points: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.find_contours_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            points.as_raw(),
            num_points.as_raw(),
        ))
    }
}

/// Minimum-area rectangle around each contour.
#[derive(Debug)]
pub struct MinAreaRect {
    op: OpHandle,
}

impl MinAreaRect {
    pub fn new(max_contour_count: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.min_area_rect_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, max_contour_count) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        num_points_in_contour: &Tensor,
        total_contours: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.min_area_rect_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            num_points_in_contour.as_raw(),
            total_contours,
        ))
    }
}

simple_op! {
    /// Draw bounding boxes on the output image.
    BndBox, bndbox_create
}

impl BndBox {
    /// `bboxes` is a pointer to an `NVCVBndBoxesI` struct whose exact
    /// layout lives in the CV-CUDA headers — baracuda doesn't provide a
    /// typed builder for it yet.
    ///
    /// # Safety
    ///
    /// `bboxes` must point at a valid `NVCVBndBoxesI`; `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        bboxes: *const c_void,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.bndbox_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            bboxes,
        ))
    }
}

simple_op! {
    /// On-screen display — draw arbitrary text/line/rect overlays.
    Osd, osd_create
}

impl Osd {
    /// `elements` is a pointer to an `NVCVElements` struct (opaque).
    ///
    /// # Safety
    ///
    /// `elements` must be valid; `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        elements: *const c_void,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.osd_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            elements,
        ))
    }
}

/// Random-resized-crop — the PyTorch-style augmentation.
#[derive(Debug)]
pub struct RandomResizedCrop {
    op: OpHandle,
}

impl RandomResizedCrop {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        min_scale: f64,
        max_scale: f64,
        min_ratio: f64,
        max_ratio: f64,
        max_batch_size: i32,
        seed: u32,
    ) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.random_resized_crop_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut h,
                min_scale,
                max_scale,
                min_ratio,
                max_ratio,
                max_batch_size,
                seed,
            )
        })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        interp: Interpolation,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.random_resized_crop_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            interp.raw(),
        ))
    }
}

/// Gaussian noise.
#[derive(Debug)]
pub struct GaussianNoise {
    op: OpHandle,
}

impl GaussianNoise {
    pub fn new(max_batch_size: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.gaussian_noise_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, max_batch_size) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        mu: &Tensor,
        sigma: &Tensor,
        per_channel: bool,
        seed: u64,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.gaussian_noise_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            mu.as_raw(),
            sigma.as_raw(),
            per_channel,
            seed,
        ))
    }
}

/// Rhomboid noise.
#[derive(Debug)]
pub struct RhomboidNoise {
    op: OpHandle,
}

impl RhomboidNoise {
    pub fn new(max_batch_size: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.rhomboid_noise_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, max_batch_size) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        beta: &Tensor,
        seed: u64,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.rhomboid_noise_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            beta.as_raw(),
            seed,
        ))
    }
}

/// Salt-and-pepper noise.
#[derive(Debug)]
pub struct SaltAndPepperNoise {
    op: OpHandle,
}

impl SaltAndPepperNoise {
    pub fn new(max_batch_size: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.salt_and_pepper_noise_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, max_batch_size) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        salt_prob: &Tensor,
        pepper_prob: &Tensor,
        per_channel: bool,
        seed: u64,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.salt_and_pepper_noise_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            output.as_raw(),
            salt_prob.as_raw(),
            pepper_prob.as_raw(),
            per_channel,
            seed,
        ))
    }
}

simple_op! {
    /// Advanced colour-space conversion (`code` + per-frame `spec`).
    AdvCvtColor, adv_cvt_color_create
}

impl AdvCvtColor {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        code: i32,
        spec: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.adv_cvt_color_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            code,
            spec,
        ))
    }
}

/// SIFT — scale-invariant feature transform keypoint extractor.
#[derive(Debug)]
pub struct Sift {
    op: OpHandle,
}

impl Sift {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_shape: NVCVSize2D,
        max_num_features: i32,
        num_octave_layers: i32,
        contrast_threshold: f32,
        edge_threshold: f32,
        init_sigma: f32,
        flags: i32,
    ) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.sift_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe {
            cu(
                &mut h,
                max_shape,
                max_num_features,
                num_octave_layers,
                contrast_threshold,
                edge_threshold,
                init_sigma,
                flags,
            )
        })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        feature_coords: &Tensor,
        feature_metadata: &Tensor,
        feature_descriptors: &Tensor,
        num_features: &Tensor,
        num_octave_layers: i32,
        contrast_threshold: f32,
        edge_threshold: f32,
        init_sigma: f32,
        flags: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.sift_submit()?;
        check(cu(
            self.op.handle,
            stream,
            input.as_raw(),
            feature_coords.as_raw(),
            feature_metadata.as_raw(),
            feature_descriptors.as_raw(),
            num_features.as_raw(),
            num_octave_layers,
            contrast_threshold,
            edge_threshold,
            init_sigma,
            flags,
        ))
    }
}

simple_op! {
    /// High-quality resize for ML preprocessing (separable with anti-aliasing).
    HQResize, hq_resize_create
}

impl HQResize {
    /// # Safety
    ///
    /// `stream` live; `roi` may be null or point at an `NVCVHQResizeRoisF`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        min_interp: Interpolation,
        mag_interp: Interpolation,
        antialias: bool,
        roi: *const c_void,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.hq_resize_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            min_interp.raw(),
            mag_interp.raw(),
            antialias,
            roi,
        ))
    }
}

// ------------------------------------------------------------------
// Round-3 operators — finishing coverage
// ------------------------------------------------------------------

simple_op! {
    /// Fused crop + flip + normalize + layout-reformat — the classic ML
    /// video-preproc bundle.
    CropFlipNormalizeReformat, crop_flip_normalize_reformat_create
}

impl CropFlipNormalizeReformat {
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        crop_rect: &Tensor,
        interp: Interpolation,
        flip_code: &Tensor,
        base: &Tensor,
        scale: &Tensor,
        global_scale: f32,
        shift: f32,
        epsilon: f32,
        flags: u32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.crop_flip_normalize_reformat_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            crop_rect.as_raw(),
            interp.raw(),
            flip_code.as_raw(),
            base.as_raw(),
            scale.as_raw(),
            global_scale,
            shift,
            epsilon,
            flags,
        ))
    }
}

simple_op! {
    /// Guided filter — edge-preserving filter that uses `guide` to
    /// steer smoothing of `input`.
    GuidedFilter, guided_filter_create
}

impl GuidedFilter {
    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        guide: &Tensor,
        output: &Tensor,
        radius: i32,
        eps: f32,
        border_mode: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.guided_filter_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            guide.as_raw(),
            output.as_raw(),
            radius,
            eps,
            border_mode,
        ))
    }
}

/// Pairwise feature-descriptor matcher (e.g. for SIFT output).
#[derive(Debug)]
pub struct PairwiseMatcher {
    op: OpHandle,
}

impl PairwiseMatcher {
    /// `algo_choice`: 0 = brute-force, 1 = brute-force-crosscheck.
    pub fn new(algo_choice: i32) -> Result<Self> {
        let c = cvcuda()?;
        let cu = c.pairwise_matcher_create()?;
        let mut h: CVCUDA_OperatorHandle = core::ptr::null_mut();
        check(unsafe { cu(&mut h, algo_choice) })?;
        Ok(Self {
            op: OpHandle { handle: h },
        })
    }

    /// # Safety
    ///
    /// `stream` live.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        set1: &Tensor,
        set2: &Tensor,
        num_set1: &Tensor,
        num_set2: &Tensor,
        matches: &Tensor,
        num_matches: &Tensor,
        distances: &Tensor,
        cross_check: bool,
        match_per_point: i32,
        norm_type: i32,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.pairwise_matcher_submit()?;
        check(cu(
            self.op.handle,
            stream,
            set1.as_raw(),
            set2.as_raw(),
            num_set1.as_raw(),
            num_set2.as_raw(),
            matches.as_raw(),
            num_matches.as_raw(),
            distances.as_raw(),
            cross_check,
            match_per_point,
            norm_type,
        ))
    }
}

simple_op! {
    /// Hausdorff distance between two point sets.
    HausdorffDistance, hausdorff_distance_create
}

impl HausdorffDistance {
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        set1: &Tensor,
        set2: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.hausdorff_distance_submit()?;
        check(cu(
            self.raw(),
            stream,
            set1.as_raw(),
            set2.as_raw(),
            output.as_raw(),
        ))
    }
}

simple_op! {
    /// Fused resize + crop + dtype-convert + layout-reformat.
    ResizeCropConvertReformat, resize_crop_convert_reformat_create
}

impl ResizeCropConvertReformat {
    /// `crop_pos` points at an `NVCVPointI { x, y }` (2 × i32).
    ///
    /// # Safety
    ///
    /// `stream` live; `crop_pos` valid.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        resize_size: NVCVSize2D,
        interp: Interpolation,
        crop_pos: *const c_void,
        manip: i32,
        scale: f32,
        offset: f32,
        cast_to_f32: bool,
        src_cast_to_f32: bool,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.resize_crop_convert_reformat_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            resize_size,
            interp.raw(),
            crop_pos,
            manip,
            scale,
            offset,
            cast_to_f32,
            src_cast_to_f32,
        ))
    }
}

simple_op! {
    /// Per-frame rotation of a batch tensor.
    RotateBatch, rotate_batch_create
}

impl RotateBatch {
    /// `angle_deg` is a per-frame `Tensor<f64>`; `shift` is a
    /// per-frame `Tensor<(f64, f64)>`.
    ///
    /// # Safety
    ///
    /// `stream` live.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        angle_deg: &Tensor,
        shift: &Tensor,
        interp: Interpolation,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.rotate_batch_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            angle_deg.as_raw(),
            shift.as_raw(),
            interp.raw(),
        ))
    }
}

simple_op! {
    /// Per-frame 2×3 affine warp across a batch tensor.
    WarpAffineBatch, warp_affine_batch_create
}

impl WarpAffineBatch {
    /// `xform` is `[batch, 2, 3]` f32.
    ///
    /// # Safety
    ///
    /// `stream` live; `border_value.len() == 4`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input: &Tensor,
        output: &Tensor,
        xform: &Tensor,
        flags: i32,
        border_mode: i32,
        border_value: &[f32; 4],
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.warp_affine_batch_submit()?;
        check(cu(
            self.raw(),
            stream,
            input.as_raw(),
            output.as_raw(),
            xform.as_raw(),
            flags,
            border_mode,
            border_value.as_ptr(),
        ))
    }
}

simple_op! {
    /// Resize on a variable-shape image batch (different sizes per item).
    ResizeVarShape, resize_var_shape_create
}

impl ResizeVarShape {
    /// # Safety
    ///
    /// `input_batch`/`output_batch` are live `NVCVImageBatchHandle`s.
    pub unsafe fn submit(
        &self,
        stream: *mut c_void,
        input_batch: *mut c_void,
        output_batch: *mut c_void,
        interp: Interpolation,
    ) -> Result<()> {
        let c = cvcuda()?;
        let cu = c.resize_var_shape_submit()?;
        check(cu(
            self.raw(),
            stream,
            input_batch,
            output_batch,
            interp.raw(),
        ))
    }
}
