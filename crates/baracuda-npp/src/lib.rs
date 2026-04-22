//! Safe Rust wrappers for NVIDIA NPP (Performance Primitives).
//!
//! NPP is organized into ~10 separate shared libraries:
//!
//! - `nppc` — core (version info).
//! - `npps` — 1-D signal processing (covered by [`signal`]).
//! - `nppial` / `nppi*` — 2-D image processing (covered by [`image`]).
//!
//! NPP has thousands of functions for every permutation of element type
//! (8u/16u/16s/32s/32f) × channel count (C1/C3/C4) × region-of-interest
//! (R). This crate wraps a curated subset: the 32f and 8u C1R variants
//! for arithmetic, geometric, color-conversion, filter, and statistics
//! ops. Further variants can be added trivially by following the
//! existing pattern.

#![warn(missing_debug_implementations)]

use core::ffi::c_void;

use baracuda_driver::DeviceBuffer;
use baracuda_npp_sys::{nppc, npps, NppLibraryVersion, NppStatus};

pub use baracuda_npp_sys::{NppiInterpolationMode, NppiPoint, NppiRect, NppiSize};

/// Error type for NPP operations.
pub type Error = baracuda_core::Error<NppStatus>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: NppStatus) -> Result<()> {
    Error::check(status)
}

/// NPP library version (queries `nppc`).
pub fn version() -> Result<NppLibraryVersion> {
    let n = nppc()?;
    let cu = n.npp_get_lib_version()?;
    let ptr = unsafe { cu() };
    if ptr.is_null() {
        return Err(Error::Status {
            status: NppStatus(-4),
        });
    }
    Ok(unsafe { *ptr })
}

/// 1-D signal-processing ops (from `npps`).
pub mod signal {
    use super::*;

    /// `src_dst[i] += src[i]` (32-bit float).
    pub fn add_32f_in_place(
        src: &DeviceBuffer<f32>,
        src_dst: &mut DeviceBuffer<f32>,
        n: i32,
    ) -> Result<()> {
        assert!(src.len() >= n as usize);
        assert!(src_dst.len() >= n as usize);
        let s = npps()?;
        let cu = s.npps_add_32f_i()?;
        check(unsafe {
            cu(
                src.as_raw().0 as *const f32,
                src_dst.as_raw().0 as *mut f32,
                n,
            )
        })
    }

    /// `src_dst[i] -= src[i]` (32-bit float).
    pub fn sub_32f_in_place(
        src: &DeviceBuffer<f32>,
        src_dst: &mut DeviceBuffer<f32>,
        n: i32,
    ) -> Result<()> {
        assert!(src.len() >= n as usize);
        assert!(src_dst.len() >= n as usize);
        let s = npps()?;
        let cu = s.npps_sub_32f_i()?;
        check(unsafe {
            cu(
                src.as_raw().0 as *const f32,
                src_dst.as_raw().0 as *mut f32,
                n,
            )
        })
    }

    /// `src_dst[i] *= src[i]` (32-bit float).
    pub fn mul_32f_in_place(
        src: &DeviceBuffer<f32>,
        src_dst: &mut DeviceBuffer<f32>,
        n: i32,
    ) -> Result<()> {
        assert!(src.len() >= n as usize);
        assert!(src_dst.len() >= n as usize);
        let s = npps()?;
        let cu = s.npps_mul_32f_i()?;
        check(unsafe {
            cu(
                src.as_raw().0 as *const f32,
                src_dst.as_raw().0 as *mut f32,
                n,
            )
        })
    }

    /// Return the device scratch-buffer size (in bytes) that
    /// [`sum_32f`] needs for an `n`-element reduction.
    pub fn sum_buffer_size_32f(n: i32) -> Result<usize> {
        let s = npps()?;
        let cu = s.npps_sum_get_buffer_size_32f()?;
        let mut bytes: i32 = 0;
        check(unsafe { cu(n, &mut bytes) })?;
        Ok(bytes as usize)
    }

    /// Compute the sum of an `f32` signal, writing the scalar into
    /// `sum_out[0]`. Caller provides the scratch buffer (use
    /// [`sum_buffer_size_32f`] to size it).
    pub fn sum_32f(
        src: &DeviceBuffer<f32>,
        n: i32,
        sum_out: &mut DeviceBuffer<f32>,
        scratch: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        assert!(src.len() >= n as usize);
        assert!(!sum_out.is_empty());
        let s = npps()?;
        let cu = s.npps_sum_32f()?;
        check(unsafe {
            cu(
                src.as_raw().0 as *const f32,
                n,
                sum_out.as_raw().0 as *mut f32,
                scratch.as_raw().0 as *mut u8,
            )
        })
    }

    /// Scratch size for [`min_max_32f`].
    pub fn min_max_buffer_size_32f(n: i32) -> Result<usize> {
        let s = npps()?;
        let cu = s.npps_min_max_get_buffer_size_32f()?;
        let mut bytes: i32 = 0;
        check(unsafe { cu(n, &mut bytes) })?;
        Ok(bytes as usize)
    }

    /// Compute (min, max) of an `f32` signal; caller provides scratch.
    pub fn min_max_32f(
        src: &DeviceBuffer<f32>,
        n: i32,
        min_out: &mut DeviceBuffer<f32>,
        max_out: &mut DeviceBuffer<f32>,
        scratch: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let s = npps()?;
        let cu = s.npps_min_max_32f()?;
        check(unsafe {
            cu(
                src.as_raw().0 as *const f32,
                n,
                min_out.as_raw().0 as *mut f32,
                max_out.as_raw().0 as *mut f32,
                scratch.as_raw().0 as *mut u8,
            )
        })
    }
}

/// 2-D image-processing ops (from `nppial`/`nppig`/`nppicc`/`nppif`/`nppist`).
pub mod image {
    use super::*;
    use baracuda_npp_sys::{nppial, nppicc, nppif, nppig, nppist};

    /// `dst = src1 + src2` for 32f single-channel images (ROI-based).
    ///
    /// Steps are in bytes between rows. Pass the pitch your allocator
    /// returned (typically `width * 4` for packed data, or the
    /// pitched-alloc pitch from [`baracuda_runtime::memcpy2d::PitchedBuffer`]).
    ///
    /// # Safety
    ///
    /// All pointers must be device-addressable and cover `size.height`
    /// rows of `size.width * 4` bytes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn add_32f_c1r(
        src1: *const f32,
        src1_step: i32,
        src2: *const f32,
        src2_step: i32,
        dst: *mut f32,
        dst_step: i32,
        size: NppiSize,
    ) -> Result<()> {
        let l = nppial()?;
        let cu = l.nppi_add_32f_c1r()?;
        check(cu(src1, src1_step, src2, src2_step, dst, dst_step, size))
    }

    /// `dst = src1 * src2` for 32f single-channel images.
    ///
    /// # Safety
    ///
    /// Same as [`add_32f_c1r`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn mul_32f_c1r(
        src1: *const f32,
        src1_step: i32,
        src2: *const f32,
        src2_step: i32,
        dst: *mut f32,
        dst_step: i32,
        size: NppiSize,
    ) -> Result<()> {
        let l = nppial()?;
        let cu = l.nppi_mul_32f_c1r()?;
        check(cu(src1, src1_step, src2, src2_step, dst, dst_step, size))
    }

    /// `dst = (src1 + src2) >> scale_factor` for 8u single-channel.
    ///
    /// # Safety
    ///
    /// Same as [`add_32f_c1r`], with steps in bytes for 8u data.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn add_8u_c1r_sfs(
        src1: *const u8,
        src1_step: i32,
        src2: *const u8,
        src2_step: i32,
        dst: *mut u8,
        dst_step: i32,
        size: NppiSize,
        scale_factor: i32,
    ) -> Result<()> {
        let l = nppial()?;
        let cu = l.nppi_add_8u_c1r_sfs()?;
        check(cu(
            src1,
            src1_step,
            src2,
            src2_step,
            dst,
            dst_step,
            size,
            scale_factor,
        ))
    }

    /// 8u single-channel image resize with the given interpolation mode.
    ///
    /// # Safety
    ///
    /// `src` must cover `src_size` at `src_step` pitch; `dst` must cover
    /// `dst_size` at `dst_step`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn resize_8u_c1r(
        src: *const u8,
        src_step: i32,
        src_size: NppiSize,
        src_rect: NppiRect,
        dst: *mut u8,
        dst_step: i32,
        dst_size: NppiSize,
        dst_rect: NppiRect,
        interpolation: i32,
    ) -> Result<()> {
        let l = nppig()?;
        let cu = l.nppi_resize_8u_c1r()?;
        check(cu(
            src,
            src_step,
            src_size,
            src_rect,
            dst,
            dst_step,
            dst_size,
            dst_rect,
            interpolation,
        ))
    }

    /// 32f single-channel image resize.
    ///
    /// # Safety
    ///
    /// Same as [`resize_8u_c1r`] with 4-byte pixels.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn resize_32f_c1r(
        src: *const f32,
        src_step: i32,
        src_size: NppiSize,
        src_rect: NppiRect,
        dst: *mut f32,
        dst_step: i32,
        dst_size: NppiSize,
        dst_rect: NppiRect,
        interpolation: i32,
    ) -> Result<()> {
        let l = nppig()?;
        let cu = l.nppi_resize_32f_c1r()?;
        check(cu(
            src,
            src_step,
            src_size,
            src_rect,
            dst,
            dst_step,
            dst_size,
            dst_rect,
            interpolation,
        ))
    }

    /// Convert a packed RGB-8u image to single-channel grayscale.
    ///
    /// # Safety
    ///
    /// `src` must cover `size.height × src_step` bytes (RGB pixels are
    /// 3 bytes each); `dst` must cover `size.height × dst_step` bytes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rgb_to_gray_8u(
        src: *const u8,
        src_step: i32,
        dst: *mut u8,
        dst_step: i32,
        size: NppiSize,
    ) -> Result<()> {
        let l = nppicc()?;
        let cu = l.nppi_rgb_to_gray_8u_c3c1r()?;
        check(cu(src, src_step, dst, dst_step, size))
    }

    /// Same as [`rgb_to_gray_8u`] but BGR order (OpenCV convention).
    ///
    /// # Safety
    ///
    /// Same as [`rgb_to_gray_8u`].
    pub unsafe fn bgr_to_gray_8u(
        src: *const u8,
        src_step: i32,
        dst: *mut u8,
        dst_step: i32,
        size: NppiSize,
    ) -> Result<()> {
        let l = nppicc()?;
        let cu = l.nppi_bgr_to_gray_8u_c3c1r()?;
        check(cu(src, src_step, dst, dst_step, size))
    }

    /// Apply an averaging (box) filter of size `mask_size`.
    ///
    /// # Safety
    ///
    /// Same as [`add_32f_c1r`]. NPP requires an apron of ceil(mask/2)
    /// pixels on the source side.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn filter_box_8u_c1r(
        src: *const u8,
        src_step: i32,
        dst: *mut u8,
        dst_step: i32,
        dst_roi: NppiSize,
        mask_size: NppiSize,
        anchor: NppiPoint,
    ) -> Result<()> {
        let l = nppif()?;
        let cu = l.nppi_filter_box_8u_c1r()?;
        check(cu(src, src_step, dst, dst_step, dst_roi, mask_size, anchor))
    }

    /// Report the device-scratch buffer size required by
    /// [`sum_32f_c1r`] for an image of size `roi`.
    pub fn sum_buffer_size_32f_c1r(roi: NppiSize) -> Result<usize> {
        let l = nppist()?;
        let cu = l.nppi_sum_get_buffer_host_size_32f_c1r()?;
        let mut bytes: i32 = 0;
        check(unsafe { cu(roi, &mut bytes) })?;
        Ok(bytes as usize)
    }

    /// Sum of a 32f single-channel image over `roi`, writing the
    /// scalar result into `sum_out` (device buffer of len ≥ 1).
    /// Caller provides the scratch buffer (use
    /// [`sum_buffer_size_32f_c1r`] to size it).
    ///
    /// # Safety
    ///
    /// `src` must cover `roi.height × src_step` bytes.
    pub unsafe fn sum_32f_c1r(
        src: *const f32,
        src_step: i32,
        roi: NppiSize,
        sum_out: *mut f64,
        scratch: *mut u8,
    ) -> Result<()> {
        let l = nppist()?;
        let cu = l.nppi_sum_32f_c1r()?;
        check(cu(src, src_step, roi, scratch, sum_out))
    }
}

// Deprecated top-level alias.
#[deprecated(since = "0.2.0", note = "use baracuda_npp::signal::add_32f_in_place")]
pub fn adds_32f_in_place(
    src: &DeviceBuffer<f32>,
    src_dst: &mut DeviceBuffer<f32>,
    n: i32,
) -> Result<()> {
    signal::add_32f_in_place(src, src_dst, n)
}

// Silence a "unused import" if nothing in this module currently needs `c_void`.
#[allow(dead_code)]
fn _touch() -> *mut c_void {
    core::ptr::null_mut()
}
