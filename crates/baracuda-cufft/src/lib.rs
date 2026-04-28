//! Safe Rust wrappers for NVIDIA cuFFT.
//!
//! v0.1 covers `cufftPlan1d`/`cufftPlan2d`/`cufftPlan3d` and the R2C/C2R/C2C
//! single-precision transforms. Multi-GPU (`cufftXt`) and batched
//! descriptor-style plans land in a follow-up.
//!
//! ```no_run
//! use baracuda_driver::{Context, Device, DeviceBuffer};
//! use baracuda_cufft::{Plan1d, Transform};
//!
//! # fn demo() -> Result<(), Box<dyn std::error::Error>> {
//! let device = Device::get(0)?;
//! let ctx = Context::new(&device)?;
//! let host: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.05).sin()).collect();
//! let mut d_in = DeviceBuffer::from_slice(&ctx, &host)?;
//! let mut d_out: DeviceBuffer<baracuda_types::Complex32> =
//!     DeviceBuffer::new(&ctx, host.len() / 2 + 1)?;
//!
//! let plan = Plan1d::new(host.len() as i32, Transform::R2C, 1)?;
//! plan.exec_r2c(&mut d_in, &mut d_out)?;
//! # Ok(()) }
//! ```

#![warn(missing_debug_implementations)]

use baracuda_cufft_sys::{
    cufft, cufftComplex, cufftDoubleComplex, cufftHandle, cufftResult, cufftType,
};
use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_types::{Complex32, Complex64};

/// Error type for cuFFT operations.
pub type Error = baracuda_core::Error<cufftResult>;
/// Result alias.
pub type Result<T, E = Error> = core::result::Result<T, E>;

#[inline]
fn check(status: cufftResult) -> Result<()> {
    Error::check(status)
}

/// Transform kind.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Transform {
    /// Real → Complex (forward), f32.
    R2C,
    /// Complex → Real (inverse), f32.
    C2R,
    /// Complex → Complex (f32, direction passed at exec time).
    C2C,
    /// Double Real → Complex (forward), f64.
    D2Z,
    /// Complex → Double Real (inverse), f64.
    Z2D,
    /// Complex → Complex (f64, direction passed at exec time).
    Z2Z,
}

impl Transform {
    fn raw(self) -> cufftType {
        match self {
            Transform::R2C => cufftType::R2C,
            Transform::C2R => cufftType::C2R,
            Transform::C2C => cufftType::C2C,
            Transform::D2Z => cufftType::D2Z,
            Transform::Z2D => cufftType::Z2D,
            Transform::Z2Z => cufftType::Z2Z,
        }
    }
}

/// Direction for `C2C` transforms.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum Direction {
    #[default]
    Forward,
    Inverse,
}

impl Direction {
    fn raw(self) -> core::ffi::c_int {
        match self {
            Direction::Forward => baracuda_cufft_sys::CUFFT_FORWARD,
            Direction::Inverse => baracuda_cufft_sys::CUFFT_INVERSE,
        }
    }
}

/// A 1-D cuFFT plan.
pub struct Plan1d {
    handle: cufftHandle,
}

unsafe impl Send for Plan1d {}

impl core::fmt::Debug for Plan1d {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Plan1d")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Plan1d {
    /// Create a 1-D plan of length `nx` and `batch` parallel transforms.
    pub fn new(nx: i32, transform: Transform, batch: i32) -> Result<Self> {
        let c = cufft()?;
        let cu = c.cufft_plan_1d()?;
        let mut plan: cufftHandle = 0;
        check(unsafe { cu(&mut plan, nx, transform.raw(), batch) })?;
        Ok(Self { handle: plan })
    }

    /// Bind subsequent exec calls on this plan to `stream`.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_set_stream()?;
        check(unsafe { cu(self.handle, stream.as_raw() as _) })
    }

    /// Execute a real-to-complex transform.
    pub fn exec_r2c(
        &self,
        input: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<Complex32>,
    ) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_exec_r2c()?;
        check(unsafe {
            cu(
                self.handle,
                input.as_raw().0 as *mut f32,
                output.as_raw().0 as *mut cufftComplex,
            )
        })
    }

    /// Execute a complex-to-real transform.
    pub fn exec_c2r(
        &self,
        input: &mut DeviceBuffer<Complex32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_exec_c2r()?;
        check(unsafe {
            cu(
                self.handle,
                input.as_raw().0 as *mut cufftComplex,
                output.as_raw().0 as *mut f32,
            )
        })
    }

    /// Execute a complex-to-complex transform in the given direction.
    pub fn exec_c2c(
        &self,
        input: &mut DeviceBuffer<Complex32>,
        output: &mut DeviceBuffer<Complex32>,
        direction: Direction,
    ) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_exec_c2c()?;
        check(unsafe {
            cu(
                self.handle,
                input.as_raw().0 as *mut cufftComplex,
                output.as_raw().0 as *mut cufftComplex,
                direction.raw(),
            )
        })
    }

    /// Raw `cufftHandle`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cufftHandle {
        self.handle
    }
}

impl Drop for Plan1d {
    fn drop(&mut self) {
        if let Ok(c) = cufft() {
            if let Ok(cu) = c.cufft_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// A 2-D cuFFT plan.
pub struct Plan2d {
    handle: cufftHandle,
}

unsafe impl Send for Plan2d {}

impl core::fmt::Debug for Plan2d {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Plan2d")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Plan2d {
    /// Create a 2-D plan of dimensions `nx × ny`.
    pub fn new(nx: i32, ny: i32, transform: Transform) -> Result<Self> {
        let c = cufft()?;
        let cu = c.cufft_plan_2d()?;
        let mut plan: cufftHandle = 0;
        check(unsafe { cu(&mut plan, nx, ny, transform.raw()) })?;
        Ok(Self { handle: plan })
    }

    /// Execute a complex-to-complex 2D transform.
    pub fn exec_c2c(
        &self,
        input: &mut DeviceBuffer<Complex32>,
        output: &mut DeviceBuffer<Complex32>,
        direction: Direction,
    ) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_exec_c2c()?;
        check(unsafe {
            cu(
                self.handle,
                input.as_raw().0 as *mut cufftComplex,
                output.as_raw().0 as *mut cufftComplex,
                direction.raw(),
            )
        })
    }

    /// Raw handle.
    #[inline]
    pub fn as_raw(&self) -> cufftHandle {
        self.handle
    }
}

impl Drop for Plan2d {
    fn drop(&mut self) {
        if let Ok(c) = cufft() {
            if let Ok(cu) = c.cufft_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// Owned 3-D cuFFT plan.
pub struct Plan3d {
    handle: cufftHandle,
}

unsafe impl Send for Plan3d {}

impl core::fmt::Debug for Plan3d {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Plan3d")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Plan3d {
    /// Create a 3-D plan of dimensions `nx × ny × nz`.
    pub fn new(nx: i32, ny: i32, nz: i32, transform: Transform) -> Result<Self> {
        let c = cufft()?;
        let cu = c.cufft_plan_3d()?;
        let mut plan: cufftHandle = 0;
        check(unsafe { cu(&mut plan, nx, ny, nz, transform.raw()) })?;
        Ok(Self { handle: plan })
    }

    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_set_stream()?;
        check(unsafe { cu(self.handle, stream.as_raw() as _) })
    }

    pub fn exec_c2c(
        &self,
        input: &mut DeviceBuffer<Complex32>,
        output: &mut DeviceBuffer<Complex32>,
        direction: Direction,
    ) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_exec_c2c()?;
        check(unsafe {
            cu(
                self.handle,
                input.as_raw().0 as *mut cufftComplex,
                output.as_raw().0 as *mut cufftComplex,
                direction.raw(),
            )
        })
    }

    #[inline]
    pub fn as_raw(&self) -> cufftHandle {
        self.handle
    }
}

impl Drop for Plan3d {
    fn drop(&mut self) {
        if let Ok(c) = cufft() {
            if let Ok(cu) = c.cufft_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

/// cuFFT library version, e.g. `11300` for cuFFT 11.3.0.
pub fn version() -> Result<i32> {
    let c = cufft()?;
    let cu = c.cufft_get_version()?;
    let mut v: core::ffi::c_int = 0;
    check(unsafe { cu(&mut v) })?;
    Ok(v)
}

// =======================================================================
// Double-precision exec + PlanMany (batched) + XT multi-GPU
// =======================================================================

macro_rules! exec_z_impls {
    ($plan:ty) => {
        impl $plan {
            /// Execute D → Z (double-precision R2C). Plan must have been
            /// built with `Transform::D2Z`.
            pub fn exec_d2z(
                &self,
                input: &mut DeviceBuffer<f64>,
                output: &mut DeviceBuffer<Complex64>,
            ) -> Result<()> {
                let c = cufft()?;
                let cu = c.cufft_exec_d2z()?;
                check(unsafe {
                    cu(
                        self.handle,
                        input.as_raw().0 as *mut f64,
                        output.as_raw().0 as *mut cufftDoubleComplex,
                    )
                })
            }

            /// Execute Z → D (double-precision C2R).
            pub fn exec_z2d(
                &self,
                input: &mut DeviceBuffer<Complex64>,
                output: &mut DeviceBuffer<f64>,
            ) -> Result<()> {
                let c = cufft()?;
                let cu = c.cufft_exec_z2d()?;
                check(unsafe {
                    cu(
                        self.handle,
                        input.as_raw().0 as *mut cufftDoubleComplex,
                        output.as_raw().0 as *mut f64,
                    )
                })
            }

            /// Execute Z → Z (double-precision C2C). Direction passed at exec time.
            pub fn exec_z2z(
                &self,
                input: &mut DeviceBuffer<Complex64>,
                output: &mut DeviceBuffer<Complex64>,
                direction: Direction,
            ) -> Result<()> {
                let c = cufft()?;
                let cu = c.cufft_exec_z2z()?;
                check(unsafe {
                    cu(
                        self.handle,
                        input.as_raw().0 as *mut cufftDoubleComplex,
                        output.as_raw().0 as *mut cufftDoubleComplex,
                        direction.raw(),
                    )
                })
            }
        }
    };
}

exec_z_impls!(Plan1d);
exec_z_impls!(Plan2d);

/// A batched / many-rank plan (`cufftPlanMany`). Handles arbitrary
/// rank + advanced-data-layout transforms.
#[derive(Debug)]
pub struct PlanMany {
    handle: cufftHandle,
}

impl PlanMany {
    /// Construct a batched plan. `n[rank]` is the transform shape;
    /// `inembed` / `onembed` are the actual memory layouts of in/out
    /// (pass `None` for packed). `istride`/`ostride` are element strides
    /// between successive elements; `idist`/`odist` are element strides
    /// between successive batches.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rank: i32,
        n: &mut [i32],
        inembed: Option<&mut [i32]>,
        istride: i32,
        idist: i32,
        onembed: Option<&mut [i32]>,
        ostride: i32,
        odist: i32,
        ty: Transform,
        batch: i32,
    ) -> Result<Self> {
        let c = cufft()?;
        let cu = c.cufft_plan_many()?;
        let mut h: cufftHandle = 0;
        check(unsafe {
            cu(
                &mut h,
                rank,
                n.as_mut_ptr(),
                inembed.map_or(core::ptr::null_mut(), |s| s.as_mut_ptr()),
                istride,
                idist,
                onembed.map_or(core::ptr::null_mut(), |s| s.as_mut_ptr()),
                ostride,
                odist,
                ty.raw(),
                batch,
            )
        })?;
        Ok(Self { handle: h })
    }

    #[inline]
    pub fn as_raw(&self) -> cufftHandle {
        self.handle
    }

    /// Bind the plan to a CUDA stream.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_set_stream()?;
        check(unsafe { cu(self.handle, stream.as_raw() as _) })
    }
}

impl Drop for PlanMany {
    fn drop(&mut self) {
        if let Ok(c) = cufft() {
            if let Ok(cu) = c.cufft_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

exec_z_impls!(PlanMany);

impl PlanMany {
    /// Execute R → C (single-precision R2C).
    pub fn exec_r2c(
        &self,
        input: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<Complex32>,
    ) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_exec_r2c()?;
        check(unsafe {
            cu(
                self.handle,
                input.as_raw().0 as *mut f32,
                output.as_raw().0 as *mut cufftComplex,
            )
        })
    }

    /// Execute C → R (single-precision C2R).
    pub fn exec_c2r(
        &self,
        input: &mut DeviceBuffer<Complex32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_exec_c2r()?;
        check(unsafe {
            cu(
                self.handle,
                input.as_raw().0 as *mut cufftComplex,
                output.as_raw().0 as *mut f32,
            )
        })
    }

    /// Execute C → C.
    pub fn exec_c2c(
        &self,
        input: &mut DeviceBuffer<Complex32>,
        output: &mut DeviceBuffer<Complex32>,
        direction: Direction,
    ) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_exec_c2c()?;
        check(unsafe {
            cu(
                self.handle,
                input.as_raw().0 as *mut cufftComplex,
                output.as_raw().0 as *mut cufftComplex,
                direction.raw(),
            )
        })
    }
}

/// Sizing estimates (workspace bytes) for a plan shape.
pub fn estimate_1d(nx: i32, ty: Transform, batch: i32) -> Result<usize> {
    let c = cufft()?;
    let cu = c.cufft_estimate_1d()?;
    let mut s: usize = 0;
    check(unsafe { cu(nx, ty.raw(), batch, &mut s) })?;
    Ok(s)
}

pub fn estimate_2d(nx: i32, ny: i32, ty: Transform) -> Result<usize> {
    let c = cufft()?;
    let cu = c.cufft_estimate_2d()?;
    let mut s: usize = 0;
    check(unsafe { cu(nx, ny, ty.raw(), &mut s) })?;
    Ok(s)
}

pub fn estimate_3d(nx: i32, ny: i32, nz: i32, ty: Transform) -> Result<usize> {
    let c = cufft()?;
    let cu = c.cufft_estimate_3d()?;
    let mut s: usize = 0;
    check(unsafe { cu(nx, ny, nz, ty.raw(), &mut s) })?;
    Ok(s)
}

/// Multi-GPU (XT) extension helpers. Use these to distribute a cuFFT
/// plan across multiple GPUs via `cufftXtSetGPUs` + `cufftXtExec`.
pub mod xt {
    use super::*;

    /// Spread a plan across `which_gpus` (CUDA device ordinals).
    ///
    /// # Safety
    ///
    /// `plan` must be a fresh (unexecuted) handle; all ordinals in
    /// `which_gpus` must be live CUDA devices.
    pub unsafe fn set_gpus(plan: cufftHandle, which_gpus: &mut [i32]) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_xt_set_gpus()?;
        check(cu(plan, which_gpus.len() as i32, which_gpus.as_mut_ptr()))
    }

    /// Allocate a multi-GPU `cudaLibXtDesc*` matching the plan.
    /// Returns an opaque pointer that must be freed with [`free`].
    ///
    /// # Safety
    ///
    /// `plan` must have been configured with [`set_gpus`] first.
    pub unsafe fn malloc(
        plan: cufftHandle,
        subformat: i32,
    ) -> Result<*mut core::ffi::c_void> {
        let c = cufft()?;
        let cu = c.cufft_xt_malloc()?;
        let mut desc: *mut core::ffi::c_void = core::ptr::null_mut();
        check(cu(plan, &mut desc, subformat))?;
        Ok(desc)
    }

    /// Free an XT descriptor from [`malloc`].
    ///
    /// # Safety
    ///
    /// `desc` must come from [`malloc`].
    pub unsafe fn free(desc: *mut core::ffi::c_void) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_xt_free()?;
        check(cu(desc))
    }

    /// Multi-GPU memcpy between host / device / XT descriptors.
    ///
    /// # Safety
    ///
    /// Pointer kinds and `ty` must agree.
    pub unsafe fn memcpy(
        plan: cufftHandle,
        dst: *mut core::ffi::c_void,
        src: *mut core::ffi::c_void,
        ty: i32,
    ) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_xt_memcpy()?;
        check(cu(plan, dst, src, ty))
    }

    /// Execute the plan on its XT descriptors.
    ///
    /// # Safety
    ///
    /// `input` / `output` must be `cudaLibXtDesc*` pointers matching the plan.
    pub unsafe fn exec_descriptor(
        plan: cufftHandle,
        input: *mut core::ffi::c_void,
        output: *mut core::ffi::c_void,
        direction: Direction,
    ) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_xt_exec_descriptor()?;
        check(cu(plan, input, output, direction.raw()))
    }
}

/// Set a user-allocated scratch work area (`cufftSetWorkArea`).
///
/// # Safety
///
/// `plan` must have `SetAutoAllocation(false)` first; `work_area` must
/// be a live device pointer.
pub unsafe fn set_work_area(plan: cufftHandle, work_area: *mut core::ffi::c_void) -> Result<()> {
    let c = cufft()?;
    let cu = c.cufft_set_work_area()?;
    check(cu(plan, work_area))
}

/// Disable / re-enable automatic work-area allocation.
pub fn set_auto_allocation(plan: cufftHandle, auto: bool) -> Result<()> {
    let c = cufft()?;
    let cu = c.cufft_set_auto_allocation()?;
    check(unsafe { cu(plan, if auto { 1 } else { 0 }) })
}

/// Scratch bytes this plan currently needs.
pub fn get_size(plan: cufftHandle) -> Result<usize> {
    let c = cufft()?;
    let cu = c.cufft_get_size()?;
    let mut s: usize = 0;
    check(unsafe { cu(plan, &mut s) })?;
    Ok(s)
}

// ============================================================================
// Two-step plan creation: cufftCreate + cufftMakePlan*
// ============================================================================

/// Generic cuFFT plan that supports the modern two-step creation flow:
/// [`Plan::create`] gets you a fresh handle, then one of the
/// `make_plan_*` methods configures the rank and reports the workspace
/// size — useful when you want to allocate the work area yourself
/// (call [`set_auto_allocation(false)`] before `make_plan_*`).
///
/// Unlike [`Plan1d`] / [`Plan2d`] / [`Plan3d`], this type can hold any
/// rank, including the batched-many / 64-bit-many forms.
pub struct Plan {
    handle: cufftHandle,
}

unsafe impl Send for Plan {}

impl core::fmt::Debug for Plan {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Plan")
            .field("handle", &self.handle)
            .finish()
    }
}

impl Plan {
    /// Allocate a fresh empty plan. The plan is unusable until you
    /// finalize it with one of the `make_plan_*` methods.
    pub fn create() -> Result<Self> {
        let c = cufft()?;
        let cu = c.cufft_create()?;
        let mut plan: cufftHandle = 0;
        check(unsafe { cu(&mut plan) })?;
        Ok(Self { handle: plan })
    }

    /// Finalize as a 1-D plan of length `nx` and `batch` parallel
    /// transforms. Returns the workspace size in bytes.
    pub fn make_plan_1d(&self, nx: i32, transform: Transform, batch: i32) -> Result<usize> {
        let c = cufft()?;
        let cu = c.cufft_make_plan_1d()?;
        let mut size: usize = 0;
        check(unsafe { cu(self.handle, nx, transform.raw(), batch, &mut size) })?;
        Ok(size)
    }

    /// Finalize as a 2-D plan. Returns workspace size in bytes.
    pub fn make_plan_2d(&self, nx: i32, ny: i32, transform: Transform) -> Result<usize> {
        let c = cufft()?;
        let cu = c.cufft_make_plan_2d()?;
        let mut size: usize = 0;
        check(unsafe { cu(self.handle, nx, ny, transform.raw(), &mut size) })?;
        Ok(size)
    }

    /// Finalize as a 3-D plan. Returns workspace size in bytes.
    pub fn make_plan_3d(
        &self,
        nx: i32,
        ny: i32,
        nz: i32,
        transform: Transform,
    ) -> Result<usize> {
        let c = cufft()?;
        let cu = c.cufft_make_plan_3d()?;
        let mut size: usize = 0;
        check(unsafe { cu(self.handle, nx, ny, nz, transform.raw(), &mut size) })?;
        Ok(size)
    }

    /// Finalize as a generic strided/batched plan. Returns workspace
    /// size in bytes.
    ///
    /// # Safety
    ///
    /// `n`, `inembed`, `onembed` must be writable arrays of length
    /// `rank` (cuFFT mutates them in some versions). Pass null for
    /// `inembed` / `onembed` to use defaults.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn make_plan_many(
        &self,
        rank: i32,
        n: &mut [i32],
        inembed: *mut i32,
        istride: i32,
        idist: i32,
        onembed: *mut i32,
        ostride: i32,
        odist: i32,
        transform: Transform,
        batch: i32,
    ) -> Result<usize> {
        assert_eq!(n.len() as i32, rank, "n.len() must equal rank");
        let c = cufft()?;
        let cu = c.cufft_make_plan_many()?;
        let mut size: usize = 0;
        check(cu(
            self.handle,
            rank,
            n.as_mut_ptr(),
            inembed,
            istride,
            idist,
            onembed,
            ostride,
            odist,
            transform.raw(),
            batch,
            &mut size,
        ))?;
        Ok(size)
    }

    /// 64-bit variant of [`make_plan_many`] — use this when any
    /// dimension or stride exceeds `i32::MAX`.
    ///
    /// # Safety
    ///
    /// Same as [`make_plan_many`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn make_plan_many64(
        &self,
        rank: i32,
        n: &mut [i64],
        inembed: *mut i64,
        istride: i64,
        idist: i64,
        onembed: *mut i64,
        ostride: i64,
        odist: i64,
        transform: Transform,
        batch: i64,
    ) -> Result<usize> {
        assert_eq!(n.len() as i32, rank, "n.len() must equal rank");
        let c = cufft()?;
        let cu = c.cufft_make_plan_many64()?;
        let mut size: usize = 0;
        check(cu(
            self.handle,
            rank,
            n.as_mut_ptr(),
            inembed,
            istride,
            idist,
            onembed,
            ostride,
            odist,
            transform.raw(),
            batch,
            &mut size,
        ))?;
        Ok(size)
    }

    /// Bind subsequent exec calls to `stream`.
    pub fn set_stream(&self, stream: &Stream) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_set_stream()?;
        check(unsafe { cu(self.handle, stream.as_raw() as _) })
    }

    /// Raw `cufftHandle`. Use with care.
    #[inline]
    pub fn as_raw(&self) -> cufftHandle {
        self.handle
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        if let Ok(c) = cufft() {
            if let Ok(cu) = c.cufft_destroy() {
                let _ = unsafe { cu(self.handle) };
            }
        }
    }
}

// ============================================================================
// Callback API (cufftXtSetCallback / Clear / SetCallbackSharedSize)
// ============================================================================

pub mod callback {
    //! Pre/post callbacks attached to a cuFFT plan via the `cufftXt*`
    //! callback entry points. The callback ABI is fixed by NVIDIA: each
    //! callback receives the input/output element index, a caller-info
    //! pointer, and the data; see the cuFFT reference for the exact
    //! signatures by callback type. We expose only the raw setters
    //! because the function-pointer types are PTX-shaped, not regular
    //! `extern "C"` functions — they ship as device-side `__device__`
    //! functions linked into the user's CUBIN.
    use super::*;

    /// Callback type values from the cuFFT header
    /// (`cufftXtCallbackType`).
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    #[repr(i32)]
    pub enum CallbackType {
        /// Load callback for complex single precision.
        LoadComplex = 0,
        /// Load callback for complex double precision.
        LoadDoubleComplex = 1,
        /// Load callback for real single precision.
        LoadReal = 2,
        /// Load callback for real double precision.
        LoadDoubleReal = 3,
        /// Store callback for complex single precision.
        StoreComplex = 4,
        /// Store callback for complex double precision.
        StoreDoubleComplex = 5,
        /// Store callback for real single precision.
        StoreReal = 6,
        /// Store callback for real double precision.
        StoreDoubleReal = 7,
    }

    /// Attach a load/store callback to the plan. `callback_routine` is
    /// an array of device function pointers — one per GPU for
    /// multi-GPU plans, otherwise a single-element array.
    /// `caller_info` is parallel; pass null for "no caller info".
    ///
    /// # Safety
    ///
    /// `callback_routine[i]` must be a `__device__` function with the
    /// signature cuFFT expects for `cb_type` (see the cuFFT reference).
    /// The routine and `caller_info` must outlive every plan exec call.
    pub unsafe fn set(
        plan: cufftHandle,
        callback_routine: &mut [*mut core::ffi::c_void],
        cb_type: CallbackType,
        caller_info: &mut [*mut core::ffi::c_void],
    ) -> Result<()> {
        assert_eq!(
            callback_routine.len(),
            caller_info.len(),
            "callback_routine and caller_info must have the same length"
        );
        let c = cufft()?;
        let cu = c.cufft_xt_set_callback()?;
        check(cu(
            plan,
            callback_routine.as_mut_ptr(),
            cb_type as i32,
            caller_info.as_mut_ptr(),
        ))
    }

    /// Detach any previously set callback of `cb_type`.
    pub fn clear(plan: cufftHandle, cb_type: CallbackType) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_xt_clear_callback()?;
        check(unsafe { cu(plan, cb_type as i32) })
    }

    /// Reserve `shared_size` bytes of dynamic shared memory per kernel
    /// for the callback. Maximum permissible value is GPU-dependent.
    pub fn set_shared_size(
        plan: cufftHandle,
        cb_type: CallbackType,
        shared_size: usize,
    ) -> Result<()> {
        let c = cufft()?;
        let cu = c.cufft_xt_set_callback_shared_size()?;
        check(unsafe { cu(plan, cb_type as i32, shared_size) })
    }
}
