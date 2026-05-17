//! Multi-dimensional FFT / IFFT (complex-to-complex) — `FftNdPlan<T>`
//! for `T = Complex32` / `Complex64`. Milestone 6.8.
//!
//! Wraps cuFFT's `cufftPlanMany` for `rank`-D transforms (2-D and 3-D
//! are the primary targets; rank-1 also works as a degenerate path
//! parallel to `FftPlan`). Trailblazer ships rank in `1..=3`; rank 4
//! is wired in the descriptor but the entry-point `select` rejects it
//! pending a real-hardware soak.
//!
//! Layout contract: the transformed axes are the **trailing**
//! `rank` axes of the input / output. Anything to the left of those
//! axes is flattened into the cuFFT `batch` dimension. The descriptor
//! therefore carries:
//!
//! - `dims: [i32; 4]` — extent of each transformed axis (only the
//!   first `rank` slots are read). `dims[0]` is the slowest, `dims[rank-1]`
//!   the fastest.
//! - `rank: u8` — number of transformed axes (`1..=3`).
//! - `batch: i32` — product of all leading axes that are *not*
//!   transformed (or `1` for the "single instance" case).
//!
//! Restriction: the transformed axes must be a **contiguous suffix**
//! of the operand's logical rank. Callers that want to FFT over a
//! non-suffix axis subset must permute before the call. cuFFT itself
//! supports arbitrary strides through `cufftPlanMany`'s `inembed` /
//! `istride` / `idist` triple, but the safe plan layer keeps the
//! contract simple and avoids the surface area of arbitrary embedding.
//!
//! Normalization: forward is unnormalized; inverse is divided by
//! `N = product(dims[..rank])` in-place via the bespoke
//! `scale_inplace_c{32,64}` kernel, matching PyTorch's `norm="backward"`.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::{DeviceSlice, DeviceSliceMut, Stream};
use baracuda_kernels_sys::{
    baracuda_kernels_scale_inplace_c32_run, baracuda_kernels_scale_inplace_c64_run, cufftComplex,
    cufftDestroy, cufftDoubleComplex, cufftExecC2C, cufftExecZ2Z, cufftHandle, cufftPlanMany,
    cufftSetStream, CUFFT_C2C, CUFFT_FORWARD, CUFFT_INVERSE, CUFFT_Z2Z,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Complex32, Complex64, Element, ElementKind, FftKind, KernelSku,
    MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, Workspace,
};

use super::fft::{cufft_to_status, map_status};

const HANDLE_UNINIT: cufftHandle = -1;
const MAX_RANK: usize = 4;

/// Descriptor for an ND C2C FFT / IFFT.
///
/// `dims[0..rank]` are the extents along the transformed axes,
/// slowest-first. `batch` is the cuFFT batch (product of any leading
/// non-transformed axes). `inverse` selects the direction; for
/// inverse the plan layer applies `1/product(dims[..rank])` in-place.
#[derive(Copy, Clone, Debug)]
pub struct FftNdDescriptor {
    /// Per-axis extents for the transformed axes. Only `dims[0..rank]`
    /// is read; the trailing slots are ignored.
    pub dims: [i32; MAX_RANK],
    /// Number of transformed axes. `1..=3` supported by the trailblazer.
    pub rank: u8,
    /// Number of independent transforms (product of leading
    /// non-transformed axes). `1` for the "single instance" case.
    pub batch: i32,
    /// `false` = forward (unnormalized), `true` = inverse (normalized
    /// by `1/N` to match PyTorch's `norm="backward"`).
    pub inverse: bool,
    /// Element type — `Complex32` / `Complex64`.
    pub element: ElementKind,
}

impl FftNdDescriptor {
    /// Total number of elements along the transformed axes
    /// (`product(dims[..rank])`). Saturates on overflow.
    #[inline]
    pub fn transform_numel(&self) -> i64 {
        let mut n: i64 = 1;
        let rank = self.rank as usize;
        let mut i = 0;
        while i < rank {
            n = n.saturating_mul(self.dims[i] as i64);
            i += 1;
        }
        n
    }
}

/// Args for an ND C2C FFT.
///
/// Both buffers are flat device slices — the descriptor's `dims` +
/// `batch` define the logical shape. Required length is
/// `batch * product(dims[..rank])` cells each.
pub struct FftNdArgs<'a, T: Element> {
    /// Input data — `batch * product(dims[..rank])` complex cells.
    pub x: DeviceSlice<'a, T>,
    /// Output data — `batch * product(dims[..rank])` complex cells.
    pub y: DeviceSliceMut<'a, T>,
}

/// Multi-dimensional FFT / IFFT (C2C) plan.
pub struct FftNdPlan<T: Element> {
    desc: FftNdDescriptor,
    sku: KernelSku,
    handle: Cell<cufftHandle>,
    _marker: PhantomData<T>,
}

impl<T: Element> FftNdPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &FftNdDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FftNdPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::Complex32 | ElementKind::Complex64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::FftNdPlan: C2C ND FFT supports Complex32 + Complex64 only",
            ));
        }
        if !(1..=3).contains(&desc.rank) {
            return Err(Error::Unsupported(
                "baracuda-kernels::FftNdPlan: rank must be in 1..=3 (trailblazer)",
            ));
        }
        if desc.batch <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftNdPlan: batch must be > 0",
            ));
        }
        for i in 0..desc.rank as usize {
            if desc.dims[i] <= 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::FftNdPlan: every transformed-axis dim must be > 0",
                ));
            }
        }

        let math_precision = match T::KIND {
            ElementKind::Complex64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: T::KIND,
            bit_stable_on_same_hardware: false,
            deterministic: true,
        };
        let op = if desc.inverse {
            FftKind::Ifft
        } else {
            FftKind::Fft
        };
        let sku = KernelSku {
            category: OpCategory::Fft,
            op: op as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Cufft,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            handle: Cell::new(HANDLE_UNINIT),
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

    /// Workspace size in bytes — cuFFT manages its own internal scratch.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    fn ensure_handle(&self) -> Result<cufftHandle> {
        let h = self.handle.get();
        if h != HANDLE_UNINIT {
            return Ok(h);
        }
        let fft_type = match T::KIND {
            ElementKind::Complex32 => CUFFT_C2C,
            ElementKind::Complex64 => CUFFT_Z2Z,
            _ => unreachable!("select() gates on Complex32 / Complex64"),
        };
        let rank = self.desc.rank as i32;
        // cufftPlanMany takes `*mut i32` for n / inembed / onembed; the
        // arrays must outlive the call. Stack copies are fine — cuFFT
        // reads them synchronously inside `cufftPlanMany`.
        let mut n: [i32; MAX_RANK] = self.desc.dims;
        let dist = self.desc.transform_numel() as i32;
        let mut handle: cufftHandle = HANDLE_UNINIT;
        let status = unsafe {
            cufftPlanMany(
                &mut handle as *mut _,
                rank,
                n.as_mut_ptr(),
                core::ptr::null_mut(),
                1,
                dist,
                core::ptr::null_mut(),
                1,
                dist,
                fft_type,
                self.desc.batch,
            )
        };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }
        self.handle.set(handle);
        Ok(handle)
    }

    fn bind_stream(&self, handle: cufftHandle, stream: &Stream) -> Result<()> {
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = unsafe { cufftSetStream(handle, stream_ptr) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }
        Ok(())
    }

    fn check_args(&self, x: &DeviceSlice<'_, T>, y: &DeviceSliceMut<'_, T>) -> Result<i64> {
        let per = self.desc.transform_numel();
        let total = per.saturating_mul(self.desc.batch as i64);
        if (x.len() as i64) < total {
            return Err(Error::BufferTooSmall {
                needed: total as usize,
                got: x.len(),
            });
        }
        if (y.len() as i64) < total {
            return Err(Error::BufferTooSmall {
                needed: total as usize,
                got: y.len(),
            });
        }
        Ok(total)
    }
}

// ----- Complex32 -------------------------------------------------------------

impl FftNdPlan<Complex32> {
    /// Run the ND C2C FFT (single precision).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FftNdArgs<'_, Complex32>,
    ) -> Result<()> {
        let total = self.check_args(&args.x, &args.y)?;
        if total == 0 {
            return Ok(());
        }
        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;
        let direction = if self.desc.inverse {
            CUFFT_INVERSE
        } else {
            CUFFT_FORWARD
        };
        let idata = args.x.as_raw().0 as *mut cufftComplex;
        let odata = args.y.as_raw().0 as *mut cufftComplex;
        let status = unsafe { cufftExecC2C(handle, idata, odata, direction) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }

        if self.desc.inverse {
            let per = self.desc.transform_numel() as f32;
            let scale = 1.0_f32 / per;
            let stream_ptr = stream.as_raw() as *mut c_void;
            let s = unsafe {
                baracuda_kernels_scale_inplace_c32_run(
                    total,
                    scale,
                    odata as *mut c_void,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            };
            map_status(s)?;
        }
        Ok(())
    }
}

// ----- Complex64 -------------------------------------------------------------

impl FftNdPlan<Complex64> {
    /// Run the ND C2C FFT (double precision).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FftNdArgs<'_, Complex64>,
    ) -> Result<()> {
        let total = self.check_args(&args.x, &args.y)?;
        if total == 0 {
            return Ok(());
        }
        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;
        let direction = if self.desc.inverse {
            CUFFT_INVERSE
        } else {
            CUFFT_FORWARD
        };
        let idata = args.x.as_raw().0 as *mut cufftDoubleComplex;
        let odata = args.y.as_raw().0 as *mut cufftDoubleComplex;
        let status = unsafe { cufftExecZ2Z(handle, idata, odata, direction) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }

        if self.desc.inverse {
            let per = self.desc.transform_numel() as f64;
            let scale = 1.0_f64 / per;
            let stream_ptr = stream.as_raw() as *mut c_void;
            let s = unsafe {
                baracuda_kernels_scale_inplace_c64_run(
                    total,
                    scale,
                    odata as *mut c_void,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            };
            map_status(s)?;
        }
        Ok(())
    }
}

impl<T: Element> Drop for FftNdPlan<T> {
    fn drop(&mut self) {
        let h = self.handle.get();
        if h != HANDLE_UNINIT {
            unsafe {
                let _ = cufftDestroy(h);
            }
            self.handle.set(HANDLE_UNINIT);
        }
    }
}
