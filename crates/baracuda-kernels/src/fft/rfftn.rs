//! Multi-dimensional RFFT (real → complex) and IRFFT (complex → real)
//! — `RfftNdPlan<T>` / `IrfftNdPlan<T>` for `T = f32` / `f64`.
//! Milestone 6.8.
//!
//! Wraps cuFFT's `cufftPlanMany` with `R2C` / `D2Z` (forward) and
//! `C2R` / `Z2D` (inverse) for `rank`-D transforms (trailblazer:
//! `rank` in `1..=3`).
//!
//! Layout contract: the transformed axes are the **trailing** `rank`
//! axes; anything earlier is flattened into the cuFFT `batch`. cuFFT's
//! Hermitian-half convention applies to the *last* transformed axis
//! only — the complex side has extent `dims[rank-1] / 2 + 1` along
//! that axis, with the earlier transformed axes carrying their full
//! length on both sides. So the buffer sizes are:
//!
//! - Real side:    `batch * dims[0] * dims[1] * ... * dims[rank-1]`
//! - Complex side: `batch * dims[0] * dims[1] * ... * (dims[rank-1]/2 + 1)`
//!
//! Normalization: forward unnormalized; inverse divided by
//! `N = product(dims[..rank])` (the *real* element count, matching
//! PyTorch's `norm="backward"`).

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::{DeviceSlice, DeviceSliceMut, Stream};
use baracuda_kernels_sys::{
    baracuda_kernels_scale_inplace_real_f32_run, baracuda_kernels_scale_inplace_real_f64_run,
    cufftComplex, cufftDestroy, cufftDoubleComplex, cufftExecC2R, cufftExecD2Z, cufftExecR2C,
    cufftExecZ2D, cufftHandle, cufftPlanMany, cufftSetStream, CUFFT_C2R, CUFFT_D2Z, CUFFT_R2C,
    CUFFT_Z2D,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Complex32, Complex64, Element, ElementKind, FftKind, KernelSku,
    MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, Workspace,
};

use super::fft::{cufft_to_status, map_status};

const HANDLE_UNINIT: cufftHandle = -1;
const MAX_RANK: usize = 4;

// =============================================================================
// RFFT-ND — real → complex
// =============================================================================

/// Descriptor for an ND RFFT.
///
/// `dims[..rank]` are the per-axis transform extents — *as measured on
/// the real side* (the last-axis Hermitian-half halving is implicit on
/// the complex output). `batch` is the cuFFT batch.
#[derive(Copy, Clone, Debug)]
pub struct RfftNdDescriptor {
    /// Real-side per-axis extents for the transformed axes
    /// (only `dims[..rank]` is read).
    pub dims: [i32; MAX_RANK],
    /// Number of transformed axes. `1..=3` supported by the trailblazer.
    pub rank: u8,
    /// Number of independent transforms.
    pub batch: i32,
    /// Real-side element type — `F32` / `F64`.
    pub element: ElementKind,
}

impl RfftNdDescriptor {
    /// Real-side element count per batched transform.
    #[inline]
    pub fn real_numel(&self) -> i64 {
        let mut n: i64 = 1;
        for i in 0..self.rank as usize {
            n = n.saturating_mul(self.dims[i] as i64);
        }
        n
    }

    /// Complex-side element count per batched transform — same as
    /// `real_numel` except the last transformed axis is replaced by
    /// `dims[rank-1] / 2 + 1`.
    #[inline]
    pub fn complex_numel(&self) -> i64 {
        let rank = self.rank as usize;
        if rank == 0 {
            return 1;
        }
        let mut n: i64 = 1;
        for i in 0..rank - 1 {
            n = n.saturating_mul(self.dims[i] as i64);
        }
        n = n.saturating_mul((self.dims[rank - 1] / 2 + 1) as i64);
        n
    }
}

/// Args for an ND RFFT. `T` is the real type, `C` the matching complex.
pub struct RfftNdArgs<'a, T: Element, C: Element> {
    /// Real input — `batch * product(dims[..rank])` cells.
    pub x: DeviceSlice<'a, T>,
    /// Complex output — `batch * complex_numel()` cells.
    pub y: DeviceSliceMut<'a, C>,
}

/// ND RFFT plan.
pub struct RfftNdPlan<T: Element> {
    desc: RfftNdDescriptor,
    sku: KernelSku,
    handle: Cell<cufftHandle>,
    _marker: PhantomData<T>,
}

impl<T: Element> RfftNdPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &RfftNdDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RfftNdPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::RfftNdPlan: R2C ND FFT supports f32 + f64 only",
            ));
        }
        if !(1..=3).contains(&desc.rank) {
            return Err(Error::Unsupported(
                "baracuda-kernels::RfftNdPlan: rank must be in 1..=3 (trailblazer)",
            ));
        }
        if desc.batch <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RfftNdPlan: batch must be > 0",
            ));
        }
        for i in 0..desc.rank as usize {
            if desc.dims[i] <= 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RfftNdPlan: every transformed-axis dim must be > 0",
                ));
            }
        }
        let math_precision = match T::KIND {
            ElementKind::F64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let aux = match T::KIND {
            ElementKind::F32 => Some(ElementKind::Complex32),
            ElementKind::F64 => Some(ElementKind::Complex64),
            _ => None,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: T::KIND,
            bit_stable_on_same_hardware: false,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Fft,
            op: FftKind::Rfft as u16,
            element: T::KIND,
            aux_element: aux,
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

    /// Workspace size in bytes.
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
            ElementKind::F32 => CUFFT_R2C,
            ElementKind::F64 => CUFFT_D2Z,
            _ => unreachable!("select() gates on F32 / F64"),
        };
        let rank = self.desc.rank as i32;
        let mut n: [i32; MAX_RANK] = self.desc.dims;
        let real_dist = self.desc.real_numel() as i32;
        let complex_dist = self.desc.complex_numel() as i32;
        let mut handle: cufftHandle = HANDLE_UNINIT;
        // Default-layout R2C: input distance is the real numel,
        // output distance is the complex (Hermitian-half) numel.
        let status = unsafe {
            cufftPlanMany(
                &mut handle as *mut _,
                rank,
                n.as_mut_ptr(),
                core::ptr::null_mut(),
                1,
                real_dist,
                core::ptr::null_mut(),
                1,
                complex_dist,
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
}

impl RfftNdPlan<f32> {
    /// Run the ND R2C FFT (single precision).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: RfftNdArgs<'_, f32, Complex32>,
    ) -> Result<()> {
        let real_total = self.desc.real_numel().saturating_mul(self.desc.batch as i64);
        let complex_total = self
            .desc
            .complex_numel()
            .saturating_mul(self.desc.batch as i64);
        if (args.x.len() as i64) < real_total {
            return Err(Error::BufferTooSmall {
                needed: real_total as usize,
                got: args.x.len(),
            });
        }
        if (args.y.len() as i64) < complex_total {
            return Err(Error::BufferTooSmall {
                needed: complex_total as usize,
                got: args.y.len(),
            });
        }
        if real_total == 0 {
            return Ok(());
        }

        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;

        let idata = args.x.as_raw().0 as *mut f32;
        let odata = args.y.as_raw().0 as *mut cufftComplex;
        let status = unsafe { cufftExecR2C(handle, idata, odata) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }
        Ok(())
    }
}

impl RfftNdPlan<f64> {
    /// Run the ND R2C FFT (double precision).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: RfftNdArgs<'_, f64, Complex64>,
    ) -> Result<()> {
        let real_total = self.desc.real_numel().saturating_mul(self.desc.batch as i64);
        let complex_total = self
            .desc
            .complex_numel()
            .saturating_mul(self.desc.batch as i64);
        if (args.x.len() as i64) < real_total {
            return Err(Error::BufferTooSmall {
                needed: real_total as usize,
                got: args.x.len(),
            });
        }
        if (args.y.len() as i64) < complex_total {
            return Err(Error::BufferTooSmall {
                needed: complex_total as usize,
                got: args.y.len(),
            });
        }
        if real_total == 0 {
            return Ok(());
        }

        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;

        let idata = args.x.as_raw().0 as *mut f64;
        let odata = args.y.as_raw().0 as *mut cufftDoubleComplex;
        let status = unsafe { cufftExecD2Z(handle, idata, odata) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }
        Ok(())
    }
}

impl<T: Element> Drop for RfftNdPlan<T> {
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

// =============================================================================
// IRFFT-ND — complex → real
// =============================================================================

/// Descriptor for an ND IRFFT.
///
/// `dims[..rank]` are the *real-side* per-axis extents — the complex
/// input has its last transformed axis halved (`dims[rank-1]/2 + 1`).
/// cuFFT cannot infer the real-side last-axis length from the complex
/// shape alone, so `dims[rank-1]` is required.
#[derive(Copy, Clone, Debug)]
pub struct IrfftNdDescriptor {
    /// Real-side per-axis extents.
    pub dims: [i32; MAX_RANK],
    /// Number of transformed axes. `1..=3`.
    pub rank: u8,
    /// Number of independent transforms.
    pub batch: i32,
    /// Real-side element type (output dtype).
    pub element: ElementKind,
}

impl IrfftNdDescriptor {
    /// Real-side element count per batched transform.
    #[inline]
    pub fn real_numel(&self) -> i64 {
        let mut n: i64 = 1;
        for i in 0..self.rank as usize {
            n = n.saturating_mul(self.dims[i] as i64);
        }
        n
    }

    /// Complex-side element count per batched transform.
    #[inline]
    pub fn complex_numel(&self) -> i64 {
        let rank = self.rank as usize;
        if rank == 0 {
            return 1;
        }
        let mut n: i64 = 1;
        for i in 0..rank - 1 {
            n = n.saturating_mul(self.dims[i] as i64);
        }
        n = n.saturating_mul((self.dims[rank - 1] / 2 + 1) as i64);
        n
    }
}

/// Args for an ND IRFFT.
pub struct IrfftNdArgs<'a, T: Element, C: Element> {
    /// Complex input — `batch * complex_numel()` cells.
    pub x: DeviceSlice<'a, C>,
    /// Real output — `batch * real_numel()` cells.
    pub y: DeviceSliceMut<'a, T>,
}

/// ND IRFFT plan.
pub struct IrfftNdPlan<T: Element> {
    desc: IrfftNdDescriptor,
    sku: KernelSku,
    handle: Cell<cufftHandle>,
    _marker: PhantomData<T>,
}

impl<T: Element> IrfftNdPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &IrfftNdDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::IrfftNdPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::IrfftNdPlan: C2R ND FFT supports f32 + f64 only",
            ));
        }
        if !(1..=3).contains(&desc.rank) {
            return Err(Error::Unsupported(
                "baracuda-kernels::IrfftNdPlan: rank must be in 1..=3 (trailblazer)",
            ));
        }
        if desc.batch <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IrfftNdPlan: batch must be > 0",
            ));
        }
        for i in 0..desc.rank as usize {
            if desc.dims[i] <= 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::IrfftNdPlan: every transformed-axis dim must be > 0",
                ));
            }
        }
        let math_precision = match T::KIND {
            ElementKind::F64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let aux = match T::KIND {
            ElementKind::F32 => Some(ElementKind::Complex32),
            ElementKind::F64 => Some(ElementKind::Complex64),
            _ => None,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: T::KIND,
            bit_stable_on_same_hardware: false,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Fft,
            op: FftKind::Irfft as u16,
            element: T::KIND,
            aux_element: aux,
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

    /// Workspace size in bytes.
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
            ElementKind::F32 => CUFFT_C2R,
            ElementKind::F64 => CUFFT_Z2D,
            _ => unreachable!("select() gates on F32 / F64"),
        };
        let rank = self.desc.rank as i32;
        let mut n: [i32; MAX_RANK] = self.desc.dims;
        let real_dist = self.desc.real_numel() as i32;
        let complex_dist = self.desc.complex_numel() as i32;
        let mut handle: cufftHandle = HANDLE_UNINIT;
        // Default-layout C2R: input distance is the complex
        // (Hermitian-half) numel, output distance is the real numel.
        let status = unsafe {
            cufftPlanMany(
                &mut handle as *mut _,
                rank,
                n.as_mut_ptr(),
                core::ptr::null_mut(),
                1,
                complex_dist,
                core::ptr::null_mut(),
                1,
                real_dist,
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
}

impl IrfftNdPlan<f32> {
    /// Run the ND C2R FFT (single precision). Normalizes by
    /// `1/product(dims[..rank])` to match PyTorch's `norm="backward"`.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: IrfftNdArgs<'_, f32, Complex32>,
    ) -> Result<()> {
        let real_total = self.desc.real_numel().saturating_mul(self.desc.batch as i64);
        let complex_total = self
            .desc
            .complex_numel()
            .saturating_mul(self.desc.batch as i64);
        if (args.x.len() as i64) < complex_total {
            return Err(Error::BufferTooSmall {
                needed: complex_total as usize,
                got: args.x.len(),
            });
        }
        if (args.y.len() as i64) < real_total {
            return Err(Error::BufferTooSmall {
                needed: real_total as usize,
                got: args.y.len(),
            });
        }
        if real_total == 0 {
            return Ok(());
        }

        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;

        let idata = args.x.as_raw().0 as *mut cufftComplex;
        let odata = args.y.as_raw().0 as *mut f32;
        let status = unsafe { cufftExecC2R(handle, idata, odata) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }

        let n = self.desc.real_numel() as f32;
        let scale = 1.0_f32 / n;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let s = unsafe {
            baracuda_kernels_scale_inplace_real_f32_run(
                real_total,
                scale,
                odata as *mut c_void,
                core::ptr::null_mut(),
                0,
                stream_ptr,
            )
        };
        map_status(s)
    }
}

impl IrfftNdPlan<f64> {
    /// Run the ND C2R FFT (double precision).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: IrfftNdArgs<'_, f64, Complex64>,
    ) -> Result<()> {
        let real_total = self.desc.real_numel().saturating_mul(self.desc.batch as i64);
        let complex_total = self
            .desc
            .complex_numel()
            .saturating_mul(self.desc.batch as i64);
        if (args.x.len() as i64) < complex_total {
            return Err(Error::BufferTooSmall {
                needed: complex_total as usize,
                got: args.x.len(),
            });
        }
        if (args.y.len() as i64) < real_total {
            return Err(Error::BufferTooSmall {
                needed: real_total as usize,
                got: args.y.len(),
            });
        }
        if real_total == 0 {
            return Ok(());
        }

        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;

        let idata = args.x.as_raw().0 as *mut cufftDoubleComplex;
        let odata = args.y.as_raw().0 as *mut f64;
        let status = unsafe { cufftExecZ2D(handle, idata, odata) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }

        let n = self.desc.real_numel() as f64;
        let scale = 1.0_f64 / n;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let s = unsafe {
            baracuda_kernels_scale_inplace_real_f64_run(
                real_total,
                scale,
                odata as *mut c_void,
                core::ptr::null_mut(),
                0,
                stream_ptr,
            )
        };
        map_status(s)
    }
}

impl<T: Element> Drop for IrfftNdPlan<T> {
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
