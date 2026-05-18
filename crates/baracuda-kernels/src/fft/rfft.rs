//! RFFT (real-to-complex) and IRFFT (complex-to-real) — `RfftPlan<T>`
//! and `IrfftPlan<T>` for `T = f32` or `T = f64`.
//!
//! Wraps cuFFT's `cufftExecR2C` / `cufftExecD2Z` (forward) and
//! `cufftExecC2R` / `cufftExecZ2D` (inverse).
//!
//! Input / output shape contract:
//! - RFFT: input `[batch, n]` real, output `[batch, n/2 + 1]` complex
//!   (Hermitian-half — the missing half is the conjugate of the
//!   present half).
//! - IRFFT: input `[batch, n/2 + 1]` complex, output `[batch, n]`
//!   real. cuFFT cannot infer the output length `n` from the Hermitian-
//!   half input alone (both `2*(n/2)` and `2*(n/2)+1` map to the same
//!   half), so `n` is a required descriptor parameter.
//!
//! For inverse transforms the plan applies `1/n` normalization
//! in-place after the cuFFT exec, matching PyTorch's `norm="backward"`.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    baracuda_kernels_scale_inplace_real_f32_run, baracuda_kernels_scale_inplace_real_f64_run,
    cufftComplex, cufftDestroy, cufftDoubleComplex, cufftExecC2R, cufftExecD2Z, cufftExecR2C,
    cufftExecZ2D, cufftHandle, cufftPlan1d, cufftSetStream, CUFFT_C2R, CUFFT_D2Z, CUFFT_R2C,
    CUFFT_Z2D,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Complex32, Complex64, Element, ElementKind, FftKind, KernelSku,
    MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::fft::{cufft_to_status, map_status};

const HANDLE_UNINIT: cufftHandle = -1;

// =============================================================================
// RFFT — real → complex (Hermitian-half)
// =============================================================================

/// Descriptor for an RFFT (real-to-complex) op.
#[derive(Copy, Clone, Debug)]
pub struct RfftDescriptor {
    /// Signal length (the real input length). Output has shape
    /// `[batch, n/2 + 1]` (Hermitian-half).
    pub n: i32,
    /// Number of independent transforms in one launch.
    pub batch: i32,
    /// Real-side element type — `F32` or `F64`. The complex output
    /// type is `Complex32` (for `F32`) or `Complex64` (for `F64`).
    pub element: ElementKind,
}

/// Args bundle for an RFFT.
///
/// `T` is the *real* element type (`f32` / `f64`). The complex output
/// uses [`Complex32`] / [`Complex64`] depending on `T`.
pub struct RfftArgs<'a, T: Element, C: Element> {
    /// Real input tensor `[batch, n]`.
    pub x: TensorRef<'a, T, 2>,
    /// Complex output tensor `[batch, n/2 + 1]`.
    pub y: TensorMut<'a, C, 2>,
}

/// 1-D RFFT plan — real input → Hermitian-half complex output.
///
/// Wraps cuFFT's `cufftExecR2C` (`f32`) / `cufftExecD2Z` (`f64`).
///
/// **When to use**: forward FFT of real-valued data; the output is the
/// non-redundant Hermitian half. Pair with [`IrfftPlan`] for the
/// inverse direction. Use [`super::FftPlan`] when the input is
/// already complex.
///
/// **Dtypes**: `f32` → `Complex32`; `f64` → `Complex64`.
///
/// **Shape**: real `[batch, n]` → complex `[batch, n/2 + 1]`.
///
/// **Normalization**: unnormalized (`norm="backward"`).
///
/// **Workspace**: zero — cuFFT manages internal workspace.
///
/// **Precision guarantee**: deterministic; not bit-stable across
/// cuFFT versions.
///
/// Owns a lazy cuFFT handle (`!Sync` / `!Send`); destroyed on `Drop`.
pub struct RfftPlan<T: Element> {
    desc: RfftDescriptor,
    sku: KernelSku,
    handle: Cell<cufftHandle>,
    _marker: PhantomData<T>,
}

impl<T: Element> RfftPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(_stream: &Stream, desc: &RfftDescriptor, _pref: PlanPreference) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RfftPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::RfftPlan: R2C FFT supports f32 + f64 only",
            ));
        }
        if desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RfftPlan: n must be > 0",
            ));
        }
        if desc.batch <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RfftPlan: batch must be > 0",
            ));
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

    /// Workspace size in bytes — cuFFT-internal, no caller-supplied
    /// workspace needed.
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
        let mut handle: cufftHandle = HANDLE_UNINIT;
        let status = unsafe {
            cufftPlan1d(
                &mut handle as *mut _,
                self.desc.n,
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

impl RfftPlan<f32> {
    /// Run the R2C FFT (single precision).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: RfftArgs<'_, f32, Complex32>,
    ) -> Result<()> {
        let n = self.desc.n;
        let batch = self.desc.batch;
        let in_shape = [batch, n];
        let out_shape = [batch, n / 2 + 1];
        if args.x.shape != in_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RfftPlan<f32>: x shape != [batch, n]",
            ));
        }
        if args.y.shape != out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RfftPlan<f32>: y shape != [batch, n/2 + 1]",
            ));
        }
        let in_numel = (batch as i64) * (n as i64);
        let out_numel = (batch as i64) * ((n / 2 + 1) as i64);
        if (args.x.data.len() as i64) < in_numel {
            return Err(Error::BufferTooSmall {
                needed: in_numel as usize,
                got: args.x.data.len(),
            });
        }
        if (args.y.data.len() as i64) < out_numel {
            return Err(Error::BufferTooSmall {
                needed: out_numel as usize,
                got: args.y.data.len(),
            });
        }
        if in_numel == 0 {
            return Ok(());
        }

        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;

        let idata = args.x.data.as_raw().0 as *mut f32;
        let odata = args.y.data.as_raw().0 as *mut cufftComplex;
        let status = unsafe { cufftExecR2C(handle, idata, odata) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }
        Ok(())
    }
}

impl RfftPlan<f64> {
    /// Run the R2C FFT (double precision).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: RfftArgs<'_, f64, Complex64>,
    ) -> Result<()> {
        let n = self.desc.n;
        let batch = self.desc.batch;
        let in_shape = [batch, n];
        let out_shape = [batch, n / 2 + 1];
        if args.x.shape != in_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RfftPlan<f64>: x shape != [batch, n]",
            ));
        }
        if args.y.shape != out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RfftPlan<f64>: y shape != [batch, n/2 + 1]",
            ));
        }
        let in_numel = (batch as i64) * (n as i64);
        let out_numel = (batch as i64) * ((n / 2 + 1) as i64);
        if (args.x.data.len() as i64) < in_numel {
            return Err(Error::BufferTooSmall {
                needed: in_numel as usize,
                got: args.x.data.len(),
            });
        }
        if (args.y.data.len() as i64) < out_numel {
            return Err(Error::BufferTooSmall {
                needed: out_numel as usize,
                got: args.y.data.len(),
            });
        }
        if in_numel == 0 {
            return Ok(());
        }

        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;

        let idata = args.x.data.as_raw().0 as *mut f64;
        let odata = args.y.data.as_raw().0 as *mut cufftDoubleComplex;
        let status = unsafe { cufftExecD2Z(handle, idata, odata) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }
        Ok(())
    }
}

impl<T: Element> Drop for RfftPlan<T> {
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
// IRFFT — complex (Hermitian-half) → real
// =============================================================================

/// Descriptor for an IRFFT (complex-to-real) op.
///
/// Note: cuFFT cannot infer the output length `n` from the Hermitian-
/// half input alone (both `2 * (n/2)` and `2 * (n/2) + 1` produce
/// inputs of length `n/2 + 1`). The descriptor carries `n` explicitly;
/// the input shape is then `[batch, n/2 + 1]` and output is
/// `[batch, n]`.
#[derive(Copy, Clone, Debug)]
pub struct IrfftDescriptor {
    /// Real output length. Input shape is `[batch, n/2 + 1]`.
    pub n: i32,
    /// Number of independent transforms in one launch.
    pub batch: i32,
    /// Real-side element type (output dtype) — `F32` or `F64`.
    pub element: ElementKind,
}

/// Args bundle for an IRFFT.
///
/// `T` is the *real* output type; `C` is the matching complex input
/// type ([`Complex32`] for `f32`, [`Complex64`] for `f64`).
pub struct IrfftArgs<'a, T: Element, C: Element> {
    /// Complex input tensor `[batch, n/2 + 1]`.
    pub x: TensorRef<'a, C, 2>,
    /// Real output tensor `[batch, n]`.
    pub y: TensorMut<'a, T, 2>,
}

/// 1-D IRFFT plan — Hermitian-half complex input → real output.
///
/// Wraps cuFFT's `cufftExecC2R` (`f32`) / `cufftExecZ2D` (`f64`),
/// followed by a `scale_inplace_real_*` launch that applies the
/// `1/n` normalization (PyTorch `norm="backward"`).
///
/// **When to use**: inverse FFT producing real-valued data. The
/// caller supplies `n` explicitly because the Hermitian-half input
/// length is ambiguous between `2 * (n/2)` and `2 * (n/2) + 1`.
///
/// **Dtypes**: `Complex32` → `f32`; `Complex64` → `f64`.
///
/// **Shape**: complex `[batch, n/2 + 1]` → real `[batch, n]`.
///
/// **Normalization**: normalized by `1/n` to match PyTorch's
/// `norm="backward"`.
///
/// **Workspace**: zero — cuFFT manages internal workspace.
///
/// **Precision guarantee**: deterministic; not bit-stable across
/// cuFFT versions.
///
/// Owns a lazy cuFFT handle (`!Sync` / `!Send`); destroyed on `Drop`.
pub struct IrfftPlan<T: Element> {
    desc: IrfftDescriptor,
    sku: KernelSku,
    handle: Cell<cufftHandle>,
    _marker: PhantomData<T>,
}

impl<T: Element> IrfftPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &IrfftDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::IrfftPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::IrfftPlan: C2R FFT supports f32 + f64 only",
            ));
        }
        if desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IrfftPlan: n must be > 0",
            ));
        }
        if desc.batch <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IrfftPlan: batch must be > 0",
            ));
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
        let mut handle: cufftHandle = HANDLE_UNINIT;
        let status = unsafe {
            cufftPlan1d(
                &mut handle as *mut _,
                self.desc.n,
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

impl IrfftPlan<f32> {
    /// Run the C2R FFT (single precision). Applies `1/n` normalization
    /// to the output to match PyTorch's `norm="backward"`.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: IrfftArgs<'_, f32, Complex32>,
    ) -> Result<()> {
        let n = self.desc.n;
        let batch = self.desc.batch;
        let in_shape = [batch, n / 2 + 1];
        let out_shape = [batch, n];
        if args.x.shape != in_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IrfftPlan<f32>: x shape != [batch, n/2 + 1]",
            ));
        }
        if args.y.shape != out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IrfftPlan<f32>: y shape != [batch, n]",
            ));
        }
        let in_numel = (batch as i64) * ((n / 2 + 1) as i64);
        let out_numel = (batch as i64) * (n as i64);
        if (args.x.data.len() as i64) < in_numel {
            return Err(Error::BufferTooSmall {
                needed: in_numel as usize,
                got: args.x.data.len(),
            });
        }
        if (args.y.data.len() as i64) < out_numel {
            return Err(Error::BufferTooSmall {
                needed: out_numel as usize,
                got: args.y.data.len(),
            });
        }
        if out_numel == 0 {
            return Ok(());
        }

        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;

        let idata = args.x.data.as_raw().0 as *mut cufftComplex;
        let odata = args.y.data.as_raw().0 as *mut f32;
        let status = unsafe { cufftExecC2R(handle, idata, odata) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }

        // Apply 1/n normalization.
        let scale = 1.0_f32 / (n as f32);
        let stream_ptr = stream.as_raw() as *mut c_void;
        let s = unsafe {
            baracuda_kernels_scale_inplace_real_f32_run(
                out_numel,
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

impl IrfftPlan<f64> {
    /// Run the C2R FFT (double precision).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: IrfftArgs<'_, f64, Complex64>,
    ) -> Result<()> {
        let n = self.desc.n;
        let batch = self.desc.batch;
        let in_shape = [batch, n / 2 + 1];
        let out_shape = [batch, n];
        if args.x.shape != in_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IrfftPlan<f64>: x shape != [batch, n/2 + 1]",
            ));
        }
        if args.y.shape != out_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::IrfftPlan<f64>: y shape != [batch, n]",
            ));
        }
        let in_numel = (batch as i64) * ((n / 2 + 1) as i64);
        let out_numel = (batch as i64) * (n as i64);
        if (args.x.data.len() as i64) < in_numel {
            return Err(Error::BufferTooSmall {
                needed: in_numel as usize,
                got: args.x.data.len(),
            });
        }
        if (args.y.data.len() as i64) < out_numel {
            return Err(Error::BufferTooSmall {
                needed: out_numel as usize,
                got: args.y.data.len(),
            });
        }
        if out_numel == 0 {
            return Ok(());
        }

        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;

        let idata = args.x.data.as_raw().0 as *mut cufftDoubleComplex;
        let odata = args.y.data.as_raw().0 as *mut f64;
        let status = unsafe { cufftExecZ2D(handle, idata, odata) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }

        let scale = 1.0_f64 / (n as f64);
        let stream_ptr = stream.as_raw() as *mut c_void;
        let s = unsafe {
            baracuda_kernels_scale_inplace_real_f64_run(
                out_numel,
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

impl<T: Element> Drop for IrfftPlan<T> {
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
