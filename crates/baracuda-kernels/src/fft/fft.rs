//! FFT / IFFT (complex-to-complex) — `FftPlan<T>` for `T = Complex32`
//! or `T = Complex64`.
//!
//! Wraps cuFFT's `cufftExecC2C` / `cufftExecZ2Z`. The descriptor's
//! `inverse: bool` selects the direction; both branches share the same
//! plan shape and the same handle (the cuFFT plan type — `CUFFT_C2C`
//! or `CUFFT_Z2Z` — is direction-agnostic, the direction tag is passed
//! at exec time).
//!
//! For inverse transforms the safe-plan layer applies the `1/N`
//! normalization by chaining a `scale_inplace_c{32,64}` launch onto the
//! same stream right after `cufftExec*`, baking PyTorch's
//! `norm="backward"` convention into the output without exposing the
//! split to the caller.

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    baracuda_kernels_scale_inplace_c32_run, baracuda_kernels_scale_inplace_c64_run, cufftComplex,
    cufftDestroy, cufftDoubleComplex, cufftExecC2C, cufftExecZ2Z, cufftHandle, cufftPlan1d,
    cufftSetStream, CUFFT_C2C, CUFFT_FORWARD, CUFFT_INVERSE, CUFFT_Z2Z,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Complex32, Complex64, Element, ElementKind, FftKind, KernelSku,
    MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Sentinel value for "cuFFT handle not yet created". cuFFT handles are
/// non-negative integers when live; `-1` is a safe out-of-band marker.
const HANDLE_UNINIT: cufftHandle = -1;

/// Descriptor for an FFT / IFFT (C2C) op.
#[derive(Copy, Clone, Debug)]
pub struct FftDescriptor {
    /// Signal length. Both input and output have shape `[batch, n]`
    /// (complex on both sides).
    pub n: i32,
    /// Number of independent FFTs to run in one launch. cuFFT lays the
    /// `batch` signals out contiguously — `x[b, i]` is at element
    /// offset `b * n + i`.
    pub batch: i32,
    /// `false` for forward (`cufftExec*` with `CUFFT_FORWARD`), `true`
    /// for inverse. Inverse is normalized by `1/n`.
    pub inverse: bool,
    /// Output / input element type. Must be `Complex32` or `Complex64`.
    pub element: ElementKind,
}

/// Args bundle for a C2C FFT.
///
/// Input and output are separate buffers (out-of-place). cuFFT does
/// support in-place exec (passing the same pointer for `idata` and
/// `odata`); for simplicity the plan layer routes out-of-place only —
/// callers wanting in-place semantics can alias `x` and `y` to the
/// same `DeviceBuffer` at the plan-args layer (which the lifetime
/// system would block, so for now we leave that to a follow-up that
/// exposes a dedicated in-place args shape).
pub struct FftArgs<'a, T: Element> {
    /// Input tensor `[batch, n]` (complex).
    pub x: TensorRef<'a, T, 2>,
    /// Output tensor `[batch, n]` (complex).
    pub y: TensorMut<'a, T, 2>,
}

/// 1-D FFT / IFFT (complex-to-complex) plan.
///
/// Wraps cuFFT's `cufftExecC2C` (`f32`) / `cufftExecZ2Z` (`f64`).
/// `descriptor.inverse` toggles direction at exec time; the cuFFT
/// plan type itself is direction-agnostic.
///
/// **When to use**: 1-D complex FFT in either direction. Use
/// [`super::RfftPlan`] / [`super::IrfftPlan`] for real input / output;
/// [`super::FftNdPlan`] for multi-axis transforms.
///
/// **Dtypes**: `Complex32`, `Complex64`. cuFFT does not expose
/// `f16` / `bf16` transforms.
///
/// **Shape**: `[batch, n]` — both buffers complex. Out-of-place only
/// in the trailblazer (in-place exec is a deferred follow-up).
///
/// **Normalization**: forward is unnormalized (matches PyTorch
/// `norm="backward"` default); inverse is normalized by `1/n` via a
/// chained `scale_inplace_c*` launch.
///
/// **Workspace**: zero — cuFFT manages internal workspace.
///
/// **Precision guarantee**: deterministic; not bit-stable across
/// different cuFFT build versions (cuFFT picks among several radix
/// algorithms based on `n`).
///
/// Owns a lazy cuFFT handle (`!Sync` / `!Send` via the
/// `Cell<cufftHandle>`). Destroyed on `Drop`.
pub struct FftPlan<T: Element> {
    desc: FftDescriptor,
    sku: KernelSku,
    handle: Cell<cufftHandle>,
    _marker: PhantomData<T>,
}

impl<T: Element> FftPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(_stream: &Stream, desc: &FftDescriptor, _pref: PlanPreference) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FftPlan: descriptor.element != T::KIND",
            ));
        }
        if !matches!(T::KIND, ElementKind::Complex32 | ElementKind::Complex64) {
            return Err(Error::Unsupported(
                "baracuda-kernels::FftPlan: C2C FFT supports Complex32 + Complex64 only",
            ));
        }
        if desc.n <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftPlan: n must be > 0",
            ));
        }
        if desc.batch <= 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftPlan: batch must be > 0",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::Complex64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: T::KIND,
            // cuFFT picks block-radix decompositions based on `n`'s
            // factorization; the floating-point reduction order is
            // implementation-defined and not guaranteed bit-stable
            // across cuFFT minor versions or arch generations.
            bit_stable_on_same_hardware: false,
            // The library is deterministic within a single run on a
            // fixed plan / stream / arch — no atomic accumulation, no
            // randomized scheduling.
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

    /// Workspace size in bytes. cuFFT manages its own internal
    /// workspace via `cufftPlan1d`; no caller-supplied workspace is
    /// required.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Lazily create the cuFFT plan. Idempotent.
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

    /// Bind the cuFFT plan to the caller's stream.
    fn bind_stream(&self, handle: cufftHandle, stream: &Stream) -> Result<()> {
        let stream_ptr = stream.as_raw() as *mut c_void;
        let status = unsafe { cufftSetStream(handle, stream_ptr) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }
        Ok(())
    }

    fn check_args(&self, x: &TensorRef<'_, T, 2>, y: &TensorMut<'_, T, 2>) -> Result<i64> {
        let expected = [self.desc.batch, self.desc.n];
        if x.shape != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftPlan: x shape != [batch, n]",
            ));
        }
        if y.shape != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftPlan: y shape != [batch, n]",
            ));
        }
        let numel = (self.desc.batch as i64) * (self.desc.n as i64);
        if (x.data.len() as i64) < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: x.data.len(),
            });
        }
        if (y.data.len() as i64) < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: y.data.len(),
            });
        }
        Ok(numel)
    }
}

// ----- Complex32 -------------------------------------------------------------

impl FftPlan<Complex32> {
    /// Run the C2C FFT (single precision).
    ///
    /// Performs `cufftExecC2C` with the descriptor's direction, then
    /// (for inverse) chains an in-place multiply-by-`1/n` to bake the
    /// PyTorch `norm="backward"` normalization into the output.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FftArgs<'_, Complex32>,
    ) -> Result<()> {
        let numel = self.check_args(&args.x, &args.y)?;
        if numel == 0 {
            return Ok(());
        }
        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;

        let direction = if self.desc.inverse {
            CUFFT_INVERSE
        } else {
            CUFFT_FORWARD
        };
        let idata = args.x.data.as_raw().0 as *mut cufftComplex;
        let odata = args.y.data.as_raw().0 as *mut cufftComplex;
        let status = unsafe { cufftExecC2C(handle, idata, odata, direction) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }

        if self.desc.inverse {
            let scale = 1.0_f32 / (self.desc.n as f32);
            let stream_ptr = stream.as_raw() as *mut c_void;
            let s = unsafe {
                baracuda_kernels_scale_inplace_c32_run(
                    numel,
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

impl FftPlan<Complex64> {
    /// Run the C2C FFT (double precision).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FftArgs<'_, Complex64>,
    ) -> Result<()> {
        let numel = self.check_args(&args.x, &args.y)?;
        if numel == 0 {
            return Ok(());
        }
        let handle = self.ensure_handle()?;
        self.bind_stream(handle, stream)?;

        let direction = if self.desc.inverse {
            CUFFT_INVERSE
        } else {
            CUFFT_FORWARD
        };
        let idata = args.x.data.as_raw().0 as *mut cufftDoubleComplex;
        let odata = args.y.data.as_raw().0 as *mut cufftDoubleComplex;
        let status = unsafe { cufftExecZ2Z(handle, idata, odata, direction) };
        if status != 0 {
            return Err(Error::CutlassInternal(cufft_to_status(status)));
        }

        if self.desc.inverse {
            let scale = 1.0_f64 / (self.desc.n as f64);
            let stream_ptr = stream.as_raw() as *mut c_void;
            let s = unsafe {
                baracuda_kernels_scale_inplace_c64_run(
                    numel,
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

impl<T: Element> Drop for FftPlan<T> {
    fn drop(&mut self) {
        let h = self.handle.get();
        if h != HANDLE_UNINIT {
            // Best-effort destroy. cuFFT may return non-zero if the
            // context is already torn down (e.g. process exit); we
            // don't surface that to the caller — the resource is
            // reclaimed by the driver regardless.
            unsafe {
                let _ = cufftDestroy(h);
            }
            self.handle.set(HANDLE_UNINIT);
        }
    }
}

/// Map a cuFFT result code into the safe-plan status integer. cuFFT
/// success is `0`; any non-zero is surfaced as a negative
/// `CutlassInternal(-code)` so the origin remains visible.
pub(crate) fn cufft_to_status(cufft_code: i32) -> i32 {
    if cufft_code == 0 {
        0
    } else {
        -cufft_code
    }
}

/// Map a bespoke-kernel status integer (the `0/1/2/3/4/5` ABI shared
/// across baracuda-kernels-sys) into a `Result<()>`.
pub(crate) fn map_status(code: i32) -> Result<()> {
    match code {
        0 => Ok(()),
        1 => Err(Error::MisalignedOperand),
        2 => Err(Error::InvalidProblem(
            "baracuda-kernels-sys reported invalid problem",
        )),
        3 => Err(Error::Unsupported(
            "baracuda-kernels-sys reported unsupported configuration",
        )),
        4 => Err(Error::WorkspaceTooSmall { needed: 0, got: 0 }),
        n => Err(Error::CutlassInternal(n)),
    }
}
