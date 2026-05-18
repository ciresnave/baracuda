//! `fftshift` / `ifftshift` — bespoke index-permutation kernels.
//!
//! cuFFT has no native shift, so these are hand-rolled kernels in
//! `baracuda-kernels-sys`. The kernel is element-width-generic (4 / 8
//! / 16-byte cells) so the same code covers `f32`, `f64`,
//! [`Complex32`], and [`Complex64`] without per-type templating —
//! shift is a pure index permutation (no arithmetic on the element
//! values), so the element type is irrelevant beyond its byte width.
//!
//! 1-D shifts along the last axis of a `[batch, n]` tensor — matches
//! NumPy / PyTorch convention:
//!
//! - `fftshift`:  `y[b, i] = x[b, (i + (n+1)/2) % n]` (equiv. roll(x, n//2))
//! - `ifftshift`: `y[b, i] = x[b, (i + n/2)     % n]` (equiv. roll(x, -(n//2)))
//!
//! For even `n` the two are identical (the `n/2` offset is self-
//! inverse mod `n`). For odd `n` the cyclic offsets differ by one
//! cell and `ifftshift` is the genuine inverse of `fftshift`
//! (`ifftshift(fftshift(x)) == x` for any `n`).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    baracuda_kernels_fftshift_16_run, baracuda_kernels_fftshift_4_run,
    baracuda_kernels_fftshift_8_run, baracuda_kernels_ifftshift_16_run,
    baracuda_kernels_ifftshift_4_run, baracuda_kernels_ifftshift_8_run,
};
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, FftKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::fft::map_status;

/// Descriptor for an fftshift / ifftshift op.
#[derive(Copy, Clone, Debug)]
pub struct FftShiftDescriptor {
    /// Length of the last axis (the axis being shifted). For 1-D
    /// fftshift `[batch, n]` this is the size of the shifted axis.
    pub n: i32,
    /// Number of independent rows. Each row is shifted independently.
    pub batch: i32,
    /// `true` selects `ifftshift` (cyclic offset `n/2`), `false`
    /// selects `fftshift` (cyclic offset `(n+1)/2`). Identical for
    /// even `n`; the two diverge for odd `n` and `ifftshift` is then
    /// the true inverse of `fftshift`.
    pub inverse: bool,
    /// Element type. Any [`Element`]; the kernel dispatches on
    /// `size_of::<T>()` (4 / 8 / 16 bytes).
    pub element: ElementKind,
}

/// Args bundle for an fftshift / ifftshift.
pub struct FftShiftArgs<'a, T: Element> {
    /// Input tensor `[batch, n]`.
    pub x: TensorRef<'a, T, 2>,
    /// Output tensor `[batch, n]`. Must be distinct from `x` (the
    /// kernel reads from `x` and writes to `y` without scratch — in-
    /// place shift would require a 2-phase swap, which the trailblazer
    /// doesn't ship).
    pub y: TensorMut<'a, T, 2>,
}

/// 1-D `fftshift` / `ifftshift` plan — bespoke index-permutation
/// kernel.
///
/// Cyclically shifts the last axis of a `[batch, n]` tensor by `n/2`
/// (ifftshift) or `(n+1)/2` (fftshift). Matches NumPy / PyTorch
/// conventions. For even `n` the two directions are identical; for odd
/// `n` `ifftshift` is the genuine inverse of `fftshift`.
///
/// **When to use**: place the DC component at the centre of an FFT
/// output (or vice versa). Use [`super::FftShiftNdPlan`] for shifts
/// over multiple axes.
///
/// **Dtypes**: any [`Element`] — kernel dispatches on
/// `size_of::<T>()` (4 / 8 / 16-byte cells), so `f32`, `f64`,
/// `Complex32`, `Complex64` all work without per-type templating.
///
/// **Shape**: `[batch, n]`. Out-of-place only (in-place shift would
/// need a 2-phase swap).
///
/// **Workspace**: zero.
///
/// **Precision guarantee**: bit-exact (pure index permutation, no
/// arithmetic).
///
/// No cuFFT handle / state — the plan is just configuration.
pub struct FftShiftPlan<T: Element> {
    desc: FftShiftDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> FftShiftPlan<T> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &FftShiftDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FftShiftPlan: descriptor.element != T::KIND",
            ));
        }
        // Kernel handles cells of 4, 8, or 16 bytes — any baracuda
        // [`Element`] with one of those widths is supported. Today
        // that's f32 / f64 / Complex32 / Complex64; rejecting the
        // others up front keeps the supported set narrow and obvious.
        let size = core::mem::size_of::<T>();
        if !matches!(size, 4 | 8 | 16) {
            return Err(Error::Unsupported(
                "baracuda-kernels::FftShiftPlan: only 4/8/16-byte element types supported",
            ));
        }
        if desc.n < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftShiftPlan: n must be >= 0",
            ));
        }
        if desc.batch < 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftShiftPlan: batch must be >= 0",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 | ElementKind::Complex64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: T::KIND,
            // Pure index permutation — bit-exact, no arithmetic.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let op = if desc.inverse {
            FftKind::IfftShift
        } else {
            FftKind::FftShift
        };
        let sku = KernelSku {
            category: OpCategory::Fft,
            op: op as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::Bespoke,
            precision_guarantee,
        };

        Ok(Self {
            desc: *desc,
            sku,
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

    /// Run the fftshift / ifftshift.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FftShiftArgs<'_, T>,
    ) -> Result<()> {
        let expected = [self.desc.batch, self.desc.n];
        if args.x.shape != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftShiftPlan: x shape != [batch, n]",
            ));
        }
        if args.y.shape != expected {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftShiftPlan: y shape != [batch, n]",
            ));
        }
        let numel = (self.desc.batch as i64) * (self.desc.n as i64);
        if (args.x.data.len() as i64) < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: args.x.data.len(),
            });
        }
        if (args.y.data.len() as i64) < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: args.y.data.len(),
            });
        }
        if numel == 0 {
            return Ok(());
        }

        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let batch = self.desc.batch as i64;
        let n = self.desc.n;

        let size = core::mem::size_of::<T>();
        let status = unsafe {
            match (size, self.desc.inverse) {
                (4, false) => baracuda_kernels_fftshift_4_run(
                    batch, n, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                (4, true) => baracuda_kernels_ifftshift_4_run(
                    batch, n, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                (8, false) => baracuda_kernels_fftshift_8_run(
                    batch, n, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                (8, true) => baracuda_kernels_ifftshift_8_run(
                    batch, n, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                (16, false) => baracuda_kernels_fftshift_16_run(
                    batch, n, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                (16, true) => baracuda_kernels_ifftshift_16_run(
                    batch, n, x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                ),
                _ => unreachable!("select() gates on size_of::<T>() in 4 / 8 / 16"),
            }
        };
        map_status(status)
    }
}
