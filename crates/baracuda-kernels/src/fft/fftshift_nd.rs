//! N-D `fftshift` / `ifftshift` — single-pass general-permutation plan.
//!
//! Companion to the 1-D [`super::FftShiftPlan`] that shifts a subset of
//! axes of a rank-`N` tensor (`1 <= N <= 8`) in one kernel launch
//! instead of chaining axis-by-axis 1-D shifts. One thread per output
//! cell — each thread decomposes its flat output index into per-axis
//! coords using the (dense, contiguous) output strides, rotates each
//! shifted axis by its per-axis offset, then recomposes the source flat
//! index against the same strides. Bandwidth-bound; bit-exact (pure
//! index permutation, no arithmetic on values).
//!
//! The same kernel covers both forward and inverse shifts — the
//! direction lives entirely in the per-axis shift amounts:
//!
//! - `fftshift`:  axis offset `n / 2`
//! - `ifftshift`: axis offset `n - n / 2`
//!
//! For even axis lengths the two are identical (the offset is self-
//! inverse mod `n`); for odd lengths they differ by one cell and
//! `ifftshift` is the genuine inverse of `fftshift`.
//!
//! ## Layout contract
//!
//! Input and output must both be dense and contiguous (the kernel
//! decomposes the output's flat index using its strides and recomposes
//! the source index with the same strides — strides cancel out as long
//! as both tensors share them). The plan checks
//! `stride == contiguous_stride(shape)` for both operands. The strided
//! variant is not shipped in the trailblazer.
//!
//! ## Rank cap
//!
//! `1 <= N <= 8` matches the kernel's compile-time `kFftShiftNdMaxRank`.
//! The number of *shifted* axes is independently capped at 4 in the
//! descriptor — plenty for real-world use (2-D and 3-D spectral data
//! is the dominant case; rank-4 shifts come up for batched complex
//! volumes).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_sys::{
    baracuda_kernels_fftshift_nd_16_run, baracuda_kernels_fftshift_nd_4_run,
    baracuda_kernels_fftshift_nd_8_run,
};
use baracuda_kernels_types::{
    contiguous_stride, ArchSku, BackendKind, Element, ElementKind, FftKind, KernelSku,
    MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::fft::map_status;

/// Maximum rank supported by the N-D fftshift kernel. Matches the
/// kernel's compile-time `kFftShiftNdMaxRank`.
pub const FFTSHIFT_ND_MAX_RANK: usize = 8;

/// Maximum number of *shifted* axes carried by the descriptor.
pub const FFTSHIFT_ND_MAX_SHIFT_AXES: usize = 4;

/// Descriptor for an N-D `fftshift` / `ifftshift` op.
///
/// Carries the operand shape, the rank, the indices of the axes that
/// are shifted (the rest are pass-through), the direction (`inverse`),
/// and the element type.
#[derive(Copy, Clone, Debug)]
pub struct FftShiftNdDescriptor {
    /// Operand shape. Only `shape[0..ndim]` is read; the trailing
    /// slots are ignored. `shape[0]` is the outermost (slowest) axis.
    pub shape: [i32; FFTSHIFT_ND_MAX_RANK],
    /// Rank of the operand. `1..=8`.
    pub ndim: u8,
    /// Indices of the axes to shift, in `[0, ndim)`. Only the first
    /// `num_shift_axes` slots are read. Duplicates are rejected by
    /// [`Self::select`]. The remaining axes are pass-through.
    pub shift_axes: [u8; FFTSHIFT_ND_MAX_SHIFT_AXES],
    /// Number of shifted axes. `0..=ndim.min(4)`. `0` is the identity
    /// (degenerate but legal — emits a pure memcpy).
    pub num_shift_axes: u8,
    /// `true` selects `ifftshift` (per-axis offset `n - n/2`), `false`
    /// selects `fftshift` (per-axis offset `n/2`). The two are
    /// identical for even axis lengths; they differ by one cell on odd
    /// lengths and `ifftshift` is then the genuine inverse of
    /// `fftshift`.
    pub inverse: bool,
    /// Element type. Any [`Element`]; the kernel dispatches on
    /// `size_of::<T>()` (4 / 8 / 16 bytes).
    pub element: ElementKind,
}

/// Args bundle for an N-D `fftshift` / `ifftshift` launch.
pub struct FftShiftNdArgs<'a, T: Element, const N: usize> {
    /// Input tensor (rank `N`). Must be dense and contiguous.
    pub input: TensorRef<'a, T, N>,
    /// Output tensor (rank `N`, same shape as input). Must be dense /
    /// contiguous and distinct from the input — the kernel reads from
    /// `input` and writes to `output` without scratch, so in-place
    /// would alias the permutation.
    pub output: TensorMut<'a, T, N>,
}

/// N-D `fftshift` / `ifftshift` plan — single-pass general-permutation
/// kernel.
///
/// Shifts a caller-selected subset of axes (up to
/// [`FFTSHIFT_ND_MAX_SHIFT_AXES`] = 4) of a rank-up-to-
/// [`FFTSHIFT_ND_MAX_RANK`] = 8 tensor in one kernel launch. The
/// kernel decomposes / recomposes the flat index against the
/// (dense, contiguous) strides.
///
/// **When to use**: ND fftshift over 2-D / 3-D / 4-D spectral data.
/// Use [`super::FftShiftPlan`] for the 1-D fast path. Chaining 1-D
/// shifts axis-by-axis works too but costs an extra launch each.
///
/// **Dtypes**: any [`Element`] — kernel dispatches on
/// `size_of::<T>()` (4 / 8 / 16-byte cells).
///
/// **Shape**: rank in `1..=8`; both operands must be dense and
/// contiguous. Out-of-place only.
///
/// **Workspace**: zero.
///
/// **Precision guarantee**: bit-exact (pure index permutation).
///
/// No cuFFT handle / state — bespoke kernel.
pub struct FftShiftNdPlan<T: Element, const N: usize> {
    desc: FftShiftNdDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> FftShiftNdPlan<T, N> {
    /// Pick a kernel + validate the descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &FftShiftNdDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FftShiftNdPlan: descriptor.element != T::KIND",
            ));
        }
        let size = core::mem::size_of::<T>();
        if !matches!(size, 4 | 8 | 16) {
            return Err(Error::Unsupported(
                "baracuda-kernels::FftShiftNdPlan: only 4/8/16-byte element types supported",
            ));
        }
        let ndim = desc.ndim as usize;
        if ndim == 0 || ndim > FFTSHIFT_ND_MAX_RANK {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftShiftNdPlan: ndim must be in 1..=8",
            ));
        }
        if ndim != N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftShiftNdPlan: descriptor.ndim != const N",
            ));
        }
        for &d in &desc.shape[..ndim] {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::FftShiftNdPlan: shape dims must be non-negative",
                ));
            }
        }
        let num_shift_axes = desc.num_shift_axes as usize;
        if num_shift_axes > FFTSHIFT_ND_MAX_SHIFT_AXES || num_shift_axes > ndim {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftShiftNdPlan: num_shift_axes out of range",
            ));
        }
        // Validate shift_axes indices and reject duplicates.
        let mut seen = [false; FFTSHIFT_ND_MAX_RANK];
        for &axis in &desc.shift_axes[..num_shift_axes] {
            let a = axis as usize;
            if a >= ndim {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::FftShiftNdPlan: shift_axes entry out of range",
                ));
            }
            if seen[a] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::FftShiftNdPlan: duplicate entry in shift_axes",
                ));
            }
            seen[a] = true;
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

    /// Validate the arg bundle against the descriptor.
    pub fn can_implement(&self, args: &FftShiftNdArgs<'_, T, N>) -> Result<()> {
        let ndim = self.desc.ndim as usize;
        let mut expected_shape = [0i32; N];
        // Copy the leading `ndim` entries (guaranteed == N by `select`).
        for i in 0..ndim {
            expected_shape[i] = self.desc.shape[i];
        }
        if args.input.shape != expected_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftShiftNdPlan: input shape != descriptor.shape",
            ));
        }
        if args.output.shape != expected_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FftShiftNdPlan: output shape != descriptor.shape",
            ));
        }
        // Kernel decomposes flat indices using contiguous strides — both
        // operands must be dense / contiguous.
        let contig = contiguous_stride(expected_shape);
        if args.input.stride != contig {
            return Err(Error::Unsupported(
                "baracuda-kernels::FftShiftNdPlan: input must be contiguous",
            ));
        }
        if args.output.stride != contig {
            return Err(Error::Unsupported(
                "baracuda-kernels::FftShiftNdPlan: output must be contiguous",
            ));
        }
        let numel = args.output.numel();
        let x_len = args.input.data.len() as i64;
        let y_len = args.output.data.len() as i64;
        if x_len < numel || y_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: x_len.min(y_len) as usize,
            });
        }
        Ok(())
    }

    /// Launch the N-D shift.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FftShiftNdArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.output.numel();
        if numel == 0 {
            return Ok(());
        }

        // Marshal per-axis arrays at full kernel rank (= ndim = N).
        let ndim = self.desc.ndim as usize;
        let mut shape_arr = [0i32; FFTSHIFT_ND_MAX_RANK];
        let mut shift_amt_arr = [0i32; FFTSHIFT_ND_MAX_RANK];
        let mut stride_arr = [0i64; FFTSHIFT_ND_MAX_RANK];

        for i in 0..ndim {
            shape_arr[i] = self.desc.shape[i];
            // Stride layout matches the kernel's expectation
            // (`stride[d] = product(shape[d+1..ndim])` — dense row-
            // major).
            stride_arr[i] = args.output.stride[i];
        }
        // Build the per-axis shift map from the shifted-axes set.
        let num_shift_axes = self.desc.num_shift_axes as usize;
        for &axis in &self.desc.shift_axes[..num_shift_axes] {
            let a = axis as usize;
            let n = shape_arr[a];
            let half = n / 2;
            shift_amt_arr[a] = if self.desc.inverse { n - half } else { half };
        }

        let x_ptr = args.input.data.as_raw().0 as *const c_void;
        let y_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let rank = ndim as i32;

        let size = core::mem::size_of::<T>();
        let status = unsafe {
            match size {
                4 => baracuda_kernels_fftshift_nd_4_run(
                    numel,
                    rank,
                    shape_arr.as_ptr(),
                    shift_amt_arr.as_ptr(),
                    stride_arr.as_ptr(),
                    x_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                ),
                8 => baracuda_kernels_fftshift_nd_8_run(
                    numel,
                    rank,
                    shape_arr.as_ptr(),
                    shift_amt_arr.as_ptr(),
                    stride_arr.as_ptr(),
                    x_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                ),
                16 => baracuda_kernels_fftshift_nd_16_run(
                    numel,
                    rank,
                    shape_arr.as_ptr(),
                    shift_amt_arr.as_ptr(),
                    stride_arr.as_ptr(),
                    x_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                ),
                _ => unreachable!("select() gates on size_of::<T>() in 4 / 8 / 16"),
            }
        };
        map_status(status)
    }
}
