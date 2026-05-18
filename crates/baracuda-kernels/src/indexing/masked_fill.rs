//! `masked_fill` plan — Category L.
//!
//! `out[i] = mask[i] ? value : src[i]`. PyTorch
//! `torch.Tensor.masked_fill`. Mask is `u8` (Bool storage) — non-zero
//! is treated as true.
//!
//! Trailblazer dtype coverage: `f32, f64, i32, bool`. The kernel is pure
//! element-select — no arithmetic — so output is bit-exact at every
//! dtype.
//!
//! Trailblazer constraint: same-shape only (no broadcast). The
//! descriptor's `fill_bits` field carries the fill value as a 64-bit
//! payload that the kernel reinterprets into the element type at
//! launch. Helper constructors are provided for each supported dtype.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IndexingKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::gather::map_status;

/// Descriptor for a `masked_fill` op.
#[derive(Copy, Clone, Debug)]
pub struct MaskedFillDescriptor<const N: usize> {
    /// Shape of src / mask / out (must match across all three).
    pub shape: [i32; N],
    /// Fill value reinterpreted into T at the kernel entry. Caller
    /// encodes via the appropriate helper:
    /// - `f32`: `value.to_bits() as i64`
    /// - `f64`: `value.to_bits() as i64`
    /// - `i32`: `value as i64`
    /// - `bool`: `if value { 1 } else { 0 }`
    pub fill_bits: i64,
    /// Value element type.
    pub element: ElementKind,
}

impl<const N: usize> MaskedFillDescriptor<N> {
    /// Build a descriptor with an `f32` fill value.
    pub fn new_f32(shape: [i32; N], value: f32) -> Self {
        Self {
            shape,
            fill_bits: value.to_bits() as i64,
            element: ElementKind::F32,
        }
    }

    /// Build a descriptor with an `f64` fill value.
    pub fn new_f64(shape: [i32; N], value: f64) -> Self {
        Self {
            shape,
            fill_bits: value.to_bits() as i64,
            element: ElementKind::F64,
        }
    }

    /// Build a descriptor with an `i32` fill value.
    pub fn new_i32(shape: [i32; N], value: i32) -> Self {
        Self {
            shape,
            fill_bits: value as i64,
            element: ElementKind::I32,
        }
    }

    /// Build a descriptor with a `bool` fill value.
    pub fn new_bool(shape: [i32; N], value: bool) -> Self {
        Self {
            shape,
            fill_bits: if value { 1 } else { 0 },
            element: ElementKind::Bool,
        }
    }
}

/// Args bundle for a `masked_fill` launch.
pub struct MaskedFillArgs<'a, T: Element, const N: usize> {
    /// Source tensor.
    pub src: TensorRef<'a, T, N>,
    /// Boolean mask (`u8`, 0 = false, non-zero = true). Same shape as
    /// `src` and `out`.
    pub mask: TensorRef<'a, u8, N>,
    /// Output.
    pub out: TensorMut<'a, T, N>,
}

/// `masked_fill` plan.
///
/// `out[i] = mask[i] ? value : src[i]` (PyTorch
/// `torch.Tensor.masked_fill`).
///
/// **When to use**: forward `masked_fill`. Pair with
/// [`MaskedFillBackwardPlan`](crate::MaskedFillBackwardPlan) — the BW
/// simply zeros gradient at mask=true positions, so it's pure
/// element-select like the FW.
///
/// **Dtypes**: `{f32, f64, i32, bool}`. Mask is always `u8` (Bool
/// storage; non-zero = true).
///
/// **Shape limits**: rank in `[1, 8]`; same-shape only (no broadcast
/// in the trailblazer). Use one of the `MaskedFillDescriptor::new_*`
/// constructors to encode the fill value into `fill_bits`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable on same
/// hardware. Pure element-select, no arithmetic — bit-exact.
pub struct MaskedFillPlan<T: Element, const N: usize> {
    desc: MaskedFillDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> MaskedFillPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &MaskedFillDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::MaskedFillPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::MaskedFillPlan: shape dims must be non-negative",
                ));
            }
        }

        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::I32 | ElementKind::Bool
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::MaskedFillPlan: today only `f32`, `f64`, `i32`, `bool` wired",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Indexing,
            op: IndexingKind::MaskedFill as u16,
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

    /// Validate args.
    pub fn can_implement(&self, args: &MaskedFillArgs<'_, T, N>) -> Result<()> {
        if args.src.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MaskedFillPlan: src shape mismatch with descriptor",
            ));
        }
        if args.mask.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MaskedFillPlan: mask shape mismatch with descriptor",
            ));
        }
        if args.out.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MaskedFillPlan: out shape mismatch with descriptor",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::MaskedFillPlan: tensor rank > 8 not supported",
            ));
        }
        let numel = args.out.numel();
        let src_len = args.src.data.len() as i64;
        let mask_len = args.mask.data.len() as i64;
        let out_len = args.out.data.len() as i64;
        if src_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: src_len as usize,
            });
        }
        if mask_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: mask_len as usize,
            });
        }
        if out_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: out_len as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees for this plan's kernel.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: MaskedFillArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.out.numel();
        if numel == 0 {
            return Ok(());
        }
        let src_ptr = args.src.data.as_raw().0 as *const c_void;
        let mask_ptr = args.mask.data.as_raw().0 as *const c_void;
        let out_ptr = args.out.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_masked_fill_f32_run(
                    numel,
                    src_ptr,
                    mask_ptr,
                    out_ptr,
                    self.desc.fill_bits,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_masked_fill_f64_run(
                    numel,
                    src_ptr,
                    mask_ptr,
                    out_ptr,
                    self.desc.fill_bits,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_masked_fill_i32_run(
                    numel,
                    src_ptr,
                    mask_ptr,
                    out_ptr,
                    self.desc.fill_bits,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bool => unsafe {
                baracuda_kernels_sys::baracuda_kernels_masked_fill_bool_run(
                    numel,
                    src_ptr,
                    mask_ptr,
                    out_ptr,
                    self.desc.fill_bits,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::MaskedFillPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
