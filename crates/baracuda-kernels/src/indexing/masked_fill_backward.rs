//! `masked_fill_backward` plan — Category L BW.
//!
//! Adjoint of [`crate::indexing::MaskedFillPlan`]:
//! `dsrc[i] = mask[i] ? 0 : dout[i]`. The fill `value` is a
//! non-differentiable scalar — no `dvalue` is produced.
//!
//! Trailblazer dtype coverage: `f32, f64, i32, bool`. No arithmetic
//! (pure mask + copy + zero), so bit-exact at every dtype.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IndexingKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::gather::map_status;

/// Descriptor for a `masked_fill_backward` op.
#[derive(Copy, Clone, Debug)]
pub struct MaskedFillBackwardDescriptor<const N: usize> {
    /// Shape of dout / mask / dsrc.
    pub shape: [i32; N],
    /// Value element type.
    pub element: ElementKind,
}

/// Args bundle for a `masked_fill_backward` launch.
pub struct MaskedFillBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dout: TensorRef<'a, T, N>,
    /// Boolean mask from the FW pass (`u8`, 0 = false).
    pub mask: TensorRef<'a, u8, N>,
    /// Gradient w.r.t. `src` (the differentiable input of the FW).
    pub dsrc: TensorMut<'a, T, N>,
}

/// `masked_fill_backward` plan.
pub struct MaskedFillBackwardPlan<T: Element, const N: usize> {
    desc: MaskedFillBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> MaskedFillBackwardPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &MaskedFillBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::MaskedFillBackwardPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::MaskedFillBackwardPlan: shape dims must be non-negative",
                ));
            }
        }

        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F64 | ElementKind::I32 | ElementKind::Bool
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::MaskedFillBackwardPlan: today only \
                 `f32`, `f64`, `i32`, `bool` wired",
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
            op: IndexingKind::MaskedFillBackward as u16,
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
    pub fn can_implement(&self, args: &MaskedFillBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dout.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MaskedFillBackwardPlan: dout shape mismatch with descriptor",
            ));
        }
        if args.mask.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MaskedFillBackwardPlan: mask shape mismatch with descriptor",
            ));
        }
        if args.dsrc.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::MaskedFillBackwardPlan: dsrc shape mismatch with descriptor",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::MaskedFillBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        let numel = args.dout.numel();
        let dout_len = args.dout.data.len() as i64;
        let mask_len = args.mask.data.len() as i64;
        let dsrc_len = args.dsrc.data.len() as i64;
        if dout_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dout_len as usize,
            });
        }
        if mask_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: mask_len as usize,
            });
        }
        if dsrc_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dsrc_len as usize,
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
        args: MaskedFillBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dout.numel();
        if numel == 0 {
            return Ok(());
        }
        let dout_ptr = args.dout.data.as_raw().0 as *const c_void;
        let mask_ptr = args.mask.data.as_raw().0 as *const c_void;
        let dsrc_ptr = args.dsrc.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_masked_fill_backward_f32_run(
                    numel,
                    dout_ptr,
                    mask_ptr,
                    dsrc_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_masked_fill_backward_f64_run(
                    numel,
                    dout_ptr,
                    mask_ptr,
                    dsrc_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_masked_fill_backward_i32_run(
                    numel,
                    dout_ptr,
                    mask_ptr,
                    dsrc_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bool => unsafe {
                baracuda_kernels_sys::baracuda_kernels_masked_fill_backward_bool_run(
                    numel,
                    dout_ptr,
                    mask_ptr,
                    dsrc_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::MaskedFillBackwardPlan::run reached an unimplemented dtype \
                     — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}
