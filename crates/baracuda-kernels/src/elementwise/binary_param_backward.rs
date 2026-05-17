//! Backward plan for the parameterized binary elementwise family.
//!
//! Sibling of [`crate::BinaryParamPlan`]. For `Lerp`, the BW formula is
//! `da = (1 - weight)·dy`, `db = weight·dy` — no saved tensors are
//! needed because the gradient is a pure linear scaling of `dy` by
//! constants derived from the scalar weight.
//!
//! Today wired: `Lerp × {f32, f16, bf16, f64}`. The scalar `weight` is a
//! constant w.r.t. both inputs — no gradient flows to it.
//!
//! Trailblazer constraints: contig-only;
//! `dy.shape == da.shape == db.shape == desc.shape`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, BinaryKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a parameterized binary backward op. Same shape as the
/// FW descriptor.
#[derive(Copy, Clone, Debug)]
pub struct BinaryParamBackwardDescriptor<const N: usize> {
    /// Which forward parameterized binary op this is the backward of.
    pub kind: BinaryKind,
    /// Tensor shape (shared by dy / da / db).
    pub shape: [i32; N],
    /// Element type.
    pub element: ElementKind,
    /// Op-specific scalar parameter; same semantics as the FW
    /// descriptor's `param` field.
    pub param: f32,
}

/// Args bundle for a parameterized binary backward launch.
///
/// `Lerp` BW doesn't need saved forward inputs — the gradient is a pure
/// function of `dy` and the scalar param.
pub struct BinaryParamBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. `a`.
    pub da: TensorMut<'a, T, N>,
    /// Gradient w.r.t. `b`.
    pub db: TensorMut<'a, T, N>,
}

/// Parameterized binary backward plan.
pub struct BinaryParamBackwardPlan<T: Element, const N: usize> {
    desc: BinaryParamBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> BinaryParamBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &BinaryParamBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryParamBackwardPlan: descriptor element != T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BinaryParamBackwardPlan: shape dims must be non-negative",
                ));
            }
        }

        let kind_in_scope = matches!(desc.kind, BinaryKind::Lerp);
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !(kind_in_scope && dtype_in_scope) {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryParamBackwardPlan: today only `Lerp × \
                 {f32, f16, bf16, f64}` is wired.",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::BinaryElementwise,
            op: desc.kind as u16,
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
    pub fn can_implement(&self, args: &BinaryParamBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BinaryParamBackwardPlan: dy shape mismatch",
            ));
        }
        if args.da.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BinaryParamBackwardPlan: da shape mismatch",
            ));
        }
        if args.db.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BinaryParamBackwardPlan: db shape mismatch",
            ));
        }
        if !args.dy.is_contiguous() || !args.da.is_contiguous() || !args.db.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryParamBackwardPlan: contig-only trailblazer",
            ));
        }
        let numel = args.dy.numel();
        let dy_len = args.dy.data.len() as i64;
        let da_len = args.da.data.len() as i64;
        let db_len = args.db.data.len() as i64;
        if dy_len < numel || da_len < numel || db_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dy_len.min(da_len).min(db_len) as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
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

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: BinaryParamBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dy.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let da_ptr = args.da.data.as_raw().0 as *mut c_void;
        let db_ptr = args.db.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let p = self.desc.param;

        let status = match (self.desc.kind, T::KIND) {
            (BinaryKind::Lerp, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_lerp_backward_f32_run(
                    numel, dy_ptr, da_ptr, db_ptr, p,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Lerp, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_lerp_backward_f16_run(
                    numel, dy_ptr, da_ptr, db_ptr, p,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Lerp, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_lerp_backward_bf16_run(
                    numel, dy_ptr, da_ptr, db_ptr, p,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Lerp, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_lerp_backward_f64_run(
                    numel, dy_ptr, da_ptr, db_ptr, p,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BinaryParamBackwardPlan: dispatcher reached an \
                     unimplemented (kind, dtype) pair — select() should have caught this",
                ));
            }
        };
        map_status(status)
    }
}

fn map_status(code: i32) -> Result<()> {
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
