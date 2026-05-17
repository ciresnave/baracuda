//! Parameterized binary elementwise plan.
//!
//! Sibling of [`crate::BinaryPlan`] for ops that carry a scalar parameter
//! alongside their two tensor inputs. Today wired: `Lerp`
//! (`y = a + weight·(b - a)`; `weight = param`) across
//! `{f32, f16, bf16, f64}` — FW only here, the BW lives in
//! [`crate::BinaryParamBackwardPlan`].
//!
//! Single-scalar-param shape (one `f32` field, not the unary's
//! `[f32; 2]`) — binary ops in this family take exactly one broadcast
//! scalar per the PyTorch / JAX `lerp(a, b, w)` precedent. Future
//! single-param binary ops (none today) would extend the dispatcher's
//! match arms.
//!
//! Trailblazer constraints: contig-only (no strided variant);
//! `a.shape == b.shape == y.shape == desc.shape`. No broadcasting.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, BinaryKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a parameterized binary elementwise op.
#[derive(Copy, Clone, Debug)]
pub struct BinaryParamDescriptor<const N: usize> {
    /// Which parameterized binary op to apply.
    pub kind: BinaryKind,
    /// Tensor shape (shared by `a` / `b` / `y`).
    pub shape: [i32; N],
    /// Primary element type.
    pub element: ElementKind,
    /// Op-specific scalar parameter. For `Lerp`, this is the broadcast
    /// `weight`. Half-precision kernels do the arithmetic in `f32`; the
    /// `f64` kernel widens the `f32` parameter losslessly.
    pub param: f32,
}

/// Args bundle for a parameterized binary elementwise launch.
pub struct BinaryParamArgs<'a, T: Element, const N: usize> {
    /// First input.
    pub a: TensorRef<'a, T, N>,
    /// Second input.
    pub b: TensorRef<'a, T, N>,
    /// Output.
    pub y: TensorMut<'a, T, N>,
}

/// Parameterized binary elementwise plan.
pub struct BinaryParamPlan<T: Element, const N: usize> {
    desc: BinaryParamDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> BinaryParamPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &BinaryParamDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryParamPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::BinaryParamPlan: shape dims must be non-negative",
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
                "baracuda-kernels::BinaryParamPlan: today only `Lerp × {f32, f16, bf16, f64}` \
                 is wired; other parameterized binary ops join in later fanout.",
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
    pub fn can_implement(&self, args: &BinaryParamArgs<'_, T, N>) -> Result<()> {
        if args.a.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BinaryParamPlan: A shape mismatch",
            ));
        }
        if args.b.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BinaryParamPlan: B shape mismatch",
            ));
        }
        if args.y.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::BinaryParamPlan: Y shape mismatch",
            ));
        }
        if !args.a.is_contiguous() || !args.b.is_contiguous() || !args.y.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::BinaryParamPlan: contig-only trailblazer; strided fanout \
                 lands later",
            ));
        }
        let numel = args.y.numel();
        let a_len = args.a.data.len() as i64;
        let b_len = args.b.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if a_len < numel || b_len < numel || y_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: a_len.min(b_len).min(y_len) as usize,
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
        args: BinaryParamArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.y.numel();
        if numel == 0 {
            return Ok(());
        }
        let a_ptr = args.a.data.as_raw().0 as *const c_void;
        let b_ptr = args.b.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let p = self.desc.param;

        let status = match (self.desc.kind, T::KIND) {
            (BinaryKind::Lerp, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_lerp_f32_run(
                    numel, a_ptr, b_ptr, y_ptr, p,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Lerp, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_lerp_f16_run(
                    numel, a_ptr, b_ptr, y_ptr, p,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Lerp, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_lerp_bf16_run(
                    numel, a_ptr, b_ptr, y_ptr, p,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (BinaryKind::Lerp, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_binary_lerp_f64_run(
                    numel, a_ptr, b_ptr, y_ptr, p,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::BinaryParamPlan: dispatcher reached an unimplemented \
                     (kind, dtype) pair — select() should have caught this",
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
