//! `triu` backward plan (Phase 13.4).
//!
//! The triu mask is a constant linear projector: the forward zeros the
//! sub-diagonal region of the last two dims. Its adjoint is the same
//! mask applied to the upstream gradient: `d_input = triu(d_output,
//! diagonal)` — entries kept by the forward pass receive `d_output`,
//! entries zeroed by the forward pass receive `0`. No saved forward
//! tensors are needed.
//!
//! Structurally identical to [`crate::TriuPlan`] with renamed
//! input/output → grad_output/grad_input; the kernel dispatch reuses
//! the FW launch symbol.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `triu` backward op.
///
/// Mirrors [`crate::TriuDescriptor`] — same `shape` and `diagonal`
/// because the mask is its own adjoint.
#[derive(Copy, Clone, Debug)]
pub struct TriuBackwardDescriptor<const N: usize> {
    /// Tensor shape (triu preserves shape — both `dy` and `dx` agree).
    pub shape: [i32; N],
    /// Diagonal offset (same as the forward).
    pub diagonal: i32,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a Triu backward launch.
///
/// No saved forward tensors are needed: `d_input = triu(d_output,
/// diagonal)` doesn't reference the forward `input` or `output`.
pub struct TriuBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient — same shape as the forward output (== input).
    pub grad_output: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input — same shape as `grad_output`.
    pub grad_input: TensorMut<'a, T, N>,
}

/// `triu` backward plan.
///
/// `d_input = torch.triu(d_output, diagonal)` — the same mask applied
/// to the upstream gradient. Adjoint of [`crate::TriuPlan`].
///
/// **When to use**: BW for [`TriuPlan`](crate::TriuPlan).
///
/// **Dtypes**: `{f16, bf16, f32, f64, i32, i64, Bool}`.
///
/// **Shape limits**: rank in `[2, 8]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact —
/// pure select-or-zero, reuses the FW kernel.
pub struct TriuBackwardPlan<T: Element, const N: usize> {
    desc: TriuBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> TriuBackwardPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &TriuBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::TriuBackwardPlan: descriptor element != type parameter T",
            ));
        }
        if N < 2 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TriuBackwardPlan: tensor rank must be >= 2",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::TriuBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::TriuBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        if !dtype_in_scope(T::KIND) {
            return Err(Error::Unsupported(
                "baracuda-kernels::TriuBackwardPlan: dtype not wired; supported set is \
                 {f16, bf16, f32, f64, i32, i64, Bool}",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::Triu as u16,
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
    pub fn can_implement(&self, args: &TriuBackwardArgs<'_, T, N>) -> Result<()> {
        if args.grad_output.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TriuBackwardPlan: grad_output shape mismatch with descriptor",
            ));
        }
        if args.grad_input.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::TriuBackwardPlan: grad_input shape mismatch with descriptor",
            ));
        }
        let numel = args.grad_input.numel();
        let dy_len = args.grad_output.data.len() as i64;
        let dx_len = args.grad_input.data.len() as i64;
        if dy_len < numel || dx_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dy_len.min(dx_len) as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes. Always `0`.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }
    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }
    /// Numerical guarantees.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch — dispatches to the forward `triu_<dtype>_run` symbol with
    /// `grad_output` as the input and `grad_input` as the output.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: TriuBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.grad_input.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.grad_output.data.as_raw().0 as *const c_void;
        let dx_ptr = args.grad_input.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;
        let shape = self.desc.shape;
        let rank = N as i32;
        let diagonal = self.desc.diagonal;

        let status = match T::KIND {
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_f16_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_bf16_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_f32_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_f64_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::I32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_i32_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::I64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_i64_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            ElementKind::Bool => unsafe {
                baracuda_kernels_sys::baracuda_kernels_triu_bool_run(
                    dy_ptr, dx_ptr, shape.as_ptr(), rank, diagonal, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::TriuBackwardPlan::run: dtype not wired",
                ));
            }
        };
        map_status(status)
    }
}

fn dtype_in_scope(k: ElementKind) -> bool {
    matches!(
        k,
        ElementKind::F16
            | ElementKind::Bf16
            | ElementKind::F32
            | ElementKind::F64
            | ElementKind::I32
            | ElementKind::I64
            | ElementKind::Bool
    )
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
