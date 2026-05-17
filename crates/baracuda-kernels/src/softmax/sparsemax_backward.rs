//! Sparsemax backward plan — Jacobian-vector product.
//!
//! For active positions (`y > 0`):
//!   `dx[i] = dy[i] - sum_dy_active / n_active`
//! where `sum_dy_active = Σ_{j: y[j] > 0} dy[j]` and `n_active` counts
//! the actives in the row. Inactive positions get `dx[i] = 0`.
//!
//! Needs saved forward output `y` (drives the active mask).
//!
//! Wired today: `T ∈ {f32, f16, bf16, f64}`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, SoftmaxKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a Sparsemax backward op.
#[derive(Copy, Clone, Debug)]
pub struct SparsemaxBackwardDescriptor<const N: usize> {
    /// Tensor shape (dy / y / dx share it).
    pub input_shape: [i32; N],
    /// Forward sparsemax axis.
    pub softmax_axis: u8,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a Sparsemax backward launch.
///
/// `y` is the SAVED forward output (used to derive the active mask).
pub struct SparsemaxBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward output.
    pub y: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input.
    pub dx: TensorMut<'a, T, N>,
}

/// Sparsemax backward plan.
pub struct SparsemaxBackwardPlan<T: Element, const N: usize> {
    desc: SparsemaxBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> SparsemaxBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SparsemaxBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SparsemaxBackwardPlan: descriptor element != T",
            ));
        }
        if (desc.softmax_axis as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SparsemaxBackwardPlan: softmax_axis out of range",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::SparsemaxBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::SparsemaxBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_fp_family {
            return Err(Error::Unsupported(
                "baracuda-kernels::SparsemaxBackwardPlan: wired today: {f32, f16, bf16, f64}",
            ));
        }

        let math_precision = match T::KIND {
            ElementKind::F64 => MathPrecision::F64,
            _ => MathPrecision::F32,
        };
        let precision_guarantee = PrecisionGuarantee {
            math_precision,
            accumulator: match T::KIND {
                ElementKind::F64 => ElementKind::F64,
                _ => ElementKind::F32,
            },
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Softmax,
            op: SoftmaxKind::Sparsemax as u16,
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
    pub fn can_implement(&self, args: &SparsemaxBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SparsemaxBackwardPlan: dy shape mismatch",
            ));
        }
        if args.y.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SparsemaxBackwardPlan: y shape mismatch",
            ));
        }
        if args.dx.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SparsemaxBackwardPlan: dx shape mismatch",
            ));
        }
        let numel = args.dx.numel();
        let dy_len = args.dy.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        let dx_len = args.dx.data.len() as i64;
        if dy_len < numel || y_len < numel || dx_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dy_len.min(y_len).min(dx_len) as usize,
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
        args: SparsemaxBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dx.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let axis = self.desc.softmax_axis as usize;
        let shape = self.desc.input_shape;
        let stride_dy = args.dy.stride;
        let stride_y = args.y.stride;
        let stride_dx = args.dx.stride;
        let rank = N as i32;
        let extent = shape[axis];
        let stride_dy_axis = stride_dy[axis];
        let stride_y_axis = stride_y[axis];

        macro_rules! dispatch {
            ($sym:ident) => {
                unsafe {
                    baracuda_kernels_sys::$sym(
                        numel,
                        rank,
                        shape.as_ptr(),
                        stride_dy.as_ptr(),
                        stride_y.as_ptr(),
                        stride_dx.as_ptr(),
                        axis as i32,
                        extent,
                        stride_dy_axis,
                        stride_y_axis,
                        dy_ptr,
                        y_ptr,
                        dx_ptr,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            };
        }
        let status = match T::KIND {
            ElementKind::F32 => dispatch!(baracuda_kernels_sparsemax_backward_f32_run),
            ElementKind::F16 => dispatch!(baracuda_kernels_sparsemax_backward_f16_run),
            ElementKind::Bf16 => dispatch!(baracuda_kernels_sparsemax_backward_bf16_run),
            ElementKind::F64 => dispatch!(baracuda_kernels_sparsemax_backward_f64_run),
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SparsemaxBackwardPlan::run unimplemented dtype",
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
