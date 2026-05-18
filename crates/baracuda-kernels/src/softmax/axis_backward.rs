//! Softmax backward plan.
//!
//! **Formulas**:
//! - `Softmax` BW:    `dx[k] = y[k] · (dy[k] - Σ_j y[j] · dy[j])`
//! - `LogSoftmax` BW: `dx[k] = dy[k] - exp(y[k]) · Σ_j dy[j]`
//!
//! Both reference the **saved forward output** `y` — the BW formula has
//! no dependence on the forward input `x` once `y` is known.
//!
//! **When to use**: autograd backward for [`SoftmaxPlan`](super::SoftmaxPlan).
//! Caller saves `y` from the FW pass and feeds it as `args.y`.
//!
//! **Dtypes / shape**: `{Softmax, LogSoftmax} × {f32, f16, bf16, f64}`,
//! tensor rank `1..=8`. f16 / bf16 reduce in f32 (FP detour) then cast
//! back.
//!
//! **Workspace**: none.
//!
//! **Precision**: deterministic, bit-stable on the same hardware
//! (two-pass per-row scan; no atomic-add).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, SoftmaxKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a softmax-family BW op.
#[derive(Copy, Clone, Debug)]
pub struct SoftmaxBackwardDescriptor<const N: usize> {
    /// Which softmax variant this is the BW of.
    pub kind: SoftmaxKind,
    /// Tensor shape (shared by dy / y / dx).
    pub input_shape: [i32; N],
    /// Forward softmax axis.
    pub softmax_axis: u8,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a softmax BW launch.
///
/// `y` is the SAVED forward output. Required by all softmax-family BW
/// kernels (the gradient formula references it).
pub struct SoftmaxBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward output.
    pub y: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input.
    pub dx: TensorMut<'a, T, N>,
}

/// Softmax backward plan — see the module-level docs for formulas,
/// dtypes, workspace, and precision guarantees.
///
/// `T: Element` is the element type (`f32` / `f64` / `f16` / `bf16`).
/// `const N: usize` is the tensor rank (1..=8).
pub struct SoftmaxBackwardPlan<T: Element, const N: usize> {
    desc: SoftmaxBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> SoftmaxBackwardPlan<T, N> {
    /// Pick a kernel for `desc`. Validates `softmax_axis < N`, the dtype
    /// is in the wired FP family, and tensor rank ≤ 8. Returns
    /// [`Error::Unsupported`] for cells outside the matrix and
    /// [`Error::InvalidProblem`] for malformed shapes / axes.
    pub fn select(
        _stream: &Stream,
        desc: &SoftmaxBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SoftmaxBackwardPlan: descriptor element != T",
            ));
        }
        if (desc.softmax_axis as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SoftmaxBackwardPlan: softmax_axis out of range for rank N",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::SoftmaxBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::SoftmaxBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        let kind_supported = matches!(desc.kind, SoftmaxKind::Softmax | SoftmaxKind::LogSoftmax);
        if !kind_supported || !dtype_in_fp_family {
            return Err(Error::Unsupported(
                "baracuda-kernels::SoftmaxBackwardPlan: wired today: \
                 `{Softmax, LogSoftmax} × {f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Softmax,
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
    pub fn can_implement(&self, args: &SoftmaxBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SoftmaxBackwardPlan: dy shape mismatch",
            ));
        }
        if args.y.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SoftmaxBackwardPlan: y shape mismatch",
            ));
        }
        if args.dx.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SoftmaxBackwardPlan: dx shape mismatch",
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

    /// Workspace size in bytes. Always zero.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }
    /// Identity of the kernel this plan picked.
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }
    /// Numerical guarantees for this plan's kernel — deterministic,
    /// bit-stable on the same hardware, f32 accumulator for f16 / bf16
    /// inputs (FP detour).
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Launch the kernel against `args`. Calls `can_implement` first;
    /// returns `Ok(())` for empty tensors.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: SoftmaxBackwardArgs<'_, T, N>,
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

        let status = match (self.desc.kind, T::KIND) {
            (SoftmaxKind::Softmax, ElementKind::F32) => {
                dispatch!(baracuda_kernels_softmax_backward_f32_run)
            }
            (SoftmaxKind::Softmax, ElementKind::F16) => {
                dispatch!(baracuda_kernels_softmax_backward_f16_run)
            }
            (SoftmaxKind::Softmax, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_softmax_backward_bf16_run)
            }
            (SoftmaxKind::Softmax, ElementKind::F64) => {
                dispatch!(baracuda_kernels_softmax_backward_f64_run)
            }
            (SoftmaxKind::LogSoftmax, ElementKind::F32) => {
                dispatch!(baracuda_kernels_log_softmax_backward_f32_run)
            }
            (SoftmaxKind::LogSoftmax, ElementKind::F16) => {
                dispatch!(baracuda_kernels_log_softmax_backward_f16_run)
            }
            (SoftmaxKind::LogSoftmax, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_log_softmax_backward_bf16_run)
            }
            (SoftmaxKind::LogSoftmax, ElementKind::F64) => {
                dispatch!(baracuda_kernels_log_softmax_backward_f64_run)
            }
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SoftmaxBackwardPlan::run reached an unimplemented \
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
