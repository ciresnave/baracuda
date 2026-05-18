//! Softmax forward plan — single-axis numerically-stable softmax /
//! log-softmax.
//!
//! **Formulas**:
//! - `Softmax`:    `y[k] = exp(x[k] - max(x)) / Σ_j exp(x[j] - max(x))`
//! - `LogSoftmax`: `y[k] = x[k] - logsumexp(x)`
//!
//! Numerically stable via max subtraction.
//!
//! **When to use**: forward pass of softmax / log-softmax along a single
//! axis. Pair with [`SoftmaxBackwardPlan`](super::SoftmaxBackwardPlan)
//! for autograd (the BW kernel needs the saved forward output `y`).
//!
//! **Dtypes / shape**: `{Softmax, LogSoftmax} × {f32, f16, bf16, f64}`,
//! tensor rank `1..=8`. Half-precision (`f16` / `bf16`) reduces / exps in
//! `f32` (FP detour) then casts back; `f64` keeps everything in double.
//!
//! **Workspace**: none.
//!
//! **Precision**: deterministic, bit-stable on the same hardware. The
//! per-output-cell two-pass scan has no atomic-add / warp-reduction
//! ordering dependence.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, SoftmaxKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a softmax-family op.
#[derive(Copy, Clone, Debug)]
pub struct SoftmaxDescriptor<const N: usize> {
    /// Which softmax variant.
    pub kind: SoftmaxKind,
    /// Tensor shape — input and output share it.
    pub input_shape: [i32; N],
    /// Axis along which to compute softmax. Must be in `[0, N)`.
    pub softmax_axis: u8,
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a softmax launch.
pub struct SoftmaxArgs<'a, T: Element, const N: usize> {
    /// Input tensor.
    pub x: TensorRef<'a, T, N>,
    /// Output tensor — same shape as input.
    pub y: TensorMut<'a, T, N>,
}

/// Softmax forward plan — see the module-level docs for formulas,
/// dtypes, workspace, and precision guarantees.
///
/// `T: Element` is the element type (`f32` / `f64` / `f16` / `bf16`).
/// `const N: usize` is the tensor rank (1..=8).
pub struct SoftmaxPlan<T: Element, const N: usize> {
    desc: SoftmaxDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> SoftmaxPlan<T, N> {
    /// Pick a kernel for `desc`. Validates `softmax_axis < N`, the dtype
    /// is in the wired FP family, and tensor rank ≤ 8. Returns
    /// [`Error::Unsupported`] for cells outside the matrix and
    /// [`Error::InvalidProblem`] for malformed shapes / axes.
    pub fn select(
        _stream: &Stream,
        desc: &SoftmaxDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SoftmaxPlan: descriptor element != T",
            ));
        }
        if (desc.softmax_axis as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SoftmaxPlan: softmax_axis out of range for rank N",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::SoftmaxPlan: shape dims must be non-negative",
                ));
            }
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::SoftmaxPlan: tensor rank > 8 not supported",
            ));
        }
        let dtype_in_fp_family = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        let kind_supported = matches!(desc.kind, SoftmaxKind::Softmax | SoftmaxKind::LogSoftmax);
        if !kind_supported || !dtype_in_fp_family {
            return Err(Error::Unsupported(
                "baracuda-kernels::SoftmaxPlan: wired today: \
                 `{Softmax, LogSoftmax} × {f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Bit-stable across runs (deterministic per-cell two-pass scan).
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
    pub fn can_implement(&self, args: &SoftmaxArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SoftmaxPlan: x shape mismatch",
            ));
        }
        if args.y.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SoftmaxPlan: y shape mismatch",
            ));
        }
        let numel = args.x.numel();
        let x_len = args.x.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if x_len < numel || y_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: x_len.min(y_len) as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes. Always zero — the kernel does its
    /// two-pass scan in registers.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }
    /// Identity of the kernel this plan picked (for telemetry +
    /// autotuner cache keying).
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }
    /// Numerical guarantees for this plan's kernel — deterministic,
    /// bit-stable on the same hardware, f32 accumulator for the FP-detour
    /// half / bf16 inputs and f32 / f64 native for those dtypes.
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
        args: SoftmaxArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.x.numel();
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let axis = self.desc.softmax_axis as usize;
        let shape = self.desc.input_shape;
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;
        let extent = shape[axis];
        let stride_x_axis = stride_x[axis];
        let stride_y_axis = stride_y[axis];

        macro_rules! dispatch {
            ($sym:ident) => {
                unsafe {
                    baracuda_kernels_sys::$sym(
                        numel,
                        rank,
                        shape.as_ptr(),
                        stride_x.as_ptr(),
                        stride_y.as_ptr(),
                        axis as i32,
                        extent,
                        stride_x_axis,
                        stride_y_axis,
                        x_ptr,
                        y_ptr,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            };
        }

        let status = match (self.desc.kind, T::KIND) {
            (SoftmaxKind::Softmax, ElementKind::F32) => dispatch!(baracuda_kernels_softmax_f32_run),
            (SoftmaxKind::Softmax, ElementKind::F16) => dispatch!(baracuda_kernels_softmax_f16_run),
            (SoftmaxKind::Softmax, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_softmax_bf16_run)
            }
            (SoftmaxKind::Softmax, ElementKind::F64) => dispatch!(baracuda_kernels_softmax_f64_run),
            (SoftmaxKind::LogSoftmax, ElementKind::F32) => {
                dispatch!(baracuda_kernels_log_softmax_f32_run)
            }
            (SoftmaxKind::LogSoftmax, ElementKind::F16) => {
                dispatch!(baracuda_kernels_log_softmax_f16_run)
            }
            (SoftmaxKind::LogSoftmax, ElementKind::Bf16) => {
                dispatch!(baracuda_kernels_log_softmax_bf16_run)
            }
            (SoftmaxKind::LogSoftmax, ElementKind::F64) => {
                dispatch!(baracuda_kernels_log_softmax_f64_run)
            }
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::SoftmaxPlan::run reached an unimplemented \
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
