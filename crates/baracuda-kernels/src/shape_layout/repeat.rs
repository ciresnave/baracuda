//! `repeat` plan — per-axis tile (output > input). PyTorch
//! `torch.repeat(x, *repeats)`: `output.shape[d] = input.shape[d] *
//! repeats[d]`. The kernel walks output cells and computes input
//! coords as `output_coord[d] % input.shape[d]`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `repeat` op.
#[derive(Copy, Clone, Debug)]
pub struct RepeatDescriptor<const N: usize> {
    /// Input tensor shape.
    pub input_shape: [i32; N],
    /// Per-axis repeat factors. Must be `>= 1`.
    pub repeats: [i32; N],
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> RepeatDescriptor<N> {
    /// Compute the output shape: `input.shape[d] * repeats[d]` per axis.
    pub fn output_shape(&self) -> [i32; N] {
        let mut out = [0i32; N];
        for d in 0..N {
            out[d] = self.input_shape[d] * self.repeats[d];
        }
        out
    }
}

/// Args bundle for a Repeat launch.
pub struct RepeatArgs<'a, T: Element, const N: usize> {
    /// Input.
    pub x: TensorRef<'a, T, N>,
    /// Output — shape matches `desc.output_shape()`.
    pub y: TensorMut<'a, T, N>,
}

/// `repeat` plan.
///
/// Per-axis tile: `output.shape[d] = input.shape[d] * repeats[d]`
/// (PyTorch `torch.repeat`). Kernel walks output cells and computes
/// input coords as `output_coord[d] % input.shape[d]`.
///
/// **When to use**: forward repeat. Pair with
/// [`RepeatBackwardPlan`](crate::RepeatBackwardPlan).
///
/// **Dtypes**: `{f32, f64, f16, bf16}`. Pure load + store.
///
/// **Shape limits**: rank in `[1, 8]`; `repeats[d] ≥ 1`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact.
pub struct RepeatPlan<T: Element, const N: usize> {
    desc: RepeatDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> RepeatPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &RepeatDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RepeatPlan: descriptor element != type parameter T",
            ));
        }
        for d in 0..N {
            if desc.input_shape[d] < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RepeatPlan: input_shape dims must be non-negative",
                ));
            }
            if desc.repeats[d] < 1 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RepeatPlan: repeats[d] must be >= 1",
                ));
            }
        }
        if !matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        ) {
            return Err(Error::Unsupported(
                "baracuda-kernels::RepeatPlan: supported dtypes are \
                 `{f32, f16, bf16, f64}`",
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
            op: ShapeLayoutKind::Repeat as u16,
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
    pub fn can_implement(&self, args: &RepeatArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RepeatPlan: X shape mismatch",
            ));
        }
        let expected_out = self.desc.output_shape();
        if args.y.shape != expected_out {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RepeatPlan: Y shape mismatch with derived output \
                 (output[d] = input.shape[d] * repeats[d])",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::RepeatPlan: tensor rank > 8 not supported",
            ));
        }
        let x_numel = args.x.numel();
        let y_numel = args.y.numel();
        let x_len = args.x.data.len() as i64;
        let y_len = args.y.data.len() as i64;
        if x_len < x_numel {
            return Err(Error::BufferTooSmall {
                needed: x_numel as usize,
                got: x_len as usize,
            });
        }
        if y_len < y_numel {
            return Err(Error::BufferTooSmall {
                needed: y_numel as usize,
                got: y_len as usize,
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
        args: RepeatArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let output_numel = args.y.numel();
        if output_numel == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let input_shape = self.desc.input_shape;
        let output_shape = self.desc.output_shape();
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;

        // Every Repeat FFI symbol shares the same parameter shape.
        macro_rules! dispatch {
            ($sym:ident) => {{
                unsafe {
                    baracuda_kernels_sys::$sym(
                        output_numel,
                        rank,
                        input_shape.as_ptr(),
                        output_shape.as_ptr(),
                        stride_x.as_ptr(),
                        stride_y.as_ptr(),
                        x_ptr,
                        y_ptr,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            }};
        }

        let status = match T::KIND {
            ElementKind::F32 => dispatch!(baracuda_kernels_repeat_f32_run),
            ElementKind::F16 => dispatch!(baracuda_kernels_repeat_f16_run),
            ElementKind::Bf16 => dispatch!(baracuda_kernels_repeat_bf16_run),
            ElementKind::F64 => dispatch!(baracuda_kernels_repeat_f64_run),
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RepeatPlan::run: this dtype is not wired",
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
