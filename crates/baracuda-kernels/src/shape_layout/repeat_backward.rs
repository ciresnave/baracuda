//! `repeat` backward plan — Category N (Phase 3 BW).
//!
//! Backward of `y = repeat(x, repeats)` (PyTorch `torch.repeat`):
//! `dx[c_in] = sum_{k} dy[c_in + k * input_shape]` per axis — i.e. every
//! `dy` cell whose `c_out[d] % input_shape[d] == c_in[d]` for all `d`
//! contributes to `dx[c_in]`. One thread per dx cell loops the per-axis
//! repeats grid (`prod(repeats[d])` cells) and accumulates. f16 / bf16
//! accumulate in f32 inside the kernel for numerical stability; f32 /
//! f64 accumulate in their native dtype.
//!
//! Not bit-stable across same-hardware reruns in principle (the grid
//! iteration order is fixed today, but summation order matters in FP
//! semantics, so we conservatively report
//! `bit_stable_on_same_hardware: false`).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `repeat` backward op.
///
/// Mirrors [`crate::RepeatDescriptor`] for the params the BW needs:
/// `input_shape` (= dx shape) and `repeats` (per-axis factor).
#[derive(Copy, Clone, Debug)]
pub struct RepeatBackwardDescriptor<const N: usize> {
    /// Input tensor shape (= dx shape).
    pub input_shape: [i32; N],
    /// Per-axis repeat factors. Must be `>= 1` (same as forward).
    pub repeats: [i32; N],
    /// Element type of dy and dx.
    pub element: ElementKind,
}

impl<const N: usize> RepeatBackwardDescriptor<N> {
    /// Compute the dy shape (= forward output shape):
    /// `input_shape[d] * repeats[d]` per axis.
    pub fn dy_shape(&self) -> [i32; N] {
        let mut out = [0i32; N];
        for d in 0..N {
            out[d] = self.input_shape[d] * self.repeats[d];
        }
        out
    }
}

/// Args bundle for a Repeat backward launch.
///
/// `dy.shape` must match the forward output shape (`input_shape[d] *
/// repeats[d]` per axis). `dx.shape` must match `desc.input_shape`. No
/// saved forward tensors are needed — the BW formula is a pure sum over
/// dy.
pub struct RepeatBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient — full forward output shape.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input — input shape.
    pub dx: TensorMut<'a, T, N>,
}

/// `repeat` backward plan.
pub struct RepeatBackwardPlan<T: Element, const N: usize> {
    desc: RepeatBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> RepeatBackwardPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &RepeatBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RepeatBackwardPlan: descriptor element != type parameter T",
            ));
        }
        for d in 0..N {
            if desc.input_shape[d] < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RepeatBackwardPlan: input_shape dims must be \
                     non-negative",
                ));
            }
            if desc.repeats[d] < 1 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RepeatBackwardPlan: repeats[d] must be >= 1",
                ));
            }
        }
        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::RepeatBackwardPlan: today only `f32`, `f16`, `bf16`, \
                 `f64` are wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Sum order matters in FP — not bit-stable in principle.
            bit_stable_on_same_hardware: false,
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
    pub fn can_implement(&self, args: &RepeatBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dx.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RepeatBackwardPlan: dx shape mismatch with descriptor \
                 input_shape",
            ));
        }
        let expected_dy = self.desc.dy_shape();
        if args.dy.shape != expected_dy {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RepeatBackwardPlan: dy shape mismatch with derived \
                 output shape (= input_shape[d] * repeats[d] per axis)",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::RepeatBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        let dx_numel = args.dx.numel();
        let dy_numel = args.dy.numel();
        if (args.dx.data.len() as i64) < dx_numel {
            return Err(Error::BufferTooSmall {
                needed: dx_numel as usize,
                got: args.dx.data.len(),
            });
        }
        if (args.dy.data.len() as i64) < dy_numel {
            return Err(Error::BufferTooSmall {
                needed: dy_numel as usize,
                got: args.dy.data.len(),
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

    /// Launch.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: RepeatBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let input_numel = args.dx.numel();
        if input_numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let input_shape = self.desc.input_shape;
        let repeats = self.desc.repeats;
        let stride_dy = args.dy.stride;
        let stride_dx = args.dx.stride;
        let rank = N as i32;

        // All four FFI symbols share the same parameter shape.
        macro_rules! dispatch {
            ($sym:ident) => {{
                unsafe {
                    baracuda_kernels_sys::$sym(
                        input_numel,
                        rank,
                        input_shape.as_ptr(),
                        repeats.as_ptr(),
                        stride_dy.as_ptr(),
                        stride_dx.as_ptr(),
                        dy_ptr,
                        dx_ptr,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            }};
        }

        let status = match T::KIND {
            ElementKind::F32 => dispatch!(baracuda_kernels_repeat_backward_f32_run),
            ElementKind::F16 => dispatch!(baracuda_kernels_repeat_backward_f16_run),
            ElementKind::Bf16 => dispatch!(baracuda_kernels_repeat_backward_bf16_run),
            ElementKind::F64 => dispatch!(baracuda_kernels_repeat_backward_f64_run),
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RepeatBackwardPlan::run: only f32/f16/bf16/f64 \
                     wired today",
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
