//! `roll` backward plan — Category N (Phase 3 BW).
//!
//! `roll` is invertible: the backward of `y = roll(x, shifts)` is
//! `dx = roll(dy, -shifts)` — same op with negated shifts. No new
//! CUDA kernel is needed; this plan dispatches to the existing forward
//! `roll_<dtype>` launcher with `dy → x_in`, `dx → y_out`, and
//! `shifts → -shifts`. Bit-exact across all wired dtypes.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `roll` backward op.
///
/// Mirrors [`crate::RollDescriptor`] — same `shape` and `shifts` as
/// the forward. The BW negates `shifts` internally before dispatch.
#[derive(Copy, Clone, Debug)]
pub struct RollBackwardDescriptor<const N: usize> {
    /// Tensor shape (roll preserves shape).
    pub shape: [i32; N],
    /// Forward per-axis shift amounts. The BW applies `-shifts`.
    pub shifts: [i32; N],
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a Roll backward launch.
pub struct RollBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient.
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input.
    pub dx: TensorMut<'a, T, N>,
}

/// `roll` backward plan.
pub struct RollBackwardPlan<T: Element, const N: usize> {
    desc: RollBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> RollBackwardPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &RollBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RollBackwardPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RollBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::RollBackwardPlan: today only `f32`, `f16`, `bf16`, `f64` \
                 are wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Pure cyclic-index copy — no arithmetic.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::Roll as u16,
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
    pub fn can_implement(&self, args: &RollBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RollBackwardPlan: dy shape mismatch",
            ));
        }
        if args.dx.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RollBackwardPlan: dx shape mismatch",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::RollBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        let numel = args.dx.numel();
        let dy_len = args.dy.data.len() as i64;
        let dx_len = args.dx.data.len() as i64;
        if dy_len < numel || dx_len < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: dy_len.min(dx_len) as usize,
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

    /// Launch — dispatches to the forward `roll_<dtype>_run` with
    /// negated shifts.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: RollBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dx.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let shape = self.desc.shape;
        let mut neg_shifts = [0i32; 8];
        for d in 0..N {
            // `i32::wrapping_neg` keeps `i32::MIN` safe; the kernel
            // normalizes via `((c - s) % extent + extent) % extent`.
            neg_shifts[d] = self.desc.shifts[d].wrapping_neg();
        }
        let stride_dy = args.dy.stride;
        let stride_dx = args.dx.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roll_f32_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    neg_shifts.as_ptr(),
                    stride_dy.as_ptr(),
                    stride_dx.as_ptr(),
                    dy_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roll_f16_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    neg_shifts.as_ptr(),
                    stride_dy.as_ptr(),
                    stride_dx.as_ptr(),
                    dy_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roll_bf16_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    neg_shifts.as_ptr(),
                    stride_dy.as_ptr(),
                    stride_dx.as_ptr(),
                    dy_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roll_f64_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    neg_shifts.as_ptr(),
                    stride_dy.as_ptr(),
                    stride_dx.as_ptr(),
                    dy_ptr,
                    dx_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RollBackwardPlan::run: only f32/f16/bf16/f64 wired today",
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
