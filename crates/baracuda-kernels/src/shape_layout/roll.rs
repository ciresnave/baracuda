//! `roll` plan — cyclic shift along axes. Output shape == input shape.
//! Today only `f32` is wired (trailblazer).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `roll` op.
///
/// `shifts[d]` is the shift amount along axis `d`. Can be negative.
/// Normalized to `[0, shape[d])` by the kernel.
#[derive(Copy, Clone, Debug)]
pub struct RollDescriptor<const N: usize> {
    /// Input/output tensor shape.
    pub shape: [i32; N],
    /// Per-axis shift amounts (positive or negative).
    pub shifts: [i32; N],
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a Roll launch.
pub struct RollArgs<'a, T: Element, const N: usize> {
    /// Input.
    pub x: TensorRef<'a, T, N>,
    /// Output — same shape as input.
    pub y: TensorMut<'a, T, N>,
}

/// `roll` plan.
pub struct RollPlan<T: Element, const N: usize> {
    desc: RollDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> RollPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &RollDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RollPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RollPlan: shape dims must be non-negative",
                ));
            }
        }
        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::RollPlan: today only `f32`, `f16`, `bf16`, `f64` \
                 are wired; other dtypes land in future fanout",
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
    pub fn can_implement(&self, args: &RollArgs<'_, T, N>) -> Result<()> {
        if args.x.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RollPlan: X shape mismatch",
            ));
        }
        if args.y.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RollPlan: Y shape mismatch",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::RollPlan: tensor rank > 8 not supported",
            ));
        }
        let numel = args.y.numel();
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
        args: RollArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.y.numel();
        if numel == 0 {
            return Ok(());
        }
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let shape = self.desc.shape;
        let shifts = self.desc.shifts;
        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_roll_f32_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    shifts.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
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
                    shifts.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
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
                    shifts.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
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
                    shifts.as_ptr(),
                    stride_x.as_ptr(),
                    stride_y.as_ptr(),
                    x_ptr,
                    y_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RollPlan::run: only f32/f16/bf16/f64 wired today",
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
