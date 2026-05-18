//! `flip` backward plan — Category N (Phase 3 BW).
//!
//! `flip` is involutive: `flip(flip(x, dims), dims) == x`. Therefore the
//! backward of `y = flip(x, dims)` is `dx = flip(dy, dims)` — the same
//! reversal, applied to the upstream gradient. No new CUDA kernel is
//! needed; this plan dispatches to the existing forward `flip_<dtype>`
//! launcher with `dy → x_in` and `dx → y_out`.
//!
//! Output shape == input shape (flip preserves shape), so `dy.shape ==
//! dx.shape`. Bit-exact across all wired dtypes — pure element copy, no
//! arithmetic.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `flip` backward op.
///
/// Mirrors [`crate::FlipDescriptor`] — same `shape` and `flip_axes` as
/// the forward, since flip is involutive.
#[derive(Copy, Clone, Debug)]
pub struct FlipBackwardDescriptor<const N: usize> {
    /// Tensor shape (flip preserves shape, so this is both `dy.shape`
    /// and `dx.shape`).
    pub shape: [i32; N],
    /// Per-axis mask: `true` = the forward reversed this axis (the BW
    /// reverses it again to recover the gradient at the original input
    /// coords).
    pub flip_axes: [bool; N],
    /// Element type.
    pub element: ElementKind,
}

/// Args bundle for a Flip backward launch.
///
/// No saved forward tensors are needed — the BW formula `dx = flip(dy,
/// dims)` doesn't reference `x` or `y`.
pub struct FlipBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient — same shape as the forward output (= input).
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input — same shape as `dy`.
    pub dx: TensorMut<'a, T, N>,
}

/// `flip` backward plan.
///
/// Adjoint of [`crate::FlipPlan`]: `dx = dy.flip(axes)` — `flip` is
/// an involution, so the BW reuses the FW kernel.
///
/// **When to use**: BW for [`FlipPlan`](crate::FlipPlan).
///
/// **Dtypes**: `{f32, f64, f16, bf16}`.
///
/// **Shape limits**: rank in `[1, 8]`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable, bit-exact.
pub struct FlipBackwardPlan<T: Element, const N: usize> {
    desc: FlipBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> FlipBackwardPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &FlipBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlipBackwardPlan: descriptor element != type parameter T",
            ));
        }
        for &d in desc.shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::FlipBackwardPlan: shape dims must be non-negative",
                ));
            }
        }
        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlipBackwardPlan: today only `f32`, `f16`, `bf16`, `f64` \
                 are wired; other dtypes land in future fanout",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Pure element copy via reversed index — no arithmetic.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::Flip as u16,
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
    pub fn can_implement(&self, args: &FlipBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dy.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlipBackwardPlan: dy shape mismatch with descriptor",
            ));
        }
        if args.dx.shape != self.desc.shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlipBackwardPlan: dx shape mismatch with descriptor",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlipBackwardPlan: tensor rank > 8 not supported",
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

    /// Launch — dispatches to the forward `flip_<dtype>_run` symbol with
    /// `dy` as the input and `dx` as the output. Flip is involutive.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: FlipBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dx.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let mut flip_axes_i32 = [0i32; 8];
        for d in 0..N {
            flip_axes_i32[d] = if self.desc.flip_axes[d] { 1 } else { 0 };
        }

        let shape = self.desc.shape;
        let stride_dy = args.dy.stride;
        let stride_dx = args.dx.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_flip_f32_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    flip_axes_i32.as_ptr(),
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
                baracuda_kernels_sys::baracuda_kernels_flip_f16_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    flip_axes_i32.as_ptr(),
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
                baracuda_kernels_sys::baracuda_kernels_flip_bf16_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    flip_axes_i32.as_ptr(),
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
                baracuda_kernels_sys::baracuda_kernels_flip_f64_run(
                    numel,
                    rank,
                    shape.as_ptr(),
                    flip_axes_i32.as_ptr(),
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
                    "baracuda-kernels::FlipBackwardPlan::run: only f32/f16/bf16/f64 wired today",
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
