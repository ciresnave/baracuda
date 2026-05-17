//! `permute` backward plan — Category N (Phase 3 BW).
//!
//! The backward of `y = x.permute(dims)` is `dx = dy.permute(inv_dims)`
//! where `inv_dims[dims[d]] = d` — the inverse permutation. No new
//! CUDA kernel is needed; this plan dispatches to the existing forward
//! `permute_<dtype>` launcher with `dy → x_in`, `dx → y_out`, and
//! `dims → inv_dims`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, ShapeLayoutKind, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a `permute` backward op.
///
/// Mirrors [`crate::PermuteDescriptor`] — same `input_shape` (= dx
/// shape) and `dims` (the forward permutation) as the forward. The BW
/// derives the inverse permutation internally.
#[derive(Copy, Clone, Debug)]
pub struct PermuteBackwardDescriptor<const N: usize> {
    /// Forward input shape (= dx shape).
    pub input_shape: [i32; N],
    /// Forward permutation: `dims[d]` is the input axis that became
    /// forward output axis `d`. Must be a permutation of `[0, N)`.
    pub dims: [i32; N],
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> PermuteBackwardDescriptor<N> {
    /// Compute the dy shape (= forward output shape).
    pub fn dy_shape(&self) -> [i32; N] {
        let mut out = [0i32; N];
        for d in 0..N {
            out[d] = self.input_shape[self.dims[d] as usize];
        }
        out
    }

    /// Compute the inverse permutation: `inv[dims[d]] = d`.
    pub fn inverse_dims(&self) -> [i32; N] {
        let mut inv = [0i32; N];
        for d in 0..N {
            inv[self.dims[d] as usize] = d as i32;
        }
        inv
    }
}

/// Args bundle for a Permute backward launch.
pub struct PermuteBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient — forward output shape (= `dy_shape()`).
    pub dy: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the forward input — `input_shape`.
    pub dx: TensorMut<'a, T, N>,
}

/// `permute` backward plan.
pub struct PermuteBackwardPlan<T: Element, const N: usize> {
    desc: PermuteBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> PermuteBackwardPlan<T, N> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &PermuteBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::PermuteBackwardPlan: descriptor element != type parameter T",
            ));
        }
        let mut seen = [false; 8];
        for d in 0..N {
            let v = desc.dims[d];
            if v < 0 || (v as usize) >= N {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::PermuteBackwardPlan: dims values must be in [0, N)",
                ));
            }
            if seen[v as usize] {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::PermuteBackwardPlan: dims must be a permutation \
                     (no duplicates)",
                ));
            }
            seen[v as usize] = true;
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::PermuteBackwardPlan: input_shape dims must be non-negative",
                ));
            }
        }
        let supported = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !supported {
            return Err(Error::Unsupported(
                "baracuda-kernels::PermuteBackwardPlan: today only `f32`, `f16`, `bf16`, `f64` \
                 are wired",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Pure axis-reorder copy — no arithmetic.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::ShapeLayout,
            op: ShapeLayoutKind::Permute as u16,
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
    pub fn can_implement(&self, args: &PermuteBackwardArgs<'_, T, N>) -> Result<()> {
        if args.dx.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PermuteBackwardPlan: dx shape != desc.input_shape",
            ));
        }
        let expected_dy = self.desc.dy_shape();
        if args.dy.shape != expected_dy {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::PermuteBackwardPlan: dy shape != input_shape.permute(dims)",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::PermuteBackwardPlan: tensor rank > 8 not supported",
            ));
        }
        let numel = args.dx.numel();
        if (args.dx.data.len() as i64) < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: args.dx.data.len(),
            });
        }
        if (args.dy.data.len() as i64) < numel {
            return Err(Error::BufferTooSmall {
                needed: numel as usize,
                got: args.dy.data.len(),
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

    /// Launch — dispatches to the forward `permute_<dtype>_run` with
    /// the inverse permutation. The kernel iterates `input_numel` of
    /// its "x" argument, which here is `dy` (forward output shape).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: PermuteBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let numel = args.dy.numel();
        if numel == 0 {
            return Ok(());
        }
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        // Reuse forward permute: walks "x" = dy and writes "y" = dx,
        // applying `inv_dims` so the source coord on each output axis
        // is the dy coord that ended up at the matching input axis.
        let dy_shape = self.desc.dy_shape();
        let inv_dims = self.desc.inverse_dims();
        let stride_dy = args.dy.stride;
        let stride_dx = args.dx.stride;
        let rank = N as i32;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_permute_f32_run(
                    numel,
                    rank,
                    dy_shape.as_ptr(),
                    inv_dims.as_ptr(),
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
                baracuda_kernels_sys::baracuda_kernels_permute_f16_run(
                    numel,
                    rank,
                    dy_shape.as_ptr(),
                    inv_dims.as_ptr(),
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
                baracuda_kernels_sys::baracuda_kernels_permute_bf16_run(
                    numel,
                    rank,
                    dy_shape.as_ptr(),
                    inv_dims.as_ptr(),
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
                baracuda_kernels_sys::baracuda_kernels_permute_f64_run(
                    numel,
                    rank,
                    dy_shape.as_ptr(),
                    inv_dims.as_ptr(),
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
                    "baracuda-kernels::PermuteBackwardPlan::run: only f32/f16/bf16/f64 wired today",
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
