//! Gated-activation backward plan (Phase 3 Category C′).
//!
//! Sibling of [`crate::GatedActivationPlan`] for gradient computation:
//! `dx = backward(dy, x)` where `x` is the **saved forward input** (the
//! full pre-split tensor, shape `desc.input_shape`) and `dy` has the
//! half-size output shape.
//!
//! Gradient formulas (with `(a, b)` = the two halves of `x` along
//! `split_dim`, `y = a · gate(b)`):
//!
//! | Variant   | `da`                      | `db`                                       |
//! |-----------|---------------------------|--------------------------------------------|
//! | `Glu`     | `dy · σ(b)`               | `dy · a · σ(b)·(1-σ(b))`                   |
//! | `ReGlu`   | `(b>0) ? dy·b : 0`        | `(b>0) ? dy·a : 0`                         |
//! | `SwiGlu`  | `dy · b·σ(b)`             | `dy · a · σ(b)·(1 + b·(1-σ(b)))`           |
//! | `GeGlu`   | `dy · b·Φ(b)`             | `dy · a · (Φ(b) + b·φ(b))`                 |
//!
//! Today the BW is wired for `{Glu, ReGlu, SwiGlu, GeGlu} × {f32, f16,
//! bf16, f64}`. Contig only.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, GatedActivationKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a gated-activation backward op.
#[derive(Copy, Clone, Debug)]
pub struct GatedActivationBackwardDescriptor<const N: usize> {
    /// Which forward gated activation this is the backward of.
    pub kind: GatedActivationKind,
    /// Full input tensor shape (shared by `x` and `dx`).
    /// `input_shape[split_dim]` must be even.
    pub input_shape: [i32; N],
    /// Axis along which the input splits into `(a, b)`.
    pub split_dim: u8,
    /// Element type.
    pub element: ElementKind,
}

impl<const N: usize> GatedActivationBackwardDescriptor<N> {
    /// Compute the half-size output shape (shared by `dy`).
    #[inline]
    pub fn output_shape(&self) -> [i32; N] {
        let mut out = self.input_shape;
        out[self.split_dim as usize] /= 2;
        out
    }
}

/// Args bundle for a gated-activation backward launch.
pub struct GatedActivationBackwardArgs<'a, T: Element, const N: usize> {
    /// Upstream gradient — shape `desc.output_shape()`.
    pub dy: TensorRef<'a, T, N>,
    /// Saved forward input — shape `desc.input_shape` (full).
    pub x: TensorRef<'a, T, N>,
    /// Gradient w.r.t. the input — shape `desc.input_shape` (full).
    pub dx: TensorMut<'a, T, N>,
}

/// Gated-activation backward plan.
pub struct GatedActivationBackwardPlan<T: Element, const N: usize> {
    desc: GatedActivationBackwardDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> GatedActivationBackwardPlan<T, N> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &GatedActivationBackwardDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatedActivationBackwardPlan: descriptor element != T",
            ));
        }
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatedActivationBackwardPlan: rank 0 has no splittable axis",
            ));
        }
        if (desc.split_dim as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatedActivationBackwardPlan: split_dim out of range",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::GatedActivationBackwardPlan: input_shape dims must be \
                     non-negative",
                ));
            }
        }
        let sd = desc.split_dim as usize;
        if desc.input_shape[sd] % 2 != 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatedActivationBackwardPlan: input_shape[split_dim] must be even",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatedActivationBackwardPlan: tensor rank > 8 not supported \
                 (kernel param block fixes MAX_RANK = 8)",
            ));
        }

        let kind_in_scope = matches!(
            desc.kind,
            GatedActivationKind::Glu
                | GatedActivationKind::ReGlu
                | GatedActivationKind::SwiGlu
                | GatedActivationKind::GeGlu
        );
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !(kind_in_scope && dtype_in_scope) {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatedActivationBackwardPlan: wired today: \
                 `{Glu, ReGlu, SwiGlu, GeGlu}` × `{f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::GatedActivation,
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
    pub fn can_implement(&self, args: &GatedActivationBackwardArgs<'_, T, N>) -> Result<()> {
        let output_shape = self.desc.output_shape();
        if args.dy.shape != output_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatedActivationBackwardPlan: dy shape mismatch with \
                 desc.output_shape()",
            ));
        }
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatedActivationBackwardPlan: x shape mismatch with \
                 desc.input_shape",
            ));
        }
        if args.dx.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatedActivationBackwardPlan: dx shape mismatch with \
                 desc.input_shape",
            ));
        }
        if !args.dy.is_contiguous() || !args.x.is_contiguous() || !args.dx.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatedActivationBackwardPlan: trailblazer requires contiguous \
                 dy / x / dx; strided fanout lands later",
            ));
        }
        let dy_len = args.dy.data.len() as i64;
        let x_len = args.x.data.len() as i64;
        let dx_len = args.dx.data.len() as i64;
        let output_numel = args.dy.numel();
        let input_numel = args.x.numel();
        if dy_len < output_numel {
            return Err(Error::BufferTooSmall {
                needed: output_numel as usize,
                got: dy_len as usize,
            });
        }
        if x_len < input_numel || dx_len < input_numel {
            return Err(Error::BufferTooSmall {
                needed: input_numel as usize,
                got: x_len.min(dx_len) as usize,
            });
        }
        Ok(())
    }

    /// Workspace size in bytes.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// SKU identity.
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
        args: GatedActivationBackwardArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let output_shape = self.desc.output_shape();
        let output_numel = args.dy.numel();
        if output_numel == 0 {
            return Ok(());
        }

        // x_half_offset / dx_half_offset both reduce to
        // `half * stride[split_dim]` for the respective tensor — the
        // element offset between an a-half cell and its b-half twin.
        let sd = self.desc.split_dim as usize;
        let half = self.desc.input_shape[sd] as i64 / 2;
        let x_half_offset = half * args.x.stride[sd];
        let dx_half_offset = half * args.dx.stride[sd];

        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let dy_ptr = args.dy.data.as_raw().0 as *const c_void;
        let dx_ptr = args.dx.data.as_raw().0 as *mut c_void;

        let stride_x = args.x.stride;
        let stride_dy = args.dy.stride;
        let stride_dx = args.dx.stride;
        let rank = N as i32;
        let split_dim = self.desc.split_dim as i32;

        let status = match (self.desc.kind, T::KIND) {
            (GatedActivationKind::SwiGlu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_swiglu_backward_f32_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::SwiGlu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_swiglu_backward_f16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::SwiGlu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_swiglu_backward_bf16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::SwiGlu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_swiglu_backward_f64_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::Glu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_glu_backward_f32_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::Glu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_glu_backward_f16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::Glu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_glu_backward_bf16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::Glu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_glu_backward_f64_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::ReGlu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_reglu_backward_f32_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::ReGlu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_reglu_backward_f16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::ReGlu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_reglu_backward_bf16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::ReGlu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_reglu_backward_f64_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::GeGlu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_geglu_backward_f32_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::GeGlu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_geglu_backward_f16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::GeGlu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_geglu_backward_bf16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::GeGlu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_geglu_backward_f64_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim,
                    x_half_offset, dx_half_offset,
                    stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                    x_ptr, dy_ptr, dx_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GatedActivationBackwardPlan::run reached an unimplemented \
                     (kind, dtype) pair — select() should have caught this",
                ))
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
