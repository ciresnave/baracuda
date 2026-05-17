//! Gated-activation forward plan (Phase 3 Category C′).
//!
//! Plan shape: split the rank-`N` input `x` along `split_dim` into two
//! equal halves `(a, b)`; the output `y = a · gate(b)` has the same shape
//! as `x` except `output_shape[split_dim] = x.shape[split_dim] / 2`. The
//! `gate` function is selected by [`GatedActivationKind`]:
//!
//! | Variant   | `gate(b)`            | Notes                                        |
//! |-----------|----------------------|----------------------------------------------|
//! | `Glu`     | `sigmoid(b)`         | PyTorch `torch.nn.functional.glu`.           |
//! | `ReGlu`   | `max(b, 0)`          | Bit-exact at f32 / f64.                      |
//! | `SwiGlu`  | `b · sigmoid(b)`     | Llama / Mistral / Gemma — load-bearing.      |
//! | `GeGlu`   | `0.5·b·(1+erf(b/√2))`| Exact erf-based (not the tanh approximation).|
//!
//! Today the FW is wired for `{Glu, ReGlu, SwiGlu, GeGlu} × {f32, f16,
//! bf16, f64}`. Contig only — strided fanout lands later; the kernel
//! plumbing already accepts per-operand strides so the strided path will
//! reuse the same ABI.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    contiguous_stride, ArchSku, BackendKind, Element, ElementKind, GatedActivationKind, KernelSku,
    MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

/// Descriptor for a gated-activation forward op.
///
/// `input_shape` is the **input** tensor shape — the output shape is
/// derived by halving `input_shape[split_dim]` (see
/// [`Self::output_shape`]). `input_shape[split_dim]` must be even.
#[derive(Copy, Clone, Debug)]
pub struct GatedActivationDescriptor<const N: usize> {
    /// Which gated activation to apply.
    pub kind: GatedActivationKind,
    /// Full input tensor shape. `input_shape[split_dim]` must be even.
    pub input_shape: [i32; N],
    /// Axis along which the input splits into `(a, b)`.
    pub split_dim: u8,
    /// Element type — must match the type parameter `T`.
    pub element: ElementKind,
}

impl<const N: usize> GatedActivationDescriptor<N> {
    /// Compute the output tensor shape: `input_shape` with
    /// `shape[split_dim]` halved.
    #[inline]
    pub fn output_shape(&self) -> [i32; N] {
        let mut out = self.input_shape;
        out[self.split_dim as usize] /= 2;
        out
    }
}

/// Args bundle for a gated-activation forward launch.
pub struct GatedActivationArgs<'a, T: Element, const N: usize> {
    /// Input tensor — shape `desc.input_shape`.
    pub x: TensorRef<'a, T, N>,
    /// Output tensor — shape `desc.output_shape()`.
    pub y: TensorMut<'a, T, N>,
}

/// Gated-activation forward plan.
pub struct GatedActivationPlan<T: Element, const N: usize> {
    desc: GatedActivationDescriptor<N>,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element, const N: usize> GatedActivationPlan<T, N> {
    /// Pick a kernel. Returns [`Error::Unsupported`] for unwired
    /// `(kind, T::KIND)` pairs.
    pub fn select(
        _stream: &Stream,
        desc: &GatedActivationDescriptor<N>,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatedActivationPlan: descriptor element != type parameter T",
            ));
        }
        if N == 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatedActivationPlan: rank 0 has no splittable axis",
            ));
        }
        if (desc.split_dim as usize) >= N {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatedActivationPlan: split_dim out of range",
            ));
        }
        for &d in desc.input_shape.iter() {
            if d < 0 {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::GatedActivationPlan: input_shape dims must be non-negative",
                ));
            }
        }
        let sd = desc.split_dim as usize;
        if desc.input_shape[sd] % 2 != 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatedActivationPlan: input_shape[split_dim] must be even",
            ));
        }
        if N > 8 {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatedActivationPlan: tensor rank > 8 not supported \
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
                "baracuda-kernels::GatedActivationPlan: wired today: \
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
    pub fn can_implement(&self, args: &GatedActivationArgs<'_, T, N>) -> Result<()> {
        let output_shape = self.desc.output_shape();
        if args.x.shape != self.desc.input_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatedActivationPlan: x shape mismatch with descriptor input_shape",
            ));
        }
        if args.y.shape != output_shape {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::GatedActivationPlan: y shape mismatch with desc.output_shape()",
            ));
        }
        // Contig-only today — defer strided fanout.
        if !args.x.is_contiguous() || !args.y.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::GatedActivationPlan: trailblazer requires contiguous \
                 x / y; strided fanout lands later",
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
        args: GatedActivationArgs<'_, T, N>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let output_shape = self.desc.output_shape();
        let output_numel = args.y.numel();
        if output_numel == 0 {
            return Ok(());
        }

        // `x_half_offset` = (input_shape[split_dim]/2) * stride_x[split_dim]
        // — element offset between an a-half cell and its b-half twin
        // for the same output coord. With contig input strides this
        // equals `half * product(input_shape[split_dim+1..N])`, i.e. the
        // numel of the inner block at the b-side coord. We compute it
        // from args.x.stride directly so a future strided path keeps the
        // same plumbing.
        let sd = self.desc.split_dim as usize;
        let half = self.desc.input_shape[sd] as i64 / 2;
        let x_half_offset = half * args.x.stride[sd];

        let stream_ptr = stream.as_raw() as *mut c_void;
        let x_ptr = args.x.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;

        let stride_x = args.x.stride;
        let stride_y = args.y.stride;
        // Pass shape as `i32` array via slice — kernel reads first `rank` entries.
        let rank = N as i32;
        let split_dim = self.desc.split_dim as i32;

        let status = match (self.desc.kind, T::KIND) {
            (GatedActivationKind::SwiGlu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_swiglu_f32_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::SwiGlu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_swiglu_f16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::SwiGlu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_swiglu_bf16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::SwiGlu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_swiglu_f64_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::Glu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_glu_f32_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::Glu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_glu_f16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::Glu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_glu_bf16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::Glu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_glu_f64_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::ReGlu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_reglu_f32_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::ReGlu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_reglu_f16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::ReGlu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_reglu_bf16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::ReGlu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_reglu_f64_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::GeGlu, ElementKind::F32) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_geglu_f32_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::GeGlu, ElementKind::F16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_geglu_f16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::GeGlu, ElementKind::Bf16) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_geglu_bf16_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (GatedActivationKind::GeGlu, ElementKind::F64) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_gated_geglu_f64_run(
                    output_numel, rank, output_shape.as_ptr(), split_dim, x_half_offset,
                    stride_x.as_ptr(), stride_y.as_ptr(),
                    x_ptr, y_ptr, core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::GatedActivationPlan::run reached an unimplemented \
                     (kind, dtype) pair — select() should have caught this",
                ))
            }
        };
        map_status(status)
    }
}

// Silence the unused-import warning when callers don't exercise this
// helper directly; it's retained for parity with sibling Plans.
#[allow(dead_code)]
fn _strides_used<const N: usize>(s: [i32; N]) -> [i64; N] {
    contiguous_stride(s)
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
