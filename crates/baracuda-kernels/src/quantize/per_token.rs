//! `quantize_per_token` forward plan.
//!
//! Per-row quantization for 2-D activations: input `[N, D]`; one
//! `(scale, zero_point)` pair per token row. Used by W8A8 LLM
//! activation quantization at inference time (the caller computes
//! `scale[n]` from each row's max-abs dynamic range).
//!
//! FW: `q[n, d] = clamp(round(x[n, d] / scale[n]) + zero_point[n],
//!                     qmin, qmax)`.
//!
//! BW: see [`crate::quantize::QuantizePerTokenBackwardPlan`].

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IntElement, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut, TensorRef, Workspace,
};

use super::{map_status, validate_input_element, validate_output_element};

/// Descriptor for a `quantize_per_token` forward op.
#[derive(Copy, Clone, Debug)]
pub struct QuantizePerTokenDescriptor {
    /// Number of token rows (first axis of input/output).
    pub n: i32,
    /// Feature dim (second axis of input/output).
    pub d: i32,
    /// Quantization range lower bound (e.g. `-128` for s8 symmetric).
    pub q_min: i32,
    /// Quantization range upper bound (e.g. `127` for s8 symmetric).
    pub q_max: i32,
    /// Input FP element kind. Must match `TIn::KIND`.
    pub input_element: ElementKind,
    /// Output int element kind (s8 or u8). Must match `TOut::KIND`.
    pub output_element: ElementKind,
}

/// Args bundle for a `quantize_per_token` forward launch.
pub struct QuantizePerTokenArgs<'a, TIn: Element, TOut: IntElement> {
    /// Input `[N, D]` in FP.
    pub input: TensorRef<'a, TIn, 2>,
    /// Per-row scale `[N]` in FP.
    pub scale: TensorRef<'a, TIn, 1>,
    /// Per-row zero-point `[N]` in i32.
    pub zero_point: TensorRef<'a, i32, 1>,
    /// Output `[N, D]` in int.
    pub output: TensorMut<'a, TOut, 2>,
}

/// `quantize_per_token` forward plan.
///
/// `q[n, d] = clamp(round(x[n, d] / scale[n]) + zero_point[n], qmin, qmax)`.
/// Per-row quantization for 2-D activations (W8A8 LLM-style).
///
/// **When to use**: forward activation quantization at inference (one
/// `(scale, zp)` pair per token row, computed from the row's max-abs
/// range upstream). For weight quantization use
/// [`QuantizePerChannelPlan`](crate::QuantizePerChannelPlan); for
/// global scale use [`QuantizePerTensorPlan`](crate::QuantizePerTensorPlan).
/// Pair with [`QuantizePerTokenBackwardPlan`](crate::QuantizePerTokenBackwardPlan)
/// for STE.
///
/// **Dtypes**: input FP `{f32, f64, f16, bf16}` × output int
/// `{s8, u8}`. `scale[]` is input dtype; `zero_point[]` is `i32`.
///
/// **Shape limits**: rank-2 `[N, D]`; `scale` and `zero_point` are
/// `[N]`. `q_max ≥ q_min`.
///
/// **Workspace**: none.
///
/// **Precision guarantee**: deterministic, bit-stable. One thread
/// per output cell, no atomics. Round-ties-even.
pub struct QuantizePerTokenPlan<TIn: Element, TOut: IntElement> {
    desc: QuantizePerTokenDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> QuantizePerTokenPlan<TIn, TOut> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &QuantizePerTokenDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "QuantizePerTokenPlan: descriptor input_element != type parameter TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "QuantizePerTokenPlan: descriptor output_element != type parameter TOut",
            ));
        }
        validate_input_element(TIn::KIND, "QuantizePerTokenPlan: unsupported TIn dtype")?;
        validate_output_element(TOut::KIND, "QuantizePerTokenPlan: unsupported TOut dtype")?;
        if desc.n < 0 || desc.d < 0 {
            return Err(Error::InvalidProblem(
                "QuantizePerTokenPlan: n and d must be non-negative",
            ));
        }
        if desc.q_max < desc.q_min {
            return Err(Error::InvalidProblem(
                "QuantizePerTokenPlan: q_max < q_min",
            ));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::PerToken);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args at run time.
    pub fn can_implement(&self, args: &QuantizePerTokenArgs<'_, TIn, TOut>) -> Result<()> {
        if args.input.shape != [self.desc.n, self.desc.d] {
            return Err(Error::InvalidProblem(
                "QuantizePerTokenPlan: input shape != [n, d]",
            ));
        }
        if args.output.shape != [self.desc.n, self.desc.d] {
            return Err(Error::InvalidProblem(
                "QuantizePerTokenPlan: output shape != [n, d]",
            ));
        }
        if args.scale.shape != [self.desc.n] {
            return Err(Error::InvalidProblem(
                "QuantizePerTokenPlan: scale shape != [n]",
            ));
        }
        if args.zero_point.shape != [self.desc.n] {
            return Err(Error::InvalidProblem(
                "QuantizePerTokenPlan: zero_point shape != [n]",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — none.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// Identity of the selected kernel.
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
        args: QuantizePerTokenArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.n as i64) * (self.desc.d as i64);
        if total == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let sc_ptr = args.scale.data.as_raw().0 as *const c_void;
        let zp_ptr = args.zero_point.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match (TIn::KIND, TOut::KIND) {
            (ElementKind::F32, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_f32_s8_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F32, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_f32_u8_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_f64_s8_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_f64_u8_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_f16_s8_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_f16_u8_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_bf16_s8_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::Bf16, ElementKind::U8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantize_per_token_bf16_u8_run(
                    self.desc.n, self.desc.d, self.desc.q_min, self.desc.q_max,
                    in_ptr, sc_ptr, zp_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "QuantizePerTokenPlan::run reached unsupported (TIn, TOut) combination",
                ))
            }
        };
        map_status(status)
    }
}

/// Build the [`KernelSku`] for a quantize-per-token-family plan.
pub(crate) fn build_sku<TIn: Element, TOut: IntElement>(op: QuantizeKind) -> KernelSku {
    let precision_guarantee = PrecisionGuarantee {
        math_precision: if TIn::KIND == ElementKind::F64 {
            MathPrecision::F64
        } else {
            MathPrecision::F32
        },
        accumulator: ElementKind::F32,
        // Deterministic — one thread per output cell, no atomics.
        bit_stable_on_same_hardware: true,
        deterministic: true,
    };
    KernelSku {
        category: OpCategory::Quantization,
        op: op as u16,
        element: TIn::KIND,
        aux_element: Some(TOut::KIND),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee,
    }
}
