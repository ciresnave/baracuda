//! `dynamic_range_quantize` — compose op (Phase 8 Milestone 8.3).
//!
//! Computes a `(scale, zero_point)` pair from the runtime dynamic range
//! of the input — the canonical post-training-quantization-at-inference
//! recipe — then quantizes. The Plan returns the quantized tensor AND
//! the computed `scale` vector so the caller can later dequantize.
//!
//! ## Granularity
//!
//! [`DynamicRangeMode::Symmetric`] vs [`DynamicRangeMode::Asymmetric`]:
//! - **Symmetric**: `max_abs = max |x|` over the segment; `scale =
//!   max_abs / qmax`; `zero_point = 0`. Standard for activations.
//! - **Asymmetric** (offset): `xmin / xmax` reduced separately; `scale
//!   = (xmax - xmin) / (qmax - qmin)`; `zp = qmin - round(xmin / scale)`.
//!
//! ## Scope
//!
//! [`DynamicRangeScope::Token`] is the Phase 8.3 trailblazer
//! — `[N, D]` input with one `(scale, zp)` pair per row, matching the
//! LLM W8A8 activation-quantize recipe.
//!
//! [`DynamicRangeScope::Tensor`], [`DynamicRangeScope::Channel`] and
//! [`DynamicRangeScope::Group`] are reserved scopes that return
//! [`Error::Unsupported`] from [`DynamicRangeQuantizePlan::select`]
//! today; they wire up in follow-up milestones by orchestrating the
//! matching primitive [`ReducePlan`](crate::ReducePlan) + per-channel /
//! per-group quantize plans (the per-tensor / per-channel / per-group
//! quantize plans from 8.1 / 8.2 already exist).
//!
//! ## Dtype coverage (trailblazer)
//!
//! `TIn ∈ {f32, f64}`, `TOut = S8`. `f16` / `bf16` activation, `u8`
//! output, and asymmetric mode are deferred.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IntElement, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut, TensorRef, Workspace,
};

use super::{map_status, validate_input_element, validate_output_element};

/// Symmetric vs asymmetric (offset) dynamic-range quantization.
///
/// Only [`DynamicRangeMode::Symmetric`] is wired in the 8.3
/// trailblazer; [`DynamicRangeMode::Asymmetric`] returns
/// [`Error::Unsupported`] and is reserved for a follow-up that wires
/// the dual max + min reduction + offset compute kernel.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DynamicRangeMode {
    /// `scale = max_abs / qmax`, `zero_point = 0`.
    Symmetric,
    /// `scale = (xmax - xmin) / (qmax - qmin)`,
    /// `zp = qmin - round(xmin / scale)`. Reserved.
    Asymmetric,
}

/// Per-tensor / per-channel / per-token / per-group granularity for the
/// scale + zero_point computation.
///
/// Only [`DynamicRangeScope::Token`] is wired in the 8.3 trailblazer.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DynamicRangeScope {
    /// One `(scale, zp)` for the whole tensor. Reserved.
    Tensor,
    /// One `(scale, zp)` per slice along `axis`. Reserved.
    Channel {
        /// Axis along which to slice.
        axis: i32,
    },
    /// One `(scale, zp)` per token row (first axis of a `[N, D]`
    /// tensor). Trailblazer scope.
    Token,
    /// One `(scale, zp)` per group along `axis`. Reserved.
    Group {
        /// Axis along which to group.
        axis: i32,
        /// Number of elements per group.
        group_size: i32,
    },
}

/// Descriptor for a `dynamic_range_quantize` op.
#[derive(Copy, Clone, Debug)]
pub struct DynamicRangeQuantizeDescriptor {
    /// Number of token rows (first axis of input / output).
    pub n: i32,
    /// Feature dim (second axis).
    pub d: i32,
    /// Quantization range lower bound (e.g. `-127` for symmetric s8).
    pub q_min: i32,
    /// Quantization range upper bound (e.g. `127` for symmetric s8).
    pub q_max: i32,
    /// Symmetric / asymmetric mode.
    pub mode: DynamicRangeMode,
    /// Per-tensor / per-channel / per-token / per-group scope.
    pub scope: DynamicRangeScope,
    /// Input FP element kind. Must match `TIn::KIND`.
    pub input_element: ElementKind,
    /// Output int element kind (must match `TOut::KIND`). `S8` only.
    pub output_element: ElementKind,
}

/// Args bundle for a `dynamic_range_quantize` launch.
///
/// Compared with [`super::QuantizePerTokenArgs`], the caller does NOT
/// supply `scale` / `zero_point` — those are computed by the kernel
/// from the runtime dynamic range. The plan writes `scale[N]` into the
/// caller-supplied `scale_out` buffer so a downstream dequantize step
/// has access to the same scale.
///
/// `zero_point` is implicit (= 0 for symmetric) and is not materialized.
pub struct DynamicRangeQuantizeArgs<'a, TIn: Element, TOut: IntElement> {
    /// Input `[N, D]` in FP.
    pub input: TensorRef<'a, TIn, 2>,
    /// Per-row scale `[N]` in FP — written by the kernel.
    pub scale_out: TensorMut<'a, TIn, 1>,
    /// Output `[N, D]` in int.
    pub output: TensorMut<'a, TOut, 2>,
}

/// `dynamic_range_quantize` plan.
///
/// Composes per-row max-abs reduction + scale computation + per-row
/// quantize into a single fused kernel launch (the dynamic-range
/// recipe IS the fused composition — running the reduce and quantize
/// as one kernel avoids a scale-buffer round-trip).
///
/// The trailblazer kernel is symmetric per-token only. Future fanout
/// adds asymmetric mode (requires xmin + xmax reductions) and the
/// other three scopes (tensor / channel / group) by orchestrating
/// existing primitives.
///
/// **When to use**: post-training activation quantization at
/// inference — compute scale from runtime range and quantize in
/// one launch. No BW (inference-only).
///
/// **Dtypes (trailblazer)**: `TIn ∈ {f32, f64}`, `TOut = S8`.
/// `f16` / `bf16` activation, `u8` output, and asymmetric mode
/// gated as `Unsupported` until follow-up milestones wire the
/// xmin/xmax reductions and offset-compute kernel.
///
/// **Shape limits**: rank-2 `[N, D]`; `N ≤ 65535` (block-per-row
/// grid cap, lifts when row tiling lands); `q_max > 0` (symmetric
/// divisor); `q_max ≥ q_min`.
///
/// **Workspace**: none — single-launch fused kernel.
///
/// **Precision guarantee**: deterministic, bit-stable. One block
/// per row, no atomics; block-tree reduction is associative-stable
/// on a single GPU.
pub struct DynamicRangeQuantizePlan<TIn: Element, TOut: IntElement> {
    desc: DynamicRangeQuantizeDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TOut)>,
}

impl<TIn: Element, TOut: IntElement> DynamicRangeQuantizePlan<TIn, TOut> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &DynamicRangeQuantizeDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.input_element != TIn::KIND {
            return Err(Error::Unsupported(
                "DynamicRangeQuantizePlan: descriptor input_element != TIn",
            ));
        }
        if desc.output_element != TOut::KIND {
            return Err(Error::Unsupported(
                "DynamicRangeQuantizePlan: descriptor output_element != TOut",
            ));
        }
        validate_input_element(TIn::KIND, "DynamicRangeQuantizePlan: unsupported TIn")?;
        validate_output_element(TOut::KIND, "DynamicRangeQuantizePlan: unsupported TOut")?;
        // Trailblazer: TIn ∈ {f32, f64}, TOut = S8. f16/bf16/u8 deferred.
        if !matches!(TIn::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "DynamicRangeQuantizePlan: 8.3 trailblazer only wires f32 / f64 \
                 activation (f16 / bf16 deferred)",
            ));
        }
        if TOut::KIND != ElementKind::S8 {
            return Err(Error::Unsupported(
                "DynamicRangeQuantizePlan: 8.3 trailblazer only wires s8 output \
                 (u8 deferred)",
            ));
        }
        if desc.mode != DynamicRangeMode::Symmetric {
            return Err(Error::Unsupported(
                "DynamicRangeQuantizePlan: 8.3 trailblazer only wires symmetric mode \
                 (asymmetric deferred — requires xmin + xmax reductions and a \
                 separate offset-compute kernel)",
            ));
        }
        if desc.scope != DynamicRangeScope::Token {
            return Err(Error::Unsupported(
                "DynamicRangeQuantizePlan: 8.3 trailblazer only wires per-token scope \
                 (tensor / channel / group deferred)",
            ));
        }
        if desc.n < 0 || desc.d < 0 {
            return Err(Error::InvalidProblem(
                "DynamicRangeQuantizePlan: n and d must be non-negative",
            ));
        }
        if desc.q_max <= 0 {
            return Err(Error::InvalidProblem(
                "DynamicRangeQuantizePlan: q_max must be > 0 (symmetric divisor)",
            ));
        }
        if desc.q_max < desc.q_min {
            return Err(Error::InvalidProblem(
                "DynamicRangeQuantizePlan: q_max < q_min",
            ));
        }
        // Grid limit on the kernel: one block per row, capped at 65535.
        if desc.n > 65535 {
            return Err(Error::Unsupported(
                "DynamicRangeQuantizePlan: N > 65535 — block-per-row grid limit \
                 (will be lifted when row tiling lands)",
            ));
        }
        let sku = build_sku::<TIn, TOut>(QuantizeKind::DynamicRange);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args at run time.
    pub fn can_implement(&self, args: &DynamicRangeQuantizeArgs<'_, TIn, TOut>) -> Result<()> {
        if args.input.shape != [self.desc.n, self.desc.d] {
            return Err(Error::InvalidProblem(
                "DynamicRangeQuantizePlan: input shape != [n, d]",
            ));
        }
        if args.output.shape != [self.desc.n, self.desc.d] {
            return Err(Error::InvalidProblem(
                "DynamicRangeQuantizePlan: output shape != [n, d]",
            ));
        }
        if args.scale_out.shape != [self.desc.n] {
            return Err(Error::InvalidProblem(
                "DynamicRangeQuantizePlan: scale_out shape != [n]",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — none. The kernel is single-launch.
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
        args: DynamicRangeQuantizeArgs<'_, TIn, TOut>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let total = (self.desc.n as i64) * (self.desc.d as i64);
        if total == 0 {
            return Ok(());
        }
        let in_ptr = args.input.data.as_raw().0 as *const c_void;
        let sc_ptr = args.scale_out.data.as_raw().0 as *mut c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = match (TIn::KIND, TOut::KIND) {
            (ElementKind::F32, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dynamic_range_quantize_per_token_sym_f32_s8_run(
                    self.desc.n,
                    self.desc.d,
                    self.desc.q_min,
                    self.desc.q_max,
                    in_ptr, sc_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            (ElementKind::F64, ElementKind::S8) => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dynamic_range_quantize_per_token_sym_f64_s8_run(
                    self.desc.n,
                    self.desc.d,
                    self.desc.q_min,
                    self.desc.q_max,
                    in_ptr, sc_ptr, out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "DynamicRangeQuantizePlan::run reached unsupported (TIn, TOut) \
                     (select should have caught this)",
                ))
            }
        };
        map_status(status)
    }
}

/// Build the [`KernelSku`] for a dynamic-range-quantize plan.
fn build_sku<TIn: Element, TOut: IntElement>(op: QuantizeKind) -> KernelSku {
    let precision_guarantee = PrecisionGuarantee {
        math_precision: if TIn::KIND == ElementKind::F64 {
            MathPrecision::F64
        } else {
            MathPrecision::F32
        },
        accumulator: ElementKind::F32,
        // Deterministic — one block per row, no atomics. Block-tree
        // reduction is associative-stable on a single GPU.
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
