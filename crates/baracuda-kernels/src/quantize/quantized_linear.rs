//! `quantized_linear` — fused W8A8 quantized matmul (Phase 8.3).
//!
//! The canonical inference-time LLM matmul recipe:
//!
//! 1. Quantize the FP activation per-token (dynamic-range, symmetric).
//! 2. Accumulate the int8 × int8 GEMM into int32.
//! 3. Dequantize the int32 acc by `scale_a[m] · scale_w[n]` and store as FP.
//!
//! Used by SmoothQuant, AWQ-runtime, and most production W8A8 LLM
//! kernels. The Plan owns the orchestration; the underlying bespoke
//! kernel fuses the int8 mma + dequant + FP store as one launch.
//!
//! ## Layout
//!
//! - `activation`   : `[M, K]` FP (row-major).
//! - `weight_q`     : `[C_out, K]` int8 (row-major — one row per output channel).
//! - `weight_scale` : `[C_out]` FP (per-output-channel, saved when the
//!   weight was quantized offline).
//! - `output`       : `[M, C_out]` FP.
//!
//! `weight_q` is `[C_out, K]` rather than `[K, C_out]` so the inner-K
//! reduction reads contiguous K spans from both the activation row and
//! the weight row — the natural layout for the linear-layer convention
//! `y = x · W^T` where `W` is the weight matrix in `[C_out, C_in]` form
//! (PyTorch `nn.Linear.weight` layout).
//!
//! ## Trailblazer scope
//!
//! - Symmetric + per-token activation quantization (composes
//!   [`super::DynamicRangeQuantizePlan`]).
//! - Per-output-channel weight scale (caller supplies, computed offline).
//! - `TIn ∈ {f32, f64}` activation + output; weight = `S8`.
//! - **Naive kernel** (one thread per output cell, register-only int32
//!   accumulator) — correctness scaffold, not throughput-optimized.
//!   Tiled-smem / mma.sync optimizations land in a perf milestone.
//! - **Inference-only** — no backward. The W8A8 path is forward-only
//!   by convention; if a downstream needs gradients, they should use
//!   [`super::FakeQuantizePlan`] for QAT (quant-aware training) and run
//!   a normal FP matmul.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IntElement, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, S8, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a `quantized_linear` op.
#[derive(Copy, Clone, Debug)]
pub struct QuantizedLinearDescriptor {
    /// Number of token rows in the activation (and rows of the output).
    pub m: i32,
    /// Number of output channels (rows of `weight_q`, cols of output).
    pub c_out: i32,
    /// Inner reduction dim (cols of `activation` and `weight_q`).
    pub k: i32,
    /// Activation quantization range lower bound (symmetric: `-127`).
    pub q_min: i32,
    /// Activation quantization range upper bound (symmetric: `127`).
    pub q_max: i32,
    /// Activation FP element kind. Must match `TIn::KIND`.
    pub activation_element: ElementKind,
    /// Weight int element kind. Today wired only for `S8`.
    pub weight_element: ElementKind,
}

/// Args bundle for a `quantized_linear` launch.
///
/// The caller supplies the already-quantized weight + its per-channel
/// scale (offline-computed). The activation is FP; per-token
/// activation quantization happens inside [`QuantizedLinearPlan::run`]
/// via an internally orchestrated [`super::DynamicRangeQuantizePlan`]
/// pass.
///
/// `act_q_scratch` and `act_scale_scratch` are caller-owned scratch
/// buffers for the quantized activation + computed per-row activation
/// scale. They are part of the args bundle (not workspace) so callers
/// can reuse them across launches without re-allocation — the Plan's
/// `workspace_size()` returns 0.
pub struct QuantizedLinearArgs<'a, TIn: Element, TWQ: IntElement> {
    /// FP activation `[M, K]`.
    pub activation: TensorRef<'a, TIn, 2>,
    /// Already-quantized int8 weight `[C_out, K]`.
    pub weight_q: TensorRef<'a, TWQ, 2>,
    /// Per-output-channel weight scale `[C_out]` in FP.
    pub weight_scale: TensorRef<'a, TIn, 1>,
    /// FP output `[M, C_out]`.
    pub output: TensorMut<'a, TIn, 2>,
    /// Scratch for the per-token quantized activation `[M, K]` in int8.
    /// Caller-owned; reused across launches.
    pub act_q_scratch: TensorMut<'a, S8, 2>,
    /// Scratch for the per-token activation scale `[M]` in FP.
    /// Caller-owned; reused across launches. Populated by the
    /// internally orchestrated dynamic-range pass.
    pub act_scale_scratch: TensorMut<'a, TIn, 1>,
}

/// `quantized_linear` plan (W8A8 fused).
///
/// Composes two passes internally:
///
/// 1. **Activation quantize** — per-token symmetric dynamic-range
///    quantization, fused max-abs reduce + scale compute + quantize.
///    Implemented by the same `dynamic_range_quantize_per_token_sym`
///    kernel that backs [`super::DynamicRangeQuantizePlan`].
/// 2. **Quantized matmul** — fused int8 GEMM + per-row/per-col
///    dequantize + FP store. Implemented by the bespoke
///    `quantized_linear_w8a8` kernel.
///
/// Both passes share the same stream and execute back-to-back; the Plan
/// does NOT own an internal `DynamicRangeQuantizePlan` instance — it
/// invokes the FFI directly to keep the launch ordering explicit.
///
/// **When to use**: W8A8 inference matmul (SmoothQuant / AWQ-runtime
/// style). Inference-only — no BW; for QAT use
/// [`FakeQuantizePlan`](crate::FakeQuantizePlan) + normal FP matmul.
///
/// **Dtypes (trailblazer)**: `TIn (act/out) ∈ {f32, f64}`; `TWQ = S8`.
/// f16 / bf16 activations and u8 weight not yet wired.
///
/// **Shape limits**: `activation` `[M, K]`; `weight_q` `[C_out, K]`;
/// `weight_scale` `[C_out]`; `output` `[M, C_out]`. The W4 layout
/// `[C_out, K]` matches `y = x · W^T` (PyTorch `nn.Linear.weight`).
///
/// **Workspace**: zero in [`Workspace`]. Caller supplies
/// `act_q_scratch` `[M, K]` (int8) and `act_scale_scratch` `[M]`
/// (FP) in [`QuantizedLinearArgs`] for the fused activation-quant
/// pass.
///
/// **Precision guarantee**: deterministic, bit-stable. Naive kernel
/// (one thread per output cell, register-only int32 acc) for
/// correctness; tiled-smem / mma.sync optimizations land in a perf
/// milestone — current variant is **correctness-scaffold, not
/// throughput-optimized**.
pub struct QuantizedLinearPlan<TIn: Element, TWQ: IntElement> {
    desc: QuantizedLinearDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TWQ)>,
}

impl<TIn: Element, TWQ: IntElement> QuantizedLinearPlan<TIn, TWQ> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &QuantizedLinearDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.activation_element != TIn::KIND {
            return Err(Error::Unsupported(
                "QuantizedLinearPlan: descriptor activation_element != TIn",
            ));
        }
        if desc.weight_element != TWQ::KIND {
            return Err(Error::Unsupported(
                "QuantizedLinearPlan: descriptor weight_element != TWQ",
            ));
        }
        // Trailblazer dtype matrix: TIn ∈ {f32, f64}, TWQ = S8.
        if !matches!(TIn::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "QuantizedLinearPlan: 8.3 trailblazer only wires f32 / f64 \
                 activation (f16 / bf16 deferred)",
            ));
        }
        if TWQ::KIND != ElementKind::S8 {
            return Err(Error::Unsupported(
                "QuantizedLinearPlan: 8.3 trailblazer only wires S8 weight \
                 (U8 deferred)",
            ));
        }
        if desc.m < 0 || desc.c_out < 0 || desc.k < 0 {
            return Err(Error::InvalidProblem(
                "QuantizedLinearPlan: m, c_out, k must be non-negative",
            ));
        }
        if desc.q_max <= 0 {
            return Err(Error::InvalidProblem(
                "QuantizedLinearPlan: q_max must be > 0",
            ));
        }
        if desc.q_max < desc.q_min {
            return Err(Error::InvalidProblem(
                "QuantizedLinearPlan: q_max < q_min",
            ));
        }
        if desc.m > 65535 {
            return Err(Error::Unsupported(
                "QuantizedLinearPlan: M > 65535 — the internal dynamic-range pass \
                 uses one block per row and would exceed the legacy grid limit \
                 (lift when row tiling lands)",
            ));
        }
        let sku = build_sku::<TIn, TWQ>(QuantizeKind::QuantizedLinear);
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args at run time.
    pub fn can_implement(&self, args: &QuantizedLinearArgs<'_, TIn, TWQ>) -> Result<()> {
        if args.activation.shape != [self.desc.m, self.desc.k] {
            return Err(Error::InvalidProblem(
                "QuantizedLinearPlan: activation shape != [M, K]",
            ));
        }
        if args.weight_q.shape != [self.desc.c_out, self.desc.k] {
            return Err(Error::InvalidProblem(
                "QuantizedLinearPlan: weight_q shape != [C_out, K]",
            ));
        }
        if args.weight_scale.shape != [self.desc.c_out] {
            return Err(Error::InvalidProblem(
                "QuantizedLinearPlan: weight_scale shape != [C_out]",
            ));
        }
        if args.output.shape != [self.desc.m, self.desc.c_out] {
            return Err(Error::InvalidProblem(
                "QuantizedLinearPlan: output shape != [M, C_out]",
            ));
        }
        if args.act_q_scratch.shape != [self.desc.m, self.desc.k] {
            return Err(Error::InvalidProblem(
                "QuantizedLinearPlan: act_q_scratch shape != [M, K]",
            ));
        }
        if args.act_scale_scratch.shape != [self.desc.m] {
            return Err(Error::InvalidProblem(
                "QuantizedLinearPlan: act_scale_scratch shape != [M]",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — none. Scratch buffers are caller-owned via the
    /// args bundle (`act_q_scratch` + `act_scale_scratch`), allowing
    /// reuse across launches.
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
        args: QuantizedLinearArgs<'_, TIn, TWQ>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if (self.desc.m as i64) * (self.desc.c_out as i64) == 0
            || self.desc.k == 0
        {
            return Ok(());
        }

        let stream_ptr = stream.as_raw() as *mut c_void;

        // ---- Pass 1: dynamic-range per-token symmetric quantize the
        //              FP activation into the int8 scratch. -----------
        let act_ptr = args.activation.data.as_raw().0 as *const c_void;
        let act_scale_ptr = args.act_scale_scratch.data.as_raw().0 as *mut c_void;
        let act_q_ptr = args.act_q_scratch.data.as_raw().0 as *mut c_void;
        let drq_status = match TIn::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dynamic_range_quantize_per_token_sym_f32_s8_run(
                    self.desc.m,
                    self.desc.k,
                    self.desc.q_min,
                    self.desc.q_max,
                    act_ptr, act_scale_ptr, act_q_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_dynamic_range_quantize_per_token_sym_f64_s8_run(
                    self.desc.m,
                    self.desc.k,
                    self.desc.q_min,
                    self.desc.q_max,
                    act_ptr, act_scale_ptr, act_q_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "QuantizedLinearPlan::run reached unsupported TIn at \
                     activation-quantize pass (select should have caught)",
                ))
            }
        };
        map_status(drq_status)?;

        // ---- Pass 2: fused quantized-linear (int8 GEMM + dequant + FP store). ----
        let weight_ptr = args.weight_q.data.as_raw().0 as *const c_void;
        let act_q_const = args.act_q_scratch.data.as_raw().0 as *const c_void;
        let act_scale_const = args.act_scale_scratch.data.as_raw().0 as *const c_void;
        let w_scale_ptr = args.weight_scale.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let ql_status = match TIn::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantized_linear_w8a8_f32_run(
                    self.desc.m,
                    self.desc.c_out,
                    self.desc.k,
                    weight_ptr, act_q_const,
                    act_scale_const, w_scale_ptr,
                    out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantized_linear_w8a8_f64_run(
                    self.desc.m,
                    self.desc.c_out,
                    self.desc.k,
                    weight_ptr, act_q_const,
                    act_scale_const, w_scale_ptr,
                    out_ptr,
                    core::ptr::null_mut(), 0, stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "QuantizedLinearPlan::run reached unsupported TIn at \
                     quantized-linear pass (select should have caught)",
                ))
            }
        };
        map_status(ql_status)
    }
}

/// Build the [`KernelSku`] for a quantized-linear plan.
fn build_sku<TIn: Element, TWQ: IntElement>(op: QuantizeKind) -> KernelSku {
    let precision_guarantee = PrecisionGuarantee {
        math_precision: if TIn::KIND == ElementKind::F64 {
            MathPrecision::F64
        } else {
            MathPrecision::F32
        },
        accumulator: ElementKind::F32,
        // Deterministic: int32 reduction + per-row block scan + FP
        // dequant; no atomics. Different (M, K) tile orderings would
        // change the float sum-cancellation pattern in a later tiled
        // kernel, but the naive trailblazer is fully serial within a
        // thread and bit-stable on the same hardware.
        bit_stable_on_same_hardware: true,
        deterministic: true,
    };
    KernelSku {
        category: OpCategory::Quantization,
        op: op as u16,
        element: TIn::KIND,
        aux_element: Some(TWQ::KIND),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee,
    }
}
