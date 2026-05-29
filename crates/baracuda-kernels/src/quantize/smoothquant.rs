//! `SmoothQuantLinearPlan` — Phase 45 zero-new-CUDA composition.
//!
//! Implements the inference-time linear pass of **SmoothQuant**
//! (Xiao et al. ICML 2023, MIT;
//! [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant)).
//!
//! SmoothQuant is an **offline algorithmic recipe** — a Python
//! preprocessing pass that migrates outlier difficulty from
//! activations to weights via a per-channel divisor `s[K]` so that
//! both the smoothed activation `A_smooth = A / diag(s)` and the
//! smoothed weight `W_smooth = diag(s) · W` quantize cleanly under a
//! single per-tensor activation scale + a per-output-channel weight
//! scale. The smoothing itself is **not** a CUDA kernel; it lives in
//! the Python flow at training-prep time. baracuda only needs to
//! consume the already-smoothed-and-quantized tensors.
//!
//! The inference-time math is the standard W8A8 dequant:
//!
//! ```text
//! y[m, n] = act_scale · weight_scale[n] · Σ_k a_q[m, k] · w_q[n, k]
//! ```
//!
//! Differences from [`super::QuantizedLinearPlan`]:
//!
//! - **Per-tensor activation scale** (single `f32`) vs per-token
//!   `[M]` dynamic-range scale. The whole point of SmoothQuant is
//!   that one static scalar suffices once outliers are migrated.
//! - **Caller-pre-quantized int8 activation** — no internal
//!   `dynamic_range_quantize_per_token_sym` pass.
//!
//! Composition strategy: the bespoke `quantized_linear_w8a8_*`
//! kernel (vendored Milestone 8.3) consumes `scale_a: [M]` — we
//! reuse it verbatim by having the caller supply an `[M]` scratch
//! buffer that we fill with the constant `act_scale` via
//! [`super::super::FillPlan`] before the matmul. Zero new CUDA.
//!
//! ## Layout
//!
//! - `act_q`        : `[M, K]` int8 (row-major).
//! - `weight_q`     : `[N, K]` int8 (row-major — one row per output channel,
//!   matching PyTorch `nn.Linear.weight` layout, same as
//!   [`super::QuantizedLinearPlan`]).
//! - `weight_scale` : `[N]` FP (per-output-channel; saved alongside
//!   the smoothed-then-quantized weights from the offline flow).
//! - `act_scale`    : single `f32` scalar in the descriptor (per-tensor).
//! - `output`       : `[M, N]` FP.
//!
//! ## Trailblazer scope
//!
//! - `TIn ∈ {f32, f64}` activation-scale + weight-scale + output;
//!   `TWQ = S8` weight; activation = `S8`.
//!   (f16 / bf16 / u8-weight follow the same matrix as
//!   [`QuantizedLinearPlan`]; deferred until the underlying bespoke
//!   `quantized_linear_w8a8` kernel grows the dtypes.)
//! - **Inference-only** — no backward. The W8A8 path is forward-only
//!   by convention; if a downstream needs gradients, they should use
//!   [`super::FakeQuantizePlan`] for QAT.
//! - **No bias fusion in trailblazer** — bias addition is a separate
//!   downstream op (Affine / Binary Add). The underlying
//!   `quantized_linear_w8a8` kernel doesn't take a bias today and
//!   we don't synthesize one here.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, IntElement, KernelSku, MathPrecision, OpCategory,
    PlanPreference, PrecisionGuarantee, QuantizeKind, S8, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a `SmoothQuant` linear op.
///
/// The per-tensor activation scale lives in the descriptor (not the
/// args) because in the SmoothQuant flow it's part of the model's
/// frozen quantization metadata — it doesn't change between launches
/// for the same layer.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct SmoothQuantLinearDescriptor {
    /// Number of token rows in the activation (and rows of the output).
    pub m: i32,
    /// Number of output channels (rows of `weight_q`, cols of output).
    pub n: i32,
    /// Inner reduction dim (cols of `act_q` and `weight_q`).
    pub k: i32,
    /// Per-tensor activation scale produced by the offline SmoothQuant
    /// Python flow. Always `f32` regardless of `TIn` — the underlying
    /// `quantized_linear_w8a8` kernel does the scale multiply in float
    /// space irrespective of output dtype.
    pub act_scale: f32,
    /// Activation int element kind. Today wired only for `S8`.
    pub activation_element: ElementKind,
    /// Weight int element kind. Today wired only for `S8`.
    pub weight_element: ElementKind,
    /// Output FP element kind. Must match `TIn::KIND`.
    pub output_element: ElementKind,
}

impl SmoothQuantLinearDescriptor {
    /// Construct a `SmoothQuantLinearDescriptor` for the given problem
    /// shape and per-tensor activation scale. Defaults `S8` for both
    /// activation and weight; output element matches `TIn::KIND`.
    pub fn new<TIn: Element>(m: i32, n: i32, k: i32, act_scale: f32) -> Self {
        Self {
            m,
            n,
            k,
            act_scale,
            activation_element: ElementKind::S8,
            weight_element: ElementKind::S8,
            output_element: TIn::KIND,
        }
    }
}

/// Args bundle for a `SmoothQuant` linear launch.
///
/// `act_scale_scratch` is a caller-owned `[M]` FP scratch buffer used
/// to broadcast the descriptor's per-tensor `act_scale` into the
/// per-row form the underlying `quantized_linear_w8a8` kernel
/// consumes. Caller-owned so it can be reused across launches without
/// re-allocation — the Plan's `workspace_size()` returns 0.
pub struct SmoothQuantLinearArgs<'a, TIn: Element, TWQ: IntElement> {
    /// Pre-quantized int8 activation `[M, K]`.
    pub act_q: TensorRef<'a, S8, 2>,
    /// Pre-smoothed-then-quantized int8 weight `[N, K]`.
    pub weight_q: TensorRef<'a, TWQ, 2>,
    /// Per-output-channel weight scale `[N]` in FP.
    pub weight_scale: TensorRef<'a, TIn, 1>,
    /// FP output `[M, N]`.
    pub output: TensorMut<'a, TIn, 2>,
    /// Scratch for the per-row broadcast of `act_scale`. `[M]` FP.
    /// Caller-owned; reused across launches. Populated by the plan
    /// before the matmul launch.
    pub act_scale_scratch: TensorMut<'a, TIn, 1>,
}

/// `SmoothQuant` linear plan — pure Rust composition over the
/// bespoke `quantized_linear_w8a8` kernel.
///
/// **When to use**: SmoothQuant inference matmul. Activation has
/// already been smoothed (divided by per-channel `s[K]`) and
/// quantized per-tensor to int8; weight has already been smoothed
/// (multiplied by `s[K]`) and quantized per-output-channel to int8;
/// caller passes both, plus the static per-tensor act-scale + per-N
/// weight-scale, to this plan.
///
/// **Dtypes (trailblazer)**: `TIn (scales/out) ∈ {f32, f64}`;
/// `TWQ = S8` weight; activation is fixed at `S8`. f16 / bf16 / u8
/// weight follow once the underlying `quantized_linear_w8a8` kernel
/// grows those dtypes (same matrix as
/// [`super::QuantizedLinearPlan`]).
///
/// **Shape limits**: `act_q` `[M, K]`; `weight_q` `[N, K]`;
/// `weight_scale` `[N]`; `output` `[M, N]`. `[N, K]` weight layout
/// matches `y = x · W^T` (PyTorch `nn.Linear.weight`).
///
/// **Workspace**: zero in [`Workspace`]. Caller supplies
/// `act_scale_scratch` `[M]` (FP) in [`SmoothQuantLinearArgs`] for
/// the act-scale broadcast.
///
/// **Precision guarantee**: deterministic, bit-stable on the same
/// hardware (inherits from the underlying `quantized_linear_w8a8`
/// kernel — register-only int32 accumulator + serial FP scale
/// multiply, no atomics).
pub struct SmoothQuantLinearPlan<TIn: Element, TWQ: IntElement> {
    desc: SmoothQuantLinearDescriptor,
    sku: KernelSku,
    _marker: PhantomData<(TIn, TWQ)>,
}

impl<TIn: Element, TWQ: IntElement> SmoothQuantLinearPlan<TIn, TWQ> {
    /// Pick a kernel for `desc`.
    pub fn select(
        _stream: &Stream,
        desc: &SmoothQuantLinearDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.output_element != TIn::KIND {
            return Err(Error::Unsupported(
                "SmoothQuantLinearPlan: descriptor output_element != TIn",
            ));
        }
        if desc.weight_element != TWQ::KIND {
            return Err(Error::Unsupported(
                "SmoothQuantLinearPlan: descriptor weight_element != TWQ",
            ));
        }
        if desc.activation_element != ElementKind::S8 {
            return Err(Error::Unsupported(
                "SmoothQuantLinearPlan: trailblazer only wires S8 activation \
                 (matches underlying quantized_linear_w8a8 kernel)",
            ));
        }
        // Trailblazer dtype matrix mirrors QuantizedLinearPlan exactly
        // (same underlying kernel): TIn ∈ {f32, f64}, TWQ = S8.
        if !matches!(TIn::KIND, ElementKind::F32 | ElementKind::F64) {
            return Err(Error::Unsupported(
                "SmoothQuantLinearPlan: trailblazer only wires f32 / f64 \
                 output (f16 / bf16 follow when quantized_linear_w8a8 grows them)",
            ));
        }
        if TWQ::KIND != ElementKind::S8 {
            return Err(Error::Unsupported(
                "SmoothQuantLinearPlan: trailblazer only wires S8 weight (U8 deferred)",
            ));
        }
        if desc.m < 0 || desc.n < 0 || desc.k < 0 {
            return Err(Error::InvalidProblem(
                "SmoothQuantLinearPlan: m, n, k must be non-negative",
            ));
        }
        if !desc.act_scale.is_finite() {
            return Err(Error::InvalidProblem(
                "SmoothQuantLinearPlan: act_scale must be finite",
            ));
        }
        // We don't require act_scale > 0 strictly — a zero scale produces
        // a zero output (degenerate but well-defined). Negative scales
        // are unusual but mathematically valid (SmoothQuant's offline
        // flow always yields positive scales; we don't enforce here).
        let precision_guarantee = PrecisionGuarantee {
            math_precision: if TIn::KIND == ElementKind::F64 {
                MathPrecision::F64
            } else {
                MathPrecision::F32
            },
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Quantization,
            // SmoothQuant rides on the same op-kind discriminant as the
            // existing W8A8 path — they're variants of the same logical
            // op (W8A8 fused matmul), just with different
            // activation-scale provenance.
            op: QuantizeKind::QuantizedLinear as u16,
            element: TIn::KIND,
            aux_element: Some(TWQ::KIND),
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

    /// Validate args at run time.
    pub fn can_implement(&self, args: &SmoothQuantLinearArgs<'_, TIn, TWQ>) -> Result<()> {
        if args.act_q.shape != [self.desc.m, self.desc.k] {
            return Err(Error::InvalidProblem(
                "SmoothQuantLinearPlan: act_q shape != [M, K]",
            ));
        }
        if args.weight_q.shape != [self.desc.n, self.desc.k] {
            return Err(Error::InvalidProblem(
                "SmoothQuantLinearPlan: weight_q shape != [N, K]",
            ));
        }
        if args.weight_scale.shape != [self.desc.n] {
            return Err(Error::InvalidProblem(
                "SmoothQuantLinearPlan: weight_scale shape != [N]",
            ));
        }
        if args.output.shape != [self.desc.m, self.desc.n] {
            return Err(Error::InvalidProblem(
                "SmoothQuantLinearPlan: output shape != [M, N]",
            ));
        }
        if args.act_scale_scratch.shape != [self.desc.m] {
            return Err(Error::InvalidProblem(
                "SmoothQuantLinearPlan: act_scale_scratch shape != [M]",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — none. The act-scale `[M]` broadcast buffer is
    /// caller-owned via the args bundle (`act_scale_scratch`),
    /// allowing reuse across launches.
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
    ///
    /// Two-pass: (1) fill the `[M]` scratch with the descriptor's
    /// `act_scale`; (2) launch the `quantized_linear_w8a8` kernel
    /// directly via the FFI (skips the dynamic-range-quantize pass
    /// that [`super::QuantizedLinearPlan`] does).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: SmoothQuantLinearArgs<'_, TIn, TWQ>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if (self.desc.m as i64) * (self.desc.n as i64) == 0 || self.desc.k == 0 {
            return Ok(());
        }

        let stream_ptr = stream.as_raw() as *mut c_void;

        // ---- Pass 1: broadcast `act_scale` across [M]. ---------------
        //
        // We invoke the fill FFI directly rather than constructing a
        // FillPlan — both paths land on the same underlying kernel,
        // and the direct call sidesteps the FillPlan's Element trait
        // bound which would force us through transmute_copy gymnastics
        // (the descriptor knows the scale as f32; the scratch is TIn).
        let fill_ptr = args.act_scale_scratch.data.as_raw().0 as *mut c_void;
        let fill_status = match TIn::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_fill_f32_run(
                    self.desc.m as i64,
                    fill_ptr,
                    self.desc.act_scale,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_fill_f64_run(
                    self.desc.m as i64,
                    fill_ptr,
                    self.desc.act_scale as f64,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "SmoothQuantLinearPlan::run reached unsupported TIn at \
                     act-scale broadcast (select should have caught)",
                ))
            }
        };
        map_status(fill_status)?;

        // ---- Pass 2: fused quantized-linear (int8 GEMM + dequant + FP store). ----
        let weight_ptr = args.weight_q.data.as_raw().0 as *const c_void;
        let act_q_ptr = args.act_q.data.as_raw().0 as *const c_void;
        let act_scale_const = args.act_scale_scratch.data.as_raw().0 as *const c_void;
        let w_scale_ptr = args.weight_scale.data.as_raw().0 as *const c_void;
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;
        let ql_status = match TIn::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantized_linear_w8a8_f32_run(
                    self.desc.m,
                    self.desc.n,
                    self.desc.k,
                    weight_ptr,
                    act_q_ptr,
                    act_scale_const,
                    w_scale_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_quantized_linear_w8a8_f64_run(
                    self.desc.m,
                    self.desc.n,
                    self.desc.k,
                    weight_ptr,
                    act_q_ptr,
                    act_scale_const,
                    w_scale_ptr,
                    out_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "SmoothQuantLinearPlan::run reached unsupported TIn at \
                     quantized-linear pass (select should have caught)",
                ))
            }
        };
        map_status(ql_status)
    }
}
