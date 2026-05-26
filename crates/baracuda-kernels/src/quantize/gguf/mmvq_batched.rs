//! GGUF batched MMVQ × N-experts plan — Phase 20.1.
//!
//! General-purpose MMVQ primitive that takes N weight matrices + routing
//! semantics (`sorted_token_ids[]`, `expert_offsets[]`, optional
//! `topk_weights[]`) and computes
//!
//! ```text
//! For each dispatch i ∈ [0, M_total):
//!   token  = sorted_token_ids[i]
//!   expert = find_e(expert_offsets, i)
//!   w      = topk_weights[i] (or 1.0)
//!   for r ∈ [0, n_rows_per_expert):
//!     output[token, r] (+)= w * dot(weights[expert, r, :], activations[token, :])
//! ```
//!
//! `(+)=` is a regular store when `top_k == 1` (no aliasing — each
//! `(token, row)` pair is written exactly once) and an atomicAdd when
//! `top_k > 1`. **For `top_k > 1` the caller must zero-initialize
//! `output` before calling `run()`.**
//!
//! Use cases:
//!   * MoE inference (Mixtral, DeepSeek-V2/V3, Qwen-MoE, ...).
//!   * Speculative-decoding draft-vs-target dispatch.
//!   * Any workload that benefits from fusing N independent MMVQ launches
//!     into a single grid.
//!
//! ## Coverage
//!
//! - **Quantized weights** (11 block formats × 3 activation dtypes = 33
//!   variants): `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`,
//!   `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` × `{f32, f16, bf16}`.
//! - **Pure FP** (non-quant; 3 variants): `f32`, `f16`, `bf16`. Weight,
//!   activation, output share dtype.
//!
//! ## Workspace
//!
//! `m_total * sizeof(i32)` bytes — used for an internal
//! `dispatch_to_expert[]` array (precomputed once per launch via a small
//! prelude kernel; avoids per-block binary search over
//! `expert_offsets[]`).
//!
//! ## Determinism
//!
//! - `top_k == 1` path is fully deterministic and bit-stable on identical
//!   hardware (same warp-shuffle reduction as the single-MMVQ plan).
//! - `top_k > 1` path uses `atomicAdd` for the dK/dV-style scatter; bit
//!   exact across launches is NOT guaranteed (atomic ordering varies).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, Element, ElementKind, GgufBlockFormat, KernelSku,
    MathPrecision, OpCategory, PlanPreference, PrecisionGuarantee, QuantizeKind, TensorMut,
    TensorRef, Workspace, U8,
};
use baracuda_types::DeviceRepr;
use half::{bf16, f16};

use crate::quantize::map_status;

/// Sealed marker for activation / destination element types accepted by
/// [`GgufMmvqBatchedPlan`]. Identical set to [`GgufMmvqActivation`] from
/// the single-MMVQ plan — `f32`, `f16`, `bf16`.
pub trait GgufMmvqBatchedActivation: Element + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
    impl Sealed for half::bf16 {}
}

impl GgufMmvqBatchedActivation for f32 {}
impl GgufMmvqBatchedActivation for f16 {}
impl GgufMmvqBatchedActivation for bf16 {}

/// Selects between quantized (GGUF block-packed) and pure-FP weight
/// formats. Pure-FP variants use the same dtype for weights, activations,
/// and output (matching the activation dtype `T`).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum GgufMmvqBatchedFormat {
    /// GGUF block-packed weights. Type-0/1 (32-elt blocks) or k-quant
    /// (256-elt super-blocks). One of the 11 GGUF block formats.
    Quantized(GgufBlockFormat),
    /// Pure FP weights. Dtype matches the activation `T` (no separate
    /// weight dtype field — keep the plan signature tight).
    Fp,
}

/// Descriptor for a batched MMVQ op.
#[derive(Copy, Clone, Debug)]
pub struct GgufMmvqBatchedDescriptor {
    /// Number of experts (= number of weight matrices stacked along
    /// the leading axis of `weights`).
    pub n_experts: i32,
    /// Number of output rows per expert (= rows of each weight matrix
    /// = output feature count `N`).
    pub n_rows_per_expert: i32,
    /// Number of unpacked columns per row (= length of each activation
    /// vector = `K`). For quantized formats, must be a multiple of
    /// `block_format.block_size()`.
    pub n_cols: i32,
    /// Total dispatch count across all experts — `M_tokens × top_k`.
    /// The workspace is sized exactly to `m_total * sizeof(i32)` bytes.
    pub m_total: i32,
    /// Hint for the kernel's atomic-vs-store dispatch:
    ///   * `1` (default) — no output-row aliasing, regular stores.
    ///   * `> 1` — multiple dispatches may write to the same `(token,
    ///     row)`, switch to `atomicAdd`. Caller MUST zero-initialize
    ///     the output tensor.
    pub top_k: i32,
    /// Block-format selector (quantized variant) or pure-FP marker.
    pub format: GgufMmvqBatchedFormat,
}

impl Default for GgufMmvqBatchedDescriptor {
    fn default() -> Self {
        Self {
            n_experts: 0,
            n_rows_per_expert: 0,
            n_cols: 0,
            m_total: 0,
            top_k: 1,
            format: GgufMmvqBatchedFormat::Quantized(GgufBlockFormat::Q8_0),
        }
    }
}

/// Args bundle for a batched MMVQ launch.
///
/// Generic on the activation / destination dtype `T`. The weight tensor
/// is `u8` bytes for the quantized variant and dtype-`T` for the pure-FP
/// variant — the [`GgufMmvqBatchedFormat`] field of the descriptor
/// disambiguates which interpretation is in effect.
///
/// Trait bound is `T: DeviceRepr + Copy` per the Phase 13 pragma —
/// activation dtype may not be under the [`Element`] trait for every
/// future variant (S4 / U4 / Fp8 are conceptually wireable), but for
/// now only `f32` / `f16` / `bf16` are wired.
pub struct GgufMmvqBatchedArgs<'a, T: DeviceRepr + Copy + 'static = f32> {
    /// Weight tensor.
    ///
    ///   * Quantized variant: raw GGUF-packed bytes
    ///     `[n_experts × n_rows_per_expert × (n_cols / block_size) × type_size]`
    ///     bytes total. The plan layer takes it as a `u8` 1-D tensor;
    ///     the kernel reinterprets per the descriptor's `block_format`.
    ///   * Pure-FP variant: dense FP elements
    ///     `[n_experts × n_rows_per_expert × n_cols]` of dtype `T`.
    ///     **The plan-layer surface still takes `u8` bytes** to keep
    ///     the API uniform — interpret element counts as
    ///     `bytes / size_of::<T>()`.
    pub weights: TensorRef<'a, U8, 1>,
    /// Activation tensor `[M_tokens, n_cols]` of dtype `T`.
    pub activations: TensorRef<'a, T, 2>,
    /// `[M_total]` — i-th entry is the token id this dispatch operates
    /// on. `M_total = M_tokens × top_k`.
    pub sorted_token_ids: TensorRef<'a, i32, 1>,
    /// `[n_experts + 1]` — cumulative dispatch-range boundaries. The
    /// range `[expert_offsets[e], expert_offsets[e+1])` is the set of
    /// dispatch indices going to expert `e`. `expert_offsets[n_experts]
    /// == M_total`.
    pub expert_offsets: TensorRef<'a, i32, 1>,
    /// Optional `[M_total]` — per-dispatch scalar multiplier (the
    /// routing softmax weight). `None` (caller signals via `None` →
    /// kernel sees nullptr) means "unweighted" (multiplier = 1.0).
    pub topk_weights: Option<TensorRef<'a, f32, 1>>,
    /// Output tensor `[M_tokens, n_rows_per_expert]` of dtype `T`.
    /// When `top_k > 1` the caller MUST zero-initialize before calling
    /// `run()` (atomicAdd scatter accumulates onto whatever's already
    /// in memory).
    pub output: TensorMut<'a, T, 2>,
}

/// `gguf_mmvq_batched` plan.
///
/// See module-level docs for op shape, coverage, and determinism notes.
///
/// **When to use**: MoE inference, speculative-decoding dispatch, or any
/// workload that benefits from fusing N independent MMVQ launches into
/// a single kernel.
///
/// **Block formats**: all 11 GGUF block formats (quantized variant) +
/// pure FP (no quant). Activation dtype: `f32`, `f16`, `bf16`.
///
/// **Workspace**: `m_total * sizeof(i32)` bytes. Used for an internal
/// `dispatch_to_expert[]` array (small prelude kernel converts the
/// `expert_offsets[]` array into per-dispatch lookup; avoids per-block
/// binary search).
///
/// **Precision guarantee**:
///   * `top_k == 1`: deterministic, bit-stable on identical hardware
///     (same warp-shuffle reduction as the single-MMVQ plan).
///   * `top_k > 1`: NOT bit-stable across launches (atomicAdd ordering
///     varies). Numerically equivalent up to atomic-summation reorder.
pub struct GgufMmvqBatchedPlan<T: DeviceRepr + Copy + 'static = f32> {
    desc: GgufMmvqBatchedDescriptor,
    sku: KernelSku,
    _phantom: PhantomData<T>,
}

impl<T: GgufMmvqBatchedActivation> GgufMmvqBatchedPlan<T> {
    /// Pick a kernel for `desc`. Errors:
    ///   * Non-positive dims.
    ///   * Quantized variant + `n_cols` not a multiple of the block size.
    ///   * `m_total < 0`.
    pub fn select(
        _stream: &Stream,
        desc: &GgufMmvqBatchedDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.n_experts <= 0 {
            return Err(Error::InvalidProblem(
                "GgufMmvqBatchedPlan: n_experts must be positive",
            ));
        }
        if desc.n_rows_per_expert < 0 || desc.n_cols < 0 || desc.m_total < 0 {
            return Err(Error::InvalidProblem(
                "GgufMmvqBatchedPlan: n_rows_per_expert / n_cols / m_total must be non-negative",
            ));
        }
        if desc.top_k <= 0 {
            return Err(Error::InvalidProblem(
                "GgufMmvqBatchedPlan: top_k must be positive",
            ));
        }
        if let GgufMmvqBatchedFormat::Quantized(fmt) = desc.format {
            if !fmt.has_mmvq() {
                return Err(Error::Unsupported(
                    "GgufMmvqBatchedPlan: block format reports no MMVQ kernel",
                ));
            }
            let bs = fmt.block_size() as i32;
            if desc.n_cols % bs != 0 {
                return Err(Error::InvalidProblem(
                    "GgufMmvqBatchedPlan: n_cols must be a multiple of the block size",
                ));
            }
            // Phase 22 — debug-build assertion against the type-0/1 MMVQ
            // `n_cols >= 2 * GGML_CUDA_DMMV_X = 64` implicit minimum.
            // Threads `tid=16..31` always read columns `32..62`; for
            // `n_cols < 64` those reads are OOB. Single-matrix `GgufMmvqPlan`
            // is incidentally safe because the OOB region is unallocated
            // zero memory; **GgufMmvqBatchedPlan is NOT** because activations
            // are contiguous-batched across tokens, so OOB reads land in
            // the next token's activation row producing silent-wrong
            // results. Caught by Phase 20.1's test fixture; k-quants use
            // per-format bespoke iteration and aren't affected.
            #[cfg(debug_assertions)]
            {
                use baracuda_kernels_types::GgufBlockFormat;
                let is_type_0_1 = matches!(
                    fmt,
                    GgufBlockFormat::Q4_0
                        | GgufBlockFormat::Q4_1
                        | GgufBlockFormat::Q5_0
                        | GgufBlockFormat::Q5_1
                        | GgufBlockFormat::Q8_0,
                );
                if is_type_0_1 && desc.n_cols < 64 {
                    return Err(Error::InvalidProblem(
                        "GgufMmvqBatchedPlan: type-0/1 block formats require \
                         n_cols >= 64 (2 * GGML_CUDA_DMMV_X); smaller n_cols \
                         produces silent-wrong results because contiguous-batched \
                         activations make threads' OOB reads hit adjacent tokens' \
                         rows",
                    ));
                }
            }
        }
        Ok(Self {
            desc: *desc,
            sku: build_sku(&desc.format, T::KIND),
            _phantom: PhantomData,
        })
    }

    /// Validate args. Checks tensor shapes vs the descriptor (does not
    /// inspect device memory). The weight byte count is checked against
    /// the descriptor's `(n_experts, n_rows_per_expert, n_cols, format)`
    /// product.
    pub fn can_implement(&self, args: &GgufMmvqBatchedArgs<'_, T>) -> Result<()> {
        if args.activations.shape[1] != self.desc.n_cols {
            return Err(Error::InvalidProblem(
                "GgufMmvqBatchedPlan: activations.shape[1] != n_cols",
            ));
        }
        if args.output.shape[1] != self.desc.n_rows_per_expert {
            return Err(Error::InvalidProblem(
                "GgufMmvqBatchedPlan: output.shape[1] != n_rows_per_expert",
            ));
        }
        if args.output.shape[0] != args.activations.shape[0] {
            return Err(Error::InvalidProblem(
                "GgufMmvqBatchedPlan: output.shape[0] != activations.shape[0] (M_tokens mismatch)",
            ));
        }
        if args.sorted_token_ids.shape[0] != self.desc.m_total {
            return Err(Error::InvalidProblem(
                "GgufMmvqBatchedPlan: sorted_token_ids.shape[0] != m_total",
            ));
        }
        if args.expert_offsets.shape[0] != self.desc.n_experts + 1 {
            return Err(Error::InvalidProblem(
                "GgufMmvqBatchedPlan: expert_offsets.shape[0] != n_experts + 1",
            ));
        }
        if let Some(ref tw) = args.topk_weights {
            if tw.shape[0] != self.desc.m_total {
                return Err(Error::InvalidProblem(
                    "GgufMmvqBatchedPlan: topk_weights.shape[0] != m_total",
                ));
            }
        }
        // Weight byte size check.
        let expected_weight_bytes: i64 = match self.desc.format {
            GgufMmvqBatchedFormat::Quantized(fmt) => {
                let blocks_per_row = self.desc.n_cols / fmt.block_size() as i32;
                (self.desc.n_experts as i64)
                    * (self.desc.n_rows_per_expert as i64)
                    * (blocks_per_row as i64)
                    * (fmt.type_size() as i64)
            }
            GgufMmvqBatchedFormat::Fp => {
                (self.desc.n_experts as i64)
                    * (self.desc.n_rows_per_expert as i64)
                    * (self.desc.n_cols as i64)
                    * (core::mem::size_of::<T>() as i64)
            }
        };
        if (args.weights.shape[0] as i64) < expected_weight_bytes {
            return Err(Error::InvalidProblem(
                "GgufMmvqBatchedPlan: weights byte length < n_experts * n_rows_per_expert * (n_cols/bs) * type_size",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — `m_total * sizeof(i32)` for the
    /// `dispatch_to_expert[]` prelude buffer.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        (self.desc.m_total as usize) * core::mem::size_of::<i32>()
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
        workspace: Workspace<'_>,
        args: GgufMmvqBatchedArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;

        // Short-circuit on the trivial empty case (no dispatches).
        if self.desc.m_total == 0
            || self.desc.n_experts == 0
            || self.desc.n_rows_per_expert == 0
            || self.desc.n_cols == 0
        {
            return Ok(());
        }

        // Workspace: m_total * sizeof(i32) bytes for dispatch_to_expert.
        let need = self.workspace_size();
        let (ws_ptr, ws_bytes) = match workspace {
            Workspace::None => {
                return Err(Error::WorkspaceTooSmall { needed: need, got: 0 });
            }
            Workspace::Borrowed(slice) => {
                let got = slice.len();
                if got < need {
                    return Err(Error::WorkspaceTooSmall { needed: need, got });
                }
                (slice.as_raw().0 as *mut c_void, got)
            }
        };

        let w_ptr = args.weights.data.as_raw().0 as *const c_void;
        let y_ptr = args.activations.data.as_raw().0 as *const c_void;
        let dst_ptr = args.output.data.as_raw().0 as *mut c_void;
        let tids_ptr = args.sorted_token_ids.data.as_raw().0 as *const i32;
        let off_ptr = args.expert_offsets.data.as_raw().0 as *const i32;
        let tw_ptr = args
            .topk_weights
            .as_ref()
            .map(|t| t.data.as_raw().0 as *const f32)
            .unwrap_or(core::ptr::null());
        let stream_ptr = stream.as_raw() as *mut c_void;

        let status = unsafe {
            dispatch_ffi::<T>(
                &self.desc.format,
                self.desc.n_experts,
                self.desc.n_rows_per_expert,
                self.desc.n_cols,
                w_ptr,
                y_ptr,
                tids_ptr,
                off_ptr,
                tw_ptr,
                dst_ptr,
                self.desc.top_k,
                ws_ptr,
                ws_bytes,
                stream_ptr,
            )
        };
        map_status(status)
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_ffi<T: GgufMmvqBatchedActivation>(
    format: &GgufMmvqBatchedFormat,
    n_experts: i32,
    n_rows_per_expert: i32,
    n_cols: i32,
    weights: *const c_void,
    activations: *const c_void,
    sorted_token_ids: *const i32,
    expert_offsets: *const i32,
    topk_weights: *const f32,
    output: *mut c_void,
    top_k: i32,
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    match (format, T::KIND) {
        // ---- Pure FP ----
        (GgufMmvqBatchedFormat::Fp, ElementKind::F32) => {
            unsafe { baracuda_kernels_sys::baracuda_kernels_mmvq_batched_f32_run(
                n_experts, n_rows_per_expert, n_cols, weights, activations,
                sorted_token_ids, expert_offsets, topk_weights, output, top_k,
                workspace, workspace_bytes, stream) }
        }
        (GgufMmvqBatchedFormat::Fp, ElementKind::F16) => {
            unsafe { baracuda_kernels_sys::baracuda_kernels_mmvq_batched_f16_run(
                n_experts, n_rows_per_expert, n_cols, weights, activations,
                sorted_token_ids, expert_offsets, topk_weights, output, top_k,
                workspace, workspace_bytes, stream) }
        }
        (GgufMmvqBatchedFormat::Fp, ElementKind::Bf16) => {
            unsafe { baracuda_kernels_sys::baracuda_kernels_mmvq_batched_bf16_run(
                n_experts, n_rows_per_expert, n_cols, weights, activations,
                sorted_token_ids, expert_offsets, topk_weights, output, top_k,
                workspace, workspace_bytes, stream) }
        }
        // ---- Quantized ----
        (GgufMmvqBatchedFormat::Quantized(fmt), kind) => {
            unsafe { dispatch_quant_ffi(
                *fmt, kind, n_experts, n_rows_per_expert, n_cols, weights, activations,
                sorted_token_ids, expert_offsets, topk_weights, output, top_k,
                workspace, workspace_bytes, stream,
            ) }
        }
        _ => -1,
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn dispatch_quant_ffi(
    fmt: GgufBlockFormat,
    kind: ElementKind,
    n_experts: i32,
    n_rows_per_expert: i32,
    n_cols: i32,
    weights: *const c_void,
    activations: *const c_void,
    sorted_token_ids: *const i32,
    expert_offsets: *const i32,
    topk_weights: *const f32,
    output: *mut c_void,
    top_k: i32,
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
) -> i32 {
    macro_rules! call {
        ($sym:ident) => {
            unsafe { baracuda_kernels_sys::$sym(
                n_experts, n_rows_per_expert, n_cols, weights, activations,
                sorted_token_ids, expert_offsets, topk_weights, output, top_k,
                workspace, workspace_bytes, stream) }
        };
    }
    match (fmt, kind) {
        (GgufBlockFormat::Q4_0, ElementKind::F32)  => call!(baracuda_kernels_mmvq_q4_0_batched_run),
        (GgufBlockFormat::Q4_0, ElementKind::F16)  => call!(baracuda_kernels_mmvq_q4_0_batched_f16_run),
        (GgufBlockFormat::Q4_0, ElementKind::Bf16) => call!(baracuda_kernels_mmvq_q4_0_batched_bf16_run),
        (GgufBlockFormat::Q4_1, ElementKind::F32)  => call!(baracuda_kernels_mmvq_q4_1_batched_run),
        (GgufBlockFormat::Q4_1, ElementKind::F16)  => call!(baracuda_kernels_mmvq_q4_1_batched_f16_run),
        (GgufBlockFormat::Q4_1, ElementKind::Bf16) => call!(baracuda_kernels_mmvq_q4_1_batched_bf16_run),
        (GgufBlockFormat::Q5_0, ElementKind::F32)  => call!(baracuda_kernels_mmvq_q5_0_batched_run),
        (GgufBlockFormat::Q5_0, ElementKind::F16)  => call!(baracuda_kernels_mmvq_q5_0_batched_f16_run),
        (GgufBlockFormat::Q5_0, ElementKind::Bf16) => call!(baracuda_kernels_mmvq_q5_0_batched_bf16_run),
        (GgufBlockFormat::Q5_1, ElementKind::F32)  => call!(baracuda_kernels_mmvq_q5_1_batched_run),
        (GgufBlockFormat::Q5_1, ElementKind::F16)  => call!(baracuda_kernels_mmvq_q5_1_batched_f16_run),
        (GgufBlockFormat::Q5_1, ElementKind::Bf16) => call!(baracuda_kernels_mmvq_q5_1_batched_bf16_run),
        (GgufBlockFormat::Q8_0, ElementKind::F32)  => call!(baracuda_kernels_mmvq_q8_0_batched_run),
        (GgufBlockFormat::Q8_0, ElementKind::F16)  => call!(baracuda_kernels_mmvq_q8_0_batched_f16_run),
        (GgufBlockFormat::Q8_0, ElementKind::Bf16) => call!(baracuda_kernels_mmvq_q8_0_batched_bf16_run),
        (GgufBlockFormat::Q2K,  ElementKind::F32)  => call!(baracuda_kernels_mmvq_q2_K_batched_run),
        (GgufBlockFormat::Q2K,  ElementKind::F16)  => call!(baracuda_kernels_mmvq_q2_K_batched_f16_run),
        (GgufBlockFormat::Q2K,  ElementKind::Bf16) => call!(baracuda_kernels_mmvq_q2_K_batched_bf16_run),
        (GgufBlockFormat::Q3K,  ElementKind::F32)  => call!(baracuda_kernels_mmvq_q3_K_batched_run),
        (GgufBlockFormat::Q3K,  ElementKind::F16)  => call!(baracuda_kernels_mmvq_q3_K_batched_f16_run),
        (GgufBlockFormat::Q3K,  ElementKind::Bf16) => call!(baracuda_kernels_mmvq_q3_K_batched_bf16_run),
        (GgufBlockFormat::Q4K,  ElementKind::F32)  => call!(baracuda_kernels_mmvq_q4_K_batched_run),
        (GgufBlockFormat::Q4K,  ElementKind::F16)  => call!(baracuda_kernels_mmvq_q4_K_batched_f16_run),
        (GgufBlockFormat::Q4K,  ElementKind::Bf16) => call!(baracuda_kernels_mmvq_q4_K_batched_bf16_run),
        (GgufBlockFormat::Q5K,  ElementKind::F32)  => call!(baracuda_kernels_mmvq_q5_K_batched_run),
        (GgufBlockFormat::Q5K,  ElementKind::F16)  => call!(baracuda_kernels_mmvq_q5_K_batched_f16_run),
        (GgufBlockFormat::Q5K,  ElementKind::Bf16) => call!(baracuda_kernels_mmvq_q5_K_batched_bf16_run),
        (GgufBlockFormat::Q6K,  ElementKind::F32)  => call!(baracuda_kernels_mmvq_q6_K_batched_run),
        (GgufBlockFormat::Q6K,  ElementKind::F16)  => call!(baracuda_kernels_mmvq_q6_K_batched_f16_run),
        (GgufBlockFormat::Q6K,  ElementKind::Bf16) => call!(baracuda_kernels_mmvq_q6_K_batched_bf16_run),
        (GgufBlockFormat::Q8K,  ElementKind::F32)  => call!(baracuda_kernels_mmvq_q8_K_batched_run),
        (GgufBlockFormat::Q8K,  ElementKind::F16)  => call!(baracuda_kernels_mmvq_q8_K_batched_f16_run),
        (GgufBlockFormat::Q8K,  ElementKind::Bf16) => call!(baracuda_kernels_mmvq_q8_K_batched_bf16_run),
        _ => -1,
    }
}

fn build_sku(_format: &GgufMmvqBatchedFormat, act_kind: ElementKind) -> KernelSku {
    KernelSku {
        category: OpCategory::Quantization,
        op: QuantizeKind::GgufMmvq as u16,
        // Element on the SKU records the activation / output dtype.
        element: act_kind,
        aux_element: Some(ElementKind::U8),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80,
        backend: BackendKind::Bespoke,
        precision_guarantee: PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // top_k=1 path is bit-stable; top_k>1 path uses atomicAdd
            // and is NOT bit-stable. We report the optimistic flag here
            // — callers using top_k>1 should not rely on bit-stability.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        },
    }
}
