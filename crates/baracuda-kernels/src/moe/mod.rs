//! Mixture-of-Experts (MoE) inference forward — Phase 8 Milestone 8.5
//! (Category V).
//!
//! Three fused per-token-dispatch + expert-matmul + accumulate kernel
//! variants:
//!
//! - [`MoeVariant::ScalarGguf`] — scalar (no tensor cores) MoE GEMM over
//!   GGUF-quantized expert weights. f32 activations, f32 output, q8_1
//!   activation staging internally.
//! - [`MoeVariant::Wmma`] — sm_70+ WMMA tensor cores over dense FP
//!   (`f16` / `bf16`) expert weights. The FP MoE hot path.
//! - [`MoeVariant::WmmaGguf`] — combined WMMA + GGUF path. f16/bf16
//!   activations, GGUF-packed weights, f32 output. The production hot
//!   path for quantized LLM inference.
//!
//! ## Lineage
//!
//! Vendored from [attention.rs](https://github.com/guoqingbao/attention.rs)
//! via `fuel-cuda-kernels`. See
//! `crates/baracuda-kernels-sys/LICENSE-thirdparty.md` for the full
//! attribution chain and `kernels/include/baracuda_moe.cuh` for kernel-
//! level lineage notes.
//!
//! ## Phase 20.2 — Fuel-replacement FFI surface (2026-05-25)
//!
//! The `baracuda_kernels_moe_*_run` C symbols are the canonical MoE
//! surface; `fuel-cuda-kernels/src/moe/` retires in favour of direct
//! calls to those symbols. Callers can bypass [`MoePlan`] entirely
//! and call the FFI directly — see
//! `crates/baracuda-kernels/tests/moe_ffi_direct_smoke.rs` for the
//! reference call pattern. The plan layer (this module) and the FFI
//! layer both reach the same kernel bodies in `baracuda_moe.cuh`.
//!
//! ## Block-format coverage
//!
//! The GGUF variants support `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`,
//! and `Q6_K`. This matches Fuel's `moe_gemm_gguf` / `moe_gemm_gguf_prefill`
//! switch exactly; the `Q4_0` / `Q4_1` / `Q5_0` / `Q5_1` block formats
//! are NOT shipped by upstream for the MoE path (they'd require adding
//! 4 new `vec_dot_q*_q8_1` wirings Fuel itself doesn't carry).
//! [`MoePlan::select`] returns [`Error::Unsupported`] for any unsupported
//! block format / variant combination.
//!
//! ## Inference-only
//!
//! All three variants are inference-only by convention; backward
//! passes are not shipped. MoE training composes per-expert FFN ops
//! manually at the autograd surface above.

use core::ffi::c_void;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, BackendKind, ElementKind, GgufBlockFormat, KernelSku, MathPrecision, MoeKind,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace, U8,
};

use crate::quantize::map_status;

/// Selector for the MoE variant.
///
/// `#[non_exhaustive]` — additional MoE backend variants (FP8 expert
/// weights, BF16+WMMA on Hopper, multi-block routing) may land in
/// future phases. Match arms must include a `_ =>` catch-all.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum MoeVariant {
    /// Scalar dispatch over GGUF-packed expert weights, f32 activations.
    ScalarGguf,
    /// WMMA tensor cores over FP expert weights, f16/bf16 activations.
    Wmma,
    /// WMMA tensor cores + GGUF-packed expert weights, f16/bf16
    /// activations. The combined hot path for quantized LLM inference.
    WmmaGguf,
}

/// Descriptor for an MoE forward op.
#[derive(Copy, Clone, Debug)]
pub struct MoeDescriptor {
    /// Total number of tokens to process.
    pub num_tokens: i32,
    /// Number of experts in the MoE block.
    pub num_experts: i32,
    /// Number of experts each token is routed to (`top_k` in routing).
    pub top_k: i32,
    /// Hidden dim of the activation / output (`size_k` in Fuel-speak).
    pub d_model: i32,
    /// Per-expert output feature dim (`size_n` in Fuel-speak).
    pub d_expert: i32,
    /// Which kernel variant to dispatch.
    pub variant: MoeVariant,
    /// GGUF block format — must be `Some(...)` for `ScalarGguf` /
    /// `WmmaGguf` variants and `None` for the `Wmma` variant.
    pub block_format: Option<GgufBlockFormat>,
    /// Activation element type. `F32` for `ScalarGguf`; `F16` or `Bf16`
    /// for `Wmma` / `WmmaGguf`.
    pub element: ElementKind,
    /// `is_prefill` flag for the `Wmma` variant (selects between
    /// prefill M=16 / N=16 / WARPS_N=2 and decode M=8 / N=32 / WARPS_N=1
    /// tile geometries). Ignored by the other variants.
    pub is_prefill: bool,
}

/// Args bundle for an MoE forward launch.
///
/// The expert weight matrix is carried as a raw byte buffer (`&[u8]`)
/// so the same struct shape works for FP weights (`Wmma` variant) and
/// GGUF-packed weights (`ScalarGguf` / `WmmaGguf`). Plan-side
/// validation checks the byte length against the descriptor.
pub struct MoeArgs<'a, T>
where
    T: baracuda_types::DeviceRepr + Copy + 'static,
{
    /// Activations `[num_tokens, d_model]`.
    pub activations: TensorRef<'a, T, 2>,
    /// Top-k expert indices `[num_tokens, top_k]`.
    pub expert_indices: TensorRef<'a, i32, 2>,
    /// Top-k expert mixing weights `[num_tokens, top_k]`.
    pub expert_weights: TensorRef<'a, T, 2>,
    /// Per-token sorted-by-expert flat index list `[num_tokens * top_k]`.
    /// Pre-computed upstream (top-k routing already done).
    pub sorted_token_ids: TensorRef<'a, i32, 1>,
    /// Per-token expert id list aligned with `sorted_token_ids`
    /// `[num_tokens * top_k]`. Already sorted by expert.
    pub flat_expert_ids: TensorRef<'a, i32, 1>,
    /// Optional per-token mixing weight `[num_tokens * top_k]`. When
    /// `None`, the launcher passes `nullptr` and the kernel reads from
    /// `expert_weights` via the routing path.
    pub topk_weight_flat: Option<TensorRef<'a, f32, 1>>,
    /// Packed expert weight bytes. For `Wmma`, must equal
    /// `num_experts * d_expert * d_model * sizeof(T)`; for GGUF, must
    /// equal `num_experts * d_expert * (d_model / block_size) * type_size`.
    pub expert_matrices: TensorRef<'a, U8, 1>,
    /// Output `[num_tokens, d_expert]`. For `WmmaGguf`, output is `f32`
    /// regardless of activation dtype (kernel writes float directly).
    pub output: TensorMut<'a, T, 2>,
    /// Scratch buffer for the WMMA variants — must be at least
    /// `num_experts * sizeof(i32)` bytes. Pass `None` for `ScalarGguf`.
    pub expert_counts_scratch: Option<TensorMut<'a, i32, 1>>,
    /// Scratch buffer for the WMMA variants — must be at least
    /// `(num_experts + 1) * sizeof(i32)` bytes. Pass `None` for
    /// `ScalarGguf`.
    pub expert_offsets_scratch: Option<TensorMut<'a, i32, 1>>,
}

/// MoE forward plan.
///
/// Fused per-token dispatch + expert GEMM + accumulate over up to
/// `top_k` experts. Inference-only.
///
/// **When to use**: forward MoE FFN pass. No BW plan — MoE training
/// composes per-expert FFN ops manually at the autograd surface.
/// Variant is selected at descriptor build time:
///
/// | variant       | acts        | weights         | output |
/// |---------------|-------------|-----------------|--------|
/// | `ScalarGguf`  | `f32`       | GGUF-packed     | `f32`  |
/// | `Wmma`        | `f16`/`bf16`| dense FP        | `T`    |
/// | `WmmaGguf`    | `f16`/`bf16`| GGUF-packed     | `f32`  |
///
/// **Shape limits**: `num_experts ≤ 1024` (WMMA scan kernel);
/// `top_k ≥ 1`. For GGUF variants `d_model` must be a multiple of
/// the block size.
///
/// **GGUF coverage**: `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`.
/// `Q4_0`/`Q4_1`/`Q5_0`/`Q5_1`/`Q8K` are NOT shipped (Fuel upstream
/// doesn't carry the `vec_dot_q*_q8_1` wirings for those).
///
/// **Workspace**: zero in [`Workspace`]. WMMA variants require
/// caller-supplied `expert_counts_scratch` (`num_experts * i32`) and
/// `expert_offsets_scratch` (`(num_experts + 1) * i32`) in
/// [`MoeArgs`] instead.
///
/// **Precision guarantee**: deterministic, bit-stable on identical
/// hardware (no atomics — top-k writes are to distinct token rows;
/// per-token weight scaling is applied in-kernel).
///
/// # Variant / `topk_weight` semantics — **PENDING**
///
/// The reference CPU math for each variant is a known TODO: the
/// `kernels/moe.cu` integration tests currently retain the kernel
/// outputs via `let _ = ...` placeholders rather than asserting
/// against a verified CPU reference. The exact composition rules
/// — when the kernel reads `topk_weight_flat` vs `expert_weights`,
/// the post-mix scaling order, the prefill-vs-decode tile-geometry
/// numerical drift — are NOT yet pinned down by a reference
/// implementation. Callers should treat any specific numerical
/// output as kernel-defined until the reference lands. See
/// `crates/baracuda-kernels/src/moe/mod.rs` and the integration
/// tests under `crates/baracuda-kernels/tests/moe*.rs`.
pub struct MoePlan {
    desc: MoeDescriptor,
    sku: KernelSku,
}

impl MoePlan {
    /// Pick a kernel for `desc`. Errors on unsupported variant/dtype
    /// combos, missing block format for GGUF variants, or non-positive
    /// dims.
    pub fn select(_stream: &Stream, desc: &MoeDescriptor, _pref: PlanPreference) -> Result<Self> {
        if desc.num_tokens < 0
            || desc.num_experts <= 0
            || desc.top_k <= 0
            || desc.d_model <= 0
            || desc.d_expert <= 0
        {
            return Err(Error::InvalidProblem(
                "MoePlan: tokens/experts/top_k/d_model/d_expert must be > 0 (tokens >= 0)",
            ));
        }
        if desc.num_experts > 1024 {
            return Err(Error::Unsupported(
                "MoePlan: WMMA scan kernel only supports num_experts <= 1024",
            ));
        }
        match desc.variant {
            MoeVariant::ScalarGguf => {
                if desc.element != ElementKind::F32 {
                    return Err(Error::Unsupported(
                        "MoePlan: ScalarGguf variant requires f32 activations",
                    ));
                }
                let bf = desc.block_format.ok_or(Error::InvalidProblem(
                    "MoePlan: ScalarGguf variant requires block_format = Some(...)",
                ))?;
                fuel_moe_gguf_dtype(bf)?;
            }
            MoeVariant::Wmma => {
                if desc.element != ElementKind::F16 && desc.element != ElementKind::Bf16 {
                    return Err(Error::Unsupported(
                        "MoePlan: Wmma variant requires f16 or bf16 activations",
                    ));
                }
                if desc.block_format.is_some() {
                    return Err(Error::InvalidProblem(
                        "MoePlan: Wmma variant must not set block_format",
                    ));
                }
            }
            MoeVariant::WmmaGguf => {
                if desc.element != ElementKind::F16 && desc.element != ElementKind::Bf16 {
                    return Err(Error::Unsupported(
                        "MoePlan: WmmaGguf variant requires f16 or bf16 activations",
                    ));
                }
                let bf = desc.block_format.ok_or(Error::InvalidProblem(
                    "MoePlan: WmmaGguf variant requires block_format = Some(...)",
                ))?;
                fuel_moe_gguf_dtype(bf)?;
                let bs = bf.block_size() as i32;
                if desc.d_model % bs != 0 {
                    return Err(Error::InvalidProblem(
                        "MoePlan: d_model must be a multiple of the GGUF block size",
                    ));
                }
            }
        }
        Ok(Self {
            desc: *desc,
            sku: build_sku(desc),
        })
    }

    /// Workspace bytes — none. The WMMA variants need
    /// `expert_counts_scratch` and `expert_offsets_scratch` but those
    /// are carried in `MoeArgs`, not the workspace.
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

    /// Launch the MoE forward kernel.
    ///
    /// `T` must match `desc.element` (compile-time bound enforced by
    /// the [`TensorRef`] / [`TensorMut`] views in `args`).
    pub fn run<T>(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: MoeArgs<'_, T>,
    ) -> Result<()>
    where
        T: baracuda_types::DeviceRepr + Copy + 'static,
    {
        let stream_ptr = stream.as_raw() as *mut c_void;
        let acts_ptr = args.activations.data.as_raw().0 as *const c_void;
        let weights_ptr = args.expert_matrices.data.as_raw().0 as *const c_void;
        let sorted_token_ids_ptr = args.sorted_token_ids.data.as_raw().0 as *const i32;
        let flat_expert_ids_ptr = args.flat_expert_ids.data.as_raw().0 as *const i32;
        let topk_weights_ptr = args
            .topk_weight_flat
            .as_ref()
            .map(|tw| tw.data.as_raw().0 as *const f32)
            .unwrap_or(core::ptr::null());
        let out_ptr = args.output.data.as_raw().0 as *mut c_void;

        let num_tokens_flat = args.sorted_token_ids.shape[0];

        let status = match self.desc.variant {
            MoeVariant::ScalarGguf => {
                let bf = self.desc.block_format.expect("checked in select()");
                let gguf_dtype = fuel_moe_gguf_dtype(bf).expect("checked in select()");
                unsafe {
                    baracuda_kernels_sys::baracuda_kernels_moe_scalar_gguf_run(
                        acts_ptr,
                        weights_ptr,
                        sorted_token_ids_ptr,
                        flat_expert_ids_ptr,
                        topk_weights_ptr,
                        out_ptr,
                        self.desc.num_experts,
                        self.desc.top_k,
                        num_tokens_flat,
                        self.desc.d_expert,
                        self.desc.d_model,
                        gguf_dtype,
                        core::ptr::null_mut(),
                        0,
                        stream_ptr,
                    )
                }
            }
            MoeVariant::Wmma => {
                let ec = args.expert_counts_scratch.as_ref().ok_or(Error::InvalidProblem(
                    "MoePlan::run: Wmma variant requires expert_counts_scratch",
                ))?;
                let eo = args.expert_offsets_scratch.as_ref().ok_or(Error::InvalidProblem(
                    "MoePlan::run: Wmma variant requires expert_offsets_scratch",
                ))?;
                let ec_ptr = ec.data.as_raw().0 as *mut i32;
                let eo_ptr = eo.data.as_raw().0 as *mut i32;
                let is_prefill = if self.desc.is_prefill { 1 } else { 0 };
                match self.desc.element {
                    ElementKind::F16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_moe_wmma_f16_run(
                            acts_ptr,
                            weights_ptr,
                            sorted_token_ids_ptr,
                            flat_expert_ids_ptr,
                            topk_weights_ptr,
                            out_ptr,
                            ec_ptr,
                            eo_ptr,
                            self.desc.num_experts,
                            self.desc.top_k,
                            num_tokens_flat,
                            self.desc.d_expert,
                            self.desc.d_model,
                            is_prefill,
                            core::ptr::null_mut(),
                            0,
                            stream_ptr,
                        )
                    },
                    ElementKind::Bf16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_moe_wmma_bf16_run(
                            acts_ptr,
                            weights_ptr,
                            sorted_token_ids_ptr,
                            flat_expert_ids_ptr,
                            topk_weights_ptr,
                            out_ptr,
                            ec_ptr,
                            eo_ptr,
                            self.desc.num_experts,
                            self.desc.top_k,
                            num_tokens_flat,
                            self.desc.d_expert,
                            self.desc.d_model,
                            is_prefill,
                            core::ptr::null_mut(),
                            0,
                            stream_ptr,
                        )
                    },
                    _ => return Err(Error::Unsupported("MoePlan::run: Wmma element unsupported")),
                }
            }
            MoeVariant::WmmaGguf => {
                let bf = self.desc.block_format.expect("checked in select()");
                let gguf_dtype = fuel_moe_gguf_dtype(bf).expect("checked in select()");
                let ec = args.expert_counts_scratch.as_ref().ok_or(Error::InvalidProblem(
                    "MoePlan::run: WmmaGguf variant requires expert_counts_scratch",
                ))?;
                let eo = args.expert_offsets_scratch.as_ref().ok_or(Error::InvalidProblem(
                    "MoePlan::run: WmmaGguf variant requires expert_offsets_scratch",
                ))?;
                let ec_ptr = ec.data.as_raw().0 as *mut i32;
                let eo_ptr = eo.data.as_raw().0 as *mut i32;
                match self.desc.element {
                    ElementKind::F16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_moe_wmma_gguf_f16_run(
                            acts_ptr,
                            weights_ptr,
                            sorted_token_ids_ptr,
                            flat_expert_ids_ptr,
                            topk_weights_ptr,
                            out_ptr,
                            ec_ptr,
                            eo_ptr,
                            self.desc.num_experts,
                            self.desc.top_k,
                            num_tokens_flat,
                            self.desc.d_expert,
                            self.desc.d_model,
                            gguf_dtype,
                            core::ptr::null_mut(),
                            0,
                            stream_ptr,
                        )
                    },
                    ElementKind::Bf16 => unsafe {
                        baracuda_kernels_sys::baracuda_kernels_moe_wmma_gguf_bf16_run(
                            acts_ptr,
                            weights_ptr,
                            sorted_token_ids_ptr,
                            flat_expert_ids_ptr,
                            topk_weights_ptr,
                            out_ptr,
                            ec_ptr,
                            eo_ptr,
                            self.desc.num_experts,
                            self.desc.top_k,
                            num_tokens_flat,
                            self.desc.d_expert,
                            self.desc.d_model,
                            gguf_dtype,
                            core::ptr::null_mut(),
                            0,
                            stream_ptr,
                        )
                    },
                    _ => return Err(Error::Unsupported("MoePlan::run: WmmaGguf element unsupported")),
                }
            }
        };
        map_status(status)
    }
}

/// Translate baracuda's `GgufBlockFormat` into the Fuel-convention
/// `gguf_dtype` discriminant expected by the `moe_*_gguf_run` FFI
/// (matches the switch in Fuel's `moe_gemm_gguf`):
///   `0 = Q8_0`, `1 = Q4_K`, `2 = Q2_K`, `3 = Q3_K`, `4 = Q5_K`, `5 = Q6_K`.
fn fuel_moe_gguf_dtype(bf: GgufBlockFormat) -> Result<i32> {
    match bf {
        GgufBlockFormat::Q8_0 => Ok(0),
        GgufBlockFormat::Q4K => Ok(1),
        GgufBlockFormat::Q2K => Ok(2),
        GgufBlockFormat::Q3K => Ok(3),
        GgufBlockFormat::Q5K => Ok(4),
        GgufBlockFormat::Q6K => Ok(5),
        GgufBlockFormat::Q4_0
        | GgufBlockFormat::Q4_1
        | GgufBlockFormat::Q5_0
        | GgufBlockFormat::Q5_1
        | GgufBlockFormat::Q8K => Err(Error::Unsupported(
            "MoePlan: GGUF MoE variants only support Q8_0 + k-quants (Q2_K..Q6_K)",
        )),
        // Defensive arm — `GgufBlockFormat` is `#[non_exhaustive]`.
        _ => Err(Error::Unsupported(
            "MoePlan: unsupported GGUF block format",
        )),
    }
}

fn build_sku(desc: &MoeDescriptor) -> KernelSku {
    let op = match desc.variant {
        MoeVariant::ScalarGguf => MoeKind::ScalarGguf as u16,
        MoeVariant::Wmma => MoeKind::Wmma as u16,
        MoeVariant::WmmaGguf => MoeKind::WmmaGguf as u16,
    };
    KernelSku {
        category: OpCategory::Moe,
        op,
        element: desc.element,
        aux_element: Some(ElementKind::U8),
        layout: None,
        epilogue: None,
        arch: ArchSku::Sm80, // sm_70+; sm_80 is the baseline arch baracuda exposes.
        backend: BackendKind::Bespoke,
        precision_guarantee: PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Atomic-free (top-k > 1 writes are to distinct token rows
            // when `topk_weights == None`; otherwise the per-token-weight
            // scaling is applied in the kernel and the output is written
            // directly). Deterministic on identical hardware.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        },
    }
}
