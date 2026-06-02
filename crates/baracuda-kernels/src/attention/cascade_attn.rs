//! Cascade attention LSE merge — Phase 46 (FlashInfer cherry-pick).
//!
//! Cascade attention is the building block for shared-prefix serving
//! (system prompts, RAG context reuse): the prefix tokens are attended
//! over ONCE per batch, producing partial attention state `(v_prefix,
//! lse_prefix)`. Each request then attends to its unique-suffix K/V,
//! producing per-request state `(v_suffix, lse_suffix)`. The two states
//! are MERGED — that's the op this plan implements.
//!
//! The merge formula (FlashInfer's base-2 LSE convention):
//!
//! ```text
//! m'       = max(s_prefix, s_suffix)
//! w_prefix = 2^(s_prefix - m')
//! w_suffix = 2^(s_suffix - m')
//! v_out    = (w_prefix * v_prefix + w_suffix * v_suffix) / (w_prefix + w_suffix)
//! s_out    = m' + log2(w_prefix + w_suffix)
//! ```
//!
//! All `s_*` are log-sum-exp in **base 2** (matches FlashInfer's
//! internal convention; bespoke baracuda LSE is base-e and needs
//! `* 1/ln(2)` before being passed to this plan).
//!
//! ## Variants
//!
//! - [`Self::merge_in_place`] — pairwise in-place merge:
//!   `(v, s) <- merge((v, s), (v_other, s_other))`. Fastest for the
//!   common "merge prefix into per-request output" case.
//! - Many-way merge via FlashInfer's `MergeStates` (cascade depth > 2)
//!   is wired through a separate plan-method in a follow-up tier —
//!   not in the Phase 46 Tier-1 surface.
//!
//! ## Constraints
//!
//! - `head_dim ∈ {64, 128, 256}`.
//! - Element type `T ∈ {f16, bf16, f32}` for `v`. LSE always `f32`.
//! - Contiguous `[seq_len, num_heads, head_dim]` for `v`,
//!   `[seq_len, num_heads]` for `s`.
//! - No mask (pairwise merge always merges; pass equal LSE values
//!   if you want to weight one side out).

use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};


/// Descriptor for a cascade-attention LSE merge.
#[derive(Copy, Clone, Debug)]
pub struct CascadeAttentionDescriptor {
    /// Sequence length (per-request output query length; typically 1
    /// for decode, `q_len` for prefill).
    pub seq_len: i32,
    /// Number of heads.
    pub num_heads: i32,
    /// Per-head dimension. Must be 64, 128, or 256.
    pub head_dim: i32,
    /// Element type of `v`.
    pub element: ElementKind,
}

/// Args bundle for a cascade in-place pairwise merge launch.
///
/// `v` / `s` are merged in place; `v_other` / `s_other` are read-only
/// inputs.
pub struct CascadeAttentionArgs<'a, T: Element> {
    /// Merged `v` — `[seq_len, num_heads, head_dim]`, in-place.
    pub v: TensorMut<'a, T, 3>,
    /// Merged `s` (base-2 LSE) — `[seq_len, num_heads]` f32, in-place.
    pub s: TensorMut<'a, f32, 2>,
    /// Other-side `v` — `[seq_len, num_heads, head_dim]`.
    pub v_other: TensorRef<'a, T, 3>,
    /// Other-side `s` (base-2 LSE) — `[seq_len, num_heads]` f32.
    pub s_other: TensorRef<'a, f32, 2>,
}

/// Cascade-attention LSE-merge plan.
///
/// Routes to FlashInfer's `MergeStateInPlace<DType>`.
/// Requires the `flashinfer` cargo feature.
///
/// **Workspace**: zero (pure in-place merge).
///
/// **Precision guarantee**: deterministic on same-hardware repeat.
/// f16/bf16 accumulate in f32 internally.
pub struct CascadeAttentionPlan<T: Element> {
    desc: CascadeAttentionDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> CascadeAttentionPlan<T> {
    /// Pick a kernel + validate head_dim + element gates.
    pub fn select(
        _stream: &Stream,
        desc: &CascadeAttentionDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "CascadeAttentionPlan: descriptor element != T",
            ));
        }
        if desc.seq_len <= 0 || desc.num_heads <= 0 {
            return Err(Error::InvalidProblem(
                "CascadeAttentionPlan: extents must be positive",
            ));
        }
        if !matches!(desc.head_dim, 64 | 128 | 256) {
            return Err(Error::Unsupported(
                "CascadeAttentionPlan: head_dim must be 64, 128, or 256",
            ));
        }
        if !matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16 | ElementKind::F32) {
            return Err(Error::Unsupported(
                "CascadeAttentionPlan: element type must be f16, bf16, or f32",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            // Re-uses the PagedAttention discriminant slot — cascade is
            // a cascade-attention-specific helper, no dedicated
            // AttentionKind variant. Considered acceptable because
            // KernelSku's discriminant is for telemetry, not dispatch.
            op: AttentionKind::PagedAttention as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::FlashInfer,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args against the descriptor (shape + contiguity checks).
    pub fn can_implement(&self, args: &CascadeAttentionArgs<'_, T>) -> Result<()> {
        let v_shape = [self.desc.seq_len, self.desc.num_heads, self.desc.head_dim];
        let s_shape = [self.desc.seq_len, self.desc.num_heads];
        if args.v.shape != v_shape || args.v_other.shape != v_shape {
            return Err(Error::InvalidProblem(
                "CascadeAttentionPlan: v / v_other shape mismatch",
            ));
        }
        if args.s.shape != s_shape || args.s_other.shape != s_shape {
            return Err(Error::InvalidProblem(
                "CascadeAttentionPlan: s / s_other shape mismatch",
            ));
        }
        if !args.v.is_contiguous()
            || !args.s.is_contiguous()
            || !args.v_other.is_contiguous()
            || !args.s_other.is_contiguous()
        {
            return Err(Error::Unsupported(
                "CascadeAttentionPlan: tensors must be contiguous (Tier 1)",
            ));
        }
        Ok(())
    }

    /// Required workspace bytes (always 0 — pure in-place merge).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// SKU identity (telemetry / autotuner key).
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees of this plan.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Pairwise in-place merge.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: CascadeAttentionArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = (stream, &args);
            Err(Error::Unsupported(
                "CascadeAttentionPlan: `flashinfer` cargo feature is not enabled",
            ))
        }
        #[cfg(feature = "flashinfer")]
        {
            let stream_ptr = stream.as_raw() as *mut c_void;
            let v_ptr = args.v.data.as_raw().0 as *mut c_void;
            let s_ptr = args.s.data.as_raw().0 as *mut c_void;
            let v_other_ptr = args.v_other.data.as_raw().0 as *const c_void;
            let s_other_ptr = args.s_other.data.as_raw().0 as *const c_void;

            let status = match T::KIND {
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_merge_state_in_place_f16_run(
                        self.desc.seq_len, self.desc.num_heads, self.desc.head_dim,
                        v_ptr, s_ptr, v_other_ptr, s_other_ptr, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_merge_state_in_place_bf16_run(
                        self.desc.seq_len, self.desc.num_heads, self.desc.head_dim,
                        v_ptr, s_ptr, v_other_ptr, s_other_ptr, stream_ptr,
                    )
                },
                ElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_merge_state_in_place_f32_run(
                        self.desc.seq_len, self.desc.num_heads, self.desc.head_dim,
                        v_ptr, s_ptr, v_other_ptr, s_other_ptr, stream_ptr,
                    )
                },
                _ => {
                    return Err(Error::Unsupported(
                        "CascadeAttentionPlan::run reached an unimplemented dtype",
                    ));
                }
            };
            map_status(status)
        }
    }
}

// =========================================================================
// Many-way merge (cascade depth > 2) — Phase 66.
// =========================================================================
//
// Phase 46 wired only the pairwise in-place merge above. This plan wraps
// FlashInfer's `MergeStates` many-way merge: given `num_index_sets`
// partial attention states stacked per `(seq_pos, head)` cell, collapse
// them to a single merged state in one launch. This is the building block
// for cascade depth > 2 — e.g. a multi-level shared-prefix tree (global
// system prompt → per-tenant prefix → per-request suffix) or fusing the
// partial states of several overlapping prefix caches / LoRA adapters.

/// Descriptor for a many-way cascade state merge.
#[derive(Copy, Clone, Debug)]
pub struct CascadeMergeStatesDescriptor {
    /// Number of partial states to merge per cell (cascade fan-in).
    pub num_index_sets: i32,
    /// Sequence length (per-request output query length).
    pub seq_len: i32,
    /// Number of heads.
    pub num_heads: i32,
    /// Per-head dimension. Must be 64, 128, or 256.
    pub head_dim: i32,
    /// Element type of `v` / `v_merged`.
    pub element: ElementKind,
}

/// Args bundle for a many-way cascade merge launch.
///
/// Inputs are stacked along the `num_index_sets` axis; outputs collapse
/// that axis away.
pub struct CascadeMergeStatesArgs<'a, T: Element> {
    /// Stacked partial `v` —
    /// `[seq_len, num_index_sets, num_heads, head_dim]`.
    pub v: TensorRef<'a, T, 4>,
    /// Stacked partial `s` (base-2 LSE) —
    /// `[seq_len, num_index_sets, num_heads]` f32.
    pub s: TensorRef<'a, f32, 3>,
    /// Merged `v` output — `[seq_len, num_heads, head_dim]`.
    pub v_merged: TensorMut<'a, T, 3>,
    /// Merged `s` (base-2 LSE) output — `[seq_len, num_heads]` f32.
    pub s_merged: TensorMut<'a, f32, 2>,
}

/// Many-way cascade-attention LSE-merge plan.
///
/// Routes to FlashInfer's `MergeStates<DType, DType>`. Requires the
/// `flashinfer` cargo feature.
///
/// **Workspace**: zero. **Precision**: f32 accumulation; deterministic on
/// same-hardware repeat.
pub struct CascadeMergeStatesPlan<T: Element> {
    desc: CascadeMergeStatesDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> CascadeMergeStatesPlan<T> {
    /// Pick a kernel + validate fan-in / head_dim / element gates.
    pub fn select(
        _stream: &Stream,
        desc: &CascadeMergeStatesDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "CascadeMergeStatesPlan: descriptor element != T",
            ));
        }
        if desc.num_index_sets <= 0 || desc.seq_len <= 0 || desc.num_heads <= 0 {
            return Err(Error::InvalidProblem(
                "CascadeMergeStatesPlan: extents must be positive",
            ));
        }
        if !matches!(desc.head_dim, 64 | 128 | 256) {
            return Err(Error::Unsupported(
                "CascadeMergeStatesPlan: head_dim must be 64, 128, or 256",
            ));
        }
        if !matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16 | ElementKind::F32) {
            return Err(Error::Unsupported(
                "CascadeMergeStatesPlan: element type must be f16, bf16, or f32",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::PagedAttention as u16,
            element: T::KIND,
            aux_element: None,
            layout: None,
            epilogue: None,
            arch: ArchSku::Sm80,
            backend: BackendKind::FlashInfer,
            precision_guarantee,
        };
        Ok(Self {
            desc: *desc,
            sku,
            _marker: PhantomData,
        })
    }

    /// Validate args against the descriptor (shape + contiguity checks).
    pub fn can_implement(&self, args: &CascadeMergeStatesArgs<'_, T>) -> Result<()> {
        let v_shape = [
            self.desc.seq_len,
            self.desc.num_index_sets,
            self.desc.num_heads,
            self.desc.head_dim,
        ];
        let s_shape = [self.desc.seq_len, self.desc.num_index_sets, self.desc.num_heads];
        let v_merged_shape = [self.desc.seq_len, self.desc.num_heads, self.desc.head_dim];
        let s_merged_shape = [self.desc.seq_len, self.desc.num_heads];
        if args.v.shape != v_shape {
            return Err(Error::InvalidProblem("CascadeMergeStatesPlan: v shape mismatch"));
        }
        if args.s.shape != s_shape {
            return Err(Error::InvalidProblem("CascadeMergeStatesPlan: s shape mismatch"));
        }
        if args.v_merged.shape != v_merged_shape {
            return Err(Error::InvalidProblem(
                "CascadeMergeStatesPlan: v_merged shape mismatch",
            ));
        }
        if args.s_merged.shape != s_merged_shape {
            return Err(Error::InvalidProblem(
                "CascadeMergeStatesPlan: s_merged shape mismatch",
            ));
        }
        if !args.v.is_contiguous()
            || !args.s.is_contiguous()
            || !args.v_merged.is_contiguous()
            || !args.s_merged.is_contiguous()
        {
            return Err(Error::Unsupported(
                "CascadeMergeStatesPlan: tensors must be contiguous (Tier 1)",
            ));
        }
        Ok(())
    }

    /// Required workspace bytes (always 0).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        0
    }

    /// SKU identity (telemetry / autotuner key).
    #[inline]
    pub fn sku(&self) -> KernelSku {
        self.sku
    }

    /// Numerical guarantees of this plan.
    #[inline]
    pub fn precision_guarantee(&self) -> PrecisionGuarantee {
        self.sku.precision_guarantee
    }

    /// Many-way merge of `num_index_sets` partial states into one.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: CascadeMergeStatesArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = (stream, &args);
            Err(Error::Unsupported(
                "CascadeMergeStatesPlan: `flashinfer` cargo feature is not enabled",
            ))
        }
        #[cfg(feature = "flashinfer")]
        {
            let stream_ptr = stream.as_raw() as *mut c_void;
            let v_ptr = args.v.data.as_raw().0 as *const c_void;
            let s_ptr = args.s.data.as_raw().0 as *const c_void;
            let v_merged_ptr = args.v_merged.data.as_raw().0 as *mut c_void;
            let s_merged_ptr = args.s_merged.data.as_raw().0 as *mut c_void;

            let status = match T::KIND {
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_merge_states_f16_run(
                        self.desc.num_index_sets, self.desc.seq_len, self.desc.num_heads,
                        self.desc.head_dim, v_ptr, s_ptr, v_merged_ptr, s_merged_ptr, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_merge_states_bf16_run(
                        self.desc.num_index_sets, self.desc.seq_len, self.desc.num_heads,
                        self.desc.head_dim, v_ptr, s_ptr, v_merged_ptr, s_merged_ptr, stream_ptr,
                    )
                },
                ElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_merge_states_f32_run(
                        self.desc.num_index_sets, self.desc.seq_len, self.desc.num_heads,
                        self.desc.head_dim, v_ptr, s_ptr, v_merged_ptr, s_merged_ptr, stream_ptr,
                    )
                },
                _ => {
                    return Err(Error::Unsupported(
                        "CascadeMergeStatesPlan::run reached an unimplemented dtype",
                    ));
                }
            };
            map_status(status)
        }
    }
}
