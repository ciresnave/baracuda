//! Batched paged-KV prefill — Phase 66 Tier 2 (FlashInfer).
//!
//! The prompt-ingestion counterpart to [`BatchPagedDecodePlan`]: every
//! request contributes *multiple* query rows (its prompt tokens), all
//! concatenated raggedly across the batch via `q_indptr`, each attending
//! over that request's paged K/V history. Routes to FlashInfer's
//! `BatchPrefillWithPagedKVCacheDispatched`.
//!
//! ## Layout
//!
//! Ragged query / output (rows concatenated across the batch):
//!
//! - `q`        : `[total_num_rows, num_qo_heads, head_dim]` element `T`.
//! - `q_indptr` : `[batch + 1]` i32 — request `b`'s rows are
//!   `q[q_indptr[b] .. q_indptr[b+1]]`.
//! - `o`        : `[total_num_rows, num_qo_heads, head_dim]` element `T`.
//! - `lse`      : `[total_num_rows, num_qo_heads]` **f32**.
//!
//! Paged store (kHND, shared with [`BatchPagedDecodePlan`]):
//!
//! - `k_data` / `v_data` : `[max_num_pages, num_kv_heads, page_size, head_dim]`.
//! - `kv_indices`        : `[total_used_pages]` i32 — physical page IDs.
//! - `kv_indptr`         : `[batch + 1]` i32 — PAGE prefix-sum.
//! - `last_page_len`     : `[batch]` i32 in `[1, page_size]`.
//!
//! ## Constraints (Phase 66 Tier 2, v1)
//!
//! - `head_dim ∈ {64, 128, 256}`.
//! - `num_qo_heads % num_kv_heads == 0` (integer GQA group).
//! - Element type `T ∈ {f16, bf16}` — prefill is tensor-core (mma) based;
//!   f32 Q/K is not supported by the kernel (use [`BatchPagedDecodePlan`]
//!   or the bespoke SDPA path for f32).
//! - `causal` selects standard autoregressive masking; `false` attends to
//!   the full history.
//! - No positional encoding inside the kernel (apply RoPE before
//!   populating the cache), no sliding window / soft-cap / ALiBi / custom
//!   mask, no KV-split, no CUDA Graph.
//!
//! ## Workspace + synchronization
//!
//! The plan reports `workspace_size() == 0`: the FlashInfer scheduler
//! workspaces are allocated *internally* per call. Because the host-side
//! plan needs host copies of the indptr arrays, **`run` is synchronous**
//! in v1 (it syncs the stream before returning). Prefill is not the
//! per-token hot path, so this is acceptable.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::batch_paged_decode::PagedKvCacheDescriptor;
use super::map_status;

/// Descriptor for a batched paged-KV prefill op.
#[derive(Copy, Clone, Debug)]
pub struct BatchPagedPrefillDescriptor {
    /// Number of requests in the batch.
    pub batch_size: i32,
    /// Total query rows summed across all requests (`q_indptr[batch]`).
    pub total_num_rows: i32,
    /// Query / output attention heads.
    pub num_qo_heads: i32,
    /// Score scaling factor — typically `1.0 / sqrt(head_dim)`.
    pub sm_scale: f32,
    /// Apply causal (autoregressive) masking.
    pub causal: bool,
    /// Opt into FlashInfer's KV-split parallelism (the scheduler may split
    /// long KV across CTAs; the kernel merges partial states internally).
    /// Costs a transient float workspace per call — enable for
    /// long-context / few-request prefill where the GPU would otherwise be
    /// under-utilized. `false` keeps the lighter no-split path.
    pub enable_kv_split: bool,
    /// Paged cache descriptor (shared between K and V).
    pub paged_kv: PagedKvCacheDescriptor,
}

/// Args bundle for a batched paged-KV prefill launch.
pub struct BatchPagedPrefillArgs<'a, T: Element> {
    /// Ragged query rows — `[total_num_rows, num_qo_heads, head_dim]`.
    pub q: TensorRef<'a, T, 3>,
    /// Query row prefix-sum `[batch + 1]` i32.
    pub q_indptr: TensorRef<'a, i32, 1>,
    /// Paged K cache — `[max_num_pages, num_kv_heads, page_size, head_dim]`.
    pub k_data: TensorRef<'a, T, 4>,
    /// Paged V cache — same layout as `k_data`.
    pub v_data: TensorRef<'a, T, 4>,
    /// Page indices `[total_used_pages]` i32.
    pub kv_indices: TensorRef<'a, i32, 1>,
    /// Page indptr `[batch + 1]` i32.
    pub kv_indptr: TensorRef<'a, i32, 1>,
    /// Last-page-len `[batch]` i32 (values in `[1, page_size]`).
    pub last_page_len: TensorRef<'a, i32, 1>,
    /// Output rows — `[total_num_rows, num_qo_heads, head_dim]`.
    pub o: TensorMut<'a, T, 3>,
    /// Per-row log-sum-exp — `[total_num_rows, num_qo_heads]` **f32**.
    pub lse: TensorMut<'a, f32, 2>,
}

/// Batched paged-KV prefill plan.
///
/// Routes to FlashInfer's `BatchPrefillWithPagedKVCacheDispatched`.
/// Requires the `flashinfer` cargo feature. f16 / bf16 only.
pub struct BatchPagedPrefillPlan<T: Element> {
    desc: BatchPagedPrefillDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> BatchPagedPrefillPlan<T> {
    /// Pick a kernel + validate shape limits.
    pub fn select(
        _stream: &Stream,
        desc: &BatchPagedPrefillDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.paged_kv.element != T::KIND {
            return Err(Error::Unsupported(
                "BatchPagedPrefillPlan: descriptor element != T",
            ));
        }
        if desc.batch_size <= 0
            || desc.total_num_rows <= 0
            || desc.num_qo_heads <= 0
            || desc.paged_kv.num_kv_heads <= 0
            || desc.paged_kv.page_size <= 0
            || desc.paged_kv.num_total_pages <= 0
        {
            return Err(Error::InvalidProblem(
                "BatchPagedPrefillPlan: extents must be positive",
            ));
        }
        if desc.num_qo_heads % desc.paged_kv.num_kv_heads != 0 {
            return Err(Error::InvalidProblem(
                "BatchPagedPrefillPlan: num_qo_heads must be a multiple of num_kv_heads",
            ));
        }
        if !matches!(desc.paged_kv.head_dim, 64 | 128 | 256) {
            return Err(Error::Unsupported(
                "BatchPagedPrefillPlan: head_dim must be 64, 128, or 256",
            ));
        }
        // Prefill is mma (tensor-core) based — f16/bf16 only.
        if !matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16) {
            return Err(Error::Unsupported(
                "BatchPagedPrefillPlan: element type must be f16 or bf16 (prefill is mma-based)",
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

    /// Validate args against the descriptor.
    pub fn can_implement(&self, args: &BatchPagedPrefillArgs<'_, T>) -> Result<()> {
        let d = &self.desc;
        let qo_shape = [d.total_num_rows, d.num_qo_heads, d.paged_kv.head_dim];
        if args.q.shape != qo_shape || args.o.shape != qo_shape {
            return Err(Error::InvalidProblem("BatchPagedPrefillPlan: q/o shape mismatch"));
        }
        if args.q_indptr.shape != [d.batch_size + 1] {
            return Err(Error::InvalidProblem(
                "BatchPagedPrefillPlan: q_indptr shape must be [batch + 1]",
            ));
        }
        let cache_shape = [
            d.paged_kv.num_total_pages,
            d.paged_kv.num_kv_heads,
            d.paged_kv.page_size,
            d.paged_kv.head_dim,
        ];
        if args.k_data.shape != cache_shape || args.v_data.shape != cache_shape {
            return Err(Error::InvalidProblem(
                "BatchPagedPrefillPlan: k_data/v_data shape mismatch",
            ));
        }
        if args.kv_indptr.shape != [d.batch_size + 1] {
            return Err(Error::InvalidProblem(
                "BatchPagedPrefillPlan: kv_indptr shape must be [batch + 1]",
            ));
        }
        if args.last_page_len.shape != [d.batch_size] {
            return Err(Error::InvalidProblem(
                "BatchPagedPrefillPlan: last_page_len shape must be [batch]",
            ));
        }
        if args.lse.shape != [d.total_num_rows, d.num_qo_heads] {
            return Err(Error::InvalidProblem(
                "BatchPagedPrefillPlan: lse shape must be [total_num_rows, num_qo_heads]",
            ));
        }
        if !args.q.is_contiguous()
            || !args.k_data.is_contiguous()
            || !args.v_data.is_contiguous()
            || !args.o.is_contiguous()
            || !args.lse.is_contiguous()
        {
            return Err(Error::Unsupported(
                "BatchPagedPrefillPlan: tensors must be contiguous (Tier 1)",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — zero; the scheduler workspace is internal.
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

    /// Launch the FlashInfer paged-KV prefill kernel. Synchronous in v1
    /// (see the module docs). Requires the `flashinfer` cargo feature.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: BatchPagedPrefillArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = (stream, &args);
            Err(Error::Unsupported(
                "BatchPagedPrefillPlan: `flashinfer` cargo feature is not enabled",
            ))
        }
        #[cfg(feature = "flashinfer")]
        {
            let d = &self.desc;
            let stream_ptr = stream.as_raw() as *mut c_void;
            let q_ptr = args.q.data.as_raw().0 as *const c_void;
            let q_indptr_ptr = args.q_indptr.data.as_raw().0 as *mut c_void;
            let k_ptr = args.k_data.data.as_raw().0 as *mut c_void;
            let v_ptr = args.v_data.data.as_raw().0 as *mut c_void;
            let kv_indices_ptr = args.kv_indices.data.as_raw().0 as *mut c_void;
            let kv_indptr_ptr = args.kv_indptr.data.as_raw().0 as *mut c_void;
            let last_page_len_ptr = args.last_page_len.data.as_raw().0 as *mut c_void;
            let o_ptr = args.o.data.as_raw().0 as *mut c_void;
            let lse_ptr = args.lse.data.as_raw().0 as *mut c_void;
            let causal = if d.causal { 1 } else { 0 };
            let enable_split = if d.enable_kv_split { 1 } else { 0 };

            let status = match T::KIND {
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_prefill_f16_run(
                        d.batch_size, d.total_num_rows, d.paged_kv.page_size, d.paged_kv.head_dim,
                        d.num_qo_heads, d.paged_kv.num_kv_heads, d.sm_scale, causal, enable_split,
                        k_ptr, v_ptr, kv_indices_ptr, kv_indptr_ptr, last_page_len_ptr,
                        q_ptr, q_indptr_ptr, o_ptr, lse_ptr, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_prefill_bf16_run(
                        d.batch_size, d.total_num_rows, d.paged_kv.page_size, d.paged_kv.head_dim,
                        d.num_qo_heads, d.paged_kv.num_kv_heads, d.sm_scale, causal, enable_split,
                        k_ptr, v_ptr, kv_indices_ptr, kv_indptr_ptr, last_page_len_ptr,
                        q_ptr, q_indptr_ptr, o_ptr, lse_ptr, stream_ptr,
                    )
                },
                _ => {
                    return Err(Error::Unsupported(
                        "BatchPagedPrefillPlan::run reached an unimplemented dtype",
                    ));
                }
            };
            map_status(status)
        }
    }
}
