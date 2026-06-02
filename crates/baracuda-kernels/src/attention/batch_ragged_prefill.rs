//! Batched ragged-KV prefill — Phase 66 Tier 2 (FlashInfer).
//!
//! Like [`BatchPagedPrefillPlan`](crate::BatchPagedPrefillPlan) but the
//! K/V history is stored CONTIGUOUSLY (no page table), ragged across the
//! batch via `kv_indptr`. Routes to FlashInfer's
//! `BatchPrefillWithRaggedKVCacheDispatched`. Useful when the KV is not
//! paged — e.g. the initial full-prompt prefill before the cache is paged,
//! or a non-paged serving path.
//!
//! ## Layout
//!
//! - `q`        : `[total_num_rows, num_qo_heads, head_dim]` element `T`.
//! - `q_indptr` : `[batch + 1]` i32 — query row prefix-sum.
//! - `k_data` / `v_data` : `[total_kv_rows, num_kv_heads, head_dim]` `T`.
//! - `kv_indptr`: `[batch + 1]` i32 — KV row prefix-sum.
//! - `o`        : `[total_num_rows, num_qo_heads, head_dim]` element `T`.
//! - `lse`      : `[total_num_rows, num_qo_heads]` f32.
//!
//! Constraints + synchronization match [`BatchPagedPrefillPlan`]:
//! `head_dim ∈ {64,128,256}`, integer GQA, `T ∈ {f16, bf16}`, causal or
//! full, optional KV-split, synchronous `run`.

use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};


/// Descriptor for a batched ragged-KV prefill op.
#[derive(Copy, Clone, Debug)]
pub struct BatchRaggedPrefillDescriptor {
    /// Number of requests in the batch.
    pub batch_size: i32,
    /// Total query rows across all requests (`q_indptr[batch]`).
    pub total_num_rows: i32,
    /// Total KV rows across all requests (`kv_indptr[batch]`).
    pub total_kv_rows: i32,
    /// Query / output attention heads.
    pub num_qo_heads: i32,
    /// KV heads.
    pub num_kv_heads: i32,
    /// Per-head dimension. Must be 64, 128, or 256.
    pub head_dim: i32,
    /// Score scaling factor.
    pub sm_scale: f32,
    /// Apply causal masking.
    pub causal: bool,
    /// Opt into KV-split parallelism (see [`BatchPagedPrefillPlan`]).
    pub enable_kv_split: bool,
    /// Element type of q / k / v / o.
    pub element: ElementKind,
}

/// Args bundle for a batched ragged-KV prefill launch.
pub struct BatchRaggedPrefillArgs<'a, T: Element> {
    /// Ragged query rows — `[total_num_rows, num_qo_heads, head_dim]`.
    pub q: TensorRef<'a, T, 3>,
    /// Query row prefix-sum `[batch + 1]` i32.
    pub q_indptr: TensorRef<'a, i32, 1>,
    /// Ragged K — `[total_kv_rows, num_kv_heads, head_dim]`.
    pub k_data: TensorRef<'a, T, 3>,
    /// Ragged V — same layout as `k_data`.
    pub v_data: TensorRef<'a, T, 3>,
    /// KV row prefix-sum `[batch + 1]` i32.
    pub kv_indptr: TensorRef<'a, i32, 1>,
    /// Output rows — `[total_num_rows, num_qo_heads, head_dim]`.
    pub o: TensorMut<'a, T, 3>,
    /// Per-row log-sum-exp — `[total_num_rows, num_qo_heads]` f32.
    pub lse: TensorMut<'a, f32, 2>,
}

/// Batched ragged-KV prefill plan. f16 / bf16. Requires the `flashinfer`
/// cargo feature.
pub struct BatchRaggedPrefillPlan<T: Element> {
    desc: BatchRaggedPrefillDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> BatchRaggedPrefillPlan<T> {
    /// Pick a kernel + validate shape limits.
    pub fn select(
        _stream: &Stream,
        desc: &BatchRaggedPrefillDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported("BatchRaggedPrefillPlan: descriptor element != T"));
        }
        if desc.batch_size <= 0
            || desc.total_num_rows <= 0
            || desc.total_kv_rows <= 0
            || desc.num_qo_heads <= 0
            || desc.num_kv_heads <= 0
        {
            return Err(Error::InvalidProblem("BatchRaggedPrefillPlan: extents must be positive"));
        }
        if desc.num_qo_heads % desc.num_kv_heads != 0 {
            return Err(Error::InvalidProblem(
                "BatchRaggedPrefillPlan: num_qo_heads must be a multiple of num_kv_heads",
            ));
        }
        if !matches!(desc.head_dim, 64 | 128 | 256) {
            return Err(Error::Unsupported("BatchRaggedPrefillPlan: head_dim must be 64, 128, or 256"));
        }
        if !matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16) {
            return Err(Error::Unsupported(
                "BatchRaggedPrefillPlan: element type must be f16 or bf16 (prefill is mma-based)",
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
        Ok(Self { desc: *desc, sku, _marker: PhantomData })
    }

    /// Validate args against the descriptor.
    pub fn can_implement(&self, args: &BatchRaggedPrefillArgs<'_, T>) -> Result<()> {
        let d = &self.desc;
        let qo_shape = [d.total_num_rows, d.num_qo_heads, d.head_dim];
        if args.q.shape != qo_shape || args.o.shape != qo_shape {
            return Err(Error::InvalidProblem("BatchRaggedPrefillPlan: q/o shape mismatch"));
        }
        let kv_shape = [d.total_kv_rows, d.num_kv_heads, d.head_dim];
        if args.k_data.shape != kv_shape || args.v_data.shape != kv_shape {
            return Err(Error::InvalidProblem("BatchRaggedPrefillPlan: k_data/v_data shape mismatch"));
        }
        if args.q_indptr.shape != [d.batch_size + 1] || args.kv_indptr.shape != [d.batch_size + 1] {
            return Err(Error::InvalidProblem(
                "BatchRaggedPrefillPlan: q_indptr/kv_indptr shape must be [batch + 1]",
            ));
        }
        if args.lse.shape != [d.total_num_rows, d.num_qo_heads] {
            return Err(Error::InvalidProblem(
                "BatchRaggedPrefillPlan: lse shape must be [total_num_rows, num_qo_heads]",
            ));
        }
        if !args.q.is_contiguous()
            || !args.k_data.is_contiguous()
            || !args.v_data.is_contiguous()
            || !args.o.is_contiguous()
            || !args.lse.is_contiguous()
        {
            return Err(Error::Unsupported("BatchRaggedPrefillPlan: tensors must be contiguous"));
        }
        Ok(())
    }

    /// Workspace bytes — zero; the scheduler workspace is internal.
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

    /// Launch the FlashInfer ragged-KV prefill kernel (synchronous).
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: BatchRaggedPrefillArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = (stream, &args);
            Err(Error::Unsupported(
                "BatchRaggedPrefillPlan: `flashinfer` cargo feature is not enabled",
            ))
        }
        #[cfg(feature = "flashinfer")]
        {
            let d = &self.desc;
            let stream_ptr = stream.as_raw() as *mut c_void;
            let q_ptr = args.q.data.as_raw().0 as *const c_void;
            let q_indptr_ptr = args.q_indptr.data.as_raw().0 as *mut c_void;
            let k_ptr = args.k_data.data.as_raw().0 as *const c_void;
            let v_ptr = args.v_data.data.as_raw().0 as *const c_void;
            let kv_indptr_ptr = args.kv_indptr.data.as_raw().0 as *mut c_void;
            let o_ptr = args.o.data.as_raw().0 as *mut c_void;
            let lse_ptr = args.lse.data.as_raw().0 as *mut c_void;
            let causal = if d.causal { 1 } else { 0 };
            let enable_split = if d.enable_kv_split { 1 } else { 0 };

            let status = match T::KIND {
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_ragged_prefill_f16_run(
                        d.batch_size, d.total_num_rows, d.total_kv_rows, d.head_dim,
                        d.num_qo_heads, d.num_kv_heads, d.sm_scale, causal, enable_split,
                        k_ptr, v_ptr, kv_indptr_ptr, q_ptr, q_indptr_ptr, o_ptr, lse_ptr, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_ragged_prefill_bf16_run(
                        d.batch_size, d.total_num_rows, d.total_kv_rows, d.head_dim,
                        d.num_qo_heads, d.num_kv_heads, d.sm_scale, causal, enable_split,
                        k_ptr, v_ptr, kv_indptr_ptr, q_ptr, q_indptr_ptr, o_ptr, lse_ptr, stream_ptr,
                    )
                },
                _ => {
                    return Err(Error::Unsupported(
                        "BatchRaggedPrefillPlan::run reached an unimplemented dtype",
                    ))
                }
            };
            map_status(status)
        }
    }
}
