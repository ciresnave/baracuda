//! Batched paged-KV decode — Phase 46 (FlashInfer cherry-pick).
//!
//! The key decode primitive for vLLM-style serving stacks: every request
//! in the batch contributes exactly ONE query row, and its K/V history
//! lives in a **paged** store (fixed-size pages allocated from a free
//! list and tracked through a per-request page table). The plan launches
//! FlashInfer's `BatchDecodeWithPagedKVCacheDispatched` kernel against
//! the supplied page table and produces one output row per request plus
//! a per-row log-sum-exp.
//!
//! ## Layout
//!
//! Paged store (HND layout — heads-major within each page):
//!
//! - `k_data` / `v_data` : `[max_num_pages, num_kv_heads, page_size, head_dim]`
//!   element type `T`, contiguous, kHND.
//! - `indices`           : `[total_used_pages]` i32 — physical page IDs.
//! - `indptr`            : `[batch + 1]` i32 — prefix sum into `indices`.
//!                         Request `b`'s pages are at
//!                         `indices[indptr[b] .. indptr[b+1]]`.
//! - `last_page_len`     : `[batch]` i32 — number of valid rows in
//!                         each request's last page; values in
//!                         `[0, page_size]`.
//!
//! Query / output:
//!
//! - `q`   : `[batch, num_qo_heads, head_dim]` element type `T`.
//! - `o`   : `[batch, num_qo_heads, head_dim]` element type `T`.
//! - `lse` : `[batch, num_qo_heads]` **f32** (FlashInfer convention —
//!   always f32 regardless of `T`).
//!
//! ## Constraints (Phase 46 Tier 1)
//!
//! - `head_dim ∈ {64, 128, 256}`.
//! - `num_qo_heads % num_kv_heads == 0` (GQA grouping factor must be
//!   integer).
//! - Element type `T ∈ {f16, bf16, f32}`.
//! - No positional encoding inside the kernel — apply RoPE BEFORE
//!   populating the cache (use [`crate::RopePlan`]).
//! - No sliding window, no logits soft-cap, no ALiBi, no custom mask.
//! - No CUDA Graph capture mode (auto-fallback returns `Unsupported`).
//!
//! ## Companion plans
//!
//! - [`PagedKvAppendPlan`](crate::PagedKvAppendPlan) — writes new K/V
//!   slices into the paged store at decode time.
//! - [`crate::KvCacheAppendPlan`] — the *contiguous*-cache append
//!   (rank-4 cache, no page table). Unchanged from Phase 6.5.
//!
//! The caller owns the page-table memory + the free-list / refcount
//! allocator (`BlockManager` / `BlockTable` in vLLM terminology). This
//! plan only consumes the page table once it's been built.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a paged KV-cache descriptor (kHND layout). Mirrors
/// FlashInfer's `paged_kv_t` host-side fields.
#[derive(Copy, Clone, Debug)]
pub struct PagedKvCacheDescriptor {
    /// Number of rows per page (e.g. 16 or 32 in vLLM defaults).
    pub page_size: i32,
    /// Total physical pages allocated by the caller's `BlockManager`.
    pub num_total_pages: i32,
    /// KV heads (typically `num_qo_heads / gqa_group_size`).
    pub num_kv_heads: i32,
    /// Per-head feature dimension.
    pub head_dim: i32,
    /// Element type of `k_data` / `v_data`.
    pub element: ElementKind,
}

/// Descriptor for a batched paged-KV decode op.
#[derive(Copy, Clone, Debug)]
pub struct BatchPagedDecodeDescriptor {
    /// Number of requests in the batch (each contributes 1 query row).
    pub batch_size: i32,
    /// Query / output attention heads.
    pub num_qo_heads: i32,
    /// Score scaling factor — typically `1.0 / sqrt(head_dim)`.
    pub sm_scale: f32,
    /// Paged cache descriptor (shared between K and V — same shape/dtype).
    pub paged_kv: PagedKvCacheDescriptor,
}

/// Args bundle for a batched paged-KV decode launch.
pub struct BatchPagedDecodeArgs<'a, T: Element> {
    /// Query rows — shape `[batch, num_qo_heads, head_dim]`.
    pub q: TensorRef<'a, T, 3>,
    /// Paged K cache — shape `[max_num_pages, num_kv_heads, page_size,
    /// head_dim]`. Modified by upstream [`PagedKvAppendPlan`] writes;
    /// read-only here.
    pub k_data: TensorRef<'a, T, 4>,
    /// Paged V cache — same layout as `k_data`.
    pub v_data: TensorRef<'a, T, 4>,
    /// Page indices array `[total_used_pages]` i32.
    pub indices: TensorRef<'a, i32, 1>,
    /// Page indptr `[batch + 1]` i32.
    pub indptr: TensorRef<'a, i32, 1>,
    /// Last-page-len `[batch]` i32 (values in `[0, page_size]`).
    pub last_page_len: TensorRef<'a, i32, 1>,
    /// Output rows — shape `[batch, num_qo_heads, head_dim]`.
    pub o: TensorMut<'a, T, 3>,
    /// Per-row log-sum-exp — shape `[batch, num_qo_heads]` **f32**.
    pub lse: TensorMut<'a, f32, 2>,
}

/// Batched paged-KV decode plan.
///
/// Routes to FlashInfer's `BatchDecodeWithPagedKVCacheDispatched`.
/// Requires the `flashinfer` cargo feature.
///
/// **Workspace**: required (auxiliary index buffers). Query the size
/// with [`Self::workspace_size`].
///
/// **Precision guarantee**: deterministic on same-hardware repeat
/// (FlashInfer's decode path is reduction-free across CTAs in the
/// no-split mode this plan uses). f16/bf16 accumulate in f32.
pub struct BatchPagedDecodePlan<T: Element> {
    desc: BatchPagedDecodeDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> BatchPagedDecodePlan<T> {
    /// Pick a kernel + validate shape limits.
    pub fn select(
        _stream: &Stream,
        desc: &BatchPagedDecodeDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.paged_kv.element != T::KIND {
            return Err(Error::Unsupported(
                "BatchPagedDecodePlan: descriptor element != T",
            ));
        }
        if desc.batch_size <= 0
            || desc.num_qo_heads <= 0
            || desc.paged_kv.num_kv_heads <= 0
            || desc.paged_kv.page_size <= 0
            || desc.paged_kv.num_total_pages <= 0
        {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodePlan: extents must be positive",
            ));
        }
        if desc.num_qo_heads % desc.paged_kv.num_kv_heads != 0 {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodePlan: num_qo_heads must be a multiple of num_kv_heads (GQA group size must be integer)",
            ));
        }
        let head_dim = desc.paged_kv.head_dim;
        if !matches!(head_dim, 64 | 128 | 256) {
            return Err(Error::Unsupported(
                "BatchPagedDecodePlan: head_dim must be 64, 128, or 256",
            ));
        }
        if !matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16 | ElementKind::F32) {
            return Err(Error::Unsupported(
                "BatchPagedDecodePlan: element type must be f16, bf16, or f32",
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
    pub fn can_implement(&self, args: &BatchPagedDecodeArgs<'_, T>) -> Result<()> {
        let q_shape = [
            self.desc.batch_size,
            self.desc.num_qo_heads,
            self.desc.paged_kv.head_dim,
        ];
        if args.q.shape != q_shape {
            return Err(Error::InvalidProblem("BatchPagedDecodePlan: q shape mismatch"));
        }
        let cache_shape = [
            self.desc.paged_kv.num_total_pages,
            self.desc.paged_kv.num_kv_heads,
            self.desc.paged_kv.page_size,
            self.desc.paged_kv.head_dim,
        ];
        if args.k_data.shape != cache_shape || args.v_data.shape != cache_shape {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodePlan: k_data/v_data shape mismatch",
            ));
        }
        if args.indptr.shape != [self.desc.batch_size + 1] {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodePlan: indptr shape must be [batch + 1]",
            ));
        }
        if args.last_page_len.shape != [self.desc.batch_size] {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodePlan: last_page_len shape must be [batch]",
            ));
        }
        if args.o.shape != q_shape {
            return Err(Error::InvalidProblem("BatchPagedDecodePlan: o shape mismatch"));
        }
        if args.lse.shape != [self.desc.batch_size, self.desc.num_qo_heads] {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodePlan: lse shape must be [batch, num_qo_heads]",
            ));
        }
        if !args.q.is_contiguous()
            || !args.k_data.is_contiguous()
            || !args.v_data.is_contiguous()
            || !args.o.is_contiguous()
            || !args.lse.is_contiguous()
        {
            return Err(Error::Unsupported(
                "BatchPagedDecodePlan: tensors must be contiguous (Tier 1)",
            ));
        }
        Ok(())
    }

    /// Workspace bytes — required for the auxiliary index buffers that
    /// the launcher initializes per call.
    #[inline]
    pub fn workspace_size(&self) -> usize {
        // Mirrors the C-side computation in
        // `flashinfer_paged_decode_launcher.cu::compute_workspace_bytes`.
        ((3 * self.desc.batch_size as usize) + 2) * core::mem::size_of::<i32>()
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

    /// Launch the FlashInfer paged-KV decode kernel on the supplied
    /// stream. Requires the `flashinfer` cargo feature.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: BatchPagedDecodeArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let need = self.workspace_size();
        let (ws_ptr, ws_bytes) = match workspace {
            Workspace::None => {
                return Err(Error::WorkspaceTooSmall { needed: need, got: 0 });
            }
            Workspace::Borrowed(slice) => {
                if slice.len() < need {
                    return Err(Error::WorkspaceTooSmall {
                        needed: need,
                        got: slice.len(),
                    });
                }
                (slice.as_raw().0 as *mut c_void, slice.len())
            }
        };
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = (stream, ws_ptr, ws_bytes, &args);
            Err(Error::Unsupported(
                "BatchPagedDecodePlan: `flashinfer` cargo feature is not enabled",
            ))
        }
        #[cfg(feature = "flashinfer")]
        {
            let stream_ptr = stream.as_raw() as *mut c_void;
            let q_ptr = args.q.data.as_raw().0 as *const c_void;
            let k_ptr = args.k_data.data.as_raw().0 as *mut c_void;
            let v_ptr = args.v_data.data.as_raw().0 as *mut c_void;
            let indices_ptr = args.indices.data.as_raw().0 as *mut c_void;
            let indptr_ptr = args.indptr.data.as_raw().0 as *mut c_void;
            let last_page_len_ptr = args.last_page_len.data.as_raw().0 as *mut c_void;
            let o_ptr = args.o.data.as_raw().0 as *mut c_void;
            let lse_ptr = args.lse.data.as_raw().0 as *mut c_void;

            let status = match T::KIND {
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_decode_f16_run(
                        self.desc.batch_size,
                        self.desc.paged_kv.page_size,
                        self.desc.paged_kv.head_dim,
                        self.desc.num_qo_heads,
                        self.desc.paged_kv.num_kv_heads,
                        self.desc.sm_scale,
                        k_ptr, v_ptr, indices_ptr, indptr_ptr, last_page_len_ptr,
                        q_ptr, o_ptr, lse_ptr,
                        ws_ptr, ws_bytes, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_decode_bf16_run(
                        self.desc.batch_size,
                        self.desc.paged_kv.page_size,
                        self.desc.paged_kv.head_dim,
                        self.desc.num_qo_heads,
                        self.desc.paged_kv.num_kv_heads,
                        self.desc.sm_scale,
                        k_ptr, v_ptr, indices_ptr, indptr_ptr, last_page_len_ptr,
                        q_ptr, o_ptr, lse_ptr,
                        ws_ptr, ws_bytes, stream_ptr,
                    )
                },
                ElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_decode_f32_run(
                        self.desc.batch_size,
                        self.desc.paged_kv.page_size,
                        self.desc.paged_kv.head_dim,
                        self.desc.num_qo_heads,
                        self.desc.paged_kv.num_kv_heads,
                        self.desc.sm_scale,
                        k_ptr, v_ptr, indices_ptr, indptr_ptr, last_page_len_ptr,
                        q_ptr, o_ptr, lse_ptr,
                        ws_ptr, ws_bytes, stream_ptr,
                    )
                },
                _ => {
                    return Err(Error::Unsupported(
                        "BatchPagedDecodePlan::run reached an unimplemented dtype",
                    ));
                }
            };
            map_status(status)
        }
    }
}
