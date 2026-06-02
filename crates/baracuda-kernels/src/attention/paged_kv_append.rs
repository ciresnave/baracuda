//! Paged KV-cache append — Phase 46 (FlashInfer cherry-pick).
//!
//! Companion to [`crate::KvCacheAppendPlan`] (which writes into a
//! CONTIGUOUS rank-4 cache). This plan writes new K/V slices into the
//! PAGED store used by [`crate::BatchPagedDecodePlan`].
//!
//! Decode-time variant (1 token per request). The launcher routes to
//! FlashInfer's `AppendPagedKVCacheDecode<DType, IdType>` kernel.
//!
//! Caller contract:
//! - Increment `last_page_len[b]` by 1 *after* the call returns (this
//!   plan does NOT mutate the page table).
//! - If `last_page_len[b] == page_size`, the caller's `BlockManager`
//!   must allocate a new page and extend `indices` / `indptr` BEFORE
//!   the next call.

use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::batch_paged_decode::PagedKvCacheDescriptor;

/// Descriptor for a paged KV-cache append op (decode-time).
#[derive(Copy, Clone, Debug)]
pub struct PagedKvAppendDescriptor {
    /// Number of requests in the batch (one new K/V row each).
    pub batch_size: i32,
    /// Paged cache descriptor.
    pub paged_kv: PagedKvCacheDescriptor,
}

/// Args bundle for a paged KV-cache append launch.
pub struct PagedKvAppendArgs<'a, T: Element> {
    /// New K rows — shape `[batch, num_kv_heads, head_dim]`.
    pub key: TensorRef<'a, T, 3>,
    /// New V rows — shape `[batch, num_kv_heads, head_dim]`.
    pub value: TensorRef<'a, T, 3>,
    /// Paged K cache — `[max_num_pages, num_kv_heads, page_size, head_dim]`,
    /// modified in place.
    pub k_data: TensorMut<'a, T, 4>,
    /// Paged V cache — same layout as `k_data`.
    pub v_data: TensorMut<'a, T, 4>,
    /// Page indices `[total_used_pages]` i32.
    pub indices: TensorRef<'a, i32, 1>,
    /// Page indptr `[batch + 1]` i32.
    pub indptr: TensorRef<'a, i32, 1>,
    /// Last-page-len `[batch]` i32 (values in `[0, page_size]`).
    /// **Read-only** in this plan — caller increments after the call.
    pub last_page_len: TensorRef<'a, i32, 1>,
}

/// Paged KV-cache append plan (decode-time, 1 token per request).
pub struct PagedKvAppendPlan<T: Element> {
    desc: PagedKvAppendDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> PagedKvAppendPlan<T> {
    /// Pick a kernel + validate descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &PagedKvAppendDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.paged_kv.element != T::KIND {
            return Err(Error::Unsupported(
                "PagedKvAppendPlan: descriptor element != T",
            ));
        }
        if desc.batch_size <= 0
            || desc.paged_kv.num_kv_heads <= 0
            || desc.paged_kv.page_size <= 0
            || desc.paged_kv.num_total_pages <= 0
        {
            return Err(Error::InvalidProblem(
                "PagedKvAppendPlan: extents must be positive",
            ));
        }
        if !matches!(desc.paged_kv.head_dim, 64 | 128 | 256) {
            return Err(Error::Unsupported(
                "PagedKvAppendPlan: head_dim must be 64, 128, or 256",
            ));
        }
        if !matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16 | ElementKind::F32) {
            return Err(Error::Unsupported(
                "PagedKvAppendPlan: element type must be f16, bf16, or f32",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: T::KIND,
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
    pub fn can_implement(&self, args: &PagedKvAppendArgs<'_, T>) -> Result<()> {
        let kv_row_shape = [
            self.desc.batch_size,
            self.desc.paged_kv.num_kv_heads,
            self.desc.paged_kv.head_dim,
        ];
        if args.key.shape != kv_row_shape || args.value.shape != kv_row_shape {
            return Err(Error::InvalidProblem(
                "PagedKvAppendPlan: key/value shape mismatch",
            ));
        }
        let cache_shape = [
            self.desc.paged_kv.num_total_pages,
            self.desc.paged_kv.num_kv_heads,
            self.desc.paged_kv.page_size,
            self.desc.paged_kv.head_dim,
        ];
        if args.k_data.shape != cache_shape || args.v_data.shape != cache_shape {
            return Err(Error::InvalidProblem(
                "PagedKvAppendPlan: k_data/v_data shape mismatch",
            ));
        }
        if args.indptr.shape != [self.desc.batch_size + 1]
            || args.last_page_len.shape != [self.desc.batch_size]
        {
            return Err(Error::InvalidProblem(
                "PagedKvAppendPlan: indptr / last_page_len shape mismatch",
            ));
        }
        if !args.key.is_contiguous()
            || !args.value.is_contiguous()
            || !args.k_data.is_contiguous()
            || !args.v_data.is_contiguous()
        {
            return Err(Error::Unsupported(
                "PagedKvAppendPlan: K/V tensors must be contiguous",
            ));
        }
        Ok(())
    }

    /// Required workspace bytes (always 0 — pure copy).
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

    /// Launch the FlashInfer paged-KV append kernel on the supplied
    /// stream. Requires the `flashinfer` cargo feature.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: PagedKvAppendArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = (stream, &args);
            Err(Error::Unsupported(
                "PagedKvAppendPlan: `flashinfer` cargo feature is not enabled",
            ))
        }
        #[cfg(feature = "flashinfer")]
        {
            let stream_ptr = stream.as_raw() as *mut c_void;
            let key_ptr = args.key.data.as_raw().0 as *const c_void;
            let value_ptr = args.value.data.as_raw().0 as *const c_void;
            let k_ptr = args.k_data.data.as_raw().0 as *mut c_void;
            let v_ptr = args.v_data.data.as_raw().0 as *mut c_void;
            let indices_ptr = args.indices.data.as_raw().0 as *mut c_void;
            let indptr_ptr = args.indptr.data.as_raw().0 as *mut c_void;
            let last_page_len_ptr = args.last_page_len.data.as_raw().0 as *mut c_void;

            let status = match T::KIND {
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_f16_run(
                        self.desc.batch_size,
                        self.desc.paged_kv.page_size,
                        self.desc.paged_kv.num_kv_heads,
                        self.desc.paged_kv.head_dim,
                        k_ptr, v_ptr, indices_ptr, indptr_ptr, last_page_len_ptr,
                        key_ptr, value_ptr, stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_bf16_run(
                        self.desc.batch_size,
                        self.desc.paged_kv.page_size,
                        self.desc.paged_kv.num_kv_heads,
                        self.desc.paged_kv.head_dim,
                        k_ptr, v_ptr, indices_ptr, indptr_ptr, last_page_len_ptr,
                        key_ptr, value_ptr, stream_ptr,
                    )
                },
                ElementKind::F32 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_flashinfer_paged_kv_append_decode_f32_run(
                        self.desc.batch_size,
                        self.desc.paged_kv.page_size,
                        self.desc.paged_kv.num_kv_heads,
                        self.desc.paged_kv.head_dim,
                        k_ptr, v_ptr, indices_ptr, indptr_ptr, last_page_len_ptr,
                        key_ptr, value_ptr, stream_ptr,
                    )
                },
                _ => {
                    return Err(Error::Unsupported(
                        "PagedKvAppendPlan::run reached an unimplemented dtype",
                    ));
                }
            };
            map_status(status)
        }
    }
}
