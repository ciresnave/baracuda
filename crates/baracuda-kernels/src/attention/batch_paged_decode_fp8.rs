//! FP8 KV-cache paged decode — Phase 66 Tier 2 (FlashInfer).
//!
//! A variant of [`BatchPagedDecodePlan`](crate::BatchPagedDecodePlan) where
//! the paged K/V cache is stored in 8-bit floating point (e4m3 or e5m2)
//! while the query / output stay in f16 or bf16. The decode kernel
//! `cast_load`s the fp8 KV to float on the fly (the fp8 format carries the
//! value directly — no separate dequant scale), halving KV-cache memory
//! and bandwidth.
//!
//! The fp8 cache tensors are passed as raw `u8` (one byte per fp8 element);
//! select the encoding via [`Fp8KvDtype`] in the descriptor.
//!
//! Constraints mirror the homogeneous decode plan: `head_dim ∈ {64, 128,
//! 256}`, integer GQA, Q/O ∈ {f16, bf16}, RoPE applied before the cache,
//! no mask / sliding-window / soft-cap / ALiBi. Same auxiliary workspace.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};


/// FP8 encoding of the KV cache.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Fp8KvDtype {
    /// `e4m3` (1 sign, 4 exponent, 3 mantissa) — higher precision, range ±448.
    E4M3,
    /// `e5m2` (1 sign, 5 exponent, 2 mantissa) — higher range, lower precision.
    E5M2,
}

/// Descriptor for an FP8 KV-cache paged decode op.
#[derive(Copy, Clone, Debug)]
pub struct BatchPagedDecodeFp8Descriptor {
    /// Number of requests (one query row each).
    pub batch_size: i32,
    /// Query / output attention heads.
    pub num_qo_heads: i32,
    /// Score scaling factor — typically `1.0 / sqrt(head_dim)`.
    pub sm_scale: f32,
    /// Rows per page.
    pub page_size: i32,
    /// Total physical pages.
    pub num_total_pages: i32,
    /// KV heads.
    pub num_kv_heads: i32,
    /// Per-head dimension. Must be 64, 128, or 256.
    pub head_dim: i32,
    /// FP8 encoding of `k_data` / `v_data`.
    pub kv_dtype: Fp8KvDtype,
}

/// Args bundle for an FP8 KV-cache paged decode launch.
pub struct BatchPagedDecodeFp8Args<'a, T: Element> {
    /// Query rows — `[batch, num_qo_heads, head_dim]` element `T`.
    pub q: TensorRef<'a, T, 3>,
    /// Paged K cache (fp8 bytes) —
    /// `[max_num_pages, num_kv_heads, page_size, head_dim]` `u8`.
    pub k_data: TensorRef<'a, u8, 4>,
    /// Paged V cache (fp8 bytes) — same layout as `k_data`.
    pub v_data: TensorRef<'a, u8, 4>,
    /// Page indices `[total_used_pages]` i32.
    pub indices: TensorRef<'a, i32, 1>,
    /// Page indptr `[batch + 1]` i32.
    pub indptr: TensorRef<'a, i32, 1>,
    /// Last-page-len `[batch]` i32.
    pub last_page_len: TensorRef<'a, i32, 1>,
    /// Output rows — `[batch, num_qo_heads, head_dim]` element `T`.
    pub o: TensorMut<'a, T, 3>,
    /// Per-row log-sum-exp — `[batch, num_qo_heads]` f32.
    pub lse: TensorMut<'a, f32, 2>,
}

/// FP8 KV-cache paged decode plan (Q/O in `T` = f16/bf16, KV in fp8).
pub struct BatchPagedDecodeFp8Plan<T: Element> {
    desc: BatchPagedDecodeFp8Descriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> BatchPagedDecodeFp8Plan<T> {
    /// Pick a kernel + validate shape limits.
    pub fn select(
        _stream: &Stream,
        desc: &BatchPagedDecodeFp8Descriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.batch_size <= 0
            || desc.num_qo_heads <= 0
            || desc.num_kv_heads <= 0
            || desc.page_size <= 0
            || desc.num_total_pages <= 0
        {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodeFp8Plan: extents must be positive",
            ));
        }
        if desc.num_qo_heads % desc.num_kv_heads != 0 {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodeFp8Plan: num_qo_heads must be a multiple of num_kv_heads",
            ));
        }
        if !matches!(desc.head_dim, 64 | 128 | 256) {
            return Err(Error::Unsupported(
                "BatchPagedDecodeFp8Plan: head_dim must be 64, 128, or 256",
            ));
        }
        if !matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16) {
            return Err(Error::Unsupported(
                "BatchPagedDecodeFp8Plan: Q/O element must be f16 or bf16",
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
            aux_element: Some(match desc.kv_dtype {
                Fp8KvDtype::E4M3 => ElementKind::Fp8E4M3,
                Fp8KvDtype::E5M2 => ElementKind::Fp8E5M2,
            }),
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
    pub fn can_implement(&self, args: &BatchPagedDecodeFp8Args<'_, T>) -> Result<()> {
        let d = &self.desc;
        let q_shape = [d.batch_size, d.num_qo_heads, d.head_dim];
        if args.q.shape != q_shape || args.o.shape != q_shape {
            return Err(Error::InvalidProblem("BatchPagedDecodeFp8Plan: q/o shape mismatch"));
        }
        let cache_shape = [d.num_total_pages, d.num_kv_heads, d.page_size, d.head_dim];
        if args.k_data.shape != cache_shape || args.v_data.shape != cache_shape {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodeFp8Plan: k_data/v_data shape mismatch",
            ));
        }
        if args.indptr.shape != [d.batch_size + 1] {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodeFp8Plan: indptr shape must be [batch + 1]",
            ));
        }
        if args.last_page_len.shape != [d.batch_size] {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodeFp8Plan: last_page_len shape must be [batch]",
            ));
        }
        if args.lse.shape != [d.batch_size, d.num_qo_heads] {
            return Err(Error::InvalidProblem(
                "BatchPagedDecodeFp8Plan: lse shape must be [batch, num_qo_heads]",
            ));
        }
        if !args.q.is_contiguous()
            || !args.k_data.is_contiguous()
            || !args.v_data.is_contiguous()
            || !args.o.is_contiguous()
            || !args.lse.is_contiguous()
        {
            return Err(Error::Unsupported(
                "BatchPagedDecodeFp8Plan: tensors must be contiguous (Tier 1)",
            ));
        }
        Ok(())
    }

    /// Auxiliary index workspace bytes (same as the homogeneous decode plan).
    #[inline]
    pub fn workspace_size(&self) -> usize {
        ((3 * self.desc.batch_size as usize) + 2) * core::mem::size_of::<i32>()
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

    /// Launch the FP8 KV-cache decode kernel.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: BatchPagedDecodeFp8Args<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        let need = self.workspace_size();
        let (ws_ptr, ws_bytes) = match workspace {
            Workspace::None => return Err(Error::WorkspaceTooSmall { needed: need, got: 0 }),
            Workspace::Borrowed(slice) => {
                if slice.len() < need {
                    return Err(Error::WorkspaceTooSmall { needed: need, got: slice.len() });
                }
                (slice.as_raw().0 as *mut c_void, slice.len())
            }
        };
        #[cfg(not(feature = "flashinfer"))]
        {
            let _ = (stream, ws_ptr, ws_bytes, &args);
            Err(Error::Unsupported(
                "BatchPagedDecodeFp8Plan: `flashinfer` cargo feature is not enabled",
            ))
        }
        #[cfg(feature = "flashinfer")]
        {
            let d = &self.desc;
            let stream_ptr = stream.as_raw() as *mut c_void;
            let q_ptr = args.q.data.as_raw().0 as *const c_void;
            let k_ptr = args.k_data.data.as_raw().0 as *mut c_void;
            let v_ptr = args.v_data.data.as_raw().0 as *mut c_void;
            let indices_ptr = args.indices.data.as_raw().0 as *mut c_void;
            let indptr_ptr = args.indptr.data.as_raw().0 as *mut c_void;
            let last_page_len_ptr = args.last_page_len.data.as_raw().0 as *mut c_void;
            let o_ptr = args.o.data.as_raw().0 as *mut c_void;
            let lse_ptr = args.lse.data.as_raw().0 as *mut c_void;

            macro_rules! call {
                ($f:ident) => {{
                    unsafe {
                        baracuda_kernels_sys::$f(
                            d.batch_size, d.page_size, d.head_dim, d.num_qo_heads, d.num_kv_heads,
                            d.sm_scale, k_ptr, v_ptr, indices_ptr, indptr_ptr, last_page_len_ptr,
                            q_ptr, o_ptr, lse_ptr, ws_ptr, ws_bytes, stream_ptr,
                        )
                    }
                }};
            }
            let status = match (T::KIND, d.kv_dtype) {
                (ElementKind::F16, Fp8KvDtype::E4M3) => {
                    call!(baracuda_kernels_flashinfer_paged_decode_f16_e4m3_run)
                }
                (ElementKind::F16, Fp8KvDtype::E5M2) => {
                    call!(baracuda_kernels_flashinfer_paged_decode_f16_e5m2_run)
                }
                (ElementKind::Bf16, Fp8KvDtype::E4M3) => {
                    call!(baracuda_kernels_flashinfer_paged_decode_bf16_e4m3_run)
                }
                (ElementKind::Bf16, Fp8KvDtype::E5M2) => {
                    call!(baracuda_kernels_flashinfer_paged_decode_bf16_e5m2_run)
                }
                _ => {
                    return Err(Error::Unsupported(
                        "BatchPagedDecodeFp8Plan::run reached an unimplemented dtype",
                    ))
                }
            };
            map_status(status)
        }
    }
}
