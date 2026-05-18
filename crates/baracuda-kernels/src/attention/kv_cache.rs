//! KV-cache append — decoder-inference helper (Milestone 6.5).
//!
//! At every autoregressive decoding step the model produces fresh
//! `K_new` / `V_new` slices that need to be appended to running caches
//! `K_cache` / `V_cache` shared across the step's attention call. Each
//! batch sample keeps its own offset (`cache_offsets[b]`) — the next
//! slot to fill in its slice of the cache — so ragged-batch inference
//! (different cumulative cache lengths per sample) is natively
//! supported.
//!
//! Op semantics, for each `b, h, l_new, d`:
//!
//! ```text
//! K_cache[b, h, cache_offsets[b] + l_new, d_k] = K_new[b, h, l_new, d_k]
//! V_cache[b, h, cache_offsets[b] + l_new, d_v] = V_new[b, h, l_new, d_v]
//! ```
//!
//! Shape conventions (all rank-4, contiguous, row-major):
//!
//! | tensor          | shape                                    |
//! |-----------------|------------------------------------------|
//! | `K_new`         | `[B, H, L_new, D_k]`                     |
//! | `V_new`         | `[B, H, L_new, D_v]`                     |
//! | `cache_offsets` | `[B]` (i64)                              |
//! | `K_cache`       | `[B, H, L_max, D_k]` (modified in place) |
//! | `V_cache`       | `[B, H, L_max, D_v]` (modified in place) |
//!
//! Pure copy — bit-exact across every wired dtype (`{f32, f16, bf16,
//! f64}`). Cells where `cache_offsets[b] + l_new >= L_max` are silently
//! skipped; the caller is responsible for sizing the cache so writes
//! land in bounds. No BW — KV-cache is an inference-time op.
//!
//! After the call the caller updates `cache_offsets[b] += L_new` (host-
//! or device-side, the plan doesn't touch the offset vector).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Descriptor for a KV-cache append op.
#[derive(Copy, Clone, Debug)]
pub struct KvCacheAppendDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Number of attention heads (`H`).
    pub num_heads: i32,
    /// Number of new K/V rows to append per sample (`L_new`).
    pub new_len: i32,
    /// Capacity of the cache along the sequence axis (`L_max`).
    pub max_cache_len: i32,
    /// Head dimension of K (`D_k`).
    pub d_k: i32,
    /// Head dimension of V (`D_v`). May differ from `d_k`.
    pub d_v: i32,
    /// Element type — must match the plan's type parameter.
    pub element: ElementKind,
}

/// Args bundle for a KV-cache append launch.
pub struct KvCacheAppendArgs<'a, T: Element> {
    /// New K rows — shape `[B, H, L_new, D_k]`, contiguous.
    pub k_new: TensorRef<'a, T, 4>,
    /// New V rows — shape `[B, H, L_new, D_v]`, contiguous.
    pub v_new: TensorRef<'a, T, 4>,
    /// Per-sample insert offsets — shape `[B]`, `i64`. Values must
    /// satisfy `0 <= cache_offsets[b]` and `cache_offsets[b] + L_new
    /// <= L_max` for every cell that should land (out-of-range cells
    /// are silently skipped by the kernel).
    pub cache_offsets: TensorRef<'a, i64, 1>,
    /// Destination K cache — shape `[B, H, L_max, D_k]`, contiguous;
    /// modified in place.
    pub k_cache: TensorMut<'a, T, 4>,
    /// Destination V cache — shape `[B, H, L_max, D_v]`, contiguous;
    /// modified in place.
    pub v_cache: TensorMut<'a, T, 4>,
}

/// KV-cache append plan.
///
/// Writes new `K_new` / `V_new` slices into running `K_cache` /
/// `V_cache` buffers at per-sample offsets supplied via
/// `cache_offsets[b]`. Pure copy — bit-exact across all wired dtypes.
///
/// **When to use**: autoregressive decoder inference. Call once per
/// generation step to extend the cache before the attention op for the
/// next step. Ragged-batch insertion is supported natively because each
/// sample carries its own offset. No backward — KV-cache is an
/// inference-time op.
///
/// **Dtypes**: `f32`, `f64`, `f16`, `bf16`.
///
/// **Shape limits**: rank-4 contiguous K/V tensors. `d_k != d_v` is
/// allowed. Cells where `cache_offsets[b] + l_new >= max_cache_len`
/// are silently skipped — the caller sizes the cache so writes land in
/// bounds.
///
/// **Workspace**: zero (pure in-place copy).
///
/// **Precision guarantee**: bit-exact (no math at all).
pub struct KvCacheAppendPlan<T: Element> {
    desc: KvCacheAppendDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> KvCacheAppendPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &KvCacheAppendDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::KvCacheAppendPlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0
            || desc.num_heads < 0
            || desc.new_len < 0
            || desc.max_cache_len < 0
            || desc.d_k < 0
            || desc.d_v < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KvCacheAppendPlan: extents must be non-negative",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::KvCacheAppendPlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            // Pure copy — no accumulator math happens at all. The
            // value carried here is cosmetic; bit-stability is
            // guaranteed regardless.
            accumulator: T::KIND,
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::KvCache as u16,
            element: T::KIND,
            aux_element: None,
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

    /// Validate args against the descriptor.
    pub fn can_implement(&self, args: &KvCacheAppendArgs<'_, T>) -> Result<()> {
        let shape_k_new = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.new_len,
            self.desc.d_k,
        ];
        let shape_v_new = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.new_len,
            self.desc.d_v,
        ];
        let shape_k_cache = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.max_cache_len,
            self.desc.d_k,
        ];
        let shape_v_cache = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.max_cache_len,
            self.desc.d_v,
        ];
        if args.k_new.shape != shape_k_new {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KvCacheAppendPlan: k_new shape mismatch",
            ));
        }
        if args.v_new.shape != shape_v_new {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KvCacheAppendPlan: v_new shape mismatch",
            ));
        }
        if args.k_cache.shape != shape_k_cache {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KvCacheAppendPlan: k_cache shape mismatch",
            ));
        }
        if args.v_cache.shape != shape_v_cache {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KvCacheAppendPlan: v_cache shape mismatch",
            ));
        }
        if args.cache_offsets.shape != [self.desc.batch_size] {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::KvCacheAppendPlan: cache_offsets shape must be [batch_size]",
            ));
        }
        if !args.k_new.is_contiguous()
            || !args.v_new.is_contiguous()
            || !args.k_cache.is_contiguous()
            || !args.v_cache.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::KvCacheAppendPlan: trailblazer requires contiguous K/V tensors",
            ));
        }
        if args.cache_offsets.stride != [1] {
            return Err(Error::Unsupported(
                "baracuda-kernels::KvCacheAppendPlan: cache_offsets must be unit-stride",
            ));
        }
        let k_new_n = args.k_new.numel();
        let v_new_n = args.v_new.numel();
        let k_cache_n = args.k_cache.numel();
        let v_cache_n = args.v_cache.numel();
        if (args.k_new.data.len() as i64) < k_new_n
            || (args.v_new.data.len() as i64) < v_new_n
            || (args.k_cache.data.len() as i64) < k_cache_n
            || (args.v_cache.data.len() as i64) < v_cache_n
        {
            return Err(Error::BufferTooSmall {
                needed: k_new_n.max(v_new_n).max(k_cache_n).max(v_cache_n) as usize,
                got: 0,
            });
        }
        if (args.cache_offsets.data.len() as i64) < self.desc.batch_size as i64 {
            return Err(Error::BufferTooSmall {
                needed: self.desc.batch_size as usize,
                got: args.cache_offsets.data.len(),
            });
        }
        Ok(())
    }

    /// Workspace size in bytes — zero (pure in-place copy).
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

    /// Launch the K + V copy kernels on the supplied stream.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: KvCacheAppendArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        // Empty problem — nothing to do.
        if self.desc.batch_size == 0
            || self.desc.num_heads == 0
            || self.desc.new_len == 0
        {
            return Ok(());
        }
        let stream_ptr = stream.as_raw() as *mut c_void;
        let k_new_ptr = args.k_new.data.as_raw().0 as *const c_void;
        let v_new_ptr = args.v_new.data.as_raw().0 as *const c_void;
        let offsets_ptr = args.cache_offsets.data.as_raw().0 as *const c_void;
        let k_cache_ptr = args.k_cache.data.as_raw().0 as *mut c_void;
        let v_cache_ptr = args.v_cache.data.as_raw().0 as *mut c_void;

        let status = match T::KIND {
            ElementKind::F32 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_kv_cache_append_f32_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.new_len,
                    self.desc.max_cache_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    k_new_ptr,
                    v_new_ptr,
                    offsets_ptr,
                    k_cache_ptr,
                    v_cache_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_kv_cache_append_f16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.new_len,
                    self.desc.max_cache_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    k_new_ptr,
                    v_new_ptr,
                    offsets_ptr,
                    k_cache_ptr,
                    v_cache_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_kv_cache_append_bf16_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.new_len,
                    self.desc.max_cache_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    k_new_ptr,
                    v_new_ptr,
                    offsets_ptr,
                    k_cache_ptr,
                    v_cache_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            ElementKind::F64 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_kv_cache_append_f64_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.new_len,
                    self.desc.max_cache_len,
                    self.desc.d_k,
                    self.desc.d_v,
                    k_new_ptr,
                    v_new_ptr,
                    offsets_ptr,
                    k_cache_ptr,
                    v_cache_ptr,
                    core::ptr::null_mut(),
                    0,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::KvCacheAppendPlan::run reached an unimplemented dtype",
                ));
            }
        };
        map_status(status)
    }
}
