//! Block-sparse Scaled Dot-Product Attention forward plan (Phase 54).
//!
//! Clean-room hand-port of facebookresearch/xFormers
//! `components/attention/blocksparse.py` algorithmic reference
//! (BSD-3-Clause). See
//! [`crates/baracuda-kernels-sys/vendor/xformers/VENDOR.md`] for the
//! attribution + cherry-pick scope documentation.
//!
//! ## Algorithm
//!
//! The attention mask is supplied as a **per-block** boolean pattern
//! of shape `[B, H, num_blocks_q * num_blocks_k]` (uint8 — 0 = skip,
//! non-zero = compute). Only the active (q_block, k_block) pairs
//! contribute to the QK^T matmul + online-softmax accumulation;
//! masked blocks are skipped entirely (no K/V load, no compute).
//!
//! This is the **differentiator** from baracuda's existing attention
//! surface:
//!
//! - [`super::FlashSdpaPlan`] (dense) — iterates every k-block.
//! - Phase 51's arbitrary additive-mask path — iterates every k-block
//!   and adds an additive bias to S = QK^T before softmax. Still O(QK)
//!   compute.
//! - **This plan (block-sparse)** — actually skips compute on masked
//!   blocks. Real wall-clock speedup on long-context attention with
//!   known sparse patterns (sliding-window with sinks, BigBird-style
//!   strided patterns, dilated attention, etc.).
//!
//! Block size is supplied at launch time (typical values 32 / 64;
//! capped at 64 in the Tier-1 trailblazer per the SMEM budget at
//! `d_k = d_v = 128`).
//!
//! ## Shape conventions (rank-4, contiguous, row-major)
//!
//! | tensor | shape |
//! |--------|-------|
//! | `Q`              | `[B, H, Q_len, D_k]` |
//! | `K`              | `[B, H, K_len, D_k]` |
//! | `V`              | `[B, H, K_len, D_v]` |
//! | `block_pattern`  | `[B, H, num_blocks_q * num_blocks_k]` (`u8`) |
//! | `y`              | `[B, H, Q_len, D_v]` |
//! | `lse`            | `[B, H, Q_len]` |
//!
//! `num_blocks_q = ceil(Q_len / block_size)`,
//! `num_blocks_k = ceil(K_len / block_size)`.
//!
//! ## Constraints (Tier 1)
//!
//! - FW only. BW deferred (same Tier-1 cadence as FA2 + arbmask).
//! - `block_size ∈ [1, 64]`.
//! - `d_k == d_v ≤ 128`.
//! - Optional causal mask; composes with block pattern (block-pattern
//!   = 0 takes precedence — if a block is masked off-pattern AND
//!   causal-suppressed, it's still skipped first via the pattern check).
//! - Wired dtypes: `{f32, f16, bf16, f64}`.
//!
//! ## Workspace
//!
//! Zero — `lse` is the only saved tensor (for prospective BW; FW
//! doesn't need it as scratch).

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Maximum supported block size for the Tier-1 block-sparse trailblazer.
/// Capped at 64 by the dynamic SMEM budget at `d_k = d_v = 128`.
pub const SDPA_BLOCK_SPARSE_MAX_BLOCK: i32 = 64;

/// Maximum supported head dimension (matches `FLASH_SDPA_MAX_D`).
pub const SDPA_BLOCK_SPARSE_MAX_D: i32 = 128;

/// Descriptor for a block-sparse SDPA forward op.
#[derive(Copy, Clone, Debug)]
pub struct SdpaBlockSparseDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Number of attention heads (`H`).
    pub num_heads: i32,
    /// Query sequence length (`Q_len`).
    pub query_len: i32,
    /// Key / value sequence length (`K_len`).
    pub key_len: i32,
    /// Head dimension of Q and K (`D_k`).
    pub d_k: i32,
    /// Head dimension of V (`D_v`). Must equal `d_k` in Tier 1.
    pub d_v: i32,
    /// Block size (== `Br == Bc`). Typical values: 32, 64. Must be in
    /// `1..=SDPA_BLOCK_SPARSE_MAX_BLOCK`.
    pub block_size: i32,
    /// Score scaling factor — typically `1.0 / sqrt(d_k)`.
    pub scale: f32,
    /// Apply upper-triangular causal mask inside the scores kernel.
    /// Composes with `block_pattern` (block-pattern check takes
    /// precedence; causal-suppressed cells inside an active block are
    /// still set to `-INF`).
    pub is_causal: bool,
    /// Element type — must match the plan's type parameter.
    pub element: ElementKind,
}

/// Args bundle for a block-sparse SDPA forward launch.
pub struct SdpaBlockSparseArgs<'a, T: Element> {
    /// Query — `[B, H, Q_len, D_k]`, contiguous.
    pub q: TensorRef<'a, T, 4>,
    /// Key — `[B, H, K_len, D_k]`, contiguous.
    pub k: TensorRef<'a, T, 4>,
    /// Value — `[B, H, K_len, D_v]`, contiguous.
    pub v: TensorRef<'a, T, 4>,
    /// Per-block boolean pattern — shape
    /// `[B, H, num_blocks_q * num_blocks_k]`, `u8` cells. `0` = skip
    /// the block; non-zero = compute it. Row-major contiguous.
    pub block_pattern: TensorRef<'a, u8, 3>,
    /// Output — `[B, H, Q_len, D_v]`, contiguous.
    pub y: TensorMut<'a, T, 4>,
    /// Log-sum-exp save — `[B, H, Q_len]`, contiguous. Saved for the
    /// prospective Tier-2 BW pass.
    pub lse: TensorMut<'a, T, 3>,
}

/// Block-sparse scaled-dot-product attention forward plan.
///
/// **When to use**: long-context attention with a known sparse mask
/// pattern (sliding-window with sinks, BigBird-style global+local,
/// dilated attention, etc.). The block-level skipping is the value
/// prop — at 75% sparsity the kernel does ~4× less work than the dense
/// path.
///
/// **Dtypes**: `f32`, `f16`, `bf16`, `f64`. f16/bf16 accumulate in f32.
///
/// **Workspace**: zero.
///
/// **Precision guarantee**: deterministic; bit-stable on the same
/// hardware. No atomicAdd anywhere (one CUDA block per `(b, h, qb)`
/// owns its output rows entirely).
pub struct SdpaBlockSparsePlan<T: Element> {
    desc: SdpaBlockSparseDescriptor,
    sku: KernelSku,
    num_blocks_q: i32,
    num_blocks_k: i32,
    _marker: PhantomData<T>,
}

impl<T: Element> SdpaBlockSparsePlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &SdpaBlockSparseDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaBlockSparsePlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0
            || desc.num_heads < 0
            || desc.query_len < 0
            || desc.key_len < 0
            || desc.d_k < 0
            || desc.d_v < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBlockSparsePlan: extents must be non-negative",
            ));
        }
        if desc.d_k != desc.d_v {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaBlockSparsePlan: Tier 1 requires d_k == d_v",
            ));
        }
        if desc.d_k > SDPA_BLOCK_SPARSE_MAX_D {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaBlockSparsePlan: d_k > 128 unsupported in Tier 1",
            ));
        }
        if desc.block_size <= 0 || desc.block_size > SDPA_BLOCK_SPARSE_MAX_BLOCK {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaBlockSparsePlan: block_size must be in 1..=64",
            ));
        }
        if !desc.scale.is_finite() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBlockSparsePlan: scale must be finite",
            ));
        }
        let dtype_in_scope = matches!(
            T::KIND,
            ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16 | ElementKind::F64
        );
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaBlockSparsePlan: wired today: `{f32, f16, bf16, f64}`",
            ));
        }

        // Pre-flight C-side guard.
        #[cfg(feature = "xformers_blocksparse")]
        {
            let probe = unsafe {
                match T::KIND {
                    ElementKind::F32 =>
                        baracuda_kernels_sys::baracuda_kernels_sdpa_f32_block_sparse_can_implement(
                            desc.batch_size, desc.num_heads, desc.query_len, desc.key_len,
                            desc.d_k, desc.d_v, desc.block_size,
                        ),
                    ElementKind::F16 =>
                        baracuda_kernels_sys::baracuda_kernels_sdpa_f16_block_sparse_can_implement(
                            desc.batch_size, desc.num_heads, desc.query_len, desc.key_len,
                            desc.d_k, desc.d_v, desc.block_size,
                        ),
                    ElementKind::Bf16 =>
                        baracuda_kernels_sys::baracuda_kernels_sdpa_bf16_block_sparse_can_implement(
                            desc.batch_size, desc.num_heads, desc.query_len, desc.key_len,
                            desc.d_k, desc.d_v, desc.block_size,
                        ),
                    ElementKind::F64 =>
                        baracuda_kernels_sys::baracuda_kernels_sdpa_f64_block_sparse_can_implement(
                            desc.batch_size, desc.num_heads, desc.query_len, desc.key_len,
                            desc.d_k, desc.d_v, desc.block_size,
                        ),
                    _ => 3,
                }
            };
            map_status(probe)?;
        }

        let num_blocks_q = (desc.query_len + desc.block_size - 1) / desc.block_size;
        let num_blocks_k = (desc.key_len + desc.block_size - 1) / desc.block_size;

        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // One block per (b, h, qb); deterministic per-row online
            // softmax; no atomicAdd.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::BlockSparseAttention as u16,
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
            num_blocks_q,
            num_blocks_k,
            _marker: PhantomData,
        })
    }

    /// Validate args against the descriptor.
    pub fn can_implement(&self, args: &SdpaBlockSparseArgs<'_, T>) -> Result<()> {
        let shape_q = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
            self.desc.d_k,
        ];
        let shape_k = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.key_len,
            self.desc.d_k,
        ];
        let shape_v = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.key_len,
            self.desc.d_v,
        ];
        let shape_y = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
            self.desc.d_v,
        ];
        let shape_lse = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
        ];
        let shape_bp = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.num_blocks_q * self.num_blocks_k,
        ];
        if args.q.shape != shape_q {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBlockSparsePlan: Q shape mismatch",
            ));
        }
        if args.k.shape != shape_k {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBlockSparsePlan: K shape mismatch",
            ));
        }
        if args.v.shape != shape_v {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBlockSparsePlan: V shape mismatch",
            ));
        }
        if args.y.shape != shape_y {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBlockSparsePlan: y shape mismatch",
            ));
        }
        if args.lse.shape != shape_lse {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBlockSparsePlan: lse shape mismatch",
            ));
        }
        if args.block_pattern.shape != shape_bp {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::SdpaBlockSparsePlan: block_pattern shape mismatch \
                 (expected [B, H, num_blocks_q * num_blocks_k])",
            ));
        }
        if !args.q.is_contiguous()
            || !args.k.is_contiguous()
            || !args.v.is_contiguous()
            || !args.y.is_contiguous()
            || !args.lse.is_contiguous()
            || !args.block_pattern.is_contiguous()
        {
            return Err(Error::Unsupported(
                "baracuda-kernels::SdpaBlockSparsePlan: all tensors must be contiguous in Tier 1",
            ));
        }
        Ok(())
    }

    /// Workspace size in bytes — zero.
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

    /// Number of `Q`-side blocks (= `ceil(query_len / block_size)`).
    #[inline]
    pub fn num_blocks_q(&self) -> i32 {
        self.num_blocks_q
    }

    /// Number of `K`-side blocks (= `ceil(key_len / block_size)`).
    #[inline]
    pub fn num_blocks_k(&self) -> i32 {
        self.num_blocks_k
    }

    /// Launch the kernel.
    pub fn run(
        &self,
        stream: &Stream,
        _workspace: Workspace<'_>,
        args: SdpaBlockSparseArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.y.numel() == 0 {
            return Ok(());
        }
        #[cfg(feature = "xformers_blocksparse")]
        {
            let stream_ptr = stream.as_raw() as *mut c_void;
            let q_ptr = args.q.data.as_raw().0 as *const c_void;
            let k_ptr = args.k.data.as_raw().0 as *const c_void;
            let v_ptr = args.v.data.as_raw().0 as *const c_void;
            let bp_ptr = args.block_pattern.data.as_raw().0 as *const c_void;
            let y_ptr = args.y.data.as_raw().0 as *mut c_void;
            let lse_ptr = args.lse.data.as_raw().0 as *mut c_void;
            let is_causal_flag = if self.desc.is_causal { 1 } else { 0 };

            let status = unsafe {
                match T::KIND {
                    ElementKind::F32 =>
                        baracuda_kernels_sys::baracuda_kernels_sdpa_f32_block_sparse_run(
                            self.desc.batch_size, self.desc.num_heads,
                            self.desc.query_len, self.desc.key_len,
                            self.desc.d_k, self.desc.d_v, self.desc.block_size,
                            self.desc.scale, is_causal_flag,
                            q_ptr, k_ptr, v_ptr, bp_ptr,
                            y_ptr, lse_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        ),
                    ElementKind::F16 =>
                        baracuda_kernels_sys::baracuda_kernels_sdpa_f16_block_sparse_run(
                            self.desc.batch_size, self.desc.num_heads,
                            self.desc.query_len, self.desc.key_len,
                            self.desc.d_k, self.desc.d_v, self.desc.block_size,
                            self.desc.scale, is_causal_flag,
                            q_ptr, k_ptr, v_ptr, bp_ptr,
                            y_ptr, lse_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        ),
                    ElementKind::Bf16 =>
                        baracuda_kernels_sys::baracuda_kernels_sdpa_bf16_block_sparse_run(
                            self.desc.batch_size, self.desc.num_heads,
                            self.desc.query_len, self.desc.key_len,
                            self.desc.d_k, self.desc.d_v, self.desc.block_size,
                            self.desc.scale, is_causal_flag,
                            q_ptr, k_ptr, v_ptr, bp_ptr,
                            y_ptr, lse_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        ),
                    ElementKind::F64 =>
                        baracuda_kernels_sys::baracuda_kernels_sdpa_f64_block_sparse_run(
                            self.desc.batch_size, self.desc.num_heads,
                            self.desc.query_len, self.desc.key_len,
                            self.desc.d_k, self.desc.d_v, self.desc.block_size,
                            self.desc.scale, is_causal_flag,
                            q_ptr, k_ptr, v_ptr, bp_ptr,
                            y_ptr, lse_ptr,
                            core::ptr::null_mut(), 0, stream_ptr,
                        ),
                    _ => return Err(Error::Unsupported(
                        "baracuda-kernels::SdpaBlockSparsePlan::run reached an unimplemented dtype",
                    )),
                }
            };
            map_status(status)
        }
        #[cfg(not(feature = "xformers_blocksparse"))]
        {
            let _ = stream;
            Err(Error::Unsupported(
                "baracuda-kernels::SdpaBlockSparsePlan: build with the `xformers_blocksparse` cargo feature",
            ))
        }
    }
}
