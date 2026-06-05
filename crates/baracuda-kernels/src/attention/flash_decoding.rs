// SPDX-FileCopyrightText: 2026 Eric Evans and the baracuda contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
//! FlashDecoding — split-K parallel attention decode for `seq_q = 1`.
//!
//! Phase 73 follow-up. Closes the perf gap that both the bespoke
//! `FlashSdpaPlan` Phase 10 trailblazer AND FA2 leave at the decode
//! regime, where the seq_q dimension is too short to fill a 64-row
//! q-tile and most of the GPU sits idle.
//!
//! FlashDecoding flips the parallelism axis: split K into chunks of
//! 256 rows, launch one block per `(b, h, k_split)`, and combine the
//! per-split online-softmax partials in a second small reduction kernel.
//! For (B=1, H=32, K=2048, D=128) the split kernel launches `1 × 32 × 8
//! = 256` blocks vs the FlashAttention kernel's 32 (Q/64=32 × H=32),
//! and each block does meaningful work instead of being mostly
//! q-tile padding.
//!
//! See `kernels/include/baracuda_flash_decoding.cuh` for the kernel
//! body. This file wraps it with the standard descriptor / args / plan
//! triple.
//!
//! ## Tier-1 scope
//!
//! - dtypes: `f16`, `bf16`. f32 / f64 are decode-uncommon (typical
//!   inference is half-precision).
//! - `head_dim ∈ [1, 128]`.
//! - `seq_q == 1` strictly (decode contract — the whole point).
//! - GQA via stride-0 broadcast on K/V's head axis. Pass an actual
//!   `H_k` < H by setting `args.k.stride[1] = 0` (and same for V) with
//!   `args.k.shape[1] = H`; the safe-wrapper computes the right
//!   per-head base offset.
//! - `is_causal` is irrelevant — there's only one query row, and the
//!   caller is responsible for slicing the cache to the prefix it
//!   wants attended to.
//!
//! ## Out of scope (deferred)
//!
//! - f32 / f64 (no decode workload uses these).
//! - sliding window / ALiBi / soft-cap — pre-mask the cache.
//! - BW pass — decode is FW-only.
//! - Tensor-core MMA inside the Q·K dot product — the first cut uses
//!   warp-shuffle reduce in fp32. A tensor-core retune is the next
//!   follow-up phase once perf bench numbers land.
//!
//! ## Workspace
//!
//! Non-zero. The split kernel emits per-`(B, H, S)` partials (m, l, o)
//! into the workspace; the combine kernel reads them and writes the
//! final output. `workspace_size()` returns
//! `B * H * num_splits * (2 + head_dim) * sizeof(f32)`.

use core::ffi::c_void;
use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::Stream;
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef, Workspace,
};

use super::map_status;

/// Maximum head dimension wired in the Tier-1 trailblazer.
pub const FLASH_DECODING_MAX_D: i32 = 128;
const CHUNK_K: i32 = 256;

/// Descriptor for a FlashDecoding op.
///
/// `num_kv_heads` is the GQA grouping signal: when it equals `num_heads`
/// the workload is full MHA; when it's smaller (e.g. 8 for Llama 3 8B
/// at H_q=32) every K/V head is shared by `group_size = num_heads /
/// num_kv_heads` Q heads. The launcher uses `group_size` to pick
/// between the warp-cooperative SIMT kernel (Tier-1) and the
/// GQA-batched WMMA kernel (Tier-2, gated on group_size ≥ 4 +
/// head_dim aligned to 16).
#[derive(Copy, Clone, Debug)]
pub struct FlashDecodingDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Number of query / output heads (`H_q`).
    pub num_heads: i32,
    /// Number of K/V heads (`H_kv`). Must divide `num_heads` evenly.
    /// `num_kv_heads == num_heads` → pure MHA. `num_kv_heads == 1` →
    /// MQA. `num_kv_heads < num_heads && > 1` → GQA.
    pub num_kv_heads: i32,
    /// K/V sequence length (the full attended prefix, not just the new
    /// step). Arbitrary; the split-K factor adapts via [`CHUNK_K`].
    pub k_len: i32,
    /// Per-head feature dimension. `d_q == d_k == d_v` is enforced —
    /// the decode regime doesn't justify the d_k != d_v complication
    /// the prefill kernel handles.
    pub head_dim: i32,
    /// Score scaling factor — typically `1.0 / sqrt(head_dim)`.
    pub scale: f32,
    /// Element type — must match the plan's type parameter.
    pub element: ElementKind,
}

impl FlashDecodingDescriptor {
    /// Convenience constructor for pure MHA (`num_kv_heads == num_heads`)
    /// with the standard `1/sqrt(D)` scale.
    #[inline]
    pub fn new(batch_size: i32, num_heads: i32, k_len: i32, head_dim: i32, element: ElementKind) -> Self {
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        Self {
            batch_size,
            num_heads,
            num_kv_heads: num_heads,
            k_len,
            head_dim,
            scale,
            element,
        }
    }

    /// Convenience constructor for GQA / MQA. `num_kv_heads` must
    /// divide `num_heads`.
    #[inline]
    pub fn new_gqa(
        batch_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        k_len: i32,
        head_dim: i32,
        element: ElementKind,
    ) -> Self {
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        Self {
            batch_size,
            num_heads,
            num_kv_heads,
            k_len,
            head_dim,
            scale,
            element,
        }
    }

    /// Builder: override the score scale (e.g. for QK-norm models that
    /// pre-divide by something other than `sqrt(head_dim)`).
    #[inline]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// GQA group size — number of Q heads sharing each K/V head.
    #[inline]
    pub fn group_size(&self) -> i32 {
        if self.num_kv_heads == 0 {
            0
        } else {
            self.num_heads / self.num_kv_heads
        }
    }
}

/// Args bundle for a FlashDecoding launch.
///
/// Q is rank-3 because `seq_q == 1` is encoded in the descriptor — no
/// need to thread a unit axis through the API.
///
/// K/V take shape `[B, H_kv, K_len, D]` (the PHYSICAL layout, not the
/// broadcast-replicated H_q view). The kernel handles the Q→KV head
/// mapping via integer division `kv_head = q_head / group_size`. For
/// pure MHA the caller just passes `H_kv == H_q` and the same data
/// shape as before.
pub struct FlashDecodingArgs<'a, T: Element> {
    /// Query tensor — shape `[B, H_q, D]`. Arbitrary strides via the
    /// supplied stride array; typical case is contig.
    pub q: TensorRef<'a, T, 3>,
    /// Key tensor — shape `[B, H_kv, K_len, D]`, physical layout.
    pub k: TensorRef<'a, T, 4>,
    /// Value tensor — shape `[B, H_kv, K_len, D]`, physical layout.
    pub v: TensorRef<'a, T, 4>,
    /// Output tensor — shape `[B, H_q, D]`.
    pub y: TensorMut<'a, T, 3>,
}

/// FlashDecoding forward plan (Dao 2023).
///
/// Split-K parallel attention decode for `seq_q = 1`. Replaces both
/// [`FlashSdpaPlan`](crate::FlashSdpaPlan) and FA2 at the decode regime
/// — both of those tile the Q dimension and waste work when seq_q < 64.
///
/// **When to use**: autoregressive decoder inference token loop. After
/// the prefill step (which uses [`FlashSdpaPlan`] with `fa2` for the
/// long initial context), each generated token calls this plan with
/// `seq_q = 1` and the full grown KV cache.
///
/// **Dtypes**: `f16`, `bf16` (the only dtypes inference uses).
///
/// **Shape limits**: `head_dim ≤ 128`. Arbitrary `B`, `H`, `K_len`.
///
/// **Workspace**: non-zero. See [`Self::workspace_size`].
///
/// **Precision guarantee**: f32 accumulators throughout the split AND
/// combine kernels. Deterministic — each output cell is written by
/// exactly one block; no atomicAdd.
pub struct FlashDecodingPlan<T: Element> {
    desc: FlashDecodingDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> FlashDecodingPlan<T> {
    /// Pick a kernel for the supplied descriptor.
    pub fn select(
        _stream: &Stream,
        desc: &FlashDecodingDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashDecodingPlan: descriptor element != T",
            ));
        }
        if desc.batch_size <= 0
            || desc.num_heads <= 0
            || desc.num_kv_heads <= 0
            || desc.k_len < 0
            || desc.head_dim <= 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashDecodingPlan: extents must be positive (k_len may be 0)",
            ));
        }
        if desc.num_heads % desc.num_kv_heads != 0 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::FlashDecodingPlan: num_heads must be a multiple of num_kv_heads",
            ));
        }
        if desc.head_dim > FLASH_DECODING_MAX_D {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashDecodingPlan: head_dim > 128 not supported",
            ));
        }
        if !matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16) {
            return Err(Error::Unsupported(
                "baracuda-kernels::FlashDecodingPlan: wired today: {f16, bf16}",
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
            op: AttentionKind::FlashAttention as u16,
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
    pub fn can_implement(&self, args: &FlashDecodingArgs<'_, T>) -> Result<()> {
        let d = self.desc.head_dim;
        let b = self.desc.batch_size;
        let h_q = self.desc.num_heads;
        let h_kv = self.desc.num_kv_heads;
        let k = self.desc.k_len;

        if args.q.shape != [b, h_q, d] {
            return Err(Error::InvalidProblem(
                "FlashDecodingPlan: q.shape mismatch (expected [B, H_q, D])",
            ));
        }
        if args.y.shape != [b, h_q, d] {
            return Err(Error::InvalidProblem(
                "FlashDecodingPlan: y.shape mismatch (expected [B, H_q, D])",
            ));
        }
        if args.k.shape != [b, h_kv, k, d] {
            return Err(Error::InvalidProblem(
                "FlashDecodingPlan: k.shape mismatch (expected [B, H_kv, K_len, D])",
            ));
        }
        if args.v.shape != [b, h_kv, k, d] {
            return Err(Error::InvalidProblem(
                "FlashDecodingPlan: v.shape mismatch (expected [B, H_kv, K_len, D])",
            ));
        }
        Ok(())
    }

    /// Backend selected by `select`.
    #[inline]
    pub fn backend(&self) -> BackendKind {
        BackendKind::Bespoke
    }

    /// Kernel SKU descriptor.
    #[inline]
    pub fn sku(&self) -> &KernelSku {
        &self.sku
    }

    /// Workspace requirement in bytes for the (split + combine) pipeline.
    pub fn workspace_size(&self) -> usize {
        let b = self.desc.batch_size as i64;
        let h = self.desc.num_heads as i64;
        let s = num_splits(self.desc.k_len) as i64;
        let d = self.desc.head_dim as i64;
        if s == 0 || b == 0 || h == 0 {
            return 0;
        }
        // partial_m + partial_l + partial_o[D] → (2 + D) * f32 per
        // (b, h, split).
        (b * h * s * (2 + d) * 4) as usize
    }

    /// Run the FlashDecoding pipeline.
    pub fn run(
        &self,
        stream: &Stream,
        workspace: Workspace<'_>,
        args: FlashDecodingArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;

        let needed = self.workspace_size();
        let (ws_ptr, ws_bytes) = match workspace {
            Workspace::None => {
                if needed > 0 {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: 0,
                    });
                }
                (core::ptr::null_mut::<c_void>(), 0_usize)
            }
            Workspace::Borrowed(buf) => {
                if buf.len() < needed {
                    return Err(Error::WorkspaceTooSmall {
                        needed,
                        got: buf.len(),
                    });
                }
                (buf.as_raw().0 as *mut c_void, buf.len())
            }
        };

        let stream_ptr = stream.as_raw() as *mut c_void;
        let q_ptr = args.q.data.as_raw().0 as *const c_void;
        let k_ptr = args.k.data.as_raw().0 as *const c_void;
        let v_ptr = args.v.data.as_raw().0 as *const c_void;
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;

        let status = unsafe {
            match T::KIND {
                ElementKind::F16 => baracuda_kernels_sys::baracuda_kernels_flash_decoding_f16_run(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    y_ptr,
                    ws_ptr,
                    ws_bytes,
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.num_kv_heads,
                    self.desc.k_len,
                    self.desc.head_dim,
                    args.q.stride[0],
                    args.q.stride[1],
                    args.k.stride[0],
                    args.k.stride[1],
                    args.k.stride[2],
                    args.v.stride[0],
                    args.v.stride[1],
                    args.v.stride[2],
                    args.y.stride[0],
                    args.y.stride[1],
                    self.desc.scale,
                    stream_ptr,
                ),
                ElementKind::Bf16 => baracuda_kernels_sys::baracuda_kernels_flash_decoding_bf16_run(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    y_ptr,
                    ws_ptr,
                    ws_bytes,
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.num_kv_heads,
                    self.desc.k_len,
                    self.desc.head_dim,
                    args.q.stride[0],
                    args.q.stride[1],
                    args.k.stride[0],
                    args.k.stride[1],
                    args.k.stride[2],
                    args.v.stride[0],
                    args.v.stride[1],
                    args.v.stride[2],
                    args.y.stride[0],
                    args.y.stride[1],
                    self.desc.scale,
                    stream_ptr,
                ),
                _ => {
                    return Err(Error::Unsupported(
                        "baracuda-kernels::FlashDecodingPlan: only f16 / bf16 wired",
                    ));
                }
            }
        };
        map_status(status)
    }
}

#[inline]
fn num_splits(k_len: i32) -> i32 {
    if k_len <= 0 {
        return 0;
    }
    (k_len + CHUNK_K - 1) / CHUNK_K
}
