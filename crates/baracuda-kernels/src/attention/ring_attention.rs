//! Ring Attention — sequence-parallel attention (Phase 56, Tier 1).
//!
//! Hand-port of the algorithm from:
//!
//! - **Liu, H., Yan, M., & Abbeel, P.** (2023). *Ring Attention with
//!   Blockwise Transformers for Near-Infinite Context.* NeurIPS 2023.
//!   arXiv:2310.01889.
//! - Reference implementation: <https://github.com/lhao499/RingAttention>
//!   (Apache-2.0 with §3 patent grant; JAX, not vendored — clean-room
//!   CUDA port).
//!
//! ## Motivation
//!
//! Standard attention is O(N²) in memory and compute. FlashAttention
//! (Phase 6.6) drops the memory term to O(N) by streaming the softmax.
//! Ring Attention drops the memory term further to **O(N/P)** where
//! `P = world_size` by sharding the Q tensor along its sequence axis
//! and rotating K/V chunks around a NCCL ring. This unlocks
//! million-token context on `P`-GPU clusters where a single GPU could
//! never hold the full attention state.
//!
//! Ring Attention is complementary to tensor-parallelism (TP): Q-slices
//! shard the sequence dim; TP shards the head dim. They compose
//! naturally (a future phase wires this up).
//!
//! ## Algorithm summary
//!
//! Per rank `r ∈ [0, P)`:
//!
//! 1. **Owned data**:
//!    - `Q[my_slice]`: `[B, H, Q_local, D]` where `Q_local = Q_total / P`.
//!    - `K_local`: `[B, H, K_chunk, D]` — initially the rank's own K
//!      chunk; rotates each step.
//!    - `V_local`: `[B, H, K_chunk, D]` — initially the rank's own V
//!      chunk; rotates each step.
//!
//! 2. **Per-step kernel** folds the resident K/V chunk's contribution
//!    into a persistent online-softmax accumulator
//!    `(o_acc, m_acc, l_acc)` (Tri Dao 2022 streaming-softmax math).
//!
//! 3. **NCCL ring rotation** (bidirectional, batched in a
//!    `group_start` / `group_end` bracket): send current K/V to
//!    `(rank + 1) % world_size`, recv next K/V from
//!    `(rank - 1 + world_size) % world_size`.
//!
//! 4. After `world_size` iterations, every rank's accumulator holds
//!    the partial sums for the FULL global K/V range against its local
//!    Q slice. A finalize kernel divides `o_acc / l_acc` and emits the
//!    final `y` in the operand dtype.
//!
//! ```text
//! for step in 0..world_size:
//!     k_global_base = ((rank - step + P) % P) * K_chunk_size
//!     step_kernel(Q[my_slice], K_local, V_local,
//!                 o_acc, m_acc, l_acc,
//!                 q_global_base=rank * Q_local,
//!                 k_global_base=k_global_base)
//!     if step + 1 < world_size:
//!         group_start()
//!         send(K_local, V_local) → (rank + 1) % P
//!         recv(K_local, V_local) ← (rank - 1 + P) % P
//!         group_end()
//! finalize(o_acc, m_acc, l_acc → y)
//! ```
//!
//! ## Causal masking
//!
//! Causal cells are masked on **global** indices. Each step kernel
//! receives `q_global_base` and `k_global_base` as launch parameters;
//! the kernel applies `q_idx_abs > k_idx_abs → mask` consistently
//! regardless of which rotation step is active. Whole-block early-exit
//! is preserved for chunks whose global K range is entirely past every
//! owned Q's index.
//!
//! ## Tier 1 scope
//!
//! - **FW only** (BW deferred to Tier 2).
//! - **f16 / bf16** dtypes (f32 / f64 deferred to Tier 2).
//! - **head_dim = 128** (Tier 1; the kernel reads `D` as a runtime
//!   parameter but the launcher rejects anything else for clarity).
//! - **No GQA broadcast** (deferred to Tier 2).
//! - **No arbitrary additive mask** (deferred to Tier 2 — composes
//!   with the Phase 51 arbmask path conceptually).
//! - **Caller owns the NCCL `Communicator`** — the plan takes a
//!   `&Communicator` and does not manage NCCL state. This keeps the
//!   plan stateless and lets the caller integrate Ring Attention into
//!   an existing process-group setup.
//!
//! ## Single-rank degenerate case
//!
//! When `world_size == 1` the ring rotation is a no-op — the algorithm
//! reduces to standard FlashAttention. This is the validation path for
//! the kernel math on single-GPU hardware. The smoke test compares the
//! single-rank Ring Attention output against
//! [`crate::FlashSdpaPlan`] as ground truth.
//!
//! ## NCCL coordination contract
//!
//! - The plan's [`RingAttentionPlan::run`] takes a `&Communicator` arg
//!   alongside the workspace. Caller owns the communicator's lifetime
//!   and rank/world_size identity.
//! - K/V buffers are rotated **in place** — the plan ping-pongs
//!   between two scratch buffers (owned by the caller via the
//!   `kv_scratch` field of [`RingAttentionArgs`]) to avoid an extra
//!   global-memory roundtrip per step.
//! - The plan calls `Communicator::group_start` / `send` / `recv` /
//!   `group_end` internally for the K/V rotation. On error the plan
//!   surfaces a `Error::CutlassInternal(7000)` wrapper code; the raw
//!   `ncclResult_t` is logged but not threaded into baracuda-cutlass's
//!   typed error set.
//!
//! ## Workspace
//!
//! Caller-supplied via [`RingAttentionArgs::accumulator_scratch`]:
//! `batch * heads * q_local * (d * 4 + 4 + 4)` bytes of f32 storage
//! for the persistent `(o_acc, m_acc, l_acc)` state. Query the exact
//! byte count via [`RingAttentionPlan::accumulator_scratch_bytes`].

use core::marker::PhantomData;

use baracuda_cutlass::{Error, Result};
use baracuda_driver::{DeviceSliceMut, Stream};
use baracuda_kernels_types::{
    ArchSku, AttentionKind, BackendKind, Element, ElementKind, KernelSku, MathPrecision,
    OpCategory, PlanPreference, PrecisionGuarantee, TensorMut, TensorRef,
};

#[cfg(feature = "ring_attention")]
use core::ffi::c_void;

#[cfg(feature = "ring_attention")]
use super::map_status;

/// Tier-1 fixed head dimension. The launcher rejects anything else.
pub const RING_ATTENTION_HEAD_DIM: i32 = 128;

/// Descriptor for a Ring Attention forward op.
///
/// Note that `query_len` / `key_len` here are the **per-rank** extents
/// — i.e. `q_local` (Q slice owned by this rank) and `k_chunk`
/// (K/V chunk size that rotates around the ring). The global sequence
/// length implied by the descriptor is `world_size * key_len`.
#[derive(Copy, Clone, Debug)]
pub struct RingAttentionDescriptor {
    /// Batch size (`B`).
    pub batch_size: i32,
    /// Number of attention heads (`H`).
    pub num_heads: i32,
    /// Per-rank query sequence length (`Q_local = Q_total / world_size`).
    pub query_len: i32,
    /// Per-rank K/V chunk size (`K_chunk`). The global K/V sequence
    /// length is `world_size * key_len`.
    pub key_len: i32,
    /// Head dimension. Tier 1 requires `head_dim == 128`.
    pub head_dim: i32,
    /// Score scaling factor — typically `1.0 / sqrt(head_dim)`.
    pub scale: f32,
    /// Apply causal mask on **global** indices across rotation steps.
    pub is_causal: bool,
    /// Element type — must match the plan's type parameter.
    /// Tier 1: `F16` or `Bf16`.
    pub element: ElementKind,
}

/// Args bundle for a Ring Attention forward launch.
///
/// The caller supplies all device buffers; the plan does NOT allocate.
///
/// **K/V scratch staging contract**: the caller MUST pre-stage this
/// rank's initial K chunk + V chunk concatenated into `kv_scratch_a`
/// before the launch (K first, then V). Layout is
/// `[k_chunk_elems, v_chunk_elems]` where
/// `k_chunk_elems = v_chunk_elems = B * H * K_chunk * D`. This keeps
/// the plan zero-copy on the K/V data — no D2D copies, just NCCL
/// `send`/`recv` ping-pong between `kv_scratch_a` and `kv_scratch_b`.
pub struct RingAttentionArgs<'a, T: Element> {
    /// Query tensor — shape `[B, H, Q_local, D]`, contiguous.
    pub q: TensorRef<'a, T, 4>,
    /// Output tensor — shape `[B, H, Q_local, D]`, contiguous. Holds
    /// the per-rank Q slice's attention output against the FULL global
    /// K/V range after the ring completes.
    pub y: TensorMut<'a, T, 4>,
    /// Optional saved LSE — shape `[B, H, Q_local]`, contiguous in
    /// operand dtype. Holds `m + log(l)` after finalize; consumed by a
    /// future Ring Attention BW pass (Tier 2). Pass `None` for FW-only
    /// inference.
    pub lse: Option<TensorMut<'a, T, 3>>,
    /// First K/V scratch buffer — `2 * B * H * K_chunk * D` elements
    /// of `T`. The plan reads/writes this across rotation steps.
    /// MUST be pre-staged by the caller with the rank's initial
    /// K chunk + V chunk (K first then V, contiguous). See struct doc.
    pub kv_scratch_a: DeviceSliceMut<'a, T>,
    /// Second K/V scratch buffer — same layout as `kv_scratch_a`. Used
    /// as the receive-side ping-pong target during rotation. Initial
    /// contents don't matter (overwritten on the first NCCL recv).
    pub kv_scratch_b: DeviceSliceMut<'a, T>,
    /// Persistent f32 accumulator scratch — see
    /// [`RingAttentionPlan::accumulator_scratch_bytes`] for sizing.
    /// Holds `(o_acc, m_acc, l_acc)` concatenated.
    pub accumulator_scratch: DeviceSliceMut<'a, u8>,
}

/// Ring Attention forward plan.
///
/// See the module-level documentation for the algorithm narrative,
/// causal-masking semantics, and NCCL coordination contract.
///
/// **When to use**: long-context attention split across `P > 1` GPUs.
/// On single-GPU hardware (`world_size == 1`) the plan still works
/// (and is the test/validation path) but [`crate::FlashSdpaPlan`] is
/// strictly faster — it skips the rotation-loop overhead.
///
/// **Dtypes**: `f16`, `bf16` (Tier 1). `f32` / `f64` deferred to
/// Tier 2.
///
/// **Shape limits**: `head_dim == 128`; `Br = Bc = 64` tile (per-step
/// kernel inherits Phase 6.6 FlashAttention tile geometry); arbitrary
/// `Q_local` and `K_chunk`.
///
/// **Workspace**: caller-supplied f32 scratch for the persistent
/// `(o_acc, m_acc, l_acc)` accumulator + two K/V ping-pong buffers
/// (see [`RingAttentionArgs`]). The plan itself allocates nothing.
///
/// **Precision guarantee**: deterministic per-rank; bit-stable on the
/// same hardware AND same `(world_size, rank)` pair. Across different
/// world_size values the float-order changes (more rotation steps =
/// different summation order) so results are NOT bit-identical across
/// different cluster sizes. Comparable to within typical f16/bf16
/// streaming-softmax tolerances.
pub struct RingAttentionPlan<T: Element> {
    desc: RingAttentionDescriptor,
    sku: KernelSku,
    _marker: PhantomData<T>,
}

impl<T: Element> RingAttentionPlan<T> {
    /// Pick a kernel.
    pub fn select(
        _stream: &Stream,
        desc: &RingAttentionDescriptor,
        _pref: PlanPreference,
    ) -> Result<Self> {
        if desc.element != T::KIND {
            return Err(Error::Unsupported(
                "baracuda-kernels::RingAttentionPlan: descriptor element != T",
            ));
        }
        if desc.batch_size < 0
            || desc.num_heads < 0
            || desc.query_len < 0
            || desc.key_len < 0
            || desc.head_dim < 0
        {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RingAttentionPlan: extents must be non-negative",
            ));
        }
        if !desc.scale.is_finite() {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RingAttentionPlan: scale must be finite",
            ));
        }
        if desc.head_dim != RING_ATTENTION_HEAD_DIM {
            return Err(Error::Unsupported(
                "baracuda-kernels::RingAttentionPlan: Tier 1 requires head_dim == 128",
            ));
        }
        let dtype_in_scope = matches!(T::KIND, ElementKind::F16 | ElementKind::Bf16);
        if !dtype_in_scope {
            return Err(Error::Unsupported(
                "baracuda-kernels::RingAttentionPlan: Tier 1 wired for `{f16, bf16}`",
            ));
        }
        let precision_guarantee = PrecisionGuarantee {
            math_precision: MathPrecision::F32,
            accumulator: ElementKind::F32,
            // Single-rank: bit-stable on same hardware. Multi-rank:
            // bit-stable on same hardware + same world_size + same rank.
            // Different world_size changes the rotation-step count and
            // hence the float-summation order.
            bit_stable_on_same_hardware: true,
            deterministic: true,
        };
        let sku = KernelSku {
            category: OpCategory::Attention,
            op: AttentionKind::RingAttention as u16,
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
    pub fn can_implement(&self, args: &RingAttentionArgs<'_, T>) -> Result<()> {
        let shape_q = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
            self.desc.head_dim,
        ];
        let shape_y = shape_q;
        let shape_lse = [
            self.desc.batch_size,
            self.desc.num_heads,
            self.desc.query_len,
        ];
        if args.q.shape != shape_q {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RingAttentionPlan: Q shape mismatch",
            ));
        }
        if args.y.shape != shape_y {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RingAttentionPlan: y shape mismatch",
            ));
        }
        if let Some(lse) = args.lse.as_ref() {
            if lse.shape != shape_lse {
                return Err(Error::InvalidProblem(
                    "baracuda-kernels::RingAttentionPlan: lse shape must be [B, H, Q_local]",
                ));
            }
            if !lse.is_contiguous() {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RingAttentionPlan: lse must be contiguous",
                ));
            }
        }
        if !args.q.is_contiguous() || !args.y.is_contiguous() {
            return Err(Error::Unsupported(
                "baracuda-kernels::RingAttentionPlan: Tier 1 requires contiguous tensors",
            ));
        }
        let kv_scratch_min = self.kv_scratch_elements();
        if args.kv_scratch_a.len() < kv_scratch_min {
            return Err(Error::BufferTooSmall {
                needed: kv_scratch_min,
                got: args.kv_scratch_a.len(),
            });
        }
        if args.kv_scratch_b.len() < kv_scratch_min {
            return Err(Error::BufferTooSmall {
                needed: kv_scratch_min,
                got: args.kv_scratch_b.len(),
            });
        }
        let acc_min = self.accumulator_scratch_bytes();
        if args.accumulator_scratch.len() < acc_min {
            return Err(Error::WorkspaceTooSmall {
                needed: acc_min,
                got: args.accumulator_scratch.len(),
            });
        }
        Ok(())
    }

    /// Number of `T` elements needed per K/V scratch buffer.
    ///
    /// Each scratch holds one full K chunk + one full V chunk for this
    /// rank: `2 * B * H * K_chunk * D` elements.
    #[inline]
    pub fn kv_scratch_elements(&self) -> usize {
        let n = (self.desc.batch_size as i64)
            * (self.desc.num_heads as i64)
            * (self.desc.key_len as i64)
            * (self.desc.head_dim as i64)
            * 2;
        n.max(0) as usize
    }

    /// Bytes needed for the persistent f32 accumulator scratch
    /// (`o_acc + m_acc + l_acc`).
    ///
    /// Layout: `o_acc` (`B * H * Q_local * D` × f32) followed by
    /// `m_acc` (`B * H * Q_local` × f32) followed by `l_acc`
    /// (`B * H * Q_local` × f32).
    #[inline]
    pub fn accumulator_scratch_bytes(&self) -> usize {
        let bhq = (self.desc.batch_size as i64)
            * (self.desc.num_heads as i64)
            * (self.desc.query_len as i64);
        let o = bhq * (self.desc.head_dim as i64);
        let ml = bhq;
        ((o + 2 * ml) * 4).max(0) as usize
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

    /// Launch the Ring Attention forward pass.
    ///
    /// Orchestrates the per-step kernel + NCCL K/V rotation loop and
    /// the finalize kernel. The caller supplies the NCCL communicator;
    /// the plan does not manage NCCL state.
    ///
    /// **Single-rank degenerate case** (`comm.world_size() == 1`):
    /// the rotation loop is a no-op — the plan runs the step kernel
    /// once with `q_global_base = 0`, `k_global_base = 0` and then
    /// finalizes. The result is mathematically equivalent to
    /// [`crate::FlashSdpaPlan`] (different float order so not
    /// bit-identical, but within streaming-softmax tolerance).
    #[cfg(feature = "ring_attention")]
    pub fn run(
        &self,
        stream: &Stream,
        comm: &baracuda_nccl::Communicator,
        mut args: RingAttentionArgs<'_, T>,
    ) -> Result<()> {
        self.can_implement(&args)?;
        if args.y.numel() == 0 {
            return Ok(());
        }
        let world_size = comm.world_size();
        let rank = comm.rank();
        if world_size < 1 {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RingAttentionPlan: world_size must be >= 1",
            ));
        }
        if rank < 0 || rank >= world_size {
            return Err(Error::InvalidProblem(
                "baracuda-kernels::RingAttentionPlan: rank out of range",
            ));
        }

        let stream_ptr = stream.as_raw() as *mut c_void;
        let kv_scratch_elems = self.kv_scratch_elements();
        let k_chunk_elems = kv_scratch_elems / 2;
        // Caller has pre-staged this rank's initial K/V into
        // kv_scratch_a (K first then V — see RingAttentionArgs doc).
        // The rotation loop ping-pongs from there.

        // Carve out o_acc / m_acc / l_acc views over the accumulator scratch.
        let bhq = (self.desc.batch_size as usize)
            * (self.desc.num_heads as usize)
            * (self.desc.query_len as usize);
        let o_len_f32 = bhq * (self.desc.head_dim as usize);
        let ml_len_f32 = bhq;
        let o_bytes = o_len_f32 * 4;
        let m_bytes = ml_len_f32 * 4;
        let acc_raw = args.accumulator_scratch.as_raw().0 as *mut u8;
        let o_ptr = acc_raw as *mut c_void;
        let m_ptr = unsafe { acc_raw.add(o_bytes) } as *mut c_void;
        let l_ptr = unsafe { acc_raw.add(o_bytes + m_bytes) } as *mut c_void;

        // Zero accumulators.
        let init_status = unsafe {
            baracuda_kernels_sys::baracuda_kernels_ring_attention_init_run(
                o_ptr,
                m_ptr,
                l_ptr,
                o_len_f32 as i64,
                ml_len_f32 as i64,
                stream_ptr,
            )
        };
        map_status(init_status)?;

        // Ring rotation loop. `current_in_a` tracks which scratch holds
        // the currently-resident K/V chunk. Initial chunks are in
        // kv_scratch_a (we staged them there above).
        let mut current_in_a = true;
        let q_global_base = rank * self.desc.query_len;
        let q_ptr = args.q.data.as_raw().0 as *const c_void;

        for step in 0..world_size {
            // Which rank originally owned the chunk that's resident now?
            let origin_rank = (rank - step + world_size) % world_size;
            let k_global_base = origin_rank * self.desc.key_len;

            // Resolve current K/V pointers.
            let (cur_ptr, _other_ptr) = if current_in_a {
                (
                    args.kv_scratch_a.as_raw().0 as *mut c_void,
                    args.kv_scratch_b.as_raw().0 as *mut c_void,
                )
            } else {
                (
                    args.kv_scratch_b.as_raw().0 as *mut c_void,
                    args.kv_scratch_a.as_raw().0 as *mut c_void,
                )
            };
            let k_cur = cur_ptr as *const c_void;
            // V follows K in the scratch layout.
            let v_cur = unsafe {
                (cur_ptr as *mut u8).add(k_chunk_elems * core::mem::size_of::<T>())
                    as *const c_void
            };

            // Launch the per-step kernel.
            let step_status = match T::KIND {
                ElementKind::F16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_ring_attention_f16_step_run(
                        self.desc.batch_size,
                        self.desc.num_heads,
                        self.desc.query_len,
                        self.desc.key_len,
                        self.desc.head_dim,
                        q_global_base,
                        k_global_base,
                        self.desc.scale,
                        if self.desc.is_causal { 1 } else { 0 },
                        q_ptr,
                        k_cur,
                        v_cur,
                        o_ptr,
                        m_ptr,
                        l_ptr,
                        stream_ptr,
                    )
                },
                ElementKind::Bf16 => unsafe {
                    baracuda_kernels_sys::baracuda_kernels_ring_attention_bf16_step_run(
                        self.desc.batch_size,
                        self.desc.num_heads,
                        self.desc.query_len,
                        self.desc.key_len,
                        self.desc.head_dim,
                        q_global_base,
                        k_global_base,
                        self.desc.scale,
                        if self.desc.is_causal { 1 } else { 0 },
                        q_ptr,
                        k_cur,
                        v_cur,
                        o_ptr,
                        m_ptr,
                        l_ptr,
                        stream_ptr,
                    )
                },
                _ => {
                    return Err(Error::Unsupported(
                        "baracuda-kernels::RingAttentionPlan: dtype not in {f16, bf16}",
                    ));
                }
            };
            map_status(step_status)?;

            // Rotate K/V to the next rank (skip on the final step).
            if step + 1 < world_size {
                let next_peer = (rank + 1) % world_size;
                let prev_peer = (rank - 1 + world_size) % world_size;
                rotate_kv(
                    comm,
                    stream,
                    current_in_a,
                    &mut args.kv_scratch_a,
                    &mut args.kv_scratch_b,
                    k_chunk_elems,
                    next_peer,
                    prev_peer,
                )?;
                current_in_a = !current_in_a;
            }
        }

        // Finalize: divide o_acc by l_acc and emit y (+ optional lse).
        let y_ptr = args.y.data.as_raw().0 as *mut c_void;
        let lse_ptr = args
            .lse
            .as_mut()
            .map(|t| t.data.as_raw().0 as *mut c_void)
            .unwrap_or(core::ptr::null_mut());
        let fin_status = match T::KIND {
            ElementKind::F16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ring_attention_f16_finalize_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.head_dim,
                    o_ptr,
                    m_ptr,
                    l_ptr,
                    y_ptr,
                    lse_ptr,
                    stream_ptr,
                )
            },
            ElementKind::Bf16 => unsafe {
                baracuda_kernels_sys::baracuda_kernels_ring_attention_bf16_finalize_run(
                    self.desc.batch_size,
                    self.desc.num_heads,
                    self.desc.query_len,
                    self.desc.head_dim,
                    o_ptr,
                    m_ptr,
                    l_ptr,
                    y_ptr,
                    lse_ptr,
                    stream_ptr,
                )
            },
            _ => {
                return Err(Error::Unsupported(
                    "baracuda-kernels::RingAttentionPlan: dtype not in {f16, bf16}",
                ));
            }
        };
        map_status(fin_status)?;
        Ok(())
    }
}

/// Helper: rotate K/V around the ring. The current chunk lives in
/// `kv_scratch_a` (when `current_in_a` is true) or `kv_scratch_b`;
/// after the rotation the new chunk is in the OTHER buffer.
///
/// Uses NCCL `group_start` / `send` / `recv` / `group_end` to batch
/// the bidirectional point-to-point. K and V are sent as a single
/// concatenated chunk (the scratch layout puts them back-to-back).
#[cfg(feature = "ring_attention")]
#[allow(clippy::too_many_arguments)]
fn rotate_kv<T: Element>(
    comm: &baracuda_nccl::Communicator,
    stream: &Stream,
    current_in_a: bool,
    scratch_a: &mut DeviceSliceMut<'_, T>,
    scratch_b: &mut DeviceSliceMut<'_, T>,
    k_chunk_elems: usize,
    next_peer: i32,
    prev_peer: i32,
) -> Result<()> {
    // SAFETY: we re-frame the existing scratch slices as transient
    // DeviceBuffer views just to satisfy the NCCL API. The send buffer
    // is the currently-resident chunk; the recv buffer is the other
    // scratch.
    //
    // NCCL `send` / `recv` take `&DeviceBuffer<T>` references, but
    // baracuda's NCCL wrapper accesses them via `as_raw()`, so the
    // important contract is that the device pointer + element count
    // are correct.
    //
    // We work around the type signature mismatch (DeviceSliceMut vs
    // DeviceBuffer) by reframing through the raw-ptr / count APIs.
    // K and V are concatenated in scratch, so 2 * k_chunk_elems is
    // the total transfer size per rotation.
    let total_elems = 2 * k_chunk_elems;

    // Reframe as ad-hoc DeviceBuffer views via the back door — we
    // can't directly call NCCL with a DeviceSliceMut. The pattern used
    // by other distributed code is to allocate a DeviceBuffer and
    // call the safe API. Here we need zero-copy, so we use the raw
    // NCCL FFI directly through the `baracuda_nccl_sys` crate.
    use baracuda_nccl_sys::{nccl, ncclDataType_t};
    let n = nccl().map_err(|_| {
        Error::Unsupported(
            "baracuda-kernels::RingAttentionPlan: NCCL library not available at runtime",
        )
    })?;

    // Determine NCCL dtype tag from T::KIND.
    let dtype = match T::KIND {
        ElementKind::F16 => ncclDataType_t::Float16,
        ElementKind::Bf16 => ncclDataType_t::BFloat16,
        _ => {
            return Err(Error::Unsupported(
                "baracuda-kernels::RingAttentionPlan::rotate_kv: dtype not in {f16, bf16}",
            ));
        }
    };

    let (send_ptr, recv_ptr) = if current_in_a {
        (
            scratch_a.as_raw().0 as *mut c_void,
            scratch_b.as_raw().0 as *mut c_void,
        )
    } else {
        (
            scratch_b.as_raw().0 as *mut c_void,
            scratch_a.as_raw().0 as *mut c_void,
        )
    };

    let group_start = n.nccl_group_start().map_err(|_| {
        Error::CutlassInternal(7000)
    })?;
    let group_end = n.nccl_group_end().map_err(|_| {
        Error::CutlassInternal(7001)
    })?;
    let send_fn = n.nccl_send().map_err(|_| Error::CutlassInternal(7002))?;
    let recv_fn = n.nccl_recv().map_err(|_| Error::CutlassInternal(7003))?;

    let comm_handle = comm.as_raw();
    let stream_raw = stream.as_raw() as _;

    // group_start
    let s = unsafe { group_start() };
    if !s.is_success() {
        return Err(Error::CutlassInternal(7100 + s.0));
    }
    let s = unsafe {
        send_fn(
            send_ptr as *const c_void,
            total_elems,
            dtype,
            next_peer,
            comm_handle,
            stream_raw,
        )
    };
    if !s.is_success() {
        // Try to close the group anyway so NCCL state stays consistent.
        let _ = unsafe { group_end() };
        return Err(Error::CutlassInternal(7200 + s.0));
    }
    let s = unsafe {
        recv_fn(
            recv_ptr,
            total_elems,
            dtype,
            prev_peer,
            comm_handle,
            stream_raw,
        )
    };
    if !s.is_success() {
        let _ = unsafe { group_end() };
        return Err(Error::CutlassInternal(7300 + s.0));
    }
    let s = unsafe { group_end() };
    if !s.is_success() {
        return Err(Error::CutlassInternal(7400 + s.0));
    }
    Ok(())
}

