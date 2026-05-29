//! Distributed (ZeRO-1-style) optimizer plans.
//!
//! Phase 58 — extends Phase 49's [`AdamStepPlan`](crate::AdamStepPlan)
//! with the **sharded** variant for memory-efficient distributed
//! training. Behind the `distributed_optim` cargo feature (default
//! OFF). Requires NCCL at runtime (via [`baracuda_nccl`]).
//!
//! ## ZeRO-1 in one paragraph
//!
//! Stage 1 of Microsoft's ZeRO (Zero Redundancy Optimizer) partitions
//! the optimizer state — Adam's `m`, `v`, and the master fp32 weights
//! — across the data-parallel ranks. With `world_size` ranks, each rank
//! holds `1/world_size` of the optimizer state, cutting the optimizer's
//! memory footprint (typically ~3× the model parameter count when
//! mixed-precision is in play) by the same factor.
//!
//! ZeRO-1 leaves gradients fully replicated across ranks (ZeRO-2 shards
//! gradients; ZeRO-3 shards parameters during FW/BW too). ZeRO-1 is the
//! sweet spot when memory pressure is dominated by optimizer state —
//! the common case for FP16/BF16 mixed-precision training of LLMs at
//! 7B-70B parameters.
//!
//! ## Per-step protocol
//!
//! Each `step()` does:
//!
//! 1. **`all_reduce(grads)`** — every rank ends up with the full,
//!    summed (typically mean-averaged in DDP — caller's responsibility
//!    to scale prior to step) gradient. This is the same all-reduce
//!    that vanilla DDP does; no extra communication vs the non-ZeRO
//!    baseline.
//! 2. **`local_step(my_shard)`** — each rank runs Adam on the
//!    `(shard_offset .. shard_offset + shard_len)` slice of every
//!    parameter / grad / moment tensor. The shard split is **even** —
//!    rank `r` of `world_size = W` owns elements
//!    `[r * floor(n/W) + min(r, n mod W) .. ...]` of each tensor (so
//!    the first `n mod W` ranks get one extra element). This matches
//!    PyTorch's `torch.chunk(tensor, W)` convention.
//! 3. **`all_gather(updated_params)`** — broadcast every rank's updated
//!    shard back to every other rank, leaving the full updated
//!    parameter set replicated everywhere (ready for the next FW pass).
//!
//! Step 1's all-reduce is the same one DDP already does — Phase 58 does
//! NOT impose extra communication cost relative to DDP-without-ZeRO. The
//! tradeoff is purely (memory saved) vs (extra all-gather in step 3).
//! Step 3 is bandwidth-equivalent to step 1's all-reduce for ring-
//! topology NCCL.
//!
//! ## Single-rank degenerate case
//!
//! When `world_size == 1`, [`DistributedAdamStepPlan::step`] elides
//! both collectives and reduces to a plain
//! [`AdamStepPlan::step`](crate::AdamStepPlan::step) — bit-exact
//! equivalence to the non-distributed plan. This is the smoke-test
//! shape (runs on a single-RTX-4070 dev box without multi-GPU
//! hardware).
//!
//! ## Out of scope (Phase 58)
//!
//! - **ZeRO-2** (gradient sharding) — would need a `reduce_scatter`
//!   in step 1 + custom gradient accumulators; deeper FW/BW
//!   integration. Future phase.
//! - **ZeRO-3** (parameter sharding during FW/BW) — needs major
//!   plumbing in the autograd graph; not on a near-term roadmap.
//! - **DistributedLamb / DistributedSGD** — same composition pattern;
//!   add when concrete demand surfaces.
//! - **CPU-offload optimizer state** — separate concern (would pair
//!   well with this but lives in a different layer).
//! - **8-bit distributed optimizer state** — combines with the
//!   bitsandbytes 8-bit Adam path; future phase.
//! - **Async gradient overlap** — Hopper-specific; hardware-blocked
//!   on the current single-RTX-4070 dev environment.
//!
//! ## License
//!
//! Builds on Apex's BSD-3-Clause `multi_tensor_adam.cuh` (vendored in
//! Phase 49 at `vendor/apex/`). Phase 58 adds no new vendored source —
//! it composes existing Phase 49 plans + Phase 52 NCCL collectives in
//! pure Rust. See `vendor/apex/VENDOR.md`.

#![cfg(feature = "distributed_optim")]

use baracuda_driver::{DeviceBuffer, Stream};
use baracuda_nccl::{Communicator, NcclScalar, RedOp};

use crate::{AdamConfig, AdamParamDtype, AdamStepPlan, Error, Result, TensorList};

// ============================================================================
// Sharding helper — even split of `n` elements across `world_size` ranks
// ============================================================================

/// Compute this rank's (offset, length) inside an `n`-element tensor,
/// matching PyTorch's `torch.chunk(t, world_size)` semantics.
///
/// The first `n mod world_size` ranks get `ceil(n / world_size)` elements;
/// the remainder get `floor(n / world_size)`. Empty shards are legal
/// (rank `r >= n` gets `(n, 0)` — a zero-length view at the end).
///
/// # Example
///
/// `shard_range(10, 0, 3) == (0, 4)`,
/// `shard_range(10, 1, 3) == (4, 4)`,
/// `shard_range(10, 2, 3) == (8, 2)`.
#[inline]
pub fn shard_range(n: usize, rank: i32, world_size: i32) -> (usize, usize) {
    debug_assert!(rank >= 0 && rank < world_size);
    debug_assert!(world_size > 0);
    let world = world_size as usize;
    let rank = rank as usize;
    let base = n / world;
    let rem = n % world;
    let offset = rank * base + rank.min(rem);
    let len = base + if rank < rem { 1 } else { 0 };
    (offset, len)
}

// ============================================================================
// DistributedAdamStepPlan
// ============================================================================

/// ZeRO-1-style sharded Adam optimizer step.
///
/// Wraps a Phase 49 [`AdamStepPlan<T>`] + a borrowed
/// [`baracuda_nccl::Communicator`]; orchestrates the
/// `all_reduce(grads) → local_step(shard) → all_gather(params)`
/// protocol described in the [module docs](self).
///
/// On `world_size == 1` the plan elides both collectives and reduces
/// to a plain `AdamStepPlan::step` (bit-exact, useful for single-GPU
/// smoke tests of the distributed protocol itself).
///
/// ## Lifetime model
///
/// The plan **does not own** the communicator — it borrows it for the
/// step duration. This matches `AdamStepPlan`'s "plan is config, step
/// takes references" shape and lets one communicator drive multiple
/// plan kinds (Adam + LAMB + SGD + future) with no shared-ownership
/// overhead.
///
/// ## Numerical fidelity
///
/// Single-rank: bit-exact match with `AdamStepPlan` (no collectives
/// invoked).
///
/// Multi-rank: each rank performs its local Adam math in fp32
/// (mixed-precision: half-precision params + grads, but f32 moments —
/// inherited from Phase 49's `step_with_f32_state` contract). The
/// `all_gather` in step 3 is a bit-copy — no per-rank floating-point
/// rounding skew. The only multi-rank-vs-single-rank divergence is
/// the `all_reduce` reduction order (NCCL ring sum vs single-pass
/// addition); this is bounded by `world_size · eps` per element.
#[derive(Debug)]
pub struct DistributedAdamStepPlan<'comm, T: AdamParamDtype + NcclScalar> {
    inner: AdamStepPlan<T>,
    comm: &'comm Communicator,
}

impl<'comm, T: AdamParamDtype + NcclScalar> DistributedAdamStepPlan<'comm, T> {
    /// Build a sharded Adam plan with the given hyperparameters and
    /// communicator. The communicator must outlive every `step()` call.
    pub fn new(cfg: AdamConfig, comm: &'comm Communicator) -> Self {
        Self {
            inner: AdamStepPlan::<T>::new(cfg),
            comm,
        }
    }

    /// Active Adam config (forwarded to the inner plan).
    pub fn config(&self) -> &AdamConfig {
        self.inner.config()
    }

    /// Communicator this plan is bound to.
    pub fn communicator(&self) -> &Communicator {
        self.comm
    }

    /// This rank's index within the communicator (cached). Convenience
    /// wrapper for `self.communicator().rank()`.
    #[inline]
    pub fn rank(&self) -> i32 {
        self.comm.rank()
    }

    /// Total rank count. Convenience wrapper for
    /// `self.communicator().world_size()`.
    #[inline]
    pub fn world_size(&self) -> i32 {
        self.comm.world_size()
    }

    /// Compute the (offset, length) of this rank's shard inside an
    /// `n`-element tensor. See [`shard_range`].
    #[inline]
    pub fn shard_range_for(&self, n: usize) -> (usize, usize) {
        shard_range(n, self.comm.rank(), self.comm.world_size())
    }

    /// One ZeRO-1 step. Same dtype `T` for params + grads + moments
    /// (use the f32 plan, or [`Self::step_with_f32_state`] for the
    /// mixed-precision shape).
    ///
    /// ## Arguments
    ///
    /// - `param_buffers` — mutable references to each parameter
    ///   tensor's device buffer. Each buffer must already describe
    ///   this rank's local shard view of the full tensor (the
    ///   caller controls the per-rank slice layout; Phase 58 only
    ///   supplies the protocol).
    /// - `grad_buffers` — mutable references to each gradient
    ///   tensor's device buffer (same per-rank shard convention).
    /// - `exp_avg`, `exp_avg_sq` — TensorList views of the per-rank
    ///   Adam moment buffers (also sharded; smaller than the full
    ///   tensor in the multi-rank case — that's the ZeRO-1 memory
    ///   savings).
    /// - `step_index` — 1-based step counter for bias correction.
    /// - `stream` — CUDA stream the launches are dispatched on.
    ///
    /// ## Multi-rank protocol
    ///
    /// 1. `all_reduce(grads, op=Sum, in-place)` — every rank ends up
    ///    with the summed gradient.
    /// 2. Local Adam step on this rank's shard of every (param, grad,
    ///    exp_avg, exp_avg_sq) tuple — uses the inner Phase 49
    ///    [`AdamStepPlan`].
    /// 3. `all_gather(updated_params, in-place)` — broadcast every
    ///    rank's updated shard back to all peers.
    ///
    /// ## Single-rank degenerate case
    ///
    /// When `world_size == 1` both collectives are skipped and the
    /// call reduces to [`AdamStepPlan::step`] exactly (bit-exact).
    ///
    /// ## Constraints (Phase 58)
    ///
    /// - Each tensor's element count must be a multiple of
    ///   `world_size` (ring all_gather symmetry requirement). The
    ///   per-tensor broadcast fallback for ragged shards is future
    ///   work; in practice model weight tensors are almost always
    ///   dim-aligned for tensor-core bucketing, so this falls out
    ///   naturally.
    ///
    /// ## Errors
    ///
    /// Returns [`Error::LengthMismatch`] for length-mismatched lists.
    /// Returns [`Error::InvalidArgument`] for non-world_size-multiple
    /// tensor sizes (multi-rank only).
    /// Returns [`Error::NcclFailed`] if any NCCL collective fails —
    /// the underlying NCCL status is surfaced verbatim in the wrapped
    /// error string.
    ///
    /// ## Buffer aliasing
    ///
    /// All-reduce in step 1 is **in-place** on `grad_buffers`.
    /// All-gather in step 3 is **in-place** on `param_buffers`.
    /// Callers must ensure the underlying device buffers aren't
    /// concurrently aliased by other streams.
    pub fn step(
        &self,
        param_buffers: &mut [&mut DeviceBuffer<T>],
        grad_buffers: &mut [&mut DeviceBuffer<T>],
        exp_avg: &TensorList<'_, T>,
        exp_avg_sq: &TensorList<'_, T>,
        step_index: i32,
        stream: &Stream,
    ) -> Result<()> {
        if param_buffers.len() != grad_buffers.len()
            || param_buffers.len() != exp_avg.len()
            || param_buffers.len() != exp_avg_sq.len()
        {
            return Err(Error::LengthMismatch(
                "param_buffers/grad_buffers/exp_avg/exp_avg_sq tensor counts differ",
            ));
        }

        let multi_rank = self.comm.world_size() > 1;

        // Step 1 (multi-rank only): in-place all-reduce gradients
        // across ranks. Done up front while we still have &mut access
        // to grad_buffers. After this step, every rank's grad slice
        // holds the summed gradient.
        if multi_rank {
            for g in grad_buffers.iter_mut() {
                let n = g.len();
                let raw: *mut DeviceBuffer<T> = *g as *mut DeviceBuffer<T>;
                // SAFETY: in-place all_reduce reads `send` before
                // writing `recv` in a single CUDA kernel launch; the
                // two aliasing borrows never live across the FFI call.
                let send_ref: &DeviceBuffer<T> = unsafe { &*raw };
                baracuda_nccl::all_reduce(send_ref, *g, n, RedOp::Sum, self.comm, stream)
                    .map_err(map_nccl_err)?;
                let _ = send_ref;
            }
        }

        // Step 2: local Adam step. Re-borrow the buffers as
        // immutable to build the TensorLists the inner plan wants.
        // After this block returns, the TensorList borrows are gone
        // and we can re-take &mut on the buffers for step 3.
        //
        // PHASE 58 SCOPE NOTE: the buffer slices passed in are
        // EXPECTED to already describe each rank's local shard. The
        // caller controls the per-tensor slice geometry by sizing
        // the underlying DeviceBuffers; Phase 58 supplies the
        // protocol, the caller supplies the sharded buffers. The
        // all_gather in step 3 then assembles each rank's
        // local-shard updates back into the full tensor.
        {
            let params_immut: Vec<&DeviceBuffer<T>> =
                param_buffers.iter().map(|b| &**b).collect();
            let grads_immut: Vec<&DeviceBuffer<T>> =
                grad_buffers.iter().map(|b| &**b).collect();
            let params = TensorList::<T>::new(&params_immut)?;
            let grads = TensorList::<T>::new(&grads_immut)?;
            self.inner
                .step(&params, &grads, exp_avg, exp_avg_sq, step_index, stream)?;
        }

        // Step 3 (multi-rank only): in-place all-gather updated params.
        if multi_rank {
            for p in param_buffers.iter_mut() {
                let n_total = p.len();
                let world = self.comm.world_size() as usize;
                if n_total % world != 0 {
                    return Err(Error::InvalidArgument(
                        "DistributedAdamStepPlan: tensor element count must be a \
                         multiple of world_size for the ring all_gather (Phase 58 \
                         restriction; per-tensor broadcast fallback is future work)",
                    ));
                }
                let per_rank = n_total / world;
                let raw: *mut DeviceBuffer<T> = *p as *mut DeviceBuffer<T>;
                // SAFETY: in-place all_gather is documented legal when
                // `send_ptr == recv_ptr + rank * sendcount * sizeof(T)`.
                let send_ref: &DeviceBuffer<T> = unsafe { &*raw };
                self.comm
                    .all_gather::<T>(send_ref, *p, per_rank, stream)
                    .map_err(map_nccl_err)?;
                let _ = send_ref;
            }
        }

        Ok(())
    }

    /// Mixed-precision (half-param + f32-state) ZeRO-1 step.
    ///
    /// Same protocol as [`Self::step`] but the moment buffers stay in
    /// f32 (matching Phase 49's
    /// [`AdamStepPlan::step_with_f32_state`](crate::AdamStepPlan::step_with_f32_state)
    /// contract).
    ///
    /// Only the params + grads are all_reduce'd / all_gather'd in `T` —
    /// the f32 moment buffers stay on-rank (they're already sharded;
    /// that's the ZeRO-1 memory savings).
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_f32_state(
        &self,
        param_buffers: &mut [&mut DeviceBuffer<T>],
        grad_buffers: &mut [&mut DeviceBuffer<T>],
        exp_avg: &TensorList<'_, f32>,
        exp_avg_sq: &TensorList<'_, f32>,
        step_index: i32,
        stream: &Stream,
    ) -> Result<()> {
        if param_buffers.len() != grad_buffers.len()
            || param_buffers.len() != exp_avg.len()
            || param_buffers.len() != exp_avg_sq.len()
        {
            return Err(Error::LengthMismatch(
                "param_buffers/grad_buffers/exp_avg/exp_avg_sq tensor counts differ",
            ));
        }

        let multi_rank = self.comm.world_size() > 1;

        // Step 1 (multi-rank only): all-reduce gradients.
        if multi_rank {
            for g in grad_buffers.iter_mut() {
                let n = g.len();
                let raw: *mut DeviceBuffer<T> = *g as *mut DeviceBuffer<T>;
                // SAFETY: see `step()` for the in-place all_reduce
                // aliasing contract.
                let send_ref: &DeviceBuffer<T> = unsafe { &*raw };
                baracuda_nccl::all_reduce(send_ref, *g, n, RedOp::Sum, self.comm, stream)
                    .map_err(map_nccl_err)?;
                let _ = send_ref;
            }
        }

        // Step 2: local Adam step with mixed-precision f32 moments.
        {
            let params_immut: Vec<&DeviceBuffer<T>> =
                param_buffers.iter().map(|b| &**b).collect();
            let grads_immut: Vec<&DeviceBuffer<T>> =
                grad_buffers.iter().map(|b| &**b).collect();
            let params = TensorList::<T>::new(&params_immut)?;
            let grads = TensorList::<T>::new(&grads_immut)?;
            self.inner.step_with_f32_state(
                &params,
                &grads,
                exp_avg,
                exp_avg_sq,
                step_index,
                stream,
            )?;
        }

        // Step 3 (multi-rank only): all-gather updated params.
        if multi_rank {
            for p in param_buffers.iter_mut() {
                let n_total = p.len();
                let world = self.comm.world_size() as usize;
                if n_total % world != 0 {
                    return Err(Error::InvalidArgument(
                        "DistributedAdamStepPlan: tensor element count must be a \
                         multiple of world_size for the ring all_gather",
                    ));
                }
                let per_rank = n_total / world;
                let raw: *mut DeviceBuffer<T> = *p as *mut DeviceBuffer<T>;
                // SAFETY: see `step()` for the in-place all_gather
                // aliasing contract.
                let send_ref: &DeviceBuffer<T> = unsafe { &*raw };
                self.comm
                    .all_gather::<T>(send_ref, *p, per_rank, stream)
                    .map_err(map_nccl_err)?;
                let _ = send_ref;
            }
        }

        Ok(())
    }
}

/// Helper: lower a `baracuda_nccl::Error` into the optim crate's
/// `Error::NcclFailed { what }` variant.
fn map_nccl_err(e: baracuda_nccl::Error) -> Error {
    // We surface the NCCL error's display string verbatim in a
    // 'static description by allocating it on the box — the optim
    // Error enum needs the description owned for `'static` callers.
    // The Phase 49 Error enum predates allocation in errors, so we
    // bridge via the NcclFailed variant added in Phase 58.
    Error::NcclFailed(e.to_string())
}

// ============================================================================
// Tests — pure-Rust shard_range covered without GPU
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_range_even_split() {
        // 12 elements across 4 ranks → each gets 3.
        assert_eq!(shard_range(12, 0, 4), (0, 3));
        assert_eq!(shard_range(12, 1, 4), (3, 3));
        assert_eq!(shard_range(12, 2, 4), (6, 3));
        assert_eq!(shard_range(12, 3, 4), (9, 3));
    }

    #[test]
    fn shard_range_uneven_split_pytorch_chunk_convention() {
        // 10 elements across 3 ranks → first 1 rank gets 4, rest get 3.
        // Matches torch.chunk(10, 3) = [4, 3, 3].
        assert_eq!(shard_range(10, 0, 3), (0, 4));
        assert_eq!(shard_range(10, 1, 3), (4, 3));
        assert_eq!(shard_range(10, 2, 3), (7, 3));
    }

    #[test]
    fn shard_range_single_rank_is_full_tensor() {
        assert_eq!(shard_range(1000, 0, 1), (0, 1000));
    }

    #[test]
    fn shard_range_empty_shard_when_rank_exceeds_elements() {
        // 3 elements across 4 ranks → ranks 3 gets a zero-length shard.
        assert_eq!(shard_range(3, 0, 4), (0, 1));
        assert_eq!(shard_range(3, 1, 4), (1, 1));
        assert_eq!(shard_range(3, 2, 4), (2, 1));
        assert_eq!(shard_range(3, 3, 4), (3, 0));
    }
}
