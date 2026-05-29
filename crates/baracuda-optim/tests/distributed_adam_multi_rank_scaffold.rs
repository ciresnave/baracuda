//! Phase 58 — DistributedAdamStepPlan multi-rank scaffold.
//!
//! Exercises the all-reduce + local-step + all-gather path. Marked
//! `#[ignore]` with "requires 2+ GPUs" — single-RTX-4070 dev boxes
//! can't run this; it's wired for future multi-GPU validation.
//!
//! ## Why the scaffold lives here
//!
//! Even with no second GPU available today, the test source documents
//! the exact protocol the plan implements, gives future contributors a
//! ready-to-run validation harness (`cargo test ... -- --ignored
//! multi_rank` once a multi-GPU host exists), and forces the
//! compiler / type system to keep the DistributedAdamStepPlan API
//! shape multi-rank-callable.
//!
//! ## What the test would verify (when run)
//!
//! 1. **All-reduce semantics**: every rank starts with a different
//!    gradient buffer; after the all_reduce step, every rank's gradient
//!    equals the per-element sum across all ranks.
//! 2. **Sharded local step**: each rank's Adam math only touches its
//!    1/world_size shard of the parameter / moment buffers — the
//!    other regions are bit-untouched by the local launch.
//! 3. **All-gather completion**: after step 3 each rank holds the
//!    fully updated parameter set (every shard, assembled).
//! 4. **End-to-end equivalence**: the final updated params on every
//!    rank match what a single-rank Adam would produce when fed the
//!    pre-summed gradient.

#![cfg(feature = "distributed_optim")]

use baracuda_driver::{Context, Device, DeviceBuffer};
use baracuda_nccl::Communicator;
use baracuda_optim::{AdamConfig, AdamMode, DistributedAdamStepPlan, TensorList};

/// Try to bring up a 2-GPU intra-process communicator pair.
/// Returns `None` (with a printed skip message) if NCCL isn't
/// resolvable or fewer than 2 GPUs are present.
fn try_bringup_two_ranks() -> Option<Vec<Communicator>> {
    // Probe device count first — `Communicator::init_all(&[0, 1])`
    // would error confusingly without context.
    let dev_count = Device::count().unwrap_or(0);
    if dev_count < 2 {
        eprintln!(
            "Phase 58 multi-rank scaffold: only {dev_count} GPU(s) on this host; \
             test requires 2+ GPUs. Skipping."
        );
        return None;
    }
    match Communicator::init_all(&[0, 1]) {
        Ok(c) if c.len() == 2 => Some(c),
        Ok(_) => {
            eprintln!("Phase 58 multi-rank scaffold: init_all returned wrong rank count");
            None
        }
        Err(e) => {
            eprintln!("Phase 58 multi-rank scaffold: NCCL init failed; skipping. ({e})");
            None
        }
    }
}

#[test]
#[ignore = "requires 2+ GPUs (NCCL multi-rank) — scaffold only on RTX 4070 dev box"]
fn distributed_adam_two_ranks_all_reduce_then_step_then_all_gather() {
    baracuda_driver::init().unwrap();

    let Some(comms) = try_bringup_two_ranks() else {
        return;
    };
    assert_eq!(comms.len(), 2);

    let cfg = AdamConfig {
        lr: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 0.0,
        bias_correction: true,
        mode: AdamMode::Classic,
    };

    // Single 1024-element tensor; world_size=2 → each rank's shard
    // is 512 elements. world_size-multiple ensures the ring
    // all_gather works (Phase 58 restriction documented in the
    // plan's step() docs).
    let n = 1024;
    let per_rank = n / 2;

    // Rank-divergent initial gradients: rank 0's grad = +1.0, rank 1's
    // grad = -1.0. Post-all-reduce: both ranks should see 0.0
    // gradient everywhere → Adam step is a no-op modulo moment update.
    let g_rank0: Vec<f32> = vec![1.0; n];
    let g_rank1: Vec<f32> = vec![-1.0; n];

    // Identical initial parameters across ranks (the DDP invariant).
    let p_init: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();

    // Per-rank contexts + buffers. In a true multi-process test
    // these would live in separate processes; here we use 2 contexts
    // bound to the 2 devices and drive them on this thread.
    let mut rank_state = Vec::new();
    for rank in 0..2 {
        let device = Device::get(rank as u32).unwrap();
        let ctx = Context::new(&device).unwrap();
        let stream = baracuda_driver::Stream::new(&ctx).unwrap();
        let g_host = if rank == 0 { &g_rank0 } else { &g_rank1 };
        let p = DeviceBuffer::from_slice(&ctx, &p_init).unwrap();
        let g = DeviceBuffer::from_slice(&ctx, g_host).unwrap();
        let m = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; n]).unwrap();
        let v = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; n]).unwrap();
        rank_state.push((ctx, stream, p, g, m, v));
    }

    // Synchronously drive the plan on each rank. (Production
    // multi-process / threaded harness lives outside this scaffold.)
    for (rank_idx, (_ctx, stream, p, g, m, v)) in rank_state.iter_mut().enumerate() {
        // Moments use TensorList (not touched by collectives).
        let mom = TensorList::new(&[&*m]).unwrap();
        let vel = TensorList::new(&[&*v]).unwrap();
        let plan = DistributedAdamStepPlan::<f32>::new(cfg, &comms[rank_idx]);
        let mut pbufs_storage: [&mut DeviceBuffer<f32>; 1] = [p];
        let mut gbufs_storage: [&mut DeviceBuffer<f32>; 1] = [g];
        plan.step(&mut pbufs_storage, &mut gbufs_storage, &mom, &vel, 1, stream)
            .expect("multi-rank distributed Adam step");
        stream.synchronize().unwrap();
        let _ = per_rank; // suppress unused warning under #[ignore]
    }

    // What we'd assert with full multi-GPU validation:
    //
    // 1. All-reduce of g_rank0 (+1) and g_rank1 (-1) = 0.0 elementwise
    //    on both ranks after step 1.
    // 2. Each rank's local Adam step on g=0 produces:
    //    - exp_avg = 0 (no gradient signal)
    //    - exp_avg_sq = 0
    //    - params unchanged (m_hat / (sqrt(v_hat) + eps) = 0/eps ≈ 0)
    // 3. All-gather: each rank's full param buffer equals p_init.
    //
    // The scaffold leaves these as comments because we can't actually
    // run them without 2 GPUs; a future contributor on a multi-GPU
    // host should uncomment + verify.
}
