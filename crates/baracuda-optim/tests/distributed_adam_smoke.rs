//! Phase 58 — DistributedAdamStepPlan single-rank smoke test.
//!
//! Validates the degenerate single-rank case where
//! `world_size == 1`: the plan elides both NCCL collectives and
//! reduces to a plain `AdamStepPlan::step`. This must produce bit-
//! exact equivalence to the non-distributed Adam plan.
//!
//! Multi-rank correctness validation is in
//! `distributed_adam_multi_rank_scaffold.rs` (gated behind
//! `--ignored`, marked "requires 2+ GPUs").
//!
//! All tests are `#[ignore]`-gated — they need an NVIDIA GPU + a
//! working NCCL loader path. `try_bringup()` returns `None` if NCCL
//! can't be resolved, printing a skip notice. The single-rank Adam
//! step (without NCCL) still runs unconditionally inside
//! `adam_smoke.rs`; this file is specifically about the distributed
//! plan's protocol orchestration.

#![cfg(feature = "distributed_optim")]

use baracuda_driver::{Context, Device, DeviceBuffer};
use baracuda_nccl::Communicator;
use baracuda_optim::{
    AdamConfig, AdamMode, AdamStepPlan, DistributedAdamStepPlan, TensorList, shard_range,
};

/// Try to bring up a single-rank NCCL communicator on device 0.
/// Returns `None` (with a printed skip message) if NCCL isn't
/// resolvable on this host — keeps the test green on Windows /
/// CUDA-without-NCCL setups.
fn try_bringup_single_rank() -> Option<Communicator> {
    match Communicator::new_single_gpu(0) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("Phase 58 smoke: NCCL not available; skipping. ({e})");
            None
        }
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU + working NCCL loader"]
fn distributed_adam_single_rank_matches_plain_adam() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = baracuda_driver::Stream::new(&ctx).unwrap();

    let Some(comm) = try_bringup_single_rank() else {
        return;
    };

    // World size must be 1 for the degenerate-case property.
    assert_eq!(
        comm.world_size(),
        1,
        "single-rank communicator should report world_size == 1"
    );
    assert_eq!(comm.rank(), 0);

    let cfg = AdamConfig {
        lr: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: 0.01,
        bias_correction: true,
        mode: AdamMode::Classic,
    };

    // Build two parallel parameter sets — one for the plain Adam
    // plan, one for the distributed-Adam plan. They must produce
    // identical results when world_size == 1.
    let n = 1024;
    let p_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let g_host: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) * 0.01 - 0.05).collect();
    let m_init: Vec<f32> = vec![0.0; n];
    let v_init: Vec<f32> = vec![0.0; n];

    // Plain Adam side.
    let p_plain = DeviceBuffer::from_slice(&ctx, &p_host).unwrap();
    let g_plain = DeviceBuffer::from_slice(&ctx, &g_host).unwrap();
    let m_plain = DeviceBuffer::from_slice(&ctx, &m_init).unwrap();
    let v_plain = DeviceBuffer::from_slice(&ctx, &v_init).unwrap();

    // Distributed Adam side.
    let mut p_dist = DeviceBuffer::from_slice(&ctx, &p_host).unwrap();
    let mut g_dist = DeviceBuffer::from_slice(&ctx, &g_host).unwrap();
    let m_dist = DeviceBuffer::from_slice(&ctx, &m_init).unwrap();
    let v_dist = DeviceBuffer::from_slice(&ctx, &v_init).unwrap();

    // Run plain Adam.
    {
        let params = TensorList::new(&[&p_plain]).unwrap();
        let grads = TensorList::new(&[&g_plain]).unwrap();
        let mom = TensorList::new(&[&m_plain]).unwrap();
        let vel = TensorList::new(&[&v_plain]).unwrap();
        AdamStepPlan::<f32>::new(cfg)
            .step(&params, &grads, &mom, &vel, 1, &stream)
            .expect("plain Adam step");
    }

    // Run distributed Adam — single rank, so it must elide NCCL and
    // call the same kernel. The new API takes param + grad buffers
    // as &mut [&mut ...] (the plan builds the inner TensorLists
    // internally) and moment buffers as TensorList (they're not
    // touched by collectives, only the inner Adam kernel).
    {
        let mom = TensorList::new(&[&m_dist]).unwrap();
        let vel = TensorList::new(&[&v_dist]).unwrap();
        let plan = DistributedAdamStepPlan::<f32>::new(cfg, &comm);
        let mut pbufs_storage = [&mut p_dist];
        let mut gbufs_storage = [&mut g_dist];
        let pbufs: &mut [&mut DeviceBuffer<f32>] = &mut pbufs_storage;
        let gbufs: &mut [&mut DeviceBuffer<f32>] = &mut gbufs_storage;
        plan.step(pbufs, gbufs, &mom, &vel, 1, &stream)
            .expect("distributed Adam step (single rank)");
    }

    stream.synchronize().unwrap();

    // Read back both sides — must match bit-exactly.
    let mut p_plain_got = vec![0.0f32; n];
    let mut p_dist_got = vec![0.0f32; n];
    p_plain.copy_to_host(&mut p_plain_got).unwrap();
    p_dist.copy_to_host(&mut p_dist_got).unwrap();

    for i in 0..n {
        assert_eq!(
            p_plain_got[i], p_dist_got[i],
            "single-rank DistributedAdam must bit-exact-match plain Adam at \
             index {i}; got plain={} dist={}",
            p_plain_got[i], p_dist_got[i]
        );
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU + working NCCL loader"]
fn distributed_adam_single_rank_three_tensors_no_collective_overhead() {
    // Exercises the multi-tensor path under single-rank: still must
    // forward to the inner AdamStepPlan and skip all collectives.
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = baracuda_driver::Stream::new(&ctx).unwrap();

    let Some(comm) = try_bringup_single_rank() else {
        return;
    };

    let cfg = AdamConfig::default();

    // Three tensors of varying sizes — exercises the multi-tensor
    // pack + chunk-clamp path.
    let n1 = 1024;
    let n2 = 3000;
    let n3 = 512;
    let p1: Vec<f32> = (0..n1).map(|i| (i as f32) * 0.001).collect();
    let p2: Vec<f32> = (0..n2).map(|i| (i as f32) * -0.0005).collect();
    let p3: Vec<f32> = (0..n3).map(|i| (i as f32) * 0.002).collect();
    let g1: Vec<f32> = (0..n1).map(|i| ((i % 17) as f32) * 0.03).collect();
    let g2: Vec<f32> = (0..n2).map(|i| ((i % 7) as f32) * -0.01).collect();
    let g3: Vec<f32> = (0..n3).map(|i| ((i % 11) as f32) * 0.02).collect();

    let mut bp1 = DeviceBuffer::from_slice(&ctx, &p1).unwrap();
    let mut bp2 = DeviceBuffer::from_slice(&ctx, &p2).unwrap();
    let mut bp3 = DeviceBuffer::from_slice(&ctx, &p3).unwrap();
    let mut bg1 = DeviceBuffer::from_slice(&ctx, &g1).unwrap();
    let mut bg2 = DeviceBuffer::from_slice(&ctx, &g2).unwrap();
    let mut bg3 = DeviceBuffer::from_slice(&ctx, &g3).unwrap();
    let bm1 = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; n1]).unwrap();
    let bm2 = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; n2]).unwrap();
    let bm3 = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; n3]).unwrap();
    let bv1 = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; n1]).unwrap();
    let bv2 = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; n2]).unwrap();
    let bv3 = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; n3]).unwrap();

    let mom = TensorList::new(&[&bm1, &bm2, &bm3]).unwrap();
    let vel = TensorList::new(&[&bv1, &bv2, &bv3]).unwrap();

    let plan = DistributedAdamStepPlan::<f32>::new(cfg, &comm);
    let mut pbufs_storage = [&mut bp1, &mut bp2, &mut bp3];
    let mut gbufs_storage = [&mut bg1, &mut bg2, &mut bg3];
    plan.step(&mut pbufs_storage, &mut gbufs_storage, &mom, &vel, 1, &stream)
        .expect("distributed Adam step (3 tensors, single rank)");
    stream.synchronize().unwrap();

    // We don't compare against a reference here — the bit-exact
    // match against plain AdamStepPlan is covered in
    // `distributed_adam_single_rank_matches_plain_adam`. This test's
    // contract is just "the multi-tensor path runs cleanly under the
    // distributed wrapper at world_size == 1".
}

/// Pure-Rust shard_range coverage — runs unconditionally (no GPU,
/// no NCCL). Mirrors the unit tests in src/distributed.rs but lives
/// here too as the externally-callable contract.
#[test]
fn shard_range_public_api_matches_pytorch_chunk() {
    // torch.chunk(10, 3) = [4, 3, 3]
    assert_eq!(shard_range(10, 0, 3), (0, 4));
    assert_eq!(shard_range(10, 1, 3), (4, 3));
    assert_eq!(shard_range(10, 2, 3), (7, 3));
    // torch.chunk(8, 4) = [2, 2, 2, 2]
    for r in 0..4 {
        assert_eq!(shard_range(8, r, 4), (r as usize * 2, 2));
    }
    // Single rank
    assert_eq!(shard_range(1_000_000, 0, 1), (0, 1_000_000));
}
