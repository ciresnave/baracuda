//! Multi-rank scaffold — `#[ignore]`-gated.
//!
//! This test is **expected to fail** on single-GPU dev environments
//! because it requires `world_size >= 2`. It's checked in as a
//! scaffold so that a future multi-GPU CI step can exercise it
//! end-to-end with no setup. The test is kept in the test binary
//! (rather than in `examples/`) so `cargo test --ignored -- --list`
//! advertises it as a smoke target.
//!
//! Real multi-rank validation requires either:
//! - 2+ physical GPUs in the same process (NCCL's `init_all`
//!   single-process multi-GPU pattern), OR
//! - 2+ processes coordinating via `NcclUniqueId` (the multi-
//!   process pattern — typically `mpirun -n 2 …` or a manual
//!   `std::process::Command::spawn` pair).
//!
//! This scaffold takes the simpler single-process-multi-GPU path.

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_megatron::{
    ColumnParallelLinearPlan, RowParallelLinearPlan, TensorParallelContext,
};
use baracuda_nccl::Communicator;

#[test]
#[ignore = "requires 2+ NVIDIA GPUs + NCCL (single-GPU dev env returns early)"]
fn column_parallel_world_size_2_smoke() {
    baracuda_driver::init().unwrap();

    let count = Device::count().unwrap_or(0);
    if count < 2 {
        eprintln!(
            "multi_rank_scaffold: only {count} GPU(s) detected; this test \
             requires 2+. Skipping cleanly."
        );
        return;
    }

    // Single-process multi-GPU: ncclCommInitAll for 2 devices, one
    // communicator per device.
    let comms = match Communicator::init_all(&[0, 1]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NCCL init_all failed: {e:?}; skipping.");
            return;
        }
    };
    assert_eq!(comms.len(), 2);
    assert_eq!(comms[0].world_size(), 2);
    assert_eq!(comms[1].world_size(), 2);

    // Synthetic shape — out_features divisible by 2.
    let batch = 4i32;
    let in_f = 32i32;
    let out_f = 16i32; // out_per_rank = 8

    // Run ColumnParallel on rank 0 only (per-rank state isolated for
    // the smoke test — full cross-rank correctness would need
    // coordinated multi-thread execution; this is just bring-up).
    let ctx_0 = Context::new(&Device::get(0u32).unwrap()).unwrap();
    let stream_0 = Stream::new(&ctx_0).unwrap();

    let tpctx = TensorParallelContext::new(&comms[0], in_f, out_f);
    assert_eq!(tpctx.partitioned_out_features(), out_f / 2);

    let plan = ColumnParallelLinearPlan::<f32>::new(&tpctx, batch).expect("col plan");
    assert_eq!(plan.out_per_rank(), out_f / 2);

    // We don't actually run the FW here — the per-rank buffer
    // allocations would need to be sized correctly across both
    // contexts, and the cross-rank all_gather requires both ranks to
    // call into NCCL synchronously. That's a richer harness than this
    // scaffold provides. The compile + plan-construction here is
    // enough to assert the bring-up path is intact.
    let _ = (batch, in_f, out_f, &plan);
    drop(stream_0);
    drop(ctx_0);
}

#[test]
#[ignore = "requires 2+ NVIDIA GPUs + NCCL (single-GPU dev env returns early)"]
fn row_parallel_world_size_2_smoke() {
    baracuda_driver::init().unwrap();

    let count = Device::count().unwrap_or(0);
    if count < 2 {
        eprintln!("multi_rank_scaffold: only {count} GPU(s); skipping.");
        return;
    }

    let comms = match Communicator::init_all(&[0, 1]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NCCL init_all failed: {e:?}; skipping.");
            return;
        }
    };

    let batch = 4i32;
    let in_f = 32i32; // in_per_rank = 16
    let out_f = 16i32;

    let ctx_0 = Context::new(&Device::get(0u32).unwrap()).unwrap();
    let stream_0 = Stream::new(&ctx_0).unwrap();

    let tpctx = TensorParallelContext::new(&comms[0], in_f, out_f);
    assert_eq!(tpctx.partitioned_in_features(), in_f / 2);

    let plan = RowParallelLinearPlan::<f32>::new(&tpctx, batch).expect("row plan");
    assert_eq!(plan.in_per_rank(), in_f / 2);

    // Bring-up only — same scaffold-shape as ColumnParallel above.
    let _ = (batch, in_f, out_f, &plan);
    let _ = DeviceBuffer::<f32>::new(&ctx_0, 16);
    drop(stream_0);
    drop(ctx_0);
}
