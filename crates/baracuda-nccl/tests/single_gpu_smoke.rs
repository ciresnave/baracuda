//! Single-GPU NCCL smoke tests — exercise the full collective surface
//! against a `Communicator::new_single_gpu` (world_size = 1) communicator.
//!
//! With `world_size = 1` every collective degenerates to either a stream-
//! ordered no-op (AllReduce / Reduce / Broadcast on the lone rank) or
//! a per-rank-shard copy (AllGather / ReduceScatter with `count = N/1`).
//! The point is to validate the safe-wrapper API surface — type
//! conversions, stream-pointer plumbing, RAII destruction — without
//! needing multi-GPU hardware.
//!
//! Multi-rank validation (correctness of the reductions across ranks) is
//! out of scope for Phase 52 — it requires either 2+ physical GPUs or
//! a process-spawning test harness, neither of which is available in
//! the standard dev environment. Phase 53+ (Megatron TP, FSDP) will
//! introduce a multi-rank harness.
//!
//! These tests are `#[ignore]`-d by default. Run with
//! `cargo test -p baracuda-nccl --test single_gpu_smoke -- --ignored`
//! on a host that has NCCL installed.

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_nccl::{Communicator, NcclReduceOp};

fn try_bringup() -> Option<(Context, Stream, Communicator)> {
    if baracuda_driver::init().is_err() {
        eprintln!("CUDA driver init failed; skipping.");
        return None;
    }
    let device = Device::get(0).ok()?;
    let ctx = Context::new(&device).ok()?;
    let stream = Stream::new(&ctx).ok()?;
    let comm = match Communicator::new_single_gpu(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("NCCL not available: {e}. Skipping.");
            return None;
        }
    };
    Some((ctx, stream, comm))
}

#[test]
#[ignore = "requires NCCL installed (typically Linux multi-GPU hosts)"]
fn new_single_gpu_reports_rank0_worldsize1() {
    let Some((_ctx, _stream, comm)) = try_bringup() else { return };
    assert_eq!(comm.rank(), 0, "single-GPU communicator must be rank 0");
    assert_eq!(comm.world_size(), 1, "single-GPU communicator must have world_size 1");
}

#[test]
#[ignore = "requires NCCL installed"]
fn all_reduce_single_gpu_passes_through() {
    let Some((ctx, stream, comm)) = try_bringup() else { return };

    let host_in: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let send = DeviceBuffer::from_slice(&ctx, &host_in).unwrap();
    let mut recv: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, host_in.len()).unwrap();

    comm.all_reduce(&send, &mut recv, NcclReduceOp::Sum, &stream)
        .expect("all_reduce on single-GPU communicator must succeed");
    stream.synchronize().unwrap();

    let mut host_out = vec![0f32; host_in.len()];
    recv.copy_to_host(&mut host_out).unwrap();
    assert_eq!(host_out, host_in, "world_size=1 AllReduce(Sum) is identity");
}

#[test]
#[ignore = "requires NCCL installed"]
fn reduce_single_gpu_passes_through() {
    let Some((ctx, stream, comm)) = try_bringup() else { return };

    let host_in: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
    let send = DeviceBuffer::from_slice(&ctx, &host_in).unwrap();
    let mut recv: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, host_in.len()).unwrap();

    comm.reduce(&send, &mut recv, host_in.len(), NcclReduceOp::Sum, 0, &stream)
        .expect("reduce on single-GPU communicator must succeed");
    stream.synchronize().unwrap();

    let mut host_out = vec![0f32; host_in.len()];
    recv.copy_to_host(&mut host_out).unwrap();
    assert_eq!(host_out, host_in, "world_size=1 Reduce(root=0) is identity");
}

#[test]
#[ignore = "requires NCCL installed"]
fn broadcast_single_gpu_passes_through() {
    let Some((ctx, stream, comm)) = try_bringup() else { return };

    let host_in: Vec<i32> = (0..24).collect();
    let mut buf = DeviceBuffer::from_slice(&ctx, &host_in).unwrap();

    comm.broadcast(&mut buf, 0, &stream)
        .expect("broadcast on single-GPU communicator must succeed");
    stream.synchronize().unwrap();

    let mut host_out = vec![0i32; host_in.len()];
    buf.copy_to_host(&mut host_out).unwrap();
    assert_eq!(host_out, host_in, "world_size=1 Broadcast(root=0) is identity");
}

#[test]
#[ignore = "requires NCCL installed"]
fn all_gather_single_gpu_passes_through() {
    let Some((ctx, stream, comm)) = try_bringup() else { return };

    // sendcount = N, world_size = 1 → recv buffer matches send buffer.
    let host_in: Vec<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
    let send = DeviceBuffer::from_slice(&ctx, &host_in).unwrap();
    let mut recv: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, host_in.len()).unwrap();

    comm.all_gather(&send, &mut recv, host_in.len(), &stream)
        .expect("all_gather on single-GPU communicator must succeed");
    stream.synchronize().unwrap();

    let mut host_out = vec![0f32; host_in.len()];
    recv.copy_to_host(&mut host_out).unwrap();
    assert_eq!(host_out, host_in, "world_size=1 AllGather is identity");
}

#[test]
#[ignore = "requires NCCL installed"]
fn reduce_scatter_single_gpu_passes_through() {
    let Some((ctx, stream, comm)) = try_bringup() else { return };

    // recvcount × world_size = sendlen → with world_size=1 they match.
    let host_in: Vec<f32> = (0..28).map(|i| i as f32).collect();
    let send = DeviceBuffer::from_slice(&ctx, &host_in).unwrap();
    let mut recv: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, host_in.len()).unwrap();

    comm.reduce_scatter(&send, &mut recv, host_in.len(), NcclReduceOp::Sum, &stream)
        .expect("reduce_scatter on single-GPU communicator must succeed");
    stream.synchronize().unwrap();

    let mut host_out = vec![0f32; host_in.len()];
    recv.copy_to_host(&mut host_out).unwrap();
    assert_eq!(host_out, host_in, "world_size=1 ReduceScatter(Sum) is identity");
}

#[test]
#[ignore = "requires NCCL installed"]
fn send_recv_self_loop_in_group() {
    // P2P self-loop: rank 0 sends to rank 0. NCCL requires `send` and
    // `recv` paired inside a group bracket on the same communicator.
    let Some((ctx, stream, comm)) = try_bringup() else { return };

    let host_in: Vec<f32> = (0..12).map(|i| i as f32 + 100.0).collect();
    let send = DeviceBuffer::from_slice(&ctx, &host_in).unwrap();
    let mut recv: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, host_in.len()).unwrap();

    Communicator::group_start().expect("group_start must succeed");
    comm.send(&send, host_in.len(), 0, &stream)
        .expect("self-send must succeed inside group");
    comm.recv(&mut recv, host_in.len(), 0, &stream)
        .expect("self-recv must succeed inside group");
    Communicator::group_end().expect("group_end must succeed");
    stream.synchronize().unwrap();

    let mut host_out = vec![0f32; host_in.len()];
    recv.copy_to_host(&mut host_out).unwrap();
    assert_eq!(host_out, host_in, "self-loop send/recv must produce identity");
}
