//! Single-process multi-GPU NCCL `AllReduce(Sum)` demo.
//!
//! Uses `ncclCommInitAll` to bring up one communicator per visible
//! device in this process, fills each rank's send buffer with its rank
//! id, runs an `AllReduce(Sum)`, and prints the reduced value on rank
//! 0. With `R` ranks the expected sum is `0 + 1 + ... + (R-1)`.
//!
//! If fewer than 2 GPUs are visible (or NCCL isn't installed — common
//! on Windows or single-GPU dev boxes) the example reports the
//! situation and exits cleanly without running the collective.
//!
//! Run with:
//!
//! ```text
//! cargo run --example all_reduce_single_proc -p baracuda-nccl
//! ```

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_nccl::{all_reduce, Communicator, RedOp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    baracuda_driver::init()?;
    let device_count = Device::count()?;
    println!("visible CUDA devices: {device_count}");
    if device_count < 2 {
        println!(
            "skipping AllReduce demo: NCCL single-process AllReduce needs ≥2 GPUs \
             (have {device_count}). A 1-GPU communicator works too, but the \
             reduction would be the identity."
        );
        return Ok(());
    }

    let devices: Vec<i32> = (0..device_count as i32).collect();
    println!("bringing up {} NCCL communicators...", devices.len());
    let comms = match Communicator::init_all(&devices) {
        Ok(c) => c,
        Err(e) => {
            // NCCL is typically Linux-only — Windows boxes won't have it.
            eprintln!("NCCL init_all failed ({e}). NCCL is normally Linux-only.");
            return Ok(());
        }
    };
    assert_eq!(comms.len(), devices.len());

    // Per-rank: pick the matching device, allocate a context + stream,
    // fill the send buffer with the rank id, allocate a recv buffer,
    // then run AllReduce(Sum).
    let n: usize = 16;
    let mut ctxs = Vec::with_capacity(comms.len());
    let mut streams = Vec::with_capacity(comms.len());
    let mut sends = Vec::with_capacity(comms.len());
    let mut recvs = Vec::with_capacity(comms.len());

    for comm in &comms {
        let dev = Device::get(comm.rank() as u32)?;
        let ctx = Context::new(&dev)?;
        let stream = Stream::new(&ctx)?;
        let host_in = vec![comm.rank() as f32; n];
        let send = DeviceBuffer::from_slice(&ctx, &host_in)?;
        let recv: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n)?;
        ctxs.push(ctx);
        streams.push(stream);
        sends.push(send);
        recvs.push(recv);
    }

    // Issue the AllReduce on every rank. NCCL serializes the collective
    // through each communicator's stream; we sync afterwards.
    for ((send, recv), (stream, comm)) in sends
        .iter()
        .zip(recvs.iter_mut())
        .zip(streams.iter().zip(comms.iter()))
    {
        all_reduce(send, recv, n, RedOp::Sum, comm, stream)?;
    }
    for stream in &streams {
        stream.synchronize()?;
    }

    // Pull rank 0's result back and verify.
    let mut host_out = vec![0.0f32; n];
    recvs[0].copy_to_host(&mut host_out)?;
    let expected_sum: f32 = (0..device_count as i32).map(|r| r as f32).sum();
    println!("rank 0 result (first 4): {:?}", &host_out[..4]);
    println!("expected sum: {expected_sum}");
    for (i, v) in host_out.iter().enumerate() {
        assert!(
            (v - expected_sum).abs() < 1e-3,
            "rank 0 cell {i}: got {v}, expected {expected_sum}"
        );
    }
    println!("OK");
    Ok(())
}
