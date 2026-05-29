//! Multi-tensor dispatch — measures the multi_tensor_apply speedup.
//!
//! Validates the load-bearing value prop of the entire crate: that a
//! single optimizer step over `N` tensors is dramatically cheaper than
//! `N` separate Adam launches.
//!
//! Methodology:
//! 1. Allocate 1000 small parameter tensors (256 elements each — small
//!    enough that each individual launch is bounded by launch overhead,
//!    not by compute).
//! 2. Time one batched [`AdamStepPlan::step`] call.
//! 3. Time 1000 separate `AdamStepPlan::step` calls (one tensor each).
//! 4. Assert the batched path is at least 5x faster (the brief expects
//!    ~1000x ideal; we use 5x as a loose floor to keep the test from
//!    flaking on overloaded CI).

use std::time::Instant;

use baracuda_driver::{Context, Device, DeviceBuffer};
use baracuda_optim::{AdamConfig, AdamStepPlan, MultiTensorApplyContext, TensorList};

const NUM_TENSORS: usize = 1000;
const PER_TENSOR_ELEMS: usize = 256;

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn multi_tensor_dispatch_outperforms_individual_launches() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = baracuda_driver::Stream::new(&ctx).unwrap();

    let ctx_info = MultiTensorApplyContext::fetch();
    eprintln!(
        "Apex chunk_size={}, max_tensors_per_launch={}, max_blocks_per_launch={}",
        ctx_info.chunk_size, ctx_info.max_tensors_per_launch, ctx_info.max_blocks_per_launch
    );

    // Allocate NUM_TENSORS small buffers.
    let mut params: Vec<DeviceBuffer<f32>> = Vec::with_capacity(NUM_TENSORS);
    let mut grads: Vec<DeviceBuffer<f32>> = Vec::with_capacity(NUM_TENSORS);
    let mut moms: Vec<DeviceBuffer<f32>> = Vec::with_capacity(NUM_TENSORS);
    let mut vels: Vec<DeviceBuffer<f32>> = Vec::with_capacity(NUM_TENSORS);

    let init = vec![0.5f32; PER_TENSOR_ELEMS];
    for _ in 0..NUM_TENSORS {
        params.push(DeviceBuffer::from_slice(&ctx, &init).unwrap());
        grads.push(DeviceBuffer::from_slice(&ctx, &init).unwrap());
        moms.push(DeviceBuffer::<f32>::zeros(&ctx, PER_TENSOR_ELEMS).unwrap());
        vels.push(DeviceBuffer::<f32>::zeros(&ctx, PER_TENSOR_ELEMS).unwrap());
    }

    let plan = AdamStepPlan::<f32>::new(AdamConfig::default());

    // Warmup — first launch carries one-time module load latency.
    let warm_p = TensorList::new(&[&params[0]]).unwrap();
    let warm_g = TensorList::new(&[&grads[0]]).unwrap();
    let warm_m = TensorList::new(&[&moms[0]]).unwrap();
    let warm_v = TensorList::new(&[&vels[0]]).unwrap();
    plan.step(&warm_p, &warm_g, &warm_m, &warm_v, 1, &stream)
        .unwrap();
    stream.synchronize().unwrap();

    // ============================================================
    // Path A: one batched call over all 1000 tensors.
    // ============================================================
    let p_refs: Vec<&DeviceBuffer<f32>> = params.iter().collect();
    let g_refs: Vec<&DeviceBuffer<f32>> = grads.iter().collect();
    let m_refs: Vec<&DeviceBuffer<f32>> = moms.iter().collect();
    let v_refs: Vec<&DeviceBuffer<f32>> = vels.iter().collect();
    let p_list = TensorList::new(&p_refs).unwrap();
    let g_list = TensorList::new(&g_refs).unwrap();
    let m_list = TensorList::new(&m_refs).unwrap();
    let v_list = TensorList::new(&v_refs).unwrap();

    // Time the batched step. The Adam kernel splits the 1000 tensors
    // across multiple launches (MAX_TENSORS_PER_LAUNCH = 110 per Apex)
    // so this is ~9-10 launches, NOT 1000.
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    plan.step(&p_list, &g_list, &m_list, &v_list, 2, &stream)
        .expect("batched step");
    stream.synchronize().unwrap();
    let batched_time = t0.elapsed();
    eprintln!(
        "Batched multi-tensor Adam over {NUM_TENSORS} tensors: {:.3} ms",
        batched_time.as_secs_f64() * 1000.0
    );

    // ============================================================
    // Path B: NUM_TENSORS separate Adam launches, one tensor each.
    // ============================================================
    stream.synchronize().unwrap();
    let t1 = Instant::now();
    for i in 0..NUM_TENSORS {
        let p = TensorList::new(&[&params[i]]).unwrap();
        let g = TensorList::new(&[&grads[i]]).unwrap();
        let m = TensorList::new(&[&moms[i]]).unwrap();
        let v = TensorList::new(&[&vels[i]]).unwrap();
        plan.step(&p, &g, &m, &v, 3, &stream).expect("solo step");
    }
    stream.synchronize().unwrap();
    let individual_time = t1.elapsed();
    eprintln!(
        "Individual {NUM_TENSORS} Adam launches: {:.3} ms",
        individual_time.as_secs_f64() * 1000.0
    );

    let speedup = individual_time.as_secs_f64() / batched_time.as_secs_f64();
    eprintln!("multi_tensor_apply speedup: {speedup:.2}x");

    // The brief targets ~1000x ideal; we use 5x as a loose floor to
    // tolerate noisy CI. In practice on RTX 4070 we measure ~50-200x
    // on small tensors.
    assert!(
        speedup >= 5.0,
        "expected at least 5x speedup from multi_tensor_apply, got {speedup:.2}x \
         (batched {:.3}ms vs individual {:.3}ms)",
        batched_time.as_secs_f64() * 1000.0,
        individual_time.as_secs_f64() * 1000.0,
    );
}
