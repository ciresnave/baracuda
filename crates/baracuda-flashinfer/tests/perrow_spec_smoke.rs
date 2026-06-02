//! Smoke tests for per-row sampling + speculative-decode verification —
//! Phase 66 Tier 2. `#[ignore]` + `flashinfer` feature.

#![cfg(feature = "flashinfer")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_flashinfer::sampling::{
    PerRowSampler, PerRowSamplingArgs, PerRowSamplingDescriptor, PerRowSamplingPlan,
    SpeculativeSamplingArgs, SpeculativeSamplingDescriptor, SpeculativeSamplingPlan,
};
use baracuda_flashinfer::{contiguous_stride, PlanPreference, TensorMut, TensorRef, Workspace};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Per-row top-K+top-P over one-hot rows must return each row's hot index
/// (per-row top_k=1 i32 array, per-row top_p=1.0 f32 array).
#[test]
#[ignore]
fn perrow_top_k_top_p_one_hot() {
    let (ctx, stream) = setup();
    const VOCAB: usize = 16;
    let hot = [3usize, 11];
    let mut probs = vec![0.0f32; 2 * VOCAB];
    probs[0 * VOCAB + hot[0]] = 1.0;
    probs[1 * VOCAB + hot[1]] = 1.0;
    let probs_dev = DeviceBuffer::from_slice(&ctx, &probs).expect("probs");
    let top_k_dev = DeviceBuffer::from_slice(&ctx, &[1i32, 1]).expect("top_k");
    let top_p_dev = DeviceBuffer::from_slice(&ctx, &[1.0f32, 1.0]).expect("top_p");
    let mut out_dev: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 2).expect("out");
    let mut valid_dev: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 2).expect("valid");

    let desc = PerRowSamplingDescriptor {
        batch_size: 2,
        vocab_size: VOCAB as i32,
        sampler: PerRowSampler::TopKTopP,
        deterministic: false,
    };
    let plan = PerRowSamplingPlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let pshape = [2, VOCAB as i32];
    plan.run(
        &stream,
        Workspace::None,
        PerRowSamplingArgs {
            probs: TensorRef { data: probs_dev.as_slice(), shape: pshape, stride: contiguous_stride(pshape) },
            top_k_arr: Some(TensorRef { data: top_k_dev.as_slice(), shape: [2], stride: [1] }),
            top_p_arr: Some(TensorRef { data: top_p_dev.as_slice(), shape: [2], stride: [1] }),
            min_p_arr: None,
            output: TensorMut { data: out_dev.as_slice_mut(), shape: [2], stride: [1] },
            valid: Some(TensorMut { data: valid_dev.as_slice_mut(), shape: [2], stride: [1] }),
            seed_val: 7,
            offset_val: 0,
        },
    )
    .expect("perrow run");
    stream.synchronize().expect("sync");

    let mut got = [0i32; 2];
    out_dev.copy_to_host(&mut got).expect("download");
    assert_eq!(got[0] as usize, hot[0], "row 0");
    assert_eq!(got[1] as usize, hot[1], "row 1");
}

/// Identical draft + target one-hot distributions ⇒ every draft token is
/// accepted; the bonus token is sampled from the target's extra step.
#[test]
#[ignore]
fn speculative_all_accept() {
    let (ctx, stream) = setup();
    const VOCAB: usize = 4;
    const NSPEC: usize = 2;

    // draft step0 -> token 1, step1 -> token 2 (one-hot).
    let mut draft = vec![0.0f32; NSPEC * VOCAB];
    draft[0 * VOCAB + 1] = 1.0;
    draft[1 * VOCAB + 2] = 1.0;
    let draft_ids = [1i32, 2];
    // target matches the two draft steps + a bonus step -> token 3.
    let mut target = vec![0.0f32; (NSPEC + 1) * VOCAB];
    target[0 * VOCAB + 1] = 1.0;
    target[1 * VOCAB + 2] = 1.0;
    target[2 * VOCAB + 3] = 1.0;

    let draft_dev = DeviceBuffer::from_slice(&ctx, &draft).expect("draft");
    let draft_ids_dev = DeviceBuffer::from_slice(&ctx, &draft_ids).expect("draft_ids");
    let target_dev = DeviceBuffer::from_slice(&ctx, &target).expect("target");
    let mut out_ids_dev: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, NSPEC + 1).expect("out_ids");
    let mut accepted_dev: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("accepted");
    let mut emitted_dev: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("emitted");

    let desc = SpeculativeSamplingDescriptor {
        batch_size: 1,
        num_speculative_tokens: NSPEC as i32,
        vocab_size: VOCAB as i32,
        deterministic: true,
    };
    let plan =
        SpeculativeSamplingPlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let dshape = [1, NSPEC as i32, VOCAB as i32];
    let tshape = [1, NSPEC as i32 + 1, VOCAB as i32];
    plan.run(
        &stream,
        Workspace::None,
        SpeculativeSamplingArgs {
            draft_probs: TensorRef { data: draft_dev.as_slice(), shape: dshape, stride: contiguous_stride(dshape) },
            draft_token_ids: TensorRef { data: draft_ids_dev.as_slice(), shape: [1, NSPEC as i32], stride: contiguous_stride([1, NSPEC as i32]) },
            target_probs: TensorRef { data: target_dev.as_slice(), shape: tshape, stride: contiguous_stride(tshape) },
            output_token_ids: TensorMut { data: out_ids_dev.as_slice_mut(), shape: [1, NSPEC as i32 + 1], stride: contiguous_stride([1, NSPEC as i32 + 1]) },
            output_accepted_token_num: TensorMut { data: accepted_dev.as_slice_mut(), shape: [1], stride: [1] },
            output_emitted_draft_token_num: TensorMut { data: emitted_dev.as_slice_mut(), shape: [1], stride: [1] },
            seed_val: 42,
            offset_val: 0,
        },
    )
    .expect("speculative run");
    stream.synchronize().expect("sync");

    let mut out_ids = [0i32; NSPEC + 1];
    out_ids_dev.copy_to_host(&mut out_ids).expect("download ids");
    let mut accepted = [0i32];
    accepted_dev.copy_to_host(&mut accepted).expect("download accepted");

    assert_eq!(accepted[0], NSPEC as i32, "all draft tokens should be accepted");
    assert_eq!(out_ids[0], 1, "token 0");
    assert_eq!(out_ids[1], 2, "token 1");
    assert_eq!(out_ids[2], 3, "bonus token from target");
}
