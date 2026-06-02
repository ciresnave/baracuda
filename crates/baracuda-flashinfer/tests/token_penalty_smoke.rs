//! Smoke test for the bespoke token-penalty logit transform — Phase 66
//! Tier 2. Native baracuda op (not behind the `flashinfer` feature).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_flashinfer::sampling::{TokenPenaltyArgs, TokenPenaltyDescriptor, TokenPenaltyPlan};
use baracuda_flashinfer::{contiguous_stride, PlanPreference, TensorMut, TensorRef, Workspace};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn token_penalty_closed_form() {
    let (ctx, stream) = setup();

    // logits = [1.0, 2.0, -1.0, 0.5]; counts = [0, 2, 1, 0].
    // rep=2.0, freq=0.5, pres=1.0. Expected per cell:
    //   c0: count 0 -> unchanged           -> 1.0
    //   c1: 2.0/2 - 0.5*2 - 1.0            -> 1.0 - 1.0 - 1.0 = -1.0
    //   c2: -1.0*2 - 0.5*1 - 1.0           -> -2.0 - 0.5 - 1.0 = -3.5
    //   c3: count 0 -> unchanged           -> 0.5
    let logits = [1.0f32, 2.0, -1.0, 0.5];
    let counts = [0i32, 2, 1, 0];
    let expected = [1.0f32, -1.0, -3.5, 0.5];

    let mut logits_dev = DeviceBuffer::from_slice(&ctx, &logits).expect("upload logits");
    let counts_dev = DeviceBuffer::from_slice(&ctx, &counts).expect("upload counts");

    let desc = TokenPenaltyDescriptor {
        batch_size: 1,
        vocab_size: 4,
        rep_penalty: 2.0,
        freq_penalty: 0.5,
        pres_penalty: 1.0,
    };
    let plan = TokenPenaltyPlan::select(&stream, &desc, PlanPreference::default()).expect("select");
    let shape = [1, 4];
    plan.run(
        &stream,
        Workspace::None,
        TokenPenaltyArgs {
            logits: TensorMut { data: logits_dev.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            counts: TensorRef { data: counts_dev.as_slice(), shape, stride: contiguous_stride(shape) },
        },
    )
    .expect("token penalty run");
    stream.synchronize().expect("sync");

    let mut got = [0.0f32; 4];
    logits_dev.copy_to_host(&mut got).expect("download");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-5, "logit[{i}] = {g}, expected {e}");
    }
}
