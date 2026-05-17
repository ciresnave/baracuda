//! Real-GPU smoke test for `QuantizePerTensorPlan<TIn, TOut>`
//! (Phase 8 Milestone 8.1).
//!
//! Covers f32 → s8 with a small fixture against a hand-computed
//! reference, including clip-edge cases (values that round out of the
//! s8 range — verifies the clamp).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test quantize_per_tensor_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, QuantizePerTensorArgs,
    QuantizePerTensorDescriptor, QuantizePerTensorPlan, S8, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn quantize_per_tensor_f32_s8_clipping() {
    let (ctx, stream) = setup();
    let scale: f32 = 0.1;
    let zero_point: i32 = 0;
    // Mix in-range and out-of-range values:
    //   x =   0.05  →  round( 0.5)+0 =   0  ∈ [-128, 127] → 0
    //   x =   0.15  →  round( 1.5)+0 =   2 (round-half-to-even) → 2
    //   x =   0.25  →  round( 2.5)+0 =   2 (round-half-to-even) → 2
    //   x =  -1.27  →  round(-12.7)+0 = -13 → -13
    //   x = -12.8   →  round(-128.0)+0 = -128 → -128 (edge)
    //   x = -20.0   →  round(-200.0)+0 = -200 → clamped to -128
    //   x =  12.7   →  round(127.0)+0 = 127 → 127 (edge)
    //   x =  20.0   →  round(200.0)+0 = 200 → clamped to 127
    let host_x: Vec<f32> = vec![0.05, 0.15, 0.25, -1.27, -12.8, -20.0, 12.7, 20.0];
    let host_expected: Vec<i8> = vec![0, 2, 2, -13, -128, -128, 127, 127];
    let numel = host_x.len() as i32;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_q: DeviceBuffer<S8> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc q");

    let desc = QuantizePerTensorDescriptor {
        numel,
        q_min: -128,
        q_max: 127,
        input_element: ElementKind::F32,
        output_element: ElementKind::S8,
    };
    let plan = QuantizePerTensorPlan::<f32, S8>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = QuantizePerTensorArgs::<f32, S8> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
        scale,
        zero_point,
        output: TensorMut {
            data: dev_q.as_slice_mut(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_storage = vec![S8(0); host_x.len()];
    dev_q.copy_to_host(&mut got_storage).expect("dl");
    let got: Vec<i8> = got_storage.iter().map(|s| s.0).collect();
    assert_eq!(got, host_expected, "quantize_per_tensor f32→s8 mismatch");
}
