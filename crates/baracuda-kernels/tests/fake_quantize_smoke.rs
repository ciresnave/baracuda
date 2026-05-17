//! Real-GPU smoke test for `FakeQuantizePlan<TIn>` — Phase 8 Milestone 8.1.
//!
//! Verifies `fake_quant(x) == dequant(quant(x))` cellwise — i.e. the
//! FW kernel computes a single fused round-trip that matches the
//! composition of [`QuantizePerTensorPlan`] followed by
//! [`DequantizePerTensorPlan`].

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FakeQuantizeArgs, FakeQuantizeDescriptor, FakeQuantizePlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
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
fn fake_quantize_f32_matches_quant_dequant_roundtrip() {
    let (ctx, stream) = setup();
    let scale: f32 = 0.1;
    let zero_point: i32 = 0;
    let q_min: i32 = -128;
    let q_max: i32 = 127;

    // Mix in-range and clipped values.
    let host_x: Vec<f32> = vec![
        0.0, 0.05, -0.05, 0.15, -0.15, 1.0, -1.0, 12.7, -12.8, 20.0, -20.0,
    ];
    let numel = host_x.len() as i32;

    // CPU reference: quantize-then-dequantize.
    let host_expected: Vec<f32> = host_x
        .iter()
        .map(|&x| {
            let r = ((x / scale).round_ties_even() as i32 + zero_point).clamp(q_min, q_max);
            scale * ((r - zero_point) as f32)
        })
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let desc = FakeQuantizeDescriptor {
        numel,
        q_min,
        q_max,
        input_element: ElementKind::F32,
    };
    let plan = FakeQuantizePlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = FakeQuantizeArgs::<f32> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
        scale,
        zero_point,
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(
            diff < 1e-5,
            "fake_quantize roundtrip @ {i}: got {g}, expected {e}, diff {diff}"
        );
    }
}
