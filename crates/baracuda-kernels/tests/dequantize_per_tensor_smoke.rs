//! Real-GPU smoke test for `DequantizePerTensorPlan<TIn, TOut>` —
//! Phase 8 Milestone 8.1.
//!
//! Verifies s8 → f32 is the exact inverse of `quantize_per_tensor` when
//! no clipping happens — i.e. for any `q ∈ [qmin, qmax]`,
//! `dequant(q) = scale * (q - zp)` exactly.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, DequantizePerTensorArgs, DequantizePerTensorDescriptor,
    DequantizePerTensorPlan, ElementKind, PlanPreference, S8, TensorMut, TensorRef, Workspace,
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
fn dequantize_per_tensor_s8_f32_exact_inverse() {
    let (ctx, stream) = setup();
    let scale: f32 = 0.5;
    let zero_point: i32 = 3;
    // Choose q values entirely in [-128, 127] so there's no clipping in
    // the (notional) FW. Reference: x[i] = scale * (q[i] - zp).
    let host_q_storage: Vec<S8> = (-5i32..5)
        .map(|i| S8(i as i8))
        .collect();
    let host_expected: Vec<f32> = host_q_storage
        .iter()
        .map(|q| scale * ((q.0 as i32 - zero_point) as f32))
        .collect();
    let numel = host_q_storage.len() as i32;

    let dev_q = DeviceBuffer::from_slice(&ctx, &host_q_storage).expect("up q");
    let mut dev_x: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_q_storage.len()).expect("alloc x");

    let desc = DequantizePerTensorDescriptor {
        numel,
        input_element: ElementKind::F32,
        output_element: ElementKind::S8,
    };
    let plan = DequantizePerTensorPlan::<f32, S8>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .expect("select");
    let args = DequantizePerTensorArgs::<f32, S8> {
        input: TensorRef {
            data: dev_q.as_slice(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
        scale,
        zero_point,
        output: TensorMut {
            data: dev_x.as_slice_mut(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; host_q_storage.len()];
    dev_x.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "dequant s8 → f32 mismatch @ {i}: got {g}, expected {e}"
        );
    }
}
