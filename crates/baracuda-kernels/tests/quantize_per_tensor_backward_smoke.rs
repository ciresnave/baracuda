//! Real-GPU smoke test for `QuantizePerTensorBackwardPlan<TIn, TOut>`
//! (Phase 8 Milestone 8.1, STE).
//!
//! Verifies:
//! - `dx[i] = dy[i] / scale` for cells where `round(x[i]/scale)+zp` is in
//!   `[q_min, q_max]`.
//! - `dx[i] = 0` for cells where the rounded result is out of range
//!   (clipped in FW → no gradient flow).
//!
//! `#[ignore]` by default.

use std::marker::PhantomData;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, QuantizePerTensorBackwardArgs,
    QuantizePerTensorBackwardDescriptor, QuantizePerTensorBackwardPlan, S8, TensorMut, TensorRef,
    Workspace,
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
fn quantize_per_tensor_backward_f32_s8_ste() {
    let (ctx, stream) = setup();
    let scale: f32 = 0.1;
    let zero_point: i32 = 0;
    // Mix in-range and out-of-range:
    //   x =  0.05  → r =   0  → in_range  → dx = dy / 0.1 = 10·dy
    //   x = -1.27  → r = -13  → in_range  → dx = 10·dy
    //   x = -20.0  → r = -200 → out_range → dx = 0
    //   x =  20.0  → r =  200 → out_range → dx = 0
    let host_x: Vec<f32> = vec![0.05, -1.27, -20.0, 20.0];
    let host_dy: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    // Expected: 10·dy where in_range, else 0.
    let host_expected: Vec<f32> = vec![10.0, 20.0, 0.0, 0.0];
    let numel = host_x.len() as i32;

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc dx");

    let desc = QuantizePerTensorBackwardDescriptor {
        numel,
        q_min: -128,
        q_max: 127,
        input_element: ElementKind::F32,
        output_element: ElementKind::S8,
    };
    let plan = QuantizePerTensorBackwardPlan::<f32, S8>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .expect("select");
    let args = QuantizePerTensorBackwardArgs::<f32, S8> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
        scale,
        zero_point,
        d_output: TensorRef {
            data: dev_dy.as_slice(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
        d_input: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: [numel],
            stride: contiguous_stride([numel]),
        },
        _phantom: PhantomData,
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; host_x.len()];
    dev_dx.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(
            diff < 1e-5,
            "quantize BW STE @ {i}: got {g}, expected {e}, diff {diff}"
        );
    }
}
