//! Real-GPU smoke test for `QuantizePerGroupBackwardPlan<f32>` (STE).
//!
//! Gated behind the `device-tests` feature.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    ElementKind, PlanPreference, QuantizePerGroupBackwardArgs, QuantizePerGroupBackwardDescriptor,
    QuantizePerGroupBackwardPlan, TensorMut, TensorRef, Workspace, contiguous_stride,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
fn quantize_per_group_backward_f32_ste() {
    let (ctx, stream) = setup();
    let outer: i32 = 2;
    let axis_size: i32 = 8;
    let group_size: i32 = 4;
    let num_groups: i32 = axis_size / group_size;
    let qmin: i32 = -10;
    let qmax: i32 = 10;

    let host_x: Vec<f32> = vec![
        0.1, 0.2, 0.3, -0.4, 1.0, 2.0, -3.0, 4.0, 15.0, 1.0, 0.5, -1.0, 0.05, 0.10, 0.15, -0.20,
    ];
    let host_dy: Vec<f32> = (0..(outer * axis_size) as usize)
        .map(|i| (i as f32) * 0.5 + 1.0)
        .collect();
    let host_scale: Vec<f32> = vec![0.1, 1.0, 0.5, 0.1];
    let host_zp: Vec<i32> = vec![0, 0, 0, 0];

    let total = (outer * axis_size) as usize;
    let mut expected = vec![0f32; total];
    for o in 0..outer as usize {
        for j in 0..axis_size as usize {
            let g_idx = j / group_size as usize;
            let sg = o * num_groups as usize + g_idx;
            let s = host_scale[sg];
            let zp = host_zp[sg];
            let q = ((host_x[o * axis_size as usize + j] / s).round_ties_even() as i32) + zp;
            let in_range = q > qmin && q < qmax;
            expected[o * axis_size as usize + j] = if in_range {
                host_dy[o * axis_size as usize + j] / s
            } else {
                0.0
            };
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_scale = DeviceBuffer::from_slice(&ctx, &host_scale).expect("up s");
    let dev_zp = DeviceBuffer::from_slice(&ctx, &host_zp).expect("up zp");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, total).expect("alloc dx");

    let desc = QuantizePerGroupBackwardDescriptor {
        outer_size: outer,
        axis_size,
        group_size,
        q_min: qmin,
        q_max: qmax,
        input_element: ElementKind::F32,
    };
    let plan =
        QuantizePerGroupBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    let args = QuantizePerGroupBackwardArgs::<f32> {
        d_output: TensorRef::new(
            dev_dy.as_slice(),
            [outer, axis_size],
            contiguous_stride([outer, axis_size]),
        )
        .unwrap(),
        input: TensorRef::new(
            dev_x.as_slice(),
            [outer, axis_size],
            contiguous_stride([outer, axis_size]),
        )
        .unwrap(),
        scale: TensorRef::new(
            dev_scale.as_slice(),
            [outer, num_groups],
            contiguous_stride([outer, num_groups]),
        )
        .unwrap(),
        zero_point: TensorRef::new(
            dev_zp.as_slice(),
            [outer, num_groups],
            contiguous_stride([outer, num_groups]),
        )
        .unwrap(),
        d_input: TensorMut::new(
            dev_dx.as_slice_mut(),
            [outer, axis_size],
            contiguous_stride([outer, axis_size]),
        )
        .unwrap(),
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; total];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "BW mismatch @ {i}: got {g} expected {e}"
        );
    }
}
