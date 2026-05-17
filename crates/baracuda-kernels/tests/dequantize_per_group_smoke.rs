//! Real-GPU smoke test for `DequantizePerGroupPlan<f32, S8>`.
//!
//! `#[ignore]`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, DequantizePerGroupArgs, DequantizePerGroupDescriptor, DequantizePerGroupPlan,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace, S8,
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
fn dequantize_per_group_f32_s8_basic() {
    let (ctx, stream) = setup();
    let outer: i32 = 2;
    let axis_size: i32 = 8;
    let group_size: i32 = 4;
    let num_groups: i32 = axis_size / group_size;

    let host_q: Vec<S8> = vec![
        S8(1), S8(2), S8(-3), S8(4),
        S8(5), S8(-6), S8(7), S8(-8),
        S8(10), S8(2), S8(0), S8(-1),
        S8(3), S8(-3), S8(1), S8(-2),
    ];
    let host_scale: Vec<f32> = vec![0.1, 1.0, 0.5, 0.1];
    let host_zp: Vec<i32> = vec![0, 0, 0, 0];

    let total = (outer * axis_size) as usize;
    let mut expected = vec![0f32; total];
    for o in 0..outer as usize {
        for j in 0..axis_size as usize {
            let g_idx = j / group_size as usize;
            let sg = o * num_groups as usize + g_idx;
            let q = host_q[o * axis_size as usize + j].0 as f32;
            expected[o * axis_size as usize + j] =
                (q - host_zp[sg] as f32) * host_scale[sg];
        }
    }

    let dev_q = DeviceBuffer::from_slice(&ctx, &host_q).expect("up q");
    let dev_scale = DeviceBuffer::from_slice(&ctx, &host_scale).expect("up s");
    let dev_zp = DeviceBuffer::from_slice(&ctx, &host_zp).expect("up zp");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, total).expect("alloc y");

    let desc = DequantizePerGroupDescriptor {
        outer_size: outer,
        axis_size,
        group_size,
        input_element: ElementKind::F32,
        output_element: ElementKind::S8,
    };
    let plan =
        DequantizePerGroupPlan::<f32, S8>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    let args = DequantizePerGroupArgs::<f32, S8> {
        input: TensorRef {
            data: dev_q.as_slice(),
            shape: [outer, axis_size],
            stride: contiguous_stride([outer, axis_size]),
        },
        scale: TensorRef {
            data: dev_scale.as_slice(),
            shape: [outer, num_groups],
            stride: contiguous_stride([outer, num_groups]),
        },
        zero_point: TensorRef {
            data: dev_zp.as_slice(),
            shape: [outer, num_groups],
            stride: contiguous_stride([outer, num_groups]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [outer, axis_size],
            stride: contiguous_stride([outer, axis_size]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; total];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "dequant mismatch @ {i}: got {g} expected {e}"
        );
    }
}
