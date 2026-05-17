//! Real-GPU smoke test for `QuantizePerGroupPlan<f32, S8>`.
//! `[outer=2, axis_size=8]`, `group_size=4` → 2 groups per row, so
//! `scale` / `zp` shape is `[2, 2]`.
//!
//! `#[ignore]`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, QuantizePerGroupArgs,
    QuantizePerGroupDescriptor, QuantizePerGroupPlan, TensorMut, TensorRef, Workspace, S8,
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
fn quantize_per_group_f32_s8_basic() {
    let (ctx, stream) = setup();
    let outer: i32 = 2;
    let axis_size: i32 = 8;
    let group_size: i32 = 4;
    let num_groups: i32 = axis_size / group_size; // = 2
    let qmin: i32 = -10;
    let qmax: i32 = 10;

    // Two rows, two groups per row. Construct so first group of row 1
    // has a saturating element.
    let host_x: Vec<f32> = vec![
        // row 0:
        0.1, 0.2, 0.3, -0.4, // group (0, 0): scale 0.1, in-range
        1.0, 2.0, -3.0, 4.0, // group (0, 1): scale 1.0
        // row 1:
        15.0, 1.0, 0.5, -1.0, // group (1, 0): scale 0.5, 15.0/0.5 = 30 → saturates to 10
        0.05, 0.10, 0.15, -0.20, // group (1, 1): scale 0.1
    ];
    let host_scale: Vec<f32> = vec![0.1, 1.0, 0.5, 0.1];
    let host_zp: Vec<i32> = vec![0, 0, 0, 0];

    let n_per_row = axis_size as usize;
    let total = (outer * axis_size) as usize;
    let mut expected = vec![0i8; total];
    for o in 0..outer as usize {
        for j in 0..axis_size as usize {
            let g_idx = j / group_size as usize;
            let sg = o * num_groups as usize + g_idx;
            let s = host_scale[sg];
            let zp = host_zp[sg];
            let x = host_x[o * n_per_row + j];
            let q = ((x / s).round_ties_even() as i32 + zp).clamp(qmin, qmax);
            expected[o * n_per_row + j] = q as i8;
        }
    }
    // sanity — at least one saturation occurs.
    assert!(expected.iter().any(|&v| v == qmax as i8 || v == qmin as i8));

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_scale = DeviceBuffer::from_slice(&ctx, &host_scale).expect("up s");
    let dev_zp = DeviceBuffer::from_slice(&ctx, &host_zp).expect("up zp");
    let mut dev_q: DeviceBuffer<S8> = DeviceBuffer::zeros(&ctx, total).expect("alloc q");

    let desc = QuantizePerGroupDescriptor {
        outer_size: outer,
        axis_size,
        group_size,
        q_min: qmin,
        q_max: qmax,
        input_element: ElementKind::F32,
        output_element: ElementKind::S8,
    };
    let plan = QuantizePerGroupPlan::<f32, S8>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = QuantizePerGroupArgs::<f32, S8> {
        input: TensorRef {
            data: dev_x.as_slice(),
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
            data: dev_q.as_slice_mut(),
            shape: [outer, axis_size],
            stride: contiguous_stride([outer, axis_size]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![S8(0); total];
    dev_q.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.0, *e, "quant mismatch at idx {i}: got {} expected {}", g.0, e);
    }
}
