//! Real-GPU smoke test for `QuantizePerTokenBackwardPlan<f32>`.
//! STE verification: gradient passes through (divided by scale) inside
//! the un-clipped range, zero outside.
//!
//! `#[ignore]`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, QuantizePerTokenBackwardArgs,
    QuantizePerTokenBackwardDescriptor, QuantizePerTokenBackwardPlan, TensorMut, TensorRef,
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
fn quantize_per_token_backward_f32_ste() {
    let (ctx, stream) = setup();
    let n: i32 = 2;
    let d: i32 = 4;
    let host_x: Vec<f32> = vec![
        0.05, 0.15, -0.07, 0.30,    // row 0 — all in range with scale=0.1, qmin=-20, qmax=20
        12.5, 1.0, -0.25, 4.0,      // row 1 — first elem will saturate with scale=0.5, qmin=-20, qmax=20
    ];
    let host_dy: Vec<f32> = (0..(n * d) as usize).map(|i| (i as f32) + 1.0).collect();
    let host_scale: Vec<f32> = vec![0.1, 0.5];
    let host_zp: Vec<i32> = vec![0, 0];
    let qmin: i32 = -20;
    let qmax: i32 = 20;

    // CPU reference for STE BW.
    let mut expected = vec![0f32; (n * d) as usize];
    for row in 0..n as usize {
        for col in 0..d as usize {
            let i = row * d as usize + col;
            let s = host_scale[row];
            let zp = host_zp[row];
            let q = ((host_x[i] / s).round_ties_even() as i32) + zp;
            let in_range = q > qmin && q < qmax;
            expected[i] = if in_range { host_dy[i] / s } else { 0.0 };
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_scale = DeviceBuffer::from_slice(&ctx, &host_scale).expect("up s");
    let dev_zp = DeviceBuffer::from_slice(&ctx, &host_zp).expect("up zp");
    let mut dev_dx: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * d) as usize).expect("alloc dx");

    let desc = QuantizePerTokenBackwardDescriptor {
        n,
        d,
        q_min: qmin,
        q_max: qmax,
        input_element: ElementKind::F32,
    };
    let plan = QuantizePerTokenBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = QuantizePerTokenBackwardArgs::<f32> {
        d_output: TensorRef {
            data: dev_dy.as_slice(),
            shape: [n, d],
            stride: contiguous_stride([n, d]),
        },
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [n, d],
            stride: contiguous_stride([n, d]),
        },
        scale: TensorRef {
            data: dev_scale.as_slice(),
            shape: [n],
            stride: contiguous_stride([n]),
        },
        zero_point: TensorRef {
            data: dev_zp.as_slice(),
            shape: [n],
            stride: contiguous_stride([n]),
        },
        d_input: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: [n, d],
            stride: contiguous_stride([n, d]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (n * d) as usize];
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
