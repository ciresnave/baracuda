//! Real-GPU smoke test for `DequantizePerTokenPlan<f32, S8>`.
//! Verifies `y[n, d] = (q[n, d] - zp[n]) * scale[n]` — exact inverse of
//! `quantize_per_token` (modulo FW rounding).
//!
//! `#[ignore]`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, DequantizePerTokenArgs, DequantizePerTokenDescriptor, DequantizePerTokenPlan,
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
fn dequantize_per_token_f32_s8_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 2;
    let d: i32 = 4;
    let host_q: Vec<S8> = vec![
        S8(1), S8(2), S8(-3), S8(4),
        S8(20), S8(2), S8(-1), S8(8),
    ];
    let host_scale: Vec<f32> = vec![0.1, 0.5];
    let host_zp: Vec<i32> = vec![0, 0];

    let mut expected = vec![0f32; (n * d) as usize];
    for row in 0..n as usize {
        for col in 0..d as usize {
            let i = row * d as usize + col;
            let q = host_q[i].0 as f32;
            expected[i] = (q - host_zp[row] as f32) * host_scale[row];
        }
    }

    let dev_q = DeviceBuffer::from_slice(&ctx, &host_q).expect("up q");
    let dev_scale = DeviceBuffer::from_slice(&ctx, &host_scale).expect("up s");
    let dev_zp = DeviceBuffer::from_slice(&ctx, &host_zp).expect("up zp");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * d) as usize).expect("alloc y");

    let desc = DequantizePerTokenDescriptor {
        n,
        d,
        input_element: ElementKind::F32,
        output_element: ElementKind::S8,
    };
    let plan =
        DequantizePerTokenPlan::<f32, S8>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    let args = DequantizePerTokenArgs::<f32, S8> {
        input: TensorRef {
            data: dev_q.as_slice(),
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
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [n, d],
            stride: contiguous_stride([n, d]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (n * d) as usize];
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
