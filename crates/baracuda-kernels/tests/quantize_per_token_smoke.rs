//! Real-GPU smoke test for `QuantizePerTokenPlan<f32, S8>` (Phase 8 8.2).
//!
//! `[N=2, D=4]` fixture with two distinct per-row scales. CPU reference
//! verifies the kernel matches `clamp(round(x/s)+zp, qmin, qmax)`. Test
//! includes one clipped element so the saturation path is exercised.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, QuantizePerTokenArgs,
    QuantizePerTokenDescriptor, QuantizePerTokenPlan, TensorMut, TensorRef, Workspace, S8,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn cpu_quantize_per_token_s8(
    n: usize,
    d: usize,
    x: &[f32],
    scale: &[f32],
    zp: &[i32],
    qmin: i32,
    qmax: i32,
) -> Vec<i8> {
    let mut out = vec![0i8; n * d];
    for row in 0..n {
        let s = scale[row];
        let z = zp[row];
        for col in 0..d {
            let v = x[row * d + col];
            let q_f = v / s;
            let q = (q_f.round_ties_even() as i32) + z;
            let q_clip = q.clamp(qmin, qmax);
            out[row * d + col] = q_clip as i8;
        }
    }
    out
}

#[test]
#[ignore]
fn quantize_per_token_f32_s8_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 2;
    let d: i32 = 4;
    // Row 0: small values, scale=0.1 → no clipping.
    // Row 1: contains a big value (12.5) with scale=0.5 → q=25, clipped
    //        at 25 if qmax >= 25; we choose qmax=20 to force clipping
    //        for at least one cell.
    let host_x: Vec<f32> = vec![
        // row 0:
        0.05, 0.15, -0.07, 0.30,
        // row 1: includes a clipping case.
        12.5, 1.0, -0.25, 4.0,
    ];
    let host_scale: Vec<f32> = vec![0.1, 0.5];
    let host_zp: Vec<i32> = vec![0, 0];
    let qmin: i32 = -20;
    let qmax: i32 = 20;

    let expected = cpu_quantize_per_token_s8(
        n as usize,
        d as usize,
        &host_x,
        &host_scale,
        &host_zp,
        qmin,
        qmax,
    );
    // Sanity: at least one element should saturate.
    assert!(expected.iter().any(|&v| v == 20 || v == -20));

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_scale = DeviceBuffer::from_slice(&ctx, &host_scale).expect("up s");
    let dev_zp = DeviceBuffer::from_slice(&ctx, &host_zp).expect("up zp");
    let mut dev_q: DeviceBuffer<S8> =
        DeviceBuffer::zeros(&ctx, (n * d) as usize).expect("alloc q");

    let desc = QuantizePerTokenDescriptor {
        n,
        d,
        q_min: qmin,
        q_max: qmax,
        input_element: ElementKind::F32,
        output_element: ElementKind::S8,
    };
    let plan = QuantizePerTokenPlan::<f32, S8>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = QuantizePerTokenArgs::<f32, S8> {
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
        output: TensorMut {
            data: dev_q.as_slice_mut(),
            shape: [n, d],
            stride: contiguous_stride([n, d]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![S8(0); (n * d) as usize];
    dev_q.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.0, *e, "quant mismatch at idx {i}: got {} expected {}", g.0, e);
    }
}
