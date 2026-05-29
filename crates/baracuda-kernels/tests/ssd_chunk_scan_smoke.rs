//! Real-GPU smoke test for Phase 50 [`SsdChunkScanPlan`] (FW).

#![cfg(feature = "mamba")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SsdChunkScanArgs, SsdChunkScanDescriptor,
    SsdChunkScanPlan, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn cpu_ref_ssd_f32(
    x: &[f32], dt: &[f32], a_h: &[f32], b: &[f32], c: &[f32],
    bsz: usize, l: usize, h: usize, d: usize, n: usize,
) -> Vec<f32> {
    let mut y = vec![0.0f32; bsz * l * h * d];
    for bi in 0..bsz {
        for hi in 0..h {
            let mut state = vec![0.0f32; d * n];
            let a = a_h[hi];
            for t in 0..l {
                let dt_t = dt[bi * l * h + t * h + hi];
                let decay = (dt_t * a).exp();
                for di in 0..d {
                    let xd = x[bi * l * h * d + t * h * d + hi * d + di];
                    for ni in 0..n {
                        let b_n = b[bi * l * h * n + t * h * n + hi * n + ni];
                        let idx = di * n + ni;
                        state[idx] = decay * state[idx] + dt_t * b_n * xd;
                    }
                }
                for di in 0..d {
                    let mut acc = 0.0f32;
                    for ni in 0..n {
                        let c_n = c[bi * l * h * n + t * h * n + hi * n + ni];
                        acc += c_n * state[di * n + ni];
                    }
                    y[bi * l * h * d + t * h * d + hi * d + di] = acc;
                }
            }
        }
    }
    y
}

fn check_close(a: &[f32], b: &[f32], tol: f32, tag: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", tag);
    for (i, (av, bv)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (av - bv).abs();
        let scale = av.abs().max(bv.abs()).max(1e-3);
        if diff > tol * scale {
            panic!("{}: mismatch at idx {} — got {}, expected {}, diff {}", tag, i, av, bv, diff);
        }
    }
}

#[test]
#[ignore]
fn ssd_chunk_scan_f32_tiny_matches_cpu_ref() {
    let (ctx, stream) = setup();

    let b = 1;
    let l = 8;
    let h = 2;
    let d = 4;
    let n = 4;
    let chunk = 4;

    let x_host: Vec<f32> = (0..b * l * h * d).map(|i| (i as f32) * 0.01).collect();
    let dt_host: Vec<f32> = (0..b * l * h).map(|i| 0.1 + (i as f32) * 0.01).collect();
    let a_host: Vec<f32> = (0..h).map(|i| -0.5 - (i as f32) * 0.1).collect();
    let b_host: Vec<f32> = (0..b * l * h * n).map(|i| ((i as f32) * 0.07).sin() * 0.3).collect();
    let c_host: Vec<f32> = (0..b * l * h * n).map(|i| ((i as f32) * 0.11).cos() * 0.3).collect();

    let x_dev = DeviceBuffer::from_slice(&ctx, &x_host).expect("x");
    let dt_dev = DeviceBuffer::from_slice(&ctx, &dt_host).expect("dt");
    let a_dev = DeviceBuffer::from_slice(&ctx, &a_host).expect("a");
    let b_dev = DeviceBuffer::from_slice(&ctx, &b_host).expect("b");
    let c_dev = DeviceBuffer::from_slice(&ctx, &c_host).expect("c");
    let mut y_dev = DeviceBuffer::<f32>::zeros(&ctx, b * l * h * d).expect("y");

    let desc = SsdChunkScanDescriptor {
        batch_size: b as i32, seq_len: l as i32, num_heads: h as i32,
        head_dim: d as i32, state_dim: n as i32, chunk_size: chunk,
        element: ElementKind::F32,
    };
    let plan = SsdChunkScanPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let shape_x: [i32; 4] = [b as i32, l as i32, h as i32, d as i32];
    let shape_dt: [i32; 3] = [b as i32, l as i32, h as i32];
    let shape_a: [i32; 1] = [h as i32];
    let shape_bn: [i32; 4] = [b as i32, l as i32, h as i32, n as i32];

    plan.run(&stream, Workspace::None, SsdChunkScanArgs {
        x: TensorRef { data: x_dev.as_slice(), shape: shape_x, stride: contiguous_stride(shape_x) },
        dt: TensorRef { data: dt_dev.as_slice(), shape: shape_dt, stride: contiguous_stride(shape_dt) },
        a: TensorRef { data: a_dev.as_slice(), shape: shape_a, stride: contiguous_stride(shape_a) },
        b: TensorRef { data: b_dev.as_slice(), shape: shape_bn, stride: contiguous_stride(shape_bn) },
        c: TensorRef { data: c_dev.as_slice(), shape: shape_bn, stride: contiguous_stride(shape_bn) },
        y: TensorMut { data: y_dev.as_slice_mut(), shape: shape_x, stride: contiguous_stride(shape_x) },
    }).expect("run");
    stream.synchronize().expect("sync");

    let mut y_got = vec![0.0f32; b * l * h * d];
    y_dev.copy_to_host(&mut y_got).expect("download");
    let y_expected = cpu_ref_ssd_f32(&x_host, &dt_host, &a_host, &b_host, &c_host,
        b, l, h, d, n);
    check_close(&y_got, &y_expected, 1e-4, "ssd_chunk_scan_f32_tiny");
}

#[test]
#[ignore]
fn ssd_chunk_scan_state_dim_too_large_rejected() {
    let (_ctx, stream) = setup();
    let desc = SsdChunkScanDescriptor {
        batch_size: 1, seq_len: 8, num_heads: 1,
        head_dim: 64, state_dim: 512,
        chunk_size: 4, element: ElementKind::F32,
    };
    let res = SsdChunkScanPlan::<f32>::select(&stream, &desc, PlanPreference::default());
    assert!(res.is_err(), "state_dim > 256 should be rejected");
}
