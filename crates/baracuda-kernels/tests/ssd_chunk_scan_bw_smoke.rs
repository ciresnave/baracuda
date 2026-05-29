//! Real-GPU smoke test for Phase 50 [`SsdChunkScanBackwardPlan`].
//!
//! Validates dx via finite difference at a small handful of probe
//! cells. Other grads (dB, dC, ddt, dA) sanity-checked for finiteness.

#![cfg(feature = "mamba")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SsdChunkScanArgs,
    SsdChunkScanBackwardArgs, SsdChunkScanBackwardDescriptor, SsdChunkScanBackwardPlan,
    SsdChunkScanDescriptor, SsdChunkScanPlan, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const B: usize = 1;
const L: usize = 4;
const H: usize = 1;
const D: usize = 4;
const N: usize = 4;
const CHUNK: i32 = 2;

fn run_fw_and_sum(
    ctx: &Context, stream: &Stream,
    x_host: &[f32], dt_host: &[f32], a_host: &[f32],
    b_host: &[f32], c_host: &[f32],
) -> f32 {
    let x_dev = DeviceBuffer::from_slice(ctx, x_host).expect("x");
    let dt_dev = DeviceBuffer::from_slice(ctx, dt_host).expect("dt");
    let a_dev = DeviceBuffer::from_slice(ctx, a_host).expect("a");
    let b_dev = DeviceBuffer::from_slice(ctx, b_host).expect("b");
    let c_dev = DeviceBuffer::from_slice(ctx, c_host).expect("c");
    let mut y_dev = DeviceBuffer::<f32>::zeros(ctx, B * L * H * D).expect("y");

    let desc = SsdChunkScanDescriptor {
        batch_size: B as i32, seq_len: L as i32, num_heads: H as i32,
        head_dim: D as i32, state_dim: N as i32, chunk_size: CHUNK,
        element: ElementKind::F32,
    };
    let plan = SsdChunkScanPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select");
    let shape_x: [i32; 4] = [B as i32, L as i32, H as i32, D as i32];
    let shape_dt: [i32; 3] = [B as i32, L as i32, H as i32];
    let shape_a: [i32; 1] = [H as i32];
    let shape_bn: [i32; 4] = [B as i32, L as i32, H as i32, N as i32];
    plan.run(stream, Workspace::None, SsdChunkScanArgs {
        x: TensorRef { data: x_dev.as_slice(), shape: shape_x, stride: contiguous_stride(shape_x) },
        dt: TensorRef { data: dt_dev.as_slice(), shape: shape_dt, stride: contiguous_stride(shape_dt) },
        a: TensorRef { data: a_dev.as_slice(), shape: shape_a, stride: contiguous_stride(shape_a) },
        b: TensorRef { data: b_dev.as_slice(), shape: shape_bn, stride: contiguous_stride(shape_bn) },
        c: TensorRef { data: c_dev.as_slice(), shape: shape_bn, stride: contiguous_stride(shape_bn) },
        y: TensorMut { data: y_dev.as_slice_mut(), shape: shape_x, stride: contiguous_stride(shape_x) },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0.0f32; B * L * H * D];
    y_dev.copy_to_host(&mut got).expect("dl");
    got.iter().sum()
}

#[test]
#[ignore]
fn ssd_bw_dx_matches_fd_and_others_finite() {
    let (ctx, stream) = setup();
    let x_host: Vec<f32> = (0..B * L * H * D).map(|i| (i as f32) * 0.01).collect();
    let dt_host: Vec<f32> = (0..B * L * H).map(|i| 0.1 + (i as f32) * 0.01).collect();
    let a_host: Vec<f32> = (0..H).map(|i| -0.5 - (i as f32) * 0.1).collect();
    let b_host: Vec<f32> = (0..B * L * H * N).map(|i| ((i as f32) * 0.07).sin() * 0.3).collect();
    let c_host: Vec<f32> = (0..B * L * H * N).map(|i| ((i as f32) * 0.11).cos() * 0.3).collect();

    let x_dev = DeviceBuffer::from_slice(&ctx, &x_host).expect("x");
    let dt_dev = DeviceBuffer::from_slice(&ctx, &dt_host).expect("dt");
    let a_dev = DeviceBuffer::from_slice(&ctx, &a_host).expect("a");
    let b_dev = DeviceBuffer::from_slice(&ctx, &b_host).expect("b");
    let c_dev = DeviceBuffer::from_slice(&ctx, &c_host).expect("c");
    let dy_host = vec![1.0f32; B * L * H * D];
    let dy_dev = DeviceBuffer::from_slice(&ctx, &dy_host).expect("dy");
    let mut dx_dev = DeviceBuffer::<f32>::zeros(&ctx, B * L * H * D).expect("dx");
    let mut db_dev = DeviceBuffer::<f32>::zeros(&ctx, B * L * H * N).expect("dB");
    let mut dc_dev = DeviceBuffer::<f32>::zeros(&ctx, B * L * H * N).expect("dC");
    let mut ddt_dev = DeviceBuffer::<f32>::zeros(&ctx, B * L * H).expect("ddt");
    let mut da_dev = DeviceBuffer::<f32>::zeros(&ctx, H).expect("dA");

    let bw_desc = SsdChunkScanBackwardDescriptor {
        batch_size: B as i32, seq_len: L as i32, num_heads: H as i32,
        head_dim: D as i32, state_dim: N as i32, chunk_size: CHUNK,
        element: ElementKind::F32,
    };
    let bw_plan = SsdChunkScanBackwardPlan::<f32>::select(&stream, &bw_desc, PlanPreference::default())
        .expect("bw select");

    let ws_bytes = bw_plan.workspace_size();
    let mut ws = DeviceBuffer::<u8>::zeros(&ctx, ws_bytes.max(1)).expect("ws");

    let shape_x: [i32; 4] = [B as i32, L as i32, H as i32, D as i32];
    let shape_dt: [i32; 3] = [B as i32, L as i32, H as i32];
    let shape_a: [i32; 1] = [H as i32];
    let shape_bn: [i32; 4] = [B as i32, L as i32, H as i32, N as i32];

    bw_plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()),
        SsdChunkScanBackwardArgs {
            x: TensorRef { data: x_dev.as_slice(), shape: shape_x, stride: contiguous_stride(shape_x) },
            dt: TensorRef { data: dt_dev.as_slice(), shape: shape_dt, stride: contiguous_stride(shape_dt) },
            a: TensorRef { data: a_dev.as_slice(), shape: shape_a, stride: contiguous_stride(shape_a) },
            b: TensorRef { data: b_dev.as_slice(), shape: shape_bn, stride: contiguous_stride(shape_bn) },
            c: TensorRef { data: c_dev.as_slice(), shape: shape_bn, stride: contiguous_stride(shape_bn) },
            dy: TensorRef { data: dy_dev.as_slice(), shape: shape_x, stride: contiguous_stride(shape_x) },
            dx: TensorMut { data: dx_dev.as_slice_mut(), shape: shape_x, stride: contiguous_stride(shape_x) },
            d_b: TensorMut { data: db_dev.as_slice_mut(), shape: shape_bn, stride: contiguous_stride(shape_bn) },
            d_c: TensorMut { data: dc_dev.as_slice_mut(), shape: shape_bn, stride: contiguous_stride(shape_bn) },
            d_dt: TensorMut { data: ddt_dev.as_slice_mut(), shape: shape_dt, stride: contiguous_stride(shape_dt) },
            d_a: TensorMut { data: da_dev.as_slice_mut(), shape: shape_a, stride: contiguous_stride(shape_a) },
        }).expect("bw run");
    stream.synchronize().expect("sync");

    let mut dx_got = vec![0.0f32; B * L * H * D];
    let mut db_got = vec![0.0f32; B * L * H * N];
    let mut dc_got = vec![0.0f32; B * L * H * N];
    let mut ddt_got = vec![0.0f32; B * L * H];
    let mut da_got = vec![0.0f32; H];
    dx_dev.copy_to_host(&mut dx_got).expect("dl");
    db_dev.copy_to_host(&mut db_got).expect("dl");
    dc_dev.copy_to_host(&mut dc_got).expect("dl");
    ddt_dev.copy_to_host(&mut ddt_got).expect("dl");
    da_dev.copy_to_host(&mut da_got).expect("dl");

    for v in db_got.iter().chain(dc_got.iter()).chain(ddt_got.iter()).chain(da_got.iter()) {
        assert!(v.is_finite(), "non-finite grad component: {}", v);
    }

    let eps = 1e-3f32;
    let tol = 5e-2f32;
    for &i in &[0usize, B * L * H * D - 1] {
        let mut xp = x_host.clone(); xp[i] += eps;
        let mut xm = x_host.clone(); xm[i] -= eps;
        let yp = run_fw_and_sum(&ctx, &stream, &xp, &dt_host, &a_host, &b_host, &c_host);
        let ym = run_fw_and_sum(&ctx, &stream, &xm, &dt_host, &a_host, &b_host, &c_host);
        let fd = (yp - ym) / (2.0 * eps);
        let analytic = dx_got[i];
        let diff = (fd - analytic).abs();
        let scale = fd.abs().max(analytic.abs()).max(1e-3);
        assert!(diff < tol * scale,
            "ssd dx[{}]: analytic={} fd={} diff={}", i, analytic, fd, diff);
    }
}
