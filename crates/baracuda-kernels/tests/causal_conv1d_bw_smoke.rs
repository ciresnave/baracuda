//! Real-GPU smoke test for Phase 50 [`CausalConv1dBackwardPlan`].
//!
//! Validates the BW kernel via finite-difference (FD) on a tiny shape.
//! Each gradient (`dx`, `dw`, `db`) is compared against a central-
//! difference FD estimate at a handful of probe cells. Uses generous
//! tolerance (5e-2 relative) because FD at f32 has its own noise floor.

#![cfg(feature = "mamba")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CausalConv1dArgs, CausalConv1dBackwardArgs,
    CausalConv1dBackwardDescriptor, CausalConv1dBackwardPlan, CausalConv1dDescriptor,
    CausalConv1dPlan, ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const B: usize = 1;
const C: usize = 2;
const L: usize = 8;
const W: usize = 3;

fn run_fw_and_sum(
    ctx: &Context, stream: &Stream,
    x_host: &[f32], weight_host: &[f32], bias_host: &[f32],
    use_silu: bool,
) -> f32 {
    let x_dev = DeviceBuffer::from_slice(ctx, x_host).expect("x");
    let weight_dev = DeviceBuffer::from_slice(ctx, weight_host).expect("w");
    let bias_dev = DeviceBuffer::from_slice(ctx, bias_host).expect("b");
    let mut y_dev = DeviceBuffer::<f32>::zeros(ctx, B * C * L).expect("y");

    let desc = CausalConv1dDescriptor {
        batch_size: B as i32, channels: C as i32, seq_len: L as i32,
        width: W as i32, use_silu, element: ElementKind::F32,
    };
    let plan = CausalConv1dPlan::<f32>::select(stream, &desc, PlanPreference::default())
        .expect("select");
    let shape_x: [i32; 3] = [B as i32, C as i32, L as i32];
    let shape_w: [i32; 2] = [C as i32, W as i32];
    let shape_b: [i32; 1] = [C as i32];
    plan.run(stream, Workspace::None, CausalConv1dArgs {
        x: TensorRef { data: x_dev.as_slice(), shape: shape_x, stride: contiguous_stride(shape_x) },
        weight: TensorRef { data: weight_dev.as_slice(), shape: shape_w, stride: contiguous_stride(shape_w) },
        bias: Some(TensorRef { data: bias_dev.as_slice(), shape: shape_b, stride: contiguous_stride(shape_b) }),
        y: TensorMut { data: y_dev.as_slice_mut(), shape: shape_x, stride: contiguous_stride(shape_x) },
    }).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0.0f32; B * C * L];
    y_dev.copy_to_host(&mut got).expect("dl");
    got.iter().sum()
}

#[test]
#[ignore]
fn causal_conv1d_bw_dx_dw_db_match_fd() {
    let (ctx, stream) = setup();
    let use_silu = false;

    let x_host: Vec<f32> = (0..B * C * L).map(|i| (i as f32) * 0.05).collect();
    let weight_host: Vec<f32> = (0..C * W).map(|i| ((i as f32) * 0.13).sin() * 0.3).collect();
    let bias_host: Vec<f32> = (0..C).map(|i| (i as f32) * 0.07).collect();

    let x_dev = DeviceBuffer::from_slice(&ctx, &x_host).expect("x");
    let weight_dev = DeviceBuffer::from_slice(&ctx, &weight_host).expect("w");
    let bias_dev = DeviceBuffer::from_slice(&ctx, &bias_host).expect("b");
    let dy_host = vec![1.0f32; B * C * L];
    let dy_dev = DeviceBuffer::from_slice(&ctx, &dy_host).expect("dy");
    let mut dx_dev = DeviceBuffer::<f32>::zeros(&ctx, B * C * L).expect("dx");
    let mut dw_dev = DeviceBuffer::<f32>::zeros(&ctx, C * W).expect("dw");
    let mut db_dev = DeviceBuffer::<f32>::zeros(&ctx, C).expect("db");

    let bw_desc = CausalConv1dBackwardDescriptor {
        batch_size: B as i32, channels: C as i32, seq_len: L as i32,
        width: W as i32, use_silu, element: ElementKind::F32,
    };
    let bw_plan = CausalConv1dBackwardPlan::<f32>::select(&stream, &bw_desc, PlanPreference::default())
        .expect("bw select");

    let shape_x: [i32; 3] = [B as i32, C as i32, L as i32];
    let shape_w: [i32; 2] = [C as i32, W as i32];
    let shape_b: [i32; 1] = [C as i32];
    bw_plan.run(&stream, Workspace::None, CausalConv1dBackwardArgs {
        x: TensorRef { data: x_dev.as_slice(), shape: shape_x, stride: contiguous_stride(shape_x) },
        weight: TensorRef { data: weight_dev.as_slice(), shape: shape_w, stride: contiguous_stride(shape_w) },
        bias: Some(TensorRef { data: bias_dev.as_slice(), shape: shape_b, stride: contiguous_stride(shape_b) }),
        dy: TensorRef { data: dy_dev.as_slice(), shape: shape_x, stride: contiguous_stride(shape_x) },
        dx: TensorMut { data: dx_dev.as_slice_mut(), shape: shape_x, stride: contiguous_stride(shape_x) },
        dw: TensorMut { data: dw_dev.as_slice_mut(), shape: shape_w, stride: contiguous_stride(shape_w) },
        db: Some(TensorMut { data: db_dev.as_slice_mut(), shape: shape_b, stride: contiguous_stride(shape_b) }),
    }).expect("bw run");
    stream.synchronize().expect("sync");

    let mut dx_got = vec![0.0f32; B * C * L];
    let mut dw_got = vec![0.0f32; C * W];
    let mut db_got = vec![0.0f32; C];
    dx_dev.copy_to_host(&mut dx_got).expect("dl dx");
    dw_dev.copy_to_host(&mut dw_got).expect("dl dw");
    db_dev.copy_to_host(&mut db_got).expect("dl db");

    let eps = 1e-3f32;
    let tol = 5e-2f32;

    for &i in &[0usize, B*C*L / 2, B*C*L - 1] {
        let mut xp = x_host.clone(); xp[i] += eps;
        let mut xm = x_host.clone(); xm[i] -= eps;
        let yp = run_fw_and_sum(&ctx, &stream, &xp, &weight_host, &bias_host, use_silu);
        let ym = run_fw_and_sum(&ctx, &stream, &xm, &weight_host, &bias_host, use_silu);
        let fd = (yp - ym) / (2.0 * eps);
        let analytic = dx_got[i];
        let diff = (fd - analytic).abs();
        let scale = fd.abs().max(analytic.abs()).max(1e-3);
        assert!(diff < tol * scale, "dx[{}]: analytic={} fd={} diff={}", i, analytic, fd, diff);
    }

    for &i in &[0usize, C*W / 2, C*W - 1] {
        let mut wp = weight_host.clone(); wp[i] += eps;
        let mut wm = weight_host.clone(); wm[i] -= eps;
        let yp = run_fw_and_sum(&ctx, &stream, &x_host, &wp, &bias_host, use_silu);
        let ym = run_fw_and_sum(&ctx, &stream, &x_host, &wm, &bias_host, use_silu);
        let fd = (yp - ym) / (2.0 * eps);
        let analytic = dw_got[i];
        let diff = (fd - analytic).abs();
        let scale = fd.abs().max(analytic.abs()).max(1e-3);
        assert!(diff < tol * scale, "dw[{}]: analytic={} fd={} diff={}", i, analytic, fd, diff);
    }

    for i in 0..C {
        let mut bp = bias_host.clone(); bp[i] += eps;
        let mut bm = bias_host.clone(); bm[i] -= eps;
        let yp = run_fw_and_sum(&ctx, &stream, &x_host, &weight_host, &bp, use_silu);
        let ym = run_fw_and_sum(&ctx, &stream, &x_host, &weight_host, &bm, use_silu);
        let fd = (yp - ym) / (2.0 * eps);
        let analytic = db_got[i];
        let diff = (fd - analytic).abs();
        let scale = fd.abs().max(analytic.abs()).max(1e-3);
        assert!(diff < tol * scale, "db[{}]: analytic={} fd={} diff={}", i, analytic, fd, diff);
    }
}
