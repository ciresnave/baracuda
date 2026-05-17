//! Real-GPU smoke test for `InstanceNormBackwardPlan`.
//!
//! Verifies the GroupNorm-wrapper path (num_groups == C).
//! Numerical formula is identical to GroupNorm BW (covered by
//! `group_norm_backward_smoke`), so this is a shape / dispatch test
//! across all four FP dtypes — values are only checked finite.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, InstanceNormBackwardArgs, InstanceNormBackwardDescriptor,
    InstanceNormBackwardPlan, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn compute_saves_f32(
    n: usize, c: usize, s: usize, x: &[f32], eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let gcount = n * c;
    let mut saved_mean = vec![0f32; gcount];
    let mut saved_rstd = vec![0f32; gcount];
    for ni in 0..n {
        for ci in 0..c {
            let mut sum = 0f64;
            let mut sq = 0f64;
            for si in 0..s {
                let v = x[ni * c * s + ci * s + si] as f64;
                sum += v;
                sq += v * v;
            }
            let mean = sum / s as f64;
            let var = (sq / s as f64 - mean * mean).max(0.0);
            let k = ni * c + ci;
            saved_mean[k] = mean as f32;
            saved_rstd[k] = (1.0 / (var + eps as f64).sqrt()) as f32;
        }
    }
    (saved_mean, saved_rstd)
}

#[test]
#[ignore]
fn instance_norm_bw_f32_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 3i32;
    let h = 2i32;
    let w = 2i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let host_gamma: Vec<f32> = (0..c).map(|i| 1.0 + 0.1 * i as f32).collect();
    let (saved_mean, saved_rstd) =
        compute_saves_f32(n as usize, c as usize, s, &host_x, 1e-5);
    let gcount = (n as usize) * (c as usize);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("g");
    let dev_sm = DeviceBuffer::from_slice(&ctx, &saved_mean).expect("sm");
    let dev_sr = DeviceBuffer::from_slice(&ctx, &saved_rstd).expect("sr");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("dx");
    let mut dev_dg: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, c as usize).expect("dg");
    let mut dev_db: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, c as usize).expect("db");
    let ws_bytes = 2 * gcount * core::mem::size_of::<f32>();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let desc = InstanceNormBackwardDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        has_affine: true,
        element: ElementKind::F32,
    };
    let plan = InstanceNormBackwardPlan::<f32, 4>::select(
        &stream, &desc, PlanPreference::default()
    ).expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        InstanceNormBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
            saved_mean: TensorRef { data: dev_sm.as_slice(), shape: [gcount as i32], stride: [1] },
            saved_rstd: TensorRef { data: dev_sr.as_slice(), shape: [gcount as i32], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dgamma: Some(TensorMut { data: dev_dg.as_slice_mut(), shape: [c], stride: [1] }),
            dbeta: Some(TensorMut { data: dev_db.as_slice_mut(), shape: [c], stride: [1] }),
        },
    ).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for (i, v) in got_dx.iter().enumerate() {
        assert!(v.is_finite(), "in bw f32 dx[{i}] not finite: {v}");
    }
}

#[test]
#[ignore]
fn instance_norm_bw_f64_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 3i32;
    let h = 2i32;
    let w = 2i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 1.0 + 0.1 * i as f32).collect();
    let (saved_mean_f32, saved_rstd_f32) =
        compute_saves_f32(n as usize, c as usize, s, &host_x_f32, 1e-5);
    let gcount = (n as usize) * (c as usize);

    let host_x: Vec<f64> = host_x_f32.iter().map(|&v| v as f64).collect();
    let host_dy: Vec<f64> = host_dy_f32.iter().map(|&v| v as f64).collect();
    let host_gamma: Vec<f64> = host_gamma_f32.iter().map(|&v| v as f64).collect();
    let saved_mean: Vec<f64> = saved_mean_f32.iter().map(|&v| v as f64).collect();
    let saved_rstd: Vec<f64> = saved_rstd_f32.iter().map(|&v| v as f64).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("g");
    let dev_sm = DeviceBuffer::from_slice(&ctx, &saved_mean).expect("sm");
    let dev_sr = DeviceBuffer::from_slice(&ctx, &saved_rstd).expect("sr");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("dx");
    let mut dev_dg: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, c as usize).expect("dg");
    let mut dev_db: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, c as usize).expect("db");
    let ws_bytes = 2 * gcount * core::mem::size_of::<f64>();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let desc = InstanceNormBackwardDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        has_affine: true,
        element: ElementKind::F64,
    };
    let plan = InstanceNormBackwardPlan::<f64, 4>::select(
        &stream, &desc, PlanPreference::default()
    ).expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        InstanceNormBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
            saved_mean: TensorRef { data: dev_sm.as_slice(), shape: [gcount as i32], stride: [1] },
            saved_rstd: TensorRef { data: dev_sr.as_slice(), shape: [gcount as i32], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dgamma: Some(TensorMut { data: dev_dg.as_slice_mut(), shape: [c], stride: [1] }),
            dbeta: Some(TensorMut { data: dev_db.as_slice_mut(), shape: [c], stride: [1] }),
        },
    ).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for (i, v) in got_dx.iter().enumerate() {
        assert!(v.is_finite(), "in bw f64 dx[{i}] not finite: {v}");
    }
}

#[test]
#[ignore]
fn instance_norm_bw_f16_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 3i32;
    let h = 2i32;
    let w = 2i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 1.0 + 0.1 * i as f32).collect();
    let (saved_mean_f32, saved_rstd_f32) =
        compute_saves_f32(n as usize, c as usize, s, &host_x_f32, 1e-5);
    let gcount = (n as usize) * (c as usize);

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_gamma: Vec<f16> = host_gamma_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let saved_mean: Vec<f16> = saved_mean_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let saved_rstd: Vec<f16> = saved_rstd_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("g");
    let dev_sm = DeviceBuffer::from_slice(&ctx, &saved_mean).expect("sm");
    let dev_sr = DeviceBuffer::from_slice(&ctx, &saved_rstd).expect("sr");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("dx");
    let mut dev_dg: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, c as usize).expect("dg");
    let mut dev_db: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, c as usize).expect("db");
    let ws_bytes = 2 * gcount * core::mem::size_of::<f32>();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let desc = InstanceNormBackwardDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        has_affine: true,
        element: ElementKind::F16,
    };
    let plan = InstanceNormBackwardPlan::<f16, 4>::select(
        &stream, &desc, PlanPreference::default()
    ).expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        InstanceNormBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
            saved_mean: TensorRef { data: dev_sm.as_slice(), shape: [gcount as i32], stride: [1] },
            saved_rstd: TensorRef { data: dev_sr.as_slice(), shape: [gcount as i32], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dgamma: Some(TensorMut { data: dev_dg.as_slice_mut(), shape: [c], stride: [1] }),
            dbeta: Some(TensorMut { data: dev_db.as_slice_mut(), shape: [c], stride: [1] }),
        },
    ).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![f16::ZERO; numel];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for (i, v) in got_dx.iter().enumerate() {
        assert!(v.to_f32().is_finite(), "in bw f16 dx[{i}] not finite: {}", v.to_f32());
    }
}

#[test]
#[ignore]
fn instance_norm_bw_bf16_with_affine() {
    let (ctx, stream) = setup();
    let n = 2i32;
    let c = 3i32;
    let h = 2i32;
    let w = 2i32;
    let shape = [n, c, h, w];
    let s = (h * w) as usize;
    let numel = (n * c * h * w) as usize;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();
    let host_gamma_f32: Vec<f32> = (0..c).map(|i| 1.0 + 0.1 * i as f32).collect();
    let (saved_mean_f32, saved_rstd_f32) =
        compute_saves_f32(n as usize, c as usize, s, &host_x_f32, 1e-5);
    let gcount = (n as usize) * (c as usize);

    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_gamma: Vec<bf16> = host_gamma_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let saved_mean: Vec<bf16> = saved_mean_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let saved_rstd: Vec<bf16> = saved_rstd_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("x");
    let dev_g = DeviceBuffer::from_slice(&ctx, &host_gamma).expect("g");
    let dev_sm = DeviceBuffer::from_slice(&ctx, &saved_mean).expect("sm");
    let dev_sr = DeviceBuffer::from_slice(&ctx, &saved_rstd).expect("sr");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("dx");
    let mut dev_dg: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, c as usize).expect("dg");
    let mut dev_db: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, c as usize).expect("db");
    let ws_bytes = 2 * gcount * core::mem::size_of::<f32>();
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let desc = InstanceNormBackwardDescriptor::<4> {
        input_shape: shape,
        channel_axis: 1,
        has_affine: true,
        element: ElementKind::Bf16,
    };
    let plan = InstanceNormBackwardPlan::<bf16, 4>::select(
        &stream, &desc, PlanPreference::default()
    ).expect("sel");
    plan.run(
        &stream,
        Workspace::Borrowed(dev_ws.as_slice_mut()),
        InstanceNormBackwardArgs {
            dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
            x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
            gamma: Some(TensorRef { data: dev_g.as_slice(), shape: [c], stride: [1] }),
            saved_mean: TensorRef { data: dev_sm.as_slice(), shape: [gcount as i32], stride: [1] },
            saved_rstd: TensorRef { data: dev_sr.as_slice(), shape: [gcount as i32], stride: [1] },
            dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
            dgamma: Some(TensorMut { data: dev_dg.as_slice_mut(), shape: [c], stride: [1] }),
            dbeta: Some(TensorMut { data: dev_db.as_slice_mut(), shape: [c], stride: [1] }),
        },
    ).expect("run");
    stream.synchronize().expect("sync");

    let mut got_dx = vec![bf16::ZERO; numel];
    dev_dx.copy_to_host(&mut got_dx).expect("dl dx");
    for (i, v) in got_dx.iter().enumerate() {
        assert!(v.to_f32().is_finite(), "in bw bf16 dx[{i}] not finite: {}", v.to_f32());
    }
}
