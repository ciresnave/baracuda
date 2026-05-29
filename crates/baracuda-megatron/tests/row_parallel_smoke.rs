//! RowParallelLinear — single-rank smoke test.
//!
//! Same shape as `column_parallel_smoke` but with the input-split
//! plan. On a single-rank communicator `X_local == X` (the input
//! isn't actually sharded) and `all_reduce` is a D2D copy, so the
//! plan reduces to a plain Linear layer.

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_megatron::{RowParallelLinearPlan, TensorParallelContext};
use baracuda_nccl::Communicator;

fn linear_forward_cpu(x: &[f32], w: &[f32], batch: usize, in_f: usize, out_f: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; batch * out_f];
    for b in 0..batch {
        for o in 0..out_f {
            let mut acc = 0.0f64;
            for i in 0..in_f {
                acc += x[b * in_f + i] as f64 * w[o * in_f + i] as f64;
            }
            y[b * out_f + o] = acc as f32;
        }
    }
    y
}

fn linear_backward_cpu(
    x: &[f32],
    w: &[f32],
    dy: &[f32],
    batch: usize,
    in_f: usize,
    out_f: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0f32; batch * in_f];
    let mut dw = vec![0.0f32; out_f * in_f];
    for b in 0..batch {
        for i in 0..in_f {
            let mut acc = 0.0f64;
            for o in 0..out_f {
                acc += dy[b * out_f + o] as f64 * w[o * in_f + i] as f64;
            }
            dx[b * in_f + i] = acc as f32;
        }
    }
    for o in 0..out_f {
        for i in 0..in_f {
            let mut acc = 0.0f64;
            for b in 0..batch {
                acc += dy[b * out_f + o] as f64 * x[b * in_f + i] as f64;
            }
            dw[o * in_f + i] = acc as f32;
        }
    }
    (dx, dw)
}

fn try_bringup(device: i32) -> Option<Communicator> {
    match Communicator::new_single_gpu(device) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("NCCL not available; skipping: {e:?}");
            None
        }
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU + NCCL"]
fn row_parallel_f32_forward_single_rank_matches_linear() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0u32).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let comm = match try_bringup(0) {
        Some(c) => c,
        None => return,
    };
    assert_eq!(comm.world_size(), 1);

    let batch = 4usize;
    let in_f = 24usize;
    let out_f = 16usize;

    let x_host: Vec<f32> = (0..batch * in_f).map(|i| (i as f32) * 0.013 - 0.4).collect();
    let w_host: Vec<f32> = (0..out_f * in_f)
        .map(|i| ((i % 19) as f32) * 0.06 - 0.35)
        .collect();

    let expected = linear_forward_cpu(&x_host, &w_host, batch, in_f, out_f);

    let x_dev = DeviceBuffer::from_slice(&ctx, &x_host).unwrap();
    let w_dev = DeviceBuffer::from_slice(&ctx, &w_host).unwrap();
    let mut y_partial: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, batch * out_f).unwrap();
    let mut y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, batch * out_f).unwrap();

    let tpctx = TensorParallelContext::new(&comm, in_f as i32, out_f as i32);
    let plan = RowParallelLinearPlan::<f32>::new(&tpctx, batch as i32).expect("RowParallel plan");
    assert_eq!(plan.in_per_rank(), in_f as i32);

    plan.forward(&stream, &x_dev, &w_dev, None, &mut y_partial, &mut y)
        .expect("forward");
    stream.synchronize().unwrap();

    let mut got = vec![0.0f32; batch * out_f];
    y.copy_to_host(&mut got).unwrap();

    for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "[{i}] forward mismatch: got {g}, expected {e}"
        );
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU + NCCL"]
fn row_parallel_f32_backward_single_rank_matches_linear() {
    baracuda_driver::init().unwrap();
    let device = Device::get(0u32).unwrap();
    let ctx = Context::new(&device).unwrap();
    let stream = Stream::new(&ctx).unwrap();

    let comm = match try_bringup(0) {
        Some(c) => c,
        None => return,
    };

    let batch = 5usize;
    let in_f = 16usize;
    let out_f = 12usize;

    let x_host: Vec<f32> = (0..batch * in_f).map(|i| (i as f32) * 0.014).collect();
    let w_host: Vec<f32> = (0..out_f * in_f)
        .map(|i| ((i % 23) as f32) * 0.04 - 0.45)
        .collect();
    let dy_host: Vec<f32> = (0..batch * out_f)
        .map(|i| ((i % 7) as f32) * 0.05 - 0.1)
        .collect();

    let (expected_dx, expected_dw) =
        linear_backward_cpu(&x_host, &w_host, &dy_host, batch, in_f, out_f);

    let x_dev = DeviceBuffer::from_slice(&ctx, &x_host).unwrap();
    let w_dev = DeviceBuffer::from_slice(&ctx, &w_host).unwrap();
    let dy_dev = DeviceBuffer::from_slice(&ctx, &dy_host).unwrap();
    let mut dx_local: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, batch * in_f).unwrap();
    let mut dw_local: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_f * in_f).unwrap();

    let tpctx = TensorParallelContext::new(&comm, in_f as i32, out_f as i32);
    let plan = RowParallelLinearPlan::<f32>::new(&tpctx, batch as i32).unwrap();

    plan.backward(
        &stream,
        &x_dev,
        &w_dev,
        &dy_dev,
        &mut dx_local,
        &mut dw_local,
    )
    .expect("backward");
    stream.synchronize().unwrap();

    let mut got_dx = vec![0.0f32; batch * in_f];
    dx_local.copy_to_host(&mut got_dx).unwrap();
    let mut got_dw = vec![0.0f32; out_f * in_f];
    dw_local.copy_to_host(&mut got_dw).unwrap();

    for (i, (g, e)) in got_dx.iter().zip(&expected_dx).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "[{i}] dx mismatch: got {g}, expected {e}"
        );
    }
    for (i, (g, e)) in got_dw.iter().zip(&expected_dw).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "[{i}] dw mismatch: got {g}, expected {e}"
        );
    }
}
