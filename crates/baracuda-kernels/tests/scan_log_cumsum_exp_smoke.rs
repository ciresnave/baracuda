//! Real-GPU smoke test for `ScanPlan<T, N> + ScanKind::LogCumsumExp`.
//!
//! Forward: `y[k] = log(Σ_{j ≤ k} exp(x[j]))` (or `y[k] =
//! log(Σ_{j ≥ k} exp(x[j]))` when `reverse == true`). Stable-LSE
//! algorithm — kernel uses the online running-max trick to keep `exp`
//! arguments bounded.
//!
//! Input range capped at `|x| ≤ 5` so the CPU reference can compute
//! `exp(x)` directly without overflow (we use the stable formulation
//! on the device, but the host reference is the naive
//! `log(Σ exp(x - m)) + m` two-pass — same answer, easier to read).
//!
//! Run with: `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test scan_log_cumsum_exp_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ScanArgs, ScanDescriptor, ScanKind, ScanPlan,
    TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn for_each_coord<const N: usize, F: FnMut([i32; N], i64)>(shape: [i32; N], mut f: F) {
    let numel: i64 = shape.iter().map(|&d| d as i64).product();
    for linear in 0..numel {
        let mut coord = [0i32; N];
        let mut rem = linear;
        for d in (0..N).rev() {
            coord[d] = (rem % shape[d] as i64) as i32;
            rem /= shape[d] as i64;
        }
        f(coord, linear);
    }
}

// CPU reference. Two-pass stable LSE per output cell: track prefix max
// then sum `exp(x - max)`. Mathematically identical to the device
// kernel's online algorithm but easier to inspect.
fn host_log_cumsum_exp_f32<const N: usize>(
    shape: [i32; N],
    axis: usize,
    reverse: bool,
    x: &[f32],
) -> Vec<f32> {
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let mut stride = [1usize; N];
    for d in (0..N).rev().skip(1) {
        stride[d] = stride[d + 1] * shape[d + 1] as usize;
    }
    let mut y = vec![0f32; numel];
    let extent = shape[axis];
    for_each_coord::<N, _>(shape, |coord, linear| {
        let k = coord[axis];
        // First pass: max over the prefix/suffix.
        let (j_lo, j_hi) = if reverse { (k, extent - 1) } else { (0, k) };
        let mut m = f32::NEG_INFINITY;
        for j in j_lo..=j_hi {
            let mut src = coord;
            src[axis] = j;
            let mut idx = 0usize;
            for d in 0..N {
                idx += src[d] as usize * stride[d];
            }
            if x[idx] > m {
                m = x[idx];
            }
        }
        // Second pass: sum exp(x - m).
        let mut s = 0f32;
        for j in j_lo..=j_hi {
            let mut src = coord;
            src[axis] = j;
            let mut idx = 0usize;
            for d in 0..N {
                idx += src[d] as usize * stride[d];
            }
            s += (x[idx] - m).exp();
        }
        y[linear as usize] = s.ln() + m;
    });
    y
}

#[test]
#[ignore]
fn lcse_fw_f32_2d_axis_1_reverse() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let host_x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.3) % 5.0 - 2.5).collect();
    let expected = host_log_cumsum_exp_f32(shape, 1, true, &host_x);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::LogCumsumExp,
        input_shape: shape,
        scan_axis: 1,
        reverse: true,
        element: ElementKind::F32,
    };
    let plan = ScanPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = ScanArgs::<f32, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 32];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f32::EPSILON;
    for i in 0..32 {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!(
            (got[i] - expected[i]).abs() <= tol,
            "f32 lcse FW reverse @ {i}: got={} want={}",
            got[i],
            expected[i]
        );
    }
}

#[test]
#[ignore]
fn lcse_fw_f64_2d_axis_0_forward() {
    let (ctx, stream) = setup();
    let shape = [5i32, 4];
    let numel = 20;
    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.2 - 2.0).collect();
    // Build f64 reference directly.
    let mut stride = [1usize; 2];
    stride[0] = shape[1] as usize;
    let mut expected = vec![0f64; numel];
    let extent = shape[0];
    for_each_coord::<2, _>(shape, |coord, linear| {
        let k = coord[0];
        let mut m = f64::NEG_INFINITY;
        for j in 0..=k {
            let mut src = coord;
            src[0] = j;
            let mut idx = 0usize;
            for d in 0..2 {
                idx += src[d] as usize * stride[d];
            }
            if host_x[idx] > m {
                m = host_x[idx];
            }
        }
        let mut s = 0f64;
        for j in 0..=k {
            let mut src = coord;
            src[0] = j;
            let mut idx = 0usize;
            for d in 0..2 {
                idx += src[d] as usize * stride[d];
            }
            s += (host_x[idx] - m).exp();
        }
        let _ = extent; // silence unused
        expected[linear as usize] = s.ln() + m;
    });

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::LogCumsumExp,
        input_shape: shape,
        scan_axis: 0,
        reverse: false,
        element: ElementKind::F64,
    };
    let plan = ScanPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = ScanArgs::<f64, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f64::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!(
            (got[i] - expected[i]).abs() <= tol,
            "f64 lcse FW @ {i}: got={} want={}",
            got[i],
            expected[i]
        );
    }
}

#[test]
#[ignore]
fn lcse_fw_f16_2d_axis_1_forward() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.4) % 4.0 - 2.0).collect();
    let expected_f32 = host_log_cumsum_exp_f32::<2>(shape, 1, false, &host_x_f32);
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::LogCumsumExp,
        input_shape: shape,
        scan_axis: 1,
        reverse: false,
        element: ElementKind::F16,
    };
    let plan = ScanPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = ScanArgs::<f16, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    // 8 ULP-equivalent for f16 (≈ 9.77e-4 per ULP).
    let eps = 8.0 * 9.77e-4_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(
            diff <= tol,
            "f16 lcse FW @ {i}: got={} want={} diff={}",
            got[i].to_f32(),
            expected_f32[i],
            diff
        );
    }
}

#[test]
#[ignore]
fn lcse_fw_bf16_2d_axis_1_reverse() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.4) % 4.0 - 2.0).collect();
    let expected_f32 = host_log_cumsum_exp_f32::<2>(shape, 1, true, &host_x_f32);
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::LogCumsumExp,
        input_shape: shape,
        scan_axis: 1,
        reverse: true,
        element: ElementKind::Bf16,
    };
    let plan = ScanPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = ScanArgs::<bf16, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    // 8 ULP-equivalent for bf16 (≈ 7.81e-3 per ULP).
    let eps = 8.0 * 7.81e-3_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(
            diff <= tol,
            "bf16 lcse FW reverse @ {i}: got={} want={} diff={}",
            got[i].to_f32(),
            expected_f32[i],
            diff
        );
    }
}
