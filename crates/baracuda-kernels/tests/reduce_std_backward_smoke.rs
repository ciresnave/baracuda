//! Real-GPU smoke test for reduce-std backward
//! (`ReduceBackwardPlan<T, N> + ReduceKind::Std`).
//!
//! Forward: `y = std = sqrt(var(x, axis=k, correction=c))`.
//! Backward:
//!   `dx[c] = dy[c_reduced] * (x[c] - mean[c_reduced]) /
//!            (m * y[c_reduced])`
//!   where `m = max(n - correction, 1)`.
//!
//! Needs saved `x` (full shape) AND saved `y` (keepdim shape; the
//! forward std). Caller must ensure `y != 0` per reduce group.
//!
//! Wired for all four FP dtypes (Phase 4 deferral 4.2 close-out):
//! - f32 / f64 accumulate at native precision.
//! - f16 / bf16 accumulate at f32 (Welford state stays at f32; final
//!   cast to T introduces 1 store-time ULP).
//!
//! Run with: `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test reduce_std_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ReduceBackwardArgs,
    ReduceBackwardDescriptor, ReduceBackwardPlan, ReduceKind, TensorMut, TensorRef, Workspace,
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

fn dy_index<const N: usize>(dx_coord: [i32; N], axis: usize, dy_shape: [i32; N]) -> i64 {
    let mut idx = 0i64;
    let mut stride = 1i64;
    for d in (0..N).rev() {
        let c = if d == axis { 0 } else { dx_coord[d] };
        idx += (c as i64) * stride;
        stride *= dy_shape[d] as i64;
    }
    idx
}

/// Host reference: compute mean / std per reduce group, then apply
/// Std BW: `dx[c] = dy[c_r] * (x[c] - mean[c_r]) / (m * y[c_r])`.
/// Returns (y, dx) where y = std.
fn host_std_bw_f32(
    input_shape: [i32; 3],
    axis: usize,
    correction: i32,
    x: &[f32],
    dy: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let dy_shape = {
        let mut s = input_shape;
        s[axis] = 1;
        s
    };
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let n = input_shape[axis] as f32;
    let m = ((input_shape[axis] - correction).max(1)) as f32;

    let mut sum = vec![0f32; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        sum[dy_lin] += x[x_linear as usize];
    });
    let mean: Vec<f32> = sum.iter().map(|&s| s / n).collect();

    let mut m2 = vec![0f32; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        let d = x[x_linear as usize] - mean[dy_lin];
        m2[dy_lin] += d * d;
    });
    let y: Vec<f32> = m2.iter().map(|&v| (v / m).sqrt()).collect();

    let mut dx = vec![0f32; dx_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        let diff = x[x_linear as usize] - mean[dy_lin];
        dx[x_linear as usize] = dy[dy_lin] * diff / (m * y[dy_lin]);
    });
    (y, dx)
}

fn run_case(axis: usize, correction: i32) {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    // Pick values that produce a clearly non-zero std per reduce group.
    let host_x: Vec<f32> = (0..dx_numel)
        .map(|i| {
            let f = i as f32;
            0.5 + 0.1 * f + 0.3 * (f * 0.37).sin()
        })
        .collect();
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| 0.5 + 0.25 * (i as f32)).collect();
    let (host_y, expected_dx) =
        host_std_bw_f32(input_shape, axis, correction, &host_x, &host_dy);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");

    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Std,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F32,
        correction,
    };
    let plan = ReduceBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ReduceBackwardArgs::<f32, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        x: Some(TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        }),
        y: Some(TensorRef {
            data: dev_y.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        }),
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    // Welford sum + (x-mean) cancellation + 1/(m·y) division: use a
    // relative tolerance with a moderate absolute floor — the
    // host single-pass sum and the GPU single-pass sum diverge in the
    // last bit or two when the reduce group is small (n=5 with
    // correction=0 amplifies tail-bit slop through the 1/m·y divide).
    let eps = 16.0 * f32::EPSILON;
    for i in 0..dx_numel {
        let tol = (expected_dx[i].abs() * eps).max(1e-4);
        assert!(
            (got[i] - expected_dx[i]).abs() <= tol,
            "f32 std BW axis={axis} corr={correction} @ {i}: got={} want={} diff={}",
            got[i],
            expected_dx[i],
            (got[i] - expected_dx[i]).abs()
        );
    }
}

#[test]
#[ignore]
fn std_backward_f32_3d_axis1_corr1() {
    run_case(1, 1);
}

#[test]
#[ignore]
fn std_backward_f32_3d_axis2_corr0() {
    run_case(2, 0);
}

#[test]
#[ignore]
fn std_backward_f32_3d_axis0_corr1() {
    run_case(0, 1);
}

// ----- f16 / bf16 / f64 dtype fanout (Phase 4 deferral 4.2 close-out) -----

fn run_case_f16(axis: usize, correction: i32) {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    // Larger spread so std stays well above zero (Std BW divides by y;
    // y close to 0 amplifies error).
    let host_x_f32: Vec<f32> = (0..dx_numel)
        .map(|i| {
            let f = i as f32;
            0.5 + 0.1 * f + 0.3 * (f * 0.37).sin()
        })
        .collect();
    let host_dy_f32: Vec<f32> = (0..dy_numel).map(|i| 0.5 + 0.25 * (i as f32)).collect();

    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_round: Vec<f32> = host_x.iter().map(|v| v.to_f32()).collect();
    let dy_round: Vec<f32> = host_dy.iter().map(|v| v.to_f32()).collect();
    let (host_y_f32, expected_dx_f32) =
        host_std_bw_f32(input_shape, axis, correction, &x_round, &dy_round);
    let host_y: Vec<f16> = host_y_f32.iter().map(|&v| f16::from_f32(v)).collect();
    // Sanity: every per-group host_y must round to a strictly nonzero
    // f16 so the kernel's `1/(m*y)` is well-defined.
    for (i, &yv) in host_y.iter().enumerate() {
        assert!(
            yv.to_f32() != 0.0,
            "f16 std BW test setup: host_y[{i}] = 0 (axis={axis} corr={correction}) — \
             pick inputs with larger spread"
        );
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");

    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Std,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F16,
        correction,
    };
    let plan = ReduceBackwardPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ReduceBackwardArgs::<f16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("dl");

    // f16 mantissa ~9.77e-4. Std BW does a divide by (m*y); when y is
    // small the result blows up but the relative error stays bounded.
    // Allow 8 ULP relative + a modest absolute floor.
    let eps = 8.0 * 9.77e-4_f32;
    for i in 0..dx_numel {
        let g = got[i].to_f32();
        let e = expected_dx_f32[i];
        let tol = (e.abs() * eps).max(eps);
        let diff = (g - e).abs();
        assert!(
            diff <= tol,
            "f16 std BW axis={axis} corr={correction} @ {i}: got={g} want={e} diff={diff} tol={tol}"
        );
    }
}

fn run_case_bf16(axis: usize, correction: i32) {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    // bf16 has only 7 mantissa bits — a small std along the reduce
    // axis rounds to zero, which then makes both the GPU `1/(m*y)` and
    // the host reference divide by zero. Use a wider spread so the
    // bf16-quantized std is comfortably nonzero on every reduce group.
    let host_x_f32: Vec<f32> = (0..dx_numel)
        .map(|i| {
            let f = i as f32;
            0.5 + 0.4 * f + 1.5 * (f * 0.41).sin()
        })
        .collect();
    let host_dy_f32: Vec<f32> = (0..dy_numel).map(|i| 0.5 + 0.25 * (i as f32)).collect();

    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let x_round: Vec<f32> = host_x.iter().map(|v| v.to_f32()).collect();
    let dy_round: Vec<f32> = host_dy.iter().map(|v| v.to_f32()).collect();
    let (host_y_f32, expected_dx_f32) =
        host_std_bw_f32(input_shape, axis, correction, &x_round, &dy_round);
    let host_y: Vec<bf16> = host_y_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    // Sanity: every per-group host_y must round to a strictly nonzero
    // bf16 so the kernel's `1/(m*y)` is well-defined.
    for (i, &yv) in host_y.iter().enumerate() {
        assert!(
            yv.to_f32() != 0.0,
            "bf16 std BW test setup: host_y[{i}] = 0 (axis={axis} corr={correction}) — \
             pick inputs with larger spread"
        );
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");

    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Std,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::Bf16,
        correction,
    };
    let plan = ReduceBackwardPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ReduceBackwardArgs::<bf16, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("dl");

    let eps = 8.0 * 7.81e-3_f32;
    for i in 0..dx_numel {
        let g = got[i].to_f32();
        let e = expected_dx_f32[i];
        let tol = (e.abs() * eps).max(eps);
        let diff = (g - e).abs();
        assert!(
            diff <= tol,
            "bf16 std BW axis={axis} corr={correction} @ {i}: got={g} want={e} diff={diff} tol={tol}"
        );
    }
}

/// f64 host reference (Welford in f64 end-to-end).
fn host_std_bw_f64(
    input_shape: [i32; 3],
    axis: usize,
    correction: i32,
    x: &[f64],
    dy: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let dy_shape = {
        let mut s = input_shape;
        s[axis] = 1;
        s
    };
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let n = input_shape[axis] as f64;
    let m = ((input_shape[axis] - correction).max(1)) as f64;

    let mut sum = vec![0f64; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        sum[dy_lin] += x[x_linear as usize];
    });
    let mean: Vec<f64> = sum.iter().map(|&s| s / n).collect();

    let mut m2 = vec![0f64; dy_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        let d = x[x_linear as usize] - mean[dy_lin];
        m2[dy_lin] += d * d;
    });
    let y: Vec<f64> = m2.iter().map(|&v| (v / m).sqrt()).collect();

    let mut dx = vec![0f64; dx_numel];
    for_each_coord::<3, _>(input_shape, |coord, x_linear| {
        let dy_lin = dy_index(coord, axis, dy_shape) as usize;
        let diff = x[x_linear as usize] - mean[dy_lin];
        dx[x_linear as usize] = dy[dy_lin] * diff / (m * y[dy_lin]);
    });
    (y, dx)
}

fn run_case_f64(axis: usize, correction: i32) {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_shape = input_shape;
    dy_shape[axis] = 1;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f64> = (0..dx_numel)
        .map(|i| {
            let f = i as f64;
            0.5 + 0.1 * f + 0.3 * (f * 0.37).sin()
        })
        .collect();
    let host_dy: Vec<f64> = (0..dy_numel).map(|i| 0.5 + 0.25 * (i as f64)).collect();
    let (host_y, expected_dx) =
        host_std_bw_f64(input_shape, axis, correction, &host_x, &host_dy);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc dx");

    let desc = ReduceBackwardDescriptor {
        kind: ReduceKind::Std,
        input_shape,
        reduce_axis: axis as u8,
        element: ElementKind::F64,
        correction,
    };
    let plan = ReduceBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ReduceBackwardArgs::<f64, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape: input_shape, stride: contiguous_stride(input_shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape: dy_shape, stride: contiguous_stride(dy_shape) }),
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape: input_shape, stride: contiguous_stride(input_shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("dl");

    let eps = 16.0 * f64::EPSILON;
    for i in 0..dx_numel {
        let g = got[i];
        let e = expected_dx[i];
        let tol = (e.abs() * eps).max(1e-12);
        let diff = (g - e).abs();
        assert!(
            diff <= tol,
            "f64 std BW axis={axis} corr={correction} @ {i}: got={g} want={e} diff={diff} tol={tol}"
        );
    }
}

#[test]
#[ignore]
fn std_backward_f16_3d_axis0_corr1() {
    run_case_f16(0, 1);
}
#[test]
#[ignore]
fn std_backward_f16_3d_axis1_corr1() {
    run_case_f16(1, 1);
}
#[test]
#[ignore]
fn std_backward_f16_3d_axis2_corr0() {
    run_case_f16(2, 0);
}

#[test]
#[ignore]
fn std_backward_bf16_3d_axis0_corr1() {
    run_case_bf16(0, 1);
}
#[test]
#[ignore]
fn std_backward_bf16_3d_axis1_corr0() {
    run_case_bf16(1, 0);
}
#[test]
#[ignore]
fn std_backward_bf16_3d_axis2_corr1() {
    run_case_bf16(2, 1);
}

#[test]
#[ignore]
fn std_backward_f64_3d_axis0_corr1() {
    run_case_f64(0, 1);
}
#[test]
#[ignore]
fn std_backward_f64_3d_axis1_corr0() {
    run_case_f64(1, 0);
}
#[test]
#[ignore]
fn std_backward_f64_3d_axis2_corr1() {
    run_case_f64(2, 1);
}
