//! Real-GPU smoke test for `ScanBackwardPlan<T, N> + ScanKind::Cummax`.
//!
//! Gradient flows to the first-occurrence argmax (PyTorch semantics).
//! Tests use strictly non-monotone (but tie-free) input sequences.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ScanBackwardArgs, ScanBackwardDescriptor,
    ScanBackwardPlan, ScanKind, TensorMut, TensorRef, Workspace,
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

/// Walk every (non-axis) coord; along the scan axis from i_start to i_end,
/// track running first-occurrence argmax; for each step, dy at that
/// step is routed to dx[argmax_pos].
fn host_cummax_bw_f32<const N: usize>(
    shape: [i32; N],
    axis: usize,
    fw_reverse: bool,
    dy: &[f32],
    x: &[f32],
) -> Vec<f32> {
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let mut stride = [1usize; N];
    for d in (0..N).rev().skip(1) {
        stride[d] = stride[d + 1] * shape[d + 1] as usize;
    }
    let mut dx = vec![0f32; numel];
    let extent = shape[axis];
    // Enumerate every "row" of the scan axis: pick a base coord with
    // axis index 0, then iterate the axis.
    let mut base_shape = shape;
    base_shape[axis] = 1;
    for_each_coord::<N, _>(base_shape, |base_coord, _| {
        let mut running_winner = f32::NEG_INFINITY;
        let mut running_arg: i32 = -1;
        let (i_start, i_end, i_step) = if fw_reverse { (extent - 1, -1, -1) } else { (0, extent, 1) };
        let mut ii = i_start;
        while ii != i_end {
            let mut coord = base_coord;
            coord[axis] = ii;
            let mut idx = 0usize;
            for d in 0..N { idx += coord[d] as usize * stride[d]; }
            let v = x[idx];
            let better = running_arg < 0 || v > running_winner;
            if better {
                running_winner = v;
                running_arg = ii;
            }
            // Route dy[ii] to dx[running_arg].
            let mut dx_coord = base_coord;
            dx_coord[axis] = running_arg;
            let mut dx_idx = 0usize;
            for d in 0..N { dx_idx += dx_coord[d] as usize * stride[d]; }
            dx[dx_idx] += dy[idx];
            ii += i_step;
        }
    });
    dx
}

#[test]
#[ignore]
fn cummax_bw_f32_1d_forward_fw() {
    let (ctx, stream) = setup();
    let shape = [8i32];
    let host_x: Vec<f32> = vec![1.0, -2.0, 3.0, 0.5, 4.5, -1.0, 5.0, 2.0];
    let host_dy: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let expected = host_cummax_bw_f32::<1>(shape, 0, false, &host_dy, &host_x);
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 8).expect("alloc dx");
    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cummax,
        input_shape: shape,
        scan_axis: 0,
        reverse: false,
        element: ElementKind::F32,
    };
    let plan = ScanBackwardPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ScanBackwardArgs::<f32, 1> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) }),
        y: None,
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 8];
    dev_dx.copy_to_host(&mut got).expect("dl");
    for i in 0..8 {
        assert!((got[i] - expected[i]).abs() <= 4.0 * f32::EPSILON * expected[i].abs().max(1.0),
            "f32 cummax BW @ {i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn cummax_bw_f64_2d_reverse_fw() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15;
    let host_x_f32: Vec<f32> = vec![
        1.0, -2.0, 3.0, 0.5, 4.5,
        -1.0, 2.0, 5.0, 0.0, 3.5,
        4.0, 1.5, -3.0, 2.5, 0.25,
    ];
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| 0.1 + 0.05 * i as f32).collect();
    let expected_f32 = host_cummax_bw_f32::<2>(shape, 1, true, &host_dy_f32, &host_x_f32);
    let host_x: Vec<f64> = host_x_f32.iter().map(|&v| v as f64).collect();
    let host_dy: Vec<f64> = host_dy_f32.iter().map(|&v| v as f64).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cummax,
        input_shape: shape,
        scan_axis: 1,
        reverse: true,
        element: ElementKind::F64,
    };
    let plan = ScanBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ScanBackwardArgs::<f64, 2> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) }),
        y: None,
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * f32::EPSILON as f64;
    for i in 0..numel {
        let want = expected_f32[i] as f64;
        let tol = (want.abs() * eps).max(eps);
        assert!((got[i] - want).abs() <= tol, "f64 cummax BW @ {i}");
    }
}

#[test]
#[ignore]
fn cummax_bw_f16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15;
    let host_x_f32: Vec<f32> = vec![
        1.0, -2.0, 3.0, 0.5, 4.5,
        -1.0, 2.0, 5.0, 0.0, 3.5,
        4.0, 1.5, -3.0, 2.5, 0.25,
    ];
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| 0.125 * (i as f32 + 1.0)).collect();
    let expected_f32 = host_cummax_bw_f32::<2>(shape, 1, false, &host_dy_f32, &host_x_f32);
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cummax,
        input_shape: shape,
        scan_axis: 1,
        reverse: false,
        element: ElementKind::F16,
    };
    let plan = ScanBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ScanBackwardArgs::<f16, 2> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) }),
        y: None,
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    // Per scan-axis extent (5), the BW kernel sums up to 5 dy
    // terms in f32 then rounds once to f16. 4 * f16-eps covers that.
    let eps = 4.0 * 9.77e-4_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "f16 cummax BW @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn cummax_bw_bf16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15;
    let host_x_f32: Vec<f32> = vec![
        1.0, -2.0, 3.0, 0.5, 4.5,
        -1.0, 2.0, 5.0, 0.0, 3.5,
        4.0, 1.5, -3.0, 2.5, 0.25,
    ];
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| 0.125 * (i as f32 + 1.0)).collect();
    let expected_f32 = host_cummax_bw_f32::<2>(shape, 1, false, &host_dy_f32, &host_x_f32);
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cummax,
        input_shape: shape,
        scan_axis: 1,
        reverse: false,
        element: ElementKind::Bf16,
    };
    let plan = ScanBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ScanBackwardArgs::<bf16, 2> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) }),
        y: None,
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * 7.81e-3_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "bf16 cummax BW @ {i}: diff={diff}");
    }
}
