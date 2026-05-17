//! Real-GPU smoke test for `ScanBackwardPlan<T, N> + ScanKind::Cumsum`.
//!
//! Forward: `y[i] = Σ_{j ≤ i} x[j]` (inclusive prefix sum).
//! Backward: `dx[j] = Σ_{i ≥ j} dy[i]` = reverse cumsum of `dy`.
//!
//! So Cumsum BW dispatches the SAME forward kernel with `reverse`
//! flipped. No new kernel; pure plan-shape work.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ScanBackwardArgs,
    ScanBackwardDescriptor, ScanBackwardPlan, ScanKind, TensorMut, TensorRef, Workspace,
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

/// CPU reference: expected dx = (reverse if !fw_reverse else forward)
/// cumsum of dy.
fn host_cumsum_bw_f32<const N: usize>(
    shape: [i32; N],
    axis: usize,
    fw_reverse: bool,
    dy: &[f32],
) -> Vec<f32> {
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let mut stride = [1usize; N];
    for d in (0..N).rev().skip(1) { stride[d] = stride[d + 1] * shape[d + 1] as usize; }
    let mut dx = vec![0f32; numel];
    let extent = shape[axis];
    let bw_reverse = !fw_reverse;
    for_each_coord::<N, _>(shape, |coord, linear| {
        let k = coord[axis];
        let mut acc = 0f32;
        if bw_reverse {
            for j in (k..extent).rev() {
                let mut src = coord; src[axis] = j;
                let mut idx = 0usize;
                for d in 0..N { idx += src[d] as usize * stride[d]; }
                acc += dy[idx];
            }
        } else {
            for j in 0..=k {
                let mut src = coord; src[axis] = j;
                let mut idx = 0usize;
                for d in 0..N { idx += src[d] as usize * stride[d]; }
                acc += dy[idx];
            }
        }
        dx[linear as usize] = acc;
    });
    dx
}

#[test]
#[ignore]
fn cumsum_bw_f32_2d_axis_1_forward_fw() {
    // Forward fw_reverse=false → BW reverse=true (reverse cumsum of dy).
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let numel = 32;
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.25 - 4.0).collect();
    let expected = host_cumsum_bw_f32::<2>(shape, 1, false, &host_dy);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cumsum,
        input_shape: shape,
        scan_axis: 1,
        reverse: false,
        element: ElementKind::F32,
    };
    let plan = ScanBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ScanBackwardArgs::<f32, 2> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        x: None,
        y: None,
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol, "f32 cumsum BW @ {i}");
    }
}

#[test]
#[ignore]
fn cumsum_bw_f32_3d_axis_0_reverse_fw() {
    // fw_reverse=true → BW reverse=false (forward cumsum of dy).
    let (ctx, stream) = setup();
    let shape = [4i32, 3, 5];
    let numel = 60;
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 2.5).collect();
    let expected = host_cumsum_bw_f32::<3>(shape, 0, true, &host_dy);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cumsum,
        input_shape: shape,
        scan_axis: 0,
        reverse: true,
        element: ElementKind::F32,
    };
    let plan = ScanBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ScanBackwardArgs::<f32, 3> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        x: None,
        y: None,
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol, "f32 cumsum BW reverse-fw @ {i}");
    }
}

#[test]
#[ignore]
fn cumsum_bw_f16_2d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let numel = 32;
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 1.5).collect();
    let expected_f32 = host_cumsum_bw_f32::<2>(shape, 1, false, &host_dy_f32);
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cumsum,
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
        x: None,
        y: None,
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * 9.77e-4_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "f16 cumsum BW @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn cumsum_bw_bf16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18;
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.15 - 1.0).collect();
    let expected_f32 = host_cumsum_bw_f32::<2>(shape, 1, true, &host_dy_f32);
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cumsum,
        input_shape: shape,
        scan_axis: 1,
        reverse: true,
        element: ElementKind::Bf16,
    };
    let plan = ScanBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ScanBackwardArgs::<bf16, 2> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        x: None,
        y: None,
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 8.0 * 7.81e-3_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "bf16 cumsum BW reverse-fw @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn cumsum_bw_f64_1d() {
    let (ctx, stream) = setup();
    let shape = [16i32];
    let host_dy: Vec<f64> = (0..16).map(|i| (i as f64) * 0.125 - 1.0).collect();
    // BW of forward cumsum (fw_reverse=false) = reverse cumsum of dy.
    let mut expected = vec![0f64; 16];
    for k in 0..16 {
        let mut acc = 0f64;
        for j in (k..16).rev() { acc += host_dy[j]; }
        expected[k] = acc;
    }

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 16).expect("alloc");
    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cumsum,
        input_shape: shape,
        scan_axis: 0,
        reverse: false,
        element: ElementKind::F64,
    };
    let plan = ScanBackwardPlan::<f64, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ScanBackwardArgs::<f64, 1> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        x: None,
        y: None,
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; 16];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * f64::EPSILON;
    for i in 0..16 {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol, "f64 cumsum BW @ {i}");
    }
}
