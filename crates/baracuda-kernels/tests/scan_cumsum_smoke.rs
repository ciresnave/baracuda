//! Real-GPU smoke test for `ScanPlan<T, N> + ScanKind::Cumsum`
//! — inclusive prefix sum along a single axis. Both forward and
//! reverse directions covered.
//!
//! Run with: `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test scan_cumsum_smoke -- --ignored`.

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

// CPU reference: contiguous-row-major inclusive cumsum along `axis`.
fn host_cumsum_f32<const N: usize>(
    input_shape: [i32; N],
    axis: usize,
    reverse: bool,
    x: &[f32],
) -> Vec<f32> {
    let numel: usize = input_shape.iter().map(|&d| d as usize).product();
    // Compute strides (row-major).
    let mut stride = [1usize; N];
    for d in (0..N).rev().skip(1) {
        stride[d] = stride[d + 1] * input_shape[d + 1] as usize;
    }
    let mut y = vec![0f32; numel];
    let extent = input_shape[axis] as i32;
    for_each_coord::<N, _>(input_shape, |coord, linear| {
        let k = coord[axis];
        let mut acc = 0f32;
        if reverse {
            for j in (k..extent).rev() {
                let mut src_coord = coord;
                src_coord[axis] = j;
                let mut idx = 0usize;
                for d in 0..N {
                    idx += src_coord[d] as usize * stride[d];
                }
                acc += x[idx];
            }
        } else {
            for j in 0..=k {
                let mut src_coord = coord;
                src_coord[axis] = j;
                let mut idx = 0usize;
                for d in 0..N {
                    idx += src_coord[d] as usize * stride[d];
                }
                acc += x[idx];
            }
        }
        y[linear as usize] = acc;
    });
    y
}

#[test]
#[ignore]
fn cumsum_f32_1d_forward() {
    let (ctx, stream) = setup();
    let shape = [16i32];
    let host_x: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let expected = host_cumsum_f32(shape, 0, false, &host_x);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 16).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cumsum,
        input_shape: shape,
        scan_axis: 0,
        reverse: false,
        element: ElementKind::F32,
    };
    let plan = ScanPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = ScanArgs::<f32, 1> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 16];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * f32::EPSILON;
    for i in 0..16 {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol,
            "f32 cumsum @ {i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn cumsum_f32_2d_axis_1_reverse() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let host_x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.25 - 4.0).collect();
    let expected = host_cumsum_f32(shape, 1, true, &host_x);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 32).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cumsum,
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
    let eps = 4.0 * f32::EPSILON;
    for i in 0..32 {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol,
            "f32 cumsum reverse @ {i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn cumsum_f64_3d_axis_1_forward() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5, 4];
    let numel = 60;
    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.125 - 2.0).collect();
    let expected = host_cumsum_f32::<3>(shape, 1, false, &host_x.iter().map(|&v| v as f32).collect::<Vec<_>>())
        .into_iter().collect::<Vec<f32>>();
    // Build f64 reference directly to avoid double-precision loss
    let mut expected_f64 = vec![0f64; numel];
    let mut stride = [1usize; 3];
    for d in (0..3).rev().skip(1) { stride[d] = stride[d + 1] * shape[d + 1] as usize; }
    for_each_coord::<3, _>(shape, |coord, linear| {
        let k = coord[1];
        let mut acc = 0f64;
        for j in 0..=k {
            let mut src_coord = coord;
            src_coord[1] = j;
            let mut idx = 0usize;
            for d in 0..3 { idx += src_coord[d] as usize * stride[d]; }
            acc += host_x[idx];
        }
        expected_f64[linear as usize] = acc;
    });
    let _ = expected; // silence unused-warning

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cumsum,
        input_shape: shape,
        scan_axis: 1,
        reverse: false,
        element: ElementKind::F64,
    };
    let plan = ScanPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = ScanArgs::<f64, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride: contiguous_stride(shape) },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("dl");
    let eps = 4.0 * f64::EPSILON;
    for i in 0..numel {
        let tol = (expected_f64[i].abs() * eps).max(eps);
        assert!((got[i] - expected_f64[i]).abs() <= tol, "f64 cumsum @ {i}");
    }
}

#[test]
#[ignore]
fn cumsum_f16_2d_axis_0_forward() {
    let (ctx, stream) = setup();
    let shape = [8i32, 4];
    let numel = 32;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 1.5).collect();
    let expected_f32 = host_cumsum_f32::<2>(shape, 0, false, &host_x_f32);
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cumsum,
        input_shape: shape,
        scan_axis: 0,
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
    // f16 cumulative accumulates rounding error; per-cell extent up to 8.
    let eps = 8.0 * 9.77e-4_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol,
            "f16 cumsum @ {i}: got={} want={} diff={}",
            got[i].to_f32(), expected_f32[i], diff);
    }
}

#[test]
#[ignore]
fn cumsum_bf16_2d_axis_1_reverse() {
    let (ctx, stream) = setup();
    let shape = [4i32, 6];
    let numel = 24;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.2 - 2.0).collect();
    let expected_f32 = host_cumsum_f32::<2>(shape, 1, true, &host_x_f32);
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cumsum,
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
    let eps = 8.0 * 7.81e-3_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "bf16 cumsum reverse @ {i}: diff={diff}");
    }
}
