//! Real-GPU smoke test for `ScanPlan<T, N> + ScanKind::Cumprod` —
//! inclusive prefix product along a single axis.
//!
//! Run with: `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test scan_cumprod_smoke -- --ignored`.

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

fn host_cumprod_f32<const N: usize>(
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
        let mut acc = 1f32;
        if reverse {
            for j in (k..extent).rev() {
                let mut src = coord;
                src[axis] = j;
                let mut idx = 0usize;
                for d in 0..N {
                    idx += src[d] as usize * stride[d];
                }
                acc *= x[idx];
            }
        } else {
            for j in 0..=k {
                let mut src = coord;
                src[axis] = j;
                let mut idx = 0usize;
                for d in 0..N {
                    idx += src[d] as usize * stride[d];
                }
                acc *= x[idx];
            }
        }
        y[linear as usize] = acc;
    });
    y
}

fn host_cumprod_f64<const N: usize>(
    shape: [i32; N],
    axis: usize,
    reverse: bool,
    x: &[f64],
) -> Vec<f64> {
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let mut stride = [1usize; N];
    for d in (0..N).rev().skip(1) {
        stride[d] = stride[d + 1] * shape[d + 1] as usize;
    }
    let mut y = vec![0f64; numel];
    let extent = shape[axis];
    for_each_coord::<N, _>(shape, |coord, linear| {
        let k = coord[axis];
        let mut acc = 1f64;
        if reverse {
            for j in (k..extent).rev() {
                let mut src = coord;
                src[axis] = j;
                let mut idx = 0usize;
                for d in 0..N {
                    idx += src[d] as usize * stride[d];
                }
                acc *= x[idx];
            }
        } else {
            for j in 0..=k {
                let mut src = coord;
                src[axis] = j;
                let mut idx = 0usize;
                for d in 0..N {
                    idx += src[d] as usize * stride[d];
                }
                acc *= x[idx];
            }
        }
        y[linear as usize] = acc;
    });
    y
}

#[test]
#[ignore]
fn cumprod_f32_1d_forward() {
    let (ctx, stream) = setup();
    let shape = [16i32];
    // Values in [0.7, 1.3] — product stays in bounds.
    let host_x: Vec<f32> = (0..16).map(|i| 0.7 + ((i as f32) * 0.04)).collect();
    let expected = host_cumprod_f32(shape, 0, false, &host_x);
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 16).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cumprod,
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
    let eps = 8.0 * f32::EPSILON;
    for i in 0..16 {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol,
            "f32 cumprod @ {i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn cumprod_f64_2d_axis_1_reverse() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18;
    let host_x: Vec<f64> = (0..numel).map(|i| 0.8 + ((i as f64) * 0.05)).collect();
    let expected = host_cumprod_f64(shape, 1, true, &host_x);
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cumprod,
        input_shape: shape,
        scan_axis: 1,
        reverse: true,
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
        assert!((got[i] - expected[i]).abs() <= tol, "f64 cumprod reverse @ {i}");
    }
}

#[test]
#[ignore]
fn cumprod_f16_2d_axis_0_forward() {
    let (ctx, stream) = setup();
    let shape = [6i32, 4];
    let numel = 24;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| 0.9 + ((i as f32) * 0.02)).collect();
    let expected_f32 = host_cumprod_f32::<2>(shape, 0, false, &host_x_f32);
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cumprod,
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
    let eps = 8.0 * 9.77e-4_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "f16 cumprod @ {i}: got={} want={} diff={}",
            got[i].to_f32(), expected_f32[i], diff);
    }
}

#[test]
#[ignore]
fn cumprod_bf16_2d_axis_1_forward() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let numel = 20;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| 0.9 + ((i as f32) * 0.02)).collect();
    let expected_f32 = host_cumprod_f32::<2>(shape, 1, false, &host_x_f32);
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cumprod,
        input_shape: shape,
        scan_axis: 1,
        reverse: false,
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
        assert!(diff <= tol, "bf16 cumprod @ {i}: diff={diff}");
    }
}
