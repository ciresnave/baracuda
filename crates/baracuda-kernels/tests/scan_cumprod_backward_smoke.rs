//! Real-GPU smoke test for `ScanBackwardPlan<T, N> + ScanKind::Cumprod`.
//!
//! Forward: `y[i] = ∏_{j≤i} x[j]`.
//! Backward: `dx[j] = Σ_{i in suffix} dy[i] * y[i] / x[j]` — suffix is
//! `{i ≥ j}` for fw_reverse=false, `{i ≤ j}` for fw_reverse=true.
//!
//! Tests use x in [0.9, 1.1] (no zeros) to avoid the divide-by-zero
//! footgun.

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

fn host_cumprod_fw_f32<const N: usize>(
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
                let mut src = coord; src[axis] = j;
                let mut idx = 0usize;
                for d in 0..N { idx += src[d] as usize * stride[d]; }
                acc *= x[idx];
            }
        } else {
            for j in 0..=k {
                let mut src = coord; src[axis] = j;
                let mut idx = 0usize;
                for d in 0..N { idx += src[d] as usize * stride[d]; }
                acc *= x[idx];
            }
        }
        y[linear as usize] = acc;
    });
    y
}

fn host_cumprod_bw_f32<const N: usize>(
    shape: [i32; N],
    axis: usize,
    fw_reverse: bool,
    dy: &[f32],
    x: &[f32],
    y: &[f32],
) -> Vec<f32> {
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let mut stride = [1usize; N];
    for d in (0..N).rev().skip(1) {
        stride[d] = stride[d + 1] * shape[d + 1] as usize;
    }
    let mut dx = vec![0f32; numel];
    let extent = shape[axis];
    for_each_coord::<N, _>(shape, |coord, linear| {
        let j = coord[axis];
        let mut idx_j = 0usize;
        for d in 0..N { idx_j += coord[d] as usize * stride[d]; }
        let x_j = x[idx_j];
        let inv_x_j = 1.0f32 / x_j;
        let mut acc = 0f32;
        // forward FW (fw_reverse=false): suffix = i in [j, extent);
        // reverse FW: suffix = i in [0, j].
        let (i_lo, i_hi) = if fw_reverse { (0, j) } else { (j, extent - 1) };
        for i in i_lo..=i_hi {
            let mut src = coord; src[axis] = i;
            let mut idx_i = 0usize;
            for d in 0..N { idx_i += src[d] as usize * stride[d]; }
            acc += dy[idx_i] * y[idx_i] * inv_x_j;
        }
        dx[linear as usize] = acc;
    });
    dx
}

#[test]
#[ignore]
fn cumprod_bw_f32_2d_forward_fw() {
    let (ctx, stream) = setup();
    let shape = [3i32, 6];
    let numel = 18;
    let host_x: Vec<f32> = (0..numel).map(|i| 0.9 + ((i as f32) * 0.02)).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| 0.1 + (i as f32) * 0.05).collect();
    let host_y = host_cumprod_fw_f32::<2>(shape, 1, false, &host_x);
    let expected = host_cumprod_bw_f32::<2>(shape, 1, false, &host_dy, &host_x, &host_y);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cumprod,
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
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) }),
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 16.0 * f32::EPSILON;
    for i in 0..numel {
        let tol = (expected[i].abs() * eps).max(eps);
        assert!((got[i] - expected[i]).abs() <= tol,
            "f32 cumprod BW @ {i}: got={} want={}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn cumprod_bw_f64_2d_reverse_fw() {
    let (ctx, stream) = setup();
    let shape = [4i32, 5];
    let numel = 20;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| 0.9 + ((i as f32) * 0.02)).collect();
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| 0.05 + (i as f32) * 0.03).collect();
    let host_y_f32 = host_cumprod_fw_f32::<2>(shape, 0, true, &host_x_f32);
    let expected_f32 = host_cumprod_bw_f32::<2>(shape, 0, true, &host_dy_f32, &host_x_f32, &host_y_f32);
    let host_x: Vec<f64> = host_x_f32.iter().map(|&v| v as f64).collect();
    let host_dy: Vec<f64> = host_dy_f32.iter().map(|&v| v as f64).collect();
    let host_y: Vec<f64> = host_y_f32.iter().map(|&v| v as f64).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cumprod,
        input_shape: shape,
        scan_axis: 0,
        reverse: true,
        element: ElementKind::F64,
    };
    let plan = ScanBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ScanBackwardArgs::<f64, 2> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride: contiguous_stride(shape) },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride: contiguous_stride(shape) },
        x: Some(TensorRef { data: dev_x.as_slice(), shape, stride: contiguous_stride(shape) }),
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) }),
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    // f64 reference still uses f32 host buffer; tolerance set for f32-precision ref.
    let eps = 32.0 * f32::EPSILON as f64;
    for i in 0..numel {
        let tol = ((expected_f32[i] as f64).abs() * eps).max(eps);
        let diff = (got[i] - expected_f32[i] as f64).abs();
        assert!(diff <= tol, "f64 cumprod BW @ {i}: diff={diff}");
    }
}

#[test]
#[ignore]
fn cumprod_bw_f16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| 0.95 + ((i as f32) * 0.01)).collect();
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| 0.1 + (i as f32) * 0.05).collect();
    let host_y_f32 = host_cumprod_fw_f32::<2>(shape, 1, false, &host_x_f32);
    let expected_f32 = host_cumprod_bw_f32::<2>(shape, 1, false, &host_dy_f32, &host_x_f32, &host_y_f32);
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_dy: Vec<f16> = host_dy_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let host_y: Vec<f16> = host_y_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cumprod,
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
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) }),
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    // f16 BW accumulates rounding plus reference is f32 — generous tol.
    let eps = 16.0 * 9.77e-4_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "f16 cumprod BW @ {i}: got={} want={} diff={}",
            got[i].to_f32(), expected_f32[i], diff);
    }
}

#[test]
#[ignore]
fn cumprod_bw_bf16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel = 15;
    let host_x_f32: Vec<f32> = (0..numel).map(|i| 0.95 + ((i as f32) * 0.01)).collect();
    let host_dy_f32: Vec<f32> = (0..numel).map(|i| 0.1 + (i as f32) * 0.05).collect();
    let host_y_f32 = host_cumprod_fw_f32::<2>(shape, 1, false, &host_x_f32);
    let expected_f32 = host_cumprod_bw_f32::<2>(shape, 1, false, &host_dy_f32, &host_x_f32, &host_y_f32);
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_dy: Vec<bf16> = host_dy_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_y: Vec<bf16> = host_y_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("up y");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");

    let desc = ScanBackwardDescriptor {
        kind: ScanKind::Cumprod,
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
        y: Some(TensorRef { data: dev_y.as_slice(), shape, stride: contiguous_stride(shape) }),
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; numel];
    dev_dx.copy_to_host(&mut got).expect("dl");
    let eps = 16.0 * 7.81e-3_f32;
    for i in 0..numel {
        let tol = (expected_f32[i].abs() * eps).max(eps);
        let diff = (got[i].to_f32() - expected_f32[i]).abs();
        assert!(diff <= tol, "bf16 cumprod BW @ {i}: diff={diff}");
    }
}
