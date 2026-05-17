//! Real-GPU smoke test for `ScanPlan<T, N> + ScanKind::Cummax`.

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

fn host_cummax_f32<const N: usize>(
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
        let mut acc = f32::NEG_INFINITY;
        if reverse {
            for j in (k..extent).rev() {
                let mut src = coord; src[axis] = j;
                let mut idx = 0usize;
                for d in 0..N { idx += src[d] as usize * stride[d]; }
                if x[idx] > acc { acc = x[idx]; }
            }
        } else {
            for j in 0..=k {
                let mut src = coord; src[axis] = j;
                let mut idx = 0usize;
                for d in 0..N { idx += src[d] as usize * stride[d]; }
                if x[idx] > acc { acc = x[idx]; }
            }
        }
        y[linear as usize] = acc;
    });
    y
}

#[test]
#[ignore]
fn cummax_f32_1d_forward() {
    let (ctx, stream) = setup();
    let shape = [10i32];
    // Non-monotone signal so cummax actually tracks.
    let host_x: Vec<f32> = vec![1.0, -2.0, 3.0, 0.5, 4.5, -1.0, 2.0, 5.0, 0.0, 3.5];
    let expected = host_cummax_f32(shape, 0, false, &host_x);
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 10).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cummax,
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
    let mut got = vec![0f32; 10];
    dev_y.copy_to_host(&mut got).expect("dl");
    for i in 0..10 {
        assert_eq!(got[i], expected[i], "f32 cummax @ {i}");
    }
}

#[test]
#[ignore]
fn cummax_f64_2d_axis_1_reverse() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let host_x: Vec<f64> = vec![
        1.0, -2.0, 3.0, 0.5, 4.5,
        -1.0, 2.0, 5.0, 0.0, 3.5,
        4.0, 1.5, -3.0, 2.5, 0.25,
    ];
    let host_x_f32: Vec<f32> = host_x.iter().map(|&v| v as f32).collect();
    let expected_f32 = host_cummax_f32::<2>(shape, 1, true, &host_x_f32);
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 15).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cummax,
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
    let mut got = vec![0f64; 15];
    dev_y.copy_to_host(&mut got).expect("dl");
    for i in 0..15 {
        assert_eq!(got[i], expected_f32[i] as f64, "f64 cummax reverse @ {i}");
    }
}

#[test]
#[ignore]
fn cummax_f16_2d_axis_0_forward() {
    let (ctx, stream) = setup();
    let shape = [5i32, 3];
    let host_x_f32: Vec<f32> = vec![
        1.0, -2.0, 3.0,
        -1.0, 2.0, 0.5,
        4.0, 1.5, -3.0,
        0.0, 3.5, 2.0,
        -1.5, 0.25, 1.0,
    ];
    let expected_f32 = host_cummax_f32::<2>(shape, 0, false, &host_x_f32);
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 15).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cummax,
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
    let mut got = vec![f16::ZERO; 15];
    dev_y.copy_to_host(&mut got).expect("dl");
    for i in 0..15 {
        // Inputs are exact f16 — bit-exact compare.
        assert_eq!(got[i].to_f32(), expected_f32[i], "f16 cummax @ {i}");
    }
}

#[test]
#[ignore]
fn cummax_bf16_2d_axis_1_forward() {
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let host_x_f32: Vec<f32> = vec![
        1.0, -2.0, 3.0, 0.5, 4.5,
        -1.0, 2.0, 5.0, 0.0, 3.5,
        4.0, 1.5, -3.0, 2.5, 0.25,
    ];
    let expected_f32 = host_cummax_f32::<2>(shape, 1, false, &host_x_f32);
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 15).expect("alloc y");
    let desc = ScanDescriptor {
        kind: ScanKind::Cummax,
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
    let mut got = vec![bf16::ZERO; 15];
    dev_y.copy_to_host(&mut got).expect("dl");
    for i in 0..15 {
        assert_eq!(got[i].to_f32(), expected_f32[i], "bf16 cummax @ {i}");
    }
}
