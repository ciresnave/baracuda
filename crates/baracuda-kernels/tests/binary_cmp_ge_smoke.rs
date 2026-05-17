//! Real-GPU smoke test for `BinaryCmpPlan<T, N> + BinaryCmpKind::Ge`
//! across the 4 FP dtypes (f32 / f16 / bf16 / f64).
//!
//! IEEE semantics: any comparison with NaN returns 0. `a >= b` is true
//! when `a > b` or `a == b`. Mix three regimes (equal / a smaller / a
//! larger) to exercise both true outcomes (equal, strict greater) and
//! the false outcome (a strictly less).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_cmp_ge_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BinaryCmpArgs, BinaryCmpDescriptor, BinaryCmpKind, BinaryCmpPlan,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn cmp_ge_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_a: Vec<f32> = (0..numel).map(|i| ((i % 41) as f32) * 0.5 - 10.0).collect();
    let host_b: Vec<f32> = (0..numel)
        .map(|i| match i % 3 {
            0 => host_a[i],
            1 => host_a[i] - 1.0,
            _ => host_a[i] + 1.0,
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a >= b { 1 } else { 0 })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor { kind: BinaryCmpKind::Ge, shape, element: ElementKind::F32 };
    let plan = BinaryCmpPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = BinaryCmpArgs::<f32, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cmp ge f32 mismatch @ {i}");
    }
    let zeros = got.iter().filter(|&&x| x == 0).count();
    let ones = got.iter().filter(|&&x| x == 1).count();
    assert!(zeros > 0 && ones > 0, "expected mixed, got zeros={zeros} ones={ones}");
}

#[test]
#[ignore]
fn cmp_ge_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_a: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_b: Vec<f16> = (0..numel)
        .map(|i| match i % 3 {
            0 => host_a[i],
            1 => f16::from_f32(host_a[i].to_f32() - 1.0),
            _ => f16::from_f32(host_a[i].to_f32() + 1.0),
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a >= b { 1 } else { 0 })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor { kind: BinaryCmpKind::Ge, shape, element: ElementKind::F16 };
    let plan = BinaryCmpPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = BinaryCmpArgs::<f16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cmp ge f16 mismatch @ {i}");
    }
    let zeros = got.iter().filter(|&&x| x == 0).count();
    let ones = got.iter().filter(|&&x| x == 1).count();
    assert!(zeros > 0 && ones > 0, "expected mixed, got zeros={zeros} ones={ones}");
}

#[test]
#[ignore]
fn cmp_ge_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_a: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_b: Vec<bf16> = (0..numel)
        .map(|i| match i % 3 {
            0 => host_a[i],
            1 => bf16::from_f32(host_a[i].to_f32() - 1.0),
            _ => bf16::from_f32(host_a[i].to_f32() + 1.0),
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a >= b { 1 } else { 0 })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor { kind: BinaryCmpKind::Ge, shape, element: ElementKind::Bf16 };
    let plan = BinaryCmpPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = BinaryCmpArgs::<bf16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cmp ge bf16 mismatch @ {i}");
    }
    let zeros = got.iter().filter(|&&x| x == 0).count();
    let ones = got.iter().filter(|&&x| x == 1).count();
    assert!(zeros > 0 && ones > 0, "expected mixed, got zeros={zeros} ones={ones}");
}

#[test]
#[ignore]
fn cmp_ge_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_a: Vec<f64> = (0..numel).map(|i| ((i % 41) as f64) * 0.5 - 10.0).collect();
    let host_b: Vec<f64> = (0..numel)
        .map(|i| match i % 3 {
            0 => host_a[i],
            1 => host_a[i] - 1.0,
            _ => host_a[i] + 1.0,
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a >= b { 1 } else { 0 })
        .collect();
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor { kind: BinaryCmpKind::Ge, shape, element: ElementKind::F64 };
    let plan = BinaryCmpPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = BinaryCmpArgs::<f64, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cmp ge f64 mismatch @ {i}");
    }
    let zeros = got.iter().filter(|&&x| x == 0).count();
    let ones = got.iter().filter(|&&x| x == 1).count();
    assert!(zeros > 0 && ones > 0, "expected mixed, got zeros={zeros} ones={ones}");
}

#[test]
#[ignore]
fn cmp_ge_f32_strided_transposed() {
    let (ctx, stream) = setup();
    const M: usize = 48;
    const N_DIM: usize = 32;
    let m = M as i32;
    let n = N_DIM as i32;

    let a_buf: Vec<f32> = (0..(N_DIM * M)).map(|i| ((i % 41) as f32) * 0.5 - 10.0).collect();
    let b_buf: Vec<f32> = (0..(M * N_DIM))
        .map(|i| {
            let row = i / N_DIM;
            let col = i % N_DIM;
            let a_val = a_buf[col * M + row];
            match i % 3 {
                0 => a_val,
                1 => a_val - 1.0,
                _ => a_val + 1.0,
            }
        })
        .collect();

    let a_shape = [m, n];
    let a_stride = [1i64, M as i64];
    let b_shape = [m, n];
    let b_stride = contiguous_stride([m, n]);
    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);

    let numel = M * N_DIM;
    let mut expected = vec![0u8; numel];
    for i in 0..M {
        for j in 0..N_DIM {
            let a_val = a_buf[j * M + i];
            let b_val = b_buf[i * N_DIM + j];
            expected[i * N_DIM + j] = if a_val >= b_val { 1 } else { 0 };
        }
    }

    let dev_a = DeviceBuffer::from_slice(&ctx, &a_buf).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &b_buf).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = BinaryCmpDescriptor { kind: BinaryCmpKind::Ge, shape: y_shape, element: ElementKind::F32 };
    let plan = BinaryCmpPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = BinaryCmpArgs::<f32, 2> {
        a: TensorRef { data: dev_a.as_slice(), shape: a_shape, stride: a_stride },
        b: TensorRef { data: dev_b.as_slice(), shape: b_shape, stride: b_stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: y_stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cmp ge f32 strided mismatch @ {i}");
    }
    let zeros = got.iter().filter(|&&x| x == 0).count();
    let ones = got.iter().filter(|&&x| x == 1).count();
    assert!(zeros > 0 && ones > 0, "expected mixed, got zeros={zeros} ones={ones}");
}
