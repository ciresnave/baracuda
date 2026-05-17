//! Dtype-fill smoke test for `BinaryCmpPlan<T, N> + BinaryCmpKind::Eq`
//! on the 3 dtypes (f16 / bf16 / f64) that landed in the comparison
//! fanout. The f32 trailblazer cell stays in `binary_cmp_eq_smoke.rs`.
//!
//! Inputs follow the same strategy as the trailblazer file: every 3rd
//! cell is deliberately equal, others differ. Comparisons produce
//! exact u8 results (0 / 1) so bit-exact compare against the host
//! reference applies for all dtypes.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_cmp_eq_dtype_smoke -- --ignored`.

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
fn cmp_eq_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_b: Vec<f16> = (0..numel)
        .map(|i| {
            if i % 3 == 0 {
                host_a[i]
            } else {
                f16::from_f32(((i % 37) as f32) * 0.25 - 4.5)
            }
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a == b { 1 } else { 0 })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor {
        kind: BinaryCmpKind::Eq,
        shape,
        element: ElementKind::F16,
    };
    let plan = BinaryCmpPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
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
        assert_eq!(g, e, "cmp eq f16 mismatch @ {i}");
    }
    let trues = got.iter().filter(|&&x| x == 1).count();
    assert!(trues > 0, "expected at least some true cells");
}

#[test]
#[ignore]
fn cmp_eq_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_b: Vec<bf16> = (0..numel)
        .map(|i| {
            if i % 3 == 0 {
                host_a[i]
            } else {
                bf16::from_f32(((i % 37) as f32) * 0.25 - 4.5)
            }
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a == b { 1 } else { 0 })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor {
        kind: BinaryCmpKind::Eq,
        shape,
        element: ElementKind::Bf16,
    };
    let plan = BinaryCmpPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
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
        assert_eq!(g, e, "cmp eq bf16 mismatch @ {i}");
    }
    let trues = got.iter().filter(|&&x| x == 1).count();
    assert!(trues > 0, "expected at least some true cells");
}

#[test]
#[ignore]
fn cmp_eq_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<f64> = (0..numel)
        .map(|i| ((i % 41) as f64) * 0.5 - 10.0)
        .collect();
    let host_b: Vec<f64> = (0..numel)
        .map(|i| {
            if i % 3 == 0 {
                host_a[i]
            } else {
                ((i % 37) as f64) * 0.25 - 4.5
            }
        })
        .collect();
    let expected: Vec<u8> = host_a
        .iter()
        .zip(host_b.iter())
        .map(|(a, b)| if a == b { 1 } else { 0 })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = BinaryCmpDescriptor {
        kind: BinaryCmpKind::Eq,
        shape,
        element: ElementKind::F64,
    };
    let plan = BinaryCmpPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
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
        assert_eq!(g, e, "cmp eq f64 mismatch @ {i}");
    }
    let trues = got.iter().filter(|&&x| x == 1).count();
    assert!(trues > 0, "expected at least some true cells");
}
