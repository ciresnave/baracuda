//! Real-GPU smoke test for the Phase 3 unary fanout kernel
//! `UnaryPlan<T, N> + UnaryKind::Frac` across f32 / f16 / bf16 / f64.
//!
//! Four contig tests (one per dtype, shape `[8, 128, 128]`) + one
//! f32 strided test. f16 / bf16 lift through f32 via the standard
//! detour pattern; the host reference mirrors that lift so the
//! comparison stays apples-to-apples.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \`
//!   `--test unary_frac_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryArgs,
    UnaryDescriptor, UnaryKind, UnaryPlan, Workspace,
};
use half::{bf16, f16};

fn assert_eq_bits_f32(got: f32, expected: f32, idx: usize) {
    assert!(
        got.to_bits() == expected.to_bits() || (got.is_nan() && expected.is_nan()),
        "frac f32 mismatch @ {idx}: got {got} expected {expected}"
    );
}

fn assert_eq_bits_f64(got: f64, expected: f64, idx: usize) {
    assert!(
        got.to_bits() == expected.to_bits() || (got.is_nan() && expected.is_nan()),
        "frac f64 mismatch @ {idx}: got {got} expected {expected}"
    );
}

fn assert_eq_bits_half(got_bits: u16, expected_bits: u16, idx: usize, label: &str) {
    assert!(
        got_bits == expected_bits,
        "frac {label} mismatch @ {idx}: got bits 0x{got_bits:04x} expected 0x{expected_bits:04x}"
    );
}

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// --- f32 contig --------------------------------------------------------------

fn run_f32_contig<const N: usize>(shape: [i32; N]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.3333 - 10.0)).collect();
    let expected: Vec<f32> = host_x.iter().map(|x| ((*x) - (*x).trunc())).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Frac,
        shape,
        element: ElementKind::F32,
    };
    let plan = UnaryPlan::<f32, N>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f32, N>");
    let args = UnaryArgs::<f32, N> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("frac f32 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq_bits_f32(*g, *e, i);
    }
}

// --- f16 contig --------------------------------------------------------------

fn run_f16_contig_3d(shape: [i32; 3]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32) * 0.3333 - 10.0))
        .collect();
    let host_expected: Vec<f16> = host_x
        .iter()
        .map(|x| f16::from_f32({ let x = (x.to_f32() - x.to_f32().trunc()); x }))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Frac,
        shape,
        element: ElementKind::F16,
    };
    let plan = UnaryPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f16, 3>");
    let args = UnaryArgs::<f16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("frac f16 run");
    stream.synchronize().expect("sync");

    let mut host_got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut host_got).expect("download");
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        assert_eq_bits_half(g.to_bits(), e.to_bits(), i, "f16");
    }
}

// --- bf16 contig -------------------------------------------------------------

fn run_bf16_contig_3d(shape: [i32; 3]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32) * 0.3333 - 10.0))
        .collect();
    let host_expected: Vec<bf16> = host_x
        .iter()
        .map(|x| bf16::from_f32({ let x = (x.to_f32() - x.to_f32().trunc()); x }))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Frac,
        shape,
        element: ElementKind::Bf16,
    };
    let plan = UnaryPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<bf16, 3>");
    let args = UnaryArgs::<bf16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("frac bf16 run");
    stream.synchronize().expect("sync");

    let mut host_got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut host_got).expect("download");
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        assert_eq_bits_half(g.to_bits(), e.to_bits(), i, "bf16");
    }
}

// --- f64 contig --------------------------------------------------------------

fn run_f64_contig_3d(shape: [i32; 3]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f64> = (0..numel).map(|i| ((i as f64) * 0.3333 - 10.0)).collect();
    let expected: Vec<f64> = host_x.iter().map(|x| ((*x) - (*x).trunc())).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Frac,
        shape,
        element: ElementKind::F64,
    };
    let plan = UnaryPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f64, 3>");
    let args = UnaryArgs::<f64, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("frac f64 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq_bits_f64(*g, *e, i);
    }
}

// --- f32 strided (transposed view) ------------------------------------------

#[test]
#[ignore]
fn frac_f32_strided_transposed() {
    let (ctx, stream) = setup();
    const M: usize = 48;
    const N_DIM: usize = 32;
    let m = M as i32;
    let n = N_DIM as i32;

    let x_buf: Vec<f32> = (0..(N_DIM * M)).map(|i| ((i as f32) * 0.3333 - 10.0)).collect();
    let x_shape = [m, n];
    let x_stride = [1i64, M as i64]; // transposed
    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);

    let numel = M * N_DIM;
    let mut expected = vec![0f32; numel];
    for i in 0..M {
        for j in 0..N_DIM {
            let x_lin = j * M + i;
            let y_lin = i * N_DIM + j;
            expected[y_lin] = { let x = (x_buf[x_lin] - x_buf[x_lin].trunc()); x };
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &x_buf).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = UnaryDescriptor {
        kind: UnaryKind::Frac,
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = UnaryPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryArgs::<f32, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: x_stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: y_stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq_bits_f32(*g, *e, i);
    }
}

#[test]
#[ignore]
fn frac_f32_3d() {
    run_f32_contig::<3>([8, 128, 128]);
}

#[test]
#[ignore]
fn frac_f16_3d() {
    run_f16_contig_3d([8, 128, 128]);
}

#[test]
#[ignore]
fn frac_bf16_3d() {
    run_bf16_contig_3d([8, 128, 128]);
}

#[test]
#[ignore]
fn frac_f64_3d() {
    run_f64_contig_3d([8, 128, 128]);
}