//! Real-GPU smoke test for the Phase 3 unary fanout kernel
//! `UnaryPlan<T, N> + UnaryKind::Abs` across f32 / f16 / bf16 / f64.
//!
//! Four contig tests (one per dtype, shape `[8, 128, 128]`) + one
//! f32 strided test (transposed view — the strided dispatcher is
//! dtype-uniform so f32-only proves the wiring). f32 / f64 use
//! bit-exact compare; f16 / bf16 use the same 1-ULP relative tolerance
//! scheme as `binary_add_smoke.rs`.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test unary_abs_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryArgs,
    UnaryDescriptor, UnaryKind, UnaryPlan, Workspace,
};
use half::{bf16, f16};

// 1-ULP relative tolerance constants — same scheme as binary_add_smoke.rs.
const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

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

    // Inputs in [-10, 10] with mixed signs.
    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i % 41) as f32) * 0.5 - 10.0)
        .collect();
    let expected: Vec<f32> = host_x.iter().map(|x| x.abs()).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Abs,
        shape,
        element: ElementKind::F32,
    };
    let plan = UnaryPlan::<f32, N>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f32, N>");
    let args = UnaryArgs::<f32, N> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("abs f32 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "abs f32 contig mismatch @ {i}: got {g} expected {e}"
        );
    }
}

// --- f16 contig --------------------------------------------------------------

fn run_f16_contig_3d(shape: [i32; 3]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_expected: Vec<f16> = host_x.iter().map(|x| f16::from_f32(x.to_f32().abs())).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Abs,
        shape,
        element: ElementKind::F16,
    };
    let plan = UnaryPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f16, 3>");
    let args = UnaryArgs::<f16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("abs f16 run");
    stream.synchronize().expect("sync");

    let mut host_got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut host_got).expect("download");
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let tol = ef.abs().max(1.0) * F16_EPS;
        assert!(
            (gf - ef).abs() <= tol,
            "abs f16 mismatch @ {i}: got {gf} expected {ef}"
        );
    }
}

// --- bf16 contig -------------------------------------------------------------

fn run_bf16_contig_3d(shape: [i32; 3]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let host_expected: Vec<bf16> = host_x.iter().map(|x| bf16::from_f32(x.to_f32().abs())).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Abs,
        shape,
        element: ElementKind::Bf16,
    };
    let plan = UnaryPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<bf16, 3>");
    let args = UnaryArgs::<bf16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("abs bf16 run");
    stream.synchronize().expect("sync");

    let mut host_got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut host_got).expect("download");
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let tol = ef.abs().max(1.0) * BF16_EPS;
        assert!(
            (gf - ef).abs() <= tol,
            "abs bf16 mismatch @ {i}: got {gf} expected {ef}"
        );
    }
}

// --- f64 contig --------------------------------------------------------------

fn run_f64_contig_3d(shape: [i32; 3]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f64> = (0..numel)
        .map(|i| ((i % 41) as f64) * 0.5 - 10.0)
        .collect();
    let expected: Vec<f64> = host_x.iter().map(|x| x.abs()).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Abs,
        shape,
        element: ElementKind::F64,
    };
    let plan = UnaryPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f64, 3>");
    let args = UnaryArgs::<f64, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("abs f64 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "abs f64 contig mismatch @ {i}: got {g} expected {e}"
        );
    }
}

// --- f32 strided (transposed view) ------------------------------------------

#[test]
#[ignore]
fn abs_f32_strided_transposed() {
    let (ctx, stream) = setup();
    const M: usize = 48;
    const N_DIM: usize = 32;
    let m = M as i32;
    let n = N_DIM as i32;

    let x_buf: Vec<f32> = (0..(N_DIM * M))
        .map(|i| ((i % 41) as f32) * 0.5 - 10.0)
        .collect();
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
            expected[y_lin] = x_buf[x_lin].abs();
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &x_buf).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = UnaryDescriptor {
        kind: UnaryKind::Abs,
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
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "abs f32 strided mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn abs_f32_3d() {
    run_f32_contig::<3>([8, 128, 128]);
}

#[test]
#[ignore]
fn abs_f16_3d() {
    run_f16_contig_3d([8, 128, 128]);
}

#[test]
#[ignore]
fn abs_bf16_3d() {
    run_bf16_contig_3d([8, 128, 128]);
}

#[test]
#[ignore]
fn abs_f64_3d() {
    run_f64_contig_3d([8, 128, 128]);
}
