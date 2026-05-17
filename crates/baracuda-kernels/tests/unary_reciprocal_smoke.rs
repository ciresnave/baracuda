//! Real-GPU smoke test for the Phase 3 unary fanout kernel
//! `UnaryPlan<T, N> + UnaryKind::Reciprocal` across f32 / f16 / bf16 /
//! f64.
//!
//! Inputs are kept bounded above 0.5 (`(i as f32) * 0.125 + 1.5`) to
//! stay away from zero — reciprocal at zero is +Inf and adds no
//! coverage value beyond the standard non-pathological cases.
//!
//! f32 / f64 bit-exact: `1.0 / x` is correctly-rounded by both the
//! Rust `f32 / f64` operators and CUDA's `nvcc` (uses the IEEE-754
//! `div.rn.f32` / `div.rn.f64` instructions). f16 / bf16 use the
//! relative-tolerance scheme since the kernel takes an f32 detour.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryArgs,
    UnaryDescriptor, UnaryKind, UnaryPlan, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// Inputs in [1.5, ~10] — strictly positive, bounded away from zero.
fn recip_input_f32(i: usize) -> f32 {
    (i as f32) * 0.125 + 1.5
}

// --- f32 contig --------------------------------------------------------------

fn run_f32_contig<const N: usize>(shape: [i32; N]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f32> = (0..numel).map(recip_input_f32).collect();
    let expected: Vec<f32> = host_x.iter().map(|x| 1.0_f32 / x).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Reciprocal,
        shape,
        element: ElementKind::F32,
    };
    let plan = UnaryPlan::<f32, N>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f32, N>");
    let args = UnaryArgs::<f32, N> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("reciprocal f32 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "reciprocal f32 contig mismatch @ {i}: got {g} expected {e}"
        );
    }
}

// --- f16 contig --------------------------------------------------------------

fn run_f16_contig_3d(shape: [i32; 3]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(recip_input_f32(i % 41)))
        .collect();
    // CPU reference matches the device's f32-detour path: round input
    // up to f32, compute 1/x in f32, round back to f16.
    let host_expected: Vec<f16> = host_x
        .iter()
        .map(|x| f16::from_f32(1.0_f32 / x.to_f32()))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Reciprocal,
        shape,
        element: ElementKind::F16,
    };
    let plan = UnaryPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f16, 3>");
    let args = UnaryArgs::<f16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("reciprocal f16 run");
    stream.synchronize().expect("sync");

    let mut host_got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut host_got).expect("download");
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let tol = ef.abs().max(1.0) * F16_EPS;
        assert!(
            (gf - ef).abs() <= tol,
            "reciprocal f16 mismatch @ {i}: got {gf} expected {ef}"
        );
    }
}

// --- bf16 contig -------------------------------------------------------------

fn run_bf16_contig_3d(shape: [i32; 3]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(recip_input_f32(i % 41)))
        .collect();
    let host_expected: Vec<bf16> = host_x
        .iter()
        .map(|x| bf16::from_f32(1.0_f32 / x.to_f32()))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Reciprocal,
        shape,
        element: ElementKind::Bf16,
    };
    let plan = UnaryPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<bf16, 3>");
    let args = UnaryArgs::<bf16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("reciprocal bf16 run");
    stream.synchronize().expect("sync");

    let mut host_got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut host_got).expect("download");
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let tol = ef.abs().max(1.0) * BF16_EPS;
        assert!(
            (gf - ef).abs() <= tol,
            "reciprocal bf16 mismatch @ {i}: got {gf} expected {ef}"
        );
    }
}

// --- f64 contig --------------------------------------------------------------

fn run_f64_contig_3d(shape: [i32; 3]) {
    let (ctx, stream) = setup();
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f64> = (0..numel)
        .map(|i| (i as f64) * 0.125 + 1.5)
        .collect();
    let expected: Vec<f64> = host_x.iter().map(|x| 1.0_f64 / x).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Reciprocal,
        shape,
        element: ElementKind::F64,
    };
    let plan = UnaryPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f64, 3>");
    let args = UnaryArgs::<f64, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("reciprocal f64 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "reciprocal f64 contig mismatch @ {i}: got {g} expected {e}"
        );
    }
}

// --- f32 strided (transposed view) ------------------------------------------

#[test]
#[ignore]
fn reciprocal_f32_strided_transposed() {
    let (ctx, stream) = setup();
    const M: usize = 48;
    const N_DIM: usize = 32;
    let m = M as i32;
    let n = N_DIM as i32;

    let x_buf: Vec<f32> = (0..(N_DIM * M)).map(recip_input_f32).collect();
    let x_shape = [m, n];
    let x_stride = [1i64, M as i64];
    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);

    let numel = M * N_DIM;
    let mut expected = vec![0f32; numel];
    for i in 0..M {
        for j in 0..N_DIM {
            let x_lin = j * M + i;
            let y_lin = i * N_DIM + j;
            expected[y_lin] = 1.0_f32 / x_buf[x_lin];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &x_buf).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = UnaryDescriptor {
        kind: UnaryKind::Reciprocal,
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
            "reciprocal f32 strided mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn reciprocal_f32_3d() {
    run_f32_contig::<3>([8, 128, 128]);
}

#[test]
#[ignore]
fn reciprocal_f16_3d() {
    run_f16_contig_3d([8, 128, 128]);
}

#[test]
#[ignore]
fn reciprocal_bf16_3d() {
    run_bf16_contig_3d([8, 128, 128]);
}

#[test]
#[ignore]
fn reciprocal_f64_3d() {
    run_f64_contig_3d([8, 128, 128]);
}
