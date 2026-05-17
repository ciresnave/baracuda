//! Real-GPU smoke test for `addcmul` — `y = a + scale * b * c`.
//!
//! Covers all 4 FP dtypes contig + an f32 strided (per-row scalar `c`
//! broadcast — typical Adam-optimizer pattern: `m = m + lr * grad *
//! something`). f32 / f64 are bit-exact against host (we use unfused
//! `__fmul_rn` / `__fadd_rn` on device + plain Rust mul+add on host,
//! matching PyTorch's plain mul+add convention). f16 / bf16 use the
//! `4 * dtype_eps` relative tolerance band (same as other Phase 3
//! transcendental smoke tests).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test ternary_addcmul_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TernaryArgs,
    TernaryDescriptor, TernaryKind, TernaryPlan, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4_f32; // 2^-10
const BF16_EPS: f32 = 7.81e-3_f32; // 2^-7

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn addcmul_f32_contig() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.125;

    let host_a: Vec<f32> = (0..numel).map(|i| (i as f32 % 100.0) * 0.05 - 2.5).collect();
    let host_b: Vec<f32> = (0..numel).map(|i| (i as f32 % 100.0) * 0.05 + 0.1).collect();
    let host_c: Vec<f32> = (0..numel).map(|i| (i as f32 % 100.0) * 0.05 - 1.0).collect();
    // PyTorch convention: a + scale * b * c (plain mul+add, no FMA)
    let expected: Vec<f32> = (0..numel)
        .map(|i| {
            let t = scale * host_b[i];
            let t = t * host_c[i];
            host_a[i] + t
        })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Addcmul,
        shape,
        element: ElementKind::F32,
        scale,
    };
    let plan = TernaryPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryArgs::<f32, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "addcmul f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn addcmul_f64_contig() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.125;

    let host_a: Vec<f64> = (0..numel)
        .map(|i| (i as f64 % 100.0) * 0.05 - 2.5)
        .collect();
    let host_b: Vec<f64> = (0..numel)
        .map(|i| (i as f64 % 100.0) * 0.05 + 0.1)
        .collect();
    let host_c: Vec<f64> = (0..numel)
        .map(|i| (i as f64 % 100.0) * 0.05 - 1.0)
        .collect();
    let scale_d = scale as f64;
    let expected: Vec<f64> = (0..numel)
        .map(|i| {
            let t = scale_d * host_b[i];
            let t = t * host_c[i];
            host_a[i] + t
        })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Addcmul,
        shape,
        element: ElementKind::F64,
        scale,
    };
    let plan = TernaryPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryArgs::<f64, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "addcmul f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

fn run_half<F>(eps: f32, make_t: F)
where
    F: Fn(f32) -> u16, // -> bit pattern
{
    // Smaller shape to keep f16/bf16 magnitudes bounded.
    let _ = (eps, make_t);
}

#[test]
#[ignore]
fn addcmul_f16_contig() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.5;

    let host_a: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32 % 80.0) * 0.05 - 2.0))
        .collect();
    let host_b: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32 % 80.0) * 0.04 + 0.25))
        .collect();
    let host_c: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32 % 80.0) * 0.03 - 1.0))
        .collect();
    // Match the kernel's f32-detour: convert each to f32, compute in
    // f32 with plain mul+add, round back to f16 once at the store.
    let expected: Vec<f16> = (0..numel)
        .map(|i| {
            let a = host_a[i].to_f32();
            let b = host_b[i].to_f32();
            let c = host_c[i].to_f32();
            let t = scale * b;
            let t = t * c;
            f16::from_f32(a + t)
        })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Addcmul,
        shape,
        element: ElementKind::F16,
        scale,
    };
    let plan = TernaryPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryArgs::<f16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g.to_f32() - e.to_f32()).abs();
        let allow = e.to_f32().abs().max(1.0) * 4.0 * F16_EPS;
        assert!(
            diff <= allow,
            "addcmul f16 mismatch @ {i}: got {g} expected {e} (diff {diff} > allow {allow})"
        );
    }
    let _ = run_half::<fn(f32) -> u16>;
}

#[test]
#[ignore]
fn addcmul_bf16_contig() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let scale: f32 = 0.5;

    let host_a: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32 % 80.0) * 0.05 - 2.0))
        .collect();
    let host_b: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32 % 80.0) * 0.04 + 0.25))
        .collect();
    let host_c: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32 % 80.0) * 0.03 - 1.0))
        .collect();
    let expected: Vec<bf16> = (0..numel)
        .map(|i| {
            let a = host_a[i].to_f32();
            let b = host_b[i].to_f32();
            let c = host_c[i].to_f32();
            let t = scale * b;
            let t = t * c;
            bf16::from_f32(a + t)
        })
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Addcmul,
        shape,
        element: ElementKind::Bf16,
        scale,
    };
    let plan = TernaryPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryArgs::<bf16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g.to_f32() - e.to_f32()).abs();
        let allow = e.to_f32().abs().max(1.0) * 4.0 * BF16_EPS;
        assert!(
            diff <= allow,
            "addcmul bf16 mismatch @ {i}: got {g} expected {e} (diff {diff} > allow {allow})"
        );
    }
}

/// Strided: per-row scalar `c` broadcast (`c.shape = [M, 1]`).
/// Realistic optimizer pattern (per-parameter learning rates).
#[test]
#[ignore]
fn addcmul_f32_broadcast_per_row_c() {
    let (ctx, stream) = setup();
    const M: usize = 32;
    const N_DIM: usize = 64;
    let m = M as i32;
    let n = N_DIM as i32;
    let scale: f32 = 0.25;

    let host_a: Vec<f32> = (0..(M * N_DIM)).map(|i| (i as f32) * 0.01 - 1.5).collect();
    let host_b: Vec<f32> = (0..(M * N_DIM)).map(|i| (i as f32) * 0.02 + 0.5).collect();
    let host_c: Vec<f32> = (0..M).map(|i| (i as f32) * 0.05 + 1.0).collect();

    let mut expected = vec![0f32; M * N_DIM];
    for i in 0..M {
        for j in 0..N_DIM {
            let idx = i * N_DIM + j;
            let t = scale * host_b[idx];
            let t = t * host_c[i]; // broadcast c across cols
            expected[idx] = host_a[idx] + t;
        }
    }

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let ab_shape = [m, n];
    let ab_stride = contiguous_stride([m, n]);
    let c_shape = [m, 1];
    let c_stride = [1i64, 0]; // varies along rows, broadcast across cols
    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);

    let desc = TernaryDescriptor {
        kind: TernaryKind::Addcmul,
        shape: y_shape,
        element: ElementKind::F32,
        scale,
    };
    let plan = TernaryPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = TernaryArgs::<f32, 2> {
        a: TensorRef { data: dev_a.as_slice(), shape: ab_shape, stride: ab_stride },
        b: TensorRef { data: dev_b.as_slice(), shape: ab_shape, stride: ab_stride },
        c: TensorRef { data: dev_c.as_slice(), shape: c_shape, stride: c_stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: y_stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; M * N_DIM];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "addcmul broadcast-c mismatch @ {i}: got {g} expected {e}"
        );
    }
}
