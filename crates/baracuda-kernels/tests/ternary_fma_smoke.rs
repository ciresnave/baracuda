//! Real-GPU smoke test for `TernaryPlan<T, N> + TernaryKind::Fma`
//! across f32 / f16 / bf16 / f64 (contig) and f32 (strided with
//! broadcast on `c`).
//!
//! Fma here is `y = a * b + c` with two separate rounding steps
//! (multiply then add) — NOT the IEEE single-rounding fma intrinsic.
//! This matches PyTorch's `torch.addcmul(c, a, b)` (with implicit
//! `value=1`).
//!
//! Tolerance:
//! - f32 / f64: bit-exact. Plain mul+add is two deterministic
//!   IEEE-754 rounding steps; the GPU `a*b + c` agrees with the host
//!   `a*b + c` at the last bit.
//! - f16 / bf16: relative `4 * dtype_eps`. The f32-detour goes
//!   through `operator*` then `operator+` on `__half` / `__nv_bfloat16`
//!   on the device, with each op promoted to f32 internally; the host
//!   reference does the same via `half::f16` / `half::bf16` (`*` and
//!   `+` overloads also detour through f32). In principle the two
//!   paths agree bit-for-bit, but the relative tolerance absorbs any
//!   rounding mode drift across compilers / versions.
//!
//! Input ranges: `|a|, |b|, |c|` ≤ 5 so `a*b + c` stays well within
//! f16 / bf16 range (max magnitude ~30, easily representable).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test ternary_fma_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TernaryArgs,
    TernaryDescriptor, TernaryKind, TernaryPlan, Workspace,
};
use half::{bf16, f16};

// 1-ULP relative-tolerance constants (same scheme as the unary
// transcendental / activation smoke tests).
const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn assert_close_f16(got: f32, expected: f32, idx: usize) {
    let diff = (got - expected).abs();
    let allow = expected.abs().max(1.0) * 4.0 * F16_EPS;
    assert!(
        diff <= allow,
        "fma f16 mismatch @ {idx}: got {got} expected {expected} (diff {diff} > allow {allow})"
    );
}

fn assert_close_bf16(got: f32, expected: f32, idx: usize) {
    let diff = (got - expected).abs();
    let allow = expected.abs().max(1.0) * 4.0 * BF16_EPS;
    assert!(
        diff <= allow,
        "fma bf16 mismatch @ {idx}: got {got} expected {expected} (diff {diff} > allow {allow})"
    );
}

#[test]
#[ignore]
fn fma_f32_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<f32> = (0..numel).map(|i| ((i % 100) as f32) * 0.05 - 2.5).collect();
    let host_b: Vec<f32> = (0..numel)
        .map(|i| (((i + 17) % 100) as f32) * 0.05 - 2.5)
        .collect();
    let host_c: Vec<f32> = (0..numel)
        .map(|i| (((i + 53) % 100) as f32) * 0.05 - 2.5)
        .collect();
    // Host reference: two separate rounding steps, matching the
    // kernel's plain `a * b + c` (NOT the IEEE fma).
    let expected: Vec<f32> = host_a
        .iter()
        .zip(host_b.iter())
        .zip(host_c.iter())
        .map(|((&a, &b), &c)| a * b + c)
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Fma,
        shape,
        element: ElementKind::F32,
        scale: 1.0,
    };
    let plan = TernaryPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select TernaryPlan<f32, 3>");
    let args = TernaryArgs::<f32, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("fma f32 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "fma f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn fma_f16_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i % 100) as f32) * 0.05 - 2.5))
        .collect();
    let host_b: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((((i + 17) % 100) as f32) * 0.05 - 2.5))
        .collect();
    let host_c: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((((i + 53) % 100) as f32) * 0.05 - 2.5))
        .collect();
    // Host reference: use `half::f16` operator overloads — same
    // f32-detour pipeline as the device's `__half operator* / +`.
    let expected: Vec<f16> = host_a
        .iter()
        .zip(host_b.iter())
        .zip(host_c.iter())
        .map(|((&a, &b), &c)| a * b + c)
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Fma,
        shape,
        element: ElementKind::F16,
        scale: 1.0,
    };
    let plan = TernaryPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select TernaryPlan<f16, 3>");
    let args = TernaryArgs::<f16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("fma f16 run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_close_f16(g.to_f32(), e.to_f32(), i);
    }
}

#[test]
#[ignore]
fn fma_bf16_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i % 100) as f32) * 0.05 - 2.5))
        .collect();
    let host_b: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((((i + 17) % 100) as f32) * 0.05 - 2.5))
        .collect();
    let host_c: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((((i + 53) % 100) as f32) * 0.05 - 2.5))
        .collect();
    let expected: Vec<bf16> = host_a
        .iter()
        .zip(host_b.iter())
        .zip(host_c.iter())
        .map(|((&a, &b), &c)| a * b + c)
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Fma,
        shape,
        element: ElementKind::Bf16,
        scale: 1.0,
    };
    let plan = TernaryPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select TernaryPlan<bf16, 3>");
    let args = TernaryArgs::<bf16, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("fma bf16 run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_close_bf16(g.to_f32(), e.to_f32(), i);
    }
}

#[test]
#[ignore]
fn fma_f64_3d() {
    let (ctx, stream) = setup();
    let shape: [i32; 3] = [8, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_a: Vec<f64> = (0..numel).map(|i| ((i % 100) as f64) * 0.05 - 2.5).collect();
    let host_b: Vec<f64> = (0..numel)
        .map(|i| (((i + 17) % 100) as f64) * 0.05 - 2.5)
        .collect();
    let host_c: Vec<f64> = (0..numel)
        .map(|i| (((i + 53) % 100) as f64) * 0.05 - 2.5)
        .collect();
    let expected: Vec<f64> = host_a
        .iter()
        .zip(host_b.iter())
        .zip(host_c.iter())
        .map(|((&a, &b), &c)| a * b + c)
        .collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = TernaryDescriptor {
        kind: TernaryKind::Fma,
        shape,
        element: ElementKind::F64,
        scale: 1.0,
    };
    let plan = TernaryPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select TernaryPlan<f64, 3>");
    let args = TernaryArgs::<f64, 3> {
        a: TensorRef { data: dev_a.as_slice(), shape, stride },
        b: TensorRef { data: dev_b.as_slice(), shape, stride },
        c: TensorRef { data: dev_c.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("fma f64 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "fma f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

/// Strided test: f32 fma with `c` broadcast as a per-row scalar
/// (`shape=[M, 1]` with `stride=[1, 0]`), exercising the strided
/// dispatch path. `a` and `b` are full contig `[M, N]`; `y` is
/// contig `[M, N]`. The non-contig `c` forces the dispatcher down
/// the `run_strided` arm.
#[test]
#[ignore]
fn fma_f32_strided_broadcast_c() {
    let (ctx, stream) = setup();
    const M: usize = 64;
    const N_DIM: usize = 128;
    let m = M as i32;
    let n = N_DIM as i32;

    let host_a: Vec<f32> = (0..(M * N_DIM))
        .map(|i| ((i % 100) as f32) * 0.05 - 2.5)
        .collect();
    let host_b: Vec<f32> = (0..(M * N_DIM))
        .map(|i| (((i + 17) % 100) as f32) * 0.05 - 2.5)
        .collect();
    // `c` is a per-row scalar — one value per row, broadcast across
    // all columns. Storage is M f32s; stride is [1, 0] so axis 0
    // (rows) moves +1 element and axis 1 (cols) is broadcast.
    let host_c: Vec<f32> = (0..M).map(|i| (i as f32) * 0.05 - 1.5).collect();

    let mut expected: Vec<f32> = vec![0.0; M * N_DIM];
    for row in 0..M {
        for col in 0..N_DIM {
            let a = host_a[row * N_DIM + col];
            let b = host_b[row * N_DIM + col];
            let c = host_c[row];
            expected[row * N_DIM + col] = a * b + c;
        }
    }

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload c");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);
    let a_shape = y_shape;
    let a_stride = y_stride;
    let b_shape = y_shape;
    let b_stride = y_stride;
    let c_shape = [m, 1i32];
    // Row stride = 1 element (M scalars laid out contig along axis 0);
    // col stride = 0 (broadcast).
    let c_stride = [1i64, 0i64];

    let desc = TernaryDescriptor {
        kind: TernaryKind::Fma,
        shape: y_shape,
        element: ElementKind::F32,
        scale: 1.0,
    };
    let plan = TernaryPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select TernaryPlan<f32, 2>");
    let args = TernaryArgs::<f32, 2> {
        a: TensorRef { data: dev_a.as_slice(), shape: a_shape, stride: a_stride },
        b: TensorRef { data: dev_b.as_slice(), shape: b_shape, stride: b_stride },
        c: TensorRef { data: dev_c.as_slice(), shape: c_shape, stride: c_stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: y_stride },
    };
    plan.run(&stream, Workspace::None, args).expect("fma f32 strided run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; M * N_DIM];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "fma f32 strided mismatch @ {i}: got {g} expected {e}"
        );
    }
}
