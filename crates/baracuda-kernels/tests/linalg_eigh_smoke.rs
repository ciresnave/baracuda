//! Real-GPU smoke test for `EighPlan` (cuSOLVER syevd / heevd wrap).
//!
//! Builds a real symmetric matrix with closed-form eigenvalues, runs
//! the plan, and verifies:
//!   1. `info == 0`.
//!   2. The returned eigenvalues match the analytical values.
//!   3. Each returned eigenvector satisfies `A · v_i ≈ λ_i · v_i`.
//!
//! Storage is column-major end-to-end (cuSOLVER native). For a symmetric
//! matrix the column-major view equals the row-major view bit-for-bit,
//! so the smoke test's column-major matmul against the original `A`
//! matches the (column-major) eigenvector matrix returned in `A`'s
//! storage on output.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, EighArgs, EighDescriptor, EighPlan, ElementKind, FillMode, PlanPreference,
    TensorMut, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Column-major view of the 3×3 symmetric test matrix
///
/// ```text
///   A = [ 4 1 0 ]
///       [ 1 3 0 ]
///       [ 0 0 2 ]
/// ```
///
/// Eigenvalues: `λ = 2` and the two roots of `λ² - 7λ + 11 = 0`, i.e.
/// `λ = (7 ± √5) / 2`. cuSOLVER returns these sorted ascending, so the
/// triple is `[(7-√5)/2, 2, (7+√5)/2]`.
fn symmetric_3x3_f32() -> Vec<f32> {
    // Column-major.
    vec![
        4.0, 1.0, 0.0, // column 0
        1.0, 3.0, 0.0, // column 1
        0.0, 0.0, 2.0, // column 2
    ]
}

fn symmetric_3x3_f64() -> Vec<f64> {
    vec![4.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0]
}

fn expected_eigenvalues_f64() -> [f64; 3] {
    // cuSOLVER syevd returns eigenvalues in ascending order. Block-diag
    // input gives roots {2, (7-√5)/2 ≈ 2.382, (7+√5)/2 ≈ 4.618}; sorted
    // ascending that's [2, 2.382, 4.618].
    let sqrt5 = 5.0f64.sqrt();
    [2.0, (7.0 - sqrt5) / 2.0, (7.0 + sqrt5) / 2.0]
}

#[test]
#[ignore]
fn eigh_f32_symmetric_3x3() {
    let (ctx, stream) = setup();
    let n: i32 = 3;
    let a_host = symmetric_3x3_f32();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_w: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n as usize).expect("alloc w");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = EighDescriptor {
        n,
        uplo: FillMode::Upper,
        element: ElementKind::F32,
    };
    let plan = EighPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select EighPlan<f32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    assert!(ws_bytes > 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = EighArgs::<f32, f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [n, n],
            stride: contiguous_stride([n, n]),
        },
        w: TensorMut {
            data: dev_w.as_slice_mut(),
            shape: [n],
            stride: [1],
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run eigh f32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "eigh f32 info != 0");

    let mut w_host = vec![0f32; n as usize];
    let mut v_host = vec![0f32; (n * n) as usize];
    dev_w.copy_to_host(&mut w_host).expect("dl w");
    dev_a.copy_to_host(&mut v_host).expect("dl v");

    let expected = expected_eigenvalues_f64();
    let tol = 1e-5f64;
    for i in 0..3 {
        let got = w_host[i] as f64;
        let want = expected[i];
        assert!(
            (got - want).abs() <= tol * (want.abs().max(1.0)),
            "f32 eigh w[{i}] got={got} want={want}",
        );
    }

    // For each column i (eigenvector v_i), check A·v_i ≈ λ_i·v_i.
    // a_host is column-major [3,3]; v_host is column-major [3,3].
    for i in 0..3 {
        let v0 = v_host[i * 3];
        let v1 = v_host[i * 3 + 1];
        let v2 = v_host[i * 3 + 2];
        // A·v computed column-major: y_i = sum_j A[i,j] * v[j] = sum_j a_host[j*3+i] * v[j].
        let av0 = a_host[0] * v0 + a_host[3] * v1 + a_host[6] * v2;
        let av1 = a_host[1] * v0 + a_host[4] * v1 + a_host[7] * v2;
        let av2 = a_host[2] * v0 + a_host[5] * v1 + a_host[8] * v2;
        let lam = w_host[i];
        let tv0 = lam * v0;
        let tv1 = lam * v1;
        let tv2 = lam * v2;
        let r0 = (av0 - tv0).abs();
        let r1 = (av1 - tv1).abs();
        let r2 = (av2 - tv2).abs();
        let r_max = r0.max(r1).max(r2);
        assert!(
            r_max <= 1e-4,
            "f32 eigh eigenpair {i}: A·v - λ·v = [{r0:.3e}, {r1:.3e}, {r2:.3e}]",
        );
    }
}

#[test]
#[ignore]
fn eigh_f64_symmetric_3x3() {
    let (ctx, stream) = setup();
    let n: i32 = 3;
    let a_host = symmetric_3x3_f64();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_w: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, n as usize).expect("alloc w");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = EighDescriptor {
        n,
        uplo: FillMode::Upper,
        element: ElementKind::F64,
    };
    let plan = EighPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select EighPlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = EighArgs::<f64, f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [n, n],
            stride: contiguous_stride([n, n]),
        },
        w: TensorMut {
            data: dev_w.as_slice_mut(),
            shape: [n],
            stride: [1],
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run eigh f64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "eigh f64 info != 0");

    let mut w_host = vec![0f64; n as usize];
    let mut v_host = vec![0f64; (n * n) as usize];
    dev_w.copy_to_host(&mut w_host).expect("dl w");
    dev_a.copy_to_host(&mut v_host).expect("dl v");

    let expected = expected_eigenvalues_f64();
    let tol = 1e-12f64;
    for i in 0..3 {
        assert!(
            (w_host[i] - expected[i]).abs() <= tol * expected[i].abs().max(1.0),
            "f64 eigh w[{i}] got={} want={}",
            w_host[i],
            expected[i],
        );
    }

    for i in 0..3 {
        let v0 = v_host[i * 3];
        let v1 = v_host[i * 3 + 1];
        let v2 = v_host[i * 3 + 2];
        let av0 = a_host[0] * v0 + a_host[3] * v1 + a_host[6] * v2;
        let av1 = a_host[1] * v0 + a_host[4] * v1 + a_host[7] * v2;
        let av2 = a_host[2] * v0 + a_host[5] * v1 + a_host[8] * v2;
        let lam = w_host[i];
        let r_max = (av0 - lam * v0)
            .abs()
            .max((av1 - lam * v1).abs())
            .max((av2 - lam * v2).abs());
        assert!(
            r_max <= 1e-11,
            "f64 eigh eigenpair {i}: max residual = {r_max:.3e}",
        );
    }
}

// Complex variants (Complex32 / Complex64 Hermitian) are out of scope
// for the trailblazer smoke; the plan implementation is wired but
// exercising it requires upload-side complex helpers that aren't on
// the smoke-test path yet. See module docs in
// `crates/baracuda-kernels/src/linalg/eigh.rs` for the Hermitian
// support contract.
