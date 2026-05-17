//! Real-GPU smoke test for `SolvePlan` (cuSOLVER `getrf` + `getrs` wrap).
//!
//! Verifies `A · X ≈ B` for a small well-conditioned `A`. Storage
//! convention is column-major end-to-end (cuSOLVER native).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SolveArgs, SolveDescriptor, SolvePlan,
    TensorMut, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Multiply a column-major `m × m` matrix `a` against a column-major
/// `m × nrhs` matrix `x`, producing column-major `m × nrhs` `b = a · x`.
fn matmul_cm_f32(a: &[f32], x: &[f32], m: usize, nrhs: usize) -> Vec<f32> {
    let mut b = vec![0f32; m * nrhs];
    for j in 0..nrhs {
        for i in 0..m {
            let mut acc = 0f32;
            for k in 0..m {
                acc += a[k * m + i] * x[j * m + k];
            }
            b[j * m + i] = acc;
        }
    }
    b
}

fn matmul_cm_f64(a: &[f64], x: &[f64], m: usize, nrhs: usize) -> Vec<f64> {
    let mut b = vec![0f64; m * nrhs];
    for j in 0..nrhs {
        for i in 0..m {
            let mut acc = 0f64;
            for k in 0..m {
                acc += a[k * m + i] * x[j * m + k];
            }
            b[j * m + i] = acc;
        }
    }
    b
}

#[test]
#[ignore]
fn solve_f32_basic() {
    let (ctx, stream) = setup();
    let m: i32 = 2;
    let nrhs: i32 = 1;

    // A = [[2, 1], [1, 3]] (column-major: col 0 = [2, 1], col 1 = [1, 3]).
    // Determinant 5, well conditioned.
    let a_host: Vec<f32> = vec![2.0, 1.0, 1.0, 3.0];
    // Expected solution X = [1, 2]^T.
    let x_truth: Vec<f32> = vec![1.0, 2.0];
    // B = A · X = [4, 7]^T.
    let b_host = matmul_cm_f32(&a_host, &x_truth, m as usize, nrhs as usize);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload b");
    let mut dev_pivot: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, m as usize).expect("alloc pivot");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = SolveDescriptor {
        m,
        nrhs,
        element: ElementKind::F32,
    };
    let plan = SolvePlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select SolvePlan<f32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    assert!(ws_bytes > 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let a_shape = [m, m];
    let b_shape = [m, nrhs];
    let pivot_shape = [m];
    let args = SolveArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        b: TensorMut {
            data: dev_b.as_slice_mut(),
            shape: b_shape,
            stride: contiguous_stride(b_shape),
        },
        pivot: TensorMut {
            data: dev_pivot.as_slice_mut(),
            shape: pivot_shape,
            stride: contiguous_stride(pivot_shape),
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run Solve f32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mut x_host = vec![0f32; (m * nrhs) as usize];
    dev_b.copy_to_host(&mut x_host).expect("dl X");

    let tol = 1e-5f32;
    for i in 0..(m as usize * nrhs as usize) {
        let got = x_host[i];
        let want = x_truth[i];
        let diff = (got - want).abs();
        assert!(
            diff <= tol * want.abs().max(1.0),
            "f32 solve idx {i}: got={got}, want={want}, diff={diff}",
        );
    }
}

#[test]
#[ignore]
fn solve_f64_multi_rhs() {
    let (ctx, stream) = setup();
    let m: i32 = 3;
    let nrhs: i32 = 2;

    // A = diag-dominant 3×3 (column-major).
    // Row-major: [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
    // Column-major flatten: col 0 [4,1,0], col 1 [1,3,1], col 2 [0,1,2].
    let a_host: Vec<f64> = vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    // Two right-hand sides: X col0 = [1, 2, 3], X col1 = [-1, 0, 1].
    let x_truth: Vec<f64> = vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0];
    let b_host = matmul_cm_f64(&a_host, &x_truth, m as usize, nrhs as usize);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload b");
    let mut dev_pivot: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, m as usize).expect("alloc pivot");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = SolveDescriptor {
        m,
        nrhs,
        element: ElementKind::F64,
    };
    let plan = SolvePlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select SolvePlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let a_shape = [m, m];
    let b_shape = [m, nrhs];
    let pivot_shape = [m];
    let args = SolveArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        b: TensorMut {
            data: dev_b.as_slice_mut(),
            shape: b_shape,
            stride: contiguous_stride(b_shape),
        },
        pivot: TensorMut {
            data: dev_pivot.as_slice_mut(),
            shape: pivot_shape,
            stride: contiguous_stride(pivot_shape),
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run Solve f64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mut x_host = vec![0f64; (m * nrhs) as usize];
    dev_b.copy_to_host(&mut x_host).expect("dl X");

    let tol = 1e-10f64;
    for i in 0..(m as usize * nrhs as usize) {
        let got = x_host[i];
        let want = x_truth[i];
        let diff = (got - want).abs();
        assert!(
            diff <= tol * want.abs().max(1.0),
            "f64 solve idx {i}: got={got}, want={want}, diff={diff}",
        );
    }
}
