//! Real-GPU smoke test for `InversePlan` (cuSOLVER `getrf` + `getrs`
//! over identity).
//!
//! Verifies `A · A^{-1} ≈ I` for a small well-conditioned `A`. Storage
//! convention is column-major end-to-end (cuSOLVER native).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, InverseArgs, InverseDescriptor, InversePlan, PlanPreference,
    TensorMut, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn matmul_cm_f32(a: &[f32], b: &[f32], m: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * m];
    for j in 0..m {
        for i in 0..m {
            let mut acc = 0f32;
            for k in 0..m {
                acc += a[k * m + i] * b[j * m + k];
            }
            c[j * m + i] = acc;
        }
    }
    c
}

fn matmul_cm_f64(a: &[f64], b: &[f64], m: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * m];
    for j in 0..m {
        for i in 0..m {
            let mut acc = 0f64;
            for k in 0..m {
                acc += a[k * m + i] * b[j * m + k];
            }
            c[j * m + i] = acc;
        }
    }
    c
}

#[test]
#[ignore]
fn inverse_f32_basic() {
    let (ctx, stream) = setup();
    let m: i32 = 2;

    // A = [[2, 1], [1, 3]] column-major. Determinant 5; A^{-1} =
    // (1/5) · [[3, -1], [-1, 2]] in row-major (i.e. column-major col 0
    // = [3, -1]/5, col 1 = [-1, 2]/5).
    let a_host: Vec<f32> = vec![2.0, 1.0, 1.0, 3.0];

    // Keep a host copy of A for the A · A^{-1} ≈ I verification (the
    // GPU buffer is overwritten in place with the LU factors).
    let a_original = a_host.clone();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_inv: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * m) as usize).expect("alloc inv");
    let mut dev_pivot: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, m as usize).expect("alloc pivot");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = InverseDescriptor {
        m,
        element: ElementKind::F32,
    };
    let plan = InversePlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select InversePlan<f32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    assert!(ws_bytes > 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let mm_shape = [m, m];
    let pivot_shape = [m];
    let args = InverseArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: mm_shape,
            stride: contiguous_stride(mm_shape),
        },
        inv: TensorMut {
            data: dev_inv.as_slice_mut(),
            shape: mm_shape,
            stride: contiguous_stride(mm_shape),
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
        .expect("run Inverse f32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mut inv_host = vec![0f32; (m * m) as usize];
    dev_inv.copy_to_host(&mut inv_host).expect("dl inv");

    // Sanity: A · A^{-1} ≈ I.
    let prod = matmul_cm_f32(&a_original, &inv_host, m as usize);
    let tol = 1e-5f32;
    for i in 0..(m as usize) {
        for j in 0..(m as usize) {
            let got = prod[j * (m as usize) + i];
            let want = if i == j { 1.0 } else { 0.0 };
            let diff = (got - want).abs();
            assert!(
                diff <= tol,
                "f32 A·A^-1 ({i},{j}): got={got}, want={want}, diff={diff}",
            );
        }
    }
}

#[test]
#[ignore]
fn inverse_f64_basic() {
    let (ctx, stream) = setup();
    let m: i32 = 3;

    // Symmetric tridiagonal — well conditioned.
    // Row-major: [[4, 1, 0], [1, 3, 1], [0, 1, 2]]; column-major
    // flatten = col 0 [4,1,0], col 1 [1,3,1], col 2 [0,1,2].
    let a_host: Vec<f64> = vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let a_original = a_host.clone();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_inv: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (m * m) as usize).expect("alloc inv");
    let mut dev_pivot: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, m as usize).expect("alloc pivot");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = InverseDescriptor {
        m,
        element: ElementKind::F64,
    };
    let plan = InversePlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select InversePlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let mm_shape = [m, m];
    let pivot_shape = [m];
    let args = InverseArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: mm_shape,
            stride: contiguous_stride(mm_shape),
        },
        inv: TensorMut {
            data: dev_inv.as_slice_mut(),
            shape: mm_shape,
            stride: contiguous_stride(mm_shape),
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
        .expect("run Inverse f64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mut inv_host = vec![0f64; (m * m) as usize];
    dev_inv.copy_to_host(&mut inv_host).expect("dl inv");

    let prod = matmul_cm_f64(&a_original, &inv_host, m as usize);
    let tol = 1e-10f64;
    for i in 0..(m as usize) {
        for j in 0..(m as usize) {
            let got = prod[j * (m as usize) + i];
            let want = if i == j { 1.0 } else { 0.0 };
            let diff = (got - want).abs();
            assert!(
                diff <= tol,
                "f64 A·A^-1 ({i},{j}): got={got}, want={want}, diff={diff}",
            );
        }
    }
}
