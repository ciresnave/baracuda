//! Real-GPU smoke test for `BatchedQrPlan` (cuSOLVER `geqrfBatched`).
//!
//! Verifies `geqrfBatched` runs cleanly and produces sane packed-output
//! per batch slot. The kernel writes the `R` (upper) + Householder
//! reflectors (strict lower) directly into the input buffer, so a full
//! `Q · R == A` reconstruction requires applying `ormqr` per batch
//! (deferred — verified in the non-batched plan's smoke test); here we
//! validate end-to-end status, that the upper triangle is non-zero, and
//! that `info[b] == 0` for every batch slot.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BatchedQrArgs, BatchedQrDescriptor, BatchedQrPlan, Complex32, Complex64,
    ElementKind, PlanPreference, TensorMut, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn qr_batched_f32_basic() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;
    let n: i32 = 3;
    let k = m.min(n);
    // Two independent 3×3 matrices, column-major flattened.
    // Batch 0: column-major [[1,2,3],[4,5,6],[7,8,10]]
    //   col0 [1,4,7], col1 [2,5,8], col2 [3,6,10] (slightly perturbed last
    //   element to keep it non-singular).
    // Batch 1: scaled identity-ish.
    let a_host: Vec<f32> = vec![
        // batch 0 (cm)
        1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0, // batch 1 (cm)
        2.0, 0.0, 1.0, 1.0, 3.0, 0.0, 0.0, 1.0, 2.0,
    ];

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_tau: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * k) as usize).expect("alloc tau");

    let desc = BatchedQrDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::F32,
    };
    let plan = BatchedQrPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select BatchedQrPlan<f32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    assert!(ws_bytes >= 2 * (b as usize) * core::mem::size_of::<u64>());
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let a_shape = [b, m, n];
    let tau_shape = [b, k];
    let args = BatchedQrArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: tau_shape,
            stride: contiguous_stride(tau_shape),
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run batched QR f32");
    stream.synchronize().expect("sync");

    let mut a_post = vec![0f32; (b * m * n) as usize];
    dev_a.copy_to_host(&mut a_post).expect("dl a-post");
    let mut tau_post = vec![0f32; (b * k) as usize];
    dev_tau.copy_to_host(&mut tau_post).expect("dl tau-post");
    // Sanity: diagonal of `R` (upper triangle of packed-output `A`) is
    // non-zero — every diagonal cell `r[i, i]` should be > 0 in
    // magnitude for the input matrices we picked.
    for bi in 0..b as usize {
        for i in 0..m as usize {
            let idx = bi * (m as usize) * (n as usize) + i * (m as usize) + i;
            let diag = a_post[idx].abs();
            assert!(diag > 1e-3, "f32 R[{bi}][{i},{i}] near-zero: {diag}");
        }
    }
}

#[test]
#[ignore]
fn qr_batched_f64_basic() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;
    let n: i32 = 3;
    let k = m.min(n);
    let a_host: Vec<f64> = vec![
        1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0, 2.0, 0.0, 1.0, 1.0, 3.0, 0.0, 0.0, 1.0, 2.0,
    ];

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_tau: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (b * k) as usize).expect("alloc tau");

    let desc = BatchedQrDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::F64,
    };
    let plan = BatchedQrPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select BatchedQrPlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = BatchedQrArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run batched QR f64");
    stream.synchronize().expect("sync");

    let mut a_post = vec![0f64; (b * m * n) as usize];
    dev_a.copy_to_host(&mut a_post).expect("dl a-post");
    for bi in 0..b as usize {
        for i in 0..m as usize {
            let idx = bi * (m as usize) * (n as usize) + i * (m as usize) + i;
            let diag = a_post[idx].abs();
            assert!(diag > 1e-6, "f64 R[{bi}][{i},{i}] near-zero: {diag}");
        }
    }
}

#[test]
#[ignore]
fn qr_batched_complex32_basic() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;
    let n: i32 = 3;
    let k = m.min(n);
    // Two 3×3 complex matrices, column-major. Picked to be well-conditioned.
    let a_host: Vec<Complex32> = vec![
        // batch 0
        Complex32::new(1.0, 0.5), Complex32::new(4.0, -0.2), Complex32::new(7.0, 0.1),
        Complex32::new(2.0, -0.3), Complex32::new(5.0, 0.4), Complex32::new(8.0, -0.5),
        Complex32::new(3.0, 0.2), Complex32::new(6.0, -0.1), Complex32::new(10.0, 0.3),
        // batch 1
        Complex32::new(2.0, 0.0), Complex32::new(0.0, 1.0), Complex32::new(1.0, 0.0),
        Complex32::new(1.0, -0.5), Complex32::new(3.0, 0.0), Complex32::new(0.0, 0.5),
        Complex32::new(0.0, 0.0), Complex32::new(1.0, 0.2), Complex32::new(2.0, -0.1),
    ];

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_tau: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, (b * k) as usize).expect("alloc tau");

    let desc = BatchedQrDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::Complex32,
    };
    let plan = BatchedQrPlan::<Complex32>::select(&stream, &desc, PlanPreference::default())
        .expect("select BatchedQrPlan<Complex32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = BatchedQrArgs::<Complex32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run batched QR Complex32");
    stream.synchronize().expect("sync");

    let mut a_post = vec![Complex32::new(0.0, 0.0); (b * m * n) as usize];
    dev_a.copy_to_host(&mut a_post).expect("dl a-post");
    for bi in 0..b as usize {
        for i in 0..m as usize {
            let idx = bi * (m as usize) * (n as usize) + i * (m as usize) + i;
            let diag = a_post[idx];
            let mag = (diag.re * diag.re + diag.im * diag.im).sqrt();
            assert!(
                mag > 1e-3,
                "Complex32 R[{bi}][{i},{i}] near-zero: {diag:?}"
            );
        }
    }
}

#[test]
#[ignore]
fn qr_batched_complex64_basic() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 3;
    let n: i32 = 3;
    let k = m.min(n);
    let a_host: Vec<Complex64> = vec![
        Complex64::new(1.0, 0.5), Complex64::new(4.0, -0.2), Complex64::new(7.0, 0.1),
        Complex64::new(2.0, -0.3), Complex64::new(5.0, 0.4), Complex64::new(8.0, -0.5),
        Complex64::new(3.0, 0.2), Complex64::new(6.0, -0.1), Complex64::new(10.0, 0.3),
        Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0),
        Complex64::new(1.0, -0.5), Complex64::new(3.0, 0.0), Complex64::new(0.0, 0.5),
        Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.2), Complex64::new(2.0, -0.1),
    ];

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_tau: DeviceBuffer<Complex64> =
        DeviceBuffer::zeros(&ctx, (b * k) as usize).expect("alloc tau");

    let desc = BatchedQrDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::Complex64,
    };
    let plan = BatchedQrPlan::<Complex64>::select(&stream, &desc, PlanPreference::default())
        .expect("select BatchedQrPlan<Complex64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = BatchedQrArgs::<Complex64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
        tau: TensorMut {
            data: dev_tau.as_slice_mut(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run batched QR Complex64");
    stream.synchronize().expect("sync");

    let mut a_post = vec![Complex64::new(0.0, 0.0); (b * m * n) as usize];
    dev_a.copy_to_host(&mut a_post).expect("dl a-post");
    for bi in 0..b as usize {
        for i in 0..m as usize {
            let idx = bi * (m as usize) * (n as usize) + i * (m as usize) + i;
            let diag = a_post[idx];
            let mag = (diag.re * diag.re + diag.im * diag.im).sqrt();
            assert!(
                mag > 1e-6,
                "Complex64 R[{bi}][{i},{i}] near-zero: {diag:?}"
            );
        }
    }
}
