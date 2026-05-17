//! Real-GPU smoke test for `QrPlan` (cuSOLVER geqrf + ormqr wrap).
//!
//! Verifies `Q · R ≈ A` and `Q^T · Q ≈ I` for a general matrix `A`.
//! Storage convention is column-major end-to-end (cuSOLVER native).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, QrArgs, QrDescriptor, QrPlan, TensorMut,
    Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn build_matrix_f32(m: usize, n: usize, seed: u32) -> Vec<f32> {
    let mut a = vec![0f32; m * n];
    let mut s = seed.wrapping_mul(0x9E37_79B1);
    for cell in a.iter_mut() {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        *cell = ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
    }
    a
}

fn build_matrix_f64(m: usize, n: usize, seed: u32) -> Vec<f64> {
    build_matrix_f32(m, n, seed).into_iter().map(|v| v as f64).collect()
}

/// Column-major matmul: `c = a @ b`, `a: [ma, ka]`, `b: [kb, nb]`,
/// `c: [ma, nb]`, requires `ka == kb`.
fn matmul_cm_f32(a: &[f32], b: &[f32], ma: usize, k: usize, nb: usize) -> Vec<f32> {
    let mut c = vec![0f32; ma * nb];
    for j in 0..nb {
        for i in 0..ma {
            let mut acc = 0f32;
            for kk in 0..k {
                acc += a[kk * ma + i] * b[j * k + kk];
            }
            c[j * ma + i] = acc;
        }
    }
    c
}
fn matmul_cm_f64(a: &[f64], b: &[f64], ma: usize, k: usize, nb: usize) -> Vec<f64> {
    let mut c = vec![0f64; ma * nb];
    for j in 0..nb {
        for i in 0..ma {
            let mut acc = 0f64;
            for kk in 0..k {
                acc += a[kk * ma + i] * b[j * k + kk];
            }
            c[j * ma + i] = acc;
        }
    }
    c
}

#[test]
#[ignore]
fn qr_f32_basic() {
    let (ctx, stream) = setup();
    let m: i32 = 6;
    let n: i32 = 4;
    let a_host = build_matrix_f32(m as usize, n as usize, 0xA1A2_A3A4);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_q: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * m) as usize).expect("alloc q");
    let mut dev_r: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc r");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = QrDescriptor {
        m,
        n,
        element: ElementKind::F32,
    };
    let plan = QrPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select QrPlan<f32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    assert!(ws_bytes > 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let a_shape = [m, n];
    let a_stride = contiguous_stride(a_shape);
    let q_shape = [m, m];
    let q_stride = contiguous_stride(q_shape);
    let r_shape = [m, n];
    let r_stride = contiguous_stride(r_shape);
    let args = QrArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: a_stride,
        },
        q: TensorMut {
            data: dev_q.as_slice_mut(),
            shape: q_shape,
            stride: q_stride,
        },
        r: TensorMut {
            data: dev_r.as_slice_mut(),
            shape: r_shape,
            stride: r_stride,
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run QR f32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mut q_host = vec![0f32; (m * m) as usize];
    let mut r_host = vec![0f32; (m * n) as usize];
    dev_q.copy_to_host(&mut q_host).expect("dl Q");
    dev_r.copy_to_host(&mut r_host).expect("dl R");

    let mu = m as usize;
    let nu = n as usize;

    // Q · R ≈ A
    let reconstructed = matmul_cm_f32(&q_host, &r_host, mu, mu, nu);
    let tol = 1e-4f32;
    for j in 0..nu {
        for i in 0..mu {
            let got = reconstructed[j * mu + i];
            let expected = a_host[j * mu + i];
            let diff = (got - expected).abs();
            let t = tol * expected.abs().max(1.0);
            assert!(
                diff <= t,
                "f32 Q·R reconstruct ({i},{j}): got={got}, expected={expected}, diff={diff}",
            );
        }
    }

    // Q^T · Q ≈ I. Column-major Q^T is transposing read indices.
    let mut qtq = vec![0f32; mu * mu];
    for j in 0..mu {
        for i in 0..mu {
            let mut acc = 0f32;
            for kk in 0..mu {
                // Q^T[i, kk] = Q[kk, i] = q_host[i * m + kk]
                // Q[kk, j] = q_host[j * m + kk]
                acc += q_host[i * mu + kk] * q_host[j * mu + kk];
            }
            qtq[j * mu + i] = acc;
        }
    }
    for i in 0..mu {
        for j in 0..mu {
            let got = qtq[j * mu + i];
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-4,
                "f32 Q^T·Q ({i},{j}): got={got}, expected={expected}",
            );
        }
    }
}

#[test]
#[ignore]
fn qr_f64_basic() {
    let (ctx, stream) = setup();
    let m: i32 = 6;
    let n: i32 = 4;
    let a_host = build_matrix_f64(m as usize, n as usize, 0xB1B2_B3B4);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_q: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (m * m) as usize).expect("alloc q");
    let mut dev_r: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc r");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = QrDescriptor {
        m,
        n,
        element: ElementKind::F64,
    };
    let plan = QrPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select QrPlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let a_shape = [m, n];
    let q_shape = [m, m];
    let r_shape = [m, n];
    let args = QrArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        q: TensorMut {
            data: dev_q.as_slice_mut(),
            shape: q_shape,
            stride: contiguous_stride(q_shape),
        },
        r: TensorMut {
            data: dev_r.as_slice_mut(),
            shape: r_shape,
            stride: contiguous_stride(r_shape),
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run QR f64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mut q_host = vec![0f64; (m * m) as usize];
    let mut r_host = vec![0f64; (m * n) as usize];
    dev_q.copy_to_host(&mut q_host).expect("dl Q");
    dev_r.copy_to_host(&mut r_host).expect("dl R");

    let mu = m as usize;
    let nu = n as usize;
    let reconstructed = matmul_cm_f64(&q_host, &r_host, mu, mu, nu);
    let tol = 1e-10f64;
    for j in 0..nu {
        for i in 0..mu {
            let got = reconstructed[j * mu + i];
            let expected = a_host[j * mu + i];
            let diff = (got - expected).abs();
            let t = tol * expected.abs().max(1.0);
            assert!(
                diff <= t,
                "f64 Q·R reconstruct ({i},{j}): got={got}, expected={expected}, diff={diff}",
            );
        }
    }

    // Q^T · Q ≈ I
    let mut qtq = vec![0f64; mu * mu];
    for j in 0..mu {
        for i in 0..mu {
            let mut acc = 0f64;
            for kk in 0..mu {
                acc += q_host[i * mu + kk] * q_host[j * mu + kk];
            }
            qtq[j * mu + i] = acc;
        }
    }
    for i in 0..mu {
        for j in 0..mu {
            let got = qtq[j * mu + i];
            let expected = if i == j { 1.0 } else { 0.0 };
            let diff = (got - expected).abs();
            assert!(
                diff < 1e-10,
                "f64 Q^T·Q ({i},{j}): got={got}, expected={expected}",
            );
        }
    }
}
