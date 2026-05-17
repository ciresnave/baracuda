//! Real-GPU smoke test for `BatchedQrMaterializePlan` (Milestone 6.14).
//!
//! Pipeline:
//!   1. Build a stack of input matrices `A [B, M, N]` column-major.
//!   2. Factor with `BatchedQrPlan` → packed `A`, `tau`.
//!   3. Materialize dense `Q [B, M, M]` and `R [B, K, N]` via
//!      `BatchedQrMaterializePlan`.
//!   4. Verify `Q · R == A` (per slot, column-major) and `Q^T · Q == I`.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BatchedQrArgs, BatchedQrDescriptor, BatchedQrMaterializeArgs,
    BatchedQrMaterializeDescriptor, BatchedQrMaterializePlan, BatchedQrPlan, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn build_matrix_f32(b: usize, m: usize, n: usize, seed: u32) -> Vec<f32> {
    let mut a = vec![0f32; b * m * n];
    let mut s = seed.wrapping_mul(0x9E37_79B1);
    for cell in a.iter_mut() {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        *cell = ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
    }
    a
}

fn build_matrix_f64(b: usize, m: usize, n: usize, seed: u32) -> Vec<f64> {
    build_matrix_f32(b, m, n, seed)
        .into_iter()
        .map(|v| v as f64)
        .collect()
}

/// Column-major matmul, single slot: `c = a @ b`, `a:[ma, ka]`,
/// `b:[ka, nb]`, `c:[ma, nb]`.
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
fn qr_batched_materialize_f32_basic() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 5;
    let n: i32 = 3;
    let k = m.min(n);
    let a_host = build_matrix_f32(b as usize, m as usize, n as usize, 0xA111_2222);

    // 1 + 2. Run batched QR.
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_tau: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * k) as usize).expect("alloc tau");
    let qr_desc = BatchedQrDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::F32,
    };
    let qr_plan = BatchedQrPlan::<f32>::select(&stream, &qr_desc, PlanPreference::default())
        .expect("select QR");
    let ws_bytes = qr_plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_qr_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc qr ws");
    let qr_args = BatchedQrArgs::<f32> {
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
    qr_plan
        .run(&stream, Workspace::Borrowed(dev_qr_ws.as_slice_mut()), qr_args)
        .expect("run batched QR");

    // 3. Materialize.
    let mut dev_q: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * m * m) as usize).expect("alloc Q");
    let mut dev_r: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * k * n) as usize).expect("alloc R");
    let mat_desc = BatchedQrMaterializeDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::F32,
    };
    let mat_plan =
        BatchedQrMaterializePlan::<f32>::select(&stream, &mat_desc, PlanPreference::default())
            .expect("select materialize");
    let mat_args = BatchedQrMaterializeArgs::<f32> {
        a_packed: TensorRef {
            data: dev_a.as_slice(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
        tau: TensorRef {
            data: dev_tau.as_slice(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
        q: TensorMut {
            data: dev_q.as_slice_mut(),
            shape: [b, m, m],
            stride: contiguous_stride([b, m, m]),
        },
        r: TensorMut {
            data: dev_r.as_slice_mut(),
            shape: [b, k, n],
            stride: contiguous_stride([b, k, n]),
        },
    };
    mat_plan
        .run(&stream, Workspace::None, mat_args)
        .expect("run materialize");
    stream.synchronize().expect("sync");

    let mut q_host = vec![0f32; (b * m * m) as usize];
    let mut r_host = vec![0f32; (b * k * n) as usize];
    dev_q.copy_to_host(&mut q_host).expect("dl Q");
    dev_r.copy_to_host(&mut r_host).expect("dl R");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;

    let tol = 1e-4f32;
    for bi in 0..b as usize {
        let qb = &q_host[bi * mu * mu..(bi + 1) * mu * mu];
        let rb = &r_host[bi * ku * nu..(bi + 1) * ku * nu];
        let ab = &a_host[bi * mu * nu..(bi + 1) * mu * nu];

        // Q · R == A. Q is [M, M], R is [K, N] = [N, N] here (since K = N).
        let reconstructed = matmul_cm_f32(qb, rb, mu, ku, nu);
        for j in 0..nu {
            for i in 0..mu {
                let got = reconstructed[j * mu + i];
                let expected = ab[j * mu + i];
                let diff = (got - expected).abs();
                let t = tol * expected.abs().max(1.0);
                assert!(
                    diff <= t,
                    "f32 slot {bi} Q·R reconstruct ({i},{j}): got={got}, expected={expected}, diff={diff}",
                );
            }
        }

        // Q^T · Q == I.
        let mut qtq = vec![0f32; mu * mu];
        for j in 0..mu {
            for i in 0..mu {
                let mut acc = 0f32;
                for kk in 0..mu {
                    acc += qb[i * mu + kk] * qb[j * mu + kk];
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
                    diff <= tol,
                    "f32 slot {bi} Q^T·Q ({i},{j}): got={got}, expected={expected}",
                );
            }
        }

        // R lower triangle is zero.
        for j in 0..nu {
            for i in 0..ku {
                if i > j {
                    let got = rb[j * ku + i];
                    assert!(
                        got.abs() <= tol,
                        "f32 slot {bi} R lower-tri ({i},{j}): expected 0, got {got}",
                    );
                }
            }
        }
    }
}

#[test]
#[ignore]
fn qr_batched_materialize_f64_basic() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 5;
    let n: i32 = 3;
    let k = m.min(n);
    let a_host = build_matrix_f64(b as usize, m as usize, n as usize, 0x1357_9BDF);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_tau: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (b * k) as usize).expect("alloc tau");
    let qr_desc = BatchedQrDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::F64,
    };
    let qr_plan = BatchedQrPlan::<f64>::select(&stream, &qr_desc, PlanPreference::default())
        .expect("select QR");
    let ws_bytes = qr_plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_qr_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc qr ws");
    let qr_args = BatchedQrArgs::<f64> {
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
    qr_plan
        .run(&stream, Workspace::Borrowed(dev_qr_ws.as_slice_mut()), qr_args)
        .expect("run batched QR");

    let mut dev_q: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (b * m * m) as usize).expect("alloc Q");
    let mut dev_r: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (b * k * n) as usize).expect("alloc R");
    let mat_desc = BatchedQrMaterializeDescriptor {
        m,
        n,
        batch_size: b,
        element: ElementKind::F64,
    };
    let mat_plan =
        BatchedQrMaterializePlan::<f64>::select(&stream, &mat_desc, PlanPreference::default())
            .expect("select materialize");
    let mat_args = BatchedQrMaterializeArgs::<f64> {
        a_packed: TensorRef {
            data: dev_a.as_slice(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
        tau: TensorRef {
            data: dev_tau.as_slice(),
            shape: [b, k],
            stride: contiguous_stride([b, k]),
        },
        q: TensorMut {
            data: dev_q.as_slice_mut(),
            shape: [b, m, m],
            stride: contiguous_stride([b, m, m]),
        },
        r: TensorMut {
            data: dev_r.as_slice_mut(),
            shape: [b, k, n],
            stride: contiguous_stride([b, k, n]),
        },
    };
    mat_plan
        .run(&stream, Workspace::None, mat_args)
        .expect("run materialize");
    stream.synchronize().expect("sync");

    let mut q_host = vec![0f64; (b * m * m) as usize];
    let mut r_host = vec![0f64; (b * k * n) as usize];
    dev_q.copy_to_host(&mut q_host).expect("dl Q");
    dev_r.copy_to_host(&mut r_host).expect("dl R");

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;

    let tol = 1e-10f64;
    for bi in 0..b as usize {
        let qb = &q_host[bi * mu * mu..(bi + 1) * mu * mu];
        let rb = &r_host[bi * ku * nu..(bi + 1) * ku * nu];
        let ab = &a_host[bi * mu * nu..(bi + 1) * mu * nu];

        let reconstructed = matmul_cm_f64(qb, rb, mu, ku, nu);
        for j in 0..nu {
            for i in 0..mu {
                let got = reconstructed[j * mu + i];
                let expected = ab[j * mu + i];
                let diff = (got - expected).abs();
                let t = tol * expected.abs().max(1.0);
                assert!(
                    diff <= t,
                    "f64 slot {bi} Q·R reconstruct ({i},{j}): got={got}, expected={expected}, diff={diff}",
                );
            }
        }

        let mut qtq = vec![0f64; mu * mu];
        for j in 0..mu {
            for i in 0..mu {
                let mut acc = 0f64;
                for kk in 0..mu {
                    acc += qb[i * mu + kk] * qb[j * mu + kk];
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
                    diff <= tol,
                    "f64 slot {bi} Q^T·Q ({i},{j}): got={got}, expected={expected}",
                );
            }
        }

        for j in 0..nu {
            for i in 0..ku {
                if i > j {
                    let got = rb[j * ku + i];
                    assert!(
                        got.abs() <= tol,
                        "f64 slot {bi} R lower-tri ({i},{j}): expected 0, got {got}",
                    );
                }
            }
        }
    }
}
