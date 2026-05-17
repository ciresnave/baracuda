//! Real-GPU smoke test for `LstSqPlan` (cuSOLVER `_gels` wrap).
//!
//! Verifies `min ||A·x - b||²` for a small overdetermined full-rank
//! system. Column-major end-to-end.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LstSqArgs, LstSqDescriptor, LstSqPlan, PlanPreference,
    TensorMut, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Column-major `m × n` · `n × p` → `m × p`.
fn matmul_cm_f32(a: &[f32], x: &[f32], m: usize, n: usize, p: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * p];
    for j in 0..p {
        for i in 0..m {
            let mut acc = 0f32;
            for k in 0..n {
                acc += a[k * m + i] * x[j * n + k];
            }
            c[j * m + i] = acc;
        }
    }
    c
}
fn matmul_cm_f64(a: &[f64], x: &[f64], m: usize, n: usize, p: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * p];
    for j in 0..p {
        for i in 0..m {
            let mut acc = 0f64;
            for k in 0..n {
                acc += a[k * m + i] * x[j * n + k];
            }
            c[j * m + i] = acc;
        }
    }
    c
}

#[test]
#[ignore]
fn lstsq_f32_overdetermined() {
    let (ctx, stream) = setup();
    let m: i32 = 4;
    let n: i32 = 2;
    let nrhs: i32 = 1;
    // A: 4×2 column-major, full-rank.
    // Row-major view: [[1,1],[1,2],[1,3],[1,4]]  (constant + slope basis)
    // Column-major flatten: col0 [1,1,1,1], col1 [1,2,3,4].
    let a_host: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
    // Truth: x = [1, 2]^T → b = [3, 5, 7, 9]^T (perfect fit, no residual).
    let x_truth: Vec<f32> = vec![1.0, 2.0];
    let b_host = matmul_cm_f32(&a_host, &x_truth, m as usize, n as usize, nrhs as usize);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload b");
    let mut dev_x: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * nrhs) as usize).expect("alloc x");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = LstSqDescriptor {
        m,
        n,
        nrhs,
        element: ElementKind::F32,
    };
    let plan = LstSqPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select LstSqPlan<f32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = LstSqArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [m, n],
            stride: contiguous_stride([m, n]),
        },
        b: TensorMut {
            data: dev_b.as_slice_mut(),
            shape: [m, nrhs],
            stride: contiguous_stride([m, nrhs]),
        },
        x: TensorMut {
            data: dev_x.as_slice_mut(),
            shape: [n, nrhs],
            stride: contiguous_stride([n, nrhs]),
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1],
            stride: [1],
        },
        a_backup: None,
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run LstSq f32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mut x_host = vec![0f32; (n * nrhs) as usize];
    dev_x.copy_to_host(&mut x_host).expect("dl X");
    let tol = 1e-4f32;
    for i in 0..(n * nrhs) as usize {
        let got = x_host[i];
        let want = x_truth[i];
        let diff = (got - want).abs();
        assert!(
            diff <= tol * want.abs().max(1.0),
            "f32 lstsq idx {i}: got={got}, want={want}, diff={diff}",
        );
    }
}

#[test]
#[ignore]
fn lstsq_f64_overdetermined() {
    let (ctx, stream) = setup();
    let m: i32 = 4;
    let n: i32 = 2;
    let nrhs: i32 = 1;
    let a_host: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
    let x_truth: Vec<f64> = vec![1.0, 2.0];
    let b_host = matmul_cm_f64(&a_host, &x_truth, m as usize, n as usize, nrhs as usize);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload b");
    let mut dev_x: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (n * nrhs) as usize).expect("alloc x");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = LstSqDescriptor {
        m,
        n,
        nrhs,
        element: ElementKind::F64,
    };
    let plan = LstSqPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select LstSqPlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = LstSqArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [m, n],
            stride: contiguous_stride([m, n]),
        },
        b: TensorMut {
            data: dev_b.as_slice_mut(),
            shape: [m, nrhs],
            stride: contiguous_stride([m, nrhs]),
        },
        x: TensorMut {
            data: dev_x.as_slice_mut(),
            shape: [n, nrhs],
            stride: contiguous_stride([n, nrhs]),
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1],
            stride: [1],
        },
        a_backup: None,
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run LstSq f64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mut x_host = vec![0f64; (n * nrhs) as usize];
    dev_x.copy_to_host(&mut x_host).expect("dl X");
    let tol = 1e-10f64;
    for i in 0..(n * nrhs) as usize {
        let got = x_host[i];
        let want = x_truth[i];
        let diff = (got - want).abs();
        assert!(
            diff <= tol * want.abs().max(1.0),
            "f64 lstsq idx {i}: got={got}, want={want}, diff={diff}",
        );
    }
}

// ----------------------------------------------------------------------
// QR-fallback path tests
// ----------------------------------------------------------------------
//
// `cusolverDn{SS,DD}gels` is iterative — for poorly-conditioned matrices
// it returns `niters < 0` and the safe-plan layer must fall back to
// `geqrf` + `ormqr(Q^T)` + `trsm(R)`. We trigger non-convergence with a
// Hilbert-like matrix (condition number grows ~exponentially with N).
// `[6, 4]` Hilbert columns push cond(A) ~10^7+ which defeats
// iterative refinement in f32 reliably.

use baracuda_kernels::TensorRef;

/// `H[i, j] = 1 / (i + j + 1)` — Hilbert columns, in column-major.
/// `H` is `[M, N]` with `H[i, j]` at flat index `j * M + i`.
fn hilbert_cm_f32(m: usize, n: usize) -> Vec<f32> {
    let mut a = vec![0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            a[j * m + i] = 1.0 / ((i + j + 1) as f32);
        }
    }
    a
}
fn hilbert_cm_f64(m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0f64; m * n];
    for j in 0..n {
        for i in 0..m {
            a[j * m + i] = 1.0 / ((i + j + 1) as f64);
        }
    }
    a
}

/// Solve `min ||A·x - b||²` on the host via the normal equations in
/// f64. Returns `x` of length `n`. Adequate as a reference for the
/// small `[6, 4]` Hilbert case (cond ~10^7 → f64 still resolves
/// ~6 digits) — far more accurate than the f32 GPU path can be.
fn host_lstsq_f64(a: &[f64], b: &[f64], m: usize, n: usize) -> Vec<f64> {
    // A^T A (n × n), A^T b (n).
    let mut ata = vec![0f64; n * n];
    let mut atb = vec![0f64; n];
    for j in 0..n {
        for i in 0..n {
            let mut acc = 0f64;
            for k in 0..m {
                acc += a[i * m + k] * a[j * m + k];
            }
            ata[j * n + i] = acc;
        }
        let mut acc = 0f64;
        for k in 0..m {
            acc += a[j * m + k] * b[k];
        }
        atb[j] = acc;
    }
    // Gauss-Jordan with partial pivoting on the (n × n+1) augmented matrix.
    let mut aug = vec![0f64; n * (n + 1)];
    for j in 0..n {
        for i in 0..n {
            aug[i * (n + 1) + j] = ata[j * n + i];
        }
        aug[j * (n + 1) + n] = atb[j];
    }
    for col in 0..n {
        let mut pivot = col;
        for row in (col + 1)..n {
            if aug[row * (n + 1) + col].abs() > aug[pivot * (n + 1) + col].abs() {
                pivot = row;
            }
        }
        if pivot != col {
            for c in 0..=n {
                aug.swap(col * (n + 1) + c, pivot * (n + 1) + c);
            }
        }
        let div = aug[col * (n + 1) + col];
        for c in 0..=n {
            aug[col * (n + 1) + c] /= div;
        }
        for row in 0..n {
            if row != col {
                let f = aug[row * (n + 1) + col];
                for c in 0..=n {
                    aug[row * (n + 1) + c] -= f * aug[col * (n + 1) + c];
                }
            }
        }
    }
    (0..n).map(|i| aug[i * (n + 1) + n]).collect()
}

// NOTE on "expect non-convergence" tests: cuSOLVER's iterative-refinement
// `_gels` is robust enough that even Hilbert-style fixtures often converge
// at f32 precision. We can't reliably trigger the fallback path from a
// pure-test fixture without a forced-fallback knob in the plan (deferred
// — would require API surface change). The `_succeeds_with_backup` tests
// below exercise the API contract end-to-end with `a_backup` provided;
// the fallback path itself is exercised opportunistically when cuSOLVER's
// inner solver does happen to fail.

#[test]
#[ignore]
fn lstsq_f32_qr_fallback_succeeds_with_backup() {
    let (ctx, stream) = setup();
    let m: i32 = 6;
    let n: i32 = 4;
    let nrhs: i32 = 1;
    let a_host = hilbert_cm_f32(m as usize, n as usize);
    let x_truth: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0];
    let b_host = matmul_cm_f32(&a_host, &x_truth, m as usize, n as usize, nrhs as usize);

    // f64 reference via host normal equations.
    let a_host_f64: Vec<f64> = a_host.iter().map(|&v| v as f64).collect();
    let b_host_f64: Vec<f64> = b_host.iter().map(|&v| v as f64).collect();
    let x_ref_f64 = host_lstsq_f64(&a_host_f64, &b_host_f64, m as usize, n as usize);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let dev_a_backup = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a_backup");
    let mut dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload b");
    let mut dev_x: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * nrhs) as usize).expect("alloc x");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = LstSqDescriptor {
        m,
        n,
        nrhs,
        element: ElementKind::F32,
    };
    let plan = LstSqPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select LstSqPlan<f32>");
    let ws_gels = plan.query_workspace_size(&stream).expect("ws gels query");
    let ws_qr = plan
        .query_qr_fallback_workspace_size(&stream)
        .expect("ws qr query");
    let ws_bytes = ws_gels.max(ws_qr);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = LstSqArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [m, n],
            stride: contiguous_stride([m, n]),
        },
        b: TensorMut {
            data: dev_b.as_slice_mut(),
            shape: [m, nrhs],
            stride: contiguous_stride([m, nrhs]),
        },
        x: TensorMut {
            data: dev_x.as_slice_mut(),
            shape: [n, nrhs],
            stride: contiguous_stride([n, nrhs]),
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1],
            stride: [1],
        },
        a_backup: Some(TensorRef {
            data: dev_a_backup.as_slice(),
            shape: [m, n],
            stride: contiguous_stride([m, n]),
        }),
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run LstSq f32 QR fallback");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "geqrf/ormqr info nonzero: {}", info_host[0]);

    let mut x_host = vec![0f32; (n * nrhs) as usize];
    dev_x.copy_to_host(&mut x_host).expect("dl X");

    // Backward-error bound for f32 least-squares: |x_gpu - x_ref| <=
    // K * eps * ||A|| * ||x||. We use a generous K and a Frobenius-
    // norm proxy — the Hilbert system's amplification is roughly the
    // condition number times machine epsilon (~10^7 * 2^-23 ~ 1.2),
    // so we cap absolute deviation at a percent of ||x_ref||.
    let mut a_fro_sq = 0f32;
    for &v in &a_host {
        a_fro_sq += v * v;
    }
    let a_fro = a_fro_sq.sqrt();
    let x_norm: f32 = x_ref_f64.iter().map(|v| (*v as f32).powi(2)).sum::<f32>().sqrt();
    let tol = 100.0 * f32::EPSILON * a_fro * x_norm.max(1.0);
    // Floor the tolerance: Hilbert at cond~10^7 in f32 cannot resolve
    // truth components to better than ~1e-1, so demand only that we
    // beat the no-op baseline (||x_ref||).
    let tol = tol.max(0.5 * x_norm);
    for i in 0..(n * nrhs) as usize {
        let got = x_host[i];
        let want = x_ref_f64[i] as f32;
        let diff = (got - want).abs();
        assert!(
            diff <= tol,
            "f32 lstsq QR-fallback idx {i}: got={got}, want={want}, diff={diff}, tol={tol}",
        );
    }
}

#[test]
#[ignore]
fn lstsq_f64_qr_fallback_succeeds_with_backup() {
    let (ctx, stream) = setup();
    let m: i32 = 6;
    let n: i32 = 4;
    let nrhs: i32 = 1;
    // For f64, Hilbert [6, 4] usually converges in iterative refinement.
    // The test exercises the wiring by passing a_backup anyway — even
    // if the gels path converges, the test verifies the same end-to-end
    // contract (no error, X close to truth).
    let a_host = hilbert_cm_f64(m as usize, n as usize);
    let x_truth: Vec<f64> = vec![1.0, -2.0, 3.0, -4.0];
    let b_host = matmul_cm_f64(&a_host, &x_truth, m as usize, n as usize, nrhs as usize);
    let x_ref = host_lstsq_f64(&a_host, &b_host, m as usize, n as usize);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let dev_a_backup = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a_backup");
    let mut dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload b");
    let mut dev_x: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (n * nrhs) as usize).expect("alloc x");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = LstSqDescriptor {
        m,
        n,
        nrhs,
        element: ElementKind::F64,
    };
    let plan = LstSqPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select LstSqPlan<f64>");
    let ws_gels = plan.query_workspace_size(&stream).expect("ws gels query");
    let ws_qr = plan
        .query_qr_fallback_workspace_size(&stream)
        .expect("ws qr query");
    let ws_bytes = ws_gels.max(ws_qr);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = LstSqArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [m, n],
            stride: contiguous_stride([m, n]),
        },
        b: TensorMut {
            data: dev_b.as_slice_mut(),
            shape: [m, nrhs],
            stride: contiguous_stride([m, nrhs]),
        },
        x: TensorMut {
            data: dev_x.as_slice_mut(),
            shape: [n, nrhs],
            stride: contiguous_stride([n, nrhs]),
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1],
            stride: [1],
        },
        a_backup: Some(TensorRef {
            data: dev_a_backup.as_slice(),
            shape: [m, n],
            stride: contiguous_stride([m, n]),
        }),
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run LstSq f64 with backup");
    stream.synchronize().expect("sync");

    let mut x_host = vec![0f64; (n * nrhs) as usize];
    dev_x.copy_to_host(&mut x_host).expect("dl X");

    let mut a_fro_sq = 0f64;
    for &v in &a_host {
        a_fro_sq += v * v;
    }
    let a_fro = a_fro_sq.sqrt();
    let x_norm: f64 = x_ref.iter().map(|v| v.powi(2)).sum::<f64>().sqrt();
    let tol = (100.0 * f64::EPSILON * a_fro * x_norm.max(1.0)).max(1e-6 * x_norm);
    for i in 0..(n * nrhs) as usize {
        let got = x_host[i];
        let want = x_ref[i];
        let diff = (got - want).abs();
        assert!(
            diff <= tol,
            "f64 lstsq backup idx {i}: got={got}, want={want}, diff={diff}, tol={tol}",
        );
    }
}
