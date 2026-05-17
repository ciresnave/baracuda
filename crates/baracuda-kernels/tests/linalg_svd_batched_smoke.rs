//! Real-GPU smoke test for `BatchedSvdPlan` (cuSOLVER `gesvdjBatched`).
//!
//! Verifies `A_b = U_b · diag(S_b) · V_b^T` per batch slot for a couple
//! of small well-conditioned matrices. Singular values must be non-
//! negative and sorted descending (per cuSOLVER's default `sort_eig`).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BatchedSvdArgs, BatchedSvdDescriptor, BatchedSvdPlan, ElementKind,
    PlanPreference, TensorMut, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Column-major `m × n` · `n × p` = `m × p`. Square `n == m` here.
fn matmul_cm_f32(a: &[f32], b: &[f32], m: usize, n: usize, p: usize) -> Vec<f32> {
    let mut c = vec![0f32; m * p];
    for j in 0..p {
        for i in 0..m {
            let mut acc = 0f32;
            for k in 0..n {
                acc += a[k * m + i] * b[j * n + k];
            }
            c[j * m + i] = acc;
        }
    }
    c
}

#[test]
#[ignore]
fn svd_batched_f32_basic() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let n: i32 = 3;
    // Two 3×3 matrices (column-major) — both well-conditioned.
    // Batch 0: diag(3, 2, 1). Batch 1: small perturbation of identity.
    let a_host: Vec<f32> = vec![
        // batch 0: diag(3, 2, 1) (cm)
        3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, // batch 1
        2.0, 0.0, 1.0, 0.5, 3.0, 0.0, 0.0, 0.5, 4.0,
    ];
    let a_orig = a_host.clone();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_s: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * n) as usize).expect("alloc s");
    let mut dev_u: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * n * n) as usize).expect("alloc u");
    let mut dev_v: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * n * n) as usize).expect("alloc v");
    let mut dev_info: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, b as usize).expect("alloc info");

    let desc = BatchedSvdDescriptor {
        matrix_size: n,
        batch_size: b,
        compute_vectors: true,
        element: ElementKind::F32,
    };
    let plan = BatchedSvdPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select BatchedSvdPlan<f32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let a_shape = [b, n, n];
    let args = BatchedSvdArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        s: TensorMut {
            data: dev_s.as_slice_mut(),
            shape: [b, n],
            stride: contiguous_stride([b, n]),
        },
        u: TensorMut {
            data: dev_u.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        v: TensorMut {
            data: dev_v.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [b],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run batched SVD f32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; b as usize];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    for (bi, v) in info_host.iter().enumerate() {
        assert_eq!(*v, 0, "f32 batched SVD info[{bi}] != 0");
    }

    let mut s_host = vec![0f32; (b * n) as usize];
    let mut u_host = vec![0f32; (b * n * n) as usize];
    let mut v_host = vec![0f32; (b * n * n) as usize];
    dev_s.copy_to_host(&mut s_host).expect("dl S");
    dev_u.copy_to_host(&mut u_host).expect("dl U");
    dev_v.copy_to_host(&mut v_host).expect("dl V");

    // Per batch: S non-negative + sorted descending; reconstruct
    // `U · diag(S) · V^T` and compare against original.
    let nu = n as usize;
    for bi in 0..b as usize {
        let s_b = &s_host[bi * nu..(bi + 1) * nu];
        for i in 0..nu {
            assert!(s_b[i] >= 0.0, "f32 σ < 0 at batch {bi}, idx {i}: {}", s_b[i]);
        }
        for i in 1..nu {
            assert!(
                s_b[i - 1] + 1e-5 >= s_b[i],
                "f32 σ not descending at batch {bi}: {} < {}",
                s_b[i - 1],
                s_b[i]
            );
        }

        let u_b = &u_host[bi * nu * nu..(bi + 1) * nu * nu];
        let v_b = &v_host[bi * nu * nu..(bi + 1) * nu * nu];
        // U · diag(S): scale each column j of U by S[j].
        let mut us = vec![0f32; nu * nu];
        for j in 0..nu {
            for i in 0..nu {
                us[j * nu + i] = u_b[j * nu + i] * s_b[j];
            }
        }
        // V^T in column-major == transposed read of V; equivalently
        // we can construct V^T explicitly.
        let mut vt = vec![0f32; nu * nu];
        for i in 0..nu {
            for j in 0..nu {
                // (V^T)[i, j] == V[j, i]; cm: vt[j * n + i] = V[j, i] = v_b[i * n + j]
                vt[j * nu + i] = v_b[i * nu + j];
            }
        }
        let reconstructed = matmul_cm_f32(&us, &vt, nu, nu, nu);
        let a_b = &a_orig[bi * nu * nu..(bi + 1) * nu * nu];
        let tol = 5e-4f32;
        for k in 0..nu * nu {
            let got = reconstructed[k];
            let want = a_b[k];
            let diff = (got - want).abs();
            assert!(
                diff <= tol * want.abs().max(1.0),
                "f32 reconstruct batch {bi} idx {k}: got={got}, want={want}, diff={diff}",
            );
        }
    }
}

#[test]
#[ignore]
fn svd_batched_f64_values_only() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let n: i32 = 3;
    let a_host: Vec<f64> = vec![
        3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, // batch 1
        4.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0,
    ];

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_s: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (b * n) as usize).expect("alloc s");
    let mut dev_u: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).expect("alloc dummy u");
    let mut dev_v: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 1).expect("alloc dummy v");
    let mut dev_info: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, b as usize).expect("alloc info");

    let desc = BatchedSvdDescriptor {
        matrix_size: n,
        batch_size: b,
        compute_vectors: false,
        element: ElementKind::F64,
    };
    let plan = BatchedSvdPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select BatchedSvdPlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = BatchedSvdArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, n, n],
            stride: contiguous_stride([b, n, n]),
        },
        s: TensorMut {
            data: dev_s.as_slice_mut(),
            shape: [b, n],
            stride: contiguous_stride([b, n]),
        },
        u: TensorMut {
            data: dev_u.as_slice_mut(),
            shape: [1, 1, 1],
            stride: [1, 1, 1],
        },
        v: TensorMut {
            data: dev_v.as_slice_mut(),
            shape: [1, 1, 1],
            stride: [1, 1, 1],
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [b],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run batched SVD f64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; b as usize];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    for v in info_host.iter() {
        assert_eq!(*v, 0);
    }
    let mut s_host = vec![0f64; (b * n) as usize];
    dev_s.copy_to_host(&mut s_host).expect("dl S");
    // Diagonal matrices: singular values are abs(diag) sorted descending.
    let expected: Vec<f64> = vec![3.0, 2.0, 1.0, 4.0, 3.0, 2.0];
    for (i, (got, want)) in s_host.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff < 1e-10,
            "f64 σ[{i}]: got={got}, want={want}, diff={diff}",
        );
    }
}
