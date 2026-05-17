//! Real-GPU smoke test for `BatchedSvdaPlan` (cuSOLVER
//! `gesvdaStridedBatched` — rectangular batched approximate-SVD).
//!
//! Verifies a 4×3 overdetermined batched SVD with two different
//! fixtures per slot, in both f32 and f64. Tests both full-rank
//! reconstruction (`rank = min(M, N) = 3`) and a truncated-rank
//! configuration (`rank = 2`).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, BatchedSvdaArgs, BatchedSvdaDescriptor, BatchedSvdaPlan, ElementKind,
    PlanPreference, TensorMut, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Column-major `m × n` · `n × p` = `m × p`.
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

fn matmul_cm_f64(a: &[f64], b: &[f64], m: usize, n: usize, p: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * p];
    for j in 0..p {
        for i in 0..m {
            let mut acc = 0f64;
            for k in 0..n {
                acc += a[k * m + i] * b[j * n + k];
            }
            c[j * m + i] = acc;
        }
    }
    c
}

/// Two batch slots of `[M=4, N=3]` column-major fixtures (overdetermined).
/// Slot 0: a tall diagonal-ish pattern with a tail row. Slot 1: a small
/// dense well-conditioned matrix.
fn make_fixtures_f32() -> Vec<f32> {
    // batch 0: column-major 4×3 ([col0; col1; col2])
    let slot0: [f32; 12] = [
        3.0, 0.0, 0.0, 1.0, // col 0
        0.0, 2.0, 0.0, 1.0, // col 1
        0.0, 0.0, 1.5, 1.0, // col 2
    ];
    // batch 1: column-major 4×3
    let slot1: [f32; 12] = [
        1.0, 2.0, 0.5, 0.1, // col 0
        0.5, 1.0, 2.0, 0.2, // col 1
        0.1, 0.5, 1.0, 3.0, // col 2
    ];
    let mut v = Vec::with_capacity(24);
    v.extend_from_slice(&slot0);
    v.extend_from_slice(&slot1);
    v
}

fn make_fixtures_f64() -> Vec<f64> {
    let slot0: [f64; 12] = [
        3.0, 0.0, 0.0, 1.0, //
        0.0, 2.0, 0.0, 1.0, //
        0.0, 0.0, 1.5, 1.0, //
    ];
    let slot1: [f64; 12] = [
        1.0, 2.0, 0.5, 0.1, //
        0.5, 1.0, 2.0, 0.2, //
        0.1, 0.5, 1.0, 3.0, //
    ];
    let mut v = Vec::with_capacity(24);
    v.extend_from_slice(&slot0);
    v.extend_from_slice(&slot1);
    v
}

#[test]
#[ignore]
fn svda_batched_f32_full_rank_reconstruct() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 4;
    let n: i32 = 3;
    let rank: i32 = 3; // min(M, N)

    let a_host = make_fixtures_f32();
    let a_orig = a_host.clone();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_s: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * rank) as usize).expect("alloc s");
    let mut dev_u: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * m * rank) as usize).expect("alloc u");
    let mut dev_v: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * n * rank) as usize).expect("alloc v");
    let mut dev_info: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, b as usize).expect("alloc info");

    let desc = BatchedSvdaDescriptor {
        m,
        n,
        rank,
        batch_size: b,
        compute_vectors: true,
        element: ElementKind::F32,
    };
    let plan = BatchedSvdaPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select BatchedSvdaPlan<f32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let mut residuals = vec![0f64; b as usize];
    let args = BatchedSvdaArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
        s: TensorMut {
            data: dev_s.as_slice_mut(),
            shape: [b, rank],
            stride: contiguous_stride([b, rank]),
        },
        u: Some(TensorMut {
            data: dev_u.as_slice_mut(),
            shape: [b, m, rank],
            stride: contiguous_stride([b, m, rank]),
        }),
        v: Some(TensorMut {
            data: dev_v.as_slice_mut(),
            shape: [b, n, rank],
            stride: contiguous_stride([b, n, rank]),
        }),
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [b],
            stride: [1],
        },
        residuals: Some(&mut residuals),
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run gesvda batched f32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; b as usize];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    for (bi, v) in info_host.iter().enumerate() {
        assert_eq!(*v, 0, "f32 gesvda info[{bi}] != 0");
    }

    let r = rank as usize;
    let mu = m as usize;
    let nu = n as usize;
    let mut s_host = vec![0f32; (b * rank) as usize];
    let mut u_host = vec![0f32; (b * m * rank) as usize];
    let mut v_host = vec![0f32; (b * n * rank) as usize];
    dev_s.copy_to_host(&mut s_host).expect("dl S");
    dev_u.copy_to_host(&mut u_host).expect("dl U");
    dev_v.copy_to_host(&mut v_host).expect("dl V");

    for bi in 0..b as usize {
        // Non-negative + descending singular values.
        let s_b = &s_host[bi * r..(bi + 1) * r];
        for i in 0..r {
            assert!(
                s_b[i] >= 0.0,
                "f32 σ < 0 at batch {bi}, idx {i}: {}",
                s_b[i]
            );
        }
        for i in 1..r {
            assert!(
                s_b[i - 1] + 1e-4 >= s_b[i],
                "f32 σ not descending at batch {bi}: {} < {}",
                s_b[i - 1],
                s_b[i]
            );
        }

        // Reconstruct A = U · diag(S) · V^T. U is [M, rank] cm; V is
        // [N, rank] cm; so U·diag(S) scales each rank-column of U by S[j].
        let u_b = &u_host[bi * mu * r..(bi + 1) * mu * r];
        let v_b = &v_host[bi * nu * r..(bi + 1) * nu * r];
        let mut us = vec![0f32; mu * r];
        for j in 0..r {
            for i in 0..mu {
                us[j * mu + i] = u_b[j * mu + i] * s_b[j];
            }
        }
        // V^T is [rank, N] cm == transposed read of V [N, rank].
        let mut vt = vec![0f32; r * nu];
        for j in 0..nu {
            for i in 0..r {
                // V[j, i] in cm of shape [N, rank] is v_b[i * N + j];
                // (V^T)[i, j] in cm of shape [rank, N] is vt[j * rank + i].
                vt[j * r + i] = v_b[i * nu + j];
            }
        }
        let reconstructed = matmul_cm_f32(&us, &vt, mu, r, nu);
        let a_b = &a_orig[bi * mu * nu..(bi + 1) * mu * nu];
        let tol = 1e-3f32;
        for k in 0..mu * nu {
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
fn svda_batched_f64_full_rank_reconstruct() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 4;
    let n: i32 = 3;
    let rank: i32 = 3;

    let a_host = make_fixtures_f64();
    let a_orig = a_host.clone();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_s: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (b * rank) as usize).expect("alloc s");
    let mut dev_u: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (b * m * rank) as usize).expect("alloc u");
    let mut dev_v: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (b * n * rank) as usize).expect("alloc v");
    let mut dev_info: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, b as usize).expect("alloc info");

    let desc = BatchedSvdaDescriptor {
        m,
        n,
        rank,
        batch_size: b,
        compute_vectors: true,
        element: ElementKind::F64,
    };
    let plan = BatchedSvdaPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select BatchedSvdaPlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = BatchedSvdaArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
        s: TensorMut {
            data: dev_s.as_slice_mut(),
            shape: [b, rank],
            stride: contiguous_stride([b, rank]),
        },
        u: Some(TensorMut {
            data: dev_u.as_slice_mut(),
            shape: [b, m, rank],
            stride: contiguous_stride([b, m, rank]),
        }),
        v: Some(TensorMut {
            data: dev_v.as_slice_mut(),
            shape: [b, n, rank],
            stride: contiguous_stride([b, n, rank]),
        }),
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [b],
            stride: [1],
        },
        // Exercise the residuals=None scratch-alloc path.
        residuals: None,
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run gesvda batched f64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; b as usize];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    for (bi, v) in info_host.iter().enumerate() {
        assert_eq!(*v, 0, "f64 gesvda info[{bi}] != 0");
    }

    let r = rank as usize;
    let mu = m as usize;
    let nu = n as usize;
    let mut s_host = vec![0f64; (b * rank) as usize];
    let mut u_host = vec![0f64; (b * m * rank) as usize];
    let mut v_host = vec![0f64; (b * n * rank) as usize];
    dev_s.copy_to_host(&mut s_host).expect("dl S");
    dev_u.copy_to_host(&mut u_host).expect("dl U");
    dev_v.copy_to_host(&mut v_host).expect("dl V");

    for bi in 0..b as usize {
        let s_b = &s_host[bi * r..(bi + 1) * r];
        for i in 0..r {
            assert!(s_b[i] >= 0.0, "f64 σ < 0 at batch {bi}, idx {i}: {}", s_b[i]);
        }
        for i in 1..r {
            assert!(
                s_b[i - 1] + 1e-12 >= s_b[i],
                "f64 σ not descending at batch {bi}: {} < {}",
                s_b[i - 1],
                s_b[i]
            );
        }

        let u_b = &u_host[bi * mu * r..(bi + 1) * mu * r];
        let v_b = &v_host[bi * nu * r..(bi + 1) * nu * r];
        let mut us = vec![0f64; mu * r];
        for j in 0..r {
            for i in 0..mu {
                us[j * mu + i] = u_b[j * mu + i] * s_b[j];
            }
        }
        let mut vt = vec![0f64; r * nu];
        for j in 0..nu {
            for i in 0..r {
                vt[j * r + i] = v_b[i * nu + j];
            }
        }
        let reconstructed = matmul_cm_f64(&us, &vt, mu, r, nu);
        let a_b = &a_orig[bi * mu * nu..(bi + 1) * mu * nu];
        let tol = 1e-10f64;
        for k in 0..mu * nu {
            let got = reconstructed[k];
            let want = a_b[k];
            let diff = (got - want).abs();
            assert!(
                diff <= tol * want.abs().max(1.0),
                "f64 reconstruct batch {bi} idx {k}: got={got}, want={want}, diff={diff}",
            );
        }
    }
}

#[test]
#[ignore]
fn svda_batched_f32_truncated_rank_singular_values() {
    let (ctx, stream) = setup();
    let b: i32 = 2;
    let m: i32 = 4;
    let n: i32 = 3;
    let rank: i32 = 2; // truncated — only top-2 triplets

    let a_host = make_fixtures_f32();
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_s: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * rank) as usize).expect("alloc s");
    let mut dev_u: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * m * rank) as usize).expect("alloc u");
    let mut dev_v: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * n * rank) as usize).expect("alloc v");
    let mut dev_info: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, b as usize).expect("alloc info");

    let desc = BatchedSvdaDescriptor {
        m,
        n,
        rank,
        batch_size: b,
        compute_vectors: true,
        element: ElementKind::F32,
    };
    let plan = BatchedSvdaPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select BatchedSvdaPlan<f32> truncated");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let args = BatchedSvdaArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: [b, m, n],
            stride: contiguous_stride([b, m, n]),
        },
        s: TensorMut {
            data: dev_s.as_slice_mut(),
            shape: [b, rank],
            stride: contiguous_stride([b, rank]),
        },
        u: Some(TensorMut {
            data: dev_u.as_slice_mut(),
            shape: [b, m, rank],
            stride: contiguous_stride([b, m, rank]),
        }),
        v: Some(TensorMut {
            data: dev_v.as_slice_mut(),
            shape: [b, n, rank],
            stride: contiguous_stride([b, n, rank]),
        }),
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [b],
            stride: [1],
        },
        residuals: None,
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run gesvda batched f32 truncated");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; b as usize];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    for (bi, v) in info_host.iter().enumerate() {
        assert_eq!(*v, 0, "f32 gesvda truncated info[{bi}] != 0");
    }

    let mut s_host = vec![0f32; (b * rank) as usize];
    dev_s.copy_to_host(&mut s_host).expect("dl S");
    let r = rank as usize;
    for bi in 0..b as usize {
        let s_b = &s_host[bi * r..(bi + 1) * r];
        for i in 0..r {
            assert!(
                s_b[i] >= 0.0,
                "f32 σ < 0 at batch {bi}, idx {i}: {}",
                s_b[i]
            );
        }
        // s_b[0] >= s_b[1] (descending).
        assert!(
            s_b[0] + 1e-4 >= s_b[1],
            "f32 truncated σ not descending at batch {bi}: {} < {}",
            s_b[0],
            s_b[1]
        );
    }
}
