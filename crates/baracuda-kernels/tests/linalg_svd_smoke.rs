//! Real-GPU smoke test for `SvdPlan` (cuSOLVER gesvd wrap).
//!
//! Verifies `U · diag(S) · V^T ≈ A` for a general matrix `A`, and that
//! `U` / `V^T` are (semi-)orthogonal. Storage convention is column-
//! major end-to-end (cuSOLVER native).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SvdArgs, SvdDescriptor, SvdPlan, TensorMut,
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

/// Column-major matmul.
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
fn svd_f32_basic() {
    let (ctx, stream) = setup();
    let m: i32 = 6;
    let n: i32 = 4;
    let a_host = build_matrix_f32(m as usize, n as usize, 0xCAFE_BABE);
    let k = m.min(n);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_s: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, k as usize).expect("alloc s");
    let mut dev_u: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * m) as usize).expect("alloc u");
    let mut dev_vt: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * n) as usize).expect("alloc vt");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = SvdDescriptor {
        m,
        n,
        full_matrices: true,
        element: ElementKind::F32,
    };
    let plan = SvdPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select SvdPlan<f32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    assert!(ws_bytes > 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let a_shape = [m, n];
    let args = SvdArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        s: TensorMut {
            data: dev_s.as_slice_mut(),
            shape: [k],
            stride: [1],
        },
        u: TensorMut {
            data: dev_u.as_slice_mut(),
            shape: [m, m],
            stride: contiguous_stride([m, m]),
        },
        vt: TensorMut {
            data: dev_vt.as_slice_mut(),
            shape: [n, n],
            stride: contiguous_stride([n, n]),
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run SVD f32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let mut s_host = vec![0f32; ku];
    let mut u_host = vec![0f32; mu * mu];
    let mut vt_host = vec![0f32; nu * nu];
    dev_s.copy_to_host(&mut s_host).expect("dl s");
    dev_u.copy_to_host(&mut u_host).expect("dl u");
    dev_vt.copy_to_host(&mut vt_host).expect("dl vt");

    // Singular values are non-negative and sorted descending.
    for i in 0..ku {
        assert!(
            s_host[i] >= 0.0 && s_host[i].is_finite(),
            "f32 SVD: s[{i}] = {} (must be finite, >= 0)",
            s_host[i]
        );
        if i > 0 {
            assert!(
                s_host[i - 1] >= s_host[i] - 1e-5,
                "f32 SVD: s not descending at {i}: {} -> {}",
                s_host[i - 1],
                s_host[i]
            );
        }
    }

    // Construct US: [M, N] in column-major. For full SVD, U is [M, M].
    // We need to multiply U (first K columns) · diag(S) · V^T (first K rows
    // for full mode, but vt itself is [N, N] — only the first K rows
    // correspond to the basis vectors paired with non-zero singular values).
    //
    // For full mode, the implicit shape after the SVD is:
    //   A = U[:, :K] · diag(S) · V^T[:K, :]
    // The trailing rows / cols of U / V^T are auxiliary basis vectors.
    let mut us = vec![0f32; mu * ku]; // column-major [M, K]
    for j in 0..ku {
        for i in 0..mu {
            // u_host is column-major [M, M]; column j is u_host[j*M..(j+1)*M]
            us[j * mu + i] = u_host[j * mu + i] * s_host[j];
        }
    }
    // vt_top: first k rows of vt_host (column-major [N, N]).
    // For column-major [N, N], cell (i, j) is at vt_host[j*N + i]. We want
    // vt_top[i, j] = vt[i, j] for i in 0..K.
    let mut vt_top = vec![0f32; ku * nu]; // column-major [K, N]
    for j in 0..nu {
        for i in 0..ku {
            vt_top[j * ku + i] = vt_host[j * nu + i];
        }
    }
    let reconstructed = matmul_cm_f32(&us, &vt_top, mu, ku, nu);

    let tol = 1e-4f32;
    for j in 0..nu {
        for i in 0..mu {
            let got = reconstructed[j * mu + i];
            let expected = a_host[j * mu + i];
            let diff = (got - expected).abs();
            let t = tol * expected.abs().max(1.0);
            assert!(
                diff <= t,
                "f32 SVD reconstruct ({i},{j}): got={got}, expected={expected}, diff={diff}",
            );
        }
    }
}

#[test]
#[ignore]
fn svd_f64_basic() {
    let (ctx, stream) = setup();
    let m: i32 = 6;
    let n: i32 = 4;
    let a_host = build_matrix_f64(m as usize, n as usize, 0xFEED_FACE);
    let k = m.min(n);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_s: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, k as usize).expect("alloc s");
    let mut dev_u: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (m * m) as usize).expect("alloc u");
    let mut dev_vt: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (n * n) as usize).expect("alloc vt");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = SvdDescriptor {
        m,
        n,
        full_matrices: true,
        element: ElementKind::F64,
    };
    let plan = SvdPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select SvdPlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let a_shape = [m, n];
    let args = SvdArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: contiguous_stride(a_shape),
        },
        s: TensorMut {
            data: dev_s.as_slice_mut(),
            shape: [k],
            stride: [1],
        },
        u: TensorMut {
            data: dev_u.as_slice_mut(),
            shape: [m, m],
            stride: contiguous_stride([m, m]),
        },
        vt: TensorMut {
            data: dev_vt.as_slice_mut(),
            shape: [n, n],
            stride: contiguous_stride([n, n]),
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run SVD f64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mu = m as usize;
    let nu = n as usize;
    let ku = k as usize;
    let mut s_host = vec![0f64; ku];
    let mut u_host = vec![0f64; mu * mu];
    let mut vt_host = vec![0f64; nu * nu];
    dev_s.copy_to_host(&mut s_host).expect("dl s");
    dev_u.copy_to_host(&mut u_host).expect("dl u");
    dev_vt.copy_to_host(&mut vt_host).expect("dl vt");

    for i in 0..ku {
        assert!(s_host[i] >= 0.0 && s_host[i].is_finite());
        if i > 0 {
            assert!(s_host[i - 1] >= s_host[i] - 1e-10);
        }
    }
    let mut us = vec![0f64; mu * ku];
    for j in 0..ku {
        for i in 0..mu {
            us[j * mu + i] = u_host[j * mu + i] * s_host[j];
        }
    }
    let mut vt_top = vec![0f64; ku * nu];
    for j in 0..nu {
        for i in 0..ku {
            vt_top[j * ku + i] = vt_host[j * nu + i];
        }
    }
    let reconstructed = matmul_cm_f64(&us, &vt_top, mu, ku, nu);

    let tol = 1e-10f64;
    for j in 0..nu {
        for i in 0..mu {
            let got = reconstructed[j * mu + i];
            let expected = a_host[j * mu + i];
            let diff = (got - expected).abs();
            let t = tol * expected.abs().max(1.0);
            assert!(
                diff <= t,
                "f64 SVD reconstruct ({i},{j}): got={got}, expected={expected}, diff={diff}",
            );
        }
    }
}
