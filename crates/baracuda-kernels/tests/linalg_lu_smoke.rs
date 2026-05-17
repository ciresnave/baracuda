//! Real-GPU smoke test for `LuPlan` (cuSOLVER `getrf` wrap).
//!
//! Verifies `P · A ≈ L · U` for a general (non-singular) matrix `A`.
//! Storage convention is column-major end-to-end (cuSOLVER native).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, LuArgs, LuDescriptor, LuPlan, PlanPreference, TensorMut,
    Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Build a deterministic non-singular `n × n` matrix in column-major
/// layout. We use `A = I + B` where `B` is small-magnitude pseudo-
/// random to keep `||A|| ≈ 1` and pivoting non-trivial.
fn nonsingular_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut a = vec![0f32; n * n];
    let mut s = seed.wrapping_mul(0x9E37_79B1);
    for j in 0..n {
        for i in 0..n {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let f = ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
            // Column-major: a[j*n + i] is the (i, j) cell.
            a[j * n + i] = 0.3 * f + if i == j { 1.0 } else { 0.0 };
        }
    }
    a
}

fn nonsingular_f64(n: usize, seed: u32) -> Vec<f64> {
    nonsingular_f32(n, seed).into_iter().map(|v| v as f64).collect()
}

/// Reconstruct `L · U` from cuSOLVER's packed `LU` output (column-
/// major). `L` is unit-diagonal in strict lower; `U` is upper.
fn reconstruct_lu_f32(lu: &[f32], n: usize) -> Vec<f32> {
    let mut result = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..n {
                let l_ik = if i == kk {
                    1.0
                } else if i > kk {
                    lu[kk * n + i]
                } else {
                    0.0
                };
                let u_kj = if kk <= j { lu[j * n + kk] } else { 0.0 };
                acc += l_ik * u_kj;
            }
            result[j * n + i] = acc;
        }
    }
    result
}

fn reconstruct_lu_f64(lu: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0f64;
            for kk in 0..n {
                let l_ik = if i == kk {
                    1.0
                } else if i > kk {
                    lu[kk * n + i]
                } else {
                    0.0
                };
                let u_kj = if kk <= j { lu[j * n + kk] } else { 0.0 };
                acc += l_ik * u_kj;
            }
            result[j * n + i] = acc;
        }
    }
    result
}

/// Apply the pivot vector returned by cuSOLVER (`int[k]`, 1-based) to
/// the original matrix `A` to produce `P · A`. cuSOLVER's pivot is a
/// sequence of row swaps: at step `k`, swap rows `k-1` and `pivot[k]-1`.
fn apply_pivots_f32(a: &[f32], pivot: &[i32], n: usize) -> Vec<f32> {
    let mut pa = a.to_vec();
    // We need to permute rows in the column-major matrix. Apply the
    // sequence of pairwise swaps.
    for k in 0..n {
        let p = (pivot[k] - 1) as usize;
        if p != k {
            // Swap rows k and p across all columns.
            for j in 0..n {
                let a_idx = j * n + k;
                let b_idx = j * n + p;
                pa.swap(a_idx, b_idx);
            }
        }
    }
    pa
}

fn apply_pivots_f64(a: &[f64], pivot: &[i32], n: usize) -> Vec<f64> {
    let mut pa = a.to_vec();
    for k in 0..n {
        let p = (pivot[k] - 1) as usize;
        if p != k {
            for j in 0..n {
                let a_idx = j * n + k;
                let b_idx = j * n + p;
                pa.swap(a_idx, b_idx);
            }
        }
    }
    pa
}

#[test]
#[ignore]
fn lu_f32_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 5;
    let a_host = nonsingular_f32(n as usize, 0xC0FF_EE12);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_pivot: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, n as usize).expect("alloc pivot");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = LuDescriptor {
        m: n,
        n,
        batch_size: 1,
        element: ElementKind::F32,
    };
    let plan = LuPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select LuPlan<f32>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    assert!(ws_bytes > 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let a_shape = [1i32, n, n];
    let a_stride = contiguous_stride(a_shape);
    let pivot_shape = [1i32, n];
    let pivot_stride = contiguous_stride(pivot_shape);
    let args = LuArgs::<f32> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: a_stride,
        },
        pivot: TensorMut {
            data: dev_pivot.as_slice_mut(),
            shape: pivot_shape,
            stride: pivot_stride,
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run LU f32");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mut lu_host = vec![0f32; (n * n) as usize];
    dev_a.copy_to_host(&mut lu_host).expect("dl LU");
    let mut pivot_host = vec![0i32; n as usize];
    dev_pivot.copy_to_host(&mut pivot_host).expect("dl pivot");

    let n_usize = n as usize;
    let recon = reconstruct_lu_f32(&lu_host, n_usize);
    let pa = apply_pivots_f32(&a_host, &pivot_host, n_usize);

    let tol = 1e-4f32;
    for i in 0..n_usize {
        for j in 0..n_usize {
            let got = recon[j * n_usize + i];
            let expected = pa[j * n_usize + i];
            let diff = (got - expected).abs();
            let t = tol * expected.abs().max(1.0);
            assert!(
                diff <= t,
                "f32 LU reconstruct ({i},{j}): got={got}, expected={expected}, diff={diff}",
            );
        }
    }
}

#[test]
#[ignore]
fn lu_f64_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 5;
    let a_host = nonsingular_f64(n as usize, 0xDEAD_C0DE);

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload a");
    let mut dev_pivot: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, n as usize).expect("alloc pivot");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = LuDescriptor {
        m: n,
        n,
        batch_size: 1,
        element: ElementKind::F64,
    };
    let plan = LuPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select LuPlan<f64>");
    let ws_bytes = plan.query_workspace_size(&stream).expect("ws query");
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let a_shape = [1i32, n, n];
    let a_stride = contiguous_stride(a_shape);
    let pivot_shape = [1i32, n];
    let pivot_stride = contiguous_stride(pivot_shape);
    let args = LuArgs::<f64> {
        a: TensorMut {
            data: dev_a.as_slice_mut(),
            shape: a_shape,
            stride: a_stride,
        },
        pivot: TensorMut {
            data: dev_pivot.as_slice_mut(),
            shape: pivot_shape,
            stride: pivot_stride,
        },
        info: TensorMut {
            data: dev_info.as_slice_mut(),
            shape: [1i32],
            stride: [1],
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run LU f64");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0);

    let mut lu_host = vec![0f64; (n * n) as usize];
    dev_a.copy_to_host(&mut lu_host).expect("dl LU");
    let mut pivot_host = vec![0i32; n as usize];
    dev_pivot.copy_to_host(&mut pivot_host).expect("dl pivot");

    let n_usize = n as usize;
    let recon = reconstruct_lu_f64(&lu_host, n_usize);
    let pa = apply_pivots_f64(&a_host, &pivot_host, n_usize);

    let tol = 1e-10f64;
    for i in 0..n_usize {
        for j in 0..n_usize {
            let got = recon[j * n_usize + i];
            let expected = pa[j * n_usize + i];
            let diff = (got - expected).abs();
            let t = tol * expected.abs().max(1.0);
            assert!(
                diff <= t,
                "f64 LU reconstruct ({i},{j}): got={got}, expected={expected}, diff={diff}",
            );
        }
    }
}
