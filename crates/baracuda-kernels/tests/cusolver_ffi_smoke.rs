//! Real-GPU smoke tests for the Phase 22 cuSOLVER FFI facade.
//!
//! Each plan family gets at least one FFI symbol smoke test that
//! verifies the symbol is callable, runs successfully on a valid
//! input, and reports `info == 0` (or the analogous success flag).
//! We don't cross-check against the Rust plan-layer output for every
//! symbol — the FFI wrapper is a near-1:1 forwarder of the same
//! cuSOLVER entry points the Rust plan calls, so a status-code +
//! info check is sufficient to catch real wiring bugs (wrong dtype
//! cast, swapped arg order, missing workspace query, etc.).
//!
//! Coverage matches the deliverables of Phase 22:
//! - Cholesky (f32 + c-free)
//! - LU (f32)
//! - QR (f32)
//! - SVD (f32)
//! - SVD-batched Jacobi (f32)
//! - SVDA-batched (f32)
//! - Eigh (f32 symmetric + c32 Hermitian — complex variant)
//! - Eig (f32 — packed wr/wi)
//! - LstSq (f32)
//! - Solve (f32)
//! - Inverse (f32)
//! - ormqr (f32) — composes with QR for dense-Q materialization.
//!
//! `#[ignore]` by default — requires a real CUDA device.

// Many `dev_*` buffers are written by the FFI side (cuSOLVER mutates
// them through raw pointers), but the Rust side only reads `.as_raw()`
// — the borrow checker can't see the write-through, so it warns about
// unused-mut. Silence project-wide since every test exhibits this.
#![allow(unused_mut)]

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels_sys::{
    baracuda_kernels_cholesky_f32_run, baracuda_kernels_cholesky_f32_workspace_size,
    baracuda_kernels_eig_run, baracuda_kernels_eig_workspace_size,
    baracuda_kernels_eigh_c32_run, baracuda_kernels_eigh_c32_workspace_size,
    baracuda_kernels_eigh_f32_run, baracuda_kernels_eigh_f32_workspace_size,
    baracuda_kernels_inverse_f32_run, baracuda_kernels_inverse_f32_workspace_size,
    baracuda_kernels_lstsq_f32_run, baracuda_kernels_lstsq_f32_workspace_size,
    baracuda_kernels_lu_f32_run, baracuda_kernels_lu_f32_workspace_size,
    baracuda_kernels_ormqr_f32_run, baracuda_kernels_qr_f32_run,
    baracuda_kernels_qr_f32_workspace_size, baracuda_kernels_solve_f32_run,
    baracuda_kernels_solve_f32_workspace_size, baracuda_kernels_svd_batched_f32_run,
    baracuda_kernels_svd_batched_f32_workspace_size, baracuda_kernels_svd_f32_run,
    baracuda_kernels_svd_f32_workspace_size, baracuda_kernels_svda_batched_f32_run,
    baracuda_kernels_svda_batched_f32_workspace_size, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
    CUBLAS_SIDE_LEFT, CUDA_R_32F, CUSOLVER_EIG_MODE_VECTOR,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Build a deterministic SPD matrix `A = M·M^T + n·I` (column-major,
/// which equals row-major for the symmetric output).
fn spd_matrix_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut m = vec![0f32; n * n];
    let mut s = seed.wrapping_mul(0x9E37_79B1);
    for v in m.iter_mut() {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let f = ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
        *v = f;
    }
    let mut a = vec![0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0f32;
            for kk in 0..n {
                acc += m[i * n + kk] * m[j * n + kk];
            }
            a[i * n + j] = acc;
        }
        a[i * n + i] += n as f32;
    }
    a
}

/// Deterministic non-singular column-major matrix `A = I + 0.3 * R`
/// (R pseudo-random `[-0.5, 0.5]`).
fn nonsingular_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut a = vec![0f32; n * n];
    let mut s = seed.wrapping_mul(0x9E37_79B1);
    for j in 0..n {
        for i in 0..n {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let f = ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
            a[j * n + i] = 0.3 * f + if i == j { 1.0 } else { 0.0 };
        }
    }
    a
}

/// Deterministic tall matrix `[M, N]` column-major with `M >= N`.
fn tall_matrix_f32(m: usize, n: usize, seed: u32) -> Vec<f32> {
    let mut a = vec![0f32; m * n];
    let mut s = seed.wrapping_mul(0x9E37_79B1);
    for v in a.iter_mut() {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
        let f = ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5;
        *v = f;
    }
    // Bias the diagonal to keep it well-conditioned.
    for k in 0..n.min(m) {
        a[k * m + k] += 2.0;
    }
    a
}

#[test]
#[ignore]
fn cholesky_f32_ffi() {
    let (ctx, stream) = setup();
    let n: i32 = 6;
    let a_host = spd_matrix_f32(n as usize, 0xC0DE_F00D);
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_cholesky_f32_workspace_size(n, n, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0, "cholesky_f32_workspace_size status");
    assert!(ws_bytes > 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let status = unsafe {
        baracuda_kernels_cholesky_f32_run(
            CUBLAS_FILL_MODE_UPPER,
            n,
            n,
            dev_a.as_raw().0 as *mut c_void,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "cholesky_f32_run status");
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "cholesky info != 0 (SPD failure)");
}

#[test]
#[ignore]
fn lu_f32_ffi() {
    let (ctx, stream) = setup();
    let n: i32 = 5;
    let a_host = nonsingular_f32(n as usize, 0x42);
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_pivot: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, n as usize).expect("alloc pivot");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_lu_f32_workspace_size(n, n, n, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let status = unsafe {
        baracuda_kernels_lu_f32_run(
            n,
            n,
            n,
            dev_a.as_raw().0 as *mut c_void,
            dev_pivot.as_raw().0 as *mut i32,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "lu info != 0");
}

#[test]
#[ignore]
fn qr_f32_ffi() {
    let (ctx, stream) = setup();
    let (m, n) = (6i32, 4i32);
    let a_host = tall_matrix_f32(m as usize, n as usize, 0x77);
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_tau: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, m.min(n) as usize).expect("tau");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("info");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_qr_f32_workspace_size(m, n, m, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let status = unsafe {
        baracuda_kernels_qr_f32_run(
            m,
            n,
            m,
            dev_a.as_raw().0 as *mut c_void,
            dev_tau.as_raw().0 as *mut c_void,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "qr info != 0");
}

#[test]
#[ignore]
fn ormqr_f32_ffi() {
    // Compose qr + ormqr to materialize dense Q over a caller-staged
    // identity matrix.
    let (ctx, stream) = setup();
    let (m, n) = (5i32, 5i32);
    let k = m.min(n);
    let a_host = tall_matrix_f32(m as usize, n as usize, 0xAB);
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_tau: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, k as usize).expect("tau");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("info");

    // Identity matrix staged in the Q output.
    let mu = m as usize;
    let mut id_host = vec![0f32; mu * mu];
    for i in 0..mu {
        id_host[i * mu + i] = 1.0;
    }
    let mut dev_q = DeviceBuffer::from_slice(&ctx, &id_host).expect("stage identity");

    // QR workspace (geqrf-only — ormqr workspace will be sized separately).
    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_qr_f32_workspace_size(m, n, m, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    // 1. geqrf
    let status = unsafe {
        baracuda_kernels_qr_f32_run(
            m,
            n,
            m,
            dev_a.as_raw().0 as *mut c_void,
            dev_tau.as_raw().0 as *mut c_void,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);

    // 2. ormqr — left, op=N, K reflectors. Pre-staged identity Q gets
    //    overwritten with dense Q. ormqr internally allocates its own
    //    workspace; we re-use the QR workspace allocation which is at
    //    least as large for these small dims.
    let status = unsafe {
        baracuda_kernels_ormqr_f32_run(
            CUBLAS_SIDE_LEFT,
            CUBLAS_OP_N,
            m,
            m,
            k,
            dev_a.as_raw().0 as *const c_void,
            m,
            dev_tau.as_raw().0 as *const c_void,
            dev_q.as_raw().0 as *mut c_void,
            m,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "ormqr info != 0");
}

#[test]
#[ignore]
fn svd_f32_ffi() {
    let (ctx, stream) = setup();
    let (m, n) = (5i32, 3i32);
    let k = m.min(n);
    let a_host = tall_matrix_f32(m as usize, n as usize, 0xBE);
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_s: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, k as usize).expect("s");
    let mut dev_u: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * m) as usize).expect("u");
    let mut dev_vt: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * n) as usize).expect("vt");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("info");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_svd_f32_workspace_size(m, n, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let status = unsafe {
        baracuda_kernels_svd_f32_run(
            b'A',
            b'A',
            m,
            n,
            m,
            dev_a.as_raw().0 as *mut c_void,
            m,
            n,
            dev_s.as_raw().0 as *mut c_void,
            dev_u.as_raw().0 as *mut c_void,
            dev_vt.as_raw().0 as *mut c_void,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "svd info != 0");
}

#[test]
#[ignore]
fn svd_batched_f32_ffi() {
    let (ctx, stream) = setup();
    let n: i32 = 4;
    let b: i32 = 2;
    let bu = b as usize;
    let nu = n as usize;
    // Two SPD slots so the Jacobi-batched run converges deterministically.
    let mut a_host = vec![0f32; bu * nu * nu];
    for slot in 0..bu {
        let sub = spd_matrix_f32(nu, 0xABCD ^ slot as u32);
        a_host[slot * nu * nu..(slot + 1) * nu * nu].copy_from_slice(&sub);
    }
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_s: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * n) as usize).expect("s");
    let mut dev_u: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * n * n) as usize).expect("u");
    let mut dev_v: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * n * n) as usize).expect("v");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, bu).expect("info");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_svd_batched_f32_workspace_size(
            CUSOLVER_EIG_MODE_VECTOR,
            n,
            b,
            &mut ws_bytes as *mut _,
        )
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let status = unsafe {
        baracuda_kernels_svd_batched_f32_run(
            CUSOLVER_EIG_MODE_VECTOR,
            n,
            n,
            n,
            n,
            dev_a.as_raw().0 as *mut c_void,
            dev_s.as_raw().0 as *mut c_void,
            dev_u.as_raw().0 as *mut c_void,
            dev_v.as_raw().0 as *mut c_void,
            dev_info.as_raw().0 as *mut i32,
            b,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; bu];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    for (i, &v) in info_host.iter().enumerate() {
        assert_eq!(v, 0, "svd-batched slot {} info != 0", i);
    }
}

#[test]
#[ignore]
fn svda_batched_f32_ffi() {
    let (ctx, stream) = setup();
    let (m, n) = (4i32, 3i32);
    let rank = n; // full thin SVD per slot
    let b: i32 = 2;
    let bu = b as usize;
    let mut a_host = vec![0f32; (b * m * n) as usize];
    for slot in 0..bu {
        let sub = tall_matrix_f32(m as usize, n as usize, 0x1234 ^ slot as u32);
        let base = slot * (m * n) as usize;
        a_host[base..base + sub.len()].copy_from_slice(&sub);
    }
    let dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_s: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * rank) as usize).expect("s");
    let mut dev_u: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * m * rank) as usize).expect("u");
    let mut dev_v: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (b * n * rank) as usize).expect("v");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, bu).expect("info");
    let mut h_r_nrm_f = vec![0f64; bu];

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_svda_batched_f32_workspace_size(
            CUSOLVER_EIG_MODE_VECTOR,
            rank,
            m,
            n,
            b,
            &mut ws_bytes as *mut _,
        )
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let stride_a = (m as i64) * (n as i64);
    let stride_s = rank as i64;
    let stride_u = (m as i64) * (rank as i64);
    let stride_v = (n as i64) * (rank as i64);

    let status = unsafe {
        baracuda_kernels_svda_batched_f32_run(
            CUSOLVER_EIG_MODE_VECTOR,
            rank,
            m,
            n,
            m,
            m,
            n,
            stride_a,
            stride_s,
            stride_u,
            stride_v,
            dev_a.as_raw().0 as *const c_void,
            dev_s.as_raw().0 as *mut c_void,
            dev_u.as_raw().0 as *mut c_void,
            dev_v.as_raw().0 as *mut c_void,
            dev_info.as_raw().0 as *mut i32,
            h_r_nrm_f.as_mut_ptr(),
            b,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; bu];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    for (i, &v) in info_host.iter().enumerate() {
        assert_eq!(v, 0, "svda-batched slot {} info != 0", i);
    }
}

#[test]
#[ignore]
fn eigh_f32_ffi() {
    let (ctx, stream) = setup();
    let n: i32 = 4;
    let a_host = spd_matrix_f32(n as usize, 0xEEEE);
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_w: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n as usize).expect("w");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("info");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_eigh_f32_workspace_size(
            CUBLAS_FILL_MODE_UPPER,
            n,
            &mut ws_bytes as *mut _,
        )
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let status = unsafe {
        baracuda_kernels_eigh_f32_run(
            CUBLAS_FILL_MODE_UPPER,
            n,
            n,
            dev_a.as_raw().0 as *mut c_void,
            dev_w.as_raw().0 as *mut c_void,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "eigh info != 0");
}

#[test]
#[ignore]
fn eigh_c32_ffi() {
    // Hermitian eigh on a complex-Hermitian input (built from a real
    // SPD matrix lifted into the real component; imag = 0 keeps the
    // matrix Hermitian trivially). Caller-side complex storage is
    // `[re_0, im_0, re_1, im_1, ...]` matching `cuComplex` layout.
    let (ctx, stream) = setup();
    let n: i32 = 4;
    let nu = n as usize;
    let real = spd_matrix_f32(nu, 0xFADE);
    let mut a_host = vec![0f32; nu * nu * 2];
    for i in 0..nu * nu {
        a_host[2 * i] = real[i];
        // imag stays 0 — Hermitian with zero imag is just symmetric real.
    }
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_w: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, nu).expect("w");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("info");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_eigh_c32_workspace_size(
            CUBLAS_FILL_MODE_UPPER,
            n,
            &mut ws_bytes as *mut _,
        )
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let status = unsafe {
        baracuda_kernels_eigh_c32_run(
            CUBLAS_FILL_MODE_UPPER,
            n,
            n,
            dev_a.as_raw().0 as *mut c_void,
            dev_w.as_raw().0 as *mut c_void,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "eigh-c32 info != 0");
}

#[test]
#[ignore]
fn eig_f32_ffi() {
    let (ctx, stream) = setup();
    let n: i64 = 4;
    let nu = n as usize;
    let a_host = nonsingular_f32(nu, 0xBABE);
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    // W has shape [2 * N] for real input (LAPACK packed wr/wi convention).
    let mut dev_w: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 2 * nu).expect("w");
    let mut dev_vr: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, nu * nu).expect("vr");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("info");

    let mut ws_dev: usize = 0;
    let mut ws_host: usize = 0;
    let status = unsafe {
        baracuda_kernels_eig_workspace_size(
            CUDA_R_32F,
            0, // NOVECTOR for VL
            CUSOLVER_EIG_MODE_VECTOR,
            n,
            &mut ws_dev as *mut _,
            &mut ws_host as *mut _,
        )
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_dev.max(1)).expect("ws_dev");
    let mut host_ws_buf: Vec<u8> = vec![0u8; ws_host.max(1)];

    let status = unsafe {
        baracuda_kernels_eig_run(
            CUDA_R_32F,
            0,
            CUSOLVER_EIG_MODE_VECTOR,
            n,
            n,
            0,
            n,
            dev_a.as_raw().0 as *mut c_void,
            dev_w.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            dev_vr.as_raw().0 as *mut c_void,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_dev,
            host_ws_buf.as_mut_ptr() as *mut c_void,
            ws_host,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "eig info != 0");
}

#[test]
#[ignore]
fn lstsq_f32_ffi() {
    let (ctx, stream) = setup();
    let (m, n, nrhs) = (6i32, 4i32, 2i32);
    let a_host = tall_matrix_f32(m as usize, n as usize, 0x55);
    let b_host: Vec<f32> = (0..(m * nrhs) as usize)
        .map(|i| ((i as f32) * 0.13).sin())
        .collect();
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("up b");
    let mut dev_x: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * nrhs) as usize).expect("x");
    // `niters` is a HOST scalar — cuSOLVER's `_gels` writes it from
    // the host-side iterative-refinement loop, not from a kernel.
    // Passing a device pointer here triggers STATUS_ACCESS_VIOLATION.
    let mut niters_host: i32 = 0;
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("info");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_lstsq_f32_workspace_size(m, n, nrhs, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes.max(1)).expect("ws");

    let status = unsafe {
        baracuda_kernels_lstsq_f32_run(
            m,
            n,
            nrhs,
            m,
            m,
            n,
            dev_a.as_raw().0 as *mut c_void,
            dev_b.as_raw().0 as *mut c_void,
            dev_x.as_raw().0 as *mut c_void,
            &mut niters_host as *mut i32,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "lstsq info != 0");
}

#[test]
#[ignore]
fn solve_f32_ffi() {
    let (ctx, stream) = setup();
    let (n, nrhs) = (4i32, 2i32);
    let a_host = nonsingular_f32(n as usize, 0x33);
    let b_host: Vec<f32> = (0..(n * nrhs) as usize)
        .map(|i| 0.5 + i as f32 * 0.1)
        .collect();
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("up b");
    let mut dev_pivot: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, n as usize).expect("pivot");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("info");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_solve_f32_workspace_size(n, n, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let status = unsafe {
        baracuda_kernels_solve_f32_run(
            n,
            nrhs,
            n,
            n,
            dev_a.as_raw().0 as *mut c_void,
            dev_pivot.as_raw().0 as *mut i32,
            dev_b.as_raw().0 as *mut c_void,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "solve info != 0");
}

#[test]
#[ignore]
fn inverse_f32_ffi() {
    let (ctx, stream) = setup();
    let n: i32 = 4;
    let nu = n as usize;
    let a_host = nonsingular_f32(nu, 0x77);
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    // Stage identity in inv (column-major == row-major for I).
    let mut id_host = vec![0f32; nu * nu];
    for i in 0..nu {
        id_host[i * nu + i] = 1.0;
    }
    let mut dev_inv = DeviceBuffer::from_slice(&ctx, &id_host).expect("stage I");
    let mut dev_pivot: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, nu).expect("pivot");
    let mut dev_info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("info");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_inverse_f32_workspace_size(n, n, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);
    let mut dev_ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws");

    let status = unsafe {
        baracuda_kernels_inverse_f32_run(
            n,
            n,
            n,
            dev_a.as_raw().0 as *mut c_void,
            dev_pivot.as_raw().0 as *mut i32,
            dev_inv.as_raw().0 as *mut c_void,
            dev_info.as_raw().0 as *mut i32,
            dev_ws.as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut info_host = vec![0i32; 1];
    dev_info.copy_to_host(&mut info_host).expect("dl info");
    assert_eq!(info_host[0], 0, "inverse info != 0");
}
