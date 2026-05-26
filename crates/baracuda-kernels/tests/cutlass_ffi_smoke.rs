//! Real-GPU smoke tests for the Phase 24 Cutlass GEMM FFI re-export
//! facade.
//!
//! Each Cutlass GEMM SKU family is now reachable through the unified
//! `baracuda_kernels_gemm_*` namespace in `baracuda-kernels-sys`. This
//! smoke covers one representative SKU from each major family so the
//! trampolines are exercised end-to-end:
//!
//! - Non-bias single GEMM, f32 alpha/beta — `f32_simt_rrr` (no fp16
//!   host conversions required; SIMT fallback is universally
//!   available on Ampere+).
//! - Bias-fused single GEMM, f32 alpha/beta — `bias_f32_simt_rrr`.
//! - Strided-batched GEMM, f32 alpha/beta — `batched_f16_rcr` (uses
//!   `half::f16` for activations).
//!
//! `#[ignore]` by default — requires a real CUDA device.

#![allow(unused_mut)]

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels_sys::{
    baracuda_kernels_gemm_batched_f16_rcr_sm80_run,
    baracuda_kernels_gemm_batched_f16_rcr_sm80_workspace_size,
    baracuda_kernels_gemm_bias_f32_simt_rrr_sm80_run,
    baracuda_kernels_gemm_bias_f32_simt_rrr_sm80_workspace_size,
    baracuda_kernels_gemm_f32_simt_rrr_sm80_run,
    baracuda_kernels_gemm_f32_simt_rrr_sm80_workspace_size,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// `f32_simt_rrr` non-bias GEMM. Tests the most universally-available
/// trampoline (SIMT works on every Cutlass-supported arch).
#[test]
#[ignore]
fn gemm_f32_simt_rrr_ffi() {
    let (ctx, stream) = setup();

    // Small fixed problem: D = A * B (alpha = 1, beta = 0).
    // A: [M, K] row-major
    // B: [K, N] row-major (RRR layout)
    // D: [M, N] row-major
    let m: i32 = 16;
    let n: i32 = 16;
    let k: i32 = 16;
    let lda: i64 = k as i64;
    let ldb: i64 = n as i64;
    let ldd: i64 = n as i64;

    // Identity-like A: 0..M*K with deterministic ramp.
    let mut a_host = vec![0f32; (m * k) as usize];
    for (i, v) in a_host.iter_mut().enumerate() {
        *v = (i as f32) * 0.01;
    }
    // B = identity-ish (so D ≈ A).
    let mut b_host = vec![0f32; (k * n) as usize];
    for i in 0..k {
        b_host[(i * n + i) as usize] = 1.0;
    }

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("up b");
    let mut dev_d: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc d");

    // Query workspace.
    let ws_bytes = unsafe {
        baracuda_kernels_gemm_f32_simt_rrr_sm80_workspace_size(m, n, k)
    };
    let mut ws_buf: Option<DeviceBuffer<u8>> = if ws_bytes > 0 {
        Some(DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws"))
    } else {
        None
    };
    let (ws_ptr, ws_bytes_arg) = match ws_buf.as_mut() {
        Some(b) => (b.as_raw().0 as *mut c_void, ws_bytes),
        None => (core::ptr::null_mut(), 0usize),
    };

    let status = unsafe {
        baracuda_kernels_gemm_f32_simt_rrr_sm80_run(
            m, n, k,
            dev_a.as_raw().0 as *const c_void, lda,
            dev_b.as_raw().0 as *const c_void, ldb,
            core::ptr::null(), 0,
            dev_d.as_raw().0 as *mut c_void, ldd,
            1.0_f32, 0.0_f32,
            ws_ptr, ws_bytes_arg,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "gemm_f32_simt_rrr status");
    stream.synchronize().expect("sync");

    // Check: D == A (since B is identity).
    let mut d_host = vec![0f32; (m * n) as usize];
    dev_d.copy_to_host(&mut d_host).expect("dl d");

    for i in 0..(m as usize) {
        for j in 0..(n as usize) {
            let want = a_host[i * (k as usize) + j];
            let got = d_host[i * (n as usize) + j];
            assert!((want - got).abs() < 1e-5, "i={} j={} want={} got={}", i, j, want, got);
        }
    }
}

/// `bias_f32_simt_rrr` bias-fused GEMM. Adds a per-column broadcast
/// bias on top of the base GEMM result.
#[test]
#[ignore]
fn gemm_bias_f32_simt_rrr_ffi() {
    let (ctx, stream) = setup();

    let m: i32 = 8;
    let n: i32 = 8;
    let k: i32 = 8;

    // A = identity row-major, B = identity row-major, so A*B = identity.
    let mut a_host = vec![0f32; (m * k) as usize];
    for i in 0..m {
        a_host[(i * k + i) as usize] = 1.0;
    }
    let mut b_host = vec![0f32; (k * n) as usize];
    for i in 0..k {
        b_host[(i * n + i) as usize] = 1.0;
    }
    // Bias of length N — broadcast across all rows.
    let bias_host: Vec<f32> = (0..n).map(|i| (i as f32) + 0.5).collect();

    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("up b");
    let mut dev_bias = DeviceBuffer::from_slice(&ctx, &bias_host).expect("up bias");
    let mut dev_d: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (m * n) as usize).expect("alloc d");

    let ws_bytes = unsafe {
        baracuda_kernels_gemm_bias_f32_simt_rrr_sm80_workspace_size(m, n, k)
    };
    let mut ws_buf: Option<DeviceBuffer<u8>> = if ws_bytes > 0 {
        Some(DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws"))
    } else {
        None
    };
    let (ws_ptr, ws_bytes_arg) = match ws_buf.as_mut() {
        Some(b) => (b.as_raw().0 as *mut c_void, ws_bytes),
        None => (core::ptr::null_mut(), 0usize),
    };

    let status = unsafe {
        baracuda_kernels_gemm_bias_f32_simt_rrr_sm80_run(
            m, n, k,
            dev_a.as_raw().0 as *const c_void, k as i64,
            dev_b.as_raw().0 as *const c_void, n as i64,
            core::ptr::null(), 0,
            dev_d.as_raw().0 as *mut c_void, n as i64,
            dev_bias.as_raw().0 as *const c_void,
            1.0_f32, 0.0_f32,
            ws_ptr, ws_bytes_arg,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "gemm_bias_f32_simt_rrr status");
    stream.synchronize().expect("sync");

    // Expected: D[i,j] = I[i,j] + bias[j] (broadcast).
    let mut d_host = vec![0f32; (m * n) as usize];
    dev_d.copy_to_host(&mut d_host).expect("dl d");
    for i in 0..(m as usize) {
        for j in 0..(n as usize) {
            let want = (if i == j { 1.0 } else { 0.0 }) + bias_host[j];
            let got = d_host[i * (n as usize) + j];
            assert!((want - got).abs() < 1e-5, "i={} j={} want={} got={}", i, j, want, got);
        }
    }
}

/// `batched_f16_rcr` strided-batched GEMM. Two batches, equal stride.
#[test]
#[ignore]
fn gemm_batched_f16_rcr_ffi() {
    let (ctx, stream) = setup();

    let m: i32 = 16;
    let n: i32 = 16;
    let k: i32 = 16;
    let batch_count: i32 = 2;
    let lda: i64 = k as i64;        // row-major [M, K]
    let ldb: i64 = k as i64;        // column-major [K, N] → ld is the K dim
    let ldd: i64 = n as i64;        // row-major [M, N]
    let stride_a = (m * k) as i64;
    let stride_b = (k * n) as i64;
    let stride_d = (m * n) as i64;

    // Simple batch-0 = zero, batch-1 = identity-like A; B = identity for
    // both. We just want a non-error launch + a recognizable zero/identity
    // round-trip.
    let mut a_host = vec![f16::ZERO; (batch_count * m * k) as usize];
    // batch 1 starts at stride_a; fill its diagonal with 1.
    for i in 0..m.min(k) {
        let off = (stride_a + (i * k + i) as i64) as usize;
        a_host[off] = f16::from_f32(1.0);
    }
    // B: identity for both batches.
    let mut b_host = vec![f16::ZERO; (batch_count * k * n) as usize];
    for b_idx in 0..batch_count {
        for i in 0..k.min(n) {
            // RCR: B is column-major [K, N], so element (i, i) at
            // offset b*stride + i*ld + i = b*stride + i*K + i.
            let off = (b_idx as i64 * stride_b + (i as i64) * ldb + i as i64) as usize;
            b_host[off] = f16::from_f32(1.0);
        }
    }
    let mut dev_a = DeviceBuffer::from_slice(&ctx, &a_host).expect("up a");
    let mut dev_b = DeviceBuffer::from_slice(&ctx, &b_host).expect("up b");
    let mut dev_d: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (batch_count * m * n) as usize).expect("alloc d");

    let ws_bytes = unsafe {
        baracuda_kernels_gemm_batched_f16_rcr_sm80_workspace_size(m, n, k, batch_count)
    };
    let mut ws_buf: Option<DeviceBuffer<u8>> = if ws_bytes > 0 {
        Some(DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws"))
    } else {
        None
    };
    let (ws_ptr, ws_bytes_arg) = match ws_buf.as_mut() {
        Some(b) => (b.as_raw().0 as *mut c_void, ws_bytes),
        None => (core::ptr::null_mut(), 0usize),
    };

    let status = unsafe {
        baracuda_kernels_gemm_batched_f16_rcr_sm80_run(
            m, n, k,
            dev_a.as_raw().0 as *const c_void, lda, stride_a,
            dev_b.as_raw().0 as *const c_void, ldb, stride_b,
            core::ptr::null(), 0, 0,
            dev_d.as_raw().0 as *mut c_void, ldd, stride_d,
            1.0_f32, 0.0_f32,
            batch_count,
            ws_ptr, ws_bytes_arg,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "gemm_batched_f16_rcr status");
    stream.synchronize().expect("sync");

    let mut d_host = vec![f16::ZERO; (batch_count * m * n) as usize];
    dev_d.copy_to_host(&mut d_host).expect("dl d");

    // Batch 0: D should be all zeros (A was zero).
    for i in 0..(m * n) as usize {
        assert_eq!(d_host[i].to_f32(), 0.0, "batch0 nonzero at {}", i);
    }
    // Batch 1: D should be identity (A = I, B = I).
    for i in 0..(m as usize) {
        for j in 0..(n as usize) {
            let want = if i == j { 1.0 } else { 0.0 };
            let got = d_host[(stride_d as usize) + i * (n as usize) + j].to_f32();
            assert!((want - got).abs() < 1e-3, "batch1 i={} j={} want={} got={}", i, j, want, got);
        }
    }
}
