//! The dense GEMM facade pools cuBLAS handles per CUDA context. A caller that creates a fresh
//! context for every call (Fuel's `CudaInvoker`) gets each new context at the address the
//! previous, destroyed one had. The pool used to key on that address, so a handle bound to the
//! dead context was revived and every call after the first returned status 5. The pool now keys
//! on `cuCtxGetId`, which is never reused; this test fails without that.
//!
//! Run with `cargo gpu-test -p baracuda-kernels --features sm89,device-tests --test dense_gemm_fresh_context`.
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use half::f16;

const CALLS: usize = 40;

/// One f16 (even `i`) or f32 (odd `i`) facade call on a context made just for it; returns the
/// facade status, or -1 if the stream failed to synchronize.
fn call_on_fresh_context(dev: &Device, i: usize) -> i32 {
    let ctx = Context::new(dev).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let (m, n, k) = (31usize, 512usize, 2048usize);
    let mut status = if i.is_multiple_of(2) {
        let a = DeviceBuffer::from_slice(&ctx, &vec![f16::from_f32(0.5); m * k]).expect("A");
        let b = DeviceBuffer::from_slice(&ctx, &vec![f16::from_f32(0.25); k * n]).expect("B");
        let d: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, m * n).expect("D");
        // SAFETY: the three buffers are sized for an RRR m x k . k x n -> m x n problem and live
        // until after the synchronize below; `stream` belongs to the current context.
        unsafe {
            baracuda_kernels_sys::baracuda_kernels_gemm_dense_f16_run(
                m as i32,
                n as i32,
                k as i32,
                1,
                0,
                1.0,
                0.0,
                a.as_raw().0 as _,
                k as i64,
                0,
                b.as_raw().0 as _,
                n as i64,
                0,
                d.as_raw().0 as _,
                n as i64,
                0,
                core::ptr::null_mut(),
                0,
                stream.as_raw() as _,
            )
        }
    } else {
        let a = DeviceBuffer::from_slice(&ctx, &vec![0.5f32; m * k]).expect("A");
        let b = DeviceBuffer::from_slice(&ctx, &vec![0.25f32; k * n]).expect("B");
        let d: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, m * n).expect("D");
        // SAFETY: as above, for f32.
        unsafe {
            baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_run(
                m as i32,
                n as i32,
                k as i32,
                1,
                0,
                1.0,
                0.0,
                a.as_raw().0 as _,
                k as i64,
                0,
                b.as_raw().0 as _,
                n as i64,
                0,
                d.as_raw().0 as _,
                n as i64,
                0,
                core::ptr::null_mut(),
                0,
                stream.as_raw() as _,
            )
        }
    };
    if status == 0 && stream.synchronize().is_err() {
        status = -1;
    }
    status
}

#[test]
fn dense_gemm_survives_a_fresh_context_per_call() {
    init().expect("driver init");
    let dev = Device::get(0).expect("device 0");
    let statuses: Vec<i32> = (0..CALLS).map(|i| call_on_fresh_context(&dev, i)).collect();
    assert!(
        statuses.iter().all(|&s| s == 0),
        "facade statuses over {CALLS} fresh-context calls: {statuses:?}"
    );
}
