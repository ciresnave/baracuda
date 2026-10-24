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

const M: usize = 31;
const N: usize = 512;
const K: usize = 2048;

/// One f16 facade call on `ctx`; returns the facade status.
fn f16_call(ctx: &Context, stream: &Stream) -> i32 {
    let a = DeviceBuffer::from_slice(ctx, &vec![f16::from_f32(0.5); M * K]).expect("A");
    let b = DeviceBuffer::from_slice(ctx, &vec![f16::from_f32(0.25); K * N]).expect("B");
    let d: DeviceBuffer<f16> = DeviceBuffer::zeros(ctx, M * N).expect("D");
    // SAFETY: the three buffers are sized for an RRR M x K . K x N -> M x N problem and outlive
    // the caller's synchronize; `stream` belongs to `ctx`, which is current.
    unsafe {
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f16_run(
            M as i32,
            N as i32,
            K as i32,
            1,
            0,
            1.0,
            0.0,
            a.as_raw().0 as _,
            K as i64,
            0,
            b.as_raw().0 as _,
            N as i64,
            0,
            d.as_raw().0 as _,
            N as i64,
            0,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as _,
        )
    }
}

/// One f32 facade call on `ctx`; returns the facade status.
fn f32_call(ctx: &Context, stream: &Stream) -> i32 {
    let a = DeviceBuffer::from_slice(ctx, &vec![0.5f32; M * K]).expect("A");
    let b = DeviceBuffer::from_slice(ctx, &vec![0.25f32; K * N]).expect("B");
    let d: DeviceBuffer<f32> = DeviceBuffer::zeros(ctx, M * N).expect("D");
    // SAFETY: the three buffers are sized for an RRR M x K . K x N -> M x N problem and outlive
    // the caller's synchronize; `stream` belongs to `ctx`, which is current.
    unsafe {
        baracuda_kernels_sys::baracuda_kernels_gemm_dense_f32_run(
            M as i32,
            N as i32,
            K as i32,
            1,
            0,
            1.0,
            0.0,
            a.as_raw().0 as _,
            K as i64,
            0,
            b.as_raw().0 as _,
            N as i64,
            0,
            d.as_raw().0 as _,
            N as i64,
            0,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as _,
        )
    }
}

/// One f16 (even `i`) or f32 (odd `i`) facade call on a context made just for it; returns the
/// facade status, or -1 if the stream failed to synchronize.
fn call_on_fresh_context(dev: &Device, i: usize) -> i32 {
    let ctx = Context::new(dev).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let status = if i.is_multiple_of(2) {
        f16_call(&ctx, &stream)
    } else {
        f32_call(&ctx, &stream)
    };
    if status == 0 && stream.synchronize().is_err() {
        return -1;
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
