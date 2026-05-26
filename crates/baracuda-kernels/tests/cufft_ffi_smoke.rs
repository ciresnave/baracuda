//! Real-GPU smoke tests for the Phase 23 cuFFT FFI facade.
//!
//! Each cuFFT plan family gets at least one FFI symbol smoke test that
//! verifies the symbol is callable, executes successfully on a valid
//! input, and (where applicable) round-trips to identity via the
//! matching inverse.
//!
//! Coverage matches Phase 23 deliverables:
//! - `fft_1d` × c32 — forward + inverse round-trip
//! - `rfft_1d` × f32 + `irfft_1d` × f32 — R2C / C2R round-trip
//! - `fft_nd` × c32 — 2-D forward + inverse round-trip
//! - `rfft_nd` × f32 + `irfft_nd` × f32 — 2-D R2C / C2R round-trip
//!
//! `#[ignore]` by default — requires a real CUDA device.

#![allow(unused_mut)]

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels_sys::{
    baracuda_kernels_fft_1d_c32_run, baracuda_kernels_fft_1d_c32_workspace_size,
    baracuda_kernels_fft_nd_c32_run, baracuda_kernels_fft_nd_c32_workspace_size,
    baracuda_kernels_irfft_1d_f32_run, baracuda_kernels_irfft_1d_f32_workspace_size,
    baracuda_kernels_irfft_nd_f32_run, baracuda_kernels_irfft_nd_f32_workspace_size,
    baracuda_kernels_rfft_1d_f32_run, baracuda_kernels_rfft_1d_f32_workspace_size,
    baracuda_kernels_rfft_nd_f32_run, baracuda_kernels_rfft_nd_f32_workspace_size,
};

// Use the workspace `Complex32` so DeviceBuffer<Complex32> trait bounds
// pass without locally re-declaring a custom DeviceRepr type.
use baracuda_types::Complex32 as C32;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn fft_1d_c32_roundtrip_ffi() {
    let (ctx, stream) = setup();
    let n: i32 = 32;
    let batch: i32 = 2;
    let numel = (n as usize) * (batch as usize);

    // Deterministic input: ramp.
    let mut x_host = vec![C32::default(); numel];
    for (i, c) in x_host.iter_mut().enumerate() {
        c.re = (i as f32) * 0.125;
        c.im = ((i as f32) * 0.0625).sin();
    }

    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("up x");
    let mut dev_y: DeviceBuffer<C32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let mut dev_back: DeviceBuffer<C32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc back");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_fft_1d_c32_workspace_size(n, batch, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0, "fft_1d_c32 ws status");
    assert_eq!(ws_bytes, 0, "fft_1d_c32 ws is zero per contract");

    // Forward.
    let status = unsafe {
        baracuda_kernels_fft_1d_c32_run(
            n,
            batch,
            0,
            dev_x.as_raw().0 as *mut c_void,
            dev_y.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "fft_1d_c32 forward status");

    // Inverse (with 1/n normalization).
    let status = unsafe {
        baracuda_kernels_fft_1d_c32_run(
            n,
            batch,
            1,
            dev_y.as_raw().0 as *mut c_void,
            dev_back.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "fft_1d_c32 inverse status");
    stream.synchronize().expect("sync");

    let mut back_host = vec![C32::default(); numel];
    dev_back.copy_to_host(&mut back_host).expect("dl back");

    for (a, b) in x_host.iter().zip(back_host.iter()) {
        assert!((a.re - b.re).abs() < 1e-4, "re mismatch: {:?} vs {:?}", a, b);
        assert!((a.im - b.im).abs() < 1e-4, "im mismatch: {:?} vs {:?}", a, b);
    }
}

#[test]
#[ignore]
fn rfft_irfft_1d_f32_roundtrip_ffi() {
    let (ctx, stream) = setup();
    let n: i32 = 64;
    let batch: i32 = 1;
    let n_real = (n as usize) * (batch as usize);
    let n_complex = (n as usize / 2 + 1) * (batch as usize);

    let mut x_host = vec![0f32; n_real];
    for (i, v) in x_host.iter_mut().enumerate() {
        *v = ((i as f32) * 0.1).sin() + 0.25 * (i as f32) / (n_real as f32);
    }

    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("up x");
    let mut dev_y: DeviceBuffer<C32> = DeviceBuffer::zeros(&ctx, n_complex).expect("alloc y");
    let mut dev_back: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_real).expect("alloc back");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_rfft_1d_f32_workspace_size(n, batch, &mut ws_bytes as *mut _)
    };
    assert_eq!(status, 0);
    assert_eq!(ws_bytes, 0);

    // R2C forward.
    let status = unsafe {
        baracuda_kernels_rfft_1d_f32_run(
            n,
            batch,
            dev_x.as_raw().0 as *mut c_void,
            dev_y.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "rfft_1d_f32 status");

    let mut ws_bytes2: usize = 0;
    let status = unsafe {
        baracuda_kernels_irfft_1d_f32_workspace_size(n, batch, &mut ws_bytes2 as *mut _)
    };
    assert_eq!(status, 0);

    // C2R inverse with 1/n normalization.
    let status = unsafe {
        baracuda_kernels_irfft_1d_f32_run(
            n,
            batch,
            dev_y.as_raw().0 as *mut c_void,
            dev_back.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "irfft_1d_f32 status");
    stream.synchronize().expect("sync");

    let mut back_host = vec![0f32; n_real];
    dev_back.copy_to_host(&mut back_host).expect("dl back");

    for (a, b) in x_host.iter().zip(back_host.iter()) {
        assert!((a - b).abs() < 1e-4, "round-trip mismatch: {} vs {}", a, b);
    }
}

#[test]
#[ignore]
fn fft_nd_c32_2d_roundtrip_ffi() {
    let (ctx, stream) = setup();
    let dims_host: [i32; 2] = [8, 16];
    let rank: i32 = 2;
    let batch: i32 = 2;
    let per_transform = (dims_host[0] as usize) * (dims_host[1] as usize);
    let numel = per_transform * (batch as usize);

    let mut x_host = vec![C32::default(); numel];
    for (i, c) in x_host.iter_mut().enumerate() {
        c.re = ((i % 11) as f32) * 0.1;
        c.im = ((i % 7) as f32) * 0.07;
    }

    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("up x");
    let mut dev_y: DeviceBuffer<C32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let mut dev_back: DeviceBuffer<C32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc back");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_fft_nd_c32_workspace_size(
            rank,
            dims_host.as_ptr(),
            batch,
            &mut ws_bytes as *mut _,
        )
    };
    assert_eq!(status, 0);

    // Forward.
    let status = unsafe {
        baracuda_kernels_fft_nd_c32_run(
            rank,
            dims_host.as_ptr(),
            batch,
            0,
            dev_x.as_raw().0 as *mut c_void,
            dev_y.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "fft_nd_c32 forward status");

    // Inverse — normalized by 1 / (8 * 16).
    let status = unsafe {
        baracuda_kernels_fft_nd_c32_run(
            rank,
            dims_host.as_ptr(),
            batch,
            1,
            dev_y.as_raw().0 as *mut c_void,
            dev_back.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "fft_nd_c32 inverse status");
    stream.synchronize().expect("sync");

    let mut back_host = vec![C32::default(); numel];
    dev_back.copy_to_host(&mut back_host).expect("dl back");

    for (a, b) in x_host.iter().zip(back_host.iter()) {
        assert!((a.re - b.re).abs() < 1e-4, "re mismatch: {:?} vs {:?}", a, b);
        assert!((a.im - b.im).abs() < 1e-4, "im mismatch: {:?} vs {:?}", a, b);
    }
}

#[test]
#[ignore]
fn rfft_irfft_nd_f32_2d_roundtrip_ffi() {
    let (ctx, stream) = setup();
    let dims_host: [i32; 2] = [4, 8];
    let rank: i32 = 2;
    let batch: i32 = 1;
    let real_per = (dims_host[0] as usize) * (dims_host[1] as usize);
    let complex_per = (dims_host[0] as usize) * (dims_host[1] as usize / 2 + 1);
    let n_real = real_per * (batch as usize);
    let n_complex = complex_per * (batch as usize);

    let mut x_host = vec![0f32; n_real];
    for (i, v) in x_host.iter_mut().enumerate() {
        *v = ((i as f32) * 0.3).cos() + 0.1 * (i as f32);
    }

    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("up x");
    let mut dev_y: DeviceBuffer<C32> = DeviceBuffer::zeros(&ctx, n_complex).expect("alloc y");
    let mut dev_back: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_real).expect("alloc back");

    let mut ws_bytes: usize = 0;
    let status = unsafe {
        baracuda_kernels_rfft_nd_f32_workspace_size(
            rank,
            dims_host.as_ptr(),
            batch,
            &mut ws_bytes as *mut _,
        )
    };
    assert_eq!(status, 0);

    // R2C forward.
    let status = unsafe {
        baracuda_kernels_rfft_nd_f32_run(
            rank,
            dims_host.as_ptr(),
            batch,
            dev_x.as_raw().0 as *mut c_void,
            dev_y.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "rfft_nd_f32 status");

    // C2R inverse with 1/(4*8) normalization.
    let status = unsafe {
        baracuda_kernels_irfft_nd_f32_run(
            rank,
            dims_host.as_ptr(),
            batch,
            dev_y.as_raw().0 as *mut c_void,
            dev_back.as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "irfft_nd_f32 status");
    stream.synchronize().expect("sync");

    let mut back_host = vec![0f32; n_real];
    dev_back.copy_to_host(&mut back_host).expect("dl back");

    for (a, b) in x_host.iter().zip(back_host.iter()) {
        assert!((a - b).abs() < 1e-3, "round-trip mismatch: {} vs {}", a, b);
    }
}
