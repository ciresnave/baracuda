//! Real-GPU smoke test for Phase 39 gather extras — `u8` index dtype.
//!
//! The `u8` index variant is FFI-only today (`IndexElement` sealed
//! trait extension deferred), so this test calls the
//! `baracuda_kernels_gather_u8idx_*_run` C symbols directly with raw
//! pointer plumbing. `i64` idx for gather already had coverage in
//! `gather_smoke.rs`.
//!
//! `#[ignore]` by default.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Mirrors the kernel's coord layout: out[i,j] = src[i, idx[i,j]] along
/// `gather_dim=1`. Index dtype is u8.
#[test]
#[ignore]
fn gather_u8idx_f32_2d_dim1() {
    let (ctx, stream) = setup();
    let out_shape = [3i32, 4]; // index shape == out shape
    let src_shape = [3i32, 6]; // src extent on dim=1 is 6 (fits u8)
    let out_numel: usize = 3 * 4;
    let src_numel: usize = 3 * 6;
    let host_idx: Vec<u8> = vec![0, 1, 2, 5,  3, 0, 4, 1,  5, 4, 0, 2];
    let host_src: Vec<f32> = (0..src_numel).map(|i| (i as f32) * 0.5 + 1.0).collect();
    // Reference: out[i, j] = src[i, idx[i, j]].
    let mut expected = vec![0f32; out_numel];
    for i in 0..3usize {
        for j in 0..4usize {
            let k = host_idx[i * 4 + j] as usize;
            expected[i * 4 + j] = host_src[i * 6 + k];
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    // Contiguous row-major strides for shape [3, 4] and [3, 6].
    let stride_src = [6i64, 1];
    let stride_index = [4i64, 1];
    let stride_out = [4i64, 1];

    let src_ptr = dev_src.as_slice().as_raw().0 as *const c_void;
    let idx_ptr = dev_idx.as_slice().as_raw().0 as *const c_void;
    let out_ptr = dev_out.as_slice_mut().as_raw().0 as *mut c_void;
    let stream_ptr = stream.as_raw() as *mut c_void;

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_gather_u8idx_f32_run(
            out_numel as i64,
            /*rank=*/ 2,
            /*gather_dim=*/ 1,
            /*src_dim_size=*/ 6,
            out_shape.as_ptr(),
            stride_src.as_ptr(),
            stride_index.as_ptr(),
            stride_out.as_ptr(),
            src_ptr,
            idx_ptr,
            out_ptr,
            core::ptr::null_mut(),
            0,
            stream_ptr,
        )
    };
    assert_eq!(status, 0, "gather_u8idx_f32 returned status {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "gather u8idx f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
    // Silence unused: src_shape parameter is informational for the reader.
    let _ = src_shape;
}

#[test]
#[ignore]
fn gather_u8idx_f64_2d_dim0() {
    let (ctx, stream) = setup();
    let out_shape = [3i32, 4]; // == index shape
    let src_shape = [5i32, 4]; // src extent on dim=0 is 5
    let out_numel: usize = 3 * 4;
    let src_numel: usize = 5 * 4;
    let host_idx: Vec<u8> = vec![0, 1, 4, 3,  2, 0, 3, 1,  4, 4, 2, 0];
    let host_src: Vec<f64> = (0..src_numel).map(|i| (i as f64) * 0.25 + 2.0).collect();
    let mut expected = vec![0f64; out_numel];
    for i in 0..3usize {
        for j in 0..4usize {
            let k = host_idx[i * 4 + j] as usize;
            expected[i * 4 + j] = host_src[k * 4 + j];
        }
    }
    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("up src");
    let dev_idx = DeviceBuffer::from_slice(&ctx, &host_idx).expect("up idx");
    let mut dev_out: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc out");

    let stride_src = [4i64, 1];
    let stride_index = [4i64, 1];
    let stride_out = [4i64, 1];

    let src_ptr = dev_src.as_slice().as_raw().0 as *const c_void;
    let idx_ptr = dev_idx.as_slice().as_raw().0 as *const c_void;
    let out_ptr = dev_out.as_slice_mut().as_raw().0 as *mut c_void;
    let stream_ptr = stream.as_raw() as *mut c_void;

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_gather_u8idx_f64_run(
            out_numel as i64,
            /*rank=*/ 2,
            /*gather_dim=*/ 0,
            /*src_dim_size=*/ 5,
            out_shape.as_ptr(),
            stride_src.as_ptr(),
            stride_index.as_ptr(),
            stride_out.as_ptr(),
            src_ptr,
            idx_ptr,
            out_ptr,
            core::ptr::null_mut(),
            0,
            stream_ptr,
        )
    };
    assert_eq!(status, 0, "gather_u8idx_f64 returned status {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; out_numel];
    dev_out.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "gather u8idx f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
    let _ = src_shape;
    let _ = src_numel;
}
