//! Phase 31 — direct-FFI smoke for the new u32 / i16 cast instantiations.
//!
//! Covers the four cells of the matrix that exercise both new dtypes
//! as source and destination: f32→u32, u32→f32, f32→i16, i16→f32.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn ffi_cast_f32_u32_matches_cpp_static_cast() {
    let (ctx, stream) = setup();
    // Positive range; static_cast<uint32_t> of a negative float is
    // implementation-defined.
    let host_x: Vec<f32> = (0..1024).map(|i| (i as f32) * 1024.0 + 0.5).collect();
    let expected: Vec<u32> = host_x.iter().map(|&x| x as u32).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<u32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_cast_f32_u32_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "cast_f32_u32 returned status {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0u32; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cast_f32_u32 mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn ffi_cast_u32_f32_matches_cpp_static_cast() {
    let (ctx, stream) = setup();
    let host_x: Vec<u32> = (0..2048u32).map(|i| i * 7919).collect();
    let expected: Vec<f32> = host_x.iter().map(|&x| x as f32).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_cast_u32_f32_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(),
            "cast_u32_f32 mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn ffi_cast_f32_i16_matches_cpp_static_cast() {
    let (ctx, stream) = setup();
    // Range that exercises both positive and negative cells, and the
    // narrow i16 dynamic range.
    let host_x: Vec<f32> = (0..2048).map(|i| ((i as f32) - 1024.0) * 16.0).collect();
    let expected: Vec<i16> = host_x.iter().map(|&x| x as i16).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<i16> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_cast_f32_i16_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i16; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        // Cast may differ from Rust's saturating-`as` for out-of-range
        // floats; clamp the comparison range.
        if host_x[i] >= -32768.0 && host_x[i] < 32768.0 {
            assert_eq!(g, e, "cast_f32_i16 mismatch @ {i}: x={} got {g} expected {e}",
                host_x[i]);
        }
    }
}

#[test]
#[ignore]
fn ffi_cast_i16_f32_sign_extends() {
    let (ctx, stream) = setup();
    let host_x: Vec<i16> = vec![-32768, -1, 0, 1, 32767, 12345, -12345];
    let expected: Vec<f32> = host_x.iter().map(|&x| x as f32).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_cast_i16_f32_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(),
            "cast_i16_f32 mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn ffi_cast_u32_i16_truncates_low_bits() {
    let (ctx, stream) = setup();
    // Round-trip through u32 → i16.
    let host_x: Vec<u32> = vec![0, 1, 0xFFFF, 0x10000, 0x12345678, 0xFFFFFFFF];
    // C++ semantics: low 16 bits, interpreted as i16 (sign).
    let expected: Vec<i16> = host_x.iter().map(|&x| x as i16).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<i16> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_cast_u32_i16_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i16; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "cast_u32_i16 mismatch @ {i}: u32={} got {g} expected {e}",
            host_x[i]);
    }
}
