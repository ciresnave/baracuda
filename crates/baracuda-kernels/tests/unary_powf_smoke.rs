//! Phase 31 — direct-FFI smoke for `unary_powf_<dtype>_run`.
//!
//! `y = pow(x, exponent)`. Exercises both integer and non-integer
//! exponents (the latter distinguishing PowF from PowI's power-by-
//! squaring path).

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
fn ffi_unary_powf_f32_integer_exponent_matches_cpu() {
    let (ctx, stream) = setup();
    let exponent = 3.0_f32;
    // Positive inputs only — `pow(-x, non_integer)` is NaN, but the
    // first test uses an integer exponent so the negative branch is
    // safe here too; we keep positive to make the second case
    // straightforward to share.
    let host_x: Vec<f32> = (0..512).map(|i| 0.01 + (i as f32) * 0.01).collect();
    let expected: Vec<f32> = host_x.iter().map(|&x| x.powf(exponent)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_powf_f32_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            exponent,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "unary_powf_f32 returned status {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    // `__powf` is ≤ 4 ULP for most ranges. Use relative tolerance.
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1e-6) * 1e-4;
        assert!(
            diff <= allow,
            "powf f32 e=3.0 mismatch @ {i}: got {g} expected {e} (diff {diff} > allow {allow})",
        );
    }
}

#[test]
#[ignore]
fn ffi_unary_powf_f32_fractional_exponent_matches_cpu() {
    let (ctx, stream) = setup();
    let exponent = 0.5_f32; // sqrt
    let host_x: Vec<f32> = (0..512).map(|i| 0.01 + (i as f32) * 0.1).collect();
    let expected: Vec<f32> = host_x.iter().map(|&x| x.powf(exponent)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_powf_f32_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            exponent,
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
        let diff = (g - e).abs();
        let allow = e.abs().max(1e-6) * 1e-4;
        assert!(
            diff <= allow,
            "powf f32 e=0.5 mismatch @ {i}: got {g} expected {e} (diff {diff} > allow {allow})",
        );
    }
}

#[test]
#[ignore]
fn ffi_unary_powf_f64_matches_cpu() {
    let (ctx, stream) = setup();
    let exponent = 2.5_f32;
    let host_x: Vec<f64> = (0..256).map(|i| 0.01 + (i as f64) * 0.05).collect();
    let expected: Vec<f64> = host_x.iter().map(|&x| x.powf(exponent as f64)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_powf_f64_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            exponent,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1e-12) * 1e-12;
        assert!(
            diff <= allow,
            "powf f64 mismatch @ {i}: got {g} expected {e} (diff {diff} > allow {allow})",
        );
    }
}
