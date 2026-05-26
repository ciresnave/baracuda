//! Phase 31 — direct-FFI smoke for `unary_gelu_erf_<dtype>_run`.
//!
//! Exact-erf-based GELU: `y = 0.5 * x * (1 + erf(x / sqrt(2)))`.
//! Coexists with `unary_gelu_*` (currently bit-identical math under
//! a different symbol name) and is distinct from
//! `unary_gelu_tanh_*` (the tanh approximation).

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn cpu_gelu_erf_f32(x: f32) -> f32 {
    // 0.5 * x * (1 + erf(x / sqrt(2)))
    0.5 * x * (1.0 + libm::erff(x * std::f32::consts::FRAC_1_SQRT_2))
}

#[test]
#[ignore]
fn ffi_unary_gelu_erf_f32_matches_cpu() {
    let (ctx, stream) = setup();
    let host_x: Vec<f32> = (0..1024).map(|i| ((i as f32) - 512.0) * 0.01).collect();
    let expected: Vec<f32> = host_x.iter().map(|&x| cpu_gelu_erf_f32(x)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_gelu_erf_f32_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "unary_gelu_erf_f32 returned status {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1.0) * 4.0 * f32::EPSILON;
        assert!(
            diff <= allow,
            "gelu_erf f32 mismatch @ {i}: got {g} expected {e} (diff {diff} > allow {allow})",
        );
    }
}

/// Verify `gelu_erf` and the existing `gelu` symbols produce the same
/// output (they share the same math today). Failure indicates one of
/// the two symbols was accidentally pointed at a different formula.
#[test]
#[ignore]
fn ffi_unary_gelu_erf_matches_unary_gelu_bit_identical() {
    let (ctx, stream) = setup();
    let host_x: Vec<f32> = (0..1024).map(|i| ((i as f32) - 512.0) * 0.01).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y_erf: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y_erf");
    let mut dev_y_gelu: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y_gelu");

    unsafe {
        let st1 = baracuda_kernels_sys::baracuda_kernels_unary_gelu_erf_f32_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y_erf.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        );
        let st2 = baracuda_kernels_sys::baracuda_kernels_unary_gelu_f32_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y_gelu.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        );
        assert_eq!(st1, 0);
        assert_eq!(st2, 0);
    }
    stream.synchronize().expect("sync");

    let mut y_erf = vec![0f32; host_x.len()];
    let mut y_gelu = vec![0f32; host_x.len()];
    dev_y_erf.copy_to_host(&mut y_erf).expect("dl y_erf");
    dev_y_gelu.copy_to_host(&mut y_gelu).expect("dl y_gelu");

    for (i, (a, b)) in y_erf.iter().zip(y_gelu.iter()).enumerate() {
        assert_eq!(
            a.to_bits(), b.to_bits(),
            "gelu_erf vs gelu f32 mismatch @ {i}: erf={a} gelu={b} (must be bit-identical)",
        );
    }
}
