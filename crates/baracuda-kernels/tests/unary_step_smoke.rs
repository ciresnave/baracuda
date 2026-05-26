//! Phase 31 — direct-FFI smoke for `unary_step_<dtype>_run`.
//!
//! `y = (x > 0) ? 1 : 0`. NaN → 0 (NaN > 0 is false).

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
fn ffi_unary_step_f32_matches_cpu() {
    let (ctx, stream) = setup();
    // Span (-3.0, +3.0) plus a zero and an explicit NaN.
    let mut host_x: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 - 5.12).collect();
    host_x.push(0.0);
    host_x.push(f32::NAN);

    let expected: Vec<f32> = host_x.iter()
        .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_step_f32_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "unary_step_f32 returned status {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    // Step is exact (no math; just a compare). Use bit-equality.
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "step f32 @ {i} (x = {}): got {g} expected {e}",
            host_x[i],
        );
    }
}

#[test]
#[ignore]
fn ffi_unary_step_f64_matches_cpu() {
    let (ctx, stream) = setup();
    let host_x: Vec<f64> = (0..256).map(|i| (i as f64) * 0.02 - 2.56).collect();
    let expected: Vec<f64> = host_x.iter()
        .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_step_f64_run(
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

    let mut got = vec![0f64; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "step f64 mismatch @ {i}");
    }
}
