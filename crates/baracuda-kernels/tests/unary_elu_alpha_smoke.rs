//! Phase 31 — verify the ELU FFI threads `alpha` through correctly.
//!
//! The historical `UnaryPlan + UnaryKind::Elu` smoke test pins
//! α = 1.0 (PyTorch default); this one exercises the breaking-change
//! FFI with a non-default α to confirm the parameter is actually
//! consumed by the kernel rather than silently ignored.
//!
//! Direct FFI smoke (NOT through the plan layer) since the plan
//! still hardcodes α = 1.0 until a future descriptor-param session.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn cpu_elu_f32(x: f32, alpha: f32) -> f32 {
    if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }
}

#[test]
#[ignore]
fn ffi_unary_elu_f32_alpha_2_5_matches_cpu() {
    let (ctx, stream) = setup();
    let alpha = 2.5_f32;
    let host_x: Vec<f32> = (0..1024).map(|i| ((i as f32) - 512.0) * 0.01).collect();
    let expected: Vec<f32> = host_x.iter().map(|&x| cpu_elu_f32(x, alpha)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_elu_f32_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            alpha,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "unary_elu_f32 returned status {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1.0) * 4.0 * f32::EPSILON;
        assert!(
            diff <= allow,
            "elu α=2.5 mismatch @ {i}: got {g} expected {e} (diff {diff} > allow {allow})",
        );
    }
}

#[test]
#[ignore]
fn ffi_unary_elu_f64_alpha_negative_branch_uses_alpha() {
    // Pick a fully-negative input range so EVERY cell takes the
    // negative branch — this is where α matters. Asserts the kernel
    // is actually multiplying by the supplied α, not the old hardcoded
    // 1.0.
    let (ctx, stream) = setup();
    let alpha = 3.0_f32;
    let host_x: Vec<f64> = (0..256).map(|i| -1.0 - (i as f64) * 0.001).collect();
    let expected: Vec<f64> = host_x.iter()
        .map(|&x| (alpha as f64) * (x.exp() - 1.0))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_elu_f64_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            alpha,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "unary_elu_f64 returned status {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1.0) * 4.0 * f64::EPSILON;
        assert!(
            diff <= allow,
            "elu f64 α=3.0 mismatch @ {i}: got {g} expected {e} (diff {diff} > allow {allow})",
        );
        // Spot-check: result must NOT equal what α=1.0 would produce.
        let alpha_one = host_x[i].exp() - 1.0;
        let diff_vs_one = (g - alpha_one).abs();
        assert!(
            diff_vs_one > 1e-3,
            "elu α=3.0 result @ {i} matches α=1.0 result — α isn't being threaded through?",
        );
    }
}
