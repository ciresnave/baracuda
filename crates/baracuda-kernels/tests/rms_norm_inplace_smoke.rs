//! Direct-FFI smoke test for the **Phase 65b in-place dispatch
//! contract** on RMSNorm. Proves that calling
//! `baracuda_kernels_rms_norm_<dt>_run` with `x_ptr == y_ptr` (a
//! single device buffer for both input and output) produces the same
//! result as the non-aliased call.
//!
//! This is the load-bearing test for Phase 65b's claim that the
//! SMEM-staged RMSNorm path is in-place safe. The contig last-axis
//! eligibility check in `launch_rms_norm_fp<T>` routes the test
//! shape (rank-2 [outer, inner] with contig strides) through the
//! SMEM-staged kernel; in-place safety is therefore proven for the
//! production code path.
//!
//! Dtypes: f32 / f16 / bf16 (SMEM path eligible). f64 falls back to
//! the legacy multi-pass-global kernel which is NOT in-place safe
//! — skipped here.
//!
//! `#[ignore]` per project convention; run with
//! `cargo test -p baracuda-kernels --features sm89 \
//!    --test rms_norm_inplace_smoke -- --include-ignored`.

#![cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// CPU reference for RMSNorm over the last axis of a [outer, inner] tensor.
fn host_rms_norm_f32(host_x: &[f32], outer: usize, inner: usize, eps: f32) -> Vec<f32> {
    let mut y = vec![0_f32; host_x.len()];
    for o in 0..outer {
        let mut sum_sq = 0_f64;
        for i in 0..inner {
            let v = host_x[o * inner + i] as f64;
            sum_sq += v * v;
        }
        let mean_sq = sum_sq / (inner as f64);
        let rms = (mean_sq + eps as f64).sqrt();
        let inv_rms = 1.0 / rms;
        for i in 0..inner {
            y[o * inner + i] = (host_x[o * inner + i] as f64 * inv_rms) as f32;
        }
    }
    y
}

#[test]
#[ignore]
fn rms_norm_f32_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();

    let outer: usize = 4;
    let inner: usize = 128;
    let numel = outer * inner;
    let eps: f32 = 1e-5;
    let shape: [i32; 2] = [outer as i32, inner as i32];
    let stride_contig: [i64; 2] = [inner as i64, 1];
    // rms_out shape: [outer, 1]; stride: [1, 0]
    let stride_rms: [i64; 2] = [1, 0];
    let norm_axes_mask: i32 = 0b10; // bit 1 = last axis (axis index 1)
    let norm_total_extent: i32 = inner as i32;

    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.013 - 1.5).cos() * 0.6)
        .collect();
    let expected_y = host_rms_norm_f32(&host_x, outer, inner, eps);

    // --- Non-aliased reference run ---
    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y_ref: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let mut dev_rms_ref: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, outer).expect("alloc rms");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rms_norm_f32_run(
            eps,
            numel as i64,
            2,
            shape.as_ptr(),
            stride_contig.as_ptr(),
            stride_contig.as_ptr(),
            stride_rms.as_ptr(),
            norm_axes_mask,
            norm_total_extent,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rms_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "non-aliased run");
    stream.synchronize().expect("sync ref");
    let mut ref_out = vec![0_f32; numel];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl ref");

    // --- Aliased run: x_ptr == y_ptr ---
    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x_inplace");
    let mut dev_rms_inplace: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, outer).expect("alloc rms inplace");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rms_norm_f32_run(
            eps,
            numel as i64,
            2,
            shape.as_ptr(),
            stride_contig.as_ptr(),
            stride_contig.as_ptr(),
            stride_rms.as_ptr(),
            norm_axes_mask,
            norm_total_extent,
            p as *const c_void,
            core::ptr::null(),
            p as *mut c_void,
            dev_rms_inplace.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "aliased (in-place) run");
    stream.synchronize().expect("sync aliased");
    let mut aliased_out = vec![0_f32; numel];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl aliased");

    // Aliased output must match non-aliased reference bit-for-bit
    // (same kernel runs in both cases — Phase 65b SMEM-staged path,
    // deterministic accumulation order).
    for i in 0..numel {
        assert_eq!(
            aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "f32 in-place RMSNorm @ {i}: aliased={} non-aliased={}",
            aliased_out[i], ref_out[i]
        );
        // Also confirm we match the CPU reference within tolerance.
        let tol = (expected_y[i].abs() * 16.0 * f32::EPSILON).max(16.0 * f32::EPSILON);
        assert!((aliased_out[i] - expected_y[i]).abs() <= tol,
            "f32 RMSNorm vs CPU @ {i}: got={} want={}", aliased_out[i], expected_y[i]);
    }
}

#[test]
#[ignore]
fn rms_norm_f16_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let outer: usize = 4;
    let inner: usize = 128;
    let numel = outer * inner;
    let eps: f32 = 1e-5;
    let shape: [i32; 2] = [outer as i32, inner as i32];
    let stride_contig: [i64; 2] = [inner as i64, 1];
    let stride_rms: [i64; 2] = [1, 0];

    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.013 - 1.5).cos() * 0.6)
        .collect();
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_rms_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rms_norm_f16_run(
            eps, numel as i64, 2, shape.as_ptr(),
            stride_contig.as_ptr(), stride_contig.as_ptr(), stride_rms.as_ptr(),
            0b10, inner as i32,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rms_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![f16::ZERO; numel];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_rms_inplace: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rms_norm_f16_run(
            eps, numel as i64, 2, shape.as_ptr(),
            stride_contig.as_ptr(), stride_contig.as_ptr(), stride_rms.as_ptr(),
            0b10, inner as i32,
            p as *const c_void,
            core::ptr::null(),
            p as *mut c_void,
            dev_rms_inplace.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![f16::ZERO; numel];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..numel {
        assert_eq!(aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "f16 in-place RMSNorm @ {i}: aliased={} non-aliased={}",
            aliased_out[i].to_f32(), ref_out[i].to_f32());
    }
}

#[test]
#[ignore]
fn rms_norm_bf16_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let outer: usize = 4;
    let inner: usize = 128;
    let numel = outer * inner;
    let eps: f32 = 1e-5;
    let shape: [i32; 2] = [outer as i32, inner as i32];
    let stride_contig: [i64; 2] = [inner as i64, 1];
    let stride_rms: [i64; 2] = [1, 0];

    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.013 - 1.5).cos() * 0.6)
        .collect();
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_rms_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rms_norm_bf16_run(
            eps, numel as i64, 2, shape.as_ptr(),
            stride_contig.as_ptr(), stride_contig.as_ptr(), stride_rms.as_ptr(),
            0b10, inner as i32,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rms_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![bf16::ZERO; numel];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_rms_inplace: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rms_norm_bf16_run(
            eps, numel as i64, 2, shape.as_ptr(),
            stride_contig.as_ptr(), stride_contig.as_ptr(), stride_rms.as_ptr(),
            0b10, inner as i32,
            p as *const c_void,
            core::ptr::null(),
            p as *mut c_void,
            dev_rms_inplace.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![bf16::ZERO; numel];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..numel {
        assert_eq!(aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "bf16 in-place RMSNorm @ {i}");
    }
}
