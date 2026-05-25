//! Phase 19.2 — Real-GPU smoke tests for the upsample (nearest-2D /
//! bilinear-2D) FFI symbols exposed by `baracuda-kernels-sys`.
//!
//! Bilinear FW + BW already validated by `interpolate_smoke.rs` against
//! a host reference; this file's bilinear cases verify the
//! `upsample_bilinear_2d_*` alias re-exports yield bit-exact match to
//! the underlying `interpolate_bilinear_2d_*` symbols. The nearest
//! cases verify the new `upsample_nearest_2d_*` kernels against a
//! hand-rolled host reference.
//!
//! All tests `#[ignore]` — need a real CUDA device.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Host reference for nearest-2D upsample under `align_corners=false`.
/// Coordinate mapping: `src_idx = min(floor(dst * src/dst), src - 1)`.
fn cpu_nearest_2d_f32(
    x: &[f32], n: i32, c: i32, ih: i32, iw: i32, oh: i32, ow: i32,
) -> Vec<f32> {
    let mut out = vec![0f32; (n * c * oh * ow) as usize];
    for nn in 0..n {
        for cc in 0..c {
            for ohh in 0..oh {
                let iy_i64 = (ohh as i64) * (ih as i64) / (oh as i64);
                let iy = (iy_i64 as i32).min(ih - 1).max(0);
                for oww in 0..ow {
                    let ix_i64 = (oww as i64) * (iw as i64) / (ow as i64);
                    let ix = (ix_i64 as i32).min(iw - 1).max(0);
                    let in_off = (((nn * c + cc) * ih + iy) * iw + ix) as usize;
                    let out_off = (((nn * c + cc) * oh + ohh) * ow + oww) as usize;
                    out[out_off] = x[in_off];
                }
            }
        }
    }
    out
}

/// Host reference for the BW: for each output cell, scatter-add its
/// `dout` value into the same input cell the FW would have sampled.
fn cpu_nearest_2d_bw_f32(
    dout: &[f32], n: i32, c: i32, ih: i32, iw: i32, oh: i32, ow: i32,
) -> Vec<f32> {
    let mut din = vec![0f32; (n * c * ih * iw) as usize];
    for nn in 0..n {
        for cc in 0..c {
            for ohh in 0..oh {
                let iy_i64 = (ohh as i64) * (ih as i64) / (oh as i64);
                let iy = (iy_i64 as i32).min(ih - 1).max(0);
                for oww in 0..ow {
                    let ix_i64 = (oww as i64) * (iw as i64) / (ow as i64);
                    let ix = (ix_i64 as i32).min(iw - 1).max(0);
                    let in_off = (((nn * c + cc) * ih + iy) * iw + ix) as usize;
                    let out_off = (((nn * c + cc) * oh + ohh) * ow + oww) as usize;
                    din[in_off] += dout[out_off];
                }
            }
        }
    }
    din
}

#[test]
#[ignore]
fn ffi_upsample_nearest_2d_f32_fw_matches_cpu() {
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1, 1, 2, 2);
    let (oh, ow) = (4, 4);
    let host_in: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let expected = cpu_nearest_2d_f32(&host_in, n, c, ih, iw, oh, ow);

    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("up in");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * oh * ow) as usize).expect("alloc out");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_upsample_nearest_2d_fw_f32_run(
            n, c, ih, iw, oh, ow,
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "upsample_nearest_2d FW f32 failed");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; (n * c * oh * ow) as usize];
    dev_out.copy_to_host(&mut got).expect("dl out");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "FFI nearest FW mismatch @ {i}: ffi={g}, cpu={e}");
    }
}

#[test]
#[ignore]
fn ffi_upsample_nearest_2d_f32_bw_matches_cpu() {
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1, 2, 3, 3);
    let (oh, ow) = (6, 6);
    let dout: Vec<f32> = (0..(n * c * oh * ow) as usize)
        .map(|i| (i as f32) * 0.1 + 0.01)
        .collect();
    let expected = cpu_nearest_2d_bw_f32(&dout, n, c, ih, iw, oh, ow);

    let dev_dout = DeviceBuffer::from_slice(&ctx, &dout).expect("up dout");
    // BW kernel requires pre-zeroed dinput.
    let mut dev_din: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * ih * iw) as usize).expect("alloc din");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_upsample_nearest_2d_bw_f32_run(
            n, c, ih, iw, oh, ow,
            dev_dout.as_slice().as_raw().0 as *const c_void,
            dev_din.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "upsample_nearest_2d BW f32 failed");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; (n * c * ih * iw) as usize];
    dev_din.copy_to_host(&mut got).expect("dl din");

    // atomicAdd ordering may vary, but with a 2x exact upsample factor
    // each input cell receives exactly 4 disjoint summands of distinct
    // magnitudes — host order and device order should match within an
    // ULP for this size. Use a small relative tolerance for safety.
    let tol = 1e-5_f32;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(
            diff <= tol * e.abs().max(1.0),
            "FFI nearest BW mismatch @ {i}: ffi={g}, cpu={e}, diff={diff}",
        );
    }
}

/// The `upsample_bilinear_2d_*` symbols must produce the same output
/// as the underlying `interpolate_bilinear_2d_*` symbols — they're
/// declared as `#[inline]` Rust wrappers that forward unchanged, so
/// bit-exact match is the contract.
#[test]
#[ignore]
fn ffi_upsample_bilinear_2d_f32_fw_matches_interpolate_alias() {
    let (ctx, stream) = setup();
    let (n, c, ih, iw) = (1, 1, 2, 2);
    let (oh, ow) = (4, 4);
    let host_in: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let dev_in_a = DeviceBuffer::from_slice(&ctx, &host_in).expect("up in a");
    let dev_in_b = DeviceBuffer::from_slice(&ctx, &host_in).expect("up in b");
    let mut dev_out_a: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * oh * ow) as usize).expect("alloc out a");
    let mut dev_out_b: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * oh * ow) as usize).expect("alloc out b");

    // a: under the original `interpolate_*` symbol.
    let s_a = unsafe {
        baracuda_kernels_sys::baracuda_kernels_interpolate_bilinear_2d_f32_run(
            n, c, ih, iw, oh, ow,
            dev_in_a.as_slice().as_raw().0 as *const c_void,
            dev_out_a.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(s_a, 0);
    // b: under the new `upsample_*` alias.
    let s_b = unsafe {
        baracuda_kernels_sys::baracuda_kernels_upsample_bilinear_2d_fw_f32_run(
            n, c, ih, iw, oh, ow,
            dev_in_b.as_slice().as_raw().0 as *const c_void,
            dev_out_b.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(s_b, 0);
    stream.synchronize().expect("sync");
    let mut a = vec![0f32; (n * c * oh * ow) as usize];
    let mut b = vec![0f32; (n * c * oh * ow) as usize];
    dev_out_a.copy_to_host(&mut a).expect("dl a");
    dev_out_b.copy_to_host(&mut b).expect("dl b");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "alias mismatch @ {i}: interpolate={x}, upsample={y}");
    }
}
