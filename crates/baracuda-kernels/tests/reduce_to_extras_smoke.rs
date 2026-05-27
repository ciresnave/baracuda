//! Phase 37 Gap 1a — direct-FFI smoke for `reduce_min_to_<dtype>_run`
//! and `reduce_prod_to_<dtype>_run`.
//!
//! Verifies the broadcast-reverse min/prod against a tight Rust
//! reference. Covers f32 (trailblazer) and bf16 (half-precision).
//! Shape: `[2, 3, 4] → [1, 3, 1]` — reduces dim 0 and dim 2.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::bf16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn contig_strides(shape: &[i32]) -> Vec<i64> {
    let mut s = vec![0i64; shape.len()];
    let mut acc: i64 = 1;
    for i in (0..shape.len()).rev() {
        s[i] = acc;
        acc *= shape[i] as i64;
    }
    s
}

/// CPU reference for `reduce_min_to`: per-output cell, walk all input
/// cells that broadcast TO it and take min. Empty broadcast set ⇒ the
/// kernel returns `+inf` / `+max`.
fn cpu_min_to_f32(
    src: &[f32],
    in_shape: &[i32],
    in_stride: &[i64],
    out_shape: &[i32],
) -> Vec<f32> {
    let rank = in_shape.len();
    let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
    let mut dst = vec![f32::INFINITY; out_numel];

    let in_numel: usize = in_shape.iter().map(|&d| d as usize).product();
    let out_contig = contig_strides(out_shape);

    for in_lin in 0..in_numel {
        let mut lin = in_lin;
        let mut in_coord = vec![0i32; rank];
        for d in (0..rank).rev() {
            let s = in_shape[d] as usize;
            in_coord[d] = (lin % s) as i32;
            lin /= s;
        }
        let in_off: i64 = (0..rank)
            .map(|d| (in_coord[d] as i64) * in_stride[d])
            .sum();
        let mut out_lin: usize = 0;
        for d in 0..rank {
            let c = if out_shape[d] == 1 { 0 } else { in_coord[d] };
            out_lin += (c as usize) * (out_contig[d] as usize);
        }
        if src[in_off as usize] < dst[out_lin] {
            dst[out_lin] = src[in_off as usize];
        }
    }
    dst
}

fn cpu_prod_to_f32(
    src: &[f32],
    in_shape: &[i32],
    in_stride: &[i64],
    out_shape: &[i32],
) -> Vec<f32> {
    let rank = in_shape.len();
    let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
    let mut dst = vec![1f32; out_numel];

    let in_numel: usize = in_shape.iter().map(|&d| d as usize).product();
    let out_contig = contig_strides(out_shape);

    for in_lin in 0..in_numel {
        let mut lin = in_lin;
        let mut in_coord = vec![0i32; rank];
        for d in (0..rank).rev() {
            let s = in_shape[d] as usize;
            in_coord[d] = (lin % s) as i32;
            lin /= s;
        }
        let in_off: i64 = (0..rank)
            .map(|d| (in_coord[d] as i64) * in_stride[d])
            .sum();
        let mut out_lin: usize = 0;
        for d in 0..rank {
            let c = if out_shape[d] == 1 { 0 } else { in_coord[d] };
            out_lin += (c as usize) * (out_contig[d] as usize);
        }
        dst[out_lin] *= src[in_off as usize];
    }
    dst
}

#[test]
#[ignore]
fn ffi_reduce_min_to_f32_3d() {
    let (ctx, stream) = setup();
    let in_shape: Vec<i32> = vec![2, 3, 4];
    let out_shape: Vec<i32> = vec![1, 3, 1];
    let in_stride = contig_strides(&in_shape);

    // Deterministic but with negatives, zeros, and positives.
    let host_src: Vec<f32> = (0..24).map(|i| (i as f32) * 0.13 - 1.5).collect();
    let expected = cpu_min_to_f32(&host_src, &in_shape, &in_stride, &out_shape);

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, expected.len()).expect("alloc dst");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_reduce_min_to_f32_run(
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            in_shape.as_ptr(), in_stride.as_ptr(),
            in_shape.len() as i32,
            out_shape.as_ptr(),
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "reduce_min_to_f32 returned {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; expected.len()];
    dev_dst.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1.0) * 1e-5;
        assert!(diff <= allow,
            "min_to f32 @ {i}: got {g} expected {e} (diff {diff} > allow {allow})");
    }
}

#[test]
#[ignore]
fn ffi_reduce_prod_to_f32_3d() {
    let (ctx, stream) = setup();
    let in_shape: Vec<i32> = vec![2, 3, 4];
    let out_shape: Vec<i32> = vec![1, 3, 1];
    let in_stride = contig_strides(&in_shape);

    // Keep values close to 1 so the product stays in a safe range.
    let host_src: Vec<f32> = (0..24).map(|i| 0.85 + (i as f32) * 0.01).collect();
    let expected = cpu_prod_to_f32(&host_src, &in_shape, &in_stride, &out_shape);

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, expected.len()).expect("alloc dst");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_reduce_prod_to_f32_run(
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            in_shape.as_ptr(), in_stride.as_ptr(),
            in_shape.len() as i32,
            out_shape.as_ptr(),
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "reduce_prod_to_f32 returned {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; expected.len()];
    dev_dst.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1.0) * 1e-5;
        assert!(diff <= allow,
            "prod_to f32 @ {i}: got {g} expected {e} (diff {diff} > allow {allow})");
    }
}

/// bf16 min_to: small fixture, loose tolerance for half-precision.
#[test]
#[ignore]
fn ffi_reduce_min_to_bf16_3d() {
    let (ctx, stream) = setup();
    let in_shape: Vec<i32> = vec![2, 3, 4];
    let out_shape: Vec<i32> = vec![1, 3, 1];
    let in_stride = contig_strides(&in_shape);

    let host_src_f32: Vec<f32> = (0..24).map(|i| (i as f32) * 0.125 - 1.5).collect();
    let host_src: Vec<bf16> = host_src_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    // Reference operates in f32 to avoid double-rounding artefacts.
    let expected_f32 = cpu_min_to_f32(&host_src_f32, &in_shape, &in_stride, &out_shape);

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, expected_f32.len()).expect("alloc dst");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_reduce_min_to_bf16_run(
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            in_shape.as_ptr(), in_stride.as_ptr(),
            in_shape.len() as i32,
            out_shape.as_ptr(),
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got_bf16 = vec![bf16::from_f32(0.0); expected_f32.len()];
    dev_dst.copy_to_host(&mut got_bf16).expect("download");

    // Min is bit-comparable: as long as both sides see the same input
    // bf16 values, the comparator picks the same one. Compare via f32
    // with the bf16 round of the expected.
    for (i, (g, e)) in got_bf16.iter().zip(expected_f32.iter()).enumerate() {
        let g_f32 = g.to_f32();
        let e_bf16 = bf16::from_f32(*e).to_f32();
        let diff = (g_f32 - e_bf16).abs();
        let allow = e_bf16.abs().max(1.0) * 8e-3;
        assert!(diff <= allow,
            "min_to bf16 @ {i}: got {g_f32} expected {e_bf16} (diff {diff} > allow {allow})");
    }
}

/// f16 prod_to: keep values within 1.0 ± 0.05 to stay numerically tame.
#[test]
#[ignore]
fn ffi_reduce_prod_to_f16_3d() {
    use half::f16;

    let (ctx, stream) = setup();
    let in_shape: Vec<i32> = vec![2, 3, 4];
    let out_shape: Vec<i32> = vec![1, 3, 1];
    let in_stride = contig_strides(&in_shape);

    let host_src_f32: Vec<f32> = (0..24).map(|i| 0.97 + (i as f32) * 0.002).collect();
    let host_src: Vec<f16> = host_src_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let expected_f32 = cpu_prod_to_f32(&host_src_f32, &in_shape, &in_stride, &out_shape);

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, expected_f32.len()).expect("alloc dst");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_reduce_prod_to_f16_run(
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            in_shape.as_ptr(), in_stride.as_ptr(),
            in_shape.len() as i32,
            out_shape.as_ptr(),
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got_f16 = vec![f16::from_f32(0.0); expected_f32.len()];
    dev_dst.copy_to_host(&mut got_f16).expect("download");

    // Tolerance: 8 sequential f16 multiplies (after f32 detour) each
    // round-trip through half-precision, so ~8 * 5e-4 ULP = 4e-3 with
    // a 2x safety factor.
    for (i, (g, e)) in got_f16.iter().zip(expected_f32.iter()).enumerate() {
        let g_f32 = g.to_f32();
        let diff = (g_f32 - e).abs();
        let allow = e.abs().max(1.0) * 1.5e-2;
        assert!(diff <= allow,
            "prod_to f16 @ {i}: got {g_f32} expected {e} (diff {diff} > allow {allow})");
    }
}
