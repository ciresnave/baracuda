//! Phase 31 — direct-FFI smoke for `reduce_sum_to_<dtype>_run`.
//!
//! Verifies the broadcast-reverse Σ against a tight Rust reference.
//! Three shapes:
//!   1. [3, 4] → [1, 4]  — reduce dim 0.
//!   2. [2, 3, 4] → [1, 3, 1] — reduce dims 0 and 2.
//!   3. [5] → [1] — full reduction.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Compute contiguous (row-major) strides for the given shape.
fn contig_strides(shape: &[i32]) -> Vec<i64> {
    let mut s = vec![0i64; shape.len()];
    let mut acc: i64 = 1;
    for i in (0..shape.len()).rev() {
        s[i] = acc;
        acc *= shape[i] as i64;
    }
    s
}

/// CPU reference: for each output cell, sum every input cell that
/// broadcasts to it (matches the kernel semantics).
fn cpu_sum_to_f32(
    src: &[f32],
    in_shape: &[i32],
    in_stride: &[i64],
    out_shape: &[i32],
) -> Vec<f32> {
    let rank = in_shape.len();
    let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
    let mut dst = vec![0f32; out_numel];

    // Iterate all input cells; map each to its output cell.
    let in_numel: usize = in_shape.iter().map(|&d| d as usize).product();
    let in_contig = contig_strides(in_shape);
    let out_contig = contig_strides(out_shape);

    for in_lin in 0..in_numel {
        let mut lin = in_lin;
        let mut in_coord = vec![0i32; rank];
        for d in (0..rank).rev() {
            let s = in_shape[d] as usize;
            in_coord[d] = (lin % s) as i32;
            lin /= s;
        }

        // Compute input physical offset using in_stride (which may
        // differ from in_contig in the broadcast case here we use
        // in_contig because the test allocates contiguously).
        let in_off: i64 = (0..rank)
            .map(|d| (in_coord[d] as i64) * in_stride[d])
            .sum();
        let _ = in_contig; // (in_contig used only for the no-op identity check)

        // Output cell: for broadcast dims (out_shape[d] == 1) the
        // output coord is 0; else it matches the input coord.
        let mut out_lin: usize = 0;
        for d in 0..rank {
            let c = if out_shape[d] == 1 { 0 } else { in_coord[d] };
            out_lin += (c as usize) * (out_contig[d] as usize);
        }
        dst[out_lin] += src[in_off as usize];
    }

    dst
}

#[test]
#[ignore]
fn ffi_reduce_sum_to_f32_2d_reduce_dim0() {
    let (ctx, stream) = setup();
    let in_shape: Vec<i32> = vec![3, 4];
    let out_shape: Vec<i32> = vec![1, 4];
    let in_stride = contig_strides(&in_shape);

    let host_src: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 1.0).collect();
    let expected = cpu_sum_to_f32(&host_src, &in_shape, &in_stride, &out_shape);

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, expected.len()).expect("alloc dst");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_reduce_sum_to_f32_run(
            dev_src.as_slice().as_raw().0 as *const c_void,
            dev_dst.as_slice_mut().as_raw().0 as *mut c_void,
            in_shape.as_ptr(), in_stride.as_ptr(),
            in_shape.len() as i32,
            out_shape.as_ptr(),
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "reduce_sum_to_f32 returned {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; expected.len()];
    dev_dst.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1.0) * 1e-5;
        assert!(diff <= allow,
            "sum_to f32 [3,4]→[1,4] @ {i}: got {g} expected {e} (diff {diff} > allow {allow})");
    }
}

#[test]
#[ignore]
fn ffi_reduce_sum_to_f32_3d_reduce_dim0_dim2() {
    let (ctx, stream) = setup();
    let in_shape: Vec<i32> = vec![2, 3, 4];
    let out_shape: Vec<i32> = vec![1, 3, 1];
    let in_stride = contig_strides(&in_shape);

    let host_src: Vec<f32> = (0..24).map(|i| (i as f32) * 0.05 - 0.5).collect();
    let expected = cpu_sum_to_f32(&host_src, &in_shape, &in_stride, &out_shape);

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, expected.len()).expect("alloc dst");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_reduce_sum_to_f32_run(
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

    let mut got = vec![0f32; expected.len()];
    dev_dst.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1.0) * 1e-5;
        assert!(diff <= allow,
            "sum_to f32 [2,3,4]→[1,3,1] @ {i}: got {g} expected {e} (diff {diff} > allow {allow})");
    }
}

#[test]
#[ignore]
fn ffi_reduce_sum_to_f32_full_reduce() {
    let (ctx, stream) = setup();
    let in_shape: Vec<i32> = vec![5];
    let out_shape: Vec<i32> = vec![1];
    let in_stride = contig_strides(&in_shape);
    let host_src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let expected = vec![15.0_f32];

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, 1).expect("alloc dst");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_reduce_sum_to_f32_run(
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

    let mut got = vec![0f32; 1];
    dev_dst.copy_to_host(&mut got).expect("download");
    assert!((got[0] - expected[0]).abs() < 1e-5,
        "sum_to full reduce: got {} expected {}", got[0], expected[0]);
}
