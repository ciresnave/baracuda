//! Direct-FFI smoke test for the **Phase 65c in-place dispatch
//! contract** on LayerNorm. Proves that calling
//! `baracuda_kernels_layer_norm_<dt>_run` with `x_ptr == y_ptr`
//! (a single device buffer for both input and output) produces the
//! same result as the non-aliased call.
//!
//! Same-shape contract as the RMSNorm in-place proof: rank-2
//! [outer, inner] tensor with contig strides routes through the
//! Phase 65c SMEM-staged kernel, which performs a cooperative load
//! into SMEM before any write to global, so aliasing the output over
//! the input is safe.
//!
//! Dtypes: f32 / f16 / bf16 (SMEM path eligible). f64 falls back to
//! the legacy multi-pass-global kernel which is NOT in-place safe
//! — skipped here.
//!
//! `#[ignore]` per project convention; run with
//! `cargo test -p baracuda-kernels --features sm89 \
//!    --test layer_norm_inplace_smoke -- --include-ignored`.

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

#[test]
#[ignore]
fn layer_norm_f32_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();

    let outer: usize = 4;
    let inner: usize = 128;
    let numel = outer * inner;
    let eps: f32 = 1e-5;
    let shape: [i32; 2] = [outer as i32, inner as i32];
    let stride_contig: [i64; 2] = [inner as i64, 1];
    let stride_save: [i64; 2] = [1, 0];
    let norm_axes_mask: i32 = 0b10;
    let norm_total_extent: i32 = inner as i32;

    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.011 + 0.3).sin() * 0.7 - 0.2)
        .collect();

    // --- Non-aliased reference run ---
    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y_ref: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let mut dev_mean_ref: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, outer).expect("alloc mean");
    let mut dev_inv_std_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, outer).expect("alloc inv_std");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_layer_norm_f32_run(
            eps,
            numel as i64,
            2,
            shape.as_ptr(),
            stride_contig.as_ptr(),
            stride_contig.as_ptr(),
            stride_save.as_ptr(),
            norm_axes_mask,
            norm_total_extent,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_mean_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_inv_std_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "non-aliased run");
    stream.synchronize().expect("sync ref");
    let mut ref_out = vec![0_f32; numel];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl ref");

    // --- Aliased run: x_ptr == y_ptr ---
    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x_inplace");
    let mut dev_mean_inplace: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, outer).expect("alloc mean inplace");
    let mut dev_inv_std_inplace: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, outer).expect("alloc inv_std inplace");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_layer_norm_f32_run(
            eps,
            numel as i64,
            2,
            shape.as_ptr(),
            stride_contig.as_ptr(),
            stride_contig.as_ptr(),
            stride_save.as_ptr(),
            norm_axes_mask,
            norm_total_extent,
            p as *const c_void,
            core::ptr::null(),
            core::ptr::null(),
            p as *mut c_void,
            dev_mean_inplace.as_slice_mut().as_raw().0 as *mut c_void,
            dev_inv_std_inplace.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "aliased (in-place) run");
    stream.synchronize().expect("sync aliased");
    let mut aliased_out = vec![0_f32; numel];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl aliased");

    for i in 0..numel {
        assert_eq!(
            aliased_out[i].to_bits(),
            ref_out[i].to_bits(),
            "f32 in-place LayerNorm @ {i}: aliased={} non-aliased={}",
            aliased_out[i], ref_out[i]
        );
    }
}

#[test]
#[ignore]
fn layer_norm_f16_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let outer: usize = 4;
    let inner: usize = 128;
    let numel = outer * inner;
    let eps: f32 = 1e-5;
    let shape: [i32; 2] = [outer as i32, inner as i32];
    let stride_contig: [i64; 2] = [inner as i64, 1];
    let stride_save: [i64; 2] = [1, 0];

    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.011 + 0.3).sin() * 0.7 - 0.2)
        .collect();
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_mean_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let mut dev_inv_std_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_layer_norm_f16_run(
            eps,
            numel as i64,
            2,
            shape.as_ptr(),
            stride_contig.as_ptr(),
            stride_contig.as_ptr(),
            stride_save.as_ptr(),
            0b10,
            inner as i32,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_mean_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_inv_std_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![f16::ZERO; numel];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_mean_inplace: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let mut dev_inv_std_inplace: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_layer_norm_f16_run(
            eps,
            numel as i64,
            2,
            shape.as_ptr(),
            stride_contig.as_ptr(),
            stride_contig.as_ptr(),
            stride_save.as_ptr(),
            0b10,
            inner as i32,
            p as *const c_void,
            core::ptr::null(),
            core::ptr::null(),
            p as *mut c_void,
            dev_mean_inplace.as_slice_mut().as_raw().0 as *mut c_void,
            dev_inv_std_inplace.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![f16::ZERO; numel];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..numel {
        assert_eq!(
            aliased_out[i].to_bits(),
            ref_out[i].to_bits(),
            "f16 in-place LayerNorm @ {i}: aliased={} non-aliased={}",
            aliased_out[i].to_f32(),
            ref_out[i].to_f32()
        );
    }
}

#[test]
#[ignore]
fn layer_norm_bf16_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let outer: usize = 4;
    let inner: usize = 128;
    let numel = outer * inner;
    let eps: f32 = 1e-5;
    let shape: [i32; 2] = [outer as i32, inner as i32];
    let stride_contig: [i64; 2] = [inner as i64, 1];
    let stride_save: [i64; 2] = [1, 0];

    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.011 + 0.3).sin() * 0.7 - 0.2)
        .collect();
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_mean_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let mut dev_inv_std_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_layer_norm_bf16_run(
            eps,
            numel as i64,
            2,
            shape.as_ptr(),
            stride_contig.as_ptr(),
            stride_contig.as_ptr(),
            stride_save.as_ptr(),
            0b10,
            inner as i32,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_mean_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_inv_std_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![bf16::ZERO; numel];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_mean_inplace: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let mut dev_inv_std_inplace: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, outer).expect("alloc");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_layer_norm_bf16_run(
            eps,
            numel as i64,
            2,
            shape.as_ptr(),
            stride_contig.as_ptr(),
            stride_contig.as_ptr(),
            stride_save.as_ptr(),
            0b10,
            inner as i32,
            p as *const c_void,
            core::ptr::null(),
            core::ptr::null(),
            p as *mut c_void,
            dev_mean_inplace.as_slice_mut().as_raw().0 as *mut c_void,
            dev_inv_std_inplace.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![bf16::ZERO; numel];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..numel {
        assert_eq!(
            aliased_out[i].to_bits(),
            ref_out[i].to_bits(),
            "bf16 in-place LayerNorm @ {i}"
        );
    }
}
