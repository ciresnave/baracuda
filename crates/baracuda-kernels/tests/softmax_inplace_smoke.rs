//! Direct-FFI smoke test for the **Phase 65c in-place dispatch
//! contract** on Softmax. Proves that
//! `baracuda_kernels_softmax_<dt>_run` with `x_ptr == y_ptr`
//! produces the same result as the non-aliased call.
//!
//! The rank-2 [outer, inner] contig-strided shape used here routes
//! through the Phase 65c SMEM-staged kernel
//! (`softmax_smem_kernel<T>` in `baracuda_softmax.cuh`), which
//! cooperatively stages the entire row into SMEM before any write
//! to global, making same-pointer aliasing safe.
//!
//! Dtypes: f32 / f16 / bf16 (SMEM path eligible). f64 falls back to
//! the legacy multi-pass-global kernel which is NOT in-place safe
//! — skipped here.
//!
//! `#[ignore]` per project convention; run with
//! `cargo test -p baracuda-kernels --features sm89 \
//!    --test softmax_inplace_smoke -- --include-ignored`.

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
fn softmax_f32_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let outer: usize = 4;
    let inner: usize = 128;
    let numel = outer * inner;
    let shape: [i32; 2] = [outer as i32, inner as i32];
    let stride_contig: [i64; 2] = [inner as i64, 1];

    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.017 + 0.1).cos() * 1.4)
        .collect();

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_softmax_f32_run(
            numel as i64,
            2,
            shape.as_ptr(),
            stride_contig.as_ptr(),
            stride_contig.as_ptr(),
            1,
            inner as i32,
            1,
            1,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "non-aliased softmax");
    stream.synchronize().expect("sync ref");
    let mut ref_out = vec![0_f32; numel];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl ref");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_softmax_f32_run(
            numel as i64,
            2,
            shape.as_ptr(),
            stride_contig.as_ptr(),
            stride_contig.as_ptr(),
            1,
            inner as i32,
            1,
            1,
            p as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "aliased softmax");
    stream.synchronize().expect("sync aliased");
    let mut aliased_out = vec![0_f32; numel];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..numel {
        assert_eq!(
            aliased_out[i].to_bits(),
            ref_out[i].to_bits(),
            "f32 in-place softmax @ {i}: aliased={} non-aliased={}",
            aliased_out[i], ref_out[i]
        );
    }
}

#[test]
#[ignore]
fn softmax_f16_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let outer: usize = 4;
    let inner: usize = 128;
    let numel = outer * inner;
    let shape: [i32; 2] = [outer as i32, inner as i32];
    let stride_contig: [i64; 2] = [inner as i64, 1];

    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.017 + 0.1).cos() * 1.4)
        .collect();
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_softmax_f16_run(
            numel as i64, 2, shape.as_ptr(),
            stride_contig.as_ptr(), stride_contig.as_ptr(),
            1, inner as i32, 1, 1,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![f16::ZERO; numel];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_softmax_f16_run(
            numel as i64, 2, shape.as_ptr(),
            stride_contig.as_ptr(), stride_contig.as_ptr(),
            1, inner as i32, 1, 1,
            p as *const c_void,
            p as *mut c_void,
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
            "f16 in-place softmax @ {i}: aliased={} non-aliased={}",
            aliased_out[i].to_f32(), ref_out[i].to_f32());
    }
}

#[test]
#[ignore]
fn softmax_bf16_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let outer: usize = 4;
    let inner: usize = 128;
    let numel = outer * inner;
    let shape: [i32; 2] = [outer as i32, inner as i32];
    let stride_contig: [i64; 2] = [inner as i64, 1];

    let host_x_f32: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.017 + 0.1).cos() * 1.4)
        .collect();
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_softmax_bf16_run(
            numel as i64, 2, shape.as_ptr(),
            stride_contig.as_ptr(), stride_contig.as_ptr(),
            1, inner as i32, 1, 1,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![bf16::ZERO; numel];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_softmax_bf16_run(
            numel as i64, 2, shape.as_ptr(),
            stride_contig.as_ptr(), stride_contig.as_ptr(),
            1, inner as i32, 1, 1,
            p as *const c_void,
            p as *mut c_void,
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
            "bf16 in-place softmax @ {i}");
    }
}

#[test]
#[ignore]
fn softmax_f64_inplace_matches_non_aliased() {
    // Phase 65d-ext: f64 routes through SMEM via `block_reduce_{max,sum}_f64`.
    let (ctx, stream) = setup();
    let outer: usize = 4;
    let inner: usize = 128;
    let numel = outer * inner;
    let shape: [i32; 2] = [outer as i32, inner as i32];
    let stride_contig: [i64; 2] = [inner as i64, 1];

    let host_x: Vec<f64> = (0..numel)
        .map(|i| ((i as f64) * 0.017 + 0.1).cos() * 1.4)
        .collect();

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_softmax_f64_run(
            numel as i64, 2, shape.as_ptr(),
            stride_contig.as_ptr(), stride_contig.as_ptr(),
            1, inner as i32, 1, 1,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "f64 non-aliased softmax");
    stream.synchronize().expect("sync ref");
    let mut ref_out = vec![0_f64; numel];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl ref");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_softmax_f64_run(
            numel as i64, 2, shape.as_ptr(),
            stride_contig.as_ptr(), stride_contig.as_ptr(),
            1, inner as i32, 1, 1,
            p as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "f64 aliased softmax");
    stream.synchronize().expect("sync aliased");
    let mut aliased_out = vec![0_f64; numel];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..numel {
        assert_eq!(aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "f64 in-place softmax @ {i}: aliased={} non-aliased={}",
            aliased_out[i], ref_out[i]);
    }
}
