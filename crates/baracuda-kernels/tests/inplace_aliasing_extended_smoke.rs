//! Direct-FFI smoke tests proving the **Phase 64 extended aliasing
//! contract** for kernel families beyond the Phase 61/62 elementwise
//! set.
//!
//! Phase 61/62 covered the elementwise unary/binary/ternary contract.
//! Phase 64 extends documented aliasing safety to 5 additional families
//! whose kernels are structurally per-thread-isolated (read own cell,
//! write own cell, no cross-thread cell dependencies):
//!
//!   1. **Cast** (same-byte-width casts only)
//!   2. **Where** (ternary conditional select)
//!   3. **Triu / Tril** (triangular mask)
//!   4. **Activation BW** (gradient at a saved value)
//!   5. **Fill** (write-only, trivially in-place)
//!
//! The audit also surfaced 4 families that are NOT aliasing-safe even
//! though their kernel bodies superficially look elementwise: Flip,
//! Roll, Permute, and RoPE. Each of those has two distinct threads
//! touching the same memory cell (one as read, another as write), so
//! same-pointer dispatch is silent corruption. The FFI docstrings now
//! explicitly warn about these — no positive test here, the warning is
//! the deliverable.
//!
//! Tests pick the trailblazer per family + at least one representative
//! variant, exercise the aliased call pattern, and compare to a
//! non-aliased reference run. Equivalence proves the contract holds.
//!
//! Marked `#[ignore]` per project convention; run with
//! `cargo test -p baracuda-kernels --features sm89 \
//!    --test inplace_aliasing_extended_smoke -- --include-ignored`.

#![cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// =========================================================================
// Cast — same byte width is safe; different byte width is unsafe (not
// tested at the contract layer — caller responsibility).
// =========================================================================

/// Cast `f32 -> i32` (same byte width) with `x_ptr == y_ptr` must
/// produce the same result as the non-aliased run.
#[test]
#[ignore]
fn cast_f32_to_i32_aliasing_same_byte_width_safe() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host_f32: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 256.0).collect();

    // Reference: separate buffers.
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_f32).expect("upload f32");
    let mut dev_y: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc i32");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_cast_f32_i32_run(
            numel as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "ref cast f32→i32");
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_i32; numel];
    dev_y.copy_to_host(&mut ref_out).expect("download ref");

    // Aliased: x_ptr == y_ptr (4-byte buffer reinterpreted in-place).
    // Allocate as f32 (since we'll read as f32), reinterpret cast.
    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_f32).expect("upload");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_cast_f32_i32_run(
            numel as i64,
            p as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "aliased cast f32→i32");
    stream.synchronize().expect("sync");

    // Buffer is now i32 in-place. Read back as bytes, reinterpret as i32.
    let mut got_bytes = vec![0_u8; numel * 4];
    let dev_inplace_bytes: &baracuda_driver::DeviceBuffer<f32> = &dev_inplace;
    unsafe {
        let dst_slice = core::slice::from_raw_parts_mut(got_bytes.as_mut_ptr() as *mut f32, numel);
        dev_inplace_bytes.copy_to_host(dst_slice).expect("download bytes");
    }
    let aliased_out: Vec<i32> = (0..numel)
        .map(|i| {
            let mut b = [0_u8; 4];
            b.copy_from_slice(&got_bytes[i * 4..(i + 1) * 4]);
            i32::from_le_bytes(b)
        })
        .collect();

    assert_eq!(aliased_out, ref_out, "cast f32→i32 in-place must match non-aliased reference");
}

// =========================================================================
// Where — a_ptr == y_ptr safe.
// =========================================================================

#[test]
#[ignore]
fn where_f32_aliasing_a_eq_y_safe() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host_cond: Vec<u8> = (0..numel).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
    let host_a: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5).collect();
    let host_b: Vec<f32> = (0..numel).map(|i| -(i as f32) * 0.25).collect();

    // Reference run.
    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("up cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("up a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("up b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_f32_run(
            numel as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f32; numel];
    dev_y.copy_to_host(&mut ref_out).expect("download");

    // Aliased: a_ptr == y_ptr.
    let dev_cond2 = DeviceBuffer::from_slice(&ctx, &host_cond).expect("up cond2");
    let mut dev_a_inplace = DeviceBuffer::from_slice(&ctx, &host_a).expect("up a inp");
    let dev_b2 = DeviceBuffer::from_slice(&ctx, &host_b).expect("up b2");
    let p = dev_a_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_f32_run(
            numel as i64,
            dev_cond2.as_slice().as_raw().0 as *const c_void,
            p as *const c_void,
            dev_b2.as_slice().as_raw().0 as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; numel];
    dev_a_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..numel {
        assert_eq!(aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "where f32 in-place @ {i}: aliased {} vs ref {}", aliased_out[i], ref_out[i]);
    }
}

// =========================================================================
// Triu / Tril — input_ptr == output_ptr safe.
// =========================================================================

#[test]
#[ignore]
fn triu_f32_aliasing_input_eq_output_safe() {
    let (ctx, stream) = setup();
    // 8x8 matrix, diagonal=0 (keep upper including main diag).
    let n = 8;
    let host: Vec<f32> = (0..n * n).map(|i| 1.0 + i as f32).collect();
    let shape = [n as i32, n as i32];
    let diagonal = 0;

    // Reference.
    let dev_in = DeviceBuffer::from_slice(&ctx, &host).expect("up");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n * n).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_triu_f32_run(
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            shape.as_ptr(),
            shape.len() as i32,
            diagonal,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f32; n * n];
    dev_out.copy_to_host(&mut ref_out).expect("dl ref");

    // Aliased.
    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host).expect("up inplace");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_triu_f32_run(
            p as *const c_void,
            p as *mut c_void,
            shape.as_ptr(),
            shape.len() as i32,
            diagonal,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; n * n];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl aliased");

    assert_eq!(aliased_out, ref_out, "triu in-place must match non-aliased reference");
}

#[test]
#[ignore]
fn tril_f32_aliasing_input_eq_output_safe() {
    let (ctx, stream) = setup();
    let n = 8;
    let host: Vec<f32> = (0..n * n).map(|i| 1.0 + i as f32).collect();
    let shape = [n as i32, n as i32];
    let diagonal = 0;

    let dev_in = DeviceBuffer::from_slice(&ctx, &host).expect("up");
    let mut dev_out: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n * n).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_tril_f32_run(
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            shape.as_ptr(),
            shape.len() as i32,
            diagonal,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f32; n * n];
    dev_out.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host).expect("up inplace");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_tril_f32_run(
            p as *const c_void,
            p as *mut c_void,
            shape.as_ptr(),
            shape.len() as i32,
            diagonal,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; n * n];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl aliased");

    assert_eq!(aliased_out, ref_out, "tril in-place must match non-aliased reference");
}

// =========================================================================
// Activation BW — dx_ptr == saved_ptr OR dx_ptr == dy_ptr safe.
//
// Uses ReLU BW (saved-x) as the trailblazer. Pattern generalizes to
// all saved-x and saved-y activation backwards per the FFI docstring.
// =========================================================================

#[test]
#[ignore]
fn unary_relu_backward_f32_aliasing_dx_eq_saved_safe() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 256.0).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| 1.0 + (i as f32) * 0.001).collect();

    // Reference run.
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc dx");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_relu_backward_f32_run(
            numel as i64,
            dev_dy.as_slice().as_raw().0 as *const c_void,
            dev_x.as_slice().as_raw().0 as *const c_void, // saved-x
            dev_dx.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f32; numel];
    dev_dx.copy_to_host(&mut ref_out).expect("dl ref");

    // Aliased: dx_ptr == saved_ptr.
    let mut dev_saved_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up saved inp");
    let dev_dy2 = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy2");
    let p = dev_saved_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_relu_backward_f32_run(
            numel as i64,
            dev_dy2.as_slice().as_raw().0 as *const c_void,
            p as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; numel];
    dev_saved_inplace.copy_to_host(&mut aliased_out).expect("dl aliased");

    for i in 0..numel {
        assert_eq!(aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "relu BW dx==saved @ {i}: aliased {} vs ref {}", aliased_out[i], ref_out[i]);
    }
}

#[test]
#[ignore]
fn unary_relu_backward_f32_aliasing_dx_eq_dy_safe() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 256.0).collect();
    let host_dy: Vec<f32> = (0..numel).map(|i| 1.0 + (i as f32) * 0.001).collect();

    // Reference.
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_relu_backward_f32_run(
            numel as i64,
            dev_dy.as_slice().as_raw().0 as *const c_void,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_dx.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f32; numel];
    dev_dx.copy_to_host(&mut ref_out).expect("dl");

    // Aliased: dx_ptr == dy_ptr.
    let dev_x2 = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_dy_inplace = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy inp");
    let p = dev_dy_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_relu_backward_f32_run(
            numel as i64,
            p as *const c_void,
            dev_x2.as_slice().as_raw().0 as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; numel];
    dev_dy_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..numel {
        assert_eq!(aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "relu BW dx==dy @ {i}: aliased {} vs ref {}", aliased_out[i], ref_out[i]);
    }
}

// =========================================================================
// Fill — trivially in-place (write-only, no aliasing to test).
// Just verify the kernel runs and produces the expected constant.
// =========================================================================

#[test]
#[ignore]
fn fill_f32_writes_constant() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let value = 3.14_f32;

    let mut dev: DeviceBuffer<f32> =
        DeviceBuffer::from_slice(&ctx, &vec![999.0_f32; numel]).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fill_f32_run(
            numel as i64,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            value,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut got = vec![0_f32; numel];
    dev.copy_to_host(&mut got).expect("dl");

    for (i, &x) in got.iter().enumerate() {
        assert_eq!(x.to_bits(), value.to_bits(), "fill @ {i}: got {x} expected {value}");
    }
}
