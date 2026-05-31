//! Direct-FFI smoke tests proving baracuda's **same-pointer aliasing
//! safety contract** on the elementwise trailblazers.
//!
//! Two contracts under test:
//!
//! 1. **Phase 61 (contig)**: aliasing `y` with `x` (unary) /
//!    `a == y` or `b == y` (binary) / any input == y (ternary) on the
//!    contig launchers is safe — each thread reads its own cell once
//!    then writes back. No stride array involved.
//! 2. **Phase 62 (strided)**: same aliasing is safe on the strided
//!    launchers IF AND ONLY IF the aliased input's stride array
//!    equals `stride_y` element-for-element. With unequal strides,
//!    aliasing is UNSAFE (data corruption).
//!
//! These tests pick the trailblazer per family (`unary_neg`,
//! `binary_add`, `ternary_clamp`) at f32, exercise the aliased call
//! pattern, and compare to a non-aliased reference run (separate
//! input + output buffers). Equivalence proves the contract holds for
//! the trailblazer; by the existing "Same contract as ..." inheritance
//! line on every other family member's docstring, the contract holds
//! for those too.
//!
//! Marked `#[ignore]` per project convention; run with
//! `cargo test -p baracuda-kernels --features sm89 \
//!    --test inplace_aliasing_contract_smoke -- --include-ignored`.

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

fn close_enough_f32(a: f32, b: f32) -> bool {
    let tol = b.abs().max(1.0) * 2.0 * f32::EPSILON;
    (a - b).abs() <= tol
}

// =========================================================================
// CONTIG aliasing — Phase 61 contract.
// =========================================================================

/// Unary contig: `x_ptr == y_ptr` must produce the same result as a
/// separately-output `x → y` run.
#[test]
#[ignore]
fn unary_neg_f32_contig_aliasing_safe() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 256.0).collect();

    // Reference: separate x + y buffers.
    let dev_x = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_neg_f32_run(
            numel as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "non-aliased run");
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f32; numel];
    dev_y.copy_to_host(&mut ref_out).expect("download");

    // Aliased: x_ptr == y_ptr (single buffer).
    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_neg_f32_run(
            numel as i64,
            p as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "aliased run");
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; numel];
    dev_inplace.copy_to_host(&mut aliased_out).expect("download");

    for i in 0..numel {
        assert!(
            close_enough_f32(aliased_out[i], ref_out[i]),
            "@{i}: aliased {} vs ref {}", aliased_out[i], ref_out[i]
        );
    }
}

/// Binary contig: `a_ptr == y_ptr` must produce the same result as a
/// non-aliased run.
#[test]
#[ignore]
fn binary_add_f32_contig_aliasing_a_eq_y_safe() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host_a: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.25).collect();
    let host_b: Vec<f32> = (0..numel).map(|i| -(i as f32) * 0.1 + 5.0).collect();

    // Reference.
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_binary_add_f32_run(
            numel as i64,
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
    let mut dev_a_inplace = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b2 = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b2");
    let p = dev_a_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_binary_add_f32_run(
            numel as i64,
            p as *const c_void, // a
            dev_b2.as_slice().as_raw().0 as *const c_void, // b
            p as *mut c_void,   // y == a
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; numel];
    dev_a_inplace.copy_to_host(&mut aliased_out).expect("download");

    for i in 0..numel {
        assert!(close_enough_f32(aliased_out[i], ref_out[i]),
            "@{i}: aliased {} vs ref {}", aliased_out[i], ref_out[i]);
    }
}

/// Binary contig: `b_ptr == y_ptr` symmetric to above.
#[test]
#[ignore]
fn binary_add_f32_contig_aliasing_b_eq_y_safe() {
    let (ctx, stream) = setup();
    let numel = 512;
    let host_a: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5).collect();
    let host_b: Vec<f32> = (0..numel).map(|i| -(i as f32) * 0.2 + 3.0).collect();

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_binary_add_f32_run(
            numel as i64,
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

    let dev_a2 = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a2");
    let mut dev_b_inplace = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let p = dev_b_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_binary_add_f32_run(
            numel as i64,
            dev_a2.as_slice().as_raw().0 as *const c_void,
            p as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; numel];
    dev_b_inplace.copy_to_host(&mut aliased_out).expect("download");

    for i in 0..numel {
        assert!(close_enough_f32(aliased_out[i], ref_out[i]));
    }
}

/// Ternary contig: clamp with `a_ptr == y_ptr` (the input being clamped
/// is also the output).
#[test]
#[ignore]
fn ternary_clamp_f32_contig_aliasing_a_eq_y_safe() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host_a: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 50.0).collect();
    let host_b = vec![-10.0_f32; numel]; // lo
    let host_c = vec![10.0_f32; numel];  // hi

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_ternary_clamp_f32_run(
            numel as i64,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_c.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f32; numel];
    dev_y.copy_to_host(&mut ref_out).expect("download");

    // Aliased: a == y.
    let mut dev_a_inplace = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b2 = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload");
    let dev_c2 = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload");
    let p = dev_a_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_ternary_clamp_f32_run(
            numel as i64,
            p as *const c_void,
            dev_b2.as_slice().as_raw().0 as *const c_void,
            dev_c2.as_slice().as_raw().0 as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; numel];
    dev_a_inplace.copy_to_host(&mut aliased_out).expect("download");

    for i in 0..numel {
        assert!(close_enough_f32(aliased_out[i], ref_out[i]),
            "@{i}: aliased {} vs ref {}", aliased_out[i], ref_out[i]);
    }
}

// =========================================================================
// STRIDED aliasing — Phase 62 contract.
//
// All tests use a rank-2 [8, 16] tensor with stride [32, 1] (padded
// rows). Equal strides on input and output enable safe aliasing per
// the Phase 62 contract.
// =========================================================================

const STRIDED_SHAPE: [i32; 2] = [8, 16];
const STRIDED_STRIDE: [i64; 2] = [32, 1];
const STRIDED_PHYS_LEN: usize = 8 * 32;
const STRIDED_NUMEL: usize = 8 * 16;

fn addressed_offsets() -> Vec<usize> {
    let mut out = Vec::with_capacity(STRIDED_NUMEL);
    for r in 0..STRIDED_SHAPE[0] as usize {
        for c in 0..STRIDED_SHAPE[1] as usize {
            out.push(r * STRIDED_STRIDE[0] as usize + c * STRIDED_STRIDE[1] as usize);
        }
    }
    out
}

#[test]
#[ignore]
fn unary_neg_f32_strided_aliasing_equal_strides_safe() {
    let (ctx, stream) = setup();
    let pad: f32 = 999_999.5;
    let mut host = vec![pad; STRIDED_PHYS_LEN];
    let offsets = addressed_offsets();
    for (i, &off) in offsets.iter().enumerate() {
        host[off] = (i as f32) * 0.5 - 31.0;
    }

    // Reference: separate x + y buffers, both with the same stride.
    let dev_x = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &vec![pad; STRIDED_PHYS_LEN]).expect("alloc y");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_neg_f32_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "non-aliased run");
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f32; STRIDED_PHYS_LEN];
    dev_y.copy_to_host(&mut ref_out).expect("download");

    // Aliased: x == y, same stride.
    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_neg_f32_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            p as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "aliased run");
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; STRIDED_PHYS_LEN];
    dev_inplace.copy_to_host(&mut aliased_out).expect("download");

    // Addressed cells must match the reference run.
    for &off in &offsets {
        assert!(close_enough_f32(aliased_out[off], ref_out[off]),
            "@{off}: aliased {} vs ref {}", aliased_out[off], ref_out[off]);
    }
    // Padding must remain the sentinel value.
    let addressed_set: std::collections::HashSet<usize> = offsets.iter().copied().collect();
    for i in 0..STRIDED_PHYS_LEN {
        if !addressed_set.contains(&i) {
            assert_eq!(aliased_out[i], pad, "padding cell @{i} corrupted");
        }
    }
}

#[test]
#[ignore]
fn binary_add_f32_strided_aliasing_a_eq_y_equal_strides_safe() {
    let (ctx, stream) = setup();
    let pad: f32 = 999_999.5;
    let mut host_a = vec![pad; STRIDED_PHYS_LEN];
    let mut host_b = vec![pad; STRIDED_PHYS_LEN];
    let offsets = addressed_offsets();
    for (i, &off) in offsets.iter().enumerate() {
        host_a[off] = (i as f32) * 0.5 - 31.0;
        host_b[off] = -(i as f32) * 0.25 + 7.0;
    }

    // Reference.
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &vec![pad; STRIDED_PHYS_LEN]).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_binary_add_f32_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f32; STRIDED_PHYS_LEN];
    dev_y.copy_to_host(&mut ref_out).expect("download");

    // Aliased: a == y with stride_a == stride_y.
    let mut dev_a_inplace = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b2 = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload");
    let p = dev_a_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_binary_add_f32_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE.as_ptr(), // stride_a
            STRIDED_STRIDE.as_ptr(), // stride_b
            STRIDED_STRIDE.as_ptr(), // stride_y == stride_a
            p as *const c_void,
            dev_b2.as_slice().as_raw().0 as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; STRIDED_PHYS_LEN];
    dev_a_inplace.copy_to_host(&mut aliased_out).expect("download");

    for &off in &offsets {
        assert!(close_enough_f32(aliased_out[off], ref_out[off]));
    }
    let addressed_set: std::collections::HashSet<usize> = offsets.iter().copied().collect();
    for i in 0..STRIDED_PHYS_LEN {
        if !addressed_set.contains(&i) {
            assert_eq!(aliased_out[i], pad);
        }
    }
}

#[test]
#[ignore]
fn ternary_clamp_f32_strided_aliasing_a_eq_y_equal_strides_safe() {
    let (ctx, stream) = setup();
    let pad: f32 = 999_999.5;
    let mut host_a = vec![pad; STRIDED_PHYS_LEN];
    let offsets = addressed_offsets();
    for (i, &off) in offsets.iter().enumerate() {
        host_a[off] = (i as f32) * 0.5 - 31.0;
    }
    // Broadcast lo/hi via rank-2 [1,1] shape with stride [0,0] —
    // standard pattern. But to keep this test simple we'll use full-rank
    // broadcast tensors of size 1 in each axis, mapped via stride [0,0].
    // Easier: use a full-shape lo/hi buffer constant.
    let host_b = vec![-10.0_f32; STRIDED_PHYS_LEN];
    let host_c = vec![10.0_f32; STRIDED_PHYS_LEN];

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload");
    let dev_c = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::from_slice(&ctx, &vec![pad; STRIDED_PHYS_LEN]).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_ternary_clamp_f32_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_c.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f32; STRIDED_PHYS_LEN];
    dev_y.copy_to_host(&mut ref_out).expect("download");

    // Aliased: a == y, equal strides.
    let mut dev_a_inplace = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload");
    let dev_b2 = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload");
    let dev_c2 = DeviceBuffer::from_slice(&ctx, &host_c).expect("upload");
    let p = dev_a_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_ternary_clamp_f32_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            STRIDED_STRIDE.as_ptr(),
            p as *const c_void,
            dev_b2.as_slice().as_raw().0 as *const c_void,
            dev_c2.as_slice().as_raw().0 as *const c_void,
            p as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; STRIDED_PHYS_LEN];
    dev_a_inplace.copy_to_host(&mut aliased_out).expect("download");

    for &off in &offsets {
        assert!(close_enough_f32(aliased_out[off], ref_out[off]),
            "@{off}: aliased {} vs ref {}", aliased_out[off], ref_out[off]);
    }
    let addressed_set: std::collections::HashSet<usize> = offsets.iter().copied().collect();
    for i in 0..STRIDED_PHYS_LEN {
        if !addressed_set.contains(&i) {
            assert_eq!(aliased_out[i], pad);
        }
    }
}

// =========================================================================
// strides_equal helper integration — confirms the helper Fuel will
// call as the precondition check actually returns the right answer
// for the patterns we test above.
// =========================================================================

#[test]
fn strides_equal_helper_validates_in_place_preconditions() {
    use baracuda_kernels_types::strides_equal;

    // The aliasing test above uses equal strides — strides_equal must agree.
    assert!(strides_equal(&STRIDED_STRIDE, &STRIDED_STRIDE),
        "equal strides — in-place dispatch should be allowed");

    // A transposed view has different strides — must reject.
    let transposed: [i64; 2] = [1, 8];
    assert!(!strides_equal(&STRIDED_STRIDE, &transposed),
        "transposed view — in-place dispatch must NOT be allowed");

    // A broadcast input has zero strides — must reject.
    let broadcast: [i64; 2] = [0, 1];
    assert!(!strides_equal(&STRIDED_STRIDE, &broadcast),
        "broadcast input — in-place dispatch must NOT be allowed");
}
