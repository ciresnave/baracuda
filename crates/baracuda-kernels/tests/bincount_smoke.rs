//! Real-GPU smoke test for `BincountPlan` (Phase 9, Category O).
//!
//! Validates the FFI surface (`baracuda_kernels_bincount_{i32,i64}_run`)
//! against a CPU reference for small fixtures. Covers:
//!  - Basic counts (in-range values bucket correctly)
//!  - Out-of-range (`< 0` or `>= num_bins`) silently dropped contract
//!  - Empty input (n = 0) returns zero-filled output
//!
//! `#[ignore]` per project convention; run with
//! `cargo test -p baracuda-kernels --features sm89 \
//!    --test bincount_smoke -- --include-ignored`.

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

fn host_bincount_i32(x: &[i32], num_bins: i32) -> Vec<i32> {
    let mut counts = vec![0_i32; num_bins as usize];
    for &v in x {
        if v >= 0 && v < num_bins {
            counts[v as usize] += 1;
        }
    }
    counts
}

fn host_bincount_i64(x: &[i64], num_bins: i32) -> Vec<i32> {
    let mut counts = vec![0_i32; num_bins as usize];
    for &v in x {
        if v >= 0 && (v as i64) < num_bins as i64 {
            counts[v as usize] += 1;
        }
    }
    counts
}

#[test]
#[ignore]
fn bincount_i32_basic() {
    let (ctx, stream) = setup();
    let num_bins: i32 = 16;
    // Mixed: in-range + out-of-range + duplicates.
    let host_x: Vec<i32> = vec![0, 1, 1, 5, 5, 5, 15, 16, -1, 100, 3, 3];
    let expected = host_bincount_i32(&host_x, num_bins);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_out: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, num_bins as usize).expect("alloc");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_bincount_i32_run(
            host_x.len() as i64,
            num_bins,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "bincount_i32 status");
    stream.synchronize().expect("sync");
    let mut got = vec![0_i32; num_bins as usize];
    dev_out.copy_to_host(&mut got).expect("dl");

    assert_eq!(got, expected, "bincount_i32 result mismatch");
}

#[test]
#[ignore]
fn bincount_i32_empty_input() {
    let (ctx, stream) = setup();
    let num_bins: i32 = 8;
    let host_x: Vec<i32> = vec![];

    let dev_x: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc");
    let mut dev_out: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, num_bins as usize).expect("alloc");
    // Pre-fill output to verify the kernel observes n=0 and doesn't touch memory.
    let presets = vec![7_i32; num_bins as usize];
    dev_out.copy_from_host(&presets).expect("preset");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_bincount_i32_run(
            host_x.len() as i64,
            num_bins,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "bincount_i32 n=0 status");
    stream.synchronize().expect("sync");
    let mut got = vec![0_i32; num_bins as usize];
    dev_out.copy_to_host(&mut got).expect("dl");

    // Contract: n=0 leaves output untouched (callers are responsible
    // for zeroing if they want zeros). Our preset survives.
    assert_eq!(got, presets, "bincount_i32 n=0 should not write to output");
}

#[test]
#[ignore]
fn bincount_i64_basic() {
    let (ctx, stream) = setup();
    let num_bins: i32 = 32;
    let host_x: Vec<i64> = vec![
        0, 0, 0, 5, 10, 15, 20, 25, 31, 32, -5, 1000, 7, 7, 7, 7,
    ];
    let expected = host_bincount_i64(&host_x, num_bins);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_out: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, num_bins as usize).expect("alloc");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_bincount_i64_run(
            host_x.len() as i64,
            num_bins,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "bincount_i64 status");
    stream.synchronize().expect("sync");
    let mut got = vec![0_i32; num_bins as usize];
    dev_out.copy_to_host(&mut got).expect("dl");

    assert_eq!(got, expected, "bincount_i64 result mismatch");
}

#[test]
#[ignore]
fn bincount_i32_all_in_range_dense() {
    let (ctx, stream) = setup();
    let num_bins: i32 = 64;
    // Stress test: every bin gets exactly 3 entries.
    let host_x: Vec<i32> = (0..num_bins)
        .flat_map(|b| std::iter::repeat(b).take(3))
        .collect();
    let expected: Vec<i32> = vec![3; num_bins as usize];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_out: DeviceBuffer<i32> =
        DeviceBuffer::zeros(&ctx, num_bins as usize).expect("alloc");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_bincount_i32_run(
            host_x.len() as i64,
            num_bins,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut got = vec![0_i32; num_bins as usize];
    dev_out.copy_to_host(&mut got).expect("dl");

    assert_eq!(got, expected, "bincount dense pattern");
}
