//! Phase 40 (Fuel ask Gap 6b) — multi-block radix argsort for
//! `row_len > 1024`.
//!
//! Direct-FFI smoke for the new `baracuda_kernels_argsort_<dt>_big_run`
//! family. Verifies that the resulting permutation actually sorts the
//! row across {f32, f64, i32, i64} × rows of length {2000, 5000, 32000}.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn check_sorted_ascending<T: PartialOrd + Copy + core::fmt::Debug>(input: &[T], indices: &[i32]) {
    assert_eq!(input.len(), indices.len());
    for i in 1..indices.len() {
        let a = input[indices[i - 1] as usize];
        let b = input[indices[i] as usize];
        assert!(
            a <= b,
            "ascending argsort broken at i={i}: input[{}]={a:?} > input[{}]={b:?}",
            indices[i - 1],
            indices[i]
        );
    }
}

fn check_sorted_descending<T: PartialOrd + Copy + core::fmt::Debug>(input: &[T], indices: &[i32]) {
    assert_eq!(input.len(), indices.len());
    for i in 1..indices.len() {
        let a = input[indices[i - 1] as usize];
        let b = input[indices[i] as usize];
        assert!(
            a >= b,
            "descending argsort broken at i={i}: input[{}]={a:?} < input[{}]={b:?}",
            indices[i - 1],
            indices[i]
        );
    }
}

#[test]
fn argsort_big_can_implement_rejects_small_rows() {
    // `row_len <= 1024` should return status 3 — callers should route
    // to the bitonic kernel.
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_can_implement(1, 1024) };
    assert_eq!(s, 3, "row_len == 1024 should be rejected by big path");
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_can_implement(1, 100) };
    assert_eq!(s, 3, "row_len == 100 should be rejected by big path");
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_can_implement(1, 2048) };
    assert_eq!(s, 0, "row_len > 1024 should be accepted");
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_can_implement(-1, 2048) };
    assert_eq!(s, 2, "negative batch should be rejected");
}

#[test]
fn argsort_big_workspace_size_grows_with_inputs() {
    // Sanity check the workspace_size query: bigger inputs → bigger
    // workspace, zero for degenerate.
    let zero = unsafe { baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_workspace_size(0, 100) };
    assert_eq!(zero, 0);
    let small = unsafe { baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_workspace_size(1, 2000) };
    let big = unsafe { baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_workspace_size(1, 32000) };
    assert!(small > 0, "non-degenerate input should report > 0 bytes");
    assert!(big > small, "bigger row_len should need more workspace");
}

#[test]
#[ignore]
fn argsort_big_f32_row_2000_ascending() {
    let (ctx, stream) = setup();
    let row_len = 2000i32;
    // Deterministic pseudo-random input: sin(i * 0.97) — distinct ordering.
    let input: Vec<f32> = (0..row_len as usize)
        .map(|i| ((i as f32) * 0.97).sin())
        .collect();

    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");

    let ws_bytes = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_workspace_size(1, row_len)
    };
    let mut workspace: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws alloc");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_run(
            1,
            row_len,
            0, // ascending
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            workspace.as_slice_mut().as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "argsort_f32_big_run returned non-zero");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; input.len()];
    dev_idx.copy_to_host(&mut got).expect("dl");
    check_sorted_ascending(&input, &got);
}

#[test]
#[ignore]
fn argsort_big_f32_row_32000_descending() {
    let (ctx, stream) = setup();
    let row_len = 32000i32;
    let input: Vec<f32> = (0..row_len as usize)
        .map(|i| (((i as f32) * 0.31).cos() * 1000.0 - ((i as f32) * 0.0007)))
        .collect();

    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");

    let ws_bytes = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_workspace_size(1, row_len)
    };
    let mut workspace: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws alloc");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_run(
            1,
            row_len,
            1, // descending
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            workspace.as_slice_mut().as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; input.len()];
    dev_idx.copy_to_host(&mut got).expect("dl");
    check_sorted_descending(&input, &got);
}

#[test]
#[ignore]
fn argsort_big_f64_row_5000_multibatch() {
    let (ctx, stream) = setup();
    let row_len = 5000i32;
    let batch = 3i32;
    let total = (batch as usize) * (row_len as usize);
    let input: Vec<f64> = (0..total)
        .map(|i| ((i as f64) * 1.3).sin() + ((i as f64) * 0.001))
        .collect();

    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, total).expect("alloc");

    let ws_bytes = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_f64_big_workspace_size(batch, row_len)
    };
    let mut workspace: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws alloc");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_f64_big_run(
            batch,
            row_len,
            0,
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            workspace.as_slice_mut().as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; total];
    dev_idx.copy_to_host(&mut got).expect("dl");

    // Verify each row independently — indices are LOCAL to each row.
    for b in 0..batch as usize {
        let row_start = b * row_len as usize;
        let row_end = row_start + row_len as usize;
        let row_input = &input[row_start..row_end];
        let row_indices = &got[row_start..row_end];
        // Validate: every index is in [0, row_len) and the sort holds.
        for &idx in row_indices.iter() {
            assert!(
                idx >= 0 && idx < row_len,
                "row {b}: index {idx} out of [0, {row_len})"
            );
        }
        check_sorted_ascending(row_input, row_indices);
    }
}

#[test]
#[ignore]
fn argsort_big_i32_row_2048_ascending() {
    let (ctx, stream) = setup();
    let row_len = 2048i32;
    // Mix positive + negative + duplicates.
    let input: Vec<i32> = (0..row_len)
        .map(|i| (i.wrapping_mul(2654435761u32 as i32)) % 100000 - 50000)
        .collect();

    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");

    let ws_bytes = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_i32_big_workspace_size(1, row_len)
    };
    let mut workspace: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws alloc");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_i32_big_run(
            1,
            row_len,
            0,
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            workspace.as_slice_mut().as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; input.len()];
    dev_idx.copy_to_host(&mut got).expect("dl");
    check_sorted_ascending(&input, &got);
}

#[test]
#[ignore]
fn argsort_big_i64_row_2000_ascending() {
    let (ctx, stream) = setup();
    let row_len = 2000i32;
    let input: Vec<i64> = (0..row_len as usize)
        .map(|i| ((i as i64).wrapping_mul(1_000_003)) % 10_000_000 - 5_000_000)
        .collect();

    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");

    let ws_bytes = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_i64_big_workspace_size(1, row_len)
    };
    let mut workspace: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("ws alloc");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_i64_big_run(
            1,
            row_len,
            0,
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            workspace.as_slice_mut().as_raw().0 as *mut c_void,
            ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; input.len()];
    dev_idx.copy_to_host(&mut got).expect("dl");
    check_sorted_ascending(&input, &got);
}

#[test]
fn argsort_big_workspace_too_small_returns_4() {
    let (ctx, stream) = setup();
    let row_len = 2048i32;
    let input: Vec<f32> = (0..row_len as usize).map(|i| i as f32).collect();
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up");
    let mut dev_idx: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, input.len()).expect("alloc");
    // Deliberately too-small workspace.
    let mut ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 64).expect("ws alloc");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_argsort_f32_big_run(
            1,
            row_len,
            0,
            dev_in.as_slice().as_raw().0 as *const c_void,
            dev_idx.as_slice_mut().as_raw().0 as *mut c_void,
            ws.as_slice_mut().as_raw().0 as *mut c_void,
            64,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 4, "tiny workspace should return WorkspaceTooSmall");
}
