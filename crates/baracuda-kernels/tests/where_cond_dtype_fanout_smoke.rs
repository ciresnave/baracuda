//! Phase 38 (Fuel 6c.4 Gap 3) — direct-FFI smoke tests for the
//! `where_cond` dtype-matrix fanout.
//!
//! Covers a representative spread of (cond_dtype, value_dtype) pairs
//! across the 58 new symbol pairs landed in `where_dtype_fanout.cu`:
//!
//!   - (a) U32 / I64 cond × {f32, f16, bf16, f64} — 1 contig per cell.
//!   - (b) U8 / U32 / I64 cond × {u8, i8, u32, i16, i32, i64} — at
//!         least one contig per cond dtype (mixed value coverage), plus
//!         one strided case per cond dtype.
//!   - (c) U8-cond × Fp8E4M3 — single contig (treated as raw u8 bytes).
//!
//! Validation is bit-exact against a Rust CPU reference because the op
//! is pure element selection — no arithmetic. The reference is
//! `cond.iter().zip(a).zip(b).map(|((c, a), b)| if *c != 0 { *a } else { *b })`.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!    --test where_cond_dtype_fanout_smoke -- --ignored`.

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

// ----------------------------------------------------------------------------
// (a) U32 / I64 cond × {f32, f16, bf16, f64}
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn ffi_where_u32cond_f32_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let n: usize = 4096;
    // u32 cond: every 3rd cell zero (false), rest non-zero (true).
    let host_cond: Vec<u32> = (0..n).map(|i| if i % 3 == 0 { 0 } else { 7u32.wrapping_mul(i as u32) }).collect();
    let host_a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125 - 32.0).collect();
    let host_b: Vec<f32> = (0..n).map(|i| (i as f32) * -0.0625 + 17.5).collect();
    let expected: Vec<f32> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_u32cond_f32_run(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "where_u32cond_f32 returned status {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "u32cond_f32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn ffi_where_i64cond_f64_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let n: usize = 2048;
    // i64 cond: mix of negative, zero, and positive — non-zero = true.
    let host_cond: Vec<i64> = (0..n)
        .map(|i| {
            let s = if i % 2 == 0 { 1 } else { -1 };
            if i % 5 == 0 { 0 } else { s * (i as i64 + 1) }
        })
        .collect();
    let host_a: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let host_b: Vec<f64> = (0..n).map(|i| (i as f64) * 0.25 + 100.0).collect();
    let expected: Vec<f64> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_i64cond_f64_run(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "i64cond_f64 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn ffi_where_u32cond_f16_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let n: usize = 2048;
    let host_cond: Vec<u32> = (0..n).map(|i| (i as u32) % 4).collect(); // 0,1,2,3,...
    let host_a: Vec<f16> = (0..n).map(|i| f16::from_f32((i as f32) * 0.0125 - 10.0)).collect();
    let host_b: Vec<f16> = (0..n).map(|i| f16::from_f32((i as f32) * 0.00625 - 5.0)).collect();
    let expected: Vec<f16> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_u32cond_f16_run(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "u32cond_f16 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn ffi_where_i64cond_bf16_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let n: usize = 2048;
    let host_cond: Vec<i64> = (0..n).map(|i| if i % 4 < 2 { 0 } else { -(i as i64) }).collect();
    let host_a: Vec<bf16> = (0..n).map(|i| bf16::from_f32((i as f32) * 0.0125 - 10.0)).collect();
    let host_b: Vec<bf16> = (0..n).map(|i| bf16::from_f32((i as f32) * 0.00625 - 5.0)).collect();
    let expected: Vec<bf16> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_i64cond_bf16_run(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "i64cond_bf16 mismatch @ {i}");
    }
}

// ----------------------------------------------------------------------------
// (b) Integer value dtype coverage — at least one contig per cond
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn ffi_where_u8cond_i32_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let n: usize = 4096;
    let host_cond: Vec<u8> = (0..n).map(|i| if i % 2 == 0 { 1u8 } else { 0u8 }).collect();
    let host_a: Vec<i32> = (0..n).map(|i| (i as i32) - 2048).collect();
    let host_b: Vec<i32> = (0..n).map(|i| -(i as i32) * 3).collect();
    let expected: Vec<i32> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_u8cond_i32_run(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "u8cond_i32 mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn ffi_where_u8cond_i64_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let n: usize = 1024;
    let host_cond: Vec<u8> = (0..n).map(|i| (i % 7) as u8).collect();
    let host_a: Vec<i64> = (0..n).map(|i| (i as i64) * 7919 - 1_000_000).collect();
    let host_b: Vec<i64> = (0..n).map(|i| -(i as i64) * 3).collect();
    let expected: Vec<i64> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<i64> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_u8cond_i64_run(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i64; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "u8cond_i64 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn ffi_where_u32cond_u32_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let n: usize = 2048;
    let host_cond: Vec<u32> = (0..n).map(|i| if i % 3 == 0 { 0 } else { (i as u32) | 1 }).collect();
    let host_a: Vec<u32> = (0..n).map(|i| (i as u32).wrapping_mul(2654435761)).collect();
    let host_b: Vec<u32> = (0..n).map(|i| (i as u32) * 7).collect();
    let expected: Vec<u32> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u32> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_u32cond_u32_run(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0u32; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "u32cond_u32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn ffi_where_i64cond_i16_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let n: usize = 2048;
    let host_cond: Vec<i64> = (0..n).map(|i| (i as i64) - 1024).collect(); // negative, zero, positive
    let host_a: Vec<i16> = (0..n).map(|i| (i as i16).wrapping_mul(3)).collect();
    let host_b: Vec<i16> = (0..n).map(|i| -(i as i16).wrapping_mul(7)).collect();
    let expected: Vec<i16> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<i16> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_i64cond_i16_run(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i16; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "i64cond_i16 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn ffi_where_u32cond_i8_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let n: usize = 1024;
    let host_cond: Vec<u32> = (0..n).map(|i| if i % 5 < 3 { (i as u32) + 1 } else { 0 }).collect();
    let host_a: Vec<i8> = (0..n).map(|i| (i as i8).wrapping_sub(64)).collect();
    let host_b: Vec<i8> = (0..n).map(|i| (i as i8).wrapping_mul(3)).collect();
    let expected: Vec<i8> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<i8> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_u32cond_i8_run(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i8; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "u32cond_i8 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn ffi_where_u8cond_u8_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let n: usize = 512;
    let host_cond: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
    let host_a: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(13)).collect();
    let host_b: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_add(7)).collect();
    let expected: Vec<u8> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_u8cond_u8_run(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0u8; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "u8cond_u8 mismatch @ {i}");
    }
}

// ----------------------------------------------------------------------------
// Strided / broadcast smoke — one per cond dtype
// ----------------------------------------------------------------------------
//
// Pattern: 2-D `[rows, cols]` shape with cond broadcast across cols
// (`stride_cond = [1, 0]` — one mask value per row) and a / b laid out
// contiguously (`stride = [cols, 1]`).

#[test]
#[ignore]
fn ffi_where_u8cond_f32_strided_per_row_mask() {
    let (ctx, stream) = setup();
    let rows: i32 = 32;
    let cols: i32 = 64;
    let numel: i64 = (rows as i64) * (cols as i64);
    let n: usize = numel as usize;

    // One cond value per row (cond shape = [rows, 1]).
    let host_cond: Vec<u8> = (0..rows as usize).map(|r| if r % 2 == 0 { 1u8 } else { 0u8 }).collect();
    let host_a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 50.0).collect();
    let host_b: Vec<f32> = (0..n).map(|i| (i as f32) * -0.25 + 25.0).collect();
    let mut expected = vec![0f32; n];
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            let idx = r * (cols as usize) + c;
            expected[idx] = if host_cond[r] != 0 { host_a[idx] } else { host_b[idx] };
        }
    }

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let shape: [i32; 2] = [rows, cols];
    // cond is broadcast across cols: stride [1, 0].
    let stride_cond: [i64; 2] = [1, 0];
    let stride_ab: [i64; 2] = [cols as i64, 1];

    // We need the U8-cond strided variant of where_f32 — that already
    // exists as `where_f32_strided_run` from the original where_fp.cu
    // family. This test cross-checks that the new fanout file did not
    // perturb the original kernel template instantiation.
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_f32_strided_run(
            numel,
            shape.len() as i32,
            shape.as_ptr(),
            stride_cond.as_ptr(),
            stride_ab.as_ptr(),
            stride_ab.as_ptr(),
            stride_ab.as_ptr(),
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "u8cond_f32 strided mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn ffi_where_u32cond_i32_strided_per_row_mask() {
    let (ctx, stream) = setup();
    let rows: i32 = 32;
    let cols: i32 = 32;
    let numel: i64 = (rows as i64) * (cols as i64);
    let n: usize = numel as usize;

    let host_cond: Vec<u32> = (0..rows as usize).map(|r| if r % 3 == 0 { 0 } else { (r as u32) + 1 }).collect();
    let host_a: Vec<i32> = (0..n).map(|i| (i as i32) - 500).collect();
    let host_b: Vec<i32> = (0..n).map(|i| -(i as i32) * 2).collect();
    let mut expected = vec![0i32; n];
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            let idx = r * (cols as usize) + c;
            expected[idx] = if host_cond[r] != 0 { host_a[idx] } else { host_b[idx] };
        }
    }

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let shape: [i32; 2] = [rows, cols];
    let stride_cond: [i64; 2] = [1, 0];
    let stride_ab: [i64; 2] = [cols as i64, 1];

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_u32cond_i32_strided_run(
            numel,
            shape.len() as i32,
            shape.as_ptr(),
            stride_cond.as_ptr(),
            stride_ab.as_ptr(),
            stride_ab.as_ptr(),
            stride_ab.as_ptr(),
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "u32cond_i32 strided mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn ffi_where_i64cond_f32_strided_per_row_mask() {
    let (ctx, stream) = setup();
    let rows: i32 = 16;
    let cols: i32 = 48;
    let numel: i64 = (rows as i64) * (cols as i64);
    let n: usize = numel as usize;

    let host_cond: Vec<i64> = (0..rows as usize)
        .map(|r| if r % 4 == 0 { 0 } else { -(r as i64) - 1 })
        .collect();
    let host_a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125).collect();
    let host_b: Vec<f32> = (0..n).map(|i| (i as f32) * -0.0625).collect();
    let mut expected = vec![0f32; n];
    for r in 0..rows as usize {
        for c in 0..cols as usize {
            let idx = r * (cols as usize) + c;
            expected[idx] = if host_cond[r] != 0 { host_a[idx] } else { host_b[idx] };
        }
    }

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let shape: [i32; 2] = [rows, cols];
    let stride_cond: [i64; 2] = [1, 0];
    let stride_ab: [i64; 2] = [cols as i64, 1];

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_i64cond_f32_strided_run(
            numel,
            shape.len() as i32,
            shape.as_ptr(),
            stride_cond.as_ptr(),
            stride_ab.as_ptr(),
            stride_ab.as_ptr(),
            stride_ab.as_ptr(),
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "i64cond_f32 strided mismatch @ {i}");
    }
}

// ----------------------------------------------------------------------------
// (c) Fp8E4M3 — 1-byte storage, treated as raw u8
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn ffi_where_u8cond_fp8e4m3_matches_cpu_ref() {
    let (ctx, stream) = setup();
    let n: usize = 1024;
    // Use raw u8 host buffers — Fp8E4M3 is just a byte at the FFI level.
    let host_cond: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();
    let host_a: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(11)).collect();
    let host_b: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_add(99)).collect();
    let expected: Vec<u8> = host_cond
        .iter()
        .zip(host_a.iter())
        .zip(host_b.iter())
        .map(|((&c, &a), &b)| if c != 0 { a } else { b })
        .collect();

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, n).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_u8cond_fp8e4m3_run(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0u8; n];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "u8cond_fp8e4m3 mismatch @ {i}");
    }
}

// ----------------------------------------------------------------------------
// `_can_implement` smoke — exercises the validation companions
// ----------------------------------------------------------------------------

#[test]
#[ignore]
fn ffi_where_u32cond_f32_can_implement_accepts_valid() {
    let (ctx, _stream) = setup();
    let n: usize = 64;
    let host_cond: Vec<u32> = vec![0; n];
    let host_a: Vec<f32> = vec![0.0; n];
    let host_b: Vec<f32> = vec![0.0; n];
    let host_y: Vec<f32> = vec![0.0; n];

    let dev_cond = DeviceBuffer::from_slice(&ctx, &host_cond).expect("upload cond");
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let dev_y = DeviceBuffer::from_slice(&ctx, &host_y).expect("upload y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_where_u32cond_f32_can_implement(
            n as i64,
            dev_cond.as_slice().as_raw().0 as *const c_void,
            dev_a.as_slice().as_raw().0 as *const c_void,
            dev_b.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice().as_raw().0 as *const c_void,
        )
    };
    assert_eq!(status, 0, "can_implement returned status {status}");
}
