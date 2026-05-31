//! Direct-FFI smoke tests for the `affine_inplace_*` family.
//!
//! Covers:
//!   - **Phase 61** alpha.55 baseline (f32, f64) + alpha.61 half-precision
//!     additions (bf16, f16) — contig in-place affine. Test backfill;
//!     Phase 61 shipped without these tests because the kernels are
//!     structurally clones of the validated forward `affine_<dt>_run`
//!     family. Phase 62 lands the test coverage as part of the broader
//!     "tests for as many things as possible" sweep.
//!   - **Phase 62** int dtype contig backfill (i32, i64, u8, i8) +
//!     strided in-place for all 7 dtypes (f32, f64, i32, i64, u8, bf16,
//!     f16) matching the forward `affine_<dt>_strided_run` matrix.
//!
//! Each test calls the raw `baracuda_kernels_sys::baracuda_kernels_affine_inplace_*`
//! symbol directly with raw device pointers (no plan-layer wrapping),
//! mirroring how Fuel's in-place op family executor will dispatch.
//!
//! Marked `#[ignore]` per project convention; run with
//! `cargo test -p baracuda-kernels --features sm89 \
//!    --test affine_inplace_ffi_smoke -- --include-ignored`.

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

// =========================================================================
// Phase 61 contig in-place — backfill tests for the alpha.55 baseline
// (f32 / f64) and alpha.61 half-precision additions (bf16 / f16).
// =========================================================================

#[test]
#[ignore]
fn affine_inplace_f32_contig() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let scale: f32 = 2.5;
    let offset: f32 = -1.25;
    let expected: Vec<f32> = host.iter().map(|&x| scale * x + offset).collect();

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_f32_run(
            numel as i64,
            scale,
            offset,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "affine_inplace_f32_run status");
    stream.synchronize().expect("sync");

    let mut got = vec![0_f32; numel];
    dev.copy_to_host(&mut got).expect("download");

    let tol_eps = 2.0 * f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = e.abs().max(1.0) * tol_eps;
        assert!((g - e).abs() <= tol, "@{i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn affine_inplace_f64_contig() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.125 + 2.0).collect();
    let scale: f64 = -3.0;
    let offset: f64 = 0.75;
    let expected: Vec<f64> = host.iter().map(|&x| scale * x + offset).collect();

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_f64_run(
            numel as i64,
            scale,
            offset,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0_f64; numel];
    dev.copy_to_host(&mut got).expect("download");

    let tol_eps = 2.0 * f64::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = e.abs().max(1.0) * tol_eps;
        assert!((g - e).abs() <= tol, "@{i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn affine_inplace_bf16_contig() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32) * 0.1 - 5.0))
        .collect();
    let scale: f32 = 1.5;
    let offset: f32 = 0.25;
    // Reference: cast to f32, do math, cast back to bf16 (matches kernel pattern).
    let expected: Vec<bf16> = host
        .iter()
        .map(|&x| bf16::from_f32(scale * x.to_f32() + offset))
        .collect();

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_bf16_run(
            numel as i64,
            scale,
            offset,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::ZERO; numel];
    dev.copy_to_host(&mut got).expect("download");

    // bf16 has 8 mantissa bits → ~0.5% relative tolerance per
    // mul-then-add chain. Use 2 ulps (~1.5e-2 relative).
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let err = (gf - ef).abs();
        let tol = ef.abs().max(1.0) * 1.5e-2;
        assert!(err <= tol, "@{i}: got {gf} expected {ef}");
    }
}

#[test]
#[ignore]
fn affine_inplace_f16_contig() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32) * 0.05 - 2.5))
        .collect();
    let scale: f32 = 2.0;
    let offset: f32 = -0.5;
    let expected: Vec<f16> = host
        .iter()
        .map(|&x| f16::from_f32(scale * x.to_f32() + offset))
        .collect();

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_f16_run(
            numel as i64,
            scale,
            offset,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; numel];
    dev.copy_to_host(&mut got).expect("download");

    // f16 has 10 mantissa bits → ~0.1% relative tolerance.
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let err = (gf - ef).abs();
        let tol = ef.abs().max(1.0) * 2e-3;
        assert!(err <= tol, "@{i}: got {gf} expected {ef}");
    }
}

// =========================================================================
// Phase 62 contig in-place — int dtype backfill (i32 / i64 / u8 / i8).
// Int math wraps per C++20 two's-complement modular semantics; tests
// pick scales/offsets that DON'T overflow within the chosen range so
// reference and GPU match bit-exactly.
// =========================================================================

#[test]
#[ignore]
fn affine_inplace_i32_contig() {
    let (ctx, stream) = setup();
    let numel = 1024;
    let host: Vec<i32> = (0..numel as i32).map(|i| i - 512).collect();
    let scale: i32 = 3;
    let offset: i32 = 100;
    let expected: Vec<i32> = host.iter().map(|&x| scale * x + offset).collect();

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_i32_run(
            numel as i64,
            scale,
            offset,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0_i32; numel];
    dev.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "i32 in-place affine bit-exact");
}

#[test]
#[ignore]
fn affine_inplace_i64_contig() {
    let (ctx, stream) = setup();
    let numel = 512;
    let host: Vec<i64> = (0..numel as i64).map(|i| (i - 256) * 1_000_000).collect();
    let scale: i64 = 5;
    let offset: i64 = -42;
    let expected: Vec<i64> = host.iter().map(|&x| scale * x + offset).collect();

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_i64_run(
            numel as i64,
            scale,
            offset,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0_i64; numel];
    dev.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "i64 in-place affine bit-exact");
}

#[test]
#[ignore]
fn affine_inplace_u8_contig() {
    let (ctx, stream) = setup();
    let numel: usize = 256;
    // (0..256) iterated as u8 would loop forever; iterate as i32 and cast.
    // Wrap so values cover the full u8 range (0..=255 → wraps back to 0..255).
    let host: Vec<u8> = (0..numel).map(|i| (i & 0xFF) as u8).collect();
    // Wrap-friendly: scale=2, offset=1 → max output = 2*255 + 1 = 511 → wraps to 255.
    // Use wrapping arithmetic in the reference to match the GPU's modular behavior.
    let scale: u8 = 2;
    let offset: u8 = 1;
    let expected: Vec<u8> = host
        .iter()
        .map(|&x| scale.wrapping_mul(x).wrapping_add(offset))
        .collect();

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_u8_run(
            numel as i64,
            scale,
            offset,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0_u8; numel];
    dev.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "u8 in-place affine wraps consistently");
}

#[test]
#[ignore]
fn affine_inplace_i8_contig() {
    let (ctx, stream) = setup();
    let numel = 256;
    let host: Vec<i8> = (0..numel as i32).map(|i| (i - 128) as i8).collect();
    let scale: i8 = 2;
    let offset: i8 = 3;
    let expected: Vec<i8> = host
        .iter()
        .map(|&x| scale.wrapping_mul(x).wrapping_add(offset))
        .collect();

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_i8_run(
            numel as i64,
            scale,
            offset,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0_i8; numel];
    dev.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "i8 in-place affine wraps consistently");
}

// =========================================================================
// Phase 62 strided in-place — full 7-dtype matrix.
//
// Fixture: rank-2 [8, 16] tensor laid out non-contiguously via a custom
// stride that's still a valid permutation. We use stride [32, 1] over
// a [8, 16] shape (i.e., row-stride 32 over a 16-element row — the
// extra 16 elements per row are "padding" that the kernel must NOT
// touch). The test verifies the in-place op rewrites the addressed
// cells AND leaves the padding cells untouched.
// =========================================================================

const STRIDED_SHAPE: [i32; 2] = [8, 16];
const STRIDED_STRIDE_Y: [i64; 2] = [32, 1]; // padded-row layout
const STRIDED_PHYS_LEN: usize = 8 * 32;     // 256 cells incl. padding
const STRIDED_NUMEL: usize = 8 * 16;        // 128 addressed cells

// Compute the linearized physical offset of every addressed cell for
// the [8, 16] view with stride [32, 1] — used to construct expected
// output buffers + masks of "touched" vs "untouched" cells.
fn addressed_offsets() -> Vec<usize> {
    let mut out = Vec::with_capacity(STRIDED_NUMEL);
    for r in 0..STRIDED_SHAPE[0] as usize {
        for c in 0..STRIDED_SHAPE[1] as usize {
            out.push(r * STRIDED_STRIDE_Y[0] as usize + c * STRIDED_STRIDE_Y[1] as usize);
        }
    }
    out
}

#[test]
#[ignore]
fn affine_inplace_strided_f32() {
    let (ctx, stream) = setup();
    // Sentinel pattern in the padding so we can verify it's untouched.
    let pad_sentinel: f32 = 999_999.5;
    let mut host = vec![pad_sentinel; STRIDED_PHYS_LEN];
    let offsets = addressed_offsets();
    for (i, &off) in offsets.iter().enumerate() {
        host[off] = (i as f32) * 0.5 - 31.0;
    }

    let scale: f32 = -1.5;
    let offset_c: f32 = 7.25;
    let mut expected = host.clone();
    for &off in &offsets {
        expected[off] = scale * host[off] + offset_c;
    }

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_f32_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE_Y.as_ptr(),
            scale,
            offset_c,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0_f32; STRIDED_PHYS_LEN];
    dev.copy_to_host(&mut got).expect("download");

    // Addressed cells must match the affine; padding must stay untouched.
    let tol_eps = 2.0 * f32::EPSILON;
    for i in 0..STRIDED_PHYS_LEN {
        let tol = expected[i].abs().max(1.0) * tol_eps;
        assert!(
            (got[i] - expected[i]).abs() <= tol,
            "@{i}: got {} expected {}", got[i], expected[i]
        );
    }
    // Explicit padding-untouched check (redundant w/ above but makes
    // the intent obvious if the test ever regresses).
    let addressed_set: std::collections::HashSet<usize> = offsets.iter().copied().collect();
    for i in 0..STRIDED_PHYS_LEN {
        if !addressed_set.contains(&i) {
            assert_eq!(got[i], pad_sentinel, "padding cell @{i} was modified");
        }
    }
}

#[test]
#[ignore]
fn affine_inplace_strided_f64() {
    let (ctx, stream) = setup();
    let pad_sentinel: f64 = 999_999.5;
    let mut host = vec![pad_sentinel; STRIDED_PHYS_LEN];
    let offsets = addressed_offsets();
    for (i, &off) in offsets.iter().enumerate() {
        host[off] = (i as f64) * 0.125 + 2.0;
    }
    let scale: f64 = 0.5;
    let offset_c: f64 = -1.0;
    let mut expected = host.clone();
    for &off in &offsets {
        expected[off] = scale * host[off] + offset_c;
    }

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_f64_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE_Y.as_ptr(),
            scale,
            offset_c,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0_f64; STRIDED_PHYS_LEN];
    dev.copy_to_host(&mut got).expect("download");

    let tol_eps = 2.0 * f64::EPSILON;
    for i in 0..STRIDED_PHYS_LEN {
        let tol = expected[i].abs().max(1.0) * tol_eps;
        assert!((got[i] - expected[i]).abs() <= tol, "@{i}: got {} expected {}", got[i], expected[i]);
    }
}

#[test]
#[ignore]
fn affine_inplace_strided_i32() {
    let (ctx, stream) = setup();
    let pad_sentinel: i32 = i32::MAX;
    let mut host = vec![pad_sentinel; STRIDED_PHYS_LEN];
    let offsets = addressed_offsets();
    for (i, &off) in offsets.iter().enumerate() {
        host[off] = (i as i32) - 64;
    }
    let scale: i32 = 4;
    let offset_c: i32 = 17;
    let mut expected = host.clone();
    for &off in &offsets {
        expected[off] = scale * host[off] + offset_c;
    }

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_i32_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE_Y.as_ptr(),
            scale,
            offset_c,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0_i32; STRIDED_PHYS_LEN];
    dev.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "i32 strided in-place + padding untouched");
}

#[test]
#[ignore]
fn affine_inplace_strided_i64() {
    let (ctx, stream) = setup();
    let pad_sentinel: i64 = i64::MAX;
    let mut host = vec![pad_sentinel; STRIDED_PHYS_LEN];
    let offsets = addressed_offsets();
    for (i, &off) in offsets.iter().enumerate() {
        host[off] = (i as i64) * 1_000;
    }
    let scale: i64 = 7;
    let offset_c: i64 = -3;
    let mut expected = host.clone();
    for &off in &offsets {
        expected[off] = scale * host[off] + offset_c;
    }

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_i64_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE_Y.as_ptr(),
            scale,
            offset_c,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0_i64; STRIDED_PHYS_LEN];
    dev.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "i64 strided in-place + padding untouched");
}

#[test]
#[ignore]
fn affine_inplace_strided_u8() {
    let (ctx, stream) = setup();
    let pad_sentinel: u8 = u8::MAX;
    let mut host = vec![pad_sentinel; STRIDED_PHYS_LEN];
    let offsets = addressed_offsets();
    for (i, &off) in offsets.iter().enumerate() {
        host[off] = (i as u8).wrapping_mul(2);
    }
    let scale: u8 = 2;
    let offset_c: u8 = 5;
    let mut expected = host.clone();
    for &off in &offsets {
        expected[off] = scale.wrapping_mul(host[off]).wrapping_add(offset_c);
    }

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_u8_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE_Y.as_ptr(),
            scale,
            offset_c,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0_u8; STRIDED_PHYS_LEN];
    dev.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "u8 strided in-place wraps");
}

#[test]
#[ignore]
fn affine_inplace_strided_bf16() {
    let (ctx, stream) = setup();
    let pad_sentinel = bf16::from_f32(9999.0);
    let mut host = vec![pad_sentinel; STRIDED_PHYS_LEN];
    let offsets = addressed_offsets();
    for (i, &off) in offsets.iter().enumerate() {
        host[off] = bf16::from_f32((i as f32) * 0.1 - 6.0);
    }
    let scale: f32 = 1.25;
    let offset_c: f32 = 0.5;
    let mut expected = host.clone();
    for &off in &offsets {
        expected[off] = bf16::from_f32(scale * host[off].to_f32() + offset_c);
    }

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_bf16_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE_Y.as_ptr(),
            scale,
            offset_c,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::ZERO; STRIDED_PHYS_LEN];
    dev.copy_to_host(&mut got).expect("download");

    for i in 0..STRIDED_PHYS_LEN {
        let gf = got[i].to_f32();
        let ef = expected[i].to_f32();
        let tol = ef.abs().max(1.0) * 1.5e-2; // bf16 2-ulp
        assert!((gf - ef).abs() <= tol, "@{i}: got {gf} expected {ef}");
    }
}

#[test]
#[ignore]
fn affine_inplace_strided_f16() {
    let (ctx, stream) = setup();
    let pad_sentinel = f16::from_f32(99.0);
    let mut host = vec![pad_sentinel; STRIDED_PHYS_LEN];
    let offsets = addressed_offsets();
    for (i, &off) in offsets.iter().enumerate() {
        host[off] = f16::from_f32((i as f32) * 0.05 - 3.0);
    }
    let scale: f32 = 1.5;
    let offset_c: f32 = -0.25;
    let mut expected = host.clone();
    for &off in &offsets {
        expected[off] = f16::from_f32(scale * host[off].to_f32() + offset_c);
    }

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_f16_strided_run(
            STRIDED_NUMEL as i64,
            STRIDED_SHAPE.len() as i32,
            STRIDED_SHAPE.as_ptr(),
            STRIDED_STRIDE_Y.as_ptr(),
            scale,
            offset_c,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; STRIDED_PHYS_LEN];
    dev.copy_to_host(&mut got).expect("download");

    for i in 0..STRIDED_PHYS_LEN {
        let gf = got[i].to_f32();
        let ef = expected[i].to_f32();
        let tol = ef.abs().max(1.0) * 2e-3; // f16 2-ulp
        assert!((gf - ef).abs() <= tol, "@{i}: got {gf} expected {ef}");
    }
}

// =========================================================================
// Phase 62 strided in-place — rank-0 and rank-1 sanity (edge cases).
// =========================================================================

#[test]
#[ignore]
fn affine_inplace_strided_rank0() {
    // Rank-0 (scalar) → numel = 1. The kernel's inner unravel loop
    // runs zero iterations, so off_y stays at 0 — kernel writes to
    // y[0] only.
    let (ctx, stream) = setup();
    let host = vec![3.14_f32];
    let scale: f32 = 2.0;
    let offset: f32 = 1.0;
    let expected: f32 = scale * host[0] + offset;

    let mut dev = DeviceBuffer::from_slice(&ctx, &host).expect("upload");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_f32_strided_run(
            1,
            0, // rank
            core::ptr::null(),
            core::ptr::null(),
            scale,
            offset,
            dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut got = vec![0_f32; 1];
    dev.copy_to_host(&mut got).expect("download");
    let tol = expected.abs().max(1.0) * 2.0 * f32::EPSILON;
    assert!((got[0] - expected).abs() <= tol);
}

#[test]
fn affine_inplace_strided_rejects_rank_over_8() {
    // Host-only test: kernel returns STATUS_INVALID_ARG=2 when rank > MAX_RANK.
    // Note: kernel does the check after numel check, so we still need numel >= 0.
    let (_ctx, stream) = setup();
    // Construct a rank-9 problem. Shape/stride arrays must exist but
    // their contents don't matter — the launcher rejects before dispatching.
    let shape = [2_i32; 9];
    let stride_y = [1_i64; 9];
    // numel = 0 → kernel returns 0 (success) before checking rank.
    // Use numel = 1 to ensure we hit the rank validation.
    let mut dummy_dev: DeviceBuffer<f32> = DeviceBuffer::zeros(&_ctx, 1).expect("alloc");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_inplace_f32_strided_run(
            1,
            9, // rank > MAX_RANK
            shape.as_ptr(),
            stride_y.as_ptr(),
            1.0,
            0.0,
            dummy_dev.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 2, "rank > MAX_RANK must return STATUS_INVALID_ARG");
}
