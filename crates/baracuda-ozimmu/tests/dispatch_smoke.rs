//! Verify the ozIMMU `Handle` plumbing through `baracuda-driver`.
//!
//! Phase 44b ships the safe wrapper + the C-ABI shim; the full
//! `GemmPlan` dispatch path (`PlanPreference::prefer_backend = Some(Ozaki{slices})`)
//! is exercised by the higher-up baracuda-cutlass / baracuda-kernels
//! integration tests that gate behind the `ozimmu` cargo feature. Here
//! we cover only the `baracuda-ozimmu` surface itself:
//!
//!   - Handle creation succeeds + Drop is clean.
//!   - Stream binding works (set_stream → dgemm → sync).
//!   - Pre-grow the workspace, then run multiple dgemms on it.
//!   - `OzakiSlices::S8` works at a non-trivial shape.

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_ozimmu::{Handle, MallocMode, Op, OzakiSlices};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn handle_create_drop_default() {
    baracuda_driver::init().expect("driver init");
    let device = Device::get(0).expect("device");
    let _ctx = Context::new(&device).expect("context");
    let h = Handle::new().expect("handle");
    drop(h);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn handle_create_drop_async_mode() {
    baracuda_driver::init().expect("driver init");
    let device = Device::get(0).expect("device");
    let _ctx = Context::new(&device).expect("context");
    let h = Handle::new_with_mode(MallocMode::Async).expect("handle");
    drop(h);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn pregrown_workspace_then_dgemm() {
    baracuda_driver::init().expect("driver init");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let h = Handle::new().expect("handle");
    h.set_stream(&stream);

    // Pre-grow workspace generously; an M=N=K=512 f64 GEMM at S=8
    // is comfortably inside the 64 MB ceiling.
    let grew = h.reallocate_working_memory_bytes(64 * 1024 * 1024);
    assert!(grew > 0, "first pre-grow must allocate (got {grew})");
    let regrown = h.reallocate_working_memory_bytes(32 * 1024 * 1024);
    assert_eq!(regrown, 0, "shrinking is a no-op (got {regrown})");

    let m = 256usize;
    let a_host: Vec<f64> = (0..(m * m)).map(|i| ((i as f64) * 0.01).sin()).collect();
    let b_host: Vec<f64> = (0..(m * m)).map(|i| ((i as f64) * 0.013).cos()).collect();
    let a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload A");
    let b = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload B");
    let c: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, m * m).expect("alloc C");

    unsafe {
        h.dgemm(
            Op::N, Op::N,
            m, m, m,
            1.0,
            a.as_raw().0 as *const f64, m,
            b.as_raw().0 as *const f64, m,
            0.0,
            c.as_raw().0 as *mut f64, m,
            OzakiSlices::S8,
        )
        .expect("dgemm");
    }
    stream.synchronize().expect("stream sync");

    // Sanity: result should be non-zero (some cells will land near 0
    // by accident, but the L2 norm shouldn't).
    let mut got = vec![0.0f64; m * m];
    c.copy_to_host(&mut got).expect("D2H");
    let nonfinite = got.iter().filter(|v| !v.is_finite()).count();
    let l2: f64 = got.iter().filter(|v| v.is_finite()).map(|v| v * v).sum::<f64>().sqrt();
    assert_eq!(
        nonfinite, 0,
        "result has {} non-finite cells (ozIMMU produced inf/NaN)",
        nonfinite,
    );
    assert!(l2 > 0.0, "result has zero L2 norm (ozIMMU launch produced nothing?)");
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn invalid_shape_rejected() {
    baracuda_driver::init().expect("driver init");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let h = Handle::new().expect("handle");
    h.set_stream(&stream);

    let buf: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 64).expect("alloc");

    // m == 0 → InvalidArgument
    let r = unsafe {
        h.dgemm(
            Op::N, Op::N,
            0, 8, 8,
            1.0,
            buf.as_raw().0 as *const f64, 8,
            buf.as_raw().0 as *const f64, 8,
            0.0,
            buf.as_raw().0 as *mut f64, 8,
            OzakiSlices::S8,
        )
    };
    assert!(r.is_err(), "m=0 should be rejected");

    // ldc < m → InvalidArgument
    let r = unsafe {
        h.dgemm(
            Op::N, Op::N,
            8, 8, 8,
            1.0,
            buf.as_raw().0 as *const f64, 8,
            buf.as_raw().0 as *const f64, 8,
            0.0,
            buf.as_raw().0 as *mut f64, 4,
            OzakiSlices::S8,
        )
    };
    assert!(r.is_err(), "ldc < m should be rejected");
}

#[test]
fn slices_enum_round_trip() {
    // Pure host-side test — no GPU needed. The to_compute_mode /
    // slice_count helpers are the bridge between the safe enum and
    // the FFI integer; make sure they don't drift.
    let all = [
        (OzakiSlices::S3, Some(3)),
        (OzakiSlices::S6, Some(6)),
        (OzakiSlices::S8, Some(8)),
        (OzakiSlices::S12, Some(12)),
        (OzakiSlices::S18, Some(18)),
        (OzakiSlices::Auto, None),
    ];
    for (s, expected) in all {
        assert_eq!(s.slice_count(), expected, "slice_count mismatch for {:?}", s);
        // to_compute_mode should not return 0 (sgemm) or anything out of range.
        let cm = s.to_compute_mode();
        assert!(cm >= 1 && cm <= 18, "compute_mode {cm} out of range");
    }
}
