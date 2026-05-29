//! Phase 44c — n-blocking smoke test for the ozIMMU dispatch.
//!
//! `matmul_core` automatically chunks `n > 12288` int8 GEMM launches
//! into 8192-wide pieces (ported from
//! RIKEN-RCCS/accelerator_for_ozIMMU's `acc/gemm.cu`). This test
//! forces that path with `n = 16384` and verifies:
//!
//!   - The chunked launch succeeds without status errors.
//!   - The result matches a small-N reference (the chunking is
//!     algebraically a no-op — `C[:, j_start..j_end] = A @ B[:,
//!     j_start..j_end]` is exact for any chunking of `j`).
//!
//! The `n = 16384` shape allocates ~6 GiB of working memory on the
//! sm_89 RTX 4070 default config (m=64, k=128); kept small enough to
//! comfortably fit in 12 GB VRAM with overhead. The chunk path
//! triggers because `n > 12288`.
//!
//! Run with `cargo test -p baracuda-ozimmu --features build-vendor
//! --test n_blocking_smoke -- --ignored`.

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_ozimmu::{Handle, Op, OzakiSlices, OzakiVariant};

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn n_blocking_at_16384_matches_smaller_n_reference() {
    baracuda_driver::init().expect("driver init");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let h = Handle::new().expect("handle");
    h.set_stream(&stream);

    // Small m/k keeps the working-memory + I/O cost manageable; large
    // n forces the chunked path.
    let m = 64usize;
    let k = 128usize;
    let n_large = 16384usize;
    let n_small = 4096usize;

    let a_host: Vec<f64> =
        (0..(m * k)).map(|i| ((i as f64) * 0.011).sin() * 0.3).collect();
    let b_host: Vec<f64> = (0..(k * n_large))
        .map(|i| ((i as f64) * 0.007).cos() * 0.3)
        .collect();

    let a = DeviceBuffer::from_slice(&ctx, &a_host).expect("upload A");
    let b_large = DeviceBuffer::from_slice(&ctx, &b_host).expect("upload B");
    let c_large: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, m * n_large).expect("alloc C-large");

    // Run the large-N case — exercises n-blocking inside matmul_core.
    unsafe {
        h.dgemm_with_variant(
            Op::N, Op::N,
            m, n_large, k,
            1.0,
            a.as_raw().0 as *const f64, m,
            b_large.as_raw().0 as *const f64, k,
            0.0,
            c_large.as_raw().0 as *mut f64, m,
            OzakiSlices::S8,
            OzakiVariant::Base,
        )
        .expect("dgemm_with_variant large-N");
    }
    stream.synchronize().expect("sync large-N");

    // Now repeat the smaller-N tail (first 4096 cols of B) — no
    // chunking. The first n_small cols of c_large should match
    // c_small exactly modulo the int8 round-off (which depends on
    // per-row absolute max — independent of column index, so chunking
    // can't move the result).
    let b_small_host: Vec<f64> = b_host[0..(k * n_small)].to_vec();
    let b_small =
        DeviceBuffer::from_slice(&ctx, &b_small_host).expect("upload B-small");
    let c_small: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, m * n_small).expect("alloc C-small");
    unsafe {
        h.dgemm_with_variant(
            Op::N, Op::N,
            m, n_small, k,
            1.0,
            a.as_raw().0 as *const f64, m,
            b_small.as_raw().0 as *const f64, k,
            0.0,
            c_small.as_raw().0 as *mut f64, m,
            OzakiSlices::S8,
            OzakiVariant::Base,
        )
        .expect("dgemm_with_variant small-N");
    }
    stream.synchronize().expect("sync small-N");

    let mut large = vec![0.0f64; m * n_large];
    let mut small = vec![0.0f64; m * n_small];
    c_large.copy_to_host(&mut large).expect("D2H large");
    c_small.copy_to_host(&mut small).expect("D2H small");

    // Column-major: cell (mi, ni) is at index `ni * m + mi`. The
    // first `n_small` columns of `large` map to the entirety of
    // `small`.
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    for ni in 0..n_small {
        for mi in 0..m {
            let lv = large[ni * m + mi];
            let sv = small[ni * m + mi];
            let d = lv - sv;
            diff_sq += d * d;
            ref_sq += sv * sv;
        }
    }
    let rel = diff_sq.sqrt() / ref_sq.sqrt().max(1e-300);
    // n-blocking is supposed to be algebraically exact — split int8
    // GEMM along N produces bit-identical int32 partial sums. Tight
    // tolerance.
    assert!(
        rel < 1e-12,
        "n-blocking path diverged from small-N reference (relative {})",
        rel,
    );
}
