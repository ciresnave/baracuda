//! Phase 44c — exercise the four ozIMMU algorithmic variants (Base /
//! EF / RN / H) end-to-end via the safe wrapper.
//!
//! Each variant goes through `Handle::dgemm_with_variant` for the
//! same shape + inputs and we verify:
//!
//!   - The launch succeeds (no `Error::DgemmFailed`).
//!   - The output is non-zero (L2 norm > 0), confirming the dispatch
//!     actually reached the int8-tensor-core path for each variant.
//!   - Variants agree on the result up to their documented accuracy
//!     contract: all four should match Base at ~5e-10 relative error
//!     on a well-conditioned input (the upstream perf paper's
//!     `comparable-to-DGEMM` claim for `S = 8`). We use a generous
//!     1e-6 tolerance so well-conditioned-but-not-perfect inputs pass
//!     while genuine kernel breakage fails.
//!
//! Run with `cargo test -p baracuda-ozimmu --features build-vendor
//! --test variants_smoke -- --ignored`.

use baracuda_driver::{Context, Device, DeviceBuffer, Stream};
use baracuda_ozimmu::{Handle, Op, OzakiSlices, OzakiVariant};

fn run_variant_dgemm(
    h: &Handle,
    stream: &Stream,
    ctx: &Context,
    m: usize,
    variant: OzakiVariant,
) -> Vec<f64> {
    let a_host: Vec<f64> = (0..(m * m))
        .map(|i| ((i as f64) * 0.01).sin() * 0.5)
        .collect();
    let b_host: Vec<f64> = (0..(m * m))
        .map(|i| ((i as f64) * 0.013).cos() * 0.5)
        .collect();
    let a = DeviceBuffer::from_slice(ctx, &a_host).expect("upload A");
    let b = DeviceBuffer::from_slice(ctx, &b_host).expect("upload B");
    let c: DeviceBuffer<f64> = DeviceBuffer::zeros(ctx, m * m).expect("alloc C");

    unsafe {
        h.dgemm_with_variant(
            Op::N, Op::N,
            m, m, m,
            1.0,
            a.as_raw().0 as *const f64, m,
            b.as_raw().0 as *const f64, m,
            0.0,
            c.as_raw().0 as *mut f64, m,
            OzakiSlices::S8,
            variant,
        )
        .expect("dgemm_with_variant");
    }
    stream.synchronize().expect("stream sync");

    let mut out = vec![0.0f64; m * m];
    c.copy_to_host(&mut out).expect("D2H");
    out
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn variants_dispatch_and_produce_nonzero_output() {
    baracuda_driver::init().expect("driver init");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let h = Handle::new().expect("handle");
    h.set_stream(&stream);

    let m = 128usize;
    for variant in [
        OzakiVariant::Base,
        OzakiVariant::EF,
        OzakiVariant::RN,
        OzakiVariant::H,
    ] {
        let out = run_variant_dgemm(&h, &stream, &ctx, m, variant);
        let nonfinite = out.iter().filter(|v| !v.is_finite()).count();
        let nz_finite = out.iter().filter(|v| v.is_finite() && **v != 0.0).count();
        assert_eq!(
            nonfinite, 0,
            "variant {:?} produced {} non-finite cells (out of {})",
            variant, nonfinite, out.len(),
        );
        assert!(
            nz_finite > 0,
            "variant {:?} produced all-zero finite output (kernel didn't dispatch?)",
            variant,
        );
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn variants_agree_on_well_conditioned_input() {
    baracuda_driver::init().expect("driver init");
    let device = Device::get(0).expect("device");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let h = Handle::new().expect("handle");
    h.set_stream(&stream);

    let m = 128usize;
    let base = run_variant_dgemm(&h, &stream, &ctx, m, OzakiVariant::Base);
    let base_norm: f64 = base.iter().map(|v| v * v).sum::<f64>().sqrt();

    for variant in [OzakiVariant::EF, OzakiVariant::RN, OzakiVariant::H] {
        let got = run_variant_dgemm(&h, &stream, &ctx, m, variant);
        // Global magnitude relative error — absorbs single-cell
        // cancellation outliers in the synthetic sin/cos fixture.
        let diff_norm: f64 = base
            .iter()
            .zip(got.iter())
            .map(|(b, g)| (b - g) * (b - g))
            .sum::<f64>()
            .sqrt();
        let rel = diff_norm / base_norm.max(1e-300);
        assert!(
            rel < 1e-6,
            "variant {:?} disagreed with Base by relative {} \
             (>1e-6 tolerance; kernel may be wrong)",
            variant,
            rel,
        );
    }
}
