//! Phase 44c — ozIMMU variant timing.
//!
//! Times Base / EF / RN / H ozIMMU dispatch at LLM-typical FP64
//! shapes. Each variant runs through `baracuda-ozimmu::Handle::
//! dgemm_with_variant` directly (NOT through `GemmPlan`, which adds
//! a thin transpose-mapping overhead that's irrelevant for the
//! algorithm-level comparison).
//!
//! Compares against:
//!   - cuBLAS DGEMM (native FP64) — the "what does the hardware
//!     give you when you don't care about Ozaki" reference.
//!
//! All shapes use `S = 8` (the ozIMMU sweet spot for well-
//! conditioned input). Output dtype is FP64 column-major; layout is
//! `op_a = op_b = N` (the simplest case — n-blocking is independent
//! of layout).
//!
//! Run with `cargo bench -p baracuda-kernels-bench --features ozimmu
//! --bench gemm_ozaki_variants`.
//!
//! Without the `ozimmu` feature, this file compiles to a no-op `main`
//! so `cargo bench -p baracuda-kernels-bench --no-default-features`
//! still works.

#![allow(unexpected_cfgs)]

#[cfg(not(feature = "ozimmu"))]
fn main() {
    eprintln!(
        "gemm_ozaki_variants: built without the `ozimmu` feature; \
         skipping all benches. Re-run with `--features ozimmu` to \
         time the Phase 44c variant dispatch."
    );
}

#[cfg(feature = "ozimmu")]
mod inner {
    use baracuda_driver::DeviceBuffer;
    use baracuda_kernels_bench::{setup_device, time_with_events, warmup};
    use baracuda_ozimmu::{Handle, Op, OzakiSlices, OzakiVariant};
    use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

    // LLM-style FP64 GEMM shapes — square m=n=k matrices at three
    // useful sizes. 1024 fits comfortably in any GPU; 4096 is the
    // Llama-2-7B-class hidden dim; 8192 forces n-blocking (n > 12288
    // is the threshold; 8192 is the per-chunk size — we'd need
    // n=16384 to actually exercise blocking, but the bench is square
    // so 8192 still lands in the same compute regime).
    const OZAKI_SHAPE_SWEEP: &[usize] = &[1024, 2048, 4096];

    fn make_input_pair(
        ctx: &baracuda_driver::Context,
        m: usize,
    ) -> (DeviceBuffer<f64>, DeviceBuffer<f64>, DeviceBuffer<f64>) {
        let n = m;
        let k = m;
        let a_host: Vec<f64> = (0..(m * k))
            .map(|i| ((i as f64) * 0.01).sin() * 0.3)
            .collect();
        let b_host: Vec<f64> = (0..(k * n))
            .map(|i| ((i as f64) * 0.013).cos() * 0.3)
            .collect();
        let a = DeviceBuffer::from_slice(ctx, &a_host).expect("upload A");
        let b = DeviceBuffer::from_slice(ctx, &b_host).expect("upload B");
        let c: DeviceBuffer<f64> =
            DeviceBuffer::zeros(ctx, m * n).expect("alloc C");
        (a, b, c)
    }

    /// Bench all four ozIMMU variants for a single (m, slice) point.
    fn bench_variant_group(c: &mut Criterion, slice: OzakiSlices, label: &str) {
        let (ctx, stream) = setup_device();
        let h = Handle::new().expect("ozimmu handle");
        h.set_stream(&stream);

        let mut group = c.benchmark_group(format!("gemm_ozaki_variants/{label}"));

        for &m in OZAKI_SHAPE_SWEEP {
            let n = m;
            let k = m;
            // FP64 GEMM flops = 2*m*n*k.
            group.throughput(Throughput::Elements((2 * m * n * k) as u64));

            let (a, b, c_buf) = make_input_pair(&ctx, m);

            for variant in [
                OzakiVariant::Base,
                OzakiVariant::EF,
                OzakiVariant::RN,
                OzakiVariant::H,
            ] {
                let label_v = match variant {
                    OzakiVariant::Base => "base",
                    OzakiVariant::EF => "ef",
                    OzakiVariant::RN => "rn",
                    OzakiVariant::H => "h",
                    _ => "?",
                };
                let bid = BenchmarkId::new(label_v, m);

                let run = || unsafe {
                    h.dgemm_with_variant(
                        Op::N, Op::N,
                        m, n, k,
                        1.0,
                        a.as_raw().0 as *const f64, m,
                        b.as_raw().0 as *const f64, k,
                        0.0,
                        c_buf.as_raw().0 as *mut f64, m,
                        slice,
                        variant,
                    )
                    .expect("dgemm_with_variant");
                };

                warmup(&stream, run);

                group.bench_function(bid, |bench| {
                    bench.iter_custom(|iters| {
                        time_with_events(&ctx, &stream, iters, run)
                    });
                });
            }
        }
        group.finish();
    }

    pub fn bench_s8(c: &mut Criterion) {
        bench_variant_group(c, OzakiSlices::S8, "s8");
    }

    pub fn bench_s4(c: &mut Criterion) {
        // RN at S=4 should approach Base at S=6-S=8 accuracy.
        // Useful for the accuracy/perf tradeoff comparison.
        bench_variant_group(c, OzakiSlices::S4, "s4");
    }

    criterion_group!(benches, bench_s8, bench_s4);

    // criterion_main! at the file root (not inside this module) so
    // the symbol resolves at link time without an extra `inner::`
    // hop. See the wrapper below the module.
}

#[cfg(feature = "ozimmu")]
criterion::criterion_main!(inner::benches);
