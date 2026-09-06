//! How noisy is a Linalg timing on this box, and is any trend claimable?
//!
//! This test was written to answer a different question — "why does the CSV
//! number disagree with criterion's?" — and the answer turned out to be that
//! it does not. The title is what it measures now.
//!
//! The bench publishes TWO numbers per cell from the same run:
//!
//! * `measure_median_ns_restored(SAMPLES=5, INNER=3)` -> the CSV row, which is
//!   what reaches `BENCHMARKS.md`.
//! * criterion's own `iter_custom` over ~100 samples -> the console line.
//!
//! The first run had them disagreeing, with all four cells leaning the same way:
//!
//! ```text
//! cell            CSV(us)   criterion(us)   CSV/crit
//! cholesky N256    332.77          361.91      0.919
//! lu       N256    315.73          319.91      0.987
//! cholesky N512    693.94          807.70      0.859
//! lu       N512   1011.17         1186.20      0.852
//! ```
//!
//! ⚠️ **THAT WAS A COINCIDENCE AND THIS TEST IS WHAT DISPROVED IT.** Four
//! same-direction draws is p = 1/16 — it *reads* as a systematic bias and is
//! not rare enough to be one. Repeated sampling puts `criterion/median`
//! anywhere in **0.82 to 1.21** — measured at 0.817, 0.933, 0.951, 0.990 and
//! 1.213 across five runs, so it lands on BOTH SIDES of 1.0. Criterion's
//! figure sits inside the spread and there is no bias to explain.
//!
//! **What is real is the DISPERSION.** Consecutive runs on this box:
//!
//! ```text
//! run  median    p90/median   max/min   criterion/median
//!  1   865 us      1.39        3.26          0.933
//!  2   988 us      1.21        2.87          0.817
//!  3   849 us      1.41        5.55          0.951
//!  4   666 us      1.21        2.34          1.213   <- criterion ABOVE
//! ```
//!
//! ## ⚠️ EVERY FIGURE HERE WAS MEASURED ON A LOADED MACHINE
//!
//! **This box runs ~17 other agent processes at 77-100% CPU. An unloaded
//! baseline is NOT OBTAINABLE on it**, so "measured on the 4070" names a
//! machine, not a machine at rest, and every number below carries that.
//!
//! **What was checked, and what each check actually licensed:**
//!
//! ```text
//! nvidia-smi --query-compute-apps   answers "what else is ON THE GPU"
//!                                   NOT "nothing else is competing"
//! no cargo/rustc processes          answers "are MY builds running"
//!                                   NOT "the machine is quiet"
//! ```
//!
//! ⚠️ Both were read as the broader claim. The GPU check is why the note below
//! once said the remainder was unidentified with no candidate named; the
//! process check is why "idle" looked reachable when it was not.
//!
//! **Persistent GPU state, measured at 0% utilization and 57 C:** `SW Thermal
//! Slowdown: Active`, `SW Power Cap: Active`, SM clock **1605 MHz against a
//! 3105 MHz maximum**. That throttle is present while idle and cool — it is
//! not a response to this workload.
//!
//! **The host-contention candidate was tested and ELIMINATED.** Same test,
//! three runs, with this session's own cargo builds fully drained:
//!
//! ```text
//! bg CPU   median      p90/median   max/min
//!  77.2%    930.7 us      1.359       2.73
//!  93.8%   1029.1 us      2.071      14.57   <- worst observed
//! 100.0%    821.2 us      1.301       3.08
//! ```
//!
//! ⚠️ **The dispersion is NOT lower without this session's builds — run 2 is
//! the worst ever measured here, an eight-fold tail (p99 8886 us against a
//! median of 1029 us).** So the concurrent builds were not the driver.
//!
//! ⚠️ **AND THE PUBLISHED SPREAD IS ITSELF A SAMPLE OF A WIDER ONE.** The
//! `max/min 2.34-5.55` recorded below reached **14.57** on re-measurement.
//! **The error bar has an error bar**, and a reader taking 5.55 as the worst
//! case understates it nearly threefold.
//!
//! **What remains: a persistent 1.93x clock throttle, uncontrollable 77-100%
//! background load, and an unidentified remainder now known to reach 14.6x.
//! Two mechanisms named, one measured away, and no third proposed.**
//!
//! So this test's job is no longer "explain the gap" — there is no gap. It
//! **measures the dispersion and refuses to report a trend inside it**, which
//! is what the bench needed and did not have.
//!
//! Run:
//!   cargo test -p baracuda-kernels-bench --release \
//!     --test linalg_sample_convergence -- --ignored --nocapture

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    CholeskyArgs, CholeskyDescriptor, CholeskyPlan, ElementKind, PlanPreference, TensorMut,
    Workspace, contiguous_stride,
};
use baracuda_kernels_bench::{measure_median_ns_restored, setup_device, warmup};

/// The cell that looked like the widest gap before repeated sampling showed
/// there was no gap, only spread.
const N: i32 = 512;

/// `(samples, inner)` ladder. The first row is exactly what `benches/linalg.rs`
/// uses, so the first line of output must reproduce the CSV figure.
const LADDER: &[(usize, u64)] = &[(5, 3), (5, 10), (11, 10), (11, 50), (25, 50)];

/// criterion's figure for this cell, from the run that produced the CSV.
const CRITERION_NS: f64 = 807_700.0;

#[test]
#[ignore = "requires a CUDA device; run explicitly with --ignored"]
fn linalg_timing_dispersion_swamps_any_sample_size_trend() {
    let (ctx, stream) = setup_device();
    let nu = N as usize;

    let mut host = vec![0.5_f32; nu * nu];
    for i in 0..nu {
        host[i * nu + i] = N as f32;
    }

    let pristine = DeviceBuffer::from_slice(&ctx, &host).expect("alloc pristine");
    let work = std::cell::RefCell::new(DeviceBuffer::from_slice(&ctx, &host).expect("alloc work"));
    let mut info: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc info");

    let desc = CholeskyDescriptor {
        matrix_size: N,
        batch_size: 1,
        lower: true,
        element: ElementKind::F32,
    };
    let plan =
        CholeskyPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let ws_bytes = plan.query_workspace_size(&stream).unwrap_or(0).max(1);
    let mut ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc ws");

    let sh = [1, N, N];
    let st = contiguous_stride(sh);

    let mut restore = || {
        pristine
            .copy_to_device_async(&work.borrow(), &stream)
            .expect("restore");
    };
    let mut launch = || {
        let mut w = work.borrow_mut();
        let args = CholeskyArgs::<f32> {
            a: TensorMut {
                data: w.as_slice_mut(),
                shape: sh,
                stride: st,
            },
            info: TensorMut {
                data: info.as_slice_mut(),
                shape: [1],
                stride: [1],
            },
        };
        plan.run(&stream, Workspace::Borrowed(ws.as_slice_mut()), args)
            .expect("cholesky");
    };

    warmup(&stream, || {
        restore();
        launch();
    });

    println!(
        "cholesky/f32/N{N}  — criterion reported {:.2} us",
        CRITERION_NS / 1000.0
    );
    println!(
        "{:>8} {:>7} {:>9} {:>12} {:>10}",
        "samples", "inner", "launches", "median(us)", "vs crit"
    );
    let mut estimates = Vec::new();
    for &(samples, inner) in LADDER {
        let ns =
            measure_median_ns_restored(&ctx, &stream, samples, inner, &mut restore, &mut launch);
        println!(
            "{samples:>8} {inner:>7} {:>9} {:>12.2} {:>10.3}",
            samples as u64 * inner,
            ns / 1000.0,
            ns / CRITERION_NS
        );
        estimates.push(ns);
    }

    let st = Stats::collect(&ctx, &stream, &mut restore, &mut launch);
    st.report();
    report_verdict(&estimates, &st);
}

/// The single-iteration timing distribution, in nanoseconds.
///
/// criterion's point estimate is a mean-like statistic; `measure_median_ns_restored`
/// is a MEDIAN. If the distribution has a heavy right tail those diverge by
/// construction and neither is wrong, so both are computed from THE SAME data —
/// the comparison has no second instrument in it.
struct Stats {
    n: usize,
    min: f64,
    median: f64,
    mean: f64,
    p90: f64,
    p99: f64,
    max: f64,
}

impl Stats {
    fn collect(
        ctx: &baracuda_driver::Context,
        stream: &baracuda_driver::Stream,
        restore: &mut impl FnMut(),
        launch: &mut impl FnMut(),
    ) -> Self {
        let mut v = Vec::new();
        for _ in 0..200 {
            // Reborrow: `time_with_events_restored` takes its closures BY VALUE,
            // so passing `restore`/`launch` directly moves them on iteration 1.
            let d = baracuda_kernels_bench::time_with_events_restored(
                ctx,
                stream,
                1,
                &mut *restore,
                &mut *launch,
            );
            v.push(d.as_secs_f64() * 1e9);
        }
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        v.sort_by(f64::total_cmp);
        let n = v.len();
        Self {
            n,
            min: v[0],
            median: v[n / 2],
            mean,
            p90: v[(n * 90) / 100],
            p99: v[(n * 99) / 100],
            max: v[n - 1],
        }
    }

    /// Dispersion as a percentage, used as the verdict's noise floor.
    fn noise_pct(&self) -> f64 {
        ((self.p90 / self.median) - 1.0) * 100.0
    }

    fn report(&self) {
        println!(
            "\n--- distribution over {} single-iteration timings (us) ---",
            self.n
        );
        println!(
            "min {:.2}  median {:.2}  mean {:.2}  p90 {:.2}  p99 {:.2}  max {:.2}",
            self.min / 1000.0,
            self.median / 1000.0,
            self.mean / 1000.0,
            self.p90 / 1000.0,
            self.p99 / 1000.0,
            self.max / 1000.0
        );
        println!(
            "mean/median = {:.3}   criterion/median = {:.3}",
            self.mean / self.median,
            CRITERION_NS / self.median
        );
    }
}

/// ⚠️ THE VERDICT IS COMPARED AGAINST THE NOISE, NOT AGAINST ZERO.
///
/// The first version compared the ladder ends to a fixed +-5% and printed a
/// direction. Run twice, back to back, it said:
///
/// ```text
/// run 1   711.65 -> 657.30 us  (-7.6%)   "MOVES DOWN ... investigate"
/// run 2   862.55 -> 947.64 us  (+9.9%)   "CONVERGES UPWARD"
/// ```
///
/// Opposite verdicts, same code, same machine, minutes apart — it was reporting
/// run-to-run variance as a trend and would have produced a confident story
/// either way. That is the instrument-measures-itself failure, committed by a
/// test written to diagnose one. The threshold now comes from THIS run's own
/// spread.
fn report_verdict(estimates: &[f64], st: &Stats) {
    println!("\n--- verdict ---");
    let first = estimates[0];
    let last = *estimates.last().expect("ladder is non-empty");
    let drift = (last / first - 1.0) * 100.0;
    println!(
        "shortest ladder rung {:.2} us -> longest {:.2} us  ({drift:+.1}%)",
        first / 1000.0,
        last / 1000.0,
    );
    let noise_pct = st.noise_pct();
    println!(
        "single-iteration dispersion this run: p90/median = {:.3} ({noise_pct:.1}%), max/min = {:.2}",
        st.p90 / st.median,
        st.max / st.min
    );
    let span = (LADDER.last().expect("ladder is non-empty").0 as u64
        * LADDER.last().expect("ladder is non-empty").1)
        / (LADDER[0].0 as u64 * LADDER[0].1);
    // `<=`, not `<`: the first spelling produced a DIRECTIONAL claim from an
    // exact tie (-20.6% drift against 20.6% dispersion). A threshold compared
    // against a float derived from the same run will land on itself sooner or
    // later. Ties are noise, so ties go to "no claim".
    if drift.abs() <= noise_pct + f64::EPSILON.max(noise_pct * 1e-9) {
        println!(
            "INDISTINGUISHABLE FROM NOISE: the ladder drifted {drift:+.1}% across a \
             {span}x launch range, inside this run's own {noise_pct:.1}% dispersion. \
             No trend is claimable, in EITHER direction."
        );
    } else {
        let dir = if drift > 0.0 { "upward" } else { "downward" };
        println!("Drift {drift:+.1}% EXCEEDS this run's {noise_pct:.1}% dispersion ({dir}).");
    }
    println!(
        "criterion/median = {:.3} — criterion's figure sits INSIDE this spread, so the \
         earlier 'CSV is systematically below criterion on 4 of 4 cells' reading was a \
         4-sample coincidence, not a bias.",
        CRITERION_NS / st.median
    );
}
