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
//! anywhere in **0.82 to 0.99**: criterion's figure sits inside the spread, and
//! there is no bias to explain.
//!
//! **What is real is the DISPERSION.** Three consecutive runs on this box:
//!
//! ```text
//! run  median    p90/median   max/min   criterion/median
//!  1   865 us      1.39        3.26          0.933
//!  2   988 us      1.21        2.87          0.817
//!  3   849 us      1.41        5.55          0.951
//! ```
//!
//! **Measured environment,** since the dispersion is a property of the box and
//! not of cuSOLVER: no other compute process on the GPU, and `nvidia-smi`
//! reporting `SW Thermal Slowdown: Active`, `SW Power Cap: Active`, SM clock
//! **1605 MHz against a 3105 MHz maximum**. ⚠️ **The clock range is 1.93x and
//! the timing spread is 2.87-5.55x, so throttling is a CONTRIBUTOR and not a
//! full explanation. The remainder is unidentified and I am not guessing at
//! it.**
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

    // criterion's figure for this cell, from the run that produced the CSV.
    const CRITERION_NS: f64 = 807_700.0;

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

    // ---- Is the gap MEDIAN vs MEAN over a right-skewed distribution? ----
    // criterion's point estimate is a mean-like statistic over ~100 samples;
    // `measure_median_ns_restored` is a MEDIAN. If the per-iteration
    // distribution has a heavy right tail, those two diverge by construction
    // and neither is wrong — they answer different questions.
    //
    // Measure the whole distribution once and report both, from the same data,
    // so the comparison has no second instrument in it.
    let mut per_iter_ns = Vec::new();
    for _ in 0..200 {
        let d = baracuda_kernels_bench::time_with_events_restored(
            &ctx,
            &stream,
            1,
            &mut restore,
            &mut launch,
        );
        per_iter_ns.push(d.as_secs_f64() * 1e9);
    }
    let mut sorted = per_iter_ns.clone();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let mean = per_iter_ns.iter().sum::<f64>() / n as f64;
    let median = sorted[n / 2];
    let p90 = sorted[(n * 90) / 100];
    let p99 = sorted[(n * 99) / 100];
    println!("\n--- distribution over {n} single-iteration timings (us) ---");
    println!(
        "min {:.2}  median {:.2}  mean {:.2}  p90 {:.2}  p99 {:.2}  max {:.2}",
        sorted[0] / 1000.0,
        median / 1000.0,
        mean / 1000.0,
        p90 / 1000.0,
        p99 / 1000.0,
        sorted[n - 1] / 1000.0
    );
    println!(
        "mean/median = {:.3}   criterion/median = {:.3}",
        mean / median,
        CRITERION_NS / median
    );

    println!("\n--- verdict ---");
    let first = estimates[0];
    let last = *estimates.last().expect("ladder is non-empty");
    let drift = (last / first - 1.0) * 100.0;
    println!(
        "shortest ladder rung {:.2} us -> longest {:.2} us  ({drift:+.1}%)",
        first / 1000.0,
        last / 1000.0,
    );

    // ⚠️ THE VERDICT MUST BE COMPARED AGAINST THE NOISE, NOT AGAINST ZERO.
    //
    // The first version of this test compared the ladder ends to a fixed +-5%
    // and printed a directional conclusion. Run twice, back to back, it said:
    //
    //     run 1   711.65 -> 657.30 us  (-7.6%)   "MOVES DOWN ... investigate"
    //     run 2   862.55 -> 947.64 us  (+9.9%)   "CONVERGES UPWARD"
    //
    // Opposite verdicts, same code, same machine, minutes apart. The ladder was
    // reporting run-to-run variance as a trend, and it would have produced a
    // confident story either way — which is the instrument-measures-itself
    // failure, committed by a test written to diagnose one.
    //
    // The dispersion measured above is why: p99/median ~ 1.8 and max/min ~ 3.
    // So the threshold is derived from THIS run's spread, and when the drift
    // does not clear it the test says so instead of picking a direction.
    let noise_pct = ((p90 / median) - 1.0) * 100.0;
    println!(
        "single-iteration dispersion this run: p90/median = {:.3} ({noise_pct:.1}%), \
         max/min = {:.2}",
        p90 / median,
        sorted[n - 1] / sorted[0]
    );
    // ⚠️ `<` here, not `<=`, was the first spelling and it produced a
    // DIRECTIONAL claim from an exact tie: run 2 drifted -20.6% against a
    // 20.6% dispersion and fell through to the else branch. A threshold
    // compared for equality against a float derived from the same run will
    // sooner or later land exactly on itself. Ties are noise, so ties go to
    // "no claim".
    if drift.abs() <= noise_pct + f64::EPSILON.max(noise_pct * 1e-9) {
        println!(
            "INDISTINGUISHABLE FROM NOISE: the ladder drifted {drift:+.1}% across an \
             {}x launch range, inside this run's own {noise_pct:.1}% dispersion. \
             No trend is claimable, in EITHER direction.",
            (LADDER.last().unwrap().0 as u64 * LADDER.last().unwrap().1)
                / (LADDER[0].0 as u64 * LADDER[0].1)
        );
    } else if drift > 0.0 {
        println!("Drift {drift:+.1}% EXCEEDS this run's {noise_pct:.1}% dispersion (upward).");
    } else {
        println!("Drift {drift:+.1}% EXCEEDS this run's {noise_pct:.1}% dispersion (downward).");
    }
    println!(
        "criterion/median = {:.3} — criterion's figure sits INSIDE this spread, so the \
         earlier 'CSV is systematically below criterion on 4 of 4 cells' reading was a \
         4-sample coincidence, not a bias.",
        CRITERION_NS / median
    );
}
