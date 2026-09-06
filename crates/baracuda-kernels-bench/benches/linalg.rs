//! Dense Linalg throughput — baracuda self-bench + PyTorch reference.
//!
//! Wave-2 of the cross-implementation bench suite (Loss was Wave-1). cuSOLVER
//! backs this family and its dense API exposes **f32 + f64 only**, so unlike the
//! Loss wave there is no f16/bf16 axis — that is a property of the backend, not
//! a gap in the sweep.
//!
//! ⚠️ EVERY OP IN THIS FAMILY CONSUMES ITS INPUT IN PLACE. All eight Linalg
//! `*Args` structs take `a: TensorMut`, and `solve` additionally overwrites `b`.
//! A plain timing loop therefore measures the first iteration against the real
//! input and every later one against whatever the previous call left behind —
//! so it is not timing a fixed problem, which is what a benchmark is for.
//!
//! That is the whole reason, and it is enough. Each iteration restores the
//! pristine input by device-to-device copy, fenced out of the measurement by
//! its own event pair.
//!
//! ⚠️ AND A STRONGER REASON I ASSERTED HERE WAS FALSE — MEASURED, NOT ARGUED.
//! This doc used to claim: "after call 1 the buffer holds `L`, which is not
//! SPD, so cuSOLVER halts at the failing minor; iterations 2..N measure a
//! FAILURE PATH and the median lands far below the true cost." I never
//! measured it. `tests/repeat_destroys.rs` does, on the 4070:
//!
//! ```text
//! unrestored  info = [0, 0, 0, 0]      steady state  603.4 / 495.9 / 553.0 us
//! restored    info = [0, 0, 0, 0]      steady state  445.0 / 509.7 / 590.0 us
//! ```
//!
//! No failure path in either arm, and the two steady states INTERLEAVE — the
//! restore does not move the number at N=256. The premise was wrong because
//! `cusolverDnSpotrf(lower)` reads only the LOWER TRIANGLE, and `L`'s lower
//! triangle read as a symmetric matrix is still diagonally dominant with a
//! positive diagonal, hence still SPD. "The buffer holds L" and "the buffer
//! holds a matrix that fails to factor" are different claims and only the
//! first is true.
//!
//! ⚠️ Note the direction: that is a fixture-dependent result, so it does NOT
//! license the converse ("repeats are always safe"). It licenses exactly one
//! thing — deleting a reason I made up. The restore stays because iterations
//! 2..N factor a DIFFERENT MATRIX, which is a methodological defect whether or
//! not the clock happens to notice.
//!
//! **Sample counts are deliberately lower than the elementwise benches.** A
//! factorization is milliseconds where an elementwise kernel is microseconds;
//! the usual `samples = 11, inner = 50` would be 550 launches per cell.
//!
//! ⚠️ **READ THE NUMBERS THIS BENCH PRODUCES AS ±20-40%, NOT AS PRECISE.**
//! `tests/linalg_sample_convergence.rs` measures the single-iteration
//! distribution on this box, and it is wide. Three consecutive runs:
//!
//! ```text
//! run  median    p90/median   max/min   criterion/median
//!  1   865 us      1.39        3.26          0.933
//!  2   988 us      1.21        2.87          0.817
//!  3   849 us      1.41        5.55          0.951
//! ```
//!
//! ⚠️ **A first version of that test compared the ladder ends against a fixed
//! ±5% and printed a directional verdict. Run twice, minutes apart, it said
//! "MOVES DOWN (-7.6%)" and then "CONVERGES UPWARD (+9.9%)" — opposite
//! conclusions from the same code on the same machine.** It was reporting
//! run-to-run variance as a trend, and it would have produced a confident story
//! either way. It now derives its threshold from the run's own dispersion and
//! declines to claim a direction inside it.
//!
//! **The same coincidence nearly shipped as a finding.** The first CSV lined up
//! below criterion on 4 of 4 cells, which reads as a systematic bias; repeated
//! sampling shows `criterion/median` landing anywhere in 0.82-1.21 (measured at
//! 0.817, 0.933, 0.951, 0.990, 1.213), i.e. on BOTH SIDES of 1.0 — criterion's
//! figure sits INSIDE the spread and there is no bias. **Four same-direction
//! draws is p = 1/16 — not rare enough to mean anything.**
//!
//! **Measured environment,** because the dispersion is a property of this box
//! and not of cuSOLVER: no other compute process was on the GPU, and
//! `nvidia-smi` reported `SW Thermal Slowdown: Active`, `SW Power Cap: Active`,
//! SM clock **1605 MHz against a 3105 MHz maximum**. ⚠️ **That clock range is
//! 1.93x and the observed timing spread is 2.87-5.55x, so throttling is a
//! contributor and NOT a full explanation. I have not identified the rest and
//! am not guessing at it.**
//!
//! **Consequence: a single median from this bench is not a publishable figure
//! on its own — so it no longer ships as one.** The CSV now carries
//! `samples,min_ns,p90_ns,max_ns` beside every median, and the run prints
//! `p90/median` and `max/min` per cell.
//!
//! ⚠️ **Note WHICH dispersion that is.** Each sample is the mean over `INNER`
//! launches, so the CSV's spread is the spread of a 5-launch mean, not of a
//! single launch. Measured on the same box, same cell, minutes apart:
//!
//! ```text
//! per-SAMPLE (inner = 5, the CSV columns)   p90/median 1.01 - 1.08
//! per-LAUNCH (inner = 1, the convergence test) p90/median 1.21 - 1.41
//! ```
//!
//! The per-sample figure is the right error bar for the published median —
//! which is itself a median of those samples. Quoting it as the per-launch
//! variation would understate the noise by roughly a factor of four.
//!
//! Tranche 1 covers `cholesky` and `lu`. Deliberately excluded, so
//! the absence is a decision rather than an oversight:
//!
//! - `svd` / `eigh` — iterative, and an order of magnitude slower per call.
//!   They need their own shape sweep rather than sharing this one.
//! - `lstsq` — mixed-precision iterative refinement, so its cost is
//!   data-dependent in a way that needs a stated conditioning fixture first.
//! - `eig` (non-symmetric) — `torch.linalg.eig` returns complex even for real
//!   input, so the reference is not a like-for-like comparison.
//! - `qr` / `solve` — same in-place shape and same restore, but each needs its
//!   own output buffers (`q`/`r`; `b`/`pivot`); they follow once this tranche's
//!   restore discipline is validated on device.
//! - every batched variant — a separate tranche.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    CholeskyArgs, CholeskyDescriptor, CholeskyPlan, ElementKind, LuArgs, LuDescriptor, LuPlan,
    PlanPreference, TensorMut, Workspace, contiguous_stride,
};
use baracuda_kernels_bench::{
    PhaseTwentyNineRow, PytorchBaseline, append_csv_row_with_spread, measure_spread_restored,
    setup_device, time_with_events_restored, warmup,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const BENCH_NAME: &str = "linalg";

/// Square sizes. `O(n^3)` work, so the sweep stays modest: at n=512 an f64
/// factorization is already milliseconds.
const N_SWEEP: &[i32] = &[256, 512];

/// Median and spread over `SAMPLES` samples of `INNER` launches each.
///
/// ⚠️ **These were 5 and 3, and the justification was wrong in the same way
/// the failure-path claim was.** The module note said *"a factorization is
/// milliseconds where an elementwise kernel is microseconds"*, so 15 launches
/// per cell looked generous. Measured, these cells are **306 us to 1.19 ms** —
/// sub-millisecond at three of four sizes. I over-corrected against a cost I
/// had estimated rather than timed.
///
/// It matters now because the CSV carries a p90: **a p90 over five samples is
/// essentially the maximum**, so the spread column would have been a second
/// name for `max_ns`. At ~700 us a cell, 21x5 is 105 launches ~ 75 ms — still
/// nothing, and enough for the percentile to mean something.
const SAMPLES: usize = 21;
const INNER: u64 = 5;

fn leak_str(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

/// A symmetric, diagonally dominant `n x n` matrix — SPD by Gershgorin, so
/// Cholesky succeeds, and well-conditioned enough that LU / QR / solve are not
/// measuring a pathological case.
fn spd_host<T: From<f32> + Copy>(n: usize) -> Vec<T> {
    let mut a = vec![T::from(0.5_f32); n * n];
    for i in 0..n {
        a[i * n + i] = T::from(n as f32);
    }
    a
}

/// Everything that identifies a cell and is the same for every op in a run.
/// Bundled because `record` otherwise takes nine arguments, and the four that
/// vary per call (`op`, `n`, and the two closures) are the ones a reader is
/// actually checking at the call site.
struct Cell<'a> {
    ctx: &'a baracuda_driver::Context,
    stream: &'a baracuda_driver::Stream,
    dtype_label: &'static str,
    baseline: Option<&'a PytorchBaseline>,
}

/// One `(op, dtype, n)` cell: time it, print, and append the CSV row that
/// reaches `BENCHMARKS.md`.
fn record(
    c: &mut Criterion,
    cell: &Cell<'_>,
    op: &str,
    n: i32,
    mut restore: impl FnMut(),
    mut launch: impl FnMut(),
) {
    let Cell {
        ctx,
        stream,
        dtype_label,
        baseline,
    } = *cell;
    let shape = format!("N{n}");
    warmup(stream, || {
        restore();
        launch();
    });

    let spread = measure_spread_restored(ctx, stream, SAMPLES, INNER, &mut restore, &mut launch);
    let baracuda_ns = spread.median_ns;
    // The dispersion goes to stdout as well as the CSV, so a reader watching
    // the run sees the error bar next to the figure rather than having to open
    // the file to find out whether the number is tight.
    println!(
        "  {op}/{dtype_label}/{shape}: median {:.2} us  p90/median {:.2}  max/min {:.2}  (n={})",
        baracuda_ns / 1000.0,
        spread.p90_over_median(),
        spread.max_over_min(),
        spread.samples,
    );

    let pytorch_ns = baseline.and_then(|b| b.lookup(op, &shape, dtype_label));
    let mut group = c.benchmark_group(format!("{op}/{dtype_label}"));
    group.bench_with_input(BenchmarkId::from_parameter(&shape), &(), |bb, _| {
        bb.iter_custom(|iters| {
            time_with_events_restored(ctx, stream, iters, &mut restore, &mut launch)
        });
    });
    group.finish();

    append_csv_row_with_spread(
        BENCH_NAME,
        &PhaseTwentyNineRow {
            op: leak_str(op),
            shape: shape.clone(),
            dtype: leak_str(dtype_label),
            baracuda_ns,
            reference_ns: None,
            reference: "",
            pytorch_ns,
        },
        Some(&spread),
    );
}

/// The three buffers every op in this bench needs, allocated together so the
/// two `bench_*` functions do not each carry a copy of the same fallible
/// preamble. `None` means the device refused an allocation and the caller
/// should skip the cell.
///
/// `work` comes back behind a `RefCell` because `restore` needs `&work` (it is
/// the copy destination) while `launch` needs `&mut work`, and `record` holds
/// both closures at once. They never run concurrently, so the runtime check
/// never trips; the compiler simply cannot see the alternation.
type Buffers = (
    DeviceBuffer<f32>,
    std::cell::RefCell<DeviceBuffer<f32>>,
    DeviceBuffer<i32>,
);

fn alloc_buffers(ctx: &baracuda_driver::Context, host: &[f32]) -> Option<Buffers> {
    let (Ok(pristine), Ok(work), Ok(info)) = (
        DeviceBuffer::from_slice(ctx, host),
        DeviceBuffer::from_slice(ctx, host),
        DeviceBuffer::<i32>::zeros(ctx, 1),
    ) else {
        return None;
    };
    Some((pristine, std::cell::RefCell::new(work), info))
}

/// cuSOLVER scratch. ⚠️ `Workspace::None` is an ERROR and not an
/// auto-allocate: `unpack_workspace` returns `WorkspaceTooSmall { got: 0 }`
/// whenever `needed > 0`, and the size is only known after the query. So ask,
/// then allocate.
fn alloc_workspace(ctx: &baracuda_driver::Context, needed: usize) -> Option<DeviceBuffer<u8>> {
    DeviceBuffer::<u8>::zeros(ctx, needed.max(1)).ok()
}

/// The `work` buffer lives behind a `RefCell` in both ops below, because
/// `restore` needs `&work` (it is the copy destination) while `launch` needs
/// `&mut work`, and `record` holds both closures at once. They never run
/// concurrently, so the runtime check never trips; the compiler simply cannot
/// see the alternation.
///
/// `Workspace::None` is likewise an ERROR and not an auto-allocate: cuSOLVER's
/// scratch is caller-supplied and `unpack_workspace` returns
/// `WorkspaceTooSmall { got: 0 }` whenever `needed > 0`. The size is only known
/// after the query, so both ops ask and then allocate.
fn bench_cholesky_f32(c: &mut Criterion, cell: &Cell<'_>, n: i32, host: &[f32]) {
    let (ctx, stream) = (cell.ctx, cell.stream);
    let Some((pristine, work, mut info)) = alloc_buffers(ctx, host) else {
        return;
    };
    let desc = CholeskyDescriptor {
        matrix_size: n,
        batch_size: 1,
        lower: true,
        element: ElementKind::F32,
    };
    let Ok(plan) = CholeskyPlan::<f32>::select(stream, &desc, PlanPreference::default()) else {
        return;
    };
    let Some(mut ws) = alloc_workspace(ctx, plan.query_workspace_size(stream).unwrap_or(0)) else {
        return;
    };
    let sh = [1, n, n];
    let st = contiguous_stride(sh);
    record(
        c,
        cell,
        "cholesky",
        n,
        || {
            pristine
                .copy_to_device_async(&work.borrow(), stream)
                .expect("restore cholesky input");
        },
        || {
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
            plan.run(stream, Workspace::Borrowed(ws.as_slice_mut()), args)
                .expect("cholesky");
        },
    );
}

/// See [`bench_cholesky_f32`] for the `RefCell` and workspace notes; `lu` adds
/// a `pivot` output.
fn bench_lu_f32(c: &mut Criterion, cell: &Cell<'_>, n: i32, host: &[f32]) {
    let (ctx, stream) = (cell.ctx, cell.stream);
    let (Some((pristine, work, mut info)), Ok(mut pivot)) = (
        alloc_buffers(ctx, host),
        DeviceBuffer::<i32>::zeros(ctx, n as usize),
    ) else {
        return;
    };
    let desc = LuDescriptor {
        m: n,
        n,
        batch_size: 1,
        element: ElementKind::F32,
    };
    let Ok(plan) = LuPlan::<f32>::select(stream, &desc, PlanPreference::default()) else {
        return;
    };
    let Some(mut ws) = alloc_workspace(ctx, plan.query_workspace_size(stream).unwrap_or(0)) else {
        return;
    };
    let sh = [1, n, n];
    let st = contiguous_stride(sh);
    record(
        c,
        cell,
        "lu",
        n,
        || {
            pristine
                .copy_to_device_async(&work.borrow(), stream)
                .expect("restore lu input");
        },
        || {
            let mut w = work.borrow_mut();
            let args = LuArgs::<f32> {
                a: TensorMut {
                    data: w.as_slice_mut(),
                    shape: sh,
                    stride: st,
                },
                pivot: TensorMut {
                    data: pivot.as_slice_mut(),
                    shape: [1, n],
                    stride: contiguous_stride([1, n]),
                },
                info: TensorMut {
                    data: info.as_slice_mut(),
                    shape: [1],
                    stride: [1],
                },
            };
            plan.run(stream, Workspace::Borrowed(ws.as_slice_mut()), args)
                .expect("lu");
        },
    );
}

fn bench_f32(c: &mut Criterion, baseline: Option<&PytorchBaseline>) {
    let (ctx, stream) = setup_device();
    let cell = Cell {
        ctx: &ctx,
        stream: &stream,
        dtype_label: "f32",
        baseline,
    };
    for &n in N_SWEEP {
        let host = spd_host::<f32>(n as usize);
        bench_cholesky_f32(c, &cell, n, &host);
        bench_lu_f32(c, &cell, n, &host);
    }
}

fn benches(c: &mut Criterion) {
    let baseline = PytorchBaseline::load_default();
    if baseline.is_none() {
        eprintln!("linalg: no PyTorch baseline found; the PyTorch column will be empty");
    }
    bench_f32(c, baseline.as_ref());
}

// `criterion_group!` expands into a `pub fn` whose signature is fixed by the
// macro - can't doc-comment it directly, so suppress the workspace
// `missing_docs = deny` lint on the generated fn. Same glue as `loss.rs`.
#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, benches);
}
criterion_main!(criterion_glue::benches_grp);
