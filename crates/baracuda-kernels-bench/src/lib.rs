//! # baracuda-kernels-bench
//!
//! Phase 10 Milestone 10.1 — benchmark harness for the `baracuda-kernels`
//! ML op surface. Ships three criterion bench binaries:
//!
//! - `gemm` — TFLOPS sweep across LLM-typical `(M, K=N)` shapes ×
//!   `{f32, f16, bf16, fp8, int8}` (fp8 gated by `sm89`).
//! - `flash_attention` — tokens/sec sweep across LLM-typical
//!   `(B, H, Q=K, D)` shapes × `{f32, f16, bf16}`.
//! - `conv2d` — GFLOPS sweep across ResNet-50 block shapes ×
//!   `{f32, f16}` (gated behind the `cudnn` feature).
//!
//! ## Measurement methodology
//!
//! GPU work isn't visible to criterion's default wall-clock timer in any
//! meaningful way — the kernel launch returns to the host almost
//! immediately, the work is queued, and host-side `Duration::elapsed`
//! mostly measures driver overhead. The right tool is **CUDA events**
//! (`cudaEventRecord` + `cudaEventElapsedTime`), which the driver records
//! on-device and exposes as `f32` milliseconds with `~0.5 us` resolution.
//!
//! `baracuda-driver` already exposes the event API as the [`Event`] type;
//! [`time_with_events`] wraps it in a criterion-friendly closure that
//! batches `iters` launches under a single event pair so the
//! per-iteration cost is the kernel itself, not the event overhead.
//!
//! Each bench follows the same shape:
//!
//! ```ignore
//! group.bench_with_input(BenchmarkId::from_parameter(shape), &shape, |b, shape| {
//!     // 1. Build the plan + buffers ONCE outside the timing loop.
//!     let plan = ...;
//!     let args = ...;
//!     // 2. 10-launch warmup so GPU clock + cache state settle.
//!     for _ in 0..10 { plan.run(...).unwrap(); }
//!     stream.synchronize().unwrap();
//!     // 3. Bench with CUDA events.
//!     b.iter_custom(|iters| time_with_events(&ctx, &stream, iters, || {
//!         plan.run(...).unwrap();
//!     }));
//! });
//! ```
//!
//! ## Running
//!
//! From the workspace root, with `sm89` + cuDNN installed:
//!
//! ```text
//! cargo bench -p baracuda-kernels-bench --features sm89,cudnn
//! ```
//!
//! The full sweep takes ~30 minutes on an RTX 4070. Use `--bench gemm`
//! to scope to one family. Use `-- --quick` for criterion's reduced-
//! sample-count fast pass (10 samples vs the default 100).

#![deny(missing_docs)]

use std::time::Duration;

use baracuda_driver::{Context, Device, Event, Stream, init, version};
use baracuda_kernels_types::{
    ArchSku, CandidateResult, DispatchEntry, HwStamp, Implementor, StructureKey, winner_of,
};

// ---------------------------------------------------------------------
// Device init
// ---------------------------------------------------------------------

/// Initialize the CUDA driver and return a `(Context, Stream)` pair on
/// device 0. Panics on any failure — the bench can't continue without a
/// live GPU context anyway.
pub fn setup_device() -> (Context, Stream) {
    init().expect("baracuda-driver init failed — is the CUDA driver loaded?");
    let device = Device::get(0).expect("Device::get(0) failed — is there a CUDA-capable GPU?");
    let ctx = Context::new(&device).expect("Context::new failed");
    let stream = Stream::new(&ctx).expect("Stream::new failed");
    (ctx, stream)
}

// ---------------------------------------------------------------------
// Dispatch bench gate — the item-07 dispatch-table *populator* (v1)
// ---------------------------------------------------------------------
//
// The dispatch-table schema + decision logic (`winner_of`/`seed_winner`/`merge`)
// live in `baracuda-kernels-types::dispatch`. This is the on-device half: measure
// each candidate for a cell and reduce to a `DispatchEntry` via `winner_of`, then
// `merge` those measured rows over the hand-seeded table. It is the v1 populator;
// Fuel's `dispatch_record` feed is the v2 populator through the same `merge` seam.

/// Map a device's `(major, minor)` compute capability to the specialization
/// `ArchSku`. Ada (sm_89) is its own SKU; other Ampere (sm_80/86/87) keys to the
/// `Sm80` cell (an sm_80 kernel is forward-compatible within Ampere); Hopper
/// (compute 9.x) elects `Sm90a`, NOT the portable `Sm90` — deliberately: Baracuda
/// specializes Hopper against the arch-exclusive instruction set (its Hopper token
/// is `sm90a`, cuda.md §2), so a detected H100 gets the accelerated cell. `Sm90`
/// (added to the vocabulary in unpopped-vocab 0.2.0) exists for token decode but is
/// not a Baracuda election target — `sm_90` and `sm_90a` are distinct compilation
/// targets that cannot share a cache key. Mirrors `unpopped-vocab
/// telemetry::arch_sku_of`, kept in lockstep so drift is caught. `None` for a
/// capability with no built cell.
#[must_use]
pub fn arch_sku_of(major: u32, minor: u32) -> Option<ArchSku> {
    match (major, minor) {
        (8, 9) => Some(ArchSku::Sm89), // Ada — RTX 4070 / RTX 6000 Ada / L40S
        (8, _) => Some(ArchSku::Sm80), // Ampere — A100 (sm_80), consumer sm_86/87
        // Hopper — elects sm90a (arch-exclusive), not the portable sm90 (see doc).
        (9, _) => Some(ArchSku::Sm90a),
        _ => None,
    }
}

/// The hardware-provenance stamp for the current device — arch + device name +
/// CUDA version + capture time. `None` if the device reports a capability with no
/// built cell (`arch_sku_of` miss). The capture time is read from the wall clock
/// here (the bench crate, not the deterministic types crate) and is dropped from
/// the committed routing artifact, so it never churns the diff.
#[must_use]
pub fn current_hwstamp(device: &Device) -> Option<HwStamp> {
    let (major, minor) = device.compute_capability().ok()?;
    let arch = arch_sku_of(major, minor)?;
    let device_name = device.name().ok()?;
    let cuda_version = version().ok().map(|v| v.to_string()).unwrap_or_default();
    let captured_unix_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Some(HwStamp {
        // 0.2.0 generalized HwStamp off cuda-specific naming: `arch: ArchSku` →
        // `target: TargetId` (via `From<ArchSku>`), `cuda_version` → `runtime_version`
        // (opaque here — the namespace owner spells it, §6.8-0004).
        target: arch.into(),
        device_name,
        runtime_version: cuda_version,
        captured_unix_s,
    })
}

/// Time every candidate for a cell and reduce to a measured [`DispatchEntry`].
///
/// Each candidate is `(implementor, entry_point, launch)`; `launch` enqueues one
/// invocation (the same contract as [`measure_median_ns`]'s closure).
/// [`winner_of`] picks the min-median winner and computes the margin; the result
/// is a `Provenance::Measured` entry ready to [`baracuda_kernels_types::merge`]
/// over the seeded table. Returns `None` for an empty candidate set.
///
/// **Correctness is the caller's precondition** (as for `winner_of`): only pass
/// candidates whose output already matched the op's numeric oracle — a
/// fast-but-wrong candidate must be rejected *before* it is timed here, never
/// ranked. A generated candidate must additionally have passed the standing
/// nvrtc/nvcc/compute-sanitizer gate.
#[must_use]
pub fn gate_cell<'a>(
    ctx: &Context,
    stream: &Stream,
    key: &StructureKey,
    measured_on: Option<HwStamp>,
    samples: usize,
    inner: u64,
    candidates: Vec<(Implementor, Option<String>, Box<dyn FnMut() + 'a>)>,
) -> Option<DispatchEntry> {
    let mut results = Vec::with_capacity(candidates.len());
    for (implementor, entry_point, launch) in candidates {
        let median_ns = measure_median_ns(ctx, stream, samples, inner, launch);
        results.push(CandidateResult {
            implementor,
            median_ns,
            entry_point,
        });
    }
    winner_of(key.to_token(), results, measured_on)
}

// ---------------------------------------------------------------------
// CUDA-event-timed measurement
// ---------------------------------------------------------------------

/// Time `iters` invocations of `launch`, running `restore` before each one
/// and **excluding it from the measurement**.
///
/// ⚠️ For an op that consumes its input IN PLACE, the plain
/// [`time_with_events`] loop measures the wrong thing, and not by a little:
/// iteration 1 sees the real input and iterations `2..iters` see whatever the
/// op left behind.
///
/// Every op in the Linalg family takes `a: TensorMut` and overwrites it
/// (`cholesky`, `lu`, `qr`, `svd`, `eigh`, `inverse`, `solve`, `lstsq` — all
/// eight). Cholesky is the sharpest case: after the first call the buffer holds
/// `L`, which is not symmetric positive-definite, so cuSOLVER halts at the
/// failing minor and returns early. The later iterations measure a FAILURE PATH,
/// the mean comes out far below the true cost, and nothing errors — the `info`
/// vector reports it and a timing loop never reads `info`.
///
/// So `restore` re-establishes the pristine input (a device-to-device copy from
/// a pristine replica) and is fenced out of the timed region by a per-iteration
/// event pair. That costs two event records per iteration, ~2 µs, against
/// millisecond-scale factorizations.
///
/// Use [`time_with_events`] for ops that do not mutate their inputs — it uses a
/// single event pair for the whole loop and has strictly less overhead.
pub fn time_with_events_restored<S, F>(
    ctx: &Context,
    stream: &Stream,
    iters: u64,
    mut restore: S,
    mut launch: F,
) -> Duration
where
    S: FnMut(),
    F: FnMut(),
{
    let start = Event::new(ctx).expect("Event::new(start)");
    let end = Event::new(ctx).expect("Event::new(end)");
    let mut total_ms = 0.0_f64;

    for _ in 0..iters {
        restore();
        // Fence the restore out of the measurement: it is enqueued on the same
        // stream, so recording `start` after it means the timed span begins when
        // the restore has completed, not when it was submitted.
        start.record(stream).expect("start.record");
        launch();
        end.record(stream).expect("end.record");
        end.synchronize().expect("end.synchronize");
        total_ms += f64::from(Event::elapsed_time_ms(&start, &end).expect("elapsed_time_ms"));
    }

    Duration::from_secs_f64(total_ms / 1000.0)
}

/// Time `iters` invocations of `launch` under a single CUDA event pair
/// and return the **total** wall-clock duration (criterion divides by
/// `iters` itself when computing per-iter cost).
///
/// Each call to `launch` should enqueue exactly one kernel launch (or
/// a tight bundle of launches that make up "one logical op"); the host
/// synchronises only at the end so the iteration cost reflects pure
/// device time, not host-launch overhead.
///
/// # Panics
///
/// Panics if event creation / record / sync fails. These would indicate
/// a misconfigured CUDA context, not a bench-harness bug, and a panic
/// inside a `b.iter_custom` closure is the cleanest way to surface it
/// to the criterion runner.
pub fn time_with_events<F>(ctx: &Context, stream: &Stream, iters: u64, mut launch: F) -> Duration
where
    F: FnMut(),
{
    let start = Event::new(ctx).expect("Event::new(start)");
    let end = Event::new(ctx).expect("Event::new(end)");

    start.record(stream).expect("start.record");
    for _ in 0..iters {
        launch();
    }
    end.record(stream).expect("end.record");
    end.synchronize().expect("end.synchronize");

    let ms = Event::elapsed_time_ms(&start, &end).expect("elapsed_time_ms");
    // `cudaEventElapsedTime` returns milliseconds as `f32` with ~0.5us
    // resolution. Convert to a `Duration` for criterion.
    Duration::from_secs_f64(ms as f64 / 1000.0)
}

/// Number of warmup launches before the first timed sample. 10 is the
/// rule-of-thumb that lets the GPU clock settle out of idle and warms
/// up SMEM caches without stretching the bench too long. Exposed as a
/// constant so all three bench binaries stay consistent.
pub const WARMUP_ITERS: usize = 10;

/// Run `WARMUP_ITERS` launches then host-sync. Use this once per shape
/// before calling `iter_custom`.
///
/// Always synchronises the stream — leaving in-flight work across the
/// warmup→timed boundary would let cold-cache launches leak into the
/// first timed sample.
pub fn warmup<F: FnMut()>(stream: &Stream, mut launch: F) {
    for _ in 0..WARMUP_ITERS {
        launch();
    }
    stream.synchronize().expect("stream sync after warmup");
}

// ---------------------------------------------------------------------
// Problem-shape iterators
// ---------------------------------------------------------------------

/// `(M, N, K)` triples for the LLM-typical GEMM sweep.
///
/// `M` is the "batch-token" axis (tiny `M` ⇒ decode; large `M` ⇒
/// prefill). `K == N` follows the square-mat-mul convention used in
/// most transformer layers (hidden_size ≈ ffn_size / 4, and the
/// attention `Q @ K^T` / FFN `x @ W` both end up `(seq, hidden) @
/// (hidden, hidden)` or `(seq, hidden) @ (hidden, 4·hidden)`). At
/// modeling level `K == N` is the right baseline.
pub const GEMM_M_SWEEP: &[i32] = &[1, 8, 32, 128, 512];

/// `K == N` values to sweep for GEMM. These cover the typical hidden /
/// FFN dimension range from 7B (4096) up to 70B (8192). Smaller `2048`
/// covers the 1B-class models. Larger `K` is bandwidth-bound; smaller
/// `K` is launch-overhead-bound.
pub const GEMM_KN_SWEEP: &[i32] = &[2048, 4096, 8192];

/// `(B, H, Q=K, D)` quadruples for the Flash-Attention sweep.
///
/// `B = 1` (single user / serving), `H ∈ {8, 16, 32}` covers
/// Llama-7B/13B/70B head counts, `Q = K ∈ {512, 1024, 2048, 4096}`
/// covers prefill from short context to mid-context, `D ∈ {64, 128}`
/// covers MHA (64 in some smaller models, 128 standard).
pub const FLASH_B: i32 = 1;
/// Head counts for the Flash sweep.
pub const FLASH_H_SWEEP: &[i32] = &[8, 16, 32];
/// Sequence lengths (`Q == K`) for the Flash sweep.
pub const FLASH_QK_SWEEP: &[i32] = &[512, 1024, 2048, 4096];
/// Head dimensions for the Flash sweep.
pub const FLASH_D_SWEEP: &[i32] = &[64, 128];

/// A single ResNet-50-style Conv2d shape. The set covers (a) the small
/// stem stage (`56×56`, 64ch), (b) a mid-stage residual block
/// (`28×28`, 128ch), (c) a deep-stage block (`14×14`, 256ch). These
/// are representative of the three working sizes one sees in a typical
/// ImageNet-class CNN.
#[derive(Copy, Clone, Debug)]
pub struct Conv2dShape {
    /// Batch size.
    pub n: i32,
    /// Input channels.
    pub c_in: i32,
    /// Output channels.
    pub c_out: i32,
    /// Input spatial extent (square, so `H == W`).
    pub hw: i32,
    /// Filter spatial extent.
    pub k: i32,
}

impl Conv2dShape {
    /// Total multiply-add count = `N · C_out · H_out · W_out · C_in · K · K`.
    /// FW pass FLOPs = `2 · macs`.
    pub fn macs(self) -> u64 {
        // Assume `pad = k/2`, `stride = 1`, so `H_out == H_in`.
        let h_out = self.hw as i64;
        let w_out = self.hw as i64;
        (self.n as i64
            * self.c_out as i64
            * h_out
            * w_out
            * self.c_in as i64
            * self.k as i64
            * self.k as i64) as u64
    }
}

/// Representative ResNet-50 block shapes (3 picks).
pub const CONV2D_SWEEP: &[Conv2dShape] = &[
    // Stem-stage residual: 56×56 spatial, 64 → 64ch, 3×3 kernel.
    Conv2dShape {
        n: 1,
        c_in: 64,
        c_out: 64,
        hw: 56,
        k: 3,
    },
    // Mid-stage: 28×28, 128 → 128ch, 3×3.
    Conv2dShape {
        n: 1,
        c_in: 128,
        c_out: 128,
        hw: 28,
        k: 3,
    },
    // Deep-stage: 14×14, 256 → 256ch, 3×3.
    Conv2dShape {
        n: 1,
        c_in: 256,
        c_out: 256,
        hw: 14,
        k: 3,
    },
];

// ---------------------------------------------------------------------
// Throughput helpers
// ---------------------------------------------------------------------

/// GEMM FLOPs = `2 · M · N · K`. Returned as `u64`; criterion uses this
/// via `Throughput::Elements`.
#[inline]
pub fn gemm_flops(m: i32, n: i32, k: i32) -> u64 {
    2u64 * (m as u64) * (n as u64) * (k as u64)
}

/// Flash-Attention FLOPs ≈ `4 · B · H · Q · K · D` (two GEMMs:
/// `Q·K^T` and `softmax(...)·V`, both `B·H·Q·K·D` macs ⇒
/// `2·B·H·Q·K·D` flops each).
#[inline]
pub fn flash_flops(b: i32, h: i32, q: i32, k: i32, d: i32) -> u64 {
    4u64 * (b as u64) * (h as u64) * (q as u64) * (k as u64) * (d as u64)
}

/// Conv2d FW FLOPs = `2 · macs` (one multiply + one add per MAC).
#[inline]
pub fn conv2d_flops(shape: Conv2dShape) -> u64 {
    2 * shape.macs()
}

// ---------------------------------------------------------------------
// Phase 29 — cross-implementation CSV emission
// ---------------------------------------------------------------------

/// One row of a cross-implementation timing table.
///
/// Phase 29 benches use this to dump `(op, shape, dtype, baracuda_ns,
/// reference_ns)` rows that `BENCHMARKS.md` reads back into the summary
/// table. The CSV is written by the bench process at finish-time under
/// `target/criterion/phase29/<bench>.csv`; criterion's HTML report is
/// the primary output, the CSV is the structured-data companion for the
/// summary table.
#[derive(Clone, Debug)]
pub struct PhaseTwentyNineRow {
    /// Op family (e.g. `"gemm"`, `"softmax"`, `"conv2d"`).
    pub op: &'static str,
    /// Shape descriptor (free-form — `"M128_N4096_K4096"` for GEMM,
    /// `"N1_C64_H56_W56_K64_F3"` for conv, etc.).
    pub shape: String,
    /// Element dtype label (`"f32"`, `"f16"`, `"bf16"`, `"q4_0"`, ...).
    pub dtype: &'static str,
    /// baracuda median wall time, nanoseconds.
    pub baracuda_ns: f64,
    /// Reference (cuBLAS / cuDNN / self) median wall time, nanoseconds.
    /// `None` when the bench is self-only (e.g. MMVQ — no cuBLAS
    /// equivalent for GGUF quant ops).
    pub reference_ns: Option<f64>,
    /// Reference label — `"cuBLAS"`, `"cuDNN"`, `""` (none).
    pub reference: &'static str,
    /// PyTorch baseline median wall time, nanoseconds (Phase 73.1).
    /// Loaded from the frozen JSON baseline at
    /// `crates/baracuda-kernels-bench/bench-baselines/`. `None` when no
    /// matching `(op, shape, dtype)` entry exists in the JSON — i.e.
    /// the bench hasn't been added to the Python refresh script yet,
    /// or the bench is running on hardware/PyTorch-version with no
    /// matching baseline file.
    pub pytorch_ns: Option<f64>,
}

impl PhaseTwentyNineRow {
    /// Delta = `reference_ns / baracuda_ns`. `< 1.0` ⇒ baracuda faster;
    /// `> 1.0` ⇒ reference faster. `None` when no reference present.
    pub fn delta(&self) -> Option<f64> {
        let r = self.reference_ns?;
        if self.baracuda_ns == 0.0 {
            None
        } else {
            Some(r / self.baracuda_ns)
        }
    }

    /// PyTorch delta = `pytorch_ns / baracuda_ns`. Same convention as
    /// `delta()`: `< 1.0` ⇒ baracuda faster than PyTorch.
    pub fn pytorch_delta(&self) -> Option<f64> {
        let r = self.pytorch_ns?;
        if self.baracuda_ns == 0.0 {
            None
        } else {
            Some(r / self.baracuda_ns)
        }
    }
}

/// Median over `iters` invocations of `launch` under CUDA-event timing,
/// returning nanoseconds. Used by the cross-impl benches to get a single
/// representative timing for the summary CSV — criterion's full
/// statistical analysis is still recorded in the HTML report.
///
/// Runs `samples` independent (start, end) event pairs each timing
/// `inner` launches, then returns the median of the per-sample averages.
/// Defaults: `samples = 11`, `inner = 50`. These are conservative — they
/// add ~550 launches per shape on top of the criterion run, which
/// is rounding error vs criterion's 100-sample sweep.
pub fn measure_median_ns<F: FnMut()>(
    ctx: &Context,
    stream: &Stream,
    samples: usize,
    inner: u64,
    mut launch: F,
) -> f64 {
    let mut measurements: Vec<f64> = Vec::with_capacity(samples);
    for _ in 0..samples {
        let dur = time_with_events(ctx, stream, inner, &mut launch);
        let ns = dur.as_secs_f64() * 1e9 / inner as f64;
        measurements.push(ns);
    }
    measurements.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    measurements[measurements.len() / 2]
}

/// [`measure_median_ns`] for an op that consumes its input IN PLACE: `restore`
/// runs before every launch and is excluded from the timing.
///
/// ⚠️ The CSV value this produces is what reaches `BENCHMARKS.md`, so an
/// in-place op measured with the plain [`measure_median_ns`] publishes a wrong
/// number rather than merely reporting one. See
/// [`time_with_events_restored`] for why the error is large and silent.
pub fn measure_median_ns_restored<S: FnMut(), F: FnMut()>(
    ctx: &Context,
    stream: &Stream,
    samples: usize,
    inner: u64,
    mut restore: S,
    mut launch: F,
) -> f64 {
    measure_spread_restored(ctx, stream, samples, inner, &mut restore, &mut launch).median_ns
}

/// The spread of a timing measurement, in nanoseconds.
///
/// ⚠️ **A median alone is not a publishable figure on a box whose dispersion
/// you have not measured.** On the RTX 4070 Laptop this crate benches on, two
/// runs of *identical* Linalg code came out 14% apart, and repeated sampling of
/// one cell gave `p90/median` between 1.21 and 1.41 with `max/min` between 2.34
/// and 5.55. A point estimate with no spread beside it invites a reader to
/// treat a 14% difference as a change when it is noise.
///
/// So the measurement carries its own dispersion into the CSV, and
/// `BENCHMARKS.md` can state a figure with an error bar instead of a number
/// that looks exact.
///
/// ⚠️ **WHAT THIS IS THE DISPERSION *OF*, because the two differ by a lot.**
/// Each element summarised here is one SAMPLE — the mean over `inner`
/// launches — not one launch. Averaging smooths the per-launch tail, so this
/// spread is narrower than the raw single-shot spread and must not be quoted
/// as the latter. Measured on the same box, same cell, within minutes:
///
/// ```text
/// per-SAMPLE (inner = 5, what this struct reports)   p90/median 1.01 - 1.08
/// per-LAUNCH (inner = 1, tests/linalg_sample_...)    p90/median 1.21 - 1.41
/// ```
///
/// The per-sample figure is the right error bar for the published median,
/// because the published median IS a median of such samples. The per-launch
/// figure is the right one for "how much does one call vary". **They are
/// different questions and this reports the first.**
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    /// How many per-sample measurements this summarises.
    pub samples: usize,
    /// Fastest per-iteration time seen.
    pub min_ns: f64,
    /// The published figure.
    pub median_ns: f64,
    /// 90th percentile — the shoulder of the right tail.
    pub p90_ns: f64,
    /// Slowest per-iteration time seen.
    pub max_ns: f64,
}

impl Spread {
    /// `p90 / median` — the dispersion the verdicts in this crate compare
    /// against. `1.0` is a perfectly tight distribution.
    #[must_use]
    pub fn p90_over_median(&self) -> f64 {
        if self.median_ns > 0.0 {
            self.p90_ns / self.median_ns
        } else {
            f64::NAN
        }
    }

    /// `max / min` — the full observed range, including outliers.
    #[must_use]
    pub fn max_over_min(&self) -> f64 {
        if self.min_ns > 0.0 {
            self.max_ns / self.min_ns
        } else {
            f64::NAN
        }
    }
}

/// Like [`measure_median_ns_restored`] but keeps the whole distribution.
///
/// ⚠️ The percentiles are only as meaningful as `samples` makes them: a p90
/// over five samples is essentially the maximum. Callers wanting a usable
/// spread should pass enough samples for the percentile to mean something —
/// see the note on `SAMPLES` in `benches/linalg.rs`.
pub fn measure_spread_restored<S: FnMut(), F: FnMut()>(
    ctx: &Context,
    stream: &Stream,
    samples: usize,
    inner: u64,
    mut restore: S,
    mut launch: F,
) -> Spread {
    let mut m: Vec<f64> = Vec::with_capacity(samples);
    for _ in 0..samples {
        let dur = time_with_events_restored(ctx, stream, inner, &mut restore, &mut launch);
        m.push(dur.as_secs_f64() * 1e9 / inner as f64);
    }
    m.sort_by(f64::total_cmp);
    let n = m.len();
    Spread {
        samples: n,
        min_ns: m[0],
        median_ns: m[n / 2],
        p90_ns: m[(n * 90) / 100],
        max_ns: m[n - 1],
    }
}

/// Append a `PhaseTwentyNineRow` to `target/criterion/phase29/<bench>.csv`,
/// creating the file (with header) if it doesn't exist.
///
/// CSV columns: `op,shape,dtype,baracuda_ns,reference_ns,reference,delta,
/// pytorch_ns,pytorch_delta` (Phase 73.1 extended the format with the
/// last two columns — see `PhaseTwentyNineRow::pytorch_ns`).
///
/// Errors are swallowed (printed to stderr) — bench correctness mustn't
/// depend on the CSV write succeeding. The criterion HTML report is the
/// primary record; the CSV is a convenience for `BENCHMARKS.md` updates.
pub fn append_csv_row(bench: &str, row: &PhaseTwentyNineRow) {
    append_csv_row_with_spread(bench, row, None);
}

/// The CSV header this crate writes. Kept as a constant because
/// [`append_csv_row_with_spread`] both writes it and CHECKS it — see the
/// stale-header note there.
pub const PHASE29_CSV_HEADER: &str = "op,shape,dtype,baracuda_ns,reference_ns,reference,delta,\
     pytorch_ns,pytorch_delta,samples,min_ns,p90_ns,max_ns";

/// Resolve (and create) the phase-29 CSV path for `bench`. `None` if the
/// directory cannot be made — the caller skips the row rather than failing the
/// bench.
fn phase29_csv_path(bench: &str) -> Option<std::path::PathBuf> {
    let dir = std::path::PathBuf::from("target")
        .join("criterion")
        .join("phase29");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("phase29 csv: mkdir {} failed: {e}", dir.display());
        return None;
    }
    Some(dir.join(format!("{bench}.csv")))
}

/// ⚠️ STALE-HEADER HAZARD. The header used to be written only when the file was
/// ABSENT, so widening the row format would have appended 13-field rows under a
/// 9-field header — every later column silently shifted by four, mapping
/// `samples` onto `pytorch_delta`, with no error anywhere. These files live
/// under `target/` and survive across runs, so "it is a fresh build" is not a
/// safe assumption.
///
/// So: read the existing header and, if it is not the one we write, start the
/// file over. A truncate loses rows from a PREVIOUS run only — the same thing
/// `cargo clean` does, and strictly better than emitting misaligned ones.
fn drop_csv_if_header_is_stale(path: &std::path::Path) {
    let Ok(existing) = std::fs::read_to_string(path) else {
        return;
    };
    let stale = existing
        .lines()
        .next()
        .is_some_and(|first| first != PHASE29_CSV_HEADER);
    if stale {
        eprintln!(
            "phase29 csv: {} has an older header; rewriting it rather than \
             appending misaligned rows",
            path.display()
        );
        let _ = std::fs::remove_file(path);
    }
}

/// The four spread columns, empty when the bench measured no spread.
fn spread_fields(spread: Option<&Spread>) -> (String, String, String, String) {
    match spread {
        Some(s) => (
            s.samples.to_string(),
            format!("{:.3}", s.min_ns),
            format!("{:.3}", s.p90_ns),
            format!("{:.3}", s.max_ns),
        ),
        None => (String::new(), String::new(), String::new(), String::new()),
    }
}

/// Format an `Option<f64>` column: three decimals, or empty.
fn opt_ns(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.3}")).unwrap_or_default()
}

/// Format an `Option<f64>` ratio column: four decimals, or empty.
fn opt_ratio(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.4}")).unwrap_or_default()
}

/// [`append_csv_row`] plus the measurement's dispersion.
///
/// The four spread columns are always present and empty when `spread` is
/// `None`, so every bench writes the same header whether or not it measures a
/// spread — a reader never has to discover which shape a given file is.
pub fn append_csv_row_with_spread(bench: &str, row: &PhaseTwentyNineRow, spread: Option<&Spread>) {
    use std::io::Write;

    let Some(path) = phase29_csv_path(bench) else {
        return;
    };
    drop_csv_if_header_is_stale(&path);
    let exists = path.exists();
    let mut f = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => f,
        Err(e) => {
            eprintln!("phase29 csv: open {} failed: {e}", path.display());
            return;
        }
    };
    if !exists {
        let _ = writeln!(f, "{PHASE29_CSV_HEADER}");
    }
    let (sn, mn, p9, mx) = spread_fields(spread);
    let _ = writeln!(
        f,
        "{op},{shape},{dtype},{ba:.3},{rf},{rl},{dl},{pn},{pd},{sn},{mn},{p9},{mx}",
        op = row.op,
        shape = row.shape,
        dtype = row.dtype,
        ba = row.baracuda_ns,
        rf = opt_ns(row.reference_ns),
        rl = row.reference,
        dl = opt_ratio(row.delta()),
        pn = opt_ns(row.pytorch_ns),
        pd = opt_ratio(row.pytorch_delta()),
    );
}

// ---------------------------------------------------------------------
// Phase 29 — cross-implementation shape sweeps
// ---------------------------------------------------------------------

/// Small `(M, N, K)` sweep for the cross-impl GEMM bench. Smaller than
/// the Phase-10 sweep so the cross-impl run stays under ~5 min per
/// dtype. `M = 1` covers decode, `M = 32 / 128` covers prefill.
pub const CROSS_GEMM_M_SWEEP: &[i32] = &[1, 32, 128];
/// Square `K == N` values for the cross-impl GEMM bench. 7B (4096) is
/// the most common hidden-size — keep it as the central pick.
pub const CROSS_GEMM_KN_SWEEP: &[i32] = &[2048, 4096];

/// Sequence lengths to sweep for the softmax / norm benches. Cover a
/// short (512) and long (4096) row.
pub const CROSS_SEQLEN_SWEEP: &[i32] = &[512, 2048, 4096];
/// Hidden / feature dim to sweep for the softmax / norm benches.
pub const CROSS_HIDDEN_SWEEP: &[i32] = &[1024, 4096];

/// MMVQ sweep: `(nrows, ncols)`. Mirrors transformer decode-step matmul
/// shapes: 4096×4096 (Q/K/V projection), 4096×11008 (Llama-2 7B FFN
/// up_proj), 32000×4096 (Llama-2 7B LM head). 11008 is a multiple of
/// 256, satisfying every k-quant block-size constraint.
pub const CROSS_MMVQ_SHAPES: &[(i32, i32)] = &[(4096, 4096), (11008, 4096), (32000, 4096)];

/// Loss bench sweep — row count (batch·seqlen) for the Phase-29 loss
/// family (mse / l1 / cross_entropy / nll). Short (512) and long (2048).
pub const LOSS_ROW_SWEEP: &[i32] = &[512, 2048];
/// Loss bench sweep — feature / class extent. For cross_entropy / nll
/// this is the class count (vocab-like); for mse / l1 it is the per-row
/// feature width. 1024 and 4096 bracket a small head and an LLM hidden
/// size. The shape key is `R{rows}_C{cols}`, shared with the PyTorch
/// baseline generator (`tools/refresh_pytorch_baseline.py`).
pub const LOSS_COL_SWEEP: &[i32] = &[1024, 4096];

// Conv2d shape set — same as `CONV2D_SWEEP` (the Phase-10 sweep is
// already minimal at 3 picks).

/// Pool2d shape set: NCHW (1, 64, 56, 56) is the ResNet stem after
/// conv1; (1, 256, 14, 14) is a deep-stage feature map. Window 3×3,
/// stride 2, pad 1.
#[derive(Copy, Clone, Debug)]
pub struct PoolShape {
    /// Batch size.
    pub n: i32,
    /// Channels.
    pub c: i32,
    /// Input height.
    pub h: i32,
    /// Input width.
    pub w: i32,
    /// Pooling window (square).
    pub k: i32,
    /// Stride (square).
    pub stride: i32,
    /// Padding (square).
    pub pad: i32,
}

/// Pool sweep (3 picks): stem, mid, deep.
pub const POOL_SWEEP: &[PoolShape] = &[
    PoolShape {
        n: 1,
        c: 64,
        h: 56,
        w: 56,
        k: 3,
        stride: 2,
        pad: 1,
    },
    PoolShape {
        n: 1,
        c: 128,
        h: 28,
        w: 28,
        k: 3,
        stride: 2,
        pad: 1,
    },
    PoolShape {
        n: 1,
        c: 256,
        h: 14,
        w: 14,
        k: 3,
        stride: 2,
        pad: 1,
    },
];

/// GGUF block formats to sweep in the MMVQ bench. All have an MMVQ
/// kernel wired. Q4_0 / Q4_K / Q8_0 / Q6_K is a representative spread:
/// the two most common 4-bit formats, the most common 8-bit format, and
/// the 6-bit k-quant.
pub const CROSS_MMVQ_FORMATS: &[baracuda_kernels::GgufBlockFormat] = &[
    baracuda_kernels::GgufBlockFormat::Q4_0,
    baracuda_kernels::GgufBlockFormat::Q4K,
    baracuda_kernels::GgufBlockFormat::Q6K,
    baracuda_kernels::GgufBlockFormat::Q8_0,
];

// =====================================================================
// Liveness — a bench that times garbage is fast and meaningless.
// =====================================================================

/// A scalar whose finiteness the liveness check can test. Implemented for the
/// float element types the benches produce (`f32`, `f64`, `half::f16`,
/// `half::bf16`).
///
/// This is deliberately NOT a numerical reference: it says nothing about
/// whether a value is *correct*, only whether it is finite. "This op ran and
/// produced live output" vs "this op agrees with PyTorch" — only the latter is
/// the oracle's job (`kiss-ref-diff` + on-device parity tests).
pub trait LiveScalar: Copy {
    /// True if the value is finite (not NaN, not ±Inf).
    fn cell_is_finite(&self) -> bool;
    /// The additive identity, to fill a host readback buffer.
    fn live_zero() -> Self;
}

impl LiveScalar for f32 {
    fn cell_is_finite(&self) -> bool {
        f32::is_finite(*self)
    }
    fn live_zero() -> Self {
        0.0
    }
}
impl LiveScalar for f64 {
    fn cell_is_finite(&self) -> bool {
        f64::is_finite(*self)
    }
    fn live_zero() -> Self {
        0.0
    }
}
impl LiveScalar for half::f16 {
    fn cell_is_finite(&self) -> bool {
        half::f16::is_finite(*self)
    }
    fn live_zero() -> Self {
        half::f16::ZERO
    }
}
impl LiveScalar for half::bf16 {
    fn cell_is_finite(&self) -> bool {
        half::bf16::is_finite(*self)
    }
    fn live_zero() -> Self {
        half::bf16::ZERO
    }
}

/// Liveness SHAPE arm — the output's element count must match the declared
/// extent. Split out from [`assert_cell_live`] so it is testable without a GPU
/// (born-red: a length mismatch must panic). Panics on mismatch.
fn check_cell_shape(label: &str, actual_len: usize, expected_numel: usize) {
    assert_eq!(
        actual_len, expected_numel,
        "{label}: liveness — output length {actual_len} != expected numel {expected_numel} (wrong shape)",
    );
}

/// Liveness FINITENESS arm — every host-side element must be finite. Split out
/// from [`assert_cell_live`] so it is testable without a GPU (born-red: a slice
/// carrying a NaN or ±Inf must panic). Panics on the first non-finite element.
fn check_cell_finite<T: LiveScalar>(label: &str, host: &[T]) {
    if let Some(i) = host.iter().position(|v| !v.cell_is_finite()) {
        panic!(
            "{label}: liveness — element {i} of {} is non-finite (NaN/Inf); \
             a benchmark timing garbage is meaningless",
            host.len(),
        );
    }
}

/// LIVENESS assertion for one benched cell — NOT a numerical reference.
///
/// A benchmark that times an op returning NaN, the wrong shape, or exiting
/// early on a degenerate input still produces a fast number, and today that is
/// silent. This copies the op's output back once (call it OUTSIDE the timed
/// loop, after `warmup` has populated the output) and asserts the output is
/// live via its two runtime arms: element count matches `expected_numel`
/// ([`check_cell_shape`]) and every element is finite ([`check_cell_finite`]).
/// It distinguishes "this op ran" from "this op is correct"; only correctness
/// belongs to the oracle (`kiss-ref-diff` + on-device parity tests).
///
/// DTYPE is deliberately NOT a runtime arm: the signature is
/// `assert_cell_live<T>(&DeviceBuffer<T>, _)`, so the cell's dtype *is* `T` by
/// construction and a wrong-dtype buffer is UNREPRESENTABLE — the type system
/// enforces it, which is strictly stronger than a runtime check. Hence there is
/// no dtype born-red case (there is no failing input to construct), and its
/// absence is a statement, not an unproven arm.
///
/// Panics — a benchmark over dead output is not worth timing. `T: Element`
/// carries `DeviceRepr` (the `DeviceBuffer<T>` bound) as a supertrait.
pub fn assert_cell_live<T>(
    label: &str,
    output: &baracuda_driver::DeviceBuffer<T>,
    expected_numel: usize,
) where
    T: baracuda_kernels::Element + LiveScalar,
{
    check_cell_shape(label, output.len(), expected_numel);
    let mut host = vec![T::live_zero(); expected_numel];
    output
        .copy_to_host(&mut host)
        .unwrap_or_else(|e| panic!("{label}: liveness — copy_to_host failed: {e}"));
    check_cell_finite(label, &host);
}

// =====================================================================
// Phase 73.1 — PyTorch frozen-JSON baseline loader.
// =====================================================================

/// File-level metadata block from the PyTorch baseline JSON. Holds only the
/// facts that are INVARIANT across generation runs. Per-run facts (torch/cuda
/// version, device, git SHA, dirty bit, sample counts, attribution) live in
/// [`provenance_runs`](Self::provenance_runs), one record per run, and each
/// result row names the run that produced it — so a partial `--ops` refresh
/// cannot relabel untouched rows with a provenance that did not produce them.
///
/// Schema authored in `tools/refresh_pytorch_baseline.py`.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct PytorchBaselineMetadata {
    /// JSON schema version. Increment if the format changes
    /// incompatibly; loaders should refuse mismatched versions.
    pub schema_version: u32,
    /// Human-readable methodology blurb (invariant across runs).
    pub methodology: String,
    /// The stated CONDITION (never a date) under which this baseline must be
    /// regenerated, plus the torch major it was frozen at. `None` on v1 —
    /// a baseline without a regen-trigger has no detectable staleness, so
    /// [`PytorchBaseline::load_from`] warns when it is absent.
    #[serde(default)]
    pub regen_trigger: Option<PytorchRegenTrigger>,
    /// One provenance record per generation run. A full refresh yields one
    /// run that every row references; a partial `--ops` refresh appends a run
    /// and repoints only the rows it regenerated. Empty on a v1 (flat-metadata)
    /// baseline — the loader warns, since such rows are unattributed.
    #[serde(default)]
    pub provenance_runs: Vec<PytorchProvenanceRun>,
}

/// One generation run's provenance — the natural key for "what produced these
/// numbers." A generation run is the thing that actually has a torch version, a
/// device, a git SHA and a dirty bit; keying provenance on the run (rather than
/// duplicating twelve fields across every row) makes drift between a row and its
/// provenance structurally impossible instead of merely currently-absent.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct PytorchProvenanceRun {
    /// Stable id the result rows reference (e.g. `"provisional-2026-06-04"`).
    pub run_id: String,
    /// e.g. `"2.11.0+cu130"`.
    pub torch_version: String,
    /// e.g. `"13.0"`.
    pub cuda_version: String,
    /// Full device name as `torch.cuda.get_device_name(0)` reports.
    pub device_name: String,
    /// `(major, minor)` from `torch.cuda.get_device_capability(0)`.
    pub device_capability: [u32; 2],
    /// ISO-8601 UTC timestamp of the run.
    pub generated_at_utc: String,
    /// Commit SHA of the source tree the run was generated against. `None` when
    /// unknown (e.g. the provisional run predates the field) — honestly absent,
    /// not silently attributed.
    #[serde(default)]
    pub generator_git_sha: Option<String>,
    /// Whether the generating tree was dirty (`git status --porcelain`
    /// non-empty). `Some(true)` means the SHA does not fully describe the tree;
    /// `None` when unknown (predates the field). Never suppressed — a bare SHA
    /// that implies a clean tree is the lie this records against.
    #[serde(default)]
    pub generator_dirty: Option<bool>,
    /// Number of independent timing batches the median is over.
    pub sample_count: u32,
    /// Launches per timing batch.
    pub inner_iters: u32,
    /// Warmup launches before the first timed sample.
    pub warmup_launches: u32,
    /// Human-readable attribution: what device / tree / toolchain produced
    /// these numbers, and (for provisional runs) what is UNVERIFIED about that.
    #[serde(default)]
    pub attribution: Option<String>,
}

/// The regeneration-trigger policy from a v2 baseline. The bench harness does
/// not act on it directly (staleness DETECTION is the scheduled
/// `pytorch-baseline-liveness` workflow's job); it is carried so a reader —
/// human or the liveness job — can see the stated condition and the torch
/// major the baseline was frozen at.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct PytorchRegenTrigger {
    /// The stated condition under which to regenerate (a condition, not a date).
    pub policy: String,
    /// The PyTorch MAJOR version the baseline was generated under. The
    /// liveness workflow compares this against the current released major.
    pub torch_major_at_gen: u32,
    /// Notes on which covered ops' numerics the baseline depends on.
    #[serde(default)]
    pub covered_op_numerics_notes: Option<String>,
}

/// One per-cell timing entry in the baseline.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct PytorchBaselineEntry {
    /// Op family, e.g. `"gemm"`.
    pub op: String,
    /// Shape descriptor matching `PhaseTwentyNineRow::shape`.
    pub shape: String,
    /// Dtype label matching `PhaseTwentyNineRow::dtype`.
    pub dtype: String,
    /// Median per-launch wall-clock nanoseconds from PyTorch.
    pub median_ns: f64,
    /// The [`PytorchProvenanceRun::run_id`] that produced this row. Empty on a
    /// v1 (flat-metadata) baseline whose rows carried no per-run link.
    #[serde(default)]
    pub run: String,
}

/// In-memory representation of a PyTorch baseline JSON file. Built by
/// [`PytorchBaseline::load_from`] / [`PytorchBaseline::load_default`].
#[derive(Clone, Debug)]
pub struct PytorchBaseline {
    /// Self-describing metadata block.
    pub metadata: PytorchBaselineMetadata,
    /// `(op, shape, dtype) → median_ns`. O(1) lookup.
    by_key: std::collections::HashMap<(String, String, String), f64>,
}

#[derive(serde::Deserialize)]
struct PytorchBaselineFile {
    metadata: PytorchBaselineMetadata,
    results: Vec<PytorchBaselineEntry>,
}

impl PytorchBaseline {
    /// Parse a baseline JSON from `path`. Returns `Err` with a human-
    /// readable message on parse / IO failure.
    pub fn load_from(path: &std::path::Path) -> Result<Self, String> {
        let raw = std::fs::read(path)
            .map_err(|e| format!("pytorch baseline: failed to read {}: {e}", path.display()))?;
        let parsed: PytorchBaselineFile = serde_json::from_slice(&raw).map_err(|e| {
            format!(
                "pytorch baseline: failed to parse {} as JSON: {e}",
                path.display()
            )
        })?;
        // v1 (timing-only, no provenance/regen-trigger) and v2 (adds
        // generator_git_sha + attribution + regen_trigger) are both readable —
        // v2's new fields are `#[serde(default)]`, so a v1 file parses with them
        // as `None`. A future incompatible change bumps past 2 and lands here.
        if !(1..=2).contains(&parsed.metadata.schema_version) {
            return Err(format!(
                "pytorch baseline: schema_version {} not supported (expected 1 or 2)",
                parsed.metadata.schema_version
            ));
        }
        // Condition 2 (regen-trigger) visibility: a baseline with no stated
        // regen-trigger has UNDETECTABLE staleness — surface that at load time
        // rather than letting the comparison silently age. Not an error: a v1
        // baseline is still usable, it is just not self-describing about when
        // to refresh it.
        if parsed.metadata.regen_trigger.is_none() {
            eprintln!(
                "pytorch baseline: WARNING — {} (schema_version {}) carries no regen_trigger; \
                 its staleness is undetectable. Regenerate with tools/refresh_pytorch_baseline.py \
                 to stamp a v2 provenance + regen-trigger block.",
                path.display(),
                parsed.metadata.schema_version,
            );
        }
        // Provenance visibility: v2 rows name the run that produced them; a file
        // with no provenance_runs (a v1 flat-metadata baseline, or a malformed
        // one) has UNATTRIBUTED rows. Surface it rather than presenting
        // unattributed timings as if they were attributed.
        if parsed.metadata.provenance_runs.is_empty() {
            eprintln!(
                "pytorch baseline: WARNING — {} carries no provenance_runs; its timing rows \
                 are unattributed (no device / torch / git-SHA on record). Regenerate to stamp \
                 per-run provenance.",
                path.display(),
            );
        }
        let by_key = parsed
            .results
            .into_iter()
            .map(|e| ((e.op, e.shape, e.dtype), e.median_ns))
            .collect();
        Ok(Self {
            metadata: parsed.metadata,
            by_key,
        })
    }

    /// Resolve the default baseline file for the crate-local
    /// `bench-baselines/` directory.
    ///
    /// Resolution rule: prefer a single `pytorch_*.json` file in the
    /// baselines directory. If there are zero or multiple matches, log
    /// the situation and return `None` — the bench harness then runs
    /// without a PyTorch column (the `pytorch_ns` field stays `None`
    /// on every emitted row).
    ///
    /// In CI we expect exactly one baseline per (device, torch version)
    /// the run targets; matching by filename keeps the harness honest
    /// about which JSON it actually loaded (printed at startup).
    ///
    /// Path resolution uses `CARGO_MANIFEST_DIR` baked in at compile
    /// time. This sidesteps the cargo-bench-process-CWD quirk where
    /// the bench binary runs from the bench crate root, not the
    /// workspace root.
    pub fn load_default() -> Option<Self> {
        let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench-baselines");
        let entries: Vec<_> = match std::fs::read_dir(&dir) {
            Ok(it) => it
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("pytorch_") && n.ends_with(".json"))
                })
                .collect(),
            Err(_) => {
                eprintln!("pytorch baseline: {} not found — skipping", dir.display());
                return None;
            }
        };
        match entries.len() {
            0 => {
                eprintln!(
                    "pytorch baseline: no pytorch_*.json in {} — skipping",
                    dir.display()
                );
                None
            }
            1 => match Self::load_from(&entries[0]) {
                Ok(b) => {
                    // Summarize the provenance run(s) — a v2 file usually has one
                    // (a full refresh); a partial-refresh file has more. Name them
                    // so the startup line stays honest about which torch/device the
                    // cells came from, rather than implying a single run.
                    let runs = &b.metadata.provenance_runs;
                    let run_summary = if runs.is_empty() {
                        "unattributed (no provenance_runs)".to_string()
                    } else {
                        runs.iter()
                            .map(|r| {
                                format!(
                                    "{} [torch {}, {}]",
                                    r.run_id, r.torch_version, r.device_name
                                )
                            })
                            .collect::<Vec<_>>()
                            .join("; ")
                    };
                    eprintln!(
                        "pytorch baseline: loaded {} ({} cells; run(s): {run_summary})",
                        entries[0].display(),
                        b.by_key.len(),
                    );
                    Some(b)
                }
                Err(e) => {
                    eprintln!("pytorch baseline: {e}");
                    None
                }
            },
            n => {
                eprintln!(
                    "pytorch baseline: {} files in {} — ambiguous, skipping. Found:",
                    n,
                    dir.display()
                );
                for e in &entries {
                    eprintln!("  - {}", e.display());
                }
                None
            }
        }
    }

    /// O(1) lookup. Returns the PyTorch median wall-clock ns for the
    /// matching `(op, shape, dtype)` cell, or `None` if absent.
    pub fn lookup(&self, op: &str, shape: &str, dtype: &str) -> Option<f64> {
        self.by_key
            .get(&(op.to_owned(), shape.to_owned(), dtype.to_owned()))
            .copied()
    }
}

#[cfg(test)]
mod gate_tests {
    use super::*;
    use baracuda_kernels_types::{
        DispatchEntry, DispatchTable, ElementKind, OpCategory, OperandDesc, Provenance, merge,
        structure_key,
    };

    #[test]
    fn arch_sku_maps_capabilities() {
        assert_eq!(arch_sku_of(8, 9), Some(ArchSku::Sm89)); // Ada — RTX 4070
        assert_eq!(arch_sku_of(8, 0), Some(ArchSku::Sm80)); // A100
        assert_eq!(arch_sku_of(8, 6), Some(ArchSku::Sm80)); // consumer Ampere
        assert_eq!(arch_sku_of(9, 0), Some(ArchSku::Sm90a)); // Hopper
        assert_eq!(arch_sku_of(7, 5), None); // Turing — no built cell
    }

    // Liveness helper — UNIVERSAL born-red: each runtime arm reds on ITS OWN
    // violation, AND both arms pass on correct input (the sign-flipped check —
    // a guard false-of-the-fixed-state is the same error as one true-of-the-
    // broken-state). Driver-free: these exercise the pure `check_cell_*` /
    // `cell_is_finite` split-outs, so they run in the CI-safe path; the
    // device-copy path in `assert_cell_live` is validated on-device. Dtype has
    // NO born-red case by design — it is type-enforced (a wrong-dtype buffer is
    // unrepresentable), documented on `assert_cell_live`.

    #[test]
    fn cell_is_finite_discriminates_per_dtype() {
        // true on finite, false on NaN/±Inf — for every dtype the benches use.
        assert!(1.0f32.cell_is_finite());
        assert!(!f32::NAN.cell_is_finite());
        assert!(!f32::INFINITY.cell_is_finite());
        assert!(!f32::NEG_INFINITY.cell_is_finite());
        assert!(1.0f64.cell_is_finite());
        assert!(!f64::NAN.cell_is_finite());
        assert!(!f64::INFINITY.cell_is_finite());
        assert!(half::f16::ONE.cell_is_finite());
        assert!(!half::f16::NAN.cell_is_finite());
        assert!(!half::f16::INFINITY.cell_is_finite());
        assert!(half::bf16::ONE.cell_is_finite());
        assert!(!half::bf16::NAN.cell_is_finite());
        assert!(!half::bf16::INFINITY.cell_is_finite());
    }

    // FINITE arm — reds on its own violation (NaN, +Inf; a half NaN too).
    #[test]
    #[should_panic(expected = "non-finite")]
    fn check_cell_finite_reds_on_nan() {
        check_cell_finite("t", &[1.0f32, f32::NAN, 2.0]);
    }
    #[test]
    #[should_panic(expected = "non-finite")]
    fn check_cell_finite_reds_on_inf() {
        check_cell_finite("t", &[f32::INFINITY]);
    }
    #[test]
    #[should_panic(expected = "non-finite")]
    fn check_cell_finite_reds_on_f16_nan() {
        check_cell_finite("t", &[half::f16::ONE, half::f16::NAN]);
    }
    // FINITE arm — passes on correct (not false-of-the-fixed-state).
    #[test]
    fn check_cell_finite_passes_on_all_finite() {
        check_cell_finite("t", &[1.0f32, 2.0, 3.0]);
        check_cell_finite("t", &[half::bf16::ONE, half::bf16::ZERO]);
    }

    // SHAPE arm — reds on a length mismatch, passes on a match.
    #[test]
    #[should_panic(expected = "wrong shape")]
    fn check_cell_shape_reds_on_mismatch() {
        check_cell_shape("t", 5, 6);
    }
    #[test]
    fn check_cell_shape_passes_on_match() {
        check_cell_shape("t", 6, 6);
    }

    // These driver-free serde tests prove (a) a v2 baseline's per-run
    // provenance round-trips, (b) a partial-refresh file with TWO runs keeps
    // each row's provenance distinct, and (c) a v1 (flat-metadata) file still
    // parses with provenance_runs empty — unattributed, NOT falsely attributed.
    // Pure serde — no GPU, runs in the CI-safe test path.

    const V2_FILE: &str = r#"{
        "metadata": {
            "schema_version": 2,
            "methodology": "…",
            "regen_trigger": {
                "policy": "regenerate on a torch MAJOR bump or a documented numerics change",
                "torch_major_at_gen": 2,
                "covered_op_numerics_notes": "timing-only"
            },
            "provenance_runs": [
                {
                    "run_id": "provisional-2026-06-04",
                    "torch_version": "2.11.0+cu130", "cuda_version": "13.0",
                    "device_name": "NVIDIA GeForce RTX 4070 Laptop GPU",
                    "device_capability": [8, 9],
                    "generated_at_utc": "2026-06-04T15:02:50+00:00",
                    "generator_git_sha": null, "generator_dirty": null,
                    "sample_count": 11, "inner_iters": 50, "warmup_launches": 10,
                    "attribution": "provisional — device MODEL matches; machine UNVERIFIED"
                },
                {
                    "run_id": "2026-09-02T00:00:00+00:00",
                    "torch_version": "2.11.0+cu130", "cuda_version": "13.0",
                    "device_name": "NVIDIA GeForce RTX 4070 Laptop GPU",
                    "device_capability": [8, 9],
                    "generated_at_utc": "2026-09-02T00:00:00+00:00",
                    "generator_git_sha": "abc1234", "generator_dirty": true,
                    "sample_count": 11, "inner_iters": 50, "warmup_launches": 10,
                    "attribution": "same-box; dirty tree"
                }
            ]
        },
        "results": [
            {"op": "gemm", "shape": "M1_K2048", "dtype": "f32", "median_ns": 100.0, "run": "provisional-2026-06-04"},
            {"op": "gemm", "shape": "M1_K2048", "dtype": "f16", "median_ns": 90.0, "run": "2026-09-02T00:00:00+00:00"}
        ]
    }"#;

    // v1: flat metadata, no provenance_runs, rows carry no `run`.
    const V1_FILE: &str = r#"{
        "metadata": {
            "schema_version": 1,
            "torch_version": "2.11.0+cu130", "cuda_version": "13.0",
            "device_name": "NVIDIA GeForce RTX 4070 Laptop GPU",
            "device_capability": [8, 9],
            "generated_at_utc": "2026-06-04T15:02:50+00:00",
            "sample_count": 11, "inner_iters": 50, "warmup_launches": 10,
            "methodology": "…"
        },
        "results": [
            {"op": "gemm", "shape": "M1_K2048", "dtype": "f32", "median_ns": 100.0}
        ]
    }"#;

    #[test]
    fn pytorch_baseline_v2_per_run_provenance_round_trips() {
        let f: PytorchBaselineFile = serde_json::from_str(V2_FILE).expect("v2 parses");
        assert_eq!(f.metadata.schema_version, 2);
        let t = f
            .metadata
            .regen_trigger
            .expect("v2 carries a regen_trigger");
        assert_eq!(t.torch_major_at_gen, 2);
        // TWO runs, and each row names a DISTINCT one — a partial refresh cannot
        // relabel the untouched row.
        assert_eq!(f.metadata.provenance_runs.len(), 2);
        let prov = &f.metadata.provenance_runs[0];
        assert_eq!(prov.run_id, "provisional-2026-06-04");
        assert_eq!(prov.generator_git_sha, None); // honestly absent
        assert_eq!(prov.generator_dirty, None); // unknown, not "clean"
        assert!(prov.attribution.is_some());
        let dirty_run = &f.metadata.provenance_runs[1];
        assert_eq!(dirty_run.generator_git_sha.as_deref(), Some("abc1234"));
        assert_eq!(dirty_run.generator_dirty, Some(true)); // recorded, never suppressed
        assert_eq!(f.results[0].run, "provisional-2026-06-04");
        assert_eq!(f.results[1].run, "2026-09-02T00:00:00+00:00");
    }

    #[test]
    fn pytorch_baseline_v1_parses_unattributed_not_falsely_attributed() {
        // Backward compatibility: a v1 (flat-metadata) file still parses, with
        // provenance_runs EMPTY and rows carrying no run — unattributed, which
        // the loader warns on. It is NOT silently attributed to a synthesized run.
        let f: PytorchBaselineFile = serde_json::from_str(V1_FILE).expect("v1 parses");
        assert_eq!(f.metadata.schema_version, 1);
        assert!(f.metadata.regen_trigger.is_none());
        assert!(
            f.metadata.provenance_runs.is_empty(),
            "a v1 baseline has no provenance_runs; the loader warns it is unattributed"
        );
        assert_eq!(f.results[0].run, ""); // no per-run link on v1 rows
    }

    /// On-device smoke: the RTX 4070 stamps as sm89, and `gate_cell` times two
    /// candidates into a measured `DispatchEntry` that `merge` folds over a seed.
    /// Ignored by default (needs a live GPU); run with `--ignored`.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn hwstamp_and_gate_on_device() {
        let (ctx, stream) = setup_device();
        let device = Device::get(0).expect("device");
        let stamp = current_hwstamp(&device).expect("hwstamp");
        assert_eq!(
            stamp.target,
            baracuda_kernels_types::TargetId::from(ArchSku::Sm89),
            "RTX 4070 is Ada/sm89"
        );
        assert!(!stamp.device_name.is_empty());

        let a = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        // Two trivial (no-op) candidates — enough to exercise the time→reduce loop.
        let cands: Vec<(Implementor, Option<String>, Box<dyn FnMut()>)> = vec![
            (Implementor::Generated, Some("gen".into()), Box::new(|| {})),
            (Implementor::Cublas, None, Box::new(|| {})),
        ];
        let entry = gate_cell(&ctx, &stream, &key, Some(stamp), 5, 10, cands).expect("entry");
        assert_eq!(entry.provenance, Provenance::Measured);
        assert_eq!(entry.ranked.len(), 2, "both candidates timed");
        assert!(entry.margin.is_finite() && entry.margin >= 1.0);

        // The measured entry folds into a seeded table via the item-07 merge seam.
        let mut table = DispatchTable::from_entries(vec![DispatchEntry::seeded(
            key.to_token(),
            Implementor::Cublas,
        )]);
        merge(&mut table, &[entry]);
        assert!(table.lookup(&key).is_some(), "cell is routed after merge");
    }
}
