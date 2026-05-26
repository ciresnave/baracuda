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

use baracuda_driver::{init, Context, Device, Event, Stream};

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
// CUDA-event-timed measurement
// ---------------------------------------------------------------------

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
pub fn time_with_events<F>(
    ctx: &Context,
    stream: &Stream,
    iters: u64,
    mut launch: F,
) -> Duration
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

/// Append a `PhaseTwentyNineRow` to `target/criterion/phase29/<bench>.csv`,
/// creating the file (with header) if it doesn't exist.
///
/// CSV columns: `op,shape,dtype,baracuda_ns,reference_ns,reference,delta`.
///
/// Errors are swallowed (printed to stderr) — bench correctness mustn't
/// depend on the CSV write succeeding. The criterion HTML report is the
/// primary record; the CSV is a convenience for `BENCHMARKS.md` updates.
pub fn append_csv_row(bench: &str, row: &PhaseTwentyNineRow) {
    use std::io::Write;

    let dir = std::path::PathBuf::from("target")
        .join("criterion")
        .join("phase29");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("phase29 csv: mkdir {} failed: {e}", dir.display());
        return;
    }
    let path = dir.join(format!("{bench}.csv"));
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
        let _ = writeln!(f, "op,shape,dtype,baracuda_ns,reference_ns,reference,delta");
    }
    let ref_ns = row
        .reference_ns
        .map(|x| format!("{x:.3}"))
        .unwrap_or_else(|| "".into());
    let delta = row
        .delta()
        .map(|x| format!("{x:.4}"))
        .unwrap_or_else(|| "".into());
    let _ = writeln!(
        f,
        "{op},{shape},{dtype},{ba:.3},{rf},{rl},{dl}",
        op = row.op,
        shape = row.shape,
        dtype = row.dtype,
        ba = row.baracuda_ns,
        rf = ref_ns,
        rl = row.reference,
        dl = delta,
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
pub const CROSS_MMVQ_SHAPES: &[(i32, i32)] = &[
    (4096, 4096),
    (11008, 4096),
    (32000, 4096),
];

/// Conv2d shape set — same as `CONV2D_SWEEP` (the Phase-10 sweep is
/// already minimal at 3 picks).

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
