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
