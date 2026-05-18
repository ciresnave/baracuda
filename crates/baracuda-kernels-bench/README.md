# baracuda-kernels-bench

Benchmark harness for [`baracuda-kernels`] — CUDA-event-timed
throughput sweeps across GEMM, Flash Attention, and Conv2d at
LLM-typical and ResNet-typical shapes. Phase 10 of the comprehensive
plan.

This crate is **not published to crates.io** (`publish = false` in
`Cargo.toml`) — it requires a working NVIDIA GPU at test time and
exists to support the kernel-tuning workflow, not as a downstream
dependency.

## Bench files

| File | Scope |
| --- | --- |
| `benches/gemm.rs` | float + int8 + FP8 GEMM at `K = N ∈ {2048, 4096, 8192}` × `M ∈ {1, 8, 32, 128, 512}` |
| `benches/flash_attention.rs` | Flash SDPA at `D ∈ {64, 128}`, `H ∈ {8, 16, 32}`, `Q ∈ {512, 1024, 2048, 4096}`, `is_causal = false` |
| `benches/conv2d.rs` | NCHW Conv2d at ResNet-typical shapes: stem (56×56, 64→64, 3×3), mid (28×28, 128→128, 3×3), deep (14×14, 256→256, 3×3) |

## Usage

Run the full sweep (takes ~30 minutes on an RTX 4070):

```bash
cargo bench -p baracuda-kernels-bench --features sm89,cudnn
```

Scope to one family while iterating on a perf change:

```bash
cargo bench -p baracuda-kernels-bench --bench gemm --features sm89,cudnn
cargo bench -p baracuda-kernels-bench --bench flash_attention --features sm89
cargo bench -p baracuda-kernels-bench --bench conv2d --features sm89,cudnn
```

Add `-- --quick` for criterion's 10-sample fast pass — useful when you
just want to see whether a change moved the median in the right
direction before committing to a full sweep.

## Methodology

- **CUDA-event timing** (`cudaEventRecord` + `cudaEventElapsedTime`) —
  measured on-device, no host-side noise.
- **10-launch warmup** before the first timed sample so GPU clock +
  cache state settle out of idle.
- **Plan + buffers built once** per shape, reused across all timed
  samples (the bench measures `run()`, not `select()`).
- **`EpilogueKind::Identity`** for GEMM (pure kernel, no bias
  overhead).
- **`is_causal = false`** for Flash Attention (worst-case work).
- **`pad = k/2, stride = 1`** for Conv2d (`H_out == H_in` — same-pad
  convention).
- **Fill values of `1.0`** in dtype-appropriate units so the kernel
  exercises real FMA paths; zero-fill triggers early-exit code paths
  in some kernels and under-reports throughput.

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `sm80` | no | Ampere baseline. |
| `sm89` | no | Adds FP8 GEMM benches and the sm_89 Flash Attention sibling. |
| `sm90a` | no | Hopper specializations. |
| `cudnn` | no | Gates the Conv2d bench (the bench file becomes a no-op `main` when off, so the binary still builds). |

## Baseline results

Recorded baseline tables for the RTX 4070 (sm_89, CUDA 13.0, cuDNN
9.16) live in [`BENCH-sm89.md`](BENCH-sm89.md). Numbers there are
placeholders until each cell is populated from a real run on the
target hardware.

[`baracuda-kernels`]: ../baracuda-kernels
