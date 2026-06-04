# Phase 29 Cross-Implementation Benchmarks

This file is the structured summary of baracuda's load-bearing op
performance compared to NVIDIA library references (cuBLAS / cuDNN) and
self-bench baselines. The criterion HTML reports under
`target/criterion/` are the primary source; the tables below are the
hand-maintained roll-up.

**Reading the `delta` column**: `delta = reference_ns / baracuda_ns`.

- `delta < 1.0` ⇒ baracuda is faster than the reference.
- `delta > 1.0` ⇒ reference is faster than baracuda.
- `delta ≈ 1.0` ⇒ same kernel (expected for cuDNN-backed ops where
  baracuda's plan just wraps the cuDNN call).

**Hardware**: RTX 4070 (sm_89), CUDA 13.0, cuDNN 9.x.
**Build**: `cargo bench -p baracuda-kernels-bench --features sm89,cudnn`.

## Bench inventory

| Bench file | Ops | Reference | Shapes |
| --- | --- | --- | --- |
| `gemm_vs_cublas` | GEMM f32 / f16 / bf16 | cuBLAS (`sgemm` / `gemmEx`) | M ∈ {1, 32, 128}, K=N ∈ {2048, 4096} |
| `mmvq` | GGUF MMVQ (Q4_0, Q4_K, Q6_K, Q8_0) × f32 / f16 / bf16 | self (no library equiv) | (4096×4096), (11008×4096), (32000×4096) |
| `softmax_vs_cudnn` | Softmax f32 / f16 | cuDNN `softmax_forward` | rows ∈ {512, 2048, 4096}, hidden ∈ {1024, 4096} |
| `layernorm_vs_cudnn` | LayerNorm f32 / f16 | self (cuDNN classic LN not wired) | rows × hidden, same as softmax |
| `rmsnorm` | RMSNorm f32 / f16 / bf16 | self (no library equiv) | rows × hidden, same as softmax |
| `conv2d_vs_cudnn` | Conv2d f32 / f16 | raw cuDNN `convolution_forward` (baracuda is cuDNN-backed — measures wrapper overhead) | ResNet-50 picks (3) |
| `pool_vs_cudnn` | MaxPool2d f32 / f16 | raw cuDNN `pooling_forward` | ResNet-50 picks (3) |
| `reductions_vs_cudnn` | Sum / Max / Mean f32 | cuDNN `reduce_tensor` | rows × hidden, same as softmax |
| `elementwise` | Add / Mul / ReLU / GELU × f32 / f16 | self | numel ∈ {1M, 16M} |
| `sdpa_gqa` | Flash SDPA + GQA broadcast (f16 / bf16) | self | H_q=32, H_kv ∈ {32, 1}, Q=K=2048, D=128 |

Also see the Phase 10 baseline benches (`gemm.rs`, `flash_attention.rs`,
`conv2d.rs`) for wider per-dtype shape sweeps without the cross-impl
overlay.

## Running

Build all bench binaries:

```bash
cargo bench -p baracuda-kernels-bench --no-run --features sm89,cudnn
```

Run a single bench file (criterion HTML report lands at
`target/criterion/<group_name>/report/index.html`):

```bash
cargo bench -p baracuda-kernels-bench --bench gemm_vs_cublas --features sm89,cudnn
cargo bench -p baracuda-kernels-bench --bench mmvq --features sm89
cargo bench -p baracuda-kernels-bench --bench softmax_vs_cudnn --features sm89,cudnn
cargo bench -p baracuda-kernels-bench --bench layernorm_vs_cudnn --features sm89,cudnn
cargo bench -p baracuda-kernels-bench --bench rmsnorm --features sm89
cargo bench -p baracuda-kernels-bench --bench conv2d_vs_cudnn --features sm89,cudnn
cargo bench -p baracuda-kernels-bench --bench pool_vs_cudnn --features sm89,cudnn
cargo bench -p baracuda-kernels-bench --bench reductions_vs_cudnn --features sm89,cudnn
cargo bench -p baracuda-kernels-bench --bench elementwise --features sm89
cargo bench -p baracuda-kernels-bench --bench sdpa_gqa --features sm89
```

Each bench emits a CSV companion at
`target/criterion/phase29/<bench>.csv` with columns
`op,shape,dtype,baracuda_ns,reference_ns,reference,delta`. The CSV is
the input for updating the tables below.

`-- --quick` passes criterion's reduced-sample fast path (10 samples vs
the default 100) — useful while iterating on a perf change.

## Sample results — RTX 4070 (representative)

The harness validation run was executed on `gemm_vs_cublas` (the
fastest cross-impl bench, ~2 minutes total under `-- --quick`). Other
bench files compile + link cleanly but their full sweeps were not run
end-to-end in the Phase 29 harness-validation slot; runners should
rerun them as part of release-validation and update the tables below
from each bench's `target/criterion/phase29/<bench>.csv`.

### gemm (f32) — RTX 4070, 2026-05-26, `-- --quick`

| Shape (M×K=N) | baracuda (us) | cuBLAS sgemm (us) | delta (cuBLAS/baracuda) |
| --- | --- | --- | --- |
| M1_N2048_K2048 | 171.5 | 70.5 | 0.41 |
| M32_N2048_K2048 | 107.7 | 30.7 | 0.29 |
| M128_N2048_K2048 | 108.1 | 102.9 | 0.95 |
| M1_N4096_K4096 | 289.2 | 272.5 | 0.94 |
| M32_N4096_K4096 | 291.6 | 284.2 | 0.97 |
| M128_N4096_K4096 | 306.5 | 362.7 | 1.18 |

**Reading**: baracuda wins f32 GEMM at the high-M ResNet-typical shape
(M128, 4096²) by ~18%, but loses badly at the low-M decode shapes (M1
/ M32 at 2048²) where cuBLAS's tuned `sgemm` kernel reigns. This is
the canonical "Phase 27 multi-M opportunity" surface — baracuda's
CUTLASS RCR plan is tuned for prefill-scale M; low-M needs a
dedicated decode-step kernel.

### gemm (f16) — RTX 4070, 2026-05-26, `-- --quick`

**Phase 29 baseline** (CUTLASS sm_80 only):

| Shape | baracuda (us) | cuBLAS GemmEx (us) | delta |
| --- | --- | --- | --- |
| M1_N2048_K2048 | 55.8 | 13.3 | 0.24 |
| M32_N2048_K2048 | 55.6 | 14.4 | 0.26 |
| M128_N2048_K2048 | 55.8 | 29.5 | 0.53 |
| M1_N4096_K4096 | 107.9 | 34.6 | 0.32 |
| M32_N4096_K4096 | 108.3 | 64.5 | 0.60 |
| M128_N4096_K4096 | 146.5 | 115.3 | 0.79 |

**Reading**: baracuda's f16 GEMM is **~2-4× slower than cuBLAS GemmEx**
across the full sweep. The gap is largest at low-M (decode-step):
M1 at 2048² baracuda is 4.2× slower. At M128 it narrows to 1.3-1.9×.
cuBLAS is using the sm_89 tensor-core path with f32 accumulator;
baracuda's CUTLASS RCR plan emits a generic Ampere/SM80 path — this
falls inside the Phase 27 / Tier A optimization scope.

**Phase 30 after** — `GemmPlan` cuBLAS fast-path (RTX 4070, `--quick`):

| Shape | baracuda (us) | cuBLAS GemmEx (us) | delta | Backend picked |
| --- | --- | --- | --- | --- |
| M1_N2048_K2048 | ~67–86 (noisy) | ~16–19 | ~0.20 | CUTLASS (M=1 stays — see heuristic) |
| **M32_N2048_K2048** | **~18.3** | ~20.0 | **~1.10** | **cuBLAS** (3.0× speedup, parity with direct) |
| M128_N2048_K2048 | ~59.4 | ~38 | 0.64 | CUTLASS (M≥128 stays) |
| M1_N4096_K4096 | ~126 | ~65 | 0.52 | CUTLASS |
| **M32_N4096_K4096** | **~99.4** | ~89.9 | **~0.91** | **cuBLAS** (close to direct) |
| M128_N4096_K4096 | ~206 | ~178 | 0.86 | CUTLASS |

**Reading**: the Phase-30 cuBLAS routing **closes the gap to direct
cuBLAS at the 2 ≤ M < 128 decode-batch window** (M=32 hits parity with
cuBLAS direct on both K=N=2048 and K=N=4096). M=1 *stays on CUTLASS*
by the heuristic — see [`GemmPlan::backend`] rustdoc and
`should_use_cublas_for_fp` in `baracuda-cutlass/src/plan.rs` for why
(short version: cuBLAS forces a `transa=T` materialization for the
row-major-from-col-major mapping, which is slower than the
CUTLASS-sm_80 GEMV-tile at pure M=1).

`--quick` has 20-30% measurement variance at the M=1 shape, hence the
"~" prefixes; the M=32 numbers are stable to <5%.

**Force-cuBLAS override**: callers wanting cuBLAS at M=1 or M≥128 (e.g.
to validate output against a known cuBLAS reference, or because they
have profiling data the heuristic doesn't) can pass
`PlanPreference { prefer_backend: Some(BackendKind::Cublas), .. }` —
the plan will route through cuBLAS at any shape (subject to dtype
support: F32Strict / FP8 / integer have no cuBLAS path).

### gemm (bf16) — RTX 4070, 2026-05-26, `-- --quick`

**Phase 29 baseline** (CUTLASS sm_80 only):

| Shape | baracuda (us) | cuBLAS GemmEx (us) | delta |
| --- | --- | --- | --- |
| M1_N2048_K2048 | 55.8 | 13.1 | 0.24 |
| M32_N2048_K2048 | 55.8 | 19.5 | 0.35 |
| M128_N2048_K2048 | 56.1 | 29.5 | 0.53 |
| M1_N4096_K4096 | 108.1 | 33.2 | 0.31 |
| M32_N4096_K4096 | 108.5 | 64.2 | 0.59 |
| M128_N4096_K4096 | 147.2 | 115.1 | 0.78 |

**Reading**: identical shape to the f16 picture above — bf16 hits the
same tensor-core path as f16 on Ada / Hopper, so the gap is the same.
The Phase-30 cuBLAS fast-path applies identically to bf16; see the f16
"after" table above for the closed-gap numbers.

### mmvq

| Block format | dtype | Shape | baracuda (us) | Notes |
| --- | --- | --- | --- | --- |
| _Populate from `target/criterion/phase29/mmvq.csv`. No library reference._ | | | | |

This is the baseline that Phase 27's deferred multi-M MMVQ port +
Tier A k-quant micro-opts will measure improvements against.

### Softmax, LayerNorm, RMSNorm, Reductions, Elementwise, Conv2d, Pool, SDPA-GQA

| Bench | Op | Shape | dtype | baracuda (us) | reference (us) | delta |
| --- | --- | --- | --- | --- | --- | --- |
| _Populate from each bench's CSV._ | | | | | | |

## Methodology notes

- **CUDA event timing** — every bench wraps the launch loop in
  `cudaEventRecord` + `cudaEventElapsedTime` (via `time_with_events`).
- **Median over 11 samples** — each `measure_median_ns` call collects
  11 sample pairs of 20-100 inner launches, takes the per-sample
  average, then medians across samples. Criterion's own statistical
  pass runs on top for the HTML report.
- **Warmup** — 10 launches + `stream.synchronize()` before the first
  timed sample.
- **Buffer fill** — `1.0` in dtype-appropriate units. Zero-fill is
  avoided because some kernels short-circuit on zero inputs.
- **No cross-process PyTorch comparison** — PyTorch integration would
  require either a subprocess shim (high per-call latency, washes out
  microsecond-scale ops) or a CFFI bridge (substantial new code path).
  baracuda's perf relative to cuBLAS / cuDNN is the more critical
  signal for the 1.0 freeze; PyTorch comparison is left for a
  follow-up if a tractable integration appears.

## Out of scope

- Multi-M MMVQ port (Phase 27's deferred opportunity).
- Hopper / Blackwell specialization.
- Closing perf gaps. Phase 29's job is **measurement**, not
  optimization. The numbers from these benches are the inputs into
  the Phase 27 / k-quant Tier A perf workstreams that follow.

## Phase 44 — CUDA-L2 vendor validation (SKIP)

[`deepreinforce-ai/CUDA-L2`](https://github.com/deepreinforce-ai/CUDA-L2)
ships RL+LLM-tuned HGEMM kernels (MIT, commit `dbe017722194bb33bafadfbcbb4a65ab6df95dc3`,
upstream pinned at `external/cuda-l2/`). The Phase 44 question: should
we vendor them as a third `GemmPlan` backend alongside `Bespoke`
(CUTLASS sm_80) and `Cublas` (Phase 30 gemmEx fast-path)?

**Decision: SKIP.** Reproducible probes under `external/cuda-l2-probes/`
and the `gemm_vs_cuda_l2` bench file establish the numbers.

### Measured on RTX 4070 (sm_89), CUDA 13.0, 2026-05-28

| Shape (M×K=N, f16/fp32-acc) | baracuda Bespoke (us) | cuBLAS gemmEx (us) | CUDA-L2 (us) | CUDA-L2 vs cuBLAS |
| --- | ---: | ---: | ---: | --- |
| M=1, N=K=4096 | ~107.9 | ~34.6 (or ~65 via GemmPlan-cuBLAS) | **N/A — no kernel** | — |
| M=8, N=K=4096 | — | — | **N/A — no kernel** | — |
| M=32, N=K=4096 | ~108.3 | ~64.5 (~89.9 via GemmPlan-cuBLAS) | **N/A — no kernel** | — |
| M=128, N=K=4096 | ~146.5 | ~177.4 | 175.2 | **+1.2% (parity)** |
| M=2048, N=K=4096 | — | 2621.5 | 2452.7 | **+6.4%** |

### Why SKIP

1. **CUDA-L2 ships zero kernels for M ∈ {1, 8, 32}**. That's the decode
   regime where Phase 30's cuBLAS fast-path won 3× over CUTLASS. CUDA-L2's
   minimum-M is 64 in the 3090 set; their upstream FAQ recommends
   "pad to the nearest larger shape and zero-fill" — which at M=1 means
   64× the work. Not viable.

2. **At the shapes CUDA-L2 covers, wins are marginal on sm_89**. Their
   advertised +24.2% over cuBLAS is on RTX 3090 (sm_86). On the Ada
   RTX 4070 (sm_89) the same kernels deliver +1.2% at M=128 and +6.4%
   at M=2048 — the sm_89 tensor-core path in cuBLAS already saturates
   much of their tuning headroom. Their FAQ explicitly states "kernels
   trained on A100 should only be used on A100 if you are targeting
   speedup."

3. **Integration cost is high**. Per-shape, per-dtype `build.rs`
   compilation (each of CUDA-L2's 736 kernels is a distinct
   instantiation of a different BM/BN/BK/Stage tuning), per-shape FFI
   symbol declarations, and a new dispatch heuristic in `GemmPlan` to
   pick CUDA-L2 over cuBLAS/CUTLASS at the right shapes. The Phase 30
   cuBLAS integration was a single handle wrap + one heuristic; CUDA-L2
   would be ≥10× that work.

4. **The win regime is the prefill bulk-matmul tail, not the latency-
   sensitive decode**. Production LLM serving (the actual baracuda
   target) spends its tokens on decode, where we already win by routing
   to cuBLAS. The +6% at M=2048 is a real measurement, but +6% on the
   non-bottleneck regime doesn't pay for the integration cost.

### What we kept

- `external/cuda-l2/` — full upstream checkout (preserved for reference;
  per-shape kernels can be inspected if a future opportunity at larger
  M emerges).
- `external/cuda-l2-probes/` — stripped wrapper .cu files (M=128 and
  M=2048), standalone probe .cu files, and a README documenting the
  build and measurement methodology.
- `benches/gemm_vs_cuda_l2.rs` + `build.rs` — the bench harness can
  be re-armed via `--features cuda_l2,sm89` if a future CUDA-L2 release
  ships kernels for the decode regime (M < 64), or if Hopper/Blackwell
  kernels land and we want to re-evaluate.

### How to reproduce

```powershell
# Documentation-only mode (default). No nvcc needed for the bench;
# emits the reference probe numbers + cuBLAS / baracuda live timings.
cargo bench -p baracuda-kernels-bench --bench gemm_vs_cuda_l2 -- --quick

# Live measurement mode. build.rs compiles wrapper_m{128,2048}.cu
# (CUTLASS CuTe templates, ~30s nvcc per shape). Requires the
# baracuda-cutlass-sys CUTLASS cache (auto-populated by any prior
# bench / build).
cargo bench -p baracuda-kernels-bench --bench gemm_vs_cuda_l2 \
  --features cuda_l2,sm89 -- --quick
```


## Cross-implementation rollup (auto-generated)

Refresh workflow:

```bash
cargo bench -p baracuda-kernels-bench --features sm89,cudnn -- --quick
python tools/build_benchmarks_table.py
```

Known data-quality issues in the current run (2026-06-04):

- **`flash_sdpa_gqa` Hkv=32 cells report ~270ms baracuda** — orders
  of magnitude slower than PyTorch's 2.5ms at the same shape, and
  ~100× theoretical peak. The measurement is reproducible across
  runs and uses CUDA-event timing, so it reflects a real baracuda
  Flash SDPA perf issue at H=32, Q=K=2048, D=128 (not a bench
  artifact — smoke tests at smaller shapes pass cleanly). Worth
  profiling. Tracked as a ROADMAP follow-up.
- **`flash_sdpa_gqa` Hkv=1 cells emit `reference: "skipped"`** —
  `FlashSdpaPlan::can_implement` rejects the `stride[1] = 0` MQA
  broadcast pattern ("trailblazer requires contiguous tensors"),
  even though the strided sibling `FlashSdpaSm89Plan` supports it.
  Tracked as a ROADMAP follow-up ("`FlashSdpaPlan` GQA-broadcast
  routing gap"). The bench now catches the rejection gracefully so
  the run completes; the underlying baracuda gap remains.
- **`flash_sdpa_gqa` Hkv∈{8,4} cells** are intentionally skipped
  by the bench logic (no stride-0 broadcast pattern for these GQA
  ratios; would need a contig KV-repeat pre-pass that doesn't
  model real GQA inference).

<!-- BEGIN auto-generated phase29 rollup -->
This section is generated by `tools/build_benchmarks_table.py`
from the per-bench CSV outputs under
`target/criterion/phase29/`. Do not edit by hand — re-run the
script after a fresh `cargo bench` to refresh.

Hardware: RTX 4070 Laptop GPU (sm_89), CUDA 13.0, cuDNN 9.x.
PyTorch baseline: 2.11.0+cu130 (frozen JSON in `bench-baselines/`).

Speedup column convention: `library_ns / baracuda_ns`.
`> 1` (bolded) means baracuda is faster than that library at this cell.
`≈` means within ±5%.

### `gemm`

| dtype | shape | baracuda | cuBLAS | cuBLAS/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `M1_N2048_K2048` | 112.4μs | 80.6μs | 0.72× | 29.4μs | 0.26× |
| f32 | `M1_N4096_K4096` | 438.8μs | 279.4μs | 0.64× | 287.6μs | 0.66× |
| f32 | `M32_N2048_K2048` | 108.7μs | 33.8μs | 0.31× | 49.9μs | 0.46× |
| f32 | `M32_N4096_K4096` | 444.5μs | 470.9μs | **1.06×** | 473.3μs | **1.06×** |
| f32 | `M128_N2048_K2048` | 108.8μs | 170.8μs | **1.57×** | 192.0μs | **1.77×** |
| f32 | `M128_N4096_K4096` | 473.1μs | 983.3μs | **2.08×** | 897.1μs | **1.90×** |
| f16 | `M1_N2048_K2048` | 82.0μs | 20.7μs | 0.25× | 31.5μs | 0.38× |
| f16 | `M1_N4096_K4096` | 110.4μs | 35.4μs | 0.32× | 108.7μs | ≈ |
| f16 | `M32_N2048_K2048` | 21.3μs | 18.6μs | 0.88× | 33.9μs | **1.59×** |
| f16 | `M32_N4096_K4096` | 57.0μs | 65.5μs | **1.15×** | 112.7μs | **1.98×** |
| f16 | `M128_N2048_K2048` | 56.7μs | 31.6μs | 0.56× | 54.2μs | ≈ |
| f16 | `M128_N4096_K4096` | 115.0μs | 129.3μs | **1.12×** | 195.1μs | **1.70×** |
| bf16 | `M1_N2048_K2048` | 56.3μs | 20.2μs | 0.36× | 30.5μs | 0.54× |
| bf16 | `M1_N4096_K4096` | 120.5μs | 33.7μs | 0.28× | 64.6μs | 0.54× |
| bf16 | `M32_N2048_K2048` | 20.1μs | 21.3μs | **1.06×** | 31.4μs | **1.56×** |
| bf16 | `M32_N4096_K4096` | 56.7μs | 66.9μs | **1.18×** | 89.2μs | **1.57×** |
| bf16 | `M128_N2048_K2048` | 56.8μs | 31.3μs | 0.55× | 47.7μs | 0.84× |
| bf16 | `M128_N4096_K4096` | 164.7μs | 132.9μs | 0.81× | 181.1μs | **1.10×** |

### `softmax`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_C1024` | 16.0μs | 15.7μs | ≈ | 19.0μs | **1.18×** |
| f32 | `R512_C4096` | 32.2μs | 27.7μs | 0.86× | 32.6μs | ≈ |
| f32 | `R2048_C1024` | 29.2μs | 34.7μs | **1.19×** | 21.7μs | 0.74× |
| f32 | `R2048_C4096` | 362.5μs | 313.9μs | 0.87× | 341.3μs | 0.94× |
| f32 | `R4096_C1024` | 69.0μs | 96.9μs | **1.40×** | 44.0μs | 0.64× |
| f32 | `R4096_C4096` | 778.3μs | 653.2μs | 0.84× | 781.4μs | ≈ |
| f16 | `R512_C1024` | 16.5μs | 15.3μs | 0.93× | 22.4μs | **1.36×** |
| f16 | `R512_C4096` | 21.5μs | 18.4μs | 0.86× | 34.3μs | **1.60×** |
| f16 | `R2048_C1024` | 33.6μs | 32.8μs | ≈ | 19.2μs | 0.57× |
| f16 | `R2048_C4096` | 75.0μs | 69.3μs | 0.92× | 207.0μs | **2.76×** |
| f16 | `R4096_C1024` | 47.4μs | 89.9μs | **1.90×** | 62.5μs | **1.32×** |
| f16 | `R4096_C4096` | 409.3μs | 329.2μs | 0.80× | 481.9μs | **1.18×** |

### `layernorm`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 14.2μs | 25.2μs | **1.78×** |
| f32 | `R512_H4096` | 26.2μs | 27.7μs | **1.06×** |
| f32 | `R2048_H1024` | 29.2μs | 30.8μs | **1.06×** |
| f32 | `R2048_H4096` | 411.3μs | 333.0μs | 0.81× |
| f32 | `R4096_H1024` | 55.9μs | 143.2μs | **2.56×** |
| f32 | `R4096_H4096` | 756.6μs | 673.2μs | 0.89× |
| f16 | `R512_H1024` | 17.0μs | 25.3μs | **1.49×** |
| f16 | `R512_H4096` | 25.9μs | 23.9μs | 0.92× |
| f16 | `R2048_H1024` | 28.4μs | 28.4μs | ≈ |
| f16 | `R2048_H4096` | 91.8μs | 146.9μs | **1.60×** |
| f16 | `R4096_H1024` | 54.9μs | 87.6μs | **1.59×** |
| f16 | `R4096_H4096` | 617.3μs | 389.3μs | 0.63× |

### `rmsnorm`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 16.0μs | 153.9μs | **9.61×** |
| f32 | `R512_H4096` | 31.0μs | 173.1μs | **5.59×** |
| f32 | `R2048_H1024` | 23.4μs | 120.9μs | **5.16×** |
| f32 | `R2048_H4096` | 317.2μs | 1.09ms | **3.45×** |
| f32 | `R4096_H1024` | 56.7μs | 355.2μs | **6.27×** |
| f32 | `R4096_H4096` | 685.4μs | 2.32ms | **3.39×** |
| f16 | `R512_H1024` | 21.1μs | 224.0μs | **10.61×** |
| f16 | `R512_H4096` | 28.3μs | 224.1μs | **7.91×** |
| f16 | `R2048_H1024` | 22.5μs | 176.4μs | **7.83×** |
| f16 | `R2048_H4096` | 70.9μs | 1.32ms | **18.69×** |
| f16 | `R4096_H1024` | 43.5μs | 425.8μs | **9.79×** |
| f16 | `R4096_H4096` | 379.1μs | 2.95ms | **7.78×** |
| bf16 | `R512_H1024` | 20.4μs | 242.9μs | **11.90×** |
| bf16 | `R512_H4096` | 28.9μs | 181.3μs | **6.27×** |
| bf16 | `R2048_H1024` | 30.4μs | 213.3μs | **7.03×** |
| bf16 | `R2048_H4096` | 69.3μs | 1.31ms | **18.95×** |
| bf16 | `R4096_H1024` | 37.6μs | 433.6μs | **11.53×** |
| bf16 | `R4096_H4096` | 550.5μs | 2.95ms | **5.36×** |

### `reduce_sum`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 167.9μs | 47.1μs | 0.28× | 21.9μs | 0.13× |
| f32 | `R512_H4096` | 412.1μs | 156.3μs | 0.38× | 23.8μs | 0.06× |
| f32 | `R2048_H1024` | 106.2μs | 51.4μs | 0.48× | 22.0μs | 0.21× |
| f32 | `R2048_H4096` | 430.1μs | 874.5μs | **2.03×** | 33.2μs | 0.08× |
| f32 | `R4096_H1024` | 105.7μs | 250.4μs | **2.37×** | 32.7μs | 0.31× |
| f32 | `R4096_H4096` | 749.1μs | 1.79ms | **2.39×** | 318.2μs | 0.42× |

### `reduce_max`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 105.7μs | 55.0μs | 0.52× | 26.1μs | 0.25× |
| f32 | `R512_H4096` | 412.0μs | 170.9μs | 0.41× | 20.8μs | 0.05× |
| f32 | `R2048_H1024` | 105.4μs | 53.5μs | 0.51× | 21.1μs | 0.20× |
| f32 | `R2048_H4096` | 430.1μs | 896.0μs | **2.08×** | 33.5μs | 0.08× |
| f32 | `R4096_H1024` | 106.1μs | 256.0μs | **2.41×** | 27.3μs | 0.26× |
| f32 | `R4096_H4096` | 747.9μs | 1.83ms | **2.45×** | 318.5μs | 0.43× |

### `reduce_mean`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 105.9μs | 44.8μs | 0.42× | 21.8μs | 0.21× |
| f32 | `R512_H4096` | 413.3μs | 156.8μs | 0.38× | 21.1μs | 0.05× |
| f32 | `R2048_H1024` | 106.6μs | 51.3μs | 0.48× | 20.5μs | 0.19× |
| f32 | `R2048_H4096` | 432.4μs | 910.7μs | **2.11×** | 34.9μs | 0.08× |
| f32 | `R4096_H1024` | 107.0μs | 251.4μs | **2.35×** | 20.7μs | 0.19× |
| f32 | `R4096_H4096` | 751.1μs | 1.81ms | **2.41×** | 318.2μs | 0.42× |

### `add`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.0μs | 21.0μs | **1.32×** |
| f32 | `N16777216` | 951.3μs | 993.1μs | ≈ |
| f16 | `N1048576` | 18.0μs | 18.4μs | ≈ |
| f16 | `N16777216` | 467.3μs | 505.5μs | **1.08×** |

### `mul`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 19.4μs | 17.7μs | 0.91× |
| f32 | `N16777216` | 1.01ms | 861.6μs | 0.85× |
| f16 | `N1048576` | 13.0μs | 18.2μs | **1.40×** |
| f16 | `N16777216` | 470.8μs | 505.4μs | **1.07×** |

### `relu`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.2μs | 22.3μs | **1.37×** |
| f32 | `N16777216` | 719.7μs | 594.4μs | 0.83× |
| f16 | `N1048576` | 12.9μs | 18.4μs | **1.42×** |
| f16 | `N16777216` | 326.2μs | 301.6μs | 0.92× |

### `gelu`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 18.6μs | 17.4μs | 0.94× |
| f32 | `N16777216` | 679.1μs | 594.2μs | 0.88× |
| f16 | `N1048576` | 15.8μs | 18.1μs | **1.15×** |
| f16 | `N16777216` | 348.8μs | 334.2μs | ≈ |

### `conv2d`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `N1_Cin64_Cout64_HW56_K3` | 101.3μs | 43.5μs | 0.43× | 109.5μs | **1.08×** |
| f32 | `N1_Cin128_Cout128_HW28_K3` | 104.0μs | 57.7μs | 0.55× | 78.5μs | 0.76× |
| f32 | `N1_Cin256_Cout256_HW14_K3` | 99.1μs | 82.9μs | 0.84× | 79.6μs | 0.80× |
| f16 | `N1_Cin64_Cout64_HW56_K3` | 119.2μs | 321.8μs | **2.70×** | 77.0μs | 0.65× |
| f16 | `N1_Cin128_Cout128_HW28_K3` | 112.3μs | 403.8μs | **3.60×** | 86.2μs | 0.77× |
| f16 | `N1_Cin256_Cout256_HW14_K3` | 109.9μs | 574.8μs | **5.23×** | 99.5μs | 0.91× |

### `maxpool2d`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `N1_C64_H56_W56_K3_S2` | 14.1μs | 14.0μs | ≈ | 21.7μs | **1.54×** |
| f32 | `N1_C128_H28_W28_K3_S2` | 13.6μs | 15.0μs | **1.10×** | 25.3μs | **1.86×** |
| f32 | `N1_C256_H14_W14_K3_S2` | 17.7μs | 15.4μs | 0.87× | 21.6μs | **1.22×** |
| f16 | `N1_C64_H56_W56_K3_S2` | 14.8μs | 13.6μs | 0.92× | 22.5μs | **1.52×** |
| f16 | `N1_C128_H28_W28_K3_S2` | 13.6μs | 13.3μs | ≈ | 23.1μs | **1.70×** |
| f16 | `N1_C256_H14_W14_K3_S2` | 16.7μs | 14.0μs | 0.84× | 24.6μs | **1.47×** |

### `flash_sdpa_gqa`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f16 | `Hq32_Hkv1_Q2048_D128` | nanns | 2.53ms | ≈ |
| f16 | `Hq32_Hkv32_Q2048_D128` | 268.96ms | 2.48ms | 0.01× |
| bf16 | `Hq32_Hkv1_Q2048_D128` | nanns | 2.49ms | ≈ |
| bf16 | `Hq32_Hkv32_Q2048_D128` | 274.70ms | 2.45ms | 0.01× |

<!-- END auto-generated phase29 rollup -->
