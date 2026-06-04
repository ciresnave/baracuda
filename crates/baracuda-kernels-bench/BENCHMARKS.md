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
