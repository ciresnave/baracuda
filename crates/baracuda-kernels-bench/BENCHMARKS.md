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

**Hardware**: RTX 4070 **Laptop GPU** (sm_89). The generated rollup below carries its
own `Hardware:` line and that is the copy a regeneration keeps current; this one is
hand-maintained and must be made to agree with it.

> ⚠ **DISCHARGED 2026-09-06.** This line read *“RTX 4070 (sm_89), CUDA 13.0, cuDNN 9.x”*
> until today — carried unchanged from the file's creation on 2026-05-28 through every
> regeneration since, including the full-suite regen in #74 that corrected the generated
> preamble's copy on the same night. **RTX 4070 and RTX 4070 Laptop are different parts**
> (different SM count and memory bandwidth), so a reader comparing these numbers against
> published desktop figures was comparing against the wrong GPU. The toolchain versions
> are dropped rather than restated: the rollup is regenerated across runs, and a version
> pinned here is one the generator structurally cannot reach — which is how the old pair
> survived four months. `scripts/check-benchmark-provenance.sh` now fails if the two
> hardware claims name different parts.
**Build**: `cargo bench -p baracuda-kernels-bench --features sm89,cudnn`.

## Bench inventory

| Bench file | Ops | Reference | Shapes |
| --- | --- | --- | --- |
| `gemm_vs_cublas` | GEMM f32 / f16 / bf16 | cuBLAS (`sgemm` / `gemmEx`) | M ∈ {1, 32, 128}, K=N ∈ {2048, 4096} |
| `mmvq` | GGUF MMVQ (Q4_0, Q4_K, Q6_K, Q8_0) × f32 / f16 / bf16 | self (no library equiv) | (4096×4096), (11008×4096), (32000×4096) |
| `softmax_vs_cudnn` | Softmax + LogSoftmax (Phase 73.4) × f32 / f16 | cuDNN `softmax_forward` (`Accurate` / `Log`) | rows ∈ {512, 2048, 4096}, hidden ∈ {1024, 4096} |
| `layernorm_vs_cudnn` | LayerNorm f32 / f16 | self (cuDNN classic LN not wired) | rows × hidden, same as softmax |
| `rmsnorm` | RMSNorm f32 / f16 / bf16 | self (no library equiv) | rows × hidden, same as softmax |
| `conv2d_vs_cudnn` | Conv2d f32 / f16 | raw cuDNN `convolution_forward` (baracuda is cuDNN-backed — measures wrapper overhead) | ResNet-50 picks (3) |
| `pool_vs_cudnn` (Phase 73.7) | MaxPool2d + AvgPool2d (count-include-pad) × f32 / f16 | raw cuDNN `pooling_forward` | ResNet-50 picks (3) |
| `reductions_vs_cudnn` (Phase 73.6) | Sum / Max / Min / Mean / Prod / Var / Std / Norm2 / LogSumExp × f32 | cuDNN `reduce_tensor` where available (Sum/Max/Min/Mean/Prod/Norm2; Var/Std/LogSumExp have no cuDNN equivalent) | rows × hidden, same as softmax |
| `elementwise` (Phase 73.5) | 33 ops × f32 / f16 — activations (ReLU/GELU/Silu/Tanh/Sigmoid/Mish/Hardswish/Hardsigmoid/Hardtanh/LeakyReLU/Elu/Selu/ReLU6/Softplus/Softsign/GELU-Tanh), math unaries (Abs/Neg/Sign/Reciprocal/Sqrt/Rsqrt/Square/Exp/Log/Sin/Cos/Erf), binaries (Add/Sub/Mul/Div/Maximum/Minimum/Pow) | self | numel ∈ {1M, 16M} |
| `sdpa_gqa` | Flash SDPA + GQA broadcast (f16 / bf16) | self | H_q=32, H_kv ∈ {32, 1}, Q=K=2048, D=128 |
| `concat` (Phase 73.8) | 2-input torch.cat × f32 / f16 | self (no library equiv) | KV-cache decode (BH32_Ka2047_Kb1_D128) + mid-seq joins |
| `embedding` (Phase 73.8) | F.embedding × f32 / f16 | self (no library equiv) | Llama-2 7B decode (V32000_D4096_N1) + prefill (N2048) + smaller dense |
| `masked_fill` (Phase 73.8) | tensor.masked_fill(mask, -inf) × f32 | self (no library equiv) | rows × hidden, same as softmax |
| `batch_norm` (Phase 73.8) | BatchNorm training-mode FW × f32 / f16 | self (PyTorch via JSON; cuDNN BN has heavier API) | ResNet-50 picks (3) |
| `topk` (Phase 73.8) | torch.topk × f32 | self (no library equiv) | MoE-style (B32_L128_K4) + intermediate (B8_L512_K16) + cap (B1_L1024_K64) |
| `flash_decoding` (Phase 73 fu) | `FlashDecodingPlan` (split-K decode, seq_q=1) × f16 / bf16 | `FlashSdpaPlan` (the legacy fallback the new kernel replaces) | B=1, H=32, D=128, K ∈ {1024, 2048, 4096, 8192} |
| `flash_decoding` GQA cells (same bench) | GQA + MQA decode shapes — Llama-3-8B (Hq=32, Hkv=8), Llama-3-70B (Hq=64, Hkv=8), qwen2-14B (Hq=32, Hkv=4), MQA (Hq=32, Hkv=2) | self (TC dispatch disabled — see kernel doc-comment for why) | K ∈ {1024, 2048, 4096, 8192} × f16 / bf16 |

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

- ~~**`flash_sdpa_gqa` Hkv=32 cells report ~270ms baracuda**~~ — **✓ FIXED
  (2026-06-04, commit `833f862`)** by making `fa2` a default cargo
  feature. The `should_use_fa2` heuristic routes this shape to FA2 →
  **1.66ms (50% faster than PyTorch's 2.48ms)**. Re-run with `--features fa2`
  (now the default) to see the post-fix numbers.
- ~~**`flash_sdpa_gqa` Hkv=1 cells emit `reference: "skipped"`**~~ —
  **✓ FIXED (2026-06-05)**. `FlashSdpaPlan::can_implement` now accepts
  the stride-0 full-MQA-broadcast K/V convention; `run` reinterprets
  the broadcast buffer as physical `[B, 1, K, D]` and routes to FA2
  (MQA, head_dim 128) for large shapes, or to the sm89 strided sibling
  (head_dim ≤ 64) for small shapes. Measured Hkv=1 at the bench shape
  (Hq=32, Q=K=2048, D=128): **f16 1.68ms (1.52× faster than PyTorch's
  2.56ms), bf16 1.68ms (1.49× faster)** — re-run with
  `--features sm89,fa2`.
- **`flash_sdpa_gqa` Hkv∈{8,4} cells** are intentionally skipped
  by the bench logic (no stride-0 broadcast pattern for these
  intermediate-GQA ratios — partial GQA can't be expressed as a
  single stride). For these ratios, callers should use either:
  (a) `FlashSdpaPlan` with FA2 backend at prefill (passes physical
  `[B, H_kv, K, D]` natively), or (b) the new `FlashDecodingPlan`
  at decode (seq_q=1). Both shipped Phase 73 follow-up.

<!-- BEGIN auto-generated phase29 rollup -->
This section is generated by `tools/build_benchmarks_table.py`
from the per-bench CSV outputs under
`target/criterion/phase29/`. Do not edit by hand — re-run the
script after a fresh `cargo bench` to refresh.

Hardware: RTX 4070 Laptop GPU (sm_89).
PyTorch baseline: 2.14.0+cu126 (frozen JSON in `bench-baselines/`).

Speedup column convention: `library_ns / baracuda_ns`.
`> 1` (bolded) means baracuda is faster than that library at this cell.
`≈` means within ±5%.

⚠️ **READ THE ±5% BAND WITH THIS.** These figures come from a machine
running ~17 other agent processes at 77–100% CPU; an unloaded baseline
is not obtainable on it, and the GPU sits at 1605 of 3105 MHz SM clock
even at 0% utilization and 57 °C (`SW Thermal Slowdown: Active`).

**Measured, not estimated.** Of the cells that have been run more than
once and carry both sides, **12 of 12 vary by more than ±5% between
runs of identical code** — median ratio spread **2.57×**, worst
**5.34×**. ⚠️ **Three of the twelve FLIP DIRECTION**: e.g.
`mmvq_multim / M1_N11008_C4096 / f32` ranged from *baracuda 3.18×
faster* to *baracuda 1.68× slower* across ten runs, so its headline
verdict is not reproducible.

⚠️ **And only 12 of ~457 cells have been run more than once.** The
rest carry a single observation, so their reproducibility is
**unmeasured** — not good, not bad, unmeasured. A `≈` or a bolded
`> 1` on a single-run cell states a comparison, not a repeatable one.

### `gemm`

| dtype | shape | baracuda | cuBLAS | cuBLAS/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `M1_N2048_K2048` | 172.9μs | 70.9μs | 0.41× | 23.9μs | 0.14× |
| f32 | `M1_N4096_K4096` | 289.7μs | 274.0μs | 0.95× | 295.9μs | ≈ |
| f32 | `M32_N2048_K2048` | 108.7μs | 31.9μs | 0.29× | 40.5μs | 0.37× |
| f32 | `M32_N4096_K4096` | 292.2μs | 285.9μs | ≈ | 527.4μs | **1.80×** |
| f32 | `M128_N2048_K2048` | 109.1μs | 104.0μs | ≈ | 208.7μs | **1.91×** |
| f32 | `M128_N4096_K4096` | 307.3μs | 362.6μs | **1.18×** | 918.4μs | **2.99×** |
| f16 | `M1_N2048_K2048` | 56.2μs | 19.0μs | 0.34× | 24.1μs | 0.43× |
| f16 | `M1_N4096_K4096` | 109.6μs | 35.1μs | 0.32× | 103.6μs | 0.94× |
| f16 | `M32_N2048_K2048` | 22.7μs | 20.0μs | 0.88× | 25.2μs | **1.11×** |
| f16 | `M32_N4096_K4096` | 57.3μs | 65.5μs | **1.14×** | 108.6μs | **1.89×** |
| f16 | `M128_N2048_K2048` | 56.7μs | 30.8μs | 0.54× | 59.1μs | ≈ |
| f16 | `M128_N4096_K4096` | 175.9μs | 118.5μs | 0.67× | 193.5μs | **1.10×** |
| f16 | `M2048_N4096_K4096` | 1.57ms | 1.56ms | ≈ |  |  |
| bf16 | `M1_N2048_K2048` | 56.5μs | 18.9μs | 0.33× | 23.2μs | 0.41× |
| bf16 | `M1_N4096_K4096` | 109.3μs | 34.9μs | 0.32× | 69.5μs | 0.64× |
| bf16 | `M32_N2048_K2048` | 22.4μs | 19.8μs | 0.88× | 31.3μs | **1.40×** |
| bf16 | `M32_N4096_K4096` | 57.5μs | 65.4μs | **1.14×** | 90.2μs | **1.57×** |
| bf16 | `M128_N2048_K2048` | 57.0μs | 30.1μs | 0.53× | 43.3μs | 0.76× |
| bf16 | `M128_N4096_K4096` | 114.4μs | 115.6μs | ≈ | 201.2μs | **1.76×** |

### `softmax`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_C1024` | 19.0μs | 18.2μs | ≈ | 21.0μs | **1.10×** |
| f32 | `R512_C4096` | 32.4μs | 29.9μs | 0.92× | 32.5μs | ≈ |
| f32 | `R2048_C1024` | 27.4μs | 32.5μs | **1.19×** | 21.0μs | 0.77× |
| f32 | `R2048_C4096` | 292.4μs | 285.4μs | ≈ | 396.7μs | **1.36×** |
| f32 | `R4096_C1024` | 45.0μs | 62.4μs | **1.39×** | 51.4μs | **1.14×** |
| f32 | `R4096_C4096` | 586.3μs | 572.9μs | ≈ | 790.6μs | **1.35×** |
| f16 | `R512_C1024` | 32.8μs | 16.8μs | 0.51× | 20.6μs | 0.63× |
| f16 | `R512_C4096` | 22.5μs | 20.7μs | 0.92× | 32.5μs | **1.45×** |
| f16 | `R2048_C1024` | 27.6μs | 44.9μs | **1.62×** | 19.1μs | 0.69× |
| f16 | `R2048_C4096` | 70.4μs | 59.6μs | 0.85× | 199.2μs | **2.83×** |
| f16 | `R4096_C1024` | 44.9μs | 60.9μs | **1.36×** | 69.6μs | **1.55×** |
| f16 | `R4096_C4096` | 305.5μs | 285.1μs | 0.93× | 486.4μs | **1.59×** |

### `layernorm`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 15.7μs | 27.6μs | **1.76×** |
| f32 | `R512_H4096` | 30.1μs | 30.7μs | ≈ |
| f32 | `R2048_H1024` | 25.1μs | 33.0μs | **1.31×** |
| f32 | `R2048_H4096` | 294.9μs | 330.7μs | **1.12×** |
| f32 | `R4096_H1024` | 47.8μs | 151.9μs | **3.18×** |
| f32 | `R4096_H4096` | 595.4μs | 770.6μs | **1.29×** |
| f16 | `R512_H1024` | 13.9μs | 28.5μs | **2.05×** |
| f16 | `R512_H4096` | 23.6μs | 27.8μs | **1.18×** |
| f16 | `R2048_H1024` | 25.6μs | 28.1μs | **1.09×** |
| f16 | `R2048_H4096` | 78.9μs | 156.7μs | **1.99×** |
| f16 | `R4096_H1024` | 47.8μs | 99.5μs | **2.08×** |
| f16 | `R4096_H4096` | 308.3μs | 392.7μs | **1.27×** |

### `rmsnorm`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 16.0μs | 138.8μs | **8.70×** |
| f32 | `R512_H4096` | 28.5μs | 132.3μs | **4.64×** |
| f32 | `R2048_H1024` | 21.2μs | 132.7μs | **6.27×** |
| f32 | `R2048_H4096` | 294.6μs | 1.09ms | **3.70×** |
| f32 | `R4096_H1024` | 39.2μs | 363.8μs | **9.28×** |
| f32 | `R4096_H4096` | 593.1μs | 2.71ms | **4.57×** |
| f16 | `R512_H1024` | 19.9μs | 187.6μs | **9.42×** |
| f16 | `R512_H4096` | 23.4μs | 177.2μs | **7.58×** |
| f16 | `R2048_H1024` | 26.3μs | 175.4μs | **6.68×** |
| f16 | `R2048_H4096` | 69.8μs | 1.54ms | **22.07×** |
| f16 | `R4096_H1024` | 36.0μs | 434.8μs | **12.07×** |
| f16 | `R4096_H4096` | 296.0μs | 3.45ms | **11.64×** |
| bf16 | `R512_H1024` | 16.9μs | 185.8μs | **10.99×** |
| bf16 | `R512_H4096` | 21.5μs | 181.3μs | **8.43×** |
| bf16 | `R2048_H1024` | 30.4μs | 179.7μs | **5.91×** |
| bf16 | `R2048_H4096` | 69.7μs | 1.52ms | **21.76×** |
| bf16 | `R4096_H1024` | 36.0μs | 428.9μs | **11.91×** |
| bf16 | `R4096_H4096` | 296.6μs | 3.45ms | **11.63×** |

### `reduce_sum`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 33.6μs | 42.0μs | **1.25×** | 22.7μs | 0.68× |
| f32 | `R512_H4096` | 35.3μs | 152.6μs | **4.32×** | 23.0μs | 0.65× |
| f32 | `R2048_H1024` | 81.9μs | 50.9μs | 0.62× | 24.0μs | 0.29× |
| f32 | `R2048_H4096` | 150.9μs | 748.4μs | **4.96×** | 31.3μs | 0.21× |
| f32 | `R4096_H1024` | 106.4μs | 177.0μs | **1.66×** | 22.3μs | 0.21× |
| f32 | `R4096_H4096` | 297.2μs | 1.59ms | **5.35×** | 365.1μs | **1.23×** |

### `reduce_max`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 24.2μs | 45.6μs | **1.88×** | 22.0μs | 0.91× |
| f32 | `R512_H4096` | 26.3μs | 167.5μs | **6.36×** | 22.7μs | 0.86× |
| f32 | `R2048_H1024` | 82.0μs | 53.8μs | 0.66× | 23.7μs | 0.29× |
| f32 | `R2048_H4096` | 150.9μs | 753.9μs | **5.00×** | 32.6μs | 0.22× |
| f32 | `R4096_H1024` | 106.1μs | 181.4μs | **1.71×** | 21.7μs | 0.20× |
| f32 | `R4096_H4096` | 297.0μs | 1.59ms | **5.36×** | 366.5μs | **1.23×** |

### `reduce_mean`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 23.5μs | 41.8μs | **1.78×** | 22.0μs | 0.94× |
| f32 | `R512_H4096` | 26.5μs | 152.7μs | **5.76×** | 25.4μs | ≈ |
| f32 | `R2048_H1024` | 82.1μs | 51.0μs | 0.62× | 22.1μs | 0.27× |
| f32 | `R2048_H4096` | 151.3μs | 748.3μs | **4.95×** | 34.8μs | 0.23× |
| f32 | `R4096_H1024` | 106.0μs | 176.9μs | **1.67×** | 22.5μs | 0.21× |
| f32 | `R4096_H4096` | 297.0μs | 1.59ms | **5.36×** | 365.2μs | **1.23×** |

### `add`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 24.5μs | 19.9μs | 0.82× |
| f32 | `N16777216` | 883.2μs | 995.1μs | **1.13×** |
| f16 | `N1048576` | 15.7μs | 21.1μs | **1.34×** |
| f16 | `N16777216` | 419.8μs | 499.2μs | **1.19×** |

### `mul`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.3μs | 20.8μs | **1.28×** |
| f32 | `N16777216` | 883.1μs | 996.9μs | **1.13×** |
| f16 | `N1048576` | 14.0μs | 20.8μs | **1.48×** |
| f16 | `N16777216` | 419.7μs | 498.6μs | **1.19×** |

### `relu`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 15.6μs | 22.2μs | **1.42×** |
| f32 | `N16777216` | 591.1μs | 680.2μs | **1.15×** |
| f16 | `N1048576` | 13.5μs | 20.0μs | **1.48×** |
| f16 | `N16777216` | 281.3μs | 340.0μs | **1.21×** |

### `gelu`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 17.5μs | 19.5μs | **1.11×** |
| f32 | `N16777216` | 591.4μs | 679.8μs | **1.15×** |
| f16 | `N1048576` | 15.5μs | 18.6μs | **1.20×** |
| f16 | `N16777216` | 282.6μs | 339.6μs | **1.20×** |

### `conv2d`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `N1_Cin64_Cout64_HW56_K3` | 157.2μs | 51.6μs | 0.33× | 44.0μs | 0.28× |
| f32 | `N1_Cin128_Cout128_HW28_K3` | 125.7μs | 96.1μs | 0.76× | 57.6μs | 0.46× |
| f32 | `N1_Cin256_Cout256_HW14_K3` | 118.0μs | 73.2μs | 0.62× | 64.4μs | 0.55× |
| f16 | `N1_Cin64_Cout64_HW56_K3` | 131.0μs | 56.1μs | 0.43× | 75.0μs | 0.57× |
| f16 | `N1_Cin128_Cout128_HW28_K3` | 120.9μs | 99.2μs | 0.82× | 78.0μs | 0.65× |
| f16 | `N1_Cin256_Cout256_HW14_K3` | 201.9μs | 192.9μs | ≈ | 75.4μs | 0.37× |

### `maxpool2d`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `N1_C64_H56_W56_K3_S2` | 82.8μs | 72.0μs | 0.87× | 21.9μs | 0.26× |
| f32 | `N1_C128_H28_W28_K3_S2` | 73.2μs | 72.5μs | ≈ | 22.3μs | 0.30× |
| f32 | `N1_C256_H14_W14_K3_S2` | 80.9μs | 91.6μs | **1.13×** | 22.4μs | 0.28× |
| f16 | `N1_C64_H56_W56_K3_S2` | 72.0μs | 67.5μs | 0.94× | 22.1μs | 0.31× |
| f16 | `N1_C128_H28_W28_K3_S2` | 96.9μs | 73.5μs | 0.76× | 22.1μs | 0.23× |
| f16 | `N1_C256_H14_W14_K3_S2` | 80.7μs | 85.3μs | **1.06×** | 21.9μs | 0.27× |

### `flash_sdpa_gqa`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f16 | `Hq32_Hkv1_Q2048_D128` | 1.68ms | 2.51ms | **1.49×** |
| f16 | `Hq32_Hkv32_Q2048_D128` | 1.69ms | 2.45ms | **1.45×** |
| bf16 | `Hq32_Hkv1_Q2048_D128` | 1.68ms | 2.46ms | **1.47×** |
| bf16 | `Hq32_Hkv32_Q2048_D128` | 1.68ms | 2.43ms | **1.44×** |

### `mse`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `R512_C1024` | 58.6μs | 194.7μs | **3.32×** |
| f32 | `R512_C4096` | 128.8μs | 204.3μs | **1.59×** |
| f32 | `R2048_C1024` | 128.5μs | 276.5μs | **2.15×** |
| f32 | `R2048_C4096` | 1.16ms | 726.7μs | 0.63× |
| f16 | `R512_C1024` | 44.3μs | 138.4μs | **3.12×** |
| f16 | `R512_C4096` | 101.6μs | 235.0μs | **2.31×** |
| f16 | `R2048_C1024` | 101.0μs | 181.8μs | **1.80×** |
| f16 | `R2048_C4096` | 628.8μs | 248.4μs | 0.40× |
| bf16 | `R512_C1024` | 35.3μs | 283.7μs | **8.03×** |
| bf16 | `R512_C4096` | 101.7μs | 218.9μs | **2.15×** |
| bf16 | `R2048_C1024` | 101.7μs | 95.3μs | 0.94× |
| bf16 | `R2048_C4096` | 627.9μs | 246.4μs | 0.39× |

### `l1`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `R512_C1024` | 36.2μs | 187.8μs | **5.18×** |
| f32 | `R512_C4096` | 128.9μs | 219.2μs | **1.70×** |
| f32 | `R2048_C1024` | 129.1μs | 246.6μs | **1.91×** |
| f32 | `R2048_C4096` | 1.16ms | 910.9μs | 0.79× |
| f16 | `R512_C1024` | 30.8μs | 238.3μs | **7.75×** |
| f16 | `R512_C4096` | 101.4μs | 256.8μs | **2.53×** |
| f16 | `R2048_C1024` | 101.4μs | 184.2μs | **1.82×** |
| f16 | `R2048_C4096` | 664.2μs | 342.3μs | 0.52× |
| bf16 | `R512_C1024` | 37.7μs | 240.7μs | **6.38×** |
| bf16 | `R512_C4096` | 102.8μs | 271.7μs | **2.64×** |
| bf16 | `R2048_C1024` | 102.1μs | 209.5μs | **2.05×** |
| bf16 | `R2048_C4096` | 633.2μs | 342.3μs | 0.54× |

### `cross_entropy`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `R512_C1024` | 211.1μs | 112.0μs | 0.53× |
| f32 | `R512_C4096` | 829.2μs | 103.0μs | 0.12× |
| f32 | `R2048_C1024` | 211.1μs | 80.5μs | 0.38× |
| f32 | `R2048_C4096` | 829.8μs | 320.2μs | 0.39× |
| f16 | `R512_C1024` | 213.0μs | 98.0μs | 0.46× |
| f16 | `R512_C4096` | 834.7μs | 102.0μs | 0.12× |
| f16 | `R2048_C1024` | 213.5μs | 197.7μs | 0.93× |
| f16 | `R2048_C4096` | 835.6μs | 193.1μs | 0.23× |
| bf16 | `R512_C1024` | 212.2μs | 80.9μs | 0.38× |
| bf16 | `R512_C4096` | 834.0μs | 62.9μs | 0.08× |
| bf16 | `R2048_C1024` | 213.0μs | 108.2μs | 0.51× |
| bf16 | `R2048_C4096` | 834.8μs | 182.6μs | 0.22× |

### `nll`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `R512_C1024` | 26.7μs | 40.3μs | **1.51×** |
| f32 | `R512_C4096` | 31.7μs | 70.2μs | **2.21×** |
| f32 | `R2048_C1024` | 29.2μs | 41.7μs | **1.43×** |
| f32 | `R2048_C4096` | 33.2μs | 51.4μs | **1.55×** |
| f16 | `R512_C1024` | 34.8μs | 38.8μs | **1.12×** |
| f16 | `R512_C4096` | 29.6μs | 97.8μs | **3.31×** |
| f16 | `R2048_C1024` | 30.6μs | 66.3μs | **2.17×** |
| f16 | `R2048_C4096` | 27.8μs | 41.9μs | **1.50×** |
| bf16 | `R512_C1024` | 26.3μs | 39.8μs | **1.51×** |
| bf16 | `R512_C4096` | 26.4μs | 35.0μs | **1.33×** |
| bf16 | `R2048_C1024` | 26.3μs | 67.4μs | **2.57×** |
| bf16 | `R2048_C4096` | 29.6μs | 42.4μs | **1.43×** |

### `mmvq`

_Self-only: no PyTorch/library equivalent — baracuda timings below are absolute, not a comparison._

| dtype | shape | baracuda |
| --- | --- | --- |
| f32 | `q4_0_N4096_C4096` | 55.3μs |
| f32 | `q4_0_N11008_C4096` | 87.6μs |
| f32 | `q4_0_N32000_C4096` | 308.6μs |
| f32 | `q4_k_N4096_C4096` | 30.0μs |
| f32 | `q4_k_N11008_C4096` | 74.4μs |
| f32 | `q4_k_N32000_C4096` | 302.8μs |
| f32 | `q6_k_N4096_C4096` | 38.1μs |
| f32 | `q6_k_N11008_C4096` | 151.5μs |
| f32 | `q6_k_N32000_C4096` | 437.6μs |
| f32 | `q8_0_N4096_C4096` | 41.5μs |
| f32 | `q8_0_N11008_C4096` | 197.4μs |
| f32 | `q8_0_N32000_C4096` | 565.7μs |
| f16 | `q4_0_N4096_C4096` | 37.6μs |
| f16 | `q4_0_N11008_C4096` | 93.8μs |
| f16 | `q4_0_N32000_C4096` | 312.8μs |
| f16 | `q4_k_N4096_C4096` | 30.0μs |
| f16 | `q4_k_N11008_C4096` | 74.4μs |
| f16 | `q4_k_N32000_C4096` | 300.9μs |
| f16 | `q6_k_N4096_C4096` | 38.1μs |
| f16 | `q6_k_N11008_C4096` | 150.8μs |
| f16 | `q6_k_N32000_C4096` | 436.3μs |
| f16 | `q8_0_N4096_C4096` | 35.5μs |
| f16 | `q8_0_N11008_C4096` | 196.8μs |
| f16 | `q8_0_N32000_C4096` | 564.3μs |
| bf16 | `q4_0_N4096_C4096` | 37.6μs |
| bf16 | `q4_0_N11008_C4096` | 94.7μs |
| bf16 | `q4_0_N32000_C4096` | 313.9μs |
| bf16 | `q4_k_N4096_C4096` | 30.1μs |
| bf16 | `q4_k_N11008_C4096` | 74.6μs |
| bf16 | `q4_k_N32000_C4096` | 300.9μs |
| bf16 | `q6_k_N4096_C4096` | 38.2μs |
| bf16 | `q6_k_N11008_C4096` | 150.9μs |
| bf16 | `q6_k_N32000_C4096` | 436.5μs |
| bf16 | `q8_0_N4096_C4096` | 35.5μs |
| bf16 | `q8_0_N11008_C4096` | 197.0μs |
| bf16 | `q8_0_N32000_C4096` | 564.4μs |

### `abs`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 17.4μs | 19.4μs | **1.12×** |
| f32 | `N16777216` | 591.1μs | 681.8μs | **1.15×** |
| f16 | `N1048576` | 13.6μs | 21.2μs | **1.56×** |
| f16 | `N16777216` | 281.2μs | 342.2μs | **1.22×** |

### `avgpool2d`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `N1_C64_H56_W56_K3_S2` | 74.4μs | 73.2μs | ≈ | 22.7μs | 0.30× |
| f32 | `N1_C128_H28_W28_K3_S2` | 74.6μs | 91.9μs | **1.23×** | 19.5μs | 0.26× |
| f32 | `N1_C256_H14_W14_K3_S2` | 90.3μs | 79.1μs | 0.88× | 21.2μs | 0.23× |
| f16 | `N1_C64_H56_W56_K3_S2` | 56.9μs | 79.2μs | **1.39×** | 18.9μs | 0.33× |
| f16 | `N1_C128_H28_W28_K3_S2` | 80.2μs | 63.8μs | 0.80× | 19.3μs | 0.24× |
| f16 | `N1_C256_H14_W14_K3_S2` | 62.3μs | 58.7μs | 0.94× | 20.1μs | 0.32× |

### `batch_norm`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1_C64_H56_W56` | 29.0μs | 48.7μs | **1.68×** |
| f32 | `N1_C128_H28_W28` | 27.0μs | 47.1μs | **1.74×** |
| f32 | `N1_C256_H14_W14` | 32.5μs | 46.4μs | **1.43×** |
| f16 | `N1_C64_H56_W56` | 28.6μs | 86.2μs | **3.01×** |
| f16 | `N1_C128_H28_W28` | 28.2μs | 74.2μs | **2.63×** |
| f16 | `N1_C256_H14_W14` | 31.3μs | 81.9μs | **2.61×** |

### `concat`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `BH32_Ka512_Kb512_D128` | 42.2μs | 45.8μs | **1.08×** |
| f32 | `BH32_Ka1024_Kb1024_D128` | 297.2μs | 345.6μs | **1.16×** |
| f32 | `BH32_Ka2047_Kb1_D128` | 296.3μs | 337.9μs | **1.14×** |
| f16 | `BH32_Ka512_Kb512_D128` | 28.0μs | 33.1μs | **1.18×** |
| f16 | `BH32_Ka1024_Kb1024_D128` | 42.1μs | 43.2μs | ≈ |
| f16 | `BH32_Ka2047_Kb1_D128` | 43.3μs | 42.2μs | ≈ |

### `cos`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 18.1μs | 18.0μs | ≈ |
| f32 | `N16777216` | 590.5μs | 677.6μs | **1.15×** |
| f16 | `N1048576` | 14.2μs | 20.0μs | **1.40×** |
| f16 | `N16777216` | 282.7μs | 339.3μs | **1.20×** |

### `div`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.6μs | 19.4μs | **1.17×** |
| f32 | `N16777216` | 883.4μs | 986.8μs | **1.12×** |
| f16 | `N1048576` | 13.7μs | 20.2μs | **1.48×** |
| f16 | `N16777216` | 420.2μs | 493.9μs | **1.18×** |

### `elu`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 18.3μs | 17.1μs | 0.94× |
| f32 | `N16777216` | 591.5μs | 681.2μs | **1.15×** |
| f16 | `N1048576` | 13.9μs | 20.0μs | **1.44×** |
| f16 | `N16777216` | 282.2μs | 342.7μs | **1.21×** |

### `embedding`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `V8192_D1024_N512` | 15.1μs | 27.3μs | **1.82×** |
| f32 | `V32000_D4096_N1` | 16.1μs | 21.4μs | **1.33×** |
| f32 | `V32000_D4096_N2048` | 326.7μs | 335.0μs | ≈ |
| f16 | `V8192_D1024_N512` | 16.2μs | 27.9μs | **1.72×** |
| f16 | `V32000_D4096_N1` | 16.4μs | 20.9μs | **1.27×** |
| f16 | `V32000_D4096_N2048` | 127.4μs | 67.7μs | 0.53× |

### `erf`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.7μs | 19.5μs | **1.17×** |
| f32 | `N16777216` | 591.5μs | 678.5μs | **1.15×** |
| f16 | `N1048576` | 13.6μs | 19.2μs | **1.41×** |
| f16 | `N16777216` | 282.8μs | 339.0μs | **1.20×** |

### `exp`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.2μs | 20.0μs | **1.23×** |
| f32 | `N16777216` | 591.3μs | 680.5μs | **1.15×** |
| f16 | `N1048576` | 14.2μs | 20.2μs | **1.43×** |
| f16 | `N16777216` | 282.0μs | 341.2μs | **1.21×** |

### `flash_decoding`

| dtype | shape | baracuda |
| --- | --- | --- |
| f16 | `B1_H32_K1024_D128` | 53.4μs |
| f16 | `B1_H32_K2048_D128` | 55.3μs |
| f16 | `B1_H32_K4096_D128` | 292.3μs |
| f16 | `B1_H32_K8192_D128` | 556.1μs |
| bf16 | `B1_H32_K1024_D128` | 34.1μs |
| bf16 | `B1_H32_K2048_D128` | 53.4μs |
| bf16 | `B1_H32_K4096_D128` | 289.5μs |
| bf16 | `B1_H32_K8192_D128` | 555.6μs |

### `flash_decoding_gqa`

| dtype | shape | baracuda |
| --- | --- | --- |
| f16 | `qwen2-14b_Hq32_Hkv4_K1024_D128` | 38.7μs |
| f16 | `qwen2-14b_Hq32_Hkv4_K2048_D128` | 45.2μs |
| f16 | `qwen2-14b_Hq32_Hkv4_K4096_D128` | 78.2μs |
| f16 | `qwen2-14b_Hq32_Hkv4_K8192_D128` | 133.4μs |
| f16 | `llama3-8b_Hq32_Hkv8_K1024_D128` | 34.8μs |
| f16 | `llama3-8b_Hq32_Hkv8_K2048_D128` | 45.4μs |
| f16 | `llama3-8b_Hq32_Hkv8_K4096_D128` | 78.4μs |
| f16 | `llama3-8b_Hq32_Hkv8_K8192_D128` | 138.5μs |
| f16 | `llama3-70b_Hq64_Hkv8_K1024_D128` | 45.4μs |
| f16 | `llama3-70b_Hq64_Hkv8_K2048_D128` | 77.7μs |
| f16 | `llama3-70b_Hq64_Hkv8_K4096_D128` | 132.3μs |
| f16 | `llama3-70b_Hq64_Hkv8_K8192_D128` | 256.1μs |
| f16 | `mqa-group16_Hq32_Hkv2_K1024_D128` | 34.2μs |
| f16 | `mqa-group16_Hq32_Hkv2_K2048_D128` | 45.5μs |
| f16 | `mqa-group16_Hq32_Hkv2_K4096_D128` | 78.3μs |
| f16 | `mqa-group16_Hq32_Hkv2_K8192_D128` | 132.7μs |
| bf16 | `qwen2-14b_Hq32_Hkv4_K1024_D128` | 34.3μs |
| bf16 | `qwen2-14b_Hq32_Hkv4_K2048_D128` | 45.1μs |
| bf16 | `qwen2-14b_Hq32_Hkv4_K4096_D128` | 78.3μs |
| bf16 | `qwen2-14b_Hq32_Hkv4_K8192_D128` | 132.8μs |
| bf16 | `llama3-8b_Hq32_Hkv8_K1024_D128` | 34.8μs |
| bf16 | `llama3-8b_Hq32_Hkv8_K2048_D128` | 45.1μs |
| bf16 | `llama3-8b_Hq32_Hkv8_K4096_D128` | 78.5μs |
| bf16 | `llama3-8b_Hq32_Hkv8_K8192_D128` | 137.9μs |
| bf16 | `llama3-70b_Hq64_Hkv8_K1024_D128` | 45.1μs |
| bf16 | `llama3-70b_Hq64_Hkv8_K2048_D128` | 77.8μs |
| bf16 | `llama3-70b_Hq64_Hkv8_K4096_D128` | 132.5μs |
| bf16 | `llama3-70b_Hq64_Hkv8_K8192_D128` | 251.9μs |
| bf16 | `mqa-group16_Hq32_Hkv2_K1024_D128` | 34.2μs |
| bf16 | `mqa-group16_Hq32_Hkv2_K2048_D128` | 45.5μs |
| bf16 | `mqa-group16_Hq32_Hkv2_K4096_D128` | 78.1μs |
| bf16 | `mqa-group16_Hq32_Hkv2_K8192_D128` | 132.6μs |

### `gelu_tanh`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.7μs | 20.0μs | **1.20×** |
| f32 | `N16777216` | 591.2μs | 680.4μs | **1.15×** |
| f16 | `N1048576` | 13.5μs | 19.3μs | **1.43×** |
| f16 | `N16777216` | 282.5μs | 341.6μs | **1.21×** |

### `hardsigmoid`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.3μs | 20.5μs | **1.26×** |
| f32 | `N16777216` | 590.9μs | 679.7μs | **1.15×** |
| f16 | `N1048576` | 13.9μs | 18.6μs | **1.34×** |
| f16 | `N16777216` | 281.5μs | 341.2μs | **1.21×** |

### `hardswish`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.3μs | 21.8μs | **1.34×** |
| f32 | `N16777216` | 591.0μs | 678.2μs | **1.15×** |
| f16 | `N1048576` | 14.2μs | 19.4μs | **1.37×** |
| f16 | `N16777216` | 281.5μs | 339.6μs | **1.21×** |

### `hardtanh`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 15.7μs | 21.7μs | **1.38×** |
| f32 | `N16777216` | 591.3μs | 680.5μs | **1.15×** |
| f16 | `N1048576` | 12.9μs | 22.9μs | **1.77×** |
| f16 | `N16777216` | 281.5μs | 343.0μs | **1.22×** |

### `leaky_relu`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.0μs | 20.6μs | **1.29×** |
| f32 | `N16777216` | 591.2μs | 679.2μs | **1.15×** |
| f16 | `N1048576` | 13.0μs | 20.6μs | **1.58×** |
| f16 | `N16777216` | 281.8μs | 342.4μs | **1.22×** |

### `log`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 15.6μs | 20.1μs | **1.29×** |
| f32 | `N16777216` | 591.4μs | 680.2μs | **1.15×** |
| f16 | `N1048576` | 12.9μs | 19.6μs | **1.52×** |
| f16 | `N16777216` | 283.3μs | 339.8μs | **1.20×** |

### `log_softmax`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_C1024` | 17.4μs | 15.5μs | 0.89× | 20.0μs | **1.15×** |
| f32 | `R512_C4096` | 22.7μs | 21.9μs | ≈ | 39.4μs | **1.73×** |
| f32 | `R2048_C1024` | 26.7μs | 31.5μs | **1.18×** | 21.0μs | 0.79× |
| f32 | `R2048_C4096` | 292.5μs | 285.3μs | ≈ | 395.0μs | **1.35×** |
| f32 | `R4096_C1024` | 44.3μs | 58.2μs | **1.31×** | 45.5μs | ≈ |
| f32 | `R4096_C4096` | 585.0μs | 571.7μs | ≈ | 801.4μs | **1.37×** |
| f16 | `R512_C1024` | 31.6μs | 16.0μs | 0.51× | 20.1μs | 0.64× |
| f16 | `R512_C4096` | 21.5μs | 19.7μs | 0.91× | 37.4μs | **1.74×** |
| f16 | `R2048_C1024` | 25.7μs | 32.5μs | **1.26×** | 19.3μs | 0.75× |
| f16 | `R2048_C4096` | 65.6μs | 58.1μs | 0.89× | 170.6μs | **2.60×** |
| f16 | `R4096_C1024` | 43.8μs | 57.3μs | **1.31×** | 32.0μs | 0.73× |
| f16 | `R4096_C4096` | 300.1μs | 286.5μs | ≈ | 491.6μs | **1.64×** |

### `masked_fill`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 14.9μs | 40.1μs | **2.68×** |
| f32 | `R512_H4096` | 36.7μs | 39.7μs | **1.08×** |
| f32 | `R2048_H1024` | 24.5μs | 39.0μs | **1.59×** |
| f32 | `R2048_H4096` | 380.8μs | 723.3μs | **1.90×** |
| f32 | `R4096_H1024` | 152.4μs | 131.9μs | 0.87× |
| f32 | `R4096_H4096` | 640.3μs | 1.45ms | **2.26×** |

### `maximum`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 30.1μs | 19.2μs | 0.64× |
| f32 | `N16777216` | 885.4μs | 995.2μs | **1.12×** |
| f16 | `N1048576` | 15.9μs | 19.8μs | **1.24×** |
| f16 | `N16777216` | 425.0μs | 499.4μs | **1.17×** |

### `minimum`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 19.7μs | 19.3μs | ≈ |
| f32 | `N16777216` | 886.1μs | 995.4μs | **1.12×** |
| f16 | `N1048576` | 14.0μs | 19.2μs | **1.37×** |
| f16 | `N16777216` | 425.1μs | 499.9μs | **1.18×** |

### `mish`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 18.4μs | 20.5μs | **1.11×** |
| f32 | `N16777216` | 591.3μs | 679.2μs | **1.15×** |
| f16 | `N1048576` | 17.1μs | 19.2μs | **1.13×** |
| f16 | `N16777216` | 285.6μs | 342.0μs | **1.20×** |

### `mmvq_multim`

_Self-only: no PyTorch/library equivalent — baracuda timings below are absolute, not a comparison._

| dtype | shape | baracuda |
| --- | --- | --- |
| f32 | `M1_N4096_C4096` | 136.8μs |
| f32 | `M1_N11008_C4096` | 156.9μs |
| f32 | `M1_N32000_C4096` | 441.6μs |
| f32 | `M2_N4096_C4096` | 133.0μs |
| f32 | `M2_N11008_C4096` | 157.1μs |
| f32 | `M2_N32000_C4096` | 445.0μs |
| f32 | `M4_N4096_C4096` | 135.9μs |
| f32 | `M4_N11008_C4096` | 159.7μs |
| f32 | `M4_N32000_C4096` | 447.7μs |
| f32 | `M8_N4096_C4096` | 128.6μs |
| f32 | `M8_N11008_C4096` | 173.4μs |
| f32 | `M8_N32000_C4096` | 471.3μs |

### `neg`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 15.9μs | 24.3μs | **1.53×** |
| f32 | `N16777216` | 590.9μs | 681.1μs | **1.15×** |
| f16 | `N1048576` | 13.7μs | 19.7μs | **1.44×** |
| f16 | `N16777216` | 281.3μs | 341.5μs | **1.21×** |

### `pow`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 15.8μs | 20.1μs | **1.27×** |
| f32 | `N16777216` | 882.4μs | 986.2μs | **1.12×** |
| f16 | `N1048576` | 17.1μs | 19.9μs | **1.16×** |
| f16 | `N16777216` | 423.1μs | 498.6μs | **1.18×** |

### `reciprocal`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 15.1μs | 19.1μs | **1.26×** |
| f32 | `N16777216` | 591.5μs | 684.0μs | **1.16×** |
| f16 | `N1048576` | 14.2μs | 20.1μs | **1.42×** |
| f16 | `N16777216` | 282.2μs | 342.0μs | **1.21×** |

### `reduce_logsumexp`

| dtype | shape | baracuda |
| --- | --- | --- |
| f32 | `R512_H1024` | 209.6μs |
| f32 | `R512_H4096` | 826.9μs |
| f32 | `R2048_H1024` | 209.2μs |
| f32 | `R2048_H4096` | 827.3μs |
| f32 | `R4096_H1024` | 209.2μs |
| f32 | `R4096_H4096` | 830.3μs |

### `reduce_min`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 24.9μs | 45.5μs | **1.83×** | 21.2μs | 0.85× |
| f32 | `R512_H4096` | 26.2μs | 167.0μs | **6.38×** | 21.7μs | 0.83× |
| f32 | `R2048_H1024` | 82.1μs | 54.0μs | 0.66× | 23.1μs | 0.28× |
| f32 | `R2048_H4096` | 151.1μs | 753.6μs | **4.99×** | 34.2μs | 0.23× |
| f32 | `R4096_H1024` | 106.1μs | 181.7μs | **1.71×** | 22.9μs | 0.22× |
| f32 | `R4096_H4096` | 297.4μs | 1.59ms | **5.36×** | 366.2μs | **1.23×** |

### `reduce_norm2`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 24.3μs | 42.2μs | **1.74×** | 22.6μs | 0.93× |
| f32 | `R512_H4096` | 26.3μs | 154.0μs | **5.85×** | 22.4μs | 0.85× |
| f32 | `R2048_H1024` | 81.9μs | 51.2μs | 0.62× | 22.3μs | 0.27× |
| f32 | `R2048_H4096` | 150.9μs | 748.4μs | **4.96×** | 30.8μs | 0.20× |
| f32 | `R4096_H1024` | 106.4μs | 177.7μs | **1.67×** | 22.9μs | 0.22× |
| f32 | `R4096_H4096` | 297.2μs | 1.59ms | **5.35×** | 365.5μs | **1.23×** |

### `reduce_prod`

| dtype | shape | baracuda | cuDNN | cuDNN/baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- | --- | --- |
| f32 | `R512_H1024` | 24.8μs | 42.4μs | **1.71×** | 22.8μs | 0.92× |
| f32 | `R512_H4096` | 26.5μs | 154.4μs | **5.83×** | 23.2μs | 0.88× |
| f32 | `R2048_H1024` | 81.9μs | 51.1μs | 0.62× | 22.8μs | 0.28× |
| f32 | `R2048_H4096` | 151.0μs | 748.6μs | **4.96×** | 35.2μs | 0.23× |
| f32 | `R4096_H1024` | 106.4μs | 177.6μs | **1.67×** | 22.1μs | 0.21× |
| f32 | `R4096_H4096` | 296.9μs | 1.59ms | **5.36×** | 365.5μs | **1.23×** |

### `reduce_std`

| dtype | shape | baracuda |
| --- | --- | --- |
| f32 | `R512_H1024` | 167.4μs |
| f32 | `R512_H4096` | 662.2μs |
| f32 | `R2048_H1024` | 167.7μs |
| f32 | `R2048_H4096` | 745.0μs |
| f32 | `R4096_H1024` | 167.6μs |
| f32 | `R4096_H4096` | 893.2μs |

### `reduce_var`

| dtype | shape | baracuda |
| --- | --- | --- |
| f32 | `R512_H1024` | 167.9μs |
| f32 | `R512_H4096` | 665.0μs |
| f32 | `R2048_H1024` | 168.4μs |
| f32 | `R2048_H4096` | 746.0μs |
| f32 | `R4096_H1024` | 169.0μs |
| f32 | `R4096_H4096` | 893.0μs |

### `relu6`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.6μs | 22.0μs | **1.33×** |
| f32 | `N16777216` | 591.1μs | 679.1μs | **1.15×** |
| f16 | `N1048576` | 13.6μs | 22.0μs | **1.61×** |
| f16 | `N16777216` | 281.5μs | 340.7μs | **1.21×** |

### `rsqrt`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.3μs | 18.3μs | **1.12×** |
| f32 | `N16777216` | 591.3μs | 680.2μs | **1.15×** |
| f16 | `N1048576` | 14.6μs | 19.0μs | **1.30×** |
| f16 | `N16777216` | 281.9μs | 342.1μs | **1.21×** |

### `selu`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 17.3μs | 20.1μs | **1.16×** |
| f32 | `N16777216` | 591.3μs | 679.5μs | **1.15×** |
| f16 | `N1048576` | 13.9μs | 19.6μs | **1.41×** |
| f16 | `N16777216` | 282.1μs | 341.2μs | **1.21×** |

### `sigmoid`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.2μs | 20.7μs | **1.28×** |
| f32 | `N16777216` | 591.4μs | 682.6μs | **1.15×** |
| f16 | `N1048576` | 14.0μs | 18.6μs | **1.33×** |
| f16 | `N16777216` | 283.4μs | 341.4μs | **1.20×** |

### `sign`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 17.1μs | 20.2μs | **1.19×** |
| f32 | `N16777216` | 591.3μs | 680.9μs | **1.15×** |
| f16 | `N1048576` | 13.8μs | 19.2μs | **1.39×** |
| f16 | `N16777216` | 281.4μs | 341.4μs | **1.21×** |

### `silu`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 17.3μs | 18.5μs | **1.07×** |
| f32 | `N16777216` | 591.6μs | 683.2μs | **1.15×** |
| f16 | `N1048576` | 13.5μs | 18.9μs | **1.40×** |
| f16 | `N16777216` | 283.3μs | 340.8μs | **1.20×** |

### `sin`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 15.2μs | 20.4μs | **1.34×** |
| f32 | `N16777216` | 590.6μs | 679.1μs | **1.15×** |
| f16 | `N1048576` | 16.0μs | 20.4μs | **1.28×** |
| f16 | `N16777216` | 282.8μs | 339.8μs | **1.20×** |

### `softplus`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 17.4μs | 19.9μs | **1.14×** |
| f32 | `N16777216` | 591.4μs | 679.4μs | **1.15×** |
| f16 | `N1048576` | 14.4μs | 20.8μs | **1.44×** |
| f16 | `N16777216` | 283.7μs | 341.8μs | **1.20×** |

### `softsign`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.0μs | 63.8μs | **3.99×** |
| f32 | `N16777216` | 591.7μs | 2.35ms | **3.97×** |
| f16 | `N1048576` | 13.7μs | 63.5μs | **4.63×** |
| f16 | `N16777216` | 282.2μs | 1.18ms | **4.18×** |

### `sqrt`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 15.8μs | 19.4μs | **1.22×** |
| f32 | `N16777216` | 591.4μs | 681.4μs | **1.15×** |
| f16 | `N1048576` | 13.3μs | 20.0μs | **1.51×** |
| f16 | `N16777216` | 282.2μs | 342.0μs | **1.21×** |

### `square`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.3μs | 19.4μs | **1.19×** |
| f32 | `N16777216` | 591.0μs | 680.0μs | **1.15×** |
| f16 | `N1048576` | 13.9μs | 21.7μs | **1.56×** |
| f16 | `N16777216` | 281.3μs | 342.2μs | **1.22×** |

### `sub`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 17.6μs | 20.3μs | **1.15×** |
| f32 | `N16777216` | 883.3μs | 997.2μs | **1.13×** |
| f16 | `N1048576` | 13.5μs | 20.1μs | **1.49×** |
| f16 | `N16777216` | 419.9μs | 498.9μs | **1.19×** |

### `tanh`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `N1048576` | 16.4μs | 19.0μs | **1.15×** |
| f32 | `N16777216` | 591.4μs | 677.8μs | **1.15×** |
| f16 | `N1048576` | 14.2μs | 19.0μs | **1.34×** |
| f16 | `N16777216` | 282.2μs | 341.2μs | **1.21×** |

### `topk`

| dtype | shape | baracuda | PyTorch | PyTorch/baracuda |
| --- | --- | --- | --- | --- |
| f32 | `B1_L1024_K64` | 19.9μs | 40.3μs | **2.03×** |
| f32 | `B8_L512_K16` | 15.9μs | 40.6μs | **2.55×** |
| f32 | `B32_L128_K4` | 15.7μs | 42.1μs | **2.68×** |

<!-- END auto-generated phase29 rollup -->
