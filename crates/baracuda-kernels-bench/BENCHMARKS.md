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

### gemm (bf16) — RTX 4070, 2026-05-26, `-- --quick`

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
