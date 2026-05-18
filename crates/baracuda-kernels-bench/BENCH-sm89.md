# baracuda-kernels sm_89 baseline (RTX 4070, CUDA 13.0, cuDNN 9.16)

Phase 10 Milestone 10.2 — first measured baseline for the
`baracuda-kernels` ML op surface on Ada (sm_89). All numbers are
**placeholders** until the user runs the harness on real hardware:

```bash
cargo bench -p baracuda-kernels-bench --features sm89,cudnn
```

The full sweep takes ~30 minutes on an RTX 4070. Use `cargo bench -p
baracuda-kernels-bench --bench gemm --features sm89,cudnn` to scope to
one family while iterating on a perf change. Add `-- --quick` for
criterion's 10-sample fast pass.

Reported as TFLOPS / GFLOPS = `Throughput::Elements / 1e12` (or `/ 1e9`)
from criterion's printed `elem/sec`.

## Methodology

- CUDA events (`cudaEventRecord` + `cudaEventElapsedTime`) — measured
  on-device, no host-side noise.
- 10-launch warmup before the first timed sample so GPU clock + cache
  state settle out of idle.
- Per-shape plan + buffers built once, reused across all timed samples.
- `EpilogueKind::Identity` for GEMM (pure kernel, no bias overhead).
- `is_causal = false` for Flash Attention (worst-case work).
- `pad = k/2, stride = 1` for Conv2d (`H_out == H_in` — same-pad
  convention).

Fill values are `1.0` in dtype-appropriate units, so the kernel exercises
real FMA paths. Zero-fill triggers some kernels' early-exit code and
under-reports throughput.

## GEMM (TFLOPS)

Theoretical sm_89 peaks for context (RTX 4070, boost ~2475 MHz, 184
tensor cores):
- f32 (FP32 tensor): ~30 TFLOPS
- f16 / bf16 (tensor): ~120 TFLOPS dense, ~240 with sparsity
- fp8 (tensor): ~240 TFLOPS dense, ~480 with sparsity
- int8 (tensor): ~240 TOPS dense, ~480 with sparsity

### `K = N = 2048`

| Dtype | M=1 | M=8 | M=32 | M=128 | M=512 |
|-------|-----|-----|------|-------|-------|
| f32   | --  | --  | --   | --    | --    |
| f16   | --  | --  | --   | --    | --    |
| bf16  | --  | --  | --   | --    | --    |
| fp8   | --  | --  | --   | --    | --    |
| int8  | --  | --  | --   | --    | --    |

### `K = N = 4096`

| Dtype | M=1 | M=8 | M=32 | M=128 | M=512 |
|-------|-----|-----|------|-------|-------|
| f32   | --  | --  | --   | --    | --    |
| f16   | --  | --  | --   | --    | --    |
| bf16  | --  | --  | --   | --    | --    |
| fp8   | --  | --  | --   | --    | --    |
| int8  | --  | --  | --   | --    | --    |

### `K = N = 8192`

| Dtype | M=1 | M=8 | M=32 | M=128 | M=512 |
|-------|-----|-----|------|-------|-------|
| f32   | --  | --  | --   | --    | --    |
| f16   | --  | --  | --   | --    | --    |
| bf16  | --  | --  | --   | --    | --    |
| fp8   | --  | --  | --   | --    | --    |
| int8  | --  | --  | --   | --    | --    |

> Fill in after first run.

## Flash Attention (TFLOPS-equivalent)

`flops ≈ 4 · B · H · Q · K · D` (two GEMMs in `softmax(Q·K^T)·V`).
All entries with `B = 1, is_causal = false`.

### `D = 64`

| Dtype | H=8, Q=512 | H=8, Q=1024 | H=8, Q=2048 | H=8, Q=4096 | H=16, Q=512 | H=16, Q=1024 | H=16, Q=2048 | H=16, Q=4096 | H=32, Q=512 | H=32, Q=1024 | H=32, Q=2048 | H=32, Q=4096 |
|-------|------------|-------------|-------------|-------------|-------------|--------------|--------------|--------------|-------------|--------------|--------------|--------------|
| f32   | --         | --          | --          | --          | --          | --           | --           | --           | --          | --           | --           | --           |
| f16   | --         | --          | --          | --          | --          | --           | --           | --           | --          | --           | --           | --           |
| bf16  | --         | --          | --          | --          | --          | --           | --           | --           | --          | --           | --           | --           |

### `D = 128`

| Dtype | H=8, Q=512 | H=8, Q=1024 | H=8, Q=2048 | H=8, Q=4096 | H=16, Q=512 | H=16, Q=1024 | H=16, Q=2048 | H=16, Q=4096 | H=32, Q=512 | H=32, Q=1024 | H=32, Q=2048 | H=32, Q=4096 |
|-------|------------|-------------|-------------|-------------|-------------|--------------|--------------|--------------|-------------|--------------|--------------|--------------|
| f32   | --         | --          | --          | --          | --          | --           | --           | --           | --          | --           | --           | --           |
| f16   | --         | --          | --          | --          | --          | --           | --           | --           | --          | --           | --           | --           |
| bf16  | --         | --          | --          | --          | --          | --           | --           | --           | --          | --           | --           | --           |

> Fill in after first run.

## Conv2d (GFLOPS)

`pad = k/2, stride = 1` → `H_out = H_in`. `flops = 2 · macs`.

| Dtype | Stem (56×56, 64→64, 3×3) | Mid (28×28, 128→128, 3×3) | Deep (14×14, 256→256, 3×3) |
|-------|--------------------------|---------------------------|----------------------------|
| f32   | --                       | --                        | --                         |
| f16   | --                       | --                        | --                         |

> Fill in after first run.

## Observations from the kernel surface

These are sm_89-tuning targets we already see without running the
sweep — populate the tables above first, then prioritize the ones
below against the gap to theoretical peak.

1. **FlashSdpa trailblazer is `D ≤ 128, d_k == d_v`** — fine for MHA
   but constrains GQA / MLA / grouped-attention variants. If the
   sweep shows `D = 128` is throughput-bound by smem traffic, the
   sm_89 tune is bigger tile sizes (Ada has 100KB smem/SM vs Ampere's
   164KB but better tensor-core throughput per byte).
2. **Float GEMM is `LayoutSku::Rcr` only**; `LayoutSku::Rrr` is wired
   for int8 but not float yet. If the int8 RRR path measures better
   than int8 RCR on the small-M shapes (which is what motivated the
   bespoke kernel in the first place), porting the same RRR shape to
   f16/bf16 is the obvious sm_89 tune.
3. **`int8 × Identity × RCR` goes through CUTLASS's `mma.sync.m16n8k32`**
   path on sm_80. Ada doubles the int8 tensor-core throughput vs
   Ampere; if the sweep shows int8 is leaving headroom vs the f16
   peak ratio, an sm_89-specific instantiation is the fix.
4. **Conv2d is pinned to `CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM`**
   (see `crates/baracuda-kernels/src/conv/conv2d.rs:287`). cuDNN's
   `WINOGRAD_NONFUSED` is faster for `3×3, stride=1` shapes on Ada;
   adding a tiny algorithm-picker (heuristic on `(C_in, C_out, H, W,
   k, stride)`) is the obvious autotune target if Conv2d is the bottleneck.
5. **FP8 (E4M3) bias kernels are bespoke** (`gemm_fp8_e4m3_rcr_sm89.cu`
   and friends). Sm_89 already; no port work needed, but the bench
   will tell us whether the bespoke tile sizes are competitive vs
   what CUTLASS 4.x ships when its FP8 RCR path lands.
6. **No bench yet for cuBLAS / cuBLASLt** as a reference floor. Phase 10
   should add one — same shapes, same dtypes, separate
   `gemm_cublas` bench file — so we know what fraction of the SOL
   each plan delivers.
