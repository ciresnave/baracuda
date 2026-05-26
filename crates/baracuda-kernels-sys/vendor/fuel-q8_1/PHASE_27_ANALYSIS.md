# Phase 27 — Q8_1 staging vs baracuda MMVQ analysis

**Date**: 2026-05-25
**Vendored baseline**: `crates/baracuda-kernels-sys/vendor/fuel-q8_1/quantized.cu` (4537 LOC, alpha.36)
**Compared against**: `crates/baracuda-kernels-sys/kernels/gguf/mmvq.cu` (2281 LOC, alpha.37) + `kernels/include/baracuda_gguf.cuh` (696 LOC).
**Hardware reference**: RTX 4070 (sm_89).
**Result**: **No quick win to port.** The two implementations target different performance regimes; baracuda's path is well-tuned for the M=1 decode regime it is currently used in. Substantial value remains in Fuel's design for **M>1 prefill** — but capturing it is a multi-day refactor, not a constant tweak.

---

## 1. Inventory of vendored quantized.cu

| Section | Lines | Symbols of interest |
|---|---|---|
| Quant primitives + helpers | 1-300 | `WARP_SIZE`, `QK*`, `QI*`, `QR*` constants, `warp_reduce_sum/max`, `get_int_from_int8[_aligned]`, `__dp4a` shim |
| Block structs | 290-650 | `block_q4_0` … `block_q8_K` (matches baracuda's layout) |
| Type-0/1 dequant | 655-757 | `dequantize_q4_0/q4_1/q5_0/q5_1/q8_0` (identical to baracuda's primitives in `baracuda_gguf.cuh`) |
| `dequantize_block_q*_K_*` | 780-1158 | Per-format full-block dequant (matches baracuda's `dequantize_block_q*_K_tmpl`) |
| `dequantize_mul_mat_vec` (FP path) | 1160-~1800 | **The same algorithm as baracuda's `dequantize_mul_mat_vec` template** — dequant → fp32 multiply → warp-shuffle reduce. |
| `vec_dot_q*_q8_1_impl<vdr>` | 1824-2280 | **The Q8_1 staging hot path**: int8 SIMD dot product via `__dp4a`. |
| `vec_dot_q*_q8_1` (wrappers) | 2283-2620 | Loads 4-byte chunks of weights + activations via `get_int_from_int8_aligned`, calls the `_impl`. |
| `mul_mat_vec_q<ncols_y, …>` (template) | 2626-2698 | **Parameterized M=`ncols_y` MMVQ kernel**. Compile-time M ∈ {1, 2, 3, 4, 5, 6, 7, 8}. |
| `mul_mat_vec_q*_q8_1_cuda1..cuda8` | 2700-3346 | 10 qtypes × 8 batch sizes = **80 entry points**. |
| `quantize_q8_1` | 3348-3382 | The staging kernel — fp32 activations → Q8_1 in 128-byte blocks. ~10 lines core, runs once per launch. |
| `mul_mat_q<>` + `load_tiles_q*` + `vec_dot_q*_mul_mat` | 3384-end | **MMQ path** — the full tile-loaded GEMM (not MMVQ). Out of scope for Phase 27 (would be a separate "MMQ port" effort). |

The Q8_1 staging path is structured as:

1. Host-side: launch `quantize_q8_1` over the FP activations once (block 256×ny, sub-millisecond).
2. Host-side: launch `mul_mat_vec_q<M, ...>` where `M` = `ncols_y` (number of activation vectors).
3. Per block: each thread iterates `kbx` over weight blocks, loads `vdr` 32-bit chunks of weights + the matching Q8_1 quants, calls `vec_dot_q_q8_1_impl<vdr>` which is **almost entirely `__dp4a` calls** plus a single `d_x * d_y` FP fixup at block-end.

---

## 2. Per-aspect comparison

### 2a. Inner-loop arithmetic (THE big difference)

**Fuel (Q8_1 staging path)** — int8 SIMD dot:
```
sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi);  // 4 int8×int8 muls + accumulate, 1 cycle
sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);
…
return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);
```
Per 32 quants of work: ~2× `__dp4a` (each = 4 int8 MACs), 1 fp32 mul, 1 fp32 fma. **8 int8 MACs in ~2 cycles.**

**Baracuda (`dequantize_mul_mat_vec` template + per-qtype `mmvq_q*_K_tmpl`)** — fp dequant then fp multiply:
```
dequantize_kernel(vx, ib, iqs + j/qr, v);              // 2 fp32 muls + 1 fp32 add (q4_0)
tmp += v.x * mmvq_io<ActT>::load(&y[...]);             // 1 fp32 fma
tmp += v.y * mmvq_io<ActT>::load(&y[...]);             // 1 fp32 fma
```
Per 2 quants of work: 4 fp32 ops. **8 quants → ~16 fp32 ops vs Fuel's 2 cycles of `__dp4a`.**

On sm_89, `__dp4a` throughput is **256 ops/SM/cycle** (per 32-bit dot-of-4), vs fp32 MAD at **128 ops/SM/cycle**. Combined with the lower op count, the Fuel path has a theoretical **~8× arithmetic-density advantage** on the quantized dot.

**Caveat**: MMVQ is **memory-bound**, not compute-bound, for the M=1 case. Both paths load the same ~ncols/qk × sizeof(block_qX_0) bytes of weights per row. The arithmetic-density advantage only materializes when:
- Memory pressure drops (e.g., L1/L2 hits across multiple M rows reusing the same weight) → **M>1 prefill**.
- Or the FP dequant overhead actually surfaces in the active warp count (low-occupancy regimes — small models, low SM utilization).

For pure decode (M=1) on a ≥1B model, both paths likely saturate gmem bandwidth at the same rate. The Q8_1 path saves SM cycles but those cycles were never the bottleneck.

### 2b. Activation reuse across M rows (THE other big difference)

**Fuel**: `mul_mat_vec_q<ncols_y, ...>` — for compile-time `ncols_y ∈ {1..8}`, a single weight load is reused across all `ncols_y` activation vectors via `tmp[ncols_y][rows_per_cuda_block]`. **Each gmem byte of weight is consumed by up to 8 dot products.**

**Baracuda**: One launch per activation vector (effectively `ncols_y=1` only). For M=8 prefill, baracuda re-reads the entire weight tensor 8× from gmem. (The new Phase 20 batched-MMVQ kernel does support M>1 via grid-y, but each block still processes one (token, row) tuple — no weight reuse across tokens within a block.)

This is the **single largest concrete perf gap**. For a 7B-class model at prefill M=8: Fuel saves ~7× weight bandwidth vs current baracuda. On RTX 4070's 504 GB/s gmem with a 4 GB Q4_0 weight tensor, that's ~50 ms → ~7 ms per matmul; a real speedup.

### 2c. Block dimensions

| Aspect | Fuel (`mul_mat_vec_q`) | baracuda type-0/1 | baracuda k-quants |
|---|---|---|---|
| Threads/block | `WARP_SIZE × nwarps` = 32×{2,4} = 64 or 128 | 32×1 = 32 | 32×1 = 32 |
| Rows per block | `rows_per_cuda_block` ∈ {1, 2} | 1 (`GGML_CUDA_MMV_Y=1`) | 1 |
| Warps/block | 2-4 | 1 | 1 |
| M handled / block | 1-8 (compile-time) | 1 (hardcoded) | 1 (hardcoded) |
| Cross-warp reduce | `tmp_shared[nwarps-1][…][…][WARP_SIZE]` smem | None (single warp) | None (single warp) |

Fuel uses **multi-warp blocks** to share the K-loop across warps (each warp handles a slice of K, then a smem reduction). Baracuda's blocks are single-warp by design (no cross-warp sync). For M=1 this is fine — and arguably better, since the cross-warp reduction in Fuel costs an `__syncthreads` + smem traffic. For M>1 the multi-warp + multi-row geometry pays off.

### 2d. SMEM tile layouts

Neither MMVQ path pre-stages weights or activations in SMEM. (Pre-staging is the **MMQ** path — `mul_mat_q` in Fuel's file at line 438, which uses `tile_x_ql[mmq_y * 2*WARP_SIZE + mmq_y]` and `tile_y_qs[mmq_x * WARP_SIZE]`. That's a different beast and out of Phase 27 scope.)

Fuel's MMVQ only uses SMEM for the cross-warp partial-sum reduction (~`nwarps × ncols_y × rows × WARP_SIZE × 4 B`, at most ~1 KiB). Baracuda's MMVQ uses zero SMEM.

### 2e. Register usage / accumulator shape

- **Fuel**: `float tmp[ncols_y][rows_per_cuda_block]` = up to 8×2 = 16 fp32 accumulators per thread (for cuda8 + M>1 path). Plus int register pressure for the `__dp4a` dot products.
- **Baracuda**: `float tmp` = 1 fp32 accumulator per thread.

Fuel's deeper accumulator forces lower occupancy (fewer concurrent warps per SM) but it's the right call when each weight load amortizes across 16 MACs vs 1.

### 2f. Special-shape handling

- **Fuel**: Handles `ncols_x % qk` cleanly (divides at line 2641). Does **NOT** handle non-divisible K — same constraint as baracuda. Both reject K%qk != 0 upstream.
- **Fuel** has `blocks_per_iter = vdr * nwarps * WARP_SIZE / qi` — the inner loop stride is sized so all threads consume contiguous blocks. Tail handling is automatic via the `kbx < blocks_per_row_x` test.
- **Baracuda**: Same logical guard via `for (i = 0; i < ncols; i += iter_stride)` where `iter_stride = 2*GGML_CUDA_DMMV_X`.
- Neither path padds K to a multiple. Both require the caller to enforce the alignment (which the Rust plan layer does).

### 2g. The `quantize_q8_1` cost

`quantize_q8_1` runs once per inference step over the activation tensor: M × K fp32 → M × (K/32 × 36 B) int8 + scale/sum. For K=4096, M=1: one launch of 4096 threads, ~µs. For M=512, K=4096: 2M threads, still sub-ms.

This is a **fixed prelude cost** in the Q8_1 staging path. For a single matmul it might dominate vs decode-time savings. For prefill (multiple matmuls reusing the same activation × multiple weight tensors per layer × multiple layers) it amortizes well — quantize once, multiply many times.

### 2h. dequantize-per-token / fused path

The question in the brief: "Does baracuda have an equivalent fused path?"

**Answer**: No. Baracuda has no Q8_1 staging at all. The closest thing is the Phase 8 `quantize_per_token_group` (in `kernels/include/baracuda_quantize_per_token_group.cuh`) which is a different design — used for *output* quantization, not for staging fp activations before MMVQ. It produces Q8_0-format scales-per-group, not Q8_1's (scale, sum) pairs.

---

## 3. Ranked optimization opportunities

### Tier S — Significant perf wins (multi-day work)

**S1. Multi-M (ncols_y > 1) MMVQ via Q8_1 staging.** Port Fuel's `quantize_q8_1` + `mul_mat_vec_q<M, ...>` family for M ∈ {1, 2, 4, 8} into baracuda. This is the **principal win** of the staging design and the actual reason Fuel asked us to evaluate it. Estimated impact: 3-7× speedup on prefill matmuls (M=8 weight bandwidth reduction). Effort: ~3-5 days (10 qtypes × 4 M-sizes × FFI wrappers + test fixtures + a quantize-once-multiply-many host-side orchestration layer). **Not a Phase 27 deliverable** — proposal is to spec this as Phase 28 (or a "Phase 22a" prefill-MMVQ slot).

**S2. DP4A int8 fast-path for type-0/1 MMVQ (M=1).** Even without Q8_1 staging, an internal "dequantize fp activation to int8 + dot via `__dp4a`" path could replace the per-element fp32 dequant in `dequantize_mul_mat_vec` for Q4_0 / Q5_0 / Q8_0. The catch: the activation **must** be quantized first (the Q8_1 staging is unavoidable for DP4A — you can't dot `fp32 × int4` directly). So S2 ≈ "S1 restricted to M=1" — no separate work. Documenting only.

### Tier A — Marginal gains (constant tweaks, 0.5-1 day)

**A1. Block dimension tuning for k-quant MMVQ.** Baracuda's k-quants use a single warp (32 threads, 1 row/block). Fuel's `mul_mat_vec_q<1, QK_K, QI*_K, …>` for k-quants uses `nwarps=4` (128 threads, 1 row/block) when `ncols_y=1`. The 4-warp version may hide K-loop latency better on RTX 4070 due to higher per-block warp count → fewer block-launch barriers. Tentative estimate: 5-15% on Q4_K / Q5_K decode. Worth a microbench. **Not portable as a constant tweak** — would need refactoring baracuda's `mmvq_q*_K_tmpl` to support multi-warp partial-sum reduction.

**A2. `__dp4a` for the Q8_K k-quant.** Q8_K is a baracuda-only kernel (closing a Fuel gap during Phase 11.4). The current implementation does fp32 multiplies. Since Q8_K is **already** int8 quants × fp scale per super-block, an int8-version with `__dp4a` could be 2-4× faster on the dot proper (but the q8_K case is small — 256 elements/super-block, fewer matmuls in real workloads). Effort: ~half a day; modest win. Not chosen for Phase 27.

### Tier B — No measurable gain expected

**B1. Re-do baracuda's `dequantize_mul_mat_vec` reduction with explicit shuffle width.** Baracuda already uses `__shfl_xor_sync` with mask 0xffffffff — same as Fuel. No change.

**B2. `__restrict__` annotations / `--restrict` flag.** Both already use `__restrict__` on all kernel parameters. No change.

**B3. Activation half/bf16 conversion shape.** Baracuda's `mmvq_io<ActT>::load()` already inlines to a no-op for FP types (Phase 18). Fuel's path was fp32-only for the staging trip (the Q8_1 quants drop dtype info). No win to port.

---

## 4. Decision

**Nothing to port in Phase 27.** Reasoning:

1. The **only material win** in Fuel's design is the multi-M (`ncols_y`) prefill MMVQ. That is **not a constant tweak** — it requires:
   - A new `quantize_q8_1` kernel + FFI in baracuda (~50 LOC + FFI).
   - 10 qtypes × {1,2,4,8} M-sizes = 40 new MMVQ entry points (or one templated entry point per qtype with M as a runtime switch table).
   - A new host-side orchestration: "stage activations once, do N matmuls". Today, baracuda's matmul Rust plans are stateless per-call. Adding a stage-and-reuse pattern is a Plan-trait/Buf-management redesign.
   - Coverage testing across {fp32, fp16, bf16} activations × all qtypes.
   - Numerical-tolerance fixtures: Q8_1 staging changes accumulator dtype (fp32 throughout vs baracuda's current fp32 sum of fp32-dequant-products). PyTorch references would need bigger tolerance.

   This is roughly **Phase 22-sized work**. It belongs in the roadmap as its own milestone, not bolted onto a Phase 27 inspection task.

2. The **k-quant multi-warp tuning** (A1 above) is the closest thing to a "port a constant" win, but it still requires a non-trivial rewrite of the partial-sum reduction (introducing SMEM the kernels don't currently use). Without a benchmark showing the gain on actual workloads, shipping the rewrite carries net negative expected value (added complexity, possible regressions, marginal gain).

3. **Memory-bound vs compute-bound matters**: for the M=1 decode case baracuda primarily ships, both paths are ~equally gmem-bound. The arithmetic-density gain from `__dp4a` only surfaces with weight reuse (M>1) or extreme low occupancy (which we don't observe at typical 4070 GPU utilizations).

4. The vendored directory **stays in the tree** — Fuel hasn't deleted their original yet, and the Phase 28 (prefill MMVQ) work above would re-read this file. Removing it now would just mean re-vendoring later.

## 5. Recommendation for the roadmap

Add a **Phase 28 candidate** (or "Phase 22a — prefill MMVQ"):

> "Port Fuel's `mul_mat_vec_q<ncols_y>` multi-M MMVQ design into
> baracuda's `baracuda_kernels_mmvq_<qtype>_run` family. Includes
> the `quantize_q8_1` staging kernel as a new FFI symbol. Target:
> 3-7× speedup on prefill matmuls with M ∈ {2, 4, 8}. Estimated
> 3-5 days. Numerical-tolerance fixtures revisited."

Phase 27 itself closes with: **no code changes**, this report documents why, and the vendored directory stays put.

## 6. Files

- This report: `crates/baracuda-kernels-sys/vendor/fuel-q8_1/PHASE_27_ANALYSIS.md` (new)
- No source changes.
- No cargo check needed.
