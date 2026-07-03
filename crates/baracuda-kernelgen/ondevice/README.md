# On-device validation harnesses

Manual `nvcc` harnesses (not wired into `cargo test`) that launch the **generated**
`.cu` kernels on the GPU and diff against a host/CPU reference — the checks that
are catchable only on device. The `#include`d kernel names track the catalog cells
in `bin/kernelgen.rs`; update both together.

**Run (Windows):** from a Visual Studio dev shell so `nvcc` finds `cl.exe`
(`Enter-VsDevShell`), or an x64 Native Tools prompt. General shape:

```sh
cargo run -p baracuda-kernelgen --bin kernelgen -- <outdir>   # generate the catalog .cu
cp crates/baracuda-kernelgen/ondevice/<harness>.cu <outdir>/  # place harness beside them
nvcc -O3 -arch=sm_89 <outdir>/<harness>.cu -o <outdir>/<harness> && <outdir>/<harness>
```

---

## `reduce_validate.cu` — general reduction path (item 03)

Launches the general-path reduction kernels (`_reduce_{tag}_ax{hex}[_kd]`) with
small hand-checkable shapes vs a CPU reference. Validates:

- the emitter↔host **ABI** — `shape[]` / `s0[]` / `so[]` indexing and `n_out`;
- the **keepdim ⇒ `so` by input axis** vs **collapse ⇒ `so` by kept position** split;
- **NaN propagation** in the `Max` `has`-flag fold (torch.amax semantics);
- multi-axis, middle-axis (two kept axes), and reduce-all (kept empty);
- **integer accumulation** — i32 last-axis Sum/Max fold in a `long long` accumulator
  (exact, no float rounding), including negatives.

Expected: `ALL PASSED` (bit-exact, `maxerr 0`; NaN propagated).

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **all 9 cases PASS**
(incl. the InnerContig block-per-row last-axis path and the i32 exact-int Sum/Max cases).

### Benchmark — `reduce_bench.cu`

Compares the fast-path (last-axis) vs general (outer-axis) reduction on a large
`[8192,8192]` f32 tensor against a copy-bandwidth reference (reductions are
memory-bound, so GB/s vs. the copy peak is the figure of merit).

**RTX 4070 Laptop (sm_89):**

| kernel | ms | GB/s |
| --- | --- | --- |
| copy (bandwidth ref, read+write) | 2.74 | 195.8 |
| reduce **last** axis (block-per-row) | **1.18** | **227.4** |
| reduce **axis 0** (general/outer, 1 thread/col) | 2.27 | 118.3 |

The block-per-row rewrite gave a **4.4× win** on the last axis (was 5.15 ms /
52.2 GB/s with the old one-thread-per-row *sequential*, uncoalesced fold); it now
reads at ~227 GB/s — above the copy's read+write ceiling because a reduction is
read-only, i.e. memory-optimal. The outer-axis follow-up is now the **split-K
variant** (see `splitk_validate.cu` below) — regime-dependent, shipped as a
bench-gated schedule variant beside the baseline.

---

## `splitk_validate.cu` — split-K outer-axis reduction VARIANT (phase 2)

The first bench-gated schedule variant (ship-top-K policy — see
`docs/planning/foundational/11-variant-generators-backlog.md`): a two-kernel
split-K (`_splitk_partial` → workspace → `_splitk_combine`) beside the
single-pass baseline for the outer-axis Sum/Mean cell. Deterministic for a fixed
`chunk_rows`, no atomics — but a **different association** than the baseline
(`VariantFidelity::ReassociatedDeterministic`), so it is selectable only through
its honest contract, never silently.

Checks: baseline + split-K (ragged chunks) vs a CPU f64 oracle; degenerate
`n_chunks=1` **memcmp-identical** to the baseline (same association); run-to-run
determinism (memcmp).

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 — **all 3 cases PASS** (oracle
relerr 0.0), and the sweep (fixed 0.27 GB read, two stable runs):

| cols | baseline (1 thread/col) | split-K | speedup |
| --- | --- | --- | --- |
| 256 | 12.4 GB/s (starved) | 229.9 GB/s | **18.5×** |
| 1024 | 53.0 GB/s | 241.6 GB/s | **4.6×** |
| 4096 | 204.1 GB/s | 243.8 GB/s | 1.19× |
| 16384+ | ~248 GB/s | ~244 GB/s | 0.98× |

**Regime-dependent — the variant thesis in one table.** The `StructureKey`
deliberately carries no literal extents, so all these shapes are ONE cell: the
within-cell winner depends on a runtime extent (`cols` vs GPU width). That makes
this a **launch-config-class decision for the runtime selector** (Fuel, per
call), exactly why the ship-top-K policy ships both kernels: winner-only would
bake in an 18× loss or a 2% regression depending on the bench shape. (The old
"118 GB/s" figure from the item-03 session also did not reproduce at cols=8192
— laptop clock variance; the starved regime is the real, stable gap.)

---

## `dag_validate.cu` — shared-interior DAG emitter (item 02)

Launches the diamond kernels — `out = g / (g + 1)` with `g = a * b`, the shared
product hoisted to one `tmp` — vs a host oracle. Validates the one thing catchable
only on device: that the DAG rewrite (emit a shared value once, reference it twice)
is a **no-op on the computed values**, and that the hoisted-`tmp` source compiles.
Two cells exercise both hoist paths:

- `baracuda_gen_diamond_f32_scalar` — `float tmp0 = (in0[i]*in1[i]); out[i] = (tmp0 / (tmp0 + 1.0));`
- `baracuda_gen_diamond_f32_co_v4` — per-lane scoped block `{ float tmp0 = (v0.x*v1.x); vo.x = (tmp0 / (tmp0 + 1.0)); }` (no cross-lane name collision).

Expected: `ALL PASSED` (`maxerr 0` — bit-exact; dedup changes text, not values).

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **both cases PASS**
(scalar and per-lane vectorized hoist, bit-exact; PTX `.entry` present under a
headerless `-ptx` compile). The **fused-reduction epilogue** dedup (Softmax's
shared `exp(x-max)`), the DAG-based contract flops count, and the `region_to_op`
seam hash-cons are the item-02 follow-up (see `docs/planning/foundational/`).

---

## `packed_validate.cu` — packed f16/bf16 pair path (item 09 Stage 1)

Runs each **packed** kernel (`_co_v8`: half2/bf162 pairs, 128-bit accesses) and
its **scalar sibling** (the oracle, `_scalar` via a 2-byte-aligned cell) over a
corpus where input 0 sweeps **every 16-bit pattern** — all NaN payloads, ±Inf,
±0, every subnormal, max-finite — and requires the raw u16 outputs to be
memcmp-identical. Cases: `add` (Tier A native pair ops) and `relu_add` (Tier A
add + Tier B pair-scalarized relu), f16 + bf16.

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 — **all 10 cases bit-identical (add, relu_add, neg, abs, sqr × f16, bf16 — incl. every NaN payload through the Tier-A intrinsics)**
over the full sweep; compute-sanitizer `initcheck` + `memcheck` **0 errors**.

### Bench — `packed_bench.cu` (honest finding)

f16 `add`, scalar vs packed, per-kernel launcher-realistic grids:

| n (halves) | regime | packed vs scalar |
| --- | --- | --- |
| 64k | launch-bound | parity (noise) |
| 1M–4M | **L2 / instruction-bound** | **+3–8% (consistent)** |
| 16M+ | DRAM-bound | parity |

**The item-09 brief's premise ("largest unclaimed memory-bound win") is
empirically wrong on sm_89**: the coalescer merges a warp's adjacent 2-byte lane
accesses into optimal transactions, so the *scalar* f16 kernel already runs at
the DRAM ceiling (~203 GB/s) for large coalesced elementwise. The packed path is
a **modest pure win**: bit-identical always, +3–8% where instruction issue is
the limiter, never a consistent regression, and fewer issue slots burned per
element (headroom for fused compute-heavy bodies). The deferred packed stages
(Tier-A transcendentals, packed reductions) should be built as **measured
variants** gated by the item-07 bench harness, not assumed wins.

---

## `contract_validate.cu` — skinny contraction go/no-go vs cuBLAS (item 10)

The generated `_contract_tll` cell ([M≤8,K]·[K,N], f32) vs a sampled CPU f64
oracle and `cublasSgemm` (row-major via the C^T = B^T·A^T mapping), then the
long-tail bench at M ∈ {1, 8}, K = N = 4096. Needs `-lcublas`.

**Last run (RTX 4070/sm_89, CUDA 13.3): correctness EXACT (0.0 rel err vs both
oracle and cuBLAS) — but the v1 skinny SIMT schedule is a perf NO-GO: ~62 GB/s
vs cuBLAS ~245 (0.25×).** Diagnosis: one thread per column = 4096 threads in 16
blocks — the SAME occupancy starvation the outer-axis reduction baseline showed
(12–53 GB/s starved), with a sequential K load-use chain on top; cuBLAS's M=1
path split-Ks internally. The proven in-repo fix is the split-K schedule (16×
on the reduction analogue) as a bench-gated **variant** of this cell — queued
as the node's first variant. Fourth instance of measure-don't-assume: this time
the gate protected us from shipping our own thesis kernel as the default.

**Split-K rematch (same harness, variant pair added):** all 6 correctness cases
PASS (splitk exact vs cuBLAS; degenerate `n_chunks=1` memcmp-identical to base
at both shapes), and the bench:

| M | base | split-K | cuBLAS | splitk vs cuBLAS |
| --- | --- | --- | --- | --- |
| 1 | 63 GB/s | **233 GB/s** | 245 GB/s | 0.95× |
| 8 | 74 GB/s | **217 GB/s** | 234 GB/s | 0.93× |

The variant closed the 4× gap to within ~5–7% of cuBLAS on the vendor's own
plain decode cell — inside/near the `MIN_FLIP_MARGIN` noise floor, so the
honest per-cell verdict is "vendor keeps the plain cell." The generated node's
winning ground, per the §1 long-tail thesis, is the **fused-epilogue** cell
(matmul+bias/act in ONE launch, epilogue folded into `_splitk_combine`) that
the vendor serves only as a two-kernel round trip — the next rematch.

**Fused-epilogue rematch (matmul_relu):** correctness exact (0.0 vs the vendor
round-trip), but **fusion did NOT win — 0.94–0.96×**. Structural finding: at
Tiny-M the output is tiny (16 KB at M=1), so the vendor's separate relu pass
costs ~2 µs and our ~5–7% GEMM gap eats it; epilogue fusion pays only when the
epilogue's traffic is large relative to the GEMM, which at Tiny-M it never is.
The contraction long tail lives elsewhere: **dequant-fused matmul** (int4/nf4
weights dequantized in-kernel — the real quantized-decode traffic),
irregular-K, and batched-many-tiny. Sixth measure-don't-assume instance.

---

## `audit_reduce_softmax.cu` — generated-vs-bespoke audit, round 1

Generated cells vs the hand-written `baracuda-kernels-sys` kernels, called
through their own `extern "C" _run` launchers (their path selection — what
dispatch actually calls). Compile with
`-std=c++17 -Xcompiler "/Zc:preprocessor /std:c++17" -I <kernels-sys>/kernels/include`.

**Round-1 results (RTX 4070/sm_89, all correctness PASS):**

| matchup | generated | bespoke | verdict |
| --- | --- | --- | --- |
| mean last-axis [8192²] | 248 GB/s | 234 GB/s | **GEN 1.06×** |
| sum axis-0 [65536×1024] | 242 GB/s (splitk) | 171 GB/s (legacy) | **GEN 1.41×** |
| softmax [4096²] | 235 GB/s (recompute) | 229 GB/s (smem) | **GEN 1.03×** |
| softmax [2048×16384] | 200 GB/s | **0.2 GB/s** (global fallback) | **GEN 884×** |

Notes: (1) the bespoke softmax fast path IS the smem row-cache — independently
confirming the earlier gate finding that recompute ≥ smem-cache on this card;
above 47 KB rows it collapses to an O(numel·extent) fallback the generated
kernel simply doesn't have. (2) **Extract-the-delta, first application — from a
LOSING kernel:** the bespoke legacy reducer (171 GB/s) beats our general-path
BASE (55 GB/s) at identical parallelism because it passes shape/strides **by
value in kernel params** (`DimsI32/I64` → constant bank) while ours re-reads
`shape[]/s0[]/so[]` from **global pointers every loop iteration**. Our split-K
still wins the cell, but by-value dims params are a legitimate technique to
extract into the general strided/reduction emitters (queued in the backlog).
