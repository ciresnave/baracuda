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
read-only, i.e. memory-optimal. **Follow-up:** the outer-axis path (118 GB/s) is
coalesced but a sequential one-thread-per-column fold; a split-K partial-sum pass
would push it toward peak (additive via the `ReduceAxisClass` token — no re-key).

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
