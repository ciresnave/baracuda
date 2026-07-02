# On-device numeric validation — general reduction path (item 03)

`reduce_validate.cu` launches the **generated** general-path reduction kernels
(`_reduce_{tag}_ax{hex}[_kd]`) with small hand-checkable shapes and diffs against a
CPU reference on the GPU. It validates the paths the adversarial review flagged as
catchable **only** on device:

- the emitter↔host **ABI** — `shape[]` / `s0[]` / `so[]` indexing and `n_out`;
- the **keepdim ⇒ `so` by input axis** vs **collapse ⇒ `so` by kept position** split;
- **NaN propagation** in the `Max` `has`-flag fold (torch.amax semantics);
- multi-axis, middle-axis (two kept axes), and reduce-all (kept empty);
- **integer accumulation** — i32 last-axis Sum/Max fold in a `long long` accumulator
  (exact, no float rounding), including negatives.

## Run

Windows: run from a Visual Studio dev shell so `nvcc` finds `cl.exe`
(`Enter-VsDevShell`), or use an x64 Native Tools prompt.

```sh
# 1. generate the catalog .cu (includes the 6 general-path reduction cells)
cargo run -p baracuda-kernelgen --bin kernelgen -- <outdir>
# 2. copy this file next to the generated .cu, compile for sm_89, run
cp crates/baracuda-kernelgen/ondevice/reduce_validate.cu <outdir>/
nvcc -arch=sm_89 <outdir>/reduce_validate.cu -o <outdir>/reduce_validate
<outdir>/reduce_validate
```

Expected: `ALL PASSED` (bit-exact, `maxerr 0`; NaN propagated).

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **all 9 cases PASS**
(incl. the InnerContig block-per-row last-axis path and the i32 exact-int Sum/Max cases).

## Benchmark — `reduce_bench.cu`

Compares the fast-path (last-axis) vs general (outer-axis) reduction on a large
`[8192,8192]` f32 tensor against a copy-bandwidth reference. Reductions are
memory-bound, so GB/s vs. the copy peak is the figure of merit.

```sh
nvcc -O3 -arch=sm_89 reduce_bench.cu -o reduce_bench && ./reduce_bench
```

**RTX 4070 Laptop (sm_89):**

| kernel | ms | GB/s |
| --- | --- | --- |
| copy (bandwidth ref, read+write) | 2.74 | 195.8 |
| reduce **last** axis (block-per-row) | **1.18** | **227.4** |
| reduce **axis 0** (general/outer, 1 thread/col) | 2.27 | 118.3 |

**The block-per-row rewrite gave a 4.4× win on the last axis** — it was 5.15 ms /
52.2 GB/s with the old one-thread-per-row *sequential* fold (memory-**uncoalesced**,
adjacent threads a row-length apart). It now reads at ~227 GB/s — *above* the copy's
read+write ceiling because a reduction is read-only — i.e. memory-optimal. The last
axis went from the **slowest** path to the **fastest**.

**Remaining follow-up:** the general **outer-axis** path (118 GB/s) is coalesced but
still a sequential one-thread-per-column fold with low occupancy for large reduced
extents; a split-K partial-sum pass would push it toward peak. (A block-*per-column*
would be uncoalesced, so it stays one-thread-per-column.) Additive via the
`ReduceAxisClass` token — no re-key.

> This is a manual harness (needs `nvcc` + generated `.cu`), not wired into
> `cargo test`. The `#include`d kernel names track the catalog cells in
> `bin/kernelgen.rs`; update both together.
