# On-device numeric validation — general reduction path (item 03)

`reduce_validate.cu` launches the **generated** general-path reduction kernels
(`_reduce_{tag}_ax{hex}[_kd]`) with small hand-checkable shapes and diffs against a
CPU reference on the GPU. It validates the paths the adversarial review flagged as
catchable **only** on device:

- the emitter↔host **ABI** — `shape[]` / `s0[]` / `so[]` indexing and `n_out`;
- the **keepdim ⇒ `so` by input axis** vs **collapse ⇒ `so` by kept position** split;
- **NaN propagation** in the `Max` `has`-flag fold (torch.amax semantics);
- multi-axis, middle-axis (two kept axes), and reduce-all (kept empty).

## Run

Windows: run from a Visual Studio dev shell so `nvcc` finds `cl.exe`
(`Enter-VsDevShell`), or use an x64 Native Tools prompt.

```
# 1. generate the catalog .cu (includes the 6 general-path reduction cells)
cargo run -p baracuda-kernelgen --bin kernelgen -- <outdir>
# 2. copy this file next to the generated .cu, compile for sm_89, run
cp crates/baracuda-kernelgen/ondevice/reduce_validate.cu <outdir>/
nvcc -arch=sm_89 <outdir>/reduce_validate.cu -o <outdir>/reduce_validate
<outdir>/reduce_validate
```

Expected: `ALL PASSED` (bit-exact, `maxerr 0`; NaN propagated).

**Last run:** RTX 4070 Laptop (sm_89), CUDA 13.3 / nvcc 13.3 — **all 6 cases PASS**.

## Benchmark — `reduce_bench.cu`

Compares the fast-path (last-axis) vs general (outer-axis) reduction on a large
`[8192,8192]` f32 tensor against a copy-bandwidth reference. Reductions are
memory-bound, so GB/s vs. the copy peak is the figure of merit.

```
nvcc -O3 -arch=sm_89 reduce_bench.cu -o reduce_bench && ./reduce_bench
```

**RTX 4070 Laptop (sm_89):**

| kernel | ms | GB/s |
| --- | --- | --- |
| copy (bandwidth ref, read+write) | 2.74 | 195.7 |
| reduce **last** axis (fast path, 1 thread/row) | 5.15 | 52.2 |
| reduce **axis 0** (general/outer, 1 thread/col) | 2.68 | 100.2 |

**Finding:** the "fast path" (contiguous last-axis, `base=o*k`) is **1.9× SLOWER**
than the general outer-axis path — adjacent threads read different rows
(**uncoalesced**, 32 KB apart) vs. adjacent columns (**coalesced**). Both sit well
below the ~196 GB/s copy ceiling: the one-thread-per-output sequential fold
under-utilizes memory. Perf follow-ups: (1) route the last-axis reduction to a
**warp/block-cooperative** kernel (coalesced reads + shuffle reduce — the
`Access::RowReduce` emit already does this), and (2) the block-parallel outer-axis
kernel the design doc §9/§10 reserves. The `ReduceAxisClass` schedule token already
exists so these land as additive drop-ins, not a re-key.

> This is a manual harness (needs `nvcc` + generated `.cu`), not wired into
> `cargo test`. The `#include`d kernel names track the catalog cells in
> `bin/kernelgen.rs`; update both together.
