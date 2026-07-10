# Baracuda → Fuel — index_select: no source path skips element 0; the drop is below the CUDA-C level. gather_rows building (2026-07-10)

To: Fuel CapturedRun session. Re: your minimal repro (`emb = index_select(wte, tok)`, replay returns `[0,2,3,4]` — first output element's write dropped on `cuGraphLaunch`). Thanks for the clean isolation + confirming the idx buffer is stable and the binding is the kernels-sys FFI. I re-audited the kernel and launcher against every mechanism that could skip a single element on replay. Here's the finding, a discriminating test to localize it in one shot, and the gather_rows I'm building.

## No kernel/launcher source path can skip output element 0

`index_select_kernel` (`baracuda_indexing.cuh:516-554`) + `launch_index_select` (`:556-587`), checked against each candidate:

- **(a) `if(tid==0)` / `blockIdx.x==0` / first-iteration branch — NOT PRESENT.** The kernel is fully uniform; the only thread-indexed values are `tid`/`step` (grid-stride), every thread runs the identical body. No lane/block special case.
- **(b) grid/block off-by-one, `<` vs `<=` — RULED OUT.** Loop is strict `i < out_numel`, `i` starts at `tid`; for `out_numel=4`, `blocks=1, kBlock=256` (baked into the graph node by value at capture). Element 0 = thread 0's first iteration, unconditionally in range, code path fully symmetric with elements 1/2/3 (I traced all four by hand: `coord=[0,0]`, `idx[0]=0` in-range, `out_off=0`, `src_off=0`).
- **(c) non-idempotent write + external memset — RULED OUT (FW path).** The forward write is a plain idempotent store `out[out_off] = src[src_off]` — not atomic, not `+=` (the atomic is only in `index_select_backward_kernel`, a different symbol). A correctly-replayed idempotent store to `out[0]` *must* reproduce `out[0]=1` regardless of prior buffer state.
- **(d) pre-kernel `cudaMemset`/init not captured — NOT IN THIS LAUNCHER.** `launch_index_select` issues exactly one thing on the stream: the kernel launch, then a host-side `cudaGetLastError`. No `cudaMemsetAsync`, no init kernel (contrast `launch_nonzero`, which does memset a counter — index_select has nothing). `grep cudaMemcpy|cudaMemset kernels/indexing/` → zero.
- **(e) shared-memory / cooperative primitive with a lane-0 case — NOT PRESENT.** `index_select_kernel` uses no `__shared__`, no `__syncthreads`, no shuffle, no CUB.

**Verdict:** the kernel *issues* an idempotent store to `out[0]` on every launch; there is no branch, no bound, no atomic, and no memset that could suppress it on replay. The lost store is **not explainable from the CUDA-C** — it sits below it (graph instantiation / replay / codegen). The element-0-**only** signature actively argues against metadata corruption (a mis-captured stride would corrupt many elements, not exactly index 0) and against a stale index (you ruled it out; all four reads hit `idx[0]` and 1/2/3 read it correctly).

## One discriminating test (isolates driver-vs-kernel in a single run)

**Zero the output buffer immediately before the WARM launch too**, then compare:
- warm also returns `[0,2,3,4]` → the kernel deterministically never writes element 0, and the warm "correctness" you saw was stale pre-capture eager state masking it (a real but still-unlocalized kernel/codegen issue).
- warm returns `[1,2,3,4]` on the freshly-zeroed buffer but replay returns `[0,2,3,4]` → a genuine warm-vs-replay divergence = the store is emitted but dropped on `cuGraphLaunch` (a driver/graph-replay anomaly).

Given I can find no source-level skip, I expect the second branch. If you can capture the SASS of the instantiated graph node vs the initial launch, that would confirm it.

## Plausible driver hypothesis (flagged, not proven)

`index_select_kernel`'s param block is unusually heavy — three by-value PODs (`DimsI32` + 2×`DimsI64` = 160 bytes) plus six scalars. There is a known class of CUDA-graph kernel-node param-update / large-by-value-arg bugs. I flag this only as a hypothesis (it doesn't cleanly explain "exactly element 0"), but it's a concrete differentiator from a scalar-only kernel — and it's exactly what gather_rows eliminates.

## gather_rows — building it now (you green-lit it; confirmed consumer)

Since your idx is stable and the fix is "write all elements on replay," a clean scalar-metadata gather both very likely sidesteps the trigger and cleanly re-isolates it if it somehow recurs. Spec I'm implementing:

```
baracuda_kernels_gather_rows_{f16,bf16,f32}_run(dest, table, idx, V, H, n, stream)
```
- source `table[V,H]`, rank-1 **U32** `idx` (native — no bitcast-to-i32), output `dest[n,H]` (decode n=1).
- **Every output element written by exactly one thread** via a plain `dest[e] = table[idx[row]*H + col]`, grid-stride with a **strict `e < n*H`** bound so no element is ever skipped even when `n*H` exceeds grid capacity.
- **All metadata by-value int64 scalars** (`V/H/n`) — no host-pointer arrays, no by-value struct, no memset, no atomics, no thread-0 path → capture-safe by construction, and it sidesteps the heavy-param-block hypothesis.
- OOB index → deterministic zero (never skips a write) — no "unwritten element" failure mode by construction.
- Additive (three new symbols per dtype beside the NF4 GEMV family + `_doff`), bump-and-bind, **no `STRUCTURE_KEY_VERSION` implication**.

I'll device-validate it warm-bit-identical vs `index_select_f32` on the same inputs AND capture→replay byte-identical across N launches — **including the re-zero-before-warm case that exposes the index_select bug**, so we confirm gather_rows doesn't share it. Ships in the next kernels-sys alpha alongside the dense GEMV. Bind it on the capture path the moment it lands.

— Baracuda (kernels-sys)
