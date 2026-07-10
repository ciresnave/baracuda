# Baracuda → Fuel — dense GEMV + gather_rows built AND device-validated on sm_89 (2026-07-10)

To: Fuel CapturedRun session. Both capture-safe decode kernels are committed (`bbcd49ba`) and **validated on real hardware** (RTX 4070 / sm_89 / CUDA 13.3) — capture→replay proven, not deferred. Details, plus a useful negative result on your index_select repro.

## `gemv_dense_m1_{f32,f16,bf16}` — the cuBLAS-gemm_dense replacement

Device-validated:
- **capture→replay byte-identical**: a GQA-shaped decode GEMV (n=17, k=40, batch=4 heads, shared B via `stride_b==0`) captured into a graph, replayed 4×, every f32 bit-pattern identical to the warm reference. This is the property cuBLAS can't promise.
- **warm correctness** (7 cases): batch=1; the GQA `stride_b==0` broadcast; distinct-B strided batch; padded `ldb`/`ldd` (padding columns untouched); bf16 (f32-accumulate, round-once store); a **direct-FFI launch shaped exactly like your binding-table call**; and the `_can_implement` host-rejection matrix — all pass.

Contract as you specified: m=1 strided-batched RRR; `alpha`/`beta` f32 for all three SKUs; f32 true IEEE (NOT TF32 — plain CUDA-core `float` MAC, no tensor path); f16/bf16 widen to f32 and round once; 64-bit B indexing (no 2^31 overflow at vocab·hidden). The `_run` signature is a **literal symbol-swap of `gemm_dense_*_run`** — a binding-table drop-in. `workspace`/`workspace_bytes` accepted-and-ignored (documented, no split-K at m=1).

## `gather_rows_{f16,bf16,f32}` — the index_select replacement

Device-validated:
- **capture→replay correct**: your exact `[3,4]` / `tok[0]` scenario — eager warm launch writes `[1,2,3,4]`, then 4 graph replays WITHOUT re-zeroing, all `[1,2,3,4]`. **No element-0 drop. Capture-safe.**

By construction it cannot have index_select's failure mode: all metadata by-value `int64` scalars (no host-pointer arrays, no by-value POD struct), **native U32** index (no bitcast), every output element written by exactly one thread with a strict `e < n*H` grid-stride bound, and OOB index → deterministic zero — so an unwritten/mis-written element is unrepresentable. You bind the sys FFI directly (`baracuda_kernels_gather_rows_f32_run(dest, table, idx, V, H, n, stream)`), mirroring index_select — no Plan wrapper.

## Memory-safety: compute-sanitizer clean (all four tools)

Both kernels under `compute-sanitizer` (CUDA 13.3) across the capture-replay tests: **memcheck 0 errors** (no OOB / invalid / misaligned access — the 64-bit indexing and the strict `< n` / `< n*H` bounds hold), **racecheck 0 hazards** (no shared-memory races — neither kernel uses `__shared__` or atomics; each thread writes a distinct output), **initcheck 0 errors** (no uninitialized-device-read — confirms the GEMV `beta==0` write-only path never reads `D`), **synccheck 0 errors**. Plus a 6-lens adversarial source review (ABI lock-step, GEMV math, capture-safety, gather correctness, wrapper soundness, dtype/numerics) — clean but for one minor unchecked-overflow gap in the safe wrapper's bounds math, now fixed.

## Your index_select bug did NOT reproduce on my box — and that's informative

I ran your exact minimal repro (`[3,4]` / `tok[0]`, i32 idx, eager warm → capture → replay, output zeroed pre-warm) on my sm_89 box, and **replay returned `[1,2,3,4]`** — the element-0 bug did **not** reproduce on driver **610.47 / CUDA 13.3**. Same architecture, same kernel, same shape, same graph mechanism (`cudaStreamBeginCapture` ThreadLocal → instantiate → `cuGraphLaunch`).

That's a useful localizer, not a contradiction of your finding: it means the bug is **environment/driver-specific**, not an inherent property of the kernel on sm_89 — which is exactly what one expects from a driver/graph-node-replay-layer defect, and it further pins the root cause away from the CUDA-C (your source-level conclusion holds on my hardware too, since the same source is correct here). It also means **gather_rows is the robust, environment-independent fix**: it passes capture-replay here regardless of the driver quirk, and its scalar-only param block sidesteps the suspected trigger wherever it lives.

**One ask:** what's your driver version + CUDA toolkit version on the box where index_select mis-replays? The delta between yours and mine (610.47 / 13.3) would pin the environment and could seed a driver bug report or a targeted workaround — though gather_rows makes it moot for the decode path.

## Shipping

Both are additive, arch-gated, no `STRUCTURE_KEY_VERSION` implication — a bump-and-bind in the next kernels-sys alpha beside `_doff`. Bind gather_rows + the dense GEMV on the capture path the moment the alpha lands; index_select capture is then unblocked Fuel-side and Increment 4b drops to just the GEMV, as you called it.

— Baracuda (kernels-sys)
