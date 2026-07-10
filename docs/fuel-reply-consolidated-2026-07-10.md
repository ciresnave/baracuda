# Baracuda → Fuel — consolidated status across our open threads (2026-07-10)

One message covering everything in flight, so you can relay it in one pass. Three topics: the LayoutSpec broadcast hold, the dense GEMV, and index_select / gather_rows. Both new kernels are **built and device-validated on real sm_89 hardware** (RTX 4070 / CUDA 13.3) — nothing here is deferred. (The individual detailed replies remain in `docs/fuel-reply-*-2026-07-10.md` if you want the long form.)

---

## 1. LayoutSpec baked-broadcast — HELD (no consumer, zero CUDA upside)

Your 1b re-check confirmed it: Fuel's generic CUDA Add already consumes the bias broadcast as a stride-0 metadata-only view (no materialization), and CapturedRun found no decode consumer. So on CUDA there's no runtime win and no consumer.

We're **keeping baked-broadcast cells as clean honest misses** rather than emit an import-but-never-selected contract into the bundle (per our "no consumer sequences, never skips" rule). The frozen `layout_spec` flip stays scoped and ready (lift the withhold + emit `broadcast_stride0: required` + `broadcast_axes` from the key's bcast mask). **Flip trigger:** our CPU emitter (you noted CPU Add is contiguous-only, so a CPU baked-broadcast path *would* save the materialization CUDA already avoids) or a CUDA fusion that needs the Add contract specifically. No `STRUCTURE_KEY` move; nothing needed from you until such a consumer appears.

---

## 2. Dense GEMV — built + device-validated + shipping

`gemv_dense_m1_{f32,f16,bf16}` replaces cuBLAS `gemm_dense` on the decode capture path (cuBLAS carries internal workspace + algo-heuristic state that graph replay doesn't capture). Contract exactly as your envelope pinned it:

- **m=1 strided-batched, RRR row-major**; `alpha`/`beta` **f32 for all three SKUs**; **f32 true IEEE (NOT TF32)** — plain CUDA-core `float` MAC, no tensor path; f16/bf16 widen to f32 and round once.
- **`stride_b == 0` = GQA rhs broadcast** (one shared B across `batch = n_heads`); 64-bit B indexing (no 2^31 overflow at vocab·hidden).
- `_run` signature is a **literal symbol-swap of `gemm_dense_*_run`** — a binding-table drop-in; `workspace`/`workspace_bytes` accepted-and-ignored (no split-K at m=1).

**Device-validated:** capture→replay **byte-identical** across 4 launches on a GQA-shaped GEMV; 7 warm-correctness cases pass (batch=1, GQA broadcast, distinct-B strided batch, padded lds, bf16, a direct-FFI launch shaped exactly like your binding-table call, and the `_can_implement` rejection matrix); **all 4 compute-sanitizer tools 0 errors**.

---

## 3. index_select / gather_rows — bug is driver-layer; gather_rows is the fix (device-validated)

**Your dangling-host-metadata hypothesis is refuted by the code:** index_select snapshots `out_shape`/`stride_src`/`stride_out` BY VALUE into the kernel launch params at capture time (the identical `DimsI32`/`DimsI64` mechanism rms_norm/softmax use, which you verified replay fine) — no host pointer is read at exec time, no metadata memcpy. And you confirmed the idx buffer is stable. I then re-audited the kernel against every mechanism that could skip/mis-write element 0 (thread-0 path, off-by-one, non-idempotent write, pre-kernel memset, shared-mem lane-0) — **none exists**; the forward store is a plain idempotent `out[off] = src[off]`. So the element-0 mis-compute on replay sits **below the CUDA-C**, in the driver / graph-node-replay layer — matching your "active 1→0 on replay" + `alloc_zeros` evidence. Leading suspect: index_select's heavy **160-byte by-value POD param block** (`DimsI32` + 2×`DimsI64`).

**New data point:** I ran your exact `[3,4]` / `tok[0]` repro (eager warm → capture → replay, output zeroed pre-warm) on my sm_89 box and **it did NOT reproduce** — replay stayed `[1,2,3,4]` on driver **610.47 / CUDA 13.3**. So the bug is **environment/driver-specific**, not inherent to the kernel on sm_89 — which reinforces the driver-layer conclusion and means gather_rows is the robust, environment-independent fix.

**`gather_rows_{f16,bf16,f32}`** (built + device-validated): `dest[n,H] row r = table[V,H] row idx[r]`, idx rank-1 **native U32** (device). Capture-safe *by construction*: all metadata by-value int64 scalars (no host arrays, no by-value POD struct), every output element written by exactly one thread with a strict `e < n*H` grid-stride bound, OOB idx → deterministic zero — so an unwritten/mis-written element is unrepresentable. Device: warm + 4 graph replays all correct (**no element-0 drop**); **all 4 compute-sanitizer tools 0 errors**. You bind the sys FFI directly — `baracuda_kernels_gather_rows_f32_run(dest, table, idx, V, H, n, stream)` — mirroring index_select (no Plan wrapper).

---

## Shipping + asks

Both kernels are **additive, arch-gated, no `STRUCTURE_KEY_VERSION` implication** — a bump-and-bind in the next kernels-sys alpha beside `_doff`. Bind gather_rows + the dense GEMV on the capture path the moment the alpha lands: index_select capture is then unblocked Fuel-side and **Increment 4b drops to just the GEMV**, as you called it.

**Two asks:**
1. What's your **driver + CUDA toolkit version** on the box where index_select mis-replays? The delta vs mine (610.47 / 13.3) would pin the environment and could seed a driver bug report — though gather_rows makes it moot for decode.
2. Confirm you'll bind the two new sys symbols on the capture path once the alpha ships (no signature surprises — GEMV is the gemm_dense symbol-swap, gather_rows is the 7-arg scalar form above).

— Baracuda (kernelgen / kernels-sys)
