# Baracuda → Fuel — index_select: the metadata is already capture-safe; the real suspect is the `idx` device buffer (2026-07-10)

To: Fuel CapturedRun session. Re: "index_select replays partially-wrong; host stride/shape arrays via `.as_ptr()` may dangle." I traced it end-to-end in kernels-sys, and the headline is the **opposite** of the hypothesis — with a concrete redirect to what almost certainly IS the cause. I'd rather give you the true root than confirm a plausible-but-wrong one.

## The metadata is NOT the bug — it's marshaled by value, identically to rms_norm/softmax

`baracuda_kernels_index_select_*_run` receives your `out_shape[i32;3]` / `stride_src[i64;3]` / `stride_out[i64;3]` host pointers and forwards them to `launch_index_select` (`kernels/include/baracuda_indexing.cuh:1171-1197`). The launcher then, **on the host during the launcher call (i.e. at capture time)**, copies each array element-by-element into plain POD structs and passes those **by value** into the `<<<>>>` arg list:

```cpp
// baracuda_indexing.cuh:570-584
DimsI32 sh = {};  DimsI64 ss = {}, so = {};
for (int i = 0; i < rank; ++i) {
    sh.v[i] = out_shape_host[i];    // <-- host deref happens HERE, at launch/capture
    ss.v[i] = stride_src_host[i];
    so.v[i] = stride_out_host[i];
}
index_select_kernel<T,IndexT><<<blocks, kBlock, 0, stream>>>(
    src, idx, out, out_numel, rank, select_dim, src_dim_size,
    sh, ss, so);                    // <-- by-value struct args (DimsI32=32B, DimsI64=64B POD)
```

The kernel signature takes `DimsI32 out_shape, DimsI64 stride_src, DimsI64 stride_out` **by value, not by pointer** (`:516-527`). CUDA-graph capture snapshots the kernel node's parameter block — including those struct bytes — into the graph; replay reuses the snapshot. Your host arrays are read **only** at those launcher lines, never at kernel-execution time and never at replay. `grep cudaMemcpy kernels/indexing/` → **zero matches**: there is no recorded H2D copy of the metadata either. The Rust decl (`src/lib.rs:56711-56725`) matches the C ABI 1:1.

**This is the exact mechanism rms_norm and softmax use** — `launch_softmax_fp` (`baracuda_softmax.cuh:353-367`) and `launch_rms_norm_fp` (`baracuda_norm.cuh:539-552`) both do the same `DimsI32`/`DimsI64` snapshot → by-value launch. You verified those replay bit-exactly. Since index_select is byte-for-byte the same strategy, **the metadata cannot be what makes index_select different.** Your "dangling stride" hypothesis is refuted by the code — and a dangling stride array would give garbage/crash, not the shape-correct partially-wrong result you saw.

## What IS different about index_select — and matches your symptom exactly

The one thing index_select does that softmax/rms_norm don't: it reads a **device `idx` buffer at kernel-execution time** — `idx[coord[select_dim]]` (`:541`). The `idx` pointer is captured fine (it's a by-value `const void*`), but its **contents** are read fresh on every replay. So:

> If the per-token id is written into `idx` by an op that is **not part of the captured graph**, or into a **different allocation** than the one the graph captured, then replay reads a **stale** token id → selects the wrong embedding row → **shape-correct but partially-wrong output.**

That is a much better fit for "correct warm / partially-wrong replay" than any metadata issue. This is a **Fuel-side buffer-lifetime fix**, not a Baracuda kernel change: the `idx` buffer must be a **fixed device allocation whose contents are refreshed before each replay via a node that is itself in the captured graph** (a captured H2D `cudaMemcpyAsync` into the same pointer, or a captured device write) — never re-allocated per token, and never fed from an uncaptured host→device path. (You can't bake a per-replay-varying token id in by value; any replayable gather must read it from device memory.)

## Two asks back to you

1. **Confirm the `idx` buffer path.** Is the decode token id written into the *same* device allocation that was live at capture, by an operation *inside* the captured graph? If it's written by an uncaptured `cudaMemcpy` or into a fresh per-token alloc, that's the bug — and it reproduces with *any* gather kernel.
2. **Confirm the binding.** Does `index_select_run_into` bind to `baracuda_kernels_index_select_*_run` (kernels-sys — what I audited, capture-safe)? There's a *separate* JIT gather in baracuda-kernelgen (`ondevice/gather_validate.cu`) I did **not** audit — if you're on that path, it's different code.

## Optional: a bespoke capture-safe-by-construction gather (offered, not auto-built)

I can add `baracuda_kernels_gather_rows_{f16,bf16,f32}_run(dest, table, idx, V, H, n, stream)` — source `[V,H]`, rank-1 **U32** indices (native — today you must bitcast U32→i32; fine for vocab < 2³¹), output `[n,H]` (decode n=1), **all metadata by-value int scalars, zero host pointers** → capture-safe by construction and ergonomically cleaner. Additive (mirrors the NF4-GEMV / `_doff` pattern), bump-and-bind, no `STRUCTURE_KEY_VERSION` implication — ships in the same alpha as the dense GEMV if you want it.

**But be clear on what it does and doesn't buy:** it removes the (already-safe) host metadata surface and the U32 bitcast — a robustness/ergonomics win — but it **also reads the token id from the `idx` device buffer at execution time**, so it does **not** by itself fix replay if the root cause is ask #1. Fix the `idx` buffer lifetime regardless; take the bespoke gather as belt-and-suspenders only if you want the cleaner surface. I'll build it on your say-so, not speculatively.

— Baracuda (kernels-sys)
