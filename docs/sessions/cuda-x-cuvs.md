# Session prompt — Add `baracuda-cuvs` wrapper for cuVS

Working on baracuda at `c:\Users\cires\OneDrive\Documents\projects\baracuda`.
Per the Phase 65 CUDA-X audit, cuVS provides GPU vector search (ANN).
Relevant if Fuel ever adds RAG / embedding-store / vector-database
workflows. Other library-addition sessions may be running in parallel.

## Context

[cuVS](https://github.com/rapidsai/cuvs) is NVIDIA's GPU vector
search library (part of RAPIDS, MIT/BSD-licensed). Provides:

- **IVF-Flat** index (clustering-based exact search)
- **IVF-PQ** (product quantization for compressed indexes)
- **CAGRA** (NVIDIA's graph-based ANN algorithm, sm_80+)
- **Brute-force** k-NN (exact, for small databases)
- L2 / cosine / inner-product distance metrics
- Build + search in a single API

Use cases relevant to Fuel:
- RAG retrieval (find top-k similar embeddings from a vector store)
- Memory-augmented models (e.g. Memory-LLM, RWKV with external memory)
- Vector-DB-style production ANN at the inference layer

## Scope

**Crates to create:**

1. `crates/baracuda-cuvs-sys/` — `extern "C"` FFI declarations. cuVS
   has a C API (cuvs/cagra_c.h, cuvs/ivf_flat_c.h, etc.) — use that,
   not the C++ headers.
2. `crates/baracuda-cuvs/` — safe wrapper modeled on
   `baracuda-cusolver` / `baracuda-cusparse` patterns (typed handles,
   index build → search lifecycle, workspace management).

## Linking

cuVS ships as `libcuvs.so` + `libraft.so` (cuVS depends on RAFT, also
RAPIDS). Both should be available via the RAPIDS install. Lazy
`libloading` (matches baracuda-nccl pattern).

## Tier 1 deliverables

1. IVF-Flat index: build + search.
2. Brute-force k-NN search.
3. L2 + cosine distance metrics.
4. f32 + f16 vector dtype support.
5. Cargo feature `cuvs` on `-sys` and the safe wrapper.
6. Smoke tests using small embedded vectors (e.g. 100 random
   128-dim f32 vectors, k=5 nearest).

## Tier 2 deferrable

- IVF-PQ (product quantization — adds memory savings for huge
  databases)
- CAGRA graph-based index (faster but more complex)
- Inner-product (Hadamard) distance metric
- Multi-GPU index sharding
- Streaming index updates (add/remove vectors at runtime)

## Reference patterns

cuVS has clean C-API design similar to cuSOLVER. The build/search
lifecycle is:

```
cuvsResources_t res;
cuvsResourcesCreate(&res);
cuvsIvfFlatIndexParams_t params;
cuvsIvfFlatIndex_t index;
cuvsIvfFlatBuild(res, &params, &dataset, &index);
// ... search ...
cuvsIvfFlatSearch(res, &search_params, index, &queries, &neighbors, &distances);
cuvsIvfFlatIndexDestroy(index);
cuvsResourcesDestroy(res);
```

Map to a typed Rust Plan:

```rust
let plan = IvfFlatPlan::<f32>::build(&ctx, &dataset, IvfFlatBuildParams::default())?;
let (neighbors, distances) = plan.search(&queries, k, SearchParams::default())?;
```

## Out of scope

- Don't depend on / wrap RAFT directly. cuVS exposes what callers
  need; RAFT is the implementation detail.
- Don't try to integrate with a vector database library on Rust side.
  Pure low-level GPU ANN.
- Don't add CPU fallback paths. GPU-only.

## Coordination

- Working directory: `c:\Users\cires\OneDrive\Documents\projects\baracuda`
- Branch: `phase71-cuvs`
- No version bump, no publish.
- Commit on branch + push + stop.

## Stop conditions

- If cuVS is not installed on the dev machine + install is non-trivial:
  ship crates with `#[ignore]`-gated tests. Document install path
  (typically via `conda install -c rapidsai cuvs` or pip).
- If cuVS C API has lifetime / ownership quirks that don't map cleanly
  to Rust (e.g. mandatory shared state across calls): document the
  issue + raise it in the wrapper docstring + ship a best-effort safe
  wrapper.
- If the entire crate pair already exists: stop, report.

## Memory file

Write `project_phase71_complete.md`.
