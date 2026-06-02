# baracuda-cuvs-sys

Raw FFI bindings and a lazy `libloading` loader for [NVIDIA cuVS](https://github.com/rapidsai/cuvs) — GPU vector search / approximate nearest neighbours (ANN), part of NVIDIA RAPIDS.

This is the low-level `-sys` crate. For a safe, typed API see [`baracuda-cuvs`](../baracuda-cuvs).

## Status

Everything lives behind the off-by-default **`cuvs`** cargo feature. cuVS ships
only as part of RAPIDS (`libcuvs.so` + `libraft.so`) and has **no native
Windows distribution** (Linux / WSL2 only). The loader resolves symbols
lazily at runtime via `libloading`, so enabling the feature adds no link-time
dependency — [`cuvs()`] returns `LoaderError::LibraryNotFound` on hosts
without RAPIDS.

```toml
[dependencies]
baracuda-cuvs-sys = { version = "0.0.1-alpha.63", features = ["cuvs"] }
```

## Coverage (Tier 1)

* Resources lifecycle — `cuvsResourcesCreate` / `Destroy`, `cuvsStreamSet` / `Get` / `Sync`, `cuvsGetLastErrorText`.
* IVF-Flat — params create/destroy, index create/destroy, `cuvsIvfFlatBuild`, `cuvsIvfFlatSearch`.
* Brute-force — index create/destroy, `cuvsBruteForceBuild`, `cuvsBruteForceSearch`.
* DLPack `DLManagedTensor` / `DLTensor` / `DLDevice` / `DLDataType` structs (stable ABI) for handing device buffers to cuVS.
* Full `cuvsDistanceType` enum + `cuvsFilter` prefilter.

IVF-PQ and CAGRA are deferred (Tier 2).

## Gotchas

* **`cuvsError_t::SUCCESS` is `1`, not `0`** (`CUVS_ERROR == 0`). The status
  trait encodes this.
* cuVS consumes datasets / queries / outputs as DLPack tensor *pointers*, not
  bare device pointers. Input tensors are non-owning (`deleter: None`).
* `cuvsResources_t` is a `uintptr_t` (modelled as `usize`), not a pointer.

## Installing cuVS

```bash
# conda (recommended)
conda install -c rapidsai -c conda-forge -c nvidia cuvs cuda-version=12.x
# or pip
pip install cuvs-cu12
```

Point the loader at a non-standard install with `CUVS_ROOT` / `RAFT_ROOT`, or
ensure `libcuvs.so` is on `LD_LIBRARY_PATH`.

## License

`MIT OR Apache-2.0` (the bindings, matching the workspace). cuVS itself is
Apache-2.0 (NVIDIA RAPIDS).
