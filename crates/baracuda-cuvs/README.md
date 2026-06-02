# baracuda-cuvs

Safe Rust wrappers for [NVIDIA cuVS](https://github.com/rapidsai/cuvs) — GPU vector search / approximate nearest neighbours (ANN), part of NVIDIA RAPIDS.

Typed handles and a build → search lifecycle over baracuda `DeviceBuffer`s,
modelled on `baracuda-cusolver` / `baracuda-cusparse`. For RAG retrieval,
memory-augmented models, and vector-DB-style production ANN at the inference
layer.

## Status

The whole API is behind the off-by-default **`cuvs`** cargo feature. cuVS
ships only with RAPIDS (`libcuvs.so` + `libraft.so`) and has **no native
Windows distribution** (Linux / WSL2 only); symbols are resolved lazily at
runtime via `libloading`, so a host without cuVS gets a graceful loader error
instead of a link failure.

```toml
[dependencies]
baracuda-cuvs = { version = "0.0.1-alpha.63", features = ["cuvs"] }
# add "half-crate" for f16 vectors
```

## Tier 1 (shipped)

* **IVF-Flat** — clustering-based index: build + search.
* **Brute-force** — exact k-NN.
* Metrics: **L2** (expanded + sqrt), **cosine**, **inner product**.
* Vector dtypes: **f32** (always), **f16** (`half-crate` feature).
* `i64` neighbour indices + `f32` distances output.

## Tier 2 (deferred)

IVF-PQ, CAGRA graph index, multi-GPU sharding, and streaming index
add/remove are not yet wrapped.

## Example

```rust,no_run
use baracuda_cuvs::{Resources, BruteForce, Metric};
use baracuda_driver::{Context, Device, DeviceBuffer};

let ctx = Context::new(&Device::get(0)?)?;
let res = Resources::new()?;

let dataset: DeviceBuffer<f32> = /* n_rows * dim, row-major, on device */;
let index = BruteForce::<f32>::build(&res, &dataset, n_rows, dim, Metric::L2Expanded)?;

let queries: DeviceBuffer<f32> = /* n_queries * dim */;
let (neighbors, distances) = index.search(&res, &queries, n_queries, 5)?;
// neighbors: DeviceBuffer<i64>  (n_queries x 5 row-major dataset indices)
// distances: DeviceBuffer<f32>  (n_queries x 5)
```

## Design notes / gotchas

* **Brute-force borrows its dataset.** cuVS's brute-force index stores a
  non-owning view of the dataset (plus precomputed norms), so
  `BruteForce<'a, T>` carries the dataset borrow — the borrow checker
  guarantees the buffer outlives the index. IVF-Flat copies the vectors in
  during build (`add_data_on_build = true`), so `IvfFlat<T>` owns its data.
* **Synchronous by default.** `build` and `search` synchronize the bound
  stream before returning, so outputs are immediately host-readable. Bind a
  stream with `Resources::set_stream` to control where work lands.
* **`cuvsError_t::SUCCESS` is `1`, not `0`** (handled internally). On failure
  use `baracuda_cuvs::last_error_text()` to recover cuVS's message.
* cuVS exchanges data as DLPack `DLManagedTensor` views — constructed
  internally; callers only ever pass `DeviceBuffer`s.

## Installing cuVS

```bash
conda install -c rapidsai -c conda-forge -c nvidia cuvs cuda-version=12.x
# or
pip install cuvs-cu12
```

Run the hardware tests once installed:

```bash
cargo test -p baracuda-cuvs --features cuvs -- --ignored
```

## License

`MIT OR Apache-2.0`. cuVS itself is Apache-2.0 (NVIDIA RAPIDS).
