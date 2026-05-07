# baracuda-cudf

Safe Rust wrappers for **RAPIDS cuDF** — GPU DataFrames, accessed
through the emerging `libcudf_c` C ABI.

## Coverage scope

cuDF is a large C++ library; only a subset is exposed through the
official C ABI (`libcudf_c`), and only what's exposed there is wrappable
without writing C++ shims. baracuda-cudf tracks `libcudf_c` as it grows.

**Today's surface:**

- **`Column`** — RAII handle around a typed cuDF column.
- **`Table`** — collection of columns, the cuDF analog of a Pandas
  DataFrame.
- **`TypeId`** — column dtype tag (Int32, Float64, String, Timestamp,
  Decimal, ...).
- **I/O**:
  - CSV reader / writer.
  - Parquet reader / writer.

**For operations not yet exposed in `libcudf_c`** (groupbys, joins, most
aggregations, custom UDFs), you'll need to use Python's `cudf` or write
your own C++ shim — that gap is upstream's, not ours, and we extend
this crate as RAPIDS grows the C ABI.

## Platform support

RAPIDS cuDF is **Linux-only**. The loader fails fast with a clear
error on Windows / macOS.

Pairs with [`baracuda-cudf-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cudf-sys`]: https://docs.rs/baracuda-cudf-sys
