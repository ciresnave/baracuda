# baracuda-cudf-sys

Raw FFI bindings + dynamic loader for **RAPIDS cuDF** — GPU DataFrames
— through its emerging `libcudf_c` C ABI.

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libcudf.so`.

**Most users want [`baracuda-cudf`]** — that crate exposes typed
`Column`, `Table`, `TypeId`, plus the I/O entry points exposed by
`libcudf_c` (CSV, Parquet).

## Coverage scope

cuDF is a large C++ library; only a subset is exposed through the
official C ABI (`libcudf_c`), and only what's exposed there is wrappable
without writing C++ shims. baracuda-cudf-sys tracks `libcudf_c` as it
grows. Major bits today:

- Column / Table types and lifecycle.
- TypeId, scalar.
- I/O: CSV reader / writer, Parquet reader / writer.
- Selected compute kernels as RAPIDS adds C ABI exposure.

For operations not yet in `libcudf_c`, you'll need to use Python's
`cudf` or write your own C++ shim — that gap is upstream's, not ours.

## Platform support

RAPIDS cuDF is **Linux-only**. The loader fails fast with a clear
error on Windows / macOS.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cudf`]: https://docs.rs/baracuda-cudf
