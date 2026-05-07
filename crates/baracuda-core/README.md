# baracuda-core

Runtime machinery shared across baracuda crates: dynamic library loader,
error plumbing, OS / library-search-path detection, and process-wide
stream-mode selection.

You normally don't depend on this crate directly — it's a transitive
dependency of every `baracuda-*-sys` and safe-wrapper crate. The bits
you might care about as an application author are:

- **`stream_mode::init` / `stream_mode::current`** — choose how CUDA's
  "default stream" behaves process-wide (legacy or per-thread default
  stream). Call this once at startup before any baracuda CUDA call if
  you don't want the default.
- **`BaracudaError`** — library-erased error type that any per-library
  `Error<S>` can convert into via `From`. Useful as the error type for
  application-level `Result`s that span multiple baracuda libraries.
- **`LoaderError`** — surfaced when a `.so` / `.dll` can't be opened or
  a symbol is missing. Includes the searched paths and OS-level error
  detail.

## What's here

- `error` — `Error<S>` (parameterized by the per-library status type),
  `LoaderError`, and `BaracudaError`.
- `loader` — `Library` + `Symbol` wrappers around [`libloading`] with
  baracuda-specific search-path semantics.
- `platform` — OS detection (`OsFamily`) and per-library default
  search-path lists.
- `stream_mode` — process-wide default-stream-semantics selector,
  consulted by `Stream::default()` impls in the safe wrappers.

## Why this crate exists

baracuda's whole loading model — open every CUDA library at runtime,
resolve symbols lazily, never `#[link]` against a CUDA `.so` / `.dll` at
build time — needs a single source of truth for *how* libraries are
found and opened. That's this crate. It also factors out the
boilerplate of "produce a typed Error from a status code" so every
`-sys` crate doesn't re-implement it.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`libloading`]: https://docs.rs/libloading
