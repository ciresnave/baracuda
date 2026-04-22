# baracuda error model

Every baracuda safe crate exposes the same shape:

```rust
pub type Error = baracuda_core::Error<LibStatus>;
pub type Result<T, E = Error> = core::result::Result<T, E>;
```

`baracuda_core::Error<S>` is a generic enum parameterized by the library's
own status enum (`cudaError_t`, `cublasStatus_t`, `cutensorStatus_t`,
`nvcompStatus_t`, `NVCVStatus`, `CUfileOpError`, etc.). It has these
variants:

```rust
pub enum Error<S: CudaStatus> {
    Status { status: S },
    Loader(baracuda_core::LoaderError),
    FeatureNotSupported { api: &'static str, since: CudaVersion },
}
```

## `Status` — the wrapped library returned non-success

All `unsafe extern "C"` calls go through a `check(rc)` helper that
converts a non-zero status into `Err(Error::Status { status })`. The
status carries the full library enum value, which implements the
[`CudaStatus`](../../crates/baracuda-types/src/status.rs) trait:

```rust
pub trait CudaStatus: Copy + core::fmt::Debug + 'static {
    fn code(self) -> i32;
    fn name(self) -> &'static str;
    fn description(self) -> &'static str;
    fn is_success(self) -> bool;
    fn library(self) -> &'static str;
}
```

So `Display` on an `Error::Status` prints the library-specific name
("`CUTENSOR_STATUS_INVALID_VALUE`", "`nvcompErrorCannotDecompress`", …)
without you having to know which library raised it.

## `Loader` — couldn't find the library

Returned when `libloading::Library::new` fails, or when the expected
symbol isn't exported (common on version mismatches, e.g. asking for
`cudaGraphConditionalHandleCreate` on CUDA 12.2). The
[`LoaderError`](../../crates/baracuda-core/src/loader.rs) variants are:

- `LibraryNotFound { library, candidates, search_paths }` — the probe
  list exhausted without a match.
- `SymbolNotFound { library, symbol }` — the library loaded but the
  function isn't exported. Almost always a version-skew issue.
- `UnsupportedPlatform { platform }` — the library has no binaries for
  the caller's OS (macOS for anything CUDA; Windows for cuFile; etc.).

Loader errors flow up through the generic error via `From<LoaderError>`.

## `FeatureNotSupported` — the installed CUDA is too old

Used when the safe wrapper checks
[`baracuda_types::CudaVersion`](../../crates/baracuda-types/src/version.rs)
before calling into an API that landed in a specific CUDA release:

```rust
if !supports(installed, Feature::GraphConditionalNodes) {
    return Err(Error::FeatureNotSupported {
        api: "cudaGraphConditionalHandleCreate",
        since: Feature::GraphConditionalNodes.required_version(),
    });
}
```

The `since` field is a `CudaVersion`, not a string, so you can compare
against the installed version in your error handler.

## Unifying across libraries

If you want one error type spanning multiple libraries (e.g. a
pipeline that calls cuBLAS → cuFFT → cuDNN), define your own enum:

```rust
#[derive(thiserror::Error, Debug)]
enum PipelineError {
    #[error(transparent)]
    Runtime(#[from] baracuda_runtime::Error),
    #[error(transparent)]
    Cublas(#[from] baracuda_cublas::Error),
    #[error(transparent)]
    Cudnn(#[from] baracuda_cudnn::Error),
}
```

…and `?` between calls. We intentionally don't ship such a super-type
so you can tailor it to your stack.

## When to panic

Never from a library call. baracuda wraps every C call in `check()`
and returns `Result`. The only panics in the safe crates are:

1. `assert!` on *programmer* bugs (slice length mismatches in
   `memcpy`-style helpers).
2. `.expect("OnceLock set or lost race")` on infallible internal
   state.

If you see a panic from a non-assert site, it's a bug — please file it.
