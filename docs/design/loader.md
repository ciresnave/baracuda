# baracuda dynamic loader

baracuda never `#[link]`s against any NVIDIA shared library. Every
symbol is resolved at runtime via `libloading`. This means:

- `cargo build` succeeds on machines without a CUDA Toolkit installed.
- A single binary runs against any version of the driver / toolkit /
  optional libraries — we probe at first use.
- Optional libraries (cuTENSOR, nvCOMP, CV-CUDA, cuDNN, …) only need
  to be present on machines that actually call them.

## Per-library loader struct

Each `-sys` crate exposes a handle struct (`Runtime`, `Cublas`,
`Cutensor`, `Nvcomp`, `Cvcuda`, `Cufile`, …) that wraps a
`libloading::Library` plus a set of `OnceLock<FN_TYPE>` slots for each
exported function:

```rust
runtime_fns! {
    fn cuda_malloc as "cudaMalloc": PFN_cudaMalloc;
    fn cuda_free as "cudaFree": PFN_cudaFree;
    // ...
}
```

The `runtime_fns!` macro expands to a struct with one `OnceLock` per
entry, plus a method that:

1. Returns the cached pointer if already resolved.
2. Otherwise calls `lib.raw_symbol(sym_name)` via `libloading`, stores
   the result, and returns it.

So first-call latency pays the `dlsym` cost; subsequent calls are a
single-word load.

## Library probing order

### Driver API (`libcuda.so.1` / `nvcuda.dll`)

1. Try known filenames in `LD_LIBRARY_PATH` / `PATH`.
2. Linux: `/usr/local/cuda/lib64`, `/usr/local/cuda/compat`,
   `/usr/lib/x86_64-linux-gnu`, `/usr/lib/wsl/lib` (WSL2 driver stub).
3. Windows: `%CUDA_PATH%\bin`, `%CUDA_PATH_V12_*%\bin`.

### Runtime API + math/ML libraries

Same as Driver, but with versioned filename candidates:
`libcublas.so.13`, `libcublas.so.12`, `libcublas.so.11`, `libcublas.so`.
The probe tries newest-first, so a machine with both CUDA 12 and CUDA
13 binaries picks up 13.

### Optional libraries with standalone installers

cuTENSOR and nvCOMP ship in their own install paths on Windows:

- cuTENSOR: `C:\Program Files\NVIDIA cuTENSOR\<ver>\{bin,lib}\<cuda>\`
- nvCOMP: `C:\Program Files\NVIDIA nvCOMP\<ver>\bin\<cuda>\`

The `-sys` loader crates have an `extra_dirs()` helper that enumerates
these paths via `std::fs::read_dir` + CUDA-major sub-folders
(`bin\13`, `bin\12`, …). If the default probe fails, the extras are
searched before giving up.

### Linux-only: cuFile

`libcufile.so.0`. No Windows/macOS binaries exist. On non-Linux the
loader returns `LoaderError::UnsupportedPlatform { platform }` —
every safe API propagates this as an `Error::Loader` variant.

## `cuGetProcAddress_v2` (Driver API only)

The Driver API supports version-aware symbol resolution via
`cuGetProcAddress_v2(name, &fptr, version, flags)`. Passing
`CUDA_VERSION=11040` on a CUDA 13.x driver gives you the CUDA 11.4
semantics (e.g. pre-per-thread-default-stream behavior) without
introducing ABI breaks.

baracuda-cuda-sys uses `cuGetProcAddress_v2` for **every** post-bootstrap
Driver symbol after resolving it once via `dlsym`. The version is
pinned to the baracuda CUDA floor (11.4) unless you bump it via
`baracuda_core::stream_mode::init`. The Runtime loader uses plain
`dlsym` since `libcudart` doesn't expose a version-aware resolver.

## Per-thread-default-stream

CUDA added a `_ptsz` variant of most stream-accepting functions in
CUDA 7 (per-thread-default-stream). If the calling thread uses
`cudaStreamPerThread`, the runtime rewrites symbol resolution to
point at the `_ptsz` variants.

baracuda exposes this as a process-global toggle:

```rust
baracuda_core::stream_mode::init(StreamMode::PerThread); // default
baracuda_core::stream_mode::init(StreamMode::Legacy);    // before first symbol resolution
```

**Must be called before any CUDA symbol is resolved**, otherwise the
loader caches the wrong variant. Re-init with a different mode
panics.
