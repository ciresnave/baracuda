# baracuda-nvjitlink

Safe Rust wrappers for **NVIDIA nvJitLink** — the CUDA 12+ JIT linker.
Combines multiple PTX / CUBIN / library inputs into a single CUBIN at
runtime, the in-process equivalent of `nvcc --device-link`.

Useful when you have separate PTX modules (e.g. produced by independent
NVRTC compiles) that need to share `__device__` functions, or when a
PTX module references a CUDA library archive (cuBLASLt, CUTLASS, etc.).

```rust,no_run
use baracuda_nvjitlink::Linker;

# fn demo() -> Result<(), Box<dyn std::error::Error>> {
let mut linker = Linker::new(&["-arch=sm_80"])?;
linker.add_data(b"...PTX bytes for module A...\0", "modA.ptx")?;
linker.add_data(b"...PTX bytes for module B...\0", "modB.ptx")?;
let cubin = linker.complete()?;
// Hand `cubin` to baracuda_driver::Module::load_raw.
# Ok(()) }
```

## Coverage

- `Linker` create with options.
- `add_data` / `add_file` for every input kind nvJitLink accepts (PTX,
  CUBIN, fatbin, host object, archive).
- `complete` to produce the linked CUBIN.
- `info_log` / `error_log` retrieval.

## CUDA version requirement

nvJitLink is part of the **CUDA 12.0+** toolkit. On older drivers the
loader returns `LoaderError::LibraryNotFound`; this crate surfaces that
as a typed feature-unavailability error you can match on.

Pairs with [`baracuda-nvjitlink-sys`] for the raw FFI surface.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nvjitlink-sys`]: https://docs.rs/baracuda-nvjitlink-sys
