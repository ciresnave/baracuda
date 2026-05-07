# baracuda-nvjitlink-sys

Raw FFI bindings + dynamic loader for **NVIDIA nvJitLink** — the CUDA
12+ JIT linker, used to combine multiple PTX / CUBIN / archive inputs
into a single CUBIN at runtime (the in-process equivalent of `nvcc
--device-link`).

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libnvJitLink.so` / `nvJitLink_*.dll`.

**Most users want [`baracuda-nvjitlink`]** — that crate exposes a safe
`Linker` handle with `add_data` / `add_file` / `complete` and typed
output retrieval.

## CUDA version requirement

nvJitLink ships with CUDA 12.0+. On older drivers the loader returns
`LoaderError::LibraryNotFound`; baracuda-nvjitlink surfaces this as a
typed feature-unavailability error.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nvjitlink`]: https://docs.rs/baracuda-nvjitlink
