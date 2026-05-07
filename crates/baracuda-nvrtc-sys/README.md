# baracuda-nvrtc-sys

Raw FFI bindings + dynamic loader for **NVIDIA NVRTC** — runtime CUDA
C++ compilation (`.cu` source → PTX, in-process, no `nvcc` binary
required at runtime).

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libnvrtc.so` / `nvrtc64_*.dll`.

**Most users want [`baracuda-nvrtc`]** — that crate exposes a safe
`Program` builder, compile-options helper, include-path registration,
and error-log retrieval.

For *build-time* kernel compilation via `nvcc` (incremental, parallel,
multi-arch fat binaries, CUTLASS dependency management), see
[`baracuda-forge`] instead.

## What's exposed

- `nvrtcProgram` create / compile / destroy.
- PTX output retrieval (`nvrtcGetPTX`, `nvrtcGetPTXSize`).
- CUBIN output (CUDA 11.1+).
- Compile-log retrieval, error-message helpers.
- Name-mangling lookup for `__global__` C++ entry points.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-nvrtc`]: https://docs.rs/baracuda-nvrtc
[`baracuda-forge`]: https://docs.rs/baracuda-forge
