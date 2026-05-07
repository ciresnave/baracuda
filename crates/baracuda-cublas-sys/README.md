# baracuda-cublas-sys

Raw FFI bindings + dynamic loader for **NVIDIA cuBLAS**, including the
companion libraries **cuBLASLt** (lightweight matmul with descriptor /
heuristic API) and **cuBLASXt** (multi-GPU GEMM).

Symbols resolve lazily via [`libloading`](https://docs.rs/libloading);
no link-time dependency on `libcublas.so` / `cublas64_*.dll` /
`cublasLt*` / `cublasXt*`.

**Most users want [`baracuda-cublas`]** — that crate exposes typed
generic-over-scalar BLAS operations (`gemm`, `gemv`, `axpy`, ...),
batched variants, BLAS-1/2/3 coverage, and the cuBLASLt / cuBLASXt
APIs in idiomatic Rust.

## What's exposed

- **cuBLAS** core: handles, streams, math mode, atomics mode, pointer
  mode, all L1 / L2 / L3 BLAS in `S` / `D` / `C` / `Z` plus `Ex`
  variants; batched and strided-batched GEMM; batched direct solvers
  (`getrf`, `getrs`, `getri`, `matinv`).
- **cuBLASLt**: descriptors (matrix, matmul, preference), heuristics,
  matmul invocation.
- **cuBLASXt**: multi-GPU GEMM with affinity control.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-cublas`]: https://docs.rs/baracuda-cublas
