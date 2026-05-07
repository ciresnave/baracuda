# baracuda-cutlass-kernels-sys

Compiled CUTLASS template instantiations for the baracuda ecosystem.

This crate hosts the curated set of `.cu` source files that instantiate
CUTLASS templates for specific `(dtype, layout, arch)` tuples, compiles them
via [`baracuda-forge`], and exposes raw `extern "C"` entry points consumed by
the safe [`baracuda-cutlass`] crate.

**You probably want [`baracuda-cutlass`] instead.** This crate is
infrastructure — its API is a list of unsafe C functions with no type
safety. The safe layer wraps these with typed plans, descriptors, and
lifetime-checked device buffers.

## v0 scope

| Op family | Kernels (planned) | Shipped today |
| --- | --- | --- |
| GEMM | `f16 × RCR × sm80`, `bf16 × RCR × sm80`, `f16 × RCR × sm90a`, `bf16 × RCR × sm90a` | sm80 pair only |
| Grouped GEMM | same SKU set | sm80 pair (variable M per group) |

`RCR` = `A` row-major `[M,K]`, `B` column-major `[K,N]`, `C/D` row-major
`[M,N]`, `f32` accumulator, `f32` alpha/beta. **Epilogue: Identity only.**
The `Bias` epilogue was planned for v0 but removed during the Fuel team's
design review (it was advertised in the safe API but no kernel
implemented it); per their roadmap it returns "after grouped GEMM lands
and a real caller asks for it." sm90a kernels are deferred until Hopper
hardware is available for validation; the safe layer's selection wiring
is already in place for them.

## Features

| Feature | Default | Effect |
| --- | --- | --- |
| `sm80` | yes | Build Ampere (also runs on Ada / forward-compatible). |
| `sm90a` | no | Build Hopper-specialized kernels. Mutually exclusive with `cutlass-2-11`. |
| `cutlass-2-11` | no | Use CUTLASS 2.11 headers (CUDA 11.4 path). Mutually exclusive with `sm90a`. |

Enable both `sm80` and `sm90a` to ship a fat binary that runs on both.

## Build cost

Each kernel SKU costs roughly 30s of nvcc time on first build (CUTLASS 4.x
templates compile faster than I'd estimated). The shipped sm80 pair
(single + grouped × {f16, bf16}) takes ~50s end-to-end on a clean build.
Subsequent builds hit `baracuda-forge`'s SHA-256 incremental cache and
rebuild only changed kernels.

## Acknowledgments

Specification by the Fuel ML library team. CUTLASS by NVIDIA. Build-time
kernel compilation via [`baracuda-forge`] (vendored from `cudaforge` by
Guoqing Bao). See `NOTICE` for full provenance.

[`baracuda-forge`]: ../baracuda-forge
[`baracuda-cutlass`]: ../baracuda-cutlass
