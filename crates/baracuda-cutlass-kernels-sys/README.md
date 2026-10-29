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

For SKUs that **don't** go through CUTLASS templates — bespoke
hand-rolled `mma.sync` kernels for layouts / dtypes CUTLASS can't
express cleanly (today: int8 RRR; later: int4, bin, FP8) — see
[`baracuda-kernels-sys`] (raw FFI) and [`baracuda-kernels`] (safe
facade). The two -sys crates are siblings; the unified
`baracuda-kernels` facade dispatches across both.

## v0 scope

| Op family | Kernels (planned) | Shipped today |
| --- | --- | --- |
| GEMM | `f16 × RCR × sm80`, `bf16 × RCR × sm80`, `f16 × RCR × sm90a`, `bf16 × RCR × sm90a` | sm80 pair only |
| Grouped GEMM | same SKU set | sm80 pair (variable M per group) |

`RCR` = `A` row-major `[M,K]`, `B` column-major `[K,N]`, `C/D` row-major
`[M,N]`, `f32` accumulator, `f32` alpha/beta. **Epilogue: Identity only
for the v0 f16/bf16 pair above.**

> **Update 2026-10-29.** The v0 table above is the original scope, not
> today's inventory. `kernels/` now also holds `*_bias` epilogue kernels
> (f16/bf16 `gemm_rcr_sm80_bias.cu`, `gemm_rrr_sm80_bias.cu`, and the f32,
> f64, tf32 and s8/u8 siblings), plus f32 SIMT, f64, tf32, s8 and u8 GEMMs
> and a batched variant (see the file list in
> `crates/baracuda-cutlass-kernels-sys/kernels/`). The statement that the
> `Bias` epilogue was removed is historical. No sm_90a kernel exists:
> sm90a stays deferred until Hopper hardware is available for validation.

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
[`baracuda-kernels`]: ../baracuda-kernels
[`baracuda-kernels-sys`]: ../baracuda-kernels-sys
