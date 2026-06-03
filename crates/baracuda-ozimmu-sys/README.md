# baracuda-ozimmu-sys

Build + raw FFI bindings for baracuda's clean-fork of **ozIMMU** —
Hiroyuki Ootomo's Ozaki-scheme FP64 GEMM library that synthesizes a
DGEMM from `S²` int8 tensor-core matmuls.

## What it wraps

baracuda's clean-fork of [`enp1s0/ozIMMU`](https://github.com/enp1s0/ozIMMU)
(MIT). Reference: Ootomo, Ozaki & Yokota, *"DGEMM on Integer Matrix
Multiplication Unit"*, IJHPCA 2024,
[arXiv:2306.11975](https://arxiv.org/abs/2306.11975).

Phase 44b internalized the upstream sources under `cuda/` (no
`vendor/` subdir; cutf submodule eliminated). See `ATTRIBUTION.md` at
the crate root for full provenance + the unmodified MIT license text.

## When to use this crate

Almost never directly. The safe wrapper is [`baracuda-ozimmu`]. Most
baracuda callers opt into the Ozaki backend at the
[`baracuda-kernels`] level via
`PlanPreference::prefer_backend = Some(BackendKind::Ozaki { slices: ... })`
on a `GemmPlan` (gated behind the `ozimmu` cargo feature on
`baracuda-kernels`).

This crate is the build-system + FFI declarations crate; the static
archive it produces is consumed by `baracuda-ozimmu` and (transitively)
by `baracuda-kernels`.

## What this crate exposes

A small flat C ABI defined by `cuda/baracuda_shim.cu`. Direct bindgen
on the C++ header was avoided because the public API uses
`std::vector<std::tuple<...>>` for one of the
`reallocate_working_memory` overloads — bindgen-friendly only at the
cost of dragging the whole `std::` namespace into the generated
bindings. The flat C wrapper handles only the ops the safe layer needs:

- `baracuda_ozimmu_create` / `baracuda_ozimmu_destroy`
- `baracuda_ozimmu_set_stream`
- `baracuda_ozimmu_reallocate_bytes`
- `baracuda_ozimmu_dgemm`

Plus the `COMPUTE_MODE_*` constants matching the upstream
`mtk::ozimmu::compute_mode_t` enum (slices 3..18 + DGEMM passthrough +
Auto).

## Quick example

```rust,no_run
# unsafe fn demo(stream: *mut core::ffi::c_void) {
use baracuda_ozimmu_sys::{baracuda_ozimmu_create, COMPUTE_MODE_FP64_INT8_8};

let mut handle = core::ptr::null_mut();
let status = baracuda_ozimmu_create(&mut handle);
debug_assert_eq!(status, 0);
// ... use handle in baracuda_ozimmu_dgemm ...
# }
```

Real call sites should go through the safe wrapper.

## Cargo features

| Feature | Default | Effect |
| --- | --- | --- |
| `build-vendor` | no | Build the baracuda-owned ozIMMU source via nvcc. When OFF, this crate is a no-op stub — the FFI extern blocks compile but link will only resolve if a downstream crate actually exercises them. Decoupled from `default` to keep `cargo check --workspace --no-default-features` cheap and platform-agnostic. Auto-enabled by `baracuda-ozimmu`'s `default` feature, which `baracuda-kernels`'s `ozimmu` feature ultimately pulls in. |

`docs.rs` has no nvcc — the build.rs short-circuits on `DOCS_RS=1`.

## Status / scope

- Phase 44 (alpha.56) — initial Linux-only vendor.
- Phase 44b (alpha.57) — clean-fork, cutf elimination, Windows port.
  Sources internalized at `crates/baracuda-ozimmu-sys/cuda/`.
- Phase 44c — `OzakiVariant::EF` perf variant (Uchino / Ozaki /
  Imamura 2024).

## Related crates

- [`baracuda-ozimmu`] — safe, typed API; the documented entry point.
- [`baracuda-cuda-sys`] — provides `cudaStream_t` / `cublasHandle_t`
  opaque typedefs that the FFI signatures reference.
- [`baracuda-cublas-sys`] — pulled in for `cublasCreate_v2` /
  `cublasGemmEx` symbols that the static archive references at link
  time.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

[`baracuda-ozimmu`]: https://docs.rs/baracuda-ozimmu
[`baracuda-kernels`]: https://docs.rs/baracuda-kernels
[`baracuda-cuda-sys`]: https://docs.rs/baracuda-cuda-sys
[`baracuda-cublas-sys`]: https://docs.rs/baracuda-cublas-sys
