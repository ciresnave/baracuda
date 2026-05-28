# Vendored ozIMMU source

This directory holds the upstream [ozIMMU](https://github.com/enp1s0/ozIMMU)
source — Hiryuki Ootomo's Ozaki-scheme FP64 GEMM library that synthesizes a
DGEMM from S² 8-bit tensor-core matmuls. The integration sits behind the
optional `ozimmu` cargo feature on `baracuda-kernels`; when enabled it joins
the `GemmPlan` dispatch surface as `BackendKind::Ozaki { slices }` for the
FP64 path. See the workspace `README.md` for the integration notes.

## Provenance

- **Repository**: https://github.com/enp1s0/ozIMMU
- **Pinned upstream commit**: `08eea9231729d54dbfd92955f2cbfc21ec236856`
  (merge of `17-update-mateval` into `main`, March 2025-era state)
- **Pinned `src/cutf` submodule commit**:
  `c28c2025a5f3419661ce9cc632e3139a71b6f382`
- **License**: MIT — full text reproduced verbatim in `LICENSE`. Compatible
  with baracuda's MIT/Apache-2.0 dual.
- **Upstream paper**: Hiroyuki Ootomo, Katsuhisa Ozaki & Rio Yokota,
  "DGEMM on Integer Matrix Multiplication Unit", *IJHPCA* 2024.
  [arXiv:2306.11975](https://arxiv.org/abs/2306.11975)

## Why baracuda vendors instead of `git submodule`-ing

Per the workspace `feedback_vendor_with_attribution.md` rule: when integrating
a third-party crate / repo whose architectural direction diverges from
baracuda's conventions, vendor + diverge with explicit attribution rather
than pulling in as a submodule or git dependency. Specifically:

- baracuda links every CUDA library statically through its own `-sys` crates
  (e.g. `baracuda-cublas-sys`); ozIMMU upstream is structured around
  `LD_PRELOAD` interception of cuBLAS symbols. The two designs are
  fundamentally incompatible at the link layer.
- baracuda's build path is `baracuda-forge → nvcc` directly; ozIMMU upstream
  uses CMake. Vendoring lets us drive the build through the same path as
  every other CUDA crate in the workspace.
- baracuda targets cross-platform (Windows + Linux); ozIMMU upstream uses
  `<dlfcn.h>` + `<unistd.h>` + `RTLD_NEXT`, which are Linux-only.

## Patches applied to the vendored source

Two small in-place patches sit on top of the upstream commit pinned above.
Both are marked with a `baracuda patch (Phase 44):` comment so they stand
out in a diff. Future re-vendoring should re-apply them.

### 1. `src/utils.hpp` — direct-link mode

The upstream `<dlfcn.h>` / `<unistd.h>` includes and the
`ozIMMU_get_function_pointer` definition are gated behind the absence of a
new `OZIMMU_BARACUDA_DIRECT_LINK` preprocessor macro (which `build.rs` sets
unconditionally). When the macro is defined, `ozIMMU_get_function_pointer`
is rewritten to call an `extern "C" ozimmu_baracuda_lookup_cublas_symbol`
helper that the baracuda shim TU supplies.

### 2. `src/cublas.cu` and `src/culip.cu` — excluded from build

These two translation units contain the LD_PRELOAD interceptor shims
(`cublasGemmEx`, `cublasDgemm_v2`, `cublasZgemm_v2`, …) and the CULiP
profiler. Including them in the static archive would re-define cuBLAS
symbols already provided by `baracuda-cublas-sys` and pull in `dlfcn.h` on
Windows. The `build.rs` simply does not list them in its source set. The
two symbols `mtk::ozimmu::cublasCreate_org` and
`mtk::ozimmu::cublasDestroy_org` that `cublas.cu` used to supply are now
provided by `crates/baracuda-ozimmu-sys/csrc/ozimmu_baracuda_shim.cu`.

## What we add on top

The baracuda integration adds two source files of its own (not under
`vendor/`):

- `crates/baracuda-ozimmu-sys/csrc/ozimmu_baracuda_shim.cu` — supplies
  `mtk::ozimmu::cublasCreate_org`, `mtk::ozimmu::cublasDestroy_org`, and
  `ozimmu_baracuda_lookup_cublas_symbol` (as documented above), plus a
  flat C ABI (`baracuda_ozimmu_create`, `_destroy`, `_set_cuda_stream`,
  `_reallocate_working_memory_bytes`, `_dgemm`) that the Rust safe-wrapper
  layer in `baracuda-ozimmu` calls into. The flat C surface lets us skip
  bindgen on the upstream C++ headers (which use `std::vector`,
  `std::tuple`, `std::string` in their public API).

## Platform support

The integration ships Linux-only in alpha.56. Windows support requires
porting ozIMMU's two `<dlfcn.h>`-dependent translation units (`cublas.cu`,
`culip.cu`) — both of which we already drop — but also requires
verifying the `cutf` headers' Windows behavior. `culip.cu`'s
`clock_gettime` call is POSIX-only, but as we don't compile that TU at all
the only remaining gap is whether the `cutf` headers themselves use
anything POSIX-specific. They appear not to, but this is unvalidated. The
`build.rs` therefore prints a `cargo:warning` and skips the compile on
non-Linux hosts.
