# baracuda-build

Build-script helpers shared by every `baracuda-*-sys` crate.

This is a `[build-dependencies]` crate. It does not link against CUDA
itself and contains no runtime code — it just gives `build.rs` files a
common, well-tested place to detect a CUDA toolkit, find `nvcc`, parse
versions, and (optionally) configure `bindgen` for CUDA headers.

## Quick start

```rust,no_run
// build.rs
fn main() {
    baracuda_build::emit_rerun_hints();

    if let Some(install) = baracuda_build::detect_cuda() {
        baracuda_build::emit_version_cfg(&install);
        // install.root, install.include, install.lib, install.nvcc, install.version
        // are now available for downstream cc / bindgen invocations.
    }
}
```

## What's here

- **`detect_cuda()`** — returns a `CudaInstall` (toolkit root, `include/`
  path, library path, parsed `cuda.h` version, and `nvcc` path if found)
  by walking `$CUDA_PATH` / `$CUDA_HOME` / `$CUDA_ROOT` /
  `$CUDA_TOOLKIT_ROOT_DIR` plus standard locations on Linux/Windows.
- **`find_nvcc()`** — locates `nvcc` via `$NVCC`, the detected install,
  or `$PATH` walk.
- **`parse_nvcc_version(stdout)`** — parses the `release X.Y` line from
  `nvcc --version` output.
- **`emit_rerun_hints()`** — emits `cargo:rerun-if-env-changed=` for
  every env var the detector consults.
- **`emit_version_cfg(&install)`** — emits `cargo:rustc-cfg=cuda_<major>_<minor>`
  + `cuda_<major>` so downstream crates can `#[cfg]`-gate version-dependent
  code.
- **`find_library(&install, stem)`** — locates `libcuda.so.X` /
  `cublas64_X.dll` / etc. for static-link scenarios.
- **`bindgen_builder(&install)`** *(feature `bindgen`)* — preconfigured
  `bindgen::Builder` with the right `clang_arg`s for CUDA headers
  (`-D__CUDACC__`, `-D__host__=`, etc.), enum style, layout/derive flags
  baracuda relies on.

## Used by

Every `baracuda-*-sys` crate. Also re-exported via [`baracuda-forge`] for
toolkit detection inside the kernel-build pipeline.

Part of the [baracuda](https://github.com/ciresnave/baracuda) workspace.

## License

Dual MIT / Apache-2.0.

[`baracuda-forge`]: https://docs.rs/baracuda-forge
