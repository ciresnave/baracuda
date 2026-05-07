# baracuda-forge

Build-time CUDA kernel compiler for the [baracuda](https://github.com/ciresnave/baracuda)
ecosystem. Drop it into your `[build-dependencies]` and compile `.cu` kernels
to a static library or PTX with nvcc, with incremental rebuilds, parallel
compilation, GPU compute-capability auto-detection, and integrated CUTLASS
support.

`baracuda-forge` is the **build-time** companion to baracuda's runtime
wrappers ([`baracuda-driver`], [`baracuda-runtime`], etc.). Use forge to turn
your `.cu` files into a library, then use the runtime crates to launch the
kernels from Rust.

## Quick start

```rust,no_run
// build.rs
use baracuda_forge::KernelBuilder;

fn main() -> Result<(), baracuda_forge::Error> {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    KernelBuilder::new()
        .source_dir("src/kernels")
        .arg("-O3")
        .build_lib(format!("{out_dir}/libkernels.a"))?;

    println!("cargo:rustc-link-search={out_dir}");
    println!("cargo:rustc-link-lib=kernels");
    Ok(())
}
```

```toml
# Cargo.toml
[build-dependencies]
baracuda-forge = "0.0.1-alpha.5"
```

## Features

- **Incremental builds** — SHA-256 content hashing skips kernels whose source,
  args, and watched headers haven't changed since the last build.
- **Auto-detection** — toolkit location via `$CUDA_PATH` / `$CUDA_HOME` / `$NVCC`
  / `$PATH` (sharing logic with [`baracuda-build`]); GPU compute capability via
  `$CUDA_COMPUTE_CAP` or `nvidia-smi`.
- **C++ standard auto-select** — defaults to `c++20` on CUDA ≥ 12.0,
  `c++17` on older toolkits. Override with `.cpp_std("c++17")`.
- **Per-kernel compute caps** — wildcard patterns route different `.cu` files
  to different `sm_*` targets in one build.
- **Parallel compilation** — rayon thread pool, optional `--threads=N` for
  heavy template-instantiation files (CUTLASS, flash-attention).
- **CUTLASS** — `with_cutlass(commit)` does sparse-checkout fetch + caches
  under `~/.cudaforge/git/checkouts/`; pair with [`baracuda-cutlass-sys`] for
  the safe Rust-side header pin.
- **Custom git deps** — `with_git_dependency()` for header-only dependencies
  beyond CUTLASS.
- **PTX output** — `build_ptx()` instead of `build_lib()` if you want PTX
  blobs to ship and JIT at runtime via [`baracuda-driver`].

## Per-kernel compute capability

```rust,no_run
use baracuda_forge::KernelBuilder;
# fn main() -> Result<(), baracuda_forge::Error> {
KernelBuilder::new()
    .source_glob("src/**/*.cu")
    .with_compute_override("sm90_*.cu", 90)  // Hopper kernels
    .with_compute_override("sm80_*.cu", 80)  // Ampere kernels
    .build_lib("libkernels.a")?;
# Ok(())
# }
```

## With CUTLASS

```rust,no_run
use baracuda_forge::KernelBuilder;
# fn main() -> Result<(), baracuda_forge::Error> {
KernelBuilder::new()
    .source_dir("src/kernels")
    .with_cutlass(None)  // default pinned commit
    .arg("-DUSE_CUTLASS")
    .build_lib("libkernels.a")?;
# Ok(())
# }
```

## Docker / CI

`nvidia-smi` doesn't work during `docker build` (only `docker run --gpus all`).
For Docker builds, set `CUDA_COMPUTE_CAP` and use `require_explicit_compute_cap()`
to fail fast if it's missing:

```rust,no_run
use baracuda_forge::KernelBuilder;
# fn main() -> Result<(), baracuda_forge::Error> {
KernelBuilder::new()
    .require_explicit_compute_cap()?
    .source_dir("src/kernels")
    .build_lib("libkernels.a")?;
# Ok(())
# }
```

## Compatibility

| Aspect | Value | Note |
| --- | --- | --- |
| MSRV | 1.75 | Inherited from the workspace floor. |
| CUDA toolkit | 11.4 – 13.x | C++ standard auto-selects: `c++17` on 11.x, `c++20` on 12.0+. |
| Host compiler | MSVC, GCC, Clang | Whatever your nvcc accepts. Override via `NVCC_CCBIN`. |
| Target OS | Linux, Windows | Same as nvcc. macOS unsupported by NVIDIA since 2019. |
| GPU at build time | Not required | Compute cap can be set via `CUDA_COMPUTE_CAP` (use `require_explicit_compute_cap()` in Docker). |

## Feature flags

`baracuda-forge` ships no Cargo features today — it depends on `baracuda-build`
unconditionally and uses `serde` only at build time (not in the consumer's
runtime crate).

For CUTLASS version pinning, see [`baracuda-cutlass-sys`](../baracuda-cutlass-sys),
which has a `cutlass-2-11` feature flag that downgrades the default CUTLASS
4.x to the CUDA-11.4-compatible 2.11.x line.

## End-to-end example

[`examples/forge_hello.rs`](../../examples/forge_hello.rs) wires this crate up
to `baracuda-driver` for the canonical "compile → load → launch → verify" loop:

```text
cargo run -p baracuda-examples --bin forge_hello --features forge-hello --release
```

That builds [`examples/kernels/forge_hello.cu`](../../examples/kernels/forge_hello.cu)
to PTX via `KernelBuilder`, loads the PTX through `Module::load_ptx`, runs
1,048,576 element vector-add on the GPU, and asserts an exact match against
the CPU reference.

## Acknowledgments

`baracuda-forge` is a **vendored fork** of [`cudaforge`](https://github.com/guoqingbao/cudaforge)
by **Guoqing Bao**, with modifications to fit baracuda's workspace conventions
(error type alignment, shared toolkit detection via [`baracuda-build`], C++
standard auto-selection from detected CUDA version). The two projects are
heading in different directions — cudaforge stays a single focused crate, and
baracuda integrates kernel building into a broader CUDA-stack workspace — so
we vendored rather than depended-on, but the bulk of the algorithm and API is
Guoqing's work.

Big thank you to Guoqing for releasing cudaforge under permissive terms.
See `NOTICE` for the full provenance and the upstream commit hash.

## License

Dual-licensed under MIT or Apache-2.0, matching the workspace and the
upstream cudaforge license. See [LICENSE-MIT](../../LICENSE-MIT) and
[LICENSE-APACHE](../../LICENSE-APACHE) at the repository root.

[`baracuda-build`]: ../baracuda-build
[`baracuda-cutlass-sys`]: ../baracuda-cutlass-sys
[`baracuda-driver`]: ../baracuda-driver
[`baracuda-runtime`]: ../baracuda-runtime
