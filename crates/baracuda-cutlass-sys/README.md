# baracuda-cutlass-sys

Header acquisition for [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
as a baracuda workspace dependency. Use this crate as a `[build-dependencies]`
or `[dependencies]` entry to make CUTLASS headers available to other
`build.rs` files (e.g., [`baracuda-forge`]) and to downstream `cc` /
`cxx` / `bindgen` invocations.

## What it does

- Sparse-checks-out CUTLASS headers from GitHub at build time.
- Caches under `~/.baracuda-cutlass-sys/git/checkouts/` (file-locked for
  concurrent-build safety, derived from the cudaforge fetch logic).
- Emits Cargo metadata (`cargo:include`, `cargo:root`,
  `cargo:rustc-env=BARACUDA_CUTLASS_INCLUDE_DIR`) so consumers find the
  headers without hand-rolling path discovery.
- Exposes [`INCLUDE_DIR`] at the Rust level for runtime access.

## Version selection

| How | Default | Result |
|-----|---------|--------|
| (none) | CUTLASS 4.x (CUDA 12+) | tag `v4.2.0` |
| feature `cutlass-2-11` | CUTLASS 2.11.x (CUDA 11.4 floor) | tag `v2.11.0` |
| env `CUTLASS_DIR=/path` | local checkout | use that path |
| env `BARACUDA_CUTLASS_COMMIT=<sha>` | specific commit | sparse-checkout to commit |

The `cutlass-2-11` feature is the supported escape hatch for users who
want to keep baracuda's CUDA-11.4 floor across the CUTLASS path. CUTLASS 4.x
requires CUDA 12+.

## Usage

```toml
# Cargo.toml
[build-dependencies]
baracuda-cutlass-sys = "0.0.1-alpha.5"
```

```rust,no_run
// build.rs
fn main() {
    // The DEP_CUTLASS_INCLUDE env var is set by baracuda-cutlass-sys.
    let include = std::env::var("DEP_CUTLASS_INCLUDE")
        .expect("baracuda-cutlass-sys should set DEP_CUTLASS_INCLUDE");
    println!("cargo:rustc-link-search=native={include}");
    // Or feed `include` into cc::Build::include() / bindgen::Builder.
}
```

When pairing with `baracuda-forge`, the wiring is automatic — forge picks up
the `DEP_CUTLASS_INCLUDE` set by this crate and uses it for `with_cutlass`,
skipping forge's own fetch.

## License

Dual MIT / Apache-2.0 (matches the workspace and CUTLASS-adjacent ecosystem).
The CUTLASS source itself remains under NVIDIA's BSD-3-Clause; this crate
fetches it at build time but does not redistribute it.

See [`NOTICE`](NOTICE) for attribution to upstream cudaforge for the
sparse-checkout fetch logic.

[`baracuda-forge`]: ../baracuda-forge
[`INCLUDE_DIR`]: https://docs.rs/baracuda-cutlass-sys/latest/baracuda_cutlass_sys/constant.INCLUDE_DIR.html
