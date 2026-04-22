# Contributing to baracuda

Thanks for your interest! baracuda is early-stage; the best contributions right now are:

- Bug reports with minimal repros.
- Design feedback on the module layout of each safe crate (see `docs/design/`).
- Hand-written bindings for functions the current bindgen allowlist skips.
- Integration tests for libraries on hardware we don't have access to.

## Workflow

1. Fork, branch, and open a draft PR early for discussion.
2. Run the full local checklist before requesting review:
   ```
   cargo fmt --all
   cargo clippy --workspace --all-features -- -D warnings
   cargo build --workspace
   cargo test --workspace
   cargo doc --workspace --no-deps --all-features
   ```
3. If your change requires a GPU, gate the test with `#[ignore]` and run it locally with `BARACUDA_GPU_TESTS=1 cargo test -- --ignored`. CI does not have GPUs on free runners.

## Crate conventions

- Two-layer rule: every wrapped library is `*-sys` (raw FFI) + safe crate. Don't add a third tier.
- Dynamic loading: never `#[link]` an NVIDIA library except behind a `static-link` feature.
- Safe APIs never expose raw FFI handles on public signatures except via `as_raw()` / `from_raw()` escape hatches.
- Error types per safe crate; all implement `baracuda_types::CudaStatus`.

See `docs/design/` for more detail.

## License

By contributing, you agree that your work is dual-licensed under MIT and Apache-2.0, matching the repository.
