# On-device tests

Tests that need an NVIDIA GPU are **not** `#[ignore]`d. In a migrated crate (`baracuda-driver`,
`baracuda-kernels`, `baracuda-cutlass`, `baracuda-runtime`, `baracuda-flashinfer`, `baracuda-ozimmu`, `baracuda-kernels-bench`, `baracuda-cuda-emit`, `baracuda-cuda-seam`, `baracuda-cublas`, `baracuda-transformer-engine`, `baracuda-optim`, and the 21 library crates listed in `.github/workflows/device-tests.yml`) they sit behind the crate's `device-tests` cargo feature, and three
things keep that from being a way to hide them:

1. **Type-checked on every PR.** CI runs
   `cargo clippy --locked -p baracuda-driver --tests --features device-tests -- -D warnings`
   (and `cargo check -p baracuda-kernels --tests` and, on the sm80 leg, `-p baracuda-cutlass --tests`, each with and without the feature, per arch,
   under `DOCS_RS=1`) on the hosted runners (no CUDA needed to *compile* them), so a gated test cannot rot unbuilt.
2. **Actually run.** `.github/workflows/device-tests.yml` runs
   `cargo gpu-test -p baracuda-driver --features device-tests` on the self-hosted runner (the
   maintainer's RTX 4070 laptop) on push to `main`, on a daily schedule, and on manual dispatch.
   Never on fork pull requests (public repo: fork code must not run on that machine). The run
   holds the machine-wide `gpu-run` lock.
3. **A stale or missing run is red.** `.github/workflows/device-tests-detector.yml` fails if the
   last successful device run is older than 7 days or absent.

**Status (2026-10-22): the device lane is not running.** No self-hosted runner is registered for this public repo (ephemeral runners for public repos need a re-registration source; CireSnave board item 241), so the `DEVICE_RUNNER_ENABLED` gate is false and every device job is `skipped`. Measured: `device-tests.yml` (push 08:36Z, schedule 08:41Z) and the first `workflow_dispatch` of `device-tests-kernel-features.yml` (run 39751578145 at `20445e3b`) all concluded `skipped`. Nothing in the device lane has run since the runner went away, so the kernel feature sets in that workflow are UNVERIFIED on CI, and a green CI check says nothing about on-device behaviour. Results obtained by running `cargo gpu-test` locally under `gpu-run` are **LOCAL-ONLY**: label them so, and do not cite them as CI evidence.

Both workflows are gated on the repository variable `DEVICE_RUNNER_ENABLED == 'true'` and are
inert until the runner is registered and that variable is set.

## Run locally

```text
cargo gpu-test -p baracuda-driver --features device-tests
```

`xtask test-gpu` compiles unlocked, then runs under the `gpu-run` lock with
`BARACUDA_GPU_REQUIRED=1`, so a `require!`d resource that is absent is a **failure**, not a skip.
It runs with `--include-ignored`, so crates not yet migrated still have their `#[ignore]`d tests run.
A run that passes zero tests exits non-zero.

## Run record

Every `cargo gpu-test` writes `target/device-test-results.json` (override with
`BARACUDA_RESULTS_FILE`): `status`, `exit_code`, `passed` / `failed` / `ignored`, `commit`, `host`,
start/finish times, `skipped_declared`, and `skips_by_reason` (from `require_optional!`). The CI job
uploads it as an artifact and prints the counts in the job summary.

What cannot run even on that box is listed, by name, in [`UNVERIFIED-HARDWARE.md`](UNVERIFIED-HARDWARE.md).

## Feature-gated kernel tests

`device-tests.yml` builds `baracuda-kernels` with default features (sm80, fa2), so test files gated on any other feature compile to empty binaries there. `device-tests-kernel-features.yml` (weekly + manual) builds each feature set (sm89, cudnn, mamba, flashinfer, fa2_backward, bnb_nf4, xformers_sparse24, xformers_blocksparse, ring_attention, mhc) and runs exactly those binaries; marlin/awq stay in `device-tests.yml`. Static census at 34477f16: 55 binaries, 257 `#[test]` fns. Unverified: whether each feature set builds on the runner (not run).
