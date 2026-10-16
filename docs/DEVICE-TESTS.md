# On-device tests

Tests that need an NVIDIA GPU are **not** `#[ignore]`d. In a migrated crate (`baracuda-driver`,
`baracuda-kernels`) they sit behind the crate's `device-tests` cargo feature, and three
things keep that from being a way to hide them:

1. **Type-checked on every PR.** CI runs
   `cargo clippy --locked -p baracuda-driver --tests --features device-tests -- -D warnings`
   (and `cargo check -p baracuda-kernels --tests` with and without the feature, per arch,
   under `DOCS_RS=1`) on the hosted runners (no CUDA needed to *compile* them), so a gated test cannot rot unbuilt.
2. **Actually run.** `.github/workflows/device-tests.yml` runs
   `cargo gpu-test -p baracuda-driver --features device-tests` on the self-hosted runner (the
   maintainer's RTX 4070 laptop) on push to `main`, on a daily schedule, and on manual dispatch.
   Never on fork pull requests (public repo: fork code must not run on that machine). The run
   holds the machine-wide `gpu-run` lock.
3. **A stale or missing run is red.** `.github/workflows/device-tests-detector.yml` fails if the
   last successful device run is older than 7 days or absent.

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
