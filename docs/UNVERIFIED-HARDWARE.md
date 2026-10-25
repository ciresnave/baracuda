# Tests that cannot run on the device runner — UNVERIFIED

Policy (CireSnave, 2026-10-16): a test that cannot run on a GitHub-hosted runner but can
run on the self-hosted device runner runs there; it is never `#[ignore]`d and never hidden
behind a feature nobody builds. A test that cannot run **even there** is named below and
is **UNVERIFIED** — nothing in this repo may claim it as covered.

The device runner is one Windows laptop with a single RTX 4070 (sm_89). Source: `git grep`
of the `#[ignore = "…"]` reasons on `main` at the time of writing. No `#[ignore]` attribute
remains on `main` (0 line-anchored hits, 2026-10-25): the tests listed here are gated behind
`device-tests` (some with extra cfg conditions) and are not built by the runner.

## A. Needs hardware or an OS the runner lacks — UNVERIFIED

CireSnave 2026-10-25 (decision 283 / SA-21): the laptop has TWO GPUs, the NVIDIA RTX 4070 and an AMD one, so a test needing two NVIDIA GPUs waits for his desktop (a few days). What each row below needs, and whether one NVIDIA GPU can run it (checked 2026-10-25 on the laptop: `nvidia-smi -L` lists one NVIDIA GPU; no NCCL, NVSHMEM or cuFile library is installed on this Windows box):

| Test | Two NVIDIA GPUs? | Other blocker | Runs on the laptop? |
|---|---|---|---|
| `distributed_adam_two_ranks_...`, `column_parallel_world_size_2_smoke`, `row_parallel_world_size_2_smoke`, `ring_attention_multi_rank_scaffold` | yes (world size 2) | NCCL (Linux multi-GPU) | no: needs the desktop and Linux NCCL |
| the four `nvshmem` tests | no (a PE launcher can start one PE) | NVSHMEM is Linux, sm_70+ | no: the laptop is Windows |
| `cufile_write_read_roundtrip` | no | cuFile + GDS filesystem, Linux only | no: the laptop is Windows |

| Test | Needs | Where |
|---|---|---|
| `distributed_adam_two_ranks_all_reduce_then_step_then_all_gather` | 2+ GPUs, NCCL multi-rank | `baracuda-optim/tests/distributed_adam_multi_rank_scaffold.rs` |
| `ring_attention_multi_rank_scaffold` | 2+ GPUs, multi-process NCCL bring-up | `baracuda-kernels/tests/ring_attention_smoke.rs` |
| `column_parallel_world_size_2_smoke` | 2+ GPUs + NCCL | `baracuda-megatron/tests/multi_rank_scaffold.rs` |
| `row_parallel_world_size_2_smoke` | 2+ GPUs + NCCL | `baracuda-megatron/tests/multi_rank_scaffold.rs` |
| `nvshmem_loads_and_reports_version`, `context_init_and_pe_discovery`, `symmetric_heap_alloc_and_self_put`, `world_team_matches_context` | NVSHMEM runtime (Linux, sm_70+) + a PE launcher | `baracuda-nvshmem/tests/smoke.rs` |
| `cufile_write_read_roundtrip` | cuFile + GDS-capable filesystem (Linux-only) | `baracuda-cufile/tests/read_write.rs` |

## B. Needs a vendor library whose presence on the runner is not yet confirmed — UNVERIFIED until the runner inventory says otherwise

NCCL (17: `baracuda-nccl` 11, `baracuda-megatron` single-rank 4, `baracuda-optim` single-rank 2;
NCCL is described in its own test reasons as "typically Linux multi-GPU hosts"), cuDNN 3
(`baracuda-cudnn`), TensorRT 2 (`baracuda-tensorrt`, also needs the C-ABI shim), cuVS 2
(`baracuda-cuvs`, RAPIDS), nvImageCodec 3, cuTENSOR 2, nvCOMP 1, NPP 1.

Once the runner is set up, an inventory step should record which of these libraries load; each
that does moves out of this list into its crate's `device-tests` run, and each that does not
stays here by name.

## D. `baracuda-kernels` test files behind cargo features the device run does not build — UNVERIFIED

The kernels `#[ignore]` retirement (alpha.117-121) gates device tests behind `device-tests`; the
device run builds `--features device-tests` plus the defaults (`sm80`, `fa2`) and ran
2055 passed, 0 failed, 0 skipped (`cargo gpu-test`, RTX 4070 sm_89, 2026-10-16). Test files that
additionally need another feature, a vendor library or a source-only build option are NOT built
by that run and are UNVERIFIED (counts are `#[test]` attributes at `main`, alpha.121; they are
type-checked in CI, never executed):

| Feature | Files | Tests |
|---|---|---|
| `sm89` | 13 | 98 |
| `cudnn` | 19 | 73 |
| `fa2_backward` | 3 | 29 |
| `mamba` | 8 | 21 |
| `flashinfer` | 3 | 11 |
| `bnb_nf4` | 3 | 9 |
| `marlin` | 2 | 8 |
| `ring_attention` | 1 | 4 |
| `xformers_sparse24` | 2 | 3 |
| `xformers_blocksparse` | 1 | 3 |
| `awq` | 1 | 2 |
| `mhc` | 1 | 2 |

Total 57 files / 263 tests. `sm89` is the closest to runnable (the runner is sm_89); the next
step is adding feature sets to the device-run command, one at a time, each reporting its
ran-count. A feature that builds and runs moves out of this table.

## C. Runtime-declared skips

A `require_optional!` that declines at run time (an optional capability absent on the box) is
reported by reason in the run record (`skips_by_reason` in `target/device-test-results.json`,
and the job summary). It is a named skip, not a pass.
