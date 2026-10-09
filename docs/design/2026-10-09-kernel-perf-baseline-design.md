# Kernel performance baseline + cross-machine report tooling — design

Board item 162. Written per CireSnave's 2026-10-09 ruling (verbatim, excerpted):

> "Baselines should be captured on this box as the default... we should create
> the tooling so that it is as easy as possible for others to run the
> benchmarks on their hardware and report the results back to us. We should
> then test that bench-and-auto-report tooling by running it on my desktop
> PC... We can also re-run those once I receive my P40's... The benchmark
> suite should measure best, worst, and average and we should test both on a
> noisy PC and with as many things paused or shut down as possible for a
> clean run at least once."

## This is extend, not build-from-scratch

`crates/baracuda-kernels-bench/` already has most of the hard part:

- **CUDA-event timing** (`time_with_events`/`time_with_events_restored`,
  `src/lib.rs`) — on-device, no host-side noise, criterion-driven, with a
  10-launch warmup (`warmup`/`WARMUP_ITERS`) and a process-wide
  `DEVICE_TIMING_LOCK` (two concurrent device-timing runs on one GPU produce a
  *wrong* result, not just a slow one — measured, see the lock's doc comment).
- **A proven provenance pattern**: `PytorchBaselineMetadata` /
  `PytorchProvenanceRun` already records `device_name`, `device_capability`,
  `generator_git_sha`, `generator_dirty` (never suppressed — a bare SHA
  implying a clean tree is the lie this guards against), `sample_count`,
  `inner_iters`, `warmup_launches`, and a `regen_trigger` condition (never a
  date). This is the right shape for a kernel-vs-kernel baseline too; it needs
  extending, not replacing.
- 23 `[[bench]]` binaries already cover GEMM, Flash Attention, Conv2d, and
  more (`benches/*.rs`), each already using the warmup + CUDA-event pattern.

What's missing, specifically:

1. **Statistics**: criterion's own report gives mean/median/std-dev but not a
   labeled best/worst/p99 in the stored baseline. Need a wrapper over the raw
   per-sample timings (criterion exposes them) that computes `min, max, mean,
   median, p99, stddev, sample_count` and stores all seven, not just median —
   CireSnave: best/worst/average alone hide bimodality.
2. **Machine-state capture**, per result, not just per PyTorch-comparison
   run: GPU name + arch (already have `device_name`/`device_capability`),
   **driver version, CUDA toolkit version, GPU clocks (via `nvidia-smi
   --query-gpu=clocks.sm,clocks.mem`), power limit, temperature, count of
   other GPU-using processes, and (Windows) the active power plan**. None of
   this is captured today.
3. **Baseline key gains a machine-id component**: `(kernel, dtype, shape
   class, target arch, machine_id)` — CireSnave's P40 vs 4070 vs 4060 point
   means arch alone is not enough once more than one box reports. `machine_id`
   is a stable, human-assigned string (e.g. `cires-laptop-4070`,
   `cires-desktop-4060`), not something auto-derived from hardware (two
   otherwise-identical boxes should still be distinguishable if that ever
   matters, and auto-derived ids from driver/BIOS strings are not stable
   across OS reinstalls).
4. **One command for anyone to run and self-report**: `baracuda-bench run
   --report` (new thin binary in `baracuda-kernels-bench`, or a new
   `baracuda-bench-cli` — TBD at implementation time) that runs the existing
   sweep, computes the 7-number stats + machine-state block above, and emits
   one self-describing JSON result file. "Self-describing" means the file
   carries its own schema version and enough metadata that a human reading it
   cold (no README) can tell whose box it came from and when.
5. **A documented return path**, scrubbed of anything identifying beyond what
   the reporter intends (no local usernames/paths, no secrets) — a `gh issue
   create` with the JSON attached, or a PR adding the file under a
   `bench-reports/<machine_id>/` directory. CireSnave specifically wants this
   "copy-paste simple" for the desktop-PC test run.
6. **Trust boundary**: a report from anyone other than this lane's own capture
   is **untrusted data** — read, never auto-merged into the authoritative
   baseline. Promoting a reported result into `bench-baselines/` is a human
   (or PM-reviewed PR) decision, same as any other external input per the
   portfolio's "every absence needs a positive control" / untrusted-data
   discipline.
7. **Noisy-run / quiet-run protocol**: a short checklist (close browsers,
   stop other GPU processes, check `nvidia-smi` for other utilization, note
   the Windows power plan) run at least once for a "clean" capture, and the
   normal ad-hoc capture is explicitly allowed to be noisy — both get recorded
   (point 2 above is what makes a noisy run distinguishable from a quiet one
   after the fact, rather than needing to trust a label).

## Rollout order (per CireSnave, don't reorder)

1. Build the stats + machine-state extensions and the `--report` command.
   First real captures on this laptop (sm_89, RTX 4070) — both a noisy and at
   least one quiet run, to prove the two are distinguishable from their
   recorded machine-state block, not just their numbers.
2. CireSnave runs the report command on his desktop (RTX 4060) purely to
   **prove the tooling**, not because a 4060 vs 4070 comparison is the point.
   This is the test of "copy-paste simple."
3. Re-run once his P40s are installed — expected to show large deltas on many
   kernels (sm_61's fp16 arithmetic throughput story, see `half_arith`/board
   #106 B1(c)) once that work resumes.

## Out of scope for this doc

- Actually populating `BENCH-sm89.md`'s placeholder numbers (separate,
  smaller task once the stats extension lands).
- CI integration — GPU benchmarking in CI is noisy/contended; this stays an
  on-demand, human-triggered tool, consistent with board 162's own ruling
  (on-demand, this box as default).
- Any change to the PyTorch-comparison baseline (`docs/planning/foundational/
  13-benchmark-suite-pytorch-completion.md`) — separate authority, separate
  file, not touched here.

## Open items for implementation (not blocking this design)

- Exact JSON schema for the new baseline/report format (likely a sibling of
  `PytorchBaselineEntry`, not a fork of it — share the provenance struct
  shape where the fields overlap).
- Where `bench-reports/` lives and whether it's a tracked directory or
  deliberately gitignored until promoted.
