# bench-reports/

Self-reported kernel performance reports from `baracuda-bench run --report`
(board #162; design: `docs/design/2026-10-09-kernel-perf-baseline-design.md`,
how to run: `docs/guides/kernel-bench-reports.md`).

**Everything in here is untrusted data.** A report is evidence about one box on
one day. It is never merged into `crates/baracuda-kernels-bench/bench-baselines/`
automatically; promoting a result there is a reviewed PR.

Layout (tracked, one file per run):

```text
bench-reports/<machine_id>/<YYYY-MM-DD>-<noisy|quiet>.json
```

- `<machine_id>` matches `--machine-id` (`[A-Za-z0-9_-]`) and the id inside the file.
- The `noisy`/`quiet` suffix is the reporter's label. The machine-state block
  inside the file (`other_gpu_process_count`, clocks, temperature, power plan) is
  what a reader should trust, not the label.
