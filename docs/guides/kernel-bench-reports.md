# Running and reporting kernel benchmarks

Board #162. One command produces one self-describing JSON report; you send the
file back. Nothing is uploaded automatically.

## Run

```text
pwsh scripts/gpu-run.ps1 -Project baracuda -- cargo run --release -p baracuda-kernels-bench \
    --bin baracuda-bench -- run --report out.json --machine-id <your-box-id> \
    [--note "what you had running"] [--samples 101] [--inner 50]
```

`--machine-id` is a stable name you choose (`cires-desktop-4060`), `[A-Za-z0-9_-]` only.
Do not put a username or path in it or in `--note`.
Without the repo's `gpu-run` wrapper, run the same `cargo run` directly; just make
sure nothing else is using the GPU (see below).

The report carries its schema version, GPU name and arch, driver, CUDA toolkit,
SM/memory clocks, temperature, power limit (`null` where the driver doesn't
expose it, e.g. most laptops), count of other GPU processes, Windows power plan,
and seven statistics per cell: min, max, mean, median, p99, stddev, sample count.

## Noisy run vs quiet run

Record both at least once per machine. The noisy run is the everyday
state of the box; the quiet run is the best case it can do.

**Noisy:** run it as the machine normally is. Say so in `--note`.

**Quiet** (before running):

1. Close browsers, chat clients, video/streaming apps and anything GPU-accelerated.
2. `nvidia-smi` — confirm no other process is listed under compute or graphics and
   utilisation is ~0%.
3. Set the OS power plan to High performance / plugged in (laptops: on AC).
4. Let the GPU idle ~1 minute so temperature settles; check `nvidia-smi` again.
5. Run once; don't touch the machine until it finishes.

You do not need to prove a run was quiet: the machine-state block records it.
A quiet run whose report shows `other_gpu_process_count > 1` or a non-High power
plan was not quiet, whatever the filename says. Note that the process count may
include the benchmark itself, so a clean box reads 1.

## Send it back

Either:

- open a PR adding `bench-reports/<machine_id>/<YYYY-MM-DD>-<noisy|quiet>.json`, or
- attach the JSON to an issue (`gh issue create --body-file out.json` or by hand).

Read the file once before sending; it should contain nothing you don't want public.

## What happens to it

Reports are untrusted input. They are read, compared and discussed, never merged
into `bench-baselines/` automatically. See `bench-reports/README.md`.
