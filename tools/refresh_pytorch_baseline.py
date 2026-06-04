"""Generate PyTorch reference timings for baracuda's bench suite.

Phase 73.1 — replaces the abandoned tch-rs in-process integration with a
frozen-on-disk baseline. Runs each op + shape + dtype combo under
PyTorch's CUDA path with CUDA-event timing (same methodology as
baracuda-kernels-bench's `measure_median_ns`), dumps medians to a JSON
file checked into the repo under
`crates/baracuda-kernels-bench/bench-baselines/`.

The bench harness reads this JSON at process start and emits the PyTorch
median into the `pytorch_ns` column of `PhaseTwentyNineRow`. No PyTorch
dep at build or test time — refresh is an out-of-band step, run when
PyTorch updates materially or when adding new ops to the bench suite.

Usage:
    python tools/refresh_pytorch_baseline.py [--ops OP1,OP2,...]
                                              [--output PATH]
                                              [--samples N] [--inner N]

If `--output` is omitted, the script picks a filename based on detected
device + PyTorch version + CUDA version:

    bench-baselines/pytorch_<device_slug>_<torch_version>_cu<cuda>.json

Example:
    bench-baselines/pytorch_rtx4070_2.11.0_cu130.json

Notes on timing methodology:

  - 10-launch warmup (matches baracuda's `warmup()`).
  - `samples` independent batches, each containing `inner` launches.
  - Wall-clock time around `torch.cuda.synchronize()` for each batch.
  - Report the median of per-batch averages (matches baracuda's
    `measure_median_ns`).

Defaults (`samples=11`, `inner=50`) are conservative — same as bench
harness — and add ~550 launches per (op, shape, dtype) cell. For the
~10 ops × ~10 shapes × ~3 dtypes covered in Phase 29 that's ~165k
launches, roughly 3-5 minutes on RTX 4070.

Ops covered in this initial pass (extended over time):

  - gemm: `torch.matmul` × {f32, f16, bf16} × shape sweep matching
    `CROSS_GEMM_M_SWEEP` × `CROSS_GEMM_KN_SWEEP` from
    `crates/baracuda-kernels-bench/src/lib.rs`.

Future ops (planned):

  - softmax, log_softmax, layer_norm, rms_norm
  - conv2d, max_pool2d
  - reductions (sum / max / mean)
  - elementwise (add / mul / relu / gelu)
  - flash sdpa + gqa
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import torch


# ---------------------------------------------------------------------
# Shape sweep — kept in sync with `baracuda-kernels-bench/src/lib.rs`.
# ---------------------------------------------------------------------

# Matches `CROSS_GEMM_M_SWEEP`. `M=1` covers decode, `M=32 / 128` covers
# prefill at LLM-typical batch sizes.
GEMM_M_SWEEP: tuple[int, ...] = (1, 32, 128)
# Matches `CROSS_GEMM_KN_SWEEP`. 7B hidden = 4096; smaller 2048 picks up
# the K-bound regime.
GEMM_KN_SWEEP: tuple[int, ...] = (2048, 4096)

# Dtypes to cover per op. `f32` everywhere; `f16` / `bf16` for ops with
# tensor-core paths.
GEMM_DTYPES: tuple[tuple[str, torch.dtype], ...] = (
    ("f32", torch.float32),
    ("f16", torch.float16),
    ("bf16", torch.bfloat16),
)


# ---------------------------------------------------------------------
# Timing helpers — match baracuda's measure_median_ns methodology.
# ---------------------------------------------------------------------

WARMUP_LAUNCHES: int = 10


def time_median_ns(
    launch: Callable[[], None],
    samples: int,
    inner: int,
) -> float:
    """Median per-launch wall-clock ns across `samples` batches of `inner`."""
    # Warmup — let GPU clock + caches settle. Match `warmup()` in
    # `baracuda-kernels-bench/src/lib.rs`.
    for _ in range(WARMUP_LAUNCHES):
        launch()
    torch.cuda.synchronize()

    per_launch_ns: list[float] = []
    for _ in range(samples):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(inner):
            launch()
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        per_launch_ns.append((end - start) / inner)

    return statistics.median(per_launch_ns)


# ---------------------------------------------------------------------
# Per-op generators — yield (op, shape, dtype, launch_fn).
# ---------------------------------------------------------------------


def gemm_cases() -> Iterable[tuple[str, str, str, Callable[[], None]]]:
    """All (op, shape, dtype, launch) tuples for GEMM."""
    device = torch.device("cuda")
    for kn in GEMM_KN_SWEEP:
        for m in GEMM_M_SWEEP:
            n = kn
            k = kn
            shape = f"M{m}_N{n}_K{k}"
            for dtype_name, dtype in GEMM_DTYPES:
                # Bind loop vars into the closure (Python late-binding gotcha).
                # Tensors live for the duration of the cell — torch keeps them
                # alive via the closure capture; freed at next cell.
                a = torch.ones((m, k), dtype=dtype, device=device)
                b = torch.ones((k, n), dtype=dtype, device=device)
                launch = lambda a=a, b=b: torch.matmul(a, b)
                yield ("gemm", shape, dtype_name, launch)


# Op registry — name → cases generator. Extend per-op for fanout.
OP_REGISTRY: dict[str, Callable[[], Iterable[tuple[str, str, str, Callable[[], None]]]]] = {
    "gemm": gemm_cases,
}


# ---------------------------------------------------------------------
# Device + version metadata.
# ---------------------------------------------------------------------


def _device_slug() -> str:
    """Short identifier for the active CUDA device. Used in default output filename."""
    name = torch.cuda.get_device_name(0)
    # "NVIDIA GeForce RTX 4070 ..." -> "rtx4070"
    m = re.search(r"RTX\s*(\d{4})", name)
    if m:
        return f"rtx{m.group(1)}"
    # Fallback: lowercase, strip non-alnum
    return re.sub(r"[^a-z0-9]+", "", name.lower())[:20]


def _torch_version_short() -> str:
    """`'2.11.0+cu130'` → `'2.11.0'`."""
    return torch.__version__.split("+", 1)[0]


def _cuda_version_short() -> str:
    """`'13.0'` → `'130'` for filename use."""
    v = torch.version.cuda or "unknown"
    return v.replace(".", "")


def _default_output_path() -> Path:
    """Default output filename derived from device + torch + CUDA version."""
    base = Path(__file__).resolve().parent.parent / "crates" / "baracuda-kernels-bench" / "bench-baselines"
    return base / (
        f"pytorch_{_device_slug()}_{_torch_version_short()}_cu{_cuda_version_short()}.json"
    )


def _metadata(samples: int, inner: int) -> dict[str, object]:
    """Self-describing block written into the JSON header."""
    return {
        "schema_version": 1,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "sample_count": samples,
        "inner_iters": inner,
        "warmup_launches": WARMUP_LAUNCHES,
        "methodology": (
            "wall-clock around torch.cuda.synchronize() per batch; "
            "median of per-batch averages. Matches baracuda's "
            "measure_median_ns in baracuda-kernels-bench/src/lib.rs."
        ),
    }


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--ops",
        default=",".join(OP_REGISTRY.keys()),
        help=f"Comma-separated ops to refresh. Default: all ({','.join(OP_REGISTRY)}).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: bench-baselines/pytorch_<device>_<torch>_cu<cuda>.json",
    )
    p.add_argument("--samples", type=int, default=11, help="Outer batch count (default 11).")
    p.add_argument("--inner", type=int, default=50, help="Inner launches per batch (default 50).")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() == False — install a CUDA-enabled PyTorch wheel.", file=sys.stderr)
        return 2

    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    for op in ops:
        if op not in OP_REGISTRY:
            print(f"ERROR: unknown op '{op}'. Known: {','.join(OP_REGISTRY)}", file=sys.stderr)
            return 2

    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"Ops: {','.join(ops)}")
    print(f"Output: {output_path}")
    print(f"Samples: {args.samples}  inner: {args.inner}")
    print()

    results: list[dict[str, object]] = []
    total_cells = 0
    for op in ops:
        cases = list(OP_REGISTRY[op]())
        total_cells += len(cases)
    print(f"Total cells: {total_cells}")
    print()

    cell_index = 0
    for op in ops:
        for (op_label, shape, dtype_name, launch) in OP_REGISTRY[op]():
            cell_index += 1
            print(f"[{cell_index}/{total_cells}] {op_label}/{dtype_name}/{shape} ...", end=" ", flush=True)
            try:
                median_ns = time_median_ns(launch, args.samples, args.inner)
            except RuntimeError as e:
                print(f"FAILED: {e}", flush=True)
                continue
            print(f"{median_ns:.1f} ns", flush=True)
            results.append(
                {
                    "op": op_label,
                    "shape": shape,
                    "dtype": dtype_name,
                    "median_ns": round(median_ns, 3),
                }
            )

    payload = {"metadata": _metadata(args.samples, args.inner), "results": results}
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"Wrote {len(results)} cells to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
