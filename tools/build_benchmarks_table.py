"""Build the BENCHMARKS.md wide-format rollup from phase29 CSV files.

Phase 73.3 — joins per-bench CSV outputs from
`crates/baracuda-kernels-bench/target/criterion/phase29/*.csv` into one
markdown rollup with side-by-side timing columns
(baracuda / cuBLAS / cuDNN / PyTorch) per (op, shape, dtype) row.

Usage:
    cargo bench -p baracuda-kernels-bench --features sm89,cudnn -- --quick
    python tools/build_benchmarks_table.py [--in DIR] [--out FILE]

⚠️ The `cargo bench` line above is the WHOLE-SUITE form and it needs two things
that are easy to miss; both were hit regenerating this file on 2026-09-05:

  * cuDNN's DLLs must be on PATH or every cuDNN-linked bench dies at startup with
    `exit code: 0xc0000135, STATUS_DLL_NOT_FOUND` — a loader failure, so there is
    no Rust panic and no hint which library is missing. On this box:
    `C:/Program Files/NVIDIA/CUDNN/v9.23/bin/13.3/x64` (pick the subdir matching
    the CUDA toolkit, not the other one sitting beside it).

  * `--bench flash_attention` panics on main (`CutlassInternal(1001)` at its
    warmup, flash_attention.rs:124) and `cargo bench` ABORTS THE WHOLE RUN on the
    first failing target — so the suite stops partway and the rollup silently
    describes whatever ran before the abort. It emits no CSV rows, so skipping it
    costs the table nothing.

⚠️ A PARTIAL RUN IS THE REAL HAZARD HERE, because this script rewrites the whole
marked section from whatever CSVs it finds: ops whose benches did not run are
DROPPED from the table rather than left stale, and nothing in the output says so.
That is how `mmvq`, `mmvq_multim`, `flash_decoding` and `flash_decoding_gqa` went
missing from a previous regen. `append_csv_row` also APPENDS, so a re-run over a
non-empty phase29 dir DUPLICATES rows.

So: clear `target/criterion/phase29/*.csv` first, run the 19 CSV-emitting benches
(the ones calling `append_csv_row`; the other 5 contribute nothing), and check the
count is 19 before rolling up. Declaring that number first is what makes a short
count a finding instead of a table you trust.

The script does NOT run benches. It only reads CSVs that previous
`cargo bench` runs produced. Default input dir:
`crates/baracuda-kernels-bench/target/criterion/phase29/`.

Output: writes a `BENCHMARKS.md` section between the markers

    <!-- BEGIN auto-generated phase29 rollup -->
    ...
    <!-- END auto-generated phase29 rollup -->

into `crates/baracuda-kernels-bench/BENCHMARKS.md`. If the markers
don't exist, they're appended at the end of the file. Content between
the markers is replaced on every run.

CSV columns expected:

    op,shape,dtype,baracuda_ns,reference_ns,reference,delta,pytorch_ns,pytorch_delta

The script groups rows by (op, shape, dtype) and merges:
  - `baracuda_ns` from the row where it's non-zero (typically the
    `reference: "baracuda"` or `reference: ""` self-bench row).
  - `<library>_ns` from the row where `reference == <library>` and
    `reference_ns` is set.
  - `pytorch_ns` from any row that has it set (all should agree).

Speedup columns:
  - `cuBLAS / baracuda` etc. — `> 1` means library faster than baracuda.
  - `PyTorch / baracuda` — `> 1` means PyTorch faster than baracuda.

Op families are emitted in this order:
  gemm, softmax, layernorm, rmsnorm, reduce_sum, reduce_max, reduce_mean,
  add, mul, relu, gelu, conv2d, maxpool2d, flash_sdpa_gqa, mmvq, others.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------
# Reading the CSVs.
# ---------------------------------------------------------------------


def _try_float(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if v == 0.0:
        return None
    return v


def load_phase29(csv_dir: pathlib.Path) -> dict[tuple[str, str, str], dict[str, float]]:
    """Read every `*.csv` under `csv_dir`, return `{(op, shape, dtype): merged}`.

    Each value dict can carry keys `baracuda_ns`, `cuBLAS_ns`,
    `cuDNN_ns`, `PyTorch_ns`, and arbitrary other `<reference>_ns`
    columns from custom benches.
    """
    cells: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for csv_path in sorted(csv_dir.glob("*.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["op"], row["shape"], row["dtype"])
                cell = cells[key]
                bn = _try_float(row.get("baracuda_ns", ""))
                if bn is not None:
                    cell["baracuda_ns"] = bn
                rn = _try_float(row.get("reference_ns", ""))
                ref_label = row.get("reference", "").strip()
                if rn is not None and ref_label and ref_label != "baracuda":
                    cell[f"{ref_label}_ns"] = rn
                pn = _try_float(row.get("pytorch_ns", ""))
                if pn is not None:
                    cell["PyTorch_ns"] = pn
    return cells


# ---------------------------------------------------------------------
# Op family ordering + presentation.
# ---------------------------------------------------------------------

OP_ORDER: tuple[str, ...] = (
    "gemm",
    "softmax",
    "layernorm",
    "rmsnorm",
    "reduce_sum",
    "reduce_max",
    "reduce_mean",
    "add",
    "mul",
    "relu",
    "gelu",
    "conv2d",
    "maxpool2d",
    "flash_sdpa_gqa",
    "mse",
    "l1",
    "cross_entropy",
    "nll",
    "mmvq",
)

# Library labels to surface as columns (in column order). Anything not
# in this list is dropped from the rollup (e.g. "baracuda-self"
# placeholder reference labels are uninteresting).
LIBRARY_COLUMNS: tuple[str, ...] = ("cuBLAS", "cuDNN", "PyTorch", "FA2", "mHC")

# Ops with NO PyTorch (or NVIDIA-library) equivalent — their rows are baracuda
# ABSOLUTE timings, not a comparison. Declared explicitly so the artifact
# distinguishes "no equivalent exists" (a fact) from "reference not yet
# generated" (missing data): a self-only op that lacks a PyTorch column is
# CORRECT; a torch-mappable op that lacks one is a gap. Without this, both
# render as an absent column and a reader can't tell which. (mmvq is the
# existing instance — GGUF matrix-vector has no direct torch op; the
# GGUF/AWQ/Marlin quant family joins here as it is benched.)
SELF_ONLY_OPS: frozenset[str] = frozenset({"mmvq", "mmvq_multim"})


def _baseline_torch_versions() -> str:
    """Distinct torch version(s) from the committed baseline's provenance runs,
    for the rollup header — DERIVED, not hardcoded (a hardcoded version drifts
    the moment the baseline is regenerated, as it did at 2.11.0→2.14.0). Reads
    the single `pytorch_*.json` under bench-baselines; returns 'unknown' if the
    count is not exactly one (the same ambiguity rule the loader uses)."""
    base = (
        pathlib.Path(__file__).resolve().parent.parent
        / "crates" / "baracuda-kernels-bench" / "bench-baselines"
    )
    files = sorted(base.glob("pytorch_*.json")) if base.is_dir() else []
    if len(files) != 1:
        return "unknown"
    try:
        meta = json.loads(files[0].read_text(encoding="utf-8")).get("metadata", {})
        vers = sorted({r.get("torch_version", "?") for r in meta.get("provenance_runs", [])})
        return ", ".join(vers) if vers else "unknown"
    except (json.JSONDecodeError, OSError):
        return "unknown"


def _format_ns(v: float | None) -> str:
    if v is None:
        return ""
    if v >= 1e6:
        return f"{v / 1e6:.2f}ms"
    if v >= 1e3:
        return f"{v / 1e3:.1f}μs"
    return f"{v:.0f}ns"


def _speedup_vs_baracuda(library_ns: float | None, baracuda_ns: float | None) -> str:
    """`library_ns / baracuda_ns`. `> 1.0` ⇒ library faster, `< 1.0` ⇒ baracuda faster."""
    if library_ns is None or baracuda_ns is None or baracuda_ns == 0.0:
        return ""
    r = library_ns / baracuda_ns
    if r >= 1.05:
        return f"**{r:.2f}×**"  # baracuda > 5% slower
    if r <= 0.95:
        return f"{r:.2f}×"
    return "≈"  # within ±5%


def _shape_sort_key(shape: str) -> tuple[int, ...]:
    """Stable shape sort: split on letters/digits, numeric where possible."""
    parts: list[int] = []
    for tok in re.findall(r"\d+", shape):
        parts.append(int(tok))
    return tuple(parts) or (0,)


def _dtype_sort_key(dtype: str) -> int:
    return {"f32": 0, "f16": 1, "bf16": 2, "f64": 3}.get(dtype, 99)


def emit_markdown(cells: dict[tuple[str, str, str], dict[str, float]]) -> str:
    """Render the rollup as a markdown string."""
    by_op: dict[str, list[tuple[str, str, dict[str, float]]]] = defaultdict(list)
    for (op, shape, dtype), cell in cells.items():
        by_op[op].append((shape, dtype, cell))

    op_keys = list(by_op.keys())
    op_keys.sort(key=lambda o: (OP_ORDER.index(o) if o in OP_ORDER else 999, o))

    out: list[str] = []
    out.append("This section is generated by `tools/build_benchmarks_table.py`")
    out.append("from the per-bench CSV outputs under")
    out.append("`target/criterion/phase29/`. Do not edit by hand — re-run the")
    out.append("script after a fresh `cargo bench` to refresh.")
    out.append("")
    out.append("Hardware: RTX 4070 Laptop GPU (sm_89).")
    out.append(
        f"PyTorch baseline: {_baseline_torch_versions()} (frozen JSON in `bench-baselines/`)."
    )
    out.append("")
    out.append("Speedup column convention: `library_ns / baracuda_ns`.")
    out.append("`> 1` (bolded) means baracuda is faster than that library at this cell.")
    out.append("`≈` means within ±5%.")
    out.append("")

    for op in op_keys:
        rows = by_op[op]
        # Sort within an op family by (dtype, shape).
        rows.sort(key=lambda r: (_dtype_sort_key(r[1]), _shape_sort_key(r[0])))

        # Pick which library columns to show: only the ones with at least
        # one populated value in this op family.
        seen_libs = {
            lib for (_, _, c) in rows for lib in LIBRARY_COLUMNS if f"{lib}_ns" in c
        }
        libs = [lib for lib in LIBRARY_COLUMNS if lib in seen_libs]

        out.append(f"### `{op}`")
        out.append("")
        if op in SELF_ONLY_OPS:
            # State the absence as a fact, not missing data: this op has no
            # PyTorch/library equivalent, so its baracuda column is an absolute
            # timing, not a comparison. (A torch-mappable op that lacked a
            # column would instead be a not-yet-generated gap.)
            out.append(
                "_Self-only: no PyTorch/library equivalent — baracuda timings "
                "below are absolute, not a comparison._"
            )
            out.append("")
        header_cells = ["dtype", "shape", "baracuda"]
        for lib in libs:
            header_cells.append(lib)
            header_cells.append(f"{lib}/baracuda")
        out.append("| " + " | ".join(header_cells) + " |")
        out.append("| " + " | ".join(["---"] * len(header_cells)) + " |")

        for shape, dtype, cell in rows:
            baracuda = cell.get("baracuda_ns")
            row = [dtype, f"`{shape}`", _format_ns(baracuda)]
            for lib in libs:
                lib_ns = cell.get(f"{lib}_ns")
                row.append(_format_ns(lib_ns))
                row.append(_speedup_vs_baracuda(lib_ns, baracuda))
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------
# Insert into BENCHMARKS.md between markers.
# ---------------------------------------------------------------------

BEGIN_MARKER = "<!-- BEGIN auto-generated phase29 rollup -->"
END_MARKER = "<!-- END auto-generated phase29 rollup -->"


# ---------------------------------------------------------------------
# Drop guard.
#
# This script rewrites the WHOLE marked section from whatever CSVs it finds, so
# an op whose bench did not run is DELETED from the table rather than left
# stale -- and nothing in the output says so. A partial run does not produce a
# partial table; it produces a COMPLETE-LOOKING table that is missing things.
#
# Not hypothetical: mmvq, mmvq_multim, flash_decoding and flash_decoding_gqa sat
# missing from the published table for an unknown period after an earlier
# partial regen, and a reader would have concluded baracuda does not benchmark
# them. They were noticed only by diffing a fresh full regen against main by
# hand -- a comparison nothing performs and nobody was asked to.
#
# The table has no EDGE: it renders as finished at every size. This makes the
# tool answer "did this regen lose a family?" itself, and REFUSE rather than
# warn -- a warning on a 450-row rollup is one line of scrollback.
#
# ---------------------------------------------------------------------
# WHAT THIS GUARD DOES NOT DO, stated here because its limit is invisible
# exactly when things are healthy.
#
# It RATCHETS FROM THE COMMITTED FILE. Its floor is whatever BENCHMARKS.md
# already says, NOT the set of ops the bench suite can produce. So it prevents
# FURTHER drops and cannot detect EXISTING ones.
#
# Concretely: had this existed before #74, it would have ratcheted happily from
# a table that was already missing mmvq, mmvq_multim, flash_decoding and
# flash_decoding_gqa, and pronounced every later regen clean. The four ops it
# now protects are ops it would not have recovered.
#
# All 66 families are present as of #74, which is precisely the state in which
# nobody discovers that limit -- a green guard over a degraded file looks
# identical to a green guard over a correct one.
#
# Closing it needs a different instrument: a comparison against what the SUITE
# emits (the benches calling `append_csv_row`) rather than against what the file
# holds. That is not built. Recorded rather than implied, because a reader
# finding this guard green in a degraded future needs the sentence more than a
# second mechanism.  (Raised by the Claim Auditor, 2026-09-06.)
# ---------------------------------------------------------------------

FAMILY_HEADING = re.compile(r"^### `([^`]+)`", re.M)


def families_in(text: str) -> set:
    """Op families named by the ``### `name` `` headings of a rollup section."""
    if BEGIN_MARKER in text and END_MARKER in text:
        text = text.split(BEGIN_MARKER, 1)[1].split(END_MARKER, 1)[0]
    return set(FAMILY_HEADING.findall(text))


def splice_into(md_path: pathlib.Path, new_section: str) -> bool:
    """Replace content between markers in `md_path`. Returns True if file changed."""
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    block = f"{BEGIN_MARKER}\n{new_section}\n{END_MARKER}\n"
    if BEGIN_MARKER in text and END_MARKER in text:
        pattern = re.compile(
            re.escape(BEGIN_MARKER) + r"[\s\S]*?" + re.escape(END_MARKER) + r"\n?",
            re.M,
        )
        new_text = pattern.sub(block, text, count=1)
    else:
        sep = "\n\n" if text and not text.endswith("\n\n") else ""
        new_text = text + sep + "## Cross-implementation rollup (auto-generated)\n\n" + block
    if new_text == text:
        return False
    md_path.write_text(new_text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--in",
        dest="csv_dir",
        default="crates/baracuda-kernels-bench/target/criterion/phase29",
        help="Phase 29 CSV directory.",
    )
    p.add_argument(
        "--out",
        dest="md_path",
        default="crates/baracuda-kernels-bench/BENCHMARKS.md",
        help="BENCHMARKS.md path to splice into.",
    )
    p.add_argument(
        "--allow-drop",
        action="store_true",
        help=(
            "Permit this regen to REMOVE op families that the current "
            "BENCHMARKS.md has. Without it the script refuses, because a "
            "partial bench run silently deletes rows rather than leaving them "
            "stale. Pass it only when an op was deliberately retired."
        ),
    )
    args = p.parse_args()

    csv_dir = pathlib.Path(args.csv_dir)
    if not csv_dir.is_dir():
        print(f"ERROR: {csv_dir} not a directory", file=sys.stderr)
        return 2

    cells = load_phase29(csv_dir)
    if not cells:
        print(f"ERROR: no rows loaded from {csv_dir}", file=sys.stderr)
        return 2

    section = emit_markdown(cells)
    md_path = pathlib.Path(args.md_path)

    # Refuse to silently drop families. See the drop-guard note above.
    existing = families_in(md_path.read_text(encoding="utf-8")) if md_path.exists() else set()
    dropped = sorted(existing - families_in(section))
    if dropped and not args.allow_drop:
        plural = "y" if len(dropped) == 1 else "ies"
        print(
            f"ERROR: this regen would REMOVE {len(dropped)} op famil{plural} "
            f"from {md_path}:",
            file=sys.stderr,
        )
        for fam in dropped:
            print(f"  - {fam}", file=sys.stderr)
        print(
            "",
            file=sys.stderr,
        )
        print(
            "That is what a PARTIAL bench run looks like: the table is rewritten "
            "from whatever CSVs exist, so an op whose bench did not run is "
            "DELETED rather than left stale, and the result still looks "
            "complete. Run the full suite (the benches calling append_csv_row), "
            "or pass --allow-drop if these ops were deliberately retired.",
            file=sys.stderr,
        )
        return 3
    if dropped:
        print(f"WARNING: --allow-drop given; removing {len(dropped)}: {', '.join(dropped)}")

    changed = splice_into(md_path, section)
    summary = f"{len(cells)} cells, {len({op for (op, _, _) in cells})} op families"
    if changed:
        print(f"Wrote rollup ({summary}) into {md_path}")
    else:
        print(f"No change ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
