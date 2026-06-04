"""Build the BENCHMARKS.md wide-format rollup from phase29 CSV files.

Phase 73.3 — joins per-bench CSV outputs from
`crates/baracuda-kernels-bench/target/criterion/phase29/*.csv` into one
markdown rollup with side-by-side timing columns
(baracuda / cuBLAS / cuDNN / PyTorch) per (op, shape, dtype) row.

Usage:
    cargo bench -p baracuda-kernels-bench --features sm89,cudnn -- --quick
    python tools/build_benchmarks_table.py [--in DIR] [--out FILE]

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
    "mmvq",
)

# Library labels to surface as columns (in column order). Anything not
# in this list is dropped from the rollup (e.g. "baracuda-self"
# placeholder reference labels are uninteresting).
LIBRARY_COLUMNS: tuple[str, ...] = ("cuBLAS", "cuDNN", "PyTorch", "FA2", "mHC")


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
    out.append("Hardware: RTX 4070 Laptop GPU (sm_89), CUDA 13.0, cuDNN 9.x.")
    out.append("PyTorch baseline: 2.11.0+cu130 (frozen JSON in `bench-baselines/`).")
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
    changed = splice_into(md_path, section)
    summary = f"{len(cells)} cells, {len({op for (op, _, _) in cells})} op families"
    if changed:
        print(f"Wrote rollup ({summary}) into {md_path}")
    else:
        print(f"No change ({summary})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
