#!/usr/bin/env python3
"""Companion to insert_check_fits.py for plans that have NO `can_implement`.

For every `pub fn run(..)` in crates/baracuda-kernels/src whose file defines no
`can_implement`, insert at the top of the body one check per `TensorRef` /
`TensorMut` field of the Args parameter (Option<..> fields when present):

  * `check_dense` (slice holds Π shape) - these kernels ignore `stride` and read
    `numel()` contiguous elements from the slice start;
  * `check_fits`  (reachable extent <= slice) - for the fields listed in
    STRIDE_AWARE, whose kernels do read `stride[0]`.

Idempotent. Usage (repo root):  python tools/soundness/insert_run_checks.py [--count]
`--count` prints the tallies and writes nothing. Files come from `git ls-files`.
"""
import re
import subprocess
import sys

SRC = "crates/baracuda-kernels/src/"
STRIDE_AWARE = {
    "loss/cross_entropy.rs": {"input"},
    "loss/fused_linear_cross_entropy.rs": {"input", "grad_input"},
    "loss/nll.rs": {"input"},
}
files = [
    f
    for f in subprocess.check_output(["git", "ls-files", "-z", "--", SRC + "*.rs"], text=True).split("\0")
    if f
]
raw = {f: open(f, encoding="utf-8", newline="").read() for f in files}
crlf = {f: "\r\n" in s for f, s in raw.items()}
text = {f: s.replace("\r\n", "\n") for f, s in raw.items()}

STRUCT = re.compile(r"pub struct (\w+)(?:<[^>{]*>)?\s*(?:where[^{]*)?\{(.*?)\n\}", re.S)
FIELD = re.compile(r"^\s*pub (\w+):\s*(Option<\s*)?Tensor(?:Ref|Mut)<", re.M)
args_fields = {}
for f, s in text.items():
    for m in STRUCT.finditer(s):
        fields = [(fm.group(1), bool(fm.group(2))) for fm in FIELD.finditer(m.group(2))]
        if fields:
            args_fields[m.group(1)] = fields

RUN = re.compile(r"pub fn run\(([^{;]*?)\)\s*->\s*(?:crate::)?Result<[^{]*?>\s*\{\n")
PARAM_TY = re.compile(r"(\w+):\s*(\w+)<")

tally = {"files": 0, "run_fns": 0, "run_fns_without_tensor_args": 0, "dense_calls": 0, "fits_calls": 0}
for f in files:
    s = text[f]
    if "pub fn can_implement(" in s or "pub fn run(" not in s:
        continue
    rel = f[len(SRC) :]
    out, last, touched = [], 0, False
    for m in RUN.finditer(s):
        tally["run_fns"] += 1
        arg = ty = None
        for pm in PARAM_TY.finditer(m.group(1)):
            if pm.group(2) in args_fields:
                arg, ty = pm.group(1), pm.group(2)
        if arg is None:
            tally["run_fns_without_tensor_args"] += 1
            continue
        block = ""
        for name, opt in args_fields[ty]:
            strided = name in STRIDE_AWARE.get(rel, ())
            fn = "check_fits" if strided else "check_dense"
            head = s[m.end() : m.end() + 3000]
            if f"{fn}(&{arg}.{name})" in head or (f"&{arg}.{name} {{" in head and f"{fn}(t)" in head):
                continue
            if opt:
                block += (
                    f"        if let Some(t) = &{arg}.{name} {{\n"
                    f"            crate::view_check::{fn}(t)?;\n"
                    f"        }}\n"
                )
            else:
                block += f"        crate::view_check::{fn}(&{arg}.{name})?;\n"
            tally["fits_calls" if strided else "dense_calls"] += 1
        if block:
            touched = True
        out.append(s[last : m.end()] + block)
        last = m.end()
    out.append(s[last:])
    new = "".join(out)
    if touched:
        tally["files"] += 1
    if new != s and "--count" not in sys.argv:
        with open(f, "w", encoding="utf-8", newline="") as fh:
            fh.write(new.replace("\n", "\r\n") if crlf[f] else new)
print(tally)
