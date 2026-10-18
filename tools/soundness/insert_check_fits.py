#!/usr/bin/env python3
"""Insert `crate::view_check::check_fits(&args.<field>)?;` at the top of every
`can_implement` in crates/baracuda-kernels/src, one call per `TensorRef` /
`TensorMut` field of the plan's Args struct (Option<..> fields are checked when
present). Idempotent: fields that already have a check are skipped.

Usage (from the repo root):  python tools/soundness/insert_check_fits.py [--count]
`--count` prints the tallies and writes nothing.
Enumerates files with `git ls-files`, never by walking the disk.
"""
import re
import subprocess
import sys

SRC = "crates/baracuda-kernels/src/"
files = [
    f
    for f in subprocess.check_output(["git", "ls-files", "-z", "--", SRC + "*.rs"], text=True).split("\0")
    if f
]
raw = {f: open(f, encoding="utf-8", newline="").read() for f in files}
# Work in LF; restore each file's own line ending on write (Windows checkouts are CRLF).
crlf = {f: "\r\n" in s for f, s in raw.items()}
text = {f: s.replace("\r\n", "\n") for f, s in raw.items()}

# Args struct name -> [(field, optional)]
STRUCT = re.compile(r"pub struct (\w+)(?:<[^>{]*>)?\s*(?:where[^{]*)?\{(.*?)\n\}", re.S)
FIELD = re.compile(r"^\s*pub (\w+):\s*(Option<\s*)?Tensor(?:Ref|Mut)<", re.M)
args_fields = {}
for f, s in text.items():
    for m in STRUCT.finditer(s):
        fields = [(fm.group(1), bool(fm.group(2))) for fm in FIELD.finditer(m.group(2))]
        if fields:
            args_fields[m.group(1)] = fields

SIG = re.compile(r"pub fn can_implement\(\s*&self,\s*(\w+):\s*&(\w+)<[^{;]*?\)\s*->\s*Result<\(\)>\s*\{\n")


def covered(s: str, pos: int, field: str, arg: str) -> bool:
    head = s[pos : pos + 3000]
    return f"check_fits(&{arg}.{field})" in head or (f"&{arg}.{field} {{" in head and "check_fits(t)" in head)


tally = {"can_implement_fns": 0, "with_tensor_fields": 0, "args_struct_not_found": 0, "calls_added": 0}
unknown = []
for f in files:
    s = text[f]
    out, last = [], 0
    for m in SIG.finditer(s):
        tally["can_implement_fns"] += 1
        arg, ty = m.group(1), m.group(2)
        fields = args_fields.get(ty)
        if fields is None:
            tally["args_struct_not_found"] += 1
            unknown.append((f, ty))
            continue
        tally["with_tensor_fields"] += 1
        block = ""
        for name, opt in fields:
            if covered(s, m.end(), name, arg):
                continue
            if opt:
                block += (
                    f"        if let Some(t) = &{arg}.{name} {{\n"
                    f"            crate::view_check::check_fits(t)?;\n"
                    f"        }}\n"
                )
            else:
                block += f"        crate::view_check::check_fits(&{arg}.{name})?;\n"
            tally["calls_added"] += 1
        out.append(s[last : m.end()] + block)
        last = m.end()
    out.append(s[last:])
    new = "".join(out)
    if new != s and "--count" not in sys.argv:
        with open(f, "w", encoding="utf-8", newline="") as fh:
            fh.write(new.replace("\n", "\r\n") if crlf[f] else new)
print(tally)
for u in unknown:
    print("ARGS-STRUCT-NOT-FOUND", u)
