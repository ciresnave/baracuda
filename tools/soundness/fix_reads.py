#!/usr/bin/env python3
"""Apply rustc E0616 ("field `X` of struct `TensorRef|TensorMut` is private")
fixes from a `cargo check --message-format=json` log: turn the field read into
the accessor call, `.data` -> `.data()`, `.shape` -> `.shape()`, `.stride` ->
`.stride()`. Spans inside macro expansions are skipped and listed.

usage: fix_reads.py LOG.json [--dry]
"""
import json
import re
import sys
from collections import defaultdict

log = sys.argv[1]
dry = "--dry" in sys.argv
edits = defaultdict(set)
skipped = []
seen = 0
for line in open(log, encoding="utf-8"):
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get("reason") != "compiler-message":
        continue
    d = m["message"]
    if d["level"] != "error":
        continue
    mm = re.match(r"field `(data|shape|stride)` of struct `(TensorRef|TensorMut)` is private", d["message"])
    if not mm:
        continue
    seen += 1
    for sp in d["spans"]:
        if not sp.get("is_primary"):
            continue
        # Spans inside a macro BODY point at the real source text; the
        # text-equality guard below skips anything that does not.
        edits[sp["file_name"]].add((sp["byte_start"], sp["byte_end"], mm.group(1)))

changed = 0
for f, es in edits.items():
    path = f.replace("\\", "/")
    raw = open(path, "rb").read()
    out = bytearray(raw)
    for start, end, name in sorted(es, reverse=True):
        if bytes(out[start:end]) != name.encode():
            skipped.append((f, start, f"span text {bytes(out[start:end])!r} != {name}"))
            continue
        out[start:end] = (name + "()").encode()
        changed += 1
    if not dry:
        open(path, "wb").write(bytes(out))
print(f"E0616 diagnostics: {seen}; edits applied: {changed}; files: {len(edits)}; skipped: {len(skipped)}")
for s in skipped[:30]:
    print("SKIPPED", s)
