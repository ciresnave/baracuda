#!/usr/bin/env python3
"""Rewrite `TensorRef { data, shape, stride }` / `TensorMut { .. }` struct
literals into `TensorRef::new(data, shape, stride).unwrap()` calls, now that
the fields are private and `new` is the validated constructor. With
`--matrix` it does the same for `MatrixRef/MatrixMut { data, rows, cols, ld }`
and `VectorRef { data, len, stride }`.

Handles any field order, `field: expr` and shorthand `field`, an optional
turbofish (`TensorRef::<f32, 2> { .. }`) and nested brackets / strings in the
expressions. A literal it cannot rewrite safely (comment inside the braces,
`..` rest pattern, a missing or extra field, an unparsable field) is left
untouched and listed, for a by-hand fix.

Usage (repo root):  python tools/soundness/privatize_views.py [--matrix] [--count]
`--count` reports and writes nothing. Files come from `git ls-files`.
"""
import re
import subprocess
import sys

if "--matrix" in sys.argv:
    SPECS = {
        "MatrixRef": ("data", "rows", "cols", "ld"),
        "MatrixMut": ("data", "rows", "cols", "ld"),
        "VectorRef": ("data", "len", "stride"),
    }
else:
    SPECS = {"TensorRef": ("data", "shape", "stride"), "TensorMut": ("data", "shape", "stride")}
OPEN = re.compile(r"\b(" + "|".join(SPECS) + r")(\s*::\s*<[^{};]*?>)?\s*\{")
SKIP_BEFORE = re.compile(r"(?:\bstruct|\bimpl|\bfor|->|\bmatch|\blet)\s*$")

files = [
    f
    for f in subprocess.check_output(["git", "ls-files", "-z", "--", "*.rs", "*.md"], text=True).split("\0")
    if f and not f.startswith("tools/soundness/")
]


def match_close(s: str, i: int):
    """Index of the `}` closing the `{` at s[i], or None. Skips strings."""
    depth = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == '"':
            i += 1
            while i < n and s[i] != '"':
                i += 2 if s[i] == "\\" else 1
        elif c == "/" and s.startswith("//", i):
            return None  # comment inside the literal: leave for a human
        elif c == "/" and s.startswith("/*", i):
            return None
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def split_top(body: str):
    parts, depth, cur, i = [], 0, [], 0
    while i < len(body):
        c = body[i]
        if c == '"':
            j = i + 1
            while j < len(body) and body[j] != '"':
                j += 2 if body[j] == "\\" else 1
            cur.append(body[i : j + 1])
            i = j + 1
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    tail = "".join(cur)
    if tail.strip():
        parts.append(tail)
    return [p.strip() for p in parts if p.strip()]


tally = {"files": 0, "rewritten": 0, "skipped": 0}
skipped = []
for f in files:
    try:
        raw = open(f, encoding="utf-8", newline="").read()
    except UnicodeDecodeError:
        print("NOT-UTF8, skipped:", f)
        continue
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    out, last, changed = [], 0, False
    for m in OPEN.finditer(s):
        if m.start() < last:
            continue  # nested inside an already-rewritten literal
        if SKIP_BEFORE.search(s[max(0, m.start() - 12) : m.start()]):
            continue
        brace = m.end() - 1
        close = match_close(s, brace)
        where = f"{f}:{s[: m.start()].count(chr(10)) + 1}"
        if close is None:
            tally["skipped"] += 1
            skipped.append((where, "comment or unbalanced"))
            continue
        body = s[brace + 1 : close]
        if ".." in re.sub(r'"(?:[^"\\]|\\.)*"', "", body) and re.search(r"(^|,)\s*\.\.", body):
            tally["skipped"] += 1
            skipped.append((where, "rest pattern"))
            continue
        got = {}
        ok = True
        for part in split_top(body):
            fm = re.match(r"^(\w+)\s*:\s*(.+)$", part, re.S)
            if fm:
                got[fm.group(1)] = fm.group(2).strip()
            elif re.match(r"^\w+$", part):
                got[part] = part
            else:
                ok = False
        fields = SPECS[m.group(1)]
        if not ok or set(got) != set(fields):
            tally["skipped"] += 1
            skipped.append((where, f"fields={sorted(got)}"))
            continue
        turbo = m.group(2) or ""
        turbo = re.sub(r"\s+", "", turbo)
        call = f"{m.group(1)}{turbo}::new({', '.join(got[f] for f in fields)}).unwrap()"
        out.append(s[last : m.start()] + call)
        last = close + 1
        tally["rewritten"] += 1
        changed = True
    out.append(s[last:])
    if changed:
        tally["files"] += 1
        if "--count" not in sys.argv:
            new = "".join(out)
            with open(f, "w", encoding="utf-8", newline="") as fh:
                fh.write(new.replace("\n", "\r\n") if crlf else new)
print(tally)
for w, why in skipped:
    print("SKIPPED", w, why)
