#!/usr/bin/env python3
"""Phrase search over Rust doc comments that is not blind to line wraps.

WHY THIS EXISTS
===============

A phrase in a `///` doc comment is wrapped by whoever wrote it, so the bytes in
the file are not the bytes a reader sees. `rustfmt` does not reflow doc text,
so the wrap point is wherever the author's line ran out:

    /// §3a.4; Baracuda reconcile §2). For scalar-param ops the value is **not
    /// baked** — it identifies the slot the emitted `extract:` path points at

The literal text between "not" and "baked" is `\\n/// `. A line-oriented
`grep 'not baked'` cannot match it, and — this is the part that bites — it
returns **0**, which is indistinguishable from the phrase being absent.

⚠️ MEASURED 2026-09-09, TWO PROJECTS, SAME PHRASE, SAME NIGHT. Both the fuel
architect and I ran `grep 'not baked'` against `fuel-kernel-seam-types`, both
got 0, and the phrase is at `src/lib.rs:242`. One of us was about to send the
other a correction built on that zero.

    grep -rc 'not baked' fuel-kernel-seam-types/src   ->  0    (phrase IS there)
    this tool, same tree                              ->  1

THE GENERAL SHAPE
=================

⚠️ A TEXT SEARCH THAT CANNOT SPAN A LINE BREAK REPORTS ABSENCE FOR ANY PHRASE
THE AUTHOR HAPPENED TO WRAP. The failure is silent, returns a plausible number,
and correlates with nothing the searcher controls — the wrap point depends on
how long the *preceding* words were.

So this joins `///`, `//!` and `//` continuations before matching, and ships a
`--self-test` that builds a deliberately wrapped fixture and requires the naive
form to MISS it and this form to FIND it. A tool that cannot demonstrate the
failure it exists to prevent is a claim, not an instrument.

USAGE
=====

    tools/docgrep.py --self-test                 # validate the tool
    tools/docgrep.py PHRASE PATH [PATH ...]      # search (exit 1 if no hits)
"""

from __future__ import annotations

import pathlib
import re
import sys
import tempfile

CONT = re.compile(r"\n\s*(///|//!|//)\s?")


def flatten(text: str) -> str:
    """Join comment continuations so a wrapped phrase becomes contiguous."""
    return CONT.sub(" ", text)


def search(phrase: str, roots: list[str]) -> list[tuple[pathlib.Path, str]]:
    # Any run of whitespace in the needle matches any run in the haystack, so
    # the caller writes the phrase as a reader sees it.
    rx = re.compile(re.escape(phrase).replace(r"\ ", r"\s+"), re.I)
    out: list[tuple[pathlib.Path, str]] = []
    for root in roots:
        p = pathlib.Path(root)
        files = [p] if p.is_file() else [f for f in p.rglob("*.rs") if "target" not in f.parts]
        for f in files:
            try:
                flat = flatten(f.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                continue
            for m in rx.finditer(flat):
                lo, hi = max(0, m.start() - 50), m.end() + 50
                out.append((f, flat[lo:hi].replace("\n", " ")))
    return out


def self_test() -> int:
    """The tool must demonstrate the failure it prevents, on a fixture."""
    wrapped = (
        "/// some preamble text that runs on and on until the value is **not\n"
        "/// baked** and the sentence continues here.\n"
        "pub struct X;\n"
    )
    with tempfile.TemporaryDirectory() as d:
        f = pathlib.Path(d) / "fixture.rs"
        f.write_text(wrapped, encoding="utf-8")
        raw = f.read_text(encoding="utf-8")

        naive = sum(1 for line in raw.splitlines() if "not baked" in line)
        ours = len(search("not baked", [str(f)]))

        # And a control the OTHER way: an UNwrapped phrase both forms must find,
        # so "ours found it" is not just "ours matches everything".
        f2 = pathlib.Path(d) / "flat.rs"
        f2.write_text("/// the value is not baked at all\npub struct Y;\n", encoding="utf-8")
        naive2 = sum(1 for line in f2.read_text(encoding="utf-8").splitlines() if "not baked" in line)
        ours2 = len(search("not baked", [str(f2)]))

        print("  fixture: phrase WRAPPED across a `///` continuation")
        print(f"    line-oriented search : {naive}   (expect 0 — this is the blindness)")
        print(f"    this tool            : {ours}   (expect 1)")
        print("  control: phrase on ONE line")
        print(f"    line-oriented search : {naive2}   (expect 1)")
        print(f"    this tool            : {ours2}   (expect 1)")

        ok = naive == 0 and ours == 1 and naive2 == 1 and ours2 == 1
        print("\n" + ("OK: the tool finds what a line search misses, and both "
                      "agree when nothing is wrapped."
                      if ok else
                      "FAIL: the self-test did not reproduce the expected pattern."))
        return 0 if ok else 1


def main(argv: list[str]) -> int:
    if len(argv) >= 1 and argv[0] == "--self-test":
        return self_test()
    if len(argv) < 2:
        print(__doc__)
        return 2
    hits = search(argv[0], argv[1:])
    for f, ctx in hits:
        print(f"  {f}: ...{ctx}...")
    print(f"  TOTAL {len(hits)}")
    # ⚠️ A zero here means "not found by a search that CAN span wraps". That is
    # a stronger statement than grep's zero, and it is still not proof of
    # absence — run --self-test to confirm the tool works in this environment
    # before treating any zero as a finding.
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
