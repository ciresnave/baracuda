#!/usr/bin/env python3
"""Bump the workspace version `0.1.0-alpha.OLD` -> `0.1.0-alpha.NEW`, and nothing else.

Usage: python scripts/bump-version.py OLD NEW      (run from the repo root; edits tracked files)

Why this exists: the alpha.76-79 bumps were done with a blind `alpha.OLD -> alpha.NEW` replace,
which rewrote historical text (Fuel's bisect at alpha.78, pins at alpha.77, ...) in docs and test
comments. This script touches only:
  * `version = "0.1.0-alpha.OLD"` lines in tracked Cargo.toml / Cargo.lock (quoted, exact, whole token);
  * the three present-tense "current version" statements (ARCHITECTURE.md, OP-MATRIX.md, README.md).
Any other `alpha.OLD` mention is history and is left alone. CHANGELOG.md is written by hand.
Enumerates with `git ls-files` (never a disk walk). Files are handled as latin-1 bytes so CRLF and
cp1252 bytes round-trip unchanged.
"""
import re
import subprocess
import sys

CURRENT_STATEMENTS = [
    # (file, regex with one group around the version token)
    ("ARCHITECTURE.md", r"(The workspace is at \*\*v)0\.1\.0-alpha\.{old}(?!\d)"),
    ("OP-MATRIX.md", r"(the workspace is now at )0\.1\.0-alpha\.{old}(?!\d)"),
    ("README.md", r"(`main` is at `)0\.1\.0-alpha\.{old}(?!\d)"),
]


def bump_cargo(text, old, new):
    pat = re.compile(r'(?m)^(\s*(?:[\w.-]+\s*=\s*\{\s*)?version\s*=\s*")0\.1\.0-alpha\.%s(")' % re.escape(old))
    return pat.sub(lambda m: f"{m.group(1)}0.1.0-alpha.{new}{m.group(2)}", text)


def bump_statement(text, regex, old, new):
    pat = re.compile(regex.replace("{old}", re.escape(old)))
    return pat.sub(lambda m: f"{m.group(1)}0.1.0-alpha.{new}", text)


def bump_file(path, text, old, new):
    """Pure: returns the bumped text for the tracked file `path`."""
    if path.endswith(("Cargo.toml", "Cargo.lock")):
        return bump_cargo(text, old, new)
    for name, regex in CURRENT_STATEMENTS:
        if path == name:
            return bump_statement(text, regex, old, new)
    return text


def main(argv):
    if len(argv) != 3 or not (argv[1].isdigit() and argv[2].isdigit()):
        print(__doc__)
        return 2
    old, new = argv[1], argv[2]
    files = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout.decode().split("\0")
    changed = 0
    for path in filter(None, files):
        if not (path.endswith(("Cargo.toml", "Cargo.lock")) or path in dict(CURRENT_STATEMENTS)):
            continue
        with open(path, "rb") as fh:
            before = fh.read().decode("latin-1")
        after = bump_file(path, before, old, new)
        if after != before:
            with open(path, "wb") as fh:
                fh.write(after.encode("latin-1"))
            changed += 1
    print(f"bumped alpha.{old} -> alpha.{new} in {changed} files (enumerated via git ls-files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
