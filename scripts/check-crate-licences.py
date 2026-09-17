#!/usr/bin/env python3
"""Guard: every publishable crate ships the licence text its `license` field names.

The workspace licence is `MIT OR Apache-2.0`, and both licences require their
text to travel with every copy. crates.io only packages files that live inside a
crate's own directory, so the root `LICENSE-MIT` / `LICENSE-APACHE` are NOT in
any published crate unless each crate directory carries its own copy.

That is how 0.0.1-alpha.79 shipped 69 crates with no licence text at all: the
per-crate copies were added after it (#18). It is also how
`baracuda-cuda-vocab`, created a week after that sweep, came to lack them at
the alpha.80 tag while the other 69 crates had them. Nothing checked.

This guard fails when a publishable crate directory lacks either file, or holds
a copy whose git blob differs from the root file's.

- The crate set comes from `cargo metadata`, filtered exactly as
  `scripts/publish.ps1` filters it (`publish` absent, or a non-empty
  allowlist), so the guard and the publisher see the same population.
- File identity comes from the git tree of REV (default `HEAD`), not from the
  disk, so an untracked or locally edited copy cannot satisfy it.

Usage: python3 scripts/check-crate-licences.py [REV]
Exit status: 0 when every crate is covered, 1 when a crate is not, 2 when the
guard cannot run.
"""

import json
import os
import subprocess
import sys

LICENCES = ("LICENSE-MIT", "LICENSE-APACHE")


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def same_path(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def publishable_crates(meta: dict) -> list:
    """The crates `scripts/publish.ps1` uploads: `publish` absent, or a non-empty allowlist."""
    crates = [p for p in meta["packages"] if p["publish"] is None or len(p["publish"]) > 0]
    return sorted(crates, key=lambda p: p["name"])


def tree_blobs(root: str, rev: str) -> dict:
    """Map every path in the git tree at REV to its blob id."""
    blobs = {}
    for entry in run("git", "-C", root, "ls-tree", "-r", "-z", rev).split("\0"):
        if entry:
            info, path = entry.split("\t", 1)
            blobs[path] = info.split()[2]
    return blobs


def licence_problems(crates: list, root: str, blobs: dict, root_blobs: dict) -> list:
    problems = []
    for pkg in crates:
        crate_dir = os.path.relpath(os.path.dirname(pkg["manifest_path"]), root)
        crate_dir = crate_dir.replace(os.sep, "/")
        for name in LICENCES:
            blob = blobs.get(f"{crate_dir}/{name}")
            if blob is None:
                problems.append(f"{pkg['name']}: {crate_dir}/{name} is missing")
            elif blob != root_blobs[name]:
                problems.append(f"{pkg['name']}: {crate_dir}/{name} differs from the root {name}")
    return problems


def main() -> int:
    rev = sys.argv[1] if len(sys.argv) > 1 else "HEAD"

    meta = json.loads(run("cargo", "metadata", "--no-deps", "--format-version", "1"))
    root = meta["workspace_root"]
    toplevel = run("git", "-C", root, "rev-parse", "--show-toplevel").strip()
    if not same_path(root, toplevel):
        print(f"cannot run: workspace root {root!r} is not the git toplevel {toplevel!r}")
        return 2

    crates = publishable_crates(meta)
    if not crates:
        print("cannot run: cargo metadata lists no publishable crates")
        return 2

    blobs = tree_blobs(root, rev)
    root_blobs = {name: blobs.get(name) for name in LICENCES}
    absent = [name for name, blob in root_blobs.items() if blob is None]
    if absent:
        print(f"cannot run: missing from the root of the tree at {rev}: {', '.join(absent)}")
        return 2

    print(
        f"enumeration: cargo metadata publishable set ({len(crates)} crates); "
        f"files read from the git tree at {rev}"
    )

    problems = licence_problems(crates, root, blobs, root_blobs)
    if problems:
        print(f"FAIL: {len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        print(f"Fix: copy the root {' and '.join(LICENCES)} into each crate directory named above.")
        return 1

    print(f"ok: all {len(crates)} publishable crates carry both licence files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
