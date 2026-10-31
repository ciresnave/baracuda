#!/usr/bin/env python3
"""Fail if any tracked text file contains a raw NUL byte.

A backslash-zero mangled by editing tooling becomes a real NUL that still compiles
(Rust byte strings, Markdown) and only shows up as grep's "Binary file matches".
Enumerates from the index (`git ls-files`), skips known-binary extensions, and
prints the population so a scan of nothing cannot pass silently.
"""
import subprocess  # nosec B404 - fixed argv list to `git`, never a shell, never user input
import sys

BINARY = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".a", ".lib", ".dll",
          ".so", ".bin", ".cubin", ".woff", ".woff2", ".ttf", ".npy", ".npz", ".safetensors", ".jar")
NUL = b"\x00"


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout  # nosec B603 B607 - fixed argv, `git` from PATH on purpose
    return [f.decode() for f in out.split(NUL) if f]


def nul_count(path: str):
    """NUL count of a readable file, or None if it cannot be read."""
    try:
        with open(path, "rb") as fh:
            return fh.read().count(NUL)
    except OSError:
        return None


def scan(files: list[str]):
    scanned, bad = 0, []
    for f in files:
        if f.lower().endswith(BINARY):
            continue
        n = nul_count(f)
        if n is None:
            continue
        scanned += 1
        if n:
            bad.append((f, n))
    return scanned, bad


def main() -> int:
    files = tracked_files()
    scanned, bad = scan(files)
    print(f"check-no-nul-bytes: {len(files)} tracked, {scanned} scanned as text, {len(bad)} with NUL")
    if scanned == 0:
        print("FAIL: scanned nothing", file=sys.stderr)
        return 1
    for f, n in bad:
        print(f"FAIL: {f}: {n} NUL byte(s)", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
