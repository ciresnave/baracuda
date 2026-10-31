#!/usr/bin/env python3
"""Fail if any tracked text file contains a raw NUL byte.

A backslash-zero mangled by editing tooling becomes a real NUL that still compiles
(Rust byte strings, Markdown) and only shows up as grep's "Binary file matches".
Enumerates from the index (`git ls-files`), skips known-binary extensions, and
prints the population so a scan of nothing cannot pass silently.
"""
import subprocess
import sys

BINARY = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".a", ".lib", ".dll",
          ".so", ".bin", ".cubin", ".woff", ".woff2", ".ttf", ".npy", ".npz", ".safetensors", ".jar")


def main() -> int:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    files = [f.decode() for f in out.split(b"\x00") if f]
    scanned, bad = 0, []
    for f in files:
        if f.lower().endswith(BINARY):
            continue
        try:
            data = open(f, "rb").read()
        except OSError:
            continue
        scanned += 1
        if b"\x00" in data:
            bad.append((f, data.count(b"\x00")))
    print(f"check-no-nul-bytes: {len(files)} tracked, {scanned} scanned as text, {len(bad)} with NUL")
    if scanned == 0:
        print("FAIL: scanned nothing", file=sys.stderr)
        return 1
    for f, n in bad:
        print(f"FAIL: {f}: {n} NUL byte(s)", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
