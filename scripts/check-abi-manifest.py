#!/usr/bin/env python3
"""C-ABI manifest guard for baracuda-kernels-sys and baracuda-cutlass-kernels-sys.

The C ABI is an independent integer (`ABI_VERSION` in baracuda-kernels-sys/src/lib.rs, mirrored as
BARACUDA_KERNELS_ABI_VERSION in kernels/include/baracuda_abi.h and returned by the exported
`baracuda_kernels_abi_version()`), bumped by hand ONLY when the ABI breaks. This script keeps that
honest:

  1. the Rust constant equals the C header value (single source, asserted);
  2. the checked-in manifest (abi/abi-manifest.tsv) lists every exported symbol, `#[repr(C)]` type
     and `pub const` of both crates with a hash of its normalised text, and equals what the sources
     say now (additions need the manifest regenerated with --update);
  3. with --base REF: no entry of REF's manifest was removed or changed unless ABI_VERSION is
     higher than REF's. Adding entries never needs a bump.

What it cannot see: a behavioural/documented-contract change that leaves the text of a signature
alone (aliasing, stride meaning, workspace formula). Those still need a human bump.

Usage:
  python scripts/check-abi-manifest.py                 # check
  python scripts/check-abi-manifest.py --base REF      # check + breaking-change gate against REF
  python scripts/check-abi-manifest.py --update        # rewrite the manifest from the sources
"""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRATES = ["baracuda-kernels-sys", "baracuda-cutlass-kernels-sys"]
LIB_RS = "crates/baracuda-kernels-sys/src/lib.rs"
HEADER = "crates/baracuda-kernels-sys/kernels/include/baracuda_abi.h"
MANIFEST = "abi/abi-manifest.tsv"
# Entries that are the version mechanism itself, not part of the frozen surface.
# Only items WE export are frozen: names starting with `baracuda` (any case). The vendor mirrors in the
# same files (cublas*, cudnn*, curand*, CUFFT_*, ...) are imports of someone else's ABI (README 4.4).
SKIP = {("const", "ABI_VERSION"), ("type", "AbiMismatch")}


def read(path):
    with open(os.path.join(ROOT, path), "rb") as f:
        return f.read().replace(b"\r\n", b"\n").decode("utf-8", "replace")


def strip_comments(s):
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.S)
    return re.sub(r"//[^\n]*", "", s)


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def match_brace(s, i):
    """Index just past the `}` matching the `{` at s[i]."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced braces")


def extract_fns(s):
    """`pub fn` / `pub static` declarations inside `extern "C"` blocks."""
    for m in re.finditer(r'(?:unsafe\s+)?extern\s+"C"\s*\{', s):
        end = match_brace(s, m.end() - 1)
        body = s[m.end():end - 1]
        for f in re.finditer(r"pub\s+fn\s+(\w+)\s*(<[^>]*>)?\s*\(([^;{]*)\)\s*(->[^;{]*)?;", body):
            yield "fn", f.group(1), norm(f"({f.group(3)}) {f.group(4) or ''}")
        for st in re.finditer(r"pub\s+static\s+(?:mut\s+)?(\w+)\s*:([^;]*);", body):
            yield "static", st.group(1), norm(st.group(2))


def repr_type_text(s, m):
    """Text of one `#[repr(..)] pub struct/enum/union` item, through its closing brace or `;`."""
    brace = s.find("{", m.end())
    semi = s.find(";", m.end())
    if semi != -1 and (brace == -1 or semi < brace):  # unit / tuple struct
        return s[m.start():semi + 1]
    return s[m.start():match_brace(s, brace)]


def extract_types(s):
    """`#[repr(C...)]` / `#[repr(<int>)]` types and top-level `pub type` aliases."""
    for m in re.finditer(r"#\[repr\(([^)]*)\)\]\s*(?:#\[[^\]]*\]\s*)*pub\s+(struct|enum|union)\s+(\w+)", s):
        yield "type", m.group(3), norm(repr_type_text(s, m))
    for m in re.finditer(r"(?m)^pub\s+type\s+(\w+)\s*=\s*([^;]+);", s):
        yield "type", m.group(1), norm(m.group(2))


def extract_consts(s):
    """Top-level `pub const` values (status codes, enum values)."""
    for m in re.finditer(r"(?m)^pub\s+const\s+(\w+)\s*:\s*([^=]+)=\s*([^;]+);", s):
        yield "const", m.group(1), norm(f"{m.group(2)} = {m.group(3)}")


def extract(src):
    """Yield (kind, name, normalised text) for the ABI-relevant items of one source file."""
    s = strip_comments(src)
    yield from extract_fns(s)
    yield from extract_types(s)
    yield from extract_consts(s)


def collect():
    entries = {}
    for crate in CRATES:
        d = os.path.join(ROOT, "crates", crate, "src")
        for name in sorted(os.listdir(d)):
            if not name.endswith(".rs"):
                continue
            for kind, ident, text in extract(read(f"crates/{crate}/src/{name}")):
                if (kind, ident) in SKIP or not ident.lower().startswith("baracuda"):
                    continue
                entries.setdefault((kind, crate, ident), []).append(text)
    return {k: hashlib.sha256("\x1f".join(sorted(v)).encode()).hexdigest()[:16] for k, v in entries.items()}


def rust_version():
    m = re.search(r"pub const ABI_VERSION: u32 = (\d+);", read(LIB_RS))
    if not m:
        sys.exit("FAIL: `pub const ABI_VERSION: u32 = N;` not found in " + LIB_RS)
    return int(m.group(1))


def header_version():
    m = re.search(r"#define\s+BARACUDA_KERNELS_ABI_VERSION\s+(\d+)u?", read(HEADER))
    if not m:
        sys.exit("FAIL: BARACUDA_KERNELS_ABI_VERSION not found in " + HEADER)
    return int(m.group(1))


def git_show(ref, path):
    """`git show REF:PATH` as text, or None when the path does not exist at REF.

    The argv is a fixed list (no shell) and REF is a commit SHA or ref name chosen by the CI
    workflow, never user input; git is resolved once so a missing git fails clearly.
    """
    git = shutil.which("git")
    if git is None:
        sys.exit("FAIL: git not found on PATH (needed for --base)")
    r = subprocess.run([git, "show", f"{ref}:{path}"], cwd=ROOT, check=False, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"note: no {path} at {ref} ({r.stderr.strip()}); breaking-change gate skipped.")
        return None
    return r.stdout


def parse_manifest(text):
    version, entries = None, {}
    for line in text.splitlines():
        if line.startswith("# abi_version="):
            version = int(line.split("=", 1)[1])
        elif line and not line.startswith("#"):
            kind, crate, ident, h = line.split("\t")
            entries[(kind, crate, ident)] = h
    return version, entries


def render(version, entries):
    out = [
        f"# abi_version={version}",
        "# format=1: kind<TAB>crate<TAB>name<TAB>sha256[:16] of the whitespace-normalised, comment-free text.",
        "# Generated by scripts/check-abi-manifest.py --update. Do not edit by hand; see the script header.",
    ]
    out += ["\t".join((k[0], k[1], k[2], h)) for k, h in sorted(entries.items())]
    return "\n".join(out) + "\n"


def diff(old, new):
    removed = sorted(k for k in old if k not in new)
    changed = sorted(k for k in old if k in new and old[k] != new[k])
    added = sorted(k for k in new if k not in old)
    return removed, changed, added


def show(label, keys, limit=8):
    if keys:
        print(f"  {label} ({len(keys)}): " + ", ".join(k[2] for k in keys[:limit]) + (" ..." if len(keys) > limit else ""))


def check_versions(ver, hver, mver):
    """The Rust constant, the C header and the manifest header must all agree."""
    failed = False
    if ver != hver:
        print(f"FAIL: ABI_VERSION in {LIB_RS} is {ver} but {HEADER} says {hver}; they are one value.")
        failed = True
    if mver != ver:
        print(f"FAIL: {MANIFEST} records abi_version={mver} but ABI_VERSION is {ver}; regenerate with --update.")
        failed = True
    return failed


def check_manifest(ver, mver, old, cur):
    """The checked-in manifest must equal what the sources say now."""
    removed, changed, added = diff(old, cur)
    if not (removed or changed or added):
        return False
    print(f"FAIL: {MANIFEST} is out of date against the sources. Regenerate with --update.")
    show("removed", removed)
    show("changed", changed)
    show("added", added)
    if (removed or changed) and mver == ver:
        print("  removed/changed entries are an ABI BREAK: bump ABI_VERSION (lib.rs AND baracuda_abi.h), "
              "then --update, and add a CHANGELOG line.")
    return True


def check_base(base_ref, ver, cur):
    """Breaking-change gate: nothing in the base manifest may be removed or changed without a bump."""
    base_text = git_show(base_ref, MANIFEST)
    if base_text is None:
        return False
    bver, base = parse_manifest(base_text)
    brem, bchg, badd = diff(base, cur)
    print(f"base {base_ref}: abi_version={bver}, {len(base)} entries; now {ver}, {len(cur)} "
          f"(+{len(badd)} -{len(brem)} ~{len(bchg)})")
    failed = False
    if (brem or bchg) and ver <= bver:
        print(f"FAIL: ABI break against {base_ref} without bumping ABI_VERSION ({bver} -> {ver}).")
        show("removed", brem)
        show("changed", bchg)
        failed = True
    if ver < bver:
        print(f"FAIL: ABI_VERSION went DOWN ({bver} -> {ver}).")
        failed = True
    return failed


def write_manifest(ver, cur):
    with open(os.path.join(ROOT, MANIFEST), "w", newline="\n") as f:
        f.write(render(ver, cur))
    print(f"wrote {MANIFEST}: abi_version={ver}, {len(cur)} entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--base", help="git ref whose manifest is the baseline for the breaking-change gate")
    args = ap.parse_args()

    ver = rust_version()
    cur = collect()
    if not cur:
        sys.exit("FAIL: extracted 0 ABI entries; the parser is broken (this guard would assert nothing).")
    if args.update:
        write_manifest(ver, cur)
        return

    mver, old = parse_manifest(read(MANIFEST))
    failed = check_versions(ver, header_version(), mver)
    failed = check_manifest(ver, mver, old, cur) or failed
    if args.base:
        failed = check_base(args.base, ver, cur) or failed
    if failed:
        sys.exit(1)
    print(f"ok: abi_version={ver}, {len(cur)} entries, header and manifest agree")


if __name__ == "__main__":
    main()
