#!/usr/bin/env python3
"""Compile every declared cargo feature of every workspace crate, one at a time.

Why: a feature that gates code is only as good as the job that builds it. Three
`baracuda-kernels` features (`mhc`, `xformers_blocksparse`, `xformers_sparse24`)
did not compile for months because every CI step enabled arch features and never
those, and nothing reported the blind spot. The feature set is DERIVED from
`cargo metadata`, so a newly declared feature joins this check automatically.

Each (crate, feature) runs `cargo check --locked -p CRATE --features FEATURE`
(`--all-targets` unless --lib-only). With DOCS_RS=1 the nvcc crates skip their
forge, so this needs no CUDA toolkit. A feature that genuinely cannot build on a
toolkit-less runner must be listed in NEEDS_TOOLKIT with the reason, never
dropped silently; the script prints that list so the blindness stays visible.

Usage:  scripts/check-each-feature.py [--crate NAME ...] [--list] [--lib-only] [--shard K/N]
`--shard K/N` (1-based) checks every N-th (crate, feature) pair starting at K, so CI can
split the ~170 pairs across N jobs; the shards partition the full list.
Exit 1 if any (crate, feature) fails to compile.
"""
import json
import os
import subprocess
import sys
import time

# (crate, feature) -> why it cannot be built without the CUDA toolkit / hardware.
# Empty on purpose: add an entry with a reason, do not skip silently.
NEEDS_TOOLKIT: dict[tuple[str, str], str] = {}


def workspace_features():
    meta = json.loads(
        subprocess.check_output(["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"], text=True)
    )
    ws = set(meta["workspace_members"])
    out = []
    for p in sorted(meta["packages"], key=lambda p: p["name"]):
        if p["id"] not in ws:
            continue
        for f in sorted(p["features"]):
            if f != "default":
                out.append((p["name"], f))
    return out


def main(argv):
    only = [argv[i + 1] for i, a in enumerate(argv) if a == "--crate"]
    pairs = [(c, f) for c, f in workspace_features() if not only or c in only]
    for i, a in enumerate(argv):
        if a == "--shard":
            k, n = (int(x) for x in argv[i + 1].split("/"))
            assert 1 <= k <= n, "--shard K/N needs 1 <= K <= N"
            pairs = pairs[k - 1 :: n]
    if "--list" in argv:
        for c, f in pairs:
            print(f"{c} {f}")
        return 0
    env = dict(os.environ, DOCS_RS="1")
    targets = [] if "--lib-only" in argv else ["--all-targets"]
    failed, skipped = [], []
    t0 = time.time()
    for i, (c, f) in enumerate(pairs, 1):
        if (c, f) in NEEDS_TOOLKIT:
            skipped.append((c, f, NEEDS_TOOLKIT[(c, f)]))
            continue
        cmd = ["cargo", "check", "--locked", "-p", c, "--features", f, "--message-format", "short", *targets]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        status = "ok" if r.returncode == 0 else "FAIL"
        print(f"[{i}/{len(pairs)}] {status:4} {c} --features {f}", flush=True)
        if r.returncode != 0:
            failed.append((c, f))
            errs = [l for l in r.stderr.splitlines() if "error" in l][:8]
            for l in errs:
                print("      " + l, flush=True)
    print(f"\nchecked {len(pairs) - len(skipped)} (crate, feature) pairs in {time.time() - t0:.0f}s; "
          f"{len(failed)} failed; {len(skipped)} declared toolkit-only")
    for c, f, why in skipped:
        print(f"  NOT BUILT {c} --features {f}: {why}")
    for c, f in failed:
        print(f"  FAILED    {c} --features {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
