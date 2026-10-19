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
import subprocess  # nosec B404 - fixed argv lists to `cargo`, never a shell, never user input
import sys
import time

# (crate, feature) -> why it cannot be built without the CUDA toolkit / hardware.
# Empty on purpose: add an entry with a reason, do not skip silently.
NEEDS_TOOLKIT: dict[tuple[str, str], str] = {}


def workspace_features():
    """All (crate, feature) pairs of the workspace, from `cargo metadata`."""
    meta = json.loads(
        subprocess.check_output(  # nosec B603 B607 - fixed argv, `cargo` from PATH on purpose
            ["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"], text=True
        )
    )
    members = set(meta["workspace_members"])
    pairs = []
    for pkg in sorted(meta["packages"], key=lambda p: p["name"]):
        if pkg["id"] in members:
            pairs.extend((pkg["name"], f) for f in sorted(pkg["features"]) if f != "default")
    return pairs


def option_values(argv, flag):
    """Every value that follows `flag` in argv."""
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag]


def select_pairs(argv):
    """Apply --crate filters and the --shard K/N partition."""
    only = option_values(argv, "--crate")
    pairs = [(c, f) for c, f in workspace_features() if not only or c in only]
    for spec in option_values(argv, "--shard"):
        k, n = (int(x) for x in spec.split("/"))
        if not 1 <= k <= n:
            raise SystemExit("--shard K/N needs 1 <= K <= N")
        pairs = pairs[k - 1 :: n]
    return pairs


def check_pair(crate, feature, env, targets):
    """Run one `cargo check`; return (ok, first error lines)."""
    cmd = ["cargo", "check", "--locked", "-p", crate, "--features", feature, "--message-format", "short", *targets]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)  # nosec B603 B607
    errors = [line for line in r.stderr.splitlines() if "error" in line][:8]
    return r.returncode == 0, errors


def report(checked, failed, skipped, started):
    """Print the summary; the exit status is derived from `failed`."""
    print(
        f"\nchecked {checked} (crate, feature) pairs in {time.time() - started:.0f}s; "
        f"{len(failed)} failed; {len(skipped)} declared toolkit-only"
    )
    for c, f, why in skipped:
        print(f"  NOT BUILT {c} --features {f}: {why}")
    for c, f in failed:
        print(f"  FAILED    {c} --features {f}")


def main(argv):
    pairs = select_pairs(argv)
    if "--list" in argv:
        for c, f in pairs:
            print(f"{c} {f}")
        return 0
    env = dict(os.environ, DOCS_RS="1")
    targets = [] if "--lib-only" in argv else ["--all-targets"]
    failed, skipped = [], []
    started = time.time()
    for i, (c, f) in enumerate(pairs, 1):
        if (c, f) in NEEDS_TOOLKIT:
            skipped.append((c, f, NEEDS_TOOLKIT[(c, f)]))
            continue
        ok, errors = check_pair(c, f, env, targets)
        print(f"[{i}/{len(pairs)}] {'ok' if ok else 'FAIL':4} {c} --features {f}", flush=True)
        if not ok:
            failed.append((c, f))
            for line in errors:
                print("      " + line, flush=True)
    report(len(pairs) - len(skipped), failed, skipped, started)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
