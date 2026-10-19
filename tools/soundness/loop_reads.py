#!/usr/bin/env python3
"""Run `cargo check` for several feature sets, apply E0616 accessor fixes, repeat
until a set has no E0616 left. Prints the remaining (non-E0616) error summary."""
import collections
import json
import os
import subprocess
import sys

ROOT = os.getcwd()
SCRATCH = os.path.dirname(os.path.abspath(__file__))
SETS = {
    "A": "baracuda-kernels/sm89,baracuda-kernels/device-tests,baracuda-kernels-bench/device-tests",
    "B": "baracuda-kernels/sm90a,baracuda-kernels/device-tests,baracuda-kernels/cudnn,baracuda-kernels/fa2_backward,"
    "baracuda-kernels/mamba,baracuda-kernels/mhc,baracuda-kernels/marlin,baracuda-kernels/awq,"
    "baracuda-kernels/bnb_nf4,baracuda-kernels/xformers_blocksparse,baracuda-kernels/ozimmu",
    "C": "baracuda-kernels/sm89,baracuda-kernels/device-tests,baracuda-kernels/flashinfer,baracuda-flashinfer/device-tests",
    "E": "baracuda-kernels/sm90a,baracuda-kernels/device-tests,baracuda-kernels/cudnn,baracuda-kernels/fa2_backward,baracuda-kernels/mamba,baracuda-kernels/marlin,baracuda-kernels/awq,baracuda-kernels/bnb_nf4,baracuda-kernels/xformers_blocksparse,baracuda-kernels/ozimmu",
    "F": "baracuda-kernels/sm90a,baracuda-kernels/device-tests,baracuda-kernels/cudnn,baracuda-kernels/fa2_backward,baracuda-kernels/mamba,baracuda-kernels/marlin,baracuda-kernels/awq,baracuda-kernels/bnb_nf4,baracuda-kernels/ozimmu",
    "D": "baracuda-kernels/sm89,baracuda-kernels/device-tests,baracuda-kernels/xformers_sparse24",
}
PKGS = ["-p", "baracuda-kernels", "-p", "baracuda-kernels-types", "-p", "baracuda-kernels-bench", "-p", "baracuda-flashinfer"]
env = dict(os.environ, DOCS_RS="1", RUSTFLAGS="--cap-lints warn")

for name in sys.argv[1:] or list(SETS):
    for it in range(8):
        log = os.path.join(SCRATCH, f"loop_{name}.json")
        with open(log, "w", encoding="utf-8") as out:
            subprocess.run(
                ["cargo", "check", *PKGS, "--all-targets", "--features", SETS[name], "--message-format=json"],
                stdout=out, stderr=subprocess.DEVNULL, env=env, cwd=ROOT,
            )
        c = collections.Counter()
        first = {}
        for line in open(log, encoding="utf-8"):
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("reason") == "compiler-message" and m["message"]["level"] == "error":
                d = m["message"]
                key = (d.get("code") or {}).get("code") or d["message"][:60]
                c[key] += 1
                if d["spans"]:
                    first.setdefault(key, f'{d["spans"][0]["file_name"]}:{d["spans"][0]["line_start"]} {d["message"][:90]}')
        print(f"set {name} iter {it}: {dict(c)}", flush=True)
        if not c.get("E0616"):
            for k, v in first.items():
                print("   first", k, v, flush=True)
            break
        r = subprocess.run([sys.executable, os.path.join(SCRATCH, "fix_reads.py"), log], capture_output=True, text=True, cwd=ROOT)
        print("  ", r.stdout.strip().splitlines()[0], flush=True)
