#!/usr/bin/env bash
# Guard: documentation must not instruct a raw `cargo test -- --ignored` device run.
#
# CONTRIBUTING step 3 is explicit: on-device tests go through `cargo gpu-test`, which
# holds the machine-wide `gpu-run` mutex, because the box is SHARED with other
# projects and concurrent device runs collide. That is not a preference — it is the
# 2026-07-31 host-aperture postmortem. `.cargo/config.toml` states the reasoning for
# the alias in one sentence:
#
#     lock acquisition is structural (the xtask subcommand *is* the wrapper call),
#     so a device run cannot silently skip the mutex — the "convention in a docs
#     file is not a control" failure the 2026-07-31 host-aperture postmortem indicts.
#
# ⚠ And then four docs files carried exactly that convention, in the forbidden form.
# Measured 2026-09-06 at main `eb63561a`:
#
#     README.md:546                         "Full GPU integration sweep"
#     crates/baracuda-kernels/README.md:238 "run them with … on a host with a driver"
#     crates/baracuda-cuvs/README.md:81     "Run the hardware tests once installed"
#     crates/baracuda-tensorrt/AUDIT.md:147 "Run: …"
#
# All four are unambiguously device-run instructions. The rule lived in CONTRIBUTING;
# a reader asking "how do I run these tests" opens the crate README — so the
# lower-authority, higher-traffic document was the one that won, and following it
# could corrupt another project's GPU run on the shared box.
#
# ⚠ COMPLIANCE CONTAINS THE FORBIDDEN TOKEN, which is why this cannot be a plain
# grep. The sanctioned wrapped form in
# docs/superpowers/plans/2026-08-01-ir-contraction-roles-layout.md reads
#
#     pwsh scripts/gpu-run.ps1 -Project baracuda -- cargo test … -- --ignored
#
# and is CORRECT: the mutex is held by the wrapper. A guard that flagged every
# `-- --ignored` would flag compliance and absence identically.
#
# NOT scanned: docs/superpowers/plans/. Those are records of how work was done, not
# instructions to a current reader — one of them invokes `-p baracuda-kernelgen`, a
# crate that no longer exists in this repository, which is what a historical document
# looks like. Rewriting them would edit history to satisfy a guard.
set -euo pipefail

# CONTRIBUTING.md is where the rule LIVES: it must quote the forbidden command in
# order to forbid it. Exempt, with honesty check 3 below keeping the exemption honest.
RULE_FILE="CONTRIBUTING.md"
PROHIBITION="must not be used for on-device runs"

scanned=0
violations=()

while IFS= read -r f; do
  case "$f" in
    docs/superpowers/plans/*) continue ;;
    "$RULE_FILE") continue ;;
  esac
  # `git ls-files` lists the INDEX, which can name a path that is not on disk — a
  # staged deletion, or an intent-to-add whose file was removed. `cat` on one of those
  # fails, and under `set -e` that kills the guard rather than reporting anything.
  # Measured while exercising this file's own probes.
  [ -f "$f" ] || continue
  scanned=$((scanned + 1))

  # Read the file once into a variable rather than piping into an early-exiting
  # consumer: under `set -o pipefail` a `grep -q` that closes the pipe gives the
  # producer SIGPIPE (141) and pipefail promotes it. That exact failure was measured
  # on ubuntu-latest in PR #27 — see scripts/check-test-crate-locality.sh.
  body=$(cat "$f")

  while IFS= read -r line; do
    # Only lines that actually invoke a device run.
    case "$line" in
      *"-- --ignored"*) ;;
      *) continue ;;
    esac
    # Wrapped forms are the sanctioned ones.
    case "$line" in
      *"gpu-run.ps1"*|*"cargo gpu-test"*|*"xtask test-gpu"*) continue ;;
    esac
    # ⚠ A line that NAMES the raw form in order to FORBID it is not an instruction to
    # use it, and a doc warning its readers off the raw form is the behaviour this
    # guard wants more of. Without this arm the guard fires on its own remedy: the
    # four fixed docs each say "raw `cargo test -- --ignored` bypasses it and must not
    # be used", and the first run after fixing them reported all four as violations.
    # Same shape as a retraction that has to quote what it retracts.
    case "$line" in
      *"must not"*|*"bypasses"*|*"do not use"*|*"never use"*) continue ;;
    esac
    violations+=("$f: $line")
  done <<< "$body"
done < <(git ls-files -- '*.md')

# Honesty 1: an empty walk is a broken guard, not a clean tree.
if [ "$scanned" -eq 0 ]; then
  echo "GUARD BROKEN: scanned 0 markdown files — wrong cwd, or git ls-files found nothing."
  exit 2
fi

# Honesty 2: the rule's home must exist, or the exemption above is silently covering
# a file that is not there.
if [ ! -f "$RULE_FILE" ]; then
  echo "GUARD BROKEN: $RULE_FILE not found — the exempted rule-holder is missing."
  exit 2
fi

# Honesty 3: the exemption is earned only while CONTRIBUTING still STATES the rule.
# If the prohibition is ever dropped, exempting the file would hide the fact that the
# whole guard has lost its basis — the same shape as check-test-crate-locality.sh's
# "every exempt file must still be a would-be violation".
if ! grep -qF -- "$PROHIBITION" "$RULE_FILE"; then
  echo "GUARD STALE: $RULE_FILE no longer contains the prohibition"
  echo "  (\"$PROHIBITION\")."
  echo "Either the rule changed — in which case this guard should change or go — or it"
  echo "was lost. Do not just re-word this check to match."
  exit 2
fi

if [ "${#violations[@]}" -gt 0 ]; then
  echo "RAW DEVICE-RUN INSTRUCTION — these docs tell a reader to run device tests"
  echo "without the machine-wide gpu-run mutex:"
  printf '  %s\n' "${violations[@]}"
  echo
  echo "Use \`cargo gpu-test -p <crate> [cargo test args]\` instead: it compiles"
  echo "unlocked, then runs the #[ignore]d tests while HOLDING the lock. Raw"
  echo "\`cargo test -- --ignored\` bypasses it, and the box is shared with other"
  echo "projects (2026-07-31 host-aperture postmortem). A wrapped invocation via"
  echo "scripts/gpu-run.ps1 is also accepted."
  exit 1
fi

echo "device-run-invocations OK: scanned $scanned markdown files (plans/ and $RULE_FILE excluded), 0 raw on-device invocations."
