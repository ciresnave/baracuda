#!/usr/bin/env bash
# Guard: a CR detector must be validated on THIS shell before its answer means
# anything, and the natural way to validate one is single-sided and useless.
#
# Four sessions across three repos produced four different signatures for the
# same question — "does this file have CRLF line endings?" — and every wrong
# reading came from testing a detector against ONE file:
#
#   KISS 2           grep -c $'\r'   -> 0   read as "no CRLF"        (file HAD 1854)
#   KISS architect   the corrected   -> 0   read as "grep is blind"
#   baracuda (me)    grep -c $'\r'   -> 3   read as "degenerate pattern",
#                                           which was my own shell quoting
#   portfolio PM     grep -Uc $'\r'  -> 3/3 on BOTH files
#
# ⚠️ THE ANSWERS DISAGREE ACROSS SESSIONS AND THE MECHANISM IS STILL UNIDENTIFIED.
# What every session agrees on is which form discriminates, not why the others
# fail — so this script tests the instrument rather than explaining it.
#
# THE PROTOCOL: run any candidate against a pure-LF file AND a pure-CRLF file
# and require the two answers to DIFFER. A detector that returns the same number
# on opposite inputs is blind, whatever that number is — which is why "it
# returned 0" and "it returned the line count" are the same failure wearing
# different clothes, and why a single-sided test cannot tell them apart.
#
# ⚠️ AND THE FAILURE DIRECTION IS THE DANGEROUS ONE: a blind CR detector reports
# AGREEMENT. Anything gating on "these files match after normalisation" will pass
# on files that do not.
#
# Exits non-zero if the prescribed form does not discriminate HERE, because a
# prescription that has not been checked on the running shell is a rule, not a
# measurement.
#
# Usage:
#   scripts/check-crlf-detector.sh            # validate the instrument
#   scripts/check-crlf-detector.sh FILE ...   # validate, then report on FILEs
set -uo pipefail

# The form all sessions that ran it saw discriminate. `-U` (binary; do not strip
# CR) is required. `-P` is not universally redundant: `grep -Uc CR` discriminates
# in some sessions and returns the line count in others, so both flags stay.
PRESCRIBED=(-UPc '\r')

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
printf 'a\nb\nc\n'       > "$tmp/lf.txt"
printf 'a\r\nb\r\nc\r\n' > "$tmp/crlf.txt"

echo "== fixtures (od ground truth) =="
cr_lf=$(od -c "$tmp/lf.txt"   | grep -o '\\r' | wc -l | tr -d ' ')
cr_crlf=$(od -c "$tmp/crlf.txt" | grep -o '\\r' | wc -l | tr -d ' ')
printf '  lf.txt   CR bytes: %s   (must be 0)\n' "$cr_lf"
printf '  crlf.txt CR bytes: %s   (must be 3)\n' "$cr_crlf"
# ⚠️ ASSERTED, NOT PRINTED. If the fixtures do not carry what they claim, every
# verdict below describes the wrong input — and a printed number nobody checks
# is exactly the failure this script exists to catch, committed by this script.
if [ "$cr_lf" != "0" ] || [ "$cr_crlf" != "3" ]; then
    printf 'FAIL: the fixtures are wrong (%s / %s). Nothing below is a measurement.\n' \
        "$cr_lf" "$cr_crlf"
    exit 2
fi
echo

# grep -c exits 0 (match), 1 (no match), or >1 (ERROR). Emit "<status>|<output>"
# so the caller can tell a real count from a failure.
#
# ⚠️ WITHOUT THIS SPLIT THE SCRIPT CARRIES ITS OWN VERSION OF THE BUG IT HUNTS.
# Comparing two raw outputs treats an error string as a value: if grep ERRORS on
# one fixture and answers on the other, the two "differ" and the verdict reads
# DISCRIMINATES — a PASS built on a form that never ran. Two EQUAL errors compare
# equal and read BLIND, which at least fails safe; one error and one answer does
# not. Raised by the portfolio PM, who asked whether this script distinguishes
# "answered differently" from "two failures compared equal". It did not.
count() { local out s; out=$(grep "$@" 2>&1); s=$?; printf '%s|%s' "$s" "$out"; }

status=0
echo "== candidate forms: must give DIFFERENT answers on the two fixtures =="
probe() { # probe <label> <flag...> -- <pattern>
    local label=$1; shift
    local flags=() pat
    while [ "$1" != "--" ]; do flags+=("$1"); shift; done
    shift; pat=$1
    local ra rb sa sb a b verdict
    ra=$(count "${flags[@]}" "$pat" "$tmp/lf.txt");   sa=${ra%%|*}; a=${ra#*|}
    rb=$(count "${flags[@]}" "$pat" "$tmp/crlf.txt"); sb=${rb%%|*}; b=${rb#*|}
    if [ "$sa" -gt 1 ] || [ "$sb" -gt 1 ]; then
        verdict="<<< ERRORED (exit $sa/$sb) >>>"
        a="err"; b="err"
    elif [ "$a" != "$b" ]; then
        verdict="DISCRIMINATES"
    else
        verdict="<<< BLIND >>>"
    fi
    printf '  %-22s LF=%-5s CRLF=%-5s %s\n' "$label" "$a" "$b" "$verdict"
    if [ "$label" = "$PRESCRIBED_LABEL" ] && [ "$verdict" != "DISCRIMINATES" ]; then
        status=1
    fi
    return 0
}
PRESCRIBED_LABEL="grep -UPc CR-escape"
CR=$(printf '\r')
probe "grep -c CR"           -c   -- "$CR"
probe "grep -Uc CR"          -Uc  -- "$CR"
probe "grep -Pc CR-escape"   -Pc  -- '\r'
probe "$PRESCRIBED_LABEL"    "${PRESCRIBED[@]:0:1}" -- "${PRESCRIBED[1]}"
echo

# ⚠️ printf, not echo, for every line carrying a backslash escape. `echo`'s
# handling of them is shell-dependent (bash without -e leaves them alone; a shell
# with xpg_echo expands them) — and in THIS script an expanded `\r` would emit a
# real carriage return into the advice about detecting carriage returns.
if [ "$status" -ne 0 ]; then
    printf 'FAIL: the prescribed form (grep -UPc %s) is BLIND or ERRORED on this shell.\n' "'\\r'"
    printf '      Do not use it here. Use a byte-level instrument instead:\n'
    printf '        python -c "import pathlib,sys;print(pathlib.Path(sys.argv[1]).read_bytes().count(b%s))" FILE\n' "'\\r\\n'"
    printf '        git ls-files --eol FILE          # i/<index> w/<worktree>, authoritative\n'
else
    printf 'OK: grep -UPc %s discriminates here. Safe to use in this environment.\n' "'\\r'"
fi

# Optional: report on real files using ONLY the validated instrument.
if [ "$#" -gt 0 ]; then
    echo
    printf '== reporting on %s file(s) with the validated form ==\n' "$#"
    for f in "$@"; do
        if [ ! -f "$f" ]; then printf '  %-52s (not a file)\n' "$f"; continue; fi
        r=$(count "${PRESCRIBED[@]}" "$f"); s=${r%%|*}; n=${r#*|}
        [ "$s" -gt 1 ] && n="ERR$s"
        printf '  %-52s CRLF lines=%-7s %s\n' "$f" "$n" \
            "$(git ls-files --eol -- "$f" 2>/dev/null | awk '{print $1, $2}')"
    done
    echo
    printf '  (the trailing i/... w/... is git own reading — index vs worktree —\n'
    printf '   and is authoritative regardless of what grep does on this shell.)\n'
fi

exit "$status"
