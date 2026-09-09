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
printf '  lf.txt   CR bytes: %s\n' "$(od -c "$tmp/lf.txt"   | grep -o '\\r' | wc -l | tr -d ' ')"
printf '  crlf.txt CR bytes: %s\n' "$(od -c "$tmp/crlf.txt" | grep -o '\\r' | wc -l | tr -d ' ')"
echo

# grep -c exits 1 when the count is 0; capture the number, keep stderr.
count() { local out; out=$(grep "$@" 2>&1); local s=$?; [ $s -le 1 ] && echo "$out" || echo "ERR$s"; }

CR=$(printf '\r')
status=0
echo "== candidate forms: must give DIFFERENT answers on the two fixtures =="
probe() { # probe <label> <flag...> -- <pattern>
    local label=$1; shift
    local flags=() pat
    while [ "$1" != "--" ]; do flags+=("$1"); shift; done
    shift; pat=$1
    local a b verdict
    a=$(count "${flags[@]}" "$pat" "$tmp/lf.txt")
    b=$(count "${flags[@]}" "$pat" "$tmp/crlf.txt")
    if [ "$a" != "$b" ]; then verdict="DISCRIMINATES"; else verdict="<<< BLIND >>>"; fi
    printf '  %-22s LF=%-5s CRLF=%-5s %s\n' "$label" "$a" "$b" "$verdict"
    [ "$label" = "grep -UPc '\\r'" ] && [ "$verdict" != "DISCRIMINATES" ] && status=1
    return 0
}
probe "grep -c CR"       -c   -- "$CR"
probe "grep -Uc CR"      -Uc  -- "$CR"
probe "grep -Pc '\\r'"    -Pc  -- '\r'
probe "grep -UPc '\\r'"   -UPc -- '\r'
echo

if [ "$status" -ne 0 ]; then
    echo "FAIL: the prescribed form (grep -UPc '\\r') is BLIND on this shell."
    echo "      Do not use it here. Use a byte-level instrument instead:"
    echo "        python -c \"import pathlib,sys;b=pathlib.Path(sys.argv[1]).read_bytes();print(b.count(b'\\r\\n'))\" FILE"
    echo "        git ls-files --eol FILE          # i/<index> w/<worktree>, authoritative"
else
    echo "OK: grep -UPc '\\r' discriminates here. Safe to use in this environment."
fi

# Optional: report on real files using ONLY the validated instrument.
if [ "$#" -gt 0 ]; then
    echo
    echo "== reporting on ${#} file(s) with the validated form =="
    for f in "$@"; do
        if [ ! -f "$f" ]; then printf '  %-52s (not a file)\n' "$f"; continue; fi
        printf '  %-52s CRLF lines=%-7s %s\n' "$f" \
            "$(count "${PRESCRIBED[@]}" "$f")" \
            "$(git ls-files --eol -- "$f" 2>/dev/null | awk '{print $1, $2}')"
    done
    echo
    echo "  (the trailing i/… w/… is git's own reading — index vs worktree —"
    echo "   and is authoritative regardless of what grep does on this shell.)"
fi

exit "$status"
