"""A `vulkan:` token producer written from `manifest/vulkan-vocabulary.json` ALONE.

KISS-CLASSIFY-6.8-0017 condition-2 demonstration for the KISS architect.

Experimental conditions, self-enforced:
  * the ONLY input was manifest/vulkan-vocabulary.json (24 633 bytes)
  * the crate archive was DELETED after extracting that one file, so
    src/lib.rs, README.md and the four test files were unreachable
  * no docs.rs, no vulkane source, no asking them

Every inference not stated by the manifest is tagged GUESS-n in a comment and
collected in GUESSES at the bottom. Per the architect's condition, a guess that
happens to be RIGHT is still a finding: it is a thing the manifest failed to
supply, and it is invisible in a byte-match that succeeds.
"""

import json
import pathlib
import sys

# The manifest sits beside this file in the committed artifact; during the
# original run it lived in an `isolated/` directory that held nothing else.
_HERE = pathlib.Path(__file__).parent
MANIFEST = next(p for p in (_HERE / "vulkan-vocabulary.json",
                            _HERE / "isolated" / "vulkan-vocabulary.json")
                if p.is_file())
M = json.loads(MANIFEST.read_text(encoding="utf-8"))
D = M["declarative"]

GUESSES: list[str] = []


def guess(tag: str, what: str) -> None:
    if tag not in [g.split(":", 1)[0] for g in GUESSES]:
        GUESSES.append(f"{tag}: {what}")


# --------------------------------------------------------------------------
# FNV-1a-64.
#
# GUESS-1: the manifest says "FNV-1a-64" and `fnv1a64-<hex16>` and NEVER GIVES
# THE PARAMETERS. offset basis and prime are taken from the published FNV
# definition. A producer with the FNV-1 (not -1a) operand order, or a different
# basis, matches the declared marker shape and produces different bytes.
# --------------------------------------------------------------------------
FNV64_OFFSET = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3


def fnv1a64(s: str) -> str:
    guess("GUESS-1", "FNV-1a-64 offset basis 0xcbf29ce484222325 and prime "
                     "0x100000001b3 — the manifest names the algorithm and the "
                     "marker shape but not its constants")
    # GUESS-2: hashed over UTF-8 bytes. The digest_input strings are pure ASCII
    # so UTF-8 vs ASCII vs Latin-1 cannot be distinguished by any vector here.
    guess("GUESS-2", "the digest is over UTF-8 bytes — every pinned digest_input "
                     "is pure ASCII, so no vector discriminates the encoding")
    h = FNV64_OFFSET
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    # GUESS-3: lowercase, zero-padded to 16. Inferred from the two pinned
    # digests being 16 lowercase hex chars; a value with a leading zero nibble
    # would discriminate padding and neither pinned digest has one.
    guess("GUESS-3", "hex is LOWERCASE and ZERO-PADDED to 16 — both pinned "
                     "digests happen to have no leading-zero nibble, so padding "
                     "is unpinned")
    return f"fnv1a64-{h:016x}"


# --------------------------------------------------------------------------
# Fields.
# --------------------------------------------------------------------------
def field_subgroup(v) -> str:
    # `sg<width>`, or `sgdyn` for the width-agnostic kernel (field_spec note).
    # GUESS-4: how a width-agnostic input is REPRESENTED in the input struct.
    # No vector exercises `sgdyn`; all 12 pass an integer. None/"dyn" is my
    # choice of spelling for the absent case.
    if v is None or v == "dyn":
        guess("GUESS-4", "an input meaning 'width-agnostic' maps to `sgdyn` — no "
                         "vector exercises it, so its INPUT representation is "
                         "unpinned (all 12 vectors pass an integer)")
        return "sgdyn"
    return f"sg{v}"


def _empty(prefix: str) -> str:
    # `empty_set_spelling` is "<prefix>-none"; the prefixes in field_spec already
    # carry their trailing "-" (e.g. "ops-", "cm-"), except subgroup's "sg".
    return f"{prefix}none"


def field_ops(vals) -> str:
    if not vals:
        return _empty("ops-")
    order = {c: i for i, c in enumerate(D["ops_alphabet"])}
    # Juxtaposed single letters in ops_alphabet order (field_spec + vector[5]).
    return "ops-" + "".join(sorted(set(vals), key=lambda c: order[c]))


def field_arith(vals) -> str:
    if not vals:
        return _empty("arith-")
    order = {n: i for i, n in enumerate(D["arith_names"])}
    # Named parts joined by "-" in arith_names order (field_spec + vector[4]).
    return "arith-" + "-".join(sorted(set(vals), key=lambda n: order[n]))


def _coop_tuple(t: dict) -> str:
    # "M-N-K plus four component types, joined by `-`" (field_spec).
    # GUESS-5: `saturating` is NOT spelled. Every vector has saturating=false,
    # so a producer that appended it would pass all 12 — the input carries a
    # field the token has no place for, and nothing pins that.
    guess("GUESS-5", "`saturating` is not spelled into the coop tuple — all 12 "
                     "vectors have saturating=false, so a producer that appended "
                     "it would still match every one")
    return "-".join([str(t["m"]), str(t["n"]), str(t["k"]),
                     t["a"], t["b"], t["c"], t["result"]])


def _coop_key(t: dict):
    # NUMERIC on m,n,k then the component strings — pinned by vector[6], whose
    # enumeration spans 9 -> 10 and so discriminates numeric from lexicographic.
    # GUESS-6: the tiebreak ORDER AMONG THE COMPONENT fields (a,b,c,result) is
    # their tuple order. No vector has two shapes with equal m,n,k and differing
    # components, so the tiebreak is unpinned.
    guess("GUESS-6", "tuples with equal m,n,k tie-break on (a,b,c,result) in "
                     "that order, ranked by component_types index — no vector "
                     "has two shapes agreeing on m,n,k, so the tiebreak field "
                     "ORDER is unpinned (the component RANKING is derivable)")
    order = {c: i for i, c in enumerate(D["component_types"])}
    rank = lambda c: order.get(c, len(order))
    return (t["m"], t["n"], t["k"],
            rank(t["a"]), rank(t["b"]), rank(t["c"]), rank(t["result"]))


def _coopvec_tuple(t: dict) -> str:
    # "five component types plus a transpose flag" (field_spec) — but the pinned
    # tokens spell FIVE parts and no flag.
    # GUESS-7: transpose is NOT spelled. All 12 vectors have transpose=false.
    # field_spec explicitly SAYS the tuple carries a transpose flag, and no
    # pinned token contains one, so the manifest contradicts itself here and the
    # vectors win. A device reporting transpose=true is unrepresentable.
    guess("GUESS-7", "`transpose` is not spelled — field_spec says the tuple is "
                     "'five component types plus a transpose flag' but every "
                     "pinned token spells five parts and no flag, and all 12 "
                     "vectors have transpose=false. The prose and the vectors "
                     "DISAGREE and I followed the vectors")
    return "-".join([t["input"], t["input_interpretation"],
                     t["matrix_interpretation"], t["bias_interpretation"],
                     t["result"]])


def _coopvec_key(t: dict):
    # The canonical order is the `component_types` INDEX, not the spelled string.
    # vector[9] discriminates: it puts the `u8-...` group BEFORE the `u32-...`
    # group, which is impossible lexicographically ("u32" < "u8") and exact under
    # component_types (u8 -> 8, u32 -> 10).
    #
    # ⚠️ I GOT THIS WRONG FIRST TIME and vector[9] caught it. That is the vector
    # set doing its job: `component_types` is an ORDERED array and the manifest
    # never says in prose that its order is the sort order — but ops_alphabet and
    # arith_names are used exactly that way, so the convention is derivable and
    # the vector pins it. Recorded as DERIVED, not as a guess.
    order = {c: i for i, c in enumerate(D["component_types"])}

    def rank(c: str):
        if c in order:
            return (0, order[c], 0)
        # GUESS-8: the unnamed escape `x<n>` sorts NUMERICALLY on n, and after
        # every named type. vector[9] holds x0,x1,x2 and x1000..x1023 — both runs
        # are equal-width, so lexicographic and numeric agree on every pinned
        # case and NOTHING discriminates them. A device exposing x9 and x10
        # together would.
        guess("GUESS-8", "the unnamed escape `x<n>` sorts NUMERICALLY on n and "
                         "after all named types — every pinned run is "
                         "equal-width (x0,x1,x2 / x1000..x1023), so no vector "
                         "distinguishes numeric from lexicographic here")
        n = int(c[1:]) if c.startswith("x") and c[1:].isdigit() else -1
        return (1, n, 0)

    return tuple(rank(c) for c in (t["input"], t["input_interpretation"],
                                   t["matrix_interpretation"],
                                   t["bias_interpretation"], t["result"]))


def _length_conditional(prefix: str, tuples: list, spell, key) -> str:
    if not tuples:
        return _empty(prefix)
    uniq = {spell(t): t for t in tuples}                      # dedup before spelling
    body = ",".join(spell(t) for t in sorted(uniq.values(), key=key))
    # Strictly ABOVE the threshold digests; exactly AT it is spelled in full
    # (vectors 6/7 and 9/10). The measured string is the enumeration WITHOUT the
    # field prefix — pinned by the digest_input vectors 8/11.
    if len(body.encode("utf-8")) > D["digest_threshold_bytes"]:
        return prefix + fnv1a64(body)
    return prefix + body


def token(inp: dict) -> str:
    # GUESS-9: field ORDER in the emitted token comes from `grammar`, and the
    # namespace prefix is "<namespace>:" — the manifest gives the grammar string
    # and `namespace`, but never states that the token begins "vulkan:" as
    # opposed to some other rendering of the namespace.
    guess("GUESS-9", "the token begins '<namespace>:' and fields appear in "
                     "`grammar` order — both read off the grammar string, which "
                     "is prose rather than a declared assembly rule")
    fields = [
        field_subgroup(inp.get("subgroup")),
        field_ops(inp.get("ops") or []),
        field_arith(inp.get("arith") or []),
        _length_conditional("cm-", inp.get("coop") or [], _coop_tuple, _coop_key),
        _length_conditional("cv-", inp.get("coopvec") or [], _coopvec_tuple, _coopvec_key),
    ]
    return f"{M['namespace']}:" + D["field_separator"].join(fields)


# --------------------------------------------------------------------------
# Check against every pinned vector.
# --------------------------------------------------------------------------
def main() -> int:
    ok = bad = skipped = 0
    for i, v in enumerate(M["vectors"]):
        if "token" in v and "input" in v:
            got = token(v["input"])
            if got == v["token"]:
                ok += 1
                print(f"  [PASS] vector[{i:2}] pins={v['pins']:<13} field={v['field']}")
            else:
                bad += 1
                print(f"  [FAIL] vector[{i:2}] pins={v['pins']:<13} field={v['field']}")
                print(f"         want {v['token'][:110]}")
                print(f"         got  {got[:110]}")
        elif "digest_input" in v:
            got = fnv1a64(v["digest_input"])
            if got == v["digest"] and len(v["digest_input"].encode()) == v["digest_input_bytes"]:
                ok += 1
                print(f"  [PASS] vector[{i:2}] pins={v['pins']:<13} field={v['field']} "
                      f"(digest + {v['digest_input_bytes']}B length)")
            else:
                bad += 1
                print(f"  [FAIL] vector[{i:2}] digest want {v['digest']} got {got}")
        else:
            skipped += 1
    print(f"\n{ok} passed, {bad} failed, {skipped} skipped, of {len(M['vectors'])} vectors")
    print(f"\n=== {len(GUESSES)} THINGS THE MANIFEST DID NOT SUPPLY ===")
    for g in GUESSES:
        print(f"  {g}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
