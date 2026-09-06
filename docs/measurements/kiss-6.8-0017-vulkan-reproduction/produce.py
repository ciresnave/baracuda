"""A `vulkan:` token producer written from `manifest/vulkan-vocabulary.json` ALONE.

KISS-CLASSIFY-6.8-0017 condition-2 demonstration for the KISS architect.

Experimental conditions, self-enforced:
  * the ONLY input was manifest/vulkan-vocabulary.json (24 633 bytes)
  * the crate archive was DELETED after extracting that one file, so
    src/lib.rs, README.md and the four test files were unreachable
  * no docs.rs, no vulkane source, no asking them

Two lists, per KISS-CLASSIFY-6.8-0017:

  GUESSED  — supplied from OUTSIDE the manifest. A guess that happened to be
             RIGHT is still listed: the array records what the DOCUMENT failed
             to determine, not what the reproduction got wrong, and the next
             reader may guess differently.
  DERIVED  — stated by the manifest, or entailed by its own conventions. Each
             entry MUST cite WHERE, because "it was entailed" is not a claim a
             reader can check and "it appears in these three places" is.

⚠️ An item in NEITHER list is an assertion that the manifest determined it.
Moving an item from GUESSED to DERIVED flatters the manifest and this
reproduction at once, which is why every DERIVED entry carries its location.
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

# --------------------------------------------------------------------------
# The residue ledger is DECLARED HERE, not accumulated at run time.
#
# WARNING: it used to be accumulated -- each `guess(...)` call appended its
# text when the branch executed -- and that structurally hid an item. GUESS-4
# is about the width-agnostic subgroup input, and its CONTENT is "no vector
# exercises this path". So the branch never ran, the append never happened,
# and the report printed SEVEN guesses from a file that declared EIGHT. The
# exact property that makes an item a finding -- nothing in the manifest
# reaches it -- was the property that kept it out of the finding list. An
# accumulate-on-execution ledger cannot report an unexercised path BY
# CONSTRUCTION, and its output is indistinguishable from a file that simply
# had one fewer entry.
#
# So: the full residue is the literal below, `guess()`/`derived()` only MARK a
# tag exercised, an undeclared tag is a hard error rather than a new row, and
# `main` prints every declared item with the ones no vector reached called out
# as [NOT EXERCISED] -- which is the strongest thing this report can say about
# them, not a reason to omit them.
# --------------------------------------------------------------------------
LEDGER: dict[str, tuple[str, str]] = {
    "GUESS-1": (
        "guess",
        "FNV-1a-64 offset basis 0xcbf29ce484222325 and prime "
        "0x100000001b3 — the manifest names the algorithm and the "
        "marker shape but not its constants",
    ),
    "GUESS-2": (
        "guess",
        "the digest is over UTF-8 bytes — every pinned digest_input is "
        "pure ASCII, so no vector discriminates the encoding",
    ),
    "GUESS-3": (
        "guess",
        "hex is LOWERCASE and ZERO-PADDED to 16 — both pinned digests "
        "happen to have no leading-zero nibble, so padding is unpinned",
    ),
    "GUESS-4": (
        "guess",
        "an input meaning 'width-agnostic' maps to `sgdyn` — no vector "
        "exercises it, so its INPUT representation is unpinned (all 12 "
        "vectors pass an integer)",
    ),
    "GUESS-5": (
        "guess",
        "⚠️ CONCLUSION CORRECTED, FINDING STANDS. I wrote '`saturating` "
        "is not spelled into the coop tuple'. It IS — vulkane measured a "
        "trailing `-sat`, with the saturating and non-saturating shapes "
        "kept as SEPARATE tuples. My PREMISE was right and is the whole "
        "finding: all 12 vectors have saturating=false, so a producer "
        "that appended a suffix and one that did not BOTH match every "
        "vector — which is true whether or not the suffix exists, and is "
        "exactly why the vectors could not answer it. And it is WORSE "
        "than the transpose case: transpose at least appears in "
        "`field_spec` prose, whereas saturating appears in NEITHER half "
        "— the prose says 'M-N-K plus four component types' and stops. "
        "So my producer emits WRONG BYTES for any saturating=true shape "
        "and scored 12/12 against a corpus that cannot contain one.",
    ),
    "GUESS-6": (
        "guess",
        "tuples with equal m,n,k tie-break on (a,b,c,result) in that "
        "order, ranked by component_types index — no vector has two "
        "shapes agreeing on m,n,k, so the tiebreak field ORDER is "
        "unpinned (the component RANKING is derivable)",
    ),
    "GUESS-7": (
        "guess",
        "`transpose` is not spelled — 56 coopvec combos across the "
        "vectors, transpose=true in ZERO, `-t` in ZERO tokens. The flag "
        "IS real (vulkane: spell() appends `-t`) but appears only in "
        "field_spec prose, so a manifest-only reader cannot learn it "
        "exists",
    ),
    "GUESS-8": (
        "guess",
        "the unnamed escape `x<n>` sorts NUMERICALLY on n and after all "
        "named types — every pinned run is equal-width (x0,x1,x2 / "
        "x1000..x1023), so no vector distinguishes numeric from "
        "lexicographic here",
    ),
    "DERIVED-1": (
        "derived",
        "the `<namespace>:` prefix and the field ORDER — stated "
        "verbatim by the top-level `grammar` field, "
        "\"vulkan:<subgroup>.<ops>.<arith>.<coop>.<coopvec>\"",
    ),
    "DERIVED-2": (
        "derived",
        "the canonical sort order for component types is the "
        "`component_types` ARRAY INDEX — the manifest gives it as an "
        "ordered array, and `ops_alphabet` and `arith_names` are used "
        "the same way for `<ops>` and `<arith>`, so the convention "
        "appears in THREE places; vector[9] (u8 before u32) pins it",
    ),
}

EXERCISED: set[str] = set()


def _mark(tag: str, kind: str) -> None:
    entry = LEDGER.get(tag)
    if entry is None:
        raise KeyError(f"{tag} is not declared in LEDGER - declare it, do not "
                       f"invent a row at run time")
    if entry[0] != kind:
        raise ValueError(f"{tag} is declared {entry[0]!r}, marked {kind!r}")
    EXERCISED.add(tag)


def derived(tag: str) -> None:
    """Mark a DERIVED item reached.

    An item the manifest STATES, or that its own conventions entail. Per
    KISS-CLASSIFY-6.8-0017: not `guessed`, but it MUST still be named, and
    every entry must say WHERE in the manifest the derivation is available.
    "It was entailed" is not checkable; "it appears in these three places" is.
    """
    _mark(tag, "derived")


def guess(tag: str) -> None:
    """Mark a GUESSED item reached. The text lives in `LEDGER`."""
    _mark(tag, "guess")


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
    guess("GUESS-1")
    # GUESS-2: hashed over UTF-8 bytes. The digest_input strings are pure ASCII
    # so UTF-8 vs ASCII vs Latin-1 cannot be distinguished by any vector here.
    guess("GUESS-2")
    h = FNV64_OFFSET
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * FNV64_PRIME) & 0xFFFFFFFFFFFFFFFF
    # GUESS-3: lowercase, zero-padded to 16. Inferred from the two pinned
    # digests being 16 lowercase hex chars; a value with a leading zero nibble
    # would discriminate padding and neither pinned digest has one.
    guess("GUESS-3")
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
        guess("GUESS-4")
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
    #
    # ⚠️ THIS SPELLING IS KNOWN WRONG AND IS LEFT AS THE REPRODUCTION'S RECORD.
    # vulkane measured that `saturating` IS spelled, as a trailing `-sat`, with
    # the two shapes kept as separate tuples. So the line below emits wrong
    # bytes for any saturating=true shape. It still scores 12/12, because all
    # 12 vectors are saturating=false — which is the finding, not a defence.
    #
    # It is not corrected here on purpose: this file is the artifact of what a
    # manifest-only reader produced, and silently fixing it with knowledge from
    # the maintainer would destroy the only thing it measures. See GUESS-5.
    guess("GUESS-5")
    return "-".join([str(t["m"]), str(t["n"]), str(t["k"]),
                     t["a"], t["b"], t["c"], t["result"]])


def _coop_key(t: dict):
    # NUMERIC on m,n,k then the component strings — pinned by vector[6], whose
    # enumeration spans 9 -> 10 and so discriminates numeric from lexicographic.
    # GUESS-6: the tiebreak ORDER AMONG THE COMPONENT fields (a,b,c,result) is
    # their tuple order. No vector has two shapes with equal m,n,k and differing
    # components, so the tiebreak is unpinned.
    guess("GUESS-6")
    order = {c: i for i, c in enumerate(D["component_types"])}
    rank = lambda c: order.get(c, len(order))
    return (t["m"], t["n"], t["k"],
            rank(t["a"]), rank(t["b"]), rank(t["c"]), rank(t["result"]))


def _coopvec_tuple(t: dict) -> str:
    # "five component types plus a transpose flag" (field_spec) — but the pinned
    # tokens spell FIVE parts and no flag.
    # GUESS-7: transpose is NOT spelled into the tuple.
    #
    # ⚠️ CORRECTED. This originally read "field_spec contradicts the vectors and
    # transpose:true is unrepresentable". WRONG — vulkane refuted it from
    # src/lib.rs:946-985, code this reproduction could not see: `spell()` appends
    # `-t` when set and `parse()` reads a sixth part back. Five components plus
    # an OPTIONAL `-t`, and the prose is accurate.
    #
    # It remains a guess, and the corrected version is sharper: across all
    # vectors there are 56 coopvec combos, transpose=true in ZERO of them, and
    # `-t` in ZERO tokens. The flag exists only in the documentation half, so a
    # reader confined to this manifest cannot learn it exists — which is exactly
    # why this producer scores 12/12 while being unable to emit one.
    # The pass and the blindness have the same cause.
    #
    # ⚠️ The correction above briefly DELETED this call along with the old
    # comment, and the guess list silently went from 8 to 7. Fixing a wrong
    # description of a guess must not remove the guess: the array records what
    # the DOCUMENT failed to determine, and that is unchanged by my having
    # mis-diagnosed why.
    guess("GUESS-7")
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
    derived("DERIVED-2")
    order = {c: i for i, c in enumerate(D["component_types"])}

    def rank(c: str):
        if c in order:
            return (0, order[c], 0)
        # GUESS-8: the unnamed escape `x<n>` sorts NUMERICALLY on n, and after
        # every named type. vector[9] holds x0,x1,x2 and x1000..x1023 — both runs
        # are equal-width, so lexicographic and numeric agree on every pinned
        # case and NOTHING discriminates them. A device exposing x9 and x10
        # together would.
        guess("GUESS-8")
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
    # ⚠️ RECLASSIFIED from GUESS-9 to DERIVED. `grammar` literally reads
    # "vulkan:<subgroup>.<ops>.<arith>.<coop>.<coopvec>" — it STATES the prefix
    # and the field order. It was first filed as a guess because the grammar is
    # a STRING rather than a structured assembly rule, but -0017 asks whether
    # the manifest STATES the item, not whether it states it machine-readably.
    #
    # This move makes both the manifest and this reproduction look better, which
    # is the exact incentive -0017 warns about — so it carries its citation.
    derived("DERIVED-1")
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
def _check_token(i: int, v: dict) -> bool:
    """One token vector. Prints its line; returns True if it reproduced."""
    got = token(v["input"])
    if got == v["token"]:
        print(f"  [PASS] vector[{i:2}] pins={v['pins']:<13} field={v['field']}")
        return True
    print(f"  [FAIL] vector[{i:2}] pins={v['pins']:<13} field={v['field']}")
    print(f"         want {v['token'][:110]}")
    print(f"         got  {got[:110]}")
    return False


def _check_digest(i: int, v: dict) -> bool:
    """One digest vector. The declared input LENGTH is checked too: it is what
    pins the threshold comparison as strictly-greater rather than >=."""
    got = fnv1a64(v["digest_input"])
    n = len(v["digest_input"].encode())
    if got == v["digest"] and n == v["digest_input_bytes"]:
        print(f"  [PASS] vector[{i:2}] pins={v['pins']:<13} field={v['field']} "
              f"(digest + {v['digest_input_bytes']}B length)")
        return True
    print(f"  [FAIL] vector[{i:2}] digest want {v['digest']} got {got} "
          f"({n}B vs declared {v['digest_input_bytes']}B)")
    return False


def check_vectors() -> int:
    """Run every pinned vector. Returns the FAILURE count."""
    ok = bad = skipped = 0
    for i, v in enumerate(M["vectors"]):
        if "token" in v and "input" in v:
            passed = _check_token(i, v)
        elif "digest_input" in v:
            passed = _check_digest(i, v)
        else:
            skipped += 1
            continue
        ok, bad = (ok + 1, bad) if passed else (ok, bad + 1)
    print(f"\n{ok} passed, {bad} failed, {skipped} skipped, "
          f"of {len(M['vectors'])} vectors")
    return bad


def _report_kind(kind: str, heading: str) -> None:
    """Print every DECLARED row of one kind, exercised or not.

    The unreached rows are the point: see the module note on why this must not
    be driven by what the run happened to execute.
    """
    rows = [(t, e[1]) for t, e in LEDGER.items() if e[0] == kind]
    cold = [t for t, _ in rows if t not in EXERCISED]
    print(f"\n=== {len(rows)} {heading} ===")
    for tag, text in rows:
        flag = "" if tag in EXERCISED else "[NOT EXERCISED] "
        print(f"  {flag}{tag}: {text}")
    if cold:
        print(f"  ({len(cold)} of {len(rows)} reached by no vector: "
              f"{', '.join(cold)}. Declared, so they are reported; an "
              f"accumulating ledger would have printed "
              f"{len(rows) - len(cold)} and looked complete.)")


def report_residue() -> None:
    """Print the full declared residue, both kinds."""
    _report_kind("guess", "GUESSED \u2014 the manifest did not supply these")
    _report_kind("derived",
                 "DERIVED \u2014 stated by the manifest; each cites WHERE")
    print(
        "\nAn item in NEITHER list asserts that the manifest determined it "
        "(KISS-CLASSIFY-6.8-0017)."
    )


def main() -> int:
    bad = check_vectors()
    report_residue()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
