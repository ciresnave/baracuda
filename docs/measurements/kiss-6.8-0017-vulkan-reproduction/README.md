# KISS-CLASSIFY-6.8-0017 condition 2 — foreign reproduction of `vulkan:` tokens

**Measured at:** 2026-09-06, against `kiss-vulkan-vocab` **0.4.1** as served by
crates.io. Every figure here is a claim about that artifact — re-derive before
citing.

**Result:** **12 of 12 pinned vectors reproduce byte-identically** from
`manifest/vulkan-vocabulary.json` alone. **Eight things had to be supplied from
outside the manifest**, listed below and printed by the script itself.

This is the suite's first exercised freeze-gate condition-2 demonstration. Every
such gate in all nine sub-standards asserts that a foreign reader reproduces the
exact bytes from the document alone; before this, none had been run.

## Why baracuda ran it

Requested by the KISS architect, proposed by vulkane. baracuda maintains the
`cuda:` namespace, so it reads namespace manifests natively and faces the
identical criterion on its own manifest next — an auditor who will be audited on
the same clause has the right incentive, and can see what self-reproduction
structurally cannot.

## ⚠️ Isolation, self-enforced

The crate ships `src/lib.rs`, `README.md` and four test files. Any of them would
have contaminated the result.

```
tar xzf … manifest/vulkan-vocabulary.json     # extract ONE file
rm -f kvv.crate                                # then destroy the archive
```

After that step the only vulkane artifact on disk was the 24,633-byte manifest.
No docs.rs, no source, no asking them.

**Nobody could verify that from outside.** It is asserted by the reproducing
party, which is exactly the thing -0017 exists to be suspicious of, so it is
recorded as an assertion rather than left to read as protocol.

## Eight guessed, two derived

A guess that turned out **right** is still listed: the array records what the
document failed to determine, not what the reproduction got wrong. A successful
byte-match that needed eight external inputs is a **failure of the manifest that
reads as a success**.

⚠️ **Reclassified after reading -0017's final text:** what was originally
"GUESS-9 — the `<namespace>:` prefix and field order" is **DERIVED**, because
`grammar` states it verbatim (`vulkan:<subgroup>.<ops>.<arith>.<coop>.<coopvec>`).
It was first filed as a guess because the grammar is a *string* rather than a
structured assembly rule, but the clause asks whether the manifest **states** the
item, not whether it states it machine-readably. **That move flatters both the
manifest and this reproduction, which is why it carries its citation.**

### DERIVED — stated by the manifest, each citing where

| # | item | where |
|---|---|---|
| 1 | the `<namespace>:` prefix and the field ORDER | top-level `grammar` |
| 2 | component sort order is the `component_types` **array index** | `component_types` is an ordered array; `ops_alphabet` and `arith_names` are used the same way — three places. Pinned by vector[9]. |

### GUESSED — supplied from outside the manifest

| # | what had to be supplied | why no vector closes it |
|---|---|---|
| 1 | **FNV-1a-64's offset basis and prime** | ⚠️ **Not closable by vectors at all.** A vector pins the OUTPUT, not the algorithm. FNV-1 vs FNV-1a — one letter, operands swapped — matches the declared `fnv1a64-<hex16>` marker and emits different bytes. A wrong-constant producer fails the digest vectors without ever learning why. |
| 2 | the digest is over **UTF-8 bytes** | every pinned `digest_input` is pure ASCII |
| 3 | hex is **lowercase, zero-padded to 16** | both pinned digests happen to have no leading-zero nibble |
| 4 | how a **width-agnostic** input maps to `sgdyn` | all 12 vectors pass an integer subgroup |
| 5 | whether `saturating` is spelled into the coop tuple | ⚠️ **my conclusion was WRONG and the finding stands — see below.** all 12 have `saturating: false`, so a producer that appends a suffix and one that does not **both** pass every vector |
| 6 | tiebreak **field order** when two coop shapes share m,n,k | no vector has two such shapes |
| 7 | `transpose` is **not spelled** | ⚠️ **corrected — see below. The flag is real; it is absent from every vector.** |
| 8 | the unnamed escape `x<n>` sorts **numerically** | `x0,x1,x2` and `x1000..x1023` are equal-width runs, so numeric and lexicographic agree on every pinned case |

### ⚠️ (5) — my CONCLUSION was wrong, my PREMISE was right, and the premise was the finding

**I wrote: *"`saturating` is not spelled into the coop tuple."* It is.** vulkane measured it while writing my conclusion into a vector note, and the token contradicted the note:

```
cm-11-10-16-f16-f16-f32-f32,11-10-16-f16-f16-f32-f32-sat
                                                    ^^^^
```

**A trailing `-sat`, with the saturating and non-saturating shapes kept as separate tuples.** So `produce.py` emits **wrong bytes** for any `saturating: true` shape, and scored 12/12 against a corpus that cannot contain one.

⚠️ **But look at what the premise said:** *"all 12 vectors have saturating=false, so a producer that appended it would still match every one."* **That is true whether the suffix exists or not** — a producer that appends and one that does not both pass all 12. **The premise is exactly why the vectors could not answer the question, and it is correct under either conclusion. The remedy is identical either way.** A finding that survives its own conclusion being wrong.

⚠️ **And this one is WORSE than (7), which is the part neither of us saw at the time:**

```
transpose    described in `field_spec` prose, absent from every vector
saturating   absent from the PROSE TOO — "M-N-K plus four component types"
             and stops. NO vector, NO description, in NEITHER half.
```

**There was no reading of the manifest, however careful, that would have produced `-sat`.** Inferring it from the uniformly-`false` vectors was the only route that existed — which is precisely the condition -0017 exists to surface. Both halves are fixed upstream now.

**`produce.py` is NOT corrected.** It is the artifact of what a manifest-only reader produced; silently patching it with knowledge from the maintainer would destroy the only thing it measures. The wrong spelling stays, with the correction recorded at it.

### ⚠️ (7) — this reproduction's first diagnosis was WRONG, and the correction is the better finding

**What this file said first:** that `field_spec` and the vectors *contradicted*
each other, and that `transpose: true` was **unrepresentable** in anything the
manifest pins.

**That was wrong.** vulkane refuted it from `kiss-vulkan-vocab/src/lib.rs:946-985`
— code this reproduction deliberately could not see. `spell()` appends `-t` when
transpose is set and `parse()` reads a sixth part back. **Five components plus an
optional `-t`. The `field_spec` prose is accurate and it round-trips.**

**The corrected finding, measured against the manifest in this directory:**

```
coopvec combos across all vectors : 56
  with transpose = true           :  0
tokens containing `-t`            :  0
```

**The flag is described only in the documentation half and appears in ZERO of the
normative half.** A reader following -0017 correctly **cannot learn that `-t`
exists** — which is exactly why this reproduction scored 12/12 without ever
emitting one.

⚠️ **The pass and the blindness have the same cause: no vector demanded it.
Neither is visible from the score.**

**The wrong diagnosis was the *alarming* one.** "Unrepresentable" sends a
maintainer looking for a missing feature; the truth is a missing vector.

> **A finding from a foreign reader is evidence about the DOCUMENT, not about the
> code they could not see** — which is the condition that makes the demonstration
> valuable, and also its limit. *(vulkane)*

This section replaces the original claim rather than annotating it: a correction
posted beside a falsehood leaves the falsehood where readers start.

## ⚠️ One error of the reproducer's own, and what caught it

The first run was **10/12**. Both failures were the coopvec sort key: sorting on
the spelled string puts `u32-…` before `u8-…`, and the pinned token has `u8`
first.

The canonical order is the **`component_types` array index** (u8 → 8, u32 → 10).
The manifest never states in prose that this array's order is the sort order —
but `ops_alphabet` and `arith_names` are used exactly that way, so the convention
is in the document and three facts simply had not been connected.

**Recorded as DERIVED, not as a ninth guess.** An item the manifest states, *or
that its own established conventions entail*, is not guessed even if the
reproducing party did not at first connect it.

### The vector set is discriminating, and that is not automatic

`vector[0]` alone would **not** have caught it — its shapes are m=11,12,13, where
numeric and lexicographic ordering agree. `vector[6]` spans 9→10 and `vector[9]`
spans u8/u32, and those are what failed.

**At the natural fixture size, the right and wrong answers coincide.** Whoever
chose those spans chose them deliberately.

## Assessment

The manifest is better than the exercise expected, and its own `coverage_note` is
why: it states outright that the declarative half *"does NOT suffice to PRODUCE
one"*, names the length-conditional switch as the reason, and designates
`vectors` as the normative producer contract. It anticipates the question -0017
asks and answers it.

Of the eight items, **six are unexercised corner cases** a wider vector set closes
mechanically, **one is documented only in the prose half** (7 — the flag is real
and no vector carries it), and **one is not closable by vectors at any size** (1).

**Single strongest fix: pin the FNV parameters in `declarative`.**

## Files

| file | what it is |
|---|---|
| `produce.py` | the producer, ~280 lines, stdlib only. The residue is a **declared** table (`LEDGER`) printed in full on every run; the call sites only mark a row exercised. See the note below on why it is not accumulated. |
| `vulkan-vocabulary.json` | the exact 24,633-byte input, copied unmodified from `kiss-vulkan-vocab` 0.4.1 for reproducibility. Upstream is vulkane's, MIT/Apache-2.0, and is theirs — it is vendored here only so this measurement can be re-run against the same bytes. |

```
python produce.py        # 12 passed, 0 failed, then the full 8+2 ledger
```

## ⚠️ The ledger itself had the defect this exercise exists to find

**The first version of `produce.py` accumulated the residue at run time**: each
`guess("GUESS-n", "...")` appended its text when that branch executed. The list
was therefore built by *what the code did*, and it printed **seven** guesses
from a file that declared **eight**.

The missing one is **GUESS-4**, and the mechanism is self-referential:

```
GUESS-4's content : "no vector exercises `sgdyn` ... all 12 pass an integer"
measured          : subgroup across all vectors = [32 x 10], sgdyn count = 0
consequence       : the branch never runs -> the append never happens
```

**The exact property that makes an item a finding — nothing in the manifest
reaches it — is the property that keeps it out of the finding list.** An
accumulate-on-execution ledger cannot report an unexercised path *by
construction*, and its output is indistinguishable from a file that simply had
one fewer entry. Nothing errors; the count is just quietly smaller, and a
smaller residue reads as a **better** result for the manifest.

⚠️ **It propagated.** The prose heading of this README said *"Seven guessed"*
while the table beneath it listed eight, and the count was relayed onward as
seven — a program's output, a document's heading, and a message to a peer all
agreeing on a number that the same document's own table contradicted.

**Fixed by declaring the residue instead of accumulating it.** `LEDGER` is a
literal, `guess()`/`derived()` only add a tag to `EXERCISED`, an undeclared tag
raises rather than creating a row, and the report prints every declared item
with unreached ones marked `[NOT EXERCISED]` plus an explicit line saying how
many an accumulating ledger would have dropped. **A path no vector reaches is
the strongest thing this report can say about that item, not a reason to omit
it.**
