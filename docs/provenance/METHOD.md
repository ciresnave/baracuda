# Provenance audit method (reusable)

One page, written from the baracuda audit of 2026-10-14 (PRs #173–#176; results in
[`../../PROVENANCE.md`](../../PROVENANCE.md)). It implements CireSnave's rule (`C:\Projects\CIRESNAVE-EXPECTATIONS.md`
§6.4a): trace each third-party file to its **origin**, verify our copy against it, keep its licence and notices,
mark the file with the origin's licence, never fabricate a holder. For a lane inheriting files (fuel, lightbulb,
mlmf): do these steps in order, one PR per bucket.

## 1. Enumerate with the index

`git ls-files` at a captured `origin/main` SHA, never a disk walk (other lanes' worktrees are on disk). Find
candidates by three routes and union them: (a) directories named `vendor`/`third_party`/`external`;
(b) licence/NOTICE/AUTHORS files (`git ls-files | grep -i -E 'licen|notice|authors|copying'`);
(c) a content sweep (`git grep -l -i -E 'SPDX-FileCopyrightText|Copyright \(c\)|adapted from|ported from|derived from|based on|taken from|<names of libs you know you used>'`)
and triage the hits: a prose mention is not a derived file, a header naming a third party is.
Record the population count beside every "none found" (positive control: the sweep must re-find a file you
already know is derived).

## 2. Trace to the origin, not the nearest copy

For each file, follow the chain until a repo that is not itself a copy (fuel → candle → llama.cpp; fuel →
candle → attention.rs). Use the file's own header, the project's VENDOR/README, and **history**: `git log
--follow`, the first commit that added it, the root commit of a fork (a fork's root commit equals the parent's
"Initial commit"). Clone each upstream with `git clone --filter=blob:none --no-checkout <url>`; you only need blobs.

## 3. Verify by blob id (the core test)

Compare **git blob ids**, not text: `git hash-object <our file>` (or `git rev-parse HEAD:<path>`) against
`git rev-parse <upstream-commit>:<upstream-path>`. Equal ids ⇒ byte-identical, which beats any diff and survives
CRLF working trees (`core.autocrlf` makes the checkout lie; the blob does not — hash the blob, or `git show`
it). Classify each file:

| class | test |
|---|---|
| IDENTICAL | blob ids equal |
| MODIFIED | not equal; record hunk count and +/- lines (`git diff --no-index --numstat`) and one line saying what changed |
| ADAPTED / HAND-PORT | no near-verbatim text; measure shared distinct non-comment lines (>25 chars) and record the number; **positive control**: a file you know is verbatim must score ~100% on the same method |
| NO-UPSTREAM | authored here; say so |

Walk history for the *matching* upstream commit when the pin is unknown: the newest upstream commit whose blob
equals ours is a lower bound on when it was taken. A file edited here has no equal blob anywhere — then diff
against the latest upstream version and say "closest to `<sha>`".

## 4. Pins

Record `<repo>@<full 40-hex sha>` per origin, in the directory's VENDOR/README and in `PROVENANCE.md`. A tag is
not a pin: verify the tag points at that sha (`git rev-parse <tag>^{commit}`) — baracuda's FlashInfer entry said
"tag v0.6.12" for a `main` commit. If the vendor-time commit was never recorded and the content no longer
identifies it, use "last upstream commit on or before the vendor date" **and say it is date-bounded and not
recovered**. Never present HEAD-at-audit as the vendor-time commit.

## 5. Licence texts: verbatim or absent

- Carry the origin's licence text **as upstream bytes** at the pinned commit; prove it with the blob ids
  (`docs/provenance/licence-files-verified.tsv` is the template: staged blob == upstream blob, `yes/NO`).
  A reformatted, reworded or "equivalent" licence is a defect (baracuda found five).
- Carry NOTICE verbatim where the origin has one (Apache-2.0 §4(d)); state modifications (§4(b)) in the file or
  the VENDOR note.
- **Do not invent a holder, a year or a licence file.** If upstream's licence has no copyright line, or ships no
  LICENSE file at all (attention.rs, cudaforge), say exactly that in `PROVENANCE.md` and raise a [FINDING]; the
  header then names the origin ("The candle contributors … upstream names no holder"), never a person. Do not
  file upstream issues without the PM.
- An SPDX header names the **origin's** licence on files that are the origin's work; the project default
  (`MIT OR Apache-2.0`) only on files authored here.
- Where the origin's own origin has a licence (candle ← llama.cpp ← …), carry each link's text.

## 6. What to record, and where

Per file (TSV, one row each, in `docs/provenance/`): path, origin repo, origin commit, origin path, class, diff
summary, header-ok, notes. Per directory: a VENDOR/README with upstream URL, pinned sha (and whether it is
date-bounded), licence, what was kept/removed, **every modification to an otherwise-verbatim file**. One
`PROVENANCE.md`: inventory table, "what holds" (so it is not re-audited), a gap register with ids and a status
column (FIXED / needs PM / open-upstream-only).

## 7. Shipping the fixes: three buckets, one PR each

1. **Licence files** (add/replace with upstream bytes). Prove: staged blob == upstream blob.
2. **Header comments** (SPDX/origin lines, NOTICE restatements). Prove code unchanged: for every `.cu/.cuh/.rs`
   touched, `strip(old) == strip(new)` where `strip` removes `//` and `/* */` comments and whitespace, **and**
   `git diff -U0 | grep '^[-+]' | grep -v '^[-+]\s*//'` is empty (positive control: a deliberate one-character
   code edit must fail the first check). Run rustfmt/`cargo check` on touched Rust.
3. **Pins/docs** (VENDOR notes). Prove: no code or licence file in `git diff --name-only`.

Each PR bumps the version per policy (docs are not exempt) and says "Merging is not a publish". A code-quality
bot that flags a file you only touched in a comment gets an answer on the PR, never a config change.

## 8. Traps seen

CRLF checkouts vs LF blobs; a `.gitignore`d upstream checkout (the licence you "keep" is untracked); a tag that
is not the pinned commit; a VENDOR note that says "verbatim" for a file that was edited; an invented copyright
line copied from the project's own header; hand-ported kernels that credit the algorithm but carry no licence
text; `git log` of the *copy* mistaken for the upstream's history.
