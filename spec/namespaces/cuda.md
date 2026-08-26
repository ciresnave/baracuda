# `cuda:` — `target_capability` namespace vocabulary annex

**KISS-Classify §6.8.** Maintainer-owned annex, referenced by the §6.8-0003 namespace
registry. Per §6.8-0004 this document — not a KISS clause — defines the `cuda:`
vocabulary and its admission relations.

- **Namespace:** `cuda`
- **Maintainer:** Baracuda
- **Vocabulary version:** `cuda-vocab v1`
- **Registry vocabulary pointer:** this file.

## 0. Framing — specification, not description

This annex is **normative**: it defines what a conformant `cuda:` *producer* MUST emit
and what a conformant *consumer* MUST admit. It is written spec-first, not as a
description of one implementation.

Baracuda's `cuda:` codec (in `unpopped-vocab`, consumed via `baracuda-kernels-types`)
implements the §6.7 `target_capability` codec for this namespace; kiss-ref's
`kiss-classify-vocab` is dtype-only and produces no `cuda:` tokens. A second
producer (kiss-ref, or any other) is conformant when it satisfies THIS annex, not when
it matches Baracuda. The §8-0004 freeze gate's second same-namespace producer validates
against §1–§3 below.

## 1. Encoding layer (codec-neutral, byte-exact)

Pinned **separately** from the token values (§2): this is the shared codec, so any
producer that follows it emits byte-identical `cuda:` tokens and the §8-0004 diff is
meaningful.

1. **§cuda-1.1.** A `cuda:` capability token is the ASCII string `cuda:` immediately
   followed by exactly one sm-token (§2), with no whitespace: `cuda:` `<sm-token>`.
2. **§cuda-1.2.** The token occupies a single field of the SK4 structure key (fields
   `|`-delimited, e.g. `sk4|gem|f32|cuda:sm89|ix32|…`). It is compared **byte-exact**
   (§6.8-0002): not case-folded, not normalized. The literal bytes are the identity.
3. **§cuda-1.3.** The sm-token character set after the `cuda:` prefix is `[a-z0-9]` only
   (literal `sm`, decimal digits, an optional trailing `a`). Any byte outside `[a-z0-9]`
   — including the SK4 separator `|`, whitespace, `;`, `/`, or a second `:` (§6.8-0005) —
   makes the token **malformed**. No escaping rule is defined, because no byte requiring
   escaping is ever admissible.
4. **§cuda-1.4.** A `cuda:` capability-set contains **exactly one** token (a single
   scalar — §4). There is therefore **no** dedup, sort, or list-separator discipline to
   define at this layer; those apply only to list-bearing capability-sets in other
   namespaces.

## 2. Vocabulary — `cuda-vocab v1`

The **closed, exhaustive** set of valid sm-tokens. (Baracuda's `ArchSku` with its
emit/parse via `TargetId` (the `arch_code`/`arch_from_code` pair was retired) in
`unpopped-vocab`, re-exported through `baracuda-kernels-types`; the parser rejects any token not in this
set, so the vocabulary is closed, not an open grammar.)

| token | class | CUDA target | notes |
|-------|-------|-------------|-------|
| `cuda:sm80`  | base | Ampere `sm_80`   | forward-compat floor; runs on Ampere / Ada / Hopper |
| `cuda:sm89`  | base | Ada `sm_89`      | adds FP8 over `sm_80`; forward-compatible up-arch |
| `cuda:sm90`  | base | Hopper `sm_90`   | portable Hopper baseline (PTX-forward); distinct cache key from `sm90a` — `sm_90` is forward-compatible up-arch, `sm_90a` is not |
| `cuda:sm90a` | a    | Hopper `sm_90a`  | architecture-exclusive (wgmma / tma / async-barrier); SASS-locked |

**Well-formedness grammar (for future additions):** `sm` `<digits>` `[a]` — e.g.
`sm86`, `sm100`, `sm100a`. A syntactically well-formed token that is **not** in the
enumerated table above is **not conformant under `cuda-vocab v1`**; extending the set is
a version bump (§6). This distinguishes two declines: *malformed* (§cuda-1.3) vs
*well-formed but unregistered*.

**On `cuda:sm90` beside `cuda:sm90a`:** `cuda:sm90` (the portable Hopper baseline,
PTX-forward) was **added to the vocabulary** when `ArchSku::Sm90` was wired
(unpopped-vocab 0.2.0) — five of KISS's §6.7 reference vectors name `cuda:sm90`, which a
conformant reader must recognize. It is a **distinct token and cache key** from `sm90a`:
`sm_90` is forward-compatible up-arch, `sm_90a` is not, so the same H100 cannot share a
key across them. Baracuda's own Hopper **election** nonetheless remains `sm90a`
(`arch_sku_of` maps compute 9.x → `Sm90a`): it specializes Hopper against the
arch-exclusive instruction set, so `sm90` is a decodable vocabulary member for
PTX-forward targets / other emitters, not a Baracuda emission target today. Adding
`sm90` was a version bump (§6), per the well-formedness grammar note above.

**Token class** (load-bearing for §3):
- **base** — no trailing `a`. An architecture whose binaries are forward-compatible
  up-arch (PTX / JIT).
- **`a`** — trailing `a`. An architecture-**exclusive** feature set (SASS built with
  arch-specific instructions), carrying **no** forward-compatibility guarantee.

## 3. Admission relations — NORMATIVE, two independent clauses

**Definitions.** For a token `T` = `cuda:sm<digits>[a]`, **`T.sm_number`** is the decimal
`<digits>` (the run between `sm` and the optional trailing `a`, per the §2 grammar),
parsed as a `u32` — e.g. `sm90a`.`sm_number` = `90`. For a device `D`, **`D.sm_number`**
is `D`'s CUDA compute capability encoded as `major × 10 + minor` (compute 9.0 → `90`,
compute 10.0 → `100`). The `× 10` encoding is normative: it is what makes both the §3
comparisons and the §5 ordering correct across the 9.x → 10.x boundary.

A device admits a **set** of `cuda:` tokens; a kernel carries **one** token (its
specialization, §4). The deriver is `(device, choice) → token`. Admission — whether a
device `D` may execute a kernel bearing token `T` — is defined by **two independent
relations, selected by `T`'s class**.

There is deliberately **no single "`cuda:` admission is `≤`" rule.** Such a rule applied
namespace-wide would silently admit an `a` token onto hardware that cannot decode its
SASS — the silent-merge hazard, with no error raised.

- **§cuda-3.1 (base-token admission).** For a base token `T`: `D` admits `T`
  **iff `T.sm_number ≤ D.sm_number`.** (Forward-compat — `sm80` runs on any device with
  compute capability ≥ 8.0.)
- **§cuda-3.2 (`a`-token admission).** For an `a` token `T`: `D` admits `T`
  **iff `T.sm_number == D.sm_number`.** (Exact architecture match. `sm90a` admits on
  `sm_90` / Hopper hardware — its target — but, unlike the base `sm90` token (a real §2
  member, not a hypothetical), is **not** forward-compatible up-arch: an `sm_100` /
  Blackwell device does **not** admit it, because the architecture-exclusive SASS cannot
  execute there. `sm90` and `sm90a` share `sm_number` 90 but take opposite admission
  clauses — `sm90` the base `≤` of §cuda-3.1, `sm90a` the `==` of this clause — which is
  exactly why the two relations must be independent.)

A conformant deriver or validator MUST satisfy **both** clauses independently. Admitting
an `a` token under a `≤` relation (e.g. `sm90a` on `sm_100`) is a conformance failure of
§cuda-3.2 even when §cuda-3.1 holds for base tokens. There is no parent rule to satisfy
halfway.

## 4. Capability-set shape — single scalar token

A `cuda:` capability-set is exactly **one** sm-token: the single architecture the kernel
was specialized for. It is **not** a list. The multi-architecture reality of CUDA — a
device running several sm binaries, a base kernel forward-compatible up-arch — lives
entirely in the §3 admission relations, **never** in the token.

This is a **structural** property of the vocabulary, not a size bound. Consequently the
§6.8-0007 digest (FNV-1a-64 over a canonical enumeration, length-triggered) is inherited
KISS-wide but is **structurally unreachable** for `cuda:`: there is no variable-length
list to digest.

This claim is **held by a conformance assertion in this repository — not by prose, and
not by a cross-repo reference.** A test in the KISS conformance suite, landed with this
annex, asserts that every token enumerated in §2 matches `^sm[0-9]+a?$` (a single scalar,
no list/range separator); a range/list token added to the §2 table therefore fails a test
in the **same** repository that carries this claim, rather than silently falsifying this
paragraph. Baracuda's emitter *separately* guards the same invariant on its own side, but that
guard protects the emitter, not this document's claim — two guards, distinct scopes,
neither load-bearing for the other. (Any role a range token might serve is served by the §3 admission
relations, not by the token.)

## 5. Canonical ordering

`cuda:` tokens are never listed *within* a capability-set (§4), so ordering applies only
to **external** enumerations (registry rows, conformance corpora). Canonical order:

> sort by `(sm_number: u32 ascending, variant: base < a)`

Within `cuda-vocab v1` this is `sm80 < sm89 < sm90 < sm90a`. The rule is general for future
additions — illustrated here over a hypothetical superset (the extra tokens are **not**
v1, shown only to exercise the rule):

    sm80 < sm86 < sm89 < sm90 < sm90a < sm100 < sm100a

The `a` variant sorts immediately **after** its own base sm-number and never reorders
across sm-numbers (`100 > 90` dominates the tie-break; this relies on the `× 10`
encoding of §3).

## 6. Versioning

`cuda-vocab v1` = { `cuda:sm80`, `cuda:sm89`, `cuda:sm90`, `cuda:sm90a` }. Any of the following is a
**version bump**, recorded against the `cuda` row in the §6.8-0003 registry:

- adding an sm-token (a new `ArchSku` variant with its emit/parse arms), or removing one;
- altering an admission relation (§3);
- **changing the encoding layer (§1)** — the most consequential bump, since it breaks
  every previously-emitted token byte-for-byte.

The encoding layer (§1) and the admission-relation structure (§3) are otherwise stable
across token additions; a token addition is additive to the §2 table.
