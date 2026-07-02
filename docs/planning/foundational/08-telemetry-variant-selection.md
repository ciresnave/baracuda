# 08 — Telemetry variant-selection consumer — implementation brief

## 1. Objective

Build Baracuda's half of **telemetry-driven variant selection**: the consumer that
turns Fuel's real-workload dispatch/miss feed into a *ranked, per-arch AOT build
matrix* keyed on the ratified `structure_key` token, plus the emitter-side ability to
offer **top-K equivalent forms** of an op so the feed can vote on which form actually
wins on-device. Concretely this is three cooperating pieces: (1) an **equivalent-form
enumerator** over the existing e-graph (`optimize.rs`) that extracts the K lowest-cost
distinct `ScalarExpr` forms of one op body (not just the single cheapest), each of
which round-trips to a compilable kernel; (2) a **v1 JSONL ingest schema** —
`DispatchRecord` / `MissRecord` — parsed against the `StructureKey::to_token` /
`from_token` codec that already exists (`structure_key.rs:616,654`), coverage-agnostic
per the ratified reply (`fuel-reply-telemetry-fdx-fkc-2026-06-19.md` §1, §7); and (3) a
**variant-selection reducer** that aggregates choice-frequency + miss-count into the
next release's AOT cell list (feeds item 07's dispatch table). It is foundational
because it closes the design-doc §8 feedback loop — the mechanism that "defeats the
chicken-and-egg trap" (§8b): Fuel reports the cell it *wanted* even when it had to route
around it, so the matrix grows toward measured demand instead of guesswork, and every
generated variant is chosen by data, not by a hand-tuned cost heuristic.

## 2. Status & blockers

**Baracuda-unblocked (build now):** the entire Baracuda half is buildable today against
frozen types. The `StructureKey` token codec is shipped and lossless
(`to_token`/`from_token`, round-trip-tested at `structure_key.rs:992`). The e-graph
that top-K extraction extends is shipped (`optimize.rs`, `extract` at line 398). The
FKC/`link_registry` generator that a chosen variant must emit is shipped
(`contract.rs`, `link.rs`). So: the JSONL record structs + parser, the top-K
enumerator, and the reducer that produces a ranked cell list are all pure-Baracuda,
pure-CPU, no-CUDA-required code that can land and be unit-tested immediately.

**Fuel-blocked (the FEED only):** Baracuda does not *produce* the telemetry records —
Fuel does, from its planner/Judge (`fuel-reply-telemetry-fdx-fkc-2026-06-19.md` §6, "You:
build the emission layer over the confirmed retention", §8 step 3). We design and ingest
the schema now; we cannot end-to-end validate against a live feed until Fuel emits it.
This is fine and expected: the design doc §8 rollout is explicitly **v1 = batch**
(Fuel publishes an aggregated JSONL report at build/release time; no runtime coupling)
*before* v2 = live. Build the v1 batch consumer; it needs only a file of records, which
we can fixture ourselves for tests.

**Design-open (needs a decision, see §10):** (a) whether top-K variants are advertised
as *distinct FKC contracts under distinct `entry_point`s at the same `structure_key`*
(so Fuel's `candidates[]` can time them head-to-head) or held back as build-matrix
candidates only; (b) the exact `chosen_impl` / `ImplId` field spelling on the wire
(the ratified basis tuple is `(BackendId, op, dtypes, kernel_source, kernel_revision_hash)`
kept *separable*, `fuel-reply-…-2026-06-19.md` §2.3 / item 2 — but the JSONL leaf
encoding is not yet frozen); (c) `est_speedup_if_available` provenance on a miss (Fuel's
estimate vs measured).

## 3. Dependencies & sequencing

**Must land before this:** nothing hard-blocks the schema + reducer. Two soft
couplings:
- The **top-K enumerator** shares machinery with the algebraic optimizer (item on the
  sequencing list, LAST). Do **not** reactivate the full optimizer here. Item 08 needs
  only *extraction of the K best distinct forms* from the already-saturated e-graph —
  the rewrite *rule set* stays exactly the precision-safe set in `optimize.rs` today.
  Adding rules is the LAST item's job; 08 rides whatever equivalences the current rules
  already discover (identities, const-fold, `neg∘neg`, `max(x,x)`), which is enough to
  demonstrate the top-K mechanism end-to-end.
- **07 (per-arch dispatch table + bench-gate harness)** is the natural consumer of 08's
  output: 08 produces a *ranked list of cells to build*; 07 builds the dispatch table +
  the bench gate that proves each built cell beats its generic sibling. Sequence 07 to
  read 08's `RankedMatrix`. They are synergistic (the brief for 07 says "feeds 08 +
  10 routing"; the reciprocal is that 08's demand ranking tells 07 *which* cells to
  bench first).

**What this enables downstream:**
- **07** — the ranked cell list is 07's build-order input; the `candidates[]` in a
  `DispatchRecord` is 07's on-device timing corroboration.
- **10 (MatMul design spike)** — routing decisions (fused long-tail vs cuBLAS/CUTLASS)
  are exactly the kind of choice the dispatch feed measures; 08's schema must not assume
  elementwise-only records (keep `op_category` a first-class field so a future GEMM
  record is representable — it already is, `to_token` carries `OpCategory`).
- The **§8 lifecycle** (miss → JIT → AOT): 08's `MissRecord` ingest is the "aggregate
  the miss as demand" arrow (design doc §9); item 05's live JIT seam is the "JIT-generate
  on the miss" arrow. 08 is the batch/AOT-promotion half of that continuum.

## 4. Current code — what exists today

**The token codec (the join key) — shipped, lossless, round-trip-tested.**
`crates/baracuda-kernels-types/src/structure_key.rs`:
- `StructureKey::to_token()` (line 616): emits
  `sk<ver>|<op>|<dtype>|<arch>|<idx>|<work>|r<rank>|<op0>;…|<reduce>`.
- `StructureKey::from_token()` (line 654): parses it back; returns `None` on any
  malformed field or unknown op short-code. This is the ingest primitive — a
  `MissRecord.wanted_structure_key` string is validated by feeding it to `from_token`.
- `structure_key_token()` (line 450) is the single canonical entry point Fuel calls, so
  a token in the feed is byte-identical to one Baracuda would compute for the same
  operands (`fuel-reply-…-2026-06-19.md` §1: "join on the same token by construction").
- `STRUCTURE_KEY_VERSION` (line 48) is the schema version the JSONL `schema` field must
  gate against; `version: u16` rides inside the token itself.

**The e-graph the top-K enumerator extends — shipped.**
`crates/baracuda-kernelgen/src/optimize.rs`:
- `EGraph` (union-find + `class_nodes` + `memo`, line 49), `saturate` (line 331),
  `extract` (line 398) — extraction currently relaxes per-class *min* costs to a fixpoint
  and reconstructs the **single** cheapest form via `build` (line 420).
- `weight` (line 342) is the cost model (Div=8, transcendentals=16, …) — top-K reuses
  it verbatim to rank forms.
- `optimize(e: &ScalarExpr) -> ScalarExpr` (line 446) is the public entry: saturate then
  extract-cheapest. **08 adds a sibling** `optimize_top_k` that returns up to K distinct
  extracted forms, cheapest first. The `Reduced` leaf is deliberately never folded
  across rows (line 36 comment) — top-K must preserve that (no cross-form that CSE's a
  `Reduced`).

**The seam is Elementwise-only, and top-K must respect that boundary.**
`crates/baracuda-kernelgen/src/jit.rs`:
- `region_to_op` hardcodes `access: Access::Elementwise` (line 419).
- `synthesize_op` (line 335) already calls `optimize(&op.body)` (line 365) to build the
  *kernel* body while `derived` (the original region) carries the recipe unchanged (the
  `inward_optimizer_simplifies_kernel_but_keeps_the_recipe` test, line 1086, is the
  invariant top-K must not break: the pattern/decompose still describe the ORIGINAL
  region so Fuel's matcher recognizes it).

**FKC + link emission per cell — shipped (one generator, the four outputs §3 of the
reply names).** `crates/baracuda-kernelgen/src/contract.rs` `contract()` (line 58) emits
the `accept.structure_key` line verbatim (line 104) — the honesty invariant; `link.rs`
`link_entry()` (line 36) + `emit_link_registry()` (line 48) emit the
`(entry_point, structure_key, revision_hash)` roster. A top-K variant that is *promoted*
reuses these unchanged — each distinct form is just another op body fed to the same
generator.

**No telemetry ingest code exists yet.** `grep` for `DispatchRecord`/`MissRecord`/`JSONL`
across `crates/` finds only doc-comment mentions (the `to_token` "telemetry tagging"
docs and the `jit.rs` cost-expr comment). There is **no** `telemetry` module — item 08
creates it.

**The AOT catalog is a hardcoded pilot.** `crates/baracuda-kernelgen/src/bin/kernelgen.rs`
enumerates cells by hand (line 19 onward) and its own header (line 5) says "The
spec-driven matrix (ops × structure cells, eventually fed from Fuel telemetry) …
replace the hardcoded pilot next." Item 08 is that replacement's *input side* — it
produces the ranked cell list `kernelgen` will iterate.

## 5. Design / delta

Three additions, all in a new `crates/baracuda-kernelgen/src/telemetry.rs` module plus a
small `optimize.rs` extension. No IR enum changes, no cuda.rs changes, no contract.rs
changes — the top-K forms flow through the *existing* generate/contract/link path.

### 5.1 Top-K equivalent-form extraction (`optimize.rs`)

Add `optimize_top_k(e: &ScalarExpr, k: usize) -> Vec<ScalarExpr>` alongside `optimize`.
It saturates the same e-graph, then, instead of extracting only the min-cost form,
performs a **k-best extraction**: for the root e-class, enumerate distinct reconstructed
expressions in ascending total cost, deduping structurally-identical results, capped at
`k`. Reuse `weight`/`children`/`enode_cost` unchanged.

```rust
/// Up to `k` distinct algebraically-equivalent forms of `e`, cheapest first
/// (form[0] == optimize(e)). Same precision-safe rule set as `optimize`; each
/// form re-lowers to a compilable kernel and preserves the `Reduced`-no-fold
/// invariant. Deterministic order (ties broken by a stable structural key).
#[must_use]
pub fn optimize_top_k(e: &ScalarExpr, k: usize) -> Vec<ScalarExpr> { … }
```

Implementation note: the current `extract` computes one best per class. For k-best, keep
a small **sorted candidate list per class** (bounded at k) during the relaxation
fixpoint — Lawler/k-best-tree extraction over the e-class DAG. Cost is `O(k · classes)`
per relaxation round; K is tiny (≤ 8), so this is cheap. Guarantee `form[0]` is bit-equal
to today's `optimize(e)` so the existing JIT path is unchanged when `k == 1`.

**Scope guard:** with today's rule set most op bodies have exactly one extracted form
(the identities collapse redundancy but rarely produce *multiple equally-valid*
schedulings). That is *expected and fine*: 08 ships the mechanism + the schema; the
rule set that produces *interesting* K > 1 fan-out (FMA vs mul+add, `x*x` vs `sqr(x)`,
Horner vs naïve `pow`) is the LAST item's growth surface. 08's tests use a rule that
*does* fan out (see §7) to prove the machinery, and the reducer treats K = 1 as the
common case.

### 5.2 v1 JSONL ingest schema (`telemetry.rs`)

Mirror the design-doc §8 `dispatch_record` / `miss_record` shapes exactly, keyed on the
token. Parse with a zero-new-dependency hand-rolled reader (the crate has no serde dep;
records are flat, one JSON object per line). Each record's structure key is *stored as
the token string* and *validated by `StructureKey::from_token`* on ingest — a record
whose token doesn't parse (unknown op code, malformed field, future schema) is a
**skip-with-count**, never a panic (coverage-agnostic ingest, §7 of the reply).

```rust
/// One resolved dispatch (design-doc §8 dataset (a)). The winning impl and its
/// timing at a structure cell — the corroboration that a built variant is worth it.
pub struct DispatchRecord {
    pub schema: u16,                 // gate against STRUCTURE_KEY_VERSION
    pub structure_key: String,       // the join token (validated via from_token)
    pub chosen_impl: ImplId,         // five SEPARABLE fields (reply item 2)
    pub time_ns: u64,
    pub candidates: Vec<ImplId>,     // vendor-exclusion gate input (deferred, §7 reply)
}

/// One planner miss (design-doc §8 dataset (b)) — THE demand signal. "A cell for
/// K would have fit exactly, but it did not exist, so we fell back to F at cost Δ."
pub struct MissRecord {
    pub schema: u16,
    pub wanted_structure_key: String, // the cell to BUILD next (validated)
    pub fallback_impl: ImplId,
    pub est_speedup_if_available: f32,
}

/// The ImplId basis tuple (FKC §4.11 / reply §2.3), kept SEPARABLE on the wire —
/// never hashed into one opaque id (reply item 2).
pub struct ImplId {
    pub backend: String,
    pub op: String,
    pub dtypes: Vec<String>,
    pub kernel_source: String,
    pub kernel_revision_hash: String,
}
```

`ingest_jsonl(reader) -> Ingest { dispatches, misses, skipped }` reads a whole report,
tolerating unknown fields (forward-compat: v1 ⊂ v2, reply §6). `skipped` counts
malformed/future records so the consumer is honest about coverage.

### 5.3 Variant-selection reducer (`telemetry.rs`)

Turn ingested records into a ranked build matrix:

```rust
/// A cell the next release should build, with why. Sorted most-wanted first.
pub struct RankedCell {
    pub key: StructureKey,           // parsed (not just token) so kernelgen can build it
    pub miss_count: u64,             // primary rank key (reply §6: "rank by miss count first")
    pub weighted_speedup: f64,       // Σ est_speedup over misses — tiebreak / value
    pub dispatch_wins: u64,          // corroborating dispatch evidence at this cell
}

pub fn rank_matrix(ingest: &Ingest) -> Vec<RankedCell> { … }
```

Ranking policy (matches the ratified reply §6 "rank the AOT matrix by miss `count`
first, layer in vendor-exclusion as Judge coverage densifies"): **primary = miss_count
descending**, tiebreak = `weighted_speedup` descending, then a stable key ordering for
determinism. `candidates[]`/vendor-exclusion is *carried but not yet used to filter*
(deferred, reply §7 — densifies with no format change as Judge coverage grows).

For **top-K interplay**: when a `DispatchRecord`'s `chosen_impl.kernel_revision_hash`
matches one of several variants Baracuda emitted at the same `structure_key`, the reducer
records the *choice frequency per revision hash* — that is the "turn choice-frequency
into the per-arch dispatch table" deliverable. A `VariantVote { key, revision_hash,
wins }` table falls out of the same aggregation and is 07's dispatch-table seed.

### 5.4 FKC / contract implications

**None to the contract *format*.** The honest-miss invariant is *preserved by
construction*: a promoted top-K variant is emitted through the unchanged
`contract()` + `link_entry()`, so its `accept.structure_key` still equals its cell.
The only new consideration (design-open, §10a): if two variants share one
`structure_key`, they must differ in `entry_point` + `kernel_revision_hash` (they do —
`revision_hash` is over the source, and the two forms produce different `.cu`), and
Fuel's `candidates[]` must be allowed to hold both. That is a Fuel-side ratify, not a
format change.

## 6. Implementation steps

1. **`optimize.rs` — top-K extraction.** Add `optimize_top_k(e, k) -> Vec<ScalarExpr>`
   (k-best e-class extraction reusing `weight`/`children`/`enode_cost`); keep
   `form[0] == optimize(e)`. Export from `lib.rs`. *(No IR change.)*
2. **`telemetry.rs` — record types.** New module with `ImplId`, `DispatchRecord`,
   `MissRecord`, `Ingest`. Flat JSONL reader (no serde; hand-rolled per-line object
   parse) with unknown-field tolerance.
3. **`telemetry.rs` — token-validated ingest.** `ingest_jsonl` validates each record's
   token via `StructureKey::from_token`; malformed/future → `skipped += 1`, never panic.
   Gate `schema` against `STRUCTURE_KEY_VERSION`.
4. **`telemetry.rs` — reducer.** `rank_matrix(&Ingest) -> Vec<RankedCell>` (miss-count
   primary, weighted-speedup tiebreak, deterministic) and the `VariantVote` choice-
   frequency table keyed on `(StructureKey, revision_hash)`.
5. **`lib.rs` — exports.** Add `pub mod telemetry;` and re-export
   `optimize_top_k`, `DispatchRecord`, `MissRecord`, `ImplId`, `Ingest`, `RankedCell`,
   `rank_matrix`, `ingest_jsonl`.
6. **`bin/kernelgen.rs` — spec-driven matrix hook (thin).** Add an optional second CLI
   arg: `kernelgen <out-dir> [telemetry.jsonl]`. When a report is passed, `ingest_jsonl`
   + `rank_matrix` and iterate the ranked cells (fanning the op through
   `optimize_top_k` when a form is worth building) instead of the hardcoded pilot list;
   with no report, keep the pilot (backward-compatible). Emit the same
   `.cu`/FKC/link outputs per cell (unchanged generator).
7. **`OP-MATRIX.md` + `docs/design/kernel-specialization.md` §8.** Add a "telemetry
   consumer (v1 batch)" subsection documenting the JSONL schema + ranking policy; note
   that §8's `dispatch_record`/`miss_record` are now realized as
   `telemetry::{DispatchRecord, MissRecord}`. **Also** correct the stale status text the
   design doc still carries (Param/AddScalar/MulScalar listed as not-emittable though
   shipped) if touched in the same edit — flag it either way.

## 7. Test & on-device validation plan

Item 08 is **pure-CPU** (schema parse, e-graph extraction, reducing) — there is **no
new kernel**, so no nvcc-numeric or compute-sanitizer step is *intrinsically* required
by the change itself. The on-device obligation is *inherited*: any cell the reducer
selects and `kernelgen` emits must still pass the standard generated-cell validation
(nvrtc headerless compile + nvcc numeric diff vs the generic strided oracle, per house
discipline / design doc §10). So:

**Unit (pure-CPU, land with the code):**
- `optimize_top_k`: `k == 1` returns exactly `[optimize(e)]` (invariant with the existing
  JIT path); `form[0]` is cheapest; forms are distinct + cost-ascending; a body with a
  known multi-form equivalence (introduce a *test-only* rewrite fixture, e.g. treat
  `x + x` and `2*x` as equivalent, or `sqr(x)` vs `x*x` via a scratch e-graph) yields
  K > 1 in the proven order; **`Reduced` is never folded across a form** (regression
  guard for the `optimize.rs:36` invariant).
- `ingest_jsonl`: a fixtured report round-trips every record; a line with an
  **unparseable token** (unknown op code `"zzz"`, wrong field count) is skipped-with-count
  and never panics; a `schema` newer than `STRUCTURE_KEY_VERSION` is skipped honestly;
  unknown JSON fields are tolerated (forward-compat); `ImplId` fields stay separable
  (assert the parser never concatenates them).
- `rank_matrix`: given fixtured misses, cells sort by `miss_count` desc then
  `weighted_speedup` desc, deterministically (shuffle input, assert stable output); a
  cell present only in dispatches (no misses) ranks below any missed cell;
  `VariantVote` tallies choice-frequency per `(key, revision_hash)` correctly.
- **Token join fidelity:** a `structure_key_token(op, operands, arch)` produced the
  Baracuda way parses via `from_token` to the *same* `StructureKey` the reducer keys on
  (guards the "join by construction" property end-to-end).

**On-device (only for the emitted-cell path, reuse existing harness):**
- Feed a small fixtured `telemetry.jsonl` naming ~3 real cells (e.g. a broadcast add, a
  strided relu, a contiguous silu) through `kernelgen`; assert each emitted `.cu`
  compiles **headerless under nvrtc on sm_89** (the existing
  `nvrtc_compiles_*` ignored tests are the template) and each **numeric-diffs bit-exact
  (or within declared ULP) against the generic strided oracle** on the RTX 4070 — this is
  the design-doc §10 non-negotiable safety net, inherited unchanged.
- If a top-K variant is emitted (K > 1), assert **all K forms numerically agree** with
  the oracle *and each other* within tolerance — the whole point is they are equivalent;
  a top-K bug that changes semantics is caught here.

**Numeric oracle:** the generic strided kernel (design doc §10 "differential-test every
generated cell against the generic strided oracle"), same oracle the AOT matrix already
diffs against.

## 8. Adversarial-verify checklist

The skeptic pass (multi-agent find → dedup → refute, per house discipline) must probe:
- **Top-K semantic drift:** does any extracted form change the numeric result? Especially
  a form that *reorders* floating-point ops (assoc/distrib) — the current rule set is
  precision-safe, but k-best extraction must not surface a form that a rule set extension
  would make unsafe. Assert every form ≡ `optimize(e)` semantically on random inputs.
- **`Reduced` cross-row fold:** does top-K ever CSE or fold a `Reduced(i)` leaf across
  forms (the `optimize.rs:36` / plan.rs `RowReduce` invariant)? A form that hoists a row
  scalar is a silent correctness bug.
- **Recipe invariant broken:** if top-K is ever wired into the JIT path, does the
  pattern/decompose still describe the ORIGINAL region (the `jit.rs:1086` test)? A form
  that changes the *advertised* pattern makes Fuel's matcher miss.
- **Token ingest panics / lies:** a malformed, truncated, or adversarial token
  (empty, wrong `|`-field count, huge rank, non-ascii, a valid-looking future `sk2|…`)
  must skip-with-count, never panic and never be silently coerced to a wrong cell (a
  mis-keyed record would corrupt the demand ranking).
- **Ranking non-determinism / instability:** does `rank_matrix` produce a stable order
  under input permutation and under equal miss_counts? A non-deterministic matrix
  produces non-reproducible releases (house discipline: determinism is load-bearing).
- **Honest-miss corruption:** does promoting a variant ever emit a contract whose
  `accept.structure_key` ≠ its cell? (It must not — it flows through unchanged
  `contract()`.) Assert the emitted contract's token equals the ranked cell's token.
- **Silent coverage loss:** are skipped records counted and surfaced, or dropped? A
  consumer that silently drops the records it can't parse would under-count demand and
  the maintainer would never know coverage regressed.
- **ImplId hashing:** does any code path collapse the five separable `ImplId` fields into
  one opaque id (violating reply item 2)? That would break exact re-resolution.

## 9. Definition of done

- `optimize_top_k` lands with `k == 1` bit-identical to `optimize`; forms distinct,
  cost-ascending, deterministic; `Reduced` never cross-folded. Unit tests green.
- `telemetry.rs` ships `DispatchRecord`/`MissRecord`/`ImplId`/`Ingest` + `ingest_jsonl`
  (token-validated, panic-free, unknown-field-tolerant, skip-with-count) + `rank_matrix`
  + the `VariantVote` choice-frequency table. Unit tests green (round-trip, malformed,
  ranking-stability, join-fidelity).
- `ImplId` five fields stay **separable** on the wire (never hashed) — asserted.
- `kernelgen` accepts an optional `telemetry.jsonl` and emits the ranked cells through
  the **unchanged** generate/contract/link path; no-report mode keeps the pilot
  (backward-compatible).
- **FKC honest-miss preserved:** a promoted variant's emitted `accept.structure_key`
  equals its cell (asserted); the design-doc §8 loop is realized by these types.
- On-device: fixtured-report cells compile headerless under nvrtc on sm_89 and
  numeric-diff against the generic oracle; multi-form (K > 1) cells all agree.
- Adversarial-verify pass run and findings resolved (or documented).
- `OP-MATRIX.md` + design doc §8 updated to name the v1-batch consumer; the stale
  Param/AddScalar/MulScalar status text flagged/corrected.
- Full workspace test suite green (default + `--features seam`); clippy clean; lockstep
  republish shape honored (all crates bump together) when the export surface changes.

## 10. Open questions / Fuel asks

1. **(Fuel ask) Multiple variants at one `structure_key`.** May Baracuda advertise
   *several* FKC contracts (distinct `entry_point` + `kernel_revision_hash`) at the
   **same** `accept.structure_key`, so Fuel's `candidates[]` can time them head-to-head
   and the dispatch feed votes the winner? The reply's `candidates[]` model (§7)
   suggests yes, but co-freezing that "N kernels, one cell" is admissible needs an
   explicit Fuel confirm (it slightly changes the planner's "best admissible match"
   tie-break from one-per-cell to N-per-cell).
2. **(Fuel ask) JSONL leaf encoding of `chosen_impl` / `ImplId`.** The basis tuple is
   ratified and must stay separable (reply item 2), but the *JSONL field spelling*
   (`chosen_impl: {backend, op, dtypes[], kernel_source, kernel_revision_hash}` vs a
   nested object vs five flat columns) is not frozen. Confirm the object shape so the
   ingest parser targets the real wire, not a guess. This is design-doc §8 step 2
   ("Both: freeze the ImplId wire encoding").
3. **(Fuel ask) `est_speedup_if_available` provenance on a `MissRecord`.** Is it Fuel's
   *estimate* (from cost model) or a *measured* Δ against the fallback? It changes how
   the reducer weights it (a measured Δ is a stronger rank signal than an estimate).
   Coverage-agnostic ingest works either way, but the ranking policy should know.
4. **(Internal, sequence with LAST item) Rule-set fan-out.** Top-K only produces K > 1
   when the e-graph rule set discovers multiple equal-cost forms; today's precision-safe
   rules rarely do. Decide whether 08 ships a *minimal, precision-safe* fan-out rule
   (e.g. `a*b + c` ↔ FMA form, or `x*x` ↔ `sqr(x)`) to give the mechanism real fan-out,
   or defers all fan-out rules to the LAST algebraic-optimizer item and ships 08 with the
   mechanism proven only by a test fixture. Recommendation: ship the mechanism + one
   precision-safe fan-out rule (FMA), defer the rest — but flag that even FMA changes
   rounding, so it must be gated as `approximate`/opt-in, not default.
5. **(Internal, ties to 07) v1-batch report location + cadence.** Where does Fuel drop
   the aggregated JSONL (repo path in the `fuel-*.md` channel? a build artifact?) and how
   often — per release, per CI run? 08's `kernelgen` hook needs a stable path convention;
   coordinate with 07's bench-gate harness so both read the same report.
