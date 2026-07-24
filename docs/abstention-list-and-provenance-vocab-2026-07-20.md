# Abstention list + shared provenance/lineage vocab (prep)

**Status:** prep for two co-design artifacts owed to KISS (44elbk9y) and kiss-ref (nnb3tadk). Both go to Eric to bless before they're official (both move Baracuda's freeze-gate voice / touch the provenance surface).

---

## 1. Freeze-gate abstention list (C-4)

**What it is (44elbk9y accepted, wants it drawn up):** the list of §8 freeze-gate clauses where Baracuda **abstains/caveats** because kiss-ref's reading of that clause traces back to Baracuda's `oracle.rs` — so a kiss-ref↔oracle.rs agreement there is "one mind agreeing with itself," not independent evidence. Baracuda keeps its full independent voice on every clause it does *not* share lineage with (most of them). A partial mitigation, not a gate-clearing (per Eric's E6 ruling; the gate stays open pending genuinely external implementors).

**The set is an intersection** — needs both halves:
- **kiss-ref's half (authoritative, owed by nnb3tadk):** exactly which §6.13 decomposition / KISS-Ops clauses kiss-ref derived by *reading* `oracle.rs` as prior art.
- **Baracuda's half (below):** what `oracle.rs` actually covers.

**Baracuda's half — `oracle.rs` coverage (`baracuda-kernelgen/src/oracle.rs`):** an independent CPU plan-interpreter that shares **zero lowering code** with the emitter but shares the upstream `build_plan`/IR types. It re-implements each op's semantics from its *definition*. Coverage domains:
- **Scalar primitive floor** — the arithmetic / comparison / rounding / bitwise atoms + the transcendental atoms. *(Here oracle.rs is a STRONG independent check; kiss-ref reading it does NOT create meaningful comprehension correlation on the primitives — the floor is well-pinned.)*
- **Structural region** — reduce / scan / gather / scatter / matmul-contraction (the `§6.13` decompositions). *(Here is the correlation risk: a shared reading of a §6.13 decomposition between oracle.rs and kiss-ref is one comprehension. This is the region that populates the abstention list.)*

**Method to build it:** intersect kiss-ref's derived-from-oracle.rs clause list with the structural-region decompositions above. The primitive-floor clauses are NOT abstention candidates (oracle.rs is a genuinely strong independent floor check); the §6.13 structural decompositions ARE. Draft → Eric blesses → send to 44elbk9y to record against KISS freeze-gate independence tracking.

**Better mechanism (gubawx2d's refinement, adopted):** per-clause abstention, not "count Baracuda+kiss-ref as one lineage wholesale" — preserves Baracuda's independent voice on every non-shared clause.

---

## 2. Shared provenance / lineage-tag vocab (with kiss-ref)

**Status:** field set agreed with kiss-ref (nnb3tadk signed off); ready to bring to Eric (touches the provenance surface — #60 / PROVENANCE.md / the FKC contract).

**The agreed set:**
```
{ op_id,
  semantics_clause,        # KISS-Ops §ref the cell realizes
  realization_source,      # enum: reference-core | generated | ported-from<project>
  version_or_commit,
  numeric_basis,           # libm | cuda-intrinsic | tensor-core | …
  determinism_class,       # exact-byte | ulp/tolerance | order-invariant/nondeterministic
  accuracy,                # max_ulp | correctly-rounded
  evaluation_precision,    # {compute-dtype | wider-than-compute}   ← STANDARD (kiss-ref)
  derivation_lineage }     # {spec-6.13-table | external-cold-reader} ← STANDARD (kiss-ref)
```

**The two load-bearing additions (kiss-ref, both promoted to STANDARD):**
- **`evaluation_precision`** — distinguishes a hardware-precision reference (kiss-ref = `compute-dtype`) from a wide-precision one (§6.5 oracle / 256-bit core = `wider-than-compute`). Sets the comparison band (within-declared-ceiling vs truth). Single source of truth: this enum is authoritative; `certificate_precision_bits` (§6.5-0009) is a detail present only under `wider-than-compute` — never a free-standing field that can drift from it (44elbk9y's framing).
- **`derivation_lineage`** — makes the E6 comprehension-correlation *machine-visible*: `{kiss-ref, oracle.rs, §6.5-oracle}` all = `spec-6.13-table`, so tooling won't double-count them as independent. Operationalizes the abstention list (§1) in the attribution surface.

**Mapping:** Baracuda's fields → `Fresh`/`generated` → `realization_source: generated`; PROVENANCE.md cell/token → `op_id` + `version_or_commit`; contract determinism/accuracy → `determinism_class` + `accuracy`. kiss-ref's `Fresh` → `reference-core`; `PortedFromFuel` → `ported-from:fuel`.

Next: Eric blesses → send final to 44elbk9y (record) + nnb3tadk (both emit the same attribution vocab).
