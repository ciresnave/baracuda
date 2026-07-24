# kiss-ref-diff — the Baracuda ↔ kiss-ref recipe differential harness

The oracle-consolidation differential (the `oracle.rs` → kiss-ref plan): takes
Baracuda's **real emitted recipe surface** (`baracuda_kernelgen::recipe::
semantics_dag`, the KISS-Contract §2.3 Semantics text every contract carries),
converts it to a kiss-ref `FlatDag`, and differentially evaluates it against
Baracuda's own legs:

```text
OpDef ──semantics_dag──▶ functional text ──converter──▶ FlatDag
  │                                                        │
  ├── kernelgen CPU oracle ────────────┐    ┌── kiss-ref eval_recipe
  └── generated CUDA kernel (NVRTC) ───┤    │
                                       ▼    ▼
                       bit-exact / §6.8-conforming compare + DetClass checks
```

Op names resolve through `kiss_ops_vocab::Op::from_token` — the same closed
KISS-Ops token set the emitter re-bases onto — so the converter carries **no
duplicated name table** (the drift the consolidation exists to kill).

## Coverage (2026-07-23)

- **Step 2, elementwise**: relu_add / affine (runtime scalars) / max_prop over
  ±0.0, NaN, ±inf — CPU-oracle leg bit-for-bit.
- **Step 2b, folds**: `reduce[monoid,axes,kd]` (rank-relative `last`),
  `prefix_scan`, `matmul[mk.kn]`, `reduced_count` (float Mean composes);
  reverse scan = an emission-level honest miss (`flip` is not a registered
  KISS-Ops token). Honest DetClass asserted (float folds ≠ ExactByte).
- **Step 3a, on-device**: NVRTC-JIT'd generated kernels on the live GPU vs
  kiss-ref, under the §6.8 conforming comparator (both-NaN equal — arithmetic
  REMINTS NaN payloads per device; ±0 stays one ULP apart).
- **Step 3b (partial), on-device folds**: the block-tree last-axis reduce
  kernel (`sum` + NaN-propagating `max`) vs kiss-ref's serial fold —
  different association, bit-stable on exactly-representable values (the OIN
  class collapsing, per kiss-ref's tolerance-basis preview). The
  general-value tolerance leg joins when kiss-ref's fold-depth-scaled bound
  arrives.
- **Step 2c, indexed ops**: gather/scatter through `IndexRef::Slot` with the
  full lane split (value binds renumber past index operands; index data rides
  `indices`); scatter's explicit dest synthesized (the §6.11 explicit-dest
  surface, pure form of Baracuda's in-place posture). Bincount's rank-0
  `const(1)` updates is a pinned OPEN SEAM (typed `ShapeMismatch`) pending the
  §6.11/#67 broadcast-updates ruling.

## Findings this harness produced

1. `max_prop`/`min_prop` tie bias `a > b ? a : b` in ALL THREE Baracuda
   backends (spec says a-on-ties) — fixed `7297f17d`; KISS A.3 erratum #74.
2. sm_89 arithmetic canonicalizes PRODUCED NaNs to `0x7fffffff` vs x86's
   propagated `0x7fc00000` — the §6.8 both-NaN comparator rationale, now in
   kiss-ref's diff docs.
3. Reverse-scan recipes fabricated the unregistered `flip` token — withdrawn
   (`e3530a39`, the #68 anti-fork witness).
4. KISS's staged sk3 codec ACCEPTED the reserved fnuz spellings — fixed
   KISS-side with a typed `ReservedDtype` decline.
5. The bincount rank-0-updates expressibility seam (above), routed to ruling.

## Building

**NOT a workspace member** (deliberate): kiss-ref is a PRIVATE repo, so the
git deps here cannot enter the published workspace or default CI. Requires
GitHub auth for `ThinkersJournal/kiss-ref` and the CLI fetcher:

```sh
CARGO_NET_GIT_FETCH_WITH_CLI=true cargo run --manifest-path tools/kiss-ref-diff/Cargo.toml
```

The 3a device leg needs a CUDA driver + NVRTC (it JITs for the detected
compute capability). `POC_DUMP=1` prints each generated kernel's source.

When kiss-ref publishes (or the converter's final in-tree home lands with the
#67-consolidated grammar), this tool's converter is the reference
implementation to promote.
