# 11 — Variant generators: the measured-tradeoff backlog

> Phase-2 charter, following the pure-wins pass (2026-07-02). The generator's
> unconditional ("pure win") transforms are in place; everything below is a
> **tradeoff** — a transform that helps some cells and hurts others, or changes
> bits. The architecture for all of them is the same and is already built at its
> ends: emit **N candidate schedule variants per cell**, gate them through the
> item-07 bench harness (`gate_cell` → `winner_of` → `merge`), commit the
> measured winner in the dispatch table, and let item-08 telemetry refresh it.
> What's missing is only the middle: a variant-emission loop in the generator.

## Ground rules (from the pure-wins pass)

- **Value-preserving + never-worse ⇒ apply unconditionally.** No gate, no bench.
- **Value-preserving but sometimes-slower ⇒ variant.** Both kernels are correct;
  the dispatch table routes per cell/arch. Silent selection is fine.
- **Bit-changing ⇒ variant + contract change.** A reassociated reduction or an
  approximate packed transcendental must flip the FKC `determinism`/`precision`
  block honestly; it may only be selected under a policy that permits it
  (`PrecisionGuarantee`-style), never silently.
- **Measure, don't assume.** The empirical caution: the item-09 brief predicted
  packed f16 as "the largest unclaimed memory-bound win"; on sm_89 the scalar
  2-byte kernel already runs at the DRAM ceiling (the coalescer does the work),
  and packing wins only +3–8% in the L2/instruction-bound regime. Static
  intuition about GPU perf mispredicts; the gate decides.

## The variant backlog

| # | Variant | Bits | Why it's a tradeoff | Notes |
|---|---|---|---|---|
| 1 | **Cross-pass materialize-vs-recompute** (RowReduce/reduction shared values, e.g. Softmax's `exp(x−max)` reused across fold + epilogue) | preserved | registers/smem spent vs recompute; occupancy-cliff risk is unpredictable statically | needs `from_region` (one DAG spanning stages+epilogue) + a materialization schedule axis; keys off item-02 `consumers` |
| 2 | **Packed Tier-A transcendentals** (`h2exp`, `h2sin`, …) | **changed** (approximate intrinsics ≠ scalar float round-trip) | ~2× on transcendental-heavy f16 bodies vs a determinism-contract relaxation | Fuel policy ask: does any consumer rely on bitwise f16 transcendentals? |
| 3 | **Packed reductions (item 09 Stage 2)** — `__hadd2` tree fold | **changed** (reassociates the fold) | bandwidth win vs determinism contract + the load-bearing block-tree invariants | `racecheck`/`synccheck` mandatory; sequence after a measured demand signal |
| 4 | **Split-K outer-axis reduction** | changed (partial-sum association) | the 118 GB/s outer-axis path → toward peak, at extra pass + workspace cost | additive via `ReduceAxisClass`; no re-key |
| 5 | **Grid-stride ILP unroll** (×2/×4 with runtime-step remainder loop) | preserved | ILP win only when latency-bound; code bloat + icache pressure otherwise | NOT the compile-time remainder-free form: with a runtime `step`, remainder-free needs consecutive-per-thread indexing, which breaks warp coalescing — that's why this isn't a pure win |
| 6 | **WorkClass-driven schedules** (`OneWarp`/`OneBlock` single-block kernels for tiny work) | preserved | kills grid-stride overhead for tiny cells; a second kernel + launch contract per cell | the key already carries `WorkClass`; the emitter ignores it today |
| 7 | **Occupancy-aware launch config** | preserved | not codegen — a runtime-side heuristic the gate should *hold fixed* while comparing kernels | feeds `gate_cell` methodology more than the emitter |
| 8 | **Contraction schedule axes** (K-tile, register/shared blocking, MMA fragment, `cp.async` depth) | fragment-dependent | the classic autotuned GEMM surface | item 10's implementation phase; the design spike already scopes it |

**Reclassified out of the pure-wins list (with reasons, so nobody re-promotes
them without data):**

- **DivBucket-driven unroll/remainder-drop** → variant #5. The divisibility fact
  can only drop bounds checks under consecutive-per-thread indexing, which
  de-coalesces; the coalesced form keeps a remainder loop and is a plain ILP
  tradeoff. The `DivBucket` key axis still pays for itself via vec-width
  derivation.
- **InnerContig inner-loop vectorization** (2D contiguous-rows elementwise) →
  its own keyed item, not a silent variant: `classify_vec_width` returns
  `Scalar` for `InnerContig` today, so admitting it changes *keying* — a
  `STRUCTURE_KEY_VERSION` bump, Fuel-visible, propose-first.

## Who decides — ship top-K to Fuel (policy, pinned 2026-07-02)

**Default: ship every validated variant, not just the bench winner.** Fuel is
the runtime decision-maker by design (design §8: it already times and compares
every implementation available and picks the best), and three mechanisms keep it
in that seat: `DispatchEntry.ranked` is top-K, `merge()` treats Fuel's
`Reported` rows as overriding authority, and every shipped variant carries its
own FKC contract under the same `accept.structure_key` so the planner can pick
among them. The item-09 mispredict is the standing argument: our synthetic
bench measures regimes a real workload may never hit — shipping winner-only
would bake a bench guess in where Fuel could have known better.

- The dispatch table is Baracuda's **own default route and a seed/prior offered
  to Fuel** — not a gatekeeper on what ships.
- Winner-only shipping is reserved for cells where a variant is **strictly
  dominated** (never wins any regime on any measured arch) or where catalog
  size demands a cap (cap per cell, drop dominated variants first).
- **Bit-changing variants require Fuel-decides**: only the caller knows its
  precision requirements, so a reassociated/approximate variant is selectable
  only through its honest contract, never silently by Baracuda.

## The connective tissue to build first

1. **Variant-emission loop** (`bin/kernelgen.rs` or a `variants.rs` module): for
   a cell, emit `[(variant_tag, GeneratedKernel)]` instead of one kernel. Symbol
   names carry the tag (`…_co_v8`, `…_co_v8_unroll4`, …). Each variant must pass
   the same nvrtc/nvcc/sanitizer gate before it is *rankable* (the `gate_cell`
   correctness precondition).
2. **Gate wiring**: feed the variant set of a cell to `gate_cell` on the target
   arch; `merge` the measured winner into the committed dispatch table. The
   `MIN_FLIP_MARGIN` noise floor already prevents flip-flopping on ±5% noise —
   which matters, because several variants above live exactly in that band.
3. **Contract honesty per variant**: a bit-changing variant emits its own
   `determinism`/`precision` block; the table records which variant won so the
   FKC contract served for the cell always describes the kernel actually shipped.
