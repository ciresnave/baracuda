# 07 (follow-up) — Bench-gate populator for the dispatch table

> Continuation of `07-perarch-dispatch-table-bench-gate.md`. The **routing-oracle
> core** landed (schema + decision logic + committed artifact + vendor-exclusion
> seed, all off-device unit-tested). This brief is the remaining half: the
> **on-device populator** that measures candidates and merges `Measured` rows
> over the seeds. Self-contained for a fresh session.

---

## What landed (branch `feat/kernelgen-dispatch-table`)

- **Schema + logic** — `crates/baracuda-kernels-types/src/dispatch.rs`:
  `Implementor` (`#[non_exhaustive]`, `is_vendor()`), `Provenance`
  (`is_observed()`), `HwStamp`, `CandidateResult`, `DispatchEntry`,
  `DispatchTable` (arch-gated `lookup`, sorted+deduped `normalize`), and the pure
  reducers `winner_of` (min-median winner, `margin = second/first`, empty ⇒ honest
  `None`), `seed_winner` (large aligned half-GEMM → cuBLAS predicate), `merge`
  (seed-never-demotes-observed, observed-overrides-seed, `MIN_FLIP_MARGIN=1.10`
  flip guard, newer-wins, arch-gated `Reported`). 17 off-device tests.
- **Artifact serializer** — `crates/baracuda-kernelgen/src/dispatch_artifact.rs`:
  `emit_dispatch_table` (committed `@generated` `BARACUDA_DISPATCH_TABLE:
  &[(&str,&str,f64,&str)]`, sorted/deduped/byte-stable) + `parse_dispatch_table`
  (round-trip identity, token-codec-stability guard). 5 tests.
- **Generator wiring** — `bin/kernelgen.rs`: consults `seed_winner` before
  generating (honest-miss `debug_assert`), emits `dispatch_table.rs` seeded with
  the two vendor-routed GEMM cells (f16/bf16), which are *not* generated.
- **Docs** — design §7 status flipped to "mechanism landed"; `OP-MATRIX.md`
  gained the dispatch/vendor-exclusion pointer.

The `merge()` seam is item 08's Fuel-feed ingest point; nothing below changes the
schema — the populator only *produces* `DispatchEntry`s to feed it.

---

## Deferred — the populator (steps 6–8 of the parent brief)

1. **`HwStamp::current()`** — a bench-side constructor (in `baracuda-kernels-bench`
   or behind a `baracuda-driver` query) filling `arch` + `device_name` +
   `cuda_version`. Inject `captured_unix_s` from the caller (keep the types crate
   wall-clock-free). This is the provenance stamp every `Measured` row carries.
2. **`gate_cell()`** — in `crates/baracuda-kernels-bench/src/lib.rs`: given a cell
   `StructureKey` and a set of `(Implementor, entry_point, &mut dyn FnMut())`
   candidates, time each via the existing `measure_median_ns`, build
   `Vec<CandidateResult>`, and reduce with `winner_of`. **Correctness gate first**:
   a candidate is admitted only after its output matches the strided oracle within
   tolerance (design §10) — a fast-but-wrong candidate is rejected, never ranked.
   A generated candidate that has not passed nvrtc-headerless + nvcc-numeric +
   compute-sanitizer must not be rankable (precondition assert).
3. **CSV extension** — add additive `structure_key: Option<String>`, `arch`,
   `implementor` columns to `PhaseTwentyNineRow` (keep existing columns; bump the
   header). Emit one row per candidate.
4. **Wire `gemm_vs_cublas`** — for each `(M,K=N,dtype)`, build the `StructureKey`,
   run `gate_cell` over `{cuBLAS, CUTLASS-bespoke}` (generated absent — an honest
   empty slot), append gated rows. Then `elementwise.rs` (generated vs bespoke)
   once kernelgen emits the matching cell.
5. **CSV → table reducer** — a tool (extend `tools/build_benchmarks_table.py` or a
   Rust bin) that reads gated CSVs, calls `merge()` over the seed table, and
   regenerates the committed `dispatch_table.rs` via `emit_dispatch_table`.
6. **Contract provenance** — thread the cell's dispatch decision into
   `contract.rs` so `cost.provenance` reads `declared`/`measured`/`vendor`, and
   `contract()` returns `None` for a `vendor` cell (no bindable contract). Preserve
   the `fkc_dtype`-`None` honest-miss path.

## On-device validation (RTX 4070 / sm_89)

- Run gated `gemm_vs_cublas`; assert every row carries a non-empty
  `structure_key`, `arch: sm89`, ≥1 `CandidateResult`, and the produced
  `HwStamp` reports sm89 + the RTX 4070 device name.
- Numeric oracle gates ranking (wrong candidate rejected).
- Sanity: the gate's winner for a cell `BENCHMARKS.md` already characterizes
  (f32 GEMM `M128_N4096_K4096` → baracuda ~2×) must agree with the table.

## Adversarial-verify — the checklist that already shaped the core

Winner-from-noise (margin < `MIN_FLIP_MARGIN` can't flip — **done** in `merge`);
stale/foreign measurement (arch-gated `merge`/`lookup` — **done**); honest-miss
corruption (vendor cell emits no contract/link — **wire in step 6**);
fast-but-wrong ranked (oracle before ranking — **step 2**); token drift
(`parse_dispatch_table` round-trip + `from_token(to_token)` — **done**);
non-determinism (byte-stable artifact — **done**); empty-candidate cell
(`winner_of` ⇒ `None` — **done**).
