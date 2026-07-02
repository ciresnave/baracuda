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

## Populator seam — LANDED (2026-07-02, `baracuda-kernels-bench/src/lib.rs`)

1. **`current_hwstamp(device)`** — **DONE.** Bench-side constructor: maps the
   device's compute capability to `ArchSku` via `arch_sku_of` (sm_89→Sm89,
   other sm_8x→Sm80, sm_9x→Sm90a), fills `device_name` (`Device::name`) +
   `cuda_version` (`baracuda_driver::version`) + wall-clock `captured_unix_s`
   (the bench crate, not the deterministic types crate). Off-device unit test for
   the mapping; on-device smoke confirms the RTX 4070 stamps as sm89.
2. **`gate_cell(ctx, stream, key, measured_on, samples, inner, candidates)`** —
   **DONE.** Times each `(Implementor, entry_point, Box<dyn FnMut() + 'a>)`
   candidate via `measure_median_ns`, builds `Vec<CandidateResult>`, reduces with
   `winner_of` → a `Provenance::Measured` `DispatchEntry` ready to `merge`.
   Correctness stays the caller's precondition (documented on the fn, matching
   `winner_of`): only pre-validated candidates are timed; a generated candidate
   must have passed the nvrtc/nvcc/sanitizer gate first. On-device smoke times two
   candidates → measured entry → `merge` over a seed.

## Deferred — the specific-bench wiring (plumbing over the landed seam)

1. **CSV extension** — add additive `structure_key: Option<String>`, `arch`,
   `implementor` columns to `PhaseTwentyNineRow` (keep existing columns; bump the
   header). Emit one row per candidate.
2. **Wire `gemm_vs_cublas`** — for each `(M,K=N,dtype)`, build the `StructureKey`,
   run `gate_cell` over `{cuBLAS, CUTLASS-bespoke}` (generated absent — an honest
   empty slot), append gated rows. Then `elementwise.rs` (generated vs bespoke)
   once kernelgen emits the matching cell. This needs a **correctness oracle** per
   candidate (diff vs the generic strided kernel) before timing — `gate_cell`
   assumes pre-validated candidates.
3. **CSV → table reducer** — a tool (extend `tools/build_benchmarks_table.py` or a
   Rust bin) that reads gated CSVs, calls `merge()` over the seed table, and
   regenerates the committed `dispatch_table.rs` via `emit_dispatch_table`.
4. **Contract provenance** — thread the cell's dispatch decision into
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
