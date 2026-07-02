# 07 — Per-arch dispatch table + benchmark-gate harness — implementation brief

> Scope owner: Baracuda kernel-specialization initiative (`feat/kernel-specialization`).
> This brief is self-contained: a fresh Claude Code session with no prior context
> should be able to execute it end-to-end. Cite `file:line` before you touch code;
> the citations below were verified against the tree at alpha.72 (2026-06-30).

---

## 1. Objective

Build the mechanism from `docs/design/kernel-specialization.md` §7 — *"Excluding
vendor-owned cells"* — that makes **measurement**, not hand-blocklists, the durable
rule for which implementation serves a given `(op, structure-key, dtype, arch)` cell.
Concretely, deliver two artifacts: (a) a **benchmark gate** — a harness that, for a set
of cells, times every candidate implementation available (generated kernel vs
cuBLAS / CUTLASS / bespoke `.cu`) on the target arch and records the measured winner
with its margin; and (b) a **per-arch dispatch table** — a committed, `StructureKey`-token-keyed
build artifact that records, per cell, `winner + margin + provenance`, seeded with
hand-knowledge (e.g. "route large-aligned GEMM to cuBLAS") but overwritten by
measurement. This is foundational because it is the *routing oracle* the rest of the
initiative reads: item **08** (telemetry variant-selection) ingests and updates this
table from Fuel's `dispatch_record`/`miss_record` feed, and item **10** (MatMul design
spike) needs exactly this per-cell gate to justify its "route large GEMM to cuBLAS,
generate only the fused long-tail" vendor-exclusion split. Without a measured gate the
generator has no principled way to *not* generate a cell, and no honest basis to claim a
generated cell is worth shipping.

---

## 2. Status & blockers

**Baracuda-unblocked (build now):**
- The dispatch-table *schema* + serializer/deserializer (new module in
  `baracuda-kernelgen`, or a small sibling crate). The token codec it keys on already
  exists and round-trips (`StructureKey::to_token`/`from_token`,
  `crates/baracuda-kernels-types/src/structure_key.rs:616,654`).
- The **bench-gate harness** for the cells Baracuda can already synthesize *and* has a
  vendor/bespoke sibling to diff against — i.e. the elementwise + RowReduce families the
  generator emits today (`bin/kernelgen.rs`) vs the shipped bespoke kernels benched in
  `crates/baracuda-kernels-bench`. The harness (`measure_median_ns`, `time_with_events`,
  `append_csv_row`, `PytorchBaseline`) is already the right shape — this item *extends*
  it to emit a `structure_key`-tagged, per-candidate CSV and reduce it to a winner table.
- The hand-knowledge **seed rules** (a small declarative table of "these cells route to
  vendor, don't even generate") — pure data, no device needed.

**Fuel-blocked (the durable *feed*, not the mechanism):**
- The §8 feedback loop that *populates* the table from real workloads
  (`dispatch_record`/`miss_record`, design doc lines ~269-297) is item **08**'s
  consumer and depends on Fuel emitting those records against our `StructureKey` token.
  This item must build the table so that 08 can update it, but must **not** wait on the
  feed: the local bench gate is the v1 populator, Fuel's feed is the v2 populator. Design
  the ingest seam (a `merge(records) -> table` entry point) now; wire it when Fuel answers.

**Design-open (decide in §10):**
- Whether the table lives inside `baracuda-kernelgen` (build tool, `publish = false`) or
  in `baracuda-kernels-types` (so the *runtime* dispatcher in the sys/plan crates can read
  it without a dev-dep). Leaning types-crate for the *schema* + a committed artifact the
  sys crate `include!`s, mirroring how `link.rs` emits `BARACUDA_LINK_REGISTRY` today.
- Whether "winner" is per-cell-scalar or a ranked top-K (08 wants top-K; §8 line ~281
  carries `candidates_considered[]`). Recommend top-K from the start so 08 is a data feed,
  not a schema change.

---

## 3. Dependencies & sequencing

**Must land before this: nothing hard-blocks the harness + table schema.** The
`StructureKey` token codec, the `link.rs` registry pattern, and the bench harness all
exist. Two *soft* dependencies improve coverage but are not gating:
- **10 (MatMul design spike)** is the highest-value *consumer* of the vendor-exclusion
  gate but is design-only and terminal; this item should ship the gate *mechanism* with
  GEMM as the worked seed example (route-large-to-cuBLAS) even though kernelgen cannot yet
  *generate* a GEMM. The gate compares whatever candidates exist — for GEMM today that is
  {cuBLAS, CUTLASS bespoke}, with "generated" absent (an honest empty slot).
- **01 (layout/shape nodes)** and **03 (strided reductions)** widen the set of
  *generatable* cells the gate can compare; as they land, the same gate simply sees more
  candidates. No coupling in the other direction.

**What this ENABLES downstream:**
- **08 — telemetry variant-selection** reads/updates this table; the top-K emission +
  JSONL ingest schema in 08 is literally this table's `merge()` seam plus a Fuel feed.
- **10 — MatMul/contraction** uses the gate as its "vendor-exclusion" decision function:
  the design spike's schedule-axes/StructureKey work is scoped by *what cells the gate
  says are worth generating* (the fused long-tail) vs *what routes to cuBLAS/CUTLASS*.
- **05/06 (RowReduce seam + fused LayerNorm catalog)** gain a measured basis for the AOT
  catalog: "which fused-norm cells beat the bespoke sibling by enough to ship."

---

## 4. Current code — what exists today

Everything below is present; nothing here is a dispatch table or a measured gate yet — that
is the gap this item fills.

### 4.1 The key the table is keyed on (exists, complete)

`crates/baracuda-kernels-types/src/structure_key.rs`:
- `StructureKey` (`:188`) carries `op: OpCategory`, `dtype: ElementKind`,
  `arch: ArchSku`, `idx`, `work`, `rank`, and per-operand `OperandKey`s. It is `Copy` +
  `Hash` + `Eq` — "so it can be hashed into a dispatch table or an autotuner cache
  directly" (`:184-186`, the rustdoc literally names this use).
- `to_token()` (`:616`) / `from_token()` (`:654`) — a lossless, greppable string codec:
  `sk1|bin|f32|sm89|i32|grid|r2|co/00/v4/d16/f;…|-`. Round-trips (test `:992`). This is the
  join token for the table, the FKC `accept` predicate, and the telemetry schema — one
  token, three consumers (module doc `:5-11`).
- `structure_key_token()` (`:450`) — the one-call convenience Fuel uses.
- `ArchSku` (`crates/baracuda-kernels-types/src/layout.rs:43`) = `{Sm80, Sm89, Sm90a}`,
  **intentionally not `#[non_exhaustive]`** (`:34-41`) — adding an arch is a deliberate
  build-break event across every match site. The dispatch table is per-`ArchSku` by
  construction.
- `OpCategory` (`crates/baracuda-kernels-types/src/sku.rs:31`) — 24 categories incl.
  `Gemm`, `BinaryElementwise`, `Normalization`, `Softmax`, `Reduction`. Token short-codes
  in `op_code`/`op_from_code` (`structure_key.rs:860,892`).

### 4.2 The generator's emission driver + link registry (exists; the shape to extend)

- `crates/baracuda-kernelgen/src/bin/kernelgen.rs` — the CLI that fans an op over dtype
  cells and writes `.cu` + `.fkc.pattern` files. **Hardcoded matrix, no `build.rs`**
  (confirmed: `kernelgen` has no `build.rs`). This is where a "for each cell, emit +
  register a dispatch-table row" loop belongs.
- `crates/baracuda-kernelgen/src/link.rs` — `emit_link_registry(&[LinkEntry]) -> String`
  (`:48`) emits `pub static BARACUDA_LINK_REGISTRY: &[(&str, &str, u64)]`
  (`entry_point`, structure-key token, revision hash). Sorted + deduped, `@generated`, a
  committed Rust file the sys crate resolves at load. **The dispatch table is the exact
  same artifact pattern**: a committed `@generated` static keyed by token — but it records
  the *winner* per cell, not just the roster. `LinkEntry` (`:24`) is the row shape to mirror.

### 4.3 The FKC contract's `cost` block (exists; *declared*, not measured — the gap)

`crates/baracuda-kernelgen/src/contract.rs`:
- `contract()` (`:58`) emits a `cost:` block (`:135-139`) with
  `provenance: declared`, `class: elementwise`, `flops_per_elem`, `bytes_per_elem`. The
  literal string `provenance: declared` (`:136`) is the honest marker that today's cost is
  a *static estimate*, not a measurement. **The dispatch table is what lets a cell graduate
  to `provenance: measured`.** Keep this block; the table is the measured sibling.
- `awkward_layout()` (`:280`) and `required_align()` (`:289`) already project the cell's
  routing-relevant facts. Honest-miss is preserved by `fkc_dtype()` (`:329`) returning
  `None` for un-spellable dtypes so `contract()` returns `None` rather than fake a binding.

### 4.4 The bench harness (exists; the timing engine to reuse)

`crates/baracuda-kernels-bench/src/lib.rs`:
- `measure_median_ns(ctx, stream, samples, inner, launch) -> f64` (`:336`) — CUDA-event
  timed median (11 samples × 50 inner). `time_with_events` (`:96`) and `warmup` (`:133`)
  are the primitives.
- `PhaseTwentyNineRow` (`:276`) + `append_csv_row` (`:363`) — writes
  `op,shape,dtype,baracuda_ns,reference_ns,reference,delta,pytorch_ns,pytorch_delta` to
  `target/criterion/phase29/<bench>.csv`. `delta = reference_ns / baracuda_ns`. **This row
  is per-op-family, not per-`StructureKey`, and carries no arch or candidate-name column —
  the gap this item closes.**
- `PytorchBaseline` (`:567`) — a frozen-JSON external baseline with a self-describing
  metadata block (device name, capability, torch/cuda version — `:527`). The dispatch
  table needs the *same* provenance discipline: every measured row must record the arch +
  hardware it was measured on.
- Cross-impl benches exist and already time vendor vs baracuda:
  `benches/gemm_vs_cublas.rs`, `softmax_vs_cudnn.rs`, `layernorm_vs_cudnn.rs`,
  `reductions_vs_cudnn.rs`, `rmsnorm.rs`, `elementwise.rs`. `BENCHMARKS.md` is the
  hand-rolled roll-up (e.g. the GEMM `delta` table at lines ~342-361 already *is* an
  informal, un-keyed dispatch table — "cuBLAS wins at M=1, baracuda wins at M=128,4096²").
  This item formalizes that into a machine-keyed artifact.

### 4.5 What does NOT exist (the whole of this item)

- No `DispatchTable` / dispatch-record type anywhere (grep for `dispatch_table` /
  `DispatchTable` / `structure_key_token` across `crates/` finds only the token codec,
  the link registry, and the contract — no winner table).
- No `structure_key`-tagged bench row and no per-candidate (cuBLAS vs CUTLASS vs generated)
  timing emission.
- No seed rule-set for vendor-owned cells.
- The design doc §7 (`kernel-specialization.md:228-241`) describes the mechanism; §8
  (`:245-297`) describes the Fuel feed. Both are prose; no code.

---

## 5. Design / delta

### 5.1 The dispatch-table schema (new type, in `baracuda-kernels-types`)

Put the *schema* in `baracuda-kernels-types` so the runtime dispatcher (in the plan/sys
crates) can read it without depending on the build-only `baracuda-kernelgen`. The
serialized *artifact* is emitted by `baracuda-kernelgen` (like `link.rs`) and read by
whoever routes.

```rust
// crates/baracuda-kernels-types/src/dispatch.rs  (new)

/// A candidate implementation for a cell — the join target of the gate.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]                         // more backends come; match sites reject unknown
pub enum Implementor {
    Generated,   // baracuda-kernelgen emitted .cu (the thing we're deciding to ship or not)
    Cublas,
    Cutlass,
    Cudnn,
    Bespoke,     // a hand-written baracuda-kernels-sys .cu
}

/// One measured candidate result for a cell.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateResult {
    pub implementor: Implementor,
    pub median_ns: f64,
    pub entry_point: Option<String>,   // link-registry symbol when Generated/Bespoke
}

/// Why a winner is what it is — measured vs seeded vs Fuel-reported.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Provenance {
    Measured,   // this repo's bench gate, on `measured_on`
    Seeded,     // hand-knowledge default, never benched
    Reported,   // Fuel dispatch_record from a real workload (item 08)
}

/// One row: the decision for a single (op, structure-key, dtype, arch) cell.
/// `key` already encodes op+dtype+arch, so the token IS the row key.
#[derive(Clone, Debug, PartialEq)]
pub struct DispatchEntry {
    pub structure_key: String,          // StructureKey::to_token()
    pub winner: Implementor,
    pub margin: f64,                    // second_best_ns / winner_ns (>1 ⇒ real win)
    pub ranked: Vec<CandidateResult>,  // top-K, winner first (08 reads this)
    pub provenance: Provenance,
    pub measured_on: Option<HwStamp>,  // arch + device name + cuda ver (None for Seeded)
}
```

`HwStamp` mirrors `PytorchBaselineMetadata` (bench lib `:527`) — device name, compute
capability, CUDA version, timestamp — so a stale/foreign measurement is visible, never
silently trusted. Reuse `ArchSku` inside it.

### 5.2 The gate (in the bench crate) + the reducer (in the types crate)

The gate is a bench-side function that, given a cell and a set of `(Implementor, launch
closure)` candidates, times each via the existing `measure_median_ns` and produces the
`ranked` vec. The reducer `winner_of(ranked, seed) -> DispatchEntry` is pure and lives in
the types crate so it is unit-testable off-device.

```rust
// bench side — extends PhaseTwentyNineRow flow
pub fn gate_cell(
    ctx: &Context, stream: &Stream,
    key: &StructureKey,
    candidates: &mut [(Implementor, Option<String>, &mut dyn FnMut())],
) -> Vec<CandidateResult> { /* measure_median_ns each, sort ascending */ }
```

Emit each candidate as a CSV row (extend the schema with `structure_key`, `arch`,
`implementor` columns — additive, keep the existing columns) and, at end-of-run, reduce
the CSV to a `DispatchTable` artifact.

### 5.3 The seed table (hand-knowledge) + the merge seam (Fuel feed, item 08)

```rust
/// Hand-knowledge defaults per §7: cells we route to vendor without benching.
/// Keyed by a *predicate* over StructureKey, not a literal token, because the
/// seed is "large aligned GEMM" (a class), not one shape.
pub fn seed_winner(key: &StructureKey) -> Option<(Implementor, &'static str)> {
    // e.g. Gemm + Contig + V-aligned + GridStride work + f16/bf16 => (Cublas, "large aligned GEMM")
}

/// Merge freshly measured/reported results into an existing table.
/// Measured/Reported override Seeded; newer Measured overrides older on the
/// same arch; a Reported row (Fuel, item 08) is accepted only when its HwStamp
/// arch matches. This is item 08's ingest entry point.
pub fn merge(table: &mut DispatchTable, incoming: &[DispatchEntry]);
```

### 5.4 Artifact emission (mirror `link.rs`)

Add `emit_dispatch_table(&DispatchTable) -> String` in `baracuda-kernelgen` that writes a
committed `@generated` Rust static — same discipline as `emit_link_registry`
(`link.rs:48`): sorted by token, deduped, closes cleanly, human-diffable. The sys/plan
crate `include!`s it and, at dispatch, computes the live `StructureKey` (design §5,
`kernel-specialization.md:160-198`) → `to_token()` → table lookup → route.

### 5.5 FKC / contract implications

- The contract's `cost.provenance` (`contract.rs:136`) gains a third honest value:
  `declared` (static estimate, today), `measured` (this cell won its gate on the recorded
  arch), or `vendor` (this cell routes away — we deliberately do *not* ship a generated
  kernel for it). A `vendor` cell must **not** emit a bindable FKC contract or a
  `BARACUDA_LINK_REGISTRY` entry — that is the vendor-exclusion made honest end-to-end (an
  excluded cell reads as a deliberate miss, per §6's "any cap must be logged").
- **Honest-miss invariant (load-bearing):** a `vendor` decision produces *no* generated
  contract, exactly like `fkc_dtype` returning `None` today (`contract.rs:66,329`). The
  planner's miss signal stays truthful: "no generated cell here, on purpose" is
  distinguishable from "we forgot to build it."

---

## 6. Implementation steps

Ordered; each names the file it edits.

1. **Schema** — add `crates/baracuda-kernels-types/src/dispatch.rs` with `Implementor`,
   `Provenance`, `CandidateResult`, `DispatchEntry`, `DispatchTable`, `HwStamp`. Re-export
   from `crates/baracuda-kernels-types/src/lib.rs`. Pure data + `Eq`/`Hash`; no device.
2. **Reducer + seed + merge** — in the same module: `winner_of()`, `seed_winner()`,
   `merge()`. Keyed by `StructureKey` (in-memory) and by `to_token()` (serialized). Unit-
   testable off-device.
3. **Artifact serializer** — add `emit_dispatch_table()` to
   `crates/baracuda-kernelgen/src/link.rs` (or a new `dispatch_artifact.rs` beside it),
   mirroring `emit_link_registry` (`link.rs:48`): sorted, deduped, `@generated`, valid Rust.
4. **Vendor-exclusion gate in emission** — in `crates/baracuda-kernelgen/src/bin/kernelgen.rs`,
   before emitting a cell's `.cu`/contract/link-entry, consult `seed_winner()`; if the cell
   routes to vendor, **skip generation** and record a `Seeded`/`vendor` `DispatchEntry`
   instead (log it — §6 "any cap must be logged"). Emit the dispatch-table artifact
   alongside the existing outputs.
5. **Contract provenance** — in `crates/baracuda-kernelgen/src/contract.rs`, thread the
   cell's dispatch decision so `cost.provenance` reflects `declared`/`measured`/`vendor`,
   and `contract()` returns `None` for a `vendor` cell (no bindable contract). Preserve the
   existing `fkc_dtype`-`None` honest-miss path.
6. **Bench-gate harness** — in `crates/baracuda-kernels-bench/src/lib.rs`: extend
   `PhaseTwentyNineRow` with additive `structure_key: Option<String>`, `arch: &'static str`,
   `implementor: &'static str` columns (keep old columns; bump the CSV header + a schema
   note); add `gate_cell()` (times each candidate via `measure_median_ns`) and
   `HwStamp::current()` (device name/capability/cuda from `baracuda-driver`).
7. **Wire a seed cross-impl bench to the gate** — start with GEMM (the canonical
   vendor-exclusion case): in `crates/baracuda-kernels-bench/benches/gemm_vs_cublas.rs`,
   for each `(M,K=N,dtype)` shape build the `StructureKey`, run `gate_cell` over
   {cuBLAS, CUTLASS-bespoke} (generated absent), and append gated rows. Then repeat for
   `elementwise.rs` (generated vs bespoke, once kernelgen emits the matching cell).
8. **Reduce CSV → table → artifact** — add a tool (extend
   `tools/build_benchmarks_table.py`, referenced in `BENCHMARKS.md:300`, or a Rust binary)
   that reads the gated CSVs, calls `merge()` over the seed table, and regenerates the
   committed dispatch-table artifact. Document the refresh workflow.
9. **Catalog / OP-MATRIX** — add a "Dispatch / vendor-exclusion" note to `OP-MATRIX.md`
   (the authoritative "what's implemented + backend" doc, `:7-11`) pointing at the table as
   the machine-readable source, and update `docs/design/kernel-specialization.md` §7 status
   from prose-only to "mechanism landed; Fuel feed = item 08."

---

## 7. Test & on-device validation plan

**Off-device unit tests (in `baracuda-kernels-types`):**
- `winner_of` picks the min-median candidate; `margin = second/first`; a single-candidate
  cell has `margin = 1.0` and is flagged (no real contest).
- `seed_winner` returns `Cublas` for a large-aligned f16 GEMM cell and `None` for a small
  `M=1` decode GEMM cell (the regime `BENCHMARKS.md` shows cuBLAS+CUTLASS split — lines
  ~350-355). Assert on constructed `StructureKey`s, not literal tokens.
- `merge` precedence: `Measured` overrides `Seeded`; `Reported`(Fuel) accepted only on arch
  match; newer `Measured` overrides older same-arch. Token round-trip through the artifact
  (`emit_dispatch_table` → parse) equals the in-memory table (mirror
  `link.rs` test `registry_is_sorted_deduped_valid_rust` `:94`).
- Artifact is valid Rust, sorted by token, deduped, closes with `];`.

**On-device (mandatory for the gate itself — it *is* device measurement, RTX 4070 / sm_89):**
- Run the gated `gemm_vs_cublas` bench; assert every emitted row carries a non-empty
  `structure_key`, a matching `arch: sm89`, and ≥1 `CandidateResult`. Assert the produced
  `DispatchEntry.measured_on` HwStamp reports `sm89` + the RTX 4070 device name.
- **Numeric oracle (correctness gate, not just speed):** before a candidate is admitted to
  the ranked list, its output must match the generic strided oracle (design §10,
  `kernel-specialization.md:338`) — a fast candidate that is *wrong* must be rejected, not
  ranked. For generated cells this reuses the differential test that already gates codegen;
  for vendor candidates, diff against the same oracle within the op's declared tolerance.
- Sanity: the gate's measured winner for a cell `BENCHMARKS.md` already characterizes
  (e.g. f32 GEMM `M128_N4096_K4096` → baracuda wins ~2×, line ~349) must agree with the
  table's `winner`. A disagreement means the gate's timing harness diverged from the
  informal roll-up — investigate before trusting the table.

**nvrtc headerless compile / nvcc numeric / compute-sanitizer:**
- This item generates *no new device kernel* — it times existing ones and records a table.
  So there is **no new nvrtc-headerless or nvcc-numeric case to add for the table itself.**
  BUT: any generated *candidate* the gate admits must already have passed the initiative's
  standing house rule (nvrtc headerless compile + nvcc numeric on sm_89; compute-sanitizer
  synccheck/racecheck/initcheck for shared-mem/cross-thread kernels — e.g. RowReduce).
  The gate must **refuse to rank a generated candidate that has not passed** those checks;
  wire that as a precondition assert, so the gate can never promote an unvalidated kernel
  into a shipped dispatch decision.

---

## 8. Adversarial-verify checklist

Run the multi-agent find → dedup → skeptic-refute pass after the change. Probe specifically
for:

1. **Winner-from-noise.** `--quick`/small-`inner` timing variance (BENCHMARKS.md warns of
   20-30% variance at M=1, lines ~144-145) elects a false winner when the margin is inside
   the noise floor. Does the gate record `margin` and refuse to *flip* a seeded/prior
   decision unless the margin clears a threshold? A 1.01× "win" must not overwrite a vendor
   seed.
2. **Stale/foreign measurement trusted.** A `DispatchEntry` measured on sm_80 (or a
   different device) applied to an sm_89 route. Does `merge`/lookup gate on `HwStamp` arch,
   and does lookup refuse a row whose `measured_on` arch ≠ the query arch?
3. **Honest-miss corruption.** A `vendor`-excluded cell still emits a
   `BARACUDA_LINK_REGISTRY` entry or a bindable FKC contract (double-listing) — or,
   inversely, a cell that *should* generate is silently dropped as vendor and reads as a
   forgotten miss. Assert exclusion is logged and mutually exclusive with emission.
4. **Fast-but-wrong candidate ranked.** A candidate that returns garbage in ~0 ns tops the
   ranking because the correctness oracle was skipped. Confirm the oracle gate runs *before*
   ranking, not after.
5. **Token drift between build and runtime.** The table is keyed on `to_token()`; the
   runtime dispatcher must compute the *same* token from live tensors. A canonicalization or
   schema-version (`STRUCTURE_KEY_VERSION`, `structure_key.rs:48`) mismatch silently misses
   every row. Assert `from_token(to_token(k)) == k` on every emitted row and that
   `version` is carried in the token.
6. **Non-determinism leak.** The artifact must be byte-stable across runs (sorted, deduped)
   or it churns the committed diff and the lockstep release. Verify emission is
   order-independent of measurement order.
7. **Empty-candidate cell.** A cell where *no* implementor exists (neither generated nor
   vendor) — the gate must produce an honest "no route" entry, not a panic or a phantom
   winner.

---

## 9. Definition of done

- `DispatchEntry`/`DispatchTable`/`Implementor`/`Provenance`/`HwStamp` land in
  `baracuda-kernels-types`, re-exported, with off-device unit tests green
  (`winner_of`, `seed_winner`, `merge` precedence, token round-trip).
- `emit_dispatch_table` emits a committed `@generated` artifact, sorted/deduped/valid-Rust,
  with a `link.rs`-style test.
- The bench gate (`gate_cell` + extended CSV) is wired to at least `gemm_vs_cublas`, runs
  on the RTX 4070 (sm_89), and every emitted row is `structure_key`- + `arch`- +
  `implementor`-tagged with a correct `HwStamp`.
- Correctness oracle gates ranking: a wrong candidate is rejected, never promoted; a
  generated candidate that has not passed nvrtc/nvcc/sanitizer cannot be ranked.
- Vendor-exclusion is honest end-to-end: an excluded cell emits **no** contract and **no**
  link entry, is logged, and `cost.provenance` reflects `declared`/`measured`/`vendor`.
  Existing `fkc_dtype`-`None` honest-miss path (`contract.rs:66`) untouched and still green.
- Determinism preserved: artifact is byte-stable across runs; no `atomicAdd`/ordering
  change to any device kernel (this item adds none).
- `merge()` ingest seam exists and is documented as item 08's Fuel-feed entry point.
- Docs updated: `kernel-specialization.md` §7 status flipped from prose to "mechanism
  landed"; `OP-MATRIX.md` gains the dispatch/vendor-exclusion pointer; refresh workflow
  documented next to `BENCHMARKS.md`.
- Adversarial-verify pass run; findings triaged or refuted.
- Full workspace builds; lockstep-release shape (`publish_alpha*.ps1`) unaffected (new
  types crate content ships with the bump; the artifact is committed source).

---

## 10. Open questions / Fuel asks

**Local decisions (resolve during implementation):**
- **Where the schema lives** — `baracuda-kernels-types` (runtime-readable) vs
  `baracuda-kernelgen` (build-only). Recommendation: schema in types, emission in kernelgen
  (mirrors `link.rs`). Confirm the plan/sys crate can `include!` the artifact without a
  cycle.
- **Winner shape** — scalar winner vs ranked top-K. Recommendation: top-K from day one so
  item 08 is a data feed, not a schema migration (§8 line ~281 already carries
  `candidates_considered[]`).
- **Margin threshold** — the minimum `second/first` ratio at which a measured win is
  allowed to override a seed/prior decision. Needs a defensible default (e.g. ≥1.10, since
  BENCHMARKS.md treats ±5% as "≈"). Tie to the harness's measured variance, not a guess.
- **Seed predicate granularity** — `seed_winner` keys on a *class* predicate over
  `StructureKey` (e.g. "large aligned GEMM"), not literal tokens. Pin the exact predicate
  for the GEMM seed (which `WorkClass`/`VecWidth`/`Contiguity`/dtype set routes to cuBLAS).

**Fuel asks (cross-repo, gate the v2 feed only — do not block v1):**
- **Confirm the §8 record schema** (`dispatch_record` / `miss_record`,
  `kernel-specialization.md:269-282`) as the wire form item 08 ingests, keyed on our
  `StructureKey::to_token()`. In particular: does Fuel report `candidates_considered[]` with
  per-candidate `time_ns` (needed to populate `ranked`), or only the chosen winner? The
  table's top-K wants the former.
- **Arch/HW stamping on the Fuel side** — a Fuel `dispatch_record` must carry enough of an
  `HwStamp` (arch + device) that `merge()` can arch-gate a `Reported` row. Confirm Fuel can
  supply device capability, not just `ArchSku`.
- **Ownership of the merged artifact** — v1 (batch) has Baracuda maintainers consume Fuel's
  aggregated report and regenerate the committed table (design §8 rollout, `:285-297`).
  Confirm that division: Baracuda owns the artifact; Fuel supplies records; the live/v2
  in-process loop is explicitly deferred (design `:291-297` lists its hazards).
