# Fuel → Baracuda: dispatch/miss record schema + variants — grounded answers

**Status:** FINAL (2026-07-03), approved for send. Replies to the inbound
"dispatch/miss record schema + kernel variants on the wire" ask (your design
`kernel-specialization.md` §7–§8; dispatch-table schema in
`baracuda-kernels-types::dispatch`, alpha.72+ `feat/kernel-specialization`).
Every answer below is grounded in as-built Fuel code with file:line; nothing is
aspirational unless marked. TL;DR: **A confirmed** (per-candidate timings, we add
the arch stamp), **B all-clear** (no ≤1-per-cell assumption, `variant:` parses
today, wire identity is `(structure_key, ImplId)`), **C selectable-now iff
self-contained** (planned workspace rides our queued caps growth), **D opaque
today + field welcomed**, **ownership split confirmed**.

---

## A (schema) — CONFIRMED, with per-candidate timings; here is our exact wire form

The record types are **already built** (behind our `telemetry` cargo feature,
`fuel-dispatch/src/telemetry/record.rs`):

```rust
pub struct DispatchRecord {
    pub schema: u32,                              // versioned envelope
    pub structure_key: Option<StructureKeyToken>, // YOUR to_token() string, opaque to us
    pub chosen: ImplId,
    pub candidates: Vec<Candidate>,               // per-candidate, NOT winner-only
    pub count: u64,                               // aggregation count
}
pub struct Candidate { pub impl_id: ImplId, pub latency_ns: Option<u64> }
pub struct MissRecord {
    pub schema: u32,
    pub wanted: StructureKeyToken,
    pub fallback: ImplId,
    pub count: u64,
}
```

- **Per-candidate `time_ns`: yes** — `candidates[]` carries `latency_ns` per entry, so
  your `ranked` top-K ingest gets what it wants. The honest caveat is **sparseness,
  not shape**: our Judge retains per-`(op, dtype, size_class, backend, kernel_source)`
  timings (retention long verified), but its profiled matrix is currently the
  f32/square ladder and is actively broadening. `latency_ns: Option` encodes this:
  `None` = "considered, unmeasured (static-cost ranked)". Shape `merge()` to accept
  per-candidate lists with missing latencies rather than requiring all-measured rows;
  the lists densify as Judge coverage grows, with no wire change.
- **Hardware stamp: yes, we have it; we will put it in the envelope.** Our device
  probe carries, per device: `hardware_sku` (device name string), `vendor_id`,
  `device_id`, `compute_capability: Option<(u32, u32)>`, `driver_version`
  (`fuel-ir/src/probe.rs:111-132`); our `EquivalenceKey` already splits on driver
  version and silicon, so arch-gating is native to how we bucket measurements. The
  shipped record structs don't embed the stamp yet — we'll add a fingerprint field to
  the envelope (the `schema: u32` version bump covers it) carrying at minimum
  `compute_capability` + `hardware_sku` + `driver_version`. Note `compute_capability`
  is `Option` (None on non-CUDA backends) — records without it are exactly the ones
  your `merge` should drop, so "stampless ⇒ dropped, not guessed" composes cleanly.
  One question back: `driver_version` is what our probe has — if you need the
  *toolkit* version too, that lives on your side of the FFI; tell us if you want us
  to echo a toolkit string you hand us at registration.
- **Status honesty:** the record/ImplId/StructureKeyToken types + schema are shipped;
  the live oracle→JSONL **emission wiring is the remaining step** of our telemetry
  plan (`docs/session-prompts/baracuda-telemetry-plan.md`). Your ask is well-timed —
  pinning the schema now means the wiring lands against the agreed shape. The
  **miss half needs no Judge data at all** (it falls out of FKC planner matching:
  best-admissible-match-is-generic), so it can flow first.

## B (variants) — no ≤1 assumption; `variant:` tag parses today; identity confirmed (structure_key, entry_point)

- **Nothing on our side assumes ≤1 generated implementation per structure-key cell.**
  Two grounded facts: (1) our `KernelBindingTable` natively holds **multiple ranked
  alternatives per `(op, dtypes, backend)` key** — in production today the portable
  matmul and the MKL/AOCL BLAS siblings coexist at the same keys as alternatives;
  dedup at `finalize()` is by kernel fn-pointer identity only, so distinct entry
  points never collide. (2) **No Fuel registry or cache is keyed by the structure
  token at all** — grep-verified, `StructureKeyToken` appears only in the telemetry
  record types as an opaque wire join field (per our earlier K1 = OPAQUE answer; we
  never parse it). Your N variants per cell arrive as N contracts → N binding-table
  alternatives; our ranker/Judge picks among them, which is your §8 premise.
  One completeness flag: our *FusedKernelRegistry*'s per-`(FusedOpId, backend)`
  lookup takes the first impl per backend today — irrelevant to your variants (they
  bind through the Tier-1 binding table via FKC import, not the fused registry), but
  flagged in case you ever target the fused seam.
- **Opaque `variant:` front-matter: accepted.** Our FKC schema deliberately does NOT
  set `deny_unknown_fields` (additive-versioning contract,
  `fuel-dispatch/src/fkc/schema.rs` module doc), so the tag parses harmlessly against
  shipped importers **today**. We'll add it as a retained `Option<String>` so it
  survives lowering and can ride into records as an opaque annotation. Agreed: the
  tag is opaque on both sides; the entry point remains the true identity.
- **Identity caveat: confirmed a non-issue on the wire.** Our records identify
  implementations by `ImplId`, whose basis tuple is FKC kernel identity (backend, op,
  dtypes, `kernel_source`, entry-point-derived revision) — the structure token is a
  *separate* join field. A record is meaningless without its `chosen`/`candidates`
  ImplIds, so wire identity is `(structure_key, ImplId ⊇ entry_point)`, never the
  token alone. Two same-token reduction cells therefore stay distinguishable in every
  record we emit. The keepdim-form convention remains on our queue from item 03 as
  the durable fix — this answer removes the urgency, not the item.

## C (two-kernel + workspace) — not first-class yet; ship the facade now, planned workspace later

Ground truth: our kernel seam is
`KernelRef = fn(inputs, outputs, layouts, params) -> Result<()>`
(`fuel-dispatch/src/kernel.rs:152`) — **no workspace parameter** — and our
`KernelCaps` carries a single flag today (`strided_input`; the five-flag growth is
queued, and a workspace descriptor belongs to that same growth). So:

- **Selectable today iff self-contained:** if the split-K pair can run behind ONE
  entry point on your side — you allocate/free the `n_chunks × cols × sizeof(acc)`
  workspace internally and issue both launches within the one call — it is a normal
  binding to us and selectable immediately. Your `determinism` block ("deterministic;
  association differs from the single-pass kernel") rides through FKC into our
  `PrecisionGuarantee` and is exactly what our precision-filter consumes; nothing is
  silently selectable.
- **Caller-provided workspace: parked until the caps growth.** The better long-term
  shape (our allocator owns the workspace, it participates in memory planning) needs
  the workspace descriptor in FKC caps + our binding seam; we'll mirror your
  `Workspace<'_>` shape when we do the caps growth. Until then a caller-workspace
  contract would be parsed-and-retained but not consumable.
- **Recommendation:** ship the baseline as the always-selectable binding (you do
  regardless), plus the split-K behind a self-managed facade now if that's cheap for
  you; we migrate it to a planned workspace when the seam lands. That gets the win
  flowing without waiting on our caps work.

## D (count-unit) — opaque today, and the field future-proofs the cost seam; add it

Stronger than "we treat `n` as opaque": **Fuel never derives launch parameters from
your contracts at all.** Launches are provider-internal (your launcher computes its
own grid), and our Layer-1 cost model counts elements from *shapes*, not from a
contract `n`. So the count-unit field is documentation-only for us today. Add it
anyway: when our declared-cost trampoline (Task F) starts compiling contract cost
expressions that reference `n`, count-unit becomes load-bearing for us too — the 8×
launch hazard you're hardening against has an exact mirror as an 8× cost
mis-estimate. Pinning the field now means neither side re-learns it.

## Ownership — confirmed as restated

Matches our constitution ("backends advertise capabilities/costs/telemetry; the
DAG-level optimizer decides"): you own the committed table artifact and regenerate it
batch-wise from our aggregated records; we consume the table as a seed/prior
(Layer-1-adjacent), our Judge Layer-2 measurements + runtime route picker remain the
live selector, and the in-process v2 loop stays deferred with the hazards your design
notes.

---

### Summary of Fuel-side actions this reply commits us to (all small, none blocking you)

1. Envelope: add the hardware-fingerprint field to `DispatchRecord`/`MissRecord`
   (schema bump) — compute_capability + hardware_sku + driver_version.
2. FKC schema: add `variant: Option<String>` as a retained opaque field.
3. Telemetry emission wiring (oracle → JSONL) lands against this pinned schema; miss
   records first (no Judge dependency).
4. Workspace descriptor folded into the queued KernelCaps growth; split-K facade
   selectable before that whenever you ship it.
