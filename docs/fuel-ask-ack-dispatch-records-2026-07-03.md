# Baracuda → Fuel: ack of the dispatch-records reply + the toolkit answer

> Short ack to `fuel-reply-dispatch-records-variants-2026-07-03.md`. Everything
> is confirmed workable as answered; one answer to your question back, three
> spelling pins so your retained fields match what we emit, and a status note
> on what landed on our side same-day.

## Your question back — toolkit version echo: **not needed**

`driver_version` + `compute_capability` + `hardware_sku` in the envelope is
sufficient. Our `merge` arch-gates on `compute_capability` alone (mapped to our
`ArchSku`); `hardware_sku`/`driver_version` are audit fields, mirroring our own
`HwStamp` (whose `cuda_version` is likewise the *driver-reported* version — we
never gate on toolkit). Records with `compute_capability: None` are dropped at
ingest, exactly as you anticipated. No registration echo required.

## Spelling pins (so your retained `Option` fields match our emission)

1. **`variant:`** — lowercase tag, kernel-front-matter level, e.g.
   `variant: splitk`, `variant: smemrow`. Absent = the base lowering. Opaque on
   both sides; the entry point remains the true identity.
2. **`count_unit:`** — kernel-front-matter level, one of `elements` or
   `vectors_x{w}` (e.g. `vectors_x8` for a packed f16 V8 cell whose `n`
   argument counts 8-element vectors). Documentation-only for you today, per
   your D answer; load-bearing when the declared-cost trampoline compiles `n`.
3. **Workspace (future, your caps growth)** — when you mirror `Workspace<'_>`,
   our split-K contract will declare `workspace_bytes: n_chunks * cols *
   sizeof(acc)` with the chunking rule in the launch note; until then split-K
   stays facade-or-unshipped per your C recommendation.

## Landed on our side against your reply (same day)

- **Ingest converter** (`baracuda_kernels_types::reported_entry`): builds a
  `Provenance::Reported` entry from one aggregated record, shaped to your exact
  semantics — **your `chosen` is honored, never re-ranked** (your Judge weighs
  more than latency); `ranked` carries only the measured candidates (sparse
  lists accepted, `None` = considered-unmeasured); `margin` =
  best-other-measured / chosen-measured and may honestly be < 1 (such a row
  refreshes a same-route entry but cannot flip a different route past our
  noise floor — intended conservatism); token-less or stamp-less records are
  dropped, not guessed. Off-device tested. The batch tool that resolves your
  `ImplId` → `(implementor, entry_point)` and parses the JSONL sits on top of
  this and lands when your emission wiring produces real feeds — miss records
  first, as you suggested.
- The `(structure_key, ImplId)` identity confirmation is recorded at our
  variant emitter and in the dispatch-schema docs; the keepdim-form convention
  stays queued as the durable fix, urgency removed.
