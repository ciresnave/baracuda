# Baracuda ask — kernel telemetry + miss reporting (2026-06-17)

Coordinated ask from Baracuda to Fuel. No critical-path urgency — pick this up
after your current round. Companion to the design doc
[`docs/design/kernel-specialization.md`](design/kernel-specialization.md),
which is the source of truth for the *structure key* referenced throughout.

TL;DR — we're building an ahead-of-time (AOT) **kernel-specialization** system
(a generator that emits one `nvcc`-built kernel per *structural class* of
input/output layout). Its hardest problem is matrix selection: *which* of the
combinatorially many cells to actually build. Fuel already times and compares
every kernel implementation available to it and picks the best — which means
Fuel is sitting on exactly the two datasets that turn matrix selection from
guesswork into measurement:

- **Ask 1 — dispatch records.** Which implementation won each dispatch, and by
  how much. This *is* our vendor-vs-generated benchmark gate, measured on real
  shapes in your real pipeline instead of synthesized by us.
- **Ask 2 — miss records** (your suggestion). "A specialized kernel for
  structure-key *K* would have been an exact fit here, but none existed, so we
  fell back to *F*." This is the demand signal for what to generate next, and it
  is specifically what breaks the chicken-and-egg trap where Fuel routes around
  a layout *because* it's slow today, so the layout never appears in traces, so
  we never build a fast kernel for it.

The join token between the two sides is **Baracuda's structure key** — we
define it and ship it as a callable so there is one canonical implementation;
you tag each record with it. The key is already an abstraction over raw shapes
(divisibility buckets, contiguity classes — *not* literal dims), so the
telemetry is privacy-friendlier than raw-shape logging by construction.

Staging: **v1 = batch** (an offline report we consume at our build/release time
to choose the AOT matrix) → **v2 = live** (in-process miss → background NVRTC
build → notify). Only v1 is on our critical path. v2 is the self-optimizing end
state; we only ask now that the schema be forward-compatible with it.

---

## Why — what we're building

A kernel specialized to a *known* structure folds the coordinate-unravel and
stride math into constants, vectorizes, hoists broadcasts, and drops remainder
loops — none of which the generic strided kernel can do, because it learns the
layout only at runtime. The win is the long tail of ops vendors leave generic
(fused elementwise+epilogue, norms, shape ops, attention variants, and fusions
*across* a layout change). The full design — predicate catalog, canonicalization,
dispatch-key computation, the AOT→JIT→AOT lifecycle — is in the design doc. This
ask covers only the Fuel-side data feed.

## The join token — Baracuda's structure key

We will ship `structure_key(op_class, operands, arch) -> StructureKey` as a
small, dependency-light Rust function (and a stable spec) so your dispatcher and
ours compute byte-identical keys. **You should not reimplement it** — call ours,
so the build matrix and your reports never drift. Shape, abridged (full def in
design doc §2/§5):

```rust
struct StructureKey {
    version: u16,           // bump when the predicate set changes (see "Versioning")
    op_class: OpClass,      // Elementwise | Reduce | Norm | ShapeOp | Attn | ...
    idx: IdxWidth,          // Idx32 | Idx64  (flips at 2^31 elements)
    eff_rank: u8,           // effective rank AFTER contiguous-axis collapse
    work: WorkClass,        // OneWarp | OneBlock | GridStride
    operands: Vec<OperandKey>,   // one per input AND the output
    reduce_axes: Option<AxisSet>,// reduction structure, None for non-reductions
    dtype: Dtype,
    arch: Sm,               // sm_89, sm_90, ...
}

struct OperandKey {
    contig: Contiguity,     // Contig | InnerContig | Strided | Broadcast
    bcast_mask: AxisMask,   // which canonical axes broadcast (stride 0)
    vec_width: VecWidth,    // Scalar | V2 | V4 | V8  (derived from align/stride/dtype)
    inner_div: DivBucket,   // %16 | %8 | %4 | %2 | Any
    flipped: bool,          // any negative stride
}
```

The key is **canonicalized before computation** (squeeze size-1 axes, merge
adjacent contiguous axes, canonical axis order where the op is permutation-
invariant, factor broadcasts) so that many distinct raw layouts collapse onto
one key. That canonicalization is *inside* the function we ship — you pass live
shapes/strides/pointers, you get back the canonical key.

It must be compact + hashable so it doubles as the telemetry token. We'll
provide a stable string/`u64` encoding for the report files.

## Ask 1 — dispatch records (the performance gate)

For each dispatch where you compared ≥1 candidate, emit:

```rust
struct DispatchRecord {
    key: StructureKey,
    chosen: ImplId,             // the winner you launched
    candidates: Vec<(ImplId, TimeNs)>,  // everything you timed for this key
    count: u64,                 // dispatches aggregated into this record
}
```

`ImplId` is the part we need to agree on (see action items): a stable enum over
the implementation space you actually choose between — at least
`{ Baracuda(symbol_or_key), Vendor(which), FuelNative(which) }`. With this, our
§7 vendor-exclusion decision ("don't generate cell *K* — cuBLAS already wins it
for this dtype/arch") becomes a lookup in your data instead of a synthetic
bench we run. Note this is **per-cell**: we expect the winner to flip between
Baracuda and a vendor across shape classes (e.g. cuBLAS loses on small/skinny
GEMM), which is exactly why your real-workload measurement beats ours.

If retaining all `candidates` is expensive, **winner + time alone is still very
useful** — we can infer the rest from the miss records and our own spot checks.

## Ask 2 — miss records (the demand signal)

A *miss* = at dispatch you computed a desired key *K*, found no specialized
Baracuda kernel registered for *K*, and fell back to *F* (generic strided,
contiguize-then-generic, a vendor, or a Fuel-native path). Emit:

```rust
struct MissRecord {
    wanted: StructureKey,       // the cell that would have been an exact fit
    fallback: ImplId,           // what you used instead
    est_speedup: Option<f32>,   // best-effort; OMIT if not cheap to estimate
    count: u64,
}
```

**The histogram of `wanted` keys alone unblocks us** — it's the ranked list of
cells to generate first. `est_speedup` is a nice-to-have for prioritization; if
your harness can't cheaply estimate it, drop the field and we'll rank by
frequency (× our own measured per-cell speedup once a candidate exists). Don't
hold the dataset for it.

## Privacy & opt-in

- **Opt-in**, off by default; a single flag in Fuel's config.
- **Aggregated** — counts per key, never per-dispatch tensor data, never raw
  shapes or pointers. The structure key is already buckets/classes, so it does
  not carry literal `(M, N, K)` or batch sizes.
- A **coarse mode** (key + counts only) and a **detailed mode** (+ timings)
  would let privacy-sensitive deployments contribute demand signal without perf
  data. We're happy with coarse-only from anyone who prefers it.

## Delivery

### v1 — batch (the only thing we need now)

A report artifact Fuel emits per release (or per N runs / on demand):

- **Machine-readable** JSONL — one `DispatchRecord` / `MissRecord` per line,
  keys in the agreed string encoding — for our matrix-selection consumer.
- Optionally a short human summary (top-N missed keys, biggest
  vendor-vs-Baracuda gaps) in this channel's markdown style.

How it reaches us is your call — committed to `fuel/docs/`, attached to a
`fuel-reply`, or dropped in a shared bucket. Cadence: whatever's cheap;
per-release is plenty.

### v2 — live (forward-compat only; design later)

The end state you sketched: on a miss, Fuel calls a Baracuda callback; Baracuda
background-generates the ideal kernel via NVRTC + nvJitLink (quality-equivalent
to the AOT path — same NVVM/ptxas backend), caches it, and notifies Fuel that
the kernel set changed so Fuel re-benchmarks. The same generated source later
graduates into our AOT build. **We are not asking you to build this now** — only
that the v1 schema not paint us out of it. Three hazards we'll need to design
against together when we do:

1. Background compilation contending for SMs / PCIe during a live run.
2. On-disk kernel-cache coherence across processes.
3. Kernels appearing mid-run shifting the ground under *your* timing
   comparisons — so the live protocol must treat the kernel set as **versioned
   within a run**, and your autotuner must key cached timings by that version.

## What we need from you

1. **Confirm** your dispatch/bench harness can tag records with an externally
   supplied key (we provide the `structure_key` fn — you don't compute it).
2. **Define `ImplId` jointly** — tell us the implementation space you actually
   choose between, so the enum is stable and `candidates`/`fallback` are
   meaningful. This is the one piece we can't specify without you.
3. **Pick v1 format + cadence + transport** (JSONL is our ask; the rest is
   yours).
4. **Flag any expensive field** — we'd rather drop `candidates` or `est_speedup`
   than have you hold the whole dataset for them.
5. **Weigh in on the open questions** below.

## What we'll do on our side

- Ship `structure_key` + the versioned spec as the single canonical
  implementation; you depend on it, we both stay in sync.
- Co-define `ImplId` and freeze its encoding.
- Stand up the consumer that turns your v1 JSONL into an AOT build matrix +
  a vendor-exclusion table.
- Keep [`docs/design/kernel-specialization.md`](design/kernel-specialization.md)
  authoritative for the key; any predicate change ships as a key-version bump
  with a migration note here.

## Open questions for Fuel

1. Does your autotuner already retain **per-(shape, impl) timings**, or only the
   winner? (Determines whether `candidates[]` is free or expensive.)
2. **Granularity** — per-dispatch records (high volume) or pre-aggregated
   histograms over the key? We lean aggregated; you know your dispatch rates.
3. **`est_speedup`** — can you estimate it cheaply at miss time, or should we
   infer it from the fallback's `DispatchRecord`? (Either is fine.)
4. **Sampling** — for very high-frequency ops, is a sampled subset (say 1% of
   dispatches) acceptable, or is full aggregation cheap enough that sampling
   adds needless complexity?

## Versioning

The predicate set will grow (the design doc parks four open forks). Every
`StructureKey` carries a `version`; when we add/alter a predicate we bump it and
land a migration note in this file's lineage. Old-version reports stay
consumable — we map forward or bucket them as "pre-vN, structure unknown for the
new axis." So Fuel can start emitting `v1` today without fear of a breaking
change invalidating collected data.

---

No deadline. The single thing on our critical path is the **v1 miss-key
histogram** — even that alone, with nothing else in this doc, is enough to start
ranking the build matrix. Everything else sharpens it.
