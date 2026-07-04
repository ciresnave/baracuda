# Baracuda reply — JIT envelope reconcile: recipe is derivable (drop confirmed), + 2 conform-asks accepted

**Re:** Fuel's `jit-envelope-reconcile` revision (2026-07-04) — the envelope reshaped
to Baracuda's built `take_kernel` + `SynthArtifact` handover, the two conform-asks
(Q2), and the one field-drop confirm (`recipe`).
**Status:** the `recipe` drop is **confirmed lossless** — verified against the live
`baracuda-kernelgen::jit` source; `recipe` carries **nothing** not reconstructable from
the surviving `contract`. Both conform-asks are **accepted**; they land when we build
against the merged/published `jit-envelope-reconcile` bump (both are currently the way
they are *because* alpha.72's published trait doesn't yet have them — details below).
Nothing here blocks your freeze.

---

## The one confirm you asked for: **drop `recipe` — it is lossless**

You dropped a separate `recipe` field on the reasoning that the re-fuse `pattern:`
rides in the FKC `contract` and the `decompose` is derivable. **Confirmed — and
tighter than that: `recipe` is byte-for-byte reconstructable from `contract` ALONE.**
The region you hold isn't even needed. Here is the source proof.

Both recipe halves are built from **one** serialization of the region's canonical
pattern node (`synthesize`, `jit.rs:376-377`):

```rust
let pattern   = to_fkc(&derived);
let decompose = to_fkc(&derived).replacen("pattern:", "decompose:", 1);
```

So `recipe.decompose` **is** `recipe.pattern` with only the leading keyword swapped
(`pattern:` → `decompose:`) — zero additional information.

And `recipe.pattern` is the **same string** the contract already embeds for a fused op
(`contract.rs:403-404`):

```rust
if let (Some(p), true) = (&pattern, is_fusion) {
    s.push_str(&to_fkc(p));      // <-- the contract's `pattern:` block
}
```

Both `p` (in `contract`) and `derived` (in `recipe`) are the canonical pattern node of
the **same** region, run through the **same** `to_fkc`. They are identical bytes.
Therefore:

| recipe half | reconstruct from `contract` by |
|---|---|
| `recipe.pattern` | take the contract's embedded `pattern:` block **verbatim** |
| `recipe.decompose` | that same block with `replacen("pattern:", "decompose:", 1)` |

**Recommended derivation (zero-drift):** reconstruct `decompose` by the **string swap on
the contract's `pattern:` block**, NOT by re-serializing `JitRequest.region` through
Fuel's own serializer. Both routes are *semantically* the region, but only the string
swap is guaranteed byte-identical to what Baracuda would have emitted — re-serializing
the region risks spelling drift between Fuel's serializer and Baracuda's `to_fkc` (leaf
spelling, field order, whitespace). The contract already carries the canonical bytes;
transform those.

### One nuance that makes the drop *more* correct, not less

The kernel body is **optimized** before codegen, but the recipe/contract pattern is the
**un-optimized original region**. In `synthesize`: the *kernel* is `optimize(&op.body)`
(`jit.rs:363-367`), while `contract(&op, …)` and `to_fkc(&derived)` both use the
**original** `op`/region. So the surviving `contract.pattern` describes the primitive
subgraph to expand back to (2× `Neg` that the kernel cancels, `AddScalar`→`MulScalar`
routing, etc.) — exactly what a `decompose` must carry. Pinned by
`inward_optimizer_simplifies_kernel_but_keeps_the_recipe` (`jit.rs:1438-1449`:
`recipe.pattern.matches("op: Neg").count() == 2` even though the kernel has zero). The
`decompose` you reconstruct from `contract.pattern` inherits this correctness for free.

**Verdict: keep `recipe` dropped.** Nothing in it is unreconstructable from
`(contract.pattern)` — and the region you also hold is a redundant second source, not a
needed one.

## Q2 conform-ask 1 — **make `take_kernel` a trait method: accepted (post-publish)**

Agreed, and it must be on the trait since you call it through `&dyn Synthesizer`. The
reason it's inherent today (`jit.rs:899`, `impl BaracudaSynthesizer`) rather than on the
trait is purely that **alpha.72's published `fuel_kernel_seam::Synthesizer` has no
`take_kernel`** — we couldn't put it on a trait method that didn't exist yet, so we
staged it as an inherent method with the same signature you've now standardized:

```rust
fn take_kernel(&self, entry_point: &str) -> Option<SynthArtifact>
```

When `jit-envelope-reconcile` publishes with `take_kernel` on the trait, we move this
into `impl Synthesizer for BaracudaSynthesizer` unchanged. No signature change on our
side — it already matches your envelope.

## Q2 conform-ask 2 — **return the *envelope* `SynthArtifact`: accepted (post-publish)**

Agreed — Fuel must depend on none of our types (our own Q1 invariant). Today
`SynthArtifact` is Baracuda-defined (`jit.rs:855`) for the same reason: alpha.72 has no
`fuel_kernel_seam::SynthArtifact`. When the envelope ships it, we convert at the trait
boundary — build `fuel_kernel_seam::SynthArtifact` in `take_kernel` from our internal
retained artifact. One mapping note so the field shapes line up:

| your envelope `SynthArtifact` | our source |
|---|---|
| `artifact: Vec<u8>` | `SynthKernel::artifact` |
| `kind: ArtifactKind` (`Ptx`/`Cubin`/`Source`) | our `ArtifactKind` (`Ptx`/`Cubin`/`Stub`) — see note |
| `link: LinkEntry { entry_point, symbol, structure_key, revision_hash }` | our `link_entry(...)` row |
| `contract: String` | our full FKC contract markdown |

**Two small shape deltas to confirm on your side, since you're freezing the envelope:**

1. **`ArtifactKind::Source` vs our `Stub`.** Your envelope lists `Ptx | Cubin | Source`;
   ours is `Ptx | Cubin | Stub`. `Stub` is our *not-loadable* sentinel (the
   `StubCompiler` path when nvrtc isn't compiled in — a loader **must refuse** it). If
   your `Source` means "uncompiled `.cu` text to compile at load," that's a **different
   semantic** than our `Stub` (which is a non-artifact placeholder, never to be loaded).
   We emit only `Ptx` (real nvrtc) or `Stub` (test/wiring) today — never raw source as a
   loadable artifact. Options: (a) your envelope keeps `Ptx | Cubin` and we simply never
   hand you a `Stub` (a stubbed synth returns `Declined`, not a `Synthesized` with a
   junk artifact) — cleanest; or (b) you add a `Stub`/non-loadable variant if you want
   the sentinel visible. We lean (a): **a non-loadable synth should `Decline`, not
   surface an unloadable artifact.** Confirm and we conform.
2. **We also retain `source: String`** (the `.cu`) internally. Your envelope
   `SynthArtifact` has no `source` field — fine, we drop it at the boundary (it's a
   debug/repro convenience, not needed for adopt). Flagging only so you know we're not
   losing anything you wanted.

## Q3 / Q4 — **accepted as you landed them**

- **Q3 concurrency:** sync trait, Fuel drives `synthesize` on a G7 background/idle-time
  thread, adopts via `take_kernel` when it lands, no async wrapper. Exactly our design
  (`BaracudaSynthesizer` is `Send + Sync`, `registry: Mutex<…>`, `&self`). Settled.
- **Q4 budget:** coarse `max_compile_ms` only for v1, no watchdog, `budget =
  { max_compile_ms }`, no regs/smem or op-count axes yet. Agreed — synthesis is off the
  realize path so a coarse budget + your adoption cost-gate is the right granularity. If
  wasted-synthesis ever gets measurable we'll surface the register/shared-memory axis
  first, as you flagged.

## Summary

| Item | Resolution |
|---|---|
| Drop `recipe` | **Confirmed lossless.** `recipe.pattern` == the contract's embedded `pattern:` block (same `to_fkc`, byte-identical); `recipe.decompose` == that block with `pattern:`→`decompose:` swapped. Reconstruct `decompose` by the **string swap on `contract.pattern`**, not by re-serializing the region (zero drift). The region is a redundant source, not a needed one. |
| Conform 1: `take_kernel` on trait | **Accepted, post-publish.** Inherent today only because alpha.72's trait lacks it; signature already matches; moves into the trait impl unchanged when the bump ships. |
| Conform 2: return envelope `SynthArtifact` | **Accepted, post-publish.** Convert at the trait boundary when `fuel_kernel_seam::SynthArtifact` exists. Two shape deltas to confirm: **`ArtifactKind::Source` vs our `Stub`** (we lean: a non-loadable synth `Decline`s, keep `Ptx\|Cubin`), and we drop our internal `source` field at the boundary. |
| Q3 / Q4 | Accepted as landed. |

**Release note:** none of this blocks either side. We're publishing Baracuda alpha.73
now against the **current** (alpha.72) envelope — it does not touch the seam surface. A
later Baracuda release builds against the merged/published `jit-envelope-reconcile` bump
and lands both conforms. Ping us when the bump publishes.

— Baracuda

---

## ADDENDUM — Fuel FROZE the envelope (2026-07-04)

Fuel accepted all of the above and **froze the JIT envelope shape** on
`jit-envelope-reconcile`. Recorded here so the round-trip is closed in one place:

- **`recipe` drop — accepted with our zero-drift derivation.** Fuel reconstructs
  `decompose` by the **string swap on `contract.pattern`** (`pattern:` → `decompose:`),
  NOT by re-serializing the region — pinned in Fuel's `kernel-seam-interop.md §5.2`. The
  un-optimized-region nuance (contract pattern is the pre-codegen subgraph, e.g. the 2×
  `Neg` the kernel cancels) is the exact `decompose` correctness they want, inherited
  for free.
- **`ArtifactKind` — our option (a).** Envelope is `ArtifactKind { Ptx, Cubin }`,
  loadable-only; the speculative `Source` variant is dropped. A non-loadable/stub synth
  returns **`Declined`, never a `Synthesized` carrying an unloadable placeholder** — so a
  Fuel loader never has to refuse one. Our internal `Stub` maps to `Declined` at the
  boundary; our internal `source: String` drops at the boundary (debug-only).
- **Q3/Q4 stay as landed** (sync trait + Fuel-owned G7 threading; coarse
  `max_compile_ms`, no watchdog / extra axes for v1).

### FROZEN surface Baracuda builds against (when Fuel publishes the bump)

```
JitRequest    { region, operands:[OperandDesc], arch:ArchSku, budget:JitBudget{max_compile_ms} }
JitResponse   ::= Synthesized{ entry_point } | Declined{ reason }
SynthArtifact { artifact:Vec<u8>, kind:ArtifactKind(Ptx|Cubin),
                link:LinkEntry{entry_point,symbol,structure_key,revision_hash}, contract:String }
Synthesizer   { fn synthesize(&self,&JitRequest)->JitResponse;
                fn take_kernel(&self,&str)->Option<SynthArtifact> }
```

### Two post-publish conforms (Baracuda side, when the bump lands on crates.io)

1. Move `take_kernel` from the inherent `impl BaracudaSynthesizer` onto the trait
   `impl Synthesizer for BaracudaSynthesizer` (signature already matches).
2. Return `fuel_kernel_seam::SynthArtifact` (build it at the trait boundary): map our
   `SynthKernel::artifact`/`kind`/`link_entry`/`contract`; a `Stub` artifact →
   `Declined`; drop the internal `source`.

**Handshake:** alpha.73 (against the alpha.72 envelope, seam surface untouched) is
unaffected. Fuel merges `jit-envelope-reconcile` + publishes the envelope bump and pings
us; a later Baracuda release builds against it and lands both conforms. No blocker either
direction.
