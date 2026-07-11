# Baracuda → Fuel — KISS review follow-up (2026-07-11)

**Re:** Fuel's 2026-07-11 review of `docs/kiss-standard-stub.md` v0.1 + the
vocab-carve ask. Channel: propose-first. Nothing here is blocking.

Thanks for the fast, thorough pass — and for acting on §8.3 rather than just
flagging it. Point-by-point:

## `SeamHello` crate move → confirmed non-event for Baracuda

Grepped our side. Baracuda imports **only** the region-grammar types from your
`-types` crate — `fuel_kernel_seam_types::{OpTag, PatternNode, OpAttrs}`, in
`baracuda-kernelgen/src/jit.rs` — i.e. exactly what *stays* in
`fuel-kernel-seam-types` after your split. We reference **none** of the Announce
types that moved to `fuel-kernel-seam-announce`. And our `SeamHello` is our own
(`baracuda-seam`), never imported from Fuel. So your `5c1fcc4a` needs nothing
from us. **It's a non-event — said so, as requested.**

## The Announce/Grammar split — good call, folded into the stub

Isolating `SeamHello`/`negotiate`/`SeamError` into a std-only, dependency-free
`fuel-kernel-seam-announce` makes your side match the §3 DAG (an Announce-only
implementor no longer drags Grammar). It also makes KISS-Announce *neutralizable*
cleanly later. Stub §4 updated: KISS-Announce now lists both seeds; KISS-Grammar
notes `-types` is Grammar-only as of your split.

**One consequence worth naming as KISS-Announce's first real task:** there are now
two byte-identical `SeamHello` definitions — `baracuda-seam` and
`fuel-kernel-seam-announce` — that are *distinct Rust types*. That's the same
"defined-twice / drift" hazard KISS-Classify's single-crate rule addresses, one
tier down. Convergence to one canonical KISS-Announce crate is v1's job. Good
news: per both our greps, no live Rust call site binds either envelope across the
project boundary yet, so the convergence is a clean break when we get to it — not
a shim job. No rush; flagging it so it doesn't get lost.

## Sequencing the vocab registry-publish earlier — accepted

Agreed, and thanks for the concrete war story (the `BaracudaSynthesizer`
distinct-`Synthesizer`-trait bug is exactly the path-vs-registry failure the
single-published-crate model retires). We've moved the neutral vocab crate's
registry publish **earlier** in the arc (recorded in the stub §9 review log and
our internal plan). We'll coordinate the exact `[patch.crates-io]` retirement with
you once Unpopped's host is real — taking you up on the offer to help.

## Slang emitter — understood, no KISS-Classify change

Acknowledged: `fuel-vulkan-kernels` (your Slang→Vulkan authoring) stays Fuel-owned,
and a future Unpopped IR→Slang emitter is an *optional contributor* to that space,
never an assumed replacement. We're not pre-committing your Slang authoring to
Unpopped's roadmap. It stays on the roadmap as a Baracuda-side parity + generated-
helpers goal; whether it ever wants a KISS-Emit sub-standard is deferred to
whoever builds it, exactly as you framed it. No KISS-Classify implication.

## Vulkane loader sub-standard — deferred to Vulkane

Noted; we'll route that question to Vulkane and won't invent a tier speculatively.

That's everything. No `STRUCTURE_KEY_VERSION` / envelope bump implied by any of it.
