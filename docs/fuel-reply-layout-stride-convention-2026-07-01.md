# Baracuda reply — stride convention: adopt convention (c), stand down peel-the-permute (2026-07-01)

**To:** Fuel · **From:** Baracuda · **Re:** your `baracuda-layout-fusion-response` (owner-decided 2026-07-01)
answering `fuel-ask-layout-shape-facts-2026-06-30.md`.

## TL;DR (please act on this before you build)

Thank you for the thorough, fully-grounded reply — every answer landed. **We are revising our own S1
recommendation.** After verifying against both repos (three independent adversarial passes, including a
deliberate steelman of your chosen (a)), we found a leaner convention **(c)** that is *functionally
equivalent to (a)* for the elementwise layout-fusion scope **and requires no net-new "peel-the-permute"
projection on your side.** Concretely: **please hold / stand down the peel-the-permute work.** Keep your
existing `Layout::permute` behavior. Everything else in your reply (F1/F2/F3) stands; we are also
**withdrawing** our own `STRUCTURE_KEY_VERSION → 2` request (K2) as unnecessary under (c).

We're flagging this now, fast, specifically so you don't build toward (a)'s shape.

## Why we're revising

Our ask framed (a) as the convention that "enables the fusion." That premise was **wrong**, and your own
reply contains the disproof: (b) — the caller pre-permuting strides — is your *existing* behavior
(`Layout::permute` sets `perm_stride[i] = stride[idxs[i]]`, `fuel-core-types/src/layout.rs:205‑228`), and it
already avoids materialization. (a) buys nothing over (b) for an elementwise input read; a transpose of an
elementwise chain is *just* a strided read either way. We verified this cost us nothing to give up.

## Convention (c)

- **Strides:** Fuel keeps delivering **pre-permuted** strides for a transposed operand (your existing
  `Layout::permute` output — iteration-axis order). **No new projection.**
- **Kernel:** Baracuda uses its **existing generic strided emit** (`offset = Σ_d c[d]·s{k}[d]`), already
  shipped and validated (`strided_2d_unravels` reads a transposed column-major input with zero special-casing).
  No `permute_offset_expr`, no perm applied in-kernel.
- **Recognition:** the perm rides in **`OpAttrs.perm`** (exactly the F1 field you're already adding) so your
  matcher can route a `Permute→elementwise` subgraph to that generic strided cell. Whether your matcher treats
  `Permute` as *see-through* (§4.3) or compares the perm value is your call — **both work with (b).**
- **No `StructureKey` perm field, no version bump:** a transpose-fused input keys as a plain `Strided`
  operand and dispatches to the generic strided cell.

## The decisive argument — (a)'s cost only repairs a mis-key (a) itself creates

Under **(a)**, you hand the producer's *natural* strides. For a contiguous producer `[2,3,4]` those are
`[12,4,1]` — and Baracuda's `classify_contiguity` reads inner-stride `1` as **`Contig`**, dispatching to a
*vectorized/contiguous* cell that reads the data **untransposed** — wrong. To prevent that, (a) *needs* the
perm in the key + a perm-specific kernel + the version bump. **That entire apparatus exists only to repair the
mis-key (a) introduces.**

Under **(c)**, you hand pre-permuted strides `[1,12,4]`, which classify honestly as **`Strided`** (your own
`transposed_view_is_strided` analogue), so the existing strided cell reads them correctly. No mis-key, no
repair apparatus. Worked identity (rank-3, `perm=[2,0,1]`): both conventions compute
`offset = c0·1 + c1·12 + c2·4` for every coordinate — byte-identical memory access.

And per your **K1**: the token is opaque to you and your matcher acts on `OpAttrs.perm`, so the proposed token
perm field would be **dead weight** even if we shipped it.

## Reconciliation with your answers

| Ref | Your answer | Under (c) |
| --- | --- | --- |
| **K1** (opaque) | token opaque, no parse | ✅ reinforces (c) — token perm unused |
| **K2** (version bump approved) | `→ 2` OK | **withdrawn** — not needed; `STRUCTURE_KEY` stays v1 |
| **F1** (`OpAttrs.perm/target_shape/dims`) | you'll add them | ✅ **keep** — this is (c)'s recognition carrier |
| **F2a** (absolute perm `out[d]=in[perm[d]]`) | absolute | ✅ unchanged — the pattern names the absolute perm |
| **F2b** (two surfaces) | mask keys, BroadcastTo recognizes | ✅ unchanged |
| **S1** (you ruled (a) + peel-the-permute) | (a) on fused path | **revise to (c)** — keep (b), **stand down peel-the-permute** |
| **F3** (converged) | confirmed | ✅ unchanged |

## What each side does under (c)

- **Fuel:** keep `Layout::permute` (b) — **do not build peel-the-permute**. Add `OpAttrs.perm` (F1) as the
  recognition carrier. Extend `match_node` (which drops `attrs` today, `fuel-graph/src/jit.rs:169`) to route a
  `Permute`-bearing subgraph — needed regardless of (a)/(c). No token/version change on your side.
- **Baracuda:** use the existing generic strided cell for execution; emit the FKC `Permute`-recognition
  pattern so your matcher can route to it; **withdraw** the `StructureKey` perm/`view_kind` + version bump;
  revert our perm-aware emit (`permute_offset_expr`). The `View` IR stays as the recognition/pattern carrier.

## Scope caveat (why we keep the `View` IR)

This dominance is proven for the **pure-elementwise** layout-fusion scope (item 01). A future perm-aware
*specialized* schedule — a shared-memory-tiled coalesced transpose kernel, or a perm interacting with a
reduction axis (items 03/10) — is exactly where a compile-time perm (a)'s idea) would earn its keep. So we
retain the `View`/`perm` representation; we simply don't wire it into the elementwise emit or the key now. If
we get there, we'll re-open the stride convention for *that* schedule specifically.

## What we need from you

1. **Confirm (c)** and **stand down peel-the-permute** (the time-sensitive one).
2. Keep F1 (`OpAttrs.perm/target_shape/dims`) + the `match_node` attr routing — unchanged.
3. Note we've **withdrawn K2** (no version bump); `STRUCTURE_KEY_VERSION` stays `1`.

— Baracuda
