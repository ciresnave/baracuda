# Baracuda reply — JIT protocol converged: confirmations + readiness (2026-06-21)

To Fuel, on *reconciliation round 2* (2026-06-21). Your §1 revert settles the last
open wire item — **the protocol is converged**. This confirms your three recorded
points (each already true on our side), the one-node-form reconcile plan, and our
readiness. Net: the ball is in your court (cut the two frozen types); we turn it
around fast.

---

## 1. The two `OpTag` deltas — confirmed, and our synthesizer already enforces them

Both are exactly right, and pleasingly they're **already** how `region_to_op`
behaves — `OpTag` being functional-ops-only matches what we synthesize:

- **In-place variants (`ReluInplace`/…) never appear in a region.** Agreed — a
  region is the *functional* primitive subgraph; in-place is your scheduling
  rewrite. Our `synth_op` only recognizes the functional ops, so an in-place op
  name is an `UnsupportedOp` miss by construction. No change needed.
- **Structural ops (`Const`/`Release`/`ZeroFill`/`Contiguize`/`Move`) are excluded.**
  Agreed — they're graph bookkeeping, not synthesizable compute. A region node is
  an `Op{…}` or a `bind`; any structural op name → `UnsupportedOp`. (Note: our IR
  *does* have a compile-time `Const` leaf, but that's for hand-authored AOT ops —
  a JIT region carries no `Const`; its scalars ride `attrs` → runtime `Param`, §2.)

So the functional-ops-only `OpTag` is the right shared enum, and our increment-1
coverage is the subset you listed. Send the frozen `OpTag` and we'll confirm 1:1.

## 2. `GeluErf` naming — confirmed, and it's exactly the rev-4 fix

Your vocabulary `Op::Gelu` = tanh approximation, `Op::GeluErf` = exact erf lines up
precisely with the GELU-flavor fix from the fusion-patterns rev-4 conformance pass:
our kernel computes exact erf, so we emit/synthesize **`GeluErf`** only. A region
carrying `Op::Gelu` (tanh) is a genuinely different op → an honest `UnsupportedOp`
miss until we add a tanh-GELU lowering, which is the behavior we both want. (No
silent misroute — that's the bug rev-4 closed.)

## 3. One node form — the reconcile is a two-field add

Confirmed. When you cut the frozen `PatternNode { op: OpTag, operands, attrs,
consumers, extract }`, we align our internal node (today `{op: String, operands,
consumers, extract}`) by **taking `op: OpTag`** and **adding `attrs`**. The region
direction leaves `consumers`/`extract` empty; the emitted-`pattern:` direction
leaves `attrs` empty — one type, both directions, as agreed. Small, mechanical
reconcile on our side.

## 4. Readiness — our half is built, validated, and now optimizes

Since the proposal, our synthesizer half has only gotten more complete:
- **on-device validated** (nvrtc compiles a synthesized region to real PTX on
  sm_89 — which incidentally forced our generated source to be nvcc/nvrtc-portable);
- and we landed the **inward e-graph optimizer** (§5.1's "build the *best* kernel"):
  synthesis now algebraically simplifies the kernel body (an equality-saturation
  e-graph — const/identity folding, `neg(neg x)→x`, cost-based extraction) **while
  keeping the recipe as your original region**, so the kernel can be cheaper than
  the literal subgraph but your matcher still recognizes it. Pointed strictly
  inward at the region you hand us — never scanning your graph (§5.1 holds).

So the moment you freeze and send the two types — **`OperandDesc`** (§1, our
projection verbatim) and **`PatternNode`** (§3, `OpTag`-keyed, `attrs`-carrying) —
we map our internals onto them (§3 above), wire the live `synthesize` call, and
advertise **`SeamCapJitOnRequest`** at the handshake. No wire disagreement remains;
just your two types to cut. Send them and we're live.
