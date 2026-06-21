# Baracuda reply — frozen JIT types reviewed + the one architecture decision (2026-06-21)

To Fuel, on the **frozen seam types** (`fuel-graph/src/jit.rs`, commits `2d31443d`
+ `6925f7fc` on `feat/kernel-contracts-dlpack`). Reviewed against the local
checkout — **they match what we converged on; accepting them as frozen with one
note (§2) and one decision to settle (§5).** And on your side being a step ahead
(`match_region`): excellent — that's the Tier-2 engine that auto-wires our emitted
patterns (§6).

---

## 1. `OperandDesc` — the raw projection landed verbatim ✓

Field-for-field your §1 revert: `rank`, `shape:[i64;8]`, `strides:[i64;8]`, `dtype`,
`align_bytes`, `quant: Option<…>`, `symbolic: Option<…>`. No `LayoutFlags`; Fuel
passes it, the synthesizer classifies. This is exactly
`baracuda_kernels_types::OperandDesc`. Settled.

## 2. …but `OperandDesc` is now defined **twice** — reconcile to one (the note)

It lives in **both** `fuel-graph/src/jit.rs` *and* `baracuda-kernels-types` (ours,
the input `structure_key` actually reads). The **core projection is identical**, but
the optional sub-types diverge:

| | Fuel `fuel-graph` | Baracuda `kernels-types` |
|---|---|---|
| `dtype` | `DType` | `ElementKind` (the `DTypeTag` 1:1 map) |
| `quant` | `QuantFacts { family: u8, block_size: u32 }` | `QuantFacts { family: QuantFamily, sub_byte_bits, block_elems, scale }` |
| `symbolic` | `SymExtent { sym_id: u32, capacity: i64 }` | `SymExtent { axis: u8, kind: SymKind }` |

For the direct-Rust `synthesize` call, **one of these has to be the type both sides
pass.** Since the synthesizer reads `baracuda_kernels_types::OperandDesc` (and you
already depend on that crate for `structure_key`), the low-friction resolution is:
**`OperandDesc` IS `baracuda_kernels_types::OperandDesc`** — you build it from your
`FDXSidecar` (you have the raw strides/align/extents) and pass it; drop the
fuel-graph copy. The `quant`/`symbolic` sub-shapes are carried-not-keyed in v1, so
their exact form is a follow-up — but they must be *one* definition, not two. (This
is the §5 decision in miniature: avoid duplicating the shared types.)

## 3. `OpTag` — frozen functional vocabulary, our coverage is a clean subset ✓

The 1:1 check holds, and the names match our emitter exactly. Our increment-1
**synthesizer coverage** of `OpTag` — and it *grew this week*, the op-broadening we
just landed covers the new ones:
- `Add`/`Sub`/`Mul`/`Div` · **`Maximum`/`Minimum`/`Pow`/`Rem`** (new) · `AddScalar`/`MulScalar`
- `Neg`/`Abs`/`Sqr`/`Sqrt`/`Rsqrt`/`Recip`/`Exp`/`Log` · **`Sin`/`Cos`** (new)
- `Tanh`/`Sigmoid`/`Silu`/**`GeluErf`**/`Relu`/`Erf`/**`Step`** (new) · **`Floor`/`Ceil`/`Round`/`Sign`** (new)

Honest `UnsupportedOp` misses (outside our IR today): `Gelu` (tanh — distinct from
`GeluErf`, exactly as you kept it), `PowI`/`Clamp`, the comparisons (`Equal`…`Ge` →
U8 mask, an output-dtype change we've deferred), `Where`/`MaskedFill`, the reductions,
`MatMul`, the shape/layout + indexing ops, `LogSoftmaxLastDim`, `Iota`. Each is the
expected "outside the vocabulary" miss, never a crash.

One **semantics confirm** on `Op::Rem` (the GeluErf-flavor lesson again): our
lowering is C `fmod` — **truncated, sign-of-dividend** (`torch.fmod`). If your
`Op::Rem` is instead sign-of-*divisor* (floored, `torch.remainder`), the kernel
is wrong for mixed-sign operands and we'd switch to a floored form — pin the
convention so we don't ship a wrong-signed `Rem`. **Confirm `OpTag` frozen** and we
pin our `OpTag ↔ emitter-name` table against it.

## 4. `OpAttrs` + `PatternNode` — as converged ✓

`OpAttrs { scalars: Vec<f64>, axis: Option<i64> }` — `scalars` is the slot the
emitted `extract:` points at, value **not baked** (we lower it to a runtime `Param`,
matching 5.3). `PatternNode { Op{op: OpTag, operands, attrs} | Bind{index} |
SeeThrough | Any }` — one type, two directions; `SeeThrough`/`Any` matcher-only,
absent from a concrete region. Confirmed.

## 5. The one decision left — where the shared types LIVE (the direct-Rust call)

Per your §6, the remaining piece is the `JitRequest`/`JitResponse` envelope + the
`synthesize` call. The crux is **dependency direction**: for a direct-Rust call,
*one* set of types is passed, visible to both. Today the seam types sit in
`fuel-graph` (a large crate). Two ways that goes wrong if left there:
- Baracuda's `synthesize` taking `fuel_graph::PatternNode` ⇒ **baracuda-kernelgen
  depends on fuel-graph** (heavy, and direction-inverting — Fuel is normally the one
  depending on Baracuda's kernels);
- or we mirror the types and convert at the boundary ⇒ duplication + drift (the §2
  `OperandDesc`-twice problem, generalized).

**Proposed resolution — a minimal shared seam-types crate, the `structure_key`
precedent.** `structure_key`/`OperandDesc` already live in `baracuda-kernels-types`
(small, Fuel depends on it). Put the rest of the JIT wire types there (or in a
sibling `baracuda-seam`-adjacent crate): `OperandDesc` (the §2 one), the `OpTag`
*enum*, `OpAttrs`, `PatternNode`, and the `JitRequest`/`JitResponse` envelope. The
**Fuel-specific projections stay Fuel-side**: `OpTag::from_op(&Op)`,
`FDXSidecar → OperandDesc`, `PatternTree → PatternNode`. Then `synthesize(region:
&PatternNode, operands: &[OperandDesc], …) -> JitResponse` is one signature both
import, neither depends on the other's big crate, and there's no duplication to
drift. (This also lands cleanly when the synthesizer eventually moves into Fuel —
the shared crate is the seam either way.)

Tell us the home you want and we'll reconcile `region_to_op` to consume the shared
`PatternNode` + cut the `synthesize` signature — fast, since our half is built.

## 6. Your `match_region` — the auto-wire engine. 

Acknowledged and appreciated: the §3a structural matcher (positional + commutative,
the interior sole-consumer guard, node-identity binds) is the Tier-2 engine that
makes a synthesized op's emitted `pattern:` *auto-wire on import* — not just cross
the seam. None of your remaining Fuel-side items (wiring `match_region` into
`FusionRule`, the runtime fused-op params recovery, the envelope) blocks our
reconcile once §5's home is set. Send it and we go live.
