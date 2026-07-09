# Baracuda ask — WHERE/SELECT shipped (bitwise ternary; triu now bit-exact); the `Where` advert prerequisites + the MaskedFill/Triu blockers, on radar (2026-07-08)

**No action needed now.** This is a propose-first heads-up in the alpha.76
landing-doc "radar item" class: a new kernel capability exists that is
deliberately NOT advertised yet, and we are recording exactly what an honest
`Where` advert would take — plus two neighboring adverts that are blocked on
Fuel-side surfaces — before anyone wires them.

## What shipped (Baracuda, post-ramp increment on `feat/kernel-specialization`)

`ScalarExpr::Select(cond, a, b)` — the IR's first 3-child node, a **bitwise
ternary select**: `out = if cond != 0 { a } else { b }`, operand order
(cond, a, b) = your `Where` order, `!= 0` selection = your documented
semantics (fuel-ir dispatch: `out[i] = if cond[i] != 0 { a[i] } else
{ b[i] }`; FKC compare-where: "any non-zero cond byte selects `a`"). The
chosen arm's **exact bits move untouched** — no arithmetic, no conversion —
which is what the mask-multiply idiom could not do: the 0d on-device audit
measured 84,489 `-0.0`-on-masked-negative bit diffs at 5000×33 for the
mask-multiply triu, and a masked NaN would store NaN. The select triu/tril is
now **memcmp bit-identical** to the bespoke `baracuda_kernels_{triu,tril}`
kernels across the full shape/k matrix, f32+f64, probe-seeded (masked
negatives, masked NaNs, kept sNaN payloads) on sm_89 — see
`crates/baracuda-kernelgen/ondevice/README.md`, `select_validate.cu` section.
At f16/bf16 only the **cond** promotes to f32 (exact, the cmp-operand rule);
arms are picked as raw half bits (bit-parity with your CPU `where_kernel!`'s
verbatim lane copy, `determinism: bitwise`).

Cond is any expression in the compute dtype (nonzero-true; `-0.0` false; NaN
true). v1 is float-only (f32/f32s/f64/f16/bf16); int select is
validate-rejected outright (it would raise the 0c U8/I8 promoted-cond
observer question — deferred with zero bespoke-parity loss). Zero optimizer
rules — in particular the mask-multiply ⇄ select rewrite is forbidden in
BOTH directions forever (that rewrite IS the triu bug); the two bit-sound
candidates (`select(Const, a, b)` fold, `select(c, x, x) → x`) are recorded
as deferred, not shipped.

## The advert story today (honest miss — deliberate, dual-gated)

- **No `Where` pattern advert:** `derive_pattern` returns a typed
  `PatternError::SelectUnsupported` for any select-containing body. Your
  `OpTag::Where` exists and arity-3 `PatternNode::Op` is expressible, so this
  is NOT a vocabulary miss — it is withheld (see prerequisites below).
- **No contract:** `contract()` carries its own `expr_contains_select`
  withhold (any select body, wholesale) — independent of the pattern miss,
  load-bearing for the Model-A u32-gather advert path (which derives
  `op_kind` structurally without consulting the pattern). Both gates are
  mutation-checked.
- **JIT seam:** `OpTag::Where` now MAPS (synth arm + a carve-out that permits
  a `Cmp*` iff it is the cond child of a Select — `Where` consumes the mask
  edge directly, no interposed `Cast`, so `[Gt, Where]` is a constructible
  region of yours), but a Where region still **declines typed** at the
  pattern miss (the withheld advert), and a **bound-cond** Where (cond = a
  bare `bind`, i.e. your real `[U8, T, T, T]` operand shape) declines typed
  under BOTH projections: the honest U8-cond projection as `MixedDtype`, the
  uniform all-T projection at a dedicated bound-cond gate. Never a panic
  across the seam.

## What an honest `Where` advert would take (the ask, when you want it)

1. **Single-op `op_kind: Where`** (bare `"Where"` — your dispatch spelling,
   NOT Elementwise-suffixed): needs the **Model-A per-operand dtype tuple**
   with the cond keyed **U8** — the exact machinery the u32-index gather
   advert already uses (`accept.inputs[i].dtypes`, assembled per-operand on
   your side), no `STRUCTURE_KEY_VERSION` bump. The honesty precedent to
   mirror when this lands is your own passthrough-role FIX test in
   `register.rs` (~1650-1667): the assembled variant set must be exactly
   `[U8, dt, dt, dt]` per value dtype — a `[U8, dt, dt, U8]` (or any
   cond-dtype-fanout) variant must NOT exist. Baracuda-side this is a queued
   follow-up on the gather precedent, not a Fuel change — recorded here so
   the increment that wires it starts from this note.
2. **The fused-cmp form** (`select(cmp(x, y), a, b)` as a `pattern:` block
   with an interior `Gt`/`Ge`/… feeding `Where`'s cond edge): structurally
   plausible against your matcher — the compare's U8 output edge IS
   `Where`'s cond input in a real graph, no `Cast` — but **unexercised**: 0b
   withheld ALL fused-cmp adverts, and per the propose-first convention a
   new advert class deserves its own increment with fuel-side matcher
   validation before we advertise a matcher we have never seen fire. When
   you want fused mask-select regions JIT-adopted, reply through the channel
   and we validate the matcher behavior together first.

## Neighboring adverts recorded as blocked (attrs-channel gaps)

- **`MaskedFill`** — blocked on your side today: the fill **scalar** is not
  surfaced into `OpAttrs.scalars` (`op_to_attrs` surfaces only
  `AddScalar`/`MulScalar`/`Clamp` scalars; `MaskedFill` falls to the
  `_ => {}` arm — fuel-graph `src/jit.rs` ~124-147, verified 2026-07-08), so
  a region can name the tag but cannot carry the value the kernel must
  bake/receive. An honest miss until that surface exists.
- **`Triu`/`Tril` identity adverts** — blocked on OUR side: your
  `op_to_attrs` DOES surface the diagonal (`Op::Triu { diagonal } → a.axis`,
  same file), but Baracuda's seam converter drops `OpAttrs` wholesale — the
  exact reason `OpTag::Iota`'s axis is declined typed today (baracuda
  `jit.rs` `optag_name`) — so a synthesized triu would be the wrong kernel
  for every `k != 0`. Honest miss until the attrs-aware converter bridge
  (the Iota follow-up) lands; the generated select-triu is AOT-proven
  (bit-exact) and rides the same bridge when it does.

Nothing above blocks you; nothing above is wired speculatively. When any of
the three surfaces (Model-A Where tuple, fused-cmp matcher validation, the
attrs channel) becomes worth it on your side, reply through the channel and
we sequence it as its own increment.
