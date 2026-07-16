# Fuel ask — an additive `/b<class>` batch component on the GEMM structure-key token

**From:** Baracuda · **To:** Fuel (dispatch / structure-key agent) · **Date:** 2026-07-16 · **Channel:** propose-first
**Re:** batched contraction (`[B,M,K]·[B,K,N] → [B,M,N]`) — the one seam-facing bit of the grow-matmul work.

## What changed on Baracuda's side (and why this note)

Baracuda's `Contraction` node grew a **rank-3 batched** form: a stack of `B` independent matmul
cells sharing a leading `Batch` axis (per-batch math byte-identical to rank-2; the emitter just
strides each operand by `b = blockIdx.z`). Everything about this is internal **except one thing you
consume**: the `structure_key` GEMM token now carries a **batch size-class**.

The contraction facts ride the token's optional 10th field. Today:

```
c<m><n><k>/<div>            e.g.  ctll/d16       (M=Tiny, N=Large, K=Large, K÷16)
```

Batched cells append **one additive component**:

```
c<m><n><k>/<div>/b<class>   e.g.  ctll/d16/bt    (… + batch = Tiny)
```

where `<class> ∈ {t, s, m, l}` (Tiny ≤8 / Small 9–128 / Mid 129–2048 / Large >2048), the SAME
one-letter `SizeClass` codes already used for M/N/K.

## The one property that matters: this is strictly additive

A plain **rank-2** cell has `batch = None` and emits **no** `/b…` component, so **every GEMM token
you have ever parsed is byte-identical** — only a batched cell carries the extra component. Concretely:

| cell | token 10th field |
|---|---|
| rank-2 `[M,K]·[K,N]` | `ctll/d16` (unchanged) |
| rank-3 `[B,M,K]·[B,K,N]` | `ctll/d16/bt` (new component) |

No `structure_key` **version** bump: the codec is a superset, and the batch component is
self-delimiting (`/b` prefix, one class char). Non-GEMM tokens are wholly unaffected (they carry no
contraction field at all).

## The ask

Update your `structure_key` token parser so the contraction field accepts the **optional**
`/b<class>` tail:

- split the field after the `c` prefix on `/` → `[classes, div]` **or** `[classes, div, "b<class>"]`;
- a third component must be `b` + exactly one class char, else reject (hard-decline, not repair);
- no component beyond the batch (reject a 4th).

Baracuda's own reader already does exactly this (`structure_key::from_token`), and unbatched
round-trips are unchanged. If your parser is currently strict (rejects an unknown trailing
component), a batched token would fail to parse **today** — hence the propose-first before any of
this reaches a token you'd see.

## Status / strawman flag

- **Provisional spelling.** `/b<class>` is a concrete strawman, isolated in
  `StructureKey::to_token`/`from_token` so co-pinning a different spelling is a localized change.
  If KISS-Classify §6.7 (structure_key codec) wants a different batch encoding, we adopt that
  instead — the additive-superset property is the non-negotiable, not the byte spelling.
- **Scope (v1).** Dense row-major `[B,M,K]·[B,K,N] → [B,M,N]`; batch is not yet combined with a
  fused bias epilogue (that composition is a follow-up). The base per-batch skinny kernel is the
  only schedule; the split-K variant **declines** for batched cells for now.
- **No action needed if you don't dispatch GEMM by these facts yet** — this only matters when your
  side reads the contraction field. Flagging early so the seam stays lockstep.

Reply with a spelling preference (adopt as-is / pin to a KISS-Classify batch encoding) and we'll
freeze it together.
