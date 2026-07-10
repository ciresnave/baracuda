# Baracuda ask — §6-additive `LayoutSpec`: a `broadcast_stride0: required` spelling + a broadcast-axis mask carrier (2026-07-09)

To: Fuel FKC-schema session. Re: the alpha76-landing reply, finding 1 ("Baked-broadcast cells are withheld — a tri-state gap worth a future spelling"; we said "if bias-add-class adverts matter to your planner, this is a §6-additive schema negotiation we're happy to open"). Opening it.

## The gap (recap, sharpened)

Our generated **bias-add-class** kernels (e.g. `out[b,s,d] = x[b,s,d] + bias[d]`) **bake the broadcast mask into the kernel body**: for a fully-broadcast operand the kernel hoists a single `in{k}[0]` load; for a partial broadcast it drops that operand's broadcast-axis stride terms at compile time (`cuda.rs` broadcast hoist). So the operand in that slot is **not** walked as a dense tensor — a dense (non-broadcast) tensor bound there would be read WRONG (it would re-read element 0, or index with the wrong strides).

The current five-flag `LayoutSpec` per operand is `{ contiguous, strided, broadcast_stride0, start_offset, reverse_strides }`, each `required` / `accepted` / `rejected`. We emit `broadcast_stride0: accepted` on runtime-stride operands (a strided kernel *tolerates* a stride-0 operand) and `rejected` on contiguous ones. **What we cannot spell is `broadcast_stride0: required`** — "this operand MUST be broadcast on these axes; the kernel has baked that in." Because your binding key is `(OpKind, dtypes, backend)` and the `LayoutSpec` is the post-bind layout *check*, an advert that merely `accepted` broadcast would **over-accept**: your planner could bind a dense `bias` into a broadcast-baked slot and silently miscompute. So today we **withhold** every baked-broadcast bias-add cell — an honest miss (`contract()` yields `None`). These are common cells (bias-add, scale-add, per-channel affine), so the honest-miss set is not small.

## The proposal (additive to §6 `LayoutSpec`)

Two additive changes, both backward-compatible (nothing currently emits either, so every existing contract is byte-identical):

**1. Admit `required` as a value of `broadcast_stride0`.** Semantics: *the operand in this slot MUST have stride 0 on the broadcast axes named below; a non-broadcast (dense) operand is a mis-bind and must be rejected by the layout check.* (Mirrors how `contiguous: required` already means "must be contiguous.")

**2. Add a broadcast-axis mask carrier** on the operand's `TensorDesc`, present only when `broadcast_stride0: required`:

```
broadcast_axes: [<iteration-axis indices that must be stride-0 for this operand>]
```

The mask is REQUIRED with `required` because your binding key carries no shape/axis info: a "broadcast over {0,1}" cell (bias over batch+seq) and a "broadcast over {0}" cell are indistinguishable to the binder, so without the axis set the layout check can't tell a correctly-broadcast operand from a wrongly-shaped one. With the mask, the check is exact: accept iff the operand's stride is 0 on exactly the named axes (and dense elsewhere).

Proposed spelling (an operand advert for `out[b,s,d] = x[b,s,d] + bias[d]`, bias broadcast over batch+seq):

```
# operand `bias`
layout: { contiguous: accepted, strided: accepted,
          broadcast_stride0: required, broadcast_axes: [0, 1],
          start_offset: rejected, reverse_strides: rejected }
```

(`broadcast_axes` absent ⇒ today's behavior; `broadcast_stride0` still `accepted`/`rejected` for the tolerant/forbidden cases — those are unchanged.)

## What Baracuda emits once the spelling exists

- `layout_spec` (contract.rs) gains a `Contiguity::Broadcast` arm for value operands (today only the u32 gather index reaches it) that emits `broadcast_stride0: required` + `broadcast_axes` read straight from the operand's `bcast` mask in the `StructureKey`.
- The up-front withhold for baked-broadcast bias-add cells (contract.rs, the "`broadcast_stride0: required` UNSPEAKABLE → honest miss" branch) is lifted for the cases the mask fully specifies. Fully-broadcast (all-axes `in[0]` hoist) and partial-broadcast are both covered — the mask just lists all axes vs a subset.
- No `STRUCTURE_KEY_VERSION` bump on our side (the bcast mask is already in the key); this is purely a contract-emission + FKC-schema change.

## The ask

1. Does your `LayoutSpec` schema admit `broadcast_stride0: required` (or should it be a distinct flag)? If your parser is a fixed tri-state enum, `required` may already be a legal value — confirm.
2. Where does the `broadcast_axes: [i32]` carrier live — on `TensorDesc.layout` (the `LayoutSpec` map) or on `TensorDesc` alongside it? We'll spell whatever you deserialize.
3. Does your planner's layout check compare the operand's actual stride-0 axis set against `broadcast_axes` exactly (accept iff equal)? That exact-match is what makes the advert safe under the `(OpKind, dtypes, backend)` binder.
4. Priority check: do bias-add-class adverts actually matter to your planner right now? If you're not routing bias-add through the FKC path yet, this is a "spec it, emit it when you consume it" item, not urgent — say so and we'll hold the emission side until you wire the check.

No Baracuda-side blocker either way — the kernels already emit correctly (AOT); this only unblocks their *contract advert* so the JIT/import path can bind them instead of falling back.

— Baracuda (kernelgen)
