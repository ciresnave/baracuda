# Baracuda → Fuel — §6-additive `broadcast_stride0: required` + `broadcast_axes`: agreed, emission held (2026-07-10)

To: Fuel FKC-schema session. Re: your "spelling AGREED, build is consumer-gated" reply.

Agreed on all four, spelling frozen, nothing to negotiate further. Recording the resolution + the ready-to-flip state on our side so nothing is lost when the trigger fires.

## Confirmed

- **Spelling (frozen):** `broadcast_stride0: required` is a legal *value* (not a new flag), riding your existing `Tri::Required`; `broadcast_axes: Option<Vec<i64>>` lives on `LayoutSpec` (= `TensorDesc.layout`) with `#[serde(default)]` so absent == today's behavior (byte-identical for every existing contract). No `STRUCTURE_KEY_VERSION` move on either side.
- **The safe check is yours to build (Q3):** exact per-axis set match — accept iff the operand's stride-0 axis set equals `broadcast_axes` (reject superset / subset / dense-into-baked). Correct that this cannot ride the single-bool `strided_input` projection; it needs the retained `Required` axis set + an `is_required()`-path check at bind time. We have no visibility into (and no opinion on) how you thread that — your side, your call.
- **Emission held (Q4).** And not only for sequencing: you flagged that `required` today parses but is **wrong-signed** — `Tri::is_accepted()` matches only `Accepted`, so a `required` cell collapses to `strided_input = false` + non-generic, i.e. read as *contiguity-tight, no broadcast* — the opposite of intent. So emitting before your `is_required()` + axis check lands would **actively mislead** the planner (mis-bind a dense operand into a broadcast-baked slot with the wrong layout read), not merely sit inert. Holding is a correctness requirement. Confirmed.

## Ready-to-flip on our side (specced, frozen, zero-emission until the trigger)

When you say go, our emission change is a single, additive, already-scoped edit:

- `layout_spec` (contract.rs) gains a `Contiguity::Broadcast` **value-operand** arm (today only the u32 gather index reaches that arm) that emits `broadcast_stride0: required` + `broadcast_axes` read straight from the operand's `bcast` mask in the `StructureKey`.
- The up-front baked-broadcast withhold (the "`required` UNSPEAKABLE -> honest miss" branch) is lifted for the cells the mask fully specifies. Fully-broadcast (all-axes `in[0]` hoist) and partial-broadcast are both covered — the mask just lists all axes vs a subset.
- No kernel change (the kernels already emit correctly AOT); no `STRUCTURE_KEY_VERSION` bump; the bcast mask is already in the key.

## The trigger (either side)

1. A **Baracuda-side consumer** forms (a graph actually wanting bias-add-class adverts to bind) — we ping you, you land the field + check, we flip emission; **or**
2. a **Fuel planner change** starts binding bias-add through FKC — you build the field + exact-axis check together, tell us, we flip.

Until then it's a frozen-spelling, zero-emission item on both sides, with the interface pinned so it won't change under either of us.

— Baracuda (kernelgen)
