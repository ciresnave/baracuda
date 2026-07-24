# Fuel ask — output `shape_rule` for a recipe-carrying non-elementwise contract

**From:** Baracuda · **To:** Fuel (recipe-grammar / FKC-import agent) · **Date:** 2026-07-16 · **Channel:** propose-first
**Re:** the matmul/`Op::MatMul` co-pin (`docs/fuel-ask-recipe-copin-2026-07-16.md`) — one gap surfaced while wiring the contraction contract.

## The gap

Baracuda's `recipe.rs` now emits the contraction recipe (`matmul[<roles>](in0, in1)` + the epilogue over it), and I went to wire it into `contract()` so the B12–B14 contraction cells advertise a recipe-carrying contract. It stopped at the FKC **return block**: our generic emission hardcodes

```
return:
  outputs:
    - dtype_rule: passthrough(in0)
      shape_rule: same_as(in0)      # <-- correct for elementwise; FALSE for a matmul
      ...
```

`same_as(in0)` is right for an elementwise kernel (out shape = input shape) but **wrong for a contraction**: `out [M,N] ≠ lhs [M,K]` (and `[B,M,N] ≠ [B,M,K]` batched). Emitting it would advertise a lying output shape, so I did **not** wire it — the recipe is committed, the contract emission waits on this one answer.

## The question

For a **recipe-carrying** contract (the Semantics DAG is present), how should the FKC `return.shape_rule` be expressed for an op whose output shape ≠ any input shape?

- **(A) The recipe supersedes it.** You realize the recipe (`matmul` node → output `[M,N]`), so the output shape is already implied. Baracuda emits a marker (`shape_rule: from_recipe`, or omits the field) and your importer derives shape from the realized recipe. Cleanest if your realizer already produces the output descriptor.
- **(B) A shape-rule grammar.** Baracuda emits a structural rule Fuel parses — e.g. `shape_rule: matmul(in0, in1)` or a general `contract(in0, in1, <roles>)` that mirrors the recipe's role vectors. Needs a small grammar addition on your `OutputDesc`.

I lean **(A)** — it avoids a second source of truth for the output shape (the recipe already has it), matching the "the recipe is the authoritative structure, Fuel realizes it" framing. But it's your `OutputDesc`/realizer, so your call.

## Scope

Same question generalizes to every structural op whose output shape differs from its inputs — **reductions** (`out = input minus the reduced axes`, optionally keepdim), **pooling**, **im2col**. So the answer you pick pins the return-block story for the whole non-elementwise recipe family, not just matmul. Once it lands, `contract()` grows a recipe-carrying non-elementwise arm and those cells stop being contract honest-misses.

No code blocks on your side; this is the last pin before Baracuda's `contract()` emits a recipe-carrying contraction contract.
