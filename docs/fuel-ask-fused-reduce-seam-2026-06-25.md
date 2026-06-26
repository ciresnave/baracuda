# Baracuda ask — making fused RmsNorm/Softmax/LayerNorm adoptable via the §5 seam (2026-06-25)

Baracuda now synthesizes **fused RmsNorm, Softmax, LayerNorm, and weighted-RmsNorm**
(one block per row, warp-shuffle + shared-mem tree reduce, numerically validated on
sm_89). But they're **AOT-only** — they are NOT adoptable through the §5 JIT seam,
because the seam region path (`region_to_op`) and our FKC contract emission only
handle **elementwise** regions; a region containing a reduction honest-misses
(`derive_pattern` → `NotElementwise` → no contract). To make these Fuel-fusable
(region in → kernel + recipe out → cost-gated adoption), we need to agree on how a
**fused-reduce region** is encoded in the frozen grammar and matched. The codegen is
done; this is purely the seam encoding.

## 1. The region encoding (proposed — no grammar change for RmsNorm/LayerNorm)

The frozen `fuel-kernel-seam-types` grammar already expresses the reduce → broadcast
→ elementwise shape via `MeanDim`/`SumDim` + `OpAttrs.axis` + a **shared `Bind`** (a
repeated index = the node-identity guard you defined):

- **RmsNorm** `x · rsqrt(mean(x², −1) + eps)`:
  ```
  Mul( Bind0,
       Rsqrt( AddScalar{eps}( MeanDim{axis:-1}( Sqr( Bind0 ) ) ) ) )
  ```
  `Bind0` appears twice (the shared `x`); `MeanDim` carries `axis: Some(-1)`; `eps`
  rides `AddScalar` attrs. The reduced scalar's broadcast back over the row is
  implicit at the outer `Mul`.
- **LayerNorm** `(x − μ)·rsqrt(var + eps)·w + b`, two reductions:
  ```
  Add( Mul( Mul( Sub(Bind0, MeanDim{-1}(Bind0)),
                 Rsqrt(AddScalar{eps}(MeanDim{-1}(Sqr(Sub(Bind0, MeanDim{-1}(Bind0))))))),
            Bind1 /*weight*/),
       Bind2 /*bias*/ )
  ```
  (the inner `Sub(Bind0, MeanDim{-1}(Bind0))` is the centered x — shared; var is the
  stable mean of its square.)

Both are expressible in the **frozen** vocabulary as-is. **Ask (a): confirm** this is
the intended encoding — `MeanDim`/`SumDim` with `axis = Some(-1)`, shared `Bind`,
`AddScalar` for eps, broadcast-back implicit at the consuming binary op.

## 2. The gap — Softmax's last-axis max

`OpTag` has `SumDim`/`MeanDim` (per-dim) but the only max reductions are `MaxAll`
(all axes) and `ReduceMaxTo` (reduce-to-shape). Numerically-stable Softmax needs a
**last-axis max**:
```
Div( Exp(Sub(Bind0, MAX_LASTDIM(Bind0))),
     SumDim{-1}( Exp(Sub(Bind0, MAX_LASTDIM(Bind0))) ) )
```
**Ask (b): how should `MAX_LASTDIM` be spelled?** Options: (i) `ReduceMaxTo` with the
`[…,1]` target carried via attrs; (ii) add `MaxDim`/`MinDim` to the (`#[non_exhaustive]`)
`OpTag` for symmetry with `MeanDim`/`SumDim`. We lean (ii) — it's the cleanest mirror
and our `Access::RowReduce` already has `ReduceOp::Max`/`Min` — but it's your grammar,
so your call. (Until then, RmsNorm + LayerNorm can go live without it.)

## 3. The broadcast-back + `match_region`

The region grammar is **shapeless** (structural), so a reduction's `[…,1]` result
broadcasting into the surrounding elementwise op is implicit at the consuming node.
**Ask (c):** does `match_region` already match a *reduce → (implicit broadcast) →
elementwise* subgraph — i.e. a `MeanDim` result feeding a broadcasting `Mul`/`Sub`/
`Div` — or does the region need an explicit `BroadcastTo` node between them? If the
latter, we'll emit `BroadcastTo` in our `pattern:` and consume it in `region_to_op`.

## 4. Baracuda's side, once (a)–(c) are pinned

- Extend `region_to_op` (the seam adapter): recognize a fused-reduce region (last-axis
  `MeanDim`/`SumDim`/max nodes feeding an elementwise epilogue) and lower to our
  `Access::RowReduce { stages, epilogue }` — each reduction node → a stage; the rest →
  the epilogue, with each reduction's result becoming a `Reduced(i)` leaf.
- Emit the FKC **contract + `pattern:`** for `RowReduce` ops (today they honest-miss),
  so an adopted fused-reduce op re-wires via your declarative `match_region`.
- Advertise the capability so Fuel may issue fused-reduce JIT requests.
- **Extent pre-condition the seam caller must enforce:** our `structure_key` carries
  broadcast masks but **no numeric extents** (specialize on structure, not extents),
  so the synthesizer cannot verify a per-column weight/bias's feature extent equals
  `x`'s `k` — a too-short weight keys identically to a correct one and would read out
  of bounds. Like `n_out`/`k`, this is a caller pre-condition: the boundary that still
  holds the live `FdxOperandDesc`/`OperandDesc` extents (your `op_to_tag` /
  region-assembly side) must assert `weight.extent[-1] == x.extent[-1]` before the
  request crosses. (Adversarially verified on-device: with the extents present we are
  compute-sanitizer-clean; the mismatch is only reachable if a caller mis-sizes the
  operand.)

## 5. Cost-gating

A fused-reduce kernel is one launch, ~2–3 passes over the row (reduce passes +
epilogue); the primitive path is several reduction + broadcast + elementwise kernels,
each a full pass + launch. We'd emit a `cost` expr over `n` (out elems) and the row
extent `k` reflecting the fused pass count, for your `cost_expr` core to gate adoption
against the primitive estimate. **Ask (d): preferred cost variables** for a row-reduce
op (we have `n`; is a per-row `k` binding available, or should cost stay in `n`?).

## Scope

Last-axis reductions (the transformer norm/softmax cases) first; multi-axis / arbitrary
axis later. Single row-streamed input + per-column weight/bias (LayerNorm) is built;
a second row-streamed input (fused residual-add LayerNorm) is a follow-up.

---

**Summary of asks:** (a) confirm the `MeanDim`+axis+shared-`Bind` region encoding;
(b) pin Softmax's last-axis-max spelling (`ReduceMaxTo` vs a new `MaxDim`); (c) confirm
`match_region` handles reduce→broadcast→elementwise (or specify an explicit
`BroadcastTo`); (d) the cost-expr variables for a row-reduce op. With those, Baracuda
wires `region_to_op` + the FKC contract and the fused norms go live on the seam.
