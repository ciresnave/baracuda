# MatMul / contraction node — ORDER-3 design spike

> **Design spike, not an implementation.** Deliverable: the grammar, IR variant,
> schedule axes, `StructureKey` facts, FKC contract shape, and per-cell
> vendor-exclusion gate for a generated contraction node — plus the precise,
> now-checkable requirements on the two keystones it sits on. No
> `Access::Contraction` is merged to `main`; the honest-miss on the seam stays.
>
> Grounded in the **shipped** state (2026-07-02): items 01 (layout recognition),
> 02 (DAG-with-consumer-counts), 03 (strided reductions), and 07 (dispatch table)
> are merged into `feat/kernel-specialization`. This spike references the code
> that actually exists, not the pre-merge sketch in
> `docs/planning/foundational/10-matmul-contraction-design-spike.md`.

---

## 1. Scope — the long tail, explicitly not large GEMM

Per `kernel-specialization.md` §1, the generated contraction targets the cells the
tuned head (`baracuda-cutlass` / `baracuda-cublas`) leaves generic:

- **skinny / decode-time GEMV-adjacent** — FlashDecoding++ flat-GEMM (`M` or `N`
  ∈ 1..8), the archetypal `miss_record` demand signal;
- **irregular-K / unaligned** cells the vendor path pads or falls off its fast tile for;
- **the biggest win — contraction fused across a layout change or into a
  bias/activation/norm epilogue in one pass**, skipping the contiguize/round-trip
  the vendor path cannot fuse.

Large, aligned, uniform-batch `{f16,bf16,f32,f64}` GEMM **routes out** to the head
and is never generated. That boundary is the §7 gate (§8 below), and it is exactly
the decision function item 07's `seed_winner` already models — this node extends it
from a single seed to the contraction class ladder.

`v1` is **single-K-group batched matmul**: `out[b,m,n] = epilogue(Σ_k lhs[b,m,k]·rhs[b,k,n], inputs…)`.
General multi-K einsum is a deliberate follow-up; the axis vocabulary is designed
general (§3.1) so it does not need reshaping later.

## 2. What this builds on (shipped, not assumed)

| Keystone | Shipped form | What the contraction reuses |
|---|---|---|
| **02 — DAG** | `ExprDag`/`DagNode`/`from_expr` + per-node `consumers` (edge count); `lower_dag` hoists shared interiors to `tmp` | A contraction result is a **shared producer** feeding an epilogue and re-reads; `consumers > 1` is how "matmul result, consumed twice" is finally representable |
| **03 — reductions** | `Access::Reduction { op, axes: AxisMask, keepdim }`; `ReduceAxisClass{InnerContig,Outer,Middle,Multi}`; `reduce_axes` derived into `StructureKey` | K is a reduction axis; the **same axis vocabulary** names it. `reduce_axes` carries K (its first real non-empty use) |
| **07 — dispatch** | `Implementor{Generated,Cublas,Cutlass,Cudnn,Bespoke}`, `seed_winner(key)->(Implementor,reason)`, `merge`, arch-gated `DispatchTable` | The vendor gate **is** this table; the contraction seeds are new `seed_winner` arms |
| **01 — layout** | `View{Identity,Permute,Broadcast,Reshape}` recognition carrier on `OpDef` (emit reverted; generic strided read is layout-free) | Per-operand layout facts (transpose / leading-dim / row-col-major) come from the `View` + operand strides |
| **AxisRole** (paper) | `{Kept,Reduced,Batch,FreeM,FreeN,ContractedK}`; `reduce_axes` = the `{Reduced}` projection | The contraction's `ContractionAxes` is the `{Batch,FreeM,FreeN,ContractedK}` projection — one vocabulary, wired here |

The key insight the merge makes concrete: **a contraction is a K-reduction (03)
fused with two free axes, producing a DAG value (02) that an epilogue tree consumes
— and whose generate-vs-route decision is a dispatch-table cell (07).** All four
merged pieces are load-bearing; none is hypothetical.

## 3. Grammar & IR — `Access::Contraction`

### 3.1 Axis roles (the AxisRole projection)

A contraction names, per operand, each axis's **role** in the unified AxisRole
vocabulary:

```
Batch(b)        — shared, iterated, not summed        (lhs & rhs & out)
FreeM(m)        — free on lhs → row of out            (lhs & out)
FreeN(n)        — free on rhs → col of out            (rhs & out)
ContractedK(k)  — shared, summed, absent from out     (lhs & rhs)
```

`out` frees over `{Batch, FreeM, FreeN}`; `ContractedK` is summed. This is a strict
superset of 03's `{Reduced}` (a plain reduction is `ContractedK` with no `FreeN`),
so `reduce_axes` = the `{ContractedK}` projection, exactly as it is the `{Reduced}`
projection for 03 — **no re-key, one derivation**. Designing the vocabulary as the
full AxisRole set now (not matmul-only positions) is what lets multi-K einsum land
later without reshaping 01/03's axis representation.

### 3.2 The IR variant

```rust
// crates/baracuda-kernelgen/src/ir.rs — a new #[non_exhaustive] Access arm.
/// Batched contraction: out[b,m,n] = epilogue( Σ_k lhs[b,m,k]·rhs[b,k,n], inputs… ).
/// One contracted axis group (K); free axes M (input 0) and N (input 1); shared
/// batch axes. The K-reduction is fused with the two free axes — NOT expressible
/// as Reduction (one free axis) or RowReduce (row-broadcast, no second free axis).
Contraction {
    /// Per-operand axis roles in the AxisRole vocabulary (§3.1). References item
    /// 01's axis facts; does not invent a parallel axis system.
    axes: ContractionAxes,
    /// K-accumulation combine + accumulator-dtype policy (§5 acc-precision).
    accum: AccumSpec,
    /// Per-output-cell epilogue over the K-reduced scalar and the pointwise
    /// inputs (bias = a column-broadcast Input, etc.). Reuses `ScalarExpr`.
    epilogue: ScalarExpr,
}
```

`OpDef::body` continues to hold the `epilogue` (mirrors `row_reduce`'s
`body = epilogue`), so every existing body-walker (`params_used`, `count_flops`,
`ulp_bound`, and — post-02 — `ExprDag::from_expr`) operates unchanged. Add
`OpDef::contraction(...)` beside `elementwise`/`reduction`/`row_reduce`.

### 3.3 The contraction→epilogue bridge (this is where 02 is load-bearing)

The contraction core produces a scalar-per-output-cell. The epilogue reads it as a
**`Reduced`-like leaf** — reusing the exact precedent `RowReduce` already sets:
`ScalarExpr::Reduced(0)` is the K-reduced accumulator, and bias/relu/gelu epilogues
cost **zero new emitter vocabulary** (they are `Add`/`Unary` over `Reduced(0)` and
column-broadcast `Input`s, exactly as CUTLASS's `Bias*` epilogues).

Crucially, when the epilogue re-reads the accumulator (e.g. `silu(acc) = acc·σ(acc)`
→ `Mul(Reduced(0), Sigmoid(Reduced(0)))`), **`ExprDag::from_expr` already hash-conses
the two `Reduced(0)` leaves to one node** and the emitter references it without
recompute — the item-02 machinery applies to the contraction epilogue for free. The
one bridge item 02 still owes (§7) is letting a `Contraction` sit as a DAG-node
*producer* whose result is that `Reduced`-like leaf, so a *downstream fused op*
(not just the in-op epilogue) can re-read it with an honest `consumers` count.

## 4. `ContractionKey` — the StructureKey facts

Extend `StructureKey` (behind `STRUCTURE_KEY_VERSION` v1→v2) with a single optional
field so every non-GEMM key serializes byte-identically:

```rust
pub contraction: Option<ContractionKey>,   // None for every non-contraction cell
```

`ContractionKey` carries **structure classes, never literal extents** (honoring the
§1 non-negotiable):

| Fact | Class | Drives |
|---|---|---|
| M / N / K size | `{Tiny, Skinny, Small, Large}` per axis | **the vendor gate** (Skinny/Tiny → generate; Large∧Large∧Large → route out) |
| K alignment | `DivBucket` (reuse `{Div16,Div8,Div4,Div2,Any}`) | MMA-k fragment (8/16/32) legality; K-remainder tail |
| M/N alignment | `DivBucket` per free axis | tile predication / store vectorization |
| Layout per operand | `{RowMajor, ColMajor}` (from 01's `View` + strides) | Tensor-Core fragment (`Rrr`/`Rcr` analog); tile-load transpose |
| Batch | `{None, Uniform, Grouped}` | Grouped (MoE variable-M) → routes out to `GroupedGemmPlan` |
| Accumulator precision | `{TF32, F32, F32Strict, F64, S32}` | fragment choice + the determinism/ulp contract |

`reduce_axes` (an `AxisMask`, currently the `{Reduced}` projection for 03) carries
the K set — the contraction is its first non-reduction user; the derivation is the
`{ContractedK}` projection of the same AxisRole logic. The token codec gains **one
optional trailing field** (`|<contraction>` or absent), so
`from_token(to_token(k)) == k` holds and a non-GEMM token differs from v1 only by
the version prefix. Golden guard: a non-GEMM key's body-token is byte-identical
pre/post the version bump.

## 5. `Schedule::Contraction` — the schedule axes

`build_plan` classifies the `ContractionKey` into a `Schedule::Contraction { .. }`
(the axis data rides on `KernelPlan::access`, as RowReduce's stages do). The five
schedule axes (the "schedule half" of the algorithm/schedule split):

1. **K-tiling** — tile along K; multi-pass when K exceeds the shared-mem tile (same
   per-arch shared-mem ceiling that sets reduction predicate #10 — the `ArchSku`
   table).
2. **Register + shared blocking** — the `(Bm × Bn)` threadblock tile, the per-thread
   register tile, and the double-buffered shared staging of lhs/rhs K-slabs.
3. **Fragment selection** — MMA shape (`m16n8k8`/`m16n8k16` f16/bf16; `m16n8k4`
   DGEMM; `m16n8k32` s8 — the shapes the head's bespoke int path already uses) vs
   **SIMT/FMA fallback** for `F32Strict` bit-stability. Arch-gated; coordinates with
   item 09 (half2/packed emit).
4. **Double buffering / software pipelining** — `cp.async` prefetch depth (arch-gated
   to sm_80+).
5. **Epilogue schedule** — fuse bias/activation in-register before the store (no
   extra global traffic), the CUTLASS `LinearCombinationBiasElementwise` model. The
   epilogue lowers through the **item-02 `lower_dag`** path, so a shared accumulator
   in the epilogue is a single `tmp`.

**Determinism (house discipline):** Tensor-Core warp reductions are not
bit-reproducible; the SIMT/`F32Strict` schedule is the deterministic fallback. The
schedule records which it chose so the FKC `determinism`/`precision` block is honest
(§6) — mirroring CUTLASS's `F32Strict` guarantee.

## 6. FKC / contract shape

- `contract()` today hardcodes `cost.class: elementwise` + `provenance: declared`
  and `count_flops` counts pointwise nodes. A contraction needs **`class:
  contraction`** with a `flops_per_output = 2·K` (MAC) term and a **tiled** bytes
  model (not `(n_inputs+1)·dtype_size`). With item 07 landed, `cost.provenance` can
  graduate `declared → measured` (this cell won its gate) or `→ vendor` (routes out,
  emits **no** contract and **no** link entry — the honest miss).
- **Precision/determinism per fragment**: a TF32/Tensor-Core cell is `approximate` +
  non-bitwise; the SIMT/`F32Strict` cell is `correctly_rounded` + `bitwise`. The
  contract must *match what the kernel actually does* — an f16 cell that accumulates
  in f32 says so; one that claims `bitwise` while using warp reductions is a lie (a
  house-discipline red line, same class as the NaN-misroute / non-finite-margin bugs
  the reduction and dispatch adversarial passes caught).
- `derive_pattern` grows a `Contraction` shape (a `MatMul`/`Contract` graph-Op with
  the epilogue subtree) **or** the node advertises as a primitive `op_kind` (no
  `pattern:`) — the honest-miss `None` path stays default until the region grammar is
  agreed with Fuel (§9).

## 7. Requirements on the keystones — now checkable against merged code

**Item 01 (layout) must still expose** (recognition carrier is landed; these are the
gaps for a *contraction*):
- **axis roles**, not just extents — the `ContractionAxes` needs per-operand
  `{Batch,FreeM,FreeN,ContractedK}` tags. The `View` IR carries permutation/shape for
  recognition; a contraction additionally needs the *role* tag per axis. **Extend the
  axis fact with a role, or derive roles from the two-operand shape overlap.**
- per-operand **layout facts** (row/col-major, leading dim, transpose, inner-two-axis
  contiguity) — partially present via operand strides; formalize the `{RowMajor,ColMajor}`
  projection the fragment selector reads.
- **alignment per axis** (MMA-k class) — reuse `DivBucket` per axis.

**Item 02 (DAG) must still expose** (the interner + `lower_dag` are landed; one bridge
remains):
- **a `Contraction` as a DAG-node producer** whose result is the `Reduced`-like leaf a
  consuming `ScalarExpr` epilogue reads — the bridge between the structural
  (contraction) and pointwise (tree) worlds. Today `from_expr` interns a pure
  `ScalarExpr`; it needs to admit a "contraction result" leaf with a node identity and
  a `consumers` count so a *downstream* fused op re-reading the matmul output gets an
  honest `consumers > 1` (the fused-reduction epilogue follow-up is the same shape of
  work — the epilogue-dedup follow-up brief and this share the "DAG over a fused
  producer" bridge).

Both lists are now **small and concrete** because the keystones landed — the spike's
job (land 01/02 contraction-ready) is largely discharged; what remains is the
role-tag on 01 and the producer-leaf on 02.

## 8. The §7 vendor-exclusion gate — extend `seed_winner`

The gate is per-cell and measured, seeded by hand-knowledge — and item 07 already
built the mechanism. This node adds contraction arms to `seed_winner`:

```
seed_winner(key) for OpCategory::Gemm:
  route OUT (Cublas/Cutlass, do NOT generate):
    - M,N,K all Large, aligned, Uniform/None batch, {f16,bf16,f32,f64}   → Cublas
    - Grouped batch (MoE variable-M)                                     → Cutlass (GroupedGemm)
    - s8/fp8/int4/bin dtypes                                             → Bespoke head GEMM
  GENERATE (return None → the generator emits the cell):
    - Skinny/Tiny M or N (flat-GEMM / decode GEMV-adjacent)
    - irregular K-alignment (DivBucket = Any) at Small size
    - contraction fused across a layout change or into a bias/act/norm epilogue
```

The seeded route-out rows land in the committed dispatch artifact exactly like the
current f16/bf16 GEMM seeds; the generated long-tail cells get `Provenance::Measured`
rows once the item-07 bench-gate populator (the other deferred follow-up) times them
against the cuBLAS oracle. `merge`'s `MIN_FLIP_MARGIN` guard means a marginal
generated win never displaces a vendor seed on noise — the correctness the dispatch
adversarial pass already hardened applies directly here.

## 9. Prototype + validation plan (gated, not merged)

Optional single-cell prototype behind `--features contraction-spike` / an
`experiments/` dir (mirrors `experiments/elementwise_specialization.cu`), **not
merged to main**:

- one **skinny-GEMM f32 cell** (M=1 GEMV-adjacent), SIMT/`F32Strict` schedule;
- **nvcc numeric on sm_89 (RTX 4070)** vs a **cuBLAS `Sgemm` oracle** and the generic
  strided kernel — bit-for-bit for SIMT, within declared ulp for TF32; cover skinny,
  unaligned-K, and a **fused bias+relu** cell vs a two-kernel (`cuBLAS GEMM` →
  `elementwise`) reference (the fused cell's whole point is to match at lower traffic);
- **compute-sanitizer mandatory** (shared mem + cross-thread): `synccheck` (K-tile
  barriers), `racecheck` (double-buffer), `initcheck` (accumulator + K/M/N remainder);
- **the long-tail win is measured** — the go/no-go number that justifies the node.

**Adversarial-verify targets** (house discipline, the same find→dedup→refute pass the
reduction and DAG changes went through): K-remainder drop/double-count; M/N-remainder
predication OOB; **accumulator-precision misroute** (f16 silently accumulating in f16);
**determinism claim vs reality** (a Tensor-Core cell advertising `bitwise`);
vendor-gate leakage (a Large cell generated instead of routed out); transpose/layout
confusion (use non-square, non-symmetric oracles); NaN/Inf through the epilogue.

**Honest-miss preserved until the node ships:** `unsupported_op_is_rejected`
(`MatMul → UnsupportedOp`) and `derive_pattern`'s `NotElementwise` stay green; the
flip to a positive synthesis test is the *last* wiring step, gated on the Fuel region
grammar (§9 below → §10 asks).

## 10. Fuel asks (cross-repo — gate seam adoption, not the AOT node)

- **Region grammar for a generated contraction.** Fuel's frozen `OpTag` has no
  synthesizable `MatMul`/`Contract` region shape. For a generated contraction to be
  **seam-adoptable** (region in → kernel + recipe out), Fuel must define how a fused
  matmul+epilogue region is spelled and matched — exactly the pattern the fused-reduce
  seam ask established. Until answered, the node ships **AOT-only** (as the fused norms
  did), honest-miss preserved on the seam. Propose-first via the Baracuda↔Fuel channel.
- **Vendor-gate authority** — is the generate-vs-route threshold a Baracuda build-time
  seed only, or driven live by Fuel's `dispatch_record`/`miss_record`? The FlashDecoding++
  flat-GEMM is the archetypal `miss_record` demand signal; coordinate the threshold's
  home with items 07/08 (the `merge()` ingest seam is already built for it).
- **Accumulator/precision default** per dtype — TF32 (fast/approximate) vs
  `F32Strict`/SIMT (bit-stable) as the f32 default, or a per-request `PrecisionGuarantee`
  from Fuel. Decides the default determinism claim.

## 11. Status & next step

Design complete and grounded in merged code. Implementation stays blocked on two
small, now-scoped keystone gaps (§7: the 01 axis-role tag; the 02 contraction-producer
leaf) — both shaped correctly by the merged work, neither a reshape. The terminal
ORDER-3 node is ready to build once those two hooks and the Fuel region grammar (§10)
land; large GEMM continues to route to the head via the item-07 gate.
