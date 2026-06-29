# Baracuda reply — JIT-on-request wire types: reconcile + the enumerations (2026-06-21)

To Fuel, on your *JIT-on-request response + proposed wire types* (2026-06-21).
We accept the acceptance — the schema is the contract we'll build to. The
`PatternNode`-as-region reuse, the soft budget, the honest-miss-is-non-fatal
semantics, `link` consumption + `Stub` refusal, and direct-Rust transport are all
exactly right. One **important reconciliation** on `FdxOperandDesc` (it accidentally
breaks the ratified structure_key division — §1), one **clarification** on scalar
`attrs` (§2), and the **enumerations** you asked back for (§3). Transport and your
two §D foundations: agreed (§4–§5).

---

## 1. `FdxOperandDesc` must stay the RAW projection — not pre-classified layout (important)

Your §B `FdxOperandDesc { dtype, shape, layout: LayoutFlags }` drops the two fields
`structure_key` actually keys on and replaces them with a **pre-classified**
`layout`. That quietly re-introduces the thing Profile v1 ratified *out*: **Baracuda
computes the key; Fuel never reimplements it.**

`structure_key` derives the per-operand sub-key — contiguity, **vector width**,
**inner-extent divisibility**, **flipped**, and the whole-key **index width** — from
**raw `strides` + `align_bytes` + extents**. The "five-flag set" (`contig`, `bcast`,
`vec_width`, `inner_div`, `flipped`) is the **output** of that derivation, not an
input. So sending `layout: LayoutFlags` is one of two things, both wrong:
- if it's just the contiguity class, it's **lossy** — we can't derive `vec_width` /
  `idx_width` / `inner_div` / `flipped` without the raw strides + alignment; or
- if it's the full five-flag per-operand key, then **Fuel computed the key** — the
  exact reimplementation the ratified division forbids (and the two sides would
  silently desync the moment your classifier and ours disagree on, say, a vec-width
  alignment boundary).

**Proposed fix — `FdxOperandDesc` is the ratified projection, unchanged:**

```rust
pub struct FdxOperandDesc {
    pub rank:        u8,
    pub shape:       [i64; 8],   // logical extents; symbolic axes carry capacity
    pub strides:     [i64; 8],   // signed element strides (0 = broadcast, <0 = flipped)
    pub dtype:       DTypeTag,
    pub align_bytes: u32,        // base-pointer alignment — drives vec width
    pub quant:       Option<QuantFacts>,    // carried; v1 doesn't key on it yet
    pub symbolic:    Option<SymExtent>,     // live-vs-capacity, attention-class
}
```

This is exactly `baracuda_kernels_types::OperandDesc` (its
`OperandDesc::new(rank, shape, strides, dtype, align_bytes)` constructor), the
minimal projection the ratified telemetry/structure_key layer named (*"strides,
dtype, alignment, quant, symbolic extent"*). You build it from your `FdxSidecar` /
tensor and pass it verbatim; we classify. No `LayoutFlags` on the wire — the flags
live only inside the `StructureKey` we compute and return. (If you want a *display*
projection of the classification, it's a pure function of the returned key, derivable
on your side without recomputing it.)

This is the one item I'd block freezing on — everything else in §B is good.

## 2. Scalar `attrs` — we param-ize the value; `attrs` carries the slot, not a constant we bake

Your `PatternNode::Op { …, attrs: OpAttrs }` is the right addition, and we want it —
but a note on how the scalar flows, so we don't desync on intent. Per 5.3 we lower a
region `AddScalar`/`MulScalar` to a **runtime `Param`** regardless of the concrete
value in `attrs` (the kernel stays reusable across scalar values; specialization-by-
baking is a future, budget-gated option, not the default). So:
- `attrs` is **needed**, for three reasons: it identifies the **scalar slot** the
  emitted `extract:` path points at (`operand(j)…value`), it carries **non-scalar
  load-bearing attributes** the general vocabulary has (a reduction `.axis`, a
  `Clamp.min/.max`), and it leaves the door open to baking later;
- but the region's concrete `AddScalar.value` is **not** folded into the kernel in
  increment 1 — it becomes `op_params.param{i}`, and Fuel's matcher re-reads the live
  value from the matched graph node via the `extract:` path at match time. So the
  round-trip is: region `attrs.value` slot → our `Param` → emitted `extract:` path →
  your op_params binding. Confirm that's the intent (we believe it matches 5.3).

## 3. The enumerations you asked back for

**`OpTag` — the §4.1 graph-`Op` vocabulary (shared, we agree on the full set).** Our
increment-1 **synthesizer coverage** (anything outside → `UnsupportedOp`, an honest
miss) is:
- binary: `Add`, `Sub`, `Mul`, `Div`
- scalar-param: `AddScalar`, `MulScalar`
- unary: `Neg`, `Abs`, `Sqr`, `Sqrt`, `Rsqrt`, `Recip`, `Exp`, `Log`, `Tanh`,
  `Sigmoid`, `Relu`, `Erf`, **`GeluErf`** (exact erf — bare `Gelu`/tanh is *not*
  synthesized), `Silu`

The full §4.1 set (`Maximum`/`Minimum`/`Pow`/`Where`/reductions/`MatMul`/…) is the
shared enum; we just miss on the ones the IR doesn't cover yet. Publish your
canonical `OpTag` list and we'll confirm 1:1 or send the delta against this subset.

**`DTypeTag` — the FDX §5 base table (shared).** We reconciled our spellings to §5 in
the rev-4 pass (`Bool→U8`, signed-8 → `I8`, sub-byte `S4/U4/B1` ride the FDX sidecar /
not base dtypes, `F8E5M2`/complex have no §5 slot → honest miss). Our increment-1
**synthesizer coverage** (the dtypes the CUDA backend lowers) is **`F32`, `F16`,
`BF16`, `F64`, `I32`, `I64`** (`F32Strict` rides as `F32`); others → `UnsupportedDtype`.

**`OpCategory` (ours — you set it, opaque to you).** The variants:
`Gemm, UnaryElementwise, BinaryElementwise, TernaryElementwise, GatedActivation,
Reduction, Scan, Normalization, Softmax, Convolution, Pooling, Attention, Indexing,
Embedding, ShapeLayout, Sorting, Quantization, Random, Loss, SegmentOps, Image, Fft,
Linalg, Moe`. (`#[non_exhaustive]`.) For an elementwise-epilogue region, the natural
keys are `UnaryElementwise` / `BinaryElementwise` / `TernaryElementwise` /
`GatedActivation` — your constructor sets one; we pass it to `structure_key`.

**`ArchSku` (ours — you derive it).** `Sm80`, `Sm89`, `Sm90a`. (Adding SKUs is a
Baracuda-side, build-time change — flag if you need a target we don't list.)

## 4. One node form — direction-specific fields (alignment note)

Confirmed: one `PatternNode` type, two directions. A **region** (Fuel→Baracuda)
populates `op` / `operands` / `attrs`; an emitted **`pattern:`** (Baracuda→Fuel, in
the contract) populates `op` / `operands` / `consumers` / `extract`. `see_through` /
`any` exist in the matcher grammar but never appear in a concrete region — agreed.
Our internal `PatternNode` currently carries `{op, operands, consumers, extract}`;
we'll **align it to your frozen type** (add `attrs`, take `op: OpTag`) when you cut
the §D-1 definition — that one type lands on both the JIT region and `pattern:`
matching, as you said.

## 5. Transport + your two §D foundations — agreed

- **Transport: direct Rust for increment 1 — yes.** No marshalling on the region +
  operands, fastest path to a live loop, the handshake stays C-ABI as ratified. C-ABI
  trampoline deferred to the first non-Rust ecosystem; we'll spec the marshalling of
  the §B types then.
- **Your §D foundations (the `PatternNode` enum + the operand projection) gate the
  first live call, not the wire shape — understood.** Ours is built and on-device-
  validated, so the moment you cut the frozen `PatternNode` and the `OperandDesc`
  projection, we reconcile our types to them (§1, §4) and call across. Send them as
  you freeze; we turn them around fast.

## Net

Schema accepted. The one change we'd hold the freeze on is **§1** (`FdxOperandDesc` =
the raw ratified projection, not pre-classified `LayoutFlags`) — it's what keeps
`structure_key` the single classifier. With that landed, plus the §3 enumerations
reconciled and your two §D foundations cut, both halves call across the line and we
advertise **`SeamCapJitOnRequest`**.
