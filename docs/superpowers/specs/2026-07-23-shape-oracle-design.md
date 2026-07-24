# Design: the §6.20 output-shape oracle for `baracuda-kernelgen`

**Date:** 2026-07-23
**Status:** design approved; ready for an implementation plan.
**Scope owner:** Baracuda (kernelgen + kernel-vocab).

## 1. Motivation

Every non-elementwise contract Baracuda emits — matmul, softmax/reduce, pooling,
gather — currently **omits `shape_rule` entirely** and defers the output shape to
the recipe, because kernelgen's only shape-rule form is `same_as(in0)` (an
elementwise-only claim). Pooling and conv contracts are therefore *shape-silent*.
This is the exact gap Baracuda declared itself a "future consumer" of on Fuel's
#80 `Dims`/`WithDim` RFC, and KISS-Ops **§6.20** ("Op shape rules — the shape-side
oracle") just pinned the closed vocabulary for it.

This design gives kernelgen an **internal, symbolic output-shape oracle** and the
**§6.20 canonical wire codec**, validated against the CPU oracle. It deliberately
stops short of emitting shape rules into FKC contracts (that step is gated on
Fuel's evaluator, which today interprets only `same_as(role)`).

Kernelgen deliberately carries **size classes, never concrete extents**
(`StructureKey`), which fits §6.20 perfectly: a shape rule is a *symbolic* function
of operand shapes, evaluated by the consumer against concrete shapes.

## 2. The structural finding that shapes the design

§6.20's emittable vocabulary at this version is narrow, and most ops carry **no
`ShapeExpr` at all** — their output shape is *derived from semantics*
(§6.20-0007/0008):

- elementwise / broadcast → `SameAs(operand)`
- `reduce`-family → input shape with reduced axes removed (`nokd`) / set to 1
  (`kd`) — derived from semantics, not a free attr
- gather / index_select / embedding → data shape with the gathered axis replaced
  by the index shape
- matmul → role-vector-derived (M/N/K axis roles) — carried as axis roles, **not**
  a `ShapeExpr`
- the only *irreducible free cases* → a broadcast target (`SameAs`) and a
  slice/iota offset (a `DimExpr`)

`Dims([DimExpr…])` (0x0B) and `WithDim` (0x0A) — the multi-axis arithmetic
constructors — are **reserved**, not emittable at this vocabulary version; they
enter through the extension registry (umbrella §6.4), which is **exactly Fuel's
#80 RFC**. So pooling / im2col / TopK output shapes (multi-axis floor-division
arithmetic) **cannot yet be emitted as §6.20 wire** — they are the ops that wait
on #80.

## 3. Architecture — three layers

Crate split follows the `structure_key` precedent: the pinned wire codec is
driver-free vocab; the IR-to-wire derivation is kernelgen.

### Layer 1 — internal shape oracle (`baracuda-kernelgen/src/shape.rs`)

```rust
/// The concrete output-0 shape of `op` given its INPUT operand shapes, or a
/// typed error (a surfaced symbolic gap, a not-derivable in-place/runtime
/// extent, a missing input, or an out-of-range axis).
pub fn output_shape(op: &OpDef, input_shapes: &[Vec<i64>])
    -> Result<Vec<i64>, ShapeError>;
```

Covers **every shipped `Access` variant**, including pooling's arithmetic. This is
the real capability gain — kernelgen now *knows* every output shape — and is the
validation target against the CPU oracle.

**Return type (v1 decision):** the oracle returns the single **output-0** shape,
not `Vec<Vec<i64>>`. v1's only multi-output op is `RowSort` with `SortOut::Both`,
whose values and indices outputs have **identical** shapes, so a per-output
vector would be a vector of duplicates (YAGNI). The error is a typed `ShapeError`
enum — `Gap` (§6.20-0004 symbolic extent), `NotDerivable` (in-place scatter dest
/ runtime `k_out`), `MissingInput`, `AxisOutOfRange` — richer than a bare gap so
the not-derivable and out-of-range cases stay distinct at the call site. The
input is the op's **input** operand shapes only (deriving the output *from* the
inputs is the point). Should a future op gain outputs with differing shapes, the
signature widens then, when there is a real second shape to return.

### Layer 2 — §6.20 wire codec (`baracuda-kernel-vocab/src/shape_expr.rs`)

The pinned closed vocabulary, its evaluator, and the canonical byte codec. Lives
next to `structure_key.rs`, driver-free, re-exported from `lib.rs`.

### Layer 3 — the bridge (`shape.rs`)

Per `Access` variant, lower the internal rule to its §6.20 representation:
`SameAs` (elementwise/broadcast); the existing semantic attrs for
reduce/gather/matmul (no `ShapeExpr`); a `DimExpr` for the slice/iota free case;
and for pooling / im2col / TopK a typed **`NeedsReservedConstructor`** marker that
names the #80 `Dims`/`WithDim` dependency honestly rather than fabricating a shape.

**No `contract.rs` change in v1** — the shape-rule emit is Fuel-evaluator-gated.

## 4. The per-op derivation table

| `Access` | Output shape from operand shapes + attrs | §6.20 wire form |
|---|---|---|
| `Elementwise` | broadcast frame (per-axis max across inputs) | `SameAs(0)` (or the broadcast operand) |
| `Reduction{axes,keepdim}` | input with `axes` removed (nokd) / set 1 (kd) | semantic (reduce_axes attr) |
| `RowReduce` | = input (full-width epilogue) | `SameAs(0)` |
| `Contraction{axes}` | role-derived: batch dims ++ `[M, N]` | semantic (M/N/K roles) |
| `Scan` | = input | `SameAs(0)` |
| `Window{size,stride,dilation,pad_lo,pad_hi,axis}` | pooled axis → `(in + pad_lo + pad_hi − dilation·(size−1) − 1) ÷ stride + 1` (floor-div) | **`NeedsReservedConstructor`** (WithDim, #80) |
| `RowSort{limit}` | Full → = input; TopK → trailing axis = `k_out` | Full `SameAs(0)`; TopK **`NeedsReservedConstructor`** (WithDim) |
| `Im2Col` | `[N, C·∏kernel, L_out]` (multi-axis geometry) | **`NeedsReservedConstructor`** (Dims, #80) |
| gather / index_select / embedding (`read_index`) | data with `axis` replaced by index shape | semantic (gather axis) |
| scatter (`write_index`) | = dest shape | `SameAs(dest)` |

**Emit-ready today:** elementwise / rowreduce / scan / scatter (`SameAs`) + the
reduce / gather / matmul semantic forms + the slice/iota `DimExpr` free case.
**#80-blocked:** Window / TopK / Im2Col.

**`SameAs` references the frame-defining operand.** For elementwise the output
shape is the broadcast frame (per-axis max across inputs). In kernelgen the op's
own **output operand** carries that full frame (operands are inputs *then*
output), so `SameAs(<output operand index>)` is exactly correct. The degenerate
case where *no single operand* carries the full frame (a frame assembled only by
per-axis max, no operand equal to it) cannot be `SameAs` — it degrades to the same
`NeedsReservedConstructor` (`Dims`) gap; the implementation asserts frame ==
some-operand before emitting `SameAs`, else marks the gap. Layer 1's concrete
computation is unaffected either way.

## 5. The wire codec + evaluator (Layer 2 detail)

**Types**

```rust
pub enum ShapeExpr { SameAs(u8) }                 // whole-shape (operand index)
pub enum Axis { Idx(u8), Last }                   // `last` = trailing axis
pub enum DimExpr {
    Extent(u8, Axis), Const(i64), Param(u8),
    Add(Box<DimExpr>, Box<DimExpr>), Sub(..), Mul(..), Div(..),
}
```

**Evaluator (§6.20-0002/0003/0004)**
- `Last` resolves to `rank − 1` against the referenced operand's rank.
- a concrete `axis ≥ rank` (and `Last` on a rank-0 operand) → typed decline.
- `÷` is **floor-division** (quotient toward −∞); `÷ 0` → typed decline; never
  round toward zero, never panic.
- a symbolic / data-dependent operand extent → a propagated **`ShapeGap`** (the
  same type Layer 1's `output_shape` returns: never a decline, never a panic),
  including through a whole-shape `SameAs` over a partially-symbolic operand. A
  `ShapeGap` (symbolic extent that can't resolve to a concrete `i64`) and a typed
  **decline** (a malformed blob or an illegal axis/divisor) are distinct
  outcomes: a gap is surfaced to telemetry as an opaque-op, a decline rejects.

**Canonical byte codec (§6.20-0005)**
- one-byte tags: `SameAs=0x01, Extent=0x02, Const=0x03, Param=0x04, Add=0x05,
  Sub=0x06, Mul=0x07, Div=0x08`; reserved `Reduce=0x09, WithDim=0x0A, Dims=0x0B`;
  `0x00` reserved.
- fixed-width LE fields: `operand`/`field` as `u8`; `axis` as a `u8` with `0xFF`
  = the `last` sentinel (a **distinct** single-axis `u8` sentinel, high in the
  spirit of §6.19-0020's trailing-axis sentinel, **not** byte-identical to that
  `u16` axis-set mask `0xFFFE`); `Const` as `i64` LE.
- each child expression **definite-length-prefixed** with a `u16` LE byte length
  (§6.19-0010). Byte-deterministic (hashable / byte-comparable).

**Reader (§6.20-0006)** — a **typed decline, never a panic**: the reserved `0x00`
tag, a reserved-but-unregistered tag, a blob shorter than its tag's schema, and
trailing bytes after a complete expression each raise a typed decline; a
well-formed blob round-trips (`decode(encode(x)) == x`). Reuses the exact
`structure_key::from_token` decline discipline.

## 6. Validation — the oracle differential

The CPU oracle *reads* each output shape from the caller-supplied output operand
(e.g. `eval_window` reads `k_out = out.shape[rank-1]`), so it is not an
independent shape *deriver* — but it is a two-way consistency pin:

> for each op + a concrete operand-shape instance, assert
> `output_shape(op, in_shapes) == the output shape the CPU oracle was handed and
> ran with`.

Our derivation and the test's authored (hand-computed) golden shape must agree,
and the oracle must accept and run with that shape. This is the standard golden
discipline used throughout kernelgen.

Plus the six §6.20 conformance-shaped unit tests in the vocab crate
(`test_shape_expr_*`: vocabulary eval, axis / floor-div, symbolic gap,
serialization golden, decode declines, primitive-floor rules) and a never-panic
decode fuzz in the `from_token` mold.

## 7. Out of scope for v1 (named, not silently dropped)

- **`contract.rs` `shape_rule` emit** — Fuel-evaluator-gated (Fuel interprets
  only `same_as(role)` today; `from_params`/arithmetic are stubs, grown by #80).
- **The `Dims` / `WithDim` *encoder*** — reserved until #80 activates. v1 emits
  the typed `NeedsReservedConstructor` gap; lighting up #80 later turns Window /
  TopK / Im2Col from gap to emit-ready with no rework of Layers 1–2.
- **On-device tie-in** — shapes are host-side.

## 8. Convergence tie

This work makes Baracuda the concrete producer whose pooling / conv / im2col
output shapes motivate activating the reserved `Dims` / `WithDim` constructors —
the "declared future consumer" registered on Fuel's #80 cosign. Layer 1 closes
the internal shape-silence for all ops; Layer 2 delivers the fully-proven §6.20
codec; Layer 3 names exactly which ops are emit-ready vs #80-blocked, feeding the
convergence with real demand.
