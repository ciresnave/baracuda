# §6.20 Output-Shape Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `baracuda-kernelgen` an internal symbolic output-shape oracle covering every shipped `Access` variant, plus the pinned KISS-Ops §6.20 shape-expression wire codec, validated against the CPU oracle.

**Architecture:** Three layers. (1) `baracuda-kernel-vocab/src/shape_expr.rs` — the closed §6.20 `ShapeExpr`/`DimExpr` vocabulary, its evaluator, and the canonical byte codec (driver-free, sits beside `structure_key.rs`). (2) `baracuda-kernelgen/src/shape.rs` — `output_shape(op, input_shapes)`, deriving concrete output shapes per `Access` variant. (3) The bridge in the same file — lowering each op to its §6.20 wire form, with a typed `NeedsReservedConstructor` marker for the ops that need the reserved `Dims`/`WithDim` constructors (Fuel's #80).

**Tech Stack:** Rust 2024 edition, no new dependencies. `cargo test -p <crate>`, `rustfmt --style-edition 2024`.

## Global Constraints

- **No new dependencies.** Both crates build with their current dependency sets.
- **`baracuda-kernel-vocab` stays driver-free** — no CUDA, no `baracuda-kernelgen` dependency. It is the pure-data vocabulary crate.
- **Workspace lints are `deny`:** `missing_docs` is denied workspace-wide (`Cargo.toml` `[workspace.lints.rust]`). **Every** public item needs a doc comment or the build fails.
- **Never panic on untrusted input.** Decoders and evaluators return typed declines, following the `structure_key::from_token` precedent.
- **No `contract.rs` changes.** Emitting `shape_rule` into FKC contracts is out of scope (Fuel-evaluator-gated).
- **Do not emit the reserved constructors.** `Reduce` (0x09), `WithDim` (0x0A), `Dims` (0x0B) are reserved at this vocabulary version; the encoder must never produce them and the decoder must decline them.
- **Formatting:** run `rustfmt --style-edition 2024 <files>` before each commit.
- **Commit trailer** on every commit:
  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01TUKx27gRHZfivD4N72euGK
  ```

## File Structure

| File | Responsibility |
|---|---|
| `crates/baracuda-kernel-vocab/src/shape_expr.rs` (create) | §6.20 types, evaluator, byte codec. Pure data. |
| `crates/baracuda-kernel-vocab/src/lib.rs` (modify) | `pub mod shape_expr;` + re-exports. |
| `crates/baracuda-kernelgen/src/shape.rs` (create) | `output_shape` per-`Access` derivation + the §6.20 bridge. |
| `crates/baracuda-kernelgen/src/lib.rs` (modify) | `pub mod shape;` + re-exports. |

---

### Task 1: The §6.20 vocabulary and evaluator

**Files:**
- Create: `crates/baracuda-kernel-vocab/src/shape_expr.rs`
- Modify: `crates/baracuda-kernel-vocab/src/lib.rs`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces: `Axis`, `DimExpr`, `ShapeExpr`, `Extent`, `DimValue`, `ShapeDecline`, `eval_dim`, `eval_shape`.

- [ ] **Step 1: Write the failing test**

Create `crates/baracuda-kernel-vocab/src/shape_expr.rs` containing ONLY this test module for now:

```rust
//! KISS-Ops §6.20 shape expressions — the shape-side oracle vocabulary.

#[cfg(test)]
mod tests {
    use super::*;

    fn known(dims: &[i64]) -> Vec<Extent> {
        dims.iter().map(|&d| Extent::Known(d)).collect()
    }

    #[test]
    fn eval_resolves_last_sentinel_and_arithmetic() {
        // operand 0 = [4, 8]; `last` resolves to axis 1 (rank - 1).
        let a = known(&[4, 8]);
        let ops: &[&[Extent]] = &[&a];
        // Extent(0, last) = 8
        let e = DimExpr::Extent(0, Axis::Last);
        assert_eq!(eval_dim(&e, ops, &[]), Ok(DimValue::Known(8)));
        // Extent(0, 0) = 4
        let e = DimExpr::Extent(0, Axis::Idx(0));
        assert_eq!(eval_dim(&e, ops, &[]), Ok(DimValue::Known(4)));
        // (Extent(0,last) + Const(2)) * Const(3) = (8+2)*3 = 30
        let e = DimExpr::Mul(
            Box::new(DimExpr::Add(
                Box::new(DimExpr::Extent(0, Axis::Last)),
                Box::new(DimExpr::Const(2)),
            )),
            Box::new(DimExpr::Const(3)),
        );
        assert_eq!(eval_dim(&e, ops, &[]), Ok(DimValue::Known(30)));
    }

    #[test]
    fn eval_div_is_floor_toward_negative_infinity() {
        let a = known(&[7]);
        let ops: &[&[Extent]] = &[&a];
        // 7 / 2 = 3
        let e = DimExpr::Div(
            Box::new(DimExpr::Extent(0, Axis::Idx(0))),
            Box::new(DimExpr::Const(2)),
        );
        assert_eq!(eval_dim(&e, ops, &[]), Ok(DimValue::Known(3)));
        // -7 / 2 = -4 (floor), NOT -3 (toward zero) — §6.20-0003.
        let e = DimExpr::Div(
            Box::new(DimExpr::Const(-7)),
            Box::new(DimExpr::Const(2)),
        );
        assert_eq!(eval_dim(&e, ops, &[]), Ok(DimValue::Known(-4)));
    }

    #[test]
    fn eval_declines_bad_axis_and_zero_divisor() {
        let a = known(&[4, 8]);
        let rank0: Vec<Extent> = Vec::new();
        let ops: &[&[Extent]] = &[&a, &rank0];
        // axis >= rank
        assert_eq!(
            eval_dim(&DimExpr::Extent(0, Axis::Idx(2)), ops, &[]),
            Err(ShapeDecline::AxisOutOfRange { operand: 0, axis: 2, rank: 2 })
        );
        // `last` on a rank-0 operand
        assert_eq!(
            eval_dim(&DimExpr::Extent(1, Axis::Last), ops, &[]),
            Err(ShapeDecline::LastOnRank0 { operand: 1 })
        );
        // operand index out of range
        assert_eq!(
            eval_dim(&DimExpr::Extent(9, Axis::Idx(0)), ops, &[]),
            Err(ShapeDecline::OperandOutOfRange { operand: 9 })
        );
        // divide by zero
        let e = DimExpr::Div(
            Box::new(DimExpr::Const(8)),
            Box::new(DimExpr::Const(0)),
        );
        assert_eq!(eval_dim(&e, ops, &[]), Err(ShapeDecline::DivByZero));
        // param index out of range
        assert_eq!(
            eval_dim(&DimExpr::Param(3), ops, &[]),
            Err(ShapeDecline::ParamOutOfRange { field: 3 })
        );
    }

    #[test]
    fn eval_propagates_symbolic_gap_never_declines() {
        // §6.20-0004: a symbolic extent surfaces a GAP, never a decline/panic,
        // and propagates through arithmetic.
        let a = vec![Extent::Known(4), Extent::Symbolic];
        let ops: &[&[Extent]] = &[&a];
        assert_eq!(
            eval_dim(&DimExpr::Extent(0, Axis::Last), ops, &[]),
            Ok(DimValue::Gap)
        );
        let e = DimExpr::Add(
            Box::new(DimExpr::Extent(0, Axis::Last)),
            Box::new(DimExpr::Const(5)),
        );
        assert_eq!(eval_dim(&e, ops, &[]), Ok(DimValue::Gap));
        // A gap divisor is a gap, NOT a DivByZero decline.
        let e = DimExpr::Div(
            Box::new(DimExpr::Const(5)),
            Box::new(DimExpr::Extent(0, Axis::Last)),
        );
        assert_eq!(eval_dim(&e, ops, &[]), Ok(DimValue::Gap));
        // A whole-shape SameAs over a partially-symbolic operand keeps the gap.
        assert_eq!(
            eval_shape(&ShapeExpr::SameAs(0), ops),
            Ok(vec![DimValue::Known(4), DimValue::Gap])
        );
    }

    #[test]
    fn eval_shape_same_as_returns_the_whole_operand_shape() {
        let a = known(&[2, 3, 4]);
        let ops: &[&[Extent]] = &[&a];
        assert_eq!(
            eval_shape(&ShapeExpr::SameAs(0), ops),
            Ok(vec![DimValue::Known(2), DimValue::Known(3), DimValue::Known(4)])
        );
        assert_eq!(
            eval_shape(&ShapeExpr::SameAs(5), ops),
            Err(ShapeDecline::OperandOutOfRange { operand: 5 })
        );
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p baracuda-kernel-vocab shape_expr`
Expected: FAIL to compile — `cannot find type Extent in this scope` (and the module isn't registered yet).

- [ ] **Step 3: Register the module**

In `crates/baracuda-kernel-vocab/src/lib.rs`, add `pub mod shape_expr;` to the module list (alphabetical, between `plan` and `sku`):

```rust
pub mod plan;
pub mod shape_expr;
pub mod sku;
```

And add the re-export after the `pub use sku::{...};` line:

```rust
pub use shape_expr::{
    Axis, DimExpr, DimValue, Extent, ShapeDecline, ShapeExpr, eval_dim, eval_shape,
};
```

- [ ] **Step 4: Write the implementation**

Prepend to `crates/baracuda-kernel-vocab/src/shape_expr.rs` (above the test module, keeping the `//!` module doc at the very top):

```rust
//! KISS-Ops §6.20 shape expressions — the shape-side oracle vocabulary.
//!
//! §6.13 pins each op's **value** behaviour; §6.20 pins the **shape** behaviour
//! as its companion: a symbolic function from operand shapes (plus OpAttrs and
//! params) to the output shape. This module is the closed vocabulary, its
//! evaluator, and its canonical byte codec — pure data, driver-free, and a
//! sibling of [`crate::structure_key`] (the other pinned KISS wire codec).
//!
//! **Reserved constructors.** `Reduce` (0x09), `WithDim` (0x0A) and `Dims`
//! (0x0B) are reserved at this vocabulary version (§6.20-0002): they enter
//! through the extension registry (umbrella §6.4). This module never emits
//! them and declines them on decode.

/// A reference to one axis of an operand: a concrete index, or the reserved
/// `last` sentinel denoting the trailing axis (§6.20-0002). The KISS axis
/// convention is non-negative — a signed/negative axis never appears.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Axis {
    /// A concrete, non-negative operand-axis index.
    Idx(u8),
    /// The trailing axis; resolves to `rank - 1` against the referenced operand.
    Last,
}

/// One operand extent as seen by the evaluator: a concrete integer, or a
/// symbolic / data-dependent length that cannot resolve at evaluation time
/// (§6.20-0004).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Extent {
    /// A concrete extent.
    Known(i64),
    /// A symbolic / data-dependent extent (surfaces as a [`DimValue::Gap`]).
    Symbolic,
}

/// The result of evaluating a single dimension.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum DimValue {
    /// A concrete resolved extent.
    Known(i64),
    /// A **surfaced gap** (§6.20-0004): some referenced extent was symbolic, so
    /// the dimension cannot resolve. Never a decline, never a panic — a consumer
    /// surfaces it as an opaque-op / telemetry gap.
    Gap,
}

/// Why a shape expression could not be evaluated. Distinct from a
/// [`DimValue::Gap`]: a decline *rejects*, a gap *surfaces*.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ShapeDecline {
    /// A concrete `axis` was `>= rank` of the referenced operand.
    AxisOutOfRange {
        /// The referenced operand index.
        operand: u8,
        /// The offending axis index.
        axis: u8,
        /// The referenced operand's rank.
        rank: usize,
    },
    /// The `last` sentinel was used on a rank-0 operand (no trailing axis).
    LastOnRank0 {
        /// The referenced operand index.
        operand: u8,
    },
    /// The expression referenced an operand index the caller did not supply.
    OperandOutOfRange {
        /// The offending operand index.
        operand: u8,
    },
    /// The expression referenced a param field the caller did not supply.
    ParamOutOfRange {
        /// The offending param field index.
        field: u8,
    },
    /// A `Div` had a concrete zero divisor.
    DivByZero,
}

/// A single-dimension expression (§6.20-0002 `DimExpr`).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum DimExpr {
    /// The extent of `operand` along `axis`.
    Extent(u8, Axis),
    /// A literal extent.
    Const(i64),
    /// A declared param field, resolved from the caller's param list.
    Param(u8),
    /// Sum of two dimension expressions.
    Add(Box<DimExpr>, Box<DimExpr>),
    /// Difference of two dimension expressions.
    Sub(Box<DimExpr>, Box<DimExpr>),
    /// Product of two dimension expressions.
    Mul(Box<DimExpr>, Box<DimExpr>),
    /// **Floor** division (quotient toward −∞) of two dimension expressions.
    Div(Box<DimExpr>, Box<DimExpr>),
}

/// A whole-shape expression (§6.20-0002 `ShapeExpr`).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ShapeExpr {
    /// The referenced operand's whole shape.
    SameAs(u8),
}

/// Resolve `axis` against an operand of `rank`, or decline.
fn resolve_axis(operand: u8, axis: Axis, rank: usize) -> Result<usize, ShapeDecline> {
    match axis {
        Axis::Last => {
            if rank == 0 {
                Err(ShapeDecline::LastOnRank0 { operand })
            } else {
                Ok(rank - 1)
            }
        }
        Axis::Idx(a) => {
            if usize::from(a) >= rank {
                Err(ShapeDecline::AxisOutOfRange {
                    operand,
                    axis: a,
                    rank,
                })
            } else {
                Ok(usize::from(a))
            }
        }
    }
}

/// Evaluate a [`DimExpr`] against concrete operand shapes and params.
///
/// `operands[i]` is operand `i`'s per-axis extents; `params[f]` is param field
/// `f`. A symbolic extent yields [`DimValue::Gap`] and propagates through
/// arithmetic (§6.20-0004) — including as a divisor, where it is a gap rather
/// than a [`ShapeDecline::DivByZero`].
///
/// # Errors
/// Returns a [`ShapeDecline`] for an out-of-range operand/axis/param, `last` on
/// a rank-0 operand, or a concrete zero divisor. Never panics.
pub fn eval_dim(
    e: &DimExpr,
    operands: &[&[Extent]],
    params: &[Extent],
) -> Result<DimValue, ShapeDecline> {
    match e {
        DimExpr::Const(v) => Ok(DimValue::Known(*v)),
        DimExpr::Param(f) => params
            .get(usize::from(*f))
            .map(|x| match x {
                Extent::Known(v) => DimValue::Known(*v),
                Extent::Symbolic => DimValue::Gap,
            })
            .ok_or(ShapeDecline::ParamOutOfRange { field: *f }),
        DimExpr::Extent(operand, axis) => {
            let shape = operands
                .get(usize::from(*operand))
                .ok_or(ShapeDecline::OperandOutOfRange { operand: *operand })?;
            let d = resolve_axis(*operand, *axis, shape.len())?;
            Ok(match shape[d] {
                Extent::Known(v) => DimValue::Known(v),
                Extent::Symbolic => DimValue::Gap,
            })
        }
        DimExpr::Add(a, b) | DimExpr::Sub(a, b) | DimExpr::Mul(a, b) | DimExpr::Div(a, b) => {
            let (va, vb) = (
                eval_dim(a, operands, params)?,
                eval_dim(b, operands, params)?,
            );
            let (DimValue::Known(x), DimValue::Known(y)) = (va, vb) else {
                // Gap propagates through every arithmetic node, divisor included.
                return Ok(DimValue::Gap);
            };
            Ok(DimValue::Known(match e {
                DimExpr::Add(..) => x.saturating_add(y),
                DimExpr::Sub(..) => x.saturating_sub(y),
                DimExpr::Mul(..) => x.saturating_mul(y),
                DimExpr::Div(..) => {
                    if y == 0 {
                        return Err(ShapeDecline::DivByZero);
                    }
                    // FLOOR division (toward −∞), never toward zero.
                    x.div_euclid(y) - i64::from(x.rem_euclid(y) != 0 && y < 0)
                }
                _ => unreachable!("outer match pinned the binary arms"),
            }))
        }
    }
}

/// Evaluate a [`ShapeExpr`] to a whole shape.
///
/// # Errors
/// Returns [`ShapeDecline::OperandOutOfRange`] if the referenced operand was
/// not supplied. Never panics.
pub fn eval_shape(e: &ShapeExpr, operands: &[&[Extent]]) -> Result<Vec<DimValue>, ShapeDecline> {
    match e {
        ShapeExpr::SameAs(operand) => operands
            .get(usize::from(*operand))
            .map(|s| {
                s.iter()
                    .map(|x| match x {
                        Extent::Known(v) => DimValue::Known(*v),
                        Extent::Symbolic => DimValue::Gap,
                    })
                    .collect()
            })
            .ok_or(ShapeDecline::OperandOutOfRange { operand: *operand }),
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p baracuda-kernel-vocab shape_expr`
Expected: PASS — 5 tests.

Note on the floor-division line: `x.div_euclid(y)` rounds toward −∞ for positive `y` but toward +∞ for negative `y`; the `- i64::from(...)` correction makes it floor for both signs. If the `eval_div_is_floor_toward_negative_infinity` test fails on the negative-divisor case, verify with `assert_eq!((-7i64).div_euclid(2), -4)` and `assert_eq!((7i64).div_euclid(-2), -3)` — floor of `7/-2` is `-4`, so the correction is required.

- [ ] **Step 6: Format, lint, and commit**

```bash
rustfmt --style-edition 2024 crates/baracuda-kernel-vocab/src/shape_expr.rs crates/baracuda-kernel-vocab/src/lib.rs
cargo clippy -p baracuda-kernel-vocab -- -D warnings
git add crates/baracuda-kernel-vocab/src/shape_expr.rs crates/baracuda-kernel-vocab/src/lib.rs
git commit -m "feat(shape_expr): the KISS-Ops §6.20 vocabulary + evaluator

The closed shape-expression vocabulary (ShapeExpr::SameAs, DimExpr::{Extent,
Const, Param, Add, Sub, Mul, Div}) with the §6.20-0003/-0004 evaluator: the
\`last\` sentinel resolves to rank-1, \`÷\` is floor toward −∞, a concrete zero
divisor and an out-of-range axis/operand/param are typed declines, and a
symbolic extent surfaces a propagating Gap (never a decline, never a panic).

Sits beside structure_key.rs as the second pinned KISS wire vocabulary;
driver-free, no new dependencies.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TUKx27gRHZfivD4N72euGK"
```

---

### Task 2: The §6.20 canonical byte codec

**Files:**
- Modify: `crates/baracuda-kernel-vocab/src/shape_expr.rs`
- Modify: `crates/baracuda-kernel-vocab/src/lib.rs`

**Interfaces:**
- Consumes: `Axis`, `DimExpr`, `ShapeExpr` (Task 1).
- Produces: `CodecDecline`, `encode_dim(&DimExpr) -> Vec<u8>`, `decode_dim(&[u8]) -> Result<DimExpr, CodecDecline>`, `encode_shape(&ShapeExpr) -> Vec<u8>`, `decode_shape(&[u8]) -> Result<ShapeExpr, CodecDecline>`.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `crates/baracuda-kernel-vocab/src/shape_expr.rs`:

```rust
    #[test]
    fn codec_golden_bytes_are_canonical() {
        // SameAs(2) = [0x01, 0x02]
        assert_eq!(encode_shape(&ShapeExpr::SameAs(2)), vec![0x01, 0x02]);
        // Extent(1, last) = [0x02, 0x01, 0xFF] — 0xFF is the `last` sentinel.
        assert_eq!(
            encode_dim(&DimExpr::Extent(1, Axis::Last)),
            vec![0x02, 0x01, 0xFF]
        );
        // Extent(0, 3) = [0x02, 0x00, 0x03]
        assert_eq!(
            encode_dim(&DimExpr::Extent(0, Axis::Idx(3))),
            vec![0x02, 0x00, 0x03]
        );
        // Const(1) = [0x03] ++ i64 LE
        assert_eq!(
            encode_dim(&DimExpr::Const(1)),
            vec![0x03, 1, 0, 0, 0, 0, 0, 0, 0]
        );
        // Param(4) = [0x04, 0x04]
        assert_eq!(encode_dim(&DimExpr::Param(4)), vec![0x04, 0x04]);
        // Add(Param(0), Const(2)) = [0x05] ++ u16 len ++ lhs ++ u16 len ++ rhs
        assert_eq!(
            encode_dim(&DimExpr::Add(
                Box::new(DimExpr::Param(0)),
                Box::new(DimExpr::Const(2)),
            )),
            vec![
                0x05, // Add
                0x02, 0x00, // lhs len = 2
                0x04, 0x00, // Param(0)
                0x09, 0x00, // rhs len = 9
                0x03, 2, 0, 0, 0, 0, 0, 0, 0, // Const(2)
            ]
        );
    }

    #[test]
    fn codec_round_trips_every_constructor() {
        let exprs = vec![
            DimExpr::Extent(0, Axis::Last),
            DimExpr::Extent(3, Axis::Idx(7)),
            DimExpr::Const(-9_000_000_000),
            DimExpr::Param(2),
            DimExpr::Div(
                Box::new(DimExpr::Sub(
                    Box::new(DimExpr::Extent(0, Axis::Idx(1))),
                    Box::new(DimExpr::Const(3)),
                )),
                Box::new(DimExpr::Mul(
                    Box::new(DimExpr::Param(1)),
                    Box::new(DimExpr::Const(2)),
                )),
            ),
        ];
        for e in exprs {
            assert_eq!(decode_dim(&encode_dim(&e)), Ok(e.clone()), "round-trip {e:?}");
        }
        let s = ShapeExpr::SameAs(1);
        assert_eq!(decode_shape(&encode_shape(&s)), Ok(s));
    }

    #[test]
    fn codec_declines_reserved_short_and_trailing() {
        // The reserved 0x00 tag.
        assert_eq!(decode_dim(&[0x00]), Err(CodecDecline::ReservedTag(0x00)));
        // Reserved-but-unregistered constructors (Reduce/WithDim/Dims).
        for tag in [0x09u8, 0x0A, 0x0B] {
            assert_eq!(decode_dim(&[tag]), Err(CodecDecline::ReservedTag(tag)));
        }
        // An unknown tag.
        assert_eq!(decode_dim(&[0x7F]), Err(CodecDecline::UnknownTag(0x7F)));
        // A blob shorter than its tag's schema (Const needs 8 bytes of payload).
        assert_eq!(decode_dim(&[0x03, 1, 2, 3]), Err(CodecDecline::Truncated));
        // Empty input.
        assert_eq!(decode_dim(&[]), Err(CodecDecline::Truncated));
        // Trailing bytes after a complete expression.
        assert_eq!(
            decode_dim(&[0x04, 0x00, 0xDE, 0xAD]),
            Err(CodecDecline::TrailingBytes)
        );
        // A child length that overruns the blob.
        assert_eq!(
            decode_dim(&[0x05, 0xFF, 0xFF, 0x04, 0x00]),
            Err(CodecDecline::Truncated)
        );
    }

    #[test]
    fn decode_never_panics_on_arbitrary_bytes() {
        // The reader parses untrusted blobs: only Ok or a typed decline, never
        // a panic — the `structure_key::from_token` discipline.
        let mut state: u64 = 0x5EED_1234;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        for _ in 0..20_000 {
            let len = (next() % 24) as usize;
            let blob: Vec<u8> = (0..len).map(|_| (next() % 256) as u8).collect();
            let _ = decode_dim(&blob);
            let _ = decode_shape(&blob);
        }
        // Structured mutation of VALID blobs reaches the deep parsers.
        let base = encode_dim(&DimExpr::Add(
            Box::new(DimExpr::Extent(0, Axis::Last)),
            Box::new(DimExpr::Const(2)),
        ));
        for _ in 0..20_000 {
            let mut b = base.clone();
            let n = 1 + (next() % 3) as usize;
            for _ in 0..n {
                if b.is_empty() {
                    break;
                }
                let i = (next() as usize) % b.len();
                match next() % 3 {
                    0 => b[i] = (next() % 256) as u8,
                    1 => {
                        b.remove(i);
                    }
                    _ => b.insert(i, (next() % 256) as u8),
                }
            }
            let _ = decode_dim(&b);
        }
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p baracuda-kernel-vocab shape_expr`
Expected: FAIL to compile — `cannot find function encode_dim in this scope`.

- [ ] **Step 3: Write the implementation**

Append to `crates/baracuda-kernel-vocab/src/shape_expr.rs` (above the `#[cfg(test)]` module):

```rust
// ===========================================================================
// Canonical byte codec (§6.20-0005 / §6.20-0006)
// ===========================================================================

/// Tag byte for [`ShapeExpr::SameAs`].
const TAG_SAME_AS: u8 = 0x01;
/// Tag byte for [`DimExpr::Extent`].
const TAG_EXTENT: u8 = 0x02;
/// Tag byte for [`DimExpr::Const`].
const TAG_CONST: u8 = 0x03;
/// Tag byte for [`DimExpr::Param`].
const TAG_PARAM: u8 = 0x04;
/// Tag byte for [`DimExpr::Add`].
const TAG_ADD: u8 = 0x05;
/// Tag byte for [`DimExpr::Sub`].
const TAG_SUB: u8 = 0x06;
/// Tag byte for [`DimExpr::Mul`].
const TAG_MUL: u8 = 0x07;
/// Tag byte for [`DimExpr::Div`].
const TAG_DIV: u8 = 0x08;

/// The `last`-axis sentinel on the wire — a **distinct** single-axis `u8`
/// sentinel chosen high above the `0..MAX_RANK-1` concrete range, in the spirit
/// of the §6.19-0020 trailing-axis sentinel but deliberately **not**
/// byte-identical to that `u16` axis-set mask `0xFFFE` (§6.20-0005).
const AXIS_LAST_SENTINEL: u8 = 0xFF;

/// Why a shape-expression blob could not be decoded (§6.20-0006). Every
/// malformed input is one of these — the decoder never panics.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum CodecDecline {
    /// The reserved `0x00` tag, or a reserved-but-unregistered constructor
    /// (`Reduce` 0x09 / `WithDim` 0x0A / `Dims` 0x0B).
    ReservedTag(u8),
    /// A tag outside the defined and reserved set.
    UnknownTag(u8),
    /// The blob ended before the tag's schema was satisfied (or a child's
    /// declared length overran the buffer).
    Truncated,
    /// Bytes remained after a complete expression was decoded.
    TrailingBytes,
}

/// Encode a [`DimExpr`] in the §6.20-0005 canonical form.
///
/// Byte-deterministic: one-byte tag, fixed-width little-endian fields, and each
/// child definite-length-prefixed with a `u16` LE byte length. Never emits a
/// reserved tag.
#[must_use]
pub fn encode_dim(e: &DimExpr) -> Vec<u8> {
    let mut out = Vec::new();
    push_dim(e, &mut out);
    out
}

fn push_dim(e: &DimExpr, out: &mut Vec<u8>) {
    match e {
        DimExpr::Extent(operand, axis) => {
            out.push(TAG_EXTENT);
            out.push(*operand);
            out.push(match axis {
                Axis::Last => AXIS_LAST_SENTINEL,
                Axis::Idx(a) => *a,
            });
        }
        DimExpr::Const(v) => {
            out.push(TAG_CONST);
            out.extend_from_slice(&v.to_le_bytes());
        }
        DimExpr::Param(f) => {
            out.push(TAG_PARAM);
            out.push(*f);
        }
        DimExpr::Add(a, b) | DimExpr::Sub(a, b) | DimExpr::Mul(a, b) | DimExpr::Div(a, b) => {
            out.push(match e {
                DimExpr::Add(..) => TAG_ADD,
                DimExpr::Sub(..) => TAG_SUB,
                DimExpr::Mul(..) => TAG_MUL,
                DimExpr::Div(..) => TAG_DIV,
                _ => unreachable!("outer match pinned the binary arms"),
            });
            for child in [a, b] {
                let mut buf = Vec::new();
                push_dim(child, &mut buf);
                // Definite-length prefix, u16 LE (§6.19-0010). A child longer
                // than u16::MAX is unreachable for real expressions; saturate
                // rather than panic, and the decoder's length check rejects it.
                let len = u16::try_from(buf.len()).unwrap_or(u16::MAX);
                out.extend_from_slice(&len.to_le_bytes());
                out.extend_from_slice(&buf);
            }
        }
    }
}

/// Encode a [`ShapeExpr`] in the §6.20-0005 canonical form.
#[must_use]
pub fn encode_shape(e: &ShapeExpr) -> Vec<u8> {
    match e {
        ShapeExpr::SameAs(operand) => vec![TAG_SAME_AS, *operand],
    }
}

/// Decode a [`DimExpr`] blob.
///
/// # Errors
/// Returns a [`CodecDecline`] for a reserved/unknown tag, a truncated blob, or
/// trailing bytes after a complete expression. Never panics (§6.20-0006).
pub fn decode_dim(bytes: &[u8]) -> Result<DimExpr, CodecDecline> {
    let (e, rest) = take_dim(bytes)?;
    if rest.is_empty() {
        Ok(e)
    } else {
        Err(CodecDecline::TrailingBytes)
    }
}

/// Decode one `DimExpr` off the front of `bytes`, returning it and the remainder.
fn take_dim(bytes: &[u8]) -> Result<(DimExpr, &[u8]), CodecDecline> {
    let (&tag, rest) = bytes.split_first().ok_or(CodecDecline::Truncated)?;
    match tag {
        TAG_EXTENT => {
            let (&operand, rest) = rest.split_first().ok_or(CodecDecline::Truncated)?;
            let (&axis, rest) = rest.split_first().ok_or(CodecDecline::Truncated)?;
            let axis = if axis == AXIS_LAST_SENTINEL {
                Axis::Last
            } else {
                Axis::Idx(axis)
            };
            Ok((DimExpr::Extent(operand, axis), rest))
        }
        TAG_CONST => {
            if rest.len() < 8 {
                return Err(CodecDecline::Truncated);
            }
            let (v, rest) = rest.split_at(8);
            let v = i64::from_le_bytes(v.try_into().expect("split_at(8) yields 8 bytes"));
            Ok((DimExpr::Const(v), rest))
        }
        TAG_PARAM => {
            let (&f, rest) = rest.split_first().ok_or(CodecDecline::Truncated)?;
            Ok((DimExpr::Param(f), rest))
        }
        TAG_ADD | TAG_SUB | TAG_MUL | TAG_DIV => {
            let (lhs, rest) = take_child(rest)?;
            let (rhs, rest) = take_child(rest)?;
            let (l, r) = (Box::new(lhs), Box::new(rhs));
            let e = match tag {
                TAG_ADD => DimExpr::Add(l, r),
                TAG_SUB => DimExpr::Sub(l, r),
                TAG_MUL => DimExpr::Mul(l, r),
                _ => DimExpr::Div(l, r),
            };
            Ok((e, rest))
        }
        // `0x00` is reserved (§6.19-0006); 0x09/0x0A/0x0B are the reserved
        // Reduce/WithDim/Dims constructors — recognized, and declined until
        // they are registered through the extension registry.
        0x00 | 0x09 | 0x0A | 0x0B => Err(CodecDecline::ReservedTag(tag)),
        other => Err(CodecDecline::UnknownTag(other)),
    }
}

/// Read a `u16` LE definite-length-prefixed child expression.
fn take_child(bytes: &[u8]) -> Result<(DimExpr, &[u8]), CodecDecline> {
    if bytes.len() < 2 {
        return Err(CodecDecline::Truncated);
    }
    let (len, rest) = bytes.split_at(2);
    let len = usize::from(u16::from_le_bytes([len[0], len[1]]));
    if rest.len() < len {
        return Err(CodecDecline::Truncated);
    }
    let (body, after) = rest.split_at(len);
    let (e, leftover) = take_dim(body)?;
    if leftover.is_empty() {
        Ok((e, after))
    } else {
        Err(CodecDecline::TrailingBytes)
    }
}

/// Decode a [`ShapeExpr`] blob.
///
/// # Errors
/// Returns a [`CodecDecline`] for a reserved/unknown tag, a truncated blob, or
/// trailing bytes. Never panics (§6.20-0006).
pub fn decode_shape(bytes: &[u8]) -> Result<ShapeExpr, CodecDecline> {
    let (&tag, rest) = bytes.split_first().ok_or(CodecDecline::Truncated)?;
    match tag {
        TAG_SAME_AS => {
            let (&operand, rest) = rest.split_first().ok_or(CodecDecline::Truncated)?;
            if rest.is_empty() {
                Ok(ShapeExpr::SameAs(operand))
            } else {
                Err(CodecDecline::TrailingBytes)
            }
        }
        0x00 | 0x09 | 0x0A | 0x0B => Err(CodecDecline::ReservedTag(tag)),
        other => Err(CodecDecline::UnknownTag(other)),
    }
}
```

- [ ] **Step 4: Add the re-export**

In `crates/baracuda-kernel-vocab/src/lib.rs`, replace the Task-1 `pub use shape_expr::{...}` line with:

```rust
pub use shape_expr::{
    Axis, CodecDecline, DimExpr, DimValue, Extent, ShapeDecline, ShapeExpr, decode_dim,
    decode_shape, encode_dim, encode_shape, eval_dim, eval_shape,
};
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p baracuda-kernel-vocab shape_expr`
Expected: PASS — 9 tests.

- [ ] **Step 6: Format, lint, and commit**

```bash
rustfmt --style-edition 2024 crates/baracuda-kernel-vocab/src/shape_expr.rs crates/baracuda-kernel-vocab/src/lib.rs
cargo clippy -p baracuda-kernel-vocab -- -D warnings
git add crates/baracuda-kernel-vocab/src/shape_expr.rs crates/baracuda-kernel-vocab/src/lib.rs
git commit -m "feat(shape_expr): the §6.20 canonical byte codec

Encode/decode for the shape-expression vocabulary per §6.20-0005: one-byte
tags (SameAs=0x01 … Div=0x08), fixed-width LE fields, the 0xFF \`last\` axis
sentinel (distinct from §6.19-0020's u16 0xFFFE mask), i64 LE Const, and
u16 LE definite-length-prefixed children. Byte-deterministic.

Reader is a typed decline, never a panic (§6.20-0006): the reserved 0x00 tag,
the reserved-unregistered Reduce/WithDim/Dims (0x09/0x0A/0x0B), unknown tags,
truncation, and trailing bytes each decline; well-formed blobs round-trip.
Golden bytes + round-trip + 40k-case never-panic fuzz.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TUKx27gRHZfivD4N72euGK"
```

---

### Task 3: `output_shape` — the SameAs family and reductions

**Files:**
- Create: `crates/baracuda-kernelgen/src/shape.rs`
- Modify: `crates/baracuda-kernelgen/src/lib.rs`

**Interfaces:**
- Consumes: `baracuda_kernelgen::ir::{Access, OpDef}`, `baracuda_kernel_vocab::AxisMask`.
- Produces: `ShapeError`, `output_shape(op: &OpDef, input_shapes: &[Vec<i64>]) -> Result<Vec<i64>, ShapeError>`.

**Note on the signature:** `input_shapes` are the op's **input** operand shapes only — the point is deriving the output *from* the inputs. The oracle returns the single output-0 shape; multi-output ops (`RowSort` with `SortOut::Both`) share one shape across their outputs in v1.

- [ ] **Step 1: Write the failing tests**

Create `crates/baracuda-kernelgen/src/shape.rs`:

```rust
//! The §6.20 output-shape oracle — see `docs/superpowers/specs/2026-07-23-shape-oracle-design.md`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{OpDef, ReduceOp, input};
    use baracuda_kernel_vocab::{AxisMask, ElementKind};

    #[test]
    fn elementwise_output_is_the_broadcast_frame() {
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        assert_eq!(
            output_shape(&op, &[vec![128, 256], vec![128, 256]]),
            Ok(vec![128, 256])
        );
        // Per-axis max across inputs — the frame, not operand 0.
        assert_eq!(
            output_shape(&op, &[vec![1, 256], vec![128, 1]]),
            Ok(vec![128, 256])
        );
    }

    #[test]
    fn reduction_removes_or_keeps_the_reduced_axes() {
        // Empty mask = the legacy LAST-axis default.
        let last = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        assert_eq!(output_shape(&last, &[vec![4, 8]]), Ok(vec![4]));

        // Explicit axis-0 mask, collapse (nokd).
        let ax0 = OpDef::reduction_axes(
            "s0",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b01),
            false,
        );
        assert_eq!(output_shape(&ax0, &[vec![4, 8]]), Ok(vec![8]));

        // Explicit axis-0 mask, keepdim.
        let ax0kd = OpDef::reduction_axes(
            "s0kd",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b01),
            true,
        );
        assert_eq!(output_shape(&ax0kd, &[vec![4, 8]]), Ok(vec![1, 8]));
    }

    #[test]
    fn rowreduce_and_scan_preserve_the_input_shape() {
        use crate::ir::{ReduceStage, reduced};
        let stages = vec![ReduceStage {
            pre: input(0).0,
            op: ReduceOp::Sum,
        }];
        let rr = OpDef::row_reduce("rms", 1, &[ElementKind::F32], stages, reduced(0));
        assert_eq!(output_shape(&rr, &[vec![8, 4096]]), Ok(vec![8, 4096]));

        let sc = OpDef::scan_simple("cumsum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        assert_eq!(output_shape(&sc, &[vec![4, 8]]), Ok(vec![4, 8]));
    }

    #[test]
    fn in_place_scatter_output_is_not_derivable_from_inputs() {
        // Baracuda's scatter is IN-PLACE: the destination IS the output buffer,
        // not an input operand, so its extent is a caller-supplied fact. The
        // oracle says so honestly rather than guessing (§6.11 Gap 2 ratified the
        // EXPLICIT dest operand; until an OpDef carries one, this is the state).
        let sc = OpDef::scatter_add("sa", &[ElementKind::F32], 0, ElementKind::U32);
        assert!(matches!(
            output_shape(&sc, &[vec![6], vec![6]]),
            Err(ShapeError::NotDerivable { .. })
        ));
    }

    #[test]
    fn a_symbolic_input_extent_is_a_typed_gap() {
        let op = OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu());
        // -1 marks a symbolic / data-dependent extent on the caller's side.
        assert_eq!(
            output_shape(&op, &[vec![SYMBOLIC, 256]]),
            Err(ShapeError::Gap)
        );
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p baracuda-kernelgen shape::`
Expected: FAIL to compile — `cannot find function output_shape in this scope`.

- [ ] **Step 3: Write the implementation**

Prepend to `crates/baracuda-kernelgen/src/shape.rs` (above the test module):

```rust
//! The §6.20 output-shape oracle: derive an op's concrete output shape from its
//! **input** operand shapes plus its `Access` attributes.
//!
//! §6.13 pins what a kernel computes; §6.20 pins the shape that computation
//! produces. Kernelgen carries size *classes* (never literal extents) in
//! [`baracuda_kernel_vocab::StructureKey`], so this oracle is the symbolic
//! shape-side companion — a function from operand shapes to the output shape,
//! exactly the §6.20-0001 formulation.
//!
//! See `docs/superpowers/specs/2026-07-23-shape-oracle-design.md`.

use crate::ir::{Access, OpDef, WriteIndex};

/// Sentinel marking a symbolic / data-dependent extent in a caller-supplied
/// input shape. Any op whose output depends on a symbolic extent yields
/// [`ShapeError::Gap`] (§6.20-0004: surfaced, never a decline or a panic).
pub const SYMBOLIC: i64 = -1;

/// Why an output shape could not be derived.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShapeError {
    /// A referenced extent was symbolic / data-dependent, so the output shape
    /// cannot resolve (§6.20-0004). Surfaced as an opaque-op / telemetry gap.
    Gap,
    /// The output shape is genuinely not a function of the inputs — an
    /// in-place op whose destination is the output buffer rather than an input
    /// operand (scatter / bincount). The extent is a caller-supplied fact.
    NotDerivable {
        /// Human-readable reason, naming the op class.
        reason: &'static str,
    },
    /// The caller supplied fewer input shapes than the op declares inputs.
    MissingInput {
        /// The input index that was not supplied.
        index: usize,
    },
    /// An attribute referenced an axis outside the operand's rank.
    AxisOutOfRange {
        /// The offending axis.
        axis: usize,
        /// The operand's rank.
        rank: usize,
    },
}

/// The concrete output shape of `op` given its **input** operand shapes.
///
/// `input_shapes[i]` is input `i`'s shape; an extent equal to [`SYMBOLIC`]
/// marks a data-dependent length. Returns the output-0 shape (v1 multi-output
/// ops share one shape across their outputs).
///
/// # Errors
/// See [`ShapeError`]. Never panics.
pub fn output_shape(op: &OpDef, input_shapes: &[Vec<i64>]) -> Result<Vec<i64>, ShapeError> {
    let in0 = shape_of(input_shapes, 0)?;
    if input_shapes.iter().any(|s| s.contains(&SYMBOLIC)) {
        return Err(ShapeError::Gap);
    }
    // An in-place scatter writes THROUGH the output buffer: its destination is
    // not an input operand, so the output extent is caller-supplied.
    if !matches!(op.write_index, WriteIndex::Direct) {
        return Err(ShapeError::NotDerivable {
            reason: "in-place scatter: the destination is the output buffer, not an input operand",
        });
    }
    match &op.access {
        // Elementwise output = the broadcast frame: per-axis max across inputs
        // (the same frame-max rule the structure key's work class uses).
        Access::Elementwise => Ok(broadcast_frame(input_shapes)),
        // A fused reduce -> broadcast -> elementwise writes a full-width output.
        Access::RowReduce { .. } => Ok(in0.clone()),
        // A prefix scan is shape-preserving.
        Access::Scan { .. } => Ok(in0.clone()),
        // reduce-family: the input shape with the reduced axes removed
        // (keepdim = false) or set to 1 (keepdim = true) — derived from the
        // op's semantics, not a free shape attr (§6.20-0007).
        Access::Reduction { axes, keepdim, .. } => {
            let rank = in0.len();
            // The empty mask is the legacy LAST-axis sentinel.
            let reduced: Vec<usize> = if axes.is_empty() {
                if rank == 0 {
                    return Err(ShapeError::AxisOutOfRange { axis: 0, rank });
                }
                vec![rank - 1]
            } else {
                (0..rank).filter(|&d| axes.is_set(d as u8)).collect()
            };
            if let Some(&bad) = reduced.iter().find(|&&d| d >= rank) {
                return Err(ShapeError::AxisOutOfRange { axis: bad, rank });
            }
            let mut out = Vec::with_capacity(rank);
            for (d, &e) in in0.iter().enumerate() {
                match (reduced.contains(&d), *keepdim) {
                    (true, true) => out.push(1),
                    (true, false) => {}
                    (false, _) => out.push(e),
                }
            }
            Ok(out)
        }
        // Contraction / Window / RowSort / Im2Col land in later tasks.
        _ => Err(ShapeError::NotDerivable {
            reason: "access variant not yet covered by the shape oracle",
        }),
    }
}

/// The rank-aligned broadcast frame: per-axis max extent across every input.
fn broadcast_frame(input_shapes: &[Vec<i64>]) -> Vec<i64> {
    let rank = input_shapes.iter().map(Vec::len).max().unwrap_or(0);
    (0..rank)
        .map(|d| {
            input_shapes
                .iter()
                .map(|s| s.get(d).copied().unwrap_or(1))
                .max()
                .unwrap_or(1)
        })
        .collect()
}

/// Borrow input `i`'s shape, or report it missing.
fn shape_of(input_shapes: &[Vec<i64>], i: usize) -> Result<&Vec<i64>, ShapeError> {
    input_shapes
        .get(i)
        .ok_or(ShapeError::MissingInput { index: i })
}
```

- [ ] **Step 4: Register the module**

In `crates/baracuda-kernelgen/src/lib.rs`, add `pub mod shape;` to the module list (between `recipe` and `slang`):

```rust
pub mod recipe;
pub mod shape;
pub mod slang;
```

And add a re-export after the existing `pub use` block:

```rust
pub use shape::{SYMBOLIC, ShapeError, output_shape};
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p baracuda-kernelgen shape::`
Expected: PASS — 5 tests.

- [ ] **Step 6: Format, lint, and commit**

```bash
rustfmt --style-edition 2024 crates/baracuda-kernelgen/src/shape.rs crates/baracuda-kernelgen/src/lib.rs
cargo clippy -p baracuda-kernelgen -- -D warnings
git add crates/baracuda-kernelgen/src/shape.rs crates/baracuda-kernelgen/src/lib.rs
git commit -m "feat(shape): output-shape oracle — elementwise, rowreduce, scan, reduce

output_shape(op, input_shapes) derives an op's concrete output shape from its
INPUT shapes plus its Access attrs (§6.20-0001). This first slice covers the
shape-preserving family (elementwise = the per-axis-max broadcast frame,
rowreduce, scan) and the reduce family (reduced axes removed when
keepdim=false, set to 1 when true — derived from semantics per §6.20-0007).

Typed outcomes, never panics: a symbolic input extent is a surfaced Gap
(§6.20-0004); an in-place scatter is NotDerivable (its destination is the
output buffer, not an input operand — honest until an OpDef carries the
§6.11-ratified explicit dest).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TUKx27gRHZfivD4N72euGK"
```

---

### Task 4: `output_shape` — contraction and gather (the output ≠ any operand class)

**Files:**
- Modify: `crates/baracuda-kernelgen/src/shape.rs`

**Interfaces:**
- Consumes: `output_shape`, `ShapeError` (Task 3); `crate::ir::{AxisRole, ContractionAxes, ReadIndex}`.
- Produces: no new public items — extends `output_shape` coverage.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `crates/baracuda-kernelgen/src/shape.rs`:

```rust
    #[test]
    fn contraction_output_is_role_derived() {
        use crate::ir::{ContractionAxes, reduced};
        // rank-2 matmul: lhs[M,K] · rhs[K,N] -> [M,N]
        let mm = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        );
        assert_eq!(
            output_shape(&mm, &[vec![8, 4096], vec![4096, 4096]]),
            Ok(vec![8, 4096])
        );
        // batched: lhs[B,M,K] · rhs[B,K,N] -> [B,M,N]
        let bmm = OpDef::contraction(
            "bmm",
            &[ElementKind::F32],
            ContractionAxes::batched_matmul(),
            reduced(0),
        );
        assert_eq!(
            output_shape(&bmm, &[vec![8, 8, 4096], vec![8, 4096, 4096]]),
            Ok(vec![8, 8, 4096])
        );
    }

    #[test]
    fn gather_output_replaces_the_axis_with_the_index_shape() {
        use crate::ir::OobPolicy;
        // §6.20-0008: data[..axis] ++ index ++ data[axis+1..]
        // rank-1 data[6], index[5], axis 0 -> [5]
        let g0 = OpDef::gather(
            "g0",
            &[ElementKind::F32],
            0,
            OobPolicy::Clamp,
            ElementKind::I64,
        );
        assert_eq!(output_shape(&g0, &[vec![6], vec![5]]), Ok(vec![5]));

        // rank-2 data[4,6], index[3], axis 1 -> [4,3]
        let g1 = OpDef::gather(
            "g1",
            &[ElementKind::F32],
            1,
            OobPolicy::Clamp,
            ElementKind::I64,
        );
        assert_eq!(output_shape(&g1, &[vec![4, 6], vec![3]]), Ok(vec![4, 3]));

        // An axis beyond the data rank is a typed decline, not a panic.
        let g9 = OpDef::gather(
            "g9",
            &[ElementKind::F32],
            9,
            OobPolicy::Clamp,
            ElementKind::I64,
        );
        assert!(matches!(
            output_shape(&g9, &[vec![4, 6], vec![3]]),
            Err(ShapeError::AxisOutOfRange { .. })
        ));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p baracuda-kernelgen shape::`
Expected: FAIL — `contraction_output_is_role_derived` and `gather_output_replaces_the_axis_with_the_index_shape` fail with `Err(NotDerivable { reason: "access variant not yet covered by the shape oracle" })`.

- [ ] **Step 3: Write the implementation**

In `crates/baracuda-kernelgen/src/shape.rs`, extend the `use` line:

```rust
use crate::ir::{Access, AxisRole, OpDef, ReadIndex, WriteIndex};
```

Insert this gather early-out in `output_shape`, immediately **after** the `write_index` scatter check and **before** the `match &op.access` block:

```rust
    // A data-dependent gather rides `read_index`, not an `Access` variant, and
    // its output shape is the data shape with the gathered axis REPLACED by the
    // index operand's shape (§6.20-0008 — the class the oracle most exists to
    // catch; never `SameAs(data)`).
    if let Some((data, index_operand, axis)) =
        op.read_index
            .iter()
            .enumerate()
            .find_map(|(i, r)| match r {
                ReadIndex::Indexed {
                    index_operand,
                    axis,
                    ..
                } => Some((i, *index_operand as usize, *axis as usize)),
                ReadIndex::Direct => None,
            })
    {
        let data_shape = shape_of(input_shapes, data)?;
        let index_shape = shape_of(input_shapes, index_operand)?;
        if axis >= data_shape.len() {
            return Err(ShapeError::AxisOutOfRange {
                axis,
                rank: data_shape.len(),
            });
        }
        let mut out = data_shape[..axis].to_vec();
        out.extend_from_slice(index_shape);
        out.extend_from_slice(&data_shape[axis + 1..]);
        return Ok(out);
    }
```

Then replace the `Access` match's catch-all arm with the contraction arm plus a narrower catch-all:

```rust
        // A contraction's output shape is ROLE-derived (§6.20-0008): the shared
        // batch dims, then [M, N]. M comes from the lhs axis tagged FreeM, N
        // from the rhs axis tagged FreeN — read off the op's own ContractionAxes,
        // so it holds for rank-2 and batched alike with no rank special-casing.
        Access::Contraction { axes, .. } => {
            let lhs = in0;
            let rhs = shape_of(input_shapes, 1)?;
            let pick = |shape: &Vec<i64>, roles: &[AxisRole], want: AxisRole| {
                roles
                    .iter()
                    .position(|r| *r == want)
                    .and_then(|d| shape.get(d).copied())
            };
            let mut out: Vec<i64> = axes
                .lhs
                .iter()
                .enumerate()
                .filter(|(_, r)| **r == AxisRole::Batch)
                .filter_map(|(d, _)| lhs.get(d).copied())
                .collect();
            let m = pick(lhs, &axes.lhs, AxisRole::FreeM).ok_or(ShapeError::AxisOutOfRange {
                axis: 0,
                rank: lhs.len(),
            })?;
            let n = pick(rhs, &axes.rhs, AxisRole::FreeN).ok_or(ShapeError::AxisOutOfRange {
                axis: 0,
                rank: rhs.len(),
            })?;
            out.push(m);
            out.push(n);
            Ok(out)
        }
        // Window / RowSort / Im2Col land in Task 5.
        _ => Err(ShapeError::NotDerivable {
            reason: "access variant not yet covered by the shape oracle",
        }),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p baracuda-kernelgen shape::`
Expected: PASS — 7 tests.

- [ ] **Step 5: Format, lint, and commit**

```bash
rustfmt --style-edition 2024 crates/baracuda-kernelgen/src/shape.rs
cargo clippy -p baracuda-kernelgen -- -D warnings
git add crates/baracuda-kernelgen/src/shape.rs
git commit -m "feat(shape): output-shape oracle — contraction and gather

The §6.20-0008 class the oracle most exists to catch: ops whose output shape
equals NO operand's shape. A contraction's output is role-derived (batch dims
++ [M, N], read off the op's ContractionAxes so rank-2 and batched share one
path); a gather's output is the data shape with the gathered axis REPLACED by
the index operand's shape — never SameAs(data). An axis beyond the data rank
is a typed decline.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TUKx27gRHZfivD4N72euGK"
```

---

### Task 5: `output_shape` — the arithmetic cases (Window, TopK, Im2Col)

**Files:**
- Modify: `crates/baracuda-kernelgen/src/shape.rs`

**Interfaces:**
- Consumes: `output_shape`, `ShapeError` (Tasks 3–4); `crate::ir::SortLimit`.
- Produces: `pub fn windowed_extent(input: i64, size, stride, dilation, pad_lo, pad_hi: u8) -> i64` — the shared pooling/im2col geometry, reused by Task 6's `DimExpr` construction.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `crates/baracuda-kernelgen/src/shape.rs`:

```rust
    #[test]
    fn windowed_extent_matches_the_pooling_formula() {
        // (in + pad_lo + pad_hi - dilation*(size-1) - 1) / stride + 1, floor.
        assert_eq!(windowed_extent(8, 2, 2, 1, 0, 0), 4);
        assert_eq!(windowed_extent(4, 2, 1, 1, 0, 0), 3);
        // Padding widens the output.
        assert_eq!(windowed_extent(4, 3, 1, 1, 1, 1), 4);
        // Dilation shrinks it.
        assert_eq!(windowed_extent(8, 3, 1, 2, 0, 0), 4);
        // A window wider than the padded input clamps at 0, never negative.
        assert_eq!(windowed_extent(2, 8, 1, 1, 0, 0), 0);
    }

    #[test]
    fn window_output_downsamples_the_pooled_axis() {
        use crate::ir::reduced;
        let pool = OpDef::window(
            "maxpool",
            1, // n_inputs
            &[ElementKind::F32],
            ReduceOp::Max,
            1,    // axis
            2,    // size
            2,    // stride
            1,    // dilation
            0,    // pad_lo
            0,    // pad_hi
            true, // count_include_pad
            input(0),
            reduced(0),
        );
        assert_eq!(output_shape(&pool, &[vec![4, 8]]), Ok(vec![4, 4]));
    }

    #[test]
    fn rowsort_full_preserves_shape_and_topk_caps_the_trailing_axis() {
        use crate::ir::SortOrder;
        // NOTE: row_sort / row_topk take a SINGLE dtype, not a slice.
        let full = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        assert_eq!(output_shape(&full, &[vec![4, 8]]), Ok(vec![4, 8]));

        // TopK's k_out is a RUNTIME launch arg — the extent is caller-supplied,
        // so the oracle reports it honestly rather than inventing a number.
        let topk = OpDef::row_topk("topk", ElementKind::F32);
        assert!(matches!(
            output_shape(&topk, &[vec![4, 8]]),
            Err(ShapeError::NotDerivable { .. })
        ));
    }

    #[test]
    fn im2col_output_is_the_assembled_three_axis_shape() {
        // in [N=1, C=2, H=4, W=4], kernel 2x2, stride 1, pad 0, dilation 1:
        //   H_out = W_out = (4 - 1*(2-1) - 1)/1 + 1 = 3, L_out = 9
        //   C*kh*kw = 2*2*2 = 8  ->  [1, 8, 9]
        // NOTE: the constructor is `im2col_2d` and takes a SINGLE dtype.
        let ic = OpDef::im2col_2d("im2col", ElementKind::F32, (2, 2), (1, 1), (0, 0), (1, 1));
        assert_eq!(output_shape(&ic, &[vec![1, 2, 4, 4]]), Ok(vec![1, 8, 9]));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p baracuda-kernelgen shape::`
Expected: FAIL to compile — `cannot find function windowed_extent in this scope`; the other three fail with `NotDerivable`.

**If a constructor name mismatches**, check the real signature before adapting the test:
`grep -n "pub fn window\|pub fn row_sort\|pub fn row_topk\|pub fn im2col" crates/baracuda-kernelgen/src/ir.rs`

- [ ] **Step 3: Write the implementation**

In `crates/baracuda-kernelgen/src/shape.rs`, extend the `use` line:

```rust
use crate::ir::{Access, AxisRole, OpDef, ReadIndex, SortLimit, WriteIndex};
```

Add the shared geometry helper above `broadcast_frame`:

```rust
/// The windowed output extent along one axis — the shared pooling / im2col
/// geometry: `(input + pad_lo + pad_hi − dilation·(size−1) − 1) ÷ stride + 1`
/// with **floor** division, clamped at 0 (a window wider than the padded input
/// yields no output positions, never a negative extent).
#[must_use]
pub fn windowed_extent(input: i64, size: u8, stride: u8, dilation: u8, pad_lo: u8, pad_hi: u8) -> i64 {
    let effective = i64::from(dilation) * (i64::from(size) - 1) + 1;
    let padded = input + i64::from(pad_lo) + i64::from(pad_hi);
    let span = padded - effective;
    if span < 0 {
        return 0;
    }
    span / i64::from(stride) + 1
}
```

Then replace the Task-4 catch-all arm with these three arms plus a final catch-all:

```rust
        // Pooling: the pooled axis downsamples by the window geometry; every
        // other axis passes through.
        Access::Window {
            axis,
            size,
            stride,
            dilation,
            pad_lo,
            pad_hi,
            ..
        } => {
            let d = *axis as usize;
            if d >= in0.len() {
                return Err(ShapeError::AxisOutOfRange {
                    axis: d,
                    rank: in0.len(),
                });
            }
            let mut out = in0.clone();
            out[d] = windowed_extent(in0[d], *size, *stride, *dilation, *pad_lo, *pad_hi);
            Ok(out)
        }
        // A full row sort is a permutation (shape-preserving); a TopK caps the
        // trailing axis at a RUNTIME `k_out` launch arg, which is a
        // caller-supplied fact the key deliberately does not carry.
        Access::RowSort { limit, .. } => match limit {
            SortLimit::Full => Ok(in0.clone()),
            SortLimit::TopK => Err(ShapeError::NotDerivable {
                reason: "row TopK: k_out is a runtime launch arg, not an input extent",
            }),
        },
        // im2col assembles a THREE-axis output from a rank-4 NCHW input:
        // [N, C·kh·kw, H_out·W_out].
        Access::Im2Col {
            kernel,
            stride,
            pad,
            dilation,
        } => {
            if in0.len() != 4 {
                return Err(ShapeError::AxisOutOfRange {
                    axis: 3,
                    rank: in0.len(),
                });
            }
            let (n, c, h, w) = (in0[0], in0[1], in0[2], in0[3]);
            let h_out = windowed_extent(h, kernel.0, stride.0, dilation.0, pad.0, pad.0);
            let w_out = windowed_extent(w, kernel.1, stride.1, dilation.1, pad.1, pad.1);
            Ok(vec![
                n,
                c * i64::from(kernel.0) * i64::from(kernel.1),
                h_out * w_out,
            ])
        }
        // Every shipped Access variant is now covered; this arm exists so a
        // NEWLY ADDED variant is an honest miss rather than a wrong shape.
        _ => Err(ShapeError::NotDerivable {
            reason: "access variant not yet covered by the shape oracle",
        }),
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p baracuda-kernelgen shape::`
Expected: PASS — 11 tests.

- [ ] **Step 5: Add the re-export**

In `crates/baracuda-kernelgen/src/lib.rs`, update the shape re-export:

```rust
pub use shape::{SYMBOLIC, ShapeError, output_shape, windowed_extent};
```

- [ ] **Step 6: Format, lint, and commit**

```bash
rustfmt --style-edition 2024 crates/baracuda-kernelgen/src/shape.rs crates/baracuda-kernelgen/src/lib.rs
cargo clippy -p baracuda-kernelgen -- -D warnings
git add crates/baracuda-kernelgen/src/shape.rs crates/baracuda-kernelgen/src/lib.rs
git commit -m "feat(shape): output-shape oracle — Window, RowSort, Im2Col

The arithmetic cases. \`windowed_extent\` is the shared pooling/im2col geometry
((in + pad_lo + pad_hi − dilation·(size−1) − 1) ÷ stride + 1, floor, clamped
at 0); Window downsamples its pooled axis by it, Im2Col assembles the rank-4
NCHW input into [N, C·kh·kw, H_out·W_out]. A full RowSort is shape-preserving;
a TopK is NotDerivable because k_out is a runtime launch arg, not an input
extent. Every shipped Access variant is now covered, with the catch-all kept
so a newly added variant is an honest miss rather than a wrong shape.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TUKx27gRHZfivD4N72euGK"
```

---

### Task 6: The §6.20 bridge — wire form per op, with the reserved-constructor gap

**Files:**
- Modify: `crates/baracuda-kernelgen/src/shape.rs`
- Modify: `crates/baracuda-kernelgen/src/lib.rs`

**Interfaces:**
- Consumes: `windowed_extent` (Task 5); `baracuda_kernel_vocab::{Axis, DimExpr, ShapeExpr}`.
- Produces: `ShapeRuleForm`, `shape_rule_form(op: &OpDef, input_shapes: &[Vec<i64>]) -> ShapeRuleForm`.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `crates/baracuda-kernelgen/src/shape.rs`:

```rust
    #[test]
    fn elementwise_lowers_to_same_as_when_an_input_carries_the_frame() {
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        // input 0 carries the full frame -> SameAs(0)
        assert_eq!(
            shape_rule_form(&op, &[vec![128, 256], vec![128, 256]]),
            ShapeRuleForm::Whole(ShapeExpr::SameAs(0))
        );
        // input 1 carries it, input 0 does not -> SameAs(1)
        assert_eq!(
            shape_rule_form(&op, &[vec![1, 256], vec![128, 256]]),
            ShapeRuleForm::Whole(ShapeExpr::SameAs(1))
        );
        // NO input carries the frame (a[N,1] . b[1,M] -> [N,M]) -> the
        // SameAs-frame guard degrades to the reserved Dims constructor rather
        // than naming a wrong operand.
        assert_eq!(
            shape_rule_form(&op, &[vec![128, 1], vec![1, 256]]),
            ShapeRuleForm::NeedsReservedConstructor {
                constructor: "Dims",
                why: "the broadcast frame equals no single input's shape",
            }
        );
    }

    #[test]
    fn semantic_ops_carry_no_shape_expr() {
        use crate::ir::{ContractionAxes, OobPolicy, reduced};
        // §6.20-0008: a contraction's shape rides its axis ROLES, not a ShapeExpr.
        let mm = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        );
        assert_eq!(
            shape_rule_form(&mm, &[vec![8, 4096], vec![4096, 4096]]),
            ShapeRuleForm::Semantic { attr: "contraction axis roles (M/N/K)" }
        );
        // A reduce's shape rides its reduce_axes attr.
        let red = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        assert_eq!(
            shape_rule_form(&red, &[vec![4, 8]]),
            ShapeRuleForm::Semantic { attr: "reduce_axes + keepdim" }
        );
        // A gather's shape rides its gather axis + the index operand.
        let g = OpDef::gather(
            "g",
            &[ElementKind::F32],
            0,
            OobPolicy::Clamp,
            ElementKind::I64,
        );
        assert_eq!(
            shape_rule_form(&g, &[vec![6], vec![5]]),
            ShapeRuleForm::Semantic { attr: "gather axis + index operand shape" }
        );
    }

    #[test]
    fn pooling_and_im2col_need_the_reserved_constructors() {
        use crate::ir::reduced;
        // These are the ops blocked on the KISS #80 Dims/WithDim activation —
        // a typed marker naming the dependency, never a fabricated shape.
        let pool = OpDef::window(
            "maxpool",
            1, // n_inputs
            &[ElementKind::F32],
            ReduceOp::Max,
            1,    // axis
            2,    // size
            2,    // stride
            1,    // dilation
            0,    // pad_lo
            0,    // pad_hi
            true, // count_include_pad
            input(0),
            reduced(0),
        );
        assert_eq!(
            shape_rule_form(&pool, &[vec![4, 8]]),
            ShapeRuleForm::NeedsReservedConstructor {
                constructor: "WithDim",
                why: "the pooled axis is single-axis extent arithmetic on one operand",
            }
        );

        let ic = OpDef::im2col_2d("im2col", ElementKind::F32, (2, 2), (1, 1), (0, 0), (1, 1));
        assert_eq!(
            shape_rule_form(&ic, &[vec![1, 2, 4, 4]]),
            ShapeRuleForm::NeedsReservedConstructor {
                constructor: "Dims",
                why: "a fully assembled multi-axis shape with no single operand to extend",
            }
        );
    }

    #[test]
    fn pooling_dim_expr_evaluates_to_the_oracle_shape() {
        // The DimExpr the bridge WOULD emit once WithDim is registered must
        // evaluate to exactly what the concrete oracle computes — proving the
        // arithmetic form and the oracle agree before the wire opens.
        use baracuda_kernel_vocab::{Extent, eval_dim, DimValue};
        let e = pooled_axis_dim_expr(0, 1, 2, 2, 1, 0, 0);
        let shape = [Extent::Known(4), Extent::Known(8)];
        let ops: &[&[Extent]] = &[&shape];
        assert_eq!(eval_dim(&e, ops, &[]), Ok(DimValue::Known(4)));
        assert_eq!(windowed_extent(8, 2, 2, 1, 0, 0), 4);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p baracuda-kernelgen shape::`
Expected: FAIL to compile — `cannot find type ShapeRuleForm in this scope`.

- [ ] **Step 3: Write the implementation**

In `crates/baracuda-kernelgen/src/shape.rs`, extend the imports:

```rust
use crate::ir::{Access, AxisRole, OpDef, ReadIndex, SortLimit, WriteIndex};
use baracuda_kernel_vocab::{Axis, DimExpr, ShapeExpr};
```

Append (above the test module):

```rust
// ===========================================================================
// The §6.20 bridge — each op's wire-level shape rule
// ===========================================================================

/// How an op's output shape is expressed in the §6.20 surface.
///
/// Most ops carry **no** `ShapeExpr`: their shape is derived from semantics
/// (§6.20-0007/0008). Only the irreducible free cases carry an expression, and
/// the multi-axis arithmetic cases need constructors that are still reserved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShapeRuleForm {
    /// The output equals a referenced operand's whole shape.
    Whole(ShapeExpr),
    /// The shape is derived from the op's existing attributes, not a shape
    /// expression — `attr` names which ones (§6.20-0007/0008).
    Semantic {
        /// The attribute(s) the shape derives from.
        attr: &'static str,
    },
    /// A single free dimension expressed as a [`DimExpr`].
    Dim(DimExpr),
    /// The rule needs a constructor that is **reserved** at this vocabulary
    /// version (`WithDim` 0x0A / `Dims` 0x0B) and enters through the extension
    /// registry (umbrella §6.4 — KISS #80). Emitting anything else here would
    /// be a fabricated shape, so this is a typed, honest gap.
    NeedsReservedConstructor {
        /// The reserved constructor required (`"WithDim"` or `"Dims"`).
        constructor: &'static str,
        /// Why this op needs it.
        why: &'static str,
    },
    /// The output shape is not a function of the inputs at all (an in-place
    /// destination, or a runtime-supplied extent).
    NotDerivable {
        /// Why the shape is caller-supplied.
        reason: &'static str,
    },
}

/// The `DimExpr` for a windowed (pooled) axis — the wire form of
/// [`windowed_extent`]: `(Extent(operand, axis) + pad_lo + pad_hi
/// − dilation·(size−1) − 1) ÷ stride + 1`.
///
/// Built even while `WithDim` is reserved, so the arithmetic form can be
/// evaluated against the concrete oracle and proven equivalent before the wire
/// opens.
#[must_use]
pub fn pooled_axis_dim_expr(
    operand: u8,
    axis: u8,
    size: u8,
    stride: u8,
    dilation: u8,
    pad_lo: u8,
    pad_hi: u8,
) -> DimExpr {
    let c = |v: i64| Box::new(DimExpr::Const(v));
    let padded = DimExpr::Add(
        Box::new(DimExpr::Extent(operand, Axis::Idx(axis))),
        c(i64::from(pad_lo) + i64::from(pad_hi)),
    );
    // effective window span = dilation*(size-1) + 1
    let effective = i64::from(dilation) * (i64::from(size) - 1) + 1;
    let span = DimExpr::Sub(Box::new(padded), c(effective));
    DimExpr::Add(
        Box::new(DimExpr::Div(Box::new(span), c(i64::from(stride)))),
        c(1),
    )
}

/// The §6.20 wire-level shape rule for `op`.
///
/// `input_shapes` is consulted only to resolve the `SameAs`-frame guard (which
/// operand, if any, carries the broadcast frame); the classification itself is
/// structural.
#[must_use]
pub fn shape_rule_form(op: &OpDef, input_shapes: &[Vec<i64>]) -> ShapeRuleForm {
    if !matches!(op.write_index, WriteIndex::Direct) {
        return ShapeRuleForm::NotDerivable {
            reason: "in-place scatter: the destination is the output buffer, not an input operand",
        };
    }
    if op.read_index.iter().any(|r| !matches!(r, ReadIndex::Direct)) {
        return ShapeRuleForm::Semantic {
            attr: "gather axis + index operand shape",
        };
    }
    match &op.access {
        Access::Elementwise => same_as_frame_or_dims(input_shapes),
        // RowReduce and Scan are shape-preserving: the output equals input 0.
        Access::RowReduce { .. } | Access::Scan { .. } => {
            ShapeRuleForm::Whole(ShapeExpr::SameAs(0))
        }
        Access::Reduction { .. } => ShapeRuleForm::Semantic {
            attr: "reduce_axes + keepdim",
        },
        Access::Contraction { .. } => ShapeRuleForm::Semantic {
            attr: "contraction axis roles (M/N/K)",
        },
        Access::Window { .. } => ShapeRuleForm::NeedsReservedConstructor {
            constructor: "WithDim",
            why: "the pooled axis is single-axis extent arithmetic on one operand",
        },
        Access::RowSort { limit, .. } => match limit {
            SortLimit::Full => ShapeRuleForm::Whole(ShapeExpr::SameAs(0)),
            SortLimit::TopK => ShapeRuleForm::NeedsReservedConstructor {
                constructor: "WithDim",
                why: "the trailing axis is replaced by the runtime k_out param",
            },
        },
        Access::Im2Col { .. } => ShapeRuleForm::NeedsReservedConstructor {
            constructor: "Dims",
            why: "a fully assembled multi-axis shape with no single operand to extend",
        },
        _ => ShapeRuleForm::NotDerivable {
            reason: "access variant not yet covered by the shape oracle",
        },
    }
}

/// `SameAs(i)` for the first input whose shape IS the broadcast frame; if no
/// single input carries the frame, the rule needs the reserved `Dims`
/// constructor rather than naming a wrong operand (the `a[N,1] · b[1,M]`
/// degradation edge).
fn same_as_frame_or_dims(input_shapes: &[Vec<i64>]) -> ShapeRuleForm {
    let frame = broadcast_frame(input_shapes);
    match input_shapes.iter().position(|s| *s == frame) {
        Some(i) => ShapeRuleForm::Whole(ShapeExpr::SameAs(i as u8)),
        None => ShapeRuleForm::NeedsReservedConstructor {
            constructor: "Dims",
            why: "the broadcast frame equals no single input's shape",
        },
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p baracuda-kernelgen shape::`
Expected: PASS — 15 tests.

- [ ] **Step 5: Add the re-export**

In `crates/baracuda-kernelgen/src/lib.rs`, update the shape re-export:

```rust
pub use shape::{
    SYMBOLIC, ShapeError, ShapeRuleForm, output_shape, pooled_axis_dim_expr, shape_rule_form,
    windowed_extent,
};
```

- [ ] **Step 6: Run the full workspace suite**

Run: `cargo test -p baracuda-kernel-vocab -p baracuda-kernelgen`
Expected: PASS — vocab and kernelgen both green, no pre-existing test broken.

- [ ] **Step 7: Format, lint, and commit**

```bash
rustfmt --style-edition 2024 crates/baracuda-kernelgen/src/shape.rs crates/baracuda-kernelgen/src/lib.rs
cargo clippy -p baracuda-kernelgen -- -D warnings
git add crates/baracuda-kernelgen/src/shape.rs crates/baracuda-kernelgen/src/lib.rs
git commit -m "feat(shape): the §6.20 bridge — wire form per op + reserved-constructor gap

shape_rule_form(op) classifies each op's §6.20 surface: Whole(SameAs) for the
shape-preserving family, Semantic for the ops whose shape rides existing attrs
(reduce_axes/keepdim, contraction roles, gather axis — §6.20-0008 explicitly
carries these as attrs, NOT a ShapeExpr), and a typed
NeedsReservedConstructor for the ops that need the still-reserved WithDim
(pooling, TopK) and Dims (im2col) constructors — the KISS #80 dependency named
honestly rather than a fabricated shape.

Includes the SameAs-frame guard: an elementwise op whose broadcast frame
equals no single input (a[N,1]·b[1,M]) degrades to the Dims gap rather than
naming a wrong operand. pooled_axis_dim_expr builds pooling's wire arithmetic
now and pins it equal to the concrete oracle, so the form is proven before the
wire opens.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TUKx27gRHZfivD4N72euGK"
```

---

### Task 7: The CPU-oracle differential

**Files:**
- Modify: `crates/baracuda-kernelgen/src/shape.rs`

**Interfaces:**
- Consumes: `output_shape` (Tasks 3–5); `crate::oracle::{self, TypedBuffer}`, `crate::plan::build_plan`, `baracuda_kernel_vocab::{ArchSku, OpCategory, OperandDesc, structure_key}`.
- Produces: no new public items — this is the spec's §6 validation gate.

**Why:** every prior task checks our derivation against a *hand-computed* golden. This task closes the loop the spec requires: run the CPU oracle with that authored output operand and assert the shape it actually produced equals our derivation. Two-way pin — a wrong derivation *or* a wrong authored golden fails.

**Coverage note:** `oracle::evaluate` deliberately does not implement `Contraction`, `RowSort`, or gather/scatter (it panics on them — see its module docs), so this differential covers `Elementwise`, `Reduction`, `Scan`, `RowReduce`, and `Window`. The uncovered variants keep their hand-computed goldens from Tasks 4–5.

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `crates/baracuda-kernelgen/src/shape.rs`:

```rust
    /// Row-major dense strides for `shape`.
    fn dense_strides(shape: &[i64]) -> Vec<i64> {
        let mut s = vec![1i64; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            s[d] = s[d + 1] * shape[d + 1];
        }
        s
    }

    fn od(shape: &[i64]) -> baracuda_kernel_vocab::OperandDesc {
        baracuda_kernel_vocab::OperandDesc::new(
            shape.len(),
            shape,
            &dense_strides(shape),
            ElementKind::F32,
            256,
        )
    }

    fn zeros(shape: &[i64]) -> crate::oracle::TypedBuffer {
        let n: i64 = shape.iter().product();
        crate::oracle::TypedBuffer::new(
            ElementKind::F32,
            shape.to_vec(),
            dense_strides(shape),
            vec![0u8; (n.max(0) as usize) * 4],
        )
    }

    /// Run the CPU oracle for `op` over `in_shapes` with the AUTHORED
    /// `out_shape`, and assert the shape it produced equals `output_shape`'s
    /// derivation. The spec's §6 two-way pin.
    fn assert_oracle_agrees(
        op: &OpDef,
        cat: baracuda_kernel_vocab::OpCategory,
        in_shapes: &[Vec<i64>],
        out_shape: &[i64],
    ) {
        // Derivation must match the authored golden first.
        assert_eq!(
            output_shape(op, in_shapes).as_deref(),
            Ok(out_shape),
            "{}: derivation vs authored golden",
            op.name
        );
        // Then the oracle must accept that same shape and produce it.
        let mut operands: Vec<_> = in_shapes.iter().map(|s| od(s)).collect();
        operands.push(od(out_shape));
        let key = baracuda_kernel_vocab::structure_key(
            cat,
            &operands,
            baracuda_kernel_vocab::ArchSku::Sm89,
        );
        let plan = crate::plan::build_plan(op, &key);
        let inputs: Vec<_> = in_shapes.iter().map(|s| zeros(s)).collect();
        let produced = crate::oracle::evaluate(&plan, &operands, &inputs, &[]);
        assert_eq!(
            produced[0].shape, out_shape,
            "{}: oracle-produced shape vs derivation",
            op.name
        );
    }

    #[test]
    fn oracle_differential_agrees_on_every_supported_variant() {
        use crate::ir::{ReduceStage, SortOrder, reduced};
        use baracuda_kernel_vocab::OpCategory;

        // Elementwise.
        let add = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        assert_oracle_agrees(
            &add,
            OpCategory::BinaryElementwise,
            &[vec![4, 8], vec![4, 8]],
            &[4, 8],
        );

        // Reduction (last-axis, collapse).
        let sum = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        assert_oracle_agrees(&sum, OpCategory::Reduction, &[vec![4, 8]], &[4]);

        // Scan (shape-preserving).
        let cs = OpDef::scan_simple("cumsum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        assert_oracle_agrees(&cs, OpCategory::Scan, &[vec![4, 8]], &[4, 8]);

        // RowReduce (full-width epilogue).
        let stages = vec![ReduceStage {
            pre: input(0).0,
            op: ReduceOp::Sum,
        }];
        let rr = OpDef::row_reduce("rms", 1, &[ElementKind::F32], stages, reduced(0));
        assert_oracle_agrees(&rr, OpCategory::Softmax, &[vec![8, 16]], &[8, 16]);

        // Window (pooled axis downsamples) — the arithmetic case.
        let pool = OpDef::window(
            "maxpool",
            1,
            &[ElementKind::F32],
            ReduceOp::Max,
            1,
            2,
            2,
            1,
            0,
            0,
            true,
            input(0),
            reduced(0),
        );
        assert_oracle_agrees(&pool, OpCategory::Pooling, &[vec![4, 8]], &[4, 4]);

        let _ = SortOrder::Asc; // (RowSort is not oracle-supported; see the coverage note.)
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p baracuda-kernelgen shape::tests::oracle_differential`
Expected: FAIL to compile initially if `TypedBuffer::new` or `oracle::evaluate` are not in scope — add `use crate::oracle;` at the top of the test module if the fully-qualified paths do not resolve.

**If a shape assertion fails**, that is the differential working: recompute the authored golden by hand from the op's semantics before touching `output_shape`. Do not "fix" the derivation to match a wrong golden.

- [ ] **Step 3: Make it pass**

No production change should be needed — Tasks 3–5 already implement every rule this exercises. If a case fails, the fix belongs in whichever `output_shape` arm is wrong (Tasks 3–5), and the failing arm's own unit test should be corrected alongside.

- [ ] **Step 4: Run the whole shape suite**

Run: `cargo test -p baracuda-kernelgen shape::`
Expected: PASS — 16 tests.

- [ ] **Step 5: Run both crates**

Run: `cargo test -p baracuda-kernel-vocab -p baracuda-kernelgen`
Expected: PASS, with no pre-existing test broken.

- [ ] **Step 6: Format, lint, and commit**

```bash
rustfmt --style-edition 2024 crates/baracuda-kernelgen/src/shape.rs
cargo clippy -p baracuda-kernelgen -- -D warnings
git add crates/baracuda-kernelgen/src/shape.rs
git commit -m "test(shape): the CPU-oracle differential for the shape oracle

The spec's §6 validation gate: for each op, run the CPU oracle with the
AUTHORED output operand and assert the shape it actually produced equals
output_shape's derivation. A two-way pin — a wrong derivation OR a wrong
authored golden fails, so the oracle's semantics and our shape rule cannot
drift apart silently.

Covers Elementwise, Reduction, Scan, RowReduce and Window (oracle::evaluate
deliberately does not implement Contraction / RowSort / gather-scatter); those
keep their hand-computed goldens.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TUKx27gRHZfivD4N72euGK"
```

---

## Post-implementation

Tell the human:
- which `Access` variants report `NotDerivable` and why (in-place scatter, TopK `k_out`) — these are honest gaps, not bugs;
- that `Window`/`TopK`/`Im2Col` are marked `NeedsReservedConstructor`, which is the concrete Baracuda-side demand feeding KISS #80 (Fuel cosign posted 2026-07-24);
- that no `contract.rs` change was made (shape-rule emit stays Fuel-evaluator-gated).
