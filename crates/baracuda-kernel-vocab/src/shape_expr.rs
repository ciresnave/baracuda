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
        let e = DimExpr::Div(Box::new(DimExpr::Const(-7)), Box::new(DimExpr::Const(2)));
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
            Err(ShapeDecline::AxisOutOfRange {
                operand: 0,
                axis: 2,
                rank: 2
            })
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
        let e = DimExpr::Div(Box::new(DimExpr::Const(8)), Box::new(DimExpr::Const(0)));
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
            Ok(vec![
                DimValue::Known(2),
                DimValue::Known(3),
                DimValue::Known(4)
            ])
        );
        assert_eq!(
            eval_shape(&ShapeExpr::SameAs(5), ops),
            Err(ShapeDecline::OperandOutOfRange { operand: 5 })
        );
    }
}
