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
    /// A `Div` was `i64::MIN / -1`: the mathematical quotient (`-i64::MIN`)
    /// does not fit in `i64`. The lone input pair for which floor-division
    /// itself would overflow.
    DivOverflow,
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
/// a rank-0 operand, a concrete zero divisor, or the `i64::MIN / -1` overflow
/// case. Never panics.
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
                    if x == i64::MIN && y == -1 {
                        // The only (x, y) pair for which div_euclid itself
                        // would panic: the true quotient (-i64::MIN) overflows.
                        return Err(ShapeDecline::DivOverflow);
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
    fn eval_div_declines_i64_min_over_neg_one_instead_of_panicking() {
        // i64::MIN / -1 overflows i64 (the true quotient is -i64::MIN, which
        // doesn't fit). div_euclid itself panics on this exact pair, so it
        // must be intercepted before the call rather than let through.
        let e = DimExpr::Div(
            Box::new(DimExpr::Const(i64::MIN)),
            Box::new(DimExpr::Const(-1)),
        );
        assert_eq!(eval_dim(&e, &[], &[]), Err(ShapeDecline::DivOverflow));
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
            assert_eq!(
                decode_dim(&encode_dim(&e)),
                Ok(e.clone()),
                "round-trip {e:?}"
            );
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
}
