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
use baracuda_kernel_vocab::MAX_RANK;

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
                // Validate every set bit against the mask's full width, not
                // just the rank-filtered subset below: a bit at/above `rank`
                // must decline with AxisOutOfRange rather than silently drop
                // out of the `0..rank` filter and behave as if unset.
                if let Some(bad) =
                    (0..MAX_RANK as u8).find(|&d| axes.is_set(d) && usize::from(d) >= rank)
                {
                    return Err(ShapeError::AxisOutOfRange {
                        axis: bad as usize,
                        rank,
                    });
                }
                (0..rank).filter(|&d| axes.is_set(d as u8)).collect()
            };
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
    fn reduction_mask_bit_above_rank_declines_axis_out_of_range() {
        // AxisMask(0b100000) names axis 5, but the input is only rank 2 --
        // this must decline typed, not silently behave as an unreduced no-op.
        let bad = OpDef::reduction_axes(
            "sbad",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b100000),
            false,
        );
        assert_eq!(
            output_shape(&bad, &[vec![4, 8]]),
            Err(ShapeError::AxisOutOfRange { axis: 5, rank: 2 })
        );
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

        let sc = OpDef::scan_simple(
            "cumsum",
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
        );
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
