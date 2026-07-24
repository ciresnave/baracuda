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

use crate::ir::{Access, AxisRole, OpDef, ReadIndex, SortLimit, WriteIndex};
use baracuda_kernel_vocab::{Axis, DimExpr, MAX_RANK, ShapeExpr};

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
    /// A `Window`/`Im2Col` `stride` attribute was `0`. The windowed-extent
    /// formula divides by `stride`, so a zero stride has no valid geometry
    /// (a real pooling/im2col stride is always `>= 1`) — declined rather
    /// than divided.
    InvalidStride {
        /// Human-readable reason naming the op class and offending field.
        reason: &'static str,
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
    // A data-dependent gather rides `read_index`, not an `Access` variant, and
    // its output shape is the data shape with the gathered axis REPLACED by
    // the index operand's shape (§6.20-0008 — the class the oracle most exists
    // to catch; never `SameAs(data)`).
    if let Some((data, index_operand, axis)) =
        op.read_index.iter().enumerate().find_map(|(i, r)| match r {
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
            if *stride == 0 {
                return Err(ShapeError::InvalidStride {
                    reason: "windowed pooling: stride must be >= 1",
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
            if stride.0 == 0 {
                return Err(ShapeError::InvalidStride {
                    reason: "im2col: stride.0 (height stride) must be >= 1",
                });
            }
            if stride.1 == 0 {
                return Err(ShapeError::InvalidStride {
                    reason: "im2col: stride.1 (width stride) must be >= 1",
                });
            }
            let (n, c, h, w) = (in0[0], in0[1], in0[2], in0[3]);
            let h_out = windowed_extent(h, kernel.0, stride.0, dilation.0, pad.0, pad.0);
            let w_out = windowed_extent(w, kernel.1, stride.1, dilation.1, pad.1, pad.1);
            // Saturating: an adversarial channel/extent near i64::MAX saturates
            // rather than overflow-panicking (the never-panic contract).
            Ok(vec![
                n,
                c.saturating_mul(i64::from(kernel.0))
                    .saturating_mul(i64::from(kernel.1)),
                h_out.saturating_mul(w_out),
            ])
        }
        // Every shipped Access variant is now covered, so this arm is presently
        // unreachable — but `Access` grows over time (§6.20 tracks §6.13's
        // vocabulary), and it exists so a NEWLY ADDED variant is an honest miss
        // rather than a wrong shape or a silent compile-time gap.
        #[allow(unreachable_patterns)]
        _ => Err(ShapeError::NotDerivable {
            reason: "access variant not yet covered by the shape oracle",
        }),
    }
}

/// The windowed output extent along one axis — the shared pooling / im2col
/// geometry: `(input + pad_lo + pad_hi − dilation·(size−1) − 1) ÷ stride + 1`
/// with **floor** division, clamped at 0 (a window wider than the padded input
/// yields no output positions, never a negative extent).
///
/// **Saturating** on `input`: an adversarial extent near [`i64::MAX`] saturates
/// rather than overflow-panicking, honoring the module's never-panic contract.
/// (`effective` is bounded by the `u8` window params, so it never overflows.)
#[must_use]
pub fn windowed_extent(
    input: i64,
    size: u8,
    stride: u8,
    dilation: u8,
    pad_lo: u8,
    pad_hi: u8,
) -> i64 {
    let effective = i64::from(dilation) * (i64::from(size) - 1) + 1;
    let padded = input
        .saturating_add(i64::from(pad_lo))
        .saturating_add(i64::from(pad_hi));
    let span = padded.saturating_sub(effective);
    if span < 0 {
        return 0;
    }
    (span / i64::from(stride)).saturating_add(1)
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
    /// A single free dimension expressed as a [`DimExpr`] — the §6.20-0007
    /// slice/iota-offset free case.
    ///
    /// **Currently unconstructed:** no shipped [`Access`] variant is a bare
    /// slice or iota offset (pooling/im2col are multi-axis and map to
    /// [`ShapeRuleForm::NeedsReservedConstructor`] instead), so
    /// [`shape_rule_form`] never returns this today. It is kept as the §6.20
    /// vocabulary slot the bridge will emit the moment such an op ships — the
    /// `DimExpr` machinery ([`pooled_axis_dim_expr`], [`baracuda_kernel_vocab::eval_dim`])
    /// is already built and tested, so wiring a slice/iota op to it is additive.
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
    if op
        .read_index
        .iter()
        .any(|r| !matches!(r, ReadIndex::Direct))
    {
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
        // Every shipped Access variant is now covered, so this arm is presently
        // unreachable — but `Access` grows over time (§6.20 tracks §6.13's
        // vocabulary), and it exists so a NEWLY ADDED variant is an honest miss
        // rather than a wrong classification or a silent compile-time gap.
        #[allow(unreachable_patterns)]
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
    fn windowed_extent_saturates_on_adversarial_extents_never_panics() {
        // The never-panic contract: an input extent near i64::MAX must saturate,
        // not overflow-panic (input + pad, and span/stride + 1). Debug builds
        // trap on overflow — this pins the saturating arithmetic.
        assert_eq!(
            windowed_extent(i64::MAX, 1, 1, 1, 255, 255),
            i64::MAX,
            "input + pad near i64::MAX saturates; span/1 + 1 saturates"
        );
        // im2col's products likewise saturate rather than panic on a huge output.
        let ic = OpDef::im2col_2d("im2col", ElementKind::F32, (2, 2), (1, 1), (0, 0), (1, 1));
        let out = output_shape(&ic, &[vec![1, i64::MAX, 4, 4]]).expect("no panic");
        assert_eq!(out[1], i64::MAX, "c * kh * kw saturates");
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

    #[test]
    fn window_output_declines_zero_stride_instead_of_dividing() {
        use crate::ir::reduced;
        // A stride of 0 is unvalidated input at this layer (OpDef::window has
        // no runtime check enforcing stride >= 1) — the oracle must decline,
        // never divide by it.
        let pool = OpDef::window(
            "maxpool",
            1,
            &[ElementKind::F32],
            ReduceOp::Max,
            1,    // axis
            2,    // size
            0,    // stride
            1,    // dilation
            0,    // pad_lo
            0,    // pad_hi
            true, // count_include_pad
            input(0),
            reduced(0),
        );
        assert!(matches!(
            output_shape(&pool, &[vec![4, 8]]),
            Err(ShapeError::InvalidStride { .. })
        ));
    }

    #[test]
    fn im2col_output_declines_zero_stride_instead_of_dividing() {
        // Same exposure via im2col_2d's stride tuple — either axis being 0
        // must decline, never divide by it.
        let ic_h = OpDef::im2col_2d("im2col", ElementKind::F32, (2, 2), (0, 1), (0, 0), (1, 1));
        assert!(matches!(
            output_shape(&ic_h, &[vec![1, 2, 4, 4]]),
            Err(ShapeError::InvalidStride { .. })
        ));
        let ic_w = OpDef::im2col_2d("im2col", ElementKind::F32, (2, 2), (1, 0), (0, 0), (1, 1));
        assert!(matches!(
            output_shape(&ic_w, &[vec![1, 2, 4, 4]]),
            Err(ShapeError::InvalidStride { .. })
        ));
    }

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
            ShapeRuleForm::Semantic {
                attr: "contraction axis roles (M/N/K)"
            }
        );
        // A reduce's shape rides its reduce_axes attr.
        let red = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        assert_eq!(
            shape_rule_form(&red, &[vec![4, 8]]),
            ShapeRuleForm::Semantic {
                attr: "reduce_axes + keepdim"
            }
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
            ShapeRuleForm::Semantic {
                attr: "gather axis + index operand shape"
            }
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
        use baracuda_kernel_vocab::{DimValue, Extent, eval_dim};
        let e = pooled_axis_dim_expr(0, 1, 2, 2, 1, 0, 0);
        let shape = [Extent::Known(4), Extent::Known(8)];
        let ops: &[&[Extent]] = &[&shape];
        assert_eq!(eval_dim(&e, ops, &[]), Ok(DimValue::Known(4)));
        assert_eq!(windowed_extent(8, 2, 2, 1, 0, 0), 4);
    }

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
        let cs = OpDef::scan_simple(
            "cumsum",
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
        );
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
}
