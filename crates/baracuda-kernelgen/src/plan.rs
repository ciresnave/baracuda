//! Language-agnostic kernel plan — the schedule decision.
//!
//! [`build_plan`] turns an [`OpDef`] + a [`StructureKey`] cell into a neutral
//! [`KernelPlan`]: *what* to compute (the op body + dtype) and the *schedule*
//! (vectorized vs scalar) to compute it with. A [`crate::backend::Backend`]
//! lowers the plan to a concrete language. Choosing the schedule here, not in
//! the backend, keeps the decision shared across every backend.

use crate::ir::{
    Access, OpDef, ReadIndex, ReduceOp, ReduceStage, ScalarExpr, View, WriteCombine, WriteIndex,
};
use baracuda_kernels_types::{
    AxisMask, Contiguity, ElementKind, OperandKey, StructureKey, VecWidth, MAX_OPERANDS,
};

/// How the kernel iterates the data — the backend-neutral schedule.
///
/// `#[non_exhaustive]`: strided / broadcast / reduction schedules are the
/// growth path; backends match what they support and reject the rest.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Schedule {
    /// Linear access, `width` elements at a time (e.g. `float4` for width 4).
    Vectorized {
        /// Vector width in elements.
        width: u32,
    },
    /// Linear access, one element at a time. Contiguous operands only.
    Scalar,
    /// Per-element coordinate unravel over the cell's iteration rank — for
    /// non-contiguous operands (strided / broadcast). The emitter specializes
    /// it per cell: the rank is unrolled, broadcast axes drop their offset
    /// terms, and a fully-broadcast operand is hoisted to a loop-invariant load.
    Strided,
    /// One thread per output element; sequential fold over the contiguous trailing
    /// axis. The v1 reduction schedule — block/warp-parallel reduction is the perf
    /// follow-up.
    Reduction {
        /// The associative combine to apply along the axis.
        op: ReduceOp,
        /// Reduce-axis geometry (design-doc predicate #9). All classes lower to
        /// the sequential fold in v2; the class reserves the dispatch token for a
        /// future block-parallel outer-axis kernel. The full axis mask rides on
        /// [`KernelPlan::key`]'s `reduce_axes` (this enum is `Copy`).
        class: ReduceAxisClass,
        /// Keep reduced axes as size-1 (broadcast-back) vs. collapse them.
        keepdim: bool,
    },
    /// Fused reduce → broadcast → elementwise, one block per output row (warp-
    /// shuffle + shared-memory tree reduce): `n_stages` reductions then a full-width
    /// epilogue. The stages + epilogue ride on [`KernelPlan::access`] (this enum is
    /// `Copy`, so a `Vec` can't live here). `block` selects the block-parallel tree
    /// (v1 always `true`) over a sequential fallback.
    RowReduce {
        /// Number of reduction stages (each produces a `Reduced(i)`).
        n_stages: u8,
        /// Block-parallel tree reduce (`true`, v1) vs the sequential fallback.
        block: bool,
    },
    /// Batched contraction (`out[m,n] = epi(Σ_k lhs[m,k]·rhs[k,n])`) — the
    /// terminal ORDER-3 schedule. v1: the skinny SIMT kernel (thread per output
    /// column, M-row register accumulators, coalesced K-streaming of the rhs) —
    /// the decode / flat-GEMM long-tail cell; tiled/MMA schedules join as
    /// bench-gated variants. Axes/accum/epilogue ride on [`KernelPlan::access`].
    Contraction,
    /// **Prefix scan** along a single axis (increment 6) — a full-width cumulative
    /// output, one row per output slot. `block = false` is the serial-fold BASE
    /// (thread 0 walks the axis sequentially — the deterministic bit-reference,
    /// [`crate::backend::VariantFidelity::BitIdentical`]); `block = true` is the
    /// cooperative block-scan VARIANT (Kogge-Stone warp scan + cross-warp carry,
    /// produced by `cuda::scan_blockscan_variant`, never by `build_plan`). The
    /// monoid/axis/flags ride here (this enum is `Copy`); `pre`/`post` ride on
    /// [`KernelPlan::access`].
    Scan {
        /// The associative monoid combine along the axis.
        op: ReduceOp,
        /// The scanned axis (v1: `rank - 1`, innermost/contiguous).
        axis: u8,
        /// Walk the axis descending.
        reverse: bool,
        /// Exclusive (shift-by-one, identity at the first visited position) scan.
        exclusive: bool,
        /// `false` = serial-fold base; `true` = cooperative block-scan variant.
        block: bool,
    },
    /// **Sliding-window reduction** along one axis (increment 7) — the POOLING
    /// family (max_pool / avg_pool / sum_pool / min_pool). One thread per output
    /// element (grid-stride) walks the local window of `size` taps, reduces with
    /// `op`, and stores the downsampled output — [`crate::backend::VariantFidelity::BitIdentical`]
    /// (each output is an independent fixed-order fold; no cross-output
    /// dependence, unlike [`Schedule::Scan`]). The window geometry rides here
    /// (this enum is `Copy`); `pre`/`post` ride on [`KernelPlan::access`].
    Window {
        /// The window combine (`Max`/`Min`/`Sum`/`Mean`).
        op: ReduceOp,
        /// The pooled axis (v1: `rank - 1`, innermost/contiguous).
        axis: u8,
        /// Window length in taps.
        size: u8,
        /// Output downsampling stride.
        stride: u8,
        /// Inter-tap dilation.
        dilation: u8,
        /// Low-side padding.
        pad_lo: u8,
        /// High-side padding.
        pad_hi: u8,
        /// Mean divisor policy: `size` (`true`) vs. valid-tap count (`false`).
        count_include_pad: bool,
    },
}

/// Reduce-axis geometry (design-doc predicate #9). All classes lower to the same
/// sequential fold in v2; the class reserves the dispatch token so a later
/// block-parallel outer-axis kernel is an additive drop-in, not a re-key.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReduceAxisClass {
    /// Empty mask (legacy last-axis sentinel) or a single contiguous trailing
    /// axis — today's sequential fast path.
    InnerContig,
    /// A single outermost (axis 0) reduced axis.
    Outer,
    /// A single interior reduced axis (not axis 0, not a contiguous trailing axis).
    Middle,
    /// Two or more reduced axes.
    Multi,
}

/// A language-agnostic description of the kernel to emit.
#[derive(Clone, Debug)]
pub struct KernelPlan<'a> {
    /// Op name (used to form the generated symbol).
    pub op_name: &'a str,
    /// Number of input operands.
    pub n_inputs: u8,
    /// Element dtype.
    pub dtype: ElementKind,
    /// Output element dtype — the *resolved* [`crate::ir::OpDef::out_dtype`]
    /// (`out_dtype.unwrap_or(key.dtype)`), so `out_dtype == dtype` for every
    /// uniform-dtype op and `U8` only for a validated predicate op. Backends
    /// read this for the output pointer type + store conversion.
    pub out_dtype: ElementKind,
    /// The chosen schedule.
    pub schedule: Schedule,
    /// The structure cell this plan targets. Backends read structural detail
    /// from it (rank, per-operand broadcast mask, flip) for strided lowering,
    /// and its token for traceability.
    pub key: &'a StructureKey,
    /// Output `= body`, evaluated per coordinate. For [`Schedule::RowReduce`] this
    /// is the epilogue (`OpDef::row_reduce` sets `body = epilogue`). For a
    /// multi-output op this is output 0; the further outputs are in
    /// [`Self::extra_out_bodies`].
    pub body: &'a ScalarExpr,
    /// Number of outputs the kernel writes (`1` for every single-output op —
    /// `OpDef::n_outputs`). `> 1` only for a validated multi-output
    /// `Access::Elementwise` op (increment 1); the emitter then writes the last
    /// `n_outputs` operands of the key from one shared body-DAG.
    pub n_outputs: u8,
    /// Additional output bodies (`OpDef::extra_out_bodies`) — **empty for every
    /// single-output op**, so the single-output emitters are byte-identical. The
    /// multi-output emitter interns `[body] ++ extra_out_bodies` into one
    /// [`crate::ir::ExprDag`] for cross-body CSE.
    pub extra_out_bodies: &'a [ScalarExpr],
    /// The op's access pattern — the [`Schedule::RowReduce`] emitter reads its
    /// `stages` (and epilogue) off here, since `Schedule` is `Copy` and can't carry
    /// the stage `Vec`.
    pub access: &'a Access,
    /// Per-input layout [`crate::ir::View`]s (item 01) — index `i` ↔ `Input(i)`.
    /// **Empty for every view-free op** (every pre-item-01 constructor), so the
    /// strided emitter's per-operand offset is byte-identical; a non-empty slice
    /// has length `n_inputs`. Only a [`crate::ir::View::Permute`] entry changes
    /// emission (the stride-index remap in `cuda::offset_expr`); `Identity` /
    /// same-rank `Reshape` read at the iteration coordinate, and `Broadcast` is a
    /// key-driven validation-only declaration in v1. Validated at the top of
    /// [`build_plan`] ([`assert_valid_views`]) with an independent emitter backstop
    /// in [`crate::cuda::Cuda::lower`].
    pub views: &'a [crate::ir::View],
    /// Per-input **data-dependent read roles** ([`crate::ir::ReadIndex`], increment
    /// 4, GATHER) — index `i` ↔ `Input(i)`. **Empty for every index-free op**
    /// (every pre-increment-4 constructor), so the strided emitter's per-operand
    /// offset is byte-identical; a non-empty slice has length `n_inputs`. Only a
    /// [`crate::ir::ReadIndex::Indexed`] entry changes emission (the axis
    /// value-substitution in `cuda::emit_strided`); `Direct` reads at the
    /// iteration coordinate. Validated at the top of [`build_plan`]
    /// ([`assert_valid_gather`]) with an independent emitter backstop in
    /// [`crate::cuda::Cuda::lower`].
    pub read_index: &'a [crate::ir::ReadIndex],
    /// The output's **data-dependent write role** ([`crate::ir::WriteIndex`],
    /// increment 5, SCATTER) — the write-side mirror of [`Self::read_index`].
    /// [`crate::ir::WriteIndex::Direct`] for every non-scatter op (byte-identical
    /// output offset); a [`crate::ir::WriteIndex::ScatterIndexed`] role
    /// substitutes a runtime index value for one OUTPUT-axis coordinate and turns
    /// the store into a [`crate::ir::WriteCombine`] op. Validated at the top of
    /// [`build_plan`] ([`assert_valid_scatter`]) with an independent emitter
    /// backstop in [`crate::cuda::Cuda::lower`].
    pub write_index: &'a crate::ir::WriteIndex,
}

impl KernelPlan<'_> {
    /// All output bodies in order — `body` (output 0) then `extra_out_bodies`.
    /// One element for a single-output plan; the multi-output emitter interns
    /// these together for cross-body CSE, and the backstop walks gate every one.
    #[must_use]
    pub fn output_bodies(&self) -> Vec<&ScalarExpr> {
        std::iter::once(self.body)
            .chain(self.extra_out_bodies.iter())
            .collect()
    }
}

/// Choose the schedule for `op` at structure cell `key` and return a neutral
/// [`KernelPlan`].
///
/// Elementwise ops vectorize when every operand is `Contig` + `V4`, scalar/strided
/// otherwise. A reduction op maps straight to [`Schedule::Reduction`] (the fold is
/// the schedule). (Whether a backend can lower the chosen dtype is the backend's
/// call, not this function's.)
#[must_use]
pub fn build_plan<'a>(op: &'a OpDef, key: &'a StructureKey) -> KernelPlan<'a> {
    assert_valid_out_dtype(op);
    assert_valid_multi_output(op, key);
    assert_no_half_nextafter(op, key.dtype);
    assert_int_op_admissibility(op, key.dtype);
    assert_coord_admissibility(op, key);
    assert_valid_reduction_post(op);
    assert_valid_views(op, key);
    assert_valid_gather(op, key);
    assert_valid_scatter(op, key);
    let schedule = match op.access {
        Access::Reduction {
            op: rop,
            axes,
            keepdim,
            post: _,
        } => {
            // `class`/`keepdim` are consumed by the emitter in step 3; today all
            // classes lower to the same sequential fold, so the legacy last-axis
            // path (empty mask ⇒ `InnerContig`) stays byte-identical.
            let input0_contig =
                key.n_operands > 0 && key.operands[0].contig == Contiguity::Contig;
            Schedule::Reduction {
                op: rop,
                class: classify_reduce_axes(axes, key.rank, input0_contig),
                keepdim,
            }
        }
        // `ref` borrows (the Vec/expr can't move out of the borrowed `op.access`);
        // v1 always routes RowReduce to the block-parallel tree reduce.
        Access::RowReduce {
            ref stages,
            ref epilogue,
        } => {
            validate_row_reduce(stages, epilogue, op.n_inputs, key);
            Schedule::RowReduce {
                n_stages: stages.len() as u8,
                block: true,
            }
        }
        Access::Contraction { ref axes, ref epilogue, .. } => {
            // v1 admissibility: the canonical rank-2 dense matmul cell, keyed
            // with contraction facts, and an epilogue over the K-sum only.
            assert_eq!(
                (axes.lhs.as_slice(), axes.rhs.as_slice()),
                (
                    crate::ir::ContractionAxes::matmul().lhs.as_slice(),
                    crate::ir::ContractionAxes::matmul().rhs.as_slice()
                ),
                "contraction v1: canonical rank-2 matmul axis roles only"
            );
            assert!(
                key.contraction.is_some(),
                "contraction cell must carry ContractionKey facts (rank-2 dense \
                 row-major [M,K]x[K,N]->[M,N]); got token {}",
                key.to_token()
            );
            assert!(
                epilogue_reads_only_reduced0(epilogue),
                "contraction v1: epilogue over Reduced(0) only (fused bias inputs \
                 are a follow-up)"
            );
            Schedule::Contraction
        }
        // Increment 6 SCAN: validate admissibility (mirrors the RowReduce arm's
        // `validate_row_reduce` call), then derive the serial-fold BASE schedule
        // (`block: false`). The cooperative block-scan is produced separately by
        // `cuda::scan_blockscan_variant` (a `lower_variants` filter), never here.
        Access::Scan {
            op: sop,
            axis,
            reverse,
            exclusive,
            ..
        } => {
            validate_scan(op, key, axis, reverse, exclusive);
            Schedule::Scan {
                op: sop,
                axis,
                reverse,
                exclusive,
                block: false,
            }
        }
        // Increment 7 WINDOW: validate admissibility (mirrors the Scan arm's
        // `validate_scan` call), then derive the one-thread-per-output pooling
        // schedule. The window geometry is Copy, so it rides the schedule; `pre`/
        // `post` ride on `KernelPlan::access` (Scan/RowReduce precedent).
        Access::Window {
            op: wop,
            axis,
            size,
            stride,
            dilation,
            pad_lo,
            pad_hi,
            count_include_pad,
            ..
        } => {
            validate_window(op, key, axis, size, stride, dilation, pad_lo, pad_hi);
            Schedule::Window {
                op: wop,
                axis,
                size,
                stride,
                dilation,
                pad_lo,
                pad_hi,
                count_include_pad,
            }
        }
        Access::Elementwise => {
            let n = key.n_operands as usize;
            let all_contig =
                n > 0 && (0..n).all(|k| key.operands[k].contig == Contiguity::Contig);
            // The kernel vectorizes at the *narrowest* width every operand supports.
            let min_width = (0..n)
                .map(|k| vec_width_elems(key.operands[k].vec_width))
                .min()
                .unwrap_or(1);
            if op_has_gather(op) || op_has_scatter(op) {
                // Increment 4/5: a GATHERED input (read) or a SCATTERED output
                // (write) resolves a DATA-DEPENDENT address (one axis coordinate
                // is a runtime index value), which only the strided emitter folds
                // (it substitutes `idx·stride[axis]` for `c{axis}·stride[axis]`, on
                // the input side for gather / the output side for scatter). NEVER
                // vectorized/packed/scalar — a data-dependent address cannot
                // coalesce into a vector load/store, and the vector/packed emitters
                // iterate a bare linear index that would ignore the index operand
                // entirely. Pinned by the force-strided tests + the independent
                // `assert_gather_lowerable` / `assert_scatter_lowerable` backstops.
                // Index-free / write-Direct ops never reach here, so emission stays
                // byte-identical.
                Schedule::Strided
            } else if op_has_addressing_view(op) {
                // Item 01: a viewed INPUT (a `Permute`/transpose or a `Broadcast`
                // read-through) reads the producer through a layout change, which
                // only the strided emitter folds into address math (`offset_expr`
                // remaps `c{d}·stride[perm[d]]`). NEVER vectorized/packed/scalar —
                // a transposed read is non-contiguous, and the vector/packed
                // emitters iterate a bare linear index that would ignore the view
                // (silently reading the un-transposed operand). `Identity` and a
                // same-rank `Reshape` (identity linear map) are NOT addressing
                // views, so a view-free or all-identity op is unaffected here —
                // byte-identical. Pinned by the vectorize-never view test in
                // `cuda` and the independent `assert_views_lowerable` backstop.
                Schedule::Strided
            } else if expr_contains_coord(&op.body) {
                // A Coord body always takes the STRIDED schedule (increment
                // 0d): the strided emitter is the only one that materializes
                // the per-axis output coordinates `c{d}` a Coord leaf reads —
                // the Vectorized/Scalar emitters iterate a bare linear index.
                // Contiguous cells are still CORRECT under strided emission
                // (the unravel + stride dot-product reproduces the linear
                // offset exactly), just unoptimized — a coordinate-aware
                // vectorized variant is a follow-up. Pinned by the
                // vectorize-never test in `cuda`.
                Schedule::Strided
            } else if !all_contig {
                Schedule::Strided
            } else if min_width >= 2 && op.out_dtype.is_none() {
                // A hetero-output (u8 predicate) kernel takes the SCALAR path
                // in v1 — never Vectorized: the vector/packed emitters load and
                // STORE one vector type, and a u8-mask output has no packed
                // store (a contiguous u8 output even keys V8, which would
                // otherwise widen `min_width` past the inputs'). Pinned by the
                // packed-fallback golden in `cuda`.
                Schedule::Vectorized { width: min_width }
            } else {
                Schedule::Scalar
            }
        }
    };
    KernelPlan {
        op_name: &op.name,
        n_inputs: op.n_inputs,
        dtype: key.dtype,
        out_dtype: op.out_dtype.unwrap_or(key.dtype),
        schedule,
        key,
        body: &op.body,
        n_outputs: op.n_outputs(),
        extra_out_bodies: &op.extra_out_bodies,
        access: &op.access,
        views: &op.views,
        read_index: &op.read_index,
        write_index: &op.write_index,
    }
}

/// Classify a reduction's axis geometry (design-doc predicate #9) from its
/// reduced-axis mask + the input's contiguity. An empty mask is the legacy
/// last-axis sentinel; a single trailing axis over a contiguous input is the
/// existing fast path. All classes lower to the same sequential fold in v2 — the
/// class only reserves the dispatch token for a future block-parallel kernel.
fn classify_reduce_axes(axes: AxisMask, rank: u8, input0_contig: bool) -> ReduceAxisClass {
    match axes.0.count_ones() {
        0 => ReduceAxisClass::InnerContig, // empty ⇒ legacy last-axis default
        1 => {
            let d = axes.0.trailing_zeros() as u8;
            if rank > 0 && d == rank - 1 && input0_contig {
                ReduceAxisClass::InnerContig // contiguous trailing axis = fast path
            } else if d == 0 {
                ReduceAxisClass::Outer
            } else {
                ReduceAxisClass::Middle
            }
        }
        _ => ReduceAxisClass::Multi,
    }
}

#[cfg(test)]
mod reduce_class_tests {
    use super::*;

    #[test]
    fn classify_axis_geometry() {
        // empty mask ⇒ legacy last-axis fast path
        assert_eq!(
            classify_reduce_axes(AxisMask::EMPTY, 3, true),
            ReduceAxisClass::InnerContig
        );
        // trailing axis (rank-1) over a contiguous input ⇒ fast path
        assert_eq!(
            classify_reduce_axes(AxisMask(0b100), 3, true),
            ReduceAxisClass::InnerContig
        );
        // outermost axis 0
        assert_eq!(
            classify_reduce_axes(AxisMask(0b001), 3, true),
            ReduceAxisClass::Outer
        );
        // interior axis 1
        assert_eq!(
            classify_reduce_axes(AxisMask(0b010), 3, true),
            ReduceAxisClass::Middle
        );
        // two reduced axes
        assert_eq!(
            classify_reduce_axes(AxisMask(0b011), 3, true),
            ReduceAxisClass::Multi
        );
        // trailing axis but a STRIDED input ⇒ no longer the contiguous fast path
        assert_eq!(
            classify_reduce_axes(AxisMask(0b100), 3, false),
            ReduceAxisClass::Middle
        );
    }
}

/// `true` if `e` references no leaf other than `Reduced(0)` and constants — the
/// contraction-v1 epilogue admissibility (no `Input`/`Param`, no other stage).
fn epilogue_reads_only_reduced0(e: &crate::ir::ScalarExpr) -> bool {
    use crate::ir::ScalarExpr as E;
    match e {
        E::Reduced(0) | E::Const(_) => true,
        // Coord rejects here too: a contraction epilogue iterates the (m, n)
        // output space, not an elementwise cell's — Coord's v1 semantics are
        // Elementwise-only (`assert_coord_admissibility` fires first with the
        // targeted message; this arm keeps the predicate honest regardless).
        E::Input(_) | E::Param(_) | E::Reduced(_) | E::Coord(_) => false,
        E::Unary(_, x) => epilogue_reads_only_reduced0(x),
        E::Add(a, b) | E::Sub(a, b) | E::Mul(a, b) | E::Div(a, b) | E::Binary(_, a, b) => {
            epilogue_reads_only_reduced0(a) && epilogue_reads_only_reduced0(b)
        }
    }
}

/// The access role of a [`Access::RowReduce`] input operand, from its layout.
///
/// The three roles are the three broadcast geometries a row-reduce operand can
/// take, and they map one-to-one onto the emitter's load index (`last` = the
/// feature/reduced axis, `rank-1`):
///
/// | role | broadcast mask | varies along | index |
/// |---|---|---|---|
/// | [`RowStreamed`](RrRole::RowStreamed) | empty | row **and** feature | `in_i[base+j]` |
/// | [`ColBroadcast`](RrRole::ColBroadcast) | every outer axis, **not** `last` | feature only | `in_i[j]` |
/// | [`RowScalar`](RrRole::RowScalar) | `last` set, **no** outer axis | row only | `in_i[row]` (hoisted) |
///
/// `RowScalar` is the exact inverse of `ColBroadcast`: a `ColBroadcast` weight is
/// constant *across rows* and varies *along the feature axis*; a `RowScalar`
/// (a saved per-row statistic — μ, rstd, lse) is constant *along the feature
/// axis* and varies *across rows*.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum RrRole {
    /// A reduced/streamed tensor ([n_out, k], full / empty bcast) — `in_i[base+j]`.
    /// Input 0 (`x`) is always this; a second streamed input (softmax-bw's `dy`
    /// beside `y`) is now also legal (the increment-2 lift).
    RowStreamed,
    /// A per-column `[k]` weight/bias, broadcast over the row axis — `in_i[j]`.
    ColBroadcast,
    /// A per-row scalar (saved stat), broadcast over the feature axis —
    /// `in_i[row]`, loaded **once** per row (hoisted outside the feature loop).
    RowScalar,
}

/// Classify a RowReduce input by its broadcast mask, given the feature axis
/// `last` (`rank-1`). **Total / non-panicking** — the emitter calls this for the
/// load index and must never crash; all *rejection* of malformed shapes lives in
/// [`validate_row_reduce`] (one source of truth, no drift). The three-way split:
///
/// - empty bcast ⇒ [`RrRole::RowStreamed`] (the reduced/streamed tensor);
/// - `last` axis broadcast ⇒ [`RrRole::RowScalar`] (constant along the feature
///   axis ⇒ one value per row). An all-broadcast operand also lands here (last is
///   set) and is then *rejected* by validate's outer-axis-clear check — a true
///   scalar is a `Const`, not an operand;
/// - otherwise (some axis broadcast, but not `last`) ⇒ [`RrRole::ColBroadcast`]
///   (varies along the feature axis ⇒ a per-column weight/bias).
///
/// Which *specific* broadcast masks are legal for each role (every outer axis for
/// a column, no outer axis for a row-scalar, contiguity for a streamed input) is
/// validate's job, so the classification here is deliberately coarse and total.
pub(crate) fn rr_role(o: OperandKey, last: u8) -> RrRole {
    if o.bcast.is_empty() {
        RrRole::RowStreamed
    } else if o.bcast.is_set(last) {
        RrRole::RowScalar
    } else {
        RrRole::ColBroadcast
    }
}

/// Validate a [`Access::RowReduce`] op at build time (AOT — RowReduce never crosses
/// the JIT trust boundary, so a panic here is an author-error backstop, like
/// `emit_reduction`'s asserts). Catches expression errors (a `Reduced(s)` not yet
/// produced, out-of-range `Input`, a `Param`, a non-finite `Const`, a column input
/// inside a reduction stage) **and** operand-layout errors that would mis-index or
/// read out of bounds. Input 0 (`x`) must be row-streamed + contiguous. Each other
/// input takes one of three [`RrRole`]s, classified by broadcast mask ([`rr_role`])
/// and validated for the load index the emitter uses:
///
/// - **`RowStreamed`** (empty bcast, `in_i[base+j]`) — a second reduced/streamed
///   tensor (softmax-bw's `dy` beside `y`); must be contiguous. This is the
///   increment-2 lift of the former "inputs>0 must be column-broadcast" guard.
/// - **`ColBroadcast`** (every outer axis bcast, `last` not, `in_i[j]`) — a
///   per-column `[k]` weight/bias; not reversed; rank ≥ 2.
/// - **`RowScalar`** (`last` bcast, no outer axis, `in_i[row]`) — a saved per-row
///   scalar (μ, rstd, lse), the inverse of `ColBroadcast`; not reversed; rank ≥ 2.
///   An all-broadcast operand (a true scalar → `Const`) is rejected here.
///
/// The output is full-width contiguous.
///
/// v1 assumes a **uniform operand dtype** (the structure key carries one dtype) — a
/// mixed-dtype LayerNorm (fp16 `x` + fp32 weight) is unrepresentable here and must
/// be refused upstream by the caller.
///
/// **Caller pre-conditions this cannot check** (the structure key carries broadcast
/// masks but **no numeric extents** — specialize on structure, not extents), each at
/// the same trust level as the `n_out`/`k` launch args, asserted by the layer still
/// holding the `OperandDesc` extents (an AOT op author, or the live seam caller once
/// `region_to_op` wires RowReduce):
/// - a `ColBroadcast` weight's feature-axis extent must equal `x`'s `k` (else the
///   emitter reads `in_i[j]` past its buffer — a confirmed on-device OOB);
/// - a second `RowStreamed` input must be full `[n_out,k]` dense (else `in_i[base+j]`
///   over-reads — the identical trust as input 0; a bare rank-1 `[k]` has the same
///   key as a full `[n_out,k]`, so this can no longer be a key-visible rejection);
/// - a `RowScalar` must be `[n_out]`-shaped with a dense outer layout so its linear
///   offset equals `row` (else `in_i[row]` mis-indexes).
///
/// Nextafter is declared f32/f64-only at the IR level (its half lowering via
/// promote-to-f32 would step the f32 lattice — ~2^13 steps inside one half
/// step, so the demote rounds straight back: a silently wrong no-op). The
/// CUDA emitter's `cuda_binary` panic only guards the elementwise path; the
/// reduction pre-body, RowReduce stages/epilogue, and contraction epilogues
/// lower through accumulator-width helpers that never pass through it. This
/// plan-level walk covers EVERY Access arm, so no lowering path — present or
/// future backend — can bypass the honest miss.
/// Increment-1 **multi-output** admissibility gate — runs at the top of
/// [`build_plan`] (the house pattern), with an independent emitter backstop in
/// [`crate::cuda::Cuda::lower`]. A single-output op (`extra_out_bodies` empty,
/// every pre-increment-1 op) returns immediately, so nothing about the
/// established path changes and emission stays byte-identical. For a multi-output
/// op (built only by [`OpDef::elementwise_multi`]) the v1 rules, all honest
/// AOT panics:
///
/// 1. **Access**: `Access::Elementwise` only. Multi-output is meaningful only for
///    an elementwise map; the reduction-class arms reject `extra_out_bodies`
///    (a fused reduction/contraction stores one accumulator, not N bodies).
/// 2. **Uniform dtype**: `out_dtype == None`. All outputs share the key dtype;
///    hetero multi-output (a u8 mask beside a float grad — dropout fw) is the
///    follow-up.
/// 3. **Operand budget**: `1 ≤ n_outputs` and `n_inputs + n_outputs ≤
///    MAX_OPERANDS`, and the key must carry exactly `n_inputs + n_outputs`
///    operands (inputs then outputs) — the caller's `OperandDesc` list.
/// 4. **Body legality** (every output body): `Input(i) < n_inputs`; NO `Reduced`
///    (there is no reduction here); NO `Coord` (Elementwise-map only in v1 — a
///    multi-output coordinate kernel is deferred, same rejection the Coord gate
///    would give); `Const` finite. `Param` f32-only is enforced by the emitter's
///    param assert over all output bodies (same rule as the single-output path).
///    (This gate rejecting `Coord` here is also what lets the downstream
///    `assert_coord_admissibility` keep its `Access::Elementwise => {}` arm — a
///    multi-output `Coord` never reaches it.)
/// 5. **Output operands**: each of the last `n_outputs` operands must be
///    **non-broadcast** (a stride-0 output would alias its own writes across
///    iteration coordinates — a write race, and not the full output shape) and
///    **not flipped**. This is the key-visible slice of "outputs must not alias
///    and must match the output shape".
///
/// **Caller preconditions the key cannot see** (documented honestly, the same
/// trust level as the RowReduce `n_out`/`k` and `Coord` extent preconditions):
/// true buffer aliasing — an output buffer pointer equal to an input's (in-place)
/// — and exact per-output extent agreement are abstracted away by the structure
/// key (buffer identity and numeric extents are not keyed). The AOT op author (or
/// a future seam caller, once a multi-output region envelope exists) must ensure
/// distinct, correctly-shaped output buffers; v1 defers in-place entirely.
fn assert_valid_multi_output(op: &OpDef, key: &StructureKey) {
    if op.extra_out_bodies.is_empty() {
        return; // single-output — the established path, unchanged.
    }
    let name = &op.name;
    assert!(
        matches!(op.access, Access::Elementwise),
        "OpDef '{name}': multi-output is Access::Elementwise-only in v1 — a fused \
         reduction/contraction stores a single accumulator, not N bodies; \
         extra_out_bodies is rejected on a {}-class op",
        access_tag(&op.access)
    );
    assert!(
        op.out_dtype.is_none(),
        "OpDef '{name}': multi-output requires a uniform output dtype (out_dtype \
         None) in v1 — a hetero multi-output (u8 mask beside a float grad) is the \
         follow-up"
    );
    let n_inputs = op.n_inputs as usize;
    let n_outputs = op.n_outputs() as usize;
    assert!(
        n_inputs + n_outputs <= MAX_OPERANDS,
        "OpDef '{name}': n_inputs ({n_inputs}) + n_outputs ({n_outputs}) exceeds \
         MAX_OPERANDS ({MAX_OPERANDS})"
    );
    assert!(
        key.n_operands as usize == n_inputs + n_outputs,
        "OpDef '{name}': multi-output key must carry n_inputs+n_outputs operands \
         (inputs then outputs) = {}, got {} — the caller's OperandDesc list is a \
         shape mismatch",
        n_inputs + n_outputs,
        key.n_operands
    );

    // Body legality — walk every output body.
    fn check_body(e: &ScalarExpr, n_inputs: u8, name: &str) {
        match e {
            ScalarExpr::Input(i) => assert!(
                *i < n_inputs,
                "OpDef '{name}': multi-output body Input({i}) >= n_inputs {n_inputs}"
            ),
            ScalarExpr::Const(v) => assert!(
                v.is_finite(),
                "OpDef '{name}': multi-output body Const must be finite, got {v}"
            ),
            ScalarExpr::Param(_) => {}
            ScalarExpr::Reduced(s) => panic!(
                "OpDef '{name}': multi-output body must not read Reduced({s}) — there \
                 is no reduction in an Elementwise multi-output op"
            ),
            ScalarExpr::Coord(d) => panic!(
                "OpDef '{name}': multi-output body must not read Coord({d}) — v1 is \
                 elementwise-map only (a multi-output coordinate kernel is deferred)"
            ),
            ScalarExpr::Unary(_, x) => check_body(x, n_inputs, name),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check_body(a, n_inputs, name);
                check_body(b, n_inputs, name);
            }
        }
    }
    for e in op.output_bodies() {
        check_body(e, op.n_inputs, name);
    }

    // Output operands: the last n_outputs entries. A writable output must not be
    // broadcast (stride-0 → aliased writes) or flipped.
    for j in 0..n_outputs {
        let o = key.operands[n_inputs + j];
        assert!(
            o.bcast.is_empty(),
            "OpDef '{name}': multi-output output {j} is broadcast (mask {:#04x}) — a \
             stride-0 output aliases its own writes across iteration coordinates \
             (a write race) and is not the full output shape",
            o.bcast.0
        );
        assert!(
            !o.flipped,
            "OpDef '{name}': multi-output output {j} is flipped (negative stride) — \
             a reversed output view is deferred"
        );
    }
}

/// Short tag for an [`Access`] variant, for the multi-output rejection message.
fn access_tag(a: &Access) -> &'static str {
    match a {
        Access::Elementwise => "Elementwise",
        Access::Reduction { .. } => "Reduction",
        Access::RowReduce { .. } => "RowReduce",
        Access::Contraction { .. } => "Contraction",
        Access::Scan { .. } => "Scan",
        Access::Window { .. } => "Window",
    }
}

/// `true` if `v` is an **address-affecting** view — one that changes which
/// producer element the strided emitter reads at each iteration coordinate, and
/// therefore forces the [`Schedule::Strided`] schedule and cannot be
/// vectorized/packed. In v1 that is [`View::Permute`] (a transposed read, offset
/// remap `c{d}·stride[perm[d]]`) and [`View::Broadcast`] (stride-0 axes — already
/// non-contiguous). [`View::Identity`] and a same-rank [`View::Reshape`] (an
/// identity linear-index map) are NOT addressing: they read at the iteration
/// coordinate exactly like a view-free operand, so an all-identity op stays
/// byte-identical to a view-free one.
pub(crate) fn view_is_addressing(v: &View) -> bool {
    matches!(v, View::Permute { .. } | View::Broadcast { .. })
}

/// `true` if any of `op`'s per-input views is address-affecting
/// ([`view_is_addressing`]). `false` for a view-free op (empty `views`) and for
/// an all-`Identity`/same-rank-`Reshape` op — the byte-identical cases.
pub(crate) fn op_has_addressing_view(op: &OpDef) -> bool {
    op.views.iter().any(view_is_addressing)
}

/// Item-01 **layout-view** admissibility gate — runs at the TOP of
/// [`build_plan`] (the house pattern), with an independent emitter backstop in
/// [`crate::cuda::assert_views_lowerable`]. A view-free op (empty `views`, every
/// pre-item-01 constructor) returns immediately, and an all-[`View::Identity`] op
/// returns after the length check — so the established path is unchanged and
/// emission stays byte-identical. For an op carrying a real (non-`Identity`)
/// view, the v1 rules, all honest AOT panics (an author/generator error — views
/// never cross the JIT trust boundary, so a panic is the backstop, not a silent
/// wrong-bind):
///
/// 1. **Shape**: `views.len() == n_inputs` (index `i` ↔ `Input(i)`).
/// 2. **Access**: a non-`Identity` view is [`Access::Elementwise`]-only in v1.
///    A reduction/row-reduce/contraction op has its OWN axis machinery (reduced
///    axes, K-contraction, feature broadcast) that a per-input read-through would
///    double-count; those reject (pass through only a trivially-`Identity` view).
/// 3. **Single-output**: a viewed input on a multi-output op is a deferred
///    composition in v1 (the multi-store DAG × the stride remap is unproven) —
///    reject. A viewed single-output op is the whole item-01 surface.
/// 4. **Validity** (every view): [`View::is_valid`] against `key.rank` (a
///    `Permute` must be a true permutation of `0..rank`).
/// 5. **`Permute` ⊥ `Broadcast`**: a permuted input's operand key must have an
///    EMPTY broadcast mask — v1 keeps the transpose remap and the stride-0
///    broadcast orthogonal (a permuted-and-broadcast operand is deferred). The
///    offset remap `c{d}·stride[perm[d]]` then folds cleanly with no per-axis
///    broadcast-skip interaction.
/// 6. **`Broadcast` agreement**: the view's declared `bcast` axes must be a
///    SUBSET of the operand key's broadcast mask. Emission is key-driven (the
///    strided emitter reads `OperandKey::bcast`), so the view is the *named*
///    form of what the key already encodes (per the `View::Broadcast` doc) — a
///    view claiming a broadcast the key doesn't have would be a silent lie the
///    emitter ignores. Validate-only in v1: it changes no address math.
/// 7. **`Reshape` scope**: v1 accepts only a `producer_rank == key.rank`
///    (same-rank, identity linear-index map) reshape, carried for
///    recognition/keying and emitted as identity address math. A rank-change
///    reshape is genuine rank-change emit (items 03/10) and rejects here.
fn assert_valid_views(op: &OpDef, key: &StructureKey) {
    if op.views.is_empty() {
        return; // view-free — every pre-item-01 op, unchanged.
    }
    let name = &op.name;
    assert_eq!(
        op.views.len(),
        op.n_inputs as usize,
        "OpDef '{name}': views.len() ({}) must equal n_inputs ({})",
        op.views.len(),
        op.n_inputs
    );
    if op.views.iter().all(View::is_identity) {
        return; // all-Identity == view-free: byte-identical emission, no gate.
    }
    // From here at least one real (address- or recognition-bearing) view.
    assert!(
        matches!(op.access, Access::Elementwise),
        "OpDef '{name}': a non-Identity View is Access::Elementwise-only in v1 — a \
         {}-class op has its own axis machinery (reduced/contracted/feature axes) \
         that a per-input read-through would double-count; a view on it must be \
         Identity",
        access_tag(&op.access)
    );
    assert!(
        op.n_outputs() == 1,
        "OpDef '{name}': a viewed input on a multi-output op ({} outputs) is a \
         deferred composition in v1 (the multi-store DAG × the per-operand stride \
         remap is unproven) — miss honestly",
        op.n_outputs()
    );
    let rank = key.rank;
    for (i, v) in op.views.iter().enumerate() {
        assert!(
            v.is_valid(rank),
            "OpDef '{name}': input {i} view {v:?} is invalid for iteration rank \
             {rank} (a Permute must be a true permutation of 0..rank)"
        );
        // Input operands are the first `n_inputs` key operands (inputs then
        // outputs). The array is fixed [OperandKey; MAX_OPERANDS], so indexing a
        // valid input slot never panics; a smaller n_operands reads a default
        // (empty-broadcast) key, which the checks below treat conservatively.
        let o = key.operands[i];
        match v {
            View::Identity => {}
            View::Permute { .. } => {
                assert!(
                    o.bcast.is_empty(),
                    "OpDef '{name}': input {i} has a Permute view AND a broadcast \
                     mask ({:#04x}) — v1 keeps the transpose remap and stride-0 \
                     broadcast orthogonal (a permuted-and-broadcast operand is \
                     deferred)",
                    o.bcast.0
                );
            }
            View::Broadcast { bcast } => {
                assert!(
                    bcast.0 & !o.bcast.0 == 0,
                    "OpDef '{name}': input {i} Broadcast view declares axes \
                     ({:#04x}) the operand key does not broadcast ({:#04x}) — the \
                     key drives address math, so the named view must agree (a \
                     view-only broadcast the emitter ignores would be a silent lie)",
                    bcast.0,
                    o.bcast.0
                );
            }
            View::Reshape { producer_rank } => {
                assert!(
                    *producer_rank == rank,
                    "OpDef '{name}': input {i} Reshape view producer_rank \
                     ({producer_rank}) != iteration rank ({rank}) — a rank-change \
                     reshape is genuine rank-change emit (items 03/10), out of \
                     item-01 scope; v1 accepts only a same-rank (identity \
                     linear-map) reshape"
                );
            }
        }
    }
}

/// The single **gathered** input of `read_index` (increment 4), or `None` for an
/// index-free / all-[`ReadIndex::Direct`] op. Returns the gathered input's slot
/// plus its [`ReadIndex::Indexed`] fields `(gathered_input, index_operand, axis,
/// oob, index_dtype)`. v1 admits **at most one** gathered input (the plan gate
/// `assert_valid_gather` rejects more), so the first match is the only one; the
/// emitter and its backstop read the gather off this one accessor to stay in
/// lockstep.
pub(crate) fn gather_of(
    read_index: &[ReadIndex],
) -> Option<(usize, u8, u8, crate::ir::OobPolicy, ElementKind)> {
    read_index.iter().enumerate().find_map(|(i, r)| match r {
        ReadIndex::Indexed {
            index_operand,
            axis,
            oob,
            index_dtype,
        } => Some((i, *index_operand, *axis, *oob, *index_dtype)),
        ReadIndex::Direct => None,
    })
}

/// `true` if `op` has a [`ReadIndex::Indexed`] input (a gather; increment 4).
/// `false` for an index-free op (empty `read_index`) and an all-`Direct` op — the
/// byte-identical cases. Forces the [`Schedule::Strided`] schedule and is the
/// `contract`/`pattern` honest-miss trigger.
pub(crate) fn op_has_gather(op: &OpDef) -> bool {
    op.read_index.iter().any(|r| !r.is_direct())
}

/// Increment-4 **GATHER** admissibility gate — runs at the TOP of [`build_plan`]
/// (the house pattern), with an independent emitter backstop in
/// [`crate::cuda::assert_gather_lowerable`]. An index-free op (empty
/// `read_index`, every pre-increment-4 constructor) returns immediately, and an
/// all-[`ReadIndex::Direct`] op returns after the length check — so the
/// established path is unchanged and emission stays byte-identical. For an op
/// carrying a real gather the v1 rules, all honest AOT panics (an author/
/// generator error — the gather role never crosses the JIT trust boundary):
///
/// 1. **Shape**: `read_index.len() == n_inputs` (index `i` ↔ `Input(i)`).
/// 2. **Access**: [`Access::Elementwise`]-only in v1 — a reduction/row-reduce/
///    contraction op has its own axis machinery a per-input indexed read would
///    double-count.
/// 3. **Single-output** in v1 (the multi-store DAG × the address substitution is
///    unproven).
/// 4. **One gathered input** in v1 — the emitter handles exactly one substituted
///    axis; a second data-dependent address (and combining OOB predicates across
///    two gathers) is deferred. Every bespoke gather/index_select/embedding
///    gathers exactly one input, so this covers the charter surface.
/// 5. Per gathered input `g` with `Indexed { index_operand, axis, oob,
///    index_dtype }`:
///    - `index_operand < n_inputs` and `index_operand != g` (an input can't index
///      itself).
///    - `index_dtype ∈ {I32, I64}` — an **integer** index (a float index address
///      is meaningless; the emitted load type must be an int).
///    - `axis < key.rank`.
///    - the index operand must NOT itself be gathered (`read_index[index_operand]`
///      is `Direct`) — a data-dependent index-of-an-index is out of v1 scope.
///    - the gathered input must NOT also carry an address-affecting [`View`]
///      (Permute/Broadcast) — gather ⊥ view in v1 (a gathered-and-permuted
///      operand's composed address math is unproven; reject rather than
///      mis-emit).
///    - the gathered axis of the DATA operand must have a real stride (its key
///      broadcast mask must NOT set `axis`) — the substituted `idx·stride[axis]`
///      term needs a live stride; a broadcast gathered axis is a degenerate
///      no-op.
fn assert_valid_gather(op: &OpDef, key: &StructureKey) {
    if op.read_index.is_empty() {
        return; // index-free — every pre-increment-4 op, unchanged.
    }
    let name = &op.name;
    assert_eq!(
        op.read_index.len(),
        op.n_inputs as usize,
        "OpDef '{name}': read_index.len() ({}) must equal n_inputs ({})",
        op.read_index.len(),
        op.n_inputs
    );
    if op.read_index.iter().all(ReadIndex::is_direct) {
        return; // all-Direct == index-free: byte-identical emission, no gate.
    }
    assert!(
        matches!(op.access, Access::Elementwise),
        "OpDef '{name}': a gather (Indexed read) is Access::Elementwise-only in v1 \
         — a {}-class op has its own axis machinery that a per-input indexed read \
         would double-count",
        access_tag(&op.access)
    );
    assert!(
        op.n_outputs() == 1,
        "OpDef '{name}': a gathered input on a multi-output op ({} outputs) is a \
         deferred composition in v1 — miss honestly",
        op.n_outputs()
    );
    let n_gathered = op.read_index.iter().filter(|r| !r.is_direct()).count();
    assert!(
        n_gathered == 1,
        "OpDef '{name}': v1 admits exactly one gathered input, got {n_gathered} — a \
         second data-dependent address (and combined OOB predicates) is deferred"
    );
    let (g, index_operand, axis, _oob, index_dtype) =
        gather_of(&op.read_index).expect("one gathered input checked above");
    assert!(
        (index_operand as usize) < op.n_inputs as usize,
        "OpDef '{name}': gather index_operand ({index_operand}) >= n_inputs ({})",
        op.n_inputs
    );
    assert!(
        index_operand as usize != g,
        "OpDef '{name}': gather input {g} names ITSELF as its index_operand — an \
         input cannot index itself"
    );
    assert!(
        matches!(index_dtype, ElementKind::I32 | ElementKind::I64 | ElementKind::U32),
        "OpDef '{name}': gather index_dtype must be an integer index dtype \
         (I32/I64/U32), got {index_dtype:?} — a float index address is meaningless"
    );
    assert!(
        (axis as usize) < key.rank as usize,
        "OpDef '{name}': gather axis ({axis}) >= iteration rank ({})",
        key.rank
    );
    assert!(
        op.read_index[index_operand as usize].is_direct(),
        "OpDef '{name}': the gather index operand ({index_operand}) must not ITSELF \
         be gathered — an index-of-an-index is out of v1 scope"
    );
    // gather ⊥ view (v1): the gathered input must not carry an address-affecting
    // view. An index-free/all-identity `views` is fine (the common case).
    if !op.views.is_empty() {
        assert!(
            !view_is_addressing(&op.views[g]),
            "OpDef '{name}': gathered input {g} also carries an address-affecting \
             View ({:?}) — gather ⊥ view in v1 (the composed address math is \
             unproven)",
            op.views[g]
        );
    }
    // The gathered axis of the DATA operand needs a live stride (the substituted
    // `idx·stride[axis]` term); a broadcast gathered axis is a degenerate no-op.
    // Input operands are the first `n_inputs` key slots.
    assert!(
        !key.operands[g].bcast.is_set(axis),
        "OpDef '{name}': gathered input {g} broadcasts the gathered axis ({axis}) \
         — the substituted idx·stride[axis] term needs a live stride"
    );
}

/// `true` if `op` scatters (a [`WriteIndex::ScatterIndexed`] output; increment 5).
/// `false` for a [`WriteIndex::Direct`] op — the byte-identical case. Forces the
/// [`Schedule::Strided`] schedule and is the `contract`/`pattern` honest-miss
/// trigger.
pub(crate) fn op_has_scatter(op: &OpDef) -> bool {
    !op.write_index.is_direct()
}

/// The scattered output's [`WriteIndex::ScatterIndexed`] fields
/// `(index_operand, axis, combine, oob, index_dtype)`, or `None` for a
/// [`WriteIndex::Direct`] op. The emitter and its backstop read the scatter off
/// this one accessor to stay in lockstep (mirror of [`gather_of`]).
pub(crate) fn scatter_of(
    write_index: &WriteIndex,
) -> Option<(u8, u8, WriteCombine, crate::ir::OobPolicy, ElementKind)> {
    write_index.scatter()
}

/// Whether the [`WriteCombine`] is legal for `out_dtype` at the emitter's v1
/// atomic-primitive coverage (used by the gate and the emitter backstop):
///
/// - `Assign` — legal for every dtype (a plain store).
/// - `AtomicAdd` — legal for `f32`/`f64` (native FP atomicAdd) and `i32`/`i64`
///   (native / `unsigned long long` reinterpret). f16/bf16 need the bespoke CAS
///   helper (`baracuda_atomic.cuh`), which the header-light generated source
///   can't include — deferred. u8/s8 need a sub-word CAS — deferred.
/// - `AtomicMax`/`AtomicMin` — **integer only** in v1 (`i32`/`i64`); float has no
///   native `atomicMax`/`atomicMin` (a CAS emulation is a follow-up).
pub(crate) fn combine_legal_for_dtype(combine: WriteCombine, out_dtype: ElementKind) -> bool {
    match combine {
        WriteCombine::Assign => true,
        WriteCombine::AtomicAdd => matches!(
            out_dtype,
            ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64 | ElementKind::I32 | ElementKind::I64
        ),
        WriteCombine::AtomicMax | WriteCombine::AtomicMin => {
            matches!(out_dtype, ElementKind::I32 | ElementKind::I64)
        }
    }
}

/// Increment-5 **SCATTER** admissibility gate — runs at the TOP of [`build_plan`]
/// (the house pattern), with an independent emitter backstop in
/// [`crate::cuda::assert_scatter_lowerable`]. A write-Direct op (every
/// pre-increment-5 constructor) returns immediately, so the established path is
/// unchanged and emission stays byte-identical. For an op carrying a real scatter
/// the v1 rules, all honest AOT panics (an author/generator error — the scatter
/// role never crosses the JIT trust boundary):
///
/// 1. **Access**: [`Access::Elementwise`]-only in v1 (a reduction/row-reduce/
///    contraction has its own axis machinery a scatter would double-count).
/// 2. **Single-output** in v1.
/// 3. **Not also a gather** — a fused gather+scatter (address-in AND address-out)
///    is a deferred composition; each ships separately.
/// 4. For the scatter role `ScatterIndexed { index_operand, axis, combine, oob,
///    index_dtype }`:
///    - `index_operand < n_inputs`.
///    - `index_dtype ∈ {I32, I64}` — an integer index (the emitted load type must
///      be an int; a float destination address is meaningless).
///    - `axis < key.rank`.
///    - the index operand must NOT itself be gathered (no index-of-an-index).
///    - `oob == Skip` — the only bespoke-matched scatter policy in v1
///      (Clamp/ZeroFill are gather-side; a scattered ZeroFill would need a
///      separate zeroing pass over the untouched destination).
///    - the `combine` op must be legal for the OUTPUT dtype
///      ([`combine_legal_for_dtype`] — AtomicMax/Min integer-only, atomicAdd
///      f32/f64/i32/i64).
///    - the scattered axis of the DESTINATION (the OUTPUT key slot) must have a
///      real stride (its broadcast mask must NOT set `axis`) — the substituted
///      `idx·stride_out[axis]` term needs a live stride.
///    - the output must NOT carry an address-affecting [`View`] (scatter ⊥ view
///      in v1) — views are an input-read property; a scattered output view is
///      unproven.
fn assert_valid_scatter(op: &OpDef, key: &StructureKey) {
    let Some((index_operand, axis, combine, oob, index_dtype)) = scatter_of(&op.write_index) else {
        return; // write-Direct — every pre-increment-5 op, unchanged.
    };
    let name = &op.name;
    assert!(
        matches!(op.access, Access::Elementwise),
        "OpDef '{name}': a scatter (ScatterIndexed write) is Access::Elementwise-only \
         in v1 — a {}-class op has its own axis machinery a scattered write would \
         double-count",
        access_tag(&op.access)
    );
    assert!(
        op.n_outputs() == 1,
        "OpDef '{name}': a scattered multi-output op ({} outputs) is a deferred \
         composition in v1 — miss honestly",
        op.n_outputs()
    );
    assert!(
        !op_has_gather(op),
        "OpDef '{name}': a fused gather+scatter (data-dependent read AND write) is a \
         deferred composition in v1 — each ships separately"
    );
    assert!(
        (index_operand as usize) < op.n_inputs as usize,
        "OpDef '{name}': scatter index_operand ({index_operand}) >= n_inputs ({})",
        op.n_inputs
    );
    assert!(
        matches!(index_dtype, ElementKind::I32 | ElementKind::I64 | ElementKind::U32),
        "OpDef '{name}': scatter index_dtype must be an integer index dtype \
         (I32/I64/U32), got {index_dtype:?} — a float destination address is meaningless"
    );
    assert!(
        (axis as usize) < key.rank as usize,
        "OpDef '{name}': scatter axis ({axis}) >= iteration rank ({})",
        key.rank
    );
    assert!(
        matches!(oob, crate::ir::OobPolicy::Skip),
        "OpDef '{name}': scatter OOB policy must be Skip in v1 (bespoke \
         scatter/scatter_add/index_add/bincount all skip an OOB target), got {oob:?}"
    );
    let out_dtype = op.out_dtype.unwrap_or(key.dtype);
    assert!(
        combine_legal_for_dtype(combine, out_dtype),
        "OpDef '{name}': scatter combine {combine:?} is not legal for output dtype \
         {out_dtype:?} in v1 (AtomicMax/Min integer-only; atomicAdd f32/f64/i32/i64; \
         f16/bf16/u8 atomics need the bespoke CAS helper the header-light source \
         can't include)"
    );
    // The scattered axis of the DESTINATION (last key slot) needs a live stride.
    let out_slot = (key.n_operands as usize).saturating_sub(1);
    assert!(
        !key.operands[out_slot].bcast.is_set(axis),
        "OpDef '{name}': scattered output broadcasts the scattered axis ({axis}) — \
         the substituted idx·stride_out[axis] term needs a live stride"
    );
    // scatter ⊥ view (v1): `views` is a per-INPUT slice (an output view would ride
    // a separate future field, not expressible today), so there is no output-view
    // to reject here — the scatter output offset is always an identity remap. The
    // INPUT operands may still carry views (the value operand could be a transposed
    // read); that composes cleanly (input views are handled by `offset_expr`),
    // so it is deliberately NOT rejected.

    // Body must be a bare identity value read or a Const (review #5 CRITICAL): the
    // deterministic gather-sum base (`emit_scatter_gathersum`) sums `in{val_op}`
    // DIRECTLY rather than lowering `op.body`, so a composed body (e.g.
    // `Input(0)*Param(0)`) would silently compute `Sum(updates)` instead of
    // `Sum(f(updates))` — diverging from both the op and its own atomic variant
    // (which DOES lower the body). Bespoke scatter/scatter_add copy the value
    // verbatim and bincount stores a constant, so v1 pins the body accordingly; a
    // fused scatter body is a deferred v1 composition. `val_op` matches the
    // emitter's derivation exactly.
    let val_op = (0..op.n_inputs as usize)
        .find(|&k| k != index_operand as usize)
        .unwrap_or(0);
    assert!(
        matches!(&op.body, ScalarExpr::Input(v) if *v as usize == val_op)
            || matches!(&op.body, ScalarExpr::Const(_)),
        "OpDef '{name}': a v1 scatter body must be the identity value read Input({val_op}) \
         or a constant (bincount), got a composed body — a fused scatter transform is a \
         deferred v1 composition (the deterministic gather-sum base sums the value operand \
         directly and would silently drop it)"
    );
}

fn assert_no_half_nextafter(op: &OpDef, dtype: ElementKind) {
    use crate::ir::BinaryOp;
    if !matches!(dtype, ElementKind::F16 | ElementKind::Bf16) {
        return;
    }
    fn walk(e: &ScalarExpr) -> bool {
        match e {
            ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Param(_)
            | ScalarExpr::Reduced(_)
            | ScalarExpr::Coord(_) => false,
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b) => walk(a) || walk(b),
            ScalarExpr::Unary(_, a) => walk(a),
            ScalarExpr::Binary(bop, a, b) => {
                matches!(bop, BinaryOp::Nextafter) || walk(a) || walk(b)
            }
        }
    }
    let mut exprs: Vec<&ScalarExpr> = vec![&op.body];
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            exprs.extend(stages.iter().map(|s| &s.pre));
            exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => exprs.push(epilogue),
        // The reduction post-expr (0e) lowers through the accumulator-width
        // spellers too, so the honest-miss walk must cover it (body is already in).
        Access::Reduction { post, .. } => exprs.push(post),
        // Increment 6 SCAN: the `pre` (per-element pre-map) and `post` (per-element
        // epilogue) both lower through the accumulator-width spellers (body == post
        // is already in), so a half `Nextafter` hidden in `pre` must miss honestly.
        Access::Scan { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        // Increment 7 WINDOW: `pre` (per-tap pre-map) and `post` (per-output
        // epilogue) both lower through the accumulator-width spellers (body ==
        // post is already in), so a half `Nextafter` hidden in `pre` must miss
        // honestly.
        Access::Window { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        // Review-caught gate asymmetry (increment 1): a multi-output op's EXTRA
        // output bodies must be walked too — else a half `Nextafter` hidden in an
        // extra body bypasses this honest-miss gate. Non-elementwise multi-output
        // is already rejected by `assert_valid_multi_output` (runs first), so this
        // arm is the ONLY place `extra_out_bodies` is legal; empty for every
        // single-output op (byte-identical).
        Access::Elementwise => exprs.extend(op.extra_out_bodies.iter()),
    }
    for e in exprs {
        assert!(
            !walk(e),
            "Nextafter has no half-precision lowering (IR contract: f32/f64 only; \
             the promote-to-f32 path silently no-ops after the demote) — op '{}' \
             at {dtype:?} must miss honestly",
            op.name
        );
    }
}

/// The integer compute dtypes of increment 0c: `I32`/`I64` (already lowering
/// pre-0c) plus the newly-promoted `S8` (FKC `I8`) and `U8`.
pub(crate) fn is_int_dtype(dt: ElementKind) -> bool {
    matches!(
        dt,
        ElementKind::I32 | ElementKind::I64 | ElementKind::S8 | ElementKind::U8
    )
}

/// Increment-0c op × dtype admissibility gate — the plan-level enforcement of
/// the table in [`crate::ir::BinaryOp`]'s docs. Runs at the TOP of
/// [`build_plan`] and walks the body of EVERY Access arm (the 0a lesson: the
/// emitter backstops alone were bypassed by the reduction-class lowering
/// paths), so no lowering path — present or future backend — can bypass it.
///
/// Two directions, both validate-reject (honest miss, never silent):
///
/// 1. **Int-only ops** (`BitAnd`/`BitOr`/`BitXor`/`Shl`/`Shr`/`Logical*`) may
///    appear ONLY in an [`Access::Elementwise`] body at an int dtype (the
///    reduction/RowReduce/contraction paths lower through the FLOAT
///    accumulator spellers, which have no int arms), and the logical ops
///    narrow further to `U8` — the bespoke Bool surface (`uint8_t` only).
/// 2. **At an int dtype**, only the audited op set lowers: infix
///    `Add`/`Sub`/`Mul` (wrapping) and the int-only ops. Everything else —
///    every `UnaryOp`, infix `Div` (no bespoke int div; `/0` is device-UB),
///    the float binary fns, and the `Cmp*` predicates (bespoke cmp is
///    `_fp`-only) — rejects. `Const` rejects too: it is spelled as an f64 C
///    literal, so an int body would silently run double math (and f64 cannot
///    even represent all i64); an int-literal speller is a follow-up. `Param`
///    rejects at int for the same f32-only reason the emitter asserts.
/// 3. **8-bit composition pin (v1):** at `U8`/`S8`, EVERY operand of an
///    int-only op must be a leaf [`ScalarExpr::Input`]. Why: `Add`/`Sub`/
///    `Mul`/`BitAnd`/`BitOr`/`BitXor` compositions are congruent under
///    deferred truncation (the wrapping ring ops and the bit-local ops
///    commute with the final 8-bit store truncate), but `Shr`, shift
///    AMOUNTS, and the logical `!= 0` tests OBSERVE the un-truncated
///    promoted-`int` value — and the DAG emitter truncates a composed
///    interior only when sharing hoists it to an 8-bit tmp, so one body
///    could compute two different results depending on DAG sharing
///    (`(in0+in1)>>in2` at u8 with `(200,100,1)`: inlined `300>>1 = 150`,
///    hoisted `44>>1 = 22`). Rather than a per-position observer analysis,
///    v1 pins ALL int-op operands at 8-bit to leaves — the bespoke surface
///    has no 8-bit bitwise at all, so zero parity is lost; the dtype-aware
///    truncating speller is the follow-up that lifts this. At `I32`/`I64`
///    compositions stay legal: integer promotion never widens past the
///    compute width there, so no un-truncated wider value exists to observe.
fn assert_int_op_admissibility(op: &OpDef, dtype: ElementKind) {
    // Increment 5 — bincount exemption: a scatter with a bare `Const` body is the
    // integer-count histogram (`out[x[i]] += 1`). The `Const(1)` is NOT compute
    // (no int arithmetic, no double-math hazard) — it is a store literal the
    // scatter combine narrows EXACTLY to the count cell (`(int)(1.0)` = 1). The
    // input `x` is read only as an integer INDEX, never a value leaf. So the
    // int-Const rejection (which polices f64 literals inside int arithmetic) does
    // not apply; skip the walk for this one shape.
    if op_has_scatter(op) && matches!(op.body, ScalarExpr::Const(_)) {
        return;
    }
    let int_dt = is_int_dtype(dtype);
    let elementwise = matches!(op.access, Access::Elementwise);
    fn walk(
        e: &ScalarExpr,
        op_name: &str,
        dtype: ElementKind,
        int_dt: bool,
        elementwise: bool,
    ) {
        match e {
            ScalarExpr::Input(_) | ScalarExpr::Reduced(_) => {}
            // Coord's own gate (`assert_coord_admissibility`, which also runs
            // at the top of build_plan) rejects EVERY int dtype — a Coord is
            // spelled as a float cast, the same double-math hazard this walk
            // polices for Const/Param — so this arm carries no second assert
            // (one source of truth for the message). It is also structurally
            // moot for rule 3: an int-only op's operands are pinned to leaf
            // Inputs at 8-bit before Coord could ever appear there.
            ScalarExpr::Coord(_) => {}
            ScalarExpr::Const(_) => assert!(
                !int_dt,
                "op '{op_name}': Const at int dtype {dtype:?} is rejected — a Const \
                 is spelled as an f64 C literal, which would silently run double \
                 math in an integer kernel (and f64 cannot represent all i64); \
                 int-literal Const spelling is a follow-up"
            ),
            ScalarExpr::Param(_) => assert!(
                !int_dt,
                "op '{op_name}': scalar params are f32-only (int dtype {dtype:?})"
            ),
            ScalarExpr::Unary(uop, x) => {
                assert!(
                    !int_dt,
                    "op '{op_name}': {uop:?} has no integer lowering — the bespoke \
                     unary elementwise surface is float-only, so int dtype {dtype:?} \
                     must miss honestly"
                );
                walk(x, op_name, dtype, int_dt, elementwise);
            }
            ScalarExpr::Div(a, b) => {
                assert!(
                    !int_dt,
                    "op '{op_name}': integer division is rejected at {dtype:?} — the \
                     bespoke elementwise surface has no int div (binary_div_fp.cu is \
                     float-only) and C `/` division by zero is device-undefined; \
                     miss honestly"
                );
                walk(a, op_name, dtype, int_dt, elementwise);
                walk(b, op_name, dtype, int_dt, elementwise);
            }
            ScalarExpr::Binary(bop, a, b) => {
                if bop.is_int_only() {
                    assert!(
                        elementwise,
                        "op '{op_name}': {bop:?} is Elementwise-only in 0c — the \
                         reduction-class paths lower through the float accumulator \
                         spellers, which have no integer arms"
                    );
                    assert!(
                        int_dt,
                        "op '{op_name}': {bop:?} is int-only (I32/I64/S8/U8) — float \
                         dtype {dtype:?} must miss honestly (the bespoke bitwise/\
                         logical kernels have no float instantiation)"
                    );
                    assert!(
                        !bop.is_logical() || dtype == ElementKind::U8,
                        "op '{op_name}': {bop:?} is U8 (Bool)-only — the bespoke \
                         binary_logical_*_bool.cu surface instantiates exactly \
                         uint8_t, so {dtype:?} must miss honestly"
                    );
                    // Rule 3 (8-bit composition pin, v1): every operand of an
                    // int-only op at U8/S8 must be a LEAF Input — a composed
                    // operand's value differs between the inlined (un-truncated
                    // promoted-int) and hoisted (8-bit tmp, truncated) spellings,
                    // so admitting it would make the result depend on DAG
                    // sharing. See the doc comment above for the full rationale.
                    if matches!(dtype, ElementKind::U8 | ElementKind::S8) {
                        for (side, operand) in [("lhs", &**a), ("rhs", &**b)] {
                            assert!(
                                matches!(operand, ScalarExpr::Input(_)),
                                "op '{op_name}': {bop:?} at {dtype:?} requires LEAF \
                                 Input operands ({side} is a composed expression) — \
                                 at 8-bit dtypes a composed operand observes the \
                                 un-truncated promoted-int value when inlined but \
                                 the truncated 8-bit value when hoisted to a shared \
                                 tmp (one body, two results); v1 pins all int-op \
                                 operands at U8/S8 to leaves. Compose at I32/I64, \
                                 or wait for the dtype-aware truncating speller"
                            );
                        }
                    }
                } else {
                    assert!(
                        !int_dt,
                        "op '{op_name}': {bop:?} has no integer lowering — the \
                         bespoke elementwise surface instantiates it for float \
                         dtypes only, so int dtype {dtype:?} must miss honestly"
                    );
                }
                walk(a, op_name, dtype, int_dt, elementwise);
                walk(b, op_name, dtype, int_dt, elementwise);
            }
            ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) | ScalarExpr::Mul(a, b) => {
                // Wrapping two's-complement at int dtypes — the audited-legal set.
                walk(a, op_name, dtype, int_dt, elementwise);
                walk(b, op_name, dtype, int_dt, elementwise);
            }
        }
    }
    let mut exprs: Vec<&ScalarExpr> = vec![&op.body];
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            exprs.extend(stages.iter().map(|s| &s.pre));
            exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => exprs.push(epilogue),
        // The reduction post-expr (0e) lowers at the accumulator dtype — a
        // Const/Param/Div/unary there hits the same int-dtype hazards, so gate it.
        Access::Reduction { post, .. } => exprs.push(post),
        // Increment 6 SCAN: `pre`/`post` lower at the accumulator dtype (an int
        // cumsum/cummax rides the serial base), so a Const/Param/Div/unary/int-only
        // op there hits the same int hazards — gate both.
        Access::Scan { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        // Increment 7 WINDOW: `pre`/`post` lower at the accumulator dtype (an int
        // sum/max pool rides the same fold), so a Const/Param/Div/unary/int-only
        // op there hits the same int hazards — gate both.
        Access::Window { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        // Review-caught gate asymmetry (increment 1): walk the EXTRA output bodies
        // too. Without this, an int-only op with a COMPOSED operand hides in a
        // multi-output extra body at U8/S8 and bypasses the 8-bit leaf-operand pin
        // (rule 3) — cross-body CSE then hoists it into a truncated 8-bit tmp, the
        // exact 0c value-divergence ((200+100)>>1 = 22 hoisted vs 150 inlined).
        // Non-elementwise multi-output is already rejected by
        // `assert_valid_multi_output` (runs first), so this is the only place
        // `extra_out_bodies` is legal; empty for every single-output op.
        Access::Elementwise => exprs.extend(op.extra_out_bodies.iter()),
    }
    for e in exprs {
        walk(e, &op.name, dtype, int_dt, elementwise);
    }
}

/// Whether `e` contains a [`ScalarExpr::Coord`] leaf anywhere — drives the
/// increment-0d Strided schedule routing in [`build_plan`] (a Coord body must
/// reach the one emitter that materializes per-axis coordinates) and mirrors
/// `contract::expr_contains_cmp` in shape.
pub(crate) fn expr_contains_coord(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Coord(_) => true,
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_) => false,
        ScalarExpr::Unary(_, x) => expr_contains_coord(x),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => expr_contains_coord(a) || expr_contains_coord(b),
    }
}

/// Increment-0d [`ScalarExpr::Coord`] admissibility gate — runs at the TOP of
/// [`build_plan`] and walks the expressions of EVERY Access arm (the 0a
/// lesson: emitter backstops alone are bypassed by the reduction-class
/// lowering paths), with independent emitter backstops in `cuda`. Three
/// validate-reject rules, all honest misses:
///
/// 1. **Access**: Coord is legal ONLY in an [`Access::Elementwise`] body (v1).
///    A coordinate along a reduced/folded axis is ambiguous (which fold
///    iteration produced the output element?), and the RowReduce/Contraction
///    epilogues iterate their own coordinate spaces ((row, j) and (m, n)) —
///    lifting Coord into them needs explicit per-arm semantics, deferred.
/// 2. **Dtype**: `F32`/`F32Strict`/`F64` ONLY. f16/bf16 reject — the max
///    exactly-representable integer is 2048 (bf16: 256), which real axis
///    extents exceed, so a half coordinate would silently round. Int dtypes
///    reject — the coordinate lowers as a float cast (`(float)c{d}`), the
///    same double-math hazard as `Const`/`Param` at int dtypes; the
///    int-literal coordinate spelling is the queued follow-up.
/// 3. **Axis**: `axis < key.rank` — an out-of-range axis has no `c{d}` to
///    read (the emitter would spell an undefined identifier).
///
/// The exactness bound (f32 coordinates exact to 2²⁴, f64 to 2⁵³) is a CALLER
/// precondition — the key abstracts extents away, the same trust level as the
/// RowReduce column-weight extent precondition (see [`ScalarExpr::Coord`]).
fn assert_coord_admissibility(op: &OpDef, key: &StructureKey) {
    let elementwise = matches!(op.access, Access::Elementwise);
    fn walk(e: &ScalarExpr, op_name: &str, dtype: ElementKind, rank: u8, elementwise: bool) {
        match e {
            ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Param(_)
            | ScalarExpr::Reduced(_) => {}
            ScalarExpr::Coord(d) => {
                assert!(
                    elementwise,
                    "op '{op_name}': Coord({d}) is Elementwise-only in 0d — a coordinate \
                     along a reduced/folded axis is ambiguous (which fold iteration?), and \
                     the RowReduce/Contraction stages/epilogues iterate their own \
                     coordinate spaces; miss honestly"
                );
                assert!(
                    matches!(
                        dtype,
                        ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64
                    ),
                    "op '{op_name}': Coord({d}) requires an f32/f64 compute dtype, got \
                     {dtype:?} — f16/bf16 coordinates round past extent 2048 (bf16: 256) \
                     and int dtypes would inject the float-cast coordinate into integer \
                     math (int-literal coordinate spelling is a follow-up); miss honestly"
                );
                assert!(
                    *d < rank,
                    "op '{op_name}': Coord({d}) axis out of range for rank {rank} — the \
                     iteration space has no such coordinate"
                );
            }
            ScalarExpr::Unary(_, x) => walk(x, op_name, dtype, rank, elementwise),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                walk(a, op_name, dtype, rank, elementwise);
                walk(b, op_name, dtype, rank, elementwise);
            }
        }
    }
    let mut exprs: Vec<&ScalarExpr> = vec![&op.body];
    match &op.access {
        Access::RowReduce { stages, epilogue } => {
            exprs.extend(stages.iter().map(|s| &s.pre));
            exprs.push(epilogue);
        }
        Access::Contraction { epilogue, .. } => exprs.push(epilogue),
        // A Coord in a reduction post-expr is doubly rejected (here, non-
        // elementwise → the Coord arm fires; and by assert_valid_reduction_post).
        Access::Reduction { post, .. } => exprs.push(post),
        // Increment 6 SCAN: a Coord in `pre`/`post` is doubly rejected (here, the
        // scan is non-elementwise → the Coord arm fires; and the emitter's
        // panicking `coord` closure). The scan iterates the (row, j) space, not an
        // elementwise output coordinate space.
        Access::Scan { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        // Increment 7 WINDOW: a Coord in `pre`/`post` is doubly rejected (here, the
        // window is non-elementwise → the Coord arm fires; and the emitter's
        // panicking `coord` closure). The window iterates the (row, o) space, not an
        // elementwise output coordinate space.
        Access::Window { pre, post, .. } => {
            exprs.push(pre);
            exprs.push(post);
        }
        Access::Elementwise => {}
    }
    for e in exprs {
        walk(e, &op.name, key.dtype, key.rank, elementwise);
    }
}

/// Validate [`crate::ir::OpDef::out_dtype`] at plan time (AOT — like
/// `assert_no_half_nextafter`, this runs at the top of [`build_plan`] so EVERY
/// Access arm and every lowering path is covered; a panic here is an
/// author-error backstop, and the JIT never constructs a `Some` out_dtype).
///
/// Two admitted hetero-output shapes, both with an EXACT store conversion:
///
/// 1. **[`Access::Elementwise`] predicate → `Some(U8)`** (increment 0b): the
///    body ROOT must be a `Cmp*` — the value is exactly 0.0/1.0 and
///    `(unsigned char)` of that is exactly 1/0 (`OpDef::elementwise_pred`).
///
/// 2. **[`Access::Reduction`] hetero-out** (increment 0e — the roadmap "any/all
///    → U8, count → I64" reduction):
///    - `Some(U8)`: the POST-expr ROOT must be a `Cmp*`, so the stored value is
///      exactly 0.0/1.0 regardless of the fold magnitude — the honest
///      boolean-reduce (`any` = `Sum(x≠0)` with post `Reduced(0) > 0`; `all` =
///      `Sum(x=0)` with post `Reduced(0) = 0`; or `Max`/`Min` of a predicate
///      wrapped in a redundant cmp post). A non-cmp post would truncate the raw
///      accumulator silently (a count of 300 → `44` at u8), so it rejects.
///    - `Some(I64)`: the combine must be `Sum` and the post the identity
///      `Reduced(0)` — a **count** (`Sum(x≠0)`) or a sum-widening. The store is
///      `(long long)` of the accumulator: exact for an int input (i32→i64
///      widening) and exact for a float accumulator while the count ≤ 2²⁴ (a
///      documented CALLER precondition, the same trust level as `Coord`'s
///      exact-integer bound — the key abstracts extents away). `Mean`/`Max`/
///      `Min` → I64 reject (fractional / not the count shape).
///
/// Everything else panics honestly: `Some(non-U8/I64)` has no store conversion;
/// a `RowReduce`/`Contraction` stores its accumulator, not a predicate, so a
/// hetero store there would truncate silently.
///
/// A `Cmp*` NESTED inside a float body (mask-multiply `dy * (x > 0)`) is legal
/// with `out_dtype = None` — it is an inline 0.0/1.0 float, no u8 store — and a
/// top-level cmp with `out_dtype = None` (a float mask) is likewise legal.
fn assert_valid_out_dtype(op: &OpDef) {
    let Some(od) = op.out_dtype else { return };
    match &op.access {
        // Increment 5 — a SCATTER may write an `I32` counts output (bincount:
        // `out[x[i]] += 1`), a hetero store distinct from the U8 mask. The body is
        // a `Const` increment (no input value read), narrowed exactly to the count
        // cell; the write role's own gate (`assert_valid_scatter`) covers the rest.
        Access::Elementwise if op_has_scatter(op) => {
            assert!(
                od == ElementKind::I32 && matches!(&op.body, ScalarExpr::Const(_)),
                "OpDef '{}': a hetero-output scatter admits only the bincount shape \
                 (out_dtype Some(I32) with a Const increment body), got out_dtype \
                 Some({od:?})",
                op.name
            );
        }
        Access::Elementwise => {
            assert!(
                od == ElementKind::U8,
                "OpDef '{}': out_dtype Some({od:?}) is unsupported for an \
                 Elementwise op — the only hetero output dtype there is U8 (the \
                 comparison-predicate mask; use OpDef::elementwise_pred)",
                op.name
            );
            assert!(
                matches!(&op.body, ScalarExpr::Binary(bop, _, _) if bop.is_cmp()),
                "OpDef '{}': out_dtype = Some(U8) requires the body ROOT to be a \
                 comparison (ScalarExpr::Binary with a Cmp* op) — only a predicate \
                 yields exactly 0.0/1.0, so any other body would truncate silently \
                 under the u8 store; nested comparisons in a float body take \
                 out_dtype = None instead",
                op.name
            );
        }
        Access::Reduction { op: rop, post, .. } => match od {
            ElementKind::U8 => assert!(
                matches!(post, ScalarExpr::Binary(bop, _, _) if bop.is_cmp()),
                "OpDef '{}': a U8-output reduction requires the POST-expr ROOT to \
                 be a comparison (Cmp*) — only then is the stored value exactly \
                 0.0/1.0 (the honest any/all boolean-reduce); a non-cmp post would \
                 truncate the raw accumulator silently under the u8 store",
                op.name
            ),
            ElementKind::I64 => assert!(
                matches!(rop, ReduceOp::Sum) && matches!(post, ScalarExpr::Reduced(0)),
                "OpDef '{}': an I64-output reduction is the count/sum-widening shape \
                 — it requires op = Sum and the identity post (Reduced(0)); \
                 Mean/Max/Min → I64 or a non-identity post is out of scope (would \
                 not be an exact integer store)",
                op.name
            ),
            other => panic!(
                "OpDef '{}': out_dtype Some({other:?}) is unsupported for a \
                 reduction — v1 admits U8 (boolean any/all, via a Cmp* post) and \
                 I64 (count, via Sum + identity post)",
                op.name
            ),
        },
        Access::RowReduce { .. } | Access::Contraction { .. } => panic!(
            "OpDef '{}': out_dtype = Some({od:?}) is rejected under \
             RowReduce/Contraction — a fused reduction/contraction stores its \
             accumulator, not a 0/1 predicate, so a hetero store would truncate \
             silently; only Access::Elementwise (predicate → U8) and \
             Access::Reduction (any/all → U8, count → I64) carry a hetero output",
            op.name
        ),
        // Increment 6 SCAN: a cumulative op does not change dtype (the output is
        // same-shape, same-dtype as Input(0)), so a hetero out_dtype has no exact
        // store — reject it, exactly like RowReduce/Contraction.
        Access::Scan { .. } => panic!(
            "OpDef '{}': out_dtype = Some({od:?}) is rejected under Scan — a prefix \
             scan is same-dtype as its input (a cumulative op does not change \
             dtype), so there is no hetero store; use out_dtype = None",
            op.name
        ),
        // Increment 7 WINDOW: a pool preserves the input dtype (max/avg pool output
        // is same-dtype as the pooled tensor), so a hetero out_dtype has no exact
        // store — reject it, exactly like Scan/RowReduce/Contraction.
        Access::Window { .. } => panic!(
            "OpDef '{}': out_dtype = Some({od:?}) is rejected under Window — a \
             pooling reduction is same-dtype as its input (a pool does not change \
             dtype), so there is no hetero store; use out_dtype = None",
            op.name
        ),
    }
}

/// Validate the increment-0e reduction **post-expression** at plan time (AOT).
/// Runs at the top of [`build_plan`] (like the other honest-miss gates), with an
/// independent emitter backstop in `cuda::emit_reduction` (the post's `leaf`
/// closure panics if an `Input` ever reaches it). The post references the fold
/// result as `Reduced(0)` and MAY read `Const`/`Param`; it must NOT read:
///
/// - `Input(_)` — the reduced axis is gone, so an input at the output coordinate
///   is a different, ambiguous tensor (this mirrors the contraction epilogue's
///   `epilogue_reads_only_reduced0`, generalized to also admit `Param`);
/// - `Coord(_)` — reduction-class, Elementwise-only (also caught upstream);
/// - `Reduced(s)` for `s ≥ 1` — a single-fold reduction produces only
///   `Reduced(0)`.
fn assert_valid_reduction_post(op: &OpDef) {
    let Access::Reduction { post, .. } = &op.access else {
        return;
    };
    fn walk(e: &ScalarExpr, name: &str) {
        match e {
            ScalarExpr::Reduced(0) | ScalarExpr::Const(_) | ScalarExpr::Param(_) => {}
            ScalarExpr::Reduced(s) => panic!(
                "OpDef '{name}': reduction post-expr Reduced({s}) — a single-fold \
                 reduction produces only Reduced(0)"
            ),
            ScalarExpr::Input(i) => panic!(
                "OpDef '{name}': reduction post-expr must not read Input({i}) — the \
                 reduced axis is gone, so an input at the output coordinate is a \
                 different, ambiguous tensor; the post reads Reduced(0)/Const/Param"
            ),
            ScalarExpr::Coord(d) => panic!(
                "OpDef '{name}': reduction post-expr must not read Coord({d}) — Coord \
                 is Elementwise-only (a coordinate along a folded axis is ambiguous)"
            ),
            ScalarExpr::Unary(_, x) => walk(x, name),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                walk(a, name);
                walk(b, name);
            }
        }
    }
    walk(post, &op.name);
}

fn validate_row_reduce(stages: &[ReduceStage], epilogue: &ScalarExpr, n_inputs: u8, key: &StructureKey) {
    let dtype = key.dtype;
    assert!(
        matches!(
            dtype,
            ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
        ),
        "RowReduce requires a float dtype, got {dtype:?}"
    );
    let n = n_inputs as usize;
    assert!(
        (1..MAX_OPERANDS).contains(&n),
        "RowReduce n_inputs {n_inputs} out of [1, MAX_OPERANDS)"
    );
    assert!(
        key.n_operands as usize == n + 1,
        "RowReduce expects n_inputs+1 operands (inputs then output); got {}",
        key.n_operands
    );
    let rank = key.rank as usize;
    assert!(rank >= 1, "RowReduce needs a last (reduced) axis");
    let last = (rank - 1) as u8;

    // Operand roles + layout legality (the OOB / mis-index guards). Parallel index
    // over key.operands, so a range loop is the natural form. `is_col` feeds the
    // epilogue-only check below (reducing a per-column weight is rejected); a
    // RowScalar is legal in BOTH a stage `pre` and the epilogue (it is constant
    // along the reduced axis — layer-norm-bw's x_hat reads μ/rstd inside a fold),
    // so it is deliberately NOT tracked as a column.
    let mut is_col = [false; MAX_OPERANDS];
    let mut input0_streamed = false;
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let o = key.operands[i];
        match rr_role(o, last) {
            RrRole::RowStreamed => {
                assert!(
                    o.contig == Contiguity::Contig,
                    "RowReduce row-streamed input {i} must be contiguous (base = row*k assumes a dense last axis)"
                );
                // `Contiguity::Contig` is |stride|-based (classify_contiguity uses
                // strides[d].abs()), so a dense-but-REVERSED view passes the contig
                // check while carrying flipped=true — and the emitter walks memory
                // FORWARD (idx = row*k + j), reading the tensor mirrored / off the
                // end. `flipped` is a key-visible axis (the 'r'/'f' token field), so
                // reject it here as the ColBroadcast/RowScalar branches do. (Review
                // #2: the pre-lift inputs>0-must-be-column guard flip-checked every
                // extra input; lifting it to allow a 2nd row-streamed input newly
                // exposed this path.)
                assert!(
                    !o.flipped,
                    "RowReduce row-streamed input {i} must not be reversed along an axis (base = row*k reads forward-dense; a flipped view would read mirrored/OOB)"
                );
                input0_streamed |= i == 0;
            }
            RrRole::ColBroadcast => {
                assert!(
                    !o.bcast.is_set(last),
                    "RowReduce input {i}: the feature (last) axis is broadcast (mask {:#04x}) — a column weight/bias must vary along it; bake a true scalar as Const",
                    o.bcast.0
                );
                // Must broadcast EVERY outer axis (a per-column [k] vector), else
                // in_i[j] silently drops an outer-axis dependence.
                assert!(
                    (0..last).all(|d| o.bcast.is_set(d)),
                    "RowReduce column input {i} must broadcast every outer (row) axis — a per-column [k] weight/bias"
                );
                assert!(
                    !o.flipped,
                    "RowReduce column input {i} must not be reversed along the feature axis"
                );
                is_col[i] = true;
            }
            RrRole::RowScalar => {
                // Per-row scalar (a saved stat: μ, rstd, lse). The feature (last)
                // axis is broadcast (guaranteed by rr_role); NO outer (row) axis may
                // be — else it is either all-broadcast (a true scalar → bake as
                // Const) or drops a row dependence. Indexed `in_i[row]`, so it needs
                // an outer axis (rank >= 2) laid out dense (offset == row), the
                // latter a caller precondition at the same trust level as `x`'s
                // base = row*k (see the module note).
                assert!(
                    (0..last).all(|d| !o.bcast.is_set(d)),
                    "RowReduce row-scalar input {i}: an outer (row) axis is broadcast (mask {:#04x}) — a per-row scalar varies across rows and is constant only along the feature axis; an all-broadcast operand is a true scalar (bake as Const)",
                    o.bcast.0
                );
                assert!(
                    !o.flipped,
                    "RowReduce row-scalar input {i} must not be reversed"
                );
                assert!(
                    rank >= 2,
                    "RowReduce row-scalar input {i} needs rank >= 2 (an outer row axis to index by `row`)"
                );
            }
        }
    }
    assert!(
        input0_streamed,
        "RowReduce Input0 (x) must be the row-streamed reduced tensor, not a column-broadcast weight or a per-row scalar"
    );
    // Inputs 1.. may now be a second **row-streamed** tensor (softmax-bw's `dy`
    // beside `y`), a per-column weight/bias, OR a per-row scalar (layer-norm-bw's
    // μ/rstd). The former "inputs>0 must be column-broadcast" guard is LIFTED
    // (increment 2): a second row-streamed input full [n_out,k] is the point. A
    // bare rank-1 [k] passed as input>0 has an empty bcast and so is now accepted
    // as row-streamed — its full extent [n_out,k] is a caller precondition (the key
    // cannot see n_out/k), the identical trust level as input 0. See the module note.
    if n > 1 {
        assert!(rank >= 2, "RowReduce with a multi-operand epilogue needs rank >= 2");
    }
    let out = key.operands[n];
    assert!(
        out.bcast.is_empty() && out.contig == Contiguity::Contig,
        "RowReduce output must be full-width contiguous (empty bcast)"
    );

    // Expression legality. `max_reduced` = stages already produced (stage `i` may
    // read `Reduced(0..i)`; the epilogue may read all). `in_stage` forbids a column
    // input inside a reduction `pre` (reducing a per-column operand is nonsense).
    fn check(e: &ScalarExpr, n_inputs: u8, max_reduced: u8, in_stage: bool, is_col: &[bool]) {
        match e {
            ScalarExpr::Input(i) => {
                assert!(*i < n_inputs, "RowReduce Input({i}) >= n_inputs {n_inputs}");
                if in_stage {
                    assert!(
                        !is_col[*i as usize],
                        "RowReduce column input {i} used inside a reduction stage.pre — column weight/bias are epilogue-only"
                    );
                }
            }
            ScalarExpr::Reduced(s) => assert!(
                *s < max_reduced,
                "RowReduce Reduced({s}) references a stage not yet produced (have {max_reduced})"
            ),
            ScalarExpr::Param(i) => {
                panic!("RowReduce v1 forbids Param({i}) — bake scalars (eps) as Const")
            }
            ScalarExpr::Coord(d) => {
                panic!(
                    "RowReduce forbids Coord({d}) — the RowReduce stages/epilogue iterate \
                     the (row, j) space, not an elementwise output coordinate space; \
                     Coord is Elementwise-only in 0d"
                )
            }
            ScalarExpr::Const(v) => assert!(v.is_finite(), "RowReduce Const must be finite, got {v}"),
            ScalarExpr::Unary(_, x) => check(x, n_inputs, max_reduced, in_stage, is_col),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check(a, n_inputs, max_reduced, in_stage, is_col);
                check(b, n_inputs, max_reduced, in_stage, is_col);
            }
        }
    }
    for (i, st) in stages.iter().enumerate() {
        // Prod stages (0e added the combiner to Access::Reduction, not to the
        // fused RowReduce cooperative reducer) are an honest miss here — the
        // emitter has no block_prod in the row path. Gate + emitter backstop.
        assert!(
            !matches!(st.op, ReduceOp::Prod),
            "RowReduce stage {i}: the Prod combiner is not supported in the fused \
             row-reduce path (0e adds Prod to Access::Reduction only); miss honestly"
        );
        check(&st.pre, n_inputs, i as u8, true, &is_col);
    }
    check(epilogue, n_inputs, stages.len() as u8, false, &is_col);
}

/// Validate an [`Access::Scan`] op at build time (AOT — a scan never crosses the
/// JIT trust boundary, so a panic here is an author-error backstop). Mirrors
/// [`validate_row_reduce`]'s operand-role + layout checks, with three DELIBERATE
/// differences:
///
/// - **ADMITS `Prod`** — unlike RowReduce (which forbids `Prod` because the fused
///   row path has no `block_prod`), a scan explicitly wants cumprod, and the
///   serial base folds it directly. Only the block-scan VARIANT emitter (§4)
///   declines integer Sum/Prod; the serial base is `BitIdentical` for every
///   admitted dtype (integer Sum/Prod wraps exactly; Max/Min is exactly associative).
/// - **NO float-only gate** — integer `Sum`/`Prod`/`Max`/`Min` on the serial base
///   are legal and bit-exact, so (unlike RowReduce) no `float dtype` assert.
/// - **`axis == rank - 1`** — v1 scans the innermost (contiguous) axis only; a
///   non-inner axis needs a strided scan skeleton (deferred), rejected here so the
///   miss is honest, not silently wrong.
///
/// Rejects `Mean` (not a monoid). `exclusive` and `reverse` are independently legal
/// and composable — there is no illegal combination, so nothing extra is asserted
/// for them (the on-device validator covers all four cells).
fn validate_scan(op: &OpDef, key: &StructureKey, axis: u8, reverse: bool, exclusive: bool) {
    let Access::Scan {
        op: scan_op,
        pre,
        post,
        ..
    } = &op.access
    else {
        unreachable!("validate_scan on a non-Scan op");
    };
    let name = &op.name;
    // Both flags are independently legal and composable — asserted-consumed so a
    // future reader sees they were considered (no illegal (exclusive, reverse) cell).
    let _ = (reverse, exclusive);

    // Mean is not a monoid (no identity a running prefix can carry) — reject before
    // anything else so the message is unambiguous. Prod is DELIBERATELY admitted
    // (the row-reduce Prod ban does NOT carry over — a scan folds cumprod serially).
    assert!(
        !matches!(scan_op, ReduceOp::Mean),
        "OpDef '{name}': Scan combine Mean is not a monoid (no identity/associative \
         running prefix) — v1 scans Sum/Prod/Max/Min only"
    );

    let n = op.n_inputs as usize;
    assert!(
        (1..MAX_OPERANDS).contains(&n),
        "Scan n_inputs {} out of [1, MAX_OPERANDS)",
        op.n_inputs
    );
    assert!(
        key.n_operands as usize == n + 1,
        "Scan expects n_inputs+1 operands (inputs then output); got {}",
        key.n_operands
    );
    let rank = key.rank as usize;
    assert!(rank >= 1, "Scan needs a scanned axis (rank >= 1)");
    let last = (rank - 1) as u8;

    // v1: the innermost (contiguous, trailing) axis only. The row-iteration
    // skeleton scans the dense inner dimension; a non-inner axis needs a strided
    // scan skeleton (deferred). `axis < rank` is subsumed (axis == rank-1 implies it).
    assert!(
        axis == last,
        "Scan v1 scans the innermost (contiguous) axis only: axis {axis} != rank-1 \
         ({last}) — a non-inner scan axis is a deferred follow-up (reject so the \
         miss is honest, not silently wrong)"
    );

    // Operand roles + layout legality (mirrors validate_row_reduce). Input 0 is the
    // row-streamed scanned tensor: `base = row*k` + the forward `idx = base+j` walk
    // assume a dense, forward last axis, so it must be Contig and NOT flipped (a
    // reversed operand keys |stride|-Contig + flipped and would read mirrored/OOB —
    // the reverse SCAN is the `reverse` flag, never a flipped operand).
    let mut input0_streamed = false;
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let o = key.operands[i];
        match rr_role(o, last) {
            RrRole::RowStreamed => {
                assert!(
                    o.contig == Contiguity::Contig,
                    "Scan streamed input {i} must be contiguous (base = row*k assumes a dense scanned axis)"
                );
                assert!(
                    !o.flipped,
                    "Scan streamed input {i} must not be reversed along an axis (idx = base+j reads forward-dense; a flipped view reads mirrored/OOB — use the `reverse` scan flag, not a flipped operand)"
                );
                input0_streamed |= i == 0;
            }
            RrRole::ColBroadcast => {
                assert!(
                    !o.bcast.is_set(last) && (0..last).all(|d| o.bcast.is_set(d)),
                    "Scan column input {i} must broadcast every outer axis and vary along the scanned axis (in_i[j])"
                );
                assert!(!o.flipped, "Scan column input {i} must not be reversed");
            }
            RrRole::RowScalar => {
                assert!(
                    rank >= 2 && (0..last).all(|d| !o.bcast.is_set(d)),
                    "Scan row-scalar input {i} needs rank >= 2 and no outer-axis broadcast (in_i[row]); an all-broadcast operand is a true scalar (bake as Const)"
                );
                assert!(!o.flipped, "Scan row-scalar input {i} must not be reversed");
            }
        }
    }
    assert!(
        input0_streamed,
        "Scan Input0 must be the row-streamed scanned tensor, not a column-broadcast weight or a per-row scalar"
    );
    let out = key.operands[n];
    assert!(
        out.bcast.is_empty() && out.contig == Contiguity::Contig && !out.flipped,
        "Scan output must be full-width forward-dense contiguous (empty bcast, not flipped)"
    );

    // Expression legality. `pre` (the per-element pre-map) runs BEFORE the fold, so
    // it must NOT read the running prefix (`Reduced` is rejected in `pre`); `post`
    // (the per-element epilogue) reads the running prefix as the single `Reduced(0)`
    // leaf. Coord is rejected upstream by `assert_coord_admissibility` (non-
    // elementwise); Param is f32-only (emitter). Input indices must be in range.
    fn check(e: &ScalarExpr, n_inputs: u8, allow_reduced: bool, ctx: &str, name: &str) {
        match e {
            ScalarExpr::Input(i) => assert!(
                *i < n_inputs,
                "Scan '{name}' {ctx} Input({i}) >= n_inputs {n_inputs}"
            ),
            ScalarExpr::Reduced(s) => {
                assert!(
                    allow_reduced,
                    "Scan '{name}' {ctx} must not read Reduced({s}) — the running prefix does not exist in the pre-map (it reads inputs only)"
                );
                assert!(
                    *s == 0,
                    "Scan '{name}' {ctx} Reduced({s}) — the running prefix is the single Reduced(0) leaf"
                );
            }
            ScalarExpr::Const(v) => assert!(
                v.is_finite(),
                "Scan '{name}' {ctx} Const must be finite, got {v}"
            ),
            ScalarExpr::Param(_) | ScalarExpr::Coord(_) => {}
            ScalarExpr::Unary(_, x) => check(x, n_inputs, allow_reduced, ctx, name),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check(a, n_inputs, allow_reduced, ctx, name);
                check(b, n_inputs, allow_reduced, ctx, name);
            }
        }
    }
    check(pre, op.n_inputs, false, "pre-map", name);
    check(post, op.n_inputs, true, "post-epilogue", name);
}

/// Validate an [`Access::Window`] op at build time (increment 7; AOT — a window
/// never crosses the JIT trust boundary, so a panic here is an author-error
/// backstop). Mirrors [`validate_scan`]'s operand-role + layout checks, with the
/// pooling-specific window-parameter gate:
///
/// - **`op != Prod`** — `Prod` is not a pool (a windowed product is niche and not
///   in the pool family); rejected. `Max`/`Min`/`Sum`/`Mean` are admitted.
/// - **`Mean` requires a float dtype** — an integer average has rounding
///   semantics (i32 `sum/count` truncates); avg_pool is float-only. `Max`/`Min`/
///   `Sum` are legal on the integer base too (max/min-pool select; sum-pool
///   wraps, matching the reduction Sum contract).
/// - **`axis == rank - 1`** — v1 pools the innermost (contiguous) axis only; a
///   non-inner axis needs a strided window skeleton (deferred).
/// - **`size`/`stride`/`dilation >= 1`** — a zero window / stride / dilation is a
///   degenerate (empty-window) config.
/// - **`2*pad_lo <= span` and `2*pad_hi <= span`** where the tap footprint is
///   `span = dilation*(size-1) + 1` — each edge window must overlap the input by
///   at least 1 tap (the bespoke `pool1d` `pad*2 <= window` constraint,
///   generalized to dilation). A pad exceeding half the window would place an
///   entire edge window in padding.
///
/// The `in_len → out_len` window arithmetic (`out_len = floor((in_len + pad_lo +
/// pad_hi - dilation*(size-1) - 1)/stride) + 1`) is a **runtime-launch-arg caller
/// precondition**, NOT a plan-time check: [`StructureKey`] deliberately abstracts
/// numeric extents away (it carries per-operand contiguity/broadcast/flip, never
/// shapes), so the plan gate cannot see `in_len`/`out_len` — the same trust level
/// as RowReduce's `k`/`n_out` and `Coord`'s exact-integer extent bound. The output
/// operand's LAYOUT (forward-dense contiguous, downsampled extent) IS keyed and is
/// checked here.
#[allow(clippy::too_many_arguments)]
fn validate_window(
    op: &OpDef,
    key: &StructureKey,
    axis: u8,
    size: u8,
    stride: u8,
    dilation: u8,
    pad_lo: u8,
    pad_hi: u8,
) {
    let Access::Window {
        op: wop, pre, post, ..
    } = &op.access
    else {
        unreachable!("validate_window on a non-Window op");
    };
    let name = &op.name;

    // Prod is not a pool; reject before anything else so the message is
    // unambiguous. Max/Min/Sum/Mean are the admitted window combines.
    assert!(
        !matches!(wop, ReduceOp::Prod),
        "OpDef '{name}': Window combine Prod is not a pool (a windowed product is \
         out of the pooling family) — v1 pools Max/Min/Sum/Mean only"
    );
    // Mean (avg_pool) is float-only: an integer average rounds (i32 sum/count
    // truncates). Max/Min/Sum ride the integer base too.
    if matches!(wop, ReduceOp::Mean) {
        assert!(
            matches!(
                key.dtype,
                ElementKind::F16
                    | ElementKind::Bf16
                    | ElementKind::F32
                    | ElementKind::F32Strict
                    | ElementKind::F64
            ),
            "OpDef '{name}': Window Mean (avg_pool) requires a float dtype, got \
             {:?} — an integer average has rounding semantics (miss honestly; \
             integer max/min/sum-pool are legal)",
            key.dtype
        );
    }

    let n = op.n_inputs as usize;
    assert!(
        (1..MAX_OPERANDS).contains(&n),
        "Window n_inputs {} out of [1, MAX_OPERANDS)",
        op.n_inputs
    );
    assert!(
        key.n_operands as usize == n + 1,
        "Window expects n_inputs+1 operands (inputs then output); got {}",
        key.n_operands
    );
    let rank = key.rank as usize;
    assert!(rank >= 1, "Window needs a pooled axis (rank >= 1)");
    let last = (rank - 1) as u8;

    // v1: the innermost (contiguous, trailing) axis only.
    assert!(
        axis == last,
        "Window v1 pools the innermost (contiguous) axis only: axis {axis} != \
         rank-1 ({last}) — a non-inner window axis is a deferred follow-up (reject \
         so the miss is honest, not silently wrong)"
    );

    // Window-parameter legality — a degenerate (empty-window) config is a reject.
    assert!(size >= 1, "Window size must be >= 1 (an empty window has no taps)");
    assert!(stride >= 1, "Window stride must be >= 1");
    assert!(dilation >= 1, "Window dilation must be >= 1");
    // span = the tap footprint (dilation*(size-1)+1); each edge window must overlap
    // the input by >= 1 tap, i.e. 2*pad <= span (bespoke `pool1d` pad*2 <= window,
    // generalized to dilation). u32 arithmetic avoids u8 overflow for large params.
    let span = u32::from(dilation) * (u32::from(size) - 1) + 1;
    assert!(
        2 * u32::from(pad_lo) <= span,
        "Window pad_lo {pad_lo} exceeds half the window span {span} \
         (dilation*(size-1)+1) — an entire low-edge window would fall in padding; \
         2*pad_lo <= span (bespoke pool1d pad*2 <= window)"
    );
    assert!(
        2 * u32::from(pad_hi) <= span,
        "Window pad_hi {pad_hi} exceeds half the window span {span} \
         (dilation*(size-1)+1) — an entire high-edge window would fall in padding; \
         2*pad_hi <= span (bespoke pool1d pad*2 <= window)"
    );

    // Operand roles + layout legality (mirrors validate_scan). Input 0 is the
    // row-streamed pooled tensor: `base = row*k_in` + the tap walk `idx = base+p`
    // assume a dense, forward inner axis, so it must be Contig and NOT flipped.
    let mut input0_streamed = false;
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let o = key.operands[i];
        match rr_role(o, last) {
            RrRole::RowStreamed => {
                assert!(
                    o.contig == Contiguity::Contig,
                    "Window streamed input {i} must be contiguous (base = row*k_in assumes a dense pooled axis)"
                );
                assert!(
                    !o.flipped,
                    "Window streamed input {i} must not be reversed along an axis (idx = base+p reads forward-dense; a flipped view reads mirrored/OOB)"
                );
                input0_streamed |= i == 0;
            }
            RrRole::ColBroadcast => {
                assert!(
                    !o.bcast.is_set(last) && (0..last).all(|d| o.bcast.is_set(d)),
                    "Window column input {i} must broadcast every outer axis and vary along the pooled axis (in_i[p])"
                );
                assert!(!o.flipped, "Window column input {i} must not be reversed");
            }
            RrRole::RowScalar => {
                assert!(
                    rank >= 2 && (0..last).all(|d| !o.bcast.is_set(d)),
                    "Window row-scalar input {i} needs rank >= 2 and no outer-axis broadcast (in_i[row]); an all-broadcast operand is a true scalar (bake as Const)"
                );
                assert!(!o.flipped, "Window row-scalar input {i} must not be reversed");
            }
        }
    }
    assert!(
        input0_streamed,
        "Window Input0 must be the row-streamed pooled tensor, not a column-broadcast weight or a per-row scalar"
    );
    // The output is full-width forward-dense contiguous (a DOWNSAMPLED extent — the
    // caller sizes it via the window formula, a runtime precondition — but the same
    // layout class as the input's inner axis).
    let out = key.operands[n];
    assert!(
        out.bcast.is_empty() && out.contig == Contiguity::Contig && !out.flipped,
        "Window output must be full-width forward-dense contiguous (empty bcast, not flipped)"
    );

    // Expression legality (mirrors validate_scan): `pre` (per-tap pre-map) runs
    // BEFORE the fold, so it must NOT read the window result (`Reduced` rejected in
    // `pre`); `post` (per-output epilogue) reads the result as the single
    // `Reduced(0)` leaf. Coord is rejected upstream (non-elementwise); Param is
    // f32-only (emitter). Input indices must be in range.
    fn check(e: &ScalarExpr, n_inputs: u8, allow_reduced: bool, ctx: &str, name: &str) {
        match e {
            ScalarExpr::Input(i) => assert!(
                *i < n_inputs,
                "Window '{name}' {ctx} Input({i}) >= n_inputs {n_inputs}"
            ),
            ScalarExpr::Reduced(s) => {
                assert!(
                    allow_reduced,
                    "Window '{name}' {ctx} must not read Reduced({s}) — the window result does not exist in the pre-map (it reads taps only)"
                );
                assert!(
                    *s == 0,
                    "Window '{name}' {ctx} Reduced({s}) — the window result is the single Reduced(0) leaf"
                );
            }
            ScalarExpr::Const(v) => assert!(
                v.is_finite(),
                "Window '{name}' {ctx} Const must be finite, got {v}"
            ),
            ScalarExpr::Param(_) | ScalarExpr::Coord(_) => {}
            ScalarExpr::Unary(_, x) => check(x, n_inputs, allow_reduced, ctx, name),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check(a, n_inputs, allow_reduced, ctx, name);
                check(b, n_inputs, allow_reduced, ctx, name);
            }
        }
    }
    check(pre, op.n_inputs, false, "pre-map", name);
    check(post, op.n_inputs, true, "post-epilogue", name);
}

/// Vector width in elements for a [`VecWidth`] bucket.
fn vec_width_elems(v: VecWidth) -> u32 {
    match v {
        VecWidth::V8 => 8,
        VecWidth::V4 => 4,
        VecWidth::V2 => 2,
        VecWidth::Scalar => 1,
    }
}

#[cfg(test)]
mod multi_output_validate {
    //! Increment-1 multi-output gate-rejection tests. Per the house rule these
    //! call `build_plan` DIRECTLY (an emitter panic would mask a gate mutation).
    use super::build_plan;
    use crate::ir::{input, konst, Access, OpDef, ReduceOp, ScalarExpr};
    use baracuda_kernels_types::{
        structure_key, ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey,
    };

    // n_operands contiguous 1D operands of `dtype`.
    fn key_dt(dtype: ElementKind, n_operands: usize) -> StructureKey {
        let a = OperandDesc::new(1, &[1024], &[1], dtype, 256);
        let ops: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89)
    }
    // F32 shorthand for the layout-shape rejection tests.
    fn key(n_operands: usize) -> StructureKey {
        key_dt(ElementKind::F32, n_operands)
    }

    fn mul_backward() -> OpDef {
        OpDef::elementwise_multi(
            "mul_backward",
            3,
            &[ElementKind::F32],
            vec![input(0) * input(2), input(0) * input(1)],
        )
    }

    #[test]
    fn valid_multi_output_builds() {
        // The happy path: 3 inputs + 2 outputs = 5 operands.
        let _ = build_plan(&mul_backward(), &key(5));
    }

    #[test]
    #[should_panic(expected = "broadcast")]
    fn output_broadcast_aliases_its_writes_rejected() {
        // A broadcast (stride-0) OUTPUT aliases its own writes across iteration
        // coordinates (a write race) and is not the full output shape.
        let a = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::F32, 256);
        // Output operand 4 (the last): broadcast inner axis (stride 0 → bcast).
        let bcast_out = OperandDesc::new(2, &[8, 4], &[1, 0], ElementKind::F32, 256);
        let ops = vec![a, a, a, a, bcast_out];
        let k = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
        let _ = build_plan(&mul_backward(), &k);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn operand_count_mismatch_rejected() {
        // The key must carry exactly n_inputs+n_outputs = 5 operands; a 4-operand
        // key is a declared shape/operand mismatch.
        let _ = build_plan(&mul_backward(), &key(4));
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_OPERANDS")]
    fn n_outputs_overflow_max_operands_rejected() {
        // 6 inputs + 3 outputs = 9 > MAX_OPERANDS(8).
        let op = OpDef::elementwise_multi(
            "over",
            6,
            &[ElementKind::F32],
            vec![input(0), input(1), input(2)],
        );
        let _ = build_plan(&op, &key(8));
    }

    #[test]
    #[should_panic(expected = "Reduced")]
    fn reduced_in_a_body_rejected() {
        // A multi-output body must not read Reduced (no reduction in an
        // elementwise map). Built directly (no constructor produces this).
        let mut op = OpDef::elementwise_multi(
            "bad",
            2,
            &[ElementKind::F32],
            vec![input(0) * input(1), input(0)],
        );
        op.extra_out_bodies[0] = ScalarExpr::Reduced(0);
        let _ = build_plan(&op, &key(4));
    }

    #[test]
    #[should_panic(expected = "Coord")]
    fn coord_in_a_body_rejected() {
        // A multi-output body must not read Coord (elementwise-map only in v1).
        let mut op = OpDef::elementwise_multi(
            "bad",
            2,
            &[ElementKind::F32],
            vec![input(0) * input(1), input(0)],
        );
        op.extra_out_bodies[0] = ScalarExpr::Coord(0);
        let _ = build_plan(&op, &key(4));
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn multi_output_on_a_reduction_rejected() {
        // extra_out_bodies on a non-Elementwise op — a fused reduction stores one
        // accumulator, not N bodies. Built directly by pushing onto the field.
        let mut op = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        op.extra_out_bodies.push(ScalarExpr::Input(0));
        assert!(matches!(op.access, Access::Reduction { .. }));
        // A reduction key = [input, output]; the Elementwise check fires first.
        let _ = build_plan(&op, &key(2));
    }

    #[test]
    #[should_panic(expected = "uniform output dtype")]
    fn hetero_multi_output_rejected() {
        // out_dtype = Some on a multi-output op — hetero multi-out is deferred.
        // Body 0's root is a valid Cmp so the pre-existing assert_valid_out_dtype
        // predicate gate PASSES; the rejection is the multi-output uniform-dtype
        // rule specifically.
        use crate::ir::BinaryOp;
        let mut op = OpDef::elementwise_multi(
            "bad",
            2,
            &[ElementKind::F32],
            vec![input(0).binary(BinaryOp::CmpLt, input(1)), input(0)],
        );
        op.out_dtype = Some(ElementKind::U8);
        let _ = build_plan(&op, &key(4));
    }

    #[test]
    #[should_panic(expected = "finite")]
    fn non_finite_const_in_a_body_rejected() {
        let op = OpDef::elementwise_multi(
            "bad",
            2,
            &[ElementKind::F32],
            vec![input(0) * input(1), input(0) + konst(f64::INFINITY)],
        );
        let _ = build_plan(&op, &key(4));
    }

    // ---- Review-caught gate asymmetry: EXTRA output bodies must be walked by the
    // half-Nextafter and int-op-admissibility gates too (they seeded only op.body).

    #[test]
    #[should_panic(expected = "requires LEAF")]
    fn composed_int_op_operand_in_an_extra_body_rejected_at_the_plan_gate() {
        use crate::ir::BinaryOp;
        let op = OpDef::elementwise_multi(
            "bad",
            3,
            &[ElementKind::U8],
            vec![
                input(0) + input(1),
                (input(0) + input(1)).binary(BinaryOp::Shr, input(2)),
            ],
        );
        let _ = build_plan(&op, &key_dt(ElementKind::U8, 5));
    }

    #[test]
    #[should_panic(expected = "must miss honestly")]
    fn half_nextafter_in_an_extra_body_rejected_at_the_plan_gate() {
        use crate::ir::BinaryOp;
        let op = OpDef::elementwise_multi(
            "bad",
            1,
            &[ElementKind::F16],
            vec![
                input(0) * input(0),
                input(0).binary(BinaryOp::Nextafter, input(0)),
            ],
        );
        let _ = build_plan(&op, &key_dt(ElementKind::F16, 3));
    }
}

#[cfg(test)]
mod rowreduce_role_validate {
    //! Increment-2 RowReduce role tests: `rr_role` classification units + gate
    //! tests that call `build_plan` DIRECTLY (the house rule — an emitter panic
    //! would mask a gate mutation). Covers the new `RowScalar` role, the lifted
    //! "inputs>0 must be column-broadcast" restriction, and the rejected-ambiguous
    //! cases.
    use super::{build_plan, rr_role, RrRole};
    use crate::ir::{input, reduced, OpDef, ReduceOp, ReduceStage};
    use baracuda_kernels_types::{
        structure_key, ArchSku, AxisMask, Contiguity, DivBucket, ElementKind, OpCategory,
        OperandDesc, OperandKey, VecWidth,
    };

    // A minimal OperandKey carrying only the broadcast mask + flip (all rr_role /
    // validate read for classification); contig is irrelevant to the role.
    fn opkey(bcast: u8, flipped: bool) -> OperandKey {
        OperandKey {
            contig: Contiguity::Broadcast,
            bcast: AxisMask(bcast),
            vec_width: VecWidth::Scalar,
            inner_div: DivBucket::Any,
            flipped,
        }
    }

    #[test]
    fn rr_role_classifies_the_three_geometries_and_the_ambiguous_case() {
        // rank 2 ⇒ feature (last) axis = 1.
        assert_eq!(rr_role(opkey(0b00, false), 1), RrRole::RowStreamed); // nothing bcast
        assert_eq!(rr_role(opkey(0b10, false), 1), RrRole::RowScalar); // feature bcast ⇒ per-row
        assert_eq!(rr_role(opkey(0b01, false), 1), RrRole::ColBroadcast); // outer bcast ⇒ per-col
        // A varying feature axis is NEVER a RowScalar (that is exactly ColBroadcast).
        assert_ne!(rr_role(opkey(0b01, false), 1), RrRole::RowScalar);
        // All-broadcast is ambiguous: classified RowScalar (last is set), then REJECTED
        // by validate's outer-axis-clear check (a true scalar is a Const, not an operand).
        assert_eq!(rr_role(opkey(0b11, false), 1), RrRole::RowScalar);
    }

    // A full-width row-streamed operand [256,128].
    fn stream() -> OperandDesc {
        OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256)
    }
    // A per-row scalar: [n_out,k]-presented, strides [1,0] (feature-axis broadcast,
    // outer varies dense).
    fn rowscalar() -> OperandDesc {
        OperandDesc::new(2, &[256, 128], &[1, 0], ElementKind::F32, 256)
    }

    #[test]
    fn second_row_streamed_input_builds() {
        // softmax bw: y, dy both row-streamed [n,k] + dx. The former guard rejected
        // input>0 unless column-broadcast; the lift makes this the point — it PASSES.
        let op = OpDef::row_reduce(
            "softmax_bw",
            2,
            &[ElementKind::F32],
            vec![ReduceStage { pre: (input(0) * input(1)).0, op: ReduceOp::Sum }],
            input(0) * (input(1) - reduced(0)),
        );
        let s = stream();
        let key = structure_key(OpCategory::Softmax, &[s, s, s], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    fn row_scalar_inputs_build() {
        // layer_norm bw: x, dy row-streamed; mean, rstd per-row scalars (used INSIDE a
        // stage pre — x_hat — and the epilogue).
        let x_hat = (input(0) - input(2)) * input(3);
        let op = OpDef::row_reduce(
            "layer_norm_bw",
            4,
            &[ElementKind::F32],
            vec![
                ReduceStage { pre: input(1).0, op: ReduceOp::Mean },
                ReduceStage { pre: (input(1) * x_hat.clone()).0, op: ReduceOp::Mean },
            ],
            input(3) * (input(1) - reduced(0) - x_hat * reduced(1)),
        );
        let s = stream();
        let rs = rowscalar();
        let key = structure_key(OpCategory::Normalization, &[s, s, rs, rs, s], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    // A 2-input op whose input 1 is the offending operand under test.
    fn probe_op() -> OpDef {
        OpDef::row_reduce(
            "probe",
            2,
            &[ElementKind::F32],
            vec![ReduceStage { pre: input(0).0, op: ReduceOp::Sum }],
            input(0) + input(1),
        )
    }

    #[test]
    #[should_panic(expected = "true scalar")]
    fn all_broadcast_input_rejected_as_a_true_scalar() {
        // strides [0,0] ⇒ both axes broadcast ⇒ classified RowScalar (last set) then
        // rejected by the outer-axis-clear check — a genuinely ambiguous mask.
        let s = stream();
        let allb = OperandDesc::new(2, &[256, 128], &[0, 0], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[s, allb, s], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "reversed")]
    fn row_scalar_flipped_rejected() {
        // feature-axis broadcast (RowScalar) but a NEGATIVE outer stride ⇒ flipped.
        let s = stream();
        let flipped_rs = OperandDesc::new(2, &[256, 128], &[-1, 0], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[s, flipped_rs, s], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "reversed")]
    fn flipped_row_streamed_input_rejected() {
        // Review #2 CRITICAL: a dense-but-REVERSED second row-streamed input
        // (empty bcast ⇒ RowStreamed; strides [128,-1] ⇒ |stride|-contig=Contig
        // but flipped=true). Pre-fix the RowStreamed branch checked only Contig
        // and accepted it, then the emitter read it forward (mirrored/OOB). Now
        // rejected, matching the ColBroadcast/RowScalar branches. Via build_plan
        // DIRECTLY so only the plan gate can fire.
        let s = stream();
        let flipped = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[s, flipped, s], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "every outer")]
    fn rank3_partial_middle_broadcast_rejected() {
        // rank 3, only the MIDDLE axis broadcast: feature (last) varies ⇒ ColBroadcast,
        // but it fails "must broadcast every outer axis" (axis 0 is not broadcast) — an
        // ambiguous partial broadcast, neither a clean column nor a row-scalar.
        let x = OperandDesc::new(3, &[4, 8, 16], &[128, 16, 1], ElementKind::F32, 256);
        let mid = OperandDesc::new(3, &[4, 8, 16], &[16, 0, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(3, &[4, 8, 16], &[128, 16, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[x, mid, out], ArchSku::Sm89);
        let _ = build_plan(&probe_op(), &key);
    }

    #[test]
    #[should_panic(expected = "Input0")]
    fn input0_as_row_scalar_rejected() {
        // Input 0 (the reduced tensor) presented feature-broadcast (a row-scalar) is
        // illegal — it must be the row-streamed x.
        let op = OpDef::row_reduce(
            "t",
            1,
            &[ElementKind::F32],
            vec![ReduceStage { pre: input(0).0, op: ReduceOp::Sum }],
            reduced(0),
        );
        let key = structure_key(OpCategory::Softmax, &[rowscalar(), stream()], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }
}

#[cfg(test)]
mod view_gate_validate {
    //! Item-01 layout-view gate-rejection + schedule-routing tests. Per the house
    //! rule these call `build_plan` DIRECTLY — an emitter panic would mask a gate
    //! mutation (the 0c lesson).
    use super::{build_plan, Schedule};
    use crate::ir::{input, OpDef, ReduceOp, View};
    use baracuda_kernels_types::{
        structure_key, ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, StructureKey,
    };

    // A rank-2 contiguous [128,256] f32 cell (1 input + 1 output) — the input keys
    // Contig + a vector width, so a view-free relu VECTORIZES here.
    fn contig_2d_key() -> StructureKey {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    }

    fn relu() -> OpDef {
        OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu())
    }
    fn relu_t() -> OpDef {
        relu().with_views(vec![View::Permute { perm: vec![1, 0] }])
    }

    #[test]
    fn transpose_view_forces_strided_off_the_vectorized_path() {
        let key = contig_2d_key();
        // Baseline: the view-free relu vectorizes on this contiguous cell.
        assert!(
            matches!(build_plan(&relu(), &key).schedule, Schedule::Vectorized { .. }),
            "precondition: the view-free relu must vectorize on a contiguous cell"
        );
        // A Permute view forces the STRIDED schedule (a transposed read is
        // non-contiguous; only the strided emitter folds the stride remap).
        assert_eq!(build_plan(&relu_t(), &key).schedule, Schedule::Strided);
        // And the plan carries the view through to the backend.
        assert_eq!(build_plan(&relu_t(), &key).views.len(), 1);
    }

    #[test]
    fn identity_views_route_exactly_like_view_free() {
        let key = contig_2d_key();
        // An all-Identity views vec is byte-identical to view-free: same schedule.
        let identated =
            OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu())
                .with_views(vec![View::Identity]);
        assert_eq!(
            build_plan(&relu(), &key).schedule,
            build_plan(&identated, &key).schedule,
            "all-Identity views must not change the schedule"
        );
        assert!(matches!(
            build_plan(&identated, &key).schedule,
            Schedule::Vectorized { .. }
        ));
    }

    #[test]
    fn same_rank_reshape_is_not_addressing_and_does_not_force_strided() {
        // A same-rank Reshape is an identity linear map (recognition/keying only) —
        // it must NOT force Strided (unlike Permute/Broadcast).
        let op = OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu())
            .with_views(vec![View::Reshape { producer_rank: 2 }]);
        assert!(matches!(
            build_plan(&op, &contig_2d_key()).schedule,
            Schedule::Vectorized { .. }
        ));
    }

    #[test]
    #[should_panic(expected = "true permutation")]
    fn invalid_permutation_rejected() {
        // perm [0,0] is not a permutation of 0..2 (duplicate axis).
        let op = relu().with_views(vec![View::Permute { perm: vec![0, 0] }]);
        let _ = build_plan(&op, &contig_2d_key());
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn permute_view_on_reduction_rejected() {
        // A non-Identity view on a Reduction op: rejected (reductions own their
        // axis machinery). Build the OpDef with a view via with_views.
        let op = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum)
            .with_views(vec![View::Permute { perm: vec![1, 0] }]);
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "orthogonal")]
    fn permute_with_broadcast_operand_rejected() {
        // Input 0 is broadcast on an axis AND carries a Permute view — v1 keeps
        // them orthogonal.
        let bcast_in = OperandDesc::new(2, &[128, 256], &[0, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[bcast_in, out], ArchSku::Sm89);
        let _ = build_plan(&relu_t(), &key);
    }

    #[test]
    #[should_panic(expected = "does not broadcast")]
    fn broadcast_view_disagreeing_with_key_rejected() {
        // The view declares axis 0 broadcast, but the key operand is dense (no
        // broadcast) — a lie the key-driven emitter would ignore.
        let op = relu().with_views(vec![View::Broadcast { bcast: AxisMask(0b01) }]);
        let _ = build_plan(&op, &contig_2d_key());
    }

    #[test]
    #[should_panic(expected = "rank-change")]
    fn rank_change_reshape_rejected() {
        // producer_rank 3 != iteration rank 2 — genuine rank-change emit (items
        // 03/10), out of item-01 scope.
        let op = relu().with_views(vec![View::Reshape { producer_rank: 3 }]);
        let _ = build_plan(&op, &contig_2d_key());
    }

    #[test]
    #[should_panic(expected = "deferred composition")]
    fn viewed_multi_output_rejected() {
        // A viewed input on a multi-output op is deferred in v1.
        let op = OpDef::elementwise_multi(
            "dual",
            1,
            &[ElementKind::F32],
            vec![input(0).relu(), input(0)],
        )
        .with_views(vec![View::Permute { perm: vec![1, 0] }]);
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a, a], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "must equal n_inputs")]
    fn views_len_mismatch_rejected() {
        // Bypass the with_views debug_assert to prove the plan gate's own
        // release-path length check (views.len() != n_inputs).
        let mut op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        op.views = vec![View::Permute { perm: vec![1, 0] }]; // len 1, n_inputs 2
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    fn reduction_with_identity_view_passes_through() {
        // A trivially-Identity view on a reduction is allowed (pass-through).
        let op = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum)
            .with_views(vec![View::Identity]);
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let _ = build_plan(&op, &key); // no panic — trivially-Identity pass-through
    }
}

#[cfg(test)]
mod gather_gate_validate {
    //! Increment-4 GATHER gate-rejection + schedule-routing tests. Per the house
    //! rule these call `build_plan` DIRECTLY — an emitter panic would mask a gate
    //! mutation (the 0c lesson).
    use super::{build_plan, Schedule};
    use crate::ir::{input, OobPolicy, OpDef, ReadIndex, ReduceOp};
    use baracuda_kernels_types::{
        structure_key, ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey,
    };

    // rank-2 gather cell: data [4,3] (input 0), index [4,3] (input 1), out [4,3].
    fn gather_key(idx_dt: ElementKind) -> StructureKey {
        let data = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], idx_dt, 256);
        let out = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        structure_key(OpCategory::BinaryElementwise, &[data, idx, out], ArchSku::Sm89)
    }

    #[test]
    fn gather_forces_the_strided_schedule() {
        let op = OpDef::gather("g", &[ElementKind::F32], 0, OobPolicy::Skip, ElementKind::I32);
        assert_eq!(build_plan(&op, &gather_key(ElementKind::I32)).schedule, Schedule::Strided);
        // The plan carries the read_index through to the backend.
        assert_eq!(build_plan(&op, &gather_key(ElementKind::I32)).read_index.len(), 2);
    }

    #[test]
    #[should_panic(expected = "index_dtype must be an integer")]
    fn non_integer_index_operand_rejected() {
        // A float index dtype is meaningless (the emitted load type must be int).
        let op = OpDef::gather("g", &[ElementKind::F32], 0, OobPolicy::Skip, ElementKind::F32);
        let _ = build_plan(&op, &gather_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "gather axis")]
    fn axis_ge_rank_rejected() {
        // axis 2 on a rank-2 cell.
        let op = OpDef::gather("g", &[ElementKind::F32], 2, OobPolicy::Skip, ElementKind::I32);
        let _ = build_plan(&op, &gather_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "index_operand")]
    fn index_operand_ge_n_inputs_rejected() {
        // index_operand 2 but only 2 inputs (valid indices 0,1).
        let op = OpDef::elementwise("g", 2, &[ElementKind::F32], input(0)).with_indexed(vec![
            ReadIndex::Indexed {
                index_operand: 2,
                axis: 0,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            },
            ReadIndex::Direct,
        ]);
        let _ = build_plan(&op, &gather_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn gather_on_a_reduction_rejected() {
        // A gather (Indexed read) on a Reduction op: rejected (reductions own their
        // axis machinery). read_index length must equal n_inputs (1 here).
        let op = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum)
            .with_indexed(vec![ReadIndex::Indexed {
                index_operand: 0,
                axis: 0,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            }]);
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "exactly one gathered input")]
    fn two_gathered_inputs_rejected() {
        // Two Indexed inputs — v1 emitter handles exactly one substituted axis.
        let op = OpDef::elementwise("g", 3, &[ElementKind::F32], input(0) + input(1)).with_indexed(
            vec![
                ReadIndex::Indexed {
                    index_operand: 2,
                    axis: 0,
                    oob: OobPolicy::Skip,
                    index_dtype: ElementKind::I32,
                },
                ReadIndex::Indexed {
                    index_operand: 2,
                    axis: 0,
                    oob: OobPolicy::Skip,
                    index_dtype: ElementKind::I32,
                },
                ReadIndex::Direct,
            ],
        );
        let data = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[data, data, idx, data],
            ArchSku::Sm89,
        );
        let _ = build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "gather \u{22a5} view")]
    fn gather_plus_view_on_the_same_input_rejected() {
        use crate::ir::View;
        // The gathered input 0 also carries a Permute view — gather ⊥ view in v1.
        let op = OpDef::gather("g", &[ElementKind::F32], 0, OobPolicy::Skip, ElementKind::I32)
            .with_views(vec![View::Permute { perm: vec![1, 0] }, View::Identity]);
        let _ = build_plan(&op, &gather_key(ElementKind::I32));
    }
}

#[cfg(test)]
mod scatter_gate_validate {
    //! Increment-5 SCATTER gate-rejection + schedule-routing tests. Per the house
    //! rule these call `build_plan` DIRECTLY — an emitter panic would mask a gate
    //! mutation (the 0c lesson).
    use super::{build_plan, Schedule};
    use crate::ir::{input, OobPolicy, OpDef, ReduceOp, WriteCombine, WriteIndex};
    use baracuda_kernels_types::{
        structure_key, ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey,
    };

    // rank-2 scatter cell: updates [4,3] (input 0), index [4,3] (input 1), dst
    // [4,3] (out slot). The dst extent along the scattered axis rides `sext` at
    // launch; here the key dst just supplies the strides/broadcast facts.
    fn scatter_key(idx_dt: ElementKind) -> StructureKey {
        let upd = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], idx_dt, 256);
        let dst = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        structure_key(OpCategory::BinaryElementwise, &[upd, idx, dst], ArchSku::Sm89)
    }

    #[test]
    fn scatter_forces_the_strided_schedule() {
        let op = OpDef::scatter("s", &[ElementKind::F32], 0, ElementKind::I32);
        assert_eq!(build_plan(&op, &scatter_key(ElementKind::I32)).schedule, Schedule::Strided);
        // The plan carries the write role through to the backend.
        assert!(!build_plan(&op, &scatter_key(ElementKind::I32)).write_index.is_direct());
    }

    #[test]
    #[should_panic(expected = "index_dtype must be an integer")]
    fn non_integer_scatter_index_rejected() {
        let op = OpDef::scatter("s", &[ElementKind::F32], 0, ElementKind::F32);
        let _ = build_plan(&op, &scatter_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "scatter axis")]
    fn scatter_axis_ge_rank_rejected() {
        let op = OpDef::scatter("s", &[ElementKind::F32], 2, ElementKind::I32);
        let _ = build_plan(&op, &scatter_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "index_operand")]
    fn scatter_index_operand_ge_n_inputs_rejected() {
        let op = OpDef::elementwise("s", 2, &[ElementKind::F32], input(0)).with_scatter(
            WriteIndex::ScatterIndexed {
                index_operand: 2,
                axis: 0,
                combine: WriteCombine::Assign,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            },
        );
        let _ = build_plan(&op, &scatter_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn scatter_on_a_reduction_rejected() {
        let op = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum)
            .with_scatter(WriteIndex::ScatterIndexed {
                index_operand: 0,
                axis: 0,
                combine: WriteCombine::AtomicAdd,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            });
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let _ = build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "not legal for output dtype")]
    fn float_atomic_max_rejected() {
        // AtomicMax on a float output is not native — integer-only in v1.
        let op = OpDef::elementwise("smax", 2, &[ElementKind::F32], input(0)).with_scatter(
            WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: 0,
                combine: WriteCombine::AtomicMax,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            },
        );
        let _ = build_plan(&op, &scatter_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "OOB policy must be Skip")]
    fn scatter_zerofill_rejected() {
        let op = OpDef::elementwise("s", 2, &[ElementKind::F32], input(0)).with_scatter(
            WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: 0,
                combine: WriteCombine::Assign,
                oob: OobPolicy::ZeroFill,
                index_dtype: ElementKind::I32,
            },
        );
        let _ = build_plan(&op, &scatter_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "identity value read")]
    fn composed_scatter_body_rejected() {
        // Review #5 CRITICAL: a composed scatter body would be silently DROPPED by
        // the deterministic gather-sum base (it sums the value operand directly).
        // A fused `relu(updates)` scatter_add is a v1 deferral. Via build_plan
        // DIRECTLY so only the plan gate can fire (not an emitter panic).
        let op = OpDef::elementwise("s", 2, &[ElementKind::F32], input(0).relu())
            .with_scatter(WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: 0,
                combine: WriteCombine::AtomicAdd,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            });
        let _ = build_plan(&op, &scatter_key(ElementKind::I32));
    }

    #[test]
    fn integer_scatter_add_is_admitted() {
        // Integer AtomicAdd (bincount-class) is deterministic and legal.
        let op = OpDef::scatter_add("isa", &[ElementKind::I32], 0, ElementKind::I32);
        let iupd = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let dst = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[iupd, idx, dst], ArchSku::Sm89);
        assert_eq!(build_plan(&op, &key).schedule, Schedule::Strided);
    }
}

#[cfg(test)]
mod scan_gate_validate {
    //! Increment-6 SCAN gate-rejection tests. Per the house rule these call
    //! `build_plan` DIRECTLY — an emitter panic would mask a gate mutation (the 0c
    //! lesson). Every `validate_scan` (and `assert_valid_out_dtype`) rejection has a
    //! test here; each is mutation-checked both directions by a targeted reverse-edit.
    use super::{build_plan, Schedule};
    use crate::ir::{input, konst, reduced, OpDef, ReduceOp};
    use baracuda_kernels_types::{
        structure_key, ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey,
    };

    // A rank-2 [256,128] scan cell: contiguous input + full-width contiguous output.
    fn scan_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }

    #[test]
    fn valid_scan_builds_all_monoids_and_flags() {
        // Every v1 monoid × inclusive/exclusive × forward/reverse builds; the
        // schedule is the serial base (block=false), innermost axis (rank-1 = 1).
        for op in [ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Max, ReduceOp::Min] {
            for reverse in [false, true] {
                for exclusive in [false, true] {
                    let sc =
                        OpDef::scan_simple("cum", &[ElementKind::F32], op, 1, reverse, exclusive);
                    let key = scan_key(ElementKind::F32);
                    let plan = build_plan(&sc, &key);
                    assert_eq!(
                        plan.schedule,
                        Schedule::Scan { op, axis: 1, reverse, exclusive, block: false }
                    );
                }
            }
        }
    }

    #[test]
    fn integer_scan_builds_sum_max_min() {
        // Integer Sum/Max/Min ride the serial base BitIdentical — validate_scan
        // does NOT copy validate_row_reduce's float-only gate.
        for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
            let sc = OpDef::scan_simple("cumi", &[ElementKind::I32], op, 1, false, false);
            let _ = build_plan(&sc, &scan_key(ElementKind::I32));
        }
    }

    #[test]
    fn prod_is_admitted_unlike_rowreduce() {
        // DELIBERATE difference from validate_row_reduce: Prod IS admitted (cumprod).
        let sc = OpDef::scan_simple("cumprod", &[ElementKind::F32], ReduceOp::Prod, 1, false, false);
        let key = scan_key(ElementKind::F32);
        let plan = build_plan(&sc, &key);
        assert!(matches!(
            plan.schedule,
            Schedule::Scan { op: ReduceOp::Prod, .. }
        ));
    }

    #[test]
    #[should_panic(expected = "not a monoid")]
    fn mean_combine_rejected() {
        // Mean is not a monoid — rejected (unlike Sum/Prod/Max/Min).
        let sc = OpDef::scan_simple("cummean", &[ElementKind::F32], ReduceOp::Mean, 1, false, false);
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "innermost")]
    fn non_inner_axis_rejected() {
        // v1 scans axis == rank-1 only; axis 0 on a rank-2 cell is a deferred
        // follow-up, rejected so the miss is honest.
        let sc = OpDef::scan_simple("cum0", &[ElementKind::F32], ReduceOp::Sum, 0, false, false);
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "must be contiguous")]
    fn non_contig_scanned_input_rejected() {
        // A transposed (column-major) input keys non-Contig on the scanned axis —
        // base = row*k assumes a dense scanned axis, so reject.
        let a = OperandDesc::new(2, &[256, 128], &[1, 256], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "must not be reversed")]
    fn flipped_operand_rejected() {
        // A dense-but-REVERSED input keys |stride|-Contig + flipped; idx = base+j
        // reads forward, so a flipped operand reads mirrored/OOB — reject (the
        // reverse SCAN is the `reverse` flag, never a flipped operand).
        let a = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "rejected under Scan")]
    fn hetero_out_dtype_rejected() {
        // A cumulative op is same-dtype as its input — a hetero out_dtype has no
        // exact store (assert_valid_out_dtype, runs before validate_scan).
        let mut sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        sc.out_dtype = Some(ElementKind::U8);
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    // ---- pre/post expression-legality gate (the `check` closure). Reachable only
    // via the public OpDef::scan (arbitrary pre/post); scan_simple can't exercise
    // it. Review-caught: a mutation neutralizing BOTH `check` call sites passed the
    // whole suite — these four tests now kill that mutant. ----

    #[test]
    #[should_panic(expected = "must not read Reduced")]
    fn pre_map_reading_reduced_rejected() {
        // The pre-map runs BEFORE the fold — the running prefix does not exist yet,
        // so a `Reduced` read in `pre` would lower to an undefined register.
        let sc = OpDef::scan(
            "cum", 1, &[ElementKind::F32], ReduceOp::Sum, 1, false, false,
            reduced(0), reduced(0),
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "single Reduced(0) leaf")]
    fn post_reading_reduced_nonzero_rejected() {
        // The running prefix is the single Reduced(0) leaf; Reduced(1) has no source.
        let sc = OpDef::scan(
            "cum", 1, &[ElementKind::F32], ReduceOp::Sum, 1, false, false,
            input(0), reduced(1),
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = ">= n_inputs")]
    fn pre_input_out_of_range_rejected() {
        // Input(5) with n_inputs = 1 — the kernel signature has no in5.
        let sc = OpDef::scan(
            "cum", 1, &[ElementKind::F32], ReduceOp::Sum, 1, false, false,
            input(5), reduced(0),
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "Const must be finite")]
    fn nonfinite_const_in_post_rejected() {
        // A non-finite Const in the epilogue (here NaN) has no valid emission.
        let sc = OpDef::scan(
            "cum", 1, &[ElementKind::F32], ReduceOp::Sum, 1, false, false,
            input(0),
            crate::ir::Expr(crate::ir::ScalarExpr::Add(
                Box::new(reduced(0).0),
                Box::new(konst(f64::NAN).0),
            )),
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    // ---- operand-role / output-layout gates (review-caught: only the RowStreamed
    // contig + flip guards were tested). ----

    #[test]
    #[should_panic(expected = "full-width forward-dense")]
    fn flipped_output_rejected() {
        // A reversed OUTPUT keys |stride|-Contig + flipped; the scan store is
        // forward-dense (out[base+j]) — a flipped output would write mirrored.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "Input0 must be the row-streamed")]
    fn input0_not_streamed_rejected() {
        // Input0 broadcast along the scanned (last) axis keys as a per-row scalar,
        // not the row-streamed scanned tensor — there is nothing to scan.
        let a = OperandDesc::new(2, &[256, 128], &[1, 0], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        let _ = build_plan(&sc, &key);
    }
}

#[cfg(test)]
mod window_gate_validate {
    //! Increment-7 WINDOW gate-rejection tests. Per the house rule these call
    //! `build_plan` DIRECTLY — an emitter panic would mask a gate mutation (the 0c
    //! lesson). Every `validate_window` (and `assert_valid_out_dtype`) rejection
    //! has a test here; each window-specific gate is mutation-checked both
    //! directions by a targeted reverse-edit.
    use super::{build_plan, Schedule};
    use crate::ir::{input, konst, reduced, OpDef, ReduceOp};
    use baracuda_kernels_types::{
        structure_key, ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey,
    };

    // A rank-2 window cell: contiguous input [256,128] + downsampled contiguous
    // output [256,64] (the extent is NOT keyed — only the layout class is — so any
    // Contig output stands in; the k_in→k_out arithmetic is a runtime precondition).
    fn window_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }

    // A plain single-input pool: size/stride/dilation/pad_lo/pad_hi/cip.
    #[allow(clippy::too_many_arguments)]
    fn pool(
        op: ReduceOp,
        dt: ElementKind,
        axis: u8,
        size: u8,
        stride: u8,
        dilation: u8,
        pad_lo: u8,
        pad_hi: u8,
        cip: bool,
    ) -> OpDef {
        OpDef::window_simple("pool", &[dt], op, axis, size, stride, dilation, pad_lo, pad_hi, cip)
    }

    #[test]
    fn valid_window_builds_all_combines_and_geometry() {
        // Max/Min/Sum/Mean × a spread of stride/dilation/pad build; the schedule is
        // the pooling Window on the innermost axis (rank-1 = 1).
        for op in [ReduceOp::Max, ReduceOp::Min, ReduceOp::Sum, ReduceOp::Mean] {
            for &(size, stride, dilation, pad) in
                &[(2u8, 2u8, 1u8, 0u8), (3, 1, 1, 1), (3, 2, 2, 2), (5, 3, 1, 2)]
            {
                let p = pool(op, ElementKind::F32, 1, size, stride, dilation, pad, pad, false);
                let key = window_key(ElementKind::F32);
                let plan = build_plan(&p, &key);
                assert_eq!(
                    plan.schedule,
                    Schedule::Window {
                        op,
                        axis: 1,
                        size,
                        stride,
                        dilation,
                        pad_lo: pad,
                        pad_hi: pad,
                        count_include_pad: false,
                    }
                );
            }
        }
    }

    #[test]
    fn integer_max_min_sum_pool_builds() {
        // Max/Min/Sum ride the integer base (select / wrapping sum); only Mean is
        // float-gated.
        for op in [ReduceOp::Max, ReduceOp::Min, ReduceOp::Sum] {
            let p = pool(op, ElementKind::I32, 1, 2, 2, 1, 0, 0, false);
            let _ = build_plan(&p, &window_key(ElementKind::I32));
        }
    }

    #[test]
    fn avg_pool_count_include_pad_flag_rides_schedule() {
        let p = pool(ReduceOp::Mean, ElementKind::F32, 1, 3, 1, 1, 1, 1, true);
        let key = window_key(ElementKind::F32);
        let plan = build_plan(&p, &key);
        assert!(matches!(
            plan.schedule,
            Schedule::Window { op: ReduceOp::Mean, count_include_pad: true, .. }
        ));
    }

    // ---- window-specific gates (each mutation-checked both directions) ----

    #[test]
    #[should_panic(expected = "not a pool")]
    fn prod_combine_rejected() {
        let p = pool(ReduceOp::Prod, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "requires a float dtype")]
    fn mean_on_integer_rejected() {
        // avg_pool (Mean) at an integer dtype rounds — reject (Max/Min/Sum are OK).
        let p = pool(ReduceOp::Mean, ElementKind::I32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::I32));
    }

    #[test]
    #[should_panic(expected = "innermost")]
    fn non_inner_axis_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 0, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "size must be >= 1")]
    fn zero_size_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 0, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "stride must be >= 1")]
    fn zero_stride_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 0, 1, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "dilation must be >= 1")]
    fn zero_dilation_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 0, 0, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "pad_lo")]
    fn pad_lo_over_half_span_rejected() {
        // span = dilation*(size-1)+1 = 1*(2-1)+1 = 2; pad_lo=2 ⇒ 2*2=4 > 2 → reject.
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 2, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "pad_hi")]
    fn pad_hi_over_half_span_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 2, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    // ---- operand-role / output-layout / expr gates (mirror validate_scan) ----

    #[test]
    #[should_panic(expected = "must be contiguous")]
    fn non_contig_pooled_input_rejected() {
        let a = OperandDesc::new(2, &[256, 128], &[1, 256], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "must not be reversed")]
    fn flipped_input_rejected() {
        let a = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "full-width forward-dense")]
    fn flipped_output_rejected() {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "Input0 must be the row-streamed")]
    fn input0_not_streamed_rejected() {
        // Input0 broadcast along the pooled (last) axis keys as a per-row scalar.
        let a = OperandDesc::new(2, &[256, 128], &[1, 0], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "rejected under Window")]
    fn hetero_out_dtype_rejected() {
        let mut p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 0, false);
        p.out_dtype = Some(ElementKind::U8);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "must not read Reduced")]
    fn pre_map_reading_reduced_rejected() {
        let p = OpDef::window(
            "pool", 1, &[ElementKind::F32], ReduceOp::Sum, 1, 2, 2, 1, 0, 0, false,
            reduced(0), reduced(0),
        );
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "single Reduced(0) leaf")]
    fn post_reading_reduced_nonzero_rejected() {
        let p = OpDef::window(
            "pool", 1, &[ElementKind::F32], ReduceOp::Sum, 1, 2, 2, 1, 0, 0, false,
            input(0), reduced(1),
        );
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = ">= n_inputs")]
    fn pre_input_out_of_range_rejected() {
        let p = OpDef::window(
            "pool", 1, &[ElementKind::F32], ReduceOp::Sum, 1, 2, 2, 1, 0, 0, false,
            input(5), reduced(0),
        );
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "Const must be finite")]
    fn nonfinite_const_in_post_rejected() {
        let p = OpDef::window(
            "pool", 1, &[ElementKind::F32], ReduceOp::Sum, 1, 2, 2, 1, 0, 0, false,
            input(0),
            crate::ir::Expr(crate::ir::ScalarExpr::Add(
                Box::new(reduced(0).0),
                Box::new(konst(f64::NAN).0),
            )),
        );
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    // ---- review-caught coverage: pin the pad <= half-span EQUALITY boundary
    // from BOTH sides (a `<=`->`<` mutation previously survived — every positive
    // geometry had 2*pad strictly < span, yet size=2/pad=1 is the PyTorch-legal
    // kernel_size=2/padding=1 pool and MUST build). ----

    #[test]
    fn pad_equal_to_half_span_builds() {
        // span = dilation*(size-1)+1 = 2; 2*pad_lo = 2*pad_hi = 2 == span — the
        // boundary case is legal (each edge window still overlaps >= 1 tap).
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 1, 1, false);
        let key = window_key(ElementKind::F32);
        let plan = build_plan(&p, &key);
        assert!(matches!(plan.schedule, Schedule::Window { pad_lo: 1, pad_hi: 1, .. }));
    }

    #[test]
    #[should_panic(expected = "pad_lo 2 exceeds half the window span")]
    fn pad_lo_one_past_half_span_rejected() {
        // span = 2; 2*pad_lo = 4 > 2 — one past the boundary rejects.
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 2, 0, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "pad_hi 2 exceeds half the window span")]
    fn pad_hi_one_past_half_span_rejected() {
        let p = pool(ReduceOp::Max, ElementKind::F32, 1, 2, 2, 1, 0, 2, false);
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    // ---- review-caught coverage: the ColBroadcast / RowScalar operand-role gates
    // were unreachable by any test (every pool was single-input). A 2-input window
    // (pre = input(0)*input(1), the per-column-weight shape validate_scan also
    // admits) reaches them for i >= 1. ----

    // A 2-input window cell: streamed input0 + a second operand with the given
    // key, then the downsampled output.
    fn two_input_pool(second: OperandDesc) -> (OpDef, StructureKey) {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, second, o], ArchSku::Sm89);
        let p = OpDef::window(
            "wpool", 2, &[ElementKind::F32], ReduceOp::Sum, 1, 2, 2, 1, 0, 0, false,
            input(0) * input(1),
            reduced(0),
        );
        (p, key)
    }

    #[test]
    fn weighted_pool_with_column_weight_builds() {
        // input1 = a per-column weight broadcast over rows (stride 0 on the outer
        // axis, varying along the pooled axis) — the ColBroadcast happy path. Full
        // iteration shape with a 0 stride marks the broadcast (the scan-test
        // convention).
        let w = OperandDesc::new(2, &[256, 128], &[0, 1], ElementKind::F32, 256);
        let (p, key) = two_input_pool(w);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "column input 1 must not be reversed")]
    fn flipped_column_weight_rejected() {
        // The same column weight REVERSED along the pooled axis: |stride|-varying +
        // flipped — the in_i[p] read would be mirrored.
        let w = OperandDesc::new(2, &[256, 128], &[0, -1], ElementKind::F32, 256);
        let (p, key) = two_input_pool(w);
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "row-scalar input 1 must not be reversed")]
    fn flipped_row_scalar_rejected() {
        // input1 = a per-row scalar (broadcast along the pooled axis) but REVERSED
        // along the outer axis — in_i[row] would read mirrored.
        let s = OperandDesc::new(2, &[256, 128], &[-1, 0], ElementKind::F32, 256);
        let (p, key) = two_input_pool(s);
        let _ = build_plan(&p, &key);
    }
}
