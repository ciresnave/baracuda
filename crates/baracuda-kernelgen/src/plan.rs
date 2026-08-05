//! Language-agnostic kernel plan — the schedule decision.
//!
//! [`build_plan`] turns an [`OpDef`] + a [`StructureKey`] cell into a neutral
//! [`KernelPlan`]: *what* to compute (the op body + dtype) and the *schedule*
//! (vectorized vs scalar) to compute it with. A [`crate::backend::Backend`]
//! lowers the plan to a concrete language. Choosing the schedule here, not in
//! the backend, keeps the decision shared across every backend.

use crate::ir::{
    Access, BaseOffset, OpDef, ReadIndex, ReduceOp, ReduceStage, ScalarExpr, SortLimit, SortOrder,
    SortOut, View, WriteCombine, WriteIndex, is_admissible_int_reduction_operand,
};
use baracuda_kernel_vocab::{
    AxisMask, Contiguity, ElementKind, MAX_OPERANDS, OperandKey, StructureKey, VecWidth,
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
    /// **Row sort / argsort** along the innermost axis (increment 8). Base
    /// (`bitonic` produced by the emitter's flag, never keyed) is a per-output
    /// RANK sort — one thread per output element scans its row and computes the
    /// element's rank under the total order (O(k²), no smem, no barriers, any
    /// `k`), [`crate::backend::VariantFidelity::BitIdentical`] and stable by
    /// construction. The cooperative smem **bitonic** pair-sort is a
    /// `cuda::row_sort_bitonic_variant` VARIANT (also `BitIdentical` — a pair
    /// sort is a pure permutation), contract-bounded to `k <= 1024` via
    /// `launch_note`. The policy (`order`/`stable`/`out`/`limit`) rides here (this
    /// enum is `Copy`); `build_plan` always derives the base.
    RowSort {
        /// Ascending / descending (NaN orders greatest either way).
        order: crate::ir::SortOrder,
        /// Always `true` in v1 (`validate_row_sort` enforces the pair-sort).
        stable: bool,
        /// Which buffer(s) the sort writes: `Values` (raw-bit permutation),
        /// `Indices` (the `I32` sort permutation), or `Both` (the fused
        /// two-output kernel — values to `out_val`, `I32` indices to `out_idx`).
        /// See [`crate::ir::SortOut`].
        out: SortOut,
        /// Whether the writeback is capped to a runtime top-`k_out` (increment
        /// 10): `Full` = today's whole-row sort; `TopK` = the first `k_out` ranks
        /// under `order` (topk = `Desc`, bottomk = `Asc`), a `long long` launch
        /// arg. ORTHOGONAL to `out`. See [`crate::ir::SortLimit`].
        limit: SortLimit,
    },
    /// **2-D im2col / unfold** along the two spatial axes (increment 11) — the
    /// conv-lowering expanding gather. One thread per OUTPUT cell (grid-stride over
    /// `N*C*kh*kw*oH*oW`) unravels its linear index into `(n,c,ki,kj,oh,ow)`,
    /// computes the source coord (`in_h = oh*stride_h - pad_h + ki*dilation_h`,
    /// `in_w` symmetric), and RAW-BIT copies the in-bounds input element or stores
    /// the typed zero for an out-of-bounds tap —
    /// [`crate::backend::VariantFidelity::BitIdentical`] (each output is an
    /// independent read-or-zero + store; no fold, no cross-output dependence). The
    /// conv geometry rides here (this enum is `Copy`); `body == Input(0)` (no
    /// pre/post). Never routed to a variant.
    Im2Col {
        /// `(kh, kw)` — window taps per spatial axis.
        kernel: (u8, u8),
        /// `(stride_h, stride_w)` — output downsampling stride per axis.
        stride: (u8, u8),
        /// `(pad_h, pad_w)` — zero-padding per axis (symmetric).
        pad: (u8, u8),
        /// `(dilation_h, dilation_w)` — inter-tap dilation per axis.
        dilation: (u8, u8),
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
    /// Per-EXTRA-output element dtypes ([`crate::ir::OpDef::extra_out_dtypes`], the
    /// hetero multi-output / dropout-class increment) — **empty for every
    /// single-output op and every UNIFORM multi-output op**, so the store emitters
    /// are byte-identical (`out_dtype_of(j)` resolves to the compute dtype for
    /// every output). A non-empty slice has length `extra_out_bodies.len()`; a
    /// `Some(U8)` entry is a per-output hetero keep-mask whose store converts the
    /// exact `0.0/1.0` `Cmp*` predicate to `unsigned char` at the store site.
    /// Backends read per-output dtype via [`KernelPlan::out_dtype_of`].
    pub extra_out_dtypes: &'a [Option<ElementKind>],
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
    /// Per-input **runtime base element offsets** ([`crate::ir::BaseOffset`], the
    /// BASE_OFFSET SLICE increment) — index `i` ↔ `Input(i)`. **Empty for every
    /// offset-free op** (every pre-increment constructor), so the strided emitter's
    /// per-operand address math is byte-identical; a non-empty slice has length
    /// `n_inputs`. Only a [`crate::ir::BaseOffset::Runtime`] entry changes emission
    /// (a `long long off{i}` launch arg bumped onto the operand base at kernel
    /// entry). Presence forces [`Schedule::Strided`] (a runtime offset invalidates
    /// the keyed alignment fact the vectorized path relies on). Validated at the top
    /// of [`build_plan`] ([`assert_valid_offsets`]) with an independent emitter
    /// backstop in [`crate::cuda::assert_offsets_lowerable`].
    pub base_offsets: &'a [BaseOffset],
    /// The **single output's** runtime base element offset
    /// ([`crate::ir::BaseOffset`]) — the output-side mirror of [`Self::write_index`].
    /// [`crate::ir::BaseOffset::Zero`] for every non-offset op (byte-identical
    /// output address); [`crate::ir::BaseOffset::Runtime`] adds a `long long offo`
    /// launch arg bumped onto the output base at kernel entry.
    pub out_base_offset: BaseOffset,
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

    /// The resolved element dtype of output `j` (the hetero multi-output /
    /// dropout-class increment). Output 0 is [`Self::out_dtype`] (already resolved
    /// `op.out_dtype.unwrap_or(key.dtype)`). Output `j > 0` reads
    /// [`Self::extra_out_dtypes`]`[j-1]`, resolving `None` **and the empty-slice
    /// case** (a uniform multi-output op stores an empty `extra_out_dtypes`) to the
    /// compute dtype [`Self::dtype`]. So for every uniform-multi/single-output plan
    /// `out_dtype_of(j) == dtype` and the store is byte-identical; a `Some(U8)`
    /// entry is the only hetero case. Backends use this to pick each output
    /// pointer's ctype + store conversion.
    #[must_use]
    pub fn out_dtype_of(&self, j: usize) -> ElementKind {
        if j == 0 {
            self.out_dtype
        } else {
            self.extra_out_dtypes
                .get(j - 1)
                .copied()
                .flatten()
                .unwrap_or(self.dtype)
        }
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
    assert_valid_offsets(op, key);
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
            let input0_contig = key.n_operands > 0 && key.operands[0].contig == Contiguity::Contig;
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
        Access::Contraction {
            ref axes,
            ref epilogue,
            ..
        } => {
            // v1 admissibility: the canonical rank-2 dense matmul cell OR its
            // rank-3 batched form, keyed with contraction facts and an epilogue
            // over the K-sum (+ optional fused bias) only.
            let m2 = crate::ir::ContractionAxes::matmul();
            let m3 = crate::ir::ContractionAxes::batched_matmul();
            let ax = (axes.lhs.as_slice(), axes.rhs.as_slice());
            assert!(
                ax == (m2.lhs.as_slice(), m2.rhs.as_slice())
                    || ax == (m3.lhs.as_slice(), m3.rhs.as_slice()),
                "contraction v1: canonical rank-2 matmul or rank-3 batched-matmul \
                 axis roles only"
            );
            assert!(
                key.contraction.is_some(),
                "contraction cell must carry ContractionKey facts (rank-2 dense \
                 row-major [M,K]x[K,N]->[M,N]); got token {}",
                key.to_token()
            );
            assert!(
                contraction_epilogue_admissible(epilogue, op.n_inputs),
                "contraction: epilogue may read only Reduced(0), constants, and the \
                 fused bias Input(2..n_inputs); got a disallowed leaf"
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
        // Increment 8 SORT_PERM: validate admissibility (mirrors the Scan/Window
        // arms), then derive the per-output RANK-sort BASE schedule. The
        // cooperative smem bitonic pair-sort is produced separately by
        // `cuda::row_sort_bitonic_variant` (a `lower_variants` filter), never
        // here. All payload fields are Copy — no `ref` borrow.
        Access::RowSort {
            order,
            stable,
            out,
            limit,
        } => {
            validate_row_sort(op, key, order, stable, out, limit);
            Schedule::RowSort {
                order,
                stable,
                out,
                limit,
            }
        }
        // Increment 11 IM2COL: validate admissibility (mirrors the Scan/Window/Sort
        // arms), then thread the conv geometry into the one-thread-per-output-cell
        // expanding-gather schedule. All geometry fields are Copy — no `ref` borrow;
        // `body == Input(0)` (no pre/post), and im2col has no vectorize/variant gate.
        Access::Im2Col {
            kernel,
            stride,
            pad,
            dilation,
        } => {
            validate_im2col(op, key, kernel, stride, pad, dilation);
            Schedule::Im2Col {
                kernel,
                stride,
                pad,
                dilation,
            }
        }
        Access::Elementwise => {
            let n = key.n_operands as usize;
            let all_contig = n > 0 && (0..n).all(|k| key.operands[k].contig == Contiguity::Contig);
            // The kernel vectorizes at the *narrowest* width every operand supports.
            let min_width = (0..n)
                .map(|k| vec_width_elems(key.operands[k].vec_width))
                .min()
                .unwrap_or(1);
            // Every output stores its compute-dtype value raw (no U8/hetero store):
            // output 0 uniform (`out_dtype` None) AND every extra output uniform
            // (`extra_out_dtypes` all None / empty). A hetero output has no packed
            // vector store, so it must fall to Scalar/Strided (G4). This is the
            // per-output generalization of the single-output 0b `out_dtype.is_none()`
            // vectorize gate — byte-identical for every uniform op.
            let all_outputs_uniform =
                op.out_dtype.is_none() && op.extra_out_dtypes.iter().all(Option::is_none);
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
            } else if op_has_offset(op) {
                // BASE_OFFSET SLICE (G4): a Runtime base offset shifts the effective
                // base pointer at kernel entry, so the keyed `align_bytes → VecWidth`
                // fact the float4/ld.128 vectorized path relies on becomes a LIE —
                // even for a width-multiple offset VALUE (the gate keys on PRESENCE, a
                // compile-time mask, never the runtime value). Force Strided: only its
                // single per-pointer entry-bump lowering is proven, and `emit_vectorized`
                // indexes width-element vectors that would silently ignore the bump.
                // This cannot be patched in the emitter alone. An offset-free op never
                // reaches here (op_has_offset false), so emission stays byte-identical.
                // Pinned by the force-strided positive test + the independent
                // `assert_offsets_lowerable` backstop.
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
            } else if min_width >= 2 && all_outputs_uniform {
                // A hetero-output (u8 predicate) kernel takes the SCALAR path
                // in v1 — never Vectorized: the vector/packed emitters load and
                // STORE one vector type, and a u8-mask output has no packed
                // store (a contiguous u8 output even keys V8, which would
                // otherwise widen `min_width` past the inputs'). `all_outputs_uniform`
                // (G4) covers BOTH the single-output predicate (`out_dtype` Some)
                // AND a hetero MULTI-output (any `extra_out_dtypes` Some, e.g. the
                // dropout U8 keep-mask): any U8 output forces Scalar/Strided so
                // `emit_vectorized_multi` never receives a hetero plan. Pinned by
                // the packed-fallback golden + `dropout_hetero_never_vectorizes`.
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
        extra_out_dtypes: &op.extra_out_dtypes,
        access: &op.access,
        views: &op.views,
        read_index: &op.read_index,
        write_index: &op.write_index,
        base_offsets: &op.base_offsets,
        out_base_offset: op.out_base_offset,
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

/// `true` if a contraction epilogue references only admissible leaves: `Reduced(0)`
/// (the K-sum), constants, and — when `n_inputs > 2` — the fused bias `Input(i)`
/// for `2 <= i < n_inputs` (a per-column `[N]` bias broadcast over the M rows).
/// Never `Input(0)`/`Input(1)` (lhs/rhs are consumed by the contraction, not the
/// epilogue), `Param`, another reduced stage, or `Coord`. For `n_inputs == 2` no
/// `Input` is admitted, so this is byte-identical to the plain-contraction
/// "Reduced(0) only" rule.
fn contraction_epilogue_admissible(e: &crate::ir::ScalarExpr, n_inputs: u8) -> bool {
    use crate::ir::ScalarExpr as E;
    match e {
        E::Reduced(0) | E::Const(_) => true,
        // A fused bias leaf: Input(i), 2 <= i < n_inputs.
        E::Input(i) => *i >= 2 && *i < n_inputs,
        // Coord rejects here too: a contraction epilogue iterates the (m, n)
        // output space, not an elementwise cell's — Coord's v1 semantics are
        // Elementwise-only (`assert_coord_admissibility` fires first with the
        // targeted message; this arm keeps the predicate honest regardless).
        E::Param(_) | E::Reduced(_) | E::Coord(_) => false,
        E::Unary(_, x) => contraction_epilogue_admissible(x, n_inputs),
        E::Add(a, b) | E::Sub(a, b) | E::Mul(a, b) | E::Div(a, b) | E::Binary(_, a, b) => {
            contraction_epilogue_admissible(a, n_inputs)
                && contraction_epilogue_admissible(b, n_inputs)
        }
        E::Select(c, a, b) => {
            contraction_epilogue_admissible(c, n_inputs)
                && contraction_epilogue_admissible(a, n_inputs)
                && contraction_epilogue_admissible(b, n_inputs)
        }
    }
}

/// Validate that a [`crate::ir::ContractionAxes`] role assignment is legal for
/// operands of the given ranks: the role vector must match the operand's rank,
/// each side must carry exactly the free role it's allowed to produce (`lhs` ⇒
/// one `FreeM`, no `FreeN`; `rhs` ⇒ one `FreeN`, no `FreeM`), exactly one
/// `ContractedK` per operand (multi-K-group contraction is deferred past v1),
/// and `lhs`/`rhs` must agree on how many `Batch` axes they carry. A pure
/// predicate over roles only — it does not look at concrete extents/strides,
/// and (as of this function) it is not yet consulted by [`build_plan`]'s
/// `Access::Contraction` arm; that wiring is a later increment.
// `#[allow(dead_code)]`: exercised by `contraction_role_validate::role_legality`
// today; a later increment calls this from `build_plan`'s `Access::Contraction`
// arm, at which point this attribute comes off.
#[allow(dead_code)]
pub(crate) fn validate_contraction_roles(
    axes: &crate::ir::ContractionAxes,
    lhs_rank: usize,
    rhs_rank: usize,
) -> Result<(), String> {
    use crate::ir::AxisRole::*;
    if axes.lhs.len() != lhs_rank || axes.rhs.len() != rhs_rank {
        return Err("role vector length must equal operand rank".into());
    }
    let count =
        |v: &[crate::ir::AxisRole], r: crate::ir::AxisRole| v.iter().filter(|&&x| x == r).count();
    if count(&axes.lhs, FreeM) != 1 {
        return Err("lhs must have exactly one FreeM".into());
    }
    if count(&axes.lhs, FreeN) != 0 {
        return Err("lhs must not carry FreeN".into());
    }
    if count(&axes.rhs, FreeN) != 1 {
        return Err("rhs must have exactly one FreeN".into());
    }
    if count(&axes.rhs, FreeM) != 0 {
        return Err("rhs must not carry FreeM".into());
    }
    // single K-group in v1
    if count(&axes.lhs, ContractedK) != 1 || count(&axes.rhs, ContractedK) != 1 {
        return Err("v1: exactly one ContractedK per operand (multi-group deferred)".into());
    }
    // batch correspondence
    if count(&axes.lhs, Batch) != count(&axes.rhs, Batch) {
        return Err("lhs and rhs must share the same batch-axis count".into());
    }
    Ok(())
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
/// op (built by [`OpDef::elementwise_multi`] or [`OpDef::elementwise_multi_hetero`])
/// the v1 rules, all honest AOT panics:
///
/// 1. **Access**: `Access::Elementwise` only. Multi-output is meaningful only for
///    an elementwise map; the reduction-class arms reject `extra_out_bodies`
///    (a fused reduction/contraction stores one accumulator, not N bodies).
/// 2. **Per-output dtype legality** (G1): every output `j` resolves a declared
///    dtype `d_j` (`out_dtype` for `j = 0`, `extra_out_dtypes[j-1]` for `j > 0`,
///    each `unwrap_or(key dtype)`). A UNIFORM output (`d_j == key dtype`) is
///    unconstrained (any elementwise body — the established path). A HETERO output
///    must be `U8` **and** have a `Cmp*`-root body (the FKC "comparison → U8 mask"
///    convention — only a predicate yields the exact `0.0/1.0` a u8 store
///    round-trips; the per-output generalization of `assert_valid_out_dtype`'s
///    Elementwise arm). The dropout fw U8 keep-mask is the v1 vehicle.
///    `extra_out_dtypes` is EMPTY (every extra output uniform) or one entry per
///    extra output. (A key-side cross-check of `d_j` vs the caller's OperandDesc
///    dtype is NOT possible — the structure key carries no per-operand dtype, only
///    the operand-0 primary; that agreement is an honest caller precondition, like
///    buffer aliasing below. G1 authored legality + the G5 emitter backstop hold
///    regardless.)
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
    // Per-output dtype length invariant: `extra_out_dtypes` is EMPTY (every extra
    // output uniform — the byte-identical uniform-multi path) or carries exactly
    // one dtype per extra output.
    assert!(
        op.extra_out_dtypes.is_empty() || op.extra_out_dtypes.len() == op.extra_out_bodies.len(),
        "OpDef '{name}': extra_out_dtypes (len {}) must be empty or match \
         extra_out_bodies (len {})",
        op.extra_out_dtypes.len(),
        op.extra_out_bodies.len()
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
            ScalarExpr::Select(c, a, b) => {
                check_body(c, n_inputs, name);
                check_body(a, n_inputs, name);
                check_body(b, n_inputs, name);
            }
        }
    }
    let bodies = op.output_bodies();
    for &e in &bodies {
        check_body(e, op.n_inputs, name);
    }

    // Output operands: the last n_outputs entries. Per output j we check (a) the
    // write shape (non-broadcast, non-flipped) and (b) the per-output dtype
    // legality (G1: uniform == key dtype, or a U8 keep-mask with a Cmp* root).
    //
    // The authored per-output dtype vs the caller's OperandDesc dtype (the "G2
    // wrong-bind cross-check" the brief called for) is NOT checkable here: the
    // structure key does NOT carry a per-operand dtype (`OperandKey` is
    // contig/bcast/vec_width/inner_div/flipped only; the caller's `OperandDesc`
    // dtype is consumed to compute `vec_width` and then discarded — `StructureKey`
    // keeps only the operand-0 primary `dtype`). Storing a per-operand output dtype
    // would need a `baracuda-kernels-types` schema change, which is out of scope
    // (kernels-types untouched, no `STRUCTURE_KEY_VERSION` bump). So per-output
    // dtype/caller agreement joins the honest CALLER PRECONDITIONS the key cannot
    // see (documented below with buffer aliasing and exact extents): the AOT op
    // author keys each hetero output's `OperandDesc` at the authored dtype (U8 for
    // a keep-mask). G1 (authored legality) + G5 (emitter backstop) hold regardless.
    for (j, &body) in bodies.iter().enumerate() {
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
        // G1: resolve the authored per-output dtype (`out_dtype` for output 0,
        // `extra_out_dtypes[j-1]` else; each `unwrap_or(key dtype)`).
        let d_j = if j == 0 {
            op.out_dtype.unwrap_or(key.dtype)
        } else {
            op.extra_out_dtypes
                .get(j - 1)
                .copied()
                .flatten()
                .unwrap_or(key.dtype)
        };
        if d_j != key.dtype {
            // A HETERO output is a U8 keep-mask ONLY, and only when its body ROOT
            // is a `Cmp*` predicate (the exact 0.0/1.0 the u8 store round-trips —
            // the per-output analog of `assert_valid_out_dtype`'s Elementwise arm).
            assert!(
                d_j == ElementKind::U8,
                "OpDef '{name}': multi-output output {j} declares hetero dtype \
                 {d_j:?} — the only legal per-output hetero dtype is U8 (a \
                 comparison-predicate keep-mask); a non-U8 / wider-than-compute \
                 side-output has no exact store conversion and is out of scope"
            );
            assert!(
                matches!(body, ScalarExpr::Binary(bop, _, _) if bop.is_cmp()),
                "OpDef '{name}': multi-output output {j} is U8 but its body ROOT is \
                 not a comparison (ScalarExpr::Binary with a Cmp* op) — only a \
                 predicate yields exactly 0.0/1.0, so any other body would truncate \
                 silently under the u8 store"
            );
        }
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
        Access::RowSort { .. } => "RowSort",
        Access::Im2Col { .. } => "Im2Col",
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

/// `true` if `op` carries any **runtime base offset** (a [`BaseOffset::Runtime`]
/// input OR a `Runtime` output; BASE_OFFSET SLICE). `false` for an offset-free op
/// (empty `base_offsets` + `Zero` output) and for a non-empty all-`Zero` op — the
/// byte-identical cases (presence is defined as `any(Runtime)`, so a normalized
/// all-`Zero` vec is identical to empty). THE single presence oracle: the plan
/// schedule gate, the `assert_valid_offsets` validator, the emitter's name suffix/
/// launch args/pointer bumps, and the `pattern` honest-miss all read it, so they
/// can never disagree about whether an op is offsetted.
pub(crate) fn op_has_offset(op: &OpDef) -> bool {
    op.base_offsets.iter().any(|b| !b.is_zero()) || !op.out_base_offset.is_zero()
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
        matches!(
            index_dtype,
            ElementKind::I32 | ElementKind::I64 | ElementKind::U32
        ),
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
            ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
                | ElementKind::I32
                | ElementKind::I64
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
        matches!(
            index_dtype,
            ElementKind::I32 | ElementKind::I64 | ElementKind::U32
        ),
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

/// BASE_OFFSET SLICE admissibility gate — runs at the TOP of [`build_plan`] (the
/// house pattern), with an independent emitter backstop in
/// [`crate::cuda::assert_offsets_lowerable`]. An offset-free op (empty
/// `base_offsets` + a `Zero` output) returns immediately, and a non-empty all-`Zero`
/// op returns after the length check — so the established path is unchanged and
/// emission stays byte-identical. For an op carrying a real `Runtime` offset the v1
/// rules, all honest AOT panics (an author/generator error — the offset value is a
/// per-launch scalar that never crosses the JIT trust boundary, so a panic is the
/// backstop, not a silent wrong-emit):
///
/// 1. **Shape (G1)**: a non-empty `base_offsets` ⇒ `len == n_inputs` (index `i` ↔
///    `Input(i)`; the same rule as `views`/`read_index`). Checked even for an
///    all-`Zero` non-empty vec — a malformed length is an author bug either way.
/// 2. **Access (G2)**: any `Runtime` ⇒ [`Access::Elementwise`] only. A
///    reduction/row-reduce/contraction/scan/window/sort op has role-aware
///    addressing (RowStreamed/ColBroadcast/RowScalar) that needs per-role offset
///    semantics — a single entry-bump is unsound there, so it is de-scoped.
/// 3. **Single-output (G3)**: any `Runtime` ⇒ `extra_out_bodies` empty. A runtime
///    offset on a multi-output op is a deferred composition (the multi-store DAG ×
///    the pointer bump is unproven) — rope uses Option B (one single-output launch
///    per lane parity).
///
/// The schedule-forcing rule (**G4**: any `Runtime` ⇒ [`Schedule::Strided`]) lives
/// in the `Access::Elementwise` arm of [`build_plan`] — a runtime offset shifts the
/// effective base pointer, so the keyed `align_bytes → VecWidth` fact the
/// `float4`/`ld.128` path relies on becomes a lie even for a width-multiple value.
/// **OOB is a caller precondition** (the k/n_out trust model, like gather/scatter's
/// `gext`/`sext` and RowSort's `k<=1024`): `off + <maximal declared-extent address>`
/// must land in-bounds; the emitted code forms no address before the bump and clamps
/// no offset — it is a `launch_note` + on-device-validated contract, not a checked
/// bound.
fn assert_valid_offsets(op: &OpDef, _key: &StructureKey) {
    if op.base_offsets.is_empty() && op.out_base_offset.is_zero() {
        return; // offset-free — every pre-increment op, unchanged.
    }
    let name = &op.name;
    // G1 arity: a non-empty vec must be length-matched (an all-Zero non-empty vec
    // is still a length claim the author must get right — checked before the
    // all-Zero early-out below).
    if !op.base_offsets.is_empty() {
        assert_eq!(
            op.base_offsets.len(),
            op.n_inputs as usize,
            "OpDef '{name}': base_offsets.len() ({}) must equal n_inputs ({})",
            op.base_offsets.len(),
            op.n_inputs
        );
    }
    if !op_has_offset(op) {
        return; // all-Zero == offset-free: byte-identical emission, no gate.
    }
    // G2 access: a runtime offset is Elementwise-only in v1.
    assert!(
        matches!(op.access, Access::Elementwise),
        "OpDef '{name}': a runtime BaseOffset is Access::Elementwise-only in v1 — a \
         {}-class op has role-aware addressing (RowStreamed/ColBroadcast/RowScalar) \
         that needs per-role offset semantics (a later increment)",
        access_tag(&op.access)
    );
    // G3 single-output: a runtime offset on a multi-output op is a deferred
    // composition (the multi-store DAG × the pointer bump is unproven).
    assert!(
        op.n_outputs() == 1,
        "OpDef '{name}': a runtime BaseOffset on a multi-output op ({} outputs) is a \
         deferred composition in v1 — rope uses Option B (one single-output launch \
         per lane parity)",
        op.n_outputs()
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
            ScalarExpr::Select(c, a, b) => walk(c) || walk(a) || walk(b),
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
        // Increment 8 SORT_PERM: `body` is pinned `Input(0)` (already in `exprs`)
        // and RowSort has no pre/post, so there is nothing extra to walk — the arm
        // is forced only to keep the honest-miss walk exhaustive/total.
        Access::RowSort { .. } => {}
        // Increment 11 IM2COL: `body` is pinned `Input(0)` (already in `exprs`) and
        // im2col has no pre/post — a pure raw-bit gather has no arithmetic to hide a
        // half `Nextafter` in, so there is nothing extra to walk; the arm is forced
        // only to keep the honest-miss walk exhaustive/total.
        Access::Im2Col { .. } => {}
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
/// 4. **int8 any/all/count reduction-predicate lift (this increment):** rule 2
///    rejects `Cmp*`/`Const` at every int dtype UNCONDITIONALLY — but
///    `any`/`all`/`count` are exactly a `Sum`/`Max`/`Min` fold over a fused
///    `Cmp*` predicate (`count = Sum(in != 0)`, `I64` out; `any`/`all` = a
///    `Max`/`Min` fold whose `post` casts back through `Cmp*(Reduced(0),
///    Const)`, `U8` out — see `assert_valid_out_dtype`'s `U8`/`I64` branches,
///    the two authored shapes). So rule 2's `Cmp*`/`Const` rejection is lifted
///    **only** while walking an [`Access::Reduction`]'s `body` or `post` at an
///    int dtype (`in_reduction` below) — the ELEMENTWISE arm is untouched and
///    keeps rejecting outright. The lift is further pinned to keep the
///    double-math hazard closed: every admitted `Cmp*`'s operands must be a
///    leaf `Input`/`Reduced` or a `Const` of EXACTLY `0.0`/`1.0` (the only
///    values `any`/`all`/`count` ever compare against, and the only values
///    that round-trip losslessly through `Const`'s dtype-oblivious f64
///    spelling — see `backend::const_lit`); anything else still declines.
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
    // Rule 4: true while walking an `Access::Reduction`'s `body`/`post` — the
    // any/all/count fused-predicate lift is scoped to exactly this Access arm
    // (NOT RowReduce/Contraction/Scan/Window, which stay float-only/out of
    // scope per the design spec), and covers BOTH `body` (the per-element
    // `count = in != 0` predicate) and `post` (the any/all boolean cast) —
    // both are pushed into `exprs` below and walked with the same flag.
    let in_reduction = matches!(op.access, Access::Reduction { .. });
    // `at_reduction_root` is `true` ONLY for the initial `walk` call on
    // `op.body` and on the reduction `post` (the `for e in exprs` loop
    // below), and `false` for every recursive descent (whole-branch-review
    // finding, closing the composed-predicate leak: rule 4's Cmp*/Const
    // lift must admit exactly what `cuda::emit_reduction`'s
    // `int_reduction_predicate` can integer-lower — the body/post ROOT
    // only. A `Cmp*` reached as a sub-node of Add/Sub/Mul (e.g.
    // `Add(Cmp(..), Cmp(..))`) is NOT the root, so without this flag the
    // `in_reduction && bop.is_cmp()` arm below would wrongly admit it here
    // while the emitter falls through to the FLOAT `binary_f32` speller for
    // that same non-root Cmp — reopening the `long long acc += float`
    // double-math hazard commit 6fdfe478 closed. Restricting admission to
    // the root keeps every SHIPPED catalog shape (`count`: Cmp at body
    // root; `any`/`all`: Cmp at body root AND post root) admitted while a
    // nested Cmp falls to the existing fail-closed `else` reject below —
    // the same behavior as before rule 4 existed.
    fn walk(
        e: &ScalarExpr,
        op_name: &str,
        dtype: ElementKind,
        int_dt: bool,
        elementwise: bool,
        in_reduction: bool,
        at_reduction_root: bool,
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
                walk(x, op_name, dtype, int_dt, elementwise, in_reduction, false);
            }
            ScalarExpr::Div(a, b) => {
                assert!(
                    !int_dt,
                    "op '{op_name}': integer division is rejected at {dtype:?} — the \
                     bespoke elementwise surface has no int div (binary_div_fp.cu is \
                     float-only) and C `/` division by zero is device-undefined; \
                     miss honestly"
                );
                walk(a, op_name, dtype, int_dt, elementwise, in_reduction, false);
                walk(b, op_name, dtype, int_dt, elementwise, in_reduction, false);
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
                    walk(a, op_name, dtype, int_dt, elementwise, in_reduction, false);
                    walk(b, op_name, dtype, int_dt, elementwise, in_reduction, false);
                } else if int_dt && in_reduction && at_reduction_root && bop.is_cmp() {
                    // Rule 4: an int-dtype Cmp* is admitted HERE ONLY — the
                    // reduction body/post predicate position of any/all/count
                    // (`in_reduction` is false for Access::Elementwise, so the
                    // elementwise int Cmp* rejection below is untouched), AND
                    // only when this Cmp* IS the body/post ROOT
                    // (`at_reduction_root`) — a Cmp* reached as a sub-node of
                    // Add/Sub/Mul (a composed predicate like `Add(Cmp(..),
                    // Cmp(..))`) falls through to the fail-closed `else` below,
                    // because `cuda::emit_reduction`'s `int_reduction_predicate`
                    // only integer-lowers a Cmp* at that same root position —
                    // admitting a non-root Cmp here would let it fall through to
                    // the FLOAT `binary_f32` speller at emit time (whole-branch-
                    // review finding: the composed-predicate float-launder).
                    //
                    // Double-math hazard close-out: `ScalarExpr::Const` always
                    // lowers via `backend::const_lit` as an f64 C literal
                    // (`format!("{v:?}")`) REGARDLESS of dtype — that spelling
                    // is dtype-oblivious by construction and cannot be fixed
                    // from this gate alone. So every operand admitted here must
                    // be a LEAF `Input`/`Reduced` (the real per-element or
                    // folded value) or a `Const` whose value is EXACTLY `0.0`
                    // or `1.0` — the only values any/all/count ever compare
                    // against (the fixed zero threshold; the implicit 0/1
                    // keep-mask), and the only ones that round-trip losslessly
                    // through the f64 spelling into int compute (no precision
                    // loss is possible for a 1-bit value, unlike an arbitrary
                    // literal). Any other Const value, or any composed
                    // sub-expression, is a genuine authoring error smuggling
                    // float math into an int kernel and is rejected outright —
                    // this leaf-or-{0,1} pin is what keeps the lift surgical.
                    // The admission test itself is centralized in
                    // `ir::is_admissible_int_reduction_operand` — the SAME
                    // helper `cuda::assert_no_int_div_or_const` and
                    // `cuda::emit_reduction`'s `int_reduction_predicate`/
                    // `int_cmp_operand` call, so this shape cannot drift
                    // between the gate and the emitter again. This match only
                    // survives to pick the right panic message for the two
                    // ways an operand can be inadmissible.
                    for (side, operand) in [("lhs", &**a), ("rhs", &**b)] {
                        if is_admissible_int_reduction_operand(operand) {
                            continue;
                        }
                        match operand {
                            ScalarExpr::Const(v) => panic!(
                                "op '{op_name}': int reduction-predicate {bop:?} \
                                 Const({v}) at {side} must be exactly 0 or 1 — any \
                                 other value risks the f64-literal double-math \
                                 hazard this gate exists to prevent (Const always \
                                 lowers as an f64 C literal; 0/1 are the only \
                                 values that round-trip exactly into int compute)"
                            ),
                            other => panic!(
                                "op '{op_name}': int reduction-predicate {bop:?} at \
                                 {side} requires a leaf Input/Reduced or a 0/1 \
                                 Const, got {other:?} — composed operands are out \
                                 of scope for the any/all/count fused-predicate \
                                 shape"
                            ),
                        }
                    }
                } else {
                    // Fail-closed default — also the landing spot for a
                    // reduction-position Cmp* that is NOT at the body/post root
                    // (`in_reduction` true but `at_reduction_root` false): same
                    // reject as pre-rule-4, so a composed predicate misses
                    // honestly instead of being laundered through the float
                    // speller.
                    assert!(
                        !int_dt,
                        "op '{op_name}': {bop:?} has no integer lowering — the \
                         bespoke elementwise surface instantiates it for float \
                         dtypes only, so int dtype {dtype:?} must miss honestly \
                         (a reduction-predicate Cmp* only lowers to integer at \
                         the body/post ROOT — nested inside Add/Sub/Mul it is \
                         out of scope for the any/all/count fused-predicate lift)"
                    );
                    walk(a, op_name, dtype, int_dt, elementwise, in_reduction, false);
                    walk(b, op_name, dtype, int_dt, elementwise, in_reduction, false);
                }
            }
            ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) | ScalarExpr::Mul(a, b) => {
                // Wrapping two's-complement at int dtypes — the audited-legal
                // set. Neither operand is the reduction body/post root anymore
                // (this node IS the root, if anything is), so both recurse with
                // `at_reduction_root: false` — this is precisely what closes the
                // composed-predicate leak: a Cmp* nested here (e.g. `Add(Cmp(..),
                // Cmp(..))`) now falls to the fail-closed reject above instead of
                // being admitted.
                walk(a, op_name, dtype, int_dt, elementwise, in_reduction, false);
                walk(b, op_name, dtype, int_dt, elementwise, in_reduction, false);
            }
            ScalarExpr::Select(c, a, b) => {
                // G1 (WHERE/SELECT): select is rejected OUTRIGHT at every int
                // dtype — v1 select is float-only (f32/f32s/f64/f16/bf16).
                // Rationale: bespoke cmp is `_fp`-only already, so a cmp-cond
                // select is float-only by transitivity, and admitting an int
                // select would drag in the 0c U8/I8 observer problem — the
                // cond's `!= 0` test OBSERVES the un-truncated promoted-int
                // value of a composed cond (value 256: nonzero inlined, 0 when
                // hoisted to an 8-bit tmp — one body, two results) while the
                // arms are value-transparent. Rejecting outright sidesteps it
                // with zero bespoke-parity loss (`where_dtype_fanout` int
                // coverage is a later increment). Legal in EVERY Access arm at
                // float dtypes (a select in a Reduction pre-expr is the
                // masked-sum shape), so no elementwise-only assert here.
                assert!(
                    !int_dt,
                    "op '{op_name}': Select has no integer lowering — v1 select is \
                     float-only (f32/f32s/f64/f16/bf16), so int dtype {dtype:?} must \
                     miss honestly (the 0c U8/I8 cond-observer question is unresolved \
                     and bespoke where int coverage is a later increment)"
                );
                walk(c, op_name, dtype, int_dt, elementwise, in_reduction, false);
                walk(a, op_name, dtype, int_dt, elementwise, in_reduction, false);
                walk(b, op_name, dtype, int_dt, elementwise, in_reduction, false);
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
        // Increment 8 SORT_PERM: `body` is pinned `Input(0)` (already in `exprs`,
        // trivially int-clean) and RowSort has no pre/post; an int sort just
        // permutes storage bits (no int arithmetic), so nothing extra to gate.
        Access::RowSort { .. } => {}
        // Increment 11 IM2COL: `body` is pinned `Input(0)` (already in `exprs`,
        // trivially int-clean) and im2col has no pre/post; a raw-bit gather does no
        // int arithmetic (no Div/Const/unary), so nothing extra to gate.
        Access::Im2Col { .. } => {}
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
        // `at_reduction_root: in_reduction` — true here exactly because
        // `exprs` is `[body, post]` precisely when `op.access` is
        // `Access::Reduction` (the only branch that pushes `post` instead of
        // an epilogue/stage; see the match above), i.e. this IS the initial
        // call on the reduction body/post root. For every other Access arm
        // `in_reduction` is already `false`, so the value is moot there (the
        // Cmp* admission arm requires `in_reduction` regardless).
        walk(
            e,
            &op.name,
            dtype,
            int_dt,
            elementwise,
            in_reduction,
            in_reduction,
        );
    }
}

#[cfg(test)]
mod int_reduction_predicate_gate_validate {
    //! Rule 4 (`assert_int_op_admissibility`, above): the int8 any/all/count
    //! admissibility lift. Two directions, both required: the elementwise int
    //! `Cmp*` rejection must survive UNCHANGED (negative control), and the
    //! reduction-predicate `Cmp*`/`Const` shape (`count = Sum(in != 0)`) must
    //! now be admitted (positive case). Calls `assert_int_op_admissibility`
    //! directly (it is private to this module) rather than `build_plan`, to
    //! isolate the gate from unrelated key/shape plumbing.
    use super::assert_int_op_admissibility;
    use crate::ir::{BinaryOp, OpDef, ReduceOp, input, konst, reduced};
    use baracuda_kernel_vocab::ElementKind;

    // Negative control: an elementwise int Cmp* op (the FKC "comparison → U8
    // mask" shape, `OpDef::elementwise_pred`) must still decline at U8 — the
    // reduction-predicate lift must NOT loosen the Elementwise arm.
    #[test]
    fn int_elementwise_cmp_still_declines() {
        let op = OpDef::elementwise_pred(
            "elem_ne_u8",
            1,
            &[ElementKind::U8],
            input(0).binary(BinaryOp::CmpNe, konst(0.0)),
        );
        let r = std::panic::catch_unwind(|| assert_int_op_admissibility(&op, ElementKind::U8));
        assert!(
            r.is_err(),
            "elementwise int Cmp must still decline after the reduction-post lift"
        );
    }

    // Positive case: the `count` shape — `reduce-Sum` over a `Cmp*` predicate
    // (`in != 0`) on an S8 input, with a hetero `I64` count output (the
    // `assert_valid_out_dtype` I64 branch: `Sum` + identity `Reduced(0)`
    // post). Must be admitted — this is exactly what was blocked before the
    // rule-4 lift.
    #[test]
    fn int_reduction_predicate_cmp_admitted() {
        let mut op = OpDef::reduction(
            "count_s8",
            1,
            &[ElementKind::S8],
            input(0).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Sum,
        );
        op.out_dtype = Some(ElementKind::I64);
        assert_int_op_admissibility(&op, ElementKind::S8); // must not panic
    }

    // Positive case, the other authored shape: `any` — a Max fold over a
    // per-element keep-mask predicate, with `post` casting the folded result
    // back through `Cmp*(Reduced(0), Const)` (the `assert_valid_out_dtype` U8
    // branch: post ROOT must be Cmp*). Exercises Cmp*/Const admission in BOTH
    // the reduction body AND post in the same op.
    #[test]
    fn int_reduction_any_body_and_post_cmp_admitted() {
        let mut op = OpDef::reduction_post(
            "any_u8",
            1,
            &[ElementKind::U8],
            input(0).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Max,
            reduced(0).binary(BinaryOp::CmpNe, konst(0.0)),
        );
        op.out_dtype = Some(ElementKind::U8);
        assert_int_op_admissibility(&op, ElementKind::U8); // must not panic
    }

    // Double-math-hazard guard: a Cmp* Const in the admitted reduction
    // position must be exactly 0/1 — any other literal is still rejected,
    // even though the surrounding shape (Sum-fold over a Cmp* predicate) is
    // otherwise the admitted count/any/all pattern.
    #[test]
    fn int_reduction_predicate_cmp_rejects_non_01_const() {
        let op = OpDef::reduction(
            "bad_threshold_s8",
            1,
            &[ElementKind::S8],
            input(0).binary(BinaryOp::CmpGt, konst(5.0)),
            ReduceOp::Sum,
        );
        let r = std::panic::catch_unwind(|| assert_int_op_admissibility(&op, ElementKind::S8));
        assert!(
            r.is_err(),
            "a non-0/1 Const threshold must still decline — the leaf-or-{{0,1}} \
             pin is what keeps the lift from reopening the double-math hazard"
        );
    }

    // Composition guard: a Cmp* operand that is itself a composed expression
    // (not a leaf Input/Reduced or a 0/1 Const) must still decline in the
    // reduction predicate position — composition is out of scope for v1.
    #[test]
    fn int_reduction_predicate_cmp_rejects_composed_operand() {
        let op = OpDef::reduction(
            "composed_operand_s8",
            1,
            &[ElementKind::S8],
            (input(0) + input(0)).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Sum,
        );
        let r = std::panic::catch_unwind(|| assert_int_op_admissibility(&op, ElementKind::S8));
        assert!(
            r.is_err(),
            "a composed Cmp* operand must still decline in the reduction \
             predicate position — v1 pins operands to leaves or 0/1 Consts"
        );
    }

    // Root-only guard (whole-branch-review finding): a COMPOSED predicate —
    // two Cmp* nodes combined by Add, i.e. the Cmp is reached as a SUB-node
    // of the reduction body rather than the body's own root — must still
    // decline. Rule 4's lift is scoped to exactly the body/post ROOT (the
    // only shape `count = Sum(in != 0)` / `any`/`all` ever author); a Cmp
    // nested inside Add/Sub/Mul is NOT admitted by the emitter's
    // `int_reduction_predicate` (which only inspects the root node), so if
    // the gate admitted it here, the nested Cmp would fall through to the
    // FLOAT `binary_f32` speller and get folded into the `long long`
    // integer accumulator — the exact double-math hazard commit 6fdfe478
    // closed, reopened for the composed-Cmp case. Before this fix this
    // panic did NOT fire (the nested Cmp was wrongly admitted).
    #[test]
    fn int_reduction_predicate_rejects_composed_cmp_not_at_root() {
        let op = OpDef::reduction(
            "composed_predicate_s8",
            1,
            &[ElementKind::S8],
            input(0).binary(BinaryOp::CmpNe, konst(0.0))
                + input(0).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Sum,
        );
        let r = std::panic::catch_unwind(|| assert_int_op_admissibility(&op, ElementKind::S8));
        assert!(
            r.is_err(),
            "a Cmp* reached as a sub-node of Add/Sub/Mul (not the reduction \
             body/post ROOT) must decline — the emitter only integer-lowers a \
             Cmp* at the root, so a nested Cmp admitted here would silently \
             fall through to the float speller and launder float math into the \
             int accumulator"
        );
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
        ScalarExpr::Select(c, a, b) => {
            expr_contains_coord(c) || expr_contains_coord(a) || expr_contains_coord(b)
        }
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
            ScalarExpr::Select(c, a, b) => {
                walk(c, op_name, dtype, rank, elementwise);
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
        // Increment 8 SORT_PERM: `body` is pinned `Input(0)` (no Coord), and
        // RowSort is non-elementwise (`elementwise == false`), so any Coord that
        // ever appeared would be rejected by the Coord arm; nothing extra to walk.
        Access::RowSort { .. } => {}
        // Increment 11 IM2COL: `body` is pinned `Input(0)` (no Coord), and im2col is
        // non-elementwise (`elementwise == false`), so any Coord would be rejected by
        // the Coord arm; im2col has no pre/post — nothing extra to walk.
        Access::Im2Col { .. } => {}
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
        // Increment 8/9 SORT_PERM: the ONLY hetero output under RowSort is the
        // ARGSORT index output — `SortOut::Indices` with `Some(I32)` (the sort
        // permutation, riding the single-output out_dtype precedent). A values-sort
        // (`Values`) and the fused two-output (`Both`) are dtype-preserving on
        // output 0 (out_dtype None ⇒ this fn returned early; for `Both` the I32
        // out_idx is emitter-hardwired, NOT off out_dtype). Reject any other
        // Some(_), and reject Some(I32) on a Values/Both sort. Double-gated: also
        // enforced (state ⇔ out_dtype) in `validate_row_sort`.
        Access::RowSort { out, .. } => assert!(
            matches!(out, SortOut::Indices) && od == ElementKind::I32,
            "OpDef '{}': out_dtype Some({od:?}) under RowSort is admitted only as \
             the argsort index output (SortOut::Indices requires Some(I32)); a \
             values-sort and the fused Both are dtype-preserving (out_dtype None)",
            op.name
        ),
        // Increment 11 IM2COL: a pure gather preserves the input dtype (the column
        // matrix is same-dtype as the NCHW input), so a hetero out_dtype has no exact
        // store — reject it, exactly like Scan/Window/RowReduce/Contraction.
        Access::Im2Col { .. } => panic!(
            "OpDef '{}': out_dtype = Some({od:?}) is rejected under Im2Col — a 2-D \
             im2col is a dtype-preserving raw-bit gather (the column matrix keeps the \
             input dtype), so there is no hetero store; use out_dtype = None",
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
            ScalarExpr::Select(c, a, b) => {
                walk(c, name);
                walk(a, name);
                walk(b, name);
            }
        }
    }
    walk(post, &op.name);
}

fn validate_row_reduce(
    stages: &[ReduceStage],
    epilogue: &ScalarExpr,
    n_inputs: u8,
    key: &StructureKey,
) {
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
        assert!(
            rank >= 2,
            "RowReduce with a multi-operand epilogue needs rank >= 2"
        );
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
            ScalarExpr::Const(v) => {
                assert!(v.is_finite(), "RowReduce Const must be finite, got {v}")
            }
            ScalarExpr::Unary(_, x) => check(x, n_inputs, max_reduced, in_stage, is_col),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                check(a, n_inputs, max_reduced, in_stage, is_col);
                check(b, n_inputs, max_reduced, in_stage, is_col);
            }
            ScalarExpr::Select(c, a, b) => {
                check(c, n_inputs, max_reduced, in_stage, is_col);
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
            ScalarExpr::Select(c, a, b) => {
                check(c, n_inputs, allow_reduced, ctx, name);
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
    assert!(
        size >= 1,
        "Window size must be >= 1 (an empty window has no taps)"
    );
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
                assert!(
                    !o.flipped,
                    "Window row-scalar input {i} must not be reversed"
                );
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
            ScalarExpr::Select(c, a, b) => {
                check(c, n_inputs, allow_reduced, ctx, name);
                check(a, n_inputs, allow_reduced, ctx, name);
                check(b, n_inputs, allow_reduced, ctx, name);
            }
        }
    }
    check(pre, op.n_inputs, false, "pre-map", name);
    check(post, op.n_inputs, true, "post-epilogue", name);
}

/// Validate an [`Access::Im2Col`] op at build time (increment 11; AOT — an im2col
/// never crosses the JIT trust boundary, so a panic here is an author-error
/// backstop). Unlike the rank-agnostic-innermost emitters (Scan/Window/RowSort
/// pin `axis == rank-1`), im2col has a FIXED rank-4 -> rank-3 conv layout, so this
/// gate PINS the operand ranks + layout the way `validate_window` pins the pooled
/// axis:
///
/// - **single input** (`n_inputs == 1`, no weight operand — a weighted window is
///   the deferred windowed-contraction), `n_operands == 2` (input then output).
/// - **input rank-4 forward-dense NCHW** — `key.rank == 4` (the input is also the
///   widest operand, so the shared `key.rank` is the input rank); input0 `Contig`,
///   `!flipped`, empty broadcast. The emitter's `(((n*C+c)*H_in+in_h)*W_in+in_w)`
///   address math assumes a dense row-major NCHW input.
/// - **output rank-3 forward-dense** — `Contig`, `!flipped`, empty broadcast. The
///   EXPANDED extent `[N, C*kh*kw, oH*oW]` is a runtime caller precondition (only
///   LAYOUT is keyed — the layout-only gate admits the expansion).
/// - **geometry legality** — `kh,kw >= 1`, `stride.* >= 1`, `dilation.* >= 1`
///   (`pad.*` is a `u8`, always `>= 0`). A zero kernel/stride/dilation is a
///   degenerate config.
/// - **`out_dtype == None`** (a pure gather preserves dtype;
///   `assert_valid_out_dtype` double-gates this) and **`body == Input(0)`** (a pure
///   raw-bit copy — no pre/post in v1).
///
/// The `(H_in,W_in) -> (oH,oW)` conv arithmetic
/// (`oH = (H_in + 2*pad_h - dilation_h*(kh-1) - 1)/stride_h + 1`, `oW` symmetric)
/// is a **runtime-launch-arg caller precondition**, NOT a plan-time check:
/// [`StructureKey`] deliberately abstracts numeric extents away (it carries
/// per-operand contiguity/broadcast/flip, never shapes), so the plan gate cannot
/// see `H_in`/`oH` — the same trust level as Window's `k_in -> k_out` and RowReduce's
/// `k`/`n_out`. It is on-device-validated via `initcheck` (full write + in-bounds
/// source reads); see the increment-11 brief §3/§6.
fn validate_im2col(
    op: &OpDef,
    key: &StructureKey,
    kernel: (u8, u8),
    stride: (u8, u8),
    pad: (u8, u8),
    dilation: (u8, u8),
) {
    let Access::Im2Col { .. } = &op.access else {
        unreachable!("validate_im2col on a non-Im2Col op");
    };
    let name = &op.name;
    let _ = pad; // pad.* is u8 (>= 0); baked into the caller's oH/oW, checked on-device.

    // G1 — single input (no weight operand), input + output operands only.
    assert!(
        op.n_inputs == 1,
        "Im2Col '{name}': v1 takes a SINGLE input (n_inputs {} != 1) — a weighted \
         window is the deferred windowed-contraction, not an im2col",
        op.n_inputs
    );
    assert!(
        key.n_operands == 2,
        "Im2Col '{name}': expects n_inputs+1 = 2 operands (input then output); got {}",
        key.n_operands
    );

    // G2 — input rank-4 forward-dense NCHW. The shared `key.rank` is the widest
    // operand rank; the rank-4 NCHW input is the widest (output is rank-3), so
    // `key.rank == 4` pins the input rank. A rank-3 (or other) input drives the max
    // below 4 and rejects here (the miss is honest, not silently wrong).
    assert!(
        key.rank == 4,
        "Im2Col '{name}': v1 pins a rank-4 NCHW input [N,C,H_in,W_in] (key.rank {} \
         != 4) — 1-D/3-D im2col are deferred rank variants",
        key.rank
    );
    let in0 = key.operands[0];
    assert!(
        in0.contig == Contiguity::Contig,
        "Im2Col '{name}': input 0 must be dense contiguous NCHW (the \
         (((n*C+c)*H_in+in_h)*W_in+in_w) address math assumes row-major)"
    );
    assert!(
        !in0.flipped,
        "Im2Col '{name}': input 0 must not be reversed along an axis (a flipped view \
         reads mirrored/OOB source coords)"
    );
    assert!(
        in0.bcast.is_empty(),
        "Im2Col '{name}': input 0 must be a full dense tensor (no broadcast axis — \
         every NCHW coordinate is read at a real stride)"
    );

    // G3 — output rank-3 forward-dense. Only LAYOUT is keyed (the expanded extent
    // [N, C*kh*kw, oH*oW] is a runtime precondition), so the gate admits the
    // expansion so long as the output is empty-bcast forward-dense contiguous.
    let out = key.operands[1];
    assert!(
        out.bcast.is_empty() && out.contig == Contiguity::Contig && !out.flipped,
        "Im2Col '{name}': output must be forward-dense contiguous (empty broadcast, \
         not flipped) — the [N, C*kh*kw, oH*oW] column matrix is written densely"
    );

    // G4 — geometry legality (a zero kernel/stride/dilation is degenerate).
    assert!(
        kernel.0 >= 1 && kernel.1 >= 1,
        "Im2Col '{name}': kernel (kh,kw) = ({},{}) must be >= 1 on each axis",
        kernel.0,
        kernel.1
    );
    assert!(
        stride.0 >= 1 && stride.1 >= 1,
        "Im2Col '{name}': stride (sh,sw) = ({},{}) must be >= 1 on each axis",
        stride.0,
        stride.1
    );
    assert!(
        dilation.0 >= 1 && dilation.1 >= 1,
        "Im2Col '{name}': dilation (dh,dw) = ({},{}) must be >= 1 on each axis",
        dilation.0,
        dilation.1
    );

    // G5 — dtype-preserving pure gather: out_dtype None (double-gated in
    // assert_valid_out_dtype) and body pinned to a raw Input(0) (no pre/post — a
    // composed body would need arithmetic the raw-bit copy does not emit).
    assert!(
        op.out_dtype.is_none(),
        "Im2Col '{name}': a pure gather preserves dtype — out_dtype must be None, got \
         {:?}",
        op.out_dtype
    );
    assert!(
        matches!(op.body, ScalarExpr::Input(0)),
        "Im2Col '{name}': body must be exactly Input(0) (a raw-bit copy) — v1 has no \
         pre/post map"
    );
}

/// Validate an [`Access::RowSort`] op at build time (increment 8; AOT — a sort
/// never crosses the JIT trust boundary, so a panic here is an author-error
/// backstop). Mirrors [`validate_scan`]/[`validate_window`]'s operand-role +
/// layout checks, with the sort-specific gates:
///
/// - **`stable == true` only** — v1 ALWAYS emits the stable pair-sort (`(key,
///   original-index)` with index tie-break), so stability is free; an unstable
///   network would emit byte-identical code under a different symbol (dead
///   keying) and admitting-then-ignoring the flag would be dishonest keying.
///   `stable: false` rejects.
/// - **dtype** — admit `F32|F32Strict|F64|F16|Bf16|I32|I64`; reject `U32`
///   (index-only, no value dtype) and `S8|U8` (v1 de-scope; the bespoke argsort
///   covers small ints; liftable by widening this gate + the validator cells).
/// - **out_dtype ↔ [`SortOut`] state coupling** — `Indices ⇒ out_dtype ==
///   Some(I32)`; `Values | Both ⇒ out_dtype == None` (Both's I32 out_idx is
///   emitter-hardwired, off the symbol, not out_dtype). Double-gated (also
///   `assert_valid_out_dtype`).
/// - **`body` must be exactly `Input(0)`** — v1 has no pre/post; this single
///   equality replaces the recursive `check` machinery of validate_scan/window
///   (it rejects Param/Coord/Reduced/Const/composed bodies in one stroke).
/// - **`extra_out_bodies` empty** — multi-output does not ride RowSort (also
///   caught by `assert_valid_multi_output`; double-gated). `Both` carries its
///   second buffer via the `SortOut` state + the 3-operand key, NEVER a body.
/// - **operand count** — `Values | Indices` carry `n_inputs + 1` operands
///   (`[in0, out]`); `Both` carries `n_inputs + 2` (`[in0, out_val, out_idx]`).
/// - **Input 0 layout** — `RowStreamed` + `Contig` + `!flipped` (the emitters
///   read `in0[base+j]` stride-free; a flipped/strided input reads mirrored/OOB).
/// - **Output layout** — every output (out / out_val AND out_idx for `Both`) is
///   empty bcast + `Contig` + `!flipped`. This gate is LAYOUT-only (it does NOT
///   assert the output WIDTH), so a `TopK` cell's narrower `[batch, k_out]` output
///   passes as written — no relaxation needed.
///
/// There is no axis field: RowSort is innermost-axis by definition (matching
/// Scan/Window's `axis == rank-1` posture). The bitonic variant's `k <= 1024`
/// bound is NOT checkable here (the structure key carries no numeric extents) —
/// it is a `launch_note` precondition + on-device-validated contract, the same
/// trust level as smemrow/blockscan; the base rank sort has no length limit.
///
/// - **`limit` (increment 10 TOPK/BOTTOMK)** — `Full` is today's whole-row sort;
///   `TopK` caps the writeback to the first `k_out` ranks. `k_out` is the OUT
///   operand's inner extent, a `long long` launch arg (the Window `(n_out, k_in,
///   k_out)` precedent) — the structure key carries NO numeric extent, so the
///   `k_out <= k_in` precondition is NOT expressible as a plan assert. It is a
///   runtime launch precondition, on-device-validated by `initcheck` (proves all
///   `k_out` slots written AND no over-write past `k_out`), the same trust tier as
///   the bitonic `k <= 1024`. `Indices`/`Both` under `TopK` inherit argsort's
///   `k_in <= 2^31-1` cap (the `I32` index).
fn validate_row_sort(
    op: &OpDef,
    key: &StructureKey,
    _order: SortOrder,
    stable: bool,
    out: SortOut,
    _limit: SortLimit,
) {
    let name = &op.name;

    // v1 emits only the stable pair-sort. Reject stable=false before anything else
    // so the message is unambiguous.
    assert!(
        stable,
        "OpDef '{name}': row_sort v1 emits only the stable pair-sort (stable=true); \
         unstable declined (an unstable network would emit byte-identical code under \
         a different symbol — dead keying)"
    );

    // dtype gate: the v1 comparable value set. U32 is index-only (no value dtype);
    // S8/U8 are a v1 de-scope (bespoke argsort covers small ints).
    assert!(
        matches!(
            key.dtype,
            ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
                | ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::I32
                | ElementKind::I64
        ),
        "OpDef '{name}': row_sort dtype {:?} is out of the v1 set \
         (F32/F32Strict/F64/F16/Bf16/I32/I64) — U32 is index-only and S8/U8 are a \
         v1 de-scope (miss honestly)",
        key.dtype
    );

    // out_dtype ↔ SortOut state coupling (double-gated with assert_valid_out_dtype).
    // G1: Indices ⇒ Some(I32); Values | Both ⇒ None (Both's I32 out_idx is
    // emitter-hardwired off the symbol, not carried by out_dtype).
    match out {
        SortOut::Indices => assert!(
            op.out_dtype == Some(ElementKind::I32),
            "OpDef '{name}': a row argsort (SortOut::Indices) produces an I32 index \
             output — out_dtype must be Some(I32), got {:?}",
            op.out_dtype
        ),
        SortOut::Values | SortOut::Both => assert!(
            op.out_dtype.is_none(),
            "OpDef '{name}': a values-sort (Values) / fused two-output (Both) is \
             dtype-preserving on output 0 — out_dtype must be None (Both's I32 \
             out_idx is emitter-hardwired), got {:?}",
            op.out_dtype
        ),
    }

    // body must be exactly Input(0): v1 has no pre/post, so a single equality check
    // rejects Param/Coord/Reduced/Const/composed bodies in one stroke.
    assert!(
        matches!(op.body, ScalarExpr::Input(0)),
        "OpDef '{name}': row_sort body must be exactly Input(0) (v1 has no pre/post \
         map); got {:?}",
        op.body
    );

    // Multi-output does not ride RowSort (also caught by assert_valid_multi_output).
    assert!(
        op.extra_out_bodies.is_empty(),
        "OpDef '{name}': row_sort is single-output (a permutation is not a ScalarExpr \
         body) — extra_out_bodies must be empty"
    );

    // Operand count + rank. G2: Values/Indices write one output ([in0, out]);
    // Both writes two ([in0, out_val, out_idx]).
    let n = op.n_inputs as usize;
    assert!(
        n == 1,
        "OpDef '{name}': row_sort streams exactly one row operand (n_inputs=1), got {n}"
    );
    let n_outputs = if matches!(out, SortOut::Both) { 2 } else { 1 };
    assert!(
        key.n_operands as usize == n + n_outputs,
        "row_sort expects n_inputs+{n_outputs} operands (input then {n_outputs} \
         output buffer(s)); got {}",
        key.n_operands
    );
    let rank = key.rank as usize;
    assert!(rank >= 1, "row_sort needs a sorted axis (rank >= 1)");
    let last = (rank - 1) as u8;

    // Input 0 layout: the row-streamed sorted tensor. The emitters read
    // `in0[base+j]` stride-free, so it must be Contig and NOT flipped (a
    // flipped/strided operand reads mirrored/OOB).
    let in0 = key.operands[0];
    assert!(
        rr_role(in0, last) == RrRole::RowStreamed,
        "OpDef '{name}': row_sort Input0 must be the row-streamed sorted tensor, not \
         a column-broadcast or per-row-scalar operand"
    );
    assert!(
        in0.contig == Contiguity::Contig,
        "OpDef '{name}': row_sort input must be contiguous (base = row*k assumes a \
         dense sorted axis)"
    );
    assert!(
        !in0.flipped,
        "OpDef '{name}': row_sort input must not be reversed along an axis (idx = \
         base+j reads forward-dense; a flipped view reads mirrored/OOB — use the \
         Desc order, not a flipped operand)"
    );

    // Output layout: full-width forward-dense contiguous. G3: output 0 (out for
    // Values/Indices, out_val for Both) always; and for Both, operand 2 (out_idx)
    // too — it is written forward-dense out_idx[base + r/p].
    let out0 = key.operands[n];
    assert!(
        out0.bcast.is_empty() && out0.contig == Contiguity::Contig && !out0.flipped,
        "OpDef '{name}': row_sort output must be forward-dense contiguous (empty \
         bcast, not flipped); the width may be < the input for a TopK cap"
    );
    if matches!(out, SortOut::Both) {
        let out_idx = key.operands[n + 1];
        assert!(
            out_idx.bcast.is_empty() && out_idx.contig == Contiguity::Contig && !out_idx.flipped,
            "OpDef '{name}': row_sort Both out_idx (operand 2) must be forward-dense \
             contiguous (empty bcast, not flipped); width may be < input for a TopK cap"
        );
    }
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
mod contraction_role_validate {
    //! `validate_contraction_roles` unit tests: role-vector length, per-operand
    //! role-count legality (exactly one FreeM/FreeN on the correct side, exactly
    //! one ContractedK — multi-group deferred to v2), and lhs/rhs batch-count
    //! agreement. Pure predicate, not yet wired into `build_plan` (Task 4).
    use super::validate_contraction_roles;

    #[test]
    fn role_legality() {
        use crate::ir::{AxisRole::*, ContractionAxes};
        // canonical constructors pass
        assert!(validate_contraction_roles(&ContractionAxes::matmul(), 2, 2).is_ok());
        assert!(validate_contraction_roles(&ContractionAxes::batched_matmul(), 3, 3).is_ok());
        // transposed role order passes (rhs [N,K] → roles [FreeN, ContractedK])
        let t = ContractionAxes {
            lhs: vec![FreeM, ContractedK],
            rhs: vec![FreeN, ContractedK],
        };
        assert!(validate_contraction_roles(&t, 2, 2).is_ok());
        // batch axis in the middle passes
        let mid = ContractionAxes {
            lhs: vec![FreeM, Batch, ContractedK],
            rhs: vec![Batch, ContractedK, FreeN],
        };
        assert!(validate_contraction_roles(&mid, 3, 3).is_ok());
        // illegal: two FreeM
        let bad_m = ContractionAxes {
            lhs: vec![FreeM, FreeM],
            rhs: vec![ContractedK, FreeN],
        };
        assert!(validate_contraction_roles(&bad_m, 2, 2).is_err());
        // illegal: FreeN on lhs
        let bad_n = ContractionAxes {
            lhs: vec![FreeN, ContractedK],
            rhs: vec![ContractedK, FreeN],
        };
        assert!(validate_contraction_roles(&bad_n, 2, 2).is_err());
        // illegal: mismatched batch count
        let bad_b = ContractionAxes {
            lhs: vec![Batch, FreeM, ContractedK],
            rhs: vec![ContractedK, FreeN],
        };
        assert!(validate_contraction_roles(&bad_b, 3, 2).is_err());
        // illegal: two K (multi-group deferred)
        let bad_k = ContractionAxes {
            lhs: vec![FreeM, ContractedK, ContractedK],
            rhs: vec![ContractedK, ContractedK, FreeN],
        };
        assert!(validate_contraction_roles(&bad_k, 3, 3).is_err());
        // illegal: role-vector length != rank
        assert!(validate_contraction_roles(&ContractionAxes::matmul(), 3, 2).is_err());
    }
}

#[cfg(test)]
mod multi_output_validate {
    //! Increment-1 multi-output gate-rejection tests + the hetero (dropout-class)
    //! per-output-dtype gates (G1/G3/G4). Per the house rule these call
    //! `build_plan` DIRECTLY (an emitter panic would mask a gate mutation).
    use super::{Schedule, build_plan};
    use crate::ir::{Access, BinaryOp, OpDef, ReduceOp, ScalarExpr, input, konst, param};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
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

    // dropout_fw (the hetero v1 vehicle): inputs x=in0, rand=in1; params
    // keep_prob=p0, scale=p1. Output 0 (value, F32) = x·(rand<keep_prob ? scale : 0);
    // output 1 (mask, U8) = the SAME `rand<keep_prob` Cmp node (shared, hoisted by
    // cross-body CSE).
    fn dropout_fw() -> OpDef {
        let cond = || input(1).binary(BinaryOp::CmpLt, param(0));
        OpDef::elementwise_multi_hetero(
            "dropout",
            2,
            &[ElementKind::F32],
            vec![
                (input(0) * cond().select(param(1), konst(0.0)), None),
                (cond(), Some(ElementKind::U8)),
            ],
        )
    }

    // dropout key: inputs x,rand (F32) then outputs value (F32), mask (U8).
    // `even` picks the V4-vectorizable extent vs an odd extent (scalar).
    fn dropout_key_contig(even: bool) -> StructureKey {
        let ext: i64 = if even { 1 << 20 } else { 1_000_003 };
        let f = OperandDesc::new(1, &[ext], &[1], ElementKind::F32, 256);
        let u = OperandDesc::new(1, &[ext], &[1], ElementKind::U8, 256);
        structure_key(OpCategory::BinaryElementwise, &[f, f, f, u], ArchSku::Sm89)
    }
    // Transposed (column-major) rank-2 dropout key — all strided.
    fn dropout_key_strided() -> StructureKey {
        let f = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::F32, 256);
        let u = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::U8, 256);
        structure_key(OpCategory::BinaryElementwise, &[f, f, f, u], ArchSku::Sm89)
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

    // ==== Hetero multi-output (dropout-class) per-output-dtype gates ====

    #[test]
    fn dropout_hetero_multi_builds() {
        // G1 POSITIVE (build_plan-direct): dropout_fw — output 0 F32 value +
        // output 1 U8 keep-mask (a Cmp-root body) — builds. 2 inputs + 2 outputs =
        // 4 operands; the last is U8. Pins that a legal U8-with-Cmp-root hetero
        // output is admitted (kill target for M1: deleting the G1 loop / reverting
        // to `out_dtype.is_none()` makes this panic).
        let op = dropout_fw();
        let k = dropout_key_contig(false);
        let plan = build_plan(&op, &k);
        assert_eq!(plan.n_outputs, 2);
        assert_eq!(plan.out_dtype_of(0), ElementKind::F32);
        assert_eq!(plan.out_dtype_of(1), ElementKind::U8);
    }

    #[test]
    #[should_panic(expected = "the only legal per-output hetero dtype is U8")]
    fn hetero_output_non_u8_is_rejected() {
        // G1 (M3): a non-U8 hetero side-output (I32 count beside a float) has no
        // exact compute→out store conversion — rejected. Built directly (no
        // constructor produces an I32 elementwise side-output).
        let op = OpDef::elementwise_multi_hetero(
            "bad",
            2,
            &[ElementKind::F32],
            vec![
                (input(0) * input(1), None),
                (
                    input(0).binary(BinaryOp::CmpLt, input(1)),
                    Some(ElementKind::I32),
                ),
            ],
        );
        // Key output 1 keyed I32 so the operand shape checks pass; the per-output
        // dtype-legality gate (G1) is what must fire. (A key-side dtype cross-check
        // is not implementable — the structure key carries no per-operand dtype.)
        let f = OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256);
        let i = OperandDesc::new(1, &[1024], &[1], ElementKind::I32, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[f, f, f, i], ArchSku::Sm89);
        let _ = build_plan(&op, &k);
    }

    #[test]
    #[should_panic(expected = "body ROOT is not a comparison")]
    fn hetero_u8_output_with_noncmp_root_is_rejected() {
        // G1 (M2): a U8 output whose body ROOT is a Mul (not a Cmp*) would truncate
        // a raw float silently under the u8 store — rejected.
        let op = OpDef::elementwise_multi_hetero(
            "bad",
            2,
            &[ElementKind::F32],
            vec![
                (input(0), None),
                (input(0) * input(1), Some(ElementKind::U8)),
            ],
        );
        let f = OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256);
        let u = OperandDesc::new(1, &[1024], &[1], ElementKind::U8, 256);
        let k = structure_key(OpCategory::BinaryElementwise, &[f, f, f, u], ArchSku::Sm89);
        let _ = build_plan(&op, &k);
    }

    #[test]
    fn uniform_multi_output_plan_is_unchanged() {
        // G3: a UNIFORM multi-output op (empty extra_out_dtypes) resolves every
        // output to the compute dtype — `out_dtype_of(j) == dtype` for all j — so
        // the store path is byte-identical to pre-hetero. Pins the no-op guarantee.
        let op = mul_backward();
        let k = key(5);
        let plan = build_plan(&op, &k);
        assert!(plan.extra_out_dtypes.is_empty());
        for j in 0..plan.n_outputs as usize {
            assert_eq!(plan.out_dtype_of(j), plan.dtype);
        }
    }

    // ==== G4: schedule forces Scalar/Strided when any output is hetero ====

    #[test]
    fn dropout_contig_key_yields_scalar_schedule() {
        // G4: a contig V4-eligible dropout key still takes SCALAR — the U8 mask has
        // no packed vector store. (Kill target for M5: keeping `out_dtype.is_none()`
        // in the vectorize gate would route this to Vectorized.)
        let op = dropout_fw();
        let k = dropout_key_contig(true);
        let plan = build_plan(&op, &k);
        assert_eq!(
            plan.schedule,
            Schedule::Scalar,
            "hetero forces scalar on contig"
        );
    }

    #[test]
    fn dropout_strided_key_yields_strided() {
        // G4: a non-contiguous (transposed) dropout key takes STRIDED.
        let op = dropout_fw();
        let k = dropout_key_strided();
        let plan = build_plan(&op, &k);
        assert_eq!(plan.schedule, Schedule::Strided);
    }

    #[test]
    fn uniform_multi_vec4_key_still_vectorizes() {
        // G4 no-regression: a UNIFORM multi-output op on a V4 contig key STILL
        // vectorizes (all_outputs_uniform true) — the hetero gate does not steal the
        // uniform vectorized path.
        let f = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let ops: Vec<_> = std::iter::repeat_n(f, 5).collect();
        let k = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
        let op = mul_backward();
        let plan = build_plan(&op, &k);
        assert!(
            matches!(plan.schedule, Schedule::Vectorized { .. }),
            "uniform multi still vectorizes: {:?}",
            plan.schedule
        );
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
    use super::{RrRole, build_plan, rr_role};
    use crate::ir::{OpDef, ReduceOp, ReduceStage, input, reduced};
    use baracuda_kernel_vocab::{
        ArchSku, AxisMask, Contiguity, DivBucket, ElementKind, OpCategory, OperandDesc, OperandKey,
        VecWidth, structure_key,
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
            vec![ReduceStage {
                pre: (input(0) * input(1)).0,
                op: ReduceOp::Sum,
            }],
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
                ReduceStage {
                    pre: input(1).0,
                    op: ReduceOp::Mean,
                },
                ReduceStage {
                    pre: (input(1) * x_hat.clone()).0,
                    op: ReduceOp::Mean,
                },
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
            vec![ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Sum,
            }],
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
            vec![ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Sum,
            }],
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
    use super::{Schedule, build_plan};
    use crate::ir::{OpDef, ReduceOp, View, input};
    use baracuda_kernel_vocab::{
        ArchSku, AxisMask, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
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
            matches!(
                build_plan(&relu(), &key).schedule,
                Schedule::Vectorized { .. }
            ),
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
        let identated = OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu())
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
        let key = structure_key(
            OpCategory::UnaryElementwise,
            &[bcast_in, out],
            ArchSku::Sm89,
        );
        let _ = build_plan(&relu_t(), &key);
    }

    #[test]
    #[should_panic(expected = "does not broadcast")]
    fn broadcast_view_disagreeing_with_key_rejected() {
        // The view declares axis 0 broadcast, but the key operand is dense (no
        // broadcast) — a lie the key-driven emitter would ignore.
        let op = relu().with_views(vec![View::Broadcast {
            bcast: AxisMask(0b01),
        }]);
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
    use super::{Schedule, build_plan};
    use crate::ir::{OobPolicy, OpDef, ReadIndex, ReduceOp, input};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // rank-2 gather cell: data [4,3] (input 0), index [4,3] (input 1), out [4,3].
    fn gather_key(idx_dt: ElementKind) -> StructureKey {
        let data = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], idx_dt, 256);
        let out = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        structure_key(
            OpCategory::BinaryElementwise,
            &[data, idx, out],
            ArchSku::Sm89,
        )
    }

    #[test]
    fn gather_forces_the_strided_schedule() {
        let op = OpDef::gather(
            "g",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::I32,
        );
        assert_eq!(
            build_plan(&op, &gather_key(ElementKind::I32)).schedule,
            Schedule::Strided
        );
        // The plan carries the read_index through to the backend.
        assert_eq!(
            build_plan(&op, &gather_key(ElementKind::I32))
                .read_index
                .len(),
            2
        );
    }

    #[test]
    #[should_panic(expected = "index_dtype must be an integer")]
    fn non_integer_index_operand_rejected() {
        // A float index dtype is meaningless (the emitted load type must be int).
        let op = OpDef::gather(
            "g",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::F32,
        );
        let _ = build_plan(&op, &gather_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "gather axis")]
    fn axis_ge_rank_rejected() {
        // axis 2 on a rank-2 cell.
        let op = OpDef::gather(
            "g",
            &[ElementKind::F32],
            2,
            OobPolicy::Skip,
            ElementKind::I32,
        );
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
        let op = OpDef::gather(
            "g",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::I32,
        )
        .with_views(vec![View::Permute { perm: vec![1, 0] }, View::Identity]);
        let _ = build_plan(&op, &gather_key(ElementKind::I32));
    }
}

#[cfg(test)]
mod scatter_gate_validate {
    //! Increment-5 SCATTER gate-rejection + schedule-routing tests. Per the house
    //! rule these call `build_plan` DIRECTLY — an emitter panic would mask a gate
    //! mutation (the 0c lesson).
    use super::{Schedule, build_plan};
    use crate::ir::{OobPolicy, OpDef, ReduceOp, WriteCombine, WriteIndex, input};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // rank-2 scatter cell: updates [4,3] (input 0), index [4,3] (input 1), dst
    // [4,3] (out slot). The dst extent along the scattered axis rides `sext` at
    // launch; here the key dst just supplies the strides/broadcast facts.
    fn scatter_key(idx_dt: ElementKind) -> StructureKey {
        let upd = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], idx_dt, 256);
        let dst = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        structure_key(
            OpCategory::BinaryElementwise,
            &[upd, idx, dst],
            ArchSku::Sm89,
        )
    }

    #[test]
    fn scatter_forces_the_strided_schedule() {
        let op = OpDef::scatter("s", &[ElementKind::F32], 0, ElementKind::I32);
        assert_eq!(
            build_plan(&op, &scatter_key(ElementKind::I32)).schedule,
            Schedule::Strided
        );
        // The plan carries the write role through to the backend.
        assert!(
            !build_plan(&op, &scatter_key(ElementKind::I32))
                .write_index
                .is_direct()
        );
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
        let op = OpDef::elementwise("s", 2, &[ElementKind::F32], input(0).relu()).with_scatter(
            WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: 0,
                combine: WriteCombine::AtomicAdd,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            },
        );
        let _ = build_plan(&op, &scatter_key(ElementKind::I32));
    }

    #[test]
    fn integer_scatter_add_is_admitted() {
        // Integer AtomicAdd (bincount-class) is deterministic and legal.
        let op = OpDef::scatter_add("isa", &[ElementKind::I32], 0, ElementKind::I32);
        let iupd = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let dst = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[iupd, idx, dst],
            ArchSku::Sm89,
        );
        assert_eq!(build_plan(&op, &key).schedule, Schedule::Strided);
    }
}

#[cfg(test)]
mod scan_gate_validate {
    //! Increment-6 SCAN gate-rejection tests. Per the house rule these call
    //! `build_plan` DIRECTLY — an emitter panic would mask a gate mutation (the 0c
    //! lesson). Every `validate_scan` (and `assert_valid_out_dtype`) rejection has a
    //! test here; each is mutation-checked both directions by a targeted reverse-edit.
    use super::{Schedule, build_plan};
    use crate::ir::{OpDef, ReduceOp, input, konst, reduced};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
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
                        Schedule::Scan {
                            op,
                            axis: 1,
                            reverse,
                            exclusive,
                            block: false
                        }
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
        let sc = OpDef::scan_simple(
            "cumprod",
            &[ElementKind::F32],
            ReduceOp::Prod,
            1,
            false,
            false,
        );
        let key = scan_key(ElementKind::F32);
        let plan = build_plan(&sc, &key);
        assert!(matches!(
            plan.schedule,
            Schedule::Scan {
                op: ReduceOp::Prod,
                ..
            }
        ));
    }

    #[test]
    #[should_panic(expected = "not a monoid")]
    fn mean_combine_rejected() {
        // Mean is not a monoid — rejected (unlike Sum/Prod/Max/Min).
        let sc = OpDef::scan_simple(
            "cummean",
            &[ElementKind::F32],
            ReduceOp::Mean,
            1,
            false,
            false,
        );
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
            "cum",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
            reduced(0),
            reduced(0),
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "single Reduced(0) leaf")]
    fn post_reading_reduced_nonzero_rejected() {
        // The running prefix is the single Reduced(0) leaf; Reduced(1) has no source.
        let sc = OpDef::scan(
            "cum",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
            input(0),
            reduced(1),
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = ">= n_inputs")]
    fn pre_input_out_of_range_rejected() {
        // Input(5) with n_inputs = 1 — the kernel signature has no in5.
        let sc = OpDef::scan(
            "cum",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
            input(5),
            reduced(0),
        );
        let _ = build_plan(&sc, &scan_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "Const must be finite")]
    fn nonfinite_const_in_post_rejected() {
        // A non-finite Const in the epilogue (here NaN) has no valid emission.
        let sc = OpDef::scan(
            "cum",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
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
    use super::{Schedule, build_plan};
    use crate::ir::{OpDef, ReduceOp, input, konst, reduced};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
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
        OpDef::window_simple(
            "pool",
            &[dt],
            op,
            axis,
            size,
            stride,
            dilation,
            pad_lo,
            pad_hi,
            cip,
        )
    }

    #[test]
    fn valid_window_builds_all_combines_and_geometry() {
        // Max/Min/Sum/Mean × a spread of stride/dilation/pad build; the schedule is
        // the pooling Window on the innermost axis (rank-1 = 1).
        for op in [ReduceOp::Max, ReduceOp::Min, ReduceOp::Sum, ReduceOp::Mean] {
            for &(size, stride, dilation, pad) in &[
                (2u8, 2u8, 1u8, 0u8),
                (3, 1, 1, 1),
                (3, 2, 2, 2),
                (5, 3, 1, 2),
            ] {
                let p = pool(
                    op,
                    ElementKind::F32,
                    1,
                    size,
                    stride,
                    dilation,
                    pad,
                    pad,
                    false,
                );
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
            Schedule::Window {
                op: ReduceOp::Mean,
                count_include_pad: true,
                ..
            }
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
            "pool",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
            reduced(0),
            reduced(0),
        );
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "single Reduced(0) leaf")]
    fn post_reading_reduced_nonzero_rejected() {
        let p = OpDef::window(
            "pool",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
            input(0),
            reduced(1),
        );
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = ">= n_inputs")]
    fn pre_input_out_of_range_rejected() {
        let p = OpDef::window(
            "pool",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
            input(5),
            reduced(0),
        );
        let _ = build_plan(&p, &window_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "Const must be finite")]
    fn nonfinite_const_in_post_rejected() {
        let p = OpDef::window(
            "pool",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
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
        assert!(matches!(
            plan.schedule,
            Schedule::Window {
                pad_lo: 1,
                pad_hi: 1,
                ..
            }
        ));
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
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[a, second, o],
            ArchSku::Sm89,
        );
        let p = OpDef::window(
            "wpool",
            2,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
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

#[cfg(test)]
mod im2col_gate_validate {
    //! Increment-11 IM2COL gate-rejection tests. Per the house rule these call
    //! `build_plan` DIRECTLY — an emitter panic would mask a gate mutation (the 0c
    //! lesson). Every `validate_im2col` (and `assert_valid_out_dtype`) rejection has
    //! a test here; each gate is mutation-checked both directions by a targeted
    //! reverse-edit.
    use super::{Schedule, build_plan};
    use crate::ir::{Access, OpDef};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // A canonical im2col cell: rank-4 NCHW input [2,3,4,4] dense + rank-3 output
    // [2,8,8] dense. Extents are NOT keyed (only rank + layout), so any dense shapes
    // stand in; the (H,W)->(oH,oW) conv arithmetic is a runtime precondition.
    fn im2col_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], dt, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }

    fn im2col(kernel: (u8, u8), stride: (u8, u8), pad: (u8, u8), dilation: (u8, u8)) -> OpDef {
        OpDef::im2col_2d("unfold", ElementKind::F32, kernel, stride, pad, dilation)
    }

    // ---- G0: schedule threading ----

    #[test]
    fn im2col_cell_builds() {
        // A rank-4-in / rank-3-out Im2Col cell builds and threads the geometry into
        // Schedule::Im2Col (3x3, stride 1, pad 1, dilation 1 — the canonical conv).
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let key = im2col_key(ElementKind::F32);
        let plan = build_plan(&p, &key);
        assert_eq!(
            plan.schedule,
            Schedule::Im2Col {
                kernel: (3, 3),
                stride: (1, 1),
                pad: (1, 1),
                dilation: (1, 1),
            }
        );
        assert!(matches!(plan.access, Access::Im2Col { .. }));
    }

    #[test]
    fn im2col_dilated_cell_builds() {
        // A dilated, non-square (kh != kw), strided cell builds — the geometry rides
        // the schedule verbatim.
        let p = im2col((3, 5), (2, 1), (2, 0), (2, 3));
        let key = im2col_key(ElementKind::F32);
        let plan = build_plan(&p, &key);
        assert_eq!(
            plan.schedule,
            Schedule::Im2Col {
                kernel: (3, 5),
                stride: (2, 1),
                pad: (2, 0),
                dilation: (2, 3),
            }
        );
    }

    #[test]
    fn im2col_builds_every_dtype() {
        // A pure raw-bit gather is dtype-agnostic — f32/f64/f16/bf16/i32/i64 all build.
        for dt in [
            ElementKind::F32,
            ElementKind::F64,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::I32,
            ElementKind::I64,
        ] {
            let p = OpDef::im2col_2d("unfold", dt, (3, 3), (1, 1), (1, 1), (1, 1));
            let _ = build_plan(&p, &im2col_key(dt));
        }
    }

    // ---- G1: operand count + single input ----

    #[test]
    #[should_panic(expected = "expects n_inputs+1 = 2 operands")]
    fn im2col_key_needs_two_operands() {
        // A 3-operand key (input + two outputs) is not an im2col shape — reject.
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "takes a SINGLE input")]
    fn im2col_rejects_second_input() {
        // A weight operand (n_inputs == 2) is the deferred windowed-contraction, not
        // an im2col — reject.
        let mut p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        p.n_inputs = 2;
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }

    // ---- G2: input rank / layout (FIXED 4 -> 3) ----

    #[test]
    #[should_panic(expected = "pins a rank-4 NCHW input")]
    fn im2col_rejects_rank3_input() {
        // A rank-3 input drops key.rank below 4 (the input is the widest operand).
        let a = OperandDesc::new(3, &[2, 3, 16], &[48, 16, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "input 0 must be dense contiguous NCHW")]
    fn im2col_rejects_noncontig_input() {
        // Last-two-axes-swapped strides ([..,1,4] instead of [..,4,1]) — non-contig.
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 1, 4], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "input 0 must not be reversed")]
    fn im2col_rejects_flipped_input() {
        // A dense-magnitude but inner-axis-reversed input (stride -1) keys Contig but
        // flipped — the source read would be mirrored.
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, -1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    // ---- G3: output rank / layout ----

    #[test]
    #[should_panic(expected = "output must be forward-dense contiguous")]
    fn im2col_rejects_broadcast_output() {
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[0, 8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "output must be forward-dense contiguous")]
    fn im2col_rejects_flipped_output() {
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    #[test]
    #[should_panic(expected = "output must be forward-dense contiguous")]
    fn im2col_rejects_strided_output() {
        // Last-two-axes-swapped output strides ([64,1,8]): the inner axis stride is
        // 8 != 1, so classify_contiguity keys it Strided/InnerContig (not Contig) —
        // but with NO broadcast (no 0 stride) and NO flip (all positive). This
        // ISOLATES the G3 `contig == Contig` clause: the broadcast test trips
        // `bcast.is_empty()` and the flipped test trips `!flipped`, so only a pure
        // strided output pins the middle clause (a strided output would gather into
        // the wrong slabs — the emitter assumes a dense row-major write).
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 1, 8], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &key);
    }

    // ---- G4: geometry legality ----

    #[test]
    #[should_panic(expected = "kernel (kh,kw) = (0,3) must be >= 1")]
    fn im2col_rejects_zero_kernel() {
        let p = im2col((0, 3), (1, 1), (1, 1), (1, 1));
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "stride (sh,sw) = (1,0) must be >= 1")]
    fn im2col_rejects_zero_stride() {
        let p = im2col((3, 3), (1, 0), (1, 1), (1, 1));
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "dilation (dh,dw) = (0,1) must be >= 1")]
    fn im2col_rejects_zero_dilation() {
        let p = im2col((3, 3), (1, 1), (1, 1), (0, 1));
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }

    // ---- G5: out_dtype None + body == Input(0) ----

    #[test]
    #[should_panic(expected = "rejected under Im2Col")]
    fn im2col_rejects_out_dtype_some() {
        // A pure gather preserves dtype — a hetero out_dtype has no exact store
        // (double-gated: assert_valid_out_dtype fires first, then validate_im2col).
        let mut p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        p.out_dtype = Some(ElementKind::I32);
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "body must be exactly Input(0)")]
    fn im2col_rejects_composed_body() {
        // v1 has no pre/post — the body must be exactly Input(0); a composed body
        // (Input(0)+Input(0)) has arithmetic the raw-bit copy does not emit.
        use crate::ir::input;
        let mut p = im2col((3, 3), (1, 1), (1, 1), (1, 1));
        p.body = (input(0) + input(0)).0;
        let _ = build_plan(&p, &im2col_key(ElementKind::F32));
    }
}

#[cfg(test)]
mod sort_gate_validate {
    //! Increment-8 SORT_PERM gate-rejection tests. Per the house rule these call
    //! `build_plan` DIRECTLY — an emitter panic would mask a gate mutation (the 0c
    //! lesson). Every `validate_row_sort` (and the argsort `assert_valid_out_dtype`
    //! coupling + `assert_valid_multi_output`) rejection has a test here; each
    //! sort-specific gate is mutation-checked both directions by a targeted
    //! reverse-edit.
    use super::{Schedule, access_tag, build_plan};
    use crate::ir::{Access, OpDef, ScalarExpr, SortLimit, SortOrder, SortOut, input};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // A rank-2 [256,128] sort cell: contiguous input + full-width contiguous output.
    fn sort_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }
    // The same cell with an I32 index output (argsort).
    fn argsort_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::I32, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }
    // Increment 9 FUSED_ARGSORT: the three-operand `Both` key — input, values
    // output (same dtype), I32 index output. `oi` overrides the index operand so
    // the layout-rejection tests can hand a broadcast/flipped out_idx.
    fn both_key_with(dt: ElementKind, oi: OperandDesc) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let ov = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, ov, oi], ArchSku::Sm89)
    }
    fn both_key(dt: ElementKind) -> StructureKey {
        let oi = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::I32, 256);
        both_key_with(dt, oi)
    }

    // ---- positive gate tests ----

    #[test]
    fn valid_sort_builds_asc_and_desc_values() {
        for order in [SortOrder::Asc, SortOrder::Desc] {
            let sc = OpDef::row_sort("sort", ElementKind::F32, order);
            let key = sort_key(ElementKind::F32);
            let plan = build_plan(&sc, &key);
            assert_eq!(
                plan.schedule,
                Schedule::RowSort {
                    order,
                    stable: true,
                    out: SortOut::Values,
                    limit: SortLimit::Full
                }
            );
            assert_eq!(access_tag(&sc.access), "RowSort");
        }
    }

    #[test]
    fn valid_argsort_builds_desc_with_i32_out() {
        let sc = OpDef::row_argsort("argsort", ElementKind::F32, SortOrder::Desc);
        assert_eq!(sc.out_dtype, Some(ElementKind::I32));
        let key = argsort_key(ElementKind::F32);
        let plan = build_plan(&sc, &key);
        assert_eq!(
            plan.schedule,
            Schedule::RowSort {
                order: SortOrder::Desc,
                stable: true,
                out: SortOut::Indices,
                limit: SortLimit::Full
            }
        );
    }

    #[test]
    fn i64_and_f64_sort_build() {
        for dt in [ElementKind::I64, ElementKind::F64] {
            let sc = OpDef::row_sort("sort", dt, SortOrder::Asc);
            let _ = build_plan(&sc, &sort_key(dt));
        }
    }

    // ---- validate_row_sort rejections ----

    #[test]
    #[should_panic(expected = "unstable declined")]
    fn unstable_rejected() {
        // v1 emits only the stable pair-sort; stable=false is dead keying.
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.access = Access::RowSort {
            order: SortOrder::Asc,
            stable: false,
            out: SortOut::Values,
            limit: SortLimit::Full,
        };
        let _ = build_plan(&sc, &sort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "out of the v1 set")]
    fn s8_dtype_rejected() {
        // S8 is a v1 de-scope (the bespoke argsort covers small ints).
        let sc = OpDef::row_sort("sort", ElementKind::S8, SortOrder::Asc);
        let _ = build_plan(&sc, &sort_key(ElementKind::S8));
    }

    #[test]
    #[should_panic(expected = "out of the v1 set")]
    fn u32_dtype_rejected() {
        // U32 is index-only (no value dtype).
        let sc = OpDef::row_sort("sort", ElementKind::U32, SortOrder::Asc);
        let _ = build_plan(&sc, &sort_key(ElementKind::U32));
    }

    #[test]
    #[should_panic(expected = "must be exactly Input(0)")]
    fn composed_body_rejected() {
        // v1 has no pre/post — the body must be exactly Input(0); a composed body
        // (here Input(0)+Input(0)) rejects (replaces the recursive check machinery).
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.body = (input(0) + input(0)).0;
        let _ = build_plan(&sc, &sort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "must be exactly Input(0)")]
    fn param_body_rejected() {
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.body = ScalarExpr::Param(0);
        let _ = build_plan(&sc, &sort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "streams exactly one row operand")]
    fn multiple_inputs_rejected() {
        // A sort streams exactly one row operand.
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.n_inputs = 2;
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, o], ArchSku::Sm89);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "must be contiguous")]
    fn non_contig_input_rejected() {
        // A transposed (column-major) input keys non-Contig on the sorted axis.
        let a = OperandDesc::new(2, &[256, 128], &[1, 256], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "must not be reversed")]
    fn flipped_input_rejected() {
        // A dense-but-REVERSED input keys |stride|-Contig + flipped; idx = base+j
        // reads forward, so a flipped operand reads mirrored/OOB (use Desc order).
        let a = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "Input0 must be the row-streamed")]
    fn broadcast_input_rejected() {
        // Input0 broadcast along the sorted (last) axis keys as a per-row scalar,
        // not the row-streamed sorted tensor — there is nothing to sort.
        let a = OperandDesc::new(2, &[256, 128], &[1, 0], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "forward-dense contiguous")]
    fn flipped_output_rejected() {
        // A reversed OUTPUT keys |stride|-Contig + flipped; the store is
        // forward-dense (out[base+r]) — a flipped output would write mirrored.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    // ---- out_dtype ↔ argsort coupling ----

    #[test]
    #[should_panic(expected = "must be Some(I32)")]
    fn argsort_with_none_out_dtype_rejected() {
        // argsort=true but out_dtype None: assert_valid_out_dtype returns early on
        // None, so validate_row_sort's coupling is the layer that fires.
        let mut sc = OpDef::row_argsort("argsort", ElementKind::F32, SortOrder::Asc);
        sc.out_dtype = None;
        let _ = build_plan(&sc, &argsort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "admitted only as the argsort index output")]
    fn argsort_with_i64_out_dtype_rejected() {
        // argsort with a non-I32 index dtype: I64 indices are a v1 de-scope
        // (assert_valid_out_dtype fires first).
        let mut sc = OpDef::row_argsort("argsort", ElementKind::F32, SortOrder::Asc);
        sc.out_dtype = Some(ElementKind::I64);
        let _ = build_plan(&sc, &argsort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "admitted only as the argsort index output")]
    fn values_sort_with_i32_out_dtype_rejected() {
        // A values-sort is dtype-preserving — a Some(I32) out_dtype has no exact
        // store (assert_valid_out_dtype fires on the values-sort branch).
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.out_dtype = Some(ElementKind::I32);
        let _ = build_plan(&sc, &sort_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn extra_out_bodies_rejected() {
        // Multi-output does not ride RowSort (assert_valid_multi_output fires first;
        // validate_row_sort backstops it).
        let mut sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        sc.extra_out_bodies = vec![input(0).0];
        let _ = build_plan(&sc, &sort_key(ElementKind::F32));
    }

    // ---- review-caught coverage (both mutations EMPIRICALLY survived the suite):
    // the output-layout compound assert had only its `flipped` term tested, and the
    // n_operands == n_inputs+1 assert was deletable. Pin every term. ----

    #[test]
    #[should_panic(expected = "forward-dense contiguous")]
    fn non_contig_output_rejected() {
        // A transposed (column-major) output keys non-Contig — the sorted writeback
        // out[base + r] assumes a dense inner axis.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[1, 256], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "forward-dense contiguous")]
    fn broadcast_output_rejected() {
        // A stride-0 (broadcast) output axis is not a full-width store target.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[256, 128], &[0, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "expects n_inputs+1 operands")]
    fn wrong_operand_count_rejected() {
        // n_inputs=1 against a 3-operand key — the signature and the accept
        // predicate would describe different arities.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    // ---- Increment 9 FUSED_ARGSORT: the `Both` two-output gates (G1-G3, G5) ----

    #[test]
    fn valid_both_builds_asc_and_desc() {
        // Positive: a fused two-output sort builds with the 3-operand key and the
        // Both schedule; out_dtype stays None (values output 0 is dtype-preserving,
        // the I32 out_idx is emitter-hardwired). Also the M2 kill (G2 relaxed to
        // n+1 for Both would reject this valid 3-operand build).
        for order in [SortOrder::Asc, SortOrder::Desc] {
            let sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, order);
            assert_eq!(sc.out_dtype, None);
            let key = both_key(ElementKind::F32);
            let plan = build_plan(&sc, &key);
            assert_eq!(
                plan.schedule,
                Schedule::RowSort {
                    order,
                    stable: true,
                    out: SortOut::Both,
                    limit: SortLimit::Full
                }
            );
            // n_outputs() stays body-derived = 1 for Both (the corollary decision);
            // the second buffer is owned by the SortOut state + the 3-operand key.
            assert_eq!(sc.n_outputs(), 1);
        }
    }

    #[test]
    fn both_builds_for_all_v1_dtypes() {
        for dt in [
            ElementKind::F32,
            ElementKind::F32Strict,
            ElementKind::F64,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::I32,
            ElementKind::I64,
        ] {
            let sc = OpDef::row_sort_indices("sort_both", dt, SortOrder::Asc);
            let _ = build_plan(&sc, &both_key(dt));
        }
    }

    #[test]
    fn values_key_still_two_operands() {
        // Regression: a Values sort still requires exactly n_inputs+1 operands (the
        // n_outputs computation must stay 1 for non-Both). Positive build with the
        // 2-operand key; a 3-operand Values key is rejected by
        // `wrong_operand_count_rejected` above.
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let key = sort_key(ElementKind::F32);
        let plan = build_plan(&sc, &key);
        assert_eq!(
            plan.schedule,
            Schedule::RowSort {
                order: SortOrder::Asc,
                stable: true,
                out: SortOut::Values,
                limit: SortLimit::Full
            }
        );
    }

    // G1 — out_dtype ↔ state coupling. The `Both ⇒ None` arm lives in
    // `validate_row_sort`, but `assert_valid_out_dtype` (G5) rejects any `Some(_)`
    // for Both FIRST in `build_plan` (it runs before the schedule match). So this
    // test calls `validate_row_sort` DIRECTLY to isolate G1 (the mutation M1
    // target) — the build_plan-direct house rule guards against EMITTER masking,
    // which does not apply to a plan-gate-vs-plan-gate shadow; the G5 front gate is
    // pinned separately by `both_with_some_out_dtype_rejected_at_out_dtype_gate`.
    #[test]
    #[should_panic(expected = "dtype-preserving on output 0")]
    fn both_requires_out_dtype_none() {
        let mut sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        sc.out_dtype = Some(ElementKind::I32);
        super::validate_row_sort(
            &sc,
            &both_key(ElementKind::F32),
            SortOrder::Asc,
            true,
            SortOut::Both,
            SortLimit::Full,
        );
    }

    // G5 — the `assert_valid_out_dtype` RowSort arm rejects a Both carrying any
    // Some(_) out_dtype (build_plan-direct: this is the reachable front gate).
    #[test]
    #[should_panic(expected = "admitted only as the argsort index output")]
    fn both_with_some_out_dtype_rejected_at_out_dtype_gate() {
        let mut sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        sc.out_dtype = Some(ElementKind::I32);
        let _ = build_plan(&sc, &both_key(ElementKind::F32));
    }

    // G2 — Both needs a 3-operand key ([in0, out_val, out_idx]).
    #[test]
    #[should_panic(expected = "expects n_inputs+2 operands")]
    fn both_key_needs_three_operands() {
        // A 2-operand key against a Both op — the emitter's two-pointer signature
        // and the accept predicate would describe different arities.
        let sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &sort_key(ElementKind::F32)); // 2-operand key
    }

    // G3 — the out_idx (operand 2) layout: full-width forward-dense contiguous.
    #[test]
    #[should_panic(expected = "out_idx (operand 2) must be forward-dense")]
    fn both_out_idx_broadcast_rejected() {
        // A stride-0 (broadcast) out_idx inner axis is not a full-width store target.
        let oi = OperandDesc::new(2, &[256, 128], &[0, 1], ElementKind::I32, 256);
        let key = both_key_with(ElementKind::F32, oi);
        let sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    #[test]
    #[should_panic(expected = "out_idx (operand 2) must be forward-dense")]
    fn both_out_idx_flipped_rejected() {
        // A reversed out_idx keys |stride|-Contig + flipped; out_idx[base+r] writes
        // forward-dense, so a flipped index output would write mirrored.
        let oi = OperandDesc::new(2, &[256, 128], &[128, -1], ElementKind::I32, 256);
        let key = both_key_with(ElementKind::F32, oi);
        let sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        let _ = build_plan(&sc, &key);
    }

    // G4 — extra_out_bodies empty (a permutation is not a body). `build_plan`
    // rejects a Both+body at `assert_valid_multi_output` (Elementwise-only) FIRST,
    // so this calls `validate_row_sort` DIRECTLY to isolate G4 (the M4 target); the
    // front gate is the same one the existing `extra_out_bodies_rejected` pins.
    #[test]
    #[should_panic(expected = "extra_out_bodies must be empty")]
    fn both_rejects_extra_out_bodies() {
        let mut sc = OpDef::row_sort_indices("sort_both", ElementKind::F32, SortOrder::Asc);
        sc.extra_out_bodies = vec![input(0).0];
        super::validate_row_sort(
            &sc,
            &both_key(ElementKind::F32),
            SortOrder::Asc,
            true,
            SortOut::Both,
            SortLimit::Full,
        );
    }

    // ---- Increment 10 TOPK/BOTTOMK: the runtime-k cap + shrunk out-extent ----

    // The `Both` key whose OUTPUTS are NARROWER than the input (`k_out < k_in`) —
    // [256,128] input, [256,k_out] value + I32 index outputs. `validate_row_sort`'s
    // output gate is LAYOUT-only (it does NOT assert the width), so this narrower
    // output passes as written — no relaxation. The `k_out <= k_in` relationship is a
    // runtime launch precondition (the key carries no numeric extent).
    fn topk_key(dt: ElementKind, k_out: i64) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let ov = OperandDesc::new(2, &[256, k_out], &[k_out, 1], dt, 256);
        let oi = OperandDesc::new(2, &[256, k_out], &[k_out, 1], ElementKind::I32, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, ov, oi], ArchSku::Sm89)
    }

    // G0 — `limit` threading: a TopK cell with a narrower `[batch,k_out]` out
    // operand BUILDS and schedules `limit: TopK` with the direction `order` picks
    // (topk = Desc, bottomk = Asc). This is the "gate" for the runtime-k/shrunk
    // extent — there is NO plan-time `k_out <= k_in` assert (none is expressible;
    // the key carries no numeric extents); the shrink is admitted because the
    // layout gate is width-agnostic.
    #[test]
    fn topk_both_cell_builds() {
        for (sc, want_order) in [
            (OpDef::row_topk("topk", ElementKind::F32), SortOrder::Desc),
            (
                OpDef::row_bottomk("bottomk", ElementKind::F32),
                SortOrder::Asc,
            ),
        ] {
            assert_eq!(sc.out_dtype, None);
            assert_eq!(sc.n_outputs(), 1); // increment-9 corollary: Both stays 1
            let key = topk_key(ElementKind::F32, 64);
            let plan = build_plan(&sc, &key);
            assert_eq!(
                plan.schedule,
                Schedule::RowSort {
                    order: want_order,
                    stable: true,
                    out: SortOut::Both,
                    limit: SortLimit::TopK
                }
            );
            assert_eq!(access_tag(&sc.access), "RowSort");
        }
    }

    // G0 regression — a `Full` sort still builds and schedules `limit: Full` (adding
    // the field did not disturb the existing three states).
    #[test]
    fn topk_full_regression_still_builds() {
        let sc = OpDef::row_sort_indices("fused", ElementKind::F32, SortOrder::Asc);
        let key = both_key(ElementKind::F32);
        let plan = build_plan(&sc, &key);
        assert_eq!(
            plan.schedule,
            Schedule::RowSort {
                order: SortOrder::Asc,
                stable: true,
                out: SortOut::Both,
                limit: SortLimit::Full
            }
        );
    }

    // G0 — TopK builds for every v1 dtype at a narrow k_out (the cap is orthogonal
    // to the dtype set; inherits the RowSort dtype gate verbatim).
    #[test]
    fn topk_builds_for_all_v1_dtypes() {
        for dt in [
            ElementKind::F32,
            ElementKind::F32Strict,
            ElementKind::F64,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::I32,
            ElementKind::I64,
        ] {
            let sc = OpDef::row_topk("topk", dt);
            let _ = build_plan(&sc, &topk_key(dt, 32));
        }
    }

    // G1 — out_dtype ↔ SortOut coupling is UNCHANGED under TopK: a TopK+Both with a
    // Some(I32) out_dtype rejects at `validate_row_sort`'s G1 (dtype-preserving on
    // output 0). Called DIRECTLY to isolate G1 from the `assert_valid_out_dtype`
    // front gate (mirrors `both_requires_out_dtype_none`).
    #[test]
    #[should_panic(expected = "dtype-preserving on output 0")]
    fn topk_both_requires_out_dtype_none() {
        let mut sc = OpDef::row_topk("topk", ElementKind::F32);
        sc.out_dtype = Some(ElementKind::I32);
        super::validate_row_sort(
            &sc,
            &topk_key(ElementKind::F32, 64),
            SortOrder::Desc,
            true,
            SortOut::Both,
            SortLimit::TopK,
        );
    }

    // G2 — operand count is extent-independent, so TopK+Both reuses it verbatim: a
    // 2-operand key against a Both op rejects (the two-pointer signature needs three).
    #[test]
    #[should_panic(expected = "expects n_inputs+2 operands")]
    fn topk_both_key_needs_three_operands() {
        let sc = OpDef::row_topk("topk", ElementKind::F32);
        let _ = build_plan(&sc, &sort_key(ElementKind::F32)); // 2-operand key
    }

    // G3 — output layout is UNCHANGED (LAYOUT-only, admits the narrower width): a
    // TopK+Both with a broadcast out_idx still rejects (the narrower width passes,
    // but the broadcast does not).
    #[test]
    #[should_panic(expected = "out_idx (operand 2) must be forward-dense")]
    fn topk_both_out_idx_broadcast_rejected() {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let ov = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
        let oi = OperandDesc::new(2, &[256, 64], &[0, 1], ElementKind::I32, 256); // broadcast
        let key = structure_key(OpCategory::UnaryElementwise, &[a, ov, oi], ArchSku::Sm89);
        let sc = OpDef::row_topk("topk", ElementKind::F32);
        let _ = build_plan(&sc, &key);
    }

    // G4 — stable=false rejects under TopK too (the cap adds no exception to the
    // inherited RowSort gates).
    #[test]
    #[should_panic(expected = "unstable declined")]
    fn topk_rejects_unstable() {
        let mut sc = OpDef::row_topk("topk", ElementKind::F32);
        sc.access = Access::RowSort {
            order: SortOrder::Desc,
            stable: false,
            out: SortOut::Both,
            limit: SortLimit::TopK,
        };
        let _ = build_plan(&sc, &topk_key(ElementKind::F32, 64));
    }
}

#[cfg(test)]
mod select_gate_validate {
    //! WHERE/SELECT plan-gate tests (G1-G4). Per the house rule these call
    //! `build_plan` DIRECTLY (an emitter panic would mask a gate mutation);
    //! the emitter backstops are independent Tier-2 tests in `cuda`.
    use super::{Schedule, build_plan};
    use crate::ir::{BinaryOp, OpDef, ReduceOp, coord, input, konst};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // n_operands contiguous 1D operands of `dtype` (V4-eligible at f32).
    fn key_dt(dtype: ElementKind, n_operands: usize) -> StructureKey {
        let a = OperandDesc::new(1, &[1024], &[1], dtype, 256);
        let ops: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(OpCategory::TernaryElementwise, &ops, ArchSku::Sm89)
    }

    // G1: select is rejected OUTRIGHT at every int dtype (v1 float-only) —
    // the int-reject arm in `assert_int_op_admissibility`.
    #[test]
    #[should_panic(expected = "Select has no integer lowering")]
    fn select_at_i32_is_rejected_at_the_plan_gate() {
        let op = OpDef::elementwise(
            "sel",
            3,
            &[ElementKind::I32],
            input(0).select(input(1), input(2)),
        );
        let _ = build_plan(&op, &key_dt(ElementKind::I32, 4));
    }

    #[test]
    #[should_panic(expected = "Select has no integer lowering")]
    fn select_at_i64_is_rejected_at_the_plan_gate() {
        let op = OpDef::elementwise(
            "sel",
            3,
            &[ElementKind::I64],
            input(0).select(input(1), input(2)),
        );
        let _ = build_plan(&op, &key_dt(ElementKind::I64, 4));
    }

    #[test]
    #[should_panic(expected = "Select has no integer lowering")]
    fn select_at_u8_is_rejected_at_the_plan_gate() {
        // U8 is exactly the 0c cond-observer dtype the outright reject sidesteps.
        let op = OpDef::elementwise(
            "sel",
            3,
            &[ElementKind::U8],
            input(0).select(input(1), input(2)),
        );
        let _ = build_plan(&op, &key_dt(ElementKind::U8, 4));
    }

    // G2: a Select ROOT never gets u8-out powers — `assert_valid_out_dtype`
    // admits only a `Binary(is_cmp)` root for `Some(U8)`, and a select stores
    // the ARM dtype. Already true without a code change; pinned here.
    #[test]
    #[should_panic(expected = "requires the body ROOT to be a comparison")]
    fn select_root_with_u8_out_is_rejected_at_the_plan_gate() {
        let mut op = OpDef::elementwise(
            "sel_mask",
            3,
            &[ElementKind::F32],
            input(0).select(input(1), input(2)),
        );
        op.out_dtype = Some(ElementKind::U8);
        let _ = build_plan(&op, &key_dt(ElementKind::F32, 4));
    }

    // G3: select is legal in every Access arm at float dtypes — the masked-sum
    // reduction (a select in the pre-fold body) builds a plan.
    #[test]
    fn masked_sum_reduction_with_a_select_body_builds() {
        let masked = OpDef::reduction(
            "masked_sum",
            1,
            &[ElementKind::F32],
            input(0)
                .binary(BinaryOp::CmpGt, konst(0.5))
                .select(input(0), konst(0.0)),
            ReduceOp::Sum,
        );
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let plan = build_plan(&masked, &key);
        assert!(matches!(plan.schedule, Schedule::Reduction { .. }));
    }

    // G4 (no new schedule machinery): a Coord-bearing cond rides the existing
    // Strided force — the triu body's route…
    #[test]
    fn triu_select_body_takes_the_strided_schedule() {
        let triu = OpDef::elementwise(
            "triu_sel",
            1,
            &[ElementKind::F32],
            coord(1)
                .binary(BinaryOp::CmpGe, coord(0) + konst(0.0))
                .select(input(0), konst(0.0)),
        );
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a], ArchSku::Sm89);
        let plan = build_plan(&triu, &key);
        assert!(
            matches!(plan.schedule, Schedule::Strided),
            "Coord cond must force Strided, got {:?}",
            plan.schedule
        );
    }

    // …and an all-Input select on a vec-width-4 key vectorizes per-lane
    // (uniform-dtype select needs no schedule change).
    #[test]
    fn all_input_select_vectorizes_at_a_v4_cell() {
        let op = OpDef::elementwise(
            "sel",
            3,
            &[ElementKind::F32],
            input(0).select(input(1), input(2)),
        );
        let key = key_dt(ElementKind::F32, 4);
        let plan = build_plan(&op, &key);
        assert!(
            matches!(plan.schedule, Schedule::Vectorized { width: 4 }),
            "all-Input select at a V4 cell must vectorize, got {:?}",
            plan.schedule
        );
    }
}
