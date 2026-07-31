//! CUDA C++ lowering — the one backend-specific module (v1).
//!
//! Everything CUDA-shaped (the vector types, `__global__`, the
//! `blockIdx`/`blockDim` launch indexing, the fp16/bf16 headers) lives here. The
//! math itself is lowered by the language-neutral [`crate::backend::lower_expr`]
//! — and reused verbatim across dtypes, because CUDA overloads `+ - * /` for
//! `__half` / `__nv_bfloat16` the same as for `float`.

use crate::backend::{
    Backend, GeneratedKernel, Lowering, Variant, VariantFidelity, lower_dag, lower_dag_all,
    lower_dag_multi, lower_expr,
};
use crate::ir::{Access, BinaryOp, ExprDag, ReduceOp, ScalarExpr, SortOrder, SortOut, UnaryOp};
use crate::plan::{KernelPlan, ReduceAxisClass, RrRole, Schedule, rr_role};
use baracuda_kernel_vocab::{Contiguity, ElementKind, OperandKey};

/// The CUDA C++ backend. Lowers a [`KernelPlan`] to `.cu` source.
#[derive(Copy, Clone, Debug, Default)]
pub struct Cuda;

impl Backend for Cuda {
    fn name(&self) -> &str {
        "cuda"
    }

    fn supports_dtype(&self, dtype: ElementKind) -> bool {
        // Increment 0c replaced the 0b uniform-u8 hold with the audited int
        // story: U8 and S8 are now COMPUTE dtypes (wrapping add/sub/mul via
        // integer promotion + store truncation; the bitwise/shift/logical
        // vocabulary), so this gate is exactly "does the backend have a
        // scalar C type". This is deliberately dtype-only: per-OP legality
        // (Div-for-int rejected, unary/float-fn/cmp rejected at int, logical
        // U8-only — the ir.rs admissibility table) is gate 2, the JIT's
        // `dtype_compatible`, and the AOT plan gate
        // `assert_int_op_admissibility` — so a uniform-U8 Div region still
        // declines even though U8 itself is supported here.
        //
        // U32 is the EXCEPTION: it entered `scalar_ctype` ("unsigned int") ONLY
        // as the gather/scatter index-LOAD type (never as a value/key dtype — no
        // `impl Element`/`KernelDtype` for `u32`, no int-op admissibility). A U32
        // COMPUTE/key plan must still be rejected here (else a U32 elementwise add
        // would silently lower to an `unsigned int` kernel and bypass the int-div
        // backstop), so exclude it — mirroring the 0b index-only-dtype pattern.
        // (A valid gather keys `plan.dtype` = the DATA dtype; U32 rides
        // `read_index`/`write_index`, never `plan.dtype`.)
        !matches!(dtype, ElementKind::U32) && scalar_ctype(dtype).is_some()
    }

    fn lower_variants(&self, plan: &KernelPlan<'_>) -> Vec<Variant> {
        // Schedule variants per the backlog
        // (docs/planning/foundational/11-variant-generators-backlog.md):
        // split-K for the outer-axis reduction cell; the materialized row cache
        // for RowReduce cells whose epilogue recomputes a stage's per-element
        // values (cross-pass materialization — Softmax's shared exp).
        let mut vs = Vec::new();
        vs.extend(reduction_splitk_variant(plan));
        // Precision-first: a double-accumulator serial fold of an f32 Sum/Mean
        // reduction — MorePrecise, selectable only through an honest contract (the
        // caller's precision policy decides), never the silent default.
        vs.extend(precision_first_variant(plan));
        vs.extend(row_reduce_materialize_variant(plan));
        vs.extend(contraction_splitk_variant(plan));
        vs.extend(scatter_atomic_variant(plan));
        // Increment 6 SCAN: the cooperative block-scan (Kogge-Stone warp scan +
        // cross-warp carry) for the FP Sum/Prod scan cell — reassociated-
        // deterministic, so selectable only through an honest contract. Declines
        // Max/Min and integer to the serial base (which serves them bit-exact).
        vs.extend(scan_blockscan_variant(plan));
        // Increment 8 SORT_PERM: the cooperative smem bitonic pair-sort for the
        // RowSort cell (k <= 1024 launch-note precondition) — BitIdentical to the
        // rank-sort base (a pair sort is a pure permutation), so selectable
        // silently; declines a non-RowSort / param'd / non-dense cell to the base.
        vs.extend(row_sort_bitonic_variant(plan));
        // PARTIAL-SELECT TOPK: the streaming tiled-bitonic top-k (O(k_in·log²k_out),
        // smem bounded on k_out not k_in) — BitIdentical to the full-sort top-k_out,
        // the unique fast+correct path for k_in > 1024 large-vocab top-k (where the
        // bitonic declines and the rank base is O(k_in²)). TopK-only; the advantage
        // regime rides the launch_note (the extent-free key cannot gate it), the
        // bench/dispatch layer selects the measured default (ship-top-K).
        vs.extend(partial_select_topk_variant(plan));
        vs
    }

    fn lower(&self, plan: &KernelPlan<'_>) -> GeneratedKernel {
        // U32 has a `scalar_ctype` ("unsigned int") ONLY as an index/address
        // dtype — it must never reach an ARITHMETIC compute path (a U32
        // elementwise add would silently lower to an `unsigned int` kernel and
        // bypass the int-div backstop). It IS a legitimate `plan.dtype` for an
        // INDEXED op though: a gather keys the DATA dtype (U32 rides
        // `read_index`), but `bincount` self-indexes — its input IS the u32 x,
        // so `plan.dtype == U32` while the u32 is used as a scatter ADDRESS, not
        // a value in arithmetic. So reject a U32 plan ONLY for a NON-indexed
        // (plain elementwise/reduction) op; the gather/scatter emitters handle
        // the index dtype themselves. `supports_dtype` declines U32 at the JIT
        // boundary; this is the independent AOT emitter backstop.
        let is_indexed = plan
            .read_index
            .iter()
            .any(|r| !matches!(r, crate::ir::ReadIndex::Direct))
            || !matches!(plan.write_index, crate::ir::WriteIndex::Direct);
        assert!(
            !matches!(plan.dtype, ElementKind::U32) || is_indexed,
            "cuda backend: U32 is an index/address dtype only — a U32 value/key \
             plan is illegal for a non-indexed op (no u32 arithmetic)"
        );
        let Some(ctype) = scalar_ctype(plan.dtype) else {
            panic!("cuda backend: unsupported dtype {:?}", plan.dtype);
        };
        assert!(
            plan.output_bodies()
                .iter()
                .all(|b| params_used(b).is_empty())
                || matches!(
                    plan.dtype,
                    ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64
                ),
            "cuda backend v1: scalar params are f32/f64-only for now (dtype {:?})",
            plan.dtype
        );
        // Increment 0c: infix `Div` and `Const` are spelled by shared,
        // dtype-blind backend code (`lower_expr` emits C `/` and an f64
        // literal with no dtype context) — the only two REJECT rows of the
        // ir.rs admissibility table with no op-level emitter backstop, and
        // exactly the device-dangerous ones (int `/0` is device-UB; an
        // f64-spelled Const silently runs double math in an integer kernel).
        // Mirror of the Param assert above, walking every expression the plan
        // can lower (body + RowReduce stages/epilogue + Contraction epilogue,
        // the same coverage as plan.rs's assert_no_half_nextafter walk) so the
        // backstop holds independently of the plan gate — the 0a lesson: gate
        // every layer.
        // Increment 5 — bincount exemption (mirrors `plan::assert_int_op_admissibility`):
        // a scatter with a bare `Const` body is the integer-count histogram; its
        // `Const(1)` is a store literal the scatter combine narrows exactly, not
        // int compute — the double-math hazard this walk polices does not apply.
        let bincount_shape =
            plan.write_index.scatter().is_some() && matches!(plan.body, ScalarExpr::Const(_));
        if crate::plan::is_int_dtype(plan.dtype) && !bincount_shape {
            // Rule 4 mirror (`plan::assert_int_op_admissibility`'s any/all/count
            // lift, ba325509): `true` only for `Access::Reduction` — NOT
            // RowReduce/Contraction/Scan/Window, which stay out of scope for the
            // fused-Cmp*-predicate exception, exactly like the plan gate. Passed
            // uniformly to every expr below (for `Access::Reduction` every entry
            // in `exprs` is `body` or `post`, both correctly in-scope; for every
            // other access kind this stays `false`, so their walk is unchanged).
            let in_reduction = matches!(plan.access, Access::Reduction { .. });
            // Every output body (multi-output: body + extra_out_bodies), plus the
            // reduction-class stages/epilogue — the same coverage as plan.rs.
            let mut exprs: Vec<&ScalarExpr> = plan.output_bodies();
            match plan.access {
                Access::RowReduce { stages, epilogue } => {
                    exprs.extend(stages.iter().map(|s| &s.pre));
                    exprs.push(epilogue);
                }
                Access::Contraction { epilogue, .. } => exprs.push(epilogue),
                // The 0e reduction post-expr lowers at the accumulator dtype too.
                Access::Reduction { post, .. } => exprs.push(post),
                // Increment 6 SCAN: `pre`/`post` lower at the accumulator dtype (an
                // int cumsum rides the serial base), so gate both for int Div/Const.
                Access::Scan { pre, post, .. } => {
                    exprs.push(pre);
                    exprs.push(post);
                }
                // Increment 7 WINDOW: `pre`/`post` lower at the accumulator dtype
                // (an int sum/max pool rides the same fold), so gate both for int
                // Div/Const.
                Access::Window { pre, post, .. } => {
                    exprs.push(pre);
                    exprs.push(post);
                }
                // Increment 8 SORT_PERM: `body` is pinned `Input(0)` (already in
                // `exprs`); an int sort permutes storage bits (no int Div/Const
                // arithmetic), so nothing extra to gate — the arm keeps the walk total.
                Access::RowSort { .. } => {}
                // Increment 11 IM2COL: `body` is pinned `Input(0)` (already in
                // `exprs`); a raw-bit gather permutes storage bits (no int Div/Const
                // arithmetic), so nothing extra to gate — the arm keeps the walk total.
                Access::Im2Col { .. } => {}
                Access::Elementwise => {}
            }
            for e in exprs {
                // `at_reduction_root: in_reduction` — this loop is the initial
                // call on each expr in `exprs`, which is `[body, post]`
                // precisely when `in_reduction` is true (see the match above:
                // only `Access::Reduction` pushes `post`), mirroring
                // `plan::assert_int_op_admissibility`'s same `in_reduction`-as-
                // `at_reduction_root` initial call.
                assert_no_int_div_or_const(e, plan.dtype, in_reduction, in_reduction);
            }
        }
        // Increment 0d: independent Coord emitter backstop, beside the int
        // Div/Const walk above and with the SAME expression coverage (body +
        // RowReduce stages/epilogue + Contraction epilogue). The plan gate
        // (`plan::assert_coord_admissibility`) validate-rejects the same
        // three rows; this backstop holds independently of it (the 0a
        // lesson: gate every layer), with cuda-prefixed messages distinct
        // from the plan gate's. The fourth layer is per-emitter: every
        // non-strided emitter's `coord` closure panics if a Coord leaf
        // actually reaches it.
        {
            let mut exprs: Vec<&ScalarExpr> = plan.output_bodies();
            match plan.access {
                Access::RowReduce { stages, epilogue } => {
                    exprs.extend(stages.iter().map(|s| &s.pre));
                    exprs.push(epilogue);
                }
                Access::Contraction { epilogue, .. } => exprs.push(epilogue),
                Access::Reduction { post, .. } => exprs.push(post),
                // Increment 6 SCAN: a Coord in `pre`/`post` is rejected here (the
                // scan iterates the (row, j) space, not the elementwise output space).
                Access::Scan { pre, post, .. } => {
                    exprs.push(pre);
                    exprs.push(post);
                }
                // Increment 7 WINDOW: a Coord in `pre`/`post` is rejected here (the
                // window iterates the (row, o) space, not the elementwise output space).
                Access::Window { pre, post, .. } => {
                    exprs.push(pre);
                    exprs.push(post);
                }
                // Increment 8 SORT_PERM: `body` is pinned `Input(0)` (no Coord);
                // RowSort is non-elementwise, so a Coord would be rejected upstream —
                // nothing extra to walk here.
                Access::RowSort { .. } => {}
                // Increment 11 IM2COL: `body` is pinned `Input(0)` (no Coord); im2col
                // is non-elementwise, so a Coord would be rejected upstream — nothing
                // extra to walk here.
                Access::Im2Col { .. } => {}
                Access::Elementwise => {}
            }
            for e in exprs {
                assert_coord_lowerable(e, plan);
            }
        }
        // Increment 0b/0e: a hetero-output plan reaches only emitters with an
        // out_dtype-aware store: the scalar/strided elementwise emitters (0b
        // u8-predicate — no packed u8 store exists, so `build_plan` forces the
        // schedule) and the reduction emitter (0e any/all → U8, count → I64 —
        // `emit_reduction` threads `out_ctype`/`store` through the fold). This
        // emitter-level backstop keeps a future schedule change from silently
        // routing a hetero output through a vector-typed store (the 0a lesson:
        // gate every layer, not just the first).
        // Increment 8 SORT_PERM: the ARGSORT index output is a hetero out (I32 !=
        // key dtype) that the RowSort emitter stores through an out_dtype-aware
        // `int* out` store — widen the matches! so it is not rejected here. Additive
        // (existing behavior untouched); pinned by the argsort golden test.
        assert!(
            plan.out_dtype == plan.dtype
                || matches!(
                    plan.schedule,
                    Schedule::Scalar
                        | Schedule::Strided
                        | Schedule::Reduction { .. }
                        | Schedule::RowSort { .. }
                ),
            "cuda backend: hetero output (out {:?}, key {:?}) lowers scalar/strided/reduction/rowsort only; got {:?}",
            plan.out_dtype,
            plan.dtype,
            plan.schedule
        );
        // Item 01: independent layout-VIEW backstop, beside the plan gate
        // `plan::assert_valid_views` (the 0a lesson: gate every layer). A viewed
        // input reads the producer through a layout change that ONLY the strided
        // emitter's per-operand `offset_expr` remap folds — the vector/scalar/
        // packed emitters iterate a bare linear index and would silently read the
        // un-viewed operand. This pins: a real view ⇒ Elementwise + single-output
        // + the Strided schedule, and re-validates each view's structure. A
        // view-free / all-identity plan returns immediately (byte-identical).
        assert_views_lowerable(plan);
        // Increment 4: independent GATHER backstop, beside the plan gate
        // `plan::assert_valid_gather` (the 0a lesson: gate every layer). A gathered
        // input reads a data-dependent address that ONLY the strided emitter folds;
        // the vector/packed/scalar emitters iterate a bare linear index and would
        // ignore the index operand. Pins: a real gather ⇒ Elementwise +
        // single-output + one gathered input + Strided schedule. An index-free plan
        // returns immediately (byte-identical).
        assert_gather_lowerable(plan);
        // Increment 5: independent SCATTER backstop, beside the plan gate
        // `plan::assert_valid_scatter` (the 0a lesson: gate every layer). A
        // scattered output writes a data-dependent address that ONLY the strided
        // emitter folds; the vector/packed/scalar emitters iterate a bare linear
        // index and would ignore the index operand. Pins: a real scatter ⇒
        // Elementwise + single-output + Strided + a legal combine/dtype pair. A
        // write-Direct plan returns immediately (byte-identical).
        assert_scatter_lowerable(plan);
        // BASE_OFFSET SLICE: independent OFFSET backstop, beside the plan gate
        // `plan::assert_valid_offsets` (the 0a lesson: gate every layer). A runtime
        // base offset bumps the operand base pointer at kernel entry, which ONLY the
        // strided emitter does; the vectorized/scalar/packed + multi-out + every
        // non-elementwise emitter iterate a bare linear index that would silently
        // ignore the bump. Pins: a real offset ⇒ Elementwise + single-output +
        // Strided + not-an-FP-atomic-scatter. An offset-free plan returns immediately
        // (byte-identical). Runs BEFORE the multi-output dispatch so an offsetted
        // multi-output plan trips here, not in `emit_strided_multi`.
        assert_offsets_lowerable(plan);
        // Increment 1: a MULTI-OUTPUT plan routes to the dedicated N-store
        // emitters BEFORE the single-output dispatch below — so the single-output
        // emitters stay byte-for-byte untouched (extra_out_bodies is empty ⇒
        // n_outputs == 1 ⇒ this branch is never taken for any pre-increment-1 op).
        // Independent emitter backstop (the plan gate validates the same rules;
        // the 0a lesson: gate every layer): Elementwise + uniform dtype + no
        // Reduced/Coord in any body (the multi lowering has no coord/reduced
        // closure to reach).
        if plan.n_outputs > 1 {
            assert_multi_output_lowerable(plan);
            return match plan.schedule {
                Schedule::Vectorized { width } => match vector_type(plan.dtype, width) {
                    Some((vty, lanes)) => emit_vectorized_multi(plan, vty, lanes),
                    None => match packed_kind(plan.dtype, width) {
                        Some(pk) if bodies_pack(plan) => emit_vectorized_packed_multi(plan, &pk),
                        _ => emit_scalar_multi(plan, ctype),
                    },
                },
                Schedule::Scalar => emit_scalar_multi(plan, ctype),
                Schedule::Strided => emit_strided_multi(plan, ctype),
                // Multi-output is Elementwise-only (plan gate + the backstop
                // above), so only the elementwise schedules can appear.
                other => panic!(
                    "cuda backend: multi-output op '{}' reached a non-elementwise \
                     schedule {other:?} — the plan gate pins multi-output to \
                     Access::Elementwise (scalar/vectorized/strided)",
                    plan.op_name
                ),
            };
        }
        match plan.schedule {
            Schedule::Vectorized { width } => match vector_type(plan.dtype, width) {
                Some((vty, lanes)) => emit_vectorized(plan, vty, lanes),
                None => match packed_kind(plan.dtype, width) {
                    // f16/bf16: packed half2/bf162 pairs — bit-identical to the
                    // scalar kernel per lane (see the tier notes on the spellers),
                    // gated to Input-leaf bodies (a Const/Param participates in
                    // double-promoted math on the scalar path, which a pair splat
                    // would change).
                    Some(pk) if body_packs(plan.body) => emit_vectorized_packed(plan, &pk),
                    // No packed path (or a const/param body): scalar fallback —
                    // still correct, still the narrower-dtype bandwidth win.
                    _ => emit_scalar(plan, ctype),
                },
            },
            Schedule::Scalar => emit_scalar(plan, ctype),
            Schedule::Strided => {
                // Increment 5 — DETERMINISM ROUTING. An FP `atomicAdd` scatter is
                // run-to-run non-deterministic, so per the house variant rule it is
                // NEVER the silent default: the base `lower()` emits the
                // deterministic **gather-sum** reformulation (one thread per output
                // cell scans the update domain and sums matching values in a fixed
                // order — the bespoke `segment_sorted_kernel` precedent), and the
                // atomic scatter is offered separately as the `Nondeterministic`
                // variant (`lower_variants`). Every deterministic combine (Assign,
                // integer atomicAdd, atomicMax/Min) takes the direct scatter store
                // in `emit_strided` as its unconditional base.
                match plan.write_index.scatter() {
                    Some((_, _, combine, _, _)) if combine.is_fp_atomic_add(plan.out_dtype) => {
                        emit_scatter_gathersum(plan, ctype)
                    }
                    _ => emit_strided(plan, ctype),
                }
            }
            Schedule::Reduction { op, .. } => emit_reduction(plan, ctype, op, false),
            Schedule::RowReduce { .. } => emit_row_reduce(plan, ctype),
            Schedule::Contraction => emit_contraction(plan, ctype),
            // Increment 6 SCAN: `build_plan` always derives `block: false` (the
            // serial-fold base); the cooperative block-scan is produced separately
            // by `scan_blockscan_variant`, never routed through `lower()`.
            Schedule::Scan { .. } => emit_scan(plan, ctype),
            // Increment 7 WINDOW: one thread per output element (grid-stride) folds
            // the local pooling window — no variant (each output is an independent
            // fixed-order fold, BitIdentical).
            Schedule::Window { .. } => emit_window(plan, ctype),
            // Increment 8 SORT_PERM: the per-output RANK-sort base (any k, no smem,
            // no barriers). The cooperative smem bitonic pair-sort is produced
            // separately by `row_sort_bitonic_variant`, never routed through `lower()`.
            Schedule::RowSort { .. } => emit_row_sort(plan, ctype),
            // Increment 11 IM2COL: one thread per output cell (grid-stride) computes
            // the closed-form source coord and RAW-BIT copies it (or stores the typed
            // zero for an OOB tap) — no fold, no variant (each output is an
            // independent read-or-zero + store, BitIdentical).
            Schedule::Im2Col { .. } => emit_im2col(plan, ctype),
        }
    }
}

/// CUDA scalar type for a dtype, or `None` if the backend can't lower it yet.
/// `U8` (increment 0b) is the comparison-predicate mask dtype — `unsigned char`
/// per the FKC §5 Bool→U8 pinning — and, since increment 0c, an audited
/// COMPUTE dtype (wrapping mod-256 C semantics), same class as the i32/i64
/// arms. `S8` (FKC `I8`, increment 0c) is `signed char` — two's-complement
/// wrapping via integer promotion + store truncation (see the ir.rs table).
pub(crate) fn scalar_ctype(dt: ElementKind) -> Option<&'static str> {
    Some(match dt {
        ElementKind::F32 | ElementKind::F32Strict => "float",
        ElementKind::F64 => "double",
        ElementKind::F16 => "__half",
        ElementKind::Bf16 => "__nv_bfloat16",
        ElementKind::I32 => "int",
        ElementKind::I64 => "long long",
        ElementKind::S8 => "signed char",
        ElementKind::U8 => "unsigned char",
        // U32 is the gather/scatter INDEX-operand ctype (`unsigned int`) — a
        // 4-byte address dtype used ONLY for the index-load pointer type (the
        // Model-A u32-index path), never a compute operand. It has no `Element`
        // impl and no vector/packed path; a compute op never keys `plan.dtype =
        // U32` (no constructor builds one), so this arm serves the index load.
        ElementKind::U32 => "unsigned int",
        _ => return None,
    })
}

/// Extra header a dtype needs (fp16 / bf16 device operators), if any.
fn extra_include(dt: ElementKind) -> Option<&'static str> {
    match dt {
        ElementKind::F16 => Some("#include <cuda_fp16.h>\n"),
        ElementKind::Bf16 => Some("#include <cuda_bf16.h>\n"),
        _ => None,
    }
}

/// Vector type + lane names for a `(dtype, width)` with a **native CUDA vector
/// type** (f32/f64), or `None`. f16/bf16 vectorize through the packed-pair path
/// ([`packed_kind`] / [`emit_vectorized_packed`]) instead.
fn vector_type(dt: ElementKind, width: u32) -> Option<(&'static str, &'static [&'static str])> {
    match (dt, width) {
        (ElementKind::F32 | ElementKind::F32Strict, 4) => Some(("float4", &["x", "y", "z", "w"])),
        (ElementKind::F32 | ElementKind::F32Strict, 2) => Some(("float2", &["x", "y"])),
        (ElementKind::F64, 2) => Some(("double2", &["x", "y"])),
        _ => None,
    }
}

/// The unit of the emitted elementwise kernel's `n` argument for this plan:
/// `w > 1` means the kernel counts `w`-element **vectors** (a vectorized or
/// packed lowering, `n = elements / w`); `1` means elements. Must mirror
/// [`Cuda::lower`]'s `Schedule::Vectorized` dispatch exactly — the contract's
/// `count_unit:` documents the ABI the emitter actually built, including the
/// scalar fallbacks (unpackable dtype, const/param f16 body).
pub(crate) fn effective_count_width(plan: &KernelPlan<'_>) -> u32 {
    match plan.schedule {
        Schedule::Vectorized { width }
            if vector_type(plan.dtype, width).is_some()
                || (packed_kind(plan.dtype, width).is_some() && bodies_pack(plan)) =>
        {
            width
        }
        _ => 1,
    }
}

/// Whether EVERY output body of the plan is packable (all-`Input`-leaf, per
/// [`body_packs`]) — the per-DAG f16/bf16 packed-pair gate generalized to
/// multiple outputs. For a single-output plan this is exactly `body_packs(body)`
/// (byte-identical decision); a multi-output packed lowering requires all bodies
/// to pack, else the whole DAG falls back to the scalar path (one body with a
/// `Const`/`Param` disqualifies the pair splat, same rule as today).
fn bodies_pack(plan: &KernelPlan<'_>) -> bool {
    plan.output_bodies().iter().all(|b| body_packs(b))
}

/// Pair-lane field names for the per-kernel packed vector struct.
static PACKED_FIELDS: [&str; 4] = ["a", "b", "c", "d"];

/// A packed f16/bf16 vector cell: `width` halves per vector access (16 bytes for
/// V8, 8 for V4, 4 for V2), computed as `width/2` two-lane pairs.
struct PackedKind {
    /// The two-lane pair type: `__half2` or `__nv_bfloat162`.
    pair_ty: &'static str,
    /// Pair-lane fields in the per-kernel vector struct (`width/2` of them).
    fields: &'static [&'static str],
    /// Struct alignment in bytes (= `width * 2`, the full vector access size).
    align: u32,
}

/// The packed-pair kind for a `(dtype, width)`, or `None` when the dtype has no
/// pair type (or the width is not a packed width).
fn packed_kind(dt: ElementKind, width: u32) -> Option<PackedKind> {
    let pair_ty = match dt {
        ElementKind::F16 => "__half2",
        ElementKind::Bf16 => "__nv_bfloat162",
        _ => return None,
    };
    let lanes = match width {
        8 => 4,
        4 => 2,
        2 => 1,
        _ => return None,
    };
    Some(PackedKind {
        pair_ty,
        fields: &PACKED_FIELDS[..lanes],
        align: width * 2,
    })
}

/// Whether the packed pair path can emit `body` **bit-identically** to the
/// scalar kernel: every leaf must be an `Input`. A `Const`/`Param` on the scalar
/// f16/bf16 path participates in *double*-promoted math (`__half + 1.5` promotes
/// through float to double), which a pre-rounded pair splat would change — so
/// const/param bodies stay on the scalar path.
fn body_packs(e: &ScalarExpr) -> bool {
    match e {
        ScalarExpr::Input(_) => true,
        // Coord never packs: the packed dtypes (f16/bf16) are outside its
        // dtype gate anyway, and a Coord body never reaches Vectorized.
        ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => false,
        ScalarExpr::Unary(_, x) => body_packs(x),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => body_packs(a) && body_packs(b),
        // Select never packs (G7): the packed pair path has no bit-audited
        // pair-split select (a Tier-B split would work but is mostly moot —
        // any select with a konst/cmp already falls to scalar via the leaf
        // rule above), so a select body declines to the scalar/strided path.
        ScalarExpr::Select(..) => false,
    }
}

/// Split/join spellings for a pair type: (`low half`, `high half`, `join`).
fn pair_parts(dt: ElementKind) -> (&'static str, &'static str, &'static str) {
    match dt {
        ElementKind::F16 => ("__low2half", "__high2half", "__halves2half2"),
        ElementKind::Bf16 => ("__low2bfloat16", "__high2bfloat16", "__halves2bfloat162"),
        _ => unreachable!("pair_parts requires a packed dtype, got {dt:?}"),
    }
}

/// Packed unary speller. **Tier A** (native packed intrinsic) only for ops that
/// are provably bit-identical to the scalar path per lane: `Neg`/`Abs` are
/// sign-bit ops, and `Sqr`'s product of two halves is exact in f32 (≤ 22
/// significand bits), so the native packed product rounds identically to the
/// scalar float-round-trip. Everything else is **Tier B**: split the pair, run
/// the *existing* scalar speller on each half (identical text ⇒ identical bits),
/// and re-join. Operand strings are always leaf refs or hoisted `tmp` names
/// ([`crate::backend::lower_dag_all`]), so the `{x}` duplication below never
/// duplicates a computation.
fn packed_unary(op: UnaryOp, x: String, dt: ElementKind) -> String {
    match op {
        UnaryOp::Neg => format!("__hneg2({x})"),
        UnaryOp::Abs => format!("__habs2({x})"),
        UnaryOp::Sqr => format!("__hmul2({x}, {x})"),
        _ => {
            let (lo, hi, join) = pair_parts(dt);
            let l = cuda_unary(op, format!("{lo}({x})"), dt);
            let h = cuda_unary(op, format!("{hi}({x})"), dt);
            format!("{join}({l}, {h})")
        }
    }
}

/// Packed binary speller — every binary function op is **Tier B** (pair-split
/// through the existing scalar speller): `__hmax2`/`__hmin2` are IEEE maxNum
/// (NaN-*suppressing*), which would break the house NaN-propagating Max/Min
/// convention, and the rest (`Pow`/`Rem`, the increment-0a fns) have no packed
/// intrinsic. `FmaxIeee`/`FminIeee` *could* one day take `__hmax2`/`__hmin2` as
/// Tier A, but only behind the item-09 bit-identity sweep — not assumed here.
/// `Nextafter` never reaches this speller ([`cuda_binary`] refuses halves).
fn packed_binary(op: BinaryOp, a: String, b: String, dt: ElementKind) -> String {
    let (lo, hi, join) = pair_parts(dt);
    let l = cuda_binary(op, format!("{lo}({a})"), format!("{lo}({b})"), dt);
    let h = cuda_binary(op, format!("{hi}({a})"), format!("{hi}({b})"), dt);
    format!("{join}({l}, {h})")
}

/// Short dtype tag for generated symbol names. Only called for dtypes that pass
/// [`scalar_ctype`].
pub(crate) fn dtype_tag(dt: ElementKind) -> &'static str {
    match dt {
        ElementKind::F32 => "f32",
        ElementKind::F32Strict => "f32s",
        ElementKind::F64 => "f64",
        ElementKind::F16 => "f16",
        ElementKind::Bf16 => "bf16",
        ElementKind::I32 => "i32",
        ElementKind::I64 => "i64",
        ElementKind::S8 => "i8",
        ElementKind::U8 => "u8",
        // U32 index-dtype infix: `gather_f32_u32` (the Fuel-facing u32-index
        // variant's entry_point symbol).
        ElementKind::U32 => "u32",
        _ => "x",
    }
}

fn header(plan: &KernelPlan<'_>, name: &str) -> String {
    // Header-light, portable device source: no `<cstdint>` / `<cuda_runtime.h>`,
    // so the SAME source compiles under both nvcc (AOT) and nvrtc (JIT). Device
    // built-ins (`threadIdx`, `float4`, `__global__`) are implicit in both; the
    // 64-bit index type is the built-in `long long`. Only the fp16/bf16 operator
    // headers are included (nvrtc bundles them), and only when the dtype needs them.
    let mut h = format!(
        "// Generated by baracuda-kernelgen — do not edit.\n\
         // op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token(),
    );
    if let Some(inc) = extra_include(plan.dtype) {
        h.push_str(inc);
    }
    h.push('\n');
    h.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
    h
}

fn emit_vectorized(plan: &KernelPlan<'_>, vty: &str, lanes: &[&str]) -> GeneratedKernel {
    let name = format!(
        "baracuda_gen_{}_{}_co_v{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        lanes.len()
    );
    let n = plan.n_inputs;
    let mut s = header(plan, &name);
    for i in 0..n {
        s.push_str(&format!("    const {vty}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {vty}* __restrict__ out,\n"));
    s.push_str(&format!(
        "    long long nv{})\n{{\n",
        param_args(plan.body, param_ctype(plan))
    ));
    s.push_str("    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long step = (long long)gridDim.x * blockDim.x;\n");
    s.push_str("    for (; i < nv; i += step) {\n");
    for i in 0..n {
        s.push_str(&format!("        {vty} v{i} = in{i}[i];\n"));
    }
    s.push_str(&format!("        {vty} vo;\n"));
    // A lane value is a scalar of the vector's element type; a hoisted shared
    // interior lives in that scalar `tmp` type, scoped per lane so names never
    // collide across lanes.
    let sctype = scalar_ctype(plan.dtype).expect("vectorized dtype has a scalar ctype");
    for lane in lanes {
        let acc = |idx: u8| format!("v{idx}.{lane}");
        let (prelude, root) = lower_dag(
            &ExprDag::from_expr(plan.body),
            sctype,
            &Lowering {
                leaf: &acc,
                reduced: &|i| unreachable!("no Reduced leaf outside RowReduce: red{i}"),
                coord: &|d| {
                    panic!(
                        "cuda backend: Coord({d}) reached the vectorized emitter — Coord \
                         bodies lower via Strided only (the linear-index kernels have no \
                         per-axis coordinates)"
                    )
                },
                unary: &|op, x| cuda_unary(op, x, plan.dtype),
                binary: &|op, a, b| cuda_binary(op, a, b, plan.dtype),
                select: &|c, a, b| cuda_select(c, a, b, plan.dtype),
            },
        );
        if prelude.is_empty() {
            s.push_str(&format!("        vo.{lane} = {root};\n"));
        } else {
            s.push_str("        {\n");
            for decl in &prelude {
                s.push_str(&format!("            {decl}\n"));
            }
            s.push_str(&format!("            vo.{lane} = {root};\n        }}\n"));
        }
    }
    s.push_str("        out[i] = vo;\n    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Emit a **packed** f16/bf16 vectorized elementwise kernel: one `width`-halves
/// vector access per operand per iteration (128-bit for V8), computed as
/// `width/2` two-lane pairs. Infix `+ - * /` lower through the CUDA `__half2` /
/// `__nv_bfloat162` operator overloads — native per-lane packed ops,
/// bit-identical to the scalar `__half` operators the scalar kernel uses;
/// function ops go through [`packed_unary`]/[`packed_binary`] (Tier A/B).
/// Bodies are `Input`-leaf-only ([`body_packs`]), so there are no `p{i}` params.
///
/// Lowered with [`lower_dag_all`] (every non-leaf hoisted to a pair-typed `tmp`)
/// so Tier B's pair-split never duplicates a computation, only a `tmp` name —
/// deep Tier-B chains stay linear in the source.
fn emit_vectorized_packed(plan: &KernelPlan<'_>, pk: &PackedKind) -> GeneratedKernel {
    let width = pk.fields.len() * 2;
    let name = format!(
        "baracuda_gen_{}_{}_co_v{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        width
    );
    // Per-kernel struct name: generated kernels are concatenated into one
    // translation unit by the validators, so the type must never collide.
    let vec_ty = format!("{name}_vec");
    let n = plan.n_inputs;

    // Custom preamble — the vector struct must sit between the dtype include and
    // `extern "C"` (the same placement discipline as emit_row_reduce's helpers).
    let mut s = format!(
        "// Generated by baracuda-kernelgen — do not edit.\n// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    );
    if let Some(inc) = extra_include(plan.dtype) {
        s.push_str(inc);
    }
    s.push('\n');
    s.push_str(&format!(
        "struct __align__({}) {vec_ty} {{ {} {}; }};\n\n",
        pk.align,
        pk.pair_ty,
        pk.fields.join(", ")
    ));
    s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
    for i in 0..n {
        s.push_str(&format!("    const {vec_ty}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {vec_ty}* __restrict__ out,\n"));
    s.push_str("    long long nv)\n{\n");
    s.push_str("    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long step = (long long)gridDim.x * blockDim.x;\n");
    s.push_str("    for (; i < nv; i += step) {\n");
    for i in 0..n {
        s.push_str(&format!("        {vec_ty} v{i} = in{i}[i];\n"));
    }
    s.push_str(&format!("        {vec_ty} vo;\n"));
    for field in pk.fields {
        let acc = |idx: u8| format!("v{idx}.{field}");
        let (prelude, root) = lower_dag_all(
            &ExprDag::from_expr(plan.body),
            pk.pair_ty,
            &Lowering {
                leaf: &acc,
                reduced: &|i| unreachable!("no Reduced leaf outside RowReduce: red{i}"),
                // Doubly unreachable: `body_packs` is Input-leaf-only AND the
                // packed dtypes are outside Coord's f32/f64 gate.
                coord: &|d| {
                    panic!(
                        "cuda backend: Coord({d}) reached the packed emitter — Coord \
                         bodies lower via Strided only (and halves are outside the \
                         Coord dtype gate)"
                    )
                },
                unary: &|op, x| packed_unary(op, x, plan.dtype),
                binary: &|op, a, b| packed_binary(op, a, b, plan.dtype),
                // Unreachable: `body_packs` excludes Select (G7), so a select
                // body never reaches the packed pair path — panic backstop.
                select: &|_, _, _| {
                    panic!(
                        "cuda backend: Select reached the packed emitter — body_packs \
                         excludes select bodies (they decline to the scalar path)"
                    )
                },
            },
        );
        if prelude.is_empty() {
            // Body is a bare Input leaf (a copy) — no tmp needed.
            s.push_str(&format!("        vo.{field} = {root};\n"));
        } else {
            s.push_str("        {\n");
            for decl in &prelude {
                s.push_str(&format!("            {decl}\n"));
            }
            s.push_str(&format!("            vo.{field} = {root};\n        }}\n"));
        }
    }
    s.push_str("        out[i] = vo;\n    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Output `j`'s pointer scalar C type — the input `ctype` for a uniform output
/// (`out_dtype_of(j) == plan.dtype`), `unsigned char` for a u8-predicate/keep-mask
/// output (increment 0b single-output; the hetero multi-output / dropout-class
/// increment per-output). `out_ctype_of(plan, 0, ctype)` is byte-identical to the
/// pre-generalization `out_ctype` (output 0's dtype is `plan.out_dtype`), so every
/// single-output + uniform-multi emitter that passes `j = 0`/a uniform `j` is
/// unchanged.
pub(crate) fn out_ctype_of<'c>(plan: &KernelPlan<'_>, j: usize, ctype: &'c str) -> &'c str {
    let d = plan.out_dtype_of(j);
    if d == plan.dtype {
        ctype
    } else {
        scalar_ctype(d).expect("validated out dtype has a scalar ctype")
    }
}

/// The store expression for output `j`'s lowered body root. A uniform output
/// (`out_dtype_of(j) == plan.dtype`) stores the root unchanged (byte-identical to
/// pre-0b output). A u8 keep-mask output converts the exact 0.0/1.0 predicate
/// (lowered in the COMPUTE dtype `plan.dtype`) to `unsigned char` — exact by
/// construction (the G1 plan gate + G5 backstop pin the body root to a `Cmp*`).
/// The conversion is applied HERE at the store site, per output, **never baked
/// into the shared DAG node**: for dropout the same compute-dtype `Cmp*` temp is
/// consumed by output 0 inside a `Select` (tested `!= 0.0f`) AND stored as
/// `(unsigned char)` by output 1, so a cast on the shared node would corrupt
/// output 0's value (mutation M9). f16/bf16 re-promote first: the root lowered in
/// the house promote-demote convention is the demoted `__float2half(<pred_f32>)`,
/// and 1.0/0.0 round-trip f32→half→f32 bit-exactly, so the conversion pair is
/// value-exact (and folded by ptxas). `store_expr_of(plan, 0, root)` is
/// byte-identical to the pre-generalization `store_expr`.
pub(crate) fn store_expr_of(plan: &KernelPlan<'_>, j: usize, root: String) -> String {
    let d = plan.out_dtype_of(j);
    if d == plan.dtype {
        return root;
    }
    // The hetero elementwise store is exactly the U8 keep-mask (a `Cmp*`
    // predicate, pinned by `assert_valid_out_dtype`; the bincount-I32 scatter
    // narrows itself via `scatter_combine_store`, never through here). The
    // per-element narrowing is the shared [`cast_scalar`] routine — one source of
    // truth with the generated cast helper ([`emit_cast_helper`]). BYTE-IDENTICAL
    // to the prior hand-inlined forms: `cast_scalar(F16, U8, r)` =
    // `(unsigned char)__half2float(r)`, `(Bf16, U8)` =
    // `(unsigned char)__bfloat162float(r)`, every other `(_, U8)` =
    // `(unsigned char)r`.
    cast_scalar(plan.dtype, d, &root)
}

fn emit_scalar(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let name = format!(
        "baracuda_gen_{}_{}_scalar",
        plan.op_name,
        dtype_tag(plan.dtype)
    );
    let n = plan.n_inputs;
    let octype = out_ctype_of(plan, 0, ctype);
    let mut s = header(plan, &name);
    for i in 0..n {
        s.push_str(&format!("    const {ctype}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {octype}* __restrict__ out,\n"));
    s.push_str(&format!(
        "    long long n{})\n{{\n",
        param_args(plan.body, param_ctype(plan))
    ));
    s.push_str("    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long step = (long long)gridDim.x * blockDim.x;\n");
    let acc = |idx: u8| format!("in{idx}[i]");
    let (prelude, root) = lower_dag(
        &ExprDag::from_expr(plan.body),
        ctype,
        &Lowering {
            leaf: &acc,
            reduced: &|i| unreachable!("no Reduced leaf outside RowReduce: red{i}"),
            coord: &|d| {
                panic!(
                    "cuda backend: Coord({d}) reached the scalar emitter — Coord bodies \
                     lower via Strided only (the linear-index kernels have no per-axis \
                     coordinates)"
                )
            },
            unary: &|op, x| cuda_unary(op, x, plan.dtype),
            binary: &|op, a, b| cuda_binary(op, a, b, plan.dtype),
            select: &|c, a, b| cuda_select(c, a, b, plan.dtype),
        },
    );
    let store = store_expr_of(plan, 0, root);
    if prelude.is_empty() {
        s.push_str(&format!("    for (; i < n; i += step) out[i] = {store};\n"));
    } else {
        // Shared interiors: hoist the `tmp` block inside the loop (its RHS reads
        // the per-`i` inputs), so a shared value is computed once per element.
        s.push_str("    for (; i < n; i += step) {\n");
        for decl in &prelude {
            s.push_str(&format!("        {decl}\n"));
        }
        s.push_str(&format!("        out[i] = {store};\n    }}\n"));
    }
    s.push_str("}\n");
    GeneratedKernel { name, source: s }
}

fn emit_strided(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let rank = plan.key.rank as usize;
    let n = plan.n_inputs as usize;
    // Increment 4: the single GATHERED input (or `None` for an index-free op).
    // `(g, idx_op, axis, oob, index_dtype)`: `g` reads its `axis` coordinate from
    // integer operand `idx_op`. Index-free ⇒ every branch below collapses to the
    // pre-increment-4 string (byte-identical emission).
    let gather = crate::plan::gather_of(plan.read_index);
    // Increment 5: the SCATTERED output (or `None` for a write-Direct op).
    // `(idx_op, axis, combine, oob, index_dtype)`: the output writes its `axis`
    // coordinate from integer operand `idx_op`, combined by `combine`. The plan
    // gate pins gather ⊥ scatter (at most one is `Some`). Write-Direct ⇒ every
    // branch below collapses to the pre-increment-5 string (byte-identical).
    let scatter = crate::plan::scatter_of(plan.write_index);
    // The single INDEX-operand slot + its integer dtype, from EITHER a gather
    // (read side) or a scatter (write side) — they share the index-load machinery
    // (integer pointer type, own strided offset). At most one is present.
    let index_slot: Option<(usize, ElementKind)> = match (gather, scatter) {
        (Some((_, idx_op, _, _, idt)), _) => Some((idx_op as usize, idt)),
        (_, Some((idx_op, _, _, _, idt))) => Some((idx_op as usize, idt)),
        _ => None,
    };
    // The index dtype rides the ENTRY_POINT symbol (`gather_f32_i32` /
    // `scatter_f32_i64`) — it cannot ride the structure key (single operand-0
    // dtype). Index-free/write-Direct ⇒ no infix ⇒ byte-identical name.
    let idx_infix = index_slot.map_or(String::new(), |(_, idt)| format!("_{}", dtype_tag(idt)));
    // BASE_OFFSET SLICE: presence mask (ascending Runtime input indices + whether
    // the output is Runtime), the single source for the name suffix, the launch
    // args, and the entry pointer bumps below (so they can never disagree).
    let (off_inputs, out_off) = offset_operands(plan);
    let off_suffix = offset_suffix(&off_inputs, out_off);
    let name = format!(
        "baracuda_gen_{}_{}{}_strided_r{}{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        idx_infix,
        rank,
        off_suffix
    );
    let octype = out_ctype_of(plan, 0, ctype);
    let mut s = header(plan, &name);
    for i in 0..n {
        // The INDEX operand (gather or scatter) is an integer tensor — its pointer
        // takes the index ctype (`int`/`long long`), not the data ctype. Every
        // other input (and the index-free/write-Direct case) keeps the data ctype
        // ⇒ byte-identical.
        let ptype = match index_slot {
            Some((idx_op, idt)) if idx_op == i => {
                scalar_ctype(idt).expect("gate pins index dtype to I32/I64")
            }
            _ => ctype,
        };
        s.push_str(&format!("    const {ptype}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {octype}* __restrict__ out,\n"));
    // Extraction #1 (generated-vs-bespoke audit round 1): dims ride BY VALUE as
    // flattened scalar params (the constant bank) instead of global-memory
    // pointer arrays re-read every iteration — worth ~3x on the general path at
    // equal parallelism. Legal because every access below is compile-time
    // indexed (the rank is unrolled), so each `shape[d]` is simply the scalar
    // `shape{d}`.
    for d in 0..rank {
        s.push_str(&format!("    long long shape{d},\n"));
    }
    for i in 0..n {
        for d in 0..rank {
            s.push_str(&format!("    long long s{i}_{d},\n"));
        }
    }
    for d in 0..rank {
        s.push_str(&format!("    long long so_{d},\n"));
    }
    // Increment 4: the gathered DATA operand's extent along the gathered axis
    // (`src_dim_size` in the bespoke launcher) — the OOB bound. A dedicated
    // scalar slot (the iteration `shape{d}` is the OUTPUT/index shape, which
    // differs from the source extent along the gathered axis). Emitted ONLY for a
    // gather ⇒ index-free signature is byte-identical. Placed right before `n`.
    if gather.is_some() {
        s.push_str("    long long gext,\n");
    }
    // Increment 5: the scattered DESTINATION's extent along the scattered axis
    // (`out_dim_size` in the bespoke launcher) — the OOB bound + the clamp bound
    // for the write address. A dedicated scalar (the iteration `shape{d}` is the
    // UPDATES/source shape, which differs from the destination extent along the
    // scattered axis). Emitted ONLY for a scatter ⇒ write-Direct signature is
    // byte-identical. gather ⊥ scatter, so `gext`/`sext` are never both present.
    if scatter.is_some() {
        s.push_str("    long long sext,\n");
    }
    // BASE_OFFSET SLICE: a `long long off{i}` per Runtime input (ascending `i`),
    // then `long long offo` if the output is Runtime — the FROZEN ABI slot AFTER
    // gext/sext and BEFORE `n`. Element units, added to the operand base at kernel
    // entry (the bumps below). Offset-free ⇒ no args ⇒ byte-identical signature.
    for i in &off_inputs {
        s.push_str(&format!("    long long off{i},\n"));
    }
    if out_off {
        s.push_str("    long long offo,\n");
    }
    s.push_str(&format!(
        "    long long n{})\n{{\n",
        param_args(plan.body, param_ctype(plan))
    ));
    // BASE_OFFSET SLICE: bump each Runtime operand's base pointer by its launch arg
    // as the FIRST body statements — BEFORE the fully-broadcast hoist (so a hoisted
    // operand's `h{k} = in{k}[0]` reads element off{k}, not element 0) and BEFORE
    // the gather/scatter pre-passes (every composed address, including the index
    // buffer and the gathered/scattered data, is then formed relative to the bumped
    // base). Parameter pointers are local copies (`const T*`/`T*` are reassignable).
    // OOB is a CALLER PRECONDITION (the k/n_out trust model): off + max declared-
    // extent address must land in-bounds — no bound is emitted, no address is formed
    // before the bump. Offset-free ⇒ no bumps ⇒ byte-identical body.
    for i in &off_inputs {
        s.push_str(&format!("    in{i} += off{i};\n"));
    }
    if out_off {
        s.push_str("    out += offo;\n");
    }
    // Hoist fully-broadcast inputs: their offset is loop-invariant, load once.
    // The INDEX operand (gather OR scatter) is never hoisted here (it is handled
    // specially in the index pre-pass, with the correct integer ctype) — skip it.
    for k in 0..n {
        let is_index = matches!(index_slot, Some((idx_op, _)) if idx_op == k);
        if !is_index && is_fully_broadcast(plan.key.operands[k], rank) {
            s.push_str(&format!("    {ctype} h{k} = in{k}[0];\n"));
        }
    }
    s.push_str("    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long step = (long long)gridDim.x * blockDim.x;\n");
    s.push_str("    for (; i < n; i += step) {\n");
    s.push_str("        long long lin = i;\n");
    // Row-major unravel (last axis fastest), unrolled over the iteration rank.
    // Scalar constant-bank params, no empty-axis guard (n>0 in the loop body ⇒ no
    // empty axis is ever reached). The SAME shared routine backs the freestanding
    // coord-unravel helper (single source of truth); this call is byte-identical
    // to the prior hand-inlined emission.
    s.push_str(&emit_unravel_decomp(
        (0..rank).rev(),
        "        ",
        "lin",
        |d| format!("c{d}"),
        |d| format!("shape{d}"),
        false,
    ));
    // Increment 4 — GATHER pre-pass. Load the runtime index, then compute a
    // CLAMPED effective index used for the DATA load (always in-bounds for a
    // non-empty gathered axis, so the load is memcheck-clean regardless of OOB
    // policy) plus, for Skip/ZeroFill, an `goob` predicate for the store. Negative
    // indices are OOB (no from-end wrap) — bespoke parity.
    if let Some((g, idx_op, axis, oob, _idt)) = gather {
        // The index operand's own strided offset (a full-shape index varies on
        // every axis; a 1-D index_select/embedding index broadcasts to
        // `c{axis}·stride`). Never viewed (gather ⊥ view) ⇒ identity remap.
        let ioff = offset_expr(
            plan.key.operands[idx_op as usize],
            &format!("s{idx_op}"),
            rank,
            None,
        );
        s.push_str(&format!("        long long gidx_off = {ioff};\n"));
        s.push_str(&format!(
            "        long long gidx_raw = (long long)in{idx_op}[gidx_off];\n"
        ));
        // Clamp the LOAD address into `[0, gext-1]` (memcheck-safe for gext>=1):
        s.push_str(
            "        long long gidx_clamped = gidx_raw < 0 ? 0 : (gidx_raw >= gext ? gext - 1 : gidx_raw);\n",
        );
        if matches!(
            oob,
            crate::ir::OobPolicy::Skip | crate::ir::OobPolicy::ZeroFill
        ) {
            s.push_str("        bool goob = (gidx_raw < 0) || (gidx_raw >= gext);\n");
        }
        // The gathered DATA operand's offset substitutes `gidx_clamped·stride[axis]`
        // for the `c{axis}·stride[axis]` term.
        let goff = gathered_offset_expr(
            plan.key.operands[g],
            &format!("s{g}"),
            rank,
            axis as usize,
            "gidx_clamped",
        );
        s.push_str(&format!("        long long o{g} = {goff};\n"));
    }
    // Increment 5 — SCATTER pre-pass. Load the runtime destination index, clamp it
    // into `[0, sext-1]` for the WRITE address (memcheck-safe for a non-empty
    // scattered axis regardless of the value), and compute the `soob` skip
    // predicate. Negative indices are OOB (no from-end wrap) — bespoke parity. The
    // scattered OUTPUT offset `oo` substitutes `sidx_clamped·stride_out[axis]` for
    // the `c{axis}·stride_out[axis]` term (the write-side mirror of the gather
    // input substitution). Output slot is key operand `n` (after the `n_inputs`
    // input slots).
    if let Some((idx_op, _axis, _combine, _oob, _idt)) = scatter {
        // The index operand's own strided offset (a full-shape scatter index varies
        // on every axis; a 1-D index_add index broadcasts to `c{axis}·stride`).
        let ioff = offset_expr(
            plan.key.operands[idx_op as usize],
            &format!("s{idx_op}"),
            rank,
            None,
        );
        s.push_str(&format!("        long long sidx_off = {ioff};\n"));
        s.push_str(&format!(
            "        long long sidx_raw = (long long)in{idx_op}[sidx_off];\n"
        ));
        s.push_str(
            "        long long sidx_clamped = sidx_raw < 0 ? 0 : (sidx_raw >= sext ? sext - 1 : sidx_raw);\n",
        );
        // v1 scatter OOB is always Skip (plan gate) — predicate the combine store.
        s.push_str("        bool soob = (sidx_raw < 0) || (sidx_raw >= sext);\n");
    }
    for k in 0..n {
        // The gathered DATA operand's offset is emitted in the pre-pass above; the
        // INDEX operand's offset is `gidx_off`/`sidx_off` (its value is an address,
        // never a body leaf) — skip both here.
        let handled = matches!(gather, Some((g, idx_op, _, _, _))
            if k == g || idx_op as usize == k)
            || matches!(index_slot, Some((idx_op, _)) if idx_op == k && scatter.is_some());
        if !handled && !is_fully_broadcast(plan.key.operands[k], rank) {
            // Item 01: input `k` may be read through a Permute view (a transposed
            // read) — `input_perm` remaps its stride indices. Identity/view-free ⇒
            // `None` ⇒ byte-identical offset.
            let off = offset_expr(
                plan.key.operands[k],
                &format!("s{k}"),
                rank,
                input_perm(plan, k),
            );
            s.push_str(&format!("        long long o{k} = {off};\n"));
        }
    }
    // The OUTPUT offset. A scattered output (increment 5) substitutes
    // `sidx_clamped·stride_out[axis]` for the `c{axis}·stride_out[axis]` term (the
    // write-side mirror of gather); a write-Direct output is the identity offset
    // (byte-identical). The OUTPUT is never viewed (views are an input-read
    // property), so `perm` is always `None`.
    let oo = match scatter {
        Some((_, axis, _, _, _)) => gathered_offset_expr(
            plan.key.operands[n],
            "so",
            rank,
            axis as usize,
            "sidx_clamped",
        ),
        None => offset_expr(plan.key.operands[n], "so", rank, None),
    };
    s.push_str(&format!("        long long oo = {oo};\n"));
    let acc = |idx: u8| {
        if is_fully_broadcast(plan.key.operands[idx as usize], rank) {
            format!("h{idx}")
        } else {
            format!("in{idx}[o{idx}]")
        }
    };
    // Coord(d) reads the unraveled per-axis coordinate `c{d}` — emitted
    // UNCONDITIONALLY above for every axis (the output-offset unravel), so the
    // coordinate exists even in the all-contiguous-input case where no operand
    // needs a per-axis offset — cast to the compute ctype. Exact while the
    // axis extent fits the dtype's exact-integer range (f32: 2^24, f64: 2^53
    // — the documented caller precondition); the plan gate + the
    // `assert_coord_lowerable` backstop pin the dtype to f32/f32s/f64 and the
    // axis to `< rank` before this closure can run.
    let coord = |d: u8| {
        assert!(
            (d as usize) < rank,
            "cuda backend: Coord({d}) axis out of range for the rank-{rank} strided \
             unravel (the backstop should have refused this plan)"
        );
        match plan.dtype {
            ElementKind::F32 | ElementKind::F32Strict => format!("(float)c{d}"),
            ElementKind::F64 => format!("(double)c{d}"),
            other => panic!(
                "cuda backend: Coord({d}) has no {other:?} coordinate spelling — \
                 f32/f64 only (the backstop should have refused this plan)"
            ),
        }
    };
    let (prelude, root) = lower_dag(
        &ExprDag::from_expr(plan.body),
        ctype,
        &Lowering {
            leaf: &acc,
            reduced: &|i| unreachable!("no Reduced leaf outside RowReduce: red{i}"),
            coord: &coord,
            unary: &|op, x| cuda_unary(op, x, plan.dtype),
            binary: &|op, a, b| cuda_binary(op, a, b, plan.dtype),
            select: &|c, a, b| cuda_select(c, a, b, plan.dtype),
        },
    );
    for decl in &prelude {
        s.push_str(&format!("        {decl}\n"));
    }
    if let Some((_, _, combine, _, _)) = scatter {
        // Increment 5 — SCATTER combine store. The value LOAD inside `root` is
        // always in-bounds (iteration coord over the updates domain); the OOB
        // policy (v1: Skip) predicates only the WRITE — a duplicate/OOB target
        // never reads out of bounds (the write address is `sidx_clamped`). Every
        // combine is guarded by `if (!soob)` (skip the OOB target — bespoke
        // `continue;`). The scatter narrows a hetero body ITSELF (bincount's
        // `(int)(1.0)`), so it takes the RAW lowered `root`, NOT `store_expr`'s
        // u8-only narrowing. Reaching `emit_strided` for an FP `atomicAdd` scatter
        // means this is the `Nondeterministic` variant (the base is the gather-sum);
        // an integer atomicAdd / Assign / atomicMax-min is the unconditional base.
        let hetero = plan.out_dtype != plan.dtype;
        s.push_str(&format!(
            "        {}\n",
            scatter_combine_store(combine, octype, &root, hetero)
        ));
    } else {
        let stored = store_expr_of(plan, 0, root);
        // Increment 4 — GATHER store policy. The DATA load inside `stored` is
        // always in-bounds (`gidx_clamped`), so the policy only shapes the WRITE:
        //   - Clamp     → always store the clamped-index value (no predicate).
        //   - Skip      → store only when in-range (bespoke gather `continue;`).
        //   - ZeroFill  → store 0 on OOB, else the value (bespoke embedding).
        // Index-free ops take the plain store — byte-identical.
        match gather {
            Some((_, _, _, crate::ir::OobPolicy::Skip, _)) => {
                s.push_str(&format!("        if (!goob) out[oo] = {stored};\n"));
            }
            Some((_, _, _, crate::ir::OobPolicy::ZeroFill, _)) => {
                let zero = zero_store_literal(octype);
                s.push_str(&format!(
                    "        out[oo] = goob ? ({zero}) : ({stored});\n"
                ));
            }
            // Clamp or index-free: unconditional store.
            _ => {
                s.push_str(&format!("        out[oo] = {stored};\n"));
            }
        }
    }
    s.push_str("    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Spell a [`WriteCombine`] store of the lowered body `root` into `out[oo]`,
/// guarded by the OOB skip predicate `soob` (increment 5). The atomic combines
/// route to the native CUDA overload for `octype`, casting `root` to `octype` —
/// which also narrows a hetero body (bincount: a `Const(1)` `f64` literal into an
/// `I32` counts cell → `(int)(1.0)` = 1) to the right atomic overload; `i64`
/// (`long long`) atomicAdd goes through the `unsigned long long` reinterpret (CUDA
/// has no signed-64 atomicAdd; two's-complement add matches). `Assign` stores the
/// body verbatim (uniform) or `(octype)`-cast (`hetero`, matching store_expr's
/// narrow but NOT its u8-only arm). The plan gate + `assert_scatter_lowerable` pin
/// the combine/dtype pair, so an unreachable pairing panics (a mis-route).
fn scatter_combine_store(
    combine: crate::ir::WriteCombine,
    octype: &str,
    root: &str,
    hetero: bool,
) -> String {
    use crate::ir::WriteCombine;
    match combine {
        WriteCombine::Assign => {
            let val = if hetero {
                format!("({octype})({root})")
            } else {
                root.to_string()
            };
            format!("if (!soob) out[oo] = {val};")
        }
        WriteCombine::AtomicAdd => match octype {
            "long long" => format!(
                "if (!soob) atomicAdd((unsigned long long*)&out[oo], (unsigned long long)({root}));"
            ),
            "float" | "double" | "int" => {
                format!("if (!soob) atomicAdd(&out[oo], ({octype})({root}));")
            }
            other => panic!(
                "cuda backend: scatter atomicAdd has no v1 spelling for octype '{other}' \
                 (f16/bf16/u8 need the bespoke CAS helper) — the gate should have refused it"
            ),
        },
        WriteCombine::AtomicMax => scatter_atomic_minmax("atomicMax", octype, root),
        WriteCombine::AtomicMin => scatter_atomic_minmax("atomicMin", octype, root),
    }
}

/// Native integer `atomicMax`/`atomicMin` store (increment 5). Integer only in
/// v1 (`int`/`long long` have native signed overloads); the gate rejects float.
fn scatter_atomic_minmax(op: &str, octype: &str, val: &str) -> String {
    match octype {
        "int" | "long long" => format!("if (!soob) {op}(&out[oo], ({octype})({val}));"),
        other => panic!(
            "cuda backend: scatter {op} has no v1 spelling for octype '{other}' \
             (integer-only) — the gate should have refused it"
        ),
    }
}

/// Independent emitter backstop for a GATHERED plan (increment 4), beside the
/// plan gate `plan::assert_valid_gather` (the 0a lesson: gate every layer). An
/// index-free plan (empty `read_index` / all-`Direct`) returns immediately —
/// byte-identical. For a plan carrying a real gather, this pins the facts the
/// `emit_strided` substitution relies on so a future schedule change can't route
/// a gathered operand through a linear-index emitter that ignores the index:
///
/// - **Elementwise + single-output** — the only access/arity increment 4 emits.
/// - **exactly one gathered input** — the emitter substitutes exactly one axis.
/// - **`Schedule::Strided`** — the sole schedule whose offset emitter folds the
///   value-substitution (a data-dependent address never coalesces, so the
///   vectorized/scalar/packed emitters must never see it).
/// - per gathered input: `index_operand < n_inputs`, `index_operand != g`, an
///   integer `index_dtype`, `axis < rank`, the index operand is not itself
///   gathered — the same rules as the plan gate, held independently here.
fn assert_gather_lowerable(plan: &KernelPlan<'_>) {
    if crate::plan::gather_of(plan.read_index).is_none() {
        return; // index-free / all-Direct — the established path, unchanged.
    }
    let name = plan.op_name;
    assert!(
        matches!(plan.access, Access::Elementwise),
        "cuda backend: gathered op '{name}' must be Access::Elementwise (increment \
         4 gathers are Elementwise-only)"
    );
    assert!(
        plan.n_outputs == 1,
        "cuda backend: gathered op '{name}' must be single-output ({} outputs) — a \
         gathered multi-output op is a deferred composition",
        plan.n_outputs
    );
    let n_gathered = plan.read_index.iter().filter(|r| !r.is_direct()).count();
    assert!(
        n_gathered == 1,
        "cuda backend: gathered op '{name}' must have exactly one gathered input, \
         got {n_gathered}"
    );
    assert!(
        matches!(plan.schedule, Schedule::Strided),
        "cuda backend: gathered op '{name}' must lower on the Strided schedule (a \
         data-dependent address cannot coalesce; only the strided emitter folds \
         the value-substitution), got {:?}",
        plan.schedule
    );
    let (g, index_operand, axis, _oob, index_dtype) =
        crate::plan::gather_of(plan.read_index).expect("gather present (checked above)");
    assert!(
        (index_operand as usize) < plan.n_inputs as usize && index_operand as usize != g,
        "cuda backend: gathered op '{name}' index_operand ({index_operand}) invalid \
         for {} inputs / gathered input {g}",
        plan.n_inputs
    );
    assert!(
        matches!(
            index_dtype,
            ElementKind::I32 | ElementKind::I64 | ElementKind::U32
        ),
        "cuda backend: gathered op '{name}' index_dtype must be I32/I64/U32, got \
         {index_dtype:?}"
    );
    assert!(
        (axis as usize) < plan.key.rank as usize,
        "cuda backend: gathered op '{name}' axis ({axis}) >= rank ({})",
        plan.key.rank
    );
    assert!(
        plan.read_index[index_operand as usize].is_direct(),
        "cuda backend: gathered op '{name}' index operand ({index_operand}) must \
         not itself be gathered"
    );
}

/// Independent emitter backstop for a SCATTERED plan (increment 5), beside the
/// plan gate `plan::assert_valid_scatter` (the 0a lesson: gate every layer). A
/// write-Direct plan returns immediately — byte-identical. For a real scatter this
/// pins the facts the `emit_strided`/`emit_scatter_gathersum` substitution relies
/// on: Elementwise + single-output + Strided + `index_operand < n_inputs` +
/// integer index dtype + `axis < rank` + a legal combine/dtype pair + not also a
/// gather. Held independently of the plan gate.
fn assert_scatter_lowerable(plan: &KernelPlan<'_>) {
    let Some((index_operand, axis, combine, _oob, index_dtype)) = plan.write_index.scatter() else {
        return; // write-Direct — the established path, unchanged.
    };
    let name = plan.op_name;
    assert!(
        matches!(plan.access, Access::Elementwise),
        "cuda backend: scattered op '{name}' must be Access::Elementwise (increment \
         5 scatters are Elementwise-only)"
    );
    assert!(
        plan.n_outputs == 1,
        "cuda backend: scattered op '{name}' must be single-output ({} outputs)",
        plan.n_outputs
    );
    assert!(
        crate::plan::gather_of(plan.read_index).is_none(),
        "cuda backend: scattered op '{name}' must not also be a gather (a fused \
         gather+scatter is a deferred composition)"
    );
    assert!(
        matches!(plan.schedule, Schedule::Strided),
        "cuda backend: scattered op '{name}' must lower on the Strided schedule (a \
         data-dependent write address cannot coalesce), got {:?}",
        plan.schedule
    );
    assert!(
        (index_operand as usize) < plan.n_inputs as usize,
        "cuda backend: scattered op '{name}' index_operand ({index_operand}) >= \
         n_inputs ({})",
        plan.n_inputs
    );
    assert!(
        matches!(
            index_dtype,
            ElementKind::I32 | ElementKind::I64 | ElementKind::U32
        ),
        "cuda backend: scattered op '{name}' index_dtype must be I32/I64/U32, got \
         {index_dtype:?}"
    );
    assert!(
        (axis as usize) < plan.key.rank as usize,
        "cuda backend: scattered op '{name}' axis ({axis}) >= rank ({})",
        plan.key.rank
    );
    assert!(
        crate::plan::combine_legal_for_dtype(combine, plan.out_dtype),
        "cuda backend: scattered op '{name}' combine {combine:?} illegal for output \
         dtype {:?}",
        plan.out_dtype
    );
}

/// The BASE_OFFSET SLICE presence + operand list for a plan: the ascending input
/// indices carrying a [`crate::ir::BaseOffset::Runtime`] offset, and whether the
/// output does. Empty vec + `false` for an offset-free plan (byte-identical). THE
/// single place the emitter derives the entry-point name suffix, the launch args,
/// and the pointer bumps from — so they can never disagree; the backstop reads
/// presence off it too (presence == `!ins.is_empty() || out`).
fn offset_operands(plan: &KernelPlan<'_>) -> (Vec<usize>, bool) {
    let ins: Vec<usize> = plan
        .base_offsets
        .iter()
        .enumerate()
        .filter(|(_, b)| !b.is_zero())
        .map(|(i, _)| i)
        .collect();
    (ins, !plan.out_base_offset.is_zero())
}

/// The `_off<indices>[o]` entry-point suffix for a plan (BASE_OFFSET SLICE): the
/// ascending `Runtime` input indices, then a trailing `o` if the output is
/// `Runtime`. Empty for an offset-free plan ⇒ the name is byte-identical to today.
/// Examples: `_off1` (input 1 only), `_off1o` (input 1 + output), `_offo` (output
/// only).
fn offset_suffix(off_inputs: &[usize], out_off: bool) -> String {
    if off_inputs.is_empty() && !out_off {
        return String::new();
    }
    let mut sfx = String::from("_off");
    for i in off_inputs {
        sfx.push_str(&i.to_string());
    }
    if out_off {
        sfx.push('o');
    }
    sfx
}

/// Independent emitter backstop for an OFFSETTED plan (BASE_OFFSET SLICE), beside
/// the plan gate `plan::assert_valid_offsets` (the 0a lesson: gate every layer). An
/// offset-free plan (empty `base_offsets` + a `Zero` output) returns immediately —
/// byte-identical. For a plan carrying a real `Runtime` offset this pins the facts
/// the single per-pointer entry-bump lowering relies on, so a future schedule change
/// (or the FP-atomic-scatter gather-sum reroute) can't route an offsetted operand
/// through an emitter that never bumps the base:
///
/// - **Elementwise + single-output** — the only access/arity BASE_OFFSET emits.
/// - **`Schedule::Strided`** — the sole schedule whose emitter bumps the base
///   pointer; `emit_vectorized`/`emit_scalar`/the packed + multi-out variants and
///   every non-elementwise emitter iterate a bare linear index that would silently
///   ignore the bump.
/// - **not an FP-atomicAdd scatter** — that combine reroutes to
///   `emit_scatter_gathersum` (a DIFFERENT emitter under the Strided schedule, which
///   does not bump); an offset + FP-atomic-scatter is out of scope. An `Assign`
///   scatter + offset lowers in `emit_strided` and IS supported.
fn assert_offsets_lowerable(plan: &KernelPlan<'_>) {
    let (off_inputs, out_off) = offset_operands(plan);
    if off_inputs.is_empty() && !out_off {
        return; // offset-free — the established path, unchanged.
    }
    let name = plan.op_name;
    assert!(
        matches!(plan.access, Access::Elementwise),
        "cuda backend: offsetted op '{name}' must be Access::Elementwise (BASE_OFFSET \
         is Elementwise-only)"
    );
    assert!(
        plan.n_outputs == 1,
        "cuda backend: offsetted op '{name}' must be single-output ({} outputs) — a \
         runtime offset on a multi-output op is a deferred composition",
        plan.n_outputs
    );
    assert!(
        matches!(plan.schedule, Schedule::Strided),
        "cuda backend: offsetted op '{name}' must lower on the Strided schedule (only \
         the strided emitter bumps the operand base pointer; the vectorized/scalar/ \
         packed emitters iterate a bare linear index that ignores the bump), got {:?}",
        plan.schedule
    );
    // An FP-atomicAdd scatter reroutes to emit_scatter_gathersum (a different Strided
    // emitter that never bumps) — reject an offset there rather than mis-emit.
    if let Some((_, _, combine, _, _)) = plan.write_index.scatter() {
        assert!(
            !combine.is_fp_atomic_add(plan.out_dtype),
            "cuda backend: offsetted op '{name}' is an FP-atomicAdd scatter, which \
             reroutes to the gather-sum base (no pointer bump) — an offset + \
             nondeterministic-FP-scatter is out of v1 scope (Assign scatter + offset \
             is supported)"
        );
    }
}

/// The **`Nondeterministic` FP-atomic-add scatter** variant (increment 5). Only
/// offered for an FP `atomicAdd` scatter — the one order-nondeterministic
/// schedule. The base `lower()` is the deterministic gather-sum
/// ([`emit_scatter_gathersum`]); THIS variant is the fast atomic scatter
/// ([`emit_strided`]'s combine store), the same schedule the bespoke
/// `scatter_add`/`embedding_backward` use. Per the house variant rule a
/// [`VariantFidelity::Nondeterministic`] variant is NEVER selected silently —
/// only through an honest FKC contract whose determinism block flips to
/// `nondeterministic` (`VariantFidelity::determinism_str`); the `launch_note`
/// states the run-to-run non-determinism so a caller's precision policy is the
/// sole gate. Integer atomicAdd / Assign / atomicMax-min are deterministic and
/// stay the unconditional base (no variant here). `racecheck` on this variant's
/// atomics is legitimate (an `atomicAdd` is not a hazard); `memcheck` must be
/// clean (no OOB scatter).
fn scatter_atomic_variant(plan: &KernelPlan<'_>) -> Option<Variant> {
    let (_, _, combine, _, _) = plan.write_index.scatter()?;
    if !combine.is_fp_atomic_add(plan.out_dtype) {
        return None; // deterministic combine ⇒ the base IS the atomic scatter.
    }
    let ctype = scalar_ctype(plan.dtype)?;
    let atomic = emit_strided(plan, ctype);
    let entry = atomic.name.clone();
    Some(Variant {
        tag: "atomic",
        kernels: vec![atomic],
        fidelity: VariantFidelity::Nondeterministic,
        launch_note: format!(
            "single-launch FP-atomicAdd scatter ({entry}<<<ceil(n_upd/B), B>>>): iterates \
             the UPDATE domain (n_upd threads), atomicAdd's each value into out[scatter \
             index]; pass `sext` = destination extent along the scattered axis. \
             RUN-TO-RUN NON-DETERMINISTIC — floating atomicAdd completion order varies and \
             FP add is non-associative, so the result bits differ between launches. \
             determinism: {det}. Never select silently; the deterministic gather-sum base \
             (`_scatter_gathersum_`) is the default route. compute-sanitizer racecheck \
             treats the atomics as legitimate; memcheck must report 0 OOB.",
            det = VariantFidelity::Nondeterministic.determinism_str(),
        ),
    })
}

/// Emit the DETERMINISTIC **gather-sum** base kernel for an FP `atomicAdd` scatter
/// (increment 5). Because floating `atomicAdd` is order-nondeterministic, the
/// scatter form is offered only as the `Nondeterministic` variant; THIS is the
/// base `lower()` route (never silently nondeterministic — the house variant
/// rule). One thread per DESTINATION cell scans the entire update domain and sums
/// the values whose scattered target is this cell, in a FIXED (linear update
/// index) order — race-free (one writer per cell) + reproducible. This is the
/// bespoke `segment_sorted_kernel` strategy (one owner per output, in-order
/// sweep, no atomics) generalized to an arbitrary index; O(n_out · n_upd), the
/// documented cost of determinism. The destination is accumulated INTO (bespoke
/// `scatter_add`/`index_add` add into a caller-populated `dst`): `out[oo] +=
/// acc`, exact for a single owner.
fn emit_scatter_gathersum(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let rank = plan.key.rank as usize;
    let n = plan.n_inputs as usize;
    let (index_operand, axis, _combine, _oob, index_dtype) = plan
        .write_index
        .scatter()
        .expect("emit_scatter_gathersum called on a non-scatter plan");
    let idx_op = index_operand as usize;
    let idxct = scalar_ctype(index_dtype).expect("gate pins index dtype to I32/I64");
    let octype = out_ctype_of(plan, 0, ctype);
    // The value operand (updates) is the non-index input; its offset uses the
    // UPDATE-domain coordinate. bincount-style (index==value) has no separate
    // value input — but bincount is integer (deterministic), so it never routes
    // here; an FP scatter always has a distinct value operand. Guard anyway.
    let val_op = (0..n).find(|&k| k != idx_op).unwrap_or(0);
    // Emitter backstop (review #5): this path sums `in{val_op}` DIRECTLY (it does
    // not lower `plan.body`), so the body MUST be the identity value read — the
    // plan gate `assert_valid_scatter` pins it, and this independent check keeps a
    // future routing change from silently dropping a composed body. (A `Const`
    // body is integer bincount, which is deterministic and never routes here.)
    assert!(
        matches!(plan.body, ScalarExpr::Input(v) if *v as usize == val_op),
        "cuda backend: the scatter gather-sum base requires an identity Input({val_op}) \
         body (it sums the value operand directly); got {:?}",
        plan.body
    );
    let name = format!(
        "baracuda_gen_{}_{}_{}_scatter_gathersum_r{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        dtype_tag(index_dtype),
        rank
    );
    let mut s = header(plan, &name);
    // Inputs: value ctype for the value operand, index ctype for the index operand.
    for i in 0..n {
        let ptype = if i == idx_op { idxct } else { ctype };
        s.push_str(&format!("    const {ptype}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {octype}* __restrict__ out,\n"));
    // Destination shape (the iteration domain here) + update-domain shape.
    for d in 0..rank {
        s.push_str(&format!("    long long oshape{d},\n"));
    }
    for d in 0..rank {
        s.push_str(&format!("    long long ushape{d},\n"));
    }
    // Strides: value operand (s{val_op}_), index operand (s{idx_op}_), output (so_).
    for d in 0..rank {
        s.push_str(&format!("    long long sv_{d},\n"));
    }
    for d in 0..rank {
        s.push_str(&format!("    long long si_{d},\n"));
    }
    for d in 0..rank {
        s.push_str(&format!("    long long so_{d},\n"));
    }
    // n_out = destination numel (iteration count); n_upd = update-domain numel.
    s.push_str("    long long n_out,\n    long long n_upd)\n{\n");
    s.push_str("    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long step = (long long)gridDim.x * blockDim.x;\n");
    let acc = if matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict) {
        "double"
    } else {
        "float"
    };
    let zero = if acc == "double" { "0.0" } else { "0.0f" };
    // Load one update value at update-coordinate offset `uoff`, promoting f16/bf16.
    let load_val = |off: &str| match plan.dtype {
        ElementKind::F16 => format!("__half2float(in{val_op}[{off}])"),
        ElementKind::Bf16 => format!("__bfloat162float(in{val_op}[{off}])"),
        _ => format!("in{val_op}[{off}]"),
    };
    let store_acc = |v: String| match plan.dtype {
        ElementKind::F16 => format!("__float2half({v})"),
        ElementKind::Bf16 => format!("__float2bfloat16({v})"),
        _ => v,
    };
    s.push_str("    for (; i < n_out; i += step) {\n");
    // Unravel the destination coordinate.
    s.push_str("        long long lin = i;\n");
    s.push_str(&emit_unravel_decomp(
        (0..rank).rev(),
        "        ",
        "lin",
        |d| format!("oc{d}"),
        |d| format!("oshape{d}"),
        false,
    ));
    // Destination offset (identity — the owner cell for this thread), over the
    // OUTPUT coordinate `oc{d}` (NOT the default `c{d}`).
    let oo = offset_expr_coord(plan.key.operands[n], "so", "oc", rank);
    s.push_str(&format!("        long long oo = {oo};\n"));
    s.push_str(&format!("        {acc} acc = {zero};\n"));
    // Scan the update domain; sum the values whose scattered target == this cell.
    s.push_str("        for (long long j = 0; j < n_upd; ++j) {\n");
    s.push_str("            long long ulin = j;\n");
    for d in (0..rank).rev() {
        s.push_str(&format!(
            "            long long uc{d} = ulin % ushape{d}; ulin /= ushape{d};\n"
        ));
    }
    // The index value at this update coordinate `uc{d}` (full-shape, or 1-D via a
    // broadcast si_ stride that drops the non-axis terms).
    let ioff = offset_expr_coord(plan.key.operands[idx_op], "si", "uc", rank);
    s.push_str(&format!(
        "            long long sidx = (long long)in{idx_op}[{ioff}];\n"
    ));
    // Match: the scattered target coord equals this destination cell. Axis term is
    // `sidx == oc{axis}`; every other axis term is `uc{d} == oc{d}`.
    let mut conds: Vec<String> = Vec::new();
    conds.push(format!("sidx == oc{axis}"));
    for d in 0..rank {
        if d != axis as usize {
            conds.push(format!("uc{d} == oc{d}"));
        }
    }
    let cond = conds.join(" && ");
    s.push_str(&format!("            if ({cond}) {{\n"));
    // Value offset over the update coord `uc{d}`.
    let uoff = offset_expr_coord(plan.key.operands[val_op], "sv", "uc", rank);
    s.push_str(&format!("                long long uoff = {uoff};\n"));
    s.push_str(&format!("                acc += {};\n", load_val("uoff")));
    s.push_str("            }\n");
    s.push_str("        }\n");
    // Accumulate INTO the existing destination (dst += Σ), race-free (one owner).
    let existing = match plan.dtype {
        ElementKind::F16 => "__half2float(out[oo])".to_string(),
        ElementKind::Bf16 => "__bfloat162float(out[oo])".to_string(),
        _ => "out[oo]".to_string(),
    };
    s.push_str(&format!(
        "        out[oo] = {};\n",
        store_acc(format!("({existing} + acc)"))
    ));
    s.push_str("    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Independent emitter backstop for a multi-output plan (increment 1), beside
/// the plan gate `plan::assert_valid_multi_output` (the 0a lesson: gate every
/// layer). Pins the structural facts the N-store emitters rely on: Elementwise
/// access, a uniform output dtype, and a key operand count of exactly
/// `n_inputs + n_outputs`. A `Reduced`/`Coord` leaf in a body is caught by the
/// panicking `reduced`/`coord` closures the multi emitters pass (and by the
/// plan gate + `assert_coord_lowerable` run above).
fn assert_multi_output_lowerable(plan: &KernelPlan<'_>) {
    assert!(
        matches!(plan.access, Access::Elementwise),
        "cuda backend: multi-output op '{}' must be Access::Elementwise",
        plan.op_name
    );
    // Per-output dtype legality (G5) — the SAME per-output rule the plan gate
    // enforces (G1), held INDEPENDENTLY here (the 0a lesson: gate every layer). A
    // UNIFORM output (`out_dtype_of(j) == plan.dtype`) is unconstrained; a HETERO
    // output must be a U8 keep-mask with a `Cmp*`-root body (the exact 0.0/1.0 the
    // store converts). Plus: any hetero output FORBIDS the Vectorized schedule (a
    // U8 output has no packed vector store — G4 forces Scalar/Strided, so
    // `emit_vectorized_multi` never receives a hetero plan).
    let bodies = plan.output_bodies();
    let mut any_hetero = false;
    for (j, body) in bodies.iter().enumerate() {
        let d = plan.out_dtype_of(j);
        if d != plan.dtype {
            any_hetero = true;
            assert!(
                d == ElementKind::U8,
                "cuda backend: multi-output op '{}' output {j} has hetero dtype \
                 {d:?} — the only lowerable per-output hetero dtype is U8 (a \
                 comparison-predicate keep-mask)",
                plan.op_name
            );
            assert!(
                matches!(body, ScalarExpr::Binary(bop, _, _) if bop.is_cmp()),
                "cuda backend: multi-output op '{}' output {j} is U8 but its body \
                 root is not a Cmp* — the u8 store would truncate a non-predicate \
                 silently",
                plan.op_name
            );
        }
    }
    assert!(
        !(any_hetero && matches!(plan.schedule, Schedule::Vectorized { .. })),
        "cuda backend: multi-output op '{}' has a hetero (U8) output on the \
         Vectorized schedule — a U8 keep-mask has no packed vector store, so a \
         hetero plan must lower Scalar/Strided (G4/G5)",
        plan.op_name
    );
    let want = plan.n_inputs as usize + plan.n_outputs as usize;
    assert!(
        plan.key.n_operands as usize == want,
        "cuda backend: multi-output op '{}' key carries {} operands, expected \
         n_inputs+n_outputs = {want}",
        plan.op_name,
        plan.key.n_operands
    );
}

/// Independent emitter backstop for a layout-viewed plan (item 01), beside the
/// plan gate `plan::assert_valid_views` (the 0a lesson: gate every layer). A
/// view-free plan (empty `views`) or an all-`Identity`/same-rank-`Reshape` plan
/// (no address-affecting view) returns immediately — byte-identical. For a plan
/// carrying a real addressing view, this pins the facts the offset remap relies
/// on so a future schedule change can't route a viewed operand through a
/// linear-index emitter that ignores the view:
///
/// - **Elementwise + single-output** — the only access/arity item 01 emits.
/// - **`Schedule::Strided`** — the sole schedule whose `offset_expr` folds the
///   per-operand stride remap; a viewed read is non-contiguous, so the
///   vectorized/scalar/packed emitters must never see it.
/// - Each view is re-validated (`is_valid`), a `Permute` operand's broadcast mask
///   is empty, a `Broadcast` view agrees with the key, a `Reshape` is same-rank —
///   the same rules as the plan gate, held independently here.
fn assert_views_lowerable(plan: &KernelPlan<'_>) {
    if !plan.views.iter().any(crate::plan::view_is_addressing) {
        return; // view-free / all-identity — the established path, unchanged.
    }
    let name = plan.op_name;
    assert!(
        matches!(plan.access, Access::Elementwise),
        "cuda backend: viewed op '{name}' must be Access::Elementwise (item 01 \
         views are Elementwise-only)"
    );
    assert!(
        plan.n_outputs == 1,
        "cuda backend: viewed op '{name}' must be single-output ({} outputs) — a \
         viewed multi-output op is a deferred composition",
        plan.n_outputs
    );
    assert!(
        matches!(plan.schedule, Schedule::Strided),
        "cuda backend: viewed op '{name}' must lower on the Strided schedule (a \
         viewed read is non-contiguous; only the strided emitter folds the \
         per-operand stride remap), got {:?}",
        plan.schedule
    );
    let rank = plan.key.rank;
    for (i, v) in plan.views.iter().enumerate() {
        assert!(
            v.is_valid(rank),
            "cuda backend: viewed op '{name}' input {i} view {v:?} invalid for rank \
             {rank}"
        );
        let o = plan.key.operands[i];
        match v {
            crate::ir::View::Identity => {}
            crate::ir::View::Permute { .. } => assert!(
                o.bcast.is_empty(),
                "cuda backend: viewed op '{name}' input {i} Permute view with a \
                 broadcast mask ({:#04x}) — Permute ⊥ Broadcast in v1",
                o.bcast.0
            ),
            crate::ir::View::Broadcast { bcast } => assert!(
                bcast.0 & !o.bcast.0 == 0,
                "cuda backend: viewed op '{name}' input {i} Broadcast view declares \
                 axes ({:#04x}) the key does not broadcast ({:#04x})",
                bcast.0,
                o.bcast.0
            ),
            crate::ir::View::Reshape { producer_rank } => assert!(
                *producer_rank == rank,
                "cuda backend: viewed op '{name}' input {i} rank-change Reshape \
                 (producer_rank {producer_rank} != rank {rank}) is out of item-01 \
                 scope"
            ),
        }
    }
}

/// Panicking `reduced`/`coord` closures for the multi-output elementwise
/// emitters: v1 multi-output bodies carry neither leaf (rejected at the plan
/// gate + emitter backstops), so reaching one is a bug, not an honest miss.
fn multi_reduced_panic(op_name: &str) -> impl Fn(u8) -> String + '_ {
    move |i| {
        panic!(
            "cuda backend: Reduced({i}) in multi-output op '{op_name}' — multi-output v1 is elementwise-map only (no reduction)"
        )
    }
}
fn multi_coord_panic(op_name: &str) -> impl Fn(u8) -> String + '_ {
    move |d| {
        panic!(
            "cuda backend: Coord({d}) in multi-output op '{op_name}' — multi-output v1 is elementwise-map only (Coord bodies are deferred)"
        )
    }
}

/// Emit a **multi-output scalar** elementwise kernel (increment 1): one linear
/// grid-stride kernel that writes `n_outputs` contiguous outputs from a shared
/// body-DAG. All output bodies are interned into ONE [`ExprDag`]
/// ([`crate::ir::ExprDag::from_exprs`]) so a value shared between outputs — the
/// `dy` load, an interior product — is emitted once (hoisted `tmp` / shared load)
/// and referenced by each store: strictly fewer global loads than N separate
/// kernels. The store loop grows to N `out{j}[i] = …;`.
fn emit_scalar_multi(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let n_in = plan.n_inputs as usize;
    let n_out = plan.n_outputs as usize;
    let name = format!(
        "baracuda_gen_{}_{}_mo{}_scalar",
        plan.op_name,
        dtype_tag(plan.dtype),
        n_out
    );
    let mut s = header(plan, &name);
    for i in 0..n_in {
        s.push_str(&format!("    const {ctype}* __restrict__ in{i},\n"));
    }
    for j in 0..n_out {
        // Per-output pointer type (the hetero multi-output / dropout-class
        // increment): `unsigned char*` for a U8 keep-mask output, `ctype*` for a
        // uniform output — byte-identical for a uniform-multi op.
        s.push_str(&format!(
            "    {}* __restrict__ out{j},\n",
            out_ctype_of(plan, j, ctype)
        ));
    }
    s.push_str(&format!(
        "    long long n{})\n{{\n",
        param_args_multi(&plan.output_bodies(), param_ctype(plan))
    ));
    s.push_str("    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long step = (long long)gridDim.x * blockDim.x;\n");
    let acc = |idx: u8| format!("in{idx}[i]");
    let dag = ExprDag::from_exprs(&plan.output_bodies());
    let (prelude, roots) = lower_dag_multi(
        &dag,
        ctype,
        &Lowering {
            leaf: &acc,
            reduced: &multi_reduced_panic(plan.op_name),
            coord: &multi_coord_panic(plan.op_name),
            unary: &|op, x| cuda_unary(op, x, plan.dtype),
            binary: &|op, a, b| cuda_binary(op, a, b, plan.dtype),
            select: &|c, a, b| cuda_select(c, a, b, plan.dtype),
        },
        false,
    );
    s.push_str("    for (; i < n; i += step) {\n");
    for decl in &prelude {
        s.push_str(&format!("        {decl}\n"));
    }
    for (j, root) in roots.iter().enumerate() {
        // Per-output store conversion at the STORE SITE (never on the shared DAG
        // node): a U8 keep-mask stores `(unsigned char)root`, a uniform output
        // stores `root` raw — byte-identical for a uniform-multi op.
        s.push_str(&format!(
            "        out{j}[i] = {};\n",
            store_expr_of(plan, j, root.clone())
        ));
    }
    s.push_str("    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Emit a **multi-output strided** elementwise kernel (increment 1): the
/// coordinate-unravel emitter generalized to `n_outputs` outputs, each with its
/// own per-axis stride array `so{j}_{d}` and unraveled offset `oo{j}`. Inputs
/// keep the single-output address math (per-operand `o{k}`, fully-broadcast
/// hoist), and the shared body-DAG is lowered once — so a strided multi-output
/// cell (a shape the contig-only bespoke backward siblings cannot serve without
/// a materialization pass) still loads each input once.
fn emit_strided_multi(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let rank = plan.key.rank as usize;
    let n_in = plan.n_inputs as usize;
    let n_out = plan.n_outputs as usize;
    let name = format!(
        "baracuda_gen_{}_{}_mo{}_strided_r{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        n_out,
        rank
    );
    let mut s = header(plan, &name);
    for i in 0..n_in {
        s.push_str(&format!("    const {ctype}* __restrict__ in{i},\n"));
    }
    for j in 0..n_out {
        // Per-output pointer type (the hetero multi-output / dropout-class
        // increment): `unsigned char*` for a U8 keep-mask output — byte-identical
        // for a uniform-multi op.
        s.push_str(&format!(
            "    {}* __restrict__ out{j},\n",
            out_ctype_of(plan, j, ctype)
        ));
    }
    for d in 0..rank {
        s.push_str(&format!("    long long shape{d},\n"));
    }
    for i in 0..n_in {
        for d in 0..rank {
            s.push_str(&format!("    long long s{i}_{d},\n"));
        }
    }
    for j in 0..n_out {
        for d in 0..rank {
            s.push_str(&format!("    long long so{j}_{d},\n"));
        }
    }
    s.push_str(&format!(
        "    long long n{})\n{{\n",
        param_args_multi(&plan.output_bodies(), param_ctype(plan))
    ));
    for k in 0..n_in {
        if is_fully_broadcast(plan.key.operands[k], rank) {
            s.push_str(&format!("    {ctype} h{k} = in{k}[0];\n"));
        }
    }
    s.push_str("    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long step = (long long)gridDim.x * blockDim.x;\n");
    s.push_str("    for (; i < n; i += step) {\n");
    s.push_str("        long long lin = i;\n");
    s.push_str(&emit_unravel_decomp(
        (0..rank).rev(),
        "        ",
        "lin",
        |d| format!("c{d}"),
        |d| format!("shape{d}"),
        false,
    ));
    for k in 0..n_in {
        if !is_fully_broadcast(plan.key.operands[k], rank) {
            // A viewed input on a multi-output op is rejected at the plan gate
            // (deferred composition), so `input_perm` is `None` here in v1 — but
            // threading it keeps the two strided emitters in lockstep.
            let off = offset_expr(
                plan.key.operands[k],
                &format!("s{k}"),
                rank,
                input_perm(plan, k),
            );
            s.push_str(&format!("        long long o{k} = {off};\n"));
        }
    }
    for j in 0..n_out {
        let oo = offset_expr(plan.key.operands[n_in + j], &format!("so{j}"), rank, None);
        s.push_str(&format!("        long long oo{j} = {oo};\n"));
    }
    let acc = |idx: u8| {
        if is_fully_broadcast(plan.key.operands[idx as usize], rank) {
            format!("h{idx}")
        } else {
            format!("in{idx}[o{idx}]")
        }
    };
    let dag = ExprDag::from_exprs(&plan.output_bodies());
    let (prelude, roots) = lower_dag_multi(
        &dag,
        ctype,
        &Lowering {
            leaf: &acc,
            reduced: &multi_reduced_panic(plan.op_name),
            coord: &multi_coord_panic(plan.op_name),
            unary: &|op, x| cuda_unary(op, x, plan.dtype),
            binary: &|op, a, b| cuda_binary(op, a, b, plan.dtype),
            select: &|c, a, b| cuda_select(c, a, b, plan.dtype),
        },
        false,
    );
    for decl in &prelude {
        s.push_str(&format!("        {decl}\n"));
    }
    for (j, root) in roots.iter().enumerate() {
        // Per-output store conversion at the STORE SITE (never on the shared DAG
        // node): a U8 keep-mask stores `(unsigned char)root` — byte-identical for a
        // uniform-multi op.
        s.push_str(&format!(
            "        out{j}[oo{j}] = {};\n",
            store_expr_of(plan, j, root.clone())
        ));
    }
    s.push_str("    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Backstop for the multi-output VECTORIZED emitters (native + packed): each
/// stores one packed vector type per output, so a hetero (U8 keep-mask) output —
/// which has no packed vector store — must NEVER reach them. G4 forces any hetero
/// multi-output plan to Scalar/Strided, and `assert_multi_output_lowerable` (G5)
/// re-asserts `schedule != Vectorized` for a hetero plan; this pins the invariant
/// at the emitter itself (the "gate every layer" rule) so a future schedule change
/// cannot silently route a U8 output through a `float4` store.
fn assert_multi_outputs_uniform(plan: &KernelPlan<'_>, emitter: &str) {
    for j in 0..plan.n_outputs as usize {
        let d = plan.out_dtype_of(j);
        assert!(
            d == plan.dtype,
            "cuda backend: {emitter} received multi-output op '{}' with a hetero \
             output {j} ({d:?} != {:?}) — a U8 keep-mask has no packed vector store; \
             a hetero plan must lower Scalar/Strided (G4/G5)",
            plan.op_name,
            plan.dtype
        );
    }
}

/// Emit a **multi-output vectorized** elementwise kernel (increment 1): the
/// native `float4`/`float2`/`double2` path with `n_outputs` output vectors. Each
/// input vector is loaded once (`v{i} = in{i}[i]`); each lane lowers ALL output
/// bodies through the shared DAG, assigning `vo{j}.{lane}`; then N vector stores.
fn emit_vectorized_multi(plan: &KernelPlan<'_>, vty: &str, lanes: &[&str]) -> GeneratedKernel {
    assert_multi_outputs_uniform(plan, "emit_vectorized_multi");
    let n_in = plan.n_inputs as usize;
    let n_out = plan.n_outputs as usize;
    let name = format!(
        "baracuda_gen_{}_{}_mo{}_co_v{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        n_out,
        lanes.len()
    );
    let mut s = header(plan, &name);
    for i in 0..n_in {
        s.push_str(&format!("    const {vty}* __restrict__ in{i},\n"));
    }
    for j in 0..n_out {
        s.push_str(&format!("    {vty}* __restrict__ out{j},\n"));
    }
    s.push_str(&format!(
        "    long long nv{})\n{{\n",
        param_args_multi(&plan.output_bodies(), param_ctype(plan))
    ));
    s.push_str("    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long step = (long long)gridDim.x * blockDim.x;\n");
    s.push_str("    for (; i < nv; i += step) {\n");
    for i in 0..n_in {
        s.push_str(&format!("        {vty} v{i} = in{i}[i];\n"));
    }
    for j in 0..n_out {
        s.push_str(&format!("        {vty} vo{j};\n"));
    }
    let sctype = scalar_ctype(plan.dtype).expect("vectorized dtype has a scalar ctype");
    let dag = ExprDag::from_exprs(&plan.output_bodies());
    for lane in lanes {
        let acc = |idx: u8| format!("v{idx}.{lane}");
        let (prelude, roots) = lower_dag_multi(
            &dag,
            sctype,
            &Lowering {
                leaf: &acc,
                reduced: &multi_reduced_panic(plan.op_name),
                coord: &multi_coord_panic(plan.op_name),
                unary: &|op, x| cuda_unary(op, x, plan.dtype),
                binary: &|op, a, b| cuda_binary(op, a, b, plan.dtype),
                select: &|c, a, b| cuda_select(c, a, b, plan.dtype),
            },
            false,
        );
        if prelude.is_empty() {
            for (j, root) in roots.iter().enumerate() {
                s.push_str(&format!("        vo{j}.{lane} = {root};\n"));
            }
        } else {
            s.push_str("        {\n");
            for decl in &prelude {
                s.push_str(&format!("            {decl}\n"));
            }
            for (j, root) in roots.iter().enumerate() {
                s.push_str(&format!("            vo{j}.{lane} = {root};\n"));
            }
            s.push_str("        }\n");
        }
    }
    for j in 0..n_out {
        s.push_str(&format!("        out{j}[i] = vo{j};\n"));
    }
    s.push_str("    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Emit a **multi-output packed** f16/bf16 vectorized kernel (increment 1): the
/// packed-pair path with `n_outputs` output vectors. Every output body must pack
/// ([`bodies_pack`]) — all leaves `Input`, no params — so the pair spellers apply
/// per lane; hoisting-all (via [`lower_dag_multi`]'s `hoist_all`) keeps Tier-B
/// pair-splits from duplicating text across the shared DAG.
fn emit_vectorized_packed_multi(plan: &KernelPlan<'_>, pk: &PackedKind) -> GeneratedKernel {
    assert_multi_outputs_uniform(plan, "emit_vectorized_packed_multi");
    let width = pk.fields.len() * 2;
    let n_in = plan.n_inputs as usize;
    let n_out = plan.n_outputs as usize;
    let name = format!(
        "baracuda_gen_{}_{}_mo{}_co_v{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        n_out,
        width
    );
    let vec_ty = format!("{name}_vec");
    let mut s = format!(
        "// Generated by baracuda-kernelgen — do not edit.\n// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    );
    if let Some(inc) = extra_include(plan.dtype) {
        s.push_str(inc);
    }
    s.push('\n');
    s.push_str(&format!(
        "struct __align__({}) {vec_ty} {{ {} {}; }};\n\n",
        pk.align,
        pk.pair_ty,
        pk.fields.join(", ")
    ));
    s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
    for i in 0..n_in {
        s.push_str(&format!("    const {vec_ty}* __restrict__ in{i},\n"));
    }
    for j in 0..n_out {
        s.push_str(&format!("    {vec_ty}* __restrict__ out{j},\n"));
    }
    s.push_str("    long long nv)\n{\n");
    s.push_str("    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long step = (long long)gridDim.x * blockDim.x;\n");
    s.push_str("    for (; i < nv; i += step) {\n");
    for i in 0..n_in {
        s.push_str(&format!("        {vec_ty} v{i} = in{i}[i];\n"));
    }
    for j in 0..n_out {
        s.push_str(&format!("        {vec_ty} vo{j};\n"));
    }
    let dag = ExprDag::from_exprs(&plan.output_bodies());
    for field in pk.fields {
        let acc = |idx: u8| format!("v{idx}.{field}");
        let (prelude, roots) = lower_dag_multi(
            &dag,
            pk.pair_ty,
            &Lowering {
                leaf: &acc,
                reduced: &multi_reduced_panic(plan.op_name),
                coord: &multi_coord_panic(plan.op_name),
                unary: &|op, x| packed_unary(op, x, plan.dtype),
                binary: &|op, a, b| packed_binary(op, a, b, plan.dtype),
                // Unreachable: `bodies_pack` (all-`body_packs`) excludes any
                // select body from the packed multi path — panic backstop.
                select: &|_, _, _| {
                    panic!(
                        "cuda backend: Select reached the packed multi-output emitter — \
                         body_packs excludes select bodies (they decline to the scalar \
                         path)"
                    )
                },
            },
            true,
        );
        if prelude.is_empty() {
            for (j, root) in roots.iter().enumerate() {
                s.push_str(&format!("        vo{j}.{field} = {root};\n"));
            }
        } else {
            s.push_str("        {\n");
            for decl in &prelude {
                s.push_str(&format!("            {decl}\n"));
            }
            for (j, root) in roots.iter().enumerate() {
                s.push_str(&format!("            vo{j}.{field} = {root};\n"));
            }
            s.push_str("        }\n");
        }
    }
    for j in 0..n_out {
        s.push_str(&format!("        out{j}[i] = vo{j};\n"));
    }
    s.push_str("    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Emit a **reduction** (one thread per output element, sequential fold). Two
/// paths, chosen by the schedule's [`ReduceAxisClass`] (item 03):
///
/// - **`InnerContig`** (empty axis mask = legacy last-axis default, or a single
///   contiguous trailing axis): `base = o*k`, a dense contiguous run — **byte-
///   identical to before item 03**.
/// - **outer / middle / multi axis** (`Outer`/`Middle`/`Multi`): a generalized
///   kept-axis unravel + strided reduced-axis fold, supporting non-last / multiple
///   reduced axes, a **strided** input, and **keepdim** (size-1 broadcast-back)
///   vs. collapse output. All classes lower to the same *sequential* fold in v2
///   (correctness-first; a block-parallel outer-axis kernel is a later drop-in).
///
/// The accumulator is `float` — `double` for f64 / f32-strict — so f16/bf16/f32
/// fold up-converted (more precise, and it avoids the missing `__half2` reduce).
/// `Sum`/`Mean` fold from a 0 identity; `Max`/`Min` seed the first element
/// (NaN-propagating, no ±∞ literal, no OOB read on an empty axis).
///
/// v1 scope (AOT build-time asserts — reductions are not in the JIT vocabulary,
/// so these never fire across the `synthesize` trust boundary): a single input,
/// float dtype. Multi-input (weighted) reductions and integer accumulation
/// (item 04) are follow-ups; the axes/keepdim/strided generalization is this item.
fn emit_reduction(
    plan: &KernelPlan<'_>,
    ctype: &str,
    rop: ReduceOp,
    precision: bool,
) -> GeneratedKernel {
    let tag = match rop {
        ReduceOp::Sum => "sum",
        ReduceOp::Mean => "mean",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        ReduceOp::Prod => "prod",
    };
    // Precision-first (`MorePrecise` variant): force a `double` accumulator and
    // route through the serial per-output general nest (below), NOT the InnerContig
    // block-tree. Double accumulation lifts the f32 error from O(√n·eps) to
    // ~0.5 ULP(f32) of the correctly-rounded reduction, and the serial nest is a
    // fixed order — bitwise-reproducible across hardware AND matching the CPU
    // oracle's fold order. Gated so the base lowering stays byte-identical.
    let prec = if precision { "_prec" } else { "" };
    let int_acc = matches!(
        plan.dtype,
        ElementKind::I32 | ElementKind::I64 | ElementKind::S8 | ElementKind::U8
    );
    assert!(
        matches!(
            plan.dtype,
            ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
        ) || int_acc,
        "reduction: float or integer (i8/u8/i32/i64) dtypes only; got {:?}",
        plan.dtype
    );
    // Integer Mean is a float-output (mixed-dtype) op — unrepresentable in a
    // single-dtype cell; use Sum/Max/Min for int.
    assert!(
        !(int_acc && matches!(rop, ReduceOp::Mean)),
        "reduction: integer Mean is out of scope; use Sum/Max/Min for i32/i64"
    );
    assert!(
        plan.n_inputs == 1,
        "reduction v1: single-input only (multi-input weighted reduction is a follow-up); got {}",
        plan.n_inputs
    );

    // Axis geometry comes from the IR (the source of truth), the fast-path split
    // from the schedule class — both derived from `Access::Reduction.axes`, so this
    // does not depend on the (separate) `StructureKey.reduce_axes` keying (step 5).
    let class = match plan.schedule {
        Schedule::Reduction { class, .. } => class,
        _ => unreachable!("emit_reduction on a non-reduction schedule"),
    };
    let (axes, keepdim, post) = match plan.access {
        Access::Reduction {
            axes,
            keepdim,
            post,
            ..
        } => (*axes, *keepdim, post),
        _ => unreachable!("emit_reduction on a non-reduction access"),
    };
    // The output pointer's scalar C type — the input `ctype` for a uniform-dtype
    // reduction, `unsigned char`/`long long` for a 0e hetero-out (any/all/count),
    // exactly the 0b pattern (`assert_valid_out_dtype` pinned the legal set).
    let octype = out_ctype_of(plan, 0, ctype);

    // Accumulate in double for f64 / f32-strict; float otherwise. Shared by both
    // paths — the leaf load up-converts, and the body is lowered in the acc width.
    // Integer reductions accumulate in `long long` (exact, overflow-resistant);
    // f64 / f32-strict in double; everything else in float. The leaf load is native
    // for int (the `_` arm below) and up-converts only f16/bf16/f32-strict.
    // `precision` forces double even for plain f32 (the MorePrecise variant).
    let dbl = precision || matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let acc = if int_acc {
        "long long"
    } else if dbl {
        "double"
    } else {
        "float"
    };
    let zero = if int_acc {
        "0"
    } else if dbl {
        "0.0"
    } else {
        "0.0f"
    };
    // Prod identity (increment 0e): `acc = 1; acc *= elem` (matches the bespoke
    // `ProdReduce::init() = T(1)`). Additive combines keep the `0` identity.
    let one = if int_acc {
        "1"
    } else if dbl {
        "1.0"
    } else {
        "1.0f"
    };
    // Int Max/Min empty-axis identity (KISS-OPS-6.11-0002): an empty reduced
    // axis must fold to the dtype's monoid identity — dtype-MIN for Max,
    // dtype-MAX for Min — not `0`. This reuses the scan emitter's
    // `type_extreme_lit` (same exact per-dtype literals: S8 -128/127, U8
    // 0/255, I32/I64 INT_MIN·MAX/LLONG_MIN·MAX). Seeding `acc` with this
    // identity instead of `zero` is a no-op for any NON-empty fold: the
    // `has`-flag logic below unconditionally overwrites `acc` with the first
    // real element it sees, regardless of the seed. It only changes the
    // result when the reduced extent is empty and the fold loop never runs —
    // exactly the case this fixes. Strictly int-gated: the FP Max/Min path
    // (zero-seeded) is unchanged by this variable.
    let mm_seed = if int_acc && matches!(rop, ReduceOp::Max | ReduceOp::Min) {
        type_extreme_lit(plan.dtype, matches!(rop, ReduceOp::Max))
    } else {
        zero.to_string()
    };
    let load = |i: u8| match plan.dtype {
        ElementKind::F32Strict => format!("(double)in{i}[idx]"),
        // f16/bf16 widen via the shared promotion intrinsic; f32/f64/int native.
        k => promote_load_f32(k, &format!("in{i}[idx]")),
    };
    // Int-reduction Cmp*/Const predicate lowering — the any/all/count fix.
    // `binary_f32`/`binary_f64` spell `Cmp*` as a FLOAT predicate
    // (`(float)a != (float)b ? 1.0f : 0.0f`) and `backend::const_lit` spells
    // `Const` as an f64 literal (`0.0`/`1.0`), both dtype-oblivious by
    // construction. Folding that FLOAT predicate into a `long long acc` (the
    // `count` shape, `acc += elem`) silently promotes `acc` to `float` in C —
    // an accumulator that stalls at 2^24 (16,777,216) for a count past it.
    // `plan::assert_int_op_admissibility` rule 4 already guarantees an
    // admitted int-reduction `Cmp*` node's operands are each a leaf
    // `Input`/`Reduced` or an exact 0/1 `Const` (the only shape any/all/count
    // ever author), so when this reduction's accumulator is integer
    // (`int_acc`) and the body/post ROOT is exactly a `Cmp*` node, lower it
    // here as a genuine INTEGER comparison (`a OP b ? 1 : 0` — no `(float)`
    // casts, no `1.0f`/`0.0f` literals) instead of routing through
    // `lower_expr`'s float spellers. Reachable ONLY when `int_acc` is true AND
    // the node is this reduction's body/post ROOT — every other caller of
    // `binary_f32`/`binary_f64`/`const_lit` (float reductions, every
    // elementwise emitter) is untouched. A plain int Sum/Max/Min/Prod body
    // (no Cmp at the root — just `Input(0)`) never matches the `Binary(bop,
    // ..) if bop.is_cmp()` arm below and falls through to the unchanged
    // `lower_expr` call, so its emit stays byte-identical to pre-fix.
    let int_cmp_operand = |e: &ScalarExpr,
                            leaf: &dyn Fn(u8) -> String,
                            red: &dyn Fn(u8) -> String|
     -> String {
        match e {
            ScalarExpr::Input(i) => leaf(*i),
            ScalarExpr::Reduced(i) => red(*i),
            ScalarExpr::Const(v) if *v == 0.0 => "0".to_string(),
            ScalarExpr::Const(v) if *v == 1.0 => "1".to_string(),
            other => unreachable!(
                "cuda backend: int-reduction Cmp* operand {other:?} is neither a \
                 leaf Input/Reduced nor an exact 0/1 Const — \
                 plan::assert_int_op_admissibility rule 4 guarantees this shape at \
                 the gate; a mismatch means the gate and this emitter have drifted"
            ),
        }
    };
    let int_reduction_predicate = |e: &ScalarExpr,
                                    leaf: &dyn Fn(u8) -> String,
                                    red: &dyn Fn(u8) -> String|
     -> Option<String> {
        if !int_acc {
            return None;
        }
        match e {
            ScalarExpr::Binary(bop, a, b) if bop.is_cmp() => {
                let a_s = int_cmp_operand(a, leaf, red);
                let b_s = int_cmp_operand(b, leaf, red);
                let c_op = match bop {
                    BinaryOp::CmpEq => "==",
                    BinaryOp::CmpNe => "!=",
                    BinaryOp::CmpLt => "<",
                    BinaryOp::CmpLe => "<=",
                    BinaryOp::CmpGt => ">",
                    BinaryOp::CmpGe => ">=",
                    _ => unreachable!("is_cmp() guarantees one of the six Cmp* variants"),
                };
                Some(format!("({a_s} {c_op} {b_s} ? 1 : 0)"))
            }
            _ => None,
        }
    };
    let elem = int_reduction_predicate(plan.body, &load, &|i| {
        unreachable!("no Reduced leaf outside RowReduce: red{i}")
    })
    .unwrap_or_else(|| {
        lower_expr(
            plan.body,
            &Lowering {
                leaf: &load,
                reduced: &|i| unreachable!("no Reduced leaf outside RowReduce: red{i}"),
                coord: &|d| {
                    panic!(
                        "cuda backend: Coord({d}) reached the reduction emitter — Coord is \
                         Elementwise-only (a coordinate along a folded axis is ambiguous)"
                    )
                },
                unary: &|op, x| {
                    if dbl {
                        unary_f64(op, x)
                    } else {
                        unary_f32(op, x)
                    }
                },
                binary: &|op, a, b| {
                    if dbl {
                        binary_f64(op, a, b)
                    } else {
                        binary_f32(op, a, b)
                    }
                },
                select: &|c, a, b| {
                    if dbl {
                        select_f64(c, a, b)
                    } else {
                        select_f32(c, a, b)
                    }
                },
            },
        )
    });
    // Convert an accumulator-width value to the output store. Uniform-dtype is
    // byte-identical to pre-0e (f16/bf16 demote, else native). A 0e hetero-out
    // converts the accumulator to the output dtype: U8 for a boolean any/all
    // (the value is exactly 0.0/1.0 by the Cmp* post, so the cast is exact) and
    // I64 for a count/sum-widening — exact for an int accumulator PRECISELY
    // because `int_reduction_predicate` (above) lowers the folded `Cmp*` as an
    // integer expression (`a != b ? 1 : 0`), so `acc += elem` is a true
    // integer add with no magnitude limit; exact for a FLOAT accumulator
    // (dtype F32/F64/F16/Bf16, out of scope for `int_reduction_predicate`)
    // only while count ≤ 2²⁴ — the documented caller precondition.
    let store = |v: String| -> String {
        if plan.out_dtype != plan.dtype {
            return match plan.out_dtype {
                ElementKind::U8 => format!("(unsigned char)({v})"),
                ElementKind::I64 => format!("(long long)({v})"),
                other => unreachable!("validated hetero out dtype {other:?}"),
            };
        }
        demote_store_f32(plan.dtype, &v)
    };
    // Apply the 0e fused post-expression (default = identity `Reduced(0)`) to the
    // finalized fold result, then convert for the store. The post lowers through
    // the SAME accumulator-width spellers as the fold body, with `Reduced(0)`
    // bound to a hoisted `red0` register (so a post referencing it more than once
    // — or the Mean quotient — is computed once). Returns `(optional red0 decl,
    // store rhs)`: the identity post yields `(None, store(finalized))`, so the
    // call site emits a single `<lvalue> = <rhs>;` byte-identical to pre-0e; a
    // real post yields the `{acc} red0 = <finalized>;` decl plus the posted rhs.
    let post_is_identity = matches!(post, ScalarExpr::Reduced(0));
    let post_apply = |finalized: String| -> (Option<String>, String) {
        if post_is_identity {
            return (None, store(finalized));
        }
        let post_leaf = |i: u8| -> String {
            panic!(
                "cuda backend: reduction post-expr Input({i}) reached the emitter \
                 — the post reads Reduced(0)/Const/Param only (validated by \
                 plan::assert_valid_reduction_post)"
            )
        };
        let post_reduced = |s: u8| -> String {
            assert_eq!(
                s, 0,
                "reduction post references Reduced({s}); only 0 exists"
            );
            "red0".to_string()
        };
        let posted = int_reduction_predicate(post, &post_leaf, &post_reduced).unwrap_or_else(|| {
            lower_expr(
                post,
                &Lowering {
                    leaf: &post_leaf,
                    reduced: &post_reduced,
                    coord: &|d| {
                        panic!(
                            "cuda backend: reduction post-expr Coord({d}) is Elementwise-only"
                        )
                    },
                    unary: &|op, x| {
                        if dbl {
                            unary_f64(op, x)
                        } else {
                            unary_f32(op, x)
                        }
                    },
                    binary: &|op, a, b| {
                        if dbl {
                            binary_f64(op, a, b)
                        } else {
                            binary_f32(op, a, b)
                        }
                    },
                    select: &|c, a, b| {
                        if dbl {
                            select_f64(c, a, b)
                        } else {
                            select_f32(c, a, b)
                        }
                    },
                },
            )
        });
        (Some(format!("{acc} red0 = {finalized};")), store(posted))
    };

    if class == ReduceAxisClass::InnerContig && !precision {
        // ---------- Contiguous last-axis reduction: ONE BLOCK per output row. ----------
        // (A precision-first cell skips this coalesced block-tree and takes the serial
        // general nest below — the block-tree order is reassociated, not the fixed
        // bitwise-reproducible order the MorePrecise contract promises.)
        // Coalesced reads (adjacent threads read adjacent row elements) + a warp-
        // shuffle / shared-mem block reduce (reuses `emit_block_reducers`). Replaces
        // the old one-thread-per-row *sequential* fold, which was memory-UNCOALESCED
        // (adjacent threads walked different rows, a row-length apart) and ~1.9×
        // slower on-device (see `ondevice/reduce_bench`). The block-tree order is the
        // documented deterministic order for this class. Launch contract: `blockDim`
        // a multiple of 32 (warp uniformity) and ≤ 1024 (shared-mem `smem[32]`).
        assert!(
            (0..plan.key.n_operands as usize)
                .all(|i| plan.key.operands[i].contig == Contiguity::Contig),
            "reduction inner-contig path: contiguous operands only (base = row*k)"
        );
        let name = format!(
            "baracuda_gen_{}_{}_reduce_{tag}",
            plan.op_name,
            dtype_tag(plan.dtype)
        );
        // Helpers named per (op, dtype) so concatenating kernels into one translation
        // unit never collides on a `__device__` symbol.
        let stem = format!("{}_{}", plan.op_name, dtype_tag(plan.dtype));
        let mut s = format!(
            "// Generated by baracuda-kernelgen — do not edit.\n// op: {} | cell: {}\n",
            plan.op_name,
            plan.key.to_token()
        );
        if let Some(inc) = extra_include(plan.dtype) {
            s.push_str(inc);
        }
        s.push('\n');
        let ops = std::collections::HashSet::from([rop]);
        emit_block_reducers(&mut s, acc, zero, one, &ops, &stem);
        s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
        s.push_str(&format!("    const {ctype}* __restrict__ in0,\n"));
        s.push_str(&format!("    {octype}* __restrict__ out,\n"));
        s.push_str(&format!(
            "    long long n_out,\n    long long k{})\n{{\n",
            param_args_multi(&[plan.body, post], param_ctype(plan))
        ));
        s.push_str(
            "    for (long long row = blockIdx.x; row < n_out; row += (long long)gridDim.x) {\n",
        );
        s.push_str("        long long base = row * k;\n");
        match rop {
            ReduceOp::Sum | ReduceOp::Mean => {
                s.push_str(&format!("        {acc} acc = {zero};\n"));
                s.push_str("        for (long long j = threadIdx.x; j < k; j += blockDim.x) {\n");
                s.push_str("            long long idx = base + j;\n");
                s.push_str(&format!("            acc += {elem};\n"));
                s.push_str("        }\n");
                // `block_sum` broadcasts the row total to every thread; Mean divides by
                // k (k==0 ⇒ 0/0 = NaN, matching the prior sequential path).
                let fin = if matches!(rop, ReduceOp::Mean) {
                    format!("block_sum_{stem}(acc) / ({acc})k")
                } else {
                    format!("block_sum_{stem}(acc)")
                };
                s.push_str(&format!("        {acc} r = {fin};\n"));
            }
            ReduceOp::Prod => {
                // Prod (0e): identity 1, `acc *= elem`, pass-through finalize. The
                // block_prod tree matches the Sum/Max/Min cooperative pattern.
                s.push_str(&format!("        {acc} acc = {one};\n"));
                s.push_str("        for (long long j = threadIdx.x; j < k; j += blockDim.x) {\n");
                s.push_str("            long long idx = base + j;\n");
                s.push_str(&format!("            acc *= {elem};\n"));
                s.push_str("        }\n");
                s.push_str(&format!("        {acc} r = block_prod_{stem}(acc);\n"));
            }
            ReduceOp::Max | ReduceOp::Min => {
                let cmp = if matches!(rop, ReduceOp::Max) {
                    ">"
                } else {
                    "<"
                };
                let suf = if matches!(rop, ReduceOp::Max) {
                    "max"
                } else {
                    "min"
                };
                // `has` carries "this lane saw an element" so idle lanes inject nothing
                // (no ±inf seed, headerless); a NaN sticks via `e != e`. The seed
                // itself only surfaces when `k == 0` (the fold loop never runs, so
                // `acc` reaches the block-reduce unmodified): `mm_seed` is the dtype
                // monoid identity for an int Max/Min (KISS-OPS-6.11-0002), `zero`
                // (unchanged) for FP and for Sum/Prod.
                s.push_str(&format!("        {acc} acc = {mm_seed}; int has = 0;\n"));
                s.push_str("        for (long long j = threadIdx.x; j < k; j += blockDim.x) {\n");
                s.push_str("            long long idx = base + j;\n");
                s.push_str(&format!("            {acc} e = {elem};\n"));
                s.push_str(&format!(
                    "            if (!has || e != e || e {cmp} acc) {{ acc = e; has = 1; }}\n"
                ));
                s.push_str("        }\n");
                s.push_str(&format!(
                    "        {acc} r = block_{suf}_{stem}(acc, has);\n"
                ));
            }
        }
        // The block_* helpers broadcast the result to all threads; thread 0 applies
        // the 0e post-expr (identity ⇒ byte-identical) and writes.
        let (decl, rhs) = post_apply("r".to_string());
        match decl {
            None => s.push_str(&format!(
                "        if (threadIdx.x == 0) out[row] = {rhs};\n"
            )),
            Some(d) => {
                s.push_str("        if (threadIdx.x == 0) {\n");
                s.push_str(&format!("            {d}\n"));
                s.push_str(&format!("            out[row] = {rhs};\n"));
                s.push_str("        }\n");
            }
        }
        s.push_str("    }\n}\n");
        return GeneratedKernel { name, source: s };
    }

    // ---------- General path: outer / middle / multi axis, strided input, keepdim. ----------
    // Compile-time axis split (kept vs reduced, ascending). The runtime supplies
    // per-input-axis extents `shape[]` + input strides `s0[]` + output strides `so[]`.
    let rank = plan.key.rank as usize;
    // EMPTY axes = the last-axis default. The base reaches this general path only
    // with EXPLICIT axes (an empty-axes cell is contiguous ⇒ the InnerContig fast
    // path above, which folds via `k`), so this normalization is a no-op for the
    // base; a precision-first cell, however, routes an empty-axes InnerContig cell
    // HERE and must materialize the reduced set as the last axis (like the oracle).
    let reduced: Vec<usize> = if axes.is_empty() {
        vec![rank - 1]
    } else {
        (0..rank).filter(|&d| axes.is_set(d as u8)).collect()
    };
    let kept: Vec<usize> = (0..rank).filter(|&d| !reduced.contains(&d)).collect();
    // Output-store injectivity guard (§5c/§8): the store must not alias. A broadcast
    // (stride-0) output axis that a *kept* coordinate varies over would collapse
    // distinct outputs onto one `oo`; a flipped output can write out of bounds. A
    // stride-0 *reduced* axis is harmless (size-1, coord 0 — the keepdim form). This
    // is an AOT author-error backstop (reductions never cross the JIT boundary; a
    // real reduction output is freshly-allocated dense).
    let out_key = plan.key.operands[(plan.key.n_operands as usize).saturating_sub(1)];
    let out_aliases = if keepdim {
        // Output axes align with the input axes.
        kept.iter().any(|&a| out_key.bcast.is_set(a as u8))
    } else {
        // Collapse: output axes are the kept axes in order (0..kept.len()).
        (0..kept.len()).any(|j| out_key.bcast.is_set(j as u8))
    };
    assert!(
        !out_aliases && !out_key.flipped,
        "reduction general path: output store must be injective (no broadcast on a kept \
         axis, no flip) — a non-dense output would alias or write OOB; got {:?}",
        out_key.contig
    );
    // The axis set + keepdim disambiguate the symbol so two axis-sets never collide.
    let name = format!(
        "baracuda_gen_{}_{}_reduce_{tag}_ax{:x}{}{prec}",
        plan.op_name,
        dtype_tag(plan.dtype),
        axes.0,
        if keepdim { "_kd" } else { "" }
    );
    let mut s = header(plan, &name);
    s.push_str(&format!("    const {ctype}* __restrict__ in0,\n"));
    s.push_str(&format!("    {octype}* __restrict__ out,\n"));
    // Extraction #1 (audit round 1): dims ride BY VALUE as flattened scalar
    // params (the constant bank) instead of global-pointer arrays re-read every
    // iteration - every access below is compile-time indexed, so each array
    // slot is simply a scalar.
    for d in 0..rank {
        s.push_str(&format!("    long long shape{d},\n")); // per-input-axis extents
    }
    for d in 0..rank {
        s.push_str(&format!("    long long s0_{d},\n")); // input strides
    }
    let n_out_dims = if keepdim { rank } else { kept.len() };
    for d in 0..n_out_dims {
        s.push_str(&format!("    long long so_{d},\n")); // output strides
    }
    s.push_str(&format!(
        "    long long n_out{})\n{{\n",
        param_args_multi(&[plan.body, post], param_ctype(plan))
    ));
    s.push_str("    long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long gstride = (long long)gridDim.x * blockDim.x;\n");
    s.push_str("    for (; o < n_out; o += gstride) {\n");
    // Unravel the output linear index over the kept axes (row-major, last kept axis
    // fastest). Reduced axes never enter the output enumeration. (Reduce-all ⇒ no
    // kept axes ⇒ n_out == 1, so no unravel and no `lin` — avoids an unused var.)
    if !kept.is_empty() {
        s.push_str("        long long lin = o;\n");
        s.push_str(&emit_unravel_decomp(
            kept.iter().rev().copied(),
            "        ",
            "lin",
            |a| format!("ck{a}"),
            |a| format!("shape{a}"),
            false,
        ));
    }
    // Input base offset from the kept coords (a broadcast kept axis has stride 0
    // and drops out naturally).
    let base_expr = if kept.is_empty() {
        "0".to_string()
    } else {
        kept.iter()
            .map(|a| format!("ck{a}*s0_{a}"))
            .collect::<Vec<_>>()
            .join(" + ")
    };
    s.push_str(&format!("        long long base = {base_expr};\n"));
    // Output offset: keepdim ⇒ `so` indexed by input axis (reduced axes are
    // size-1, output coord 0, so only kept terms contribute); collapse ⇒ `so`
    // indexed by the kept axis's position among the kept axes.
    let oo_expr = if kept.is_empty() {
        "0".to_string()
    } else if keepdim {
        kept.iter()
            .map(|a| format!("ck{a}*so_{a}"))
            .collect::<Vec<_>>()
            .join(" + ")
    } else {
        kept.iter()
            .enumerate()
            .map(|(j, a)| format!("ck{a}*so_{j}"))
            .collect::<Vec<_>>()
            .join(" + ")
    };
    s.push_str(&format!("        long long oo = {oo_expr};\n"));
    // Reduced-axis fold offsets are STRENGTH-REDUCED (extraction #2, from the
    // bespoke-legacy delta hunt): each nest level walks `roff{i} += s0_{r}`
    // instead of recomputing `base + Σ cr{r}·s0_{r}` per iteration — the 64-bit
    // multiply is emulated multi-instruction on consumer SMs, and this serial
    // fold runs at starved occupancy where nothing hides instruction latency.
    // Identical addresses (an exact integer identity), so value-preserving. A
    // broadcast reduced axis (stride 0) still re-reads the same element
    // `shape{r}` times — correct semantics.
    // The innermost counter is INT32 when the extent fits (extraction #3,
    // counter-guided + lab-measured): ptxas unrolls and software-pipelines an
    // int-counter loop far better than a `long long` one — the 64-bit counter
    // was the true bespoke-legacy delta (int32 measured 174.5 GB/s vs 94.1
    // rolled-ll and 130.7 bespoke at equal parallelism; every manual source
    // unroll REGRESSED to 42–56 GB/s by fighting ptxas's own schedule, and a
    // pointer-vs-end loop killed unrolling entirely at 14.8). The offset walk
    // stays 64-bit (extraction #2) so addressing is unchanged; extents above
    // INT_MAX take the `long long` fallback nest — same body, same element
    // order, uniform branch, so both nests are bit-identical.
    let emit_reduced_nest = |s: &mut String, body: &[String]| {
        s.push_str("        long long roff0 = base;\n");
        for (i, &r) in reduced.iter().enumerate() {
            if i + 1 < reduced.len() {
                s.push_str(&format!(
                    "        for (long long cr{r} = 0; cr{r} < shape{r}; ++cr{r}) {{\n"
                ));
                s.push_str(&format!(
                    "            long long roff{} = roff{};\n",
                    i + 1,
                    i
                ));
            }
        }
        let inner = reduced.len() - 1;
        let r = reduced[inner];
        let emit_inner = |s: &mut String, header: &str| {
            s.push_str(header);
            s.push_str(&format!("            long long idx = roff{inner};\n"));
            for line in body {
                s.push_str(line);
            }
            s.push_str(&format!("            roff{inner} += s0_{r};\n"));
            s.push_str("                }\n");
        };
        s.push_str(&format!("            if (shape{r} <= 2147483647LL) {{\n"));
        s.push_str(&format!("                int ext{r} = (int)shape{r};\n"));
        emit_inner(
            s,
            &format!("                for (int cr{r} = 0; cr{r} < ext{r}; ++cr{r}) {{\n"),
        );
        s.push_str("            } else {\n");
        emit_inner(
            s,
            &format!("                for (long long cr{r} = 0; cr{r} < shape{r}; ++cr{r}) {{\n"),
        );
        s.push_str("            }\n");
        // Close the outer reduced loops innermost-first; each level advances
        // its own offset just before its closing brace.
        for (i, &r) in reduced.iter().enumerate().rev() {
            if i + 1 < reduced.len() {
                s.push_str(&format!("            roff{i} += s0_{r};\n"));
                s.push_str("        }\n");
            }
        }
    };
    match rop {
        ReduceOp::Sum | ReduceOp::Mean => {
            s.push_str(&format!("        {acc} acc = {zero};\n"));
            emit_reduced_nest(&mut s, &[format!("            acc += {elem};\n")]);
        }
        ReduceOp::Prod => {
            // Prod (0e): identity 1, `acc *= elem` in the reduced nest.
            s.push_str(&format!("        {acc} acc = {one};\n"));
            emit_reduced_nest(&mut s, &[format!("            acc *= {elem};\n")]);
        }
        ReduceOp::Max | ReduceOp::Min => {
            let cmp = if matches!(rop, ReduceOp::Max) {
                ">"
            } else {
                "<"
            };
            // `has` seeds the first reduced element (all cr=0) without a ±∞ literal.
            // An empty reduced extent leaves `acc` at its seed, unmodified (matching
            // the fast path): `mm_seed` is the dtype monoid identity for an int
            // Max/Min (KISS-OPS-6.11-0002) — dtype-MIN for Max, dtype-MAX for Min —
            // and `zero` (unchanged) for FP.
            s.push_str(&format!("        {acc} acc = {mm_seed};\n"));
            s.push_str("        int has = 0;\n");
            emit_reduced_nest(
                &mut s,
                &[
                    format!("            {acc} e = {elem};\n"),
                    format!(
                        "            acc = has ? ((e != e || e {cmp} acc) ? e : acc) : e; has = 1;\n"
                    ),
                ],
            );
        }
    }
    let finalized = if matches!(rop, ReduceOp::Mean) {
        // Mean divisor = product of the reduced extents (not just the last axis).
        let divisor = reduced
            .iter()
            .map(|r| format!("shape{r}"))
            .collect::<Vec<_>>()
            .join(" * ");
        format!("acc / ({acc})({divisor})")
    } else {
        // Sum/Prod/Max/Min: pass-through finalize (Prod matches the bespoke
        // `ProdReduce::finalize` no-op).
        "acc".to_string()
    };
    // 0e post-expr (identity ⇒ byte-identical single store line).
    let (decl, rhs) = post_apply(finalized);
    if let Some(d) = decl {
        s.push_str(&format!("        {d}\n"));
    }
    s.push_str(&format!("        out[oo] = {rhs};\n"));
    s.push_str("    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Precision-first variant ([`VariantFidelity::MorePrecise`]) — a `double`-
/// accumulator SERIAL fold that fixes the length-growing float error of an f32
/// reduction's running sum AND pins a fixed, hardware-independent fold order
/// (bitwise-reproducible, matching the CPU oracle). It is NEVER the silent
/// default (it costs the coalesced block-tree's throughput); only an honest FKC
/// contract whose precision block advertises the tighter bound selects it (the
/// caller's precision policy decides). Declines every cell it cannot strictly
/// improve.
///
/// Two schedules earn it: a last-/outer-axis [`Schedule::Reduction`]
/// (`precision_first_reduction`) and a fused [`Schedule::RowReduce`]
/// softmax/layernorm/rmsnorm cell (`precision_first_rowreduce`). Dispatch on the
/// schedule up front, then hand off to the arm.
fn precision_first_variant(plan: &KernelPlan<'_>) -> Option<Variant> {
    match plan.schedule {
        Schedule::Reduction { op, .. } => precision_first_reduction(plan, op),
        Schedule::RowReduce { .. } => precision_first_rowreduce(plan),
        _ => None,
    }
}

/// Precision-first arm for a plain [`Schedule::Reduction`] f32 Sum/Mean cell:
/// forces `double` accumulation through `emit_reduction`'s serial per-output
/// general nest, giving ~0.5 ULP(f32) of the correctly-rounded reduction.
fn precision_first_reduction(plan: &KernelPlan<'_>, rop: ReduceOp) -> Option<Variant> {
    // Only Sum/Mean carry a length-growing running FP sum a wider accumulator
    // fixes. Max/Min are a pick and Prod a product — order-exact, so the double
    // fold is bit-identical to the base (no distinct precision offer).
    if !matches!(rop, ReduceOp::Sum | ReduceOp::Mean) {
        return None;
    }
    // F32 ONLY. F64/F32Strict already accumulate in `double` (this would be a
    // byte-identical duplicate). F16/Bf16 up-convert to float in the fold, but
    // their low-mantissa STORE rounds the extra accumulation precision away —
    // not worth a distinct variant. Integer reductions are exact (no FP axis).
    if plan.dtype != ElementKind::F32 {
        return None;
    }
    // `emit_reduction`'s single-input contract, and the serial general nest's
    // store-injectivity guard (a flipped or broadcast-on-a-kept-axis output would
    // alias / write OOB — the base emitter asserts the same; decline rather than
    // trip the assert, keeping the variant set total).
    if plan.n_inputs != 1 || plan.key.n_operands < 2 {
        return None;
    }
    // The serial general nest requires an injective, forward-dense output (it
    // asserts as much). A fresh reduction output is dense; a broadcast (stride-0)
    // or flipped (negative-stride) output would alias / write OOB — decline. The
    // input may be strided (the general path reads runtime strides), so only the
    // output is constrained.
    let out_key = plan.key.operands[(plan.key.n_operands as usize) - 1];
    if out_key.flipped || out_key.contig != Contiguity::Contig {
        return None;
    }
    let ctype = scalar_ctype(plan.dtype)?;
    let kernel = emit_reduction(plan, ctype, rop, true);
    let det = VariantFidelity::MorePrecise.determinism_str();
    Some(Variant {
        tag: "prec",
        kernels: vec![kernel],
        fidelity: VariantFidelity::MorePrecise,
        launch_note: format!(
            "Precision-first {} reduction: double accumulator, serial per-output fold \
             (1-D grid over n_out, one thread per output, grid-stride). MORE ACCURATE \
             than the f32-accumulate base — ~0.5 ULP(f32) of the correctly-rounded \
             reduction vs the base's length-growing float error. determinism: {det} \
             (fixed serial order, hardware-independent). NOT the default: trades the \
             coalesced block-tree's throughput for accuracy; select only when the \
             caller's precision policy asks for it.",
            match rop {
                ReduceOp::Mean => "mean",
                _ => "sum",
            }
        ),
    })
}

/// Precision-first arm for a fused [`Schedule::RowReduce`] cell
/// (softmax/layernorm/rmsnorm): the `_prec` serial-per-row double fold from
/// `emit_row_reduce_impl(.., precision=true)` — one thread per row, no block-tree,
/// a `double` accumulator through the sum/mean/variance stages. Bitwise across
/// hardware.
fn precision_first_rowreduce(plan: &KernelPlan<'_>) -> Option<Variant> {
    let Access::RowReduce { stages, .. } = plan.access else {
        return None;
    };
    // F32 ONLY, exactly as the reduction arm: F64/F32Strict already fold in
    // `double`; f16/bf16's low-mantissa STORE rounds the wider accumulation away.
    if plan.dtype != ElementKind::F32 {
        return None;
    }
    // At least one Sum/Mean stage — those carry the length-growing FP sum a wider
    // accumulator fixes (softmax's exp-sum denominator, layernorm/rmsnorm's
    // mean/variance). A pure Max/Min-only rowreduce is order-exact per row, so the
    // double fold would be bit-identical to the base — no distinct offer, decline.
    if !stages
        .iter()
        .any(|st| matches!(st.op, ReduceOp::Sum | ReduceOp::Mean))
    {
        return None;
    }
    // The serial epilogue writes `out[base+j]` forward-dense for j∈[0,k), exactly
    // like the base full-width output. A flipped (negative-stride) or broadcast
    // (stride-0) output would alias / write OOB — decline (the block base serves
    // it via its own path). The last operand in the key is the output.
    if plan.key.n_operands < 2 {
        return None;
    }
    let out_key = plan.key.operands[(plan.key.n_operands as usize) - 1];
    if out_key.flipped || out_key.contig != Contiguity::Contig {
        return None;
    }
    let ctype = scalar_ctype(plan.dtype)?;
    let kernel = emit_row_reduce_impl(plan, ctype, false, true);
    let det = VariantFidelity::MorePrecise.determinism_str();
    Some(Variant {
        tag: "prec",
        kernels: vec![kernel],
        fidelity: VariantFidelity::MorePrecise,
        launch_note: format!(
            "Precision-first {} rowreduce: double accumulator, serial per-row fold \
             (1-D grid over n_out, one thread per row, grid-stride). MORE ACCURATE than \
             the f32 base (double accumulation of the sum/mean/variance stages); \
             determinism: {det} (fixed serial order, hardware-independent). NOT the \
             default: trades the block-per-row parallelism for accuracy + \
             reproducibility.",
            plan.op_name
        ),
    })
}

/// Split-K schedule **variant** for the outer-axis reduction cell
/// (`out[c] = Σ_r in[r,c]`, rank-2 contiguous, reduce axis 0, `Sum`/`Mean`,
/// float dtypes).
///
/// The baseline general path is one thread per column — coalesced, but a single
/// sequential fold per column, so occupancy collapses when `cols` is small
/// relative to the GPU (measured 118 GB/s vs a ~230 GB/s ceiling at
/// `[8192,8192]` f32 on sm_89: 8192 columns = 32 blocks). Split-K parallelizes
/// over row chunks: `_splitk_partial` gives each `(column-tile, row-chunk)`
/// block a partial fold into a caller-provided workspace; `_splitk_combine`
/// folds the `n_chunks` partials per column and applies the Mean divisor + the
/// store narrowing. Both kernels keep adjacent threads on adjacent columns —
/// fully coalesced — and there are no atomics.
///
/// **Determinism/bits:** deterministic for a fixed `chunk_rows` (a fixed
/// two-level association), but a *different* association than the baseline's
/// sequential fold — [`VariantFidelity::ReassociatedDeterministic`]. Selectable
/// only through an honest contract (the caller's precision policy), never
/// silently; the baseline stays the default route.
///
/// **Keying caveat (adversarial pass 2026-07-02, defect 3):** the collapse-form
/// `StructureKey` token cannot carry the reduced-axis set (`derive_reduce_axes`
/// is provably undetermined for a rank-collapsed output — the item-03 step-5
/// finding), so an axis-0 and an axis-1 rank-2 reduction share one token. Any
/// future consumer that joins *variants* on `accept.structure_key` alone would
/// conflate this workspace ABI with a last-axis cell. Until the keepdim-form
/// convention (or an explicit axis field in variant contract front-matter)
/// lands, the variant's identity is `(token, entry_point)` — never token alone.
fn reduction_splitk_variant(plan: &KernelPlan<'_>) -> Option<Variant> {
    let rop = match plan.schedule {
        Schedule::Reduction {
            op,
            class: ReduceAxisClass::Outer,
            ..
        } => op,
        _ => return None,
    };
    // Max/Min split-K (partial maxima are order-free but the has-flag NaN fold
    // needs care) is a follow-up variant; int needs a long-long workspace.
    if !matches!(rop, ReduceOp::Sum | ReduceOp::Mean) {
        return None;
    }
    let (axes, keepdim, post) = match plan.access {
        Access::Reduction {
            axes,
            keepdim,
            post,
            ..
        } => (*axes, *keepdim, post),
        _ => return None,
    };
    // The split-K combine store applies neither the 0e post-expr nor a hetero-out
    // conversion (its ABI predates both), so decline those cells — the baseline
    // general path serves them correctly. (A post-aware / hetero split-K combine
    // is a follow-up; declining is never-worse, only leaves the perf variant off.)
    if !matches!(post, ScalarExpr::Reduced(0)) || plan.out_dtype != plan.dtype {
        return None;
    }
    // Specialized ABI (no stride arrays): the canonical rank-2 axis-0 cell with
    // dense input and output only.
    if plan.key.rank != 2 || axes.0 != 0b1 || plan.n_inputs != 1 {
        return None;
    }
    // Exactly [input, output] — a malformed key would alias `out_key` below
    // onto the input and test the wrong operand.
    if plan.key.n_operands != 2 {
        return None;
    }
    // The fixed signature has no `p{i}` launch slots; a Param body would emit
    // an undefined identifier. (The baseline appends `param_args`; plumbing
    // params through the two-launch protocol is a follow-up if ever needed.)
    if !params_used(plan.body).is_empty() {
        return None;
    }
    if !matches!(
        plan.dtype,
        ElementKind::F16
            | ElementKind::Bf16
            | ElementKind::F32
            | ElementKind::F32Strict
            | ElementKind::F64
    ) {
        return None;
    }
    let out_key = plan.key.operands[(plan.key.n_operands as usize).saturating_sub(1)];
    // `Contig` alone is NOT forward-dense: contiguity classification matches
    // |stride| against row-major, so a reversed dense view keys Contig +
    // `flipped` — a cell the baseline serves correctly via its runtime stride
    // arrays, but this stride-free `idx = r*cols + c` ABI would read out of
    // bounds. Require forward-dense on both ends.
    let in_key = plan.key.operands[0];
    if in_key.contig != Contiguity::Contig
        || out_key.contig != Contiguity::Contig
        || in_key.flipped
        || out_key.flipped
    {
        return None;
    }
    let ctype = scalar_ctype(plan.dtype)?;

    // Same accumulator / load / store discipline as `emit_reduction`.
    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let acc = if dbl { "double" } else { "float" };
    let zero = if dbl { "0.0" } else { "0.0f" };
    let load = |i: u8| match plan.dtype {
        ElementKind::F16 => format!("__half2float(in{i}[idx])"),
        ElementKind::Bf16 => format!("__bfloat162float(in{i}[idx])"),
        ElementKind::F32Strict => format!("(double)in{i}[idx]"),
        _ => format!("in{i}[idx]"),
    };
    let elem = lower_expr(
        plan.body,
        &Lowering {
            leaf: &load,
            reduced: &|i| unreachable!("no Reduced leaf outside RowReduce: red{i}"),
            coord: &|d| {
                panic!(
                    "cuda backend: Coord({d}) reached the split-K reduction variant — \
                     Coord is Elementwise-only"
                )
            },
            unary: &|op, x| {
                if dbl {
                    unary_f64(op, x)
                } else {
                    unary_f32(op, x)
                }
            },
            binary: &|op, a, b| {
                if dbl {
                    binary_f64(op, a, b)
                } else {
                    binary_f32(op, a, b)
                }
            },
            select: &|c, a, b| {
                if dbl {
                    select_f64(c, a, b)
                } else {
                    select_f32(c, a, b)
                }
            },
        },
    );
    let store = |finalized: String| -> String {
        match plan.dtype {
            ElementKind::F16 => format!("__float2half({finalized})"),
            ElementKind::Bf16 => format!("__float2bfloat16({finalized})"),
            _ => finalized,
        }
    };

    let stem = format!(
        "baracuda_gen_{}_{}_reduce_{}_ax{:x}{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        if matches!(rop, ReduceOp::Mean) {
            "mean"
        } else {
            "sum"
        },
        axes.0,
        if keepdim { "_kd" } else { "" },
    );

    // ---- Kernel 1: per-(column-tile, row-chunk) partial folds. ----
    let pname = format!("{stem}_splitk_partial");
    let mut p = header(plan, &pname);
    p.push_str(&format!("    const {ctype}* __restrict__ in0,\n"));
    p.push_str(&format!("    {acc}* __restrict__ ws,\n"));
    p.push_str("    long long rows,\n    long long cols,\n    long long chunk_rows)\n{\n");
    p.push_str("    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    p.push_str("    if (c >= cols) return;\n");
    p.push_str("    long long r0 = (long long)blockIdx.y * chunk_rows;\n");
    p.push_str("    long long r1 = r0 + chunk_rows; if (r1 > rows) r1 = rows;\n");
    p.push_str(&format!("    {acc} acc = {zero};\n"));
    p.push_str("    for (long long r = r0; r < r1; ++r) {\n");
    p.push_str("        long long idx = r * cols + c;\n");
    p.push_str(&format!("        acc += {elem};\n"));
    p.push_str("    }\n");
    p.push_str("    ws[(long long)blockIdx.y * cols + c] = acc;\n}\n");

    // ---- Kernel 2: fold the n_chunks partials per column; finalize + store. ----
    let mean = matches!(rop, ReduceOp::Mean);
    let cname = format!("{stem}_splitk_combine");
    let mut k = header(plan, &cname);
    k.push_str(&format!("    const {acc}* __restrict__ ws,\n"));
    k.push_str(&format!("    {ctype}* __restrict__ out,\n"));
    k.push_str("    long long cols,\n    long long n_chunks");
    if mean {
        k.push_str(",\n    long long rows"); // the Mean divisor
    }
    k.push_str(")\n{\n");
    k.push_str("    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    k.push_str("    if (c >= cols) return;\n");
    // Seed from the first partial (n_chunks >= 1 by the launch contract), not
    // from 0: `0 + w` flips a -0.0 column total to +0.0, so zero-seeding would
    // break the degenerate n_chunks=1 bit-identity with the baseline fold.
    k.push_str(&format!("    {acc} acc = ws[c];\n"));
    k.push_str("    for (long long k = 1; k < n_chunks; ++k) acc += ws[k * cols + c];\n");
    let finalized = if mean {
        format!("acc / ({acc})rows")
    } else {
        "acc".to_string()
    };
    k.push_str(&format!("    out[c] = {};\n}}\n", store(finalized)));

    Some(Variant {
        tag: "splitk",
        kernels: vec![
            GeneratedKernel {
                name: pname.clone(),
                source: p,
            },
            GeneratedKernel {
                name: cname.clone(),
                source: k,
            },
        ],
        fidelity: VariantFidelity::ReassociatedDeterministic,
        launch_note: format!(
            "two-launch protocol: (1) {pname}<<<dim3(ceil(cols/B), n_chunks), B>>>(in0, ws, \
             rows, cols, chunk_rows) with chunk_rows = ceil(rows/n_chunks) and workspace ws \
             of n_chunks*cols `{acc}` elements; (2) {cname}<<<ceil(cols/B), B>>>(ws, out, \
             cols, n_chunks{}). Deterministic for a fixed chunk_rows; association differs \
             from the single-pass baseline.",
            if mean { ", rows" } else { "" }
        ),
    })
}

/// Emit a **contraction** ([`Access::Contraction`]) — the terminal ORDER-3
/// node, v1 = the **skinny SIMT** schedule for the `Tiny`-M long-tail cell
/// (decode / FlashDecoding++ flat-GEMM): `out[mm,col] = epi(Σ_k lhs[mm,k] ·
/// rhs[k,col])` with one thread per output **column**, the ≤ 8 M-rows held in
/// predicated register accumulators, and the rhs streamed **coalesced**
/// (adjacent threads read adjacent columns of each rhs row) — the rhs, which
/// dominates traffic at Tiny M, is read exactly once at full bandwidth.
///
/// Extents (`m`, `n`, `k`) are launch arguments (the key carries structure
/// classes, never literals); the launch contract requires `m <= 8` — the
/// `Tiny` class ceiling the register file is sized for (the same
/// extent-as-caller-precondition discipline as RowReduce). Accumulation is
/// `float` (`double` for f64/f32-strict), the [`AccumSpec::WideFloat`] SIMT
/// policy — deterministic, sequential K order per output. Larger M/tiled/MMA
/// schedules join as bench-gated variants; all-`Large` cells route to the
/// vendor via the §7 gate and are never generated.
fn emit_contraction(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let Access::Contraction { epilogue, .. } = plan.access else {
        unreachable!("emit_contraction requires Access::Contraction");
    };
    let c = plan
        .key
        .contraction
        .expect("build_plan asserted contraction facts");
    assert_eq!(
        c.m,
        baracuda_kernel_vocab::SizeClass::Tiny,
        "contraction v1 emits the Tiny-M skinny schedule only; larger M classes \
         are the tiled variant's territory (and all-Large routes to the vendor)"
    );
    // The CONTRACTION accumulator follows the canonical sk3 `<acc>` lattice
    // (`baracuda_kernel_vocab::structure_key`'s `contraction_acc`, mirroring
    // `GemmSku::precision_guarantee`): f16/bf16/f32/f32-strict → `float`,
    // f64 → `double`. Unlike the REDUCTION emitters (no key `<acc>`
    // coordinate), F32Strict here must NOT up-convert to double — the gem
    // cell's key says `<acc>=f32`, the contract's `accumulation_type` must
    // match it (KISS-Contract §6.8 pin), and the F32Strict element contract is
    // "full IEEE 754 binary32 multiply-add THROUGHOUT" — a double accumulator
    // would produce different (non-binary32-chain) bits than declared.
    let dbl = matches!(plan.dtype, ElementKind::F64);
    assert!(
        matches!(
            plan.dtype,
            ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
        ),
        "contraction v1: float dtypes only; got {:?}",
        plan.dtype
    );
    let acc = if dbl { "double" } else { "float" };
    let zero = if dbl { "0.0" } else { "0.0f" };
    // Loads up-convert to the accumulator width, exactly as the reductions do
    // (F32Strict is already accumulator-width — no conversion).
    let load = |expr: String| match plan.dtype {
        ElementKind::F16 => format!("__half2float({expr})"),
        ElementKind::Bf16 => format!("__bfloat162float({expr})"),
        _ => expr,
    };
    let name = format!(
        "baracuda_gen_{}_{}_contract_{}{}{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        size_tag(c.m),
        size_tag(c.n),
        size_tag(c.k),
    );

    // Epilogue over the K-sum: Reduced(0) is the per-(row, col) accumulator.
    let red = |s: u8| {
        assert_eq!(s, 0, "contraction epilogue reads Reduced(0) only");
        "r0".to_string()
    };
    let epi = lower_expr(
        epilogue,
        &Lowering {
            // The only admissible epilogue Input is the fused bias `in{i}` (i>=2),
            // a per-column `[N]` bias read at `col` and up-converted to the
            // accumulator width, exactly as the operand loads are.
            leaf: &|i| {
                assert!(
                    i >= 2,
                    "contraction epilogue Input leaf must be the fused bias (>=2): in{i}"
                );
                let e = format!("in{i}[col]");
                match plan.dtype {
                    ElementKind::F16 => format!("__half2float({e})"),
                    ElementKind::Bf16 => format!("__bfloat162float({e})"),
                    _ => e,
                }
            },
            reduced: &red,
            coord: &|d| {
                panic!(
                    "cuda backend: Coord({d}) reached the contraction emitter — the \
                     (m, n) epilogue space is not the elementwise coordinate space; \
                     Coord is Elementwise-only"
                )
            },
            unary: &|op, x| {
                if dbl {
                    unary_f64(op, x)
                } else {
                    unary_f32(op, x)
                }
            },
            binary: &|op, a, b| {
                if dbl {
                    binary_f64(op, a, b)
                } else {
                    binary_f32(op, a, b)
                }
            },
            select: &|c, a, b| {
                if dbl {
                    select_f64(c, a, b)
                } else {
                    select_f32(c, a, b)
                }
            },
        },
    );
    let stored = match plan.dtype {
        ElementKind::F16 => format!("__float2half({epi})"),
        ElementKind::Bf16 => format!("__float2bfloat16({epi})"),
        _ => epi,
    };

    // Batched cells (`c.batch` set) stride each operand by the batch index
    // `b = blockIdx.z`; the per-batch col/mm/kk math is byte-identical to rank-2,
    // so a rank-2 cell emits empty offsets (unchanged source).
    let (lb, rb, ob) = if c.batch.is_some() {
        ("b * m * k + ", "b * k * n + ", "b * m * n + ")
    } else {
        ("", "", "")
    };

    let mut s = header(plan, &name);
    s.push_str(&format!("    const {ctype}* __restrict__ in0,\n")); // lhs [m,k]
    s.push_str(&format!("    const {ctype}* __restrict__ in1,\n")); // rhs [k,n]
    if plan.n_inputs >= 3 {
        s.push_str(&format!("    const {ctype}* __restrict__ in2,\n")); // bias [n]
    }
    s.push_str(&format!("    {ctype}* __restrict__ out,\n")); // [m,n]
    s.push_str("    long long m,\n    long long n,\n    long long k)\n{\n");
    if c.batch.is_some() {
        s.push_str("    long long b = (long long)blockIdx.z;\n");
    }
    s.push_str("    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long step = (long long)gridDim.x * blockDim.x;\n");
    s.push_str("    for (; col < n; col += step) {\n");
    // Fixed register file at the Tiny ceiling; the mm loops unroll fully with
    // constant indices (predicated on the runtime m) so `accs` stays in
    // registers, never local memory.
    s.push_str(&format!("        {acc} accs[8];\n"));
    s.push_str("        #pragma unroll\n");
    s.push_str(&format!(
        "        for (int mm = 0; mm < 8; ++mm) accs[mm] = {zero};\n"
    ));
    s.push_str("        for (long long kk = 0; kk < k; ++kk) {\n");
    s.push_str(&format!(
        "            {acc} w = {};\n",
        load(format!("in1[{rb}kk * n + col]"))
    ));
    s.push_str("            #pragma unroll\n");
    s.push_str("            for (int mm = 0; mm < 8; ++mm) {\n");
    s.push_str(&format!(
        "                if (mm < m) accs[mm] += {} * w;\n",
        load(format!("in0[{lb}mm * k + kk]"))
    ));
    s.push_str("            }\n        }\n");
    s.push_str("        #pragma unroll\n");
    s.push_str("        for (int mm = 0; mm < 8; ++mm) {\n");
    s.push_str("            if (mm < m) {\n");
    s.push_str(&format!("                {acc} r0 = accs[mm];\n"));
    s.push_str(&format!(
        "                out[{ob}mm * n + col] = {stored};\n"
    ));
    s.push_str("            }\n        }\n    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Split-K schedule **variant** for the contraction cell — the fix for the
/// measured v1 pathology (one thread per column = starved occupancy + a
/// sequential K chain: 62 GB/s vs cuBLAS's 245 on the [8,4096]·[4096,4096]
/// cell; cuBLAS split-Ks its own M=1 path internally). `_splitk_partial` gives
/// each `(column-tile, K-chunk)` block a partial fold into a caller workspace
/// (`n_chunks · m · n` acc elements); `_splitk_combine` folds the chunk
/// partials per `(row, col)` — seeded from chunk 0, not zero, so the
/// degenerate `n_chunks = 1` launch is **bit-identical** to the base kernel —
/// and applies the epilogue + store narrowing. Coalesced throughout; no
/// atomics; deterministic for a fixed `chunk_k` —
/// [`VariantFidelity::ReassociatedDeterministic`] vs the base's sequential K.
fn contraction_splitk_variant(plan: &KernelPlan<'_>) -> Option<Variant> {
    if !matches!(plan.schedule, Schedule::Contraction) {
        return None;
    }
    let Access::Contraction { epilogue, .. } = plan.access else {
        return None;
    };
    // Fused-bias cells (n_inputs > 2) decline the split-K variant in v1: split-K
    // folds the epilogue into a second `_splitk_combine` kernel, and a per-column
    // bias load there is a follow-up. A bias cell gets the base skinny kernel only.
    if plan.n_inputs > 2 {
        return None;
    }
    // build_plan admissibility already ran; these mirror emit_contraction.
    let c = plan.key.contraction?;
    // Batched cells decline the split-K variant in v1 (the split-K workspace +
    // combine kernels have no batch dimension yet) — a batched cell gets the base
    // per-batch skinny kernel only.
    if c.batch.is_some() {
        return None;
    }
    if c.m != baracuda_kernel_vocab::SizeClass::Tiny {
        return None;
    }
    // Same canonical accumulator lattice as `emit_contraction` (the sk3
    // `<acc>` coordinate): double for F64 ONLY — F32Strict stays a binary32
    // chain, matching the key/contract-declared `<acc>=f32`.
    let dbl = matches!(plan.dtype, ElementKind::F64);
    if !matches!(
        plan.dtype,
        ElementKind::F16
            | ElementKind::Bf16
            | ElementKind::F32
            | ElementKind::F32Strict
            | ElementKind::F64
    ) {
        return None;
    }
    let ctype = scalar_ctype(plan.dtype)?;
    let acc = if dbl { "double" } else { "float" };
    let zero = if dbl { "0.0" } else { "0.0f" };
    let load = |expr: String| match plan.dtype {
        ElementKind::F16 => format!("__half2float({expr})"),
        ElementKind::Bf16 => format!("__bfloat162float({expr})"),
        _ => expr,
    };
    let stem = format!(
        "baracuda_gen_{}_{}_contract_{}{}{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        size_tag(c.m),
        size_tag(c.n),
        size_tag(c.k),
    );

    // ---- Kernel 1: per-(column-tile, K-chunk) partial folds → workspace. ----
    let pname = format!("{stem}_splitk_partial");
    let mut p = header(plan, &pname);
    p.push_str(&format!("    const {ctype}* __restrict__ in0,\n"));
    p.push_str(&format!("    const {ctype}* __restrict__ in1,\n"));
    p.push_str(&format!("    {acc}* __restrict__ ws,\n"));
    p.push_str("    long long m,\n    long long n,\n    long long k,\n    long long chunk_k)\n{\n");
    p.push_str("    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    p.push_str("    if (col >= n) return;\n");
    p.push_str("    long long k0 = (long long)blockIdx.y * chunk_k;\n");
    p.push_str("    long long k1 = k0 + chunk_k; if (k1 > k) k1 = k;\n");
    p.push_str(&format!("    {acc} accs[8];\n"));
    p.push_str("    #pragma unroll\n");
    p.push_str(&format!(
        "    for (int mm = 0; mm < 8; ++mm) accs[mm] = {zero};\n"
    ));
    p.push_str("    for (long long kk = k0; kk < k1; ++kk) {\n");
    p.push_str(&format!(
        "        {acc} w = {};\n",
        load("in1[kk * n + col]".to_string())
    ));
    p.push_str("        #pragma unroll\n");
    p.push_str("        for (int mm = 0; mm < 8; ++mm) {\n");
    p.push_str(&format!(
        "            if (mm < m) accs[mm] += {} * w;\n",
        load("in0[mm * k + kk]".to_string())
    ));
    p.push_str("        }\n    }\n");
    p.push_str("    #pragma unroll\n");
    p.push_str("    for (int mm = 0; mm < 8; ++mm) {\n");
    p.push_str("        if (mm < m) ws[((long long)blockIdx.y * m + mm) * n + col] = accs[mm];\n");
    p.push_str("    }\n}\n");

    // ---- Kernel 2: fold the chunk partials; epilogue + store narrowing. ----
    let red = |s: u8| {
        assert_eq!(s, 0, "contraction epilogue reads Reduced(0) only");
        "r0".to_string()
    };
    let epi = lower_expr(
        epilogue,
        &Lowering {
            leaf: &|i| unreachable!("contraction v1 epilogue has no Input leaf: in{i}"),
            reduced: &red,
            coord: &|d| {
                panic!(
                    "cuda backend: Coord({d}) reached the split-K contraction variant — \
                     Coord is Elementwise-only"
                )
            },
            unary: &|op, x| {
                if dbl {
                    unary_f64(op, x)
                } else {
                    unary_f32(op, x)
                }
            },
            binary: &|op, a, b| {
                if dbl {
                    binary_f64(op, a, b)
                } else {
                    binary_f32(op, a, b)
                }
            },
            select: &|c, a, b| {
                if dbl {
                    select_f64(c, a, b)
                } else {
                    select_f32(c, a, b)
                }
            },
        },
    );
    let stored = match plan.dtype {
        ElementKind::F16 => format!("__float2half({epi})"),
        ElementKind::Bf16 => format!("__float2bfloat16({epi})"),
        _ => epi,
    };
    let cname = format!("{stem}_splitk_combine");
    let mut kk = header(plan, &cname);
    kk.push_str(&format!("    const {acc}* __restrict__ ws,\n"));
    kk.push_str(&format!("    {ctype}* __restrict__ out,\n"));
    kk.push_str("    long long m,\n    long long n,\n    long long n_chunks)\n{\n");
    kk.push_str("    long long col = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    kk.push_str("    if (col >= n) return;\n");
    kk.push_str("    #pragma unroll\n");
    kk.push_str("    for (int mm = 0; mm < 8; ++mm) {\n");
    kk.push_str("        if (mm < m) {\n");
    // Seed from chunk 0 (n_chunks >= 1 by the launch contract): keeps the
    // degenerate single-chunk case bit-identical to the base kernel.
    kk.push_str(&format!(
        "            {acc} r0 = ws[(long long)mm * n + col];\n"
    ));
    kk.push_str(
        "            for (long long ch = 1; ch < n_chunks; ++ch) r0 += ws[(ch * m + mm) * n + col];\n",
    );
    kk.push_str(&format!("            out[mm * n + col] = {stored};\n"));
    kk.push_str("        }\n    }\n}\n");

    Some(Variant {
        tag: "splitk",
        kernels: vec![
            GeneratedKernel {
                name: pname.clone(),
                source: p,
            },
            GeneratedKernel {
                name: cname.clone(),
                source: kk,
            },
        ],
        fidelity: VariantFidelity::ReassociatedDeterministic,
        launch_note: format!(
            "two-launch protocol: (1) {pname}<<<dim3(ceil(n/B), n_chunks), B>>>(in0, in1, ws, \
             m, n, k, chunk_k) with chunk_k = ceil(k/n_chunks) and workspace ws of \
             n_chunks*m*n `{acc}` elements; (2) {cname}<<<ceil(n/B), B>>>(ws, out, m, n, \
             n_chunks). m <= 8 (the Tiny launch contract, as the base kernel). \
             Deterministic for a fixed chunk_k; association differs from the base's \
             sequential K fold."
        ),
    })
}

/// One-letter tag for a [`baracuda_kernel_vocab::SizeClass`] in symbol names.
fn size_tag(s: baracuda_kernel_vocab::SizeClass) -> char {
    match s {
        baracuda_kernel_vocab::SizeClass::Tiny => 't',
        baracuda_kernel_vocab::SizeClass::Small => 's',
        baracuda_kernel_vocab::SizeClass::Mid => 'm',
        baracuda_kernel_vocab::SizeClass::Large => 'l',
    }
}

/// Emit a **fused row reduction** ([`Access::RowReduce`]): one block per output
/// row, a warp-shuffle + shared-memory tree reduce per stage, then a full-width
/// elementwise epilogue. RmsNorm (1 stage) and Softmax (2 stages) are instances.
///
/// Each `block_*` helper broadcasts its result to *every* thread of the block, so a
/// stage's reduced scalar lives in a uniform per-thread register `r{i}` —
/// `Reduced(i)` lowers straight to it, with no `__shared__ row_red[]` round-trip
/// and no extra cross-row barrier (the helpers' internal barriers fully serialize
/// their shared reuse across stages and grid-stride rows). The accumulator is
/// `float` (`double` for f64 / f32-strict); f16/bf16 load up-convert and store
/// down-convert, exactly as [`emit_reduction`].
///
/// Correctness invariants (see the design panel): every `__syncthreads` (inside the
/// helpers) is reached by ALL threads — the row guard is the uniform grid-stride
/// `for (row = blockIdx.x; row < n_out; ...)`, NEVER a divergent per-thread early
/// return, so a refactor to per-thread row mapping would deadlock. `__shfl_down_sync`
/// uses the full `0xffffffff` mask; the launch contract caps `blockDim.x <= 1024`
/// and (for warp uniformity) a multiple of 32.
fn emit_row_reduce(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    emit_row_reduce_impl(plan, ctype, false, false)
}

/// Cross-pass materialization **variant** for a RowReduce cell whose epilogue
/// recomputes the LAST stage's per-element expression (the Softmax shape:
/// stage-2 `pre` = `exp(x − r0)`, epilogue = `exp(x − r0) / r1`).
///
/// The base kernel reads the row from global once per stage *plus* once in the
/// epilogue and recomputes the shared expression there; the variant caches the
/// last stage's per-element values in **dynamic shared memory** during the fold
/// and has the epilogue read them back — for Softmax: one fewer full-row global
/// read and half the `exp`s. The cache slots are written and read by the SAME
/// thread under the SAME `j` striding, so the cache is thread-private and needs
/// no extra barrier.
///
/// **Bits:** [`VariantFidelity::BitIdentical`] — the epilogue consumes the very
/// values the fold computed (same expression, same inputs), and an smem
/// store/load round-trip is exact. The tradeoff is *occupancy* (dynamic smem =
/// `k · sizeof(acc)` bytes per block), which is the bench gate's to measure —
/// plus a hard launch cap: `k` must fit the device's per-block smem ceiling,
/// recorded in the launch note.
fn row_reduce_materialize_variant(plan: &KernelPlan<'_>) -> Option<Variant> {
    let Access::RowReduce { stages, epilogue } = plan.access else {
        return None;
    };
    let last = stages.last()?;
    // A leaf `pre` (bare input) has nothing worth caching — the epilogue's
    // re-read costs the same as the smem read.
    if matches!(
        last.pre,
        ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Param(_)
            | ScalarExpr::Reduced(_)
            | ScalarExpr::Coord(_)
    ) {
        return None;
    }
    if !contains_subexpr(epilogue, &last.pre) {
        return None;
    }
    let ctype = scalar_ctype(plan.dtype)?;
    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let (acc, asz) = if dbl { ("double", 8) } else { ("float", 4) };
    let k = emit_row_reduce_impl(plan, ctype, true, false);
    let kname = k.name.clone();
    Some(Variant {
        tag: "smemrow",
        kernels: vec![k],
        fidelity: VariantFidelity::BitIdentical,
        launch_note: format!(
            "same launch shape as the base rowreduce ({kname}<<<n_out, B>>> with B a \
             multiple of 32, <= 1024) PLUS dynamic shared memory = k * {asz} bytes \
             (`{acc}` per element); requires k within the device per-block shared-memory \
             ceiling. Bit-identical to the base kernel; the tradeoff is occupancy."
        ),
    })
}

/// `true` if `t` occurs as a subexpression of `e` (structural equality).
fn contains_subexpr(e: &ScalarExpr, t: &ScalarExpr) -> bool {
    if e == t {
        return true;
    }
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => false,
        ScalarExpr::Unary(_, x) => contains_subexpr(x, t),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => contains_subexpr(a, t) || contains_subexpr(b, t),
        ScalarExpr::Select(c, a, b) => {
            contains_subexpr(c, t) || contains_subexpr(a, t) || contains_subexpr(b, t)
        }
    }
}

/// Clone `e`, replacing every subtree structurally equal to `t` with
/// `Reduced(marker)` — the hook the materialized row-cache read hangs off.
fn substitute_subexpr(e: &ScalarExpr, t: &ScalarExpr, marker: u8) -> ScalarExpr {
    if e == t {
        return ScalarExpr::Reduced(marker);
    }
    let bx = |x: &ScalarExpr| Box::new(substitute_subexpr(x, t, marker));
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => e.clone(),
        ScalarExpr::Unary(op, x) => ScalarExpr::Unary(*op, bx(x)),
        ScalarExpr::Add(a, b) => ScalarExpr::Add(bx(a), bx(b)),
        ScalarExpr::Sub(a, b) => ScalarExpr::Sub(bx(a), bx(b)),
        ScalarExpr::Mul(a, b) => ScalarExpr::Mul(bx(a), bx(b)),
        ScalarExpr::Div(a, b) => ScalarExpr::Div(bx(a), bx(b)),
        ScalarExpr::Binary(op, a, b) => ScalarExpr::Binary(*op, bx(a), bx(b)),
        ScalarExpr::Select(c, a, b) => ScalarExpr::Select(bx(c), bx(a), bx(b)),
    }
}

fn emit_row_reduce_impl(
    plan: &KernelPlan<'_>,
    ctype: &str,
    materialize: bool,
    precision: bool,
) -> GeneratedKernel {
    let Access::RowReduce { stages, epilogue } = plan.access else {
        unreachable!("emit_row_reduce requires Access::RowReduce");
    };
    // The materialize (smem cross-pass cache) and precision (serial double fold)
    // variants are mutually-exclusive STRUCTURES — the former is a block-per-row
    // shape with a per-thread j-strided cache, the latter a single-thread-per-row
    // serial fold with no smem. No caller composes them; the assert makes that a
    // hard invariant so the precision branches below can treat the cache paths as
    // unreachable.
    assert!(
        !(materialize && precision),
        "cuda backend: RowReduce materialize + precision are mutually-exclusive structures"
    );
    // Independent emitter backstop (belt-and-suspenders; the plan gate
    // `validate_row_reduce` validates the same, the 0a lesson: gate every layer).
    // Re-derive each operand's role from the key and assert the OOB-relevant
    // invariants the role-aware `load`/hoist below depend on, with cuda-prefixed
    // messages distinct from the plan gate's — so no future path can route a
    // malformed RowReduce operand into an out-of-bounds index. Reached by BOTH the
    // base kernel (`emit_row_reduce` → here) and the smemrow variant.
    {
        let last = plan.key.rank.saturating_sub(1);
        for i in 0..plan.n_inputs {
            let o = plan.key.operands[i as usize];
            match rr_role(o, last) {
                RrRole::RowStreamed => assert!(
                    o.contig == Contiguity::Contig && !o.flipped,
                    "cuda backend: RowReduce row-streamed input {i} must be contiguous \
                     and not flipped (in{i}[base+j] reads forward-dense; a reversed \
                     view is |stride|-contig but reads mirrored/OOB)"
                ),
                RrRole::ColBroadcast => assert!(
                    !o.flipped && !o.bcast.is_set(last) && (0..last).all(|d| o.bcast.is_set(d)),
                    "cuda backend: RowReduce column input {i} must broadcast every outer axis, \
                     vary along the feature axis, and not flip (in{i}[j])"
                ),
                RrRole::RowScalar => assert!(
                    plan.key.rank >= 2 && !o.flipped && (0..last).all(|d| !o.bcast.is_set(d)),
                    "cuda backend: RowReduce row-scalar input {i} needs rank>=2, no outer-axis \
                     broadcast, and no flip (in{i}[row])"
                ),
            }
        }
        assert!(
            plan.n_inputs == 0 || rr_role(plan.key.operands[0], last) == RrRole::RowStreamed,
            "cuda backend: RowReduce input 0 must be the row-streamed reduced tensor"
        );
    }
    // Precision forces the wider `double` accumulator (fixing the length-growing
    // float error of the sum/mean/variance stages); F64/F32Strict already do.
    let dbl = precision || matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let acc = if dbl { "double" } else { "float" };
    let zero = if dbl { "0.0" } else { "0.0f" };
    // Prod's multiplicative identity — passed through to `emit_block_reducers`
    // for its signature; a RowReduce Prod stage is rejected at the plan gate, so
    // `ops` never contains Prod here and no block_prod is emitted.
    let one = if dbl { "1.0" } else { "1.0f" };
    let n = plan.n_inputs;
    let vsuf = if materialize {
        "_smemrow"
    } else if precision {
        "_prec"
    } else {
        ""
    };
    let name = format!(
        "baracuda_gen_{}_{}_rowreduce{vsuf}",
        plan.op_name,
        dtype_tag(plan.dtype)
    );
    // Helpers are named per (op, dtype[, variant]) so concatenating generated
    // kernels — including a base/variant PAIR — into one translation unit can
    // never collide on a `__device__` symbol.
    let stem = format!(
        "{}_{}{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        if materialize {
            "_sm"
        } else if precision {
            "_prec"
        } else {
            ""
        }
    );

    // The feature (reduced/last) axis index, for the role classifier. `saturating_sub`
    // keeps `load` total even at rank 0 (validate guarantees rank >= 1 before we run).
    let last = plan.key.rank.saturating_sub(1);
    // load(i), up-converting f16/bf16/f32-strict to the accumulate type. The index
    // is role-aware (validate guarantees the roles): a row-streamed input (`x`, or a
    // second streamed operand like softmax-bw's `dy`) loads `in_i[idx]` (idx = base+j);
    // a per-column weight/bias loads `in_i[j]` (same value every row); a per-row scalar
    // (a saved stat) reads the value hoisted ONCE per row into `rs{i}` (already the
    // accumulate type). `idx`/`j` are in scope in every stage fold + the epilogue loop;
    // `rs{i}` is hoisted at the row-loop top, so it too is in scope everywhere.
    // (Validate forbids only column inputs inside a stage `pre`; a row-scalar is
    // constant along the reduced axis and is legal there — layer-norm-bw's x_hat.)
    let load = |i: u8| {
        let role = rr_role(plan.key.operands[i as usize], last);
        if role == RrRole::RowScalar {
            return format!("rs{i}");
        }
        let pos = match role {
            RrRole::RowStreamed => "idx",
            RrRole::ColBroadcast => "j",
            RrRole::RowScalar => unreachable!("row-scalar handled above"),
        };
        match plan.dtype {
            ElementKind::F16 => format!("__half2float(in{i}[{pos}])"),
            ElementKind::Bf16 => format!("__bfloat162float(in{i}[{pos}])"),
            ElementKind::F32Strict => format!("(double)in{i}[{pos}]"),
            _ => format!("in{i}[{pos}]"),
        }
    };
    // Reduced(s) is the broadcast register from stage s. In the materialized
    // variant, the one-past-the-end index is the epilogue's hook for the
    // per-element row cache (see the substitution below).
    let n_stages = stages.len() as u8;
    let red = |s: u8| {
        if materialize && s == n_stages {
            "baracuda_row_smem[j]".to_string()
        } else {
            format!("r{s}")
        }
    };
    let lower = |e: &ScalarExpr| {
        lower_expr(
            e,
            &Lowering {
                leaf: &load,
                reduced: &red,
                coord: &|d| {
                    panic!(
                        "cuda backend: Coord({d}) reached the RowReduce emitter — the \
                         (row, j) space is not the elementwise coordinate space; Coord \
                         is Elementwise-only"
                    )
                },
                unary: &|op, x| {
                    if dbl {
                        unary_f64(op, x)
                    } else {
                        unary_f32(op, x)
                    }
                },
                binary: &|op, a, b| {
                    if dbl {
                        binary_f64(op, a, b)
                    } else {
                        binary_f32(op, a, b)
                    }
                },
                select: &|c, a, b| {
                    if dbl {
                        select_f64(c, a, b)
                    } else {
                        select_f32(c, a, b)
                    }
                },
            },
        )
    };

    // Preamble: comment + dtype include + the block-reduce helpers (only the
    // combines the stages use), then the kernel signature. (We can't reuse
    // `header` here: the helpers must sit between the includes and `extern "C"`.)
    let mut s = format!(
        "// Generated by baracuda-kernelgen — do not edit.\n// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    );
    if let Some(inc) = extra_include(plan.dtype) {
        s.push_str(inc);
    }
    s.push('\n');
    let mut ops = std::collections::HashSet::new();
    for st in stages {
        ops.insert(st.op);
    }
    // The precision serial fold has one thread per row own the whole reduction, so
    // it needs NO cooperative block/warp/smem reducers — skip the helper emission.
    if !precision {
        emit_block_reducers(&mut s, acc, zero, one, &ops, &stem);
    }
    s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
    for i in 0..n {
        s.push_str(&format!("    const {ctype}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {ctype}* __restrict__ out,\n"));
    s.push_str(&format!(
        "    long long n_out,\n    long long k{})\n{{\n",
        param_args(plan.body, param_ctype(plan))
    ));
    // Uniform empty-axis guard (k is a single kernel arg): never skips a barrier
    // divergently, and defends the Max/Min seed against an OOB load.
    s.push_str("    if (k == 0) return;\n");
    if materialize {
        // Per-row element cache: the LAST stage's per-element values, written by
        // each thread into ITS OWN j-strided slots during the fold and read back
        // (same slots, same striding) in the epilogue — thread-private by
        // construction, so no extra barrier is needed. Launch contract: dynamic
        // shared memory = k * sizeof(acc) bytes.
        s.push_str(&format!(
            "    extern __shared__ {acc} baracuda_row_smem[];\n"
        ));
    }
    // Row mapping: the base kernel is one BLOCK per row (grid-stride over rows,
    // the block cooperatively folds each row). The precision variant is one THREAD
    // per row (grid-stride over rows), each thread serially folding its whole row —
    // a fixed, hardware-independent order.
    if precision {
        s.push_str(
            "    for (long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x; row < n_out; row += (long long)gridDim.x * blockDim.x) {\n",
        );
    } else {
        s.push_str(
            "    for (long long row = blockIdx.x; row < n_out; row += (long long)gridDim.x) {\n",
        );
    }
    s.push_str("        long long base = row * k;\n");
    // Per-element fold loop header: block-strided (each block thread takes a lane of
    // the row) for the base/materialize kernels; plain serial for the precision
    // variant (the single owning thread walks the whole row in order). Shared by
    // every stage fold AND the epilogue below.
    let fold_open = if precision {
        "        for (long long j = 0; j < k; ++j) {\n"
    } else {
        "        for (long long j = threadIdx.x; j < k; j += blockDim.x) {\n"
    };
    // Hoist per-row scalar operands (saved stats: μ, rstd, lse) once per row: they
    // are constant along the feature axis, so `in{i}[row]` is loaded here (outside
    // the feature loop), up-converted to the accumulate type, and referenced as
    // `rs{i}` in every stage fold + the epilogue. Emits NOTHING when there is no
    // row-scalar operand (every pre-increment-2 op), so existing emission is
    // byte-identical.
    for i in 0..n {
        if rr_role(plan.key.operands[i as usize], last) == RrRole::RowScalar {
            let conv = match plan.dtype {
                ElementKind::F16 => format!("__half2float(in{i}[row])"),
                ElementKind::Bf16 => format!("__bfloat162float(in{i}[row])"),
                ElementKind::F32Strict => format!("(double)in{i}[row]"),
                _ => format!("in{i}[row]"),
            };
            s.push_str(&format!("        {acc} rs{i} = {conv};\n"));
        }
    }

    for (i, st) in stages.iter().enumerate() {
        // The materialized variant caches the LAST stage's per-element values.
        let cache = materialize && i + 1 == stages.len();
        let pre = lower(&st.pre);
        s.push_str(&format!("        // stage {i}: {:?}\n", st.op));
        match st.op {
            // Prod stages are rejected at the plan gate (`validate_row_reduce`);
            // this is the independent emitter backstop (the 0a lesson: gate every
            // layer). The fused row path has no block_prod cooperative reducer.
            ReduceOp::Prod => panic!(
                "cuda backend: RowReduce Prod stage is unsupported (0e adds Prod to \
                 Access::Reduction only) — the plan gate should have refused this op"
            ),
            ReduceOp::Sum | ReduceOp::Mean => {
                s.push_str(&format!("        {acc} acc{i} = {zero};\n"));
                s.push_str(fold_open);
                s.push_str("            long long idx = base + j;\n");
                if cache {
                    s.push_str(&format!("            {acc} v = {pre};\n"));
                    s.push_str("            baracuda_row_smem[j] = v;\n");
                    s.push_str(&format!("            acc{i} += v;\n"));
                } else {
                    s.push_str(&format!("            acc{i} += {pre};\n"));
                }
                s.push_str("        }\n");
                // Base: block_sum broadcasts the row sum to every thread. Precision:
                // the single owning thread already holds the full serial sum in
                // acc{i}, so no cooperative combine. Mean divides by k (k>0 by the
                // guard above) either way.
                let fin = if precision {
                    if matches!(st.op, ReduceOp::Mean) {
                        format!("acc{i} / ({acc})k")
                    } else {
                        format!("acc{i}")
                    }
                } else if matches!(st.op, ReduceOp::Mean) {
                    format!("block_sum_{stem}(acc{i}) / ({acc})k")
                } else {
                    format!("block_sum_{stem}(acc{i})")
                };
                s.push_str(&format!("        {acc} r{i} = {fin};\n"));
            }
            ReduceOp::Max | ReduceOp::Min => {
                let cmp = if matches!(st.op, ReduceOp::Max) {
                    ">"
                } else {
                    "<"
                };
                let suf = if matches!(st.op, ReduceOp::Max) {
                    "max"
                } else {
                    "min"
                };
                // Carry a `has` flag so idle / short-row lanes inject nothing and no
                // ±inf seed is needed (headerless); NaN sticks via `e != e`. Kept in
                // the precision fold too: it gives the same empty/NaN semantics when
                // the single owning thread walks the row serially.
                s.push_str(&format!("        {acc} acc{i} = {zero}; int has{i} = 0;\n"));
                s.push_str(fold_open);
                s.push_str("            long long idx = base + j;\n");
                s.push_str(&format!("            {acc} e = {pre};\n"));
                if cache {
                    s.push_str("            baracuda_row_smem[j] = e;\n");
                }
                s.push_str(&format!(
                    "            if (!has{i} || e != e || e {cmp} acc{i}) {{ acc{i} = e; has{i} = 1; }}\n"
                ));
                s.push_str("        }\n");
                // Base: block_{suf} folds the per-thread extrema across the block.
                // Precision: the single owning thread already holds the row extremum
                // in acc{i} (the has{i} flag settled it), so read it directly.
                let rfin = if precision {
                    format!("acc{i}")
                } else {
                    format!("block_{suf}_{stem}(acc{i}, has{i})")
                };
                s.push_str(&format!("        {acc} r{i} = {rfin};\n"));
            }
        }
    }

    // Epilogue: full-width output (out[idx], same shape as input); Reduced(i)
    // read from the r{i} registers. Base: x re-read from global and the last
    // stage's per-element expression recomputed. Materialized: every occurrence
    // of that expression reads the row cache instead — the SAME values the fold
    // computed (bit-identical by construction), skipping the third global read
    // of x and the recompute.
    let epi_sub;
    let epi_src: &ScalarExpr = if materialize {
        epi_sub = substitute_subexpr(epilogue, &stages[stages.len() - 1].pre, n_stages);
        &epi_sub
    } else {
        epilogue
    };
    let epi = lower(epi_src);
    let stored = match plan.dtype {
        ElementKind::F16 => format!("__float2half({epi})"),
        ElementKind::Bf16 => format!("__float2bfloat16({epi})"),
        _ => epi,
    };
    s.push_str(fold_open);
    s.push_str("            long long idx = base + j;\n");
    s.push_str(&format!("            out[idx] = {stored};\n"));
    s.push_str("        }\n");
    s.push_str("    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Append the warp/block tree-reduce device helpers for the `ops` actually used,
/// typed by accumulator `acc` (float/double) and uniquely named per `stem`
/// (op+dtype). Each `block_*` broadcasts its result to all block threads.
///
/// Correctness: the full `0xffffffff` shuffle mask (all lanes run the same loop);
/// the cross-warp fold indexes `smem[threadIdx.x]` (NOT `lane` — a multi-warp
/// block must read distinct warp partials, not warp 0's duplicated) and reads only
/// the first `nwarps` slots; Sum pads out-of-range slots with the `0` identity;
/// Max/Min carry a `(value, has)` flag and peel real partials (no ±inf literal,
/// NaN-propagating to match `torch.amax`/`amin`).
fn emit_block_reducers(
    s: &mut String,
    acc: &str,
    zero: &str,
    one: &str,
    ops: &std::collections::HashSet<ReduceOp>,
    stem: &str,
) {
    // Prod (increment 0e): the same cooperative warp-shuffle + shared-mem tree as
    // block_sum, but with `*` and the multiplicative identity `one` for idle
    // lanes. Associative/commutative, so the block-tree order is deterministic.
    if ops.contains(&ReduceOp::Prod) {
        s.push_str(&format!(
            "__device__ __forceinline__ {acc} warp_prod_{stem}({acc} v) {{\n\
             \x20   for (int off = warpSize / 2; off > 0; off >>= 1)\n\
             \x20       v *= __shfl_down_sync(0xffffffffu, v, off);\n\
             \x20   return v;\n\
             }}\n\
             __device__ __forceinline__ {acc} block_prod_{stem}({acc} v) {{\n\
             \x20   __shared__ {acc} smem[32];\n\
             \x20   int lane = threadIdx.x & (warpSize - 1);\n\
             \x20   int wid = threadIdx.x / warpSize;\n\
             \x20   int nwarps = (blockDim.x + warpSize - 1) / warpSize;\n\
             \x20   v = warp_prod_{stem}(v);\n\
             \x20   if (lane == 0) smem[wid] = v;\n\
             \x20   __syncthreads();\n\
             \x20   {acc} r = ((int)threadIdx.x < nwarps) ? smem[threadIdx.x] : {one};\n\
             \x20   if (wid == 0) r = warp_prod_{stem}(r);\n\
             \x20   __shared__ {acc} bcast;\n\
             \x20   if (threadIdx.x == 0) bcast = r;\n\
             \x20   __syncthreads();\n\
             \x20   return bcast;\n\
             }}\n"
        ));
    }
    if ops
        .iter()
        .any(|o| matches!(o, ReduceOp::Sum | ReduceOp::Mean))
    {
        s.push_str(&format!(
            "__device__ __forceinline__ {acc} warp_sum_{stem}({acc} v) {{\n\
             \x20   for (int off = warpSize / 2; off > 0; off >>= 1)\n\
             \x20       v += __shfl_down_sync(0xffffffffu, v, off);\n\
             \x20   return v;\n\
             }}\n\
             __device__ __forceinline__ {acc} block_sum_{stem}({acc} v) {{\n\
             \x20   __shared__ {acc} smem[32];\n\
             \x20   int lane = threadIdx.x & (warpSize - 1);\n\
             \x20   int wid = threadIdx.x / warpSize;\n\
             \x20   int nwarps = (blockDim.x + warpSize - 1) / warpSize;\n\
             \x20   v = warp_sum_{stem}(v);\n\
             \x20   if (lane == 0) smem[wid] = v;\n\
             \x20   __syncthreads();\n\
             \x20   {acc} r = ((int)threadIdx.x < nwarps) ? smem[threadIdx.x] : {zero};\n\
             \x20   if (wid == 0) r = warp_sum_{stem}(r);\n\
             \x20   __shared__ {acc} bcast;\n\
             \x20   if (threadIdx.x == 0) bcast = r;\n\
             \x20   __syncthreads();\n\
             \x20   return bcast;\n\
             }}\n"
        ));
    }
    for (suf, cmp) in [("max", ">"), ("min", "<")] {
        let want = (suf == "max" && ops.contains(&ReduceOp::Max))
            || (suf == "min" && ops.contains(&ReduceOp::Min));
        if !want {
            continue;
        }
        s.push_str(&format!(
            "__device__ __forceinline__ void warp_{suf}_{stem}({acc}& v, int& has) {{\n\
             \x20   for (int off = warpSize / 2; off > 0; off >>= 1) {{\n\
             \x20       {acc} ov = __shfl_down_sync(0xffffffffu, v, off);\n\
             \x20       int oh = __shfl_down_sync(0xffffffffu, has, off);\n\
             \x20       if (oh && (!has || ov != ov || ov {cmp} v)) {{ v = ov; has = 1; }}\n\
             \x20   }}\n\
             }}\n\
             __device__ __forceinline__ {acc} block_{suf}_{stem}({acc} v, int has) {{\n\
             \x20   __shared__ {acc} sv[32]; __shared__ int sh[32];\n\
             \x20   int lane = threadIdx.x & (warpSize - 1);\n\
             \x20   int wid = threadIdx.x / warpSize;\n\
             \x20   int nwarps = (blockDim.x + warpSize - 1) / warpSize;\n\
             \x20   warp_{suf}_{stem}(v, has);\n\
             \x20   if (lane == 0) {{ sv[wid] = v; sh[wid] = has; }}\n\
             \x20   __syncthreads();\n\
             \x20   __shared__ {acc} bcast;\n\
             \x20   if (threadIdx.x == 0) {{\n\
             \x20       {acc} m = sv[0]; int mh = sh[0];\n\
             \x20       for (int i = 1; i < nwarps; ++i)\n\
             \x20           if (sh[i] && (!mh || sv[i] != sv[i] || sv[i] {cmp} m)) {{ m = sv[i]; mh = 1; }}\n\
             \x20       bcast = m;\n\
             \x20   }}\n\
             \x20   __syncthreads();\n\
             \x20   return bcast;\n\
             }}\n"
        ));
    }
    s.push('\n');
}

// ============================================================================
// Increment 6 SCAN — prefix scan (cumsum/cumprod/cummax/cummin), inner axis.
// ============================================================================

/// The monoid identity literal for a scan combine at `dt`, header-light (the
/// `INFINITY` FP extremes are already emitted by [`crate::backend::const_lit`], so
/// they compile under nvrtc/nvcc without a `<math.h>` include; the integer extremes
/// are plain C literals). `Sum → 0`, `Prod → 1`, `Max → the type MINIMUM` (an
/// empty-set max), `Min → the type MAXIMUM`. Used for the exclusive scan's first
/// position (the identity probe) and the block-scan's out-of-range lane padding.
fn scan_identity(sop: ReduceOp, dt: ElementKind) -> String {
    let int_acc = crate::plan::is_int_dtype(dt);
    let dbl = matches!(dt, ElementKind::F64 | ElementKind::F32Strict);
    match sop {
        ReduceOp::Sum => if int_acc {
            "0"
        } else if dbl {
            "0.0"
        } else {
            "0.0f"
        }
        .to_string(),
        ReduceOp::Prod => if int_acc {
            "1"
        } else if dbl {
            "1.0"
        } else {
            "1.0f"
        }
        .to_string(),
        // Max's identity is the type minimum; Min's is the type maximum.
        ReduceOp::Max => type_extreme_lit(dt, true),
        ReduceOp::Min => type_extreme_lit(dt, false),
        ReduceOp::Mean => unreachable!("Scan rejects Mean at validate_scan"),
    }
}

/// The type's extreme literal: `most_negative=true` → the minimum, else the
/// maximum. FP extremes are `∓INFINITY` (header-light per `const_lit`); integer
/// extremes are exact C literals.
fn type_extreme_lit(dt: ElementKind, most_negative: bool) -> String {
    match dt {
        // Header-light ±inf via the always-available bit-cast intrinsics — the
        // headerless-nvrtc discipline forbids the <cmath> `INFINITY` macro (the
        // reduce/row-reduce Max/Min path follows the same rule). This is NOT dead:
        // an EXCLUSIVE Max/Min emits the monoid identity (±inf) as the position-0
        // OUTPUT. Double accumulator for F64/F32Strict, float otherwise.
        ElementKind::F64 | ElementKind::F32Strict => if most_negative {
            "__longlong_as_double(0xfff0000000000000ULL)"
        } else {
            "__longlong_as_double(0x7ff0000000000000ULL)"
        }
        .to_string(),
        ElementKind::F16 | ElementKind::Bf16 | ElementKind::F32 => if most_negative {
            "__int_as_float(0xff800000u)"
        } else {
            "__int_as_float(0x7f800000u)"
        }
        .to_string(),
        ElementKind::I32 => if most_negative {
            "(-2147483647 - 1)"
        } else {
            "2147483647"
        }
        .to_string(),
        ElementKind::I64 => if most_negative {
            "(-9223372036854775807LL - 1)"
        } else {
            "9223372036854775807LL"
        }
        .to_string(),
        ElementKind::S8 => if most_negative {
            "((signed char)-128)"
        } else {
            "((signed char)127)"
        }
        .to_string(),
        ElementKind::U8 => if most_negative {
            "((unsigned char)0)"
        } else {
            "((unsigned char)255)"
        }
        .to_string(),
        other => unreachable!("scan type_extreme_lit on unsupported dtype {other:?}"),
    }
}

/// The serial-fold scan BASE (`block = false`) — [`crate::backend::VariantFidelity::BitIdentical`].
fn emit_scan(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    emit_scan_impl(plan, ctype, false)
}

/// Append the inline warp-scan device helper for the block-scan variant (analog of
/// [`emit_block_reducers`]): a Kogge-Stone warp-inclusive scan via `__shfl_up_sync`
/// (`log2(warpSize)` rounds, full `0xffffffff` mask — every lane runs the same
/// loop). Serves `Sum`/`Prod`/`Max`/`Min` over FP (float/double acc) and integer
/// (I32/I64 native acc); named per `stem` so a base+variant pair in one translation
/// unit never collides on the `__device__` symbol.
///
/// The up-sweep step combines `t` (the shuffled value from the EARLIER lane =
/// LEFT = `a`) into `v` (own = LATER = RIGHT = `b`). `Sum`/`Prod` are commutative,
/// so the shipped `v {assign} t` is UNCHANGED (byte-identical). `Max`/`Min` are
/// NON-commutative: emit `v = comb(a = t, b = v)` — take RIGHT (`v`) iff it is NaN
/// or STRICTLY beats LEFT (`t`); writing `comb(v, t)` here would flip the
/// signed-zero tie sign and let an earlier NaN win a later one. `is_int` drops the
/// dead `v != v` NaN test for I32/I64 (no integer NaN; ties are bit-equal).
fn emit_block_scanners(s: &mut String, acc: &str, sop: ReduceOp, stem: &str, is_int: bool) {
    let tag = match sop {
        ReduceOp::Sum => "sum",
        ReduceOp::Prod => "prod",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        ReduceOp::Mean => unreachable!("emit_block_scanners: Scan rejects Mean"),
    };
    let step = match sop {
        ReduceOp::Sum => "v += t;".to_string(),
        ReduceOp::Prod => "v *= t;".to_string(),
        ReduceOp::Max if is_int => "v = (v > t) ? v : t;".to_string(),
        ReduceOp::Min if is_int => "v = (v < t) ? v : t;".to_string(),
        ReduceOp::Max => "v = (v != v || v > t) ? v : t;".to_string(),
        ReduceOp::Min => "v = (v != v || v < t) ? v : t;".to_string(),
        ReduceOp::Mean => unreachable!("emit_block_scanners: Scan rejects Mean"),
    };
    s.push_str(&format!(
        "__device__ __forceinline__ {acc} warpscan_{tag}_{stem}({acc} v) {{\n\
         \x20   int lane = threadIdx.x & (warpSize - 1);\n\
         \x20   for (int off = 1; off < warpSize; off <<= 1) {{\n\
         \x20       {acc} t = __shfl_up_sync(0xffffffffu, v, off);\n\
         \x20       if (lane >= off) {step}\n\
         \x20   }}\n\
         \x20   return v;\n\
         }}\n\n"
    ));
}

/// Emit a scan kernel — the serial-fold BASE (`block = false`) or the cooperative
/// block-scan VARIANT (`block = true`). See [`crate::ir::Access::Scan`] and §3/§4
/// of the increment-6 brief.
///
/// The BASE is a plain per-row serial fold (thread 0 walks the axis in order — the
/// honest deterministic bit-reference). The VARIANT re-emits a Kogge-Stone warp
/// scan + cross-warp exclusive-offset carry INLINE (headerless — `smem_scan` does
/// not exist in this crate), chunking the row so a `k > blockDim` row threads its
/// running carry across tiles; it is FP `Sum`/`Prod` only.
fn emit_scan_impl(plan: &KernelPlan<'_>, ctype: &str, block: bool) -> GeneratedKernel {
    let (pre, post) = match plan.access {
        Access::Scan { pre, post, .. } => (pre, post),
        _ => unreachable!("emit_scan requires Access::Scan"),
    };
    let (sop, axis, reverse, exclusive) = match plan.schedule {
        Schedule::Scan {
            op,
            axis,
            reverse,
            exclusive,
            ..
        } => (op, axis, reverse, exclusive),
        _ => unreachable!("emit_scan on a non-Scan schedule"),
    };

    // ---- Independent emitter backstops (belt-and-suspenders; validate_scan
    // validates the same, the 0a lesson: gate every layer). cuda-prefixed messages
    // distinct from the plan gate's. ----
    let rank = plan.key.rank;
    assert!(
        rank >= 1,
        "cuda backend: Scan needs a scanned axis (rank >= 1)"
    );
    let last = rank - 1;
    assert!(
        axis == last,
        "cuda backend: Scan v1 emits the innermost (contiguous) axis only (axis {axis} != rank-1 {last})"
    );
    {
        let o0 = plan.key.operands[0];
        assert!(
            rr_role(o0, last) == RrRole::RowStreamed
                && o0.contig == Contiguity::Contig
                && !o0.flipped,
            "cuda backend: Scan input 0 must be the forward-dense contiguous scanned tensor (idx = base+j)"
        );
    }
    if block {
        // The cooperative kernel re-emits the warp scan + cross-warp carry inline.
        // Scope (BLOCKSCAN-VARIANTS): Sum/Prod/Max/Min over FP
        // {F16,Bf16,F32,F32Strict,F64} + integer {I32,I64}. S8/U8 ride the serial
        // base (`__shfl_up_sync` has no 8-bit overload; promoting to an int acc
        // would break the base's native 8-bit modular wrap). These are emitter
        // backstops — the variant gate is the primary filter, but S8/U8 must still
        // trip here. The warp_buf[32] cross-warp buffer sizes for blockDim <= 1024,
        // a multiple of 32 — a LAUNCH contract (no generation-time blockDim to
        // assert), carried in the launch note and pinned by the on-device runs.
        assert!(
            matches!(
                sop,
                ReduceOp::Sum | ReduceOp::Prod | ReduceOp::Max | ReduceOp::Min
            ),
            "cuda backend: the scan block-scan variant serves Sum/Prod/Max/Min only (Mean rejected upstream)"
        );
        assert!(
            matches!(
                plan.dtype,
                ElementKind::F16
                    | ElementKind::Bf16
                    | ElementKind::F32
                    | ElementKind::F32Strict
                    | ElementKind::F64
            ) || matches!(plan.dtype, ElementKind::I32 | ElementKind::I64),
            "cuda backend: the scan block-scan variant serves FP {{F16,Bf16,F32,F32Strict,F64}} + integer {{I32,I64}} only (S8/U8 decline to the serial base); got {:?}",
            plan.dtype
        );
    }

    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let int_acc = crate::plan::is_int_dtype(plan.dtype);
    // Base: float/double for FP, the native ctype (wrapping) for integers. Variant:
    // FP-only (asserted), so float/double.
    let acc = if dbl {
        "double"
    } else if int_acc {
        ctype
    } else {
        "float"
    };
    let ident = scan_identity(sop, plan.dtype);

    let combine_tag = match sop {
        ReduceOp::Sum => "sum",
        ReduceOp::Prod => "prod",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        ReduceOp::Mean => unreachable!("Scan rejects Mean at validate_scan"),
    };
    let dtag = dtype_tag(plan.dtype);
    let rev_suf = if reverse { "_rev" } else { "" };
    let exc_suf = if exclusive { "_excl" } else { "" };
    let blk_suf = if block { "_blockscan" } else { "" };
    let name = format!(
        "baracuda_gen_{}_{dtag}_scan_{combine_tag}{rev_suf}{exc_suf}{blk_suf}",
        plan.op_name
    );
    // The device-helper stem carries the flag suffixes too, so several block-scan
    // variants of ONE op (e.g. inclusive + exclusive cumsum) concatenated into one
    // translation unit (the on-device validator) never collide on the `__device__`
    // `warpscan_*` symbol.
    let stem = format!("{}_{dtag}{rev_suf}{exc_suf}", plan.op_name);

    // Role-aware load (mirrors emit_row_reduce_impl): the scanned input reads
    // `in_i[idx]` (idx = base+j); a per-column operand `in_i[j]`; a per-row scalar
    // the hoisted `rs{i}`. f16/bf16/f32-strict up-convert to the acc width.
    let load = |i: u8| {
        let role = rr_role(plan.key.operands[i as usize], last);
        if role == RrRole::RowScalar {
            return format!("rs{i}");
        }
        let pos = match role {
            RrRole::RowStreamed => "idx",
            RrRole::ColBroadcast => "j",
            RrRole::RowScalar => unreachable!("row-scalar handled above"),
        };
        match plan.dtype {
            ElementKind::F16 => format!("__half2float(in{i}[{pos}])"),
            ElementKind::Bf16 => format!("__bfloat162float(in{i}[{pos}])"),
            ElementKind::F32Strict => format!("(double)in{i}[{pos}]"),
            _ => format!("in{i}[{pos}]"),
        }
    };
    // `pre` (the per-element pre-map) lowers over the loaded input; it has NO
    // running prefix, so a Reduced leaf panics (validate_scan rejects it).
    let pre_str = lower_expr(
        pre,
        &Lowering {
            leaf: &load,
            reduced: &|s| {
                panic!(
                    "cuda backend: Scan pre-map read Reduced({s}) — no running prefix in the pre-map"
                )
            },
            coord: &|d| panic!("cuda backend: Scan Coord({d}) is Elementwise-only"),
            unary: &|op, x| {
                if dbl {
                    unary_f64(op, x)
                } else {
                    unary_f32(op, x)
                }
            },
            binary: &|op, a, b| {
                if dbl {
                    binary_f64(op, a, b)
                } else {
                    binary_f32(op, a, b)
                }
            },
            select: &|c, a, b| {
                if dbl {
                    select_f64(c, a, b)
                } else {
                    select_f32(c, a, b)
                }
            },
        },
    );
    // `post` (the per-element epilogue) lowers over the running-prefix register,
    // bound to the `prefix` variable (Reduced(0)); the identity post yields
    // `"prefix"`, so the store is byte-simple.
    let post_str = lower_expr(
        post,
        &Lowering {
            leaf: &load,
            reduced: &|s| {
                assert_eq!(
                    s, 0,
                    "Scan post references Reduced({s}); only 0 (the running prefix) exists"
                );
                "prefix".to_string()
            },
            coord: &|d| panic!("cuda backend: Scan Coord({d}) is Elementwise-only"),
            unary: &|op, x| {
                if dbl {
                    unary_f64(op, x)
                } else {
                    unary_f32(op, x)
                }
            },
            binary: &|op, a, b| {
                if dbl {
                    binary_f64(op, a, b)
                } else {
                    binary_f32(op, a, b)
                }
            },
            select: &|c, a, b| {
                if dbl {
                    select_f64(c, a, b)
                } else {
                    select_f32(c, a, b)
                }
            },
        },
    );
    let store = |v: &str| -> String {
        match plan.dtype {
            ElementKind::F16 => format!("__float2half({v})"),
            ElementKind::Bf16 => format!("__float2bfloat16({v})"),
            _ => v.to_string(),
        }
    };
    let stored = store(&post_str);

    // ---- Preamble + signature. ----
    let mut s = format!(
        "// Generated by baracuda-kernelgen — do not edit.\n// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    );
    if let Some(inc) = extra_include(plan.dtype) {
        s.push_str(inc);
    }
    s.push('\n');
    if block {
        emit_block_scanners(&mut s, acc, sop, &stem, int_acc);
    }
    s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
    for i in 0..plan.n_inputs {
        s.push_str(&format!("    const {ctype}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {ctype}* __restrict__ out,\n"));
    s.push_str(&format!(
        "    long long n_out,\n    long long k{})\n{{\n",
        param_args_multi(&[pre, post], param_ctype(plan))
    ));
    s.push_str("    if (k == 0) return;\n");

    // Per-row grid-stride loop (uniform — never a divergent early return).
    s.push_str(
        "    for (long long row = blockIdx.x; row < n_out; row += (long long)gridDim.x) {\n",
    );
    s.push_str("        long long base = row * k;\n");
    // Hoist per-row scalar operands once per row (emits nothing for single-input).
    for i in 0..plan.n_inputs {
        if rr_role(plan.key.operands[i as usize], last) == RrRole::RowScalar {
            let conv = match plan.dtype {
                ElementKind::F16 => format!("__half2float(in{i}[row])"),
                ElementKind::Bf16 => format!("__bfloat162float(in{i}[row])"),
                ElementKind::F32Strict => format!("(double)in{i}[row]"),
                _ => format!("in{i}[row]"),
            };
            s.push_str(&format!("        {acc} rs{i} = {conv};\n"));
        }
    }

    if !block {
        // -------------------- SERIAL FOLD BASE (thread 0 owns the row) --------------------
        s.push_str("        if (threadIdx.x == 0) {\n");
        // Forward: j = 0..k-1 ascending; reverse: j = k-1..0 descending.
        let for_hdr = if reverse {
            "            for (long long j = k - 1; j >= 0; --j) {\n"
        } else {
            "            for (long long j = 0; j < k; ++j) {\n"
        };
        match sop {
            ReduceOp::Sum | ReduceOp::Prod => {
                let opc = if matches!(sop, ReduceOp::Sum) {
                    "+"
                } else {
                    "*"
                };
                s.push_str(&format!("            {acc} acc = {ident};\n"));
                s.push_str(for_hdr);
                s.push_str("                long long idx = base + j;\n");
                s.push_str(&format!("                {acc} v = {pre_str};\n"));
                if exclusive {
                    // exclusive: write the PRE-combine acc (identity at first pos).
                    s.push_str(&format!("                {acc} prefix = acc;\n"));
                    s.push_str(&format!("                out[idx] = {stored};\n"));
                    s.push_str(&format!("                acc = acc {opc} v;\n"));
                } else {
                    s.push_str(&format!("                acc = acc {opc} v;\n"));
                    s.push_str(&format!("                {acc} prefix = acc;\n"));
                    s.push_str(&format!("                out[idx] = {stored};\n"));
                }
                s.push_str("            }\n");
            }
            ReduceOp::Max | ReduceOp::Min => {
                let cmp = if matches!(sop, ReduceOp::Max) {
                    ">"
                } else {
                    "<"
                };
                // Seed acc with a dummy (guarded by `have`); the exclusive first
                // position emits the monoid identity. NaN propagates via `v != v`.
                s.push_str(&format!("            {acc} acc = {ident}; int have = 0;\n"));
                s.push_str(for_hdr);
                s.push_str("                long long idx = base + j;\n");
                s.push_str(&format!("                {acc} v = {pre_str};\n"));
                if exclusive {
                    s.push_str(&format!(
                        "                {acc} prefix = have ? acc : ({ident});\n"
                    ));
                    s.push_str(&format!("                out[idx] = {stored};\n"));
                    s.push_str(&format!(
                        "                if (!have || v != v || v {cmp} acc) {{ acc = v; have = 1; }}\n"
                    ));
                } else {
                    s.push_str(&format!(
                        "                if (!have || v != v || v {cmp} acc) {{ acc = v; have = 1; }}\n"
                    ));
                    s.push_str(&format!("                {acc} prefix = acc;\n"));
                    s.push_str(&format!("                out[idx] = {stored};\n"));
                }
                s.push_str("            }\n");
            }
            ReduceOp::Mean => unreachable!("Scan rejects Mean"),
        }
        s.push_str("        }\n"); // if threadIdx.x == 0
    } else {
        // -------------------- BLOCK-SCAN VARIANT (chunked, cross-warp carry) --------------------
        // Kogge-Stone warp scan + cross-warp exclusive offset, chunked so a
        // k > blockDim row threads its running carry across tiles. Serves
        // Sum/Prod/Max/Min over FP + integer {I32,I64} (S8/U8 declined upstream).
        //
        // The cross-warp / cross-chunk carry combiner is LEFT-first everywhere
        // (a = earlier prefix, b = later): every call site passes (earlier, later).
        // Sum/Prod are UNCHANGED (byte-identical `(a + b)` / `(a * b)`). Max/Min
        // take b iff b is NaN or STRICTLY beats a — the same rule as the warp step,
        // fully parenthesized so a nested comb (b = a ternary) stays atomic; int
        // drops the dead `v != v` NaN test.
        let comb = |a: &str, b: &str| -> String {
            match sop {
                ReduceOp::Sum => format!("({a} + {b})"),
                ReduceOp::Prod => format!("({a} * {b})"),
                ReduceOp::Max if int_acc => format!("(({b} > {a}) ? {b} : {a})"),
                ReduceOp::Min if int_acc => format!("(({b} < {a}) ? {b} : {a})"),
                ReduceOp::Max => format!("(({b} != {b} || {b} > {a}) ? {b} : {a})"),
                ReduceOp::Min => format!("(({b} != {b} || {b} < {a}) ? {b} : {a})"),
                ReduceOp::Mean => unreachable!("Scan rejects Mean"),
            }
        };
        let tag = match sop {
            ReduceOp::Sum => "sum",
            ReduceOp::Prod => "prod",
            ReduceOp::Max => "max",
            ReduceOp::Min => "min",
            ReduceOp::Mean => unreachable!("Scan rejects Mean"),
        };
        s.push_str(&format!("        __shared__ {acc} warp_buf[32];\n"));
        s.push_str(&format!("        __shared__ {acc} warp_off[32];\n"));
        s.push_str(&format!("        __shared__ {acc} chunk_tot;\n"));
        s.push_str(&format!("        {acc} carry = {ident};\n"));
        s.push_str("        int lane = threadIdx.x & (warpSize - 1);\n");
        s.push_str("        int wid = threadIdx.x / warpSize;\n");
        s.push_str("        int nwarps = (blockDim.x + warpSize - 1) / warpSize;\n");
        s.push_str("        for (long long c0 = 0; c0 < k; c0 += (long long)blockDim.x) {\n");
        s.push_str("            long long p = c0 + threadIdx.x;\n");
        // Reverse remaps the axis position; the p-space scan is always ascending.
        if reverse {
            s.push_str("            long long j = k - 1 - p;\n");
        } else {
            s.push_str("            long long j = p;\n");
        }
        s.push_str("            long long idx = base + j;\n");
        // Out-of-range lanes contribute the identity (the ternary does NOT read
        // in_i[idx] when p >= k — no OOB load).
        s.push_str(&format!(
            "            {acc} v = (p < k) ? ({pre_str}) : ({ident});\n"
        ));
        s.push_str(&format!(
            "            {acc} winc = warpscan_{tag}_{stem}(v);\n"
        ));
        // `wexc` (the warp-exclusive value) is only consumed on the exclusive path
        // (`chunk_excl`); emitting it for an inclusive scan is a dead warp shuffle.
        if exclusive {
            s.push_str(&format!(
                "            {acc} wexc = __shfl_up_sync(0xffffffffu, winc, 1);\n"
            ));
            s.push_str(&format!("            if (lane == 0) wexc = {ident};\n"));
        }
        s.push_str(&format!(
            "            {acc} wtot = __shfl_sync(0xffffffffu, winc, warpSize - 1);\n"
        ));
        s.push_str("            if (lane == 0) warp_buf[wid] = wtot;\n");
        s.push_str("            __syncthreads();\n");
        s.push_str("            if (threadIdx.x == 0) {\n");
        s.push_str(&format!("                {acc} run = {ident};\n"));
        s.push_str("                for (int w = 0; w < nwarps; ++w) {\n");
        s.push_str("                    warp_off[w] = run;\n");
        s.push_str(&format!(
            "                    run = {};\n",
            comb("run", "warp_buf[w]")
        ));
        s.push_str("                }\n");
        s.push_str("                chunk_tot = run;\n");
        s.push_str("            }\n");
        s.push_str("            __syncthreads();\n");
        let chunk_incl = comb("warp_off[wid]", "winc");
        let chunk_excl = comb("warp_off[wid]", "wexc");
        if exclusive {
            s.push_str(&format!(
                "            {acc} prefix = {};\n",
                comb("carry", &chunk_excl)
            ));
        } else {
            s.push_str(&format!(
                "            {acc} prefix = {};\n",
                comb("carry", &chunk_incl)
            ));
        }
        s.push_str(&format!("            if (p < k) out[idx] = {stored};\n"));
        s.push_str(&format!(
            "            carry = {};\n",
            comb("carry", "chunk_tot")
        ));
        s.push_str("            __syncthreads();\n");
        s.push_str("        }\n");
    }

    s.push_str("    }\n}\n");
    GeneratedKernel { name, source: s }
}

/// Block-scan schedule VARIANT for the scan cell — a Kogge-Stone warp scan +
/// cross-warp exclusive-offset carry (re-emitted inline, headerless), one block per
/// row, chunked so a `k > blockDim` row threads its running carry across tiles. A
/// [`Variant`] filter (model: [`reduction_splitk_variant`]) — `return None` for
/// every cell it cannot serve honestly.
///
/// **Scope (BLOCKSCAN-VARIANTS):** `Sum`/`Prod`/`Max`/`Min` over FP
/// {`F16`,`Bf16`,`F32`,`F32Strict`,`F64`} and integer {`I32`,`I64`}, forward AND
/// reverse (`j = k-1-p` remaps the reverse j-scan to a forward p-space scan). `S8`
/// and `U8` DECLINE to the serial base — `__shfl_up_sync` has no 8-bit overload and
/// promoting to an int acc would break the base's native 8-bit modular wrap (a
/// different domain, not bit-identical). This is an explicit allowlist, NOT
/// [`crate::plan::is_int_dtype`] (which admits S8/U8).
///
/// **Bits (per op):** FP `Sum`/`Prod` reassociate (a two-level warp/cross-warp tree
/// vs the base's sequential fold), so [`VariantFidelity::ReassociatedDeterministic`]
/// — no bit-identical degenerate config (even a single blockDim-wide chunk
/// reassociates), within-ULP of the base. FP `Max`/`Min` (the combiner selects the
/// same element bit-for-bit, incl. signed-zero sign + NaN payload) and ALL integer
/// scans (modular `+`/`*` associative; `max`/`min` select, ties bit-equal) are
/// [`VariantFidelity::BitIdentical`] — device memcmp==0 vs the serial base. The FP
/// bit-identity requires an IEEE-strict build (no `-ffast-math`; matched FTZ) so the
/// `v != v` NaN test and signed-zero ties hold — the same requirement Sum/Prod
/// already relies on.
///
/// **Keying:** identity on the wire is `(structure_key token, entry_point)` — the
/// `_blockscan` entry_point disambiguates it from the base (never token alone).
fn scan_blockscan_variant(plan: &KernelPlan<'_>) -> Option<Variant> {
    let (sop, _axis, _reverse, _exclusive) = match plan.schedule {
        Schedule::Scan {
            op,
            axis,
            reverse,
            exclusive,
            ..
        } => (op, axis, reverse, exclusive),
        _ => return None,
    };
    // Sum/Prod/Max/Min only (Mean is rejected upstream at validate_scan).
    if !matches!(
        sop,
        ReduceOp::Sum | ReduceOp::Prod | ReduceOp::Max | ReduceOp::Min
    ) {
        return None;
    }
    // Explicit dtype allowlist: FP {F16,Bf16,F32,F32Strict,F64} + integer {I32,I64}.
    // S8/U8 (and any other dtype) DECLINE to the serial base — NOT `is_int_dtype`,
    // which would admit S8/U8 (whose native 8-bit wrap the promoted int acc breaks).
    let is_fp = matches!(
        plan.dtype,
        ElementKind::F16
            | ElementKind::Bf16
            | ElementKind::F32
            | ElementKind::F32Strict
            | ElementKind::F64
    );
    let is_int = matches!(plan.dtype, ElementKind::I32 | ElementKind::I64);
    if !(is_fp || is_int) {
        return None;
    }
    // The fixed signature has no `p{i}` slots — a Param pre/post would emit an
    // undefined identifier (mirror the split-K param decline). The base serves
    // param'd scans.
    let param_bodies: Vec<&ScalarExpr> = match plan.access {
        Access::Scan { pre, post, .. } => vec![pre, post],
        _ => return None,
    };
    if param_bodies.iter().any(|e| !params_used(e).is_empty()) {
        return None;
    }
    // Decline a hetero-out / non-dense / flipped cell (the base serves them). The
    // scanned input + output must be forward-dense contiguous (idx = base+j / the
    // out[idx] store), and single-streamed-input (the cooperative kernel v1 shape).
    if plan.out_dtype != plan.dtype || plan.n_inputs != 1 {
        return None;
    }
    let last = plan.key.rank.saturating_sub(1);
    let in0 = plan.key.operands[0];
    let out = plan.key.operands[plan.key.n_operands.saturating_sub(1) as usize];
    if rr_role(in0, last) != RrRole::RowStreamed
        || in0.contig != Contiguity::Contig
        || in0.flipped
        || !out.bcast.is_empty()
        || out.contig != Contiguity::Contig
        || out.flipped
    {
        return None;
    }
    let ctype = scalar_ctype(plan.dtype)?;
    let k = emit_scan_impl(plan, ctype, true);
    // Only FP Sum/Prod reassociate (the warp tree rounds differently than the base's
    // sequential fold). FP Max/Min, integer Sum/Prod, integer Max/Min are all
    // BitIdentical to the base (max/min select the same element bit-for-bit; modular
    // +/* is exactly associative — device memcmp==0, §7 probes).
    let (fidelity, bits_note) = if matches!(sop, ReduceOp::Sum | ReduceOp::Prod) && is_fp {
        (
            VariantFidelity::ReassociatedDeterministic,
            "FP Sum/Prod reassociate (warp tree rounds differently than the \
             sequential base) — within-ULP, no bit-identical degenerate config",
        )
    } else {
        (
            VariantFidelity::BitIdentical,
            "bit-identical to the serial base (FP Max/Min select the same element \
             incl. signed-zero/NaN; integer +/*/max/min exactly associative), \
             device memcmp==0",
        )
    };
    Some(Variant {
        tag: "blockscan",
        kernels: vec![k],
        fidelity,
        launch_note: format!(
            "block-scan (Kogge-Stone warp scan via __shfl_up_sync + cross-warp \
             exclusive-offset carry, re-emitted inline): one block per row, \
             <<<min(n_out, maxblocks), B>>> with B a multiple of 32 and <= 1024 \
             (static __shared__ warp_buf/warp_off[32]); a k > B row threads its \
             running carry across ceil(k/B) chunks. Scope: Sum/Prod/Max/Min over FP \
             {{F16,Bf16,F32,F32Strict,F64}} + integer {{I32,I64}}, forward + reverse; \
             S8/U8 ride the serial base. Fidelity (this cell): {bits_note}. Requires \
             an IEEE-strict build (no -ffast-math/--use_fast_math; matched FTZ/\
             denormal) so the FP Max/Min `v != v` NaN test and signed-zero ties hold \
             — the same requirement the Sum/Prod block-scan already relies on. \
             Determinism: {det}.",
            det = fidelity.determinism_str()
        ),
    })
}

// ============================================================================
// Increment 7 WINDOW — sliding-window reduction (max_pool/avg_pool/sum/min pool).
// ============================================================================

/// The serial-fold pooling emitter (`Schedule::Window`) —
/// [`crate::backend::VariantFidelity::BitIdentical`]. One thread per OUTPUT
/// element (grid-stride over `n_out * k_out`): each thread computes its
/// `(row, o)`, walks the local window of `size` taps at input position
/// `p = o*stride - pad_lo + kk*dilation`, reduces the in-bounds taps with `op`
/// (NaN-propagating Max/Min via `v != v`; padding contributes the additive
/// identity for Sum and is skipped for Max/Min), and stores the downsampled
/// output. Independent per-output folds ⇒ naturally parallel and bit-reproducible.
///
/// The window geometry is baked as compile-time literals; `n_out` (outer product),
/// `k_in` (input inner extent), `k_out` (downsampled output inner extent) are
/// runtime launch args — the `k_in → k_out` relationship is the caller's
/// window-arithmetic precondition (the structure key carries no extents; see
/// `plan::validate_window`).
fn emit_window(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let (pre, post) = match plan.access {
        Access::Window { pre, post, .. } => (pre, post),
        _ => unreachable!("emit_window requires Access::Window"),
    };
    let (wop, axis, size, stride, dilation, pad_lo, pad_hi, count_include_pad) = match plan.schedule
    {
        Schedule::Window {
            op,
            axis,
            size,
            stride,
            dilation,
            pad_lo,
            pad_hi,
            count_include_pad,
        } => (
            op,
            axis,
            size,
            stride,
            dilation,
            pad_lo,
            pad_hi,
            count_include_pad,
        ),
        _ => unreachable!("emit_window on a non-Window schedule"),
    };

    // ---- Independent emitter backstops (belt-and-suspenders; validate_window
    // validates the same — the 0a lesson: gate every layer). cuda-prefixed
    // messages distinct from the plan gate's. ----
    let rank = plan.key.rank;
    assert!(
        rank >= 1,
        "cuda backend: Window needs a pooled axis (rank >= 1)"
    );
    let last = rank - 1;
    assert!(
        axis == last,
        "cuda backend: Window v1 emits the innermost (contiguous) axis only (axis {axis} != rank-1 {last})"
    );
    assert!(
        size >= 1 && stride >= 1 && dilation >= 1,
        "cuda backend: Window size/stride/dilation must be >= 1"
    );
    assert!(
        !matches!(wop, ReduceOp::Prod),
        "cuda backend: Window Prod is not a pool (validate_window rejects it)"
    );
    {
        let o0 = plan.key.operands[0];
        assert!(
            rr_role(o0, last) == RrRole::RowStreamed
                && o0.contig == Contiguity::Contig
                && !o0.flipped,
            "cuda backend: Window input 0 must be the forward-dense contiguous pooled tensor (idx = base+p)"
        );
    }
    let is_mean = matches!(wop, ReduceOp::Mean);
    if is_mean {
        assert!(
            matches!(
                plan.dtype,
                ElementKind::F16
                    | ElementKind::Bf16
                    | ElementKind::F32
                    | ElementKind::F32Strict
                    | ElementKind::F64
            ),
            "cuda backend: Window Mean (avg_pool) is float-only; got {:?}",
            plan.dtype
        );
    }

    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let int_acc = crate::plan::is_int_dtype(plan.dtype);
    // Accumulator: double for f64/f32-strict, native ctype for integers (wrapping
    // sum-pool / exact max-pool), float otherwise (incl. f16/bf16 up-convert).
    let acc = if dbl {
        "double"
    } else if int_acc {
        ctype
    } else {
        "float"
    };

    let combine_tag = match wop {
        ReduceOp::Sum => "sum",
        ReduceOp::Mean => "mean",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        ReduceOp::Prod => unreachable!("Window rejects Prod at validate_window"),
    };
    let dtag = dtype_tag(plan.dtype);
    // count_include_pad only affects Mean; suffix it there so the two avg_pool
    // divisor policies never collide on the entry-point symbol.
    let cip_suf = if is_mean && count_include_pad {
        "_cip"
    } else {
        ""
    };
    let name = format!(
        "baracuda_gen_{}_{dtag}_window_{combine_tag}{cip_suf}",
        plan.op_name
    );

    // Role-aware load (mirrors emit_scan_impl): the pooled input reads `in_i[idx]`
    // (idx = base+p); a per-column operand `in_i[p]`; a per-row scalar the hoisted
    // `rs{i}`. f16/bf16/f32-strict up-convert to the acc width.
    let load = |i: u8| {
        let role = rr_role(plan.key.operands[i as usize], last);
        if role == RrRole::RowScalar {
            return format!("rs{i}");
        }
        let pos = match role {
            RrRole::RowStreamed => "idx",
            RrRole::ColBroadcast => "p",
            RrRole::RowScalar => unreachable!("row-scalar handled above"),
        };
        match plan.dtype {
            ElementKind::F16 => format!("__half2float(in{i}[{pos}])"),
            ElementKind::Bf16 => format!("__bfloat162float(in{i}[{pos}])"),
            ElementKind::F32Strict => format!("(double)in{i}[{pos}]"),
            _ => format!("in{i}[{pos}]"),
        }
    };
    // `pre` (per-tap pre-map) lowers over the loaded tap; NO window result exists
    // yet, so a Reduced leaf panics (validate_window rejects it).
    let pre_str = lower_expr(
        pre,
        &Lowering {
            leaf: &load,
            reduced: &|s| {
                panic!(
                    "cuda backend: Window pre-map read Reduced({s}) — no window result in the pre-map"
                )
            },
            coord: &|d| panic!("cuda backend: Window Coord({d}) is Elementwise-only"),
            unary: &|op, x| {
                if dbl {
                    unary_f64(op, x)
                } else {
                    unary_f32(op, x)
                }
            },
            binary: &|op, a, b| {
                if dbl {
                    binary_f64(op, a, b)
                } else {
                    binary_f32(op, a, b)
                }
            },
            select: &|c, a, b| {
                if dbl {
                    select_f64(c, a, b)
                } else {
                    select_f32(c, a, b)
                }
            },
        },
    );
    // `post` (per-output epilogue) lowers over the finalized window result, bound to
    // the `prefix` register (Reduced(0)); the identity post yields `"prefix"`.
    let post_str = lower_expr(
        post,
        &Lowering {
            leaf: &load,
            reduced: &|s| {
                assert_eq!(
                    s, 0,
                    "Window post references Reduced({s}); only 0 (the window result) exists"
                );
                "prefix".to_string()
            },
            coord: &|d| panic!("cuda backend: Window Coord({d}) is Elementwise-only"),
            unary: &|op, x| {
                if dbl {
                    unary_f64(op, x)
                } else {
                    unary_f32(op, x)
                }
            },
            binary: &|op, a, b| {
                if dbl {
                    binary_f64(op, a, b)
                } else {
                    binary_f32(op, a, b)
                }
            },
            select: &|c, a, b| {
                if dbl {
                    select_f64(c, a, b)
                } else {
                    select_f32(c, a, b)
                }
            },
        },
    );
    let store = |v: &str| -> String {
        match plan.dtype {
            ElementKind::F16 => format!("__float2half({v})"),
            ElementKind::Bf16 => format!("__float2bfloat16({v})"),
            _ => v.to_string(),
        }
    };
    let stored = store(&post_str);

    // ---- Preamble + signature. ----
    let mut s = format!(
        "// Generated by baracuda-kernelgen — do not edit.\n// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    );
    if let Some(inc) = extra_include(plan.dtype) {
        s.push_str(inc);
    }
    s.push('\n');
    s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
    for i in 0..plan.n_inputs {
        s.push_str(&format!("    const {ctype}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {ctype}* __restrict__ out,\n"));
    s.push_str(&format!(
        "    long long n_out,\n    long long k_in,\n    long long k_out{})\n{{\n",
        param_args_multi(&[pre, post], param_ctype(plan))
    ));
    s.push_str("    if (k_out == 0) return;\n");
    s.push_str("    long long total = n_out * k_out;\n");

    // Window geometry as compile-time literals (i64 to keep the tap arithmetic
    // signed — `p` can go negative under pad_lo).
    let (sz, st, dil, plo) = (
        i64::from(size),
        i64::from(stride),
        i64::from(dilation),
        i64::from(pad_lo),
    );
    let _ = pad_hi; // baked into the caller's k_out; the tap loop bounds-checks p < k_in.

    // Grid-stride over output elements. `row = t / k_out`, `o = t % k_out`.
    s.push_str("    for (long long t = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;\n");
    s.push_str("         t < total; t += (long long)gridDim.x * (long long)blockDim.x) {\n");
    s.push_str("        long long row = t / k_out;\n");
    s.push_str("        long long o = t - row * k_out;\n");
    s.push_str("        long long base = row * k_in;\n");
    // Hoist per-row scalar operands once per output (emits nothing for single-input).
    for i in 0..plan.n_inputs {
        if rr_role(plan.key.operands[i as usize], last) == RrRole::RowScalar {
            let conv = match plan.dtype {
                ElementKind::F16 => format!("__half2float(in{i}[row])"),
                ElementKind::Bf16 => format!("__bfloat162float(in{i}[row])"),
                ElementKind::F32Strict => format!("(double)in{i}[row]"),
                _ => format!("in{i}[row]"),
            };
            s.push_str(&format!("        {acc} rs{i} = {conv};\n"));
        }
    }

    match wop {
        ReduceOp::Sum | ReduceOp::Mean => {
            let zero = scan_identity(ReduceOp::Sum, plan.dtype); // "0" / "0.0" / "0.0f"
            s.push_str(&format!("        {acc} acc = {zero};\n"));
            s.push_str("        long long cnt = 0;\n");
            s.push_str(&format!("        for (int kk = 0; kk < {sz}; ++kk) {{\n"));
            s.push_str(&format!(
                "            long long p = o * {st} - {plo} + (long long)kk * {dil};\n"
            ));
            s.push_str("            if (p >= 0 && p < k_in) {\n");
            s.push_str("                long long idx = base + p;\n");
            s.push_str(&format!("                {acc} v = {pre_str};\n"));
            s.push_str("                acc = acc + v;\n");
            s.push_str("                cnt += 1;\n");
            s.push_str("            }\n");
            s.push_str("        }\n");
            if is_mean {
                // avg_pool divisor: `size` (count_include_pad) or the valid-tap count.
                // A cnt==0 window (only reachable at count_include_pad=false, and only
                // for a degenerate all-pad edge the 2*pad<=span gate makes unreachable
                // for the FIRST/LAST windows) stores 0 rather than 0/0 = NaN.
                if count_include_pad {
                    s.push_str(&format!("        {acc} prefix = acc / ({acc}){sz};\n"));
                } else {
                    s.push_str(&format!(
                        "        {acc} prefix = (cnt > 0) ? (acc / ({acc})cnt) : ({acc})0;\n"
                    ));
                }
            } else {
                s.push_str(&format!("        {acc} prefix = acc;\n"));
                s.push_str("        (void)cnt;\n");
            }
            s.push_str(&format!("        out[t] = {stored};\n"));
        }
        ReduceOp::Max | ReduceOp::Min => {
            let cmp = if matches!(wop, ReduceOp::Max) {
                ">"
            } else {
                "<"
            };
            let ident = scan_identity(wop, plan.dtype); // type min (Max) / max (Min)
            s.push_str(&format!("        {acc} best = {ident}; int have = 0;\n"));
            s.push_str(&format!("        for (int kk = 0; kk < {sz}; ++kk) {{\n"));
            s.push_str(&format!(
                "            long long p = o * {st} - {plo} + (long long)kk * {dil};\n"
            ));
            s.push_str("            if (p >= 0 && p < k_in) {\n");
            s.push_str("                long long idx = base + p;\n");
            s.push_str(&format!("                {acc} v = {pre_str};\n"));
            // NaN propagates via `v != v`; padding taps never enter this branch.
            s.push_str(&format!(
                "                if (!have || v != v || v {cmp} best) {{ best = v; have = 1; }}\n"
            ));
            s.push_str("            }\n");
            s.push_str("        }\n");
            // An all-pad window (no valid tap) emits the monoid identity.
            s.push_str(&format!(
                "        {acc} prefix = have ? best : ({ident});\n"
            ));
            s.push_str(&format!("        out[t] = {stored};\n"));
        }
        ReduceOp::Prod => unreachable!("Window rejects Prod"),
    }

    s.push_str("    }\n}\n");
    GeneratedKernel { name, source: s }
}

// ============================================================================
// Increment 11 IM2COL — 2-D im2col / unfold (expanding structured gather).
// ============================================================================

/// The 2-D im2col emitter (`Schedule::Im2Col`) —
/// [`crate::backend::VariantFidelity::BitIdentical`]. One thread per OUTPUT cell
/// (grid-stride over `N*C*kh*kw*oH*oW`): each thread unravels its linear
/// column-matrix index into `(n, c, ki, kj, oh, ow)`, computes the closed-form
/// source coord (`in_h = oh*stride_h - pad_h + ki*dilation_h`, `in_w` symmetric),
/// bounds-checks BOTH spatial axes, and RAW-BIT copies the in-bounds NCHW input
/// element `in0[(((n*C+c)*H_in+in_h)*W_in+in_w)]` — or stores the op's typed ZERO
/// for an out-of-bounds (zero-pad) tap. NO fold, NO accumulator, NO up-convert
/// wrapper: f16/bf16 copy `__half`/`__nv_bfloat16` verbatim so NaN payloads /
/// `-0` signs survive; the only typed value is the OOB zero (matching the bespoke
/// `zero_of<T>()`). Independent per-output copies ⇒ naturally parallel and
/// bit-reproducible; byte-identical to the bespoke `im2col_2d_fw_kernel`.
///
/// The conv geometry `(kh,kw,stride,pad,dilation)` is baked as compile-time
/// literals; the six runtime extents `(N, C, H_in, W_in, oH, oW)` are `long long`
/// launch args (Window's 3-scalar ABI generalized to 6). The
/// `(H_in,W_in) -> (oH,oW)` relationship is the caller's conv-arithmetic
/// precondition (the structure key carries no extents; see `plan::validate_im2col`).
fn emit_im2col(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let (kernel, stride, pad, dilation) = match plan.schedule {
        Schedule::Im2Col {
            kernel,
            stride,
            pad,
            dilation,
        } => (kernel, stride, pad, dilation),
        _ => unreachable!("emit_im2col on a non-Im2Col schedule"),
    };

    // ---- Independent emitter backstops (belt-and-suspenders; validate_im2col
    // validates the same — the 0a lesson: gate every layer). cuda-prefixed messages
    // distinct from the plan gate's. The rank-4 NCHW input is the widest operand, so
    // the shared key.rank pins the input rank; the output rank-3 is not per-operand
    // keyed, so only its LAYOUT is re-checkable here. ----
    assert!(
        plan.key.rank == 4,
        "cuda backend: Im2Col v1 emits a rank-4 NCHW input (key.rank {} != 4)",
        plan.key.rank
    );
    assert!(
        plan.n_inputs == 1 && plan.key.n_operands == 2,
        "cuda backend: Im2Col takes a single input + single output (n_inputs {}, n_operands {})",
        plan.n_inputs,
        plan.key.n_operands
    );
    {
        let i0 = plan.key.operands[0];
        assert!(
            i0.contig == Contiguity::Contig && !i0.flipped && i0.bcast.is_empty(),
            "cuda backend: Im2Col input 0 must be dense forward-contiguous NCHW (the \
             (((n*C+c)*H_in+in_h)*W_in+in_w) address math assumes row-major, no flip/bcast)"
        );
        let o = plan.key.operands[1];
        assert!(
            o.contig == Contiguity::Contig && !o.flipped && o.bcast.is_empty(),
            "cuda backend: Im2Col output must be forward-dense contiguous (empty bcast, not flipped)"
        );
    }
    let (kh, kw) = (i64::from(kernel.0), i64::from(kernel.1));
    let (sh, sw) = (i64::from(stride.0), i64::from(stride.1));
    let (ph, pw) = (i64::from(pad.0), i64::from(pad.1));
    let (dh, dw) = (i64::from(dilation.0), i64::from(dilation.1));
    assert!(
        kh >= 1 && kw >= 1 && sh >= 1 && sw >= 1 && dh >= 1 && dw >= 1,
        "cuda backend: Im2Col kernel/stride/dilation must be >= 1 on each axis"
    );

    let dtag = dtype_tag(plan.dtype);
    let name = format!("baracuda_gen_{}_{dtag}_im2col", plan.op_name);

    // The OOB typed zero — the bespoke `zero_of<T>()`: f16/bf16 need the explicit
    // `__float2half`/`__float2bfloat16` construction; the rest are the plain zero
    // literal (`0` / `0.0` / `0.0f`) from `scan_identity(Sum, dtype)`. The IN-bounds
    // store is a RAW-BIT copy (`out[t] = in0[src]`, same ctype) — NEVER wrapped in an
    // up/down-convert, so NaN payloads survive on f16/bf16.
    let typed_zero = match plan.dtype {
        ElementKind::F16 => "__float2half(0.0f)".to_string(),
        ElementKind::Bf16 => "__float2bfloat16(0.0f)".to_string(),
        _ => scan_identity(ReduceOp::Sum, plan.dtype),
    };

    // ---- Preamble + signature. ----
    let mut s = format!(
        "// Generated by baracuda-kernelgen — do not edit.\n// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    );
    if let Some(inc) = extra_include(plan.dtype) {
        s.push_str(inc);
    }
    s.push('\n');
    s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
    s.push_str(&format!("    const {ctype}* __restrict__ in0,\n"));
    s.push_str(&format!("    {ctype}* __restrict__ out,\n"));
    // Six runtime extents (Window's 3-scalar ABI generalized): N, C, H_in, W_in, oH, oW.
    s.push_str("    long long N,\n    long long C,\n    long long H_in,\n    long long W_in,\n    long long oH,\n    long long oW)\n{\n");
    // total = N*C*kh*kw*oH*oW output cells; kh/kw baked as literals.
    s.push_str(&format!(
        "    long long total = N * C * {kh} * {kw} * oH * oW;\n"
    ));
    s.push_str("    if (total == 0) return;\n");
    s.push_str("    for (long long t = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;\n");
    s.push_str("         t < total; t += (long long)gridDim.x * (long long)blockDim.x) {\n");
    // 6-way unravel: tid -> (n, c, ki, kj, oh, ow). Layout A (channel-major then tap,
    // spatial row-major): row = c*kh*kw + ki*kw + kj, col = oh*oW + ow.
    s.push_str("        long long col = t % (oH * oW);\n");
    s.push_str("        long long row_full = t / (oH * oW);\n");
    s.push_str("        long long oh = col / oW;\n");
    s.push_str("        long long ow = col - oh * oW;\n");
    s.push_str(&format!(
        "        long long patch = row_full % (C * {kh} * {kw});\n"
    ));
    s.push_str(&format!(
        "        long long n = row_full / (C * {kh} * {kw});\n"
    ));
    s.push_str(&format!("        long long c = patch / ({kh} * {kw});\n"));
    s.push_str(&format!(
        "        long long kij = patch - c * ({kh} * {kw});\n"
    ));
    s.push_str(&format!("        long long ki = kij / {kw};\n"));
    s.push_str(&format!("        long long kj = kij - ki * {kw};\n"));
    // Source spatial coord — signed (long long), can go negative under pad.
    s.push_str(&format!(
        "        long long in_h = oh * {sh} - {ph} + ki * {dh};\n"
    ));
    s.push_str(&format!(
        "        long long in_w = ow * {sw} - {pw} + kj * {dw};\n"
    ));
    // Two spatial bounds checks: raw-bit copy the in-bounds source, else typed zero.
    s.push_str("        if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {\n");
    s.push_str("            out[t] = in0[(((n * C + c) * H_in + in_h) * W_in + in_w)];\n");
    s.push_str("        } else {\n");
    s.push_str(&format!("            out[t] = {typed_zero};\n"));
    s.push_str("        }\n");
    s.push_str("    }\n}\n");
    GeneratedKernel { name, source: s }
}

// ============================================================================
// Increment 8 SORT_PERM — row sort / argsort along the innermost axis.
// ============================================================================

/// The **pad sentinel** literal for the bitonic staging — the MAXIMUM of the
/// pair order, so pad cells sort to `[k, pow2)` and the `k` real elements occupy
/// `[0, k)`. Header-light bit-cast forms only (never the `INFINITY` macro).
///
/// ⚠️ For **ascending FP**, `+inf` is WRONG: under the NaN-greatest ordering a
/// real NaN sorts AFTER `+inf`, so a `+inf` pad would push pads into `[0, k)`.
/// The correct asc-FP sentinel is a quiet NaN (the maximum of the NaN-greatest
/// order); pads then land after real NaNs via the index tie-break (each pad cell
/// carries lane index `p >= k`, greater than every real index). For descending
/// the pair order reverses, so the maximum is the type minimum (`-inf` / INT_MIN)
/// — exactly [`type_extreme_lit`]'s most-negative form. For integers the maximum
/// under the order is the type extreme (asc → max, desc → min).
///
/// Invariant (pinned by the on-device validator): a real element EQUAL to the pad
/// key (a real NaN asc, a real INT_MAX asc, a real −inf desc) has index `< k` <
/// every pad index, so it sorts before every pad ⇒ all `k` real elements occupy
/// `[0, k)`.
fn sort_pad_lit(dt: ElementKind, order: SortOrder) -> String {
    match (dt, order) {
        // Float accumulator (F32/F16/Bf16): asc → qNaN, desc → −inf.
        (ElementKind::F32 | ElementKind::F16 | ElementKind::Bf16, SortOrder::Asc) => {
            "__int_as_float(0x7fc00000u)".to_string()
        }
        // Double accumulator (F64/F32Strict): asc → qNaN, desc → −inf.
        (ElementKind::F64 | ElementKind::F32Strict, SortOrder::Asc) => {
            "__longlong_as_double(0x7ff8000000000000ULL)".to_string()
        }
        // Descending FP + every integer: the maximum of the (reversed) pair order
        // is the type extreme; desc = most-negative, asc-int = most-positive.
        (dt, SortOrder::Desc) => type_extreme_lit(dt, true),
        (dt, SortOrder::Asc) => type_extreme_lit(dt, false),
    }
}

/// The per-output RANK-sort BASE (`bitonic = false`) —
/// [`crate::backend::VariantFidelity::BitIdentical`], any `k`, no smem/barriers.
fn emit_row_sort(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    emit_row_sort_impl(plan, ctype, false)
}

/// Emit a row-sort kernel — the per-output RANK sort (`bitonic = false`) or the
/// cooperative smem **bitonic** pair-sort (`bitonic = true`). Both compute the
/// same UNIQUE permutation under a total order on `(key, original-index)` pairs
/// (index tie-break), so they are byte-identical (`BitIdentical`). See
/// [`crate::ir::Access::RowSort`] and §4 of the increment-8 brief.
///
/// The comparator is emitted per `stem` (base + variant + multiple dtype/order
/// cells concatenate into one validator TU without `__device__` collision). Both
/// kernels stage/compare an up-converted `acc` KEY but write the values output as
/// a RAW-BIT permutation of the original `ctype` storage (no round-trip), so NaN
/// payloads and `-0.0` signs are preserved bit-exactly.
fn emit_row_sort_impl(plan: &KernelPlan<'_>, ctype: &str, bitonic: bool) -> GeneratedKernel {
    let (order, out, limit) = match plan.schedule {
        Schedule::RowSort {
            order, out, limit, ..
        } => (order, out, limit),
        _ => unreachable!("emit_row_sort on a non-RowSort schedule"),
    };
    // Increment 10 TOPK/BOTTOMK: `TopK` caps the writeback to a runtime `k_out`
    // (topk = Desc, bottomk = Asc). `Full` reproduces today's whole-row sort
    // BYTE-FOR-BYTE (single `long long k`, base = row*k, no store guard, no `_topk`
    // suffix) — every split below is GATED on this so the existing sort/argsort/Both
    // emission is unchanged.
    let topk = matches!(limit, crate::ir::SortLimit::TopK);

    // ---- Independent emitter backstops (belt-and-suspenders; validate_row_sort
    // validates the same — the 0a lesson: gate every layer). cuda-prefixed
    // messages distinct from the plan gate's. ----
    let rank = plan.key.rank;
    assert!(
        rank >= 1,
        "cuda backend: RowSort needs a sorted axis (rank >= 1)"
    );
    let last = rank - 1;
    {
        let o0 = plan.key.operands[0];
        assert!(
            rr_role(o0, last) == RrRole::RowStreamed
                && o0.contig == Contiguity::Contig
                && !o0.flipped,
            "cuda backend: RowSort input 0 must be the forward-dense contiguous sorted tensor (idx = base+j)"
        );
    }
    assert!(
        matches!(
            plan.dtype,
            ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
                | ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::I32
                | ElementKind::I64
        ),
        "cuda backend: RowSort dtype {:?} is out of the v1 set",
        plan.dtype
    );
    // Increment 9 FUSED_ARGSORT backstop: the two-output `Both` kernel writes
    // [in0, out_val, out_idx] (n_inputs+2 operands), and out_idx (operand 2) is a
    // forward-dense contiguous I32 store. Re-assert both here — the two-pointer
    // signature below assumes them (belt-and-suspenders for validate_row_sort's G2/G3).
    // Increment 10 TopK rides the SAME backstop (the layout re-check is width-agnostic,
    // so the narrower `[batch,k_out]` out_idx passes; the `k_out <= k_in` relationship
    // is NOT emitter-checkable — the golden + on-device initcheck cover it).
    if matches!(out, SortOut::Both) {
        let n_inputs = plan.n_inputs as usize;
        assert!(
            plan.key.n_operands as usize == n_inputs + 2,
            "cuda backend: RowSort Both writes two outputs — key must carry \
             n_inputs+2 operands (in0, out_val, out_idx)"
        );
        let oi = plan.key.operands[n_inputs + 1];
        assert!(
            oi.bcast.is_empty() && oi.contig == Contiguity::Contig && !oi.flipped,
            "cuda backend: RowSort Both out_idx (operand 2) must be forward-dense \
             contiguous (empty bcast, not flipped); width may be < input for a TopK cap"
        );
    }

    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let int_acc = crate::plan::is_int_dtype(plan.dtype);
    let is_fp = !int_acc;
    // Comparator/staging accumulator: double for f64/f32-strict, native ctype for
    // integers, float otherwise (f32/f16/bf16 up-convert). Its byte size drives the
    // dynamic-smem launch contract.
    let (acc, acc_sz) = if dbl {
        ("double", 8usize)
    } else if int_acc {
        (
            ctype,
            if matches!(plan.dtype, ElementKind::I64) {
                8
            } else {
                4
            },
        )
    } else {
        ("float", 4usize)
    };
    // The output element ctype: `int` for argsort (I32 index), the input ctype
    // otherwise (plan.out_dtype is the resolved out dtype).
    let octype = scalar_ctype(plan.out_dtype).expect(
        "RowSort out dtype must have a scalar ctype (I32 for argsort, else the input dtype)",
    );

    let dtag = dtype_tag(plan.dtype);
    let ord = match order {
        SortOrder::Asc => "asc",
        SortOrder::Desc => "desc",
    };
    // Three-way output suffix (increment 9): Values ⇒ "", Indices ⇒ "_idx", Both
    // ⇒ "_both" — on BOTH the `__device__` stem and the exported `name`, so the
    // fused kernel's comparators never collide with the two single-output kernels
    // in the validator TU. Values/Indices reproduce today's suffixes byte-for-byte.
    let out_suf = match out {
        SortOut::Values => "",
        SortOut::Indices => "_idx",
        SortOut::Both => "_both",
    };
    // Increment 10: the runtime-k cap rides `_topk` on BOTH the exported `name` and
    // the `__device__` `stem`, AFTER `out_suf`, so capped kernels never collide with
    // the full-sort kernels in the validator TU. `Full` keeps today's exact symbols
    // (byte-identity).
    let limit_suf = match limit {
        crate::ir::SortLimit::Full => "",
        crate::ir::SortLimit::TopK => "_topk",
    };
    let bt_suf = if bitonic { "_bt" } else { "" };
    // Device-helper stem: base+variant + every dtype/order/out cell of one op get a
    // distinct `__device__` comparator symbol (no collision in the validator TU).
    let stem = format!("{}_{dtag}_{ord}{out_suf}{limit_suf}{bt_suf}", plan.op_name);
    let name = format!(
        "baracuda_gen_{}_{dtag}_rowsort_{ord}_stable{}{}{}",
        plan.op_name,
        out_suf,
        limit_suf,
        if bitonic { "_bitonic" } else { "" }
    );

    // Up-convert load of `in0[{idx}]` into the `acc` KEY (f16/bf16/f32-strict widen).
    let load_at = |idx: &str| -> String {
        match plan.dtype {
            ElementKind::F16 => format!("__half2float(in0[{idx}])"),
            ElementKind::Bf16 => format!("__bfloat162float(in0[{idx}])"),
            ElementKind::F32Strict => format!("(double)in0[{idx}]"),
            _ => format!("in0[{idx}]"),
        }
    };

    // ---- Preamble + comparator. ----
    let mut s = format!(
        "// Generated by baracuda-kernelgen — do not edit.\n// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    );
    if let Some(inc) = extra_include(plan.dtype) {
        s.push_str(inc);
    }
    s.push('\n');

    // key_lt: strict-less under the NaN-greatest total preorder (NaN > all non-NaN;
    // NaN ties NaN). The NaN branch is emitted for FP dtypes only. `-0.0 == +0.0`
    // is a key tie broken by index. pair_lt lifts it to a strict total order on
    // (key, index): the `order`-adjusted key first, then the ascending index tie
    // (stable in both directions).
    s.push_str(&format!(
        "__device__ __forceinline__ bool {stem}_key_lt({acc} a, {acc} b) {{\n"
    ));
    if is_fp {
        s.push_str("    if (a != a) return false;\n");
        s.push_str("    if (b != b) return true;\n");
    }
    s.push_str("    return a < b;\n}\n");
    let (fa, fb, sa, sb) = match order {
        SortOrder::Asc => ("ka", "kb", "kb", "ka"),
        SortOrder::Desc => ("kb", "ka", "ka", "kb"),
    };
    // Tie indices are `long long` so the any-k values-sort base is index-exact at
    // every k the address arithmetic reaches (review-caught: an `int` tie index
    // truncated the LOAD address too — an OOB read past 2^31 and a rank collision
    // past 2^32). The bitonic passes its `int sidx[]` (k <= 1024) — lossless widen.
    s.push_str(&format!(
        "__device__ __forceinline__ bool {stem}_pair_lt({acc} ka, long long ia, {acc} kb, long long ib) {{\n\
         \x20   if ({stem}_key_lt({fa}, {fb})) return true;\n\
         \x20   if ({stem}_key_lt({sa}, {sb})) return false;\n\
         \x20   return ia < ib;\n\
         }}\n\n"
    ));

    // The value writeback is the raw ctype bits (no up/down convert); an index
    // writeback is the original index narrowed to the I32 output dtype, so
    // k <= 2^31 - 1 is an INHERENT precondition of any index-writing state
    // (`Indices` and `Both`) — the values-only sort has no such cap. Both stores
    // are hand-emitted at the two store sites below (both quantities are already
    // in scope there — no new computation for the fused `Both`).

    s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
    s.push_str(&format!("    const {ctype}* __restrict__ in0,\n"));
    match out {
        // Fused two-output (increment 9): values (input ctype) to out_val AND the
        // I32 index to out_idx. out_val's ctype == octype here (Both ⇒ out_dtype
        // None ⇒ key.dtype); out_idx is the hardwired `int` (the argsort precedent).
        SortOut::Both => {
            s.push_str(&format!("    {ctype}* __restrict__ out_val,\n"));
            s.push_str("    int* __restrict__ out_idx,\n");
        }
        // Values ⇒ input ctype; Indices ⇒ int (octype = scalar_ctype(out_dtype)).
        SortOut::Values | SortOut::Indices => {
            s.push_str(&format!("    {octype}* __restrict__ out,\n"));
        }
    }
    // Increment 10: `Full` keeps today's 2-scalar `(n_out, k)` ABI byte-for-byte;
    // `TopK` rides the Window `(n_out, k_in, k_out)` 3-scalar ABI (`k_out` the OUT
    // operand's inner extent, a runtime launch arg).
    if topk {
        s.push_str("    long long n_out,\n    long long k_in,\n    long long k_out)\n{\n");
        s.push_str("    if (k_out == 0) return;\n");
    } else {
        s.push_str("    long long n_out,\n    long long k)\n{\n");
        s.push_str("    if (k == 0) return;\n");
    }

    if !bitonic {
        // -------------------- RANK-SORT BASE (one thread per output element) --------------------
        // Each thread owns output element `(row, i)`: it scans its row computing the
        // element's RANK (# of pairs strictly less), then writes element `i` to slot
        // `base + rank`. Pairs are all distinct ⇒ ranks are a permutation ⇒ every
        // slot written exactly once (no races), stable by construction, ANY k, no
        // smem, no __syncthreads. O(k²) reads — correctness base; perf is the variant.
        //
        // Increment 10 TopK: grid over ALL inputs (`n_out * k_in`, every element
        // needs its rank to know if it lands in the top `k_out`), split the bases
        // (`in_base = row*k_in`, `out_base = row*k_out`), scan `j < k_in`, and GUARD
        // the store `if (r < k_out)`. Ranks are a permutation of `[0, k_in)`, so
        // exactly `k_out` threads pass the guard ⇒ all `k_out` out slots written
        // once, none twice (initcheck-clean). `Full` emits the lines below verbatim.
        let (total_x, rowlen) = if topk {
            ("n_out * k_in", "k_in")
        } else {
            ("n_out * k", "k")
        };
        // Input base variable: `in_base` (TopK) vs `base` (Full, byte-identical).
        let ib = if topk { "in_base" } else { "base" };
        // Output base variable: `out_base` (TopK) vs `base` (Full, byte-identical).
        let ob = if topk { "out_base" } else { "base" };
        s.push_str(&format!("    long long total = {total_x};\n"));
        s.push_str("    long long grid_stride = (long long)blockDim.x * (long long)gridDim.x;\n");
        s.push_str("    for (long long t = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;\n");
        s.push_str("         t < total; t += grid_stride) {\n");
        s.push_str(&format!("        long long row = t / {rowlen};\n"));
        if topk {
            s.push_str("        long long in_base = row * k_in;\n");
            s.push_str("        long long out_base = row * k_out;\n");
            s.push_str("        long long i = t - in_base;\n");
        } else {
            s.push_str("        long long base = row * k;\n");
            s.push_str("        long long i = t - base;\n");
        }
        s.push_str(&format!(
            "        {acc} ki = {};\n",
            load_at(&format!("{ib} + i"))
        ));
        s.push_str("        long long r = 0;\n");
        s.push_str(&format!(
            "        for (long long j = 0; j < {rowlen}; ++j) {{\n"
        ));
        s.push_str(&format!(
            "            {acc} kj = {};\n",
            load_at(&format!("{ib} + j"))
        ));
        s.push_str(&format!(
            "            if ({stem}_pair_lt(kj, j, ki, i)) r++;\n"
        ));
        s.push_str("        }\n");
        // Store: Values writes the raw value, Indices the I32 index, Both writes
        // BOTH (both quantities in scope — no recomputation, the fusion payoff).
        // TopK wraps the store in `if (r < k_out)` (indent +4); Full stores flat.
        let g = if topk { "    " } else { "" }; // store-line extra indent under the guard
        if topk {
            s.push_str("        if (r < k_out) {\n");
        }
        match out {
            SortOut::Values => {
                s.push_str(&format!("        {g}out[{ob} + r] = in0[{ib} + i];\n"));
            }
            SortOut::Indices => s.push_str(&format!("        {g}out[{ob} + r] = (int)i;\n")),
            SortOut::Both => {
                s.push_str(&format!("        {g}out_val[{ob} + r] = in0[{ib} + i];\n"));
                s.push_str(&format!("        {g}out_idx[{ob} + r] = (int)i;\n"));
            }
        }
        if topk {
            s.push_str("        }\n");
        }
        s.push_str("    }\n");
    } else {
        // -------------------- BITONIC PAIR-SORT VARIANT (one block per row) --------------------
        // The whole padded row is staged in dynamic smem as (acc key, int index)
        // pairs; the network sorts by `pair_lt` (order baked into the comparator);
        // the values writeback gathers RAW input bits through the final permutation
        // (raw ctype is never staged / round-tripped). Launch contract (launch_note):
        // blockDim a multiple of 32, <= 1024; dynamic smem = pow2 * (acc_sz + 4)
        // bytes; REQUIRES k <= 1024 (no emitted guard beyond k == 0 — the structure
        // key carries no extents; on-device-validated, per smemrow/blockscan).
        // Increment 10 TopK: stage the FULL input row (`pow2` over `k_in`, smem
        // bound on `k_in` unchanged), sort it, then SHORTEN the writeback to
        // `p < k_out`. Split the bases: `base_in = row*k_in` (staging/gather),
        // `out_base = row*k_out` (writeback). `Full` emits the lines below verbatim.
        let (rowlen, ib, ob, wbound) = if topk {
            ("k_in", "base_in", "out_base", "k_out")
        } else {
            ("k", "base", "base", "k")
        };
        let pad = sort_pad_lit(plan.dtype, order);
        s.push_str(&format!(
            "    long long pow2 = 1; while (pow2 < {rowlen}) pow2 <<= 1;\n"
        ));
        s.push_str("    extern __shared__ unsigned char baracuda_sort_smem[];\n");
        s.push_str(&format!("    {acc}* skey = ({acc}*)baracuda_sort_smem;\n"));
        s.push_str(&format!(
            "    int* sidx = (int*)(baracuda_sort_smem + (size_t)pow2 * sizeof({acc}));\n"
        ));
        s.push_str(
            "    for (long long row = blockIdx.x; row < n_out; row += (long long)gridDim.x) {\n",
        );
        if topk {
            s.push_str("        long long base_in = row * k_in;\n");
            s.push_str("        long long out_base = row * k_out;\n");
        } else {
            s.push_str("        long long base = row * k;\n");
        }
        // Stage + pad (all threads reach every barrier — the p-loop is uniform).
        s.push_str("        for (long long p = threadIdx.x; p < pow2; p += blockDim.x) {\n");
        let cond = format!("if (p < {rowlen}) ");
        s.push_str(&format!(
            "            {cond}{{ skey[p] = {}; sidx[p] = (int)p; }}\n",
            load_at(&format!("{ib} + p"))
        ));
        let else_pad = " ".repeat(cond.len() - "else".len());
        s.push_str(&format!(
            "            else{else_pad}{{ skey[p] = {pad}; sidx[p] = (int)p; }}\n"
        ));
        s.push_str("        }\n");
        s.push_str("        __syncthreads();\n");
        // Bitonic network. The __syncthreads sits OUTSIDE the p-loop at a uniform
        // program point (never inside `if (q > p)`); only the `q > p` owner touches
        // a disjoint pair, so the strided p-loop is race-free within a phase.
        s.push_str("        for (long long kk = 2; kk <= pow2; kk <<= 1) {\n");
        s.push_str("            for (long long j = kk >> 1; j > 0; j >>= 1) {\n");
        s.push_str(
            "                for (long long p = threadIdx.x; p < pow2; p += blockDim.x) {\n",
        );
        s.push_str("                    long long q = p ^ j;\n");
        s.push_str("                    if (q > p) {\n");
        s.push_str("                        bool up = ((p & kk) == 0);\n");
        s.push_str(&format!(
            "                        bool q_lt_p = {stem}_pair_lt(skey[q], sidx[q], skey[p], sidx[p]);\n"
        ));
        s.push_str("                        if (up == q_lt_p) {\n");
        s.push_str(&format!(
            "                            {acc} tk = skey[p]; skey[p] = skey[q]; skey[q] = tk;\n"
        ));
        s.push_str(
            "                            int ti = sidx[p]; sidx[p] = sidx[q]; sidx[q] = ti;\n",
        );
        s.push_str("                        }\n");
        s.push_str("                    }\n");
        s.push_str("                }\n");
        s.push_str("                __syncthreads();\n");
        s.push_str("            }\n");
        s.push_str("        }\n");
        // Writeback: the first `k`/`k_out` cells are exactly the retained real
        // elements (pad invariant — the k_in real elements occupy `[0, k_in)`, so
        // the first `k_out <= k_in` are the top-k_out under `order`).
        s.push_str(&format!(
            "        for (long long p = threadIdx.x; p < {wbound}; p += blockDim.x) {{\n"
        ));
        match out {
            SortOut::Values => {
                s.push_str(&format!(
                    "            out[{ob} + p] = in0[{ib} + sidx[p]];\n"
                ));
            }
            SortOut::Indices => s.push_str(&format!("            out[{ob} + p] = sidx[p];\n")),
            SortOut::Both => {
                s.push_str(&format!(
                    "            out_val[{ob} + p] = in0[{ib} + sidx[p]];\n"
                ));
                s.push_str(&format!("            out_idx[{ob} + p] = sidx[p];\n"));
            }
        }
        s.push_str("        }\n");
        s.push_str("        __syncthreads();\n"); // before the next row reuses smem
        s.push_str("    }\n");
    }

    s.push_str("}\n");
    let _ = acc_sz;
    GeneratedKernel { name, source: s }
}

/// Emit the **PARTIAL-SELECT streaming tiled-bitonic top-k** kernel — a wholly
/// separate emitter from [`emit_row_sort_impl`] (which stays byte-untouched).
///
/// Instead of staging the whole `next_pow2(k_in)`-padded row (the bitonic
/// variant, O(k_in·log²k_in) and REQUIRES k_in ≤ 1024), this streams the row in
/// tiles of `m = next_pow2(k_out)` against a running best-buffer B of m pairs:
/// per tile it fully bitonic-sorts the `2m` window `[B | T]` under `pair_lt`
/// (ascending) and keeps the LOWER m. By the composition lemma
/// `min_m(min_m(X)∪Y) = min_m(X∪Y)` plus the never-evict property (a top-k_out
/// member has global rank `< k_out ≤ m` and zero pads rank below it, so it is
/// never in the discarded upper half), `B_final[0..k_out)` equals the full-sort
/// top-k_out prefix **bit-for-bit** — [`VariantFidelity::BitIdentical`]. The
/// smem/occupancy contract is on `k_out` (2m arena), NOT `k_in`, so there is no
/// `k_in ≤ 1024` cap: this is the unique fast+correct path for large-vocab
/// (`k_in > 1024`) top-k. See the PARTIAL-SELECT TOPK brief §1–§3.
///
/// **The three load-bearing invariants** (the panel's only divergence surfaces):
/// (1) **Pad key** = [`sort_pad_lit`] verbatim (topk→type-min, bottomk→FP qNaN /
/// INTEGER INT_MAX — never a +inf FP pad, never a NaN-bitcast int pad). (2) **The
/// stored index is the ORIGINAL GLOBAL index** `g = t*m + p` for BOTH reals AND
/// pads (NEVER the local smem slot — the shipped bitonic stores the slot only
/// because there slot == global index; here a slot `p < k_in` generally is not
/// the global index, so a slot store would break the value gather, the tie-break,
/// and would put pad indices below k_in). (3) **Keep the LOWER half** (constant
/// for both directions; direction rides ONLY in `pair_lt`'s operand swap).
///
/// The comparator (`{stem}_key_lt` + `{stem}_pair_lt`) and the `load_at`
/// up-convert are re-emitted with the SAME logic as [`emit_row_sort_impl`] (a
/// distinct `_psel` stem so the validator TU never collides), so the on-device
/// `memcmp` of `_psel` output vs the full-sort base is a like-for-like order.
fn emit_row_sort_partial(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let (order, out, limit) = match plan.schedule {
        Schedule::RowSort {
            order, out, limit, ..
        } => (order, out, limit),
        _ => unreachable!("emit_row_sort_partial on a non-RowSort schedule"),
    };
    // Emitter backstop (dual-gate; the variant gate is primary): the streaming
    // top-k serves ONLY the runtime-k cap. A Full cell has no k_out to stream to.
    assert!(
        matches!(limit, crate::ir::SortLimit::TopK),
        "cuda backend: emit_row_sort_partial serves only SortLimit::TopK (the \
         streaming top-k has no whole-row Full form)"
    );

    // ---- Independent emitter backstops (identical in0/out layout re-checks as
    // emit_row_sort_impl 5048-5091; belt-and-suspenders for the variant gate). ----
    let rank = plan.key.rank;
    assert!(
        rank >= 1,
        "cuda backend: RowSort needs a sorted axis (rank >= 1)"
    );
    let last = rank - 1;
    {
        let o0 = plan.key.operands[0];
        assert!(
            rr_role(o0, last) == RrRole::RowStreamed
                && o0.contig == Contiguity::Contig
                && !o0.flipped,
            "cuda backend: RowSort input 0 must be the forward-dense contiguous sorted tensor (idx = base+j)"
        );
    }
    assert!(
        matches!(
            plan.dtype,
            ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
                | ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::I32
                | ElementKind::I64
        ),
        "cuda backend: RowSort dtype {:?} is out of the v1 set",
        plan.dtype
    );
    if matches!(out, SortOut::Both) {
        let n_inputs = plan.n_inputs as usize;
        assert!(
            plan.key.n_operands as usize == n_inputs + 2,
            "cuda backend: RowSort Both writes two outputs — key must carry \
             n_inputs+2 operands (in0, out_val, out_idx)"
        );
        let oi = plan.key.operands[n_inputs + 1];
        assert!(
            oi.bcast.is_empty() && oi.contig == Contiguity::Contig && !oi.flipped,
            "cuda backend: RowSort Both out_idx (operand 2) must be forward-dense \
             contiguous (empty bcast, not flipped); width may be < input for a TopK cap"
        );
    }

    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let int_acc = crate::plan::is_int_dtype(plan.dtype);
    let is_fp = !int_acc;
    // Comparator/staging accumulator — EXACTLY as emit_row_sort_impl (double for
    // f64/f32-strict, native ctype for integers, float otherwise).
    let acc = if dbl {
        "double"
    } else if int_acc {
        ctype
    } else {
        "float"
    };
    let octype = scalar_ctype(plan.out_dtype).expect(
        "RowSort out dtype must have a scalar ctype (I32 for argsort, else the input dtype)",
    );

    let dtag = dtype_tag(plan.dtype);
    let ord = match order {
        SortOrder::Asc => "asc",
        SortOrder::Desc => "desc",
    };
    let out_suf = match out {
        SortOut::Values => "",
        SortOut::Indices => "_idx",
        SortOut::Both => "_both",
    };
    // `_topk` (limit) then `_psel` (this variant) — a distinct entry point AND a
    // distinct `__device__` comparator stem so nothing collides with the
    // base/bitonic in the validator TU (additive keying; no SKV bump).
    let limit_suf = "_topk";
    let stem = format!("{}_{dtag}_{ord}{out_suf}{limit_suf}_psel", plan.op_name);
    let name = format!(
        "baracuda_gen_{}_{dtag}_rowsort_{ord}_stable{out_suf}{limit_suf}_psel",
        plan.op_name
    );

    // Up-convert load of `in0[{idx}]` into the `acc` KEY (f16/bf16/f32-strict
    // widen) — SAME as emit_row_sort_impl's load_at.
    let load_at = |idx: &str| -> String {
        match plan.dtype {
            ElementKind::F16 => format!("__half2float(in0[{idx}])"),
            ElementKind::Bf16 => format!("__bfloat162float(in0[{idx}])"),
            ElementKind::F32Strict => format!("(double)in0[{idx}]"),
            _ => format!("in0[{idx}]"),
        }
    };

    // ---- Preamble + comparator (byte-identical LOGIC to emit_row_sort_impl). ----
    let mut s = format!(
        "// Generated by baracuda-kernelgen — do not edit.\n// op: {} | cell: {}\n",
        plan.op_name,
        plan.key.to_token()
    );
    if let Some(inc) = extra_include(plan.dtype) {
        s.push_str(inc);
    }
    s.push('\n');

    s.push_str(&format!(
        "__device__ __forceinline__ bool {stem}_key_lt({acc} a, {acc} b) {{\n"
    ));
    if is_fp {
        s.push_str("    if (a != a) return false;\n");
        s.push_str("    if (b != b) return true;\n");
    }
    s.push_str("    return a < b;\n}\n");
    let (fa, fb, sa, sb) = match order {
        SortOrder::Asc => ("ka", "kb", "kb", "ka"),
        SortOrder::Desc => ("kb", "ka", "ka", "kb"),
    };
    s.push_str(&format!(
        "__device__ __forceinline__ bool {stem}_pair_lt({acc} ka, long long ia, {acc} kb, long long ib) {{\n\
         \x20   if ({stem}_key_lt({fa}, {fb})) return true;\n\
         \x20   if ({stem}_key_lt({sa}, {sb})) return false;\n\
         \x20   return ia < ib;\n\
         }}\n\n"
    ));

    // ---- Signature: the (n_out, k_in, k_out) 3-scalar TopK ABI (verbatim). ----
    s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
    s.push_str(&format!("    const {ctype}* __restrict__ in0,\n"));
    match out {
        SortOut::Both => {
            s.push_str(&format!("    {ctype}* __restrict__ out_val,\n"));
            s.push_str("    int* __restrict__ out_idx,\n");
        }
        SortOut::Values | SortOut::Indices => {
            s.push_str(&format!("    {octype}* __restrict__ out,\n"));
        }
    }
    s.push_str("    long long n_out,\n    long long k_in,\n    long long k_out)\n{\n");
    s.push_str("    if (k_out == 0) return;\n");

    // -------------------- STREAMING TILED-BITONIC TOP-K --------------------
    // m = next_pow2(k_out); ONE contiguous 2m-pair arena: B = W[0..m), T = W[m..2m).
    // Dynamic smem = 2m * (sizeof(acc) + sizeof(int)) — bounded on k_out, NOT k_in.
    let pad = sort_pad_lit(plan.dtype, order);
    s.push_str("    long long m = 1; while (m < k_out) m <<= 1;\n");
    s.push_str("    extern __shared__ unsigned char baracuda_sort_smem[];\n");
    s.push_str(&format!("    {acc}* skey = ({acc}*)baracuda_sort_smem;\n"));
    s.push_str(&format!(
        "    int* sidx = (int*)(baracuda_sort_smem + (size_t)(2*m) * sizeof({acc}));\n"
    ));
    s.push_str(
        "    for (long long row = blockIdx.x; row < n_out; row += (long long)gridDim.x) {\n",
    );
    s.push_str("        long long base_in = row * k_in;\n");
    s.push_str("        long long out_base = row * k_out;\n");
    // Init the running best-buffer B (lower half) with pads. Their index is any
    // value >= k_in, so they are "<"-dominated by tile-0 reals and never output;
    // still stored so the pre-tile-0 buffer is a well-defined pair set. SATURATE at
    // INT_MAX: k_in <= 2^31-1 (the inherited i32 out_idx cap), so `k_in + p` can
    // exceed INT_MAX for the top-m band of the supported range and a bare `(int)`
    // would wrap NEGATIVE — a negative pad index would then "<"-BEAT a real on a
    // key-tie (OOB read + wrong output). A saturated INT_MAX still dominates every
    // real (< k_in <= INT_MAX), and pad/pad index ties are harmless (pads never
    // reach [0,k_out)). Reals/tile-pads store `(int)g` with g <= INT_MAX, so only
    // this init store (k_in + p, not rounded to a multiple of m) can overflow.
    s.push_str("        for (long long p = threadIdx.x; p < m; p += blockDim.x) {\n");
    s.push_str(&format!(
        "            skey[p] = {pad}; long long pi = k_in + p; sidx[p] = (int)(pi < 2147483647LL ? pi : 2147483647LL);\n"
    ));
    s.push_str("        }\n");
    s.push_str("        __syncthreads();\n");
    // Stream ceil(k_in/m) tiles of m elements each.
    s.push_str("        for (long long t = 0; t * m < k_in; ++t) {\n");
    // Load the T half (upper m). sidx = the GLOBAL index g for BOTH branches (the
    // #1 bug surface — NEVER the local slot s-m): a real (g < k_in) gathers its raw
    // value at in0[base_in + g] and tie-breaks on g; a pad (g >= k_in) is dominated.
    s.push_str("            for (long long s = m + threadIdx.x; s < 2*m; s += blockDim.x) {\n");
    s.push_str("                long long g = t * m + (s - m);\n");
    s.push_str(&format!(
        "                if (g < k_in) {{ skey[s] = {}; }}\n",
        load_at("base_in + g")
    ));
    s.push_str(&format!(
        "                else       {{ skey[s] = {pad}; }}\n"
    ));
    s.push_str("                sidx[s] = (int)g;\n");
    s.push_str("            }\n");
    s.push_str("            __syncthreads();\n");
    // FULL Batcher bitonic sort of the 2m window (ascending under pair_lt) — the
    // shipped ladder with pow2 := 2*m, the barrier OUTSIDE if(q>p) at a uniform
    // program point (byte-identical structure to emit_row_sort_impl 5361-5384).
    s.push_str("            for (long long kk = 2; kk <= 2*m; kk <<= 1) {\n");
    s.push_str("                for (long long j = kk >> 1; j > 0; j >>= 1) {\n");
    s.push_str("                    for (long long p = threadIdx.x; p < 2*m; p += blockDim.x) {\n");
    s.push_str("                        long long q = p ^ j;\n");
    s.push_str("                        if (q > p) {\n");
    s.push_str("                            bool up = ((p & kk) == 0);\n");
    s.push_str(&format!(
        "                            bool q_lt_p = {stem}_pair_lt(skey[q], sidx[q], skey[p], sidx[p]);\n"
    ));
    s.push_str("                            if (up == q_lt_p) {\n");
    s.push_str(&format!(
        "                                {acc} tk = skey[p]; skey[p] = skey[q]; skey[q] = tk;\n"
    ));
    s.push_str(
        "                                int ti = sidx[p]; sidx[p] = sidx[q]; sidx[q] = ti;\n",
    );
    s.push_str("                            }\n");
    s.push_str("                        }\n");
    s.push_str("                    }\n");
    s.push_str("                    __syncthreads();\n");
    s.push_str("                }\n");
    s.push_str("            }\n");
    // Keep the LOWER half implicitly (W[0..m) now holds the m smallest of [B|T]).
    // Barrier before the next tile overwrites W[m..2m).
    s.push_str("            __syncthreads();\n");
    s.push_str("        }\n");
    // Writeback the first k_out (the retained top-k_out reals — their sidx are real
    // global indices < k_in). Three-arm store identical to emit_row_sort_impl.
    s.push_str("        for (long long p = threadIdx.x; p < k_out; p += blockDim.x) {\n");
    match out {
        SortOut::Values => {
            s.push_str("            out[out_base + p] = in0[base_in + sidx[p]];\n");
        }
        SortOut::Indices => s.push_str("            out[out_base + p] = sidx[p];\n"),
        SortOut::Both => {
            s.push_str("            out_val[out_base + p] = in0[base_in + sidx[p]];\n");
            s.push_str("            out_idx[out_base + p] = sidx[p];\n");
        }
    }
    s.push_str("        }\n");
    s.push_str("        __syncthreads();\n"); // before the next row reuses smem
    s.push_str("    }\n");

    s.push_str("}\n");
    GeneratedKernel { name, source: s }
}

/// Bitonic-pair-sort schedule VARIANT for the [`Schedule::RowSort`] cell — one
/// block per row, the whole padded row staged in dynamic smem as `(key, index)`
/// pairs, sorted by a bitonic network under the `pair_lt` total order. A
/// [`Variant`] filter (model: [`scan_blockscan_variant`]); `return None` for
/// every cell it cannot serve.
///
/// **Bits:** a pair sort with an index tie-break is a pure PERMUTATION under a
/// unique total order — no FP arithmetic, no reassociation — so it is byte-for-
/// byte identical to the rank-sort base:
/// [`VariantFidelity::BitIdentical`] (unlike `scan_blockscan_variant`, there is
/// NO FP-only gate — int sorts ride the same network). Selectable silently; the
/// validator pins the base ≡ variant memcmp.
///
/// **Contract (`launch_note`):** blockDim a multiple of 32, `<= 1024`; grid = any
/// (grid-stride over rows; one block per row is natural); dynamic smem =
/// `next_pow2(k) * (acc_sz + 4)` bytes; **REQUIRES `k <= 1024`** — an on-device-
/// validated precondition (the structure key carries no extents), the same trust
/// model as smemrow/blockscan, with NO emitted guard beyond `k == 0`.
fn row_sort_bitonic_variant(plan: &KernelPlan<'_>) -> Option<Variant> {
    let Schedule::RowSort { limit, .. } = plan.schedule else {
        return None;
    };
    // The fixed signature has no `p{i}` slots — a Param body would emit an
    // undefined identifier (mirror blockscan/split-K). RowSort's body is pinned
    // Input(0), so this is always empty; the guard is load-bearing house style.
    if !params_used(plan.body).is_empty() {
        return None;
    }
    let ctype = scalar_ctype(plan.dtype)?;
    // Re-assert in0 layout (validate_row_sort already gated it; belt-and-suspenders).
    let last = plan.key.rank.saturating_sub(1);
    let in0 = plan.key.operands[0];
    let out = plan.key.operands[plan.key.n_operands.saturating_sub(1) as usize];
    if rr_role(in0, last) != RrRole::RowStreamed
        || in0.contig != Contiguity::Contig
        || in0.flipped
        || !out.bcast.is_empty()
        || out.contig != Contiguity::Contig
        || out.flipped
    {
        return None;
    }
    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let acc_sz = if dbl || matches!(plan.dtype, ElementKind::I64) {
        8
    } else {
        4
    };
    let k = emit_row_sort_impl(plan, ctype, true);
    // The bitonic network stages and sorts the WHOLE input row regardless of the
    // cap, so the smem/occupancy contract is on the INPUT row length: `k` for a
    // Full sort, `k_in` for a TopK (which then writes only the top `k_out`). The
    // Full note is kept byte-identical (its launch_note goldens pin it); the TopK
    // note names the correct `(k_in, k_out)` ABI so the metadata is not stale.
    let launch_note = match limit {
        crate::ir::SortLimit::Full => format!(
            "bitonic pair-sort (one block per row, grid-stride over rows; the whole \
             next_pow2(k)-padded row staged in dynamic shared memory as (key, index) \
             pairs and sorted by a bitonic network under the (key, original-index) \
             total order): <<<min(n_out, maxblocks), B, smem>>> with B a multiple of \
             32 and <= 1024, and dynamic shared memory = next_pow2(k) * {} bytes \
             (sizeof(acc)={} + sizeof(int)=4). REQUIRES k <= 1024 (the bitonic \
             network stages the whole padded row in one block); longer rows use the \
             any-k rank-sort base. BitIdentical to the base (a pair sort is a pure \
             permutation). Determinism: {}.",
            acc_sz + 4,
            acc_sz,
            VariantFidelity::BitIdentical.determinism_str()
        ),
        crate::ir::SortLimit::TopK => format!(
            "bitonic top-k (one block per row, grid-stride over rows; the whole \
             next_pow2(k_in)-padded INPUT row staged in dynamic shared memory as \
             (key, index) pairs and sorted by a bitonic network under the (key, \
             original-index) total order, then only the top k_out written): \
             <<<min(n_out, maxblocks), B, smem>>> with B a multiple of 32 and \
             <= 1024, and dynamic shared memory = next_pow2(k_in) * {} bytes \
             (sizeof(acc)={} + sizeof(int)=4). REQUIRES k_in <= 1024 (the bitonic \
             network stages the whole padded INPUT row in one block); longer input \
             rows use the any-k rank-sort base. k_out <= k_in is a caller/on-device \
             precondition. BitIdentical to the base (a pair sort is a pure \
             permutation). Determinism: {}.",
            acc_sz + 4,
            acc_sz,
            VariantFidelity::BitIdentical.determinism_str()
        ),
    };
    Some(Variant {
        tag: "bitonic",
        kernels: vec![k],
        fidelity: VariantFidelity::BitIdentical,
        launch_note,
    })
}

/// PARTIAL-SELECT streaming tiled-bitonic **top-k** schedule VARIANT for the
/// [`Schedule::RowSort`] cell — one block per row, a running best-buffer of
/// `m = next_pow2(k_out)` pairs streamed against the input in tiles of m
/// (O(k_in·log²k_out)). The unique fast+correct path for `k_in > 1024`
/// large-vocab top-k, where the bitonic variant declines (`k_in ≤ 1024` cap) and
/// the rank base is O(k_in²). A [`Variant`] filter (model:
/// [`row_sort_bitonic_variant`]); `return None` for every cell it cannot serve.
///
/// **Bits:** the streamed keep-lower-m composition returns the full-sort
/// top-k_out prefix bit-for-bit (design-panel PROVEN: composition lemma +
/// never-evict, 0 divergences / 200 000 adversarial rows), so
/// [`VariantFidelity::BitIdentical`] — selectable silently; the validator pins
/// `_psel` ≡ the full-sort base memcmp.
///
/// **Scope (TopK only):** `Full` declines to the bitonic/base — the streaming
/// buffer is a top-`k_out` selector with no whole-row form. Values / Indices /
/// Both all served (the index rides in the pair). Same in0/out layout re-check as
/// the bitonic.
///
/// **No compile-time advantage decline:** the panel's `2·next_pow2(k_out) ≥
/// next_pow2(k_in)` "no-advantage" filter is NOT gateable here — the
/// [`StructureKey`] carries size CLASSES, never literal extents (the §1
/// non-negotiable; `plan.key`/`plan.schedule`/`plan.rs` expose neither `k_in`
/// nor `k_out`), exactly as the bitonic variant cannot gate its `k ≤ 1024`
/// precondition at compile time. Both variants are `BitIdentical` measured
/// variants: per the ship-top-K policy each ships under the same
/// `accept.structure_key`, the `launch_note` documents the advantage regime, and
/// the bench/dispatch layer (which sees concrete extents) selects the measured
/// default. Offering `_psel` for every TopK cell it can serve is therefore both
/// correct (BitIdentical everywhere, `k_out ≤ k_in` a caller precondition) and
/// consistent with the extent-free key.
///
/// **Contract (`launch_note`):** one block per row, grid-stride over rows,
/// blockDim a multiple of 32 and `≤ 1024`; **dynamic smem = 2·next_pow2(k_out)·
/// (acc_sz + 4) bytes** (bounded on `k_out`, NOT `k_in` — the streaming win; NO
/// `k_in ≤ 1024` cap). `k_out ≤ k_in` is a caller/on-device precondition.
fn partial_select_topk_variant(plan: &KernelPlan<'_>) -> Option<Variant> {
    let Schedule::RowSort { limit, out, .. } = plan.schedule else {
        return None;
    };
    // TopK only — the streaming best-buffer has no whole-row Full form.
    if !matches!(limit, crate::ir::SortLimit::TopK) {
        return None;
    }
    // out ∈ {Values, Indices, Both} — all served (index rides in the pair). (The
    // match is exhaustive; kept explicit as documentation of the served set.)
    match out {
        SortOut::Values | SortOut::Indices | SortOut::Both => {}
    }
    // A Param body would emit an undefined identifier (mirror the bitonic / house
    // style). RowSort's body is pinned Input(0), so this is always empty.
    if !params_used(plan.body).is_empty() {
        return None;
    }
    let ctype = scalar_ctype(plan.dtype)?;
    // Re-assert in0 + out layout (the exact bitonic re-check; validate_row_sort
    // already gated it — belt-and-suspenders).
    let last = plan.key.rank.saturating_sub(1);
    let in0 = plan.key.operands[0];
    let out_op = plan.key.operands[plan.key.n_operands.saturating_sub(1) as usize];
    if rr_role(in0, last) != RrRole::RowStreamed
        || in0.contig != Contiguity::Contig
        || in0.flipped
        || !out_op.bcast.is_empty()
        || out_op.contig != Contiguity::Contig
        || out_op.flipped
    {
        return None;
    }
    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let acc_sz = if dbl || matches!(plan.dtype, ElementKind::I64) {
        8
    } else {
        4
    };
    let k = emit_row_sort_partial(plan, ctype);
    let launch_note = format!(
        "partial-select streaming tiled-bitonic top-k (one block per row, \
         grid-stride over rows; a running best-buffer of m = next_pow2(k_out) \
         (key, index) pairs streamed against the input row in tiles of m, each \
         tile fully bitonic-sorting the 2m window [B | T] under the (key, \
         original-index) total order and keeping the lower m): <<<min(n_out, \
         maxblocks), B, smem>>> with B a multiple of 32 and <= 1024, and dynamic \
         shared memory = 2 * next_pow2(k_out) * {} bytes (sizeof(acc)={} + \
         sizeof(int)=4) — bounded on k_out, NOT k_in, so there is NO k_in <= 1024 \
         cap (the streaming win). O(k_in*log^2(k_out)) — the unique fast path for \
         k_in > 1024 large-vocab top-k, where the bitonic declines and the rank \
         base is O(k_in^2). k_out <= k_in is a caller/on-device precondition; the \
         advantage regime is k_out << k_in (roughly 2*next_pow2(k_out) < \
         next_pow2(k_in)) — the bench/dispatch layer selects the measured default. \
         BitIdentical to the full-sort top-k_out (composition lemma + never-evict). \
         Determinism: {}.",
        acc_sz + 4,
        acc_sz,
        VariantFidelity::BitIdentical.determinism_str()
    );
    Some(Variant {
        tag: "psel",
        kernels: vec![k],
        fidelity: VariantFidelity::BitIdentical,
        launch_note,
    })
}

/// `true` if every iteration axis of `o` is a broadcast axis — its offset is
/// loop-invariant, so the load hoists out of the loop.
fn is_fully_broadcast(o: OperandKey, rank: usize) -> bool {
    rank > 0 && (0..rank).all(|d| o.bcast.is_set(d as u8))
}

/// Element-offset expression for an operand, dropping the terms for broadcast
/// axes (whose stride is known 0 at compile time).
///
/// `perm` (item 01) is the [`crate::ir::View::Permute`] this input is read
/// through: iteration axis `d` reads the producer stride at `perm[d]`, so the
/// term becomes `c{d}·stride[perm[d]]` (a transposed read — the §1 fused-view
/// win, no materialized contiguize copy). `None` (Identity / Broadcast /
/// same-rank Reshape / view-free) keeps `c{d}·stride[d]`, so the emitted string
/// is **byte-identical** to the pre-item-01 emitter for every existing op. A
/// permuted operand always has an empty broadcast mask (the plan gate
/// `assert_valid_views` + the `assert_views_lowerable` backstop pin it), so the
/// broadcast-skip below never interacts with the remap.
fn offset_expr(o: OperandKey, stride_arr: &str, rank: usize, perm: Option<&[u8]>) -> String {
    let mut terms = Vec::new();
    for d in 0..rank {
        if o.bcast.is_set(d as u8) {
            continue;
        }
        // Permute view: iteration axis `d` pairs with the producer stride at
        // `perm[d]`; Identity (perm None) pairs axis `d` with stride `d`.
        let si = perm.map_or(d, |p| p[d] as usize);
        // By-value scalar param spelling (extraction #1): `s0_1`, `so_0`, …
        terms.push(format!("c{d}*{stride_arr}_{si}"));
    }
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join(" + ")
    }
}

/// Emit the row-major (last-axis-fastest) unravel decomposition — the ONE place
/// the "linear index → per-axis coordinate" walk is spelled. For each axis in
/// `axes` (passed high→low: `(0..rank).rev()` for a dense unravel, or the kept
/// axes reversed for a reduction) it declares `long long {coord}` from
/// `{lin} % shape` and divides `{lin}` down (or the `s == 0`-guarded form when
/// `guard`).
///
/// This is the single source of truth shared by every inline strided emitter
/// (`emit_strided` / scatter / strided-multi / reduction — scalar-param,
/// unguarded calls, each **byte-identical** to the pre-factoring emission) and
/// the freestanding coord-unravel helper ([`emit_coord_unravel_helper`] — an
/// array-element, guarded call). An improvement to the decomposition lands once
/// and propagates to every generated kernel and the generated helper.
///
/// `coord(a)` names the coordinate variable for axis `a` (`c{a}` / `oc{a}` /
/// `ck{a}`); `shape(a)` spells its extent (scalar `shape{a}` inline; array
/// element `shape.v[{a}]` in the helper). `guard` emits the empty-axis
/// (`s == 0`) defense: the inline emitters omit it because an empty tensor
/// yields zero loop iterations (the modulo/divide never runs), while a generic
/// helper is called with arbitrary shapes and must defend. `lin` is the mutable
/// linear-index variable already in scope; `indent` is the leading whitespace.
fn emit_unravel_decomp(
    axes: impl IntoIterator<Item = usize>,
    indent: &str,
    lin: &str,
    coord: impl Fn(usize) -> String,
    shape: impl Fn(usize) -> String,
    guard: bool,
) -> String {
    let mut s = String::new();
    for a in axes {
        let cv = coord(a);
        let sh = shape(a);
        if guard {
            s.push_str(&format!(
                "{indent}long long {cv} = ({sh} == 0) ? 0 : ({lin} % (long long){sh}); if ({sh} != 0) {lin} /= (long long){sh};\n"
            ));
        } else {
            s.push_str(&format!(
                "{indent}long long {cv} = {lin} % {sh}; {lin} /= {sh};\n"
            ));
        }
    }
    s
}

/// Emit one `unravel_offset{s}_r{n}` device function: `k` stride arrays dotted
/// against a single unrolled rank-`n` unravel (one shared decomposition pass).
/// `k == 1` returns the offset (`unravel_offset_1_r{n}`); `k >= 2` is `void` with
/// `k` trailing `int64_t&` out-params (`unravel_offsets_{k}_r{n}`), matching the
/// hand-written `unravel_offsets_2/3/4`. The decomposition comes from the shared
/// [`emit_unravel_decomp`].
fn emit_unravel_offset_fn(s: &mut String, n: usize, k: usize) {
    const LETTERS: [&str; 4] = ["a", "b", "c", "d"];
    let ls = &LETTERS[..k];
    s.push_str("template <typename Shape, typename Stride>\n");
    if k == 1 {
        s.push_str(&format!(
            "__device__ __forceinline__ int64_t unravel_offset_1_r{n}(\n"
        ));
        s.push_str("    int64_t linear, const Shape& shape, const Stride& stride)\n{\n");
    } else {
        s.push_str(&format!(
            "__device__ __forceinline__ void unravel_offsets_{k}_r{n}(\n"
        ));
        s.push_str("    int64_t linear, const Shape& shape");
        for l in ls {
            s.push_str(&format!(", const Stride& stride_{l}"));
        }
        for l in ls {
            s.push_str(&format!(", int64_t& off_{l}"));
        }
        s.push_str(")\n{\n");
    }
    // Shared decomposition: guarded, array-element shape spelling, unrolled.
    s.push_str(&emit_unravel_decomp(
        (0..n).rev(),
        "    ",
        "linear",
        |d| format!("c{d}"),
        |d| format!("shape.v[{d}]"),
        true,
    ));
    if k == 1 {
        s.push_str("    int64_t off = 0;\n");
        for d in 0..n {
            s.push_str(&format!("    off += c{d} * (int64_t)stride.v[{d}];\n"));
        }
        s.push_str("    return off;\n}\n\n");
    } else {
        for l in ls {
            s.push_str(&format!("    off_{l} = 0;\n"));
        }
        for d in 0..n {
            for l in ls {
                s.push_str(&format!(
                    "    off_{l} += c{d} * (int64_t)stride_{l}.v[{d}];\n"
                ));
            }
        }
        s.push_str("}\n\n");
    }
}

/// Emit a freestanding, header-only coordinate-unravel helper (`.cuh`) — the
/// **generated** counterpart of the hand-written
/// `baracuda_coord_unravel.cuh`. For each rank in `1..=max_rank` it emits the
/// `unravel_offset_1_r{N}` (returns the offset) plus the multi-stride
/// `unravel_offsets_2/3/4_r{N}` (void, out-params) a hand-written CUDA kernel can
/// `#include` and call.
///
/// Unlike the hand-written helper's single runtime-`rank` loop, each generated
/// function is **specialized to a compile-time rank and fully unrolled** — the
/// generator's win: nvcc drops the loop and the runtime-rank branch. The
/// arithmetic is bit-identical to the hand-written helper (same row-major walk,
/// same `s == 0` guard, same signed-stride and stride-0-broadcast behavior), so
/// the two are drop-in-equivalent per rank. The decomposition is emitted by the
/// shared [`emit_unravel_decomp`] — the same routine the inline strided kernels
/// use — which is the single-source-of-truth point that makes generator
/// improvements propagate into the helper.
///
/// Templated on `Shape`/`Stride` (any layout-compatible POD exposing `.v[d]`),
/// matching the hand-written helper's convention. As there, `base_offset` stays
/// the caller's responsibility (this returns only the coordinate·stride dot).
pub fn emit_coord_unravel_helper(max_rank: usize) -> String {
    let mut s = String::new();
    s.push_str("// @generated by baracuda-kernelgen — do not edit.\n");
    s.push_str("// Freestanding coordinate-unravel helper: per-rank UNROLLED counterparts\n");
    s.push_str("// of baracuda_coord_unravel.cuh's runtime-rank `unravel_offset_1`. Offsets\n");
    s.push_str("// are bit-identical; nvcc unrolls the compile-time rank. `base_offset` is\n");
    s.push_str("// the caller's responsibility, exactly as in the hand-written helper.\n");
    s.push_str("#ifndef BARACUDA_COORD_UNRAVEL_GEN_CUH\n");
    s.push_str("#define BARACUDA_COORD_UNRAVEL_GEN_CUH\n");
    s.push_str("#include <cstdint>\n");
    s.push_str("#include <cuda_runtime.h>\n\n");
    s.push_str("namespace baracuda { namespace coord { namespace gen {\n\n");
    for n in 1..=max_rank {
        // `_1` (returns the offset) + `_2`/`_3`/`_4` (void, out-params — one
        // unravel pass feeds K stride arrays) per rank, matching the hand-written
        // `unravel_offset_1` / `unravel_offsets_2/3/4`.
        for k in 1..=4 {
            emit_unravel_offset_fn(&mut s, n, k);
        }
    }
    s.push_str("}}}  // namespace baracuda::coord::gen\n");
    s.push_str("#endif  // BARACUDA_COORD_UNRAVEL_GEN_CUH\n");
    s
}

/// The f16/bf16 → f32 widening intrinsic (`__half2float` / `__bfloat162float`),
/// or `None` for a dtype loaded without one. The ONE place the generator names
/// the half→float promotion — shared by the inline load sites and the generated
/// dtype-promote helper ([`emit_dtype_promote_helper`]).
fn half_load_intrinsic(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::F16 => Some("__half2float"),
        ElementKind::Bf16 => Some("__bfloat162float"),
        _ => None,
    }
}

/// The f32 → f16/bf16 narrowing intrinsic (`__float2half` / `__float2bfloat16`),
/// or `None` for a dtype stored without one. Counterpart of
/// [`half_load_intrinsic`].
fn half_store_intrinsic(kind: ElementKind) -> Option<&'static str> {
    match kind {
        ElementKind::F16 => Some("__float2half"),
        ElementKind::Bf16 => Some("__float2bfloat16"),
        _ => None,
    }
}

/// Widen a loaded `inner` expression to `float`: the half/bf16 intrinsic, else
/// the value unchanged (already ≥ f32, or an integer loaded natively).
fn promote_load_f32(kind: ElementKind, inner: &str) -> String {
    match half_load_intrinsic(kind) {
        Some(f) => format!("{f}({inner})"),
        None => inner.to_string(),
    }
}

/// Narrow a `float`-valued `inner` expression to the storage dtype: the
/// half/bf16 intrinsic, else the value unchanged (the caller adds any cast).
fn demote_store_f32(kind: ElementKind, inner: &str) -> String {
    match half_store_intrinsic(kind) {
        Some(f) => format!("{f}({inner})"),
        None => inner.to_string(),
    }
}

/// A single element-wise dtype-cast expression — the value of `expr` (of dtype
/// `from`) converted to dtype `to`, with the house f16/bf16 float-detour
/// convention. The ONE place the generator spells a per-element conversion
/// between two scalar dtypes, shared by the inline hetero store
/// ([`store_expr_of`]) and the generated cast helper ([`emit_cast_helper`]), so
/// the two can never drift.
///
/// Mirrors `baracuda_cast.cuh`'s `cast_value<TIn, TOut>` (value-identical, not
/// necessarily text-identical — the generated form uses C-style casts and the
/// shared [`promote_load_f32`] / [`demote_store_f32`] intrinsic picks):
///   * `from == to` → identity (no cast).
///   * f16/bf16 → f16/bf16 (cross) → widen to `float`, then narrow.
///   * f16/bf16 → arithmetic → widen to `float`, then a C-style cast to the target.
///   * arithmetic → f16/bf16 → cast to `float` (unless already `float`), then narrow.
///   * arithmetic → arithmetic → a plain C-style cast.
fn cast_scalar(from: ElementKind, to: ElementKind, expr: &str) -> String {
    if from == to {
        return expr.to_string();
    }
    match (
        half_load_intrinsic(from).is_some(),
        half_store_intrinsic(to).is_some(),
    ) {
        // f16/bf16 -> f16/bf16 (cross): widen to f32, then narrow.
        (true, true) => demote_store_f32(to, &promote_load_f32(from, expr)),
        // f16/bf16 -> arithmetic: widen to f32, C-cast to the target.
        (true, false) => {
            let oct = scalar_ctype(to).expect("cast target dtype has a scalar ctype");
            format!("({oct}){}", promote_load_f32(from, expr))
        }
        // arithmetic -> f16/bf16: cast to float (unless already float), then narrow.
        (false, true) => {
            let widened = if matches!(from, ElementKind::F32 | ElementKind::F32Strict) {
                expr.to_string()
            } else {
                format!("(float){expr}")
            };
            demote_store_f32(to, &widened)
        }
        // arithmetic -> arithmetic: plain C-style cast.
        (false, false) => {
            let oct = scalar_ctype(to).expect("cast target dtype has a scalar ctype");
            format!("({oct}){expr}")
        }
    }
}

/// Emit a freestanding, header-only dtype-promotion helper (`.cuh`) — the
/// **generated** counterpart of the hand-written `baracuda_dtype_promote.cuh`.
/// Emits the f16/bf16 `load_as_f32` / `store_from_f32` (+ the f64 lane) widen /
/// narrow device functions a hand-written CUDA kernel can `#include` and call.
///
/// Unlike coord-unravel there is no compile-time-specialization perf win here
/// (both the hand-written template specializations and the generator's
/// per-`ElementKind` pick resolve to the same single hardware intrinsic at
/// compile time — the emitted code is bit-identical). The value is
/// **de-duplication / one source of truth**: the intrinsic pick comes from the
/// same [`promote_load_f32`] / [`demote_store_f32`] routines the inline emitters
/// use, so it can never drift from them.
pub fn emit_dtype_promote_helper() -> String {
    let mut s = String::new();
    s.push_str("// @generated by baracuda-kernelgen — do not edit.\n");
    s.push_str("// Freestanding dtype-promotion helper: the f16/bf16 <-> f32/f64 widen/narrow\n");
    s.push_str("// intrinsics the generator uses inline, as reusable device functions — one\n");
    s.push_str("// source of truth with the inline emitters.\n");
    s.push_str("#ifndef BARACUDA_DTYPE_PROMOTE_GEN_CUH\n");
    s.push_str("#define BARACUDA_DTYPE_PROMOTE_GEN_CUH\n");
    s.push_str("#include <cuda_fp16.h>\n");
    s.push_str("#include <cuda_bf16.h>\n\n");
    s.push_str("namespace baracuda { namespace dtype { namespace gen {\n\n");
    for (kind, tag, cty) in [
        (ElementKind::F16, "f16", "__half"),
        (ElementKind::Bf16, "bf16", "__nv_bfloat16"),
    ] {
        // f32 lane.
        s.push_str(&format!(
            "__device__ __forceinline__ float load_as_f32_{tag}({cty} x) {{ return {}; }}\n",
            promote_load_f32(kind, "x")
        ));
        s.push_str(&format!(
            "__device__ __forceinline__ {cty} store_from_f32_{tag}(float v) {{ return {}; }}\n",
            demote_store_f32(kind, "v")
        ));
        // f64 lane — round-trips through f32 (no direct double↔half intrinsic;
        // the extra f32 hop is lossless relative to the half mantissa), matching
        // the hand-written helper.
        s.push_str(&format!(
            "__device__ __forceinline__ double load_as_f64_{tag}({cty} x) {{ return (double){}; }}\n",
            promote_load_f32(kind, "x")
        ));
        s.push_str(&format!(
            "__device__ __forceinline__ {cty} store_from_f64_{tag}(double v) {{ return {}; }}\n\n",
            demote_store_f32(kind, "(float)v")
        ));
    }
    s.push_str("}}}  // namespace baracuda::dtype::gen\n");
    s.push_str("#endif  // BARACUDA_DTYPE_PROMOTE_GEN_CUH\n");
    s
}

/// The scalar compute dtypes the generated cast matrix ([`emit_cast_helper`])
/// covers — one entry per distinct storage type. Mirrors `baracuda_cast.cuh`'s
/// endpoint set. `F32Strict` is intentionally absent (identical `float` ctype to
/// `F32`; it is a compute-mode distinction, not a distinct storage type) and
/// `U32` is index-only (no compute-operand cast), so both are excluded to avoid
/// duplicate / meaningless symbols.
const CAST_MATRIX_DTYPES: [ElementKind; 8] = [
    ElementKind::F16,
    ElementKind::Bf16,
    ElementKind::F32,
    ElementKind::F64,
    ElementKind::I32,
    ElementKind::I64,
    ElementKind::S8,
    ElementKind::U8,
];

/// Emit a freestanding, header-only elementwise dtype-cast helper (`.cuh`) — the
/// **generated** counterpart of the hand-written `baracuda_cast.cuh`. Emits the
/// full `N×N` matrix of `baracuda::cast::gen::cast_<sin>_<sout>` device functions
/// (`y = (TOut) x`) a hand-written CUDA kernel can `#include` and call.
///
/// Like dtype-promote (and unlike coord-unravel) this is a **de-duplication /
/// one-source-of-truth** win, not a speed win: each generated cast resolves to
/// the same `static_cast` / half-intrinsic the hand-written `cast_value<TIn,
/// TOut>` does, so the emitted code is value-identical (proven exhaustively by
/// `ondevice/cast_validate.cu`). The per-element conversion is spelled by the
/// same [`cast_scalar`] routine the inline hetero store ([`store_expr_of`]) uses
/// — which in turn routes half endpoints through the same [`promote_load_f32`] /
/// [`demote_store_f32`] the reduction load/store use — so the generated cast can
/// never drift from the live emitters. `cast` is a strict superset of
/// dtype-promote: it adds the integer endpoints, the cross-half pair, and the
/// float↔int conversions.
pub fn emit_cast_helper() -> String {
    let mut s = String::new();
    s.push_str("// @generated by baracuda-kernelgen — do not edit.\n");
    s.push_str("// Freestanding elementwise dtype-cast helper: the per-element (TOut)x\n");
    s.push_str("// conversion the generator uses inline (cast_scalar), as a reusable N×N\n");
    s.push_str("// matrix of device functions — one source of truth with the inline emitters.\n");
    s.push_str("#ifndef BARACUDA_CAST_GEN_CUH\n");
    s.push_str("#define BARACUDA_CAST_GEN_CUH\n");
    s.push_str("#include <cstdint>\n");
    s.push_str("#include <cuda_fp16.h>\n");
    s.push_str("#include <cuda_bf16.h>\n\n");
    s.push_str("namespace baracuda { namespace cast { namespace gen {\n\n");
    for from in CAST_MATRIX_DTYPES {
        let fcty = scalar_ctype(from).expect("cast source dtype has a scalar ctype");
        let ftag = dtype_tag(from);
        for to in CAST_MATRIX_DTYPES {
            let tcty = scalar_ctype(to).expect("cast target dtype has a scalar ctype");
            let ttag = dtype_tag(to);
            s.push_str(&format!(
                "__device__ __forceinline__ {tcty} cast_{ftag}_{ttag}({fcty} x) {{ return {}; }}\n",
                cast_scalar(from, to, "x")
            ));
        }
        s.push('\n');
    }
    s.push_str("}}}  // namespace baracuda::cast::gen\n");
    s.push_str("#endif  // BARACUDA_CAST_GEN_CUH\n");
    s
}

/// [`offset_expr`] with an arbitrary coordinate-variable prefix (identity layout,
/// no permutation) — the gather-sum kernel (increment 5) has TWO coordinate
/// spaces in one kernel (the destination `oc{d}` and the update-domain `uc{d}`),
/// so it can't use the fixed `c{d}` of [`offset_expr`]. Drops broadcast-axis terms
/// exactly as [`offset_expr`] does (a 1-D index_add index broadcasts its non-axis
/// terms to stride 0, degenerating to `{coord}{axis}·stride`).
fn offset_expr_coord(o: OperandKey, stride_arr: &str, coord: &str, rank: usize) -> String {
    let mut terms = Vec::new();
    for d in 0..rank {
        if o.bcast.is_set(d as u8) {
            continue;
        }
        terms.push(format!("{coord}{d}*{stride_arr}_{d}"));
    }
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join(" + ")
    }
}

/// The permutation input operand `k` is read through, or `None` for an identity
/// read (`Identity` / `Broadcast` / same-rank `Reshape` / a view-free plan).
/// Only a [`crate::ir::View::Permute`] remaps stride indices in [`offset_expr`];
/// `plan.views` is empty for every pre-item-01 op, so this is always `None` there
/// (`.get(k)` on the empty slice) and emission is byte-identical.
fn input_perm<'a>(plan: &'a KernelPlan<'_>, k: usize) -> Option<&'a [u8]> {
    match plan.views.get(k) {
        Some(crate::ir::View::Permute { perm }) => Some(perm.as_slice()),
        _ => None,
    }
}

/// Element-offset expression for a GATHERED operand (increment 4): identical to
/// [`offset_expr`] with an IDENTITY layout (gather ⊥ view), EXCEPT the gathered
/// `axis` term is `({idx_var})·stride[axis]` — the runtime index value replaces
/// the loop coordinate `c{axis}`. The gathered axis is never a broadcast axis
/// (the plan gate pins a live stride), so its term is always present; the other
/// axes drop their term when broadcast, exactly as [`offset_expr`].
fn gathered_offset_expr(
    o: OperandKey,
    stride_arr: &str,
    rank: usize,
    axis: usize,
    idx_var: &str,
) -> String {
    let mut terms = Vec::new();
    for d in 0..rank {
        if d == axis {
            // The data-dependent term: the runtime index, not the coordinate.
            terms.push(format!("({idx_var})*{stride_arr}_{d}"));
        } else if o.bcast.is_set(d as u8) {
            continue;
        } else {
            terms.push(format!("c{d}*{stride_arr}_{d}"));
        }
    }
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join(" + ")
    }
}

/// The properly-typed zero literal for the output ctype — the [`OobPolicy::ZeroFill`]
/// fill (increment 4). f16/bf16 need the intrinsic constructor (no portable
/// `T(0)`); everything else takes a plain literal.
fn zero_store_literal(octype: &str) -> &'static str {
    match octype {
        "__half" => "__float2half(0.0f)",
        "__nv_bfloat16" => "__float2bfloat16(0.0f)",
        "float" => "0.0f",
        "double" => "0.0",
        // int / long long / signed char / unsigned char
        _ => "0",
    }
}

/// Spell a [`UnaryOp`] applied to an already-lowered f32 inner expression.
/// Inner strings are atomic or parenthesized, so the function-call forms need no
/// extra wrapping; the operator forms wrap themselves. (`Sigmoid`/`Gelu`/`Silu`
/// reference the inner twice — fine for an atomic load; a temp-binding pass to
/// avoid recompute on compound inners is a follow-up.)
pub(crate) fn unary_f32(op: UnaryOp, x: String) -> String {
    match op {
        UnaryOp::Neg => format!("(-{x})"),
        UnaryOp::Abs => format!("fabsf({x})"),
        UnaryOp::Sqr => format!("({x}*{x})"),
        UnaryOp::Sqrt => format!("sqrtf({x})"),
        UnaryOp::Rsqrt => format!("rsqrtf({x})"),
        UnaryOp::Recip => format!("(1.0f/{x})"),
        UnaryOp::Exp => format!("expf({x})"),
        UnaryOp::Log => format!("logf({x})"),
        UnaryOp::Tanh => format!("tanhf({x})"),
        UnaryOp::Sigmoid => format!("(1.0f/(1.0f+expf(-{x})))"),
        // NaN-propagating: `NaN < 0` is false, so NaN passes through (matches
        // PyTorch). `fmaxf(x,0)` would scrub NaN to 0. (Inner duplicated — the
        // temp-binding pass that fixes recompute is a follow-up.)
        UnaryOp::Relu => format!("({x} < 0.0f ? 0.0f : {x})"),
        UnaryOp::Erf => format!("erff({x})"),
        UnaryOp::Gelu => format!("(0.5f*{x}*(1.0f+erff({x}*0.70710678f)))"),
        UnaryOp::Silu => format!("({x}*(1.0f/(1.0f+expf(-{x}))))"),
        UnaryOp::Sin => format!("sinf({x})"),
        UnaryOp::Cos => format!("cosf({x})"),
        UnaryOp::Floor => format!("floorf({x})"),
        UnaryOp::Ceil => format!("ceilf({x})"),
        UnaryOp::Round => format!("rintf({x})"), // ties to even
        UnaryOp::Sign => format!("({x} > 0.0f ? 1.0f : ({x} < 0.0f ? -1.0f : 0.0f))"),
        UnaryOp::Step => format!("({x} > 0.0f ? 1.0f : 0.0f)"), // heaviside(x, 0): step(0)=0
        // increment-0a scalar fns — all implicit CUDA device math (headerless
        // under nvrtc, same class as expf; no includes).
        UnaryOp::Erfc => format!("erfcf({x})"),
        UnaryOp::Trunc => format!("truncf({x})"),
        UnaryOp::Exp2 => format!("exp2f({x})"),
        UnaryOp::Expm1 => format!("expm1f({x})"),
        UnaryOp::Log2 => format!("log2f({x})"),
        UnaryOp::Log10 => format!("log10f({x})"),
        UnaryOp::Log1p => format!("log1pf({x})"),
        UnaryOp::Sinh => format!("sinhf({x})"),
        UnaryOp::Cosh => format!("coshf({x})"),
        UnaryOp::Tan => format!("tanf({x})"),
        UnaryOp::Asin => format!("asinf({x})"),
        UnaryOp::Acos => format!("acosf({x})"),
        UnaryOp::Atan => format!("atanf({x})"),
        UnaryOp::Asinh => format!("asinhf({x})"),
        UnaryOp::Acosh => format!("acoshf({x})"),
        UnaryOp::Atanh => format!("atanhf({x})"),
        UnaryOp::Cbrt => format!("cbrtf({x})"),
        UnaryOp::Lgamma => format!("lgammaf({x})"),
    }
}

/// Same as [`unary_f32`] but with f64 math-function names and double literals.
pub(crate) fn unary_f64(op: UnaryOp, x: String) -> String {
    match op {
        UnaryOp::Neg => format!("(-{x})"),
        UnaryOp::Abs => format!("fabs({x})"),
        UnaryOp::Sqr => format!("({x}*{x})"),
        UnaryOp::Sqrt => format!("sqrt({x})"),
        UnaryOp::Rsqrt => format!("rsqrt({x})"),
        UnaryOp::Recip => format!("(1.0/{x})"),
        UnaryOp::Exp => format!("exp({x})"),
        UnaryOp::Log => format!("log({x})"),
        UnaryOp::Tanh => format!("tanh({x})"),
        UnaryOp::Sigmoid => format!("(1.0/(1.0+exp(-{x})))"),
        UnaryOp::Relu => format!("({x} < 0.0 ? 0.0 : {x})"),
        UnaryOp::Erf => format!("erf({x})"),
        UnaryOp::Gelu => format!("(0.5*{x}*(1.0+erf({x}*0.7071067811865476)))"),
        UnaryOp::Silu => format!("({x}*(1.0/(1.0+exp(-{x}))))"),
        UnaryOp::Sin => format!("sin({x})"),
        UnaryOp::Cos => format!("cos({x})"),
        UnaryOp::Floor => format!("floor({x})"),
        UnaryOp::Ceil => format!("ceil({x})"),
        UnaryOp::Round => format!("rint({x})"), // ties to even
        UnaryOp::Sign => format!("({x} > 0.0 ? 1.0 : ({x} < 0.0 ? -1.0 : 0.0))"),
        UnaryOp::Step => format!("({x} > 0.0 ? 1.0 : 0.0)"), // heaviside(x, 0): step(0)=0
        // increment-0a scalar fns — the double variants of the f32 spellings.
        UnaryOp::Erfc => format!("erfc({x})"),
        UnaryOp::Trunc => format!("trunc({x})"),
        UnaryOp::Exp2 => format!("exp2({x})"),
        UnaryOp::Expm1 => format!("expm1({x})"),
        UnaryOp::Log2 => format!("log2({x})"),
        UnaryOp::Log10 => format!("log10({x})"),
        UnaryOp::Log1p => format!("log1p({x})"),
        UnaryOp::Sinh => format!("sinh({x})"),
        UnaryOp::Cosh => format!("cosh({x})"),
        UnaryOp::Tan => format!("tan({x})"),
        UnaryOp::Asin => format!("asin({x})"),
        UnaryOp::Acos => format!("acos({x})"),
        UnaryOp::Atan => format!("atan({x})"),
        UnaryOp::Asinh => format!("asinh({x})"),
        UnaryOp::Acosh => format!("acosh({x})"),
        UnaryOp::Atanh => format!("atanh({x})"),
        UnaryOp::Cbrt => format!("cbrt({x})"),
        UnaryOp::Lgamma => format!("lgamma({x})"),
    }
}

/// Lower a unary op for `dtype`. f32/f64 use native math; f16/bf16 compute in
/// float (convert → f32 math → convert) — correct for every op, and it avoids
/// the incomplete `__half2` math-intrinsic set (no `h2tanh`/`h2erf`). Packed
/// `__half2` SIMD is a perf follow-up. Integer dtypes have no unary math.
fn cuda_unary(op: UnaryOp, x: String, dtype: ElementKind) -> String {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict => unary_f32(op, x),
        ElementKind::F64 => unary_f64(op, x),
        ElementKind::F16 => {
            format!(
                "__float2half({})",
                unary_f32(op, format!("__half2float({x})"))
            )
        }
        ElementKind::Bf16 => {
            format!(
                "__float2bfloat16({})",
                unary_f32(op, format!("__bfloat162float({x})"))
            )
        }
        other => panic!("cuda backend: no unary math for dtype {other:?}"),
    }
}

/// Non-infix binary op in f32 math.
///
/// `Maximum`/`Minimum` are **NaN-propagating** (a NaN operand ⇒ NaN out) —
/// matching `torch.maximum`/`minimum` and the house reference kernel
/// `binary_maximum_fp.cu`, which deliberately reserves `fmaxf`/`fminf` (IEEE
/// `maxNum`, NaN-*suppressing*) for a *separate* op. That separate op now exists
/// as [`BinaryOp::FmaxIeee`]/[`BinaryOp::FminIeee`] below — so `Max`/`Min` emit
/// the compare-select, never `fmaxf`. (Operands appear 3× — the deferred
/// temp-binding pass, cf. relu/sigmoid, removes the recompute on compound inners.)
pub(crate) fn binary_f32(op: BinaryOp, a: String, b: String) -> String {
    match op {
        // A ON TIES (`>=`/`<=`): the KISS-Ops `max_prop`/`min_prop` normative
        // decomposition (`cmp_ge`/`cmp_le` select a) and numpy/torch
        // `where(a >= b, a, b)`. Bit-visible only on signed-zero ties
        // (`max_prop(-0.0, +0.0) = -0.0`); a `>`-spelled tie would return b.
        BinaryOp::Max => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} >= {b} ? {a} : {b})))")
        }
        BinaryOp::Min => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} <= {b} ? {a} : {b})))")
        }
        BinaryOp::Pow => format!("powf({a}, {b})"),
        // Floored remainder (torch.remainder, sign-of-divisor — Fuel's Op::Rem),
        // not C fmodf (sign-of-dividend). Operands appear twice (temp-binding TODO).
        BinaryOp::Rem => format!("({a} - floorf({a} / {b}) * {b})"),
        // increment-0a scalar fns. FmaxIeee/FminIeee are the deliberate
        // NaN-SUPPRESSING fmaxf/fminf — the separate op the house reserves them
        // for; Max/Min above stay the NaN-propagating compare-selects. RemTrunc
        // is C fmodf (sign-of-dividend) — the truncated sibling of Rem above.
        BinaryOp::Atan2 => format!("atan2f({a}, {b})"),
        BinaryOp::Copysign => format!("copysignf({a}, {b})"),
        BinaryOp::Nextafter => format!("nextafterf({a}, {b})"),
        BinaryOp::FmaxIeee => format!("fmaxf({a}, {b})"),
        BinaryOp::FminIeee => format!("fminf({a}, {b})"),
        BinaryOp::RemTrunc => format!("fmodf({a}, {b})"),
        // increment-0b comparison predicates: the C operators, with BOTH
        // operands cast to float so the compare is decided IN THE COMPUTE
        // DTYPE. Without the casts, a `Const` operand (spelled as a
        // suffix-less double literal) promotes the float side to double and
        // the compare is decided against the UNROUNDED constant — e.g.
        // `in0[i] == 0.1` is false at every x including 0.1f, while the
        // compute-dtype compare (and torch scalar promotion, and this
        // emitter's own f16 path, which rounds the constant to half first)
        // says true. The cast is a no-op for already-float operands;
        // arithmetic ops keep the double-then-round-once convention
        // (correctly rounded THROUGH the store — compares have no rounding
        // step, so they must round operands first instead). NaN semantics
        // are the C operators' (any comparison with NaN is false EXCEPT
        // `!=`, which is true); the value is EXACTLY 1.0f or 0.0f.
        BinaryOp::CmpEq => format!("((float){a} == (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpNe => format!("((float){a} != (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpLt => format!("((float){a} < (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpLe => format!("((float){a} <= (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpGt => format!("((float){a} > (float){b} ? 1.0f : 0.0f)"),
        BinaryOp::CmpGe => format!("((float){a} >= (float){b} ? 1.0f : 0.0f)"),
        // increment-0c INT-ONLY ops: an independent emitter backstop behind
        // the plan gate (assert_int_op_admissibility) — a bitwise/logical op
        // must never reach a float speller, including the f16/bf16 promote
        // path and the reduction-class accumulator lowerings, which all route
        // through here.
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => {
            panic!("cuda backend: {op:?} is int-only (I32/I64/S8/U8) — it has no f32 lowering")
        }
    }
}

/// Same as [`binary_f32`] but with f64 math-function names.
pub(crate) fn binary_f64(op: BinaryOp, a: String, b: String) -> String {
    match op {
        // A ON TIES (`>=`/`<=`) — see [`binary_f32`]'s Max/Min note.
        BinaryOp::Max => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} >= {b} ? {a} : {b})))")
        }
        BinaryOp::Min => {
            format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} <= {b} ? {a} : {b})))")
        }
        BinaryOp::Pow => format!("pow({a}, {b})"),
        BinaryOp::Rem => format!("({a} - floor({a} / {b}) * {b})"),
        BinaryOp::Atan2 => format!("atan2({a}, {b})"),
        BinaryOp::Copysign => format!("copysign({a}, {b})"),
        BinaryOp::Nextafter => format!("nextafter({a}, {b})"),
        BinaryOp::FmaxIeee => format!("fmax({a}, {b})"),
        BinaryOp::FminIeee => format!("fmin({a}, {b})"),
        BinaryOp::RemTrunc => format!("fmod({a}, {b})"),
        // increment-0b comparison predicates — double literals, same C-operator
        // NaN semantics as the f32 arms.
        BinaryOp::CmpEq => format!("({a} == {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpNe => format!("({a} != {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpLt => format!("({a} < {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpLe => format!("({a} <= {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpGt => format!("({a} > {b} ? 1.0 : 0.0)"),
        BinaryOp::CmpGe => format!("({a} >= {b} ? 1.0 : 0.0)"),
        // increment-0c INT-ONLY ops — same backstop as the f32 speller.
        BinaryOp::BitAnd
        | BinaryOp::BitOr
        | BinaryOp::BitXor
        | BinaryOp::Shl
        | BinaryOp::Shr
        | BinaryOp::LogicalAnd
        | BinaryOp::LogicalOr
        | BinaryOp::LogicalXor => {
            panic!("cuda backend: {op:?} is int-only (I32/I64/S8/U8) — it has no f64 lowering")
        }
    }
}

/// Spell an increment-0c INT-ONLY binary op (bitwise/shift/logical) over two
/// already-lowered integer operand strings — the RAW C operators, matching the
/// bespoke functors **exactly** (the 0c charter: express the bespoke
/// functionality, never "improve" it):
///
/// - `BitAnd`/`BitOr`/`BitXor`: `binary_bitwise_{and,or,xor}_int.cu`'s
///   `return a OP b;` — no rounding, no overflow concerns.
/// - `Shl`/`Shr`: `binary_bitwise_{left,right}_shift_int.cu`'s `return a << b;`
///   / `return a >> b;` — NO masking or clamping. Out-of-range amounts
///   (`b < 0` or `b >= 8*sizeof(promoted T)`) inherit the architecture's
///   behavior (the bespoke caller contract, carried verbatim); signed `>>` is
///   arithmetic on every CUDA compiler (PTX `shr.s32`/`shr.s64` — the bespoke
///   kernel's documented reliance), unsigned is logical.
/// - `LogicalAnd`/`LogicalOr`/`LogicalXor`: `binary_logical_*_bool.cu`'s
///   normalize-then-op — `(a != 0 OP b != 0) ? 1 : 0`, so the output is
///   strictly 0/1 even for unnormalized bytes (`2 && 4 == 1`). U8-only (the
///   bespoke Bool surface); the plan gate enforces it and the assert here is
///   the independent emitter backstop.
///
/// **Integer-promotion note (S8/U8):** the operand strings are `signed char`/
/// `unsigned char` loads — GUARANTEED, not assumed: the plan gate's 8-bit
/// composition pin (`plan::assert_int_op_admissibility` rule 3) requires every
/// int-op operand at `S8`/`U8` to be a leaf `Input`, so a composed operand
/// (whose inlined un-truncated value would diverge from its hoisted 8-bit-tmp
/// value under DAG sharing) can never reach this speller. The loads promote to
/// `int` (sign-/zero-extended) before any operator. NO defeating casts are
/// emitted, deliberately:
/// - and/or/xor: promote → op → store-truncate is bit-identical to a native
///   8-bit op (extension bits AND/OR/XOR among themselves and truncate away);
/// - `Shl`: the 32-bit shift result store-truncates mod 2⁸ — equal to a
///   native wrapping 8-bit shift for in-range amounts, and amounts 8..31 take
///   the promoted (well-defined-in-practice) semantics rather than native-8-bit
///   UB. This matches how the bespoke i32/i64 kernels compose with C — there
///   is no bespoke 8-bit shift to defer to, so the promotion semantics ARE the
///   documented contract (see `BinaryOp::Shl`);
/// - `Shr`: the promoted value's high bits are the extension of the 8-bit
///   value, so the shifted result always fits 8 bits — truncation is exact,
///   arithmetic for `signed char`, logical for `unsigned char`;
/// - logical ops: the `!= 0` tests and the 0/1 result are promotion-invariant.
///
/// The final `other` arm is the second half of the emitter backstop: a float
/// fn / cmp op that reaches the int speller (i.e. bypassed the plan gate at an
/// int dtype) panics rather than emitting C that happens to compile.
pub(crate) fn binary_int(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> String {
    if op.is_logical() {
        assert!(
            dtype == ElementKind::U8,
            "cuda backend: {op:?} is U8 (Bool)-only — the bespoke logical surface \
             instantiates exactly uint8_t; got {dtype:?}"
        );
    }
    match op {
        BinaryOp::BitAnd => format!("({a} & {b})"),
        BinaryOp::BitOr => format!("({a} | {b})"),
        BinaryOp::BitXor => format!("({a} ^ {b})"),
        BinaryOp::Shl => format!("({a} << {b})"),
        BinaryOp::Shr => format!("({a} >> {b})"),
        BinaryOp::LogicalAnd => format!("(({a} != 0 && {b} != 0) ? 1 : 0)"),
        BinaryOp::LogicalOr => format!("(({a} != 0 || {b} != 0) ? 1 : 0)"),
        BinaryOp::LogicalXor => format!("((({a} != 0) != ({b} != 0)) ? 1 : 0)"),
        other => panic!(
            "cuda backend: {other:?} has no integer lowering — the bespoke \
             elementwise surface instantiates it for float dtypes only \
             (int dtype {dtype:?} must miss honestly at the plan gate)"
        ),
    }
}

/// Lower a non-infix binary op for `dtype` (f32/f64 native; f16/bf16 compute in
/// float; int dtypes via the [`binary_int`] C-operator speller — bitwise/shift/
/// logical only, everything else backstop-panics there). Mirrors [`cuda_unary`].
///
/// The f16/bf16 promote→f32-fn→demote round trip is value-correct for every op
/// here **except `Nextafter`**, which is therefore refused (the JIT gates it in
/// `dtype_compatible`; reaching this panic means an AOT author declared f16/bf16
/// on a Nextafter body). Why the others are safe: half→f32 is exact, and
/// demoting rounds a correctly-computed f32 value once — for `Copysign`
/// specifically the result's magnitude is the *input's own* (exactly
/// representable) magnitude with the sign bit swapped, and a half NaN payload
/// round-trips half→f32→half bit-exactly (the same guarantee the whole scalar
/// half path leans on, swept by `packed_validate.cu`), so bit-level sign
/// transfer survives the promotion. `Nextafter` does not: the f32 neighbor of a
/// promoted half is ~2¹³ f32 steps closer than the next *half*, so the demote
/// rounds straight back to `a` — a silently wrong no-op, hence the honest miss.
/// The `Cmp*` predicates promote EXACTLY: half→f32 is a lossless,
/// order-preserving embedding (every f16/bf16 value, ±0, ±inf, and NaN maps to
/// the f32 value of the same class and order), so the f32 compare decides
/// identically to a native half compare, and demoting the exact 1.0f/0.0f
/// result is exact — no lattice is stepped, unlike Nextafter.
fn cuda_binary(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> String {
    if matches!(dtype, ElementKind::F16 | ElementKind::Bf16) && matches!(op, BinaryOp::Nextafter) {
        panic!(
            "cuda backend: Nextafter has no half-precision lowering — the \
             promote-to-f32 path would step the f32 lattice, not the {dtype:?} \
             one (declare f32/f64 only, or miss honestly)"
        );
    }
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict => binary_f32(op, a, b),
        ElementKind::F64 => binary_f64(op, a, b),
        ElementKind::F16 => format!(
            "__float2half({})",
            binary_f32(
                op,
                format!("__half2float({a})"),
                format!("__half2float({b})")
            )
        ),
        ElementKind::Bf16 => format!(
            "__float2bfloat16({})",
            binary_f32(
                op,
                format!("__bfloat162float({a})"),
                format!("__bfloat162float({b})")
            )
        ),
        // Increment 0c: the integer compute dtypes route to the C-operator
        // speller — legal for the bitwise/shift/logical vocabulary only
        // (binary_int backstop-panics on everything else, behind the plan
        // gate's validate-reject).
        ElementKind::I32 | ElementKind::I64 | ElementKind::S8 | ElementKind::U8 => {
            binary_int(op, a, b, dtype)
        }
        other => panic!("cuda backend: no binary math for dtype {other:?}"),
    }
}

/// Ternary select in f32 math: `cond != 0.0f` picks arm `a`, else `b`
/// ([`ScalarExpr::Select`] — nonzero-true, `-0.0` false, NaN true).
///
/// The `(float)` casts are MANDATORY, not cosmetic (the 0b double-promotion
/// lesson): they are identity no-ops on already-float operands but pin the
/// compare AND the ternary's type against a suffix-less double `Const`
/// literal from `const_lit` — without the arm casts, a double-literal arm
/// (`select(c, x, 0.0)`) promotes the whole ternary to double and the
/// double→float round-trip at the store QUIETS an f32 sNaN arm payload (a
/// bit diff vs the bespoke select). The cond cast makes the `!= 0` decision
/// happen in the compute dtype (the cmp-operand precedent, `binary_f32`).
/// No arithmetic ever touches an arm: the ternary is data movement
/// (setp+selp), so ±0 signs and NaN payloads (quiet and signaling) move
/// intact — byte-for-byte the bespoke `keep ? input[k] : zero_of<T>()`.
pub(crate) fn select_f32(c: String, a: String, b: String) -> String {
    format!("(((float)({c})) != 0.0f ? (float)({a}) : (float)({b}))")
}

/// [`select_f32`] with double literals/casts (the `binary_f64` cmp precedent).
pub(crate) fn select_f64(c: String, a: String, b: String) -> String {
    format!("(((double)({c})) != 0.0 ? (double)({a}) : (double)({b}))")
}

/// Lower a ternary select for `dtype` — the DEDICATED speller, never routed
/// through [`cuda_binary`]: its blanket f16/bf16 promote→f32→demote wrapper
/// would convert the ARMS, and a select's arms must move raw bits.
///
/// - **f32/f64**: [`select_f32`]/[`select_f64`] (identity-cast-pinned ternary).
/// - **f16/bf16**: promote ONLY the cond to f32 (`__half2float` — exact, the
///   same lossless embedding the cmp operands use) and test `!= 0.0f`; the
///   ARMS are picked as raw `__half`/`__nv_bfloat16` operand strings — the
///   ternary is typed in the storage dtype, a pure register pick, no
///   conversion pair, NaN payloads intact. The `(__half)`/`(__nv_bfloat16)`
///   arm casts are the half analog of the mandatory `(float)` casts above:
///   an IDENTITY no-op on an already-half operand (same-type functional cast
///   — no float bridge, bits untouched) that pins the ternary's type when an
///   arm is a suffix-less double `Const` literal (`select(c, x, 0.0)` — the
///   converting constructor rounds the literal to half once, `+0.0` bits,
///   matching bespoke `zero_of<T>`; without the cast the half arm would
///   convert THROUGH float/double to find a common ternary type, quieting
///   sNaN payloads). A composed arm is already a `__float2half(...)` string
///   (it crosses that bridge exactly once — the same bridge it would cross
///   at store), so the cast is a no-op there too.
/// - **int dtypes**: PANIC — the independent emitter backstop behind the plan
///   gate (`assert_int_op_admissibility`): v1 select is float-only (an int
///   select would raise the 0c U8/I8 cond-observer question).
fn cuda_select(c: String, a: String, b: String, dtype: ElementKind) -> String {
    match dtype {
        ElementKind::F32 | ElementKind::F32Strict => select_f32(c, a, b),
        ElementKind::F64 => select_f64(c, a, b),
        ElementKind::F16 => {
            format!("(__half2float({c}) != 0.0f ? (__half)({a}) : (__half)({b}))")
        }
        ElementKind::Bf16 => {
            format!("(__bfloat162float({c}) != 0.0f ? (__nv_bfloat16)({a}) : (__nv_bfloat16)({b}))")
        }
        other => panic!(
            "cuda backend: Select has no {other:?} lowering — v1 select is float-only \
             (f32/f32s/f64/f16/bf16); an integer select must miss honestly at the plan \
             gate (the 0c cond-observer question is unresolved)"
        ),
    }
}

/// Runtime scalar-param indices used by `e`, ascending + unique.
fn params_used(e: &ScalarExpr) -> Vec<u8> {
    fn rec(e: &ScalarExpr, out: &mut std::collections::BTreeSet<u8>) {
        match e {
            ScalarExpr::Param(i) => {
                out.insert(*i);
            }
            ScalarExpr::Unary(_, x) => rec(x, out),
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b)
            | ScalarExpr::Binary(_, a, b) => {
                rec(a, out);
                rec(b, out);
            }
            ScalarExpr::Select(c, a, b) => {
                rec(c, out);
                rec(a, out);
                rec(b, out);
            }
            ScalarExpr::Input(_)
            | ScalarExpr::Const(_)
            | ScalarExpr::Reduced(_)
            | ScalarExpr::Coord(_) => {}
        }
    }
    let mut set = std::collections::BTreeSet::new();
    rec(e, &mut set);
    set.into_iter().collect()
}

/// Emitter backstop for the two dtype-blind spellings (increment 0c): panic if
/// `e` contains an infix [`ScalarExpr::Div`] node or a [`ScalarExpr::Const`]
/// leaf while [`Cuda::lower`] is lowering an INTEGER dtype. Both are spelled by
/// shared backend code with no dtype context (`lower_expr` emits C `/` and an
/// f64 C literal for every dtype), so unlike the unary/binary-fn/int-only ops
/// they have no per-op speller panic to catch a plan-gate bypass — and they are
/// exactly the device-dangerous pair: integer `/0` is device-UB, and an
/// f64-spelled Const injects double math into an int kernel (f64 cannot even
/// represent all i64). Called from [`Cuda::lower`] over the body and every
/// reduction-class stage/epilogue, independent of `assert_int_op_admissibility`.
///
/// `in_reduction` mirrors `plan::assert_int_op_admissibility`'s rule 4 (the
/// any/all/count fused-predicate lift, ba325509/Task 3b): `true` only when the
/// expression is this plan's `Access::Reduction` body/post — CpuC/Slang (v1,
/// Elementwise-only) always pass `false`, so their coverage is unchanged.
/// `at_reduction_root` mirrors `plan::assert_int_op_admissibility`'s
/// `at_reduction_root` (whole-branch-review fix, closing the composed-
/// predicate leak): `true` ONLY for the initial call on the reduction
/// body/post root, `false` for every recursive descent — CpuC/Slang pass
/// `false` for both parameters (inert, since `in_reduction` is already
/// `false` there). Within that scope (`in_reduction && at_reduction_root`), a
/// `Cmp*` node's operands (leaf `Input`/`Reduced` or an exact 0/1 `Const`)
/// are exempted from the blanket `Const` panic below — that 0/1 `Const` is
/// safe (it lowers as an INTEGER literal in `cuda::emit_reduction`'s
/// `int_reduction_predicate`, which — like this gate — only inspects the
/// body/post ROOT node, never a nested one), never the f64 C literal this
/// backstop exists to catch); anything else in that position still recurses
/// into the general check and panics as before. A `Cmp*` reached as a
/// sub-node of Add/Sub/Mul (not the root) falls to the general
/// `ScalarExpr::Binary(_, a, b)` arm below like any other binary node, so its
/// own `Const`/`Div` operands are still policed by the blanket rules — it is
/// never itself the leaf-or-{0,1} exemption target.
pub(crate) fn assert_no_int_div_or_const(
    e: &ScalarExpr,
    dtype: ElementKind,
    in_reduction: bool,
    at_reduction_root: bool,
) {
    match e {
        ScalarExpr::Input(_) | ScalarExpr::Param(_) | ScalarExpr::Reduced(_) => {}
        // A Coord at an int dtype is the SAME hazard class as Const (its
        // spelling is a float cast) — but it has its own dedicated backstop,
        // `assert_coord_lowerable`, which runs beside this walk in
        // `Cuda::lower` for every dtype (not just int) and carries the
        // targeted message; no second assert here (one message per layer).
        ScalarExpr::Coord(_) => {}
        ScalarExpr::Const(_) => panic!(
            "cuda backend: Const at an integer dtype ({dtype:?}) — a Const is \
             spelled as an f64 C literal, which would silently run double math \
             in an integer kernel; the plan gate rejects this (an int-literal \
             speller is a follow-up)"
        ),
        ScalarExpr::Div(_, _) => panic!(
            "cuda backend: infix Div has no integer lowering ({dtype:?}) — the \
             bespoke elementwise surface has no int div and C `/` by zero is \
             device-UB; the plan gate rejects this"
        ),
        // Select at an integer dtype is validate-rejected at the plan gate
        // (v1 select is float-only); this walk — which only runs for int
        // dtypes — is its independent emitter backstop (G5), beside the
        // per-speller panic in `cuda_select` (which only the elementwise
        // paths route through; the reduction-class accumulator closures
        // don't, so the walk is the layer that covers them).
        ScalarExpr::Select(_, _, _) => panic!(
            "cuda backend: Select at an integer dtype ({dtype:?}) — v1 select is \
             float-only (the 0c U8/I8 cond-observer question is unresolved); the \
             plan gate rejects this"
        ),
        ScalarExpr::Unary(_, x) => {
            assert_no_int_div_or_const(x, dtype, in_reduction, false);
        }
        ScalarExpr::Add(a, b) | ScalarExpr::Sub(a, b) | ScalarExpr::Mul(a, b) => {
            assert_no_int_div_or_const(a, dtype, in_reduction, false);
            assert_no_int_div_or_const(b, dtype, in_reduction, false);
        }
        ScalarExpr::Binary(bop, a, b) if in_reduction && at_reduction_root && bop.is_cmp() => {
            for operand in [&**a, &**b] {
                match operand {
                    ScalarExpr::Input(_) | ScalarExpr::Reduced(_) => {}
                    ScalarExpr::Const(v) if *v == 0.0 || *v == 1.0 => {}
                    other => assert_no_int_div_or_const(other, dtype, in_reduction, false),
                }
            }
        }
        ScalarExpr::Binary(_, a, b) => {
            assert_no_int_div_or_const(a, dtype, in_reduction, false);
            assert_no_int_div_or_const(b, dtype, in_reduction, false);
        }
    }
}

#[cfg(test)]
mod int_div_or_const_root_gate_validate {
    //! Direct unit coverage for `assert_no_int_div_or_const`'s
    //! `at_reduction_root` restriction (whole-branch-review fix, mirroring
    //! `plan::int_reduction_predicate_gate_validate`). This backstop normally
    //! only runs AFTER `plan::assert_int_op_admissibility` has already
    //! validated the op at `build_plan` time, so a composed-predicate body
    //! never reaches it via the `generate()`/`Cuda::lower` path — the plan
    //! gate rejects it first. These tests call the function directly (it is
    //! `pub(crate)`) to exercise it as an independent layer in its own right
    //! (the "gate every layer" principle the surrounding code comments name
    //! throughout this file), the same way the plan-gate tests bypass
    //! `build_plan` to isolate `assert_int_op_admissibility`.
    use super::assert_no_int_div_or_const;
    use crate::ir::{BinaryOp, ScalarExpr, input, konst};
    use baracuda_kernel_vocab::ElementKind;

    // Root-Cmp positive case (mirrors the shipped `count` shape): a bare
    // `Cmp*` IS the reduction body root — admitted (0/1 Const operand
    // exempted from the blanket Const panic), must not panic.
    #[test]
    fn root_cmp_with_01_const_admitted() {
        let body = input(0).binary(BinaryOp::CmpNe, konst(0.0)).0;
        assert_no_int_div_or_const(&body, ElementKind::S8, true, true);
    }

    // Root-only guard: a COMPOSED predicate — `Add(Cmp*, Cmp*)` — reached as
    // the reduction body root. The root itself is `Add`, not `Cmp`, so the
    // exemption arm never matches at this level; the walk recurses into each
    // `Cmp*` child with `at_reduction_root: false` (this fix), lands the
    // nested Cmp on the general `Binary(_, a, b)` arm instead of the
    // exemption arm, and its `Const(0.0)` operand then hits the ordinary
    // blanket `Const` panic — closing the same composed-predicate leak this
    // gate mirrors from `plan::assert_int_op_admissibility`. Before this fix
    // the nested Cmp still matched `in_reduction && bop.is_cmp()`
    // (unconditionally, no root check) and its 0/1 Const was wrongly
    // exempted, so this call did NOT panic.
    #[test]
    fn composed_predicate_not_at_root_panics() {
        let cmp = || input(0).binary(BinaryOp::CmpNe, konst(0.0)).0;
        let body = ScalarExpr::Add(Box::new(cmp()), Box::new(cmp()));
        let r = std::panic::catch_unwind(|| {
            assert_no_int_div_or_const(&body, ElementKind::S8, true, true)
        });
        assert!(
            r.is_err(),
            "a Cmp* reached as a sub-node of Add (not the reduction body/post \
             root) must still panic on its Const operand — admitting it here \
             would mirror the plan-gate leak this fix closes"
        );
    }
}

/// Emitter backstop for [`ScalarExpr::Coord`] (increment 0d): panic if the
/// expression contains a Coord leaf at a non-float compute dtype, under a
/// non-Elementwise access, or with an out-of-range axis — the three rows the
/// plan gate (`plan::assert_coord_admissibility`) validate-rejects, enforced
/// here INDEPENDENTLY so a gate mutation cannot silently emit a rounding half
/// coordinate, an int-kernel float cast, an ambiguous folded-axis coordinate,
/// or an undefined `c{d}` identifier. Called from [`Cuda::lower`] over the
/// body and every reduction-class stage/epilogue (the same coverage as
/// [`assert_no_int_div_or_const`]); messages are cuda-prefixed, distinct from
/// the plan gate's.
fn assert_coord_lowerable(e: &ScalarExpr, plan: &KernelPlan<'_>) {
    match e {
        ScalarExpr::Input(_)
        | ScalarExpr::Const(_)
        | ScalarExpr::Param(_)
        | ScalarExpr::Reduced(_) => {}
        ScalarExpr::Coord(d) => {
            assert!(
                matches!(plan.access, Access::Elementwise),
                "cuda backend: Coord({d}) under a non-Elementwise access — the \
                 reduction-class emitters iterate fold/row/contraction coordinate \
                 spaces, not the elementwise output space; the plan gate rejects this"
            );
            assert!(
                matches!(
                    plan.dtype,
                    ElementKind::F32 | ElementKind::F32Strict | ElementKind::F64
                ),
                "cuda backend: Coord({d}) at non-float dtype {:?} — the coordinate is \
                 spelled as a float/double cast, which rounds past 2048 at half \
                 precision and injects float math into an integer kernel; the plan \
                 gate rejects this",
                plan.dtype
            );
            assert!(
                *d < plan.key.rank,
                "cuda backend: Coord({d}) axis out of range for rank {} — no c{d} \
                 coordinate exists to read; the plan gate rejects this",
                plan.key.rank
            );
        }
        ScalarExpr::Unary(_, x) => assert_coord_lowerable(x, plan),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => {
            assert_coord_lowerable(a, plan);
            assert_coord_lowerable(b, plan);
        }
        ScalarExpr::Select(c, a, b) => {
            assert_coord_lowerable(c, plan);
            assert_coord_lowerable(a, plan);
            assert_coord_lowerable(b, plan);
        }
    }
}

/// The SCALAR COMPUTE ctype for an op's runtime launch params — `scalar_ctype(
/// plan.dtype)`, i.e. `"float"` for F32/F32Strict (byte-identical to the pre-F64
/// hardcode) and `"double"` for F64. A launch param is ALWAYS a scalar arg, never
/// vectorized, so this is the declaration ctype at EVERY emitter (including the
/// vectorized ones, whose OPERANDS are `float4`/`double2` but whose param stays
/// `double p0`) — the F64-param increment's load-bearing distinction: pass THIS,
/// never `vty`/`octype`. The `Cuda::lower` param assert (see the `matches!` on
/// `plan.dtype` above) guarantees a param-bearing plan has a spellable scalar
/// ctype, so the `expect` is unreachable for any op that actually declares a param.
pub(crate) fn param_ctype(plan: &KernelPlan<'_>) -> &'static str {
    scalar_ctype(plan.dtype).expect("param dtype checked by the Cuda::lower param assert")
}

/// The trailing `, <ctype> p0, <ctype> p1, …` kernel-signature suffix for the
/// op's runtime scalar params (empty when the op has none). `param_ctype` is the
/// SCALAR COMPUTE ctype `scalar_ctype(plan.dtype)` — `"float"` for F32/F32Strict
/// (byte-identical to the pre-F64 hardcode by construction), `"double"` for F64.
/// A launch param is ALWAYS scalar: even on the vectorized emitter (operands are
/// `float4`/`double2`) the declaration stays `double p0`, NOT `double2 p0` — so
/// callers pass the SCALAR compute ctype, never `vty`/`octype` (the F64-param
/// increment's load-bearing distinction).
pub(crate) fn param_args(e: &ScalarExpr, param_ctype: &str) -> String {
    params_used(e)
        .iter()
        .map(|i| format!(", {param_ctype} p{i}"))
        .collect()
}

/// Like [`param_args`], but over the UNION of params used across several
/// expressions (increment 0e: a reduction's launch signature must declare params
/// referenced by the fold body OR the post-expr). Deduped + ascending via the
/// `BTreeSet` in [`params_used`]. For a body-only op (identity post) this is
/// byte-identical to `param_args(body, param_ctype)`.
fn param_args_multi(exprs: &[&ScalarExpr], param_ctype: &str) -> String {
    let mut set = std::collections::BTreeSet::new();
    for e in exprs {
        set.extend(params_used(e));
    }
    set.iter()
        .map(|i| format!(", {param_ctype} p{i}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::ir::{OpDef, input, param};
    use crate::{Cuda, generate};
    use baracuda_kernel_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

    fn add_op(dtypes: &[ElementKind]) -> OpDef {
        OpDef::elementwise("add", 2, dtypes, input(0) + input(1))
    }

    fn binary_key(dt: ElementKind) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 256);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    }

    // ---- coord-unravel: shared decomposition + freestanding helper (Phase 1) ----

    #[test]
    fn unravel_decomp_inline_form_is_byte_identical() {
        // The scalar-param, unguarded call MUST reproduce exactly what emit_strided
        // used to hand-inline. Pins the single-source-of-truth factoring
        // independently of the full emission-golden suite.
        let got = super::emit_unravel_decomp(
            (0..3).rev(),
            "        ",
            "lin",
            |d| format!("c{d}"),
            |d| format!("shape{d}"),
            false,
        );
        let want = concat!(
            "        long long c2 = lin % shape2; lin /= shape2;\n",
            "        long long c1 = lin % shape1; lin /= shape1;\n",
            "        long long c0 = lin % shape0; lin /= shape0;\n",
        );
        assert_eq!(got, want);
    }

    #[test]
    fn coord_unravel_helper_emits_unrolled_per_rank() {
        let src = super::emit_coord_unravel_helper(4);
        assert!(src.contains("#ifndef BARACUDA_COORD_UNRAVEL_GEN_CUH"));
        assert!(src.contains("namespace baracuda { namespace coord { namespace gen {"));
        // one function per rank 1..=4, none beyond max_rank
        for n in 1..=4 {
            assert!(
                src.contains(&format!("unravel_offset_1_r{n}(")),
                "missing rank {n}"
            );
        }
        assert!(
            !src.contains("unravel_offset_1_r5("),
            "should stop at max_rank"
        );
        // rank-4 body: unrolled (c3..c0 declared straight-line, NO runtime loop),
        // guarded, with the full offset accumulation.
        assert!(src.contains("long long c3 = (shape.v[3] == 0)"));
        assert!(src.contains("long long c0 = (shape.v[0] == 0)"));
        assert!(
            !src.contains("for ("),
            "generated helper must be unrolled — no runtime loop"
        );
        assert!(src.contains("off += c3 * (int64_t)stride.v[3];"));
        assert!(src.contains("off += c0 * (int64_t)stride.v[0];"));
        assert!(src.contains("return off;"));
        // multi-stride variants: one unravel pass feeds K offsets (void out-params),
        // matching the hand-written unravel_offsets_2/3/4.
        assert!(src.contains("void unravel_offsets_2_r4("));
        assert!(src.contains("void unravel_offsets_3_r2("));
        assert!(src.contains("void unravel_offsets_4_r4("));
        assert!(src.contains("int64_t& off_d"));
        assert!(src.contains("off_a += c0 * (int64_t)stride_a.v[0];"));
        assert!(src.contains("off_d += c3 * (int64_t)stride_d.v[3];"));
    }

    /// Regenerate the generated coord-unravel helper for the on-device
    /// `ondevice/unravel_bench.cu` harness. Run with:
    ///   `UNRAVEL_OUT=<outdir> cargo test -p baracuda-kernelgen dump_coord_unravel_helper -- --ignored --nocapture`
    #[test]
    #[ignore = "writes the generated coord-unravel helper for the on-device bench"]
    fn dump_coord_unravel_helper() {
        let out = std::env::var("UNRAVEL_OUT").unwrap_or_else(|_| ".".to_string());
        let path = format!("{out}/coord_unravel_gen.cuh");
        std::fs::write(&path, super::emit_coord_unravel_helper(8)).unwrap();
        println!("wrote {path}");
    }

    // ---- dtype-promote: shared promotion routines + freestanding helper ----

    #[test]
    fn promote_demote_routines_reproduce_inline_spelling() {
        // The shared routines must reproduce exactly what the inline sites emit
        // (the reduction load/store closures now route through them).
        assert_eq!(
            super::promote_load_f32(ElementKind::F16, "in0[idx]"),
            "__half2float(in0[idx])"
        );
        assert_eq!(
            super::promote_load_f32(ElementKind::Bf16, "in0[idx]"),
            "__bfloat162float(in0[idx])"
        );
        assert_eq!(
            super::promote_load_f32(ElementKind::F32, "in0[idx]"),
            "in0[idx]"
        );
        assert_eq!(
            super::demote_store_f32(ElementKind::F16, "v"),
            "__float2half(v)"
        );
        assert_eq!(
            super::demote_store_f32(ElementKind::Bf16, "v"),
            "__float2bfloat16(v)"
        );
        assert_eq!(super::demote_store_f32(ElementKind::F32, "v"), "v");
    }

    #[test]
    fn dtype_promote_helper_emits_lanes() {
        let src = super::emit_dtype_promote_helper();
        assert!(src.contains("#ifndef BARACUDA_DTYPE_PROMOTE_GEN_CUH"));
        assert!(src.contains("namespace baracuda { namespace dtype { namespace gen {"));
        assert!(src.contains("float load_as_f32_f16(__half x) { return __half2float(x); }"));
        assert!(src.contains("__half store_from_f32_f16(float v) { return __float2half(v); }"));
        assert!(
            src.contains("float load_as_f32_bf16(__nv_bfloat16 x) { return __bfloat162float(x); }")
        );
        assert!(
            src.contains("double load_as_f64_f16(__half x) { return (double)__half2float(x); }")
        );
        assert!(
            src.contains("__half store_from_f64_f16(double v) { return __float2half((float)v); }")
        );
    }

    /// Regenerate the generated dtype-promote helper for the on-device validator.
    ///   `DTYPE_OUT=<outdir> cargo test -p baracuda-kernelgen dump_dtype_promote_helper -- --ignored --nocapture`
    #[test]
    #[ignore = "writes the generated dtype-promote helper for the on-device validator"]
    fn dump_dtype_promote_helper() {
        let out = std::env::var("DTYPE_OUT").unwrap_or_else(|_| ".".to_string());
        let path = format!("{out}/dtype_promote_gen.cuh");
        std::fs::write(&path, super::emit_dtype_promote_helper()).unwrap();
        println!("wrote {path}");
    }

    // ---- cast: shared per-element cast routine + freestanding helper ----

    #[test]
    fn cast_scalar_reproduces_inline_u8_narrowing() {
        // The hetero elementwise store (`store_expr_of`) now routes through
        // `cast_scalar(plan.dtype, U8, root)`; these must be BYTE-IDENTICAL to the
        // prior hand-inlined `(unsigned char)…` spellings (goldens unchanged).
        assert_eq!(
            super::cast_scalar(ElementKind::F16, ElementKind::U8, "root"),
            "(unsigned char)__half2float(root)"
        );
        assert_eq!(
            super::cast_scalar(ElementKind::Bf16, ElementKind::U8, "root"),
            "(unsigned char)__bfloat162float(root)"
        );
        assert_eq!(
            super::cast_scalar(ElementKind::F32, ElementKind::U8, "root"),
            "(unsigned char)root"
        );
        assert_eq!(
            super::cast_scalar(ElementKind::I32, ElementKind::U8, "root"),
            "(unsigned char)root"
        );
        assert_eq!(
            super::cast_scalar(ElementKind::F64, ElementKind::U8, "root"),
            "(unsigned char)root"
        );
    }

    #[test]
    fn cast_scalar_matches_cast_value_semantics() {
        // Value-identical to `baracuda_cast.cuh`'s `cast_value<TIn, TOut>` (the
        // generated form uses C-style casts + the shared half intrinsics).
        // Identity.
        assert_eq!(
            super::cast_scalar(ElementKind::F16, ElementKind::F16, "x"),
            "x"
        );
        assert_eq!(
            super::cast_scalar(ElementKind::I32, ElementKind::I32, "x"),
            "x"
        );
        // Cross-half: widen then narrow.
        assert_eq!(
            super::cast_scalar(ElementKind::F16, ElementKind::Bf16, "x"),
            "__float2bfloat16(__half2float(x))"
        );
        assert_eq!(
            super::cast_scalar(ElementKind::Bf16, ElementKind::F16, "x"),
            "__float2half(__bfloat162float(x))"
        );
        // Half -> arithmetic: widen, C-cast.
        assert_eq!(
            super::cast_scalar(ElementKind::F16, ElementKind::F32, "x"),
            "(float)__half2float(x)"
        );
        assert_eq!(
            super::cast_scalar(ElementKind::Bf16, ElementKind::I32, "x"),
            "(int)__bfloat162float(x)"
        );
        // Arithmetic -> half: float already-float skips the extra cast; others cast.
        assert_eq!(
            super::cast_scalar(ElementKind::F32, ElementKind::F16, "x"),
            "__float2half(x)"
        );
        assert_eq!(
            super::cast_scalar(ElementKind::F64, ElementKind::F16, "x"),
            "__float2half((float)x)"
        );
        assert_eq!(
            super::cast_scalar(ElementKind::I32, ElementKind::Bf16, "x"),
            "__float2bfloat16((float)x)"
        );
        // Arithmetic -> arithmetic: plain C-style cast.
        assert_eq!(
            super::cast_scalar(ElementKind::F32, ElementKind::I32, "x"),
            "(int)x"
        );
        assert_eq!(
            super::cast_scalar(ElementKind::I64, ElementKind::F32, "x"),
            "(float)x"
        );
    }

    #[test]
    fn cast_helper_emits_matrix() {
        let src = super::emit_cast_helper();
        assert!(src.contains("#ifndef BARACUDA_CAST_GEN_CUH"));
        assert!(src.contains("namespace baracuda { namespace cast { namespace gen {"));
        // Representative lanes across every arm of `cast_scalar`.
        assert!(src.contains("__half cast_f16_f16(__half x) { return x; }"));
        assert!(src.contains("float cast_f16_f32(__half x) { return (float)__half2float(x); }"));
        assert!(src.contains(
            "__nv_bfloat16 cast_f16_bf16(__half x) { return __float2bfloat16(__half2float(x)); }"
        ));
        assert!(src.contains("__half cast_f32_f16(float x) { return __float2half(x); }"));
        assert!(src.contains("__half cast_f64_f16(double x) { return __float2half((float)x); }"));
        assert!(src.contains("int cast_f32_i32(float x) { return (int)x; }"));
        assert!(src.contains("float cast_i64_f32(long long x) { return (float)x; }"));
        assert!(src.contains(
            "unsigned char cast_f16_u8(__half x) { return (unsigned char)__half2float(x); }"
        ));
        // Full 8×8 matrix of device functions (identity lanes included).
        assert_eq!(src.matches("__device__ __forceinline__").count(), 8 * 8);
    }

    /// Regenerate the generated cast helper for the on-device validator.
    ///   `CAST_OUT=<outdir> cargo test -p baracuda-kernelgen dump_cast_helper -- --ignored --nocapture`
    #[test]
    #[ignore = "writes the generated cast helper for the on-device validator"]
    fn dump_cast_helper() {
        let out = std::env::var("CAST_OUT").unwrap_or_else(|_| ".".to_string());
        let path = format!("{out}/cast_gen.cuh");
        std::fs::write(&path, super::emit_cast_helper()).unwrap();
        println!("wrote {path}");
    }

    // rank-2 gather cell: data [4,3] (input 0, gathered axis 0), index [4,3] i32
    // full-shape (input 1), out [4,3]. Non-contig (strided) so it would strided
    // anyway; the gather forces strided regardless.
    fn gather_2d_key() -> baracuda_kernel_vocab::StructureKey {
        gather_2d_key_idx(ElementKind::I32)
    }

    fn gather_2d_key_idx(idx_dt: ElementKind) -> baracuda_kernel_vocab::StructureKey {
        let data = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], idx_dt, 256);
        let out = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        structure_key(
            OpCategory::BinaryElementwise,
            &[data, idx, out],
            ArchSku::Sm89,
        )
    }

    // ---- increment 4: GATHER emission goldens ----

    #[test]
    fn gather_skip_substitutes_the_index_value_for_the_gathered_axis() {
        use crate::ir::OobPolicy;
        // torch-gather along axis 0: out[c] = data[idx[c], c1], skip OOB.
        let op = OpDef::gather(
            "gather",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::I32,
        );
        let k = generate(&op, &gather_2d_key(), &Cuda);
        // The index dtype rides the ENTRY_POINT symbol.
        assert_eq!(k.name, "baracuda_gen_gather_f32_i32_strided_r2");
        // Index operand is an INT pointer (not the data dtype).
        assert!(k.source.contains("const int* __restrict__ in1,"));
        assert!(!k.source.contains("const float* __restrict__ in1,"));
        // The gathered-axis extent rides a dedicated scalar param.
        assert!(k.source.contains("long long gext,"));
        // Index load + clamp + oob predicate.
        assert!(k.source.contains("long long gidx_off = c0*s1_0 + c1*s1_1;"));
        assert!(
            k.source
                .contains("long long gidx_raw = (long long)in1[gidx_off];")
        );
        assert!(k.source.contains(
            "long long gidx_clamped = gidx_raw < 0 ? 0 : (gidx_raw >= gext ? gext - 1 : gidx_raw);"
        ));
        assert!(
            k.source
                .contains("bool goob = (gidx_raw < 0) || (gidx_raw >= gext);")
        );
        // THE increment: the gathered-axis term is idx·stride[axis], NOT c0·stride.
        // (Matches bespoke `src_off = idx_val*stride_src[0] + coord[1]*stride_src[1]`.)
        assert!(
            k.source
                .contains("long long o0 = (gidx_clamped)*s0_0 + c1*s0_1;"),
            "gathered-axis term must use the index value, got:\n{}",
            k.source
        );
        assert!(!k.source.contains("long long o0 = c0*s0_0 + c1*s0_1;"));
        // Skip policy: predicated store (bespoke `continue;`) — no OOB load, no write.
        assert!(k.source.contains("if (!goob) out[oo] = in0[o0];"));
    }

    #[test]
    fn embedding_zerofill_selects_zero_on_oob() {
        // embedding = axis-0 gather, ZeroFill (bespoke OOB/neg → zero row).
        let op = OpDef::embedding("emb", &[ElementKind::F32], ElementKind::I64);
        let k = generate(&op, &gather_2d_key_idx(ElementKind::I64), &Cuda);
        assert_eq!(k.name, "baracuda_gen_emb_f32_i64_strided_r2");
        // i64 index ⇒ long long pointer.
        assert!(k.source.contains("const long long* __restrict__ in1,"));
        // ZeroFill: store the zero fill on OOB, else the value (no skip predicate).
        assert!(k.source.contains("out[oo] = goob ? (0.0f) : (in0[o0]);"));
        assert!(!k.source.contains("if (!goob)"));
    }

    #[test]
    fn embedding_f16_zerofill_uses_the_half_zero_intrinsic() {
        // The ZeroFill literal must be the fp16 constructor, not a bare `0`.
        let data = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F16, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[data, idx, data],
            ArchSku::Sm89,
        );
        let op = OpDef::embedding("emb", &[ElementKind::F16], ElementKind::I32);
        let k = generate(&op, &key, &Cuda);
        assert!(
            k.source
                .contains("out[oo] = goob ? (__float2half(0.0f)) : (in0[o0]);")
        );
    }

    #[test]
    fn gather_clamp_has_no_store_predicate() {
        use crate::ir::OobPolicy;
        // Clamp: gidx_clamped IS the effective index; always store, no goob.
        let op = OpDef::gather(
            "gclamp",
            &[ElementKind::F32],
            0,
            OobPolicy::Clamp,
            ElementKind::I32,
        );
        let k = generate(&op, &gather_2d_key(), &Cuda);
        assert!(
            k.source
                .contains("long long o0 = (gidx_clamped)*s0_0 + c1*s0_1;")
        );
        // No OOB predicate at all (Clamp always stores the clamped-index value).
        assert!(!k.source.contains("goob"));
        assert!(k.source.contains("out[oo] = in0[o0];"));
        assert!(!k.source.contains("if (!goob)"));
    }

    #[test]
    fn index_select_1d_index_degenerates_to_the_axis_coordinate() {
        use crate::ir::OobPolicy;
        // index_select axis 0: the index is 1-D (broadcast on axis 1, stride 0), so
        // its offset drops the axis-1 term ⇒ `gidx_off = c0*s1_0` — the bespoke
        // `index_select` 1-D lookup by `coord[select_dim]`.
        let data = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[1, 0], ElementKind::I32, 256); // bcast axis 1
        let out = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::F32, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[data, idx, out],
            ArchSku::Sm89,
        );
        let op = OpDef::index_select(
            "isel",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::I32,
        );
        let k = generate(&op, &key, &Cuda);
        assert!(
            k.source.contains("long long gidx_off = c0*s1_0;"),
            "1-D index offset must drop the broadcast axis, got:\n{}",
            k.source
        );
        assert!(
            k.source
                .contains("long long o0 = (gidx_clamped)*s0_0 + c1*s0_1;")
        );
    }

    #[test]
    fn gather_forces_strided_off_the_vectorized_path() {
        use crate::ir::OobPolicy;
        use crate::plan::{Schedule, build_plan};
        // A fully-contiguous cell that a non-gather copy would VECTORIZE.
        let data = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::F32, 256);
        let idx = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::I32, 256);
        let out = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::F32, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[data, idx, out],
            ArchSku::Sm89,
        );
        // Baseline: a view-free/index-free copy vectorizes on this contiguous cell.
        let copy = OpDef::elementwise("copy", 1, &[ElementKind::F32], input(0));
        let copy_key = {
            let a = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::F32, 256);
            structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
        };
        assert!(matches!(
            build_plan(&copy, &copy_key).schedule,
            Schedule::Vectorized { .. }
        ));
        // The gather forces Strided (a data-dependent address cannot coalesce).
        let op = OpDef::gather(
            "gather",
            &[ElementKind::F32],
            0,
            OobPolicy::Skip,
            ElementKind::I32,
        );
        assert_eq!(build_plan(&op, &key).schedule, Schedule::Strided);
    }

    #[test]
    fn index_free_op_emits_byte_identical_to_pre_increment4() {
        // An all-Direct read_index vec must emit the exact same source + name as the
        // index-free op — the byte-identical guarantee.
        use crate::ir::ReadIndex;
        let key = binary_key(ElementKind::F32);
        let free = generate(&add_op(&[ElementKind::F32]), &key, &Cuda);
        let with = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1))
            .with_indexed(vec![ReadIndex::Direct, ReadIndex::Direct]);
        let k = generate(&with, &key, &Cuda);
        assert_eq!(free.name, k.name);
        assert_eq!(
            free.source, k.source,
            "an all-Direct read_index must emit byte-identical source"
        );
    }

    // ---- increment 5: SCATTER emission goldens ----

    // rank-2 scatter cell: updates [4,3] f32 (in0), index [4,3] i32 full-shape
    // (in1), dst [4,3] f32 (out). The dst extent along the scattered axis rides
    // `sext`; the key dst supplies strides/broadcast facts.
    fn scatter_2d_key(idx_dt: ElementKind) -> baracuda_kernel_vocab::StructureKey {
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
    fn scatter_assign_substitutes_index_for_the_output_axis() {
        // scatter along axis 0: out[idx[c], c1] = updates[c0, c1], skip OOB target.
        let op = OpDef::scatter("scatter", &[ElementKind::F32], 0, ElementKind::I32);
        let k = generate(&op, &scatter_2d_key(ElementKind::I32), &Cuda);
        // The index dtype rides the entry-point symbol.
        assert_eq!(k.name, "baracuda_gen_scatter_f32_i32_strided_r2");
        // The index operand is an INT pointer; the destination extent is a param.
        assert!(k.source.contains("const int* __restrict__ in1,"));
        assert!(k.source.contains("long long sext,"));
        // Index load + clamp + skip predicate (write-side mirror of gather).
        assert!(k.source.contains("long long sidx_off = c0*s1_0 + c1*s1_1;"));
        assert!(
            k.source
                .contains("long long sidx_raw = (long long)in1[sidx_off];")
        );
        assert!(k.source.contains(
            "long long sidx_clamped = sidx_raw < 0 ? 0 : (sidx_raw >= sext ? sext - 1 : sidx_raw);"
        ));
        assert!(
            k.source
                .contains("bool soob = (sidx_raw < 0) || (sidx_raw >= sext);")
        );
        // THE increment: the OUTPUT-axis term is idx·stride_out[axis], not c0·so.
        assert!(
            k.source
                .contains("long long oo = (sidx_clamped)*so_0 + c1*so_1;"),
            "scattered-axis term must use the index value, got:\n{}",
            k.source
        );
        // Assign store, predicated by the skip guard (bespoke `continue;`).
        assert!(k.source.contains("if (!soob) out[oo] = in0[o0];"));
        // No atomics for a plain assign.
        assert!(!k.source.contains("atomicAdd"));
    }

    #[test]
    fn integer_scatter_add_is_the_deterministic_atomicadd_base() {
        // i32 scatter_add: integer atomicAdd ⇒ order-independent ⇒ the base lower()
        // (NOT routed to the gather-sum; NOT a variant).
        let op = OpDef::scatter_add("scatter_add", &[ElementKind::I32], 0, ElementKind::I32);
        let iupd = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let dst = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[iupd, idx, dst],
            ArchSku::Sm89,
        );
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_scatter_add_i32_i32_strided_r2");
        assert!(
            k.source
                .contains("if (!soob) atomicAdd(&out[oo], (int)(in0[o0]));")
        );
        // Deterministic combine ⇒ no variant offered.
        let vs = crate::generate_variants(&op, &key, &Cuda);
        assert_eq!(
            vs.len(),
            1,
            "integer scatter_add ships one unconditional base"
        );
        assert_eq!(vs[0].fidelity, crate::VariantFidelity::BitIdentical);
    }

    #[test]
    fn i64_scatter_add_uses_the_ull_reinterpret() {
        let op = OpDef::scatter_add("scatter_add", &[ElementKind::I64], 0, ElementKind::I64);
        let iupd = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I64, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I64, 256);
        let dst = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I64, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[iupd, idx, dst],
            ArchSku::Sm89,
        );
        let k = generate(&op, &key, &Cuda);
        assert!(k.source.contains(
            "if (!soob) atomicAdd((unsigned long long*)&out[oo], (unsigned long long)(in0[o0]));"
        ));
    }

    #[test]
    fn bincount_scatters_const1_atomicadd_into_i32_counts() {
        // bincount: in0 = i64 data (the index), body Const(1), i32 counts out.
        let op = OpDef::bincount("bincount", ElementKind::I64);
        let x = OperandDesc::new(1, &[64], &[1], ElementKind::I64, 256);
        let out = OperandDesc::new(1, &[16], &[1], ElementKind::I32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[x, out], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_bincount_i64_i64_strided_r1");
        // The lone input is the index — an i64 (long long) pointer; i32 out pointer.
        assert!(k.source.contains("const long long* __restrict__ in0,"));
        assert!(k.source.contains("int* __restrict__ out,"));
        // Value is the constant 1 narrowed to the i32 count cell.
        assert!(
            k.source
                .contains("if (!soob) atomicAdd(&out[oo], (int)(1.0));")
        );
        // Deterministic integer counts ⇒ ships unconditionally (no variant).
        assert_eq!(crate::generate_variants(&op, &key, &Cuda).len(), 1);
    }

    #[test]
    fn fp_scatter_add_base_is_the_deterministic_gather_sum() {
        // f32 scatter_add: FP atomicAdd is non-deterministic, so the BASE lower()
        // is the deterministic gather-sum — one thread per output cell, scan the
        // update domain, sum matching values, NO atomics.
        let op = OpDef::scatter_add("scatter_add", &[ElementKind::F32], 0, ElementKind::I32);
        let k = generate(&op, &scatter_2d_key(ElementKind::I32), &Cuda);
        assert_eq!(
            k.name,
            "baracuda_gen_scatter_add_f32_i32_scatter_gathersum_r2"
        );
        // No atomics in the deterministic base.
        assert!(
            !k.source.contains("atomicAdd"),
            "gather-sum base must be atomic-free"
        );
        // The match condition: scattered target == this destination cell.
        assert!(k.source.contains("if (sidx == oc0 && uc1 == oc1) {"));
        // Deterministic accumulate into the existing destination (one owner, exact).
        assert!(k.source.contains("out[oo] = (out[oo] + acc);"));
    }

    #[test]
    fn fp_scatter_add_offers_the_nondeterministic_atomic_variant() {
        // The FP-atomic scatter ships ONLY as a Nondeterministic variant beside the
        // deterministic gather-sum base — never the silent default.
        let op = OpDef::scatter_add("scatter_add", &[ElementKind::F32], 0, ElementKind::I32);
        let vs = crate::generate_variants(&op, &scatter_2d_key(ElementKind::I32), &Cuda);
        assert_eq!(vs.len(), 2, "base (gather-sum) + atomic variant");
        // Base is the deterministic gather-sum.
        assert_eq!(vs[0].fidelity, crate::VariantFidelity::BitIdentical);
        assert!(vs[0].kernels[0].name.contains("_scatter_gathersum_"));
        // The atomic variant is Nondeterministic + carries the honest determinism flip.
        let atomic = &vs[1];
        assert_eq!(atomic.tag, "atomic");
        assert_eq!(atomic.fidelity, crate::VariantFidelity::Nondeterministic);
        assert!(
            atomic.kernels[0]
                .source
                .contains("if (!soob) atomicAdd(&out[oo], (float)(in0[o0]));")
        );
        assert!(atomic.launch_note.contains("determinism: nondeterministic"));
        assert!(atomic.launch_note.contains("NON-DETERMINISTIC"));
    }

    #[test]
    fn index_add_1d_index_degenerates_to_the_axis_coordinate() {
        // index_add axis 0 with a 1-D index (broadcast on axis 1, stride 0): the
        // index offset drops the axis-1 term ⇒ `sidx_off = c0*s1_0`. Integer value
        // dtype so this is the deterministic atomicAdd base (checks the offset).
        let upd = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let idx = OperandDesc::new(2, &[4, 3], &[1, 0], ElementKind::I32, 256); // bcast axis 1
        let dst = OperandDesc::new(2, &[4, 3], &[3, 1], ElementKind::I32, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[upd, idx, dst],
            ArchSku::Sm89,
        );
        let op = OpDef::index_add("index_add", &[ElementKind::I32], 0, ElementKind::I32);
        let k = generate(&op, &key, &Cuda);
        assert!(
            k.source.contains("long long sidx_off = c0*s1_0;"),
            "1-D index offset must drop the broadcast axis, got:\n{}",
            k.source
        );
        assert!(
            k.source
                .contains("long long oo = (sidx_clamped)*so_0 + c1*so_1;")
        );
    }

    #[test]
    fn scatter_forces_strided_off_the_vectorized_path() {
        use crate::plan::{Schedule, build_plan};
        // A fully-contiguous 1-D cell a non-scatter copy would VECTORIZE.
        let upd = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::F32, 256);
        let idx = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::I32, 256);
        let dst = OperandDesc::new(1, &[1 << 16], &[1], ElementKind::F32, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[upd, idx, dst],
            ArchSku::Sm89,
        );
        let op = OpDef::scatter("scatter", &[ElementKind::F32], 0, ElementKind::I32);
        assert_eq!(build_plan(&op, &key).schedule, Schedule::Strided);
    }

    #[test]
    fn non_scatter_op_emits_byte_identical_to_pre_increment5() {
        // A write-Direct op must emit the exact same source + name as before #5.
        let key = binary_key(ElementKind::F32);
        let free = generate(&add_op(&[ElementKind::F32]), &key, &Cuda);
        let with = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1))
            .with_scatter(crate::ir::WriteIndex::Direct);
        let k = generate(&with, &key, &Cuda);
        assert_eq!(free.name, k.name);
        assert_eq!(
            free.source, k.source,
            "write-Direct must emit byte-identical source"
        );
    }

    #[test]
    #[should_panic(expected = "must lower on the Strided schedule")]
    fn scattered_op_on_a_non_strided_schedule_is_refused_by_the_backstop() {
        use crate::backend::Backend;
        use crate::ir::{OobPolicy, WriteCombine, WriteIndex};
        use crate::plan::{KernelPlan, Schedule};
        // A scatter manually paired with the Vectorized schedule (build_plan never
        // produces this) — the independent emitter backstop must refuse it.
        let key = scatter_2d_key(ElementKind::I32);
        let body = input(0).0;
        let wi = WriteIndex::ScatterIndexed {
            index_operand: 1,
            axis: 0,
            combine: WriteCombine::Assign,
            oob: OobPolicy::Skip,
            index_dtype: ElementKind::I32,
        };
        let plan = KernelPlan {
            op_name: "sneaky",
            n_inputs: 2,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::Vectorized { width: 4 },
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &wi,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "must lower on the Strided schedule")]
    fn gathered_op_on_a_non_strided_schedule_is_refused_by_the_backstop() {
        use crate::backend::Backend;
        use crate::ir::{OobPolicy, ReadIndex};
        use crate::plan::{KernelPlan, Schedule};
        // A gather manually paired with the Vectorized schedule (build_plan never
        // produces this) — the independent emitter backstop must refuse it.
        let key = gather_2d_key();
        let body = input(0).0;
        let ri = [
            ReadIndex::Indexed {
                index_operand: 1,
                axis: 0,
                oob: OobPolicy::Skip,
                index_dtype: ElementKind::I32,
            },
            ReadIndex::Direct,
        ];
        let plan = KernelPlan {
            op_name: "sneaky",
            n_inputs: 2,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::Vectorized { width: 4 },
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &ri,
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    fn f32_contiguous_vectorizes_to_float4() {
        let k = generate(
            &add_op(&[ElementKind::F32]),
            &binary_key(ElementKind::F32),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_add_f32_co_v4");
        assert!(k.source.contains("float4 v0 = in0[i];"));
        assert!(k.source.contains("vo.x = (v0.x + v1.x);"));
        assert!(k.source.contains("vo.w = (v0.w + v1.w);"));
    }

    #[test]
    fn maximum_propagates_nan_and_step_excludes_zero() {
        use crate::ir::UnaryOp;
        // Maximum: NaN-propagating compare-select, not fmaxf (the house convention).
        let max = OpDef::elementwise("m", 2, &[ElementKind::F32], input(0).max(input(1)));
        let km = generate(&max, &binary_key(ElementKind::F32), &Cuda);
        assert!(!km.source.contains("fmaxf"));
        assert!(km.source.contains("!=")); // the NaN check
        // Step: heaviside(x, 0), strict `>` so step(0) = 0.
        let step = OpDef::elementwise("s", 1, &[ElementKind::F32], input(0).unary(UnaryOp::Step));
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        let ks = generate(&step, &key, &Cuda);
        assert!(ks.source.contains("> 0.0f ? 1.0f : 0.0f"));
        assert!(!ks.source.contains(">= 0.0f"));
    }

    #[test]
    fn f64_contiguous_vectorizes_to_double2() {
        let k = generate(
            &add_op(&[ElementKind::F64]),
            &binary_key(ElementKind::F64),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_add_f64_co_v2");
        assert!(k.source.contains("double2 v0 = in0[i];"));
        assert!(k.source.contains("vo.x = (v0.x + v1.x);"));
        assert!(k.source.contains("vo.y = (v0.y + v1.y);"));
        assert!(!k.source.contains("vo.z")); // only two lanes
    }

    #[test]
    fn f16_v8_packs_four_half2_pair_lanes() {
        // f16 keys V8: one 128-bit access = four __half2 pair lanes; infix `+`
        // lowers through the __half2 operator overload — the native packed op,
        // bit-identical per lane to the scalar __half operator+ the scalar
        // kernel uses.
        let k = generate(
            &add_op(&[ElementKind::F16]),
            &binary_key(ElementKind::F16),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_add_f16_co_v8");
        assert!(k.source.contains("#include <cuda_fp16.h>"));
        assert!(k.source.contains(
            "struct __align__(16) baracuda_gen_add_f16_co_v8_vec { __half2 a, b, c, d; };"
        ));
        assert!(
            k.source
                .contains("const baracuda_gen_add_f16_co_v8_vec* __restrict__ in0")
        );
        for f in ["a", "b", "c", "d"] {
            assert!(
                k.source
                    .contains(&format!("__half2 tmp0 = (v0.{f} + v1.{f});"))
            );
            assert!(k.source.contains(&format!("vo.{f} = tmp0;")));
        }
    }

    #[test]
    fn bf16_v8_packs_with_bfloat162() {
        let k = generate(
            &add_op(&[ElementKind::Bf16]),
            &binary_key(ElementKind::Bf16),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_add_bf16_co_v8");
        assert!(k.source.contains("#include <cuda_bf16.h>"));
        assert!(k.source.contains("__nv_bfloat162 a, b, c, d;"));
        assert!(k.source.contains("__nv_bfloat162 tmp0 = (v0.a + v1.a);"));
    }

    #[test]
    fn f16_const_body_stays_scalar_for_bit_exactness() {
        use crate::ir::konst;
        // A Const participates in double-promoted math on the scalar path
        // (`__half + 1.5` promotes through float to double); a packed pair splat
        // would pre-round it to f16 and change bits → gated to scalar.
        let op = OpDef::elementwise("addk", 1, &[ElementKind::F16], input(0) + konst(1.5));
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F16, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_addk_f16_scalar");
        assert!(k.source.contains("out[i] = (in0[i] + 1.5);"));
    }

    #[test]
    fn f16_align4_packs_single_pair_v2() {
        // 4-byte alignment fails V8 (16B) and V4 (8B) but admits V2 (4B): one
        // __half2 pair per access, struct align 4, no second lane.
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F16, 4);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let k = generate(&add_op(&[ElementKind::F16]), &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_add_f16_co_v2");
        assert!(
            k.source
                .contains("struct __align__(4) baracuda_gen_add_f16_co_v2_vec { __half2 a; };")
        );
        assert!(!k.source.contains("vo.b"));
    }

    #[test]
    fn precision_first_variant_offered_for_f32_sum() {
        use crate::ir::ReduceOp;
        use crate::{VariantFidelity, generate_variants};
        // Last-axis (empty-axes default) f32 Sum: base = InnerContig block-tree,
        // prec = the serial double general nest (split-K declines — not Outer).
        let op = OpDef::reduction("sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        let a = OperandDesc::new(2, &[128, 4096], &[4096, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[128], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let vs = generate_variants(&op, &key, &Cuda);
        assert_eq!(vs[0].tag, "base");
        assert_eq!(vs[0].fidelity, VariantFidelity::BitIdentical);
        assert!(
            vs[0].kernels[0].source.contains("block_sum"),
            "base keeps the coalesced block-tree"
        );
        let prec = vs
            .iter()
            .find(|v| v.tag == "prec")
            .expect("precision-first variant offered for f32 Sum");
        assert_eq!(prec.fidelity, VariantFidelity::MorePrecise);
        // A serial double fold is reproducible across IEEE-754 hardware.
        assert_eq!(prec.fidelity.determinism_str(), "bitwise");
        assert_eq!(prec.kernels.len(), 1);
        let k = &prec.kernels[0];
        assert!(
            k.source.contains("double acc"),
            "precision fold accumulates in double"
        );
        assert!(
            !k.source.contains("block_sum"),
            "precision routes through the serial nest, not the block-tree"
        );
        assert!(
            k.name.ends_with("_prec"),
            "distinct _prec entry point: {}",
            k.name
        );
    }

    #[test]
    fn precision_first_variant_declines_non_f32_and_order_exact() {
        use crate::generate_variants;
        use crate::ir::ReduceOp;
        let vs_of = |op: &OpDef, dt: ElementKind| {
            let a = OperandDesc::new(2, &[128, 4096], &[4096, 1], dt, 256);
            let o = OperandDesc::new(1, &[128], &[1], dt, 256);
            let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
            generate_variants(op, &key, &Cuda)
        };
        // f16 Sum: declined — the low-mantissa store rounds the extra accumulation
        // precision away, so a double variant isn't worth a distinct offer.
        let vs = vs_of(
            &OpDef::reduction("sum", 1, &[ElementKind::F16], input(0), ReduceOp::Sum),
            ElementKind::F16,
        );
        assert!(
            vs.iter().all(|v| v.tag != "prec"),
            "f16 declines precision-first"
        );
        // f32 Max: declined — a pick is order-exact, so the double fold is
        // bit-identical to the base (no distinct precision offer).
        let vs = vs_of(
            &OpDef::reduction("max", 1, &[ElementKind::F32], input(0), ReduceOp::Max),
            ElementKind::F32,
        );
        assert!(
            vs.iter().all(|v| v.tag != "prec"),
            "Max declines precision-first"
        );
    }

    #[test]
    fn precision_first_rowreduce_offered_for_f32_softmax() {
        use crate::{VariantFidelity, generate_variants};
        // Numerically-stable f32 softmax: max stage + exp-sum stage + divide
        // epilogue. The exp-sum denominator is the length-growing FP sum the
        // serial double fold fixes.
        let softmax = softmax_op(ElementKind::F32);
        let x = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[x, x], ArchSku::Sm89);
        let vs = generate_variants(&softmax, &key, &Cuda);
        // Base keeps the coalesced block-per-row tree.
        assert_eq!(vs[0].tag, "base");
        assert_eq!(vs[0].fidelity, VariantFidelity::BitIdentical);
        assert!(
            vs[0].kernels[0].source.contains("block_sum"),
            "base keeps the block-tree"
        );
        let prec = vs
            .iter()
            .find(|v| v.tag == "prec")
            .expect("precision-first rowreduce offered for f32 softmax");
        assert_eq!(prec.fidelity, VariantFidelity::MorePrecise);
        // A serial double fold is reproducible across IEEE-754 hardware.
        assert_eq!(prec.fidelity.determinism_str(), "bitwise");
        assert_eq!(prec.kernels.len(), 1);
        let k = &prec.kernels[0];
        assert!(
            k.source.contains("double acc"),
            "precision fold accumulates in double"
        );
        // Serial per-row fold (one thread per row), NOT the block-strided lane loop.
        assert!(
            k.source.contains("for (long long j = 0; j < k; ++j)"),
            "serial per-row fold"
        );
        assert!(
            !k.source.contains("block_sum"),
            "precision routes through the serial fold, not the block-tree"
        );
        assert!(
            k.name.ends_with("_prec"),
            "distinct _prec entry point: {}",
            k.name
        );
    }

    #[test]
    fn precision_first_rowreduce_declines_f16_and_max_only() {
        use crate::generate_variants;
        use crate::ir::{ReduceOp, ReduceStage, reduced};
        // f16 softmax: declined — the low-mantissa store rounds the wider
        // accumulation away, so a double rowreduce variant isn't worth an offer.
        let sm16 = softmax_op(ElementKind::F16);
        let x16 = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F16, 256);
        let key16 = structure_key(OpCategory::Softmax, &[x16, x16], ArchSku::Sm89);
        assert!(
            generate_variants(&sm16, &key16, &Cuda)
                .iter()
                .all(|v| v.tag != "prec"),
            "f16 rowreduce declines precision-first"
        );
        // Pure Max-only rowreduce (a row shift-by-max): order-exact per row, so the
        // double fold is bit-identical to the base — declined.
        let rowmax = OpDef::row_reduce(
            "rowmax",
            1,
            &[ElementKind::F32],
            vec![ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Max,
            }],
            input(0) - reduced(0),
        );
        let xf = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let keyf = structure_key(OpCategory::Softmax, &[xf, xf], ArchSku::Sm89);
        assert!(
            generate_variants(&rowmax, &keyf, &Cuda)
                .iter()
                .all(|v| v.tag != "prec"),
            "Max-only rowreduce declines precision-first (order-exact)"
        );
    }

    #[test]
    fn splitk_variant_offered_for_outer_sum() {
        use crate::ir::ReduceOp;
        use crate::{VariantFidelity, generate_variants};
        use baracuda_kernel_vocab::AxisMask;
        let op = OpDef::reduction_axes(
            "sum",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b1),
            false,
        );
        let a = OperandDesc::new(2, &[8192, 8192], &[8192, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[8192], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let vs = generate_variants(&op, &key, &Cuda);
        // base + splitk + prec (an Outer f32 Sum also earns the precision variant).
        assert_eq!(vs.len(), 3, "base + splitk + prec");
        assert_eq!(vs[0].tag, "base");
        assert_eq!(vs[0].fidelity, VariantFidelity::BitIdentical);
        assert_eq!(vs[2].tag, "prec");
        let sk = &vs[1];
        assert_eq!(sk.tag, "splitk");
        assert_eq!(sk.fidelity, VariantFidelity::ReassociatedDeterministic);
        assert_eq!(sk.kernels.len(), 2, "partial + combine, in launch order");
        // Partial: coalesced (adjacent threads, adjacent columns), chunked rows,
        // one workspace row per chunk. No atomics anywhere.
        let p = &sk.kernels[0];
        assert!(p.name.ends_with("_splitk_partial"));
        assert!(p.source.contains("long long idx = r * cols + c;"));
        assert!(
            p.source
                .contains("long long r1 = r0 + chunk_rows; if (r1 > rows) r1 = rows;")
        );
        assert!(
            p.source
                .contains("ws[(long long)blockIdx.y * cols + c] = acc;")
        );
        assert!(!p.source.contains("atomic"));
        // Combine: fixed chunk-order fold, no Mean divisor for Sum.
        let c = &sk.kernels[1];
        assert!(c.name.ends_with("_splitk_combine"));
        // Seeded from the first partial (not 0): keeps a -0.0 column total's
        // sign and makes the degenerate n_chunks=1 case bit-identical.
        assert!(c.source.contains("float acc = ws[c];"));
        assert!(
            c.source
                .contains("for (long long k = 1; k < n_chunks; ++k) acc += ws[k * cols + c];")
        );
        assert!(!c.source.contains("/ (float)rows"));
        assert!(sk.launch_note.contains("chunk_rows = ceil(rows/n_chunks)"));
    }

    #[test]
    fn splitk_mean_divides_in_combine_only() {
        use crate::generate_variants;
        use crate::ir::ReduceOp;
        use baracuda_kernel_vocab::AxisMask;
        let op = OpDef::reduction_axes(
            "mean",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Mean,
            AxisMask(0b1),
            false,
        );
        let a = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let vs = generate_variants(&op, &key, &Cuda);
        assert_eq!(vs.len(), 3, "base + splitk + prec");
        let sk = &vs[1];
        assert_eq!(sk.tag, "splitk");
        assert!(
            !sk.kernels[0].source.contains("acc /"),
            "partial never divides"
        );
        assert!(sk.kernels[1].source.contains("out[c] = acc / (float)rows;"));
    }

    #[test]
    fn contraction_emits_skinny_simt_kernel() {
        use crate::ir::{ContractionAxes, UnaryOp, reduced};
        // The decode / flat-GEMM cell: [8,4096]·[4096,4096] → [8,4096], f32.
        let mm = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        );
        let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let k = generate(&mm, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_matmul_f32_contract_tll");
        // Coalesced rhs stream (adjacent threads, adjacent columns)…
        assert!(k.source.contains("float w = in1[kk * n + col];"));
        // …predicated register accumulators at the Tiny ceiling…
        assert!(k.source.contains("float accs[8];"));
        assert!(
            k.source
                .contains("if (mm < m) accs[mm] += in0[mm * k + kk] * w;")
        );
        // …identity epilogue stores the K-sum.
        assert!(k.source.contains("out[mm * n + col] = r0;"));
        assert!(!k.source.contains("atomic"));

        // A relu epilogue lowers over Reduced(0) in the accumulator width.
        let mr = OpDef::contraction(
            "matmul_relu",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0).unary(UnaryOp::Relu),
        );
        let kr = generate(&mr, &key, &Cuda);
        assert!(kr.source.contains("< 0.0f ? 0.0f :"));

        // f16 loads up-convert and the store narrows, as the reductions do.
        let lh = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F16, 256);
        let rh = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F16, 256);
        let oh = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F16, 256);
        let mh = OpDef::contraction(
            "matmul",
            &[ElementKind::F16],
            ContractionAxes::matmul(),
            reduced(0),
        );
        let kh = generate(
            &mh,
            &structure_key(OpCategory::Gemm, &[lh, rh, oh], ArchSku::Sm89),
            &Cuda,
        );
        assert!(
            kh.source
                .contains("float w = __half2float(in1[kk * n + col]);")
        );
        assert!(kh.source.contains("out[mm * n + col] = __float2half(r0);"));
    }

    #[test]
    fn contraction_bias_emits_bias_param_and_column_load() {
        use crate::generate_variants;
        use crate::ir::{ContractionAxes, UnaryOp, input, reduced};
        // Fused matmul + per-column bias + relu: out = relu(Σ_k lhs·rhs + bias[n]).
        let mb = OpDef::contraction_bias(
            "matmul_bias_relu",
            &[ElementKind::F32],
            (reduced(0) + input(2)).unary(UnaryOp::Relu),
        );
        let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
        let bias = OperandDesc::new(1, &[4096], &[1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, bias, out], ArchSku::Sm89);
        let k = generate(&mb, &key, &Cuda);
        // The bias input pointer is a kernel param…
        assert!(
            k.source.contains("const float* __restrict__ in2,"),
            "missing bias param:\n{}",
            k.source
        );
        // …and the epilogue adds it per column, then relu-clamps the K-sum.
        assert!(
            k.source.contains("in2[col]"),
            "missing bias load:\n{}",
            k.source
        );
        assert!(
            k.source.contains("< 0.0f ? 0.0f :"),
            "missing relu over (r0 + bias):\n{}",
            k.source
        );
        // The split-K variant declines for a bias cell (folded-epilogue combine is
        // a follow-up) — only the base kernel is offered.
        let vs = generate_variants(&mb, &key, &Cuda);
        assert_eq!(
            vs.len(),
            1,
            "bias cell must offer only the base kernel (split-K declined)"
        );

        // The plain (no-bias) contraction emits NO in2 — byte-compat unchanged.
        let mm = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        );
        let kp = generate(
            &mm,
            &structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89),
            &Cuda,
        );
        assert!(
            !kp.source.contains("in2"),
            "plain contraction must not emit a bias param:\n{}",
            kp.source
        );
    }

    #[test]
    fn contraction_batched_emits_batch_offset_and_declines_splitk() {
        use crate::generate_variants;
        use crate::ir::{ContractionAxes, reduced};
        // Batched [8,8,4096]·[8,4096,4096] → [8,8,4096], f32 (B/M Tiny).
        let bmm = OpDef::contraction(
            "bmm",
            &[ElementKind::F32],
            ContractionAxes::batched_matmul(),
            reduced(0),
        );
        let lhs = OperandDesc::new(
            3,
            &[8, 8, 4096],
            &[8 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let rhs = OperandDesc::new(
            3,
            &[8, 4096, 4096],
            &[4096 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let out = OperandDesc::new(
            3,
            &[8, 8, 4096],
            &[8 * 4096, 4096, 1],
            ElementKind::F32,
            256,
        );
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let k = generate(&bmm, &key, &Cuda);
        // The batch index + per-operand batch strides (b = blockIdx.z).
        assert!(
            k.source.contains("long long b = (long long)blockIdx.z;"),
            "missing batch index:\n{}",
            k.source
        );
        assert!(
            k.source.contains("in1[b * k * n + kk * n + col]"),
            "missing rhs batch stride:\n{}",
            k.source
        );
        assert!(
            k.source.contains("in0[b * m * k + mm * k + kk]"),
            "missing lhs batch stride:\n{}",
            k.source
        );
        assert!(
            k.source.contains("out[b * m * n + mm * n + col]"),
            "missing out batch stride:\n{}",
            k.source
        );
        // The batched token carries the batch size class additively (`/b<class>`).
        assert!(
            key.to_token().contains("/bt"),
            "batched token needs /b<class>: {}",
            key.to_token()
        );
        // The split-K variant declines for a batched cell (base kernel only).
        let vs = generate_variants(&bmm, &key, &Cuda);
        assert_eq!(vs.len(), 1, "batched cell offers only the base kernel");

        // A plain rank-2 cell emits NO batch index — byte-compat unchanged.
        let mm = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        );
        let l2 = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let r2 = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
        let o2 = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let kp = generate(
            &mm,
            &structure_key(OpCategory::Gemm, &[l2, r2, o2], ArchSku::Sm89),
            &Cuda,
        );
        assert!(
            !kp.source.contains("blockIdx.z"),
            "plain contraction must not emit a batch index:\n{}",
            kp.source
        );
    }

    #[test]
    fn contraction_splitk_variant_offered_for_tiny_m_cell() {
        use crate::ir::{ContractionAxes, reduced};
        use crate::{VariantFidelity, generate_variants};
        let mm = OpDef::contraction(
            "matmul",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0),
        );
        let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let vs = generate_variants(&mm, &key, &Cuda);
        assert_eq!(vs.len(), 2, "base + splitk");
        let sk = &vs[1];
        assert_eq!(sk.tag, "splitk");
        assert_eq!(sk.fidelity, VariantFidelity::ReassociatedDeterministic);
        assert_eq!(sk.kernels.len(), 2);
        let p = &sk.kernels[0];
        assert!(p.name.ends_with("_contract_tll_splitk_partial"));
        assert!(
            p.source
                .contains("long long k1 = k0 + chunk_k; if (k1 > k) k1 = k;")
        );
        assert!(
            p.source
                .contains("if (mm < m) ws[((long long)blockIdx.y * m + mm) * n + col] = accs[mm];")
        );
        assert!(!p.source.contains("atomic"));
        let c = &sk.kernels[1];
        assert!(c.name.ends_with("_contract_tll_splitk_combine"));
        // Seeded from chunk 0 → degenerate n_chunks=1 is bit-identical to base.
        assert!(c.source.contains("float r0 = ws[(long long)mm * n + col];"));
        assert!(c.source.contains(
            "for (long long ch = 1; ch < n_chunks; ++ch) r0 += ws[(ch * m + mm) * n + col];"
        ));
        assert!(c.source.contains("out[mm * n + col] = r0;"));
        assert!(sk.launch_note.contains("chunk_k = ceil(k/n_chunks)"));
    }

    #[test]
    fn smemrow_variant_materializes_softmax_shared_exp() {
        use crate::ir::{ReduceOp, ReduceStage, reduced};
        use crate::{VariantFidelity, generate_variants};
        // Softmax: stage-2 pre = exp(x - r0) is recomputed verbatim in the
        // epilogue — the cross-pass shared value the variant caches.
        let softmax = OpDef::row_reduce(
            "softmax",
            1,
            &[ElementKind::F32],
            vec![
                ReduceStage {
                    pre: input(0).0,
                    op: ReduceOp::Max,
                },
                ReduceStage {
                    pre: (input(0) - reduced(0)).exp().0,
                    op: ReduceOp::Sum,
                },
            ],
            (input(0) - reduced(0)).exp() / reduced(1),
        );
        let x = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[x, x], ArchSku::Sm89);
        let vs = generate_variants(&softmax, &key, &Cuda);
        // An f32 softmax now also earns the precision-first variant, so this cell
        // yields base + prec + smemrow; select the materialize offer by TAG.
        let sm = vs
            .iter()
            .find(|v| v.tag == "smemrow")
            .expect("smemrow variant offered");
        assert_eq!(sm.tag, "smemrow");
        assert_eq!(sm.fidelity, VariantFidelity::BitIdentical);
        let src = &sm.kernels[0].source;
        assert!(sm.kernels[0].name.ends_with("_rowreduce_smemrow"));
        assert!(src.contains("extern __shared__ float baracuda_row_smem[];"));
        // The fold caches the per-element value it accumulates…
        assert!(src.contains("baracuda_row_smem[j] = v;"));
        // …and the epilogue reads the cache instead of recomputing: exactly ONE
        // expf remains (the stage fold's), vs two in the base kernel.
        assert!(src.contains("out[idx] = (baracuda_row_smem[j] / r1);"));
        assert_eq!(src.matches("expf").count(), 1, "epilogue exp eliminated");
        assert_eq!(
            vs[0].kernels[0].source.matches("expf").count(),
            2,
            "base has both"
        );
        // Helper symbols must not collide when base + variant share a TU.
        assert!(src.contains("block_sum_softmax_f32_sm"));
        assert!(vs[0].kernels[0].source.contains("block_sum_softmax_f32("));
        // Sanity: the base kernel is untouched by the variant machinery.
        assert!(!vs[0].kernels[0].source.contains("baracuda_row_smem"));
    }

    #[test]
    fn smemrow_not_offered_when_epilogue_does_not_recompute() {
        use crate::generate_variants;
        use crate::ir::{ReduceOp, ReduceStage, UnaryOp, konst, reduced};
        // RmsNorm: stage pre = x², epilogue = x·rsqrt(r0+eps) — the epilogue
        // never recomputes x², so there is nothing to materialize.
        let rmsnorm = OpDef::row_reduce(
            "rmsnorm",
            1,
            &[ElementKind::F32],
            vec![ReduceStage {
                pre: input(0).unary(UnaryOp::Sqr).0,
                op: ReduceOp::Mean,
            }],
            input(0) * (reduced(0) + konst(1e-5)).unary(UnaryOp::Rsqrt),
        );
        let x = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Normalization, &[x, x], ArchSku::Sm89);
        // The precision variant IS offered (an f32 Mean rowreduce), but smemrow is
        // NOT — the epilogue never recomputes a stage expr, so nothing to cache.
        let vs = generate_variants(&rmsnorm, &key, &Cuda);
        assert!(
            vs.iter().all(|v| v.tag != "smemrow"),
            "no materialize offer when the epilogue doesn't recompute"
        );
    }

    #[test]
    fn splitk_gate_refuses_flipped_param_and_malformed_cells() {
        use crate::ir::ReduceOp;
        use crate::{Backend, generate_variants};
        use baracuda_kernel_vocab::AxisMask;
        let op = OpDef::reduction_axes(
            "sum",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b1),
            false,
        );
        let o = OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256);
        // Row-reversed dense view: keys Contig + flipped — the baseline serves
        // it via runtime strides, but the stride-free split-K ABI would read
        // out of bounds. The gate must refuse.
        let rev = OperandDesc::new(2, &[4096, 1024], &[-1024, 1], ElementKind::F32, 256);
        let k_rev = structure_key(OpCategory::Reduction, &[rev, o], ArchSku::Sm89);
        // split-K refuses (its stride-free ABI can't address a reversed view); the
        // precision variant CAN (the general nest reads runtime strides), so assert
        // the split-K gate specifically, not a raw count.
        assert!(
            generate_variants(&op, &k_rev, &Cuda)
                .iter()
                .all(|v| v.tag != "splitk"),
            "flipped input: no split-K"
        );
        // Param body: the fixed splitk signature has no p{i} slot — refused
        // (the emitted source would reference an undefined identifier).
        let wsum = OpDef::reduction_axes(
            "wsum",
            1,
            &[ElementKind::F32],
            input(0) * crate::ir::param(0),
            ReduceOp::Sum,
            AxisMask(0b1),
            false,
        );
        let a = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let k_ok = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        // split-K's fixed signature has no p{i} slot; the precision variant rides
        // the general nest, which DOES thread params — so gate on the split-K tag.
        assert!(
            generate_variants(&wsum, &k_ok, &Cuda)
                .iter()
                .all(|v| v.tag != "splitk"),
            "param body: no split-K"
        );
        // Malformed 1-operand key: out_key would alias the input — refused.
        // (Exercised via lower_variants directly; generate() itself would also
        // reject such a key downstream.)
        let k_one = structure_key(OpCategory::Reduction, &[a], ArchSku::Sm89);
        let plan = crate::build_plan(&op, &k_one);
        assert!(
            Cuda.lower_variants(&plan).is_empty(),
            "1-operand key: no variant"
        );
    }

    #[test]
    fn splitk_not_offered_for_lastaxis_max_or_int() {
        use crate::generate_variants;
        use crate::ir::ReduceOp;
        use baracuda_kernel_vocab::AxisMask;
        // Last-axis (InnerContig): already block-parallel — no split-K.
        let last = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        let a = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[4096], &[1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        // Last-axis (InnerContig) is already block-parallel → no split-K (the
        // precision variant IS offered here, so gate on the split-K tag).
        assert!(
            generate_variants(&last, &k, &Cuda)
                .iter()
                .all(|v| v.tag != "splitk"),
            "last-axis: no split-K"
        );
        // Outer Max: the has-flag NaN fold needs its own variant treatment.
        let mx = OpDef::reduction_axes(
            "amax",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Max,
            AxisMask(0b1),
            false,
        );
        let ao = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let oo = OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256);
        let km = structure_key(OpCategory::Reduction, &[ao, oo], ArchSku::Sm89);
        assert_eq!(generate_variants(&mx, &km, &Cuda).len(), 1, "base only");
        // Outer i32 Sum: int workspace is a follow-up.
        let is = OpDef::reduction_axes(
            "sum",
            1,
            &[ElementKind::I32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b1),
            false,
        );
        let ai = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::I32, 256);
        let oi = OperandDesc::new(1, &[1024], &[1], ElementKind::I32, 256);
        let ki = structure_key(OpCategory::Reduction, &[ai, oi], ArchSku::Sm89);
        assert_eq!(generate_variants(&is, &ki, &Cuda).len(), 1, "base only");
    }

    #[test]
    fn strided_2d_unravels() {
        // Transposed (column-major) views: all operands strided, none broadcast.
        let t = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[t, t, t], ArchSku::Sm89);
        let k = generate(&add_op(&[ElementKind::F32]), &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_add_f32_strided_r2");
        assert!(
            k.source
                .contains("long long c1 = lin % shape1; lin /= shape1;")
        );
        assert!(k.source.contains("long long o0 = c0*s0_0 + c1*s0_1;"));
        assert!(k.source.contains("out[oo] = (in0[o0] + in1[o1]);"));
    }

    // ---- item 01: layout views (fused transpose-elementwise) ----

    // A rank-2 contiguous [128,256] f32 cell (1 input + 1 output) that would
    // VECTORIZE view-free — used to prove the view both forces Strided and remaps
    // the input offset.
    fn view_2d_key(n_operands: usize) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let ops: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        let cat = if n_operands >= 3 {
            OpCategory::BinaryElementwise
        } else {
            OpCategory::UnaryElementwise
        };
        structure_key(cat, &ops, ArchSku::Sm89)
    }

    #[test]
    fn transpose_fused_elementwise_reads_input_through_swapped_strides() {
        use crate::ir::View;
        // out[i,j] = relu(x[j,i]): input 0 read through Permute{[1,0]}.
        let op = OpDef::elementwise("relu_t", 1, &[ElementKind::F32], input(0).relu())
            .with_views(vec![View::Permute { perm: vec![1, 0] }]);
        let k = generate(&op, &view_2d_key(2), &Cuda);
        // Forced onto the strided schedule (a transposed read is non-contiguous),
        // NOT the float4 vectorized path it would take view-free.
        assert_eq!(k.name, "baracuda_gen_relu_t_f32_strided_r2");
        // The KEY of the increment: iteration axis d reads producer stride perm[d]
        // ⇒ `c0*s0_1 + c1*s0_0` (swapped), NOT the identity `c0*s0_0 + c1*s0_1`.
        assert!(
            k.source.contains("long long o0 = c0*s0_1 + c1*s0_0;"),
            "the transposed input offset must use SWAPPED strides"
        );
        assert!(!k.source.contains("long long o0 = c0*s0_0 + c1*s0_1;"));
        // The output offset is unaffected (views are an input-read property).
        assert!(k.source.contains("long long oo = c0*so_0 + c1*so_1;"));
        assert!(k.source.contains("out[oo] = ("));
    }

    #[test]
    fn two_input_view_transposed_plus_identity() {
        use crate::ir::View;
        // out[i,j] = x[j,i] + b[i,j]: in0 transposed, in1 identity.
        let op = OpDef::elementwise("add_t", 2, &[ElementKind::F32], input(0) + input(1))
            .with_views(vec![View::Permute { perm: vec![1, 0] }, View::Identity]);
        let k = generate(&op, &view_2d_key(3), &Cuda);
        assert_eq!(k.name, "baracuda_gen_add_t_f32_strided_r2");
        assert!(k.source.contains("long long o0 = c0*s0_1 + c1*s0_0;")); // transposed in0
        assert!(k.source.contains("long long o1 = c0*s1_0 + c1*s1_1;")); // identity in1
        assert!(k.source.contains("out[oo] = (in0[o0] + in1[o1]);"));
    }

    #[test]
    fn rank3_nontrivial_permute_uses_the_direct_stride_remap() {
        use crate::ir::View;
        // Review #3: every other perm golden uses [1,0], an INVOLUTION
        // (perm == perm^-1), so no rank-2 test can distinguish the direct remap
        // (c{d}*stride[perm[d]]) from the inverse — an inverse mutation passed all
        // 282 tests. A rank-3 NON-involutive perm [2,0,1] pins the direction:
        // iteration axis d reads producer axis perm[d], so producer stride at
        // perm[d]: d0->s_2, d1->s_0, d2->s_1.
        let a = OperandDesc::new(3, &[4, 8, 16], &[128, 16, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        let op =
            OpDef::elementwise("relu_p", 1, &[ElementKind::F32], input(0).relu()).with_views(vec![
                View::Permute {
                    perm: vec![2, 0, 1],
                },
            ]);
        let k = generate(&op, &key, &Cuda);
        // DIRECT remap (mathematically correct: producer stride at perm[d]).
        assert!(
            k.source
                .contains("long long o0 = c0*s0_2 + c1*s0_0 + c2*s0_1;"),
            "rank-3 perm [2,0,1] must use the DIRECT remap; got:\n{}",
            k.source
        );
        // The INVERSE remap (the mutation this test exists to catch) must NOT appear.
        assert!(
            !k.source
                .contains("long long o0 = c0*s0_1 + c1*s0_2 + c2*s0_0;")
        );
        // Identity output offset, unaffected by the input view.
        assert!(
            k.source
                .contains("long long oo = c0*so_0 + c1*so_1 + c2*so_2;")
        );
    }

    #[test]
    fn identity_view_is_byte_identical_to_view_free() {
        use crate::ir::View;
        // The byte-identical guarantee: an all-Identity views vec emits the exact
        // same source (and name) as the view-free op at the same key.
        let key = view_2d_key(3);
        let free = generate(&add_op(&[ElementKind::F32]), &key, &Cuda);
        let ident = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1))
            .with_views(vec![View::Identity, View::Identity]);
        let viewed = generate(&ident, &key, &Cuda);
        assert_eq!(free.name, viewed.name);
        assert_eq!(
            free.source, viewed.source,
            "an all-Identity view must emit byte-identical source to view-free"
        );
    }

    #[test]
    #[should_panic(expected = "must lower on the Strided schedule")]
    fn viewed_op_on_a_non_strided_schedule_is_refused_by_the_backstop() {
        use crate::backend::Backend;
        use crate::ir::View;
        use crate::plan::{KernelPlan, Schedule};
        // Construct a plan manually that pairs a Permute view with the Vectorized
        // schedule (which build_plan would never produce) — the independent emitter
        // backstop must refuse it (the vector emitter ignores views).
        let key = view_2d_key(2);
        let body = input(0).relu().0;
        let views = [View::Permute { perm: vec![1, 0] }];
        let plan = KernelPlan {
            op_name: "sneaky",
            n_inputs: 1,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::Vectorized { width: 4 },
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &views,
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    fn shared_interior_is_hoisted_to_one_tmp() {
        use crate::ir::konst;
        // g = a*b; out = g / (g + 1). The shared product must be emitted ONCE as a
        // named tmp and referenced twice — not re-rendered (the recompute + source
        // blow-up the DAG rewrite exists to kill).
        let g = input(0) * input(1);
        let op = OpDef::elementwise(
            "diamond",
            2,
            &[ElementKind::F32],
            g.clone() / (g + konst(1.0)),
        );
        // A transposed (strided) key routes through emit_strided (plain float infix).
        let t = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[t, t, t], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(
            k.source.matches("in0[o0] * in1[o1]").count(),
            1,
            "the shared product is emitted exactly once"
        );
        assert!(
            k.source.contains("float tmp0 = (in0[o0] * in1[o1]);"),
            "hoisted to a tmp"
        );
        assert!(
            k.source.contains("out[oo] = (tmp0 / (tmp0 + 1.0));"),
            "and referenced twice, not recomputed"
        );
    }

    #[test]
    fn single_use_body_emits_no_tmp() {
        // Transparency guard: a body with no shared interior must produce zero
        // `tmp` declarations (byte-identical to the pre-DAG emitter).
        let k = generate(
            &add_op(&[ElementKind::F32]),
            &binary_key(ElementKind::F32),
            &Cuda,
        );
        assert!(
            !k.source.contains("tmp"),
            "no hoisting for a single-use tree"
        );
    }

    #[test]
    fn fully_broadcast_operand_is_hoisted() {
        let a = OperandDesc::new(2, &[4, 8], &[8, 1], ElementKind::F32, 256);
        let b = OperandDesc::new(2, &[4, 8], &[0, 0], ElementKind::F32, 256); // scalar broadcast
        let out = OperandDesc::new(2, &[4, 8], &[8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, b, out], ArchSku::Sm89);
        let k = generate(&add_op(&[ElementKind::F32]), &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_add_f32_strided_r2");
        assert!(k.source.contains("float h1 = in1[0];")); // hoisted invariant load
        assert!(k.source.contains("out[oo] = (in0[o0] + h1);")); // uses the register
        assert!(!k.source.contains("long long o1 =")); // no per-iter offset for in1
    }

    #[test]
    fn fma_lane_splat_for_nested_expr() {
        let fma = OpDef::elementwise(
            "fma",
            3,
            &[ElementKind::F32],
            input(0) * input(1) + input(2),
        );
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::TernaryElementwise, &[a, a, a, a], ArchSku::Sm89);
        let k = generate(&fma, &key, &Cuda);
        assert!(k.source.contains("vo.y = ((v0.y * v1.y) + v2.y);"));
    }

    #[test]
    fn f32_unary_relu_lowers() {
        // y = relu(a + b), contiguous f32 V4.
        let op = OpDef::elementwise(
            "relu_add",
            2,
            &[ElementKind::F32],
            (input(0) + input(1)).relu(),
        );
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        // NaN-propagating relu (select, not fmaxf).
        assert!(
            k.source
                .contains("vo.x = ((v0.x + v1.x) < 0.0f ? 0.0f : (v0.x + v1.x));")
        );
    }

    #[test]
    fn f16_packed_tier_b_scalarizes_relu() {
        // relu has no bit-safe packed intrinsic → Tier B: split the pair, run
        // the EXISTING float relu spelling per half (identical text ⇒ identical
        // bits vs the scalar kernel), re-join. The add stays Tier A (native
        // __half2 operator+). Never the NaN-suppressing __hmax2.
        let op = OpDef::elementwise(
            "relu_add",
            2,
            &[ElementKind::F16],
            (input(0) + input(1)).relu(),
        );
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F16, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_relu_add_f16_co_v8");
        assert!(k.source.contains("__half2 tmp0 = (v0.a + v1.a);")); // Tier A, hoisted
        assert!(k.source.contains("__halves2half2(")); // Tier B re-join
        assert!(k.source.contains("__low2half(tmp0)"));
        assert!(k.source.contains("__high2half(tmp0)"));
        assert!(k.source.contains("< 0.0f ? 0.0f :")); // NaN-propagating relu, float, per half
        assert!(!k.source.contains("__hmax2"));
    }

    #[test]
    fn scalar_param_kernel() {
        // y = relu(x * p0 + p1) — one input, two runtime scalar params.
        let op = OpDef::elementwise(
            "affine_relu",
            1,
            &[ElementKind::F32],
            (input(0) * param(0) + param(1)).relu(),
        );
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert!(k.source.contains("long long nv, float p0, float p1)"));
        assert!(k.source.contains("((v0.x * p0) + p1)"));
    }

    fn reduce_key(in_dt: ElementKind) -> baracuda_kernel_vocab::StructureKey {
        // [256, 128] contiguous input, [256] output — reduce the last axis.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], in_dt, 256);
        let out = OperandDesc::new(1, &[256], &[1], in_dt, 256);
        structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89)
    }

    #[test]
    fn reduction_mean_of_squares_f32() {
        use crate::ir::{ReduceOp, UnaryOp};
        // mean(x²) over the last axis — the RmsNorm core.
        let op = OpDef::reduction(
            "ms",
            1,
            &[ElementKind::F32],
            input(0).unary(UnaryOp::Sqr),
            ReduceOp::Mean,
        );
        let k = generate(&op, &reduce_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_ms_f32_reduce_mean");
        assert!(k.source.contains("long long n_out,"));
        assert!(k.source.contains("    long long k)")); // runtime reduced extent
        // Block-per-row cooperative reduce: coalesced loop + block_sum + thread-0 store.
        assert!(
            k.source
                .contains("for (long long row = blockIdx.x; row < n_out;")
        );
        assert!(
            k.source
                .contains("for (long long j = threadIdx.x; j < k; j += blockDim.x)")
        );
        assert!(k.source.contains("acc += (in0[idx]*in0[idx]);"));
        assert!(
            k.source
                .contains("float r = block_sum_ms_f32(acc) / (float)k;")
        );
        assert!(k.source.contains("if (threadIdx.x == 0) out[row] = r;"));
        assert!(!k.source.contains("base = o * k")); // not the old uncoalesced sequential
    }

    #[test]
    fn reduction_max_peels_first_and_propagates_nan() {
        use crate::ir::ReduceOp;
        let op = OpDef::reduction("amax", 1, &[ElementKind::F32], input(0), ReduceOp::Max);
        let k = generate(&op, &reduce_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_amax_f32_reduce_max");
        assert!(!k.source.contains("INFINITY")); // has-flag seeding, no ±inf literal
        assert!(k.source.contains("float acc = 0.0f; int has = 0;"));
        assert!(
            k.source
                .contains("for (long long j = threadIdx.x; j < k; j += blockDim.x)")
        );
        // NaN-propagating select (e != e forces the swap), torch.amax semantics.
        assert!(
            k.source
                .contains("if (!has || e != e || e > acc) { acc = e; has = 1; }")
        );
        assert!(k.source.contains("float r = block_max_amax_f32(acc, has);"));
        assert!(k.source.contains("if (threadIdx.x == 0) out[row] = r;"));
    }

    #[test]
    fn reduction_f16_accumulates_in_float() {
        use crate::ir::ReduceOp;
        // f16 input/output, but the fold runs in float — precision + no __half2 sum.
        let op = OpDef::reduction("s", 1, &[ElementKind::F16], input(0), ReduceOp::Sum);
        let k = generate(&op, &reduce_key(ElementKind::F16), &Cuda);
        assert!(k.source.contains("#include <cuda_fp16.h>"));
        assert!(k.source.contains("const __half* __restrict__ in0"));
        assert!(k.source.contains("float acc = 0.0f;")); // float acc, not __half
        assert!(k.source.contains("acc += __half2float(in0[idx]);"));
        assert!(k.source.contains("float r = block_sum_s_f16(acc);"));
        assert!(
            k.source
                .contains("if (threadIdx.x == 0) out[row] = __float2half(r);")
        );
    }

    #[test]
    fn reduction_f32strict_folds_in_double() {
        use crate::ir::ReduceOp;
        // Strict precision mode folds in double (a plain-float fold isn't reproducible
        // / correctly-rounded), then stores the single-rounded result to the f32 out.
        let op = OpDef::reduction("s", 1, &[ElementKind::F32Strict], input(0), ReduceOp::Sum);
        let k = generate(&op, &reduce_key(ElementKind::F32Strict), &Cuda);
        assert!(k.source.contains("double acc = 0.0;"));
        assert!(k.source.contains("acc += (double)in0[idx];"));
        assert!(k.source.contains("double r = block_sum_s_")); // block reduce in double
        assert!(k.source.contains("if (threadIdx.x == 0) out[row] = r;")); // single round to f32 out
    }

    #[test]
    fn reduction_int_accumulates_in_long_long() {
        use crate::ir::ReduceOp;
        // i32 Sum reduces natively into a `long long` accumulator (exact); no float.
        let op = OpDef::reduction("s", 1, &[ElementKind::I32], input(0), ReduceOp::Sum);
        let k = generate(&op, &reduce_key(ElementKind::I32), &Cuda);
        assert!(k.source.contains("const int* __restrict__ in0"));
        assert!(k.source.contains("long long acc = 0;"));
        assert!(k.source.contains("acc += in0[idx];")); // native int load, no float convert
        assert!(k.source.contains("long long r = block_sum_"));
        assert!(k.source.contains("if (threadIdx.x == 0) out[row] = r;"));
        assert!(!k.source.contains("float acc"));
    }

    #[test]
    #[should_panic(expected = "integer Mean is out of scope")]
    fn reduction_int_mean_is_rejected() {
        use crate::ir::ReduceOp;
        // int Mean is float-output (mixed-dtype) — rejected, not silently mis-typed.
        let op = OpDef::reduction("m", 1, &[ElementKind::I32], input(0), ReduceOp::Mean);
        let _ = generate(&op, &reduce_key(ElementKind::I32), &Cuda);
    }

    // ---- item 03: general (outer/middle/multi/keepdim/strided) reduction path ----

    #[test]
    fn reduction_outer_axis_collapses() {
        use crate::ir::ReduceOp;
        use baracuda_kernel_vocab::AxisMask;
        // Reduce axis 0 of a contiguous [4,8] input → collapse to [8].
        let a = OperandDesc::new(2, &[4, 8], &[8, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(1, &[8], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89);
        let op = OpDef::reduction_axes(
            "s",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b01),
            false,
        );
        let k = generate(&op, &key, &Cuda);
        // Distinct symbol carrying the axis set; NOT the fast path.
        assert_eq!(k.name, "baracuda_gen_s_f32_reduce_sum_ax1");
        assert!(!k.source.contains("long long base = o * k;"));
        // Kept-axis unravel, strided base, strided reduced fold, collapse output.
        assert!(
            k.source
                .contains("long long ck1 = lin % shape1; lin /= shape1;")
        );
        assert!(k.source.contains("long long base = ck1*s0_1;"));
        // Innermost reduced walk: int32 counter when the extent fits, with a
        // long long fallback nest; strength-reduced offsets (extraction #2 + #3).
        assert!(k.source.contains("if (shape0 <= 2147483647LL)"));
        assert!(k.source.contains("int ext0 = (int)shape0;"));
        assert!(k.source.contains("for (int cr0 = 0; cr0 < ext0; ++cr0)"));
        assert!(
            k.source
                .contains("for (long long cr0 = 0; cr0 < shape0; ++cr0)")
        ); // fallback
        assert!(
            k.source.contains("long long roff0 = base;")
                && k.source.contains("long long idx = roff0;")
                && k.source.contains("roff0 += s0_0;")
        );
        assert!(k.source.contains("long long oo = ck1*so_0;"));
        assert!(k.source.contains("acc += in0[idx];"));
        assert!(k.source.contains("out[oo] ="));
    }

    #[test]
    fn reduction_multi_axis_mean_divisor_is_the_extent_product() {
        use crate::ir::ReduceOp;
        use baracuda_kernel_vocab::AxisMask;
        // Reduce axes {0,1} of [2,3,4] → [4], Mean: divisor = shape0 * shape1.
        let a = OperandDesc::new(3, &[2, 3, 4], &[12, 4, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(1, &[4], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89);
        let op = OpDef::reduction_axes(
            "m",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Mean,
            AxisMask(0b011),
            false,
        );
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_m_f32_reduce_mean_ax3");
        // Nested reduced loops + the extent-product divisor (not just the last axis).
        assert!(
            k.source
                .contains("for (long long cr0 = 0; cr0 < shape0; ++cr0)")
        ); // outer stays ll
        assert!(k.source.contains("for (int cr1 = 0; cr1 < ext1; ++cr1)")); // innermost int32
        assert!(k.source.contains("if (shape1 <= 2147483647LL)"));
        assert!(
            k.source.contains("long long roff1 = roff0;")
                && k.source.contains("long long idx = roff1;")
                && k.source.contains("roff1 += s0_1;")
        );
        assert!(k.source.contains("acc / (float)(shape0 * shape1)"));
        assert!(k.source.contains("long long base = ck2*s0_2;")); // kept axis 2
    }

    #[test]
    fn reduction_keepdim_outer_axis_uses_input_axis_output_stride() {
        use crate::ir::ReduceOp;
        use baracuda_kernel_vocab::AxisMask;
        // Reduce axis 0 of [4,8] with keepdim → [1,8]: the output stride is indexed
        // by INPUT axis (kept axis 1), not a collapsed position.
        let a = OperandDesc::new(2, &[4, 8], &[8, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[1, 8], &[8, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89);
        let op = OpDef::reduction_axes(
            "s",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b01),
            true,
        );
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_s_f32_reduce_sum_ax1_kd");
        assert!(k.source.contains("long long oo = ck1*so_1;")); // so[input-axis], not so[0]
    }

    #[test]
    fn reduction_general_max_seeds_and_propagates_nan() {
        use crate::ir::ReduceOp;
        use baracuda_kernel_vocab::AxisMask;
        // Max over a non-last axis still uses the NaN-propagating select, seeded via
        // the `has` flag (no ±∞ literal, empty extent leaves acc = 0).
        let a = OperandDesc::new(2, &[4, 8], &[8, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(1, &[8], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89);
        let op = OpDef::reduction_axes(
            "mx",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Max,
            AxisMask(0b01),
            false,
        );
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_mx_f32_reduce_max_ax1");
        assert!(!k.source.contains("INFINITY"));
        assert!(k.source.contains("int has = 0;"));
        assert!(
            k.source
                .contains("acc = has ? ((e != e || e > acc) ? e : acc) : e; has = 1;")
        );
    }

    #[test]
    fn reduction_strided_last_axis_takes_the_general_path() {
        use crate::ir::ReduceOp;
        use baracuda_kernel_vocab::AxisMask;
        // Reduce the last axis of a column-major (transposed, strided) [8,4] input:
        // the trailing axis over a non-contiguous input is NOT the contiguous fast
        // path, so it routes to the strided general fold.
        let a = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::F32, 256);
        let out = OperandDesc::new(1, &[8], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89);
        let op = OpDef::reduction_axes(
            "s",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b10),
            false,
        );
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_s_f32_reduce_sum_ax2");
        assert!(!k.source.contains("long long base = o * k;"));
        assert!(k.source.contains("long long idx = roff0;") && k.source.contains("roff0 += s0_1;")); // strided reduced fold
        assert!(k.source.contains("for (int cr1 = 0; cr1 < ext1; ++cr1)")); // int32 strided walk
        assert!(k.source.contains("long long base = ck0*s0_0;"));
    }

    #[test]
    #[should_panic(expected = "output store must be injective")]
    fn reduction_broadcast_output_is_rejected() {
        use crate::ir::ReduceOp;
        use baracuda_kernel_vocab::AxisMask;
        // A broadcast (stride-0) output would collapse every result onto one slot —
        // the general path must reject it, not emit an aliasing store.
        let a = OperandDesc::new(2, &[4, 8], &[8, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(1, &[8], &[0], ElementKind::F32, 256); // stride-0 = broadcast
        let key = structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89);
        let op = OpDef::reduction_axes(
            "s",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            AxisMask(0b01),
            false,
        );
        let _ = generate(&op, &key, &Cuda);
    }

    // ============================ increment 0e ============================
    // (1) ReduceOp::Prod, (2) fused reduction post-expr, (3) hetero-out reduction.

    #[test]
    fn reduction_prod_fp_folds_from_one() {
        use crate::ir::ReduceOp;
        // Prod over the last axis: identity 1, `acc *= elem`, block_prod tree,
        // pass-through finalize (no divisor). Matches bespoke reduce_prod_fp.cu.
        let op = OpDef::reduction("p", 1, &[ElementKind::F32], input(0), ReduceOp::Prod);
        let k = generate(&op, &reduce_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_p_f32_reduce_prod");
        // block_prod cooperative reducer, NOT block_sum.
        assert!(k.source.contains("float warp_prod_p_f32(float v)"));
        assert!(
            k.source
                .contains("v *= __shfl_down_sync(0xffffffffu, v, off);")
        );
        assert!(k.source.contains("float block_prod_p_f32(float v)"));
        assert!(!k.source.contains("block_sum"));
        // identity 1, multiplicative fold, thread-0 store, no Mean divisor.
        assert!(k.source.contains("float acc = 1.0f;"));
        assert!(k.source.contains("acc *= in0[idx];"));
        assert!(k.source.contains("float r = block_prod_p_f32(acc);"));
        assert!(k.source.contains("if (threadIdx.x == 0) out[row] = r;"));
        assert!(!k.source.contains("/ (float)k"));
    }

    #[test]
    fn reduction_prod_int_accumulates_in_long_long() {
        use crate::ir::ReduceOp;
        // i32 Prod: widened `long long` accumulator (the bespoke i64 reduce_prod_int
        // accumulator), native int multiply, store truncates back to i32 (wrap).
        let op = OpDef::reduction("p", 1, &[ElementKind::I32], input(0), ReduceOp::Prod);
        let k = generate(&op, &reduce_key(ElementKind::I32), &Cuda);
        assert!(k.source.contains("const int* __restrict__ in0"));
        assert!(k.source.contains("int* __restrict__ out")); // out == in dtype
        assert!(k.source.contains("long long acc = 1;"));
        assert!(k.source.contains("acc *= in0[idx];")); // native int, no float convert
        assert!(k.source.contains("long long r = block_prod_p_i32(acc);"));
        assert!(!k.source.contains("float acc"));
    }

    #[test]
    fn reduction_sum_s8_accumulates_in_long_long_and_wraps_at_store() {
        use crate::ir::ReduceOp;
        // S8 Sum: same `long long` widened-accumulate path as I32/I64 (reuse, no
        // narrower accumulator). The native leaf load is `in0[idx]` (S8 hits the
        // `_` arm of `promote_load_f32`, no half/bf16 widening detour). The store
        // is homogeneous (out_dtype == dtype == S8), so `demote_store_f32` is a
        // pass-through and the truncation to 8 bits happens via the `signed char*`
        // output pointer's implicit (wrapping, two's-complement) assignment
        // conversion — KISS-OPS-6.2-0002.
        let op = OpDef::reduction("s", 1, &[ElementKind::S8], input(0), ReduceOp::Sum);
        let k = generate(&op, &reduce_key(ElementKind::S8), &Cuda);
        assert!(k.source.contains("const signed char* __restrict__ in0"));
        assert!(k.source.contains("signed char* __restrict__ out")); // out == in dtype
        assert!(k.source.contains("long long acc = 0;"));
        assert!(k.source.contains("acc += in0[idx];")); // native int, no float convert
        assert!(k.source.contains("long long r = block_sum_s_i8(acc);"));
        assert!(k.source.contains("if (threadIdx.x == 0) out[row] = r;")); // wraps via the signed char* store
        assert!(!k.source.contains("float acc"));
    }

    #[test]
    fn reduction_max_u8_accumulates_in_long_long_and_wraps_at_store() {
        use crate::ir::ReduceOp;
        // U8 Max: same widened-accumulate path, unsigned native load/store. The
        // seed is now the U8 Max empty-axis identity `((unsigned char)0)`
        // (KISS-OPS-6.11-0002) rather than a bare `0` — numerically the same
        // value, but the explicit per-dtype literal is what makes an EMPTY
        // reduced extent correct (see the dedicated empty-axis identity tests
        // below); a non-empty fold is unaffected either way (`has` forces the
        // first real element to overwrite the seed unconditionally).
        let op = OpDef::reduction("m", 1, &[ElementKind::U8], input(0), ReduceOp::Max);
        let k = generate(&op, &reduce_key(ElementKind::U8), &Cuda);
        assert!(k.source.contains("const unsigned char* __restrict__ in0"));
        assert!(k.source.contains("unsigned char* __restrict__ out")); // out == in dtype
        assert!(k.source.contains("long long acc = ((unsigned char)0); int has = 0;"));
        assert!(k.source.contains("long long e = in0[idx];")); // native int, no float convert
        assert!(k.source.contains("if (threadIdx.x == 0) out[row] = r;")); // wraps via the unsigned char* store
        assert!(!k.source.contains("float acc"));
    }

    #[test]
    fn reduction_count_s8_predicate_lowers_as_integer_not_float() {
        use crate::ir::{BinaryOp, ReduceOp, konst};
        // count = Sum(in != 0), I64 out (the any/all/count hetero-out shape
        // admitted by Task 3, ba325509). Positive evidence for the Task 3b
        // fix: the predicate must lower as a genuine INTEGER comparison
        // (`in0[idx] != 0 ? 1 : 0`) so `acc += elem` is a true `long long`
        // add — exact for a count past 2^24 (16,777,216), unlike the float
        // predicate `((float)in0[idx] != (float)0.0 ? 1.0f : 0.0f)` this
        // replaces (which silently promotes `acc` to `float` in C and stalls
        // at 2^24).
        let mut op = OpDef::reduction(
            "count_s8",
            1,
            &[ElementKind::S8],
            input(0).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Sum,
        );
        op.out_dtype = Some(ElementKind::I64);
        let k = generate(&op, &reduce_key(ElementKind::S8), &Cuda);
        assert!(k.source.contains("acc += (in0[idx] != 0 ? 1 : 0);"));
        assert!(k.source.contains("long long* __restrict__ out")); // I64 hetero out
        assert!(k.source.contains("if (threadIdx.x == 0) out[row] = (long long)(r);"));
        // No float-precision predicate anywhere: no cast to float, no float
        // 1/0 literals — the double-math hazard Task 3b closes.
        assert!(!k.source.contains("(float)in0"));
        assert!(!k.source.contains("1.0f"));
        assert!(!k.source.contains("0.0f"));
    }

    #[test]
    fn reduction_any_u8_predicate_lowers_as_integer_body_and_post() {
        use crate::ir::{BinaryOp, ReduceOp, konst, reduced};
        // any = Max(in != 0) with post Cmp*(Reduced(0), 0) casting the fold
        // back to a U8 boolean (the other any/all/count hetero-out shape).
        // Exercises the Task 3b int-predicate fix in BOTH the fold body AND
        // the post — any/all are numerically safe either way (Max/Min of 0/1
        // never accumulates magnitude), but the emitted predicate must still
        // be a true integer comparison, not the dtype-oblivious float one.
        let mut op = OpDef::reduction_post(
            "any_u8",
            1,
            &[ElementKind::U8],
            input(0).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Max,
            reduced(0).binary(BinaryOp::CmpNe, konst(0.0)),
        );
        op.out_dtype = Some(ElementKind::U8);
        let k = generate(&op, &reduce_key(ElementKind::U8), &Cuda);
        assert!(k.source.contains("long long e = (in0[idx] != 0 ? 1 : 0);")); // body predicate
        assert!(k.source.contains("long long red0 = r;"));
        assert!(k.source.contains("out[row] = (red0 != 0 ? 1 : 0);")); // post predicate
        assert!(!k.source.contains("(float)in0"));
        assert!(!k.source.contains("(float)red0"));
        assert!(!k.source.contains("1.0f"));
        assert!(!k.source.contains("0.0f"));
    }

    #[test]
    fn reduction_max_s8_empty_axis_seeds_the_dtype_min_identity() {
        use crate::ir::ReduceOp;
        // S8 Max, contiguous last-axis (InnerContig fast path): an EMPTY reduced
        // extent must fold to the dtype MINIMUM (KISS-OPS-6.11-0002), not 0 — so
        // any real element in a non-empty fold still compares as `>` the
        // identity. Signed case: -128.
        let op = OpDef::reduction("m", 1, &[ElementKind::S8], input(0), ReduceOp::Max);
        let k = generate(&op, &reduce_key(ElementKind::S8), &Cuda);
        assert!(k.source.contains("const signed char* __restrict__ in0"));
        assert!(
            k.source
                .contains("long long acc = ((signed char)-128); int has = 0;")
        );
        // Non-empty fold is untouched: `has` still forces the first element to
        // overwrite the seed unconditionally.
        assert!(
            k.source
                .contains("if (!has || e != e || e > acc) { acc = e; has = 1; }")
        );
    }

    #[test]
    fn reduction_min_u8_empty_axis_seeds_the_dtype_max_identity() {
        use crate::ir::ReduceOp;
        // U8 Min, contiguous last-axis (InnerContig fast path): an EMPTY reduced
        // extent must fold to the dtype MAXIMUM, not 0. Unsigned case: 255.
        let op = OpDef::reduction("m", 1, &[ElementKind::U8], input(0), ReduceOp::Min);
        let k = generate(&op, &reduce_key(ElementKind::U8), &Cuda);
        assert!(k.source.contains("const unsigned char* __restrict__ in0"));
        assert!(
            k.source
                .contains("long long acc = ((unsigned char)255); int has = 0;")
        );
        assert!(
            k.source
                .contains("if (!has || e != e || e < acc) { acc = e; has = 1; }")
        );
    }

    #[test]
    fn reduction_max_general_axis_s8_empty_axis_seeds_the_dtype_min_identity() {
        use crate::ir::ReduceOp;
        use baracuda_kernel_vocab::AxisMask;
        // S8 Max over the OUTER (non-contiguous-last) axis: the general
        // strided-fold path (the second emit site) — same empty-axis identity
        // must apply here too, not just the InnerContig fast path.
        let a = OperandDesc::new(2, &[4, 8], &[8, 1], ElementKind::S8, 256);
        let out = OperandDesc::new(1, &[8], &[1], ElementKind::S8, 256);
        let key = structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89);
        let op = OpDef::reduction_axes(
            "m",
            1,
            &[ElementKind::S8],
            input(0),
            ReduceOp::Max,
            AxisMask(0b01),
            false,
        );
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_m_i8_reduce_max_ax1");
        assert!(k.source.contains("long long acc = ((signed char)-128);"));
        assert!(k.source.contains("int has = 0;"));
        assert!(
            k.source
                .contains("acc = has ? ((e != e || e > acc) ? e : acc) : e; has = 1;")
        );
    }

    #[test]
    fn reduction_min_general_axis_u8_empty_axis_seeds_the_dtype_max_identity() {
        use crate::ir::ReduceOp;
        use baracuda_kernel_vocab::AxisMask;
        // U8 Min over the OUTER axis: general path, unsigned case: 255.
        let a = OperandDesc::new(2, &[4, 8], &[8, 1], ElementKind::U8, 256);
        let out = OperandDesc::new(1, &[8], &[1], ElementKind::U8, 256);
        let key = structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89);
        let op = OpDef::reduction_axes(
            "m",
            1,
            &[ElementKind::U8],
            input(0),
            ReduceOp::Min,
            AxisMask(0b01),
            false,
        );
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_m_u8_reduce_min_ax1");
        assert!(k.source.contains("long long acc = ((unsigned char)255);"));
        assert!(k.source.contains("int has = 0;"));
        assert!(
            k.source
                .contains("acc = has ? ((e != e || e < acc) ? e : acc) : e; has = 1;")
        );
    }

    #[test]
    fn reduction_prod_general_axis_folds_from_one() {
        use crate::ir::ReduceOp;
        use baracuda_kernel_vocab::AxisMask;
        // Prod over the outer axis (general path): identity 1, `acc *= elem`,
        // no Mean divisor.
        let a = OperandDesc::new(2, &[4, 8], &[8, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(1, &[8], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89);
        let op = OpDef::reduction_axes(
            "p",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Prod,
            AxisMask(0b01),
            false,
        );
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_p_f32_reduce_prod_ax1");
        assert!(k.source.contains("float acc = 1.0f;"));
        assert!(k.source.contains("acc *= in0[idx];"));
        assert!(k.source.contains("out[oo] = acc;")); // pass-through finalize
        assert!(!k.source.contains("/ (float)"));
    }

    #[test]
    fn reduction_post_norm2_applies_sqrt_after_the_sum() {
        use crate::ir::{ReduceOp, UnaryOp, reduced};
        // norm2 = Sqrt(Sum(Sqr(x))): the pre-body squares (already worked), the
        // 0e post applies Sqrt to the fold result via a hoisted `red0` register.
        let op = OpDef::reduction_post(
            "norm2",
            1,
            &[ElementKind::F32],
            input(0).unary(UnaryOp::Sqr),
            ReduceOp::Sum,
            reduced(0).sqrt(),
        );
        let k = generate(&op, &reduce_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_norm2_f32_reduce_sum");
        assert!(k.source.contains("acc += (in0[idx]*in0[idx]);"));
        assert!(k.source.contains("float r = block_sum_norm2_f32(acc);"));
        // Post: red0 = r (the fold result), then Sqrt, then store — thread-0 only.
        assert!(k.source.contains("if (threadIdx.x == 0) {"));
        assert!(k.source.contains("float red0 = r;"));
        assert!(k.source.contains("out[row] = sqrtf(red0);"));
    }

    #[test]
    fn reduction_post_sees_the_post_mean_value() {
        use crate::ir::{ReduceOp, UnaryOp, reduced};
        // Ordering pin (documented): Mean divides FIRST, then the post applies to
        // the mean. So `red0` binds the ALREADY-divided block_sum/k, and Sqrt sees
        // sqrt(mean), not mean(sqrt).
        let op = OpDef::reduction_post(
            "rms",
            1,
            &[ElementKind::F32],
            input(0).unary(UnaryOp::Sqr),
            ReduceOp::Mean,
            reduced(0).sqrt(),
        );
        let k = generate(&op, &reduce_key(ElementKind::F32), &Cuda);
        assert!(
            k.source
                .contains("float r = block_sum_rms_f32(acc) / (float)k;")
        );
        assert!(k.source.contains("float red0 = r;")); // post sees the mean
        assert!(k.source.contains("out[row] = sqrtf(red0);"));
    }

    #[test]
    fn reduction_post_identity_is_byte_identical_to_plain() {
        use crate::ir::{ReduceOp, reduced};
        // The default post (`Reduced(0)`) must emit exactly like OpDef::reduction —
        // no red0 binding, no extra braces — the 0e no-regression guarantee.
        let plain = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        let posted = OpDef::reduction_post(
            "s",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            reduced(0),
        );
        let kp = generate(&plain, &reduce_key(ElementKind::F32), &Cuda);
        let kd = generate(&posted, &reduce_key(ElementKind::F32), &Cuda);
        assert_eq!(kp.source, kd.source, "identity post must be byte-identical");
        assert!(!kd.source.contains("red0"));
    }

    // Reduce-key with a hetero output dtype: [256,128] float input, [256] `out_dt`.
    fn reduce_key_hetero(out_dt: ElementKind) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(1, &[256], &[1], out_dt, 256);
        structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89)
    }

    #[test]
    fn reduction_hetero_out_u8_any() {
        use crate::ir::{BinaryOp, ReduceOp, konst, reduced};
        // any = Sum(x != 0) with a Cmp* post `Reduced(0) > 0` → exactly 0/1,
        // stored to a u8 mask. Input dtype stays f32 (the key dtype); the output
        // pointer + store convert to u8 (the 0b hetero pattern, on a reduction).
        let mut op = OpDef::reduction_post(
            "any",
            1,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Sum,
            reduced(0).binary(BinaryOp::CmpGt, konst(0.0)),
        );
        op.out_dtype = Some(ElementKind::U8);
        let k = generate(&op, &reduce_key_hetero(ElementKind::U8), &Cuda);
        assert!(k.source.contains("unsigned char* __restrict__ out")); // hetero out ptr
        assert!(k.source.contains("float acc = 0.0f;")); // fold still in float
        assert!(k.source.contains("float red0 = r;"));
        // Post is a Cmp (0/1); the store casts the exact predicate to u8.
        assert!(
            k.source
                .contains("out[row] = (unsigned char)(((float)red0 > (float)0.0 ? 1.0f : 0.0f));")
        );
    }

    #[test]
    fn reduction_hetero_out_i64_count() {
        use crate::ir::{BinaryOp, ReduceOp, konst};
        // count = Sum(x != 0) with the identity post, stored to i64. The float
        // accumulator converts to a long long store (exact while count ≤ 2^24).
        let mut op = OpDef::reduction(
            "count",
            1,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::CmpNe, konst(0.0)),
            ReduceOp::Sum,
        );
        op.out_dtype = Some(ElementKind::I64);
        let k = generate(&op, &reduce_key_hetero(ElementKind::I64), &Cuda);
        assert!(k.source.contains("long long* __restrict__ out")); // i64 out ptr
        assert!(k.source.contains("float acc = 0.0f;"));
        // Identity post ⇒ no red0 binding; the store casts float acc → i64.
        assert!(!k.source.contains("red0"));
        assert!(
            k.source
                .contains("if (threadIdx.x == 0) out[row] = (long long)(r);")
        );
    }

    // ---- 0e gate-rejection tests (call build_plan DIRECTLY: the emitter would
    // silently truncate a bad hetero store rather than panic, so the plan gate is
    // the load-bearing wall — exercise it without the emitter in the way). ----

    #[test]
    #[should_panic(expected = "must not read Input")]
    fn reduction_post_reading_input_is_rejected_at_the_plan_gate() {
        use crate::ir::{ReduceOp, reduced};
        // The reduced axis is gone; an Input at the output coordinate is a
        // different, ambiguous tensor — rejected (mirrors the contraction epilogue).
        let op = OpDef::reduction_post(
            "bad",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            reduced(0) + input(0),
        );
        let _ = crate::build_plan(&op, &reduce_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "requires the POST-expr ROOT to be a comparison")]
    fn reduction_u8_out_non_cmp_post_is_rejected_at_the_plan_gate() {
        use crate::ir::{ReduceOp, reduced};
        // U8 out with a non-cmp (identity) post stores the raw accumulator — a
        // silent truncation. The gate demands a Cmp* post (exact 0/1).
        let mut op = OpDef::reduction_post(
            "bad_any",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            reduced(0),
        );
        op.out_dtype = Some(ElementKind::U8);
        let _ = crate::build_plan(&op, &reduce_key_hetero(ElementKind::U8));
    }

    #[test]
    #[should_panic(expected = "requires op = Sum")]
    fn reduction_i64_out_non_sum_is_rejected_at_the_plan_gate() {
        use crate::ir::ReduceOp;
        // I64 out is the count/sum-widening shape — Max → i64 is not an exact
        // integer store shape, so it rejects.
        let mut op = OpDef::reduction("m", 1, &[ElementKind::F32], input(0), ReduceOp::Max);
        op.out_dtype = Some(ElementKind::I64);
        let _ = crate::build_plan(&op, &reduce_key_hetero(ElementKind::I64));
    }

    #[test]
    #[should_panic(expected = "Prod combiner is not supported in the fused")]
    fn rowreduce_prod_stage_is_rejected_at_the_plan_gate() {
        use crate::ir::{ReduceOp, ReduceStage, reduced};
        // Prod is a 0e Access::Reduction combiner only; the fused row path has no
        // block_prod, so a Prod stage misses honestly at the gate.
        let op = OpDef::row_reduce(
            "gp",
            1,
            &[ElementKind::F32],
            vec![ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Prod,
            }],
            reduced(0),
        );
        let _ = crate::build_plan(&op, &rr_key(ElementKind::F32, OpCategory::Softmax));
    }

    fn rr_key(dt: ElementKind, cat: OpCategory) -> baracuda_kernel_vocab::StructureKey {
        // full-width fused op: input + output share the [256, 128] contiguous shape.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(cat, &[a, a], ArchSku::Sm89)
    }

    fn rmsnorm_op(dt: ElementKind) -> OpDef {
        use crate::ir::{ReduceOp, ReduceStage, UnaryOp, konst, reduced};
        // x * rsqrt(mean(x^2) + eps), eps baked as a finite Const.
        OpDef::row_reduce(
            "rmsnorm",
            1,
            &[dt],
            vec![ReduceStage {
                pre: input(0).unary(UnaryOp::Sqr).0,
                op: ReduceOp::Mean,
            }],
            input(0) * (reduced(0) + konst(1e-5)).unary(UnaryOp::Rsqrt),
        )
    }

    fn softmax_op(dt: ElementKind) -> OpDef {
        use crate::ir::{ReduceOp, ReduceStage, reduced};
        // exp(x - rowmax) / sum(exp(x - rowmax)) — numerically stable.
        OpDef::row_reduce(
            "softmax",
            1,
            &[dt],
            vec![
                ReduceStage {
                    pre: input(0).0,
                    op: ReduceOp::Max,
                },
                ReduceStage {
                    pre: (input(0) - reduced(0)).exp().0,
                    op: ReduceOp::Sum,
                },
            ],
            (input(0) - reduced(0)).exp() / reduced(1),
        )
    }

    #[test]
    fn rowreduce_rmsnorm_block_tree_fold() {
        let k = generate(
            &rmsnorm_op(ElementKind::F32),
            &rr_key(ElementKind::F32, OpCategory::Normalization),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_rmsnorm_f32_rowreduce");
        // warp-shuffle + shared-mem block reduce, full 0xffffffff mask.
        assert!(k.source.contains("__shfl_down_sync(0xffffffffu, v, off)"));
        assert!(k.source.contains("float block_sum_rmsnorm_f32(float v)"));
        // one block per row, grid-stride over rows; grid-stride fold within the row.
        assert!(k.source.contains(
            "for (long long row = blockIdx.x; row < n_out; row += (long long)gridDim.x)"
        ));
        assert!(
            k.source
                .contains("for (long long j = threadIdx.x; j < k; j += blockDim.x)")
        );
        // mean of squares -> rsqrt; full-width output; uniform empty-axis guard.
        assert!(k.source.contains("acc0 += (in0[idx]*in0[idx]);"));
        assert!(k.source.contains("block_sum_rmsnorm_f32(acc0) / (float)k"));
        assert!(
            k.source
                .contains("out[idx] = (in0[idx] * rsqrtf((r0 + 1e-5)));")
        );
        assert!(k.source.contains("if (k == 0) return;"));
    }

    #[test]
    fn rowreduce_cross_warp_indexes_threadidx_not_lane() {
        // The must-fix from the design review: the cross-warp fold reads DISTINCT
        // warp partials at smem[threadIdx.x], NEVER smem[lane] (which would refold
        // warp 0's partials for every multi-warp block — a silent wrong reduction).
        let k = generate(
            &rmsnorm_op(ElementKind::F32),
            &rr_key(ElementKind::F32, OpCategory::Normalization),
            &Cuda,
        );
        assert!(k.source.contains("smem[threadIdx.x]"));
        assert!(!k.source.contains("smem[lane]"));
    }

    #[test]
    fn rowreduce_softmax_two_stage_nan_max() {
        let k = generate(
            &softmax_op(ElementKind::F32),
            &rr_key(ElementKind::F32, OpCategory::Softmax),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_softmax_f32_rowreduce");
        // stage 0: NaN-propagating max via the (value, has) flag, no ±inf literal.
        assert!(k.source.contains("block_max_softmax_f32"));
        assert!(!k.source.contains("INFINITY"));
        assert!(k.source.contains("if (!has0 || e != e || e > acc0)"));
        // stage 1 reads the rowmax register r0; numerically-stable exp(x - max).
        assert!(k.source.contains("acc1 += expf((in0[idx] - r0));"));
        // epilogue divides by the denom register r1.
        assert!(
            k.source
                .contains("out[idx] = (expf((in0[idx] - r0)) / r1);")
        );
    }

    #[test]
    fn rowreduce_f16_accumulates_in_float() {
        let k = generate(
            &rmsnorm_op(ElementKind::F16),
            &rr_key(ElementKind::F16, OpCategory::Normalization),
            &Cuda,
        );
        assert!(k.source.contains("#include <cuda_fp16.h>"));
        assert!(k.source.contains("const __half* __restrict__ in0"));
        assert!(k.source.contains("float block_sum_rmsnorm_f16(float v)")); // float acc
        assert!(
            k.source
                .contains("acc0 += (__half2float(in0[idx])*__half2float(in0[idx]));")
        );
        assert!(
            k.source.contains(
                "out[idx] = __float2half((__half2float(in0[idx]) * rsqrtf((r0 + 1e-5))));"
            )
        );
    }

    #[test]
    #[should_panic(expected = "references a stage not yet produced")]
    fn rowreduce_forward_reduced_ref_panics() {
        use crate::ir::{ReduceOp, ReduceStage, reduced};
        // A stage 0 `pre` referencing Reduced(0) (its own not-yet-produced result)
        // is a mis-authored op — validate_row_reduce must reject it at build_plan.
        let bad = OpDef::row_reduce(
            "bad",
            1,
            &[ElementKind::F32],
            vec![ReduceStage {
                pre: reduced(0).0,
                op: ReduceOp::Sum,
            }],
            input(0) * reduced(0),
        );
        let _ = generate(
            &bad,
            &rr_key(ElementKind::F32, OpCategory::Normalization),
            &Cuda,
        );
    }

    // --- multi-input RowReduce: weighted-RmsNorm + LayerNorm ---

    fn mi_key(dt: ElementKind, n_col: usize) -> baracuda_kernel_vocab::StructureKey {
        // x [256,128] full + n_col per-column [k] weight/bias (rank-aligned broadcast
        // view, stride [0,1]) + full-width output.
        let x = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let col = OperandDesc::new(2, &[256, 128], &[0, 1], dt, 256);
        let out = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let mut ops = vec![x];
        ops.extend(std::iter::repeat_n(col, n_col));
        ops.push(out);
        structure_key(OpCategory::Normalization, &ops, ArchSku::Sm89)
    }

    fn wrmsnorm_op(dt: ElementKind) -> OpDef {
        use crate::ir::{ReduceOp, ReduceStage, UnaryOp, konst, reduced};
        // x * rsqrt(mean(x^2) + eps) * weight; in0=x (row), in1=weight [k] (column).
        OpDef::row_reduce(
            "wrmsnorm",
            2,
            &[dt],
            vec![ReduceStage {
                pre: input(0).unary(UnaryOp::Sqr).0,
                op: ReduceOp::Mean,
            }],
            input(0) * (reduced(0) + konst(1e-5)).unary(UnaryOp::Rsqrt) * input(1),
        )
    }

    fn layernorm_op(dt: ElementKind) -> OpDef {
        use crate::ir::{ReduceOp, ReduceStage, UnaryOp, konst, reduced};
        // (x-mean)*rsqrt(var+eps)*weight + bias; in0=x, in1=weight[k], in2=bias[k].
        OpDef::row_reduce(
            "layernorm",
            3,
            &[dt],
            vec![
                ReduceStage {
                    pre: input(0).0,
                    op: ReduceOp::Mean,
                },
                ReduceStage {
                    pre: (input(0) - reduced(0)).unary(UnaryOp::Sqr).0,
                    op: ReduceOp::Mean,
                },
            ],
            (input(0) - reduced(0)) * (reduced(1) + konst(1e-5)).unary(UnaryOp::Rsqrt) * input(1)
                + input(2),
        )
    }

    #[test]
    fn rowreduce_weighted_rmsnorm_column_index() {
        let k = generate(
            &wrmsnorm_op(ElementKind::F32),
            &mi_key(ElementKind::F32, 1),
            &Cuda,
        );
        assert!(k.source.contains("const float* __restrict__ in1,")); // weight operand
        // stage reduces only the row-streamed x (in0[idx]); column weight only in
        // the epilogue, indexed in1[j] (per-column) not in1[idx] (per-element).
        assert!(k.source.contains("acc0 += (in0[idx]*in0[idx]);"));
        assert!(
            k.source
                .contains("out[idx] = ((in0[idx] * rsqrtf((r0 + 1e-5))) * in1[j]);")
        );
        assert!(!k.source.contains("in1[idx]"));
    }

    #[test]
    fn rowreduce_layernorm_stable_two_pass_with_weight_bias() {
        let k = generate(
            &layernorm_op(ElementKind::F32),
            &mi_key(ElementKind::F32, 2),
            &Cuda,
        );
        assert!(k.source.contains("const float* __restrict__ in2,")); // bias operand
        // stage 0 mean; stage 1 = mean of squared DEVIATIONS (the stable two-pass
        // var, not the cancellation-prone mean(x^2)-mean(x)^2).
        assert!(k.source.contains("acc0 += in0[idx];"));
        assert!(
            k.source
                .contains("acc1 += ((in0[idx] - r0)*(in0[idx] - r0));")
        );
        // epilogue: (x-mean)*rsqrt(var+eps)*weight[j] + bias[j].
        assert!(
            k.source.contains(
                "out[idx] = ((((in0[idx] - r0) * rsqrtf((r1 + 1e-5))) * in1[j]) + in2[j]);"
            )
        );
        assert!(!k.source.contains("in1[idx]") && !k.source.contains("in2[idx]"));
    }

    #[test]
    fn rowreduce_second_empty_bcast_input_is_row_streamed_not_rejected() {
        // Increment 2 LIFTED the former "inputs>0 must be column-broadcast" guard.
        // A second input with an EMPTY bcast mask is now classified RowStreamed (a
        // second reduced/streamed tensor — softmax-bw's `dy` beside `y`) and ACCEPTED;
        // the key genuinely cannot distinguish a bare rank-1 [k] from a full [n_out,k]
        // (both have the identical {Contig, empty-bcast} operand key), so the full
        // extent is a caller precondition at the same trust level as input 0 (see the
        // validate_row_reduce module note). Fed wrmsnorm_op's epilogue, input 1 now
        // indexes in1[idx] (row-streamed), NOT in1[j] (column) — proving the
        // reclassification the lift produces.
        let x = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let bare_w = OperandDesc::new(1, &[128], &[1], ElementKind::F32, 256); // bare [K]
        let out = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Normalization, &[x, bare_w, out], ArchSku::Sm89);
        let k = generate(&wrmsnorm_op(ElementKind::F32), &key, &Cuda);
        assert!(
            k.source.contains("in1[idx]"),
            "second empty-bcast input is row-streamed"
        );
        assert!(
            !k.source.contains("in1[j]"),
            "not classified as a column weight"
        );
    }

    #[test]
    #[should_panic(expected = "epilogue-only")]
    fn rowreduce_column_input_in_stage_rejected() {
        use crate::ir::{ReduceOp, ReduceStage};
        // Reducing a per-column operand is nonsense — a stage.pre referencing the
        // column weight (Input1) must be rejected.
        let bad = OpDef::row_reduce(
            "bad",
            2,
            &[ElementKind::F32],
            vec![ReduceStage {
                pre: (input(0) * input(1)).0,
                op: ReduceOp::Mean,
            }],
            input(0) * input(1),
        );
        let _ = generate(&bad, &mi_key(ElementKind::F32, 1), &Cuda);
    }

    // --- increment 2: compound-backward RowReduce (2nd row-streamed input +
    //     per-row saved-stat scalars). softmax bw + layer_norm bw dx. ---

    // Two full-width row-streamed inputs [256,128] + full output — softmax bw's
    // (y, dy, dx). No column/row-scalar operand.
    fn softmax_bw_key(dt: ElementKind) -> baracuda_kernel_vocab::StructureKey {
        let full = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(OpCategory::Softmax, &[full, full, full], ArchSku::Sm89)
    }

    fn softmax_bw_op(dt: ElementKind) -> OpDef {
        use crate::ir::{ReduceOp, ReduceStage, reduced};
        // in0=y, in1=dy (both row-streamed). dx = y*(dy - Σ_j y[j]·dy[j]).
        OpDef::row_reduce(
            "softmax_bw",
            2,
            &[dt],
            vec![ReduceStage {
                pre: (input(0) * input(1)).0,
                op: ReduceOp::Sum,
            }],
            input(0) * (input(1) - reduced(0)),
        )
    }

    // x, dy row-streamed [256,128]; mean, rstd per-row scalars ([n_out,k]-presented,
    // strides [1,0]: feature-axis broadcast, outer varies) + full output.
    fn layer_norm_bw_key(dt: ElementKind) -> baracuda_kernel_vocab::StructureKey {
        let stream = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let rowscalar = OperandDesc::new(2, &[256, 128], &[1, 0], dt, 256);
        structure_key(
            OpCategory::Normalization,
            &[stream, stream, rowscalar, rowscalar, stream],
            ArchSku::Sm89,
        )
    }

    fn layer_norm_bw_op(dt: ElementKind) -> OpDef {
        use crate::ir::{ReduceOp, ReduceStage, reduced};
        // in0=x, in1=dy (row-streamed); in2=mean, in3=rstd (per-row scalars).
        // x_hat=(x-mean)*rstd; dx = rstd*(dy - mean(dy) - x_hat*mean(dy*x_hat)).
        let x_hat = (input(0) - input(2)) * input(3);
        OpDef::row_reduce(
            "layer_norm_bw",
            4,
            &[dt],
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
        )
    }

    #[test]
    fn rowreduce_softmax_bw_two_row_streamed_inputs() {
        let k = generate(
            &softmax_bw_op(ElementKind::F32),
            &softmax_bw_key(ElementKind::F32),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_softmax_bw_f32_rowreduce");
        // signature carries the SECOND row-streamed input in1.
        assert!(k.source.contains("const float* __restrict__ in1,"));
        // the Sum stage folds y·dy — BOTH inputs indexed row-streamed (in_i[idx]).
        assert!(k.source.contains("acc0 += (in0[idx] * in1[idx]);"));
        assert!(
            k.source
                .contains("float r0 = block_sum_softmax_bw_f32(acc0);")
        );
        // epilogue: dx = y·(dy - rowdot), full-width, both inputs row-streamed.
        assert!(
            k.source
                .contains("out[idx] = (in0[idx] * (in1[idx] - r0));")
        );
        // no per-column (j) or per-row-scalar (row/rs) index appears — pure streamed.
        assert!(
            !k.source.contains("in1[j]") && !k.source.contains("rs0") && !k.source.contains("rs1")
        );
    }

    #[test]
    fn rowreduce_layer_norm_bw_rowscalar_hoist_two_stage() {
        let k = generate(
            &layer_norm_bw_op(ElementKind::F32),
            &layer_norm_bw_key(ElementKind::F32),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_layer_norm_bw_f32_rowreduce");
        // four inputs: x, dy row-streamed; mean, rstd per-row scalars.
        assert!(k.source.contains("const float* __restrict__ in3,"));
        // saved stats hoisted ONCE per row (in{i}[row]), outside the feature loop.
        assert!(k.source.contains("float rs2 = in2[row];"));
        assert!(k.source.contains("float rs3 = in3[row];"));
        // stage 0: mean(dy) — the streamed dy folded, Mean divides by k.
        assert!(k.source.contains("acc0 += in1[idx];"));
        assert!(
            k.source
                .contains("float r0 = block_sum_layer_norm_bw_f32(acc0) / (float)k;")
        );
        // stage 1: mean(dy·x_hat), x_hat=(x-mean)*rstd reads the hoisted rs2/rs3.
        assert!(
            k.source
                .contains("acc1 += (in1[idx] * ((in0[idx] - rs2) * rs3));")
        );
        assert!(
            k.source
                .contains("float r1 = block_sum_layer_norm_bw_f32(acc1) / (float)k;")
        );
        // epilogue: dx = rstd·(dy - r0 - x_hat·r1) — rstd/mean via rs3/rs2, NOT per-elem.
        assert!(
            k.source.contains(
                "out[idx] = (rs3 * ((in1[idx] - r0) - (((in0[idx] - rs2) * rs3) * r1)));"
            )
        );
        // the per-row scalars are NEVER read per-element (in2[idx]/in3[idx]).
        assert!(!k.source.contains("in2[idx]") && !k.source.contains("in3[idx]"));
        // ...nor per-column (in2[j]/in3[j]).
        assert!(!k.source.contains("in2[j]") && !k.source.contains("in3[j]"));
    }

    #[test]
    fn rowreduce_layer_norm_bw_f16_hoists_upconverted_stat() {
        // f16: the hoisted per-row scalar is up-converted to the f32 accumulate type
        // ONCE (__half2float(in{i}[row])), not per feature element.
        let k = generate(
            &layer_norm_bw_op(ElementKind::F16),
            &layer_norm_bw_key(ElementKind::F16),
            &Cuda,
        );
        assert!(k.source.contains("float rs2 = __half2float(in2[row]);"));
        assert!(k.source.contains("float rs3 = __half2float(in3[row]);"));
        // block reducer accumulates in float even for f16.
        assert!(
            k.source
                .contains("float block_sum_layer_norm_bw_f16(float v)")
        );
    }

    // =======================================================================
    // Increment-0a scalar-fn vocabulary — emission goldens + dtype gates
    // =======================================================================

    /// Scalar (unvectorized) unary cell: align defeats vectorization so the
    /// emitted body is the bare `out[i] = <fn>(in0[i]);` golden.
    fn unary_scalar_key(dt: ElementKind, align: u32) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    }

    fn binary_scalar_key(dt: ElementKind, align: u32) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    }

    #[test]
    fn vocab_unary_emission_goldens_f32_f64_f16() {
        use crate::ir::UnaryOp;
        // One golden per new unary fn (Erf pre-existed; pinned here so the whole
        // sweep is asserted in one place): exact C call per dtype — f32 device
        // math, f64 device math, f16 promote→f32-fn→demote.
        let cases: &[(UnaryOp, &str, &str)] = &[
            (UnaryOp::Erf, "erff", "erf"),
            (UnaryOp::Erfc, "erfcf", "erfc"),
            (UnaryOp::Trunc, "truncf", "trunc"),
            (UnaryOp::Exp2, "exp2f", "exp2"),
            (UnaryOp::Expm1, "expm1f", "expm1"),
            (UnaryOp::Log2, "log2f", "log2"),
            (UnaryOp::Log10, "log10f", "log10"),
            (UnaryOp::Log1p, "log1pf", "log1p"),
            (UnaryOp::Sinh, "sinhf", "sinh"),
            (UnaryOp::Cosh, "coshf", "cosh"),
            (UnaryOp::Tan, "tanf", "tan"),
            (UnaryOp::Asin, "asinf", "asin"),
            (UnaryOp::Acos, "acosf", "acos"),
            (UnaryOp::Atan, "atanf", "atan"),
            (UnaryOp::Asinh, "asinhf", "asinh"),
            (UnaryOp::Acosh, "acoshf", "acosh"),
            (UnaryOp::Atanh, "atanhf", "atanh"),
            (UnaryOp::Cbrt, "cbrtf", "cbrt"),
            (UnaryOp::Lgamma, "lgammaf", "lgamma"),
        ];
        for &(uop, f32_fn, f64_fn) in cases {
            let op = |dt: ElementKind| OpDef::elementwise("v", 1, &[dt], input(0).unary(uop));
            let kf = generate(
                &op(ElementKind::F32),
                &unary_scalar_key(ElementKind::F32, 4),
                &Cuda,
            );
            assert!(
                kf.source.contains(&format!("out[i] = {f32_fn}(in0[i]);")),
                "{uop:?} f32 golden missing in:\n{}",
                kf.source
            );
            let kd = generate(
                &op(ElementKind::F64),
                &unary_scalar_key(ElementKind::F64, 8),
                &Cuda,
            );
            assert!(
                kd.source.contains(&format!("out[i] = {f64_fn}(in0[i]);")),
                "{uop:?} f64 golden missing in:\n{}",
                kd.source
            );
            // f16 scalar: the existing promote-to-f32 transcendental path.
            let kh = generate(
                &op(ElementKind::F16),
                &unary_scalar_key(ElementKind::F16, 2),
                &Cuda,
            );
            assert!(
                kh.source.contains(&format!(
                    "out[i] = __float2half({f32_fn}(__half2float(in0[i])));"
                )),
                "{uop:?} f16 promote golden missing in:\n{}",
                kh.source
            );
        }
    }

    #[test]
    fn vocab_binary_emission_goldens_f32_f64_f16() {
        use crate::ir::BinaryOp;
        // (op, f32 fn, f64 fn, f16 promote path expected?) — Nextafter has NO
        // half lowering (gated; separate test).
        let cases: &[(BinaryOp, &str, &str, bool)] = &[
            (BinaryOp::Atan2, "atan2f", "atan2", true),
            (BinaryOp::Copysign, "copysignf", "copysign", true),
            (BinaryOp::Nextafter, "nextafterf", "nextafter", false),
            (BinaryOp::FmaxIeee, "fmaxf", "fmax", true),
            (BinaryOp::FminIeee, "fminf", "fmin", true),
            (BinaryOp::RemTrunc, "fmodf", "fmod", true),
        ];
        for &(bop, f32_fn, f64_fn, half_ok) in cases {
            let op =
                |dt: ElementKind| OpDef::elementwise("v", 2, &[dt], input(0).binary(bop, input(1)));
            let kf = generate(
                &op(ElementKind::F32),
                &binary_scalar_key(ElementKind::F32, 4),
                &Cuda,
            );
            assert!(
                kf.source
                    .contains(&format!("out[i] = {f32_fn}(in0[i], in1[i]);")),
                "{bop:?} f32 golden missing in:\n{}",
                kf.source
            );
            let kd = generate(
                &op(ElementKind::F64),
                &binary_scalar_key(ElementKind::F64, 8),
                &Cuda,
            );
            assert!(
                kd.source
                    .contains(&format!("out[i] = {f64_fn}(in0[i], in1[i]);")),
                "{bop:?} f64 golden missing in:\n{}",
                kd.source
            );
            if half_ok {
                let kh = generate(
                    &op(ElementKind::F16),
                    &binary_scalar_key(ElementKind::F16, 2),
                    &Cuda,
                );
                assert!(
                    kh.source.contains(&format!(
                        "out[i] = __float2half({f32_fn}(__half2float(in0[i]), __half2float(in1[i])));"
                    )),
                    "{bop:?} f16 promote golden missing in:\n{}",
                    kh.source
                );
            }
        }
    }

    #[test]
    fn fmax_ieee_is_fmaxf_and_distinct_from_nan_propagating_maximum() {
        use crate::ir::BinaryOp;
        // The two ops must never alias: FmaxIeee/FminIeee are the NaN-SUPPRESSING
        // fmaxf/fminf; Max/Min stay the NaN-propagating compare-selects.
        let key = binary_scalar_key(ElementKind::F32, 4);
        let ieee = OpDef::elementwise(
            "fmaxi",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::FmaxIeee, input(1)),
        );
        let ki = generate(&ieee, &key, &Cuda);
        assert!(ki.source.contains("fmaxf(in0[i], in1[i])"));
        assert!(!ki.source.contains("!="), "no NaN-select in the IEEE op");
        let max = OpDef::elementwise("m", 2, &[ElementKind::F32], input(0).max(input(1)));
        let km = generate(&max, &key, &Cuda);
        assert!(
            !km.source.contains("fmaxf"),
            "Maximum must NOT become fmaxf"
        );
        assert!(
            km.source.contains("!="),
            "Maximum keeps the NaN-propagating select"
        );
        // Min side — BOTH directions (mutation-caught gap: Min silently
        // becoming fminf passed the suite when only FminIeee was pinned).
        let ieee_min = OpDef::elementwise(
            "fmini",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::FminIeee, input(1)),
        );
        assert!(
            generate(&ieee_min, &key, &Cuda)
                .source
                .contains("fminf(in0[i], in1[i])")
        );
        let min = OpDef::elementwise("mn", 2, &[ElementKind::F32], input(0).min(input(1)));
        let kn = generate(&min, &key, &Cuda);
        assert!(
            !kn.source.contains("fminf"),
            "Minimum must NOT become fminf"
        );
        assert!(
            kn.source.contains("!="),
            "Minimum keeps the NaN-propagating select"
        );
    }

    #[test]
    fn remtrunc_is_fmodf_and_rem_stays_floored() {
        use crate::ir::BinaryOp;
        // RemTrunc = C fmodf (sign-of-dividend); Rem stays the floored form
        // (sign-of-divisor). Distinct spellings, never merged.
        let key = binary_scalar_key(ElementKind::F32, 4);
        let rt = OpDef::elementwise(
            "fmod",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::RemTrunc, input(1)),
        );
        let kt = generate(&rt, &key, &Cuda);
        assert!(kt.source.contains("fmodf(in0[i], in1[i])"));
        assert!(
            !kt.source.contains("floorf"),
            "RemTrunc must not take the floored form"
        );
        let rem = OpDef::elementwise(
            "rem",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::Rem, input(1)),
        );
        let kr = generate(&rem, &key, &Cuda);
        assert!(
            kr.source
                .contains("(in0[i] - floorf(in0[i] / in1[i]) * in1[i])")
        );
        assert!(!kr.source.contains("fmodf"), "Rem must not become fmodf");
        // f64 spellings pinned too — in binary_f64 the fmod/floor and
        // fmax/fmin arms sit one edit apart, an easy silent swap.
        let key64 = binary_scalar_key(ElementKind::F64, 4);
        let rt64 = OpDef::elementwise(
            "fmod64",
            2,
            &[ElementKind::F64],
            input(0).binary(BinaryOp::RemTrunc, input(1)),
        );
        let kt64 = generate(&rt64, &key64, &Cuda);
        assert!(kt64.source.contains("fmod(in0[i], in1[i])"));
        assert!(
            !kt64.source.contains("floor("),
            "f64 RemTrunc must not take the floored form"
        );
        let rem64 = OpDef::elementwise(
            "rem64",
            2,
            &[ElementKind::F64],
            input(0).binary(BinaryOp::Rem, input(1)),
        );
        let kr64 = generate(&rem64, &key64, &Cuda);
        assert!(
            kr64.source
                .contains("(in0[i] - floor(in0[i] / in1[i]) * in1[i])")
        );
        assert!(
            !kr64.source.contains("fmod("),
            "f64 Rem must not become fmod"
        );
        let max64 = OpDef::elementwise("mx64", 2, &[ElementKind::F64], input(0).max(input(1)));
        let km64 = generate(&max64, &key64, &Cuda);
        assert!(
            !km64.source.contains("fmax("),
            "f64 Maximum must NOT become fmax"
        );
        assert!(
            km64.source.contains("!="),
            "f64 Maximum keeps the NaN-propagating select"
        );
        let min64 = OpDef::elementwise("mn64", 2, &[ElementKind::F64], input(0).min(input(1)));
        let kn64 = generate(&min64, &key64, &Cuda);
        assert!(
            !kn64.source.contains("fmin("),
            "f64 Minimum must NOT become fmin"
        );
    }

    #[test]
    #[should_panic(expected = "no half-precision lowering")]
    fn nextafter_f16_is_refused_at_the_emitter() {
        use crate::ir::BinaryOp;
        // The promote-to-f32 half path would compute the f32-lattice neighbor
        // (which demotes right back to `a`) — silently wrong, so the emitter
        // refuses rather than lowers. (The JIT gates this in dtype_compatible.)
        let op = OpDef::elementwise(
            "nextafter",
            2,
            &[ElementKind::F16],
            input(0).binary(BinaryOp::Nextafter, input(1)),
        );
        let _ = generate(&op, &binary_scalar_key(ElementKind::F16, 2), &Cuda);
    }

    #[test]
    #[should_panic(expected = "no half-precision lowering")]
    fn nextafter_bf16_is_refused_at_the_emitter() {
        use crate::ir::BinaryOp;
        let op = OpDef::elementwise(
            "nextafter",
            2,
            &[ElementKind::Bf16],
            input(0).binary(BinaryOp::Nextafter, input(1)),
        );
        let _ = generate(&op, &binary_scalar_key(ElementKind::Bf16, 2), &Cuda);
    }

    // The adversarial review of increment 0a proved the cuda_binary refusal
    // alone was bypassable: the reduction pre-body, RowReduce stages/epilogue,
    // and contraction epilogues lower through accumulator-width helpers that
    // never reach it. The plan-level walk (assert_no_half_nextafter) now
    // guards EVERY Access arm — these three pin the previously-open paths.

    #[test]
    #[should_panic(expected = "must miss honestly")]
    fn nextafter_f16_is_refused_in_a_reduction_body() {
        use crate::ir::{BinaryOp, ReduceOp, konst};
        let op = OpDef::reduction(
            "na_sum",
            1,
            &[ElementKind::F16],
            input(0).binary(BinaryOp::Nextafter, konst(1.0)),
            ReduceOp::Sum,
        );
        let _ = generate(&op, &reduce_key(ElementKind::F16), &Cuda);
    }

    #[test]
    #[should_panic(expected = "must miss honestly")]
    fn nextafter_f16_is_refused_in_a_row_reduce_epilogue() {
        use crate::ir::{BinaryOp, ReduceOp, ReduceStage, reduced};
        let op = OpDef::row_reduce(
            "na_rr",
            1,
            &[ElementKind::F16],
            vec![ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Max,
            }],
            input(0).binary(BinaryOp::Nextafter, reduced(0)),
        );
        let x = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F16, 256);
        let key = structure_key(OpCategory::Softmax, &[x, x], ArchSku::Sm89);
        let _ = generate(&op, &key, &Cuda);
    }

    #[test]
    #[should_panic(expected = "must miss honestly")]
    fn nextafter_f16_is_refused_in_a_contraction_epilogue() {
        use crate::ir::{BinaryOp, ContractionAxes, konst, reduced};
        let op = OpDef::contraction(
            "na_mm",
            &[ElementKind::F16],
            ContractionAxes::matmul(),
            reduced(0).binary(BinaryOp::Nextafter, konst(1.0)),
        );
        let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F16, 256);
        let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F16, 256);
        let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F16, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let _ = generate(&op, &key, &Cuda);
    }

    #[test]
    fn vocab_f16_packed_falls_back_to_tier_b_pair_scalarization() {
        use crate::ir::UnaryOp;
        // A new transcendental at an aligned f16 cell takes the PACKED schedule
        // but the fn itself is Tier B: pair-split through the same scalar f32
        // spelling (bit-identical to the scalar sibling), never a fake packed
        // intrinsic.
        let op = OpDef::elementwise(
            "erfx",
            1,
            &[ElementKind::F16],
            input(0).unary(UnaryOp::Erfc),
        );
        let k = generate(&op, &unary_scalar_key(ElementKind::F16, 256), &Cuda);
        assert_eq!(k.name, "baracuda_gen_erfx_f16_co_v8");
        assert!(k.source.contains("__halves2half2("));
        assert!(k.source.contains("erfcf(__half2float(__low2half("));
        assert!(k.source.contains("erfcf(__half2float(__high2half("));
        assert!(!k.source.contains("h2erfc"), "no invented packed intrinsic");
    }

    // =======================================================================
    // Increment-0b comparison predicates + u8 mask output
    // =======================================================================

    use crate::ir::{BinaryOp, konst};

    /// Fully-aligned binary cell with a **U8 output** operand (inputs `dt`):
    /// the caller-side key shape of an `elementwise_pred` op. Note the aligned
    /// contiguous u8 output keys V8 on its own — the plan must still force the
    /// scalar path (no packed u8 store exists).
    fn pred_key(dt: ElementKind) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 256);
        let o = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::U8, 256);
        structure_key(OpCategory::BinaryElementwise, &[a, a, o], ArchSku::Sm89)
    }

    fn pred_op(name: &str, op: BinaryOp, dt: ElementKind) -> OpDef {
        OpDef::elementwise_pred(name, 2, &[dt], input(0).binary(op, input(1)))
    }

    #[test]
    fn cmp_emission_goldens_f32_u8_store() {
        // One golden per cmp op at f32: the exact C-operator ternary (the NaN
        // semantics carrier), the u8 output signature, and the exact store cast.
        let cases: &[(BinaryOp, &str, &str)] = &[
            (
                BinaryOp::CmpEq,
                "cmp_eq",
                "((float)in0[i] == (float)in1[i] ? 1.0f : 0.0f)",
            ),
            (
                BinaryOp::CmpNe,
                "cmp_ne",
                "((float)in0[i] != (float)in1[i] ? 1.0f : 0.0f)",
            ),
            (
                BinaryOp::CmpLt,
                "cmp_lt",
                "((float)in0[i] < (float)in1[i] ? 1.0f : 0.0f)",
            ),
            (
                BinaryOp::CmpLe,
                "cmp_le",
                "((float)in0[i] <= (float)in1[i] ? 1.0f : 0.0f)",
            ),
            (
                BinaryOp::CmpGt,
                "cmp_gt",
                "((float)in0[i] > (float)in1[i] ? 1.0f : 0.0f)",
            ),
            (
                BinaryOp::CmpGe,
                "cmp_ge",
                "((float)in0[i] >= (float)in1[i] ? 1.0f : 0.0f)",
            ),
        ];
        for (op, name, ternary) in cases {
            let k = generate(
                &pred_op(name, *op, ElementKind::F32),
                &pred_key(ElementKind::F32),
                &Cuda,
            );
            // The op name flows into the entry point (identity is (token, entry_point)).
            assert_eq!(k.name, format!("baracuda_gen_{name}_f32_scalar"));
            // Inputs stay the key dtype; ONLY the output is u8.
            assert!(k.source.contains("const float* __restrict__ in0"), "{name}");
            assert!(
                k.source.contains("unsigned char* __restrict__ out"),
                "{name}"
            );
            // Exact predicate + exact store conversion (0.0f/1.0f -> 0/1).
            let store = format!("out[i] = (unsigned char){ternary};");
            assert!(
                k.source.contains(&store),
                "{name}: missing `{store}` in:\n{}",
                k.source
            );
        }
    }

    #[test]
    fn cmp_f64_golden_uses_double_literals() {
        let k = generate(
            &pred_op("cmp_lt", BinaryOp::CmpLt, ElementKind::F64),
            &pred_key(ElementKind::F64),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_cmp_lt_f64_scalar");
        assert!(
            k.source
                .contains("out[i] = (unsigned char)(in0[i] < in1[i] ? 1.0 : 0.0);")
        );
        assert!(
            !k.source.contains("1.0f"),
            "f64 predicate must not use float literals"
        );
    }

    #[test]
    fn cmp_f16_promotes_to_f32_for_the_compare() {
        // f16 compares via promote-to-f32 — EXACT (half->f32 is a lossless,
        // order-preserving embedding, so the f32 compare decides identically to
        // a native half compare), and the store re-promotes the demoted 1.0/0.0
        // before the integer cast (exact round trip; see store_expr).
        let k = generate(
            &pred_op("cmp_lt", BinaryOp::CmpLt, ElementKind::F16),
            &pred_key(ElementKind::F16),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_cmp_lt_f16_scalar");
        assert!(k.source.contains("#include <cuda_fp16.h>"));
        assert!(
            k.source.contains(
                "((float)__half2float(in0[i]) < (float)__half2float(in1[i]) ? 1.0f : 0.0f)"
            )
        );
        assert!(
            k.source
                .contains("out[i] = (unsigned char)__half2float(__float2half(")
        );
        assert!(k.source.contains("unsigned char* __restrict__ out"));
    }

    #[test]
    fn cmp_u8_out_falls_back_to_scalar_at_aligned_vector_cells() {
        // Packed-classifier fallback golden: a u8-output op at a fully-aligned
        // f32 (V4) or f16 (V8) cell must take the SCALAR path — no float4, no
        // packed half2 struct, no invented packed u8 store. (The aligned
        // contiguous u8 output operand itself keys V8, so without the plan
        // gate the min-width rule alone would still say "vectorize".)
        let kf = generate(
            &pred_op("cmp_ge", BinaryOp::CmpGe, ElementKind::F32),
            &pred_key(ElementKind::F32),
            &Cuda,
        );
        assert!(kf.name.ends_with("_scalar"), "got {}", kf.name);
        assert!(!kf.source.contains("float4"));
        let kh = generate(
            &pred_op("cmp_eq", BinaryOp::CmpEq, ElementKind::F16),
            &pred_key(ElementKind::F16),
            &Cuda,
        );
        assert!(kh.name.ends_with("_scalar"), "got {}", kh.name);
        // `__half2 ` (the pair TYPE, trailing space) must not appear — the
        // scalar promote fn `__half2float` legitimately does.
        assert!(
            !kh.source.contains("__half2 "),
            "no packed pair type: {}",
            kh.name
        );
        assert!(!kh.source.contains("__halves2half2"), "no pair re-join");
        assert!(!kh.source.contains("_vec"), "no packed vector struct");
    }

    #[test]
    #[should_panic(expected = "hetero output")]
    fn hetero_out_vectorized_schedule_is_refused_by_the_emitter_backstop() {
        use crate::backend::Backend;
        use crate::ir::BinaryOp;
        use crate::plan::{KernelPlan, Schedule};
        // build_plan never produces this plan (it forces Scalar/Strided for
        // u8-out) — but the backstop must hold INDEPENDENTLY of the plan gate
        // (the 0a lesson: gate every layer). A future schedule-selection
        // change must trip here, never emit a float4 store into a u8 buffer.
        // (Review-caught gap: deleting the backstop left the suite green.)
        let key = pred_key(ElementKind::F32);
        let body = input(0).binary(BinaryOp::CmpLt, input(1)).0;
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 2,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::U8,
            schedule: Schedule::Vectorized { width: 4 },
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    fn cmp_bf16_u8_store_bridges_through_bfloat162float() {
        // The direct __nv_bfloat16→integer conversion is a header-configuration-
        // dependent C++ overload (a headerless-nvrtc hazard) — the store must
        // bridge through __bfloat162float. (Review-caught gap: the Bf16 arm of
        // store_expr had zero coverage; only the F16 form was pinned.)
        let k = generate(
            &pred_op("cmp_lt", BinaryOp::CmpLt, ElementKind::Bf16),
            &pred_key(ElementKind::Bf16),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_cmp_lt_bf16_scalar");
        assert!(
            k.source
                .contains("out[i] = (unsigned char)__bfloat162float("),
            "bf16 u8 store must bridge through __bfloat162float:\n{}",
            k.source
        );
    }

    #[test]
    #[should_panic(expected = "index/address dtype only")]
    fn u32_compute_plan_is_rejected_by_the_emitter() {
        // Review (Model-A wiring): U32 gained a scalar_ctype ("unsigned int") for
        // the gather/scatter index LOAD, which must NOT open a U32 arithmetic
        // compute path. A plain (non-indexed) U32 elementwise add must be rejected
        // — else it silently lowers to an `unsigned int` kernel and bypasses the
        // int-div backstop. (bincount self-indexes → exempt; covered elsewhere.)
        let op = OpDef::elementwise("add", 2, &[ElementKind::U32], input(0) + input(1));
        let _ = generate(&op, &binary_scalar_key(ElementKind::U32, 4), &Cuda);
    }

    #[test]
    fn int_compute_dtypes_are_supported_after_the_0c_audit() {
        use crate::backend::Backend;
        // Increment 0c replaced the 0b uniform-u8 hold: U8 and S8 are audited
        // COMPUTE dtypes now (wrapping semantics + the int-only op set), so
        // supports_dtype says yes — and the per-OP legality lives in
        // dtype_compatible / the plan gate, NOT here (a uniform-U8 Div region
        // still declines; pinned in jit.rs). Both directions:
        for dt in [
            ElementKind::U8,
            ElementKind::S8,
            ElementKind::I32,
            ElementKind::I64,
        ] {
            assert!(
                Cuda.supports_dtype(dt),
                "{dt:?} is an audited compute dtype"
            );
        }
        for dt in [
            ElementKind::Bool, // FKC has no Bool — masks ride as U8
            ElementKind::S4,
            ElementKind::U4,
            ElementKind::Bin,
            ElementKind::Fp8E4M3,
            ElementKind::Fp8E5M2,
            ElementKind::Complex32,
            ElementKind::Complex64,
        ] {
            assert!(!Cuda.supports_dtype(dt), "{dt:?} must keep declining");
        }
        // The u8-OUT predicate path is unchanged by the flip.
        let k = generate(
            &pred_op("cmp_gt", BinaryOp::CmpGt, ElementKind::F32),
            &pred_key(ElementKind::F32),
            &Cuda,
        );
        assert!(k.source.contains("unsigned char* __restrict__ out"));
    }

    #[test]
    fn cmp_u8_out_strided_cell_stores_u8_at_the_unraveled_offset() {
        // A strided input routes the u8-output op to the strided emitter, which
        // must carry the same u8 signature + store cast at out[oo].
        let a = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::F32, 256); // transposed
        let b = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::U8, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, b, o], ArchSku::Sm89);
        let k = generate(
            &pred_op("cmp_ne", BinaryOp::CmpNe, ElementKind::F32),
            &key,
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_cmp_ne_f32_strided_r2");
        assert!(k.source.contains("unsigned char* __restrict__ out"));
        assert!(k.source.contains(
            "out[oo] = (unsigned char)((float)in0[o0] != (float)in1[o1] ? 1.0f : 0.0f);"
        ));
    }

    #[test]
    fn nested_cmp_in_float_body_emits_inline_ternary_and_float_store() {
        // The mask-multiply (relu-backward) shape: dy * (x > 0), out_dtype NONE
        // — the cmp is an inline 0.0f/1.0f float, the output stays the key
        // dtype, and no u8 machinery appears.
        let op = OpDef::elementwise(
            "relu_bw",
            2,
            &[ElementKind::F32],
            input(0) * input(1).binary(BinaryOp::CmpGt, konst(0.0)),
        );
        assert_eq!(op.out_dtype, None);
        let k = generate(&op, &binary_scalar_key(ElementKind::F32, 4), &Cuda);
        assert_eq!(k.name, "baracuda_gen_relu_bw_f32_scalar");
        assert!(
            k.source
                .contains("out[i] = (in0[i] * ((float)in1[i] > (float)0.0 ? 1.0f : 0.0f));")
        );
        assert!(k.source.contains("float* __restrict__ out"));
        assert!(!k.source.contains("unsigned char"));
        // …and at a fully-aligned cell the SAME body vectorizes normally
        // (out_dtype None leaves the schedule untouched).
        let kv = generate(&op, &binary_scalar_key(ElementKind::F32, 256), &Cuda);
        assert_eq!(kv.name, "baracuda_gen_relu_bw_f32_co_v4");
        assert!(
            kv.source
                .contains("((float)v1.x > (float)0.0 ? 1.0f : 0.0f)")
        );
    }

    #[test]
    fn cmp_float_mask_toplevel_is_legal_with_none_out_dtype() {
        // A TOP-LEVEL cmp with out_dtype = None is a legal float-mask kernel
        // (1.0f/0.0f stored in the key dtype) — validate only gates Some(U8).
        let op = OpDef::elementwise(
            "gt_mask",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::CmpGt, input(1)),
        );
        let k = generate(&op, &binary_scalar_key(ElementKind::F32, 4), &Cuda);
        assert!(
            k.source
                .contains("out[i] = ((float)in0[i] > (float)in1[i] ? 1.0f : 0.0f);")
        );
        assert!(!k.source.contains("unsigned char"));
    }

    #[test]
    #[should_panic(expected = "requires the body ROOT to be a comparison")]
    fn u8_out_with_non_cmp_body_is_rejected() {
        // A non-predicate body under a u8 store would truncate real floats
        // silently — authoring error, panic (honest miss discipline).
        let mut op = OpDef::elementwise("bad", 1, &[ElementKind::F32], input(0).relu());
        op.out_dtype = Some(ElementKind::U8);
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::U8, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let _ = generate(&op, &key, &Cuda);
    }

    #[test]
    #[should_panic(expected = "requires the POST-expr ROOT to be a comparison")]
    fn u8_out_reduction_with_non_cmp_post_is_rejected() {
        use crate::ir::ReduceOp;
        // count(x > 0) stored as u8 with the IDENTITY post: the store is the raw
        // ACCUMULATOR (a count up to 1024), not a 0/1 predicate — u8 would
        // truncate silently. 0e admits a U8-out reduction ONLY when the POST-expr
        // root is a Cmp* (the honest any/all boolean-reduce); this must panic.
        let mut op = OpDef::reduction(
            "count_pos",
            1,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::CmpGt, konst(0.0)),
            ReduceOp::Sum,
        );
        op.out_dtype = Some(ElementKind::U8);
        let a = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[4096], &[1], ElementKind::U8, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let _ = generate(&op, &key, &Cuda);
    }

    #[test]
    #[should_panic(expected = "stores its accumulator")]
    fn u8_out_under_row_reduce_is_rejected() {
        let mut op = softmax_op(ElementKind::F32);
        op.out_dtype = Some(ElementKind::U8);
        let _ = generate(&op, &rr_key(ElementKind::F32, OpCategory::Softmax), &Cuda);
    }

    #[test]
    #[should_panic(expected = "stores its accumulator")]
    fn u8_out_under_contraction_is_rejected() {
        use crate::ir::{ContractionAxes, reduced};
        let mut op = OpDef::contraction(
            "mm",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0).binary(BinaryOp::CmpGt, konst(0.0)),
        );
        op.out_dtype = Some(ElementKind::U8);
        let lhs = OperandDesc::new(2, &[8, 64], &[64, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[64, 32], &[32, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 32], &[32, 1], ElementKind::U8, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let _ = generate(&op, &key, &Cuda);
    }

    #[test]
    #[should_panic(expected = "the only hetero output dtype there is U8")]
    fn non_u8_out_dtype_is_rejected() {
        let mut op = OpDef::elementwise(
            "bad_f16_out",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::CmpLt, input(1)),
        );
        op.out_dtype = Some(ElementKind::F16);
        let _ = generate(&op, &pred_key(ElementKind::F32), &Cuda);
    }

    // =======================================================================
    // Increment-0c integer compute dtypes + bitwise/shift/logical ops
    // =======================================================================

    fn int_op(name: &str, bop: BinaryOp, dt: ElementKind) -> OpDef {
        OpDef::elementwise(name, 2, &[dt], input(0).binary(bop, input(1)))
    }

    #[test]
    fn bitwise_emission_goldens_i32_i64() {
        // One golden per bitwise/shift op at i32 AND i64: the exact raw C
        // operator (the bespoke functor bodies verbatim — `a & b`, `a << b`,
        // `a >> b` with NO masking/clamping: out-of-range shifts inherit the
        // architecture's behavior, signed >> is arithmetic, exactly as
        // binary_bitwise_*_int.cu documents).
        let cases: &[(BinaryOp, &str, &str)] = &[
            (BinaryOp::BitAnd, "band", "&"),
            (BinaryOp::BitOr, "bor", "|"),
            (BinaryOp::BitXor, "bxor", "^"),
            (BinaryOp::Shl, "shl", "<<"),
            (BinaryOp::Shr, "shr", ">>"),
        ];
        for &(bop, name, c) in cases {
            let ki = generate(
                &int_op(name, bop, ElementKind::I32),
                &binary_scalar_key(ElementKind::I32, 4),
                &Cuda,
            );
            assert_eq!(ki.name, format!("baracuda_gen_{name}_i32_scalar"));
            assert!(ki.source.contains("const int* __restrict__ in0"), "{bop:?}");
            let store = format!("out[i] = (in0[i] {c} in1[i]);");
            assert!(
                ki.source.contains(&store),
                "{bop:?} i32: missing `{store}` in:\n{}",
                ki.source
            );
            let kl = generate(
                &int_op(name, bop, ElementKind::I64),
                &binary_scalar_key(ElementKind::I64, 8),
                &Cuda,
            );
            assert_eq!(kl.name, format!("baracuda_gen_{name}_i64_scalar"));
            assert!(
                kl.source.contains("const long long* __restrict__ in0"),
                "{bop:?}"
            );
            assert!(kl.source.contains(&store), "{bop:?} i64: missing `{store}`");
            // Header-light: int kernels need no includes (nvrtc headerless).
            assert!(
                !ki.source.contains("#include"),
                "{bop:?} i32 must be headerless"
            );
        }
    }

    #[test]
    fn bitwise_emission_goldens_i8_u8() {
        // 8-bit cells: `signed char` / `unsigned char` pointers and the SAME
        // raw operators with NO defeating casts — deliberately. C integer
        // promotion widens both operands to `int` (sign-extended for i8,
        // zero-extended for u8) and the 8-bit store truncates the result:
        // - and/or/xor: extension bits op among themselves and truncate away —
        //   bit-identical to a native 8-bit op;
        // - Shl at u8: the 32-bit shift result truncates mod 2^8 = the native
        //   wrapping 8-bit shift for in-range amounts (amounts 8..31 take the
        //   promoted semantics — there is no bespoke 8-bit shift to defer to,
        //   so promote-then-truncate IS the documented 0c contract);
        // - Shr: the promoted value's high bits are the extension, so the
        //   result always fits 8 bits — arithmetic for i8 (sign replicates),
        //   logical for u8 (zero-extension), matching the signed/unsigned
        //   split the bespoke i32/i64 kernels pin.
        let ku = generate(
            &int_op("shl", BinaryOp::Shl, ElementKind::U8),
            &binary_scalar_key(ElementKind::U8, 1),
            &Cuda,
        );
        assert_eq!(ku.name, "baracuda_gen_shl_u8_scalar");
        assert!(ku.source.contains("const unsigned char* __restrict__ in0"));
        assert!(ku.source.contains("unsigned char* __restrict__ out"));
        assert!(
            ku.source.contains("out[i] = (in0[i] << in1[i]);"),
            "{}",
            ku.source
        );
        assert!(
            !ku.source.contains("(int)"),
            "no defeating casts — promotion is the contract"
        );
        let ks = generate(
            &int_op("shr", BinaryOp::Shr, ElementKind::S8),
            &binary_scalar_key(ElementKind::S8, 1),
            &Cuda,
        );
        assert_eq!(ks.name, "baracuda_gen_shr_i8_scalar");
        assert!(ks.source.contains("const signed char* __restrict__ in0"));
        assert!(
            ks.source.contains("out[i] = (in0[i] >> in1[i]);"),
            "{}",
            ks.source
        );
        let ka = generate(
            &int_op("band", BinaryOp::BitAnd, ElementKind::U8),
            &binary_scalar_key(ElementKind::U8, 1),
            &Cuda,
        );
        assert!(ka.source.contains("out[i] = (in0[i] & in1[i]);"));
        let kx = generate(
            &int_op("bxor", BinaryOp::BitXor, ElementKind::S8),
            &binary_scalar_key(ElementKind::S8, 1),
            &Cuda,
        );
        assert!(kx.source.contains("out[i] = (in0[i] ^ in1[i]);"));
    }

    #[test]
    fn logical_emission_goldens_u8() {
        // The exact normalize-then-op ternaries of binary_logical_*_bool.cu:
        // inputs are normalized with `!= 0` BEFORE the op, so unnormalized
        // bytes behave boolean (2 && 4 == 1, never the bitwise 2 & 4 == 0)
        // and the output is strictly 0/1.
        let cases: &[(BinaryOp, &str, &str)] = &[
            (
                BinaryOp::LogicalAnd,
                "land",
                "out[i] = ((in0[i] != 0 && in1[i] != 0) ? 1 : 0);",
            ),
            (
                BinaryOp::LogicalOr,
                "lor",
                "out[i] = ((in0[i] != 0 || in1[i] != 0) ? 1 : 0);",
            ),
            (
                BinaryOp::LogicalXor,
                "lxor",
                "out[i] = (((in0[i] != 0) != (in1[i] != 0)) ? 1 : 0);",
            ),
        ];
        for &(bop, name, store) in cases {
            let k = generate(
                &int_op(name, bop, ElementKind::U8),
                &binary_scalar_key(ElementKind::U8, 1),
                &Cuda,
            );
            assert_eq!(k.name, format!("baracuda_gen_{name}_u8_scalar"));
            assert!(
                k.source.contains("const unsigned char* __restrict__ in0"),
                "{bop:?}"
            );
            assert!(
                k.source.contains(store),
                "{bop:?}: missing `{store}` in:\n{}",
                k.source
            );
        }
    }

    #[test]
    fn int_infix_wrapping_goldens() {
        // Wrapping infix arithmetic at the newly-audited 8-bit dtypes: the
        // native operators, no float detour, no casts (promotion + store
        // truncation = mod-2^8 wrapping; see the ir.rs table). i64 stays the
        // native `long long` operators (the pre-0c behavior, unregressed).
        let addu = OpDef::elementwise("add", 2, &[ElementKind::U8], input(0) + input(1));
        let ku = generate(&addu, &binary_scalar_key(ElementKind::U8, 1), &Cuda);
        assert_eq!(ku.name, "baracuda_gen_add_u8_scalar");
        assert!(ku.source.contains("out[i] = (in0[i] + in1[i]);"));
        assert!(
            !ku.source.contains("float"),
            "no float detour in a u8 kernel"
        );
        let muls = OpDef::elementwise("mul", 2, &[ElementKind::S8], input(0) * input(1));
        let ks = generate(&muls, &binary_scalar_key(ElementKind::S8, 1), &Cuda);
        assert_eq!(ks.name, "baracuda_gen_mul_i8_scalar");
        assert!(ks.source.contains("out[i] = (in0[i] * in1[i]);"));
        let subl = OpDef::elementwise("sub", 2, &[ElementKind::I64], input(0) - input(1));
        let kl = generate(&subl, &binary_scalar_key(ElementKind::I64, 8), &Cuda);
        assert!(kl.source.contains("out[i] = (in0[i] - in1[i]);"));
    }

    #[test]
    fn aligned_int_cells_stay_scalar_no_int_vectorization() {
        // Increment-0c scope pin (v1): int dtypes take the scalar/strided
        // paths — `vector_type` has NO int arm today (this was ALREADY true
        // for i32/i64 pre-0c: an aligned i32 add fell back to scalar, pinned
        // by the count_unit contract test), and 0c deliberately does not
        // invent an int4 path. A fully-aligned (V4-keying) i32 cell and a
        // (V8-keying) u8 cell must both emit the scalar kernel.
        let ki = generate(
            &int_op("band", BinaryOp::BitAnd, ElementKind::I32),
            &binary_scalar_key(ElementKind::I32, 256),
            &Cuda,
        );
        assert_eq!(
            ki.name, "baracuda_gen_band_i32_scalar",
            "aligned i32 stays scalar"
        );
        assert!(!ki.source.contains("int4"), "no invented int vectorization");
        let addu = OpDef::elementwise("add", 2, &[ElementKind::U8], input(0) + input(1));
        let ku = generate(&addu, &binary_scalar_key(ElementKind::U8, 256), &Cuda);
        assert_eq!(
            ku.name, "baracuda_gen_add_u8_scalar",
            "aligned u8 stays scalar"
        );
        assert!(!ku.source.contains("_vec"), "no packed vector struct");
    }

    #[test]
    fn int_strided_cell_unravels_with_the_same_operator() {
        // Transposed i32 views route to the strided emitter with the same raw
        // C operator at the unraveled offsets.
        let t = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::I32, 256);
        let d = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::I32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[t, d, d], ArchSku::Sm89);
        let k = generate(
            &int_op("bxor", BinaryOp::BitXor, ElementKind::I32),
            &key,
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_bxor_i32_strided_r2");
        assert!(k.source.contains("out[oo] = (in0[o0] ^ in1[o1]);"));
    }

    // ---- plan-gate validate-rejects (assert_int_op_admissibility): both
    // directions of the ir.rs admissibility table, every class of illegal cell.
    //
    // These call crate::build_plan DIRECTLY, not generate(): the panic must
    // originate from the PLAN gate. Via generate() five of these rejections
    // also passed on the EMITTER backstops (empirically: widening plan.rs's
    // logical U8-only arm to all int dtypes left the whole suite green —
    // review finding), so a widened/deleted gate arm went undetected. With
    // build_plan, a gate mutation turns into a should_panic FAILURE; the
    // emitter backstops keep their own hand-built-plan tests below.

    #[test]
    #[should_panic(expected = "is int-only")]
    fn bitand_at_f32_is_rejected_at_the_plan_gate() {
        let op = int_op("band", BinaryOp::BitAnd, ElementKind::F32);
        let key = binary_scalar_key(ElementKind::F32, 4);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "is int-only")]
    fn shl_at_f16_is_rejected_at_the_plan_gate() {
        let op = int_op("shl", BinaryOp::Shl, ElementKind::F16);
        let key = binary_scalar_key(ElementKind::F16, 2);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "U8 (Bool)-only")]
    fn logical_and_at_i32_is_rejected_at_the_plan_gate() {
        // Bespoke logical instantiates ONLY uint8_t — wider ints miss honestly.
        let op = int_op("land", BinaryOp::LogicalAnd, ElementKind::I32);
        let key = binary_scalar_key(ElementKind::I32, 4);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "integer division is rejected")]
    fn int_div_is_rejected_at_the_plan_gate() {
        // No bespoke int elementwise div; C `/0` is device-UB — a uniform-u8
        // Div must not ride in on the 0c dtype flip.
        let op = OpDef::elementwise("div", 2, &[ElementKind::U8], input(0) / input(1));
        let key = binary_scalar_key(ElementKind::U8, 1);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "has no integer lowering")]
    fn float_fn_at_u8_is_rejected_at_the_plan_gate() {
        let op = OpDef::elementwise("m", 2, &[ElementKind::U8], input(0).max(input(1)));
        let key = binary_scalar_key(ElementKind::U8, 1);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "has no integer lowering")]
    fn cmp_at_i32_is_rejected_at_the_plan_gate() {
        // Bespoke cmp is fp-only (binary_cmp_*_fp.cu) — int cmp misses honestly.
        let op = int_op("lt", BinaryOp::CmpLt, ElementKind::I32);
        let key = binary_scalar_key(ElementKind::I32, 4);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "has no integer lowering")]
    fn unary_at_int_is_rejected_at_the_plan_gate() {
        use crate::ir::UnaryOp;
        // The bespoke unary elementwise surface is fp-only — Abs at i32 must
        // miss honestly at the PLAN gate (not just cuda_unary's panic, which
        // the reduction-class paths bypass — the 0a lesson).
        let op = OpDef::elementwise("a", 1, &[ElementKind::I32], input(0).unary(UnaryOp::Abs));
        let key = unary_scalar_key(ElementKind::I32, 4);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "Const at int dtype")]
    fn const_at_int_is_rejected_at_the_plan_gate() {
        // A Const is spelled as an f64 C literal — at i64 it would silently
        // run double math (f64 cannot represent all i64). Reject, don't drift.
        let op = OpDef::elementwise("addk", 1, &[ElementKind::I64], input(0) + konst(2.0));
        let key = unary_scalar_key(ElementKind::I64, 8);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn bitand_under_reduction_is_rejected_at_the_plan_gate() {
        use crate::ir::ReduceOp;
        // Int-only ops are Elementwise-only in 0c: the reduction pre-body
        // lowers through the FLOAT accumulator spellers (binary_f32/f64),
        // which have no int arms — the gate must fire before the emitter.
        let op = OpDef::reduction(
            "s",
            1,
            &[ElementKind::I32],
            input(0).binary(BinaryOp::BitAnd, input(0)),
            ReduceOp::Sum,
        );
        let key = reduce_key(ElementKind::I32);
        let _ = crate::build_plan(&op, &key);
    }

    // ---- 8-bit composition pin (plan-gate rule 3): at U8/S8 every operand
    // of an int-only op must be a LEAF Input. A composed operand's value
    // differs between the inlined (un-truncated promoted-int) and hoisted
    // (8-bit tmp, truncated) spellings — (in0+in1)>>in1 at u8 with
    // (200,100,1) is 300>>1=150 inlined but 44>>1=22 hoisted — so one body
    // would compute two results depending on DAG sharing.

    #[test]
    #[should_panic(expected = "requires LEAF Input operands")]
    fn composed_operand_of_shr_at_u8_is_rejected_at_the_plan_gate() {
        // Add-fed Shr: the shifted value observes the un-truncated sum.
        let op = OpDef::elementwise(
            "addshr",
            2,
            &[ElementKind::U8],
            (input(0) + input(1)).binary(BinaryOp::Shr, input(1)),
        );
        let key = binary_scalar_key(ElementKind::U8, 1);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "requires LEAF Input operands")]
    fn composed_operand_of_logical_and_at_u8_is_rejected_at_the_plan_gate() {
        // Add-fed LogicalAnd: the `!= 0` test observes the un-truncated sum
        // (255+1 is 0 truncated / false, but 256 promoted / true).
        let op = OpDef::elementwise(
            "addland",
            2,
            &[ElementKind::U8],
            (input(0) + input(1)).binary(BinaryOp::LogicalAnd, input(1)),
        );
        let key = binary_scalar_key(ElementKind::U8, 1);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "requires LEAF Input operands")]
    fn composed_shift_amount_at_i8_is_rejected_at_the_plan_gate() {
        // The shift AMOUNT position observes the promoted value too — the rhs
        // operand must be a leaf just like the lhs.
        let op = OpDef::elementwise(
            "shlamt",
            2,
            &[ElementKind::S8],
            input(0).binary(BinaryOp::Shl, input(0) + input(1)),
        );
        let key = binary_scalar_key(ElementKind::S8, 1);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    fn pure_ring_composition_at_u8_still_lowers() {
        // Add(Add(x,y),x) at u8 — NO int-only op involved: wrapping ring
        // composition is congruent under deferred truncation (promote, add,
        // store-truncate ≡ native 8-bit wrapping adds), so the pin must not
        // touch it.
        let op = OpDef::elementwise(
            "add3",
            2,
            &[ElementKind::U8],
            (input(0) + input(1)) + input(0),
        );
        let k = generate(&op, &binary_scalar_key(ElementKind::U8, 1), &Cuda);
        assert_eq!(k.name, "baracuda_gen_add3_u8_scalar");
        assert!(
            k.source.contains("out[i] = ((in0[i] + in1[i]) + in0[i]);"),
            "nested wrapping add must still lower:\n{}",
            k.source
        );
    }

    #[test]
    fn composed_operand_of_bitand_at_i32_still_lowers() {
        // At I32/I64 promotion never widens past the compute width — there is
        // no un-truncated wider value to observe, so compositions stay legal.
        let op = OpDef::elementwise(
            "addband",
            2,
            &[ElementKind::I32],
            (input(0) + input(1)).binary(BinaryOp::BitAnd, input(1)),
        );
        let k = generate(&op, &binary_scalar_key(ElementKind::I32, 4), &Cuda);
        assert_eq!(k.name, "baracuda_gen_addband_i32_scalar");
        assert!(
            k.source.contains("out[i] = ((in0[i] + in1[i]) & in1[i]);"),
            "i32 composed bitand must still lower:\n{}",
            k.source
        );
    }

    // ---- emitter backstops, independent of the plan gate (the 0a lesson:
    // gate every layer; these construct the plan manually to prove the
    // speller itself refuses even if a future schedule change bypasses
    // build_plan).

    #[test]
    #[should_panic(expected = "has no f32 lowering")]
    fn int_only_op_at_float_is_refused_by_the_emitter_backstop() {
        use crate::backend::Backend;
        use crate::plan::{KernelPlan, Schedule};
        let key = binary_scalar_key(ElementKind::F32, 4);
        let body = input(0).binary(BinaryOp::BitAnd, input(1)).0;
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 2,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::Scalar,
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "must miss honestly at the plan gate")]
    fn float_fn_at_int_is_refused_by_the_emitter_backstop() {
        use crate::backend::Backend;
        use crate::plan::{KernelPlan, Schedule};
        let key = binary_scalar_key(ElementKind::U8, 1);
        let body = input(0).binary(BinaryOp::FmaxIeee, input(1)).0;
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 2,
            dtype: ElementKind::U8,
            out_dtype: ElementKind::U8,
            schedule: Schedule::Scalar,
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "bespoke logical surface")]
    fn logical_at_wide_int_is_refused_by_the_emitter_backstop() {
        use crate::backend::Backend;
        use crate::plan::{KernelPlan, Schedule};
        let key = binary_scalar_key(ElementKind::I64, 8);
        let body = input(0).binary(BinaryOp::LogicalXor, input(1)).0;
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 2,
            dtype: ElementKind::I64,
            out_dtype: ElementKind::I64,
            schedule: Schedule::Scalar,
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "infix Div has no integer lowering")]
    fn int_div_is_refused_by_the_emitter_backstop() {
        use crate::backend::Backend;
        use crate::plan::{KernelPlan, Schedule};
        // Div is spelled by the shared dtype-blind lower_expr (`a / b` for
        // every dtype) — no per-op speller exists to panic, so Cuda::lower's
        // body-walk is the ONLY emitter-level guard against a plan-gate
        // bypass emitting device-UB `/0` integer division.
        let key = binary_scalar_key(ElementKind::U8, 1);
        let body = (input(0) / input(1)).0;
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 2,
            dtype: ElementKind::U8,
            out_dtype: ElementKind::U8,
            schedule: Schedule::Scalar,
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "Const at an integer dtype")]
    fn int_const_is_refused_by_the_emitter_backstop() {
        use crate::backend::Backend;
        use crate::plan::{KernelPlan, Schedule};
        // A Const is spelled as an f64 C literal by the shared backend code —
        // without this backstop a plan-gate bypass would silently run double
        // math inside an i64 kernel (f64 cannot represent all i64).
        let key = unary_scalar_key(ElementKind::I64, 8);
        let body = (input(0) + konst(2.0)).0;
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 1,
            dtype: ElementKind::I64,
            out_dtype: ElementKind::I64,
            schedule: Schedule::Scalar,
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    // ===== increment 0d: Coord(axis) — the iota/coordinate leaf =============

    use crate::ir::coord;

    /// Rank-2 contiguous, fully aligned cell — the shape whose ALIGNED inputs
    /// would normally vectorize (V4 at f32): exactly the cell the Coord
    /// routing must force onto Strided.
    fn coord_key_2d(dt: ElementKind, n_operands: usize) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], dt, 256);
        let operands: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89)
    }

    /// The increment-0d proof-vehicle body: `out[i,j] = x[i,j] * (j >= i + k)`
    /// — the triu mask as a mask-multiply (`k` baked as a Const; `k = 0.0` is
    /// the main diagonal).
    fn triu_mask_op(name: &str, dt: ElementKind, k: f64) -> OpDef {
        OpDef::elementwise(
            name,
            1,
            &[dt],
            input(0) * coord(1).binary(BinaryOp::CmpGe, coord(0) + konst(k)),
        )
    }

    #[test]
    fn coord_triu_mask_f32_golden_strided_unravel_and_float_cast() {
        // The headline golden: a CONTIGUOUS (aligned, would-be-V4) cell still
        // takes the strided emitter, the row-major unravel is emitted even
        // though the single input is contiguous (the output offset needs the
        // same c{d}s), and Coord spells the exact `(float)c{d}` cast into the
        // compute-dtype compare.
        let k = generate(
            &triu_mask_op("triu_mask", ElementKind::F32, 0.0),
            &coord_key_2d(ElementKind::F32, 2),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_triu_mask_f32_strided_r2");
        // Unravel present despite all-contiguous operands (last axis fastest).
        assert!(
            k.source
                .contains("long long c1 = lin % shape1; lin /= shape1;")
        );
        assert!(
            k.source
                .contains("long long c0 = lin % shape0; lin /= shape0;")
        );
        // The compare is decided in the compute dtype: both Coord casts and
        // the 0b compute-dtype cmp casts compose.
        assert!(
            k.source.contains(
                "out[oo] = (in0[o0] * ((float)(float)c1 >= (float)((float)c0 + 0.0) ? 1.0f : 0.0f));"
            ),
            "triu-mask store golden missing in:\n{}",
            k.source
        );
        // No vector machinery leaked in.
        assert!(!k.source.contains("float4"));
    }

    #[test]
    fn coord_pure_iota_all_contiguous_zero_inputs_golden() {
        // out = coord(1) with ZERO inputs: nothing but the output needs the
        // unravel — the emitter must still emit it (the output-offset unravel
        // produces the c{d}s a Coord reads).
        let op = OpDef::elementwise("iota1", 0, &[ElementKind::F32], coord(1));
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_iota1_f32_strided_r2");
        assert!(
            k.source
                .contains("long long c1 = lin % shape1; lin /= shape1;")
        );
        assert!(k.source.contains("long long oo = c0*so_0 + c1*so_1;"));
        assert!(k.source.contains("out[oo] = (float)c1;"), "{}", k.source);
        assert!(
            !k.source.contains("in0"),
            "a 0-input op has no input pointers"
        );
    }

    #[test]
    fn coord_f64_golden_uses_double_cast_and_literals() {
        let k = generate(
            &triu_mask_op("triu_mask", ElementKind::F64, 0.0),
            &coord_key_2d(ElementKind::F64, 2),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_triu_mask_f64_strided_r2");
        assert!(
            k.source
                .contains("out[oo] = (in0[o0] * ((double)c1 >= ((double)c0 + 0.0) ? 1.0 : 0.0));"),
            "f64 triu-mask store golden missing in:\n{}",
            k.source
        );
        assert!(
            !k.source.contains("(float)"),
            "no float casts in the f64 kernel"
        );
    }

    #[test]
    fn coord_alibi_body_with_launch_param_golden() {
        // (coord(1) - coord(0)) * param(0) — the alibi relative-position shape
        // with a runtime slope: zero tensor inputs, one f32 launch param.
        let op = OpDef::elementwise(
            "alibi",
            0,
            &[ElementKind::F32],
            (coord(1) - coord(0)) * param(0),
        );
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_alibi_f32_strided_r2");
        assert!(
            k.source.contains("long long n, float p0)"),
            "p0 rides the launch signature"
        );
        assert!(
            k.source
                .contains("out[oo] = (((float)c1 - (float)c0) * p0);"),
            "alibi store golden missing in:\n{}",
            k.source
        );
    }

    #[test]
    fn coord_body_never_vectorizes_aligned_v4_cell_takes_strided() {
        use crate::plan::Schedule;
        // The routing pin, via build_plan DIRECTLY: a fully-aligned contiguous
        // f32 cell (which vectorizes to float4 for a coord-free body — see
        // f32_contiguous_vectorizes_to_float4) must take Strided when the body
        // contains a Coord; Scalar is equally illegal (no c{d}s there).
        let op = triu_mask_op("triu_mask", ElementKind::F32, 0.0);
        let key = coord_key_2d(ElementKind::F32, 2);
        let plan = crate::build_plan(&op, &key);
        assert!(
            matches!(plan.schedule, Schedule::Strided),
            "Coord body must route Strided, got {:?}",
            plan.schedule
        );
        // Sibling sanity: the SAME cell without the Coord still vectorizes —
        // the routing is keyed on the body, not the cell.
        let plain = OpDef::elementwise("scale", 1, &[ElementKind::F32], input(0) * konst(2.0));
        assert!(matches!(
            crate::build_plan(&plain, &key).schedule,
            Schedule::Vectorized { width: 4 }
        ));
    }

    // ---- plan-gate rejections, via build_plan DIRECTLY (a gate mutation must
    // turn into a should_panic failure; the emitter backstops have their own
    // hand-built-plan tests below).

    #[test]
    #[should_panic(expected = "requires an f32/f64 compute dtype")]
    fn coord_at_f16_is_rejected_at_the_plan_gate() {
        // f16's max exactly-representable integer is 2048 — real axis extents
        // exceed it, so a half coordinate would silently round.
        let op = triu_mask_op("triu_mask", ElementKind::F16, 0.0);
        let key = coord_key_2d(ElementKind::F16, 2);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "requires an f32/f64 compute dtype")]
    fn coord_at_i32_is_rejected_at_the_plan_gate() {
        // The coordinate is spelled as a float cast — the same double-math
        // hazard as Const/Param at int dtypes (int-literal spelling queued).
        let op = OpDef::elementwise("shift_by_col", 1, &[ElementKind::I32], input(0) + coord(1));
        let key = coord_key_2d(ElementKind::I32, 2);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "Elementwise-only in 0d")]
    fn coord_in_a_reduction_body_is_rejected_at_the_plan_gate() {
        use crate::ir::ReduceOp;
        // A coordinate along a folded axis is ambiguous — which fold
        // iteration produced the output element?
        let op = OpDef::reduction(
            "wsum",
            1,
            &[ElementKind::F32],
            input(0) * coord(0),
            ReduceOp::Sum,
        );
        let key = reduce_key(ElementKind::F32);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "Elementwise-only in 0d")]
    fn coord_in_a_row_reduce_epilogue_is_rejected_at_the_plan_gate() {
        use crate::ir::{ReduceOp, ReduceStage, reduced};
        // The RowReduce epilogue iterates the (row, j) space, not the
        // elementwise output coordinate space.
        let op = OpDef::row_reduce(
            "rr_coord",
            1,
            &[ElementKind::F32],
            vec![ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Sum,
            }],
            reduced(0) + coord(0),
        );
        let key = rr_key(ElementKind::F32, OpCategory::Normalization);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "Elementwise-only in 0d")]
    fn coord_in_a_contraction_epilogue_is_rejected_at_the_plan_gate() {
        use crate::ir::{ContractionAxes, reduced};
        // The contraction epilogue iterates (m, n), not an elementwise cell.
        let op = OpDef::contraction(
            "mm_coord",
            &[ElementKind::F32],
            ContractionAxes::matmul(),
            reduced(0) * coord(0),
        );
        let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(2, &[8, 4096], &[4096, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Gemm, &[lhs, rhs, out], ArchSku::Sm89);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "iteration space has no such coordinate")]
    fn coord_axis_out_of_range_is_rejected_at_the_plan_gate() {
        // coord(2) on a rank-2 cell — no such axis exists.
        let op = OpDef::elementwise("iota2", 0, &[ElementKind::F32], coord(2));
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a], ArchSku::Sm89);
        let _ = crate::build_plan(&op, &key);
    }

    // ---- emitter backstops, independent of the plan gate (hand-built plans:
    // a future gate mutation or schedule-selection change must trip HERE, not
    // silently emit a rounding half coordinate / an int-kernel float cast /
    // an undefined c{d} identifier).

    #[test]
    #[should_panic(expected = "at non-float dtype")]
    fn coord_at_int_dtype_is_refused_by_the_emitter_backstop() {
        use crate::backend::Backend;
        use crate::plan::{KernelPlan, Schedule};
        let key = coord_key_2d(ElementKind::I32, 2);
        let body = (input(0) + coord(1)).0;
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 1,
            dtype: ElementKind::I32,
            out_dtype: ElementKind::I32,
            schedule: Schedule::Strided,
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "under a non-Elementwise access")]
    fn coord_under_reduction_access_is_refused_by_the_emitter_backstop() {
        use crate::backend::Backend;
        use crate::ir::ReduceOp;
        use crate::plan::{KernelPlan, ReduceAxisClass, Schedule};
        use baracuda_kernel_vocab::AxisMask;
        let key = reduce_key(ElementKind::F32);
        let body = (input(0) * coord(0)).0;
        let access = crate::ir::Access::Reduction {
            op: ReduceOp::Sum,
            axes: AxisMask::EMPTY,
            keepdim: false,
            post: crate::ir::ScalarExpr::Reduced(0),
        };
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 1,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::Reduction {
                op: ReduceOp::Sum,
                class: ReduceAxisClass::InnerContig,
                keepdim: false,
            },
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &access,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "coordinate exists to read")]
    fn coord_axis_out_of_range_is_refused_by_the_emitter_backstop() {
        use crate::backend::Backend;
        use crate::plan::{KernelPlan, Schedule};
        let key = coord_key_2d(ElementKind::F32, 2);
        let body = coord(5).0; // rank is 2 — no c5 exists
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 1,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::Strided,
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "reached the scalar emitter")]
    fn coord_misrouted_to_scalar_is_refused_by_the_per_emitter_backstop() {
        use crate::backend::Backend;
        use crate::plan::{KernelPlan, Schedule};
        // A plan that passes the dtype/access/axis backstop but carries the
        // WRONG schedule (a future schedule-selection change): the scalar
        // emitter's coord closure is the layer that refuses — a linear-index
        // kernel has no c{d} to read.
        let key = coord_key_2d(ElementKind::F32, 2);
        let body = (input(0) + coord(1)).0;
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 1,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::Scalar,
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    fn coord_shared_leaf_is_free_and_shared_interiors_still_hoist() {
        // Coord is a LEAF: reusing coord(0) twice must NOT hoist a tmp for the
        // leaf itself, while a shared INTERIOR over Coords still hoists once —
        // the DAG discipline carries over to the new leaf.
        let g = coord(1) - coord(0); // shared interior
        let op = OpDef::elementwise(
            "reldist",
            1,
            &[ElementKind::F32],
            input(0) * (g.clone() * g),
        );
        let k = generate(&op, &coord_key_2d(ElementKind::F32, 2), &Cuda);
        assert_eq!(
            k.source.matches("((float)c1 - (float)c0)").count(),
            1,
            "the shared difference is emitted exactly once:\n{}",
            k.source
        );
        assert!(k.source.contains("float tmp0 = ((float)c1 - (float)c0);"));
        assert!(k.source.contains("out[oo] = (in0[o0] * (tmp0 * tmp0));"));
    }

    // ---- BASE_OFFSET SLICE: emission goldens, gate forcing, identity, and the
    // independent gate/backstop rejections (each test names the hazard it pins). ----

    /// A contiguous rank-1 unary cell (1 input + 1 output) that VECTORIZES to
    /// float4 offset-free — used to prove an offset both forces Strided and bumps
    /// the OUTPUT pointer.
    fn unary_contig_key(dt: ElementKind) -> baracuda_kernel_vocab::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    }

    #[test]
    fn offset_input_forces_strided_and_emits_suffix_arg_and_bump() {
        use crate::ir::BaseOffset;
        // add op, input 0 Runtime-offsetted, on a cell that would float4-vectorize.
        // The runtime offset invalidates the keyed alignment fact ⇒ forced Strided;
        // the emitter appends `_off0`, a `long long off0` launch arg (after so_,
        // before n), and an entry pointer bump `in0 += off0;` as the FIRST body stmt.
        let op = add_op(&[ElementKind::F32]).with_base_offsets(
            vec![BaseOffset::Runtime, BaseOffset::Zero],
            BaseOffset::Zero,
        );
        let k = generate(&op, &binary_key(ElementKind::F32), &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_add_f32_strided_r1_off0",
            "{}",
            k.source
        );
        assert!(
            k.source.contains("    long long off0,\n"),
            "off0 launch arg missing:\n{}",
            k.source
        );
        assert!(
            k.source.contains("    long long n)"),
            "n must follow the offset arg:\n{}",
            k.source
        );
        assert!(
            k.source.contains("    in0 += off0;\n"),
            "in0 entry pointer bump missing:\n{}",
            k.source
        );
        // in1 is Zero (no bump) and the output is Zero (no `out += `).
        assert!(
            !k.source.contains("in1 +="),
            "in1 is Zero — no bump:\n{}",
            k.source
        );
        assert!(
            !k.source.contains("out += "),
            "output is Zero — no bump:\n{}",
            k.source
        );
    }

    #[test]
    fn offset_output_emits_offo_suffix_arg_and_bumps_out() {
        use crate::ir::BaseOffset;
        // Output-only offset: name `_offo`, `long long offo` arg, `out += offo;`.
        let op = OpDef::elementwise("copy", 1, &[ElementKind::F32], input(0))
            .with_base_offsets(vec![BaseOffset::Zero], BaseOffset::Runtime);
        let k = generate(&op, &unary_contig_key(ElementKind::F32), &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_copy_f32_strided_r1_offo",
            "{}",
            k.source
        );
        assert!(k.source.contains("    long long offo,\n"), "{}", k.source);
        assert!(k.source.contains("    out += offo;\n"), "{}", k.source);
        assert!(
            !k.source.contains("in0 += "),
            "input is Zero — no bump:\n{}",
            k.source
        );
    }

    #[test]
    fn offset_input1_and_output_emits_off1o_in_frozen_arg_order() {
        use crate::ir::BaseOffset;
        // in1 Runtime + output Runtime — the rope ODD-lane kernel's offset shape.
        // Frozen ABI: `long long off1,` (ascending Runtime input) then `long long
        // offo,` then `long long n)`; bumps `in1 += off1;` then `out += offo;`.
        let op = add_op(&[ElementKind::F32]).with_base_offsets(
            vec![BaseOffset::Zero, BaseOffset::Runtime],
            BaseOffset::Runtime,
        );
        let k = generate(&op, &binary_key(ElementKind::F32), &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_add_f32_strided_r1_off1o",
            "{}",
            k.source
        );
        assert!(
            k.source
                .contains("    long long off1,\n    long long offo,\n    long long n)"),
            "frozen arg order off1,offo,n:\n{}",
            k.source
        );
        assert!(k.source.contains("    in1 += off1;\n"), "{}", k.source);
        assert!(k.source.contains("    out += offo;\n"), "{}", k.source);
        assert!(
            !k.source.contains("in0 += "),
            "in0 is Zero — no bump:\n{}",
            k.source
        );
    }

    #[test]
    fn all_zero_offsets_are_byte_identical_to_offset_free() {
        use crate::ir::BaseOffset;
        // G6 identity: a NON-EMPTY all-Zero base_offsets (+ Zero output) normalizes
        // to offset-free — no suffix, no args, no bumps, no forcing. The additive
        // guarantee at the SOURCE level (memcmp of emitted source).
        let key = binary_key(ElementKind::F32);
        let free = generate(&add_op(&[ElementKind::F32]), &key, &Cuda);
        let zeroed = add_op(&[ElementKind::F32])
            .with_base_offsets(vec![BaseOffset::Zero, BaseOffset::Zero], BaseOffset::Zero);
        let z = generate(&zeroed, &key, &Cuda);
        assert_eq!(free.name, z.name, "all-Zero name unchanged");
        assert_eq!(free.source, z.source, "all-Zero source byte-identical");
        assert!(!z.source.contains("off0"), "no off token:\n{}", z.source);
        assert!(!z.source.contains("offo"), "no off token:\n{}", z.source);
    }

    #[test]
    fn multi_offset_inputs_emit_ascending_frozen_abi_off02o() {
        use crate::ir::BaseOffset;
        // TWO Runtime inputs (0 and 2) + Runtime output on a ternary fma — THE
        // multi-offset ABI pin (review: previously a fully surviving-mutant
        // surface — no artifact anywhere carried two Runtime inputs, so a
        // descending reorder of `offset_operands` stayed internally consistent
        // and passed every test). One CONTIGUOUS golden pins the ascending
        // `off0, off2, offo` adjacency AND `n`-adjacency AND the bumps as the
        // first body statements in the same ascending order — a reorder, an
        // insertion between the off args, or a bump drifting below the hoist/
        // loop all break it.
        let op = OpDef::elementwise(
            "fmaoff",
            3,
            &[ElementKind::F32],
            input(0) * input(1) + input(2),
        )
        .with_base_offsets(
            vec![BaseOffset::Runtime, BaseOffset::Zero, BaseOffset::Runtime],
            BaseOffset::Runtime,
        );
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::TernaryElementwise, &[a, a, a, a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_fmaoff_f32_strided_r1_off02o",
            "{}",
            k.source
        );
        assert!(
            k.source.contains(
                "    long long off0,\n    long long off2,\n    long long offo,\n    \
                 long long n)\n{\n    in0 += off0;\n    in2 += off2;\n    out += offo;\n"
            ),
            "frozen multi-offset ABI (ascending off args adjacent to n; ascending \
             bumps as the first body stmts):\n{}",
            k.source
        );
        assert!(
            !k.source.contains("in1 +="),
            "in1 is Zero — no bump:\n{}",
            k.source
        );
    }

    #[test]
    fn offset_bump_precedes_broadcast_hoist_and_loop() {
        use crate::ir::BaseOffset;
        // THE load-bearing ordering (review: previously pinned only by the
        // manual GPU harness's bcastoff cell): a fully-broadcast (stride-0)
        // operand's hoist `h0 = in0[0]` must read element off0, so the entry
        // bump must come FIRST. One contiguous golden pins body-start → bump →
        // hoist → loop head with NOTHING in between; moving the bump after the
        // hoist (h0 reads element 0) or into the loop body (re-bumps every
        // grid-stride pass) breaks it.
        let op = add_op(&[ElementKind::F32]).with_base_offsets(
            vec![BaseOffset::Runtime, BaseOffset::Zero],
            BaseOffset::Zero,
        );
        let b = OperandDesc::new(1, &[1 << 20], &[0], ElementKind::F32, 256);
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[b, a, a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_add_f32_strided_r1_off0",
            "{}",
            k.source
        );
        assert!(
            k.source.contains(
                ")\n{\n    in0 += off0;\n    float h0 = in0[0];\n    \
                 long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n"
            ),
            "entry bump must be the FIRST body stmt, immediately before the \
             broadcast hoist, immediately before the loop head:\n{}",
            k.source
        );
        assert_eq!(
            k.source.matches("in0[").count(),
            1,
            "in0 read exactly once — via the hoist of the BUMPED base:\n{}",
            k.source
        );
    }

    #[test]
    fn offset_composes_with_gather_bump_before_index_pre_pass() {
        use crate::ir::{BaseOffset, OobPolicy};
        // Offset on the gathered DATA operand (in0): the entry bump `in0 += off0;`
        // precedes the gather pre-pass, so `o0 = gidx_clamped*s0_0` is relative to
        // the bumped base — the gather window shifts by off0 (caller precondition).
        let op = OpDef::gather(
            "gathoff",
            &[ElementKind::F32],
            0,
            OobPolicy::Clamp,
            ElementKind::I32,
        )
        .with_base_offsets(
            vec![BaseOffset::Runtime, BaseOffset::Zero],
            BaseOffset::Zero,
        );
        let k = generate(&op, &gather_2d_key(), &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_gathoff_f32_i32_strided_r2_off0",
            "{}",
            k.source
        );
        // Frozen ABI: gext BEFORE the offset args, offset args BEFORE n.
        assert!(
            k.source
                .contains("    long long gext,\n    long long off0,\n    long long n)"),
            "gext,off0,n order:\n{}",
            k.source
        );
        let bump = k.source.find("    in0 += off0;\n").expect("bump present");
        let prepass = k.source.find("gidx_off").expect("gather pre-pass present");
        assert!(
            bump < prepass,
            "bump must precede the gather pre-pass:\n{}",
            k.source
        );
    }

    #[test]
    fn offset_composes_with_scatter_bump_before_index_pre_pass() {
        use crate::ir::BaseOffset;
        // Offsets on the scatter VALUE input (in0) and the OUTPUT: bumps precede the
        // scatter pre-pass; `out[oo]` (oo = sidx_clamped*so_0) is relative to the
        // bumped out base. Assign combine (deterministic) — the FP-atomicAdd scatter
        // reroutes to gather-sum and REJECTS offsets (backstop-pinned).
        let op = OpDef::scatter("scatoff", &[ElementKind::F32], 0, ElementKind::I32)
            .with_base_offsets(
                vec![BaseOffset::Runtime, BaseOffset::Zero],
                BaseOffset::Runtime,
            );
        let k = generate(&op, &gather_2d_key(), &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_scatoff_f32_i32_strided_r2_off0o",
            "{}",
            k.source
        );
        assert!(
            k.source.contains(
                "    long long sext,\n    long long off0,\n    long long offo,\n    long long n)"
            ),
            "sext,off0,offo,n order:\n{}",
            k.source
        );
        let bump_in = k.source.find("    in0 += off0;\n").expect("in bump");
        let bump_out = k.source.find("    out += offo;\n").expect("out bump");
        let prepass = k.source.find("sidx_off").expect("scatter pre-pass");
        assert!(bump_in < prepass && bump_out < prepass, "{}", k.source);
    }

    #[test]
    fn offset_composes_with_permute_view() {
        use crate::ir::{BaseOffset, View};
        // Offset + Permute on the same input: the bump precedes the loop, and the
        // per-element offset uses the SWAPPED strides relative to the bumped base
        // (the offset is the view ORIGIN — applied before the remapped stride walk).
        let op = OpDef::elementwise("permoff", 1, &[ElementKind::F32], input(0))
            .with_views(vec![View::Permute { perm: vec![1, 0] }])
            .with_base_offsets(vec![BaseOffset::Runtime], BaseOffset::Zero);
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_permoff_f32_strided_r2_off0",
            "{}",
            k.source
        );
        assert!(k.source.contains("    in0 += off0;\n"), "{}", k.source);
        assert!(
            k.source.contains("long long o0 = c0*s0_1 + c1*s0_0;"),
            "permuted (swapped) strides after the bump:\n{}",
            k.source
        );
    }

    #[test]
    #[should_panic(expected = "FP-atomicAdd scatter")]
    fn offset_on_an_fp_atomic_scatter_is_refused_by_the_backstop() {
        use crate::ir::BaseOffset;
        // An FP `atomicAdd` scatter's BASE lowering reroutes to the deterministic
        // gather-sum (`emit_scatter_gathersum`) — a different Strided emitter that
        // never bumps the base pointer. An offset there would be silently dropped;
        // the backstop refuses it (Assign scatter + offset IS supported).
        let op = OpDef::scatter_add("scatoff_fp", &[ElementKind::F32], 0, ElementKind::I32)
            .with_base_offsets(
                vec![BaseOffset::Runtime, BaseOffset::Zero],
                BaseOffset::Zero,
            );
        let _ = generate(&op, &gather_2d_key(), &Cuda);
    }

    #[test]
    fn offset_base_variant_carries_the_oob_launch_note() {
        use crate::ir::BaseOffset;
        // The OOB story is a CALLER PRECONDITION (k/n_out trust model): the base
        // variant's launch_note must state it for an offsetted op, and stay EMPTY
        // for an offset-free op (byte-stable pre-increment behavior).
        let key = binary_key(ElementKind::F32);
        let free = crate::generate_variants(&add_op(&[ElementKind::F32]), &key, &Cuda);
        assert!(
            free[0].launch_note.is_empty(),
            "offset-free note stays empty"
        );
        let op = add_op(&[ElementKind::F32]).with_base_offsets(
            vec![BaseOffset::Runtime, BaseOffset::Zero],
            BaseOffset::Zero,
        );
        let vs = crate::generate_variants(&op, &key, &Cuda);
        assert!(
            vs[0].launch_note.contains("offset launch contract")
                && vs[0].launch_note.contains("off >= 0"),
            "offset base variant must carry the OOB precondition, got: {}",
            vs[0].launch_note
        );
    }

    // -- Tier-1: plan-gate rejections via build_plan DIRECTLY (a gate mutation must
    // turn into a should_panic failure). --

    #[test]
    #[should_panic(expected = "base_offsets.len()")]
    fn offset_arity_mismatch_is_rejected_at_the_plan_gate() {
        use crate::ir::BaseOffset;
        // G1: a non-empty base_offsets whose length != n_inputs. Built as a struct
        // literal to bypass `with_base_offsets`' debug_assert — the plan gate is the
        // release-build (author-error) backstop.
        let op = OpDef {
            base_offsets: vec![BaseOffset::Runtime],
            ..add_op(&[ElementKind::F32])
        };
        let _ = crate::build_plan(&op, &binary_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn offset_on_a_reduction_is_rejected_at_the_plan_gate() {
        use crate::ir::{BaseOffset, ReduceOp};
        // G2: a runtime offset on a non-Elementwise (Reduction) op — role-aware
        // addressing needs per-role offset semantics (de-scoped).
        let op = OpDef::reduction("rsum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum)
            .with_base_offsets(vec![BaseOffset::Runtime], BaseOffset::Zero);
        let _ = crate::build_plan(&op, &reduce_key(ElementKind::F32));
    }

    #[test]
    #[should_panic(expected = "multi-output")]
    fn offset_on_a_multi_output_op_is_rejected_at_the_plan_gate() {
        use crate::ir::BaseOffset;
        // G3: a runtime offset on a multi-output op (the multi-store DAG × the bump
        // is unproven). The op is otherwise a valid multi-output elementwise op, so
        // it passes `assert_valid_multi_output` and reaches `assert_valid_offsets`.
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let ops: Vec<_> = std::iter::repeat_n(a, 5).collect();
        let key = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
        let op = OpDef::elementwise_multi(
            "mo_off",
            3,
            &[ElementKind::F32],
            vec![input(0) * input(2), input(0) * input(1)],
        )
        .with_base_offsets(
            vec![BaseOffset::Runtime, BaseOffset::Zero, BaseOffset::Zero],
            BaseOffset::Zero,
        );
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    fn runtime_offset_forces_strided_on_a_would_be_vectorized_cell() {
        use crate::ir::BaseOffset;
        use crate::plan::{Schedule, build_plan};
        // G4 (positive): the SAME contiguous vec-width-4 cell is Vectorized{4}
        // offset-free but MUST be Strided under a runtime offset — even a
        // width-multiple offset value (the gate keys on presence, not value).
        let key = binary_key(ElementKind::F32);
        assert!(
            matches!(
                build_plan(&add_op(&[ElementKind::F32]), &key).schedule,
                Schedule::Vectorized { width: 4 }
            ),
            "sanity: offset-free vectorizes"
        );
        let op = add_op(&[ElementKind::F32]).with_base_offsets(
            vec![BaseOffset::Runtime, BaseOffset::Zero],
            BaseOffset::Zero,
        );
        assert!(
            matches!(build_plan(&op, &key).schedule, Schedule::Strided),
            "a runtime offset must force Strided"
        );
    }

    // -- Tier-2: emitter backstop, independent of the plan gate (hand-built plans:
    // a future schedule-selection change must trip HERE, not silently emit a kernel
    // that never bumps the base). --

    #[test]
    #[should_panic(expected = "must lower on the Strided schedule")]
    fn offset_on_a_non_strided_schedule_is_refused_by_the_backstop() {
        use crate::backend::Backend;
        use crate::ir::BaseOffset;
        use crate::plan::{KernelPlan, Schedule};
        let key = binary_key(ElementKind::F32);
        let body = (input(0) + input(1)).0;
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 2,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::Scalar, // WRONG: an offset needs Strided.
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[BaseOffset::Runtime, BaseOffset::Zero],
            out_base_offset: BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "single-output")]
    fn offset_on_a_multi_output_plan_is_refused_by_the_backstop() {
        use crate::backend::Backend;
        use crate::ir::BaseOffset;
        use crate::plan::{KernelPlan, Schedule};
        let key = binary_key(ElementKind::F32);
        let body = (input(0) + input(1)).0;
        let extra = [(input(0) * input(1)).0];
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 2,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::Strided,
            key: &key,
            body: &body,
            n_outputs: 2, // WRONG: an offset needs single-output.
            extra_out_bodies: &extra,
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[BaseOffset::Runtime, BaseOffset::Zero],
            out_base_offset: BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "Elementwise-only")]
    fn offset_under_a_non_elementwise_access_is_refused_by_the_backstop() {
        use crate::backend::Backend;
        use crate::ir::{BaseOffset, ReduceOp};
        use crate::plan::{KernelPlan, ReduceAxisClass, Schedule};
        use baracuda_kernel_vocab::AxisMask;
        let key = reduce_key(ElementKind::F32);
        let body = input(0).0;
        let access = crate::ir::Access::Reduction {
            op: ReduceOp::Sum,
            axes: AxisMask::EMPTY,
            keepdim: false,
            post: crate::ir::ScalarExpr::Reduced(0),
        };
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 1,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::Reduction {
                op: ReduceOp::Sum,
                class: ReduceAxisClass::InnerContig,
                keepdim: false,
            },
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &access,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[BaseOffset::Runtime], // WRONG: offset is Elementwise-only.
            out_base_offset: BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }
}

#[cfg(test)]
mod multi_output_tests {
    //! Increment 1 (MULTI_OUTPUT elementwise) goldens: one kernel writing N
    //! outputs from a shared body-DAG, with cross-body CSE (the shared `dy` load
    //! / an interior product emitted ONCE). Single-output emission stays
    //! byte-identical (extra_out_bodies empty) — pinned by
    //! `single_body_multi_matches_elementwise`.
    use crate::ir::{BinaryOp, OpDef, UnaryOp, input, konst};
    use crate::{Cuda, generate};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    /// N contiguous 1D operands. `even` picks the vectorizable extent (V4 f32 /
    /// V8 f16) vs an odd extent that forces the Scalar schedule.
    fn contig_key(dt: ElementKind, n_operands: usize, even: bool) -> StructureKey {
        let ext: i64 = if even { 1 << 20 } else { 1_000_003 };
        let a = OperandDesc::new(1, &[ext], &[1], dt, 256);
        let ops: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89)
    }

    /// N transposed (column-major) rank-2 operands: all strided, none broadcast.
    fn strided_key(dt: ElementKind, n_operands: usize) -> StructureKey {
        let t = OperandDesc::new(2, &[8, 4], &[1, 8], dt, 256);
        let ops: Vec<_> = std::iter::repeat_n(t, n_operands).collect();
        structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89)
    }

    // dy=in0, a=in1, b=in2. mul backward: da=dy*b, db=dy*a.
    fn mul_backward(dt: ElementKind) -> OpDef {
        OpDef::elementwise_multi(
            "mul_backward",
            3,
            &[dt],
            vec![input(0) * input(2), input(0) * input(1)],
        )
    }

    #[test]
    fn mul_backward_scalar_shares_the_dy_load_once() {
        // da = dy*b, db = dy*a. dy = in0 is loaded by both outputs; the shared
        // load hoists to ONE `tmp0 = in0[i]` referenced by both stores.
        let k = generate(
            &mul_backward(ElementKind::F32),
            &contig_key(ElementKind::F32, 5, false),
            &Cuda,
        );
        assert_eq!(
            k.name, "baracuda_gen_mul_backward_f32_mo2_scalar",
            "{}",
            k.source
        );
        assert_eq!(
            k.source.matches("in0[i]").count(),
            1,
            "the shared dy load appears exactly once:\n{}",
            k.source
        );
        assert!(k.source.contains("float tmp0 = in0[i];"), "{}", k.source);
        assert!(
            k.source.contains("out0[i] = (tmp0 * in2[i]);"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("out1[i] = (tmp0 * in1[i]);"),
            "{}",
            k.source
        );
        assert!(k.source.contains("float* __restrict__ out0,"));
        assert!(k.source.contains("float* __restrict__ out1,"));
    }

    #[test]
    fn div_backward_shares_the_interior_dy_over_b() {
        // da = dy/b; db = -((dy/b)*a/b). The dy/b interior is body 0's ROOT AND
        // body 1's interior, hoisted ONCE (tmp1); `b` (in2) is a shared leaf,
        // hoisted once (tmp0).
        let dyb = input(0) / input(2);
        let db = (dyb.clone() * input(1) / input(2)).unary(UnaryOp::Neg);
        let op = OpDef::elementwise_multi("div_backward", 3, &[ElementKind::F32], vec![dyb, db]);
        let k = generate(&op, &contig_key(ElementKind::F32, 5, false), &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_div_backward_f32_mo2_scalar",
            "{}",
            k.source
        );
        assert_eq!(
            k.source.matches("in0[i]").count(),
            1,
            "dy loaded once:\n{}",
            k.source
        );
        assert_eq!(
            k.source.matches("in2[i]").count(),
            1,
            "b loaded once:\n{}",
            k.source
        );
        assert!(k.source.contains("float tmp0 = in2[i];"), "{}", k.source);
        assert!(
            k.source.contains("float tmp1 = (in0[i] / tmp0);"),
            "the dy/b interior:\n{}",
            k.source
        );
        assert!(
            k.source.contains("out0[i] = tmp1;"),
            "da IS the shared interior:\n{}",
            k.source
        );
        assert!(
            k.source.contains("out1[i] = (-((tmp1 * in1[i]) / tmp0));"),
            "{}",
            k.source
        );
        assert_eq!(
            k.source.matches("tmp1").count(),
            3,
            "interior reused by both stores:\n{}",
            k.source
        );
    }

    #[test]
    fn fma_backward_three_outputs_with_a_plain_copy() {
        // Forward y = a*b + c. Backward: da = dy*b, db = dy*a, dc = dy (a plain
        // COPY reusing the hoisted shared load — dc = tmp0).
        let op = OpDef::elementwise_multi(
            "fma_backward",
            3,
            &[ElementKind::F32],
            vec![input(0) * input(2), input(0) * input(1), input(0)],
        );
        let k = generate(&op, &contig_key(ElementKind::F32, 6, false), &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_fma_backward_f32_mo3_scalar",
            "{}",
            k.source
        );
        assert_eq!(
            k.source.matches("in0[i]").count(),
            1,
            "dy loaded once:\n{}",
            k.source
        );
        assert!(k.source.contains("float tmp0 = in0[i];"), "{}", k.source);
        assert!(
            k.source.contains("out0[i] = (tmp0 * in2[i]);"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("out1[i] = (tmp0 * in1[i]);"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("out2[i] = tmp0;"),
            "dc is the shared dy copy:\n{}",
            k.source
        );
    }

    #[test]
    fn mul_backward_strided_both_stores_at_unraveled_offsets() {
        // Transposed operands → the strided emitter: each output has its own
        // per-axis stride array (so0_*, so1_*) and unraveled offset (oo0, oo1);
        // dy still loaded once.
        let k = generate(
            &mul_backward(ElementKind::F32),
            &strided_key(ElementKind::F32, 5),
            &Cuda,
        );
        assert_eq!(
            k.name, "baracuda_gen_mul_backward_f32_mo2_strided_r2",
            "{}",
            k.source
        );
        assert!(
            k.source.contains("long long oo0 = c0*so0_0 + c1*so0_1;"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("long long oo1 = c0*so1_0 + c1*so1_1;"),
            "{}",
            k.source
        );
        assert!(k.source.contains("float tmp0 = in0[o0];"), "{}", k.source);
        assert!(
            k.source.contains("out0[oo0] = (tmp0 * in2[o2]);"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("out1[oo1] = (tmp0 * in1[o1]);"),
            "{}",
            k.source
        );
        assert_eq!(k.source.matches("in0[o0]").count(), 1, "{}", k.source);
    }

    #[test]
    fn mul_backward_vectorized_float4_multi_store() {
        // Contiguous V4 f32 → the native float4 path with two output vectors;
        // each input vector loaded once, both output vectors stored.
        let k = generate(
            &mul_backward(ElementKind::F32),
            &contig_key(ElementKind::F32, 5, true),
            &Cuda,
        );
        assert_eq!(
            k.name, "baracuda_gen_mul_backward_f32_mo2_co_v4",
            "{}",
            k.source
        );
        assert!(k.source.contains("float4 v0 = in0[i];"), "{}", k.source);
        assert!(k.source.contains("vo0.x = (tmp0 * v2.x);"), "{}", k.source);
        assert!(k.source.contains("vo1.x = (tmp0 * v1.x);"), "{}", k.source);
        assert!(k.source.contains("out0[i] = vo0;"), "{}", k.source);
        assert!(k.source.contains("out1[i] = vo1;"), "{}", k.source);
    }

    #[test]
    fn mul_backward_packed_f16_both_bodies_pack() {
        // f16 V8 contiguous, all-Input-leaf bodies → the packed __half2 path with
        // two output vectors (4 pairs each).
        let k = generate(
            &mul_backward(ElementKind::F16),
            &contig_key(ElementKind::F16, 5, true),
            &Cuda,
        );
        assert_eq!(
            k.name, "baracuda_gen_mul_backward_f16_mo2_co_v8",
            "{}",
            k.source
        );
        assert!(
            k.source.contains(
                "struct __align__(16) baracuda_gen_mul_backward_f16_mo2_co_v8_vec { __half2 a, b, c, d; };"
            ),
            "{}",
            k.source
        );
        assert!(
            k.source
                .contains("baracuda_gen_mul_backward_f16_mo2_co_v8_vec v0 = in0[i];"),
            "{}",
            k.source
        );
        assert!(k.source.contains("vo0.a = "), "{}", k.source);
        assert!(k.source.contains("vo1.a = "), "{}", k.source);
        assert!(k.source.contains("out0[i] = vo0;"), "{}", k.source);
        assert!(k.source.contains("out1[i] = vo1;"), "{}", k.source);
    }

    #[test]
    fn multi_output_const_body_forces_scalar_fallback() {
        // A Const in one body disqualifies the packed pair splat → the WHOLE
        // multi-output DAG falls back to the scalar emitter even on a V8 f16 cell.
        let op = OpDef::elementwise_multi(
            "scaled_bw",
            2,
            &[ElementKind::F16],
            vec![input(0) * input(1), input(0) * konst(0.5)],
        );
        let k = generate(&op, &contig_key(ElementKind::F16, 4, true), &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_scaled_bw_f16_mo2_scalar",
            "{}",
            k.source
        );
        assert!(
            !k.source.contains("__half2"),
            "no packed pairs on the fallback:\n{}",
            k.source
        );
        assert!(k.source.contains("__half tmp0 = in0[i];"), "{}", k.source);
        assert!(
            k.source.contains("out0[i] = (tmp0 * in1[i]);"),
            "{}",
            k.source
        );
        assert!(k.source.contains("out1[i] = (tmp0 * 0.5);"), "{}", k.source);
    }

    #[test]
    fn single_body_multi_matches_elementwise() {
        // BYTE-IDENTITY: a one-body elementwise_multi is n_outputs==1 →
        // extra_out_bodies empty → the established single-output path, byte-for-
        // byte identical to OpDef::elementwise (the additive guarantee).
        let key = contig_key(ElementKind::F32, 2, true);
        let single = generate(
            &OpDef::elementwise("relu", 1, &[ElementKind::F32], input(0).relu()),
            &key,
            &Cuda,
        );
        let via_multi = generate(
            &OpDef::elementwise_multi("relu", 1, &[ElementKind::F32], vec![input(0).relu()]),
            &key,
            &Cuda,
        );
        assert_eq!(single.name, via_multi.name);
        assert_eq!(
            single.source, via_multi.source,
            "single-output byte-identical"
        );
    }

    #[test]
    fn nested_cmp_mask_multiply_composes_in_a_multi_output_body() {
        // A nested Cmp mask-multiply (relu-backward `dy*(x>0)`) composes in one
        // multi-output body and lowers per body — an inline 0/1 float, no
        // special-casing; the second output is the plain dy copy.
        let op = OpDef::elementwise_multi(
            "relu_like_bw",
            2,
            &[ElementKind::F32],
            vec![
                input(0) * input(1).binary(BinaryOp::CmpGt, konst(0.0)),
                input(0),
            ],
        );
        let k = generate(&op, &contig_key(ElementKind::F32, 4, false), &Cuda);
        assert_eq!(
            k.name, "baracuda_gen_relu_like_bw_f32_mo2_scalar",
            "{}",
            k.source
        );
        assert!(
            k.source.contains("out1[i] = tmp0;"),
            "the dy copy:\n{}",
            k.source
        );
        assert!(
            k.source
                .contains("out0[i] = (tmp0 * ((float)in1[i] > (float)0.0 ? 1.0f : 0.0f));"),
            "{}",
            k.source
        );
    }

    #[test]
    fn optimize_each_body_then_intern_preserves_cross_body_cse() {
        // E-graph order (pinned): the optimizer simplifies ONE ScalarExpr, so a
        // multi-output op is optimized per body FIRST, then interned together —
        // and the interning still collapses a subexpression shared between the
        // (independently-optimized) bodies. b0 = dy*b; b1 = (dy*b)*a share dy*b;
        // after optimize (a no-op here) from_exprs keeps them sharing one node.
        use crate::ir::ExprDag;
        use crate::optimize::optimize;
        let b0 = (input(0) * input(2)).0;
        let b1 = ((input(0) * input(2)) * input(1)).0;
        let o0 = optimize(&b0);
        let o1 = optimize(&b1);
        let dag = ExprDag::from_exprs(&[&o0, &o1]);
        // Body 0's optimized root is the shared dy*b node, referenced by body 1.
        assert!(
            dag.consumers(dag.roots()[0]) >= 1,
            "the shared subexpr survives optimize-then-intern"
        );
    }
}

#[cfg(test)]
mod dropout_hetero_tests {
    //! HETERO MULTI-OUTPUT (dropout-class) goldens + emitter backstops. One kernel
    //! writes an F32 value output AND a U8 keep-mask output from a shared body-DAG:
    //! the `rand < keep_prob` comparison hoists ONCE (cross-body CSE) and is
    //! consumed by BOTH output 0 (inside a `Select`, in compute dtype) and output 1
    //! (stored `(unsigned char)`). Per-output store conversion lives at the STORE
    //! SITE, never on the shared node (M9). On-device numeric proof is
    //! `ondevice/dropout_validate.cu`; these pin the source shape + the per-output
    //! ptr type / store cast.
    use crate::backend::Backend;
    use crate::ir::{Access, BaseOffset, BinaryOp, OpDef, WriteIndex, input, konst, param};
    use crate::plan::{KernelPlan, Schedule};
    use crate::{Cuda, build_plan, generate};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // dropout_fw: inputs x=in0, rand=in1; params keep_prob=p0, scale=p1.
    //   output 0 (value, `dt`): x·(rand<keep_prob ? scale : 0)
    //   output 1 (mask,  U8  ): rand<keep_prob   (the SAME Cmp node, shared)
    // Parametric on the value/compute dtype `dt` (F32 today, F64 the F64-param
    // increment vehicle): the U8 mask is hetero-uniform across both, and the
    // scalar params (keep_prob/scale) declare in `dt`'s scalar ctype. An F32 call
    // builds the byte-identical pre-F64 OpDef.
    fn dropout_fw(dt: ElementKind) -> OpDef {
        let cond = || input(1).binary(BinaryOp::CmpLt, param(0));
        OpDef::elementwise_multi_hetero(
            "dropout",
            2,
            &[dt],
            vec![
                (input(0) * cond().select(param(1), konst(0.0)), None),
                (cond(), Some(ElementKind::U8)),
            ],
        )
    }

    // inputs x,rand (`dt`) then outputs value (`dt`), mask (U8). `even` ⇒
    // vector-eligible ON THE VALUE OPERANDS, but the U8 mask forces Scalar/Strided
    // regardless (dropout never vectorizes — `dropout_hetero_never_vectorizes`).
    // Parametric on `dt`: an F32 call builds the byte-identical pre-F64 key.
    fn dropout_contig_key(even: bool, dt: ElementKind) -> StructureKey {
        let ext: i64 = if even { 1 << 20 } else { 1_000_003 };
        let f = OperandDesc::new(1, &[ext], &[1], dt, 256);
        let u = OperandDesc::new(1, &[ext], &[1], ElementKind::U8, 256);
        structure_key(OpCategory::BinaryElementwise, &[f, f, f, u], ArchSku::Sm89)
    }
    fn dropout_strided_key(dt: ElementKind) -> StructureKey {
        let f = OperandDesc::new(2, &[8, 4], &[1, 8], dt, 256);
        let u = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::U8, 256);
        structure_key(OpCategory::BinaryElementwise, &[f, f, f, u], ArchSku::Sm89)
    }

    #[test]
    fn dropout_scalar_hetero_golden_f32() {
        let k = generate(
            &dropout_fw(ElementKind::F32),
            &dropout_contig_key(false, ElementKind::F32),
            &Cuda,
        );
        assert_eq!(
            k.name, "baracuda_gen_dropout_f32_mo2_scalar",
            "{}",
            k.source
        );
        // Per-output pointer TYPES: value F32, mask u8 (M8).
        assert!(
            k.source.contains("float* __restrict__ out0,"),
            "value ptr f32:\n{}",
            k.source
        );
        assert!(
            k.source.contains("unsigned char* __restrict__ out1,"),
            "mask ptr u8:\n{}",
            k.source
        );
        // The shared comparison hoists ONCE (cross-body CSE) in COMPUTE dtype.
        assert_eq!(
            k.source
                .matches("float tmp0 = ((float)in1[i] < (float)p0 ? 1.0f : 0.0f);")
                .count(),
            1,
            "shared cond hoisted once:\n{}",
            k.source
        );
        // Value store: consumes the shared temp INSIDE the select (NOT the u8 cast
        // — M9); one multiply of x by the selected multiplier.
        assert!(
            k.source.contains(
                "out0[i] = (in0[i] * (((float)(tmp0)) != 0.0f ? (float)(p1) : (float)(0.0)));"
            ),
            "value store reads tmp0 raw:\n{}",
            k.source
        );
        // Mask store: the u8 conversion at the STORE SITE (M7).
        assert!(
            k.source.contains("out1[i] = (unsigned char)tmp0;"),
            "mask store (unsigned char)tmp0:\n{}",
            k.source
        );
        // Both params threaded (keep_prob p0 across both bodies, scale p1).
        assert!(
            k.source.contains(", float p0, float p1)"),
            "param signature:\n{}",
            k.source
        );
    }

    #[test]
    fn dropout_strided_hetero_golden_f32() {
        let k = generate(
            &dropout_fw(ElementKind::F32),
            &dropout_strided_key(ElementKind::F32),
            &Cuda,
        );
        assert_eq!(
            k.name, "baracuda_gen_dropout_f32_mo2_strided_r2",
            "{}",
            k.source
        );
        assert!(
            k.source.contains("unsigned char* __restrict__ out1,"),
            "mask ptr u8:\n{}",
            k.source
        );
        // Shared cond hoisted once over the strided input load `in1[o1]`.
        assert_eq!(
            k.source
                .matches("float tmp0 = ((float)in1[o1] < (float)p0 ? 1.0f : 0.0f);")
                .count(),
            1,
            "shared cond hoisted once:\n{}",
            k.source
        );
        assert!(
            k.source.contains(
                "out0[oo0] = (in0[o0] * (((float)(tmp0)) != 0.0f ? (float)(p1) : (float)(0.0)));"
            ),
            "value store at oo0 reads tmp0 raw:\n{}",
            k.source
        );
        assert!(
            k.source.contains("out1[oo1] = (unsigned char)tmp0;"),
            "mask store (unsigned char)tmp0 at oo1:\n{}",
            k.source
        );
    }

    #[test]
    fn dropout_hetero_never_vectorizes() {
        // G4: a contig V4-eligible dropout key STILL emits the `_scalar` entry (the
        // u8 mask has no packed vector store) — never a `_co_v4` kernel.
        let k = generate(
            &dropout_fw(ElementKind::F32),
            &dropout_contig_key(true, ElementKind::F32),
            &Cuda,
        );
        assert_eq!(
            k.name, "baracuda_gen_dropout_f32_mo2_scalar",
            "{}",
            k.source
        );
        assert!(!k.name.contains("_co_v"), "not vectorized: {}", k.name);
    }

    // ==================== F64-PARAM-CHANNEL (increment 12) ====================
    // The scalar-param launch channel goes dtype-aware: the param DECLARATION
    // ctype follows the SCALAR COMPUTE dtype (`scalar_ctype(plan.dtype)`) —
    // `"double"` for F64. F64 dropout is the vehicle (reuses the whole hetero-multi
    // U8-mask store; the value output is uniform F64). The #1 regression invariant
    // (byte-identical f32 param emission) is pinned by the UNCHANGED f32 goldens
    // above; these add the new f64 pins.

    #[test]
    fn dropout_scalar_hetero_golden_f64() {
        // The F64 sibling of `dropout_scalar_hetero_golden_f32`: symbol carries the
        // `f64` tag; the two scalar launch params are `double p0, double p1` (NOT
        // `float`); the value output is `double*` (uniform F64) and the mask output
        // is still `unsigned char*` (the hetero-multi U8 store is dtype-blind).
        let k = generate(
            &dropout_fw(ElementKind::F64),
            &dropout_contig_key(false, ElementKind::F64),
            &Cuda,
        );
        assert_eq!(
            k.name, "baracuda_gen_dropout_f64_mo2_scalar",
            "{}",
            k.source
        );
        // Per-output pointer TYPES: value F64 (double), mask u8 — unchanged store.
        assert!(
            k.source.contains("double* __restrict__ out0,"),
            "value ptr f64:\n{}",
            k.source
        );
        assert!(
            k.source.contains("unsigned char* __restrict__ out1,"),
            "mask ptr u8 (dtype-blind):\n{}",
            k.source
        );
        // The shared comparison hoists ONCE (cross-body CSE) in the F64 COMPUTE
        // dtype: `binary_f64`'s CmpLt has no operand casts (unlike the f32 speller,
        // whose compare casts to `(float)`), and the hoisted temp is `double`.
        assert_eq!(
            k.source
                .matches("double tmp0 = (in1[i] < p0 ? 1.0 : 0.0);")
                .count(),
            1,
            "shared cond hoisted once in double:\n{}",
            k.source
        );
        // Value store: the select arms + cond cast are `(double)`, one double
        // multiply of x by the selected multiplier (`select_f64`).
        assert!(
            k.source.contains(
                "out0[i] = (in0[i] * (((double)(tmp0)) != 0.0 ? (double)(p1) : (double)(0.0)));"
            ),
            "value store reads tmp0 raw in double:\n{}",
            k.source
        );
        // Mask store: the u8 conversion at the STORE SITE, unchanged from f32.
        assert!(
            k.source.contains("out1[i] = (unsigned char)tmp0;"),
            "mask store (unsigned char)tmp0:\n{}",
            k.source
        );
        // Both params SCALAR `double` (keep_prob p0 across both bodies, scale p1).
        assert!(
            k.source.contains(", double p0, double p1)"),
            "param signature is scalar double:\n{}",
            k.source
        );
        assert!(
            !k.source.contains("double2 p"),
            "a launch param is NEVER vectorized (no `double2 p`):\n{}",
            k.source
        );
    }

    #[test]
    fn dropout_f64_emits() {
        // G-pos-f64: an f64 param op now EMITS (no panic); the U8 mask forces the
        // scalar/strided path, so the symbol is `_f64_mo2_scalar`.
        let k = generate(
            &dropout_fw(ElementKind::F64),
            &dropout_contig_key(false, ElementKind::F64),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_dropout_f64_mo2_scalar");
        assert!(k.source.contains(", double p0, double p1)"), "{}", k.source);
    }

    #[test]
    fn mul_scalar_f64_emits() {
        // G-pos-f64: a single-output f64 `x * p0` op emits a scalar `double p0`
        // launch param — the non-dropout channel path (no panic).
        let op = OpDef::elementwise("mul_scalar", 1, &[ElementKind::F64], input(0) * param(0));
        let a = OperandDesc::new(1, &[1_000_003], &[1], ElementKind::F64, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_mul_scalar_f64_scalar", "{}", k.source);
        assert!(k.source.contains(", double p0)"), "{}", k.source);
    }

    #[test]
    fn f64_affine_scalar_param_golden() {
        // A single-output f64 `a*x + b` through the SCALAR elementwise emitter (odd
        // extent ⇒ VecWidth::Scalar): pins `, double p0, double p1)` on the
        // `param_args` (single-body) path, `double*` operands, and the double body.
        let op = OpDef::elementwise(
            "affine",
            1,
            &[ElementKind::F64],
            input(0) * param(0) + param(1),
        );
        let a = OperandDesc::new(1, &[1_000_003], &[1], ElementKind::F64, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_affine_f64_scalar", "{}", k.source);
        assert!(
            k.source.contains("const double* __restrict__ in0,"),
            "f64 operand ptr:\n{}",
            k.source
        );
        assert!(
            k.source.contains("long long n, double p0, double p1)"),
            "scalar double params:\n{}",
            k.source
        );
        assert!(
            k.source.contains("out[i] = ((in0[i] * p0) + p1);"),
            "double affine body:\n{}",
            k.source
        );
    }

    #[test]
    fn f64_affine_vectorized_param_golden() {
        // THE M2 PIN: the SAME f64 affine op forced onto the VECTORIZED path (even
        // extent ⇒ double2). Operands are `double2` but the launch params STAY
        // SCALAR `double p0, double p1` — proving the param ctype is the scalar
        // COMPUTE ctype, never `vty`. (An f32 vectorized golden — `scalar_param_kernel`
        // — pins the same for float by coincidence; this is the load-bearing f64 pin.)
        let op = OpDef::elementwise(
            "affine",
            1,
            &[ElementKind::F64],
            input(0) * param(0) + param(1),
        );
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F64, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_affine_f64_co_v2", "{}", k.source);
        // Operands vectorized to double2.
        assert!(
            k.source.contains("const double2* __restrict__ in0,"),
            "double2 operand ptr:\n{}",
            k.source
        );
        assert!(
            k.source.contains("double2 v0 = in0[i];"),
            "double2 operand load:\n{}",
            k.source
        );
        // Params SCALAR double, NOT double2 — the whole point.
        assert!(
            k.source.contains("long long nv, double p0, double p1)"),
            "scalar double params under double2 operands:\n{}",
            k.source
        );
        assert!(
            !k.source.contains("double2 p0") && !k.source.contains("double2 p1"),
            "a launch param is NEVER vectorized:\n{}",
            k.source
        );
        // Per-lane body: scalar-typed lanes, scalar params.
        assert!(
            k.source.contains("vo.x = ((v0.x * p0) + p1);"),
            "lane-x body:\n{}",
            k.source
        );
        assert!(
            k.source.contains("vo.y = ((v0.y * p0) + p1);"),
            "lane-y body:\n{}",
            k.source
        );
    }

    #[test]
    fn f64_param_cell_builds() {
        // G-build: build_plan ADMITS an f64 param cell — the plan layer is NOT the
        // gate (only INT params are plan-gated); the f32/f64-only wall is the
        // EMITTER assert. Documents that f64 rides the plan through unchanged, then
        // emits without panic.
        let op = OpDef::elementwise("mul_scalar", 1, &[ElementKind::F64], input(0) * param(0));
        let a = OperandDesc::new(1, &[1_000_003], &[1], ElementKind::F64, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        let plan = build_plan(&op, &key);
        assert_eq!(plan.dtype, ElementKind::F64);
        let k = Cuda.lower(&plan); // emits (no panic) — the plan layer passed f64 through
        assert!(k.source.contains(", double p0)"), "{}", k.source);
    }

    #[test]
    #[should_panic(expected = "f32/f64-only")]
    fn f16_param_still_rejected() {
        // G-neg-half (de-scope backstop): the assert allowlist is EXACTLY
        // {F32, F32Strict, F64} — NOT an open `is_float`. An f16 param op STILL
        // panics (its correct param ctype would be "float", a deferred numerics
        // decision with no oracle — §7).
        let op = OpDef::elementwise("mul_scalar", 1, &[ElementKind::F16], input(0) * param(0));
        let a = OperandDesc::new(1, &[1_000_003], &[1], ElementKind::F16, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        let _ = generate(&op, &key, &Cuda);
    }

    #[test]
    #[should_panic(expected = "f32/f64-only")]
    fn bf16_param_still_rejected() {
        // G-neg-half sibling: a bf16 param op STILL panics — the allowlist did not
        // over-broaden to `is_float`.
        let op = OpDef::elementwise("mul_scalar", 1, &[ElementKind::Bf16], input(0) * param(0));
        let a = OperandDesc::new(1, &[1_000_003], &[1], ElementKind::Bf16, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
        let _ = generate(&op, &key, &Cuda);
    }

    #[test]
    fn uniform_multi_output_goldens_are_byte_identical() {
        // G3: a UNIFORM multi-output op (mul_backward) is byte-identical whether the
        // hetero constructor is used with all-`None` dtypes or the plain
        // `elementwise_multi` — the additive-field no-op guarantee.
        let f = OperandDesc::new(1, &[1_000_003], &[1], ElementKind::F32, 256);
        let ops: Vec<_> = std::iter::repeat_n(f, 5).collect();
        let key = structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89);
        let via_plain = generate(
            &OpDef::elementwise_multi(
                "mul_backward",
                3,
                &[ElementKind::F32],
                vec![input(0) * input(2), input(0) * input(1)],
            ),
            &key,
            &Cuda,
        );
        let via_hetero = generate(
            &OpDef::elementwise_multi_hetero(
                "mul_backward",
                3,
                &[ElementKind::F32],
                vec![(input(0) * input(2), None), (input(0) * input(1), None)],
            ),
            &key,
            &Cuda,
        );
        assert_eq!(
            via_plain.source, via_hetero.source,
            "uniform multi byte-identical via either constructor"
        );
    }

    // ---- G5: the Tier-2 emitter backstop, independent of the plan gate ----
    // Hand-built plans handed straight to `Cuda::lower` (build_plan never produces
    // these); the backstop must hold INDEPENDENTLY (the 0a "gate every layer"
    // lesson). Kill targets for M6.

    fn hetero_plan_slots<'a>(
        key: &'a StructureKey,
        body0: &'a crate::ir::ScalarExpr,
        extra_bodies: &'a [crate::ir::ScalarExpr],
        extra_dtypes: &'a [Option<ElementKind>],
        schedule: Schedule,
    ) -> KernelPlan<'a> {
        KernelPlan {
            op_name: "backstop",
            n_inputs: 2,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule,
            key,
            body: body0,
            n_outputs: 2,
            extra_out_bodies: extra_bodies,
            extra_out_dtypes: extra_dtypes,
            access: &Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: BaseOffset::Zero,
        }
    }

    #[test]
    #[should_panic(expected = "the only lowerable per-output hetero dtype is U8")]
    fn hetero_non_u8_output_refused_by_emitter_backstop() {
        // A non-U8 hetero EXTRA output (I32) handed to `Cuda::lower` must panic in
        // `assert_multi_output_lowerable` (G5) — NOT emit C. build_plan would reject
        // it (G1); the backstop holds independently.
        let key = dropout_contig_key(false, ElementKind::F32);
        let body0 = input(0).0;
        let extra = [input(0).binary(BinaryOp::CmpLt, input(1)).0];
        let extra_dt = [Some(ElementKind::I32)];
        let plan = hetero_plan_slots(&key, &body0, &extra, &extra_dt, Schedule::Scalar);
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "has a hetero (U8) output on the Vectorized schedule")]
    fn hetero_plan_tagged_vectorized_refused_by_emitter_backstop() {
        // A LEGAL U8 hetero output but tagged Vectorized (which G4 never produces)
        // must panic in the backstop — a u8 mask has no packed vector store.
        let key = dropout_contig_key(true, ElementKind::F32);
        let body0 = input(0).0;
        let extra = [input(1).binary(BinaryOp::CmpLt, param(0)).0];
        let extra_dt = [Some(ElementKind::U8)];
        let plan = hetero_plan_slots(
            &key,
            &body0,
            &extra,
            &extra_dt,
            Schedule::Vectorized { width: 4 },
        );
        let _ = Cuda.lower(&plan);
    }

    // ---- source dump for ondevice/dropout_validate.cu ----

    /// Regenerate the generated kernels `ondevice/dropout_validate.cu` and
    /// `ondevice/dropout_f64_validate.cu` include. Run with:
    ///   `DROPOUT_OUT=<outdir> cargo test -p baracuda-kernelgen dump_dropout_sources -- --ignored --nocapture`
    #[test]
    #[ignore = "writes generated sources for the on-device dropout harness"]
    fn dump_dropout_sources() {
        let out = std::env::var("DROPOUT_OUT").expect("set DROPOUT_OUT=<outdir>");
        let write = |k: crate::GeneratedKernel| {
            let path = format!("{out}/{}.cu", k.name);
            std::fs::write(&path, &k.source).unwrap();
            println!("wrote {path}");
        };
        // F32: contiguous scalar (the acceptance-gate + bench kernel) and a strided cell.
        write(generate(
            &dropout_fw(ElementKind::F32),
            &dropout_contig_key(false, ElementKind::F32),
            &Cuda,
        ));
        write(generate(
            &dropout_fw(ElementKind::F32),
            &dropout_strided_key(ElementKind::F32),
            &Cuda,
        ));
        // F64 (the F64-param increment vehicle): the same two cells at `double`.
        // The U8 mask still forces Scalar/Strided, so these are `_f64_mo2_scalar`
        // and `_f64_mo2_strided_r2` with `double p0, double p1` launch params.
        write(generate(
            &dropout_fw(ElementKind::F64),
            &dropout_contig_key(false, ElementKind::F64),
            &Cuda,
        ));
        write(generate(
            &dropout_fw(ElementKind::F64),
            &dropout_strided_key(ElementKind::F64),
            &Cuda,
        ));
    }
}

#[cfg(test)]
mod scan_tests {
    //! Increment-6 SCAN emitter tests: the serial-fold BASE + the block-scan
    //! VARIANT generate valid, structurally-correct CUDA. On-device numeric proof
    //! is `ondevice/scan_validate.cu`; these are source-shape + variant-wiring pins.
    use crate::ir::{Access, OpDef, ReduceOp};
    use crate::plan::Schedule;
    use crate::{Cuda, build_plan, generate, generate_variants};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    fn scan_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }

    #[test]
    fn base_emits_full_width_serial_fold() {
        // Inclusive forward cumsum: thread 0 walks the axis, writes out[idx] every j.
        let sc = OpDef::scan_simple(
            "cumsum",
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
        );
        let k = generate(&sc, &scan_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_cumsum_f32_scan_sum");
        assert!(k.source.contains("if (threadIdx.x == 0)"));
        assert!(k.source.contains("for (long long j = 0; j < k; ++j)"));
        assert!(k.source.contains("acc = acc + v;"));
        assert!(k.source.contains("out[idx] ="));
        // full-width: writes every position, NOT a single reduced scalar.
        assert!(!k.source.contains("__shfl")); // base has no cooperative primitive
    }

    #[test]
    fn reverse_iterates_descending_exclusive_writes_before_combine() {
        let sc = OpDef::scan_simple("cs", &[ElementKind::F32], ReduceOp::Sum, 1, true, true);
        let k = generate(&sc, &scan_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_cs_f32_scan_sum_rev_excl");
        assert!(k.source.contains("for (long long j = k - 1; j >= 0; --j)"));
        // exclusive: prefix (the pre-combine acc) is stored BEFORE the combine.
        let pre = k.source.find("float prefix = acc;").unwrap();
        let comb = k.source.find("acc = acc + v;").unwrap();
        assert!(pre < comb, "exclusive writes before the combine");
    }

    #[test]
    fn maxmin_carry_have_flag_and_exclusive_identity() {
        let sc = OpDef::scan_simple("cmax", &[ElementKind::F32], ReduceOp::Max, 1, false, true);
        let k = generate(&sc, &scan_key(ElementKind::F32), &Cuda);
        assert!(k.source.contains("int have = 0;"));
        assert!(k.source.contains("v != v")); // NaN-propagating
        // exclusive[0] = the Max monoid identity (-inf), emitted HEADER-LIGHT via the
        // bit-cast intrinsic (NOT the <cmath> INFINITY macro the headerless-nvrtc
        // discipline forbids — the reduce/row-reduce Max/Min path follows the same).
        assert!(
            k.source
                .contains("have ? acc : (__int_as_float(0xff800000u))")
        );
        assert!(
            !k.source.contains("INFINITY"),
            "scan Max/Min must not emit the headerless-forbidden INFINITY macro:\n{}",
            k.source
        );
    }

    #[test]
    fn base_is_bit_identical_variant_index_0() {
        // generate_variants seeds base at index 0, BitIdentical.
        let sc = OpDef::scan_simple(
            "cumsum",
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
        );
        let vs = generate_variants(&sc, &scan_key(ElementKind::F32), &Cuda);
        assert_eq!(vs[0].tag, "base");
        assert_eq!(vs[0].fidelity, crate::VariantFidelity::BitIdentical);
    }

    #[test]
    fn blockscan_variant_present_for_fp_sum_prod_reassociated() {
        for op in [ReduceOp::Sum, ReduceOp::Prod] {
            let sc = OpDef::scan_simple("cum", &[ElementKind::F32], op, 1, false, false);
            let vs = generate_variants(&sc, &scan_key(ElementKind::F32), &Cuda);
            let bs = vs
                .iter()
                .find(|v| v.tag == "blockscan")
                .expect("blockscan variant");
            assert_eq!(
                bs.fidelity,
                crate::VariantFidelity::ReassociatedDeterministic,
                "FP Sum/Prod block-scan reassociates -> same_hardware_bitwise"
            );
            let src = &bs.kernels[0].source;
            assert!(src.contains("__shfl_up_sync")); // Kogge-Stone warp scan
            assert!(src.contains("warp_buf[32]")); // cross-warp carry
            assert!(src.contains("__syncthreads()"));
            assert!(bs.kernels[0].name.ends_with("_blockscan"));
        }
    }

    #[test]
    fn reverse_scan_offers_the_blockscan_variant() {
        // BLOCKSCAN-VARIANTS: reverse now OFFERS the block-scan (j = k-1-p remaps the
        // reverse j-scan to a forward p-space scan; device-validated). FP Sum/Prod
        // reverse reassociates (same as forward) -> ReassociatedDeterministic.
        for op in [ReduceOp::Sum, ReduceOp::Prod] {
            let sc = OpDef::scan_simple("cumr", &[ElementKind::F32], op, 1, true, false);
            let vs = generate_variants(&sc, &scan_key(ElementKind::F32), &Cuda);
            let bs = vs
                .iter()
                .find(|v| v.tag == "blockscan")
                .expect("reverse FP Sum/Prod now offers a blockscan variant");
            assert_eq!(
                bs.fidelity,
                crate::VariantFidelity::ReassociatedDeterministic,
                "reverse FP Sum/Prod block-scan reassociates like forward"
            );
            assert!(bs.kernels[0].source.contains("long long j = k - 1 - p;"));
            assert!(bs.kernels[0].name.ends_with("_rev_blockscan"));
        }
    }

    #[test]
    fn blockscan_variant_offers_maxmin_and_integer_bit_identical() {
        // BLOCKSCAN-VARIANTS: FP Max/Min AND integer {I32,I64} Sum/Prod/Max/Min now
        // OFFER a BitIdentical block-scan (combiner selects the same element /
        // modular +/* is associative — device memcmp==0). S8/U8 STILL decline.
        for (op, dt) in [
            (ReduceOp::Max, ElementKind::F32),
            (ReduceOp::Min, ElementKind::F32),
            (ReduceOp::Max, ElementKind::F64),
            (ReduceOp::Min, ElementKind::F16),
            (ReduceOp::Sum, ElementKind::I32),
            (ReduceOp::Prod, ElementKind::I32),
            (ReduceOp::Max, ElementKind::I32),
            (ReduceOp::Min, ElementKind::I64),
            (ReduceOp::Sum, ElementKind::I64),
        ] {
            let sc = OpDef::scan_simple("cum", &[dt], op, 1, false, false);
            let vs = generate_variants(&sc, &scan_key(dt), &Cuda);
            let bs = vs
                .iter()
                .find(|v| v.tag == "blockscan")
                .unwrap_or_else(|| panic!("block-scan must offer {op:?}/{dt:?}"));
            assert_eq!(
                bs.fidelity,
                crate::VariantFidelity::BitIdentical,
                "FP Max/Min + integer block-scan is bit-identical to the base ({op:?}/{dt:?})"
            );
        }
    }

    #[test]
    fn blockscan_variant_declines_s8_u8_to_base() {
        // Honest-scope pin (retained): S8/U8 (any op) ride the serial base — the warp
        // shuffle has no 8-bit overload and promoting breaks the native 8-bit wrap.
        // The gate uses an explicit FP+I32/I64 allowlist, NOT is_int_dtype (which
        // admits S8/U8); a mutation to is_int_dtype must fail this.
        for (op, dt) in [
            (ReduceOp::Sum, ElementKind::S8),
            (ReduceOp::Max, ElementKind::S8),
            (ReduceOp::Sum, ElementKind::U8),
            (ReduceOp::Min, ElementKind::U8),
            (ReduceOp::Prod, ElementKind::U8),
        ] {
            let sc = OpDef::scan_simple("cum", &[dt], op, 1, false, false);
            let vs = generate_variants(&sc, &scan_key(dt), &Cuda);
            assert!(
                vs.iter().all(|v| v.tag != "blockscan"),
                "block-scan must decline {op:?}/{dt:?} (S8/U8) to the base"
            );
            assert_eq!(vs[0].tag, "base");
        }
    }

    #[test]
    fn blockscan_fidelity_discriminates_reassoc_from_bit_identical() {
        // The fidelity discriminator: ONLY FP Sum/Prod reassociates; FP Max/Min and
        // ALL integer scans are BitIdentical.
        let blockscan_fidelity = |op: ReduceOp, dt: ElementKind| {
            let sc = OpDef::scan_simple("cum", &[dt], op, 1, false, false);
            generate_variants(&sc, &scan_key(dt), &Cuda)
                .into_iter()
                .find(|v| v.tag == "blockscan")
                .map(|v| v.fidelity)
        };
        use crate::VariantFidelity::{BitIdentical, ReassociatedDeterministic};
        // FP Sum/Prod -> reassociated (the ONLY reassociating case).
        assert_eq!(
            blockscan_fidelity(ReduceOp::Sum, ElementKind::F32),
            Some(ReassociatedDeterministic)
        );
        assert_eq!(
            blockscan_fidelity(ReduceOp::Prod, ElementKind::F64),
            Some(ReassociatedDeterministic)
        );
        // FP Max/Min -> BitIdentical.
        assert_eq!(
            blockscan_fidelity(ReduceOp::Max, ElementKind::F32),
            Some(BitIdentical)
        );
        assert_eq!(
            blockscan_fidelity(ReduceOp::Min, ElementKind::Bf16),
            Some(BitIdentical)
        );
        // integer Sum/Prod/Max/Min -> BitIdentical.
        assert_eq!(
            blockscan_fidelity(ReduceOp::Sum, ElementKind::I32),
            Some(BitIdentical)
        );
        assert_eq!(
            blockscan_fidelity(ReduceOp::Max, ElementKind::I64),
            Some(BitIdentical)
        );
    }

    #[test]
    fn blockscan_maxmin_emission_goldens() {
        let blockscan_src = |tag: &str, op: ReduceOp, dt: ElementKind, exc: bool| {
            let sc = OpDef::scan_simple(tag, &[dt], op, 1, false, exc);
            generate_variants(&sc, &scan_key(dt), &Cuda)
                .into_iter()
                .find(|v| v.tag == "blockscan")
                .expect("blockscan variant")
                .kernels
                .remove(0)
                .source
        };
        // F32 forward cummax: the FP Max warp step + the -inf identity via the
        // headerless bit-cast intrinsic; NEVER fmaxf/fminf.
        let cmax = blockscan_src("cummax", ReduceOp::Max, ElementKind::F32, false);
        assert!(cmax.contains("(v != v || v > t) ? v : t"));
        assert!(cmax.contains("__int_as_float(0xff800000u)"));
        assert!(!cmax.contains("fmaxf") && !cmax.contains("fminf"));
        // F32 cummin: the Min warp step (< not >) + the +inf identity.
        let cmin = blockscan_src("cummin", ReduceOp::Min, ElementKind::F32, false);
        assert!(cmin.contains("(v != v || v < t) ? v : t"));
        assert!(cmin.contains("__int_as_float(0x7f800000u)"));
        assert!(!cmin.contains("v > t"));
        // I32 cumsum: native int acc + `v += t` (no float acc anywhere).
        let isum = blockscan_src("cumsum", ReduceOp::Sum, ElementKind::I32, false);
        assert!(isum.contains("int warp_buf[32]"));
        assert!(isum.contains("v += t;"));
        assert!(!isum.contains("float"));
        // I32 cummax: the INT reduced form (no dead `v != v` NaN test in the int
        // kernel), native int acc.
        let imax = blockscan_src("cummax", ReduceOp::Max, ElementKind::I32, false);
        assert!(imax.contains("v = (v > t) ? v : t;"));
        assert!(!imax.contains("v != v"));
        // exclusive cummax: position-0 emits the Max identity (-inf) via `wexc`.
        let cmax_x = blockscan_src("cummax", ReduceOp::Max, ElementKind::F32, true);
        assert!(cmax_x.contains("if (lane == 0) wexc = __int_as_float(0xff800000u);"));
    }

    #[test]
    fn blockscan_sum_emission_stays_reassoc_free_of_maxmin_arms() {
        // Byte-identity anchor (companion to the present-test): the Sum block-scan's
        // warp step + carry combiner are the commutative `+` forms, and NONE of the
        // Max/Min ternary machinery (a `? v : t` step or a `!= ` NaN test) leaks into
        // a Sum kernel — adding Max/Min arms must not perturb the Sum arm.
        let sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        let src = generate_variants(&sc, &scan_key(ElementKind::F32), &Cuda)
            .into_iter()
            .find(|v| v.tag == "blockscan")
            .expect("blockscan variant")
            .kernels
            .remove(0)
            .source;
        assert!(src.contains("if (lane >= off) v += t;"));
        assert!(src.contains("carry = (carry + chunk_tot);"));
        assert!(
            !src.contains("? v : t"),
            "no Max/Min warp step in a Sum kernel"
        );
        assert!(!src.contains("!= "), "no NaN test in a Sum kernel");
    }

    #[test]
    fn integer_scan_uses_native_accumulator() {
        // I32 cumsum accumulates in the native ctype (wrapping), not a float acc.
        let sc = OpDef::scan_simple("cumi", &[ElementKind::I32], ReduceOp::Sum, 1, false, false);
        let k = generate(&sc, &scan_key(ElementKind::I32), &Cuda);
        assert!(k.source.contains("int acc = 0;"));
        assert!(!k.source.contains("float acc"));
    }

    #[test]
    fn base_schedule_is_block_false() {
        let sc = OpDef::scan_simple("cum", &[ElementKind::F32], ReduceOp::Sum, 1, false, false);
        let key = scan_key(ElementKind::F32);
        let plan = build_plan(&sc, &key);
        assert!(matches!(plan.schedule, Schedule::Scan { block: false, .. }));
        assert!(matches!(plan.access, Access::Scan { .. }));
    }

    /// Manual dump tool (not a wired assertion): regenerate the GENERATED relu
    /// `.cu` oracle sources the on-device validator `ondevice/relu_propagating_validate.cu`
    /// `#include`s (the semantics oracle — `UnaryOp::Relu`). Emits, per dtype
    /// {f32,f64,f16,bf16}: a SCALAR cell (align = dtype size ⇒ no vectorization ⇒
    /// the clean per-element `(const T*, T*, long long)` oracle) and a CONTIG cell
    /// (256-align ⇒ the vectorized path Fuel would adopt). Run with:
    ///   `RELU_OUT=<outdir> cargo test -p baracuda-kernelgen dump_relu_sources -- --ignored --nocapture`
    /// then copy `ondevice/relu_propagating_validate.cu` beside the emitted files
    /// and `nvcc` it (this replaces the old scratchpad `dump_relu` bin).
    #[test]
    #[ignore = "manual regeneration tool for ondevice/relu_propagating_validate.cu"]
    fn dump_relu_sources() {
        use crate::ir::input;
        use baracuda_kernel_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};
        let out = std::env::var("RELU_OUT").unwrap_or_else(|_| ".".to_string());
        for dt in [
            ElementKind::F32,
            ElementKind::F64,
            ElementKind::F16,
            ElementKind::Bf16,
        ] {
            let relu = OpDef::elementwise("relu", 1, &[dt], input(0).relu());
            let esz = match dt {
                ElementKind::F16 | ElementKind::Bf16 => 2,
                ElementKind::F32 => 4,
                ElementKind::F64 => 8,
                _ => unreachable!(),
            };
            // Scalar cell (align = element size ⇒ no vectorization).
            for align in [esz, 256] {
                let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
                let o = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
                let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
                let k = generate(&relu, &key, &Cuda);
                std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
                println!("wrote {out}/{}.cu", k.name);
            }
        }
    }

    /// Manual dump tool (not a wired assertion): regenerate the GENERATED offset
    /// `.cu` sources the on-device validator `ondevice/offset_validate.cu`
    /// `#include`s (BASE_OFFSET SLICE). Run with:
    ///   `OFFSET_OUT=<outdir> cargo test -p baracuda-kernelgen dump_offset_sources -- --ignored --nocapture`
    /// then copy `ondevice/offset_validate.cu` beside the emitted files and `nvcc` it.
    #[test]
    #[ignore = "manual regeneration tool for ondevice/offset_validate.cu"]
    fn dump_offset_sources() {
        use crate::ir::{BaseOffset, input};
        use baracuda_kernel_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};
        let out = std::env::var("OFFSET_OUT").unwrap_or_else(|_| ".".to_string());
        let write = |k: crate::GeneratedKernel| {
            std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
            println!("wrote {out}/{}.cu", k.name);
        };
        // --- addoff: out[i] = in0[off0 + i] + in1[i] (rank-1, offset forces Strided
        // even on a contiguous cell — the CPU-oracle + offset-0-identity + high-edge
        // memcheck kernel). f32 and f64. ---
        for dt in [ElementKind::F32, ElementKind::F64] {
            let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 256);
            let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
            // With offset on input 0.
            let off = OpDef::elementwise("addoff", 2, &[dt], input(0) + input(1))
                .with_base_offsets(
                    vec![BaseOffset::Runtime, BaseOffset::Zero],
                    BaseOffset::Zero,
                );
            write(generate(&off, &key, &Cuda));
            // No-offset sibling on the SAME strided addressing (built via a strided
            // cell so it is also Strided) — the offset-0 identity baseline.
            let t = OperandDesc::new(2, &[8, 4], &[1, 8], dt, 256);
            let skey = structure_key(OpCategory::BinaryElementwise, &[t, t, t], ArchSku::Sm89);
            let base = OpDef::elementwise("addoff", 2, &[dt], input(0) + input(1));
            write(generate(&base, &skey, &Cuda));
            let soff = OpDef::elementwise("addoff", 2, &[dt], input(0) + input(1))
                .with_base_offsets(
                    vec![BaseOffset::Runtime, BaseOffset::Zero],
                    BaseOffset::Zero,
                );
            write(generate(&soff, &skey, &Cuda));
        }
        // --- add3off (case 11): out[offo+i] = in0[off0+i] + in1[i] + in2[off2+i] —
        // MULTIPLE Runtime inputs (0 and 2) + a Runtime output, the `_off02o`
        // multi-offset frozen-ABI cell (ascending off args; three bumps). Chained
        // ADDs (no mul) so the CPU oracle is bit-deterministic — FADD chains are
        // never FMA-contracted, unlike a*b+c. ---
        {
            let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
            let key = structure_key(OpCategory::TernaryElementwise, &[a, a, a, a], ArchSku::Sm89);
            let op = OpDef::elementwise(
                "add3off",
                3,
                &[ElementKind::F32],
                input(0) + input(1) + input(2),
            )
            .with_base_offsets(
                vec![BaseOffset::Runtime, BaseOffset::Zero, BaseOffset::Runtime],
                BaseOffset::Runtime,
            );
            write(generate(&op, &key, &Cuda));
        }
        // --- bcastoff: out[i] = in0[off0] + in1[i], in0 FULLY BROADCAST (all strides
        // 0) + Runtime offset — proves the entry bump lands on the hoisted `in0[0]`
        // (reading element off0, not 0). ---
        {
            let b = OperandDesc::new(1, &[1 << 20], &[0], ElementKind::F32, 256); // fully broadcast
            let n = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
            let key = structure_key(OpCategory::BinaryElementwise, &[b, n, n], ArchSku::Sm89);
            let op = OpDef::elementwise("bcastoff", 2, &[ElementKind::F32], input(0) + input(1))
                .with_base_offsets(
                    vec![BaseOffset::Runtime, BaseOffset::Zero],
                    BaseOffset::Zero,
                );
            write(generate(&op, &key, &Cuda));
        }
        // --- permoff (case 4): out[r,c] = x[off0 + <transposed read>] — offset +
        // Permute view on the same input (the offset is the view ORIGIN, applied
        // before the swapped-stride walk). ---
        {
            use crate::ir::View;
            let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
            let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
            let op = OpDef::elementwise("permoff", 1, &[ElementKind::F32], input(0))
                .with_views(vec![View::Permute { perm: vec![1, 0] }])
                .with_base_offsets(vec![BaseOffset::Runtime], BaseOffset::Zero);
            write(generate(&op, &key, &Cuda));
        }
        // --- gathoff / gathoffi (case 5): gather (Clamp) with the offset on the
        // gathered DATA operand (off0) and, separately, on the INDEX operand (off1)
        // — the dual-pointer independence proof. rank-1, i32 index. ---
        {
            use crate::ir::OobPolicy;
            let d = OperandDesc::new(1, &[1 << 12], &[1], ElementKind::F32, 256);
            let ix = OperandDesc::new(1, &[1 << 12], &[1], ElementKind::I32, 256);
            let key = structure_key(OpCategory::BinaryElementwise, &[d, ix, d], ArchSku::Sm89);
            let goff = OpDef::gather(
                "gathoff",
                &[ElementKind::F32],
                0,
                OobPolicy::Clamp,
                ElementKind::I32,
            )
            .with_base_offsets(
                vec![BaseOffset::Runtime, BaseOffset::Zero],
                BaseOffset::Zero,
            );
            write(generate(&goff, &key, &Cuda));
            let gioff = OpDef::gather(
                "gathoffi",
                &[ElementKind::F32],
                0,
                OobPolicy::Clamp,
                ElementKind::I32,
            )
            .with_base_offsets(
                vec![BaseOffset::Zero, BaseOffset::Runtime],
                BaseOffset::Zero,
            );
            write(generate(&gioff, &key, &Cuda));
        }
        // --- scatoff (case 6): Assign scatter with offsets on the VALUE input
        // (off0) and the OUTPUT (offo). rank-1, i32 index, Skip OOB. ---
        {
            let u = OperandDesc::new(1, &[1 << 12], &[1], ElementKind::F32, 256);
            let ix = OperandDesc::new(1, &[1 << 12], &[1], ElementKind::I32, 256);
            let key = structure_key(OpCategory::BinaryElementwise, &[u, ix, u], ArchSku::Sm89);
            let op = OpDef::scatter("scatoff", &[ElementKind::F32], 0, ElementKind::I32)
                .with_base_offsets(
                    vec![BaseOffset::Runtime, BaseOffset::Zero],
                    BaseOffset::Runtime,
                );
            write(generate(&op, &key, &Cuda));
        }
    }

    /// Manual dump tool (not a wired assertion): regenerate the GENERATED rope
    /// pair-decomposition `.cu` sources the on-device validator `#include`s for the
    /// BASE_OFFSET SLICE acceptance gate (rope E2E vs `launch_rope_apply_fp<float>`).
    /// Two single-output launches over the pair axis `[BH,S,P]` (P=D/2), stride-2
    /// innermost; the odd-lane cross-read is the `+1` view origin (a Runtime offset
    /// on `x_o`), the odd output is the stride-2 write at origin `+1`. Run with:
    ///   `OFFSET_OUT=<outdir> cargo test -p baracuda-kernelgen dump_rope_pair_sources -- --ignored --nocapture`
    #[test]
    #[ignore = "manual regeneration tool for ondevice/offset_validate.cu (rope E2E)"]
    fn dump_rope_pair_sources() {
        use crate::ir::{BaseOffset, input};
        use baracuda_kernel_vocab::{ArchSku, OpCategory, OperandDesc, structure_key};
        let out = std::env::var("OFFSET_OUT").unwrap_or_else(|_| ".".to_string());
        let write = |k: crate::GeneratedKernel| {
            std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
            println!("wrote {out}/{}.cu", k.name);
        };
        // Rank-3 [BH,S,P] cell (placeholder extents — the KEY carries only rank/
        // contiguity/broadcast; the harness passes the true shape/strides at launch).
        // x_e/x_o: stride-2 innermost (strided); cos/sin: broadcast axis 0 (shared
        // table, stride_b=0); out: stride-2 innermost.
        let dt = ElementKind::F32;
        // BH=2, S=4, D=16 => P=8, S*D=64.
        let xstr = OperandDesc::new(3, &[2, 4, 8], &[64, 16, 2], dt, 256);
        let cs = OperandDesc::new(3, &[2, 4, 8], &[0, 8, 1], dt, 256); // broadcast axis 0
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[xstr, xstr, cs, cs, xstr],
            ArchSku::Sm89,
        );
        // even: y_e = x_e*cos - x_o*sin = Sub(Mul(in0,in2), Mul(in1,in3)); x_o (in1)
        // is read at the pair-partner +1 (Runtime offset), output at the even origin.
        let even = OpDef::elementwise(
            "rope_even",
            4,
            &[dt],
            input(0) * input(2) - input(1) * input(3),
        )
        .with_base_offsets(
            vec![
                BaseOffset::Zero,
                BaseOffset::Runtime,
                BaseOffset::Zero,
                BaseOffset::Zero,
            ],
            BaseOffset::Zero,
        );
        write(generate(&even, &key, &Cuda));
        // The no-offset even sibling (all-Zero) — the offset-0 identity baseline
        // (case 1): launching the `_off1` kernel with off1=0 must byte-match this.
        let even_base = OpDef::elementwise(
            "rope_even",
            4,
            &[dt],
            input(0) * input(2) - input(1) * input(3),
        );
        write(generate(&even_base, &key, &Cuda));
        // odd: y_o = x_o*cos + x_e*sin, AUTHORED as Add(Mul(in0,in3), Mul(in1,in2))
        // = `in0[o0]*in3[o3] + in1[o1]*in2[o2]` — commutatively identical, but the
        // operand ORDER pins nvcc's FMA contraction to the bespoke's: the PTX audit
        // (bespoke_inst.ptx vs gen_odd.ptx, 2026-07-08) showed nvcc front-end
        // contracts the FIRST product of an Add in this TU while the bespoke
        // `x_o*c + x_e*si` TU contracted the SECOND (fma(si,x_e, round(c*x_o))) —
        // authoring x_e*si first makes both sides fma(x_e,si)+round(x_o*c) ⇒
        // bit-exact odd lanes (even lanes contract identically via ptxas either
        // way). x_o (in1) reads at +1 (Runtime), output writes at the odd origin
        // +1 (Runtime).
        let odd = OpDef::elementwise(
            "rope_odd",
            4,
            &[dt],
            input(0) * input(3) + input(1) * input(2),
        )
        .with_base_offsets(
            vec![
                BaseOffset::Zero,
                BaseOffset::Runtime,
                BaseOffset::Zero,
                BaseOffset::Zero,
            ],
            BaseOffset::Runtime,
        );
        write(generate(&odd, &key, &Cuda));
    }

    /// Manual dump tool (not a wired assertion): regenerate the scan `.cu` sources
    /// the on-device validator `#include`s. Run with:
    ///   `cargo test -p baracuda-kernelgen dump_scan_sources -- --ignored --nocapture`
    /// then copy `ondevice/scan_validate.cu` beside the emitted files and `nvcc` it.
    #[test]
    #[ignore = "manual regeneration tool for ondevice/scan_validate.cu"]
    fn dump_scan_sources() {
        let out = std::env::var("SCAN_OUT").unwrap_or_else(|_| ".".to_string());
        let write = |k: crate::GeneratedKernel| {
            std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
            println!("wrote {out}/{}.cu", k.name);
        };
        // Emit EVERY variant (serial base at [0] + the block-scan variant where the
        // gate offers it) for a (tag, op, dtype, reverse, exclusive) cell — so the
        // BitIdentical base-vs-blockscan memcmp always has both kernels present.
        let emit_all = |tag: &str, op: ReduceOp, dt: ElementKind, rev: bool, exc: bool| {
            let sc = OpDef::scan_simple(tag, &[dt], op, 1, rev, exc);
            for v in generate_variants(&sc, &scan_key(dt), &Cuda) {
                for kern in v.kernels {
                    write(kern);
                }
            }
        };
        // f32: 4 combines × fwd/rev × incl/excl. Base cells for the serial matrix,
        // plus the block-scan variants now offered for ALL of them (Sum/Prod
        // reassociated, Max/Min BitIdentical, forward AND reverse).
        for (op, tag) in [
            (ReduceOp::Sum, "cumsum"),
            (ReduceOp::Prod, "cumprod"),
            (ReduceOp::Max, "cummax"),
            (ReduceOp::Min, "cummin"),
        ] {
            for reverse in [false, true] {
                for exclusive in [false, true] {
                    emit_all(tag, op, ElementKind::F32, reverse, exclusive);
                }
            }
        }
        // FP Max/Min over the remaining dtypes (base + BitIdentical block-scan):
        // f16, bf16, f32-strict, f64 — the full FP breadth for the memcmp matrix.
        for dt in [
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::F32Strict,
            ElementKind::F64,
        ] {
            for (op, tag) in [(ReduceOp::Max, "cummax"), (ReduceOp::Min, "cummin")] {
                for reverse in [false, true] {
                    for exclusive in [false, true] {
                        emit_all(tag, op, dt, reverse, exclusive);
                    }
                }
            }
        }
        // Integer Sum/Prod/Max/Min over I32/I64 (base + BitIdentical block-scan).
        for dt in [ElementKind::I32, ElementKind::I64] {
            for (op, tag) in [
                (ReduceOp::Sum, "cumsum"),
                (ReduceOp::Prod, "cumprod"),
                (ReduceOp::Max, "cummax"),
                (ReduceOp::Min, "cummin"),
            ] {
                for reverse in [false, true] {
                    for exclusive in [false, true] {
                        emit_all(tag, op, dt, reverse, exclusive);
                    }
                }
            }
        }
        // f64 serial base (Sum) for the double-precision bit-exact case.
        emit_all("cumsum", ReduceOp::Sum, ElementKind::F64, false, false);
    }
}

#[cfg(test)]
mod window_tests {
    //! Increment-7 WINDOW emitter tests: the one-thread-per-output pooling base
    //! generates valid, structurally-correct CUDA. On-device numeric proof is
    //! `ondevice/window_validate.cu`; these are source-shape + geometry pins.
    use crate::ir::{Access, OpDef, ReduceOp};
    use crate::plan::Schedule;
    use crate::{Cuda, build_plan, generate};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // input [8, k_in] contiguous, output [8, k_out] contiguous (downsampled).
    fn window_key(dt: ElementKind, k_in: i64, k_out: i64) -> StructureKey {
        let a = OperandDesc::new(2, &[8, k_in], &[k_in, 1], dt, 256);
        let o = OperandDesc::new(2, &[8, k_out], &[k_out, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }

    #[test]
    fn schedule_and_access_are_window() {
        let p = OpDef::window_simple(
            "pool",
            &[ElementKind::F32],
            ReduceOp::Max,
            1,
            3,
            2,
            1,
            1,
            1,
            false,
        );
        let key = window_key(ElementKind::F32, 128, 64);
        let plan = build_plan(&p, &key);
        assert!(matches!(plan.schedule, Schedule::Window { .. }));
        assert!(matches!(plan.access, Access::Window { .. }));
    }

    #[test]
    fn max_pool_emits_grid_stride_have_flag_and_nan_probe() {
        let p = OpDef::window_simple(
            "pool",
            &[ElementKind::F32],
            ReduceOp::Max,
            1,
            3,
            2,
            1,
            1,
            1,
            false,
        );
        let k = generate(&p, &window_key(ElementKind::F32, 128, 64), &Cuda);
        assert_eq!(k.name, "baracuda_gen_pool_f32_window_max");
        // one thread per OUTPUT element, grid-stride over n_out*k_out.
        assert!(
            k.source.contains("long long total = n_out * k_out;"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("long long o = t - row * k_out;"),
            "{}",
            k.source
        );
        // tap position with stride/pad/dilation baked as literals (size 3, stride 2, pad 1, dil 1).
        assert!(
            k.source
                .contains("long long p = o * 2 - 1 + (long long)kk * 1;"),
            "{}",
            k.source
        );
        assert!(k.source.contains("if (p >= 0 && p < k_in)"), "{}", k.source);
        // have-flag + NaN-propagate probe.
        assert!(
            k.source.contains("if (!have || v != v || v > best)"),
            "{}",
            k.source
        );
    }

    #[test]
    fn maxmin_pool_identity_is_header_light_no_infinity() {
        // The Max/Min identity must be the bit-cast ±inf intrinsic, NOT the
        // <cmath> INFINITY macro (the headerless-nvrtc discipline the scan/reduce
        // paths follow).
        for (op, ident) in [
            (ReduceOp::Max, "__int_as_float(0xff800000u)"),
            (ReduceOp::Min, "__int_as_float(0x7f800000u)"),
        ] {
            let p = OpDef::window_simple("pool", &[ElementKind::F32], op, 1, 2, 2, 1, 0, 0, false);
            let k = generate(&p, &window_key(ElementKind::F32, 128, 64), &Cuda);
            assert!(k.source.contains(ident), "{}", k.source);
            assert!(
                !k.source.contains("INFINITY"),
                "no INFINITY macro:\n{}",
                k.source
            );
        }
    }

    #[test]
    fn min_pool_uses_less_than_compare() {
        let p = OpDef::window_simple(
            "pool",
            &[ElementKind::F32],
            ReduceOp::Min,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
        );
        let k = generate(&p, &window_key(ElementKind::F32, 128, 64), &Cuda);
        assert_eq!(k.name, "baracuda_gen_pool_f32_window_min");
        assert!(k.source.contains("v != v || v < best"), "{}", k.source);
    }

    #[test]
    fn avg_pool_exclude_pad_divides_by_valid_count() {
        // count_include_pad=false ⇒ divide by the valid-tap count, guarded on cnt>0.
        let p = OpDef::window_simple(
            "pool",
            &[ElementKind::F32],
            ReduceOp::Mean,
            1,
            3,
            1,
            1,
            1,
            1,
            false,
        );
        let k = generate(&p, &window_key(ElementKind::F32, 128, 128), &Cuda);
        assert_eq!(k.name, "baracuda_gen_pool_f32_window_mean");
        assert!(k.source.contains("acc = acc + v;"), "{}", k.source);
        assert!(k.source.contains("cnt += 1;"), "{}", k.source);
        assert!(
            k.source
                .contains("(cnt > 0) ? (acc / (float)cnt) : (float)0;"),
            "{}",
            k.source
        );
    }

    #[test]
    fn avg_pool_include_pad_divides_by_size_literal() {
        // count_include_pad=true ⇒ divide by the window size literal; the entry
        // point is suffixed `_cip` so the two divisor policies never collide.
        let p = OpDef::window_simple(
            "pool",
            &[ElementKind::F32],
            ReduceOp::Mean,
            1,
            4,
            1,
            1,
            1,
            1,
            true,
        );
        let k = generate(&p, &window_key(ElementKind::F32, 128, 128), &Cuda);
        assert_eq!(k.name, "baracuda_gen_pool_f32_window_mean_cip");
        assert!(
            k.source.contains("float prefix = acc / (float)4;"),
            "{}",
            k.source
        );
    }

    #[test]
    fn sum_pool_has_no_divide() {
        let p = OpDef::window_simple(
            "pool",
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            3,
            1,
            1,
            0,
            0,
            false,
        );
        let k = generate(&p, &window_key(ElementKind::F32, 128, 126), &Cuda);
        assert_eq!(k.name, "baracuda_gen_pool_f32_window_sum");
        assert!(k.source.contains("float prefix = acc;"), "{}", k.source);
        assert!(
            !k.source.contains("/ (float)"),
            "sum-pool never divides:\n{}",
            k.source
        );
    }

    #[test]
    fn f16_pool_upconverts_and_stores_half() {
        let p = OpDef::window_simple(
            "pool",
            &[ElementKind::F16],
            ReduceOp::Mean,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
        );
        let k = generate(&p, &window_key(ElementKind::F16, 128, 64), &Cuda);
        assert!(k.source.contains("#include <cuda_fp16.h>"), "{}", k.source);
        assert!(
            k.source.contains("float v = __half2float(in0[idx]);"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("out[t] = __float2half(prefix);"),
            "{}",
            k.source
        );
    }

    #[test]
    fn integer_max_pool_uses_native_accumulator_no_infinity() {
        // I32 max-pool selects in the native ctype; the identity is the exact int
        // minimum literal (never a float INFINITY).
        let p = OpDef::window_simple(
            "pool",
            &[ElementKind::I32],
            ReduceOp::Max,
            1,
            2,
            2,
            1,
            0,
            0,
            false,
        );
        let k = generate(&p, &window_key(ElementKind::I32, 128, 64), &Cuda);
        assert!(
            k.source.contains("int best = (-2147483647 - 1);"),
            "{}",
            k.source
        );
        assert!(!k.source.contains("INFINITY"), "{}", k.source);
        assert!(
            !k.source.contains("float"),
            "no float accumulator for int pool:\n{}",
            k.source
        );
    }

    /// Manual dump tool (not a wired assertion): regenerate the window `.cu`
    /// sources the on-device validator `#include`s. Run with:
    ///   `cargo test -p baracuda-kernelgen dump_window_sources -- --ignored --nocapture`
    #[test]
    #[ignore = "manual regeneration tool for ondevice/window_validate.cu"]
    fn dump_window_sources() {
        let out = std::env::var("WINDOW_OUT").unwrap_or_else(|_| ".".to_string());
        let write = |k: crate::GeneratedKernel| {
            std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
            println!("wrote {out}/{}.cu", k.name);
        };
        // Geometry-encoded op_names so distinct (combine, geometry) cells emit
        // distinct entry symbols (the kernel name encodes op/dtype/combine but NOT
        // geometry — two geometries of one combine would otherwise collide). The
        // harness `window_validate.cu` mirrors this matrix. Fields:
        //   (op_name, dtype, combine, size, stride, dilation, pad_lo, pad_hi, cip)
        let cases: &[(&str, ElementKind, ReduceOp, u8, u8, u8, u8, u8, bool)] = &[
            // f32 — stride>1 no pad; stride1 + pad (NaN); dilated + pad both ends.
            (
                "mx_a",
                ElementKind::F32,
                ReduceOp::Max,
                2,
                2,
                1,
                0,
                0,
                false,
            ),
            (
                "mx_b",
                ElementKind::F32,
                ReduceOp::Max,
                3,
                1,
                1,
                1,
                1,
                false,
            ),
            (
                "mx_c",
                ElementKind::F32,
                ReduceOp::Max,
                3,
                2,
                2,
                2,
                2,
                false,
            ),
            (
                "mn_b",
                ElementKind::F32,
                ReduceOp::Min,
                3,
                1,
                1,
                1,
                1,
                false,
            ),
            (
                "sm_d",
                ElementKind::F32,
                ReduceOp::Sum,
                3,
                2,
                1,
                0,
                0,
                false,
            ),
            (
                "av_b",
                ElementKind::F32,
                ReduceOp::Mean,
                3,
                1,
                1,
                1,
                1,
                false,
            ),
            (
                "ai_b",
                ElementKind::F32,
                ReduceOp::Mean,
                3,
                1,
                1,
                1,
                1,
                true,
            ),
            (
                "av_c",
                ElementKind::F32,
                ReduceOp::Mean,
                3,
                2,
                2,
                2,
                2,
                false,
            ),
            // f64 oracle-exact avg + max (dilated + padded).
            (
                "av_c64",
                ElementKind::F64,
                ReduceOp::Mean,
                3,
                2,
                2,
                2,
                2,
                false,
            ),
            (
                "mx_c64",
                ElementKind::F64,
                ReduceOp::Max,
                3,
                2,
                2,
                2,
                2,
                false,
            ),
        ];
        for &(name, dt, op, size, stride, dil, plo, phi, cip) in cases {
            let p = OpDef::window_simple(name, &[dt], op, 1, size, stride, dil, plo, phi, cip);
            write(generate(&p, &window_key(dt, 64, 32), &Cuda));
        }
    }
}

#[cfg(test)]
mod im2col_tests {
    //! Increment-11 IM2COL emitter tests: the one-thread-per-output-cell expanding
    //! gather generates valid, structurally-correct CUDA. On-device numeric proof
    //! (whole-buffer memcmp vs the bespoke `im2col_2d_fw_kernel` + a CPU closed-form
    //! oracle) is `ondevice/im2col_validate.cu`; these are source-shape + closed-form
    //! index-map + raw-bit-copy pins. Goldens call `generate` (the full pipeline);
    //! gate mutations are covered `build_plan`-DIRECT in `plan::im2col_gate_validate`.
    use crate::plan::{KernelPlan, Schedule};
    use crate::{Cuda, generate};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    // rank-4 NCHW input [2,3,4,4] dense + rank-3 output [2,8,8] dense (the extent is
    // a runtime precondition — only rank + layout are keyed).
    fn im2col_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], dt, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }

    #[test]
    fn im2col_base_golden_f32() {
        // 3x3, stride 1, pad 1, dilation 1 — the canonical conv geometry.
        let p =
            crate::ir::OpDef::im2col_2d("unfold", ElementKind::F32, (3, 3), (1, 1), (1, 1), (1, 1));
        let k = generate(&p, &im2col_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_unfold_f32_im2col");
        let s = &k.source;
        // The 6-scalar signature in EXACT order N,C,H_in,W_in,oH,oW (a swapped
        // H/W or oH/oW would be silently wrong on non-square inputs).
        assert!(s.contains("const float* __restrict__ in0,"), "{s}");
        assert!(s.contains("float* __restrict__ out,"), "{s}");
        assert!(
            s.contains(
                "    long long N,\n    long long C,\n    long long H_in,\n    long long W_in,\n    long long oH,\n    long long oW)"
            ),
            "{s}"
        );
        // total = N*C*kh*kw*oH*oW (the full kh*kw expansion; kh/kw baked as literals).
        assert!(
            s.contains("long long total = N * C * 3 * 3 * oH * oW;"),
            "{s}"
        );
        // The 6-way unravel, Layout A: spatial col = oh*oW+ow, row = c*kh*kw+ki*kw+kj.
        assert!(s.contains("long long col = t % (oH * oW);"), "{s}");
        assert!(s.contains("long long row_full = t / (oH * oW);"), "{s}");
        assert!(s.contains("long long oh = col / oW;"), "{s}");
        assert!(s.contains("long long ow = col - oh * oW;"), "{s}");
        assert!(
            s.contains("long long patch = row_full % (C * 3 * 3);"),
            "{s}"
        );
        assert!(s.contains("long long n = row_full / (C * 3 * 3);"), "{s}");
        assert!(s.contains("long long c = patch / (3 * 3);"), "{s}");
        assert!(s.contains("long long kij = patch - c * (3 * 3);"), "{s}");
        assert!(s.contains("long long ki = kij / 3;"), "{s}");
        assert!(s.contains("long long kj = kij - ki * 3;"), "{s}");
        // Source coord with stride/pad/dilation baked (s1 p1 d1): oh*1 - 1 + ki*1.
        assert!(s.contains("long long in_h = oh * 1 - 1 + ki * 1;"), "{s}");
        assert!(s.contains("long long in_w = ow * 1 - 1 + kj * 1;"), "{s}");
        // BOTH spatial guards (a dropped in_w guard OOB-reads at the W-edges).
        assert!(
            s.contains("if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {"),
            "{s}"
        );
        // Raw-bit copy of the closed-form NCHW source offset.
        assert!(
            s.contains("out[t] = in0[(((n * C + c) * H_in + in_h) * W_in + in_w)];"),
            "{s}"
        );
        // OOB store is the exact typed zero (zero-pad).
        assert!(s.contains("out[t] = 0.0f;"), "{s}");
    }

    #[test]
    fn im2col_golden_f16() {
        // f16: the copied value is a RAW-BIT __half move (NO __float2half round-trip on
        // the copy — that would change NaN payloads); only the OOB zero is constructed.
        let p =
            crate::ir::OpDef::im2col_2d("unfold", ElementKind::F16, (2, 2), (1, 1), (0, 0), (1, 1));
        let k = generate(&p, &im2col_key(ElementKind::F16), &Cuda);
        assert_eq!(k.name, "baracuda_gen_unfold_f16_im2col");
        let s = &k.source;
        assert!(s.contains("#include <cuda_fp16.h>"), "{s}");
        assert!(s.contains("const __half* __restrict__ in0,"), "{s}");
        // Raw __half copy — NO conversion wrapper on the gathered value.
        assert!(
            s.contains("out[t] = in0[(((n * C + c) * H_in + in_h) * W_in + in_w)];"),
            "{s}"
        );
        assert!(
            !s.contains("__half2float(in0"),
            "the gather must NOT round-trip f16 through float:\n{s}"
        );
        // The OOB zero is the exact typed __float2half(0.0f) — matches bespoke zero_of.
        assert!(s.contains("out[t] = __float2half(0.0f);"), "{s}");
    }

    #[test]
    fn im2col_bf16_oob_zero_is_typed() {
        let p = crate::ir::OpDef::im2col_2d(
            "unfold",
            ElementKind::Bf16,
            (2, 2),
            (1, 1),
            (1, 1),
            (1, 1),
        );
        let k = generate(&p, &im2col_key(ElementKind::Bf16), &Cuda);
        let s = &k.source;
        assert!(s.contains("out[t] = __float2bfloat16(0.0f);"), "{s}");
        assert!(!s.contains("__bfloat162float(in0"), "{s}");
    }

    #[test]
    fn im2col_dilated_golden() {
        // dilation != 1 bakes + ki*dh / + kj*dw; non-square kernel (kh != kw) + stride
        // + pad all baked as distinct literals (a ki/kj or H/W swap fails the memcmp
        // on the on-device non-square cells; here it fails the baked literals).
        let p =
            crate::ir::OpDef::im2col_2d("unfold", ElementKind::F32, (3, 5), (2, 1), (2, 0), (2, 3));
        let k = generate(&p, &im2col_key(ElementKind::F32), &Cuda);
        let s = &k.source;
        assert!(
            s.contains("long long total = N * C * 3 * 5 * oH * oW;"),
            "{s}"
        );
        assert!(
            s.contains("long long patch = row_full % (C * 3 * 5);"),
            "{s}"
        );
        assert!(s.contains("long long c = patch / (3 * 5);"), "{s}");
        assert!(s.contains("long long ki = kij / 5;"), "{s}");
        assert!(s.contains("long long kj = kij - ki * 5;"), "{s}");
        // in_h uses (sh=2, ph=2, dh=2); in_w uses (sw=1, pw=0, dw=3) — distinct axes.
        assert!(s.contains("long long in_h = oh * 2 - 2 + ki * 2;"), "{s}");
        assert!(s.contains("long long in_w = ow * 1 - 0 + kj * 3;"), "{s}");
    }

    #[test]
    fn im2col_symbol_is_distinct() {
        // The _im2col symbol collides with no other family (window/sort/scan).
        let p =
            crate::ir::OpDef::im2col_2d("conv", ElementKind::F32, (3, 3), (1, 1), (1, 1), (1, 1));
        let k = generate(&p, &im2col_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_conv_f32_im2col");
        assert!(!k.source.contains("_window_"), "{}", k.source);
        assert!(!k.source.contains("_rowsort_"), "{}", k.source);
        assert!(!k.source.contains("_scan_"), "{}", k.source);
    }

    #[test]
    #[should_panic(expected = "Im2Col output must be forward-dense contiguous")]
    fn im2col_flipped_output_refused_by_backstop() {
        // Tier-2: a hand-built Im2Col plan with a FLIPPED output (build_plan never
        // produces this — validate_im2col rejects it) must be refused independently by
        // the emitter backstop.
        use crate::backend::Backend;
        let a = OperandDesc::new(4, &[2, 3, 4, 4], &[48, 16, 4, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(3, &[2, 8, 8], &[64, 8, -1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89);
        let body = crate::ir::input(0).0;
        let plan = KernelPlan {
            op_name: "sneaky",
            n_inputs: 1,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::Im2Col {
                kernel: (3, 3),
                stride: (1, 1),
                pad: (1, 1),
                dilation: (1, 1),
            },
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Im2Col {
                kernel: (3, 3),
                stride: (1, 1),
                pad: (1, 1),
                dilation: (1, 1),
            },
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    /// Manual dump tool (not a wired assertion): regenerate the im2col `.cu` sources
    /// the on-device validator `#include`s. Run with:
    ///   `IM2COL_OUT=<outdir> cargo test -p baracuda-kernelgen dump_im2col_sources -- --ignored --nocapture`
    #[test]
    #[ignore = "manual regeneration tool for ondevice/im2col_validate.cu"]
    fn dump_im2col_sources() {
        let out = std::env::var("IM2COL_OUT").unwrap_or_else(|_| ".".to_string());
        let write = |k: crate::GeneratedKernel| {
            std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
            println!("wrote {out}/{}.cu", k.name);
        };
        // Geometry-encoded op_names so distinct (dtype, geometry) cells emit distinct
        // entry symbols (the kernel name encodes op/dtype but NOT geometry — two
        // geometries would otherwise collide). The harness `im2col_validate.cu`
        // mirrors this matrix. Fields: (op_name, dtype, (kh,kw),(sh,sw),(ph,pw),(dh,dw)).
        type Geo = ((u8, u8), (u8, u8), (u8, u8), (u8, u8));
        // The geometry spread (shared across dtypes): 1x1 identity, 3x3 s1 p1 (OOB
        // every edge), kh!=kw non-square, stride>kernel holes, dilation>1.
        let geos: &[(&str, Geo)] = &[
            ("k1", ((1, 1), (1, 1), (0, 0), (1, 1))), // 1x1 identity-ish
            ("k3p1", ((3, 3), (1, 1), (1, 1), (1, 1))), // 3x3 s1 p1 (OOB edges)
            ("k3s2", ((3, 3), (2, 2), (1, 1), (1, 1))), // 3x3 stride 2 pad 1
            ("k35", ((3, 5), (1, 1), (1, 2), (1, 1))), // kh!=kw non-square
            ("s3k2", ((2, 2), (3, 3), (0, 0), (1, 1))), // stride>kernel (holes)
            ("d2", ((3, 3), (1, 1), (2, 2), (2, 2))), // dilation>1
        ];
        for &dt in &[
            ElementKind::F32,
            ElementKind::F64,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::I32,
            ElementKind::I64,
        ] {
            // The symbol is `baracuda_gen_{gname}_{dtag}_im2col` — dtag distinguishes
            // dtypes, gname distinguishes geometries, so each (dtype, geometry) cell is
            // a unique entry point (no redefinition when all are #included together).
            for &(gname, (kernel, stride, pad, dilation)) in geos {
                let p = crate::ir::OpDef::im2col_2d(gname, dt, kernel, stride, pad, dilation);
                write(generate(&p, &im2col_key(dt), &Cuda));
            }
        }
    }
}

#[cfg(test)]
mod sort_tests {
    //! Increment-8 SORT_PERM emitter tests: the per-output RANK-sort BASE + the
    //! cooperative smem BITONIC pair-sort VARIANT generate valid, structurally-
    //! correct CUDA. On-device numeric proof is `ondevice/sort_validate.cu`; these
    //! are source-shape + variant-wiring + no-INFINITY pins.
    use crate::ir::{OpDef, SortOrder};
    use crate::{Cuda, generate, generate_variants};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    fn sort_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }
    fn argsort_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::I32, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }
    // Increment 9 FUSED_ARGSORT: the 3-operand `Both` key — input, values output
    // (same dtype), I32 index output.
    fn both_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let ov = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let oi = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::I32, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, ov, oi], ArchSku::Sm89)
    }
    // Increment 10 TOPK/BOTTOMK: the 3-operand `Both` key with NARROWER
    // `[256,k_out]` outputs (k_out < k_in = 128) — the shrunk out extent.
    fn topk_both_key(dt: ElementKind, k_out: i64) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let ov = OperandDesc::new(2, &[256, k_out], &[k_out, 1], dt, 256);
        let oi = OperandDesc::new(2, &[256, k_out], &[k_out, 1], ElementKind::I32, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, ov, oi], ArchSku::Sm89)
    }
    // A single-output (Values) key with a narrower output — the free TopK+Values path.
    fn topk_values_key(dt: ElementKind, k_out: i64) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(2, &[256, k_out], &[k_out, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }
    // The PARTIAL-SELECT `_psel` variant's kernel source (it is a VARIANT, so it
    // comes off generate_variants — the default `generate` lowering is the base).
    fn psel_src(op: &OpDef, key: &StructureKey) -> String {
        generate_variants(op, key, &Cuda)
            .into_iter()
            .find(|v| v.tag == "psel")
            .expect("psel variant present")
            .kernels
            .swap_remove(0)
            .source
    }

    #[test]
    fn base_asc_values_signature_and_rank_loop() {
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let k = generate(&sc, &sort_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_sort_f32_rowsort_asc_stable");
        // Signature: single input, values output (float), n_out + k launch args.
        assert!(
            k.source.contains("const float* __restrict__ in0,"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("float* __restrict__ out,"),
            "{}",
            k.source
        );
        assert!(k.source.contains("long long n_out,"), "{}", k.source);
        assert!(k.source.contains("long long k)"), "{}", k.source);
        // Rank sort: one thread per output element, scans its row, writes the raw value.
        assert!(
            k.source.contains("long long total = n_out * k;"),
            "{}",
            k.source
        );
        assert!(
            k.source
                .contains("if (sort_f32_asc_pair_lt(kj, j, ki, i)) r++;"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("out[base + r] = in0[base + i];"),
            "{}",
            k.source
        );
        // Review-caught: the tie index (also the LOAD address) must be `long long`,
        // never `int` — an int index OOB-reads past 2^31 on the any-k base.
        assert!(k.source.contains("long long i = t - base;"), "{}", k.source);
        assert!(
            k.source
                .contains("pair_lt(float ka, long long ia, float kb, long long ib)"),
            "{}",
            k.source
        );
        assert!(
            !k.source.contains("int i = (int)(t - base);"),
            "{}",
            k.source
        );
        // Base has no cooperative primitive.
        assert!(!k.source.contains("__syncthreads"), "{}", k.source);
        assert!(!k.source.contains("__shared__"), "{}", k.source);
        // asc comparator argument order (ka,kb first).
        assert!(
            k.source
                .contains("if (sort_f32_asc_key_lt(ka, kb)) return true;"),
            "{}",
            k.source
        );
        // FP dtype: the NaN-greatest branch is present.
        assert!(
            k.source.contains("if (a != a) return false;"),
            "{}",
            k.source
        );
    }

    #[test]
    fn base_desc_argsort_hetero_out_and_comparator_order() {
        // The hetero-out backstop extension test (§4.1): argsort stores through an
        // `int* out`, and the desc comparator reverses the key argument order.
        let sc = OpDef::row_argsort("argsort", ElementKind::F32, SortOrder::Desc);
        let k = generate(&sc, &argsort_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_argsort_f32_rowsort_desc_stable_idx");
        assert!(k.source.contains("int* __restrict__ out,"), "{}", k.source);
        // argsort writes the original index, not the value — narrowed to the I32
        // output (k <= 2^31-1 is an inherent argsort precondition).
        assert!(k.source.contains("out[base + r] = (int)i;"), "{}", k.source);
        // desc comparator: key_lt(kb, ka) first (reversed key order).
        assert!(
            k.source
                .contains("if (argsort_f32_desc_idx_key_lt(kb, ka)) return true;"),
            "{}",
            k.source
        );
    }

    #[test]
    fn integer_sort_has_no_nan_branch_and_native_key() {
        // i32: the comparator is a bare `a < b` (no NaN branch), key type = int.
        let sc = OpDef::row_sort("sort", ElementKind::I32, SortOrder::Asc);
        let k = generate(&sc, &sort_key(ElementKind::I32), &Cuda);
        assert!(
            k.source.contains("bool sort_i32_asc_key_lt(int a, int b)"),
            "{}",
            k.source
        );
        assert!(
            !k.source.contains("if (a != a)"),
            "int sort has no NaN branch:\n{}",
            k.source
        );
    }

    #[test]
    fn bitonic_asc_smem_pad_and_barriers() {
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let vs = generate_variants(&sc, &sort_key(ElementKind::F32), &Cuda);
        let bt = vs
            .iter()
            .find(|v| v.tag == "bitonic")
            .expect("bitonic variant");
        let src = &bt.kernels[0].source;
        assert!(
            bt.kernels[0].name.ends_with("_bitonic"),
            "{}",
            bt.kernels[0].name
        );
        // dynamic smem as (key, index) pairs, staged uchar-typed.
        assert!(
            src.contains("extern __shared__ unsigned char baracuda_sort_smem[];"),
            "{}",
            src
        );
        assert!(
            src.contains("long long pow2 = 1; while (pow2 < k) pow2 <<= 1;"),
            "{}",
            src
        );
        // asc-FP pad sentinel = qNaN (NOT +inf, NOT the INFINITY macro).
        assert!(
            src.contains("skey[p] = __int_as_float(0x7fc00000u);"),
            "{}",
            src
        );
        // bitonic network with the standard swap predicate.
        assert!(src.contains("bool up = ((p & kk) == 0);"), "{}", src);
        assert!(src.contains("if (up == q_lt_p) {"), "{}", src);
        // at least the stage barrier + one per-phase barrier + the writeback barrier.
        assert!(src.matches("__syncthreads();").count() >= 3, "{}", src);
        // launch-note contract: k <= 1024 + the smem byte formula.
        assert!(bt.launch_note.contains("k <= 1024"), "{}", bt.launch_note);
        assert!(
            bt.launch_note.contains("next_pow2(k) * 8 bytes"),
            "{}",
            bt.launch_note
        ); // float(4)+int(4)
        assert_eq!(bt.fidelity, crate::VariantFidelity::BitIdentical);
    }

    #[test]
    fn bitonic_f64_desc_and_i64_asc_pad_literals() {
        // f64 desc pad = -inf (double bit-cast); i64 asc pad = INT64_MAX literal.
        let scd = OpDef::row_sort("sort", ElementKind::F64, SortOrder::Desc);
        let vd = generate_variants(&scd, &sort_key(ElementKind::F64), &Cuda);
        let btd = vd.iter().find(|v| v.tag == "bitonic").unwrap();
        assert!(
            btd.kernels[0]
                .source
                .contains("skey[p] = __longlong_as_double(0xfff0000000000000ULL);"),
            "{}",
            btd.kernels[0].source
        );
        assert!(
            btd.launch_note.contains("next_pow2(k) * 12 bytes"),
            "{}",
            btd.launch_note
        ); // double(8)+int(4)

        let sci = OpDef::row_sort("sort", ElementKind::I64, SortOrder::Asc);
        let vi = generate_variants(&sci, &sort_key(ElementKind::I64), &Cuda);
        let bti = vi.iter().find(|v| v.tag == "bitonic").unwrap();
        assert!(
            bti.kernels[0]
                .source
                .contains("skey[p] = 9223372036854775807LL;"),
            "{}",
            bti.kernels[0].source
        );
    }

    #[test]
    fn no_infinity_macro_in_any_sort_source() {
        // The headerless-nvrtc discipline forbids the <cmath> INFINITY macro; every
        // sort cell (base + bitonic, every dtype/order/argsort) must be header-light.
        for dt in [
            ElementKind::F32,
            ElementKind::F64,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::F32Strict,
            ElementKind::I32,
            ElementKind::I64,
        ] {
            for order in [SortOrder::Asc, SortOrder::Desc] {
                // Values, Indices, AND the fused Both — every cell must stay header-light.
                let cases = [
                    (OpDef::row_sort("sort", dt, order), sort_key(dt)),
                    (OpDef::row_argsort("argsort", dt, order), argsort_key(dt)),
                    (OpDef::row_sort_indices("fused", dt, order), both_key(dt)),
                ];
                for (sc, key) in &cases {
                    for v in generate_variants(sc, key, &Cuda) {
                        for kern in &v.kernels {
                            assert!(
                                !kern.source.contains("INFINITY"),
                                "sort cell {} must not emit INFINITY:\n{}",
                                kern.name,
                                kern.source
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn variants_are_exactly_base_then_bitonic() {
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let vs = generate_variants(&sc, &sort_key(ElementKind::F32), &Cuda);
        let tags: Vec<&str> = vs.iter().map(|v| v.tag).collect();
        assert_eq!(tags, vec!["base", "bitonic"]);
        assert_eq!(vs[0].fidelity, crate::VariantFidelity::BitIdentical);
    }

    // ---- Increment 9 FUSED_ARGSORT: the two-output `Both` goldens ----

    #[test]
    fn sort_both_base_golden_f32() {
        // The fused base kernel: a two-pointer signature (values float, I32 index),
        // BOTH stores driven by ONE shared pair_lt rank scan (the fusion payoff —
        // one sort, two outputs, not two full sorts).
        let sc = OpDef::row_sort_indices("fused", ElementKind::F32, SortOrder::Asc);
        let k = generate(&sc, &both_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_fused_f32_rowsort_asc_stable_both");
        // Two-pointer signature: values (input ctype) then I32 indices.
        assert!(
            k.source.contains("float* __restrict__ out_val,"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("int* __restrict__ out_idx,"),
            "{}",
            k.source
        );
        // NO single `out` pointer (byte-distinct from the Values/Indices signature).
        assert!(
            !k.source.contains("__restrict__ out,"),
            "Both must not emit the single-output `out` pointer:\n{}",
            k.source
        );
        // Both stores at the base site (M5/M9 pins: index store present; correct targets).
        assert!(
            k.source.contains("out_val[base + r] = in0[base + i];"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("out_idx[base + r] = (int)i;"),
            "{}",
            k.source
        );
        // ONE shared rank scan + ONE rank accumulator (proves single sort, not two).
        assert_eq!(
            k.source
                .matches("if (fused_f32_asc_both_pair_lt(kj, j, ki, i)) r++;")
                .count(),
            1,
            "{}",
            k.source
        );
        assert_eq!(
            k.source.matches("long long r = 0;").count(),
            1,
            "{}",
            k.source
        );
        // out_idx is `int*` (M8 pin) — NOT float*; there is no `float* ... out_idx`.
        assert!(
            !k.source.contains("float* __restrict__ out_idx"),
            "{}",
            k.source
        );
    }

    #[test]
    fn sort_both_bitonic_golden_f32() {
        // The fused smem bitonic variant: two-pointer signature, both writeback
        // stores gather through the final permutation, same staging/network as #8.
        let sc = OpDef::row_sort_indices("fused", ElementKind::F32, SortOrder::Asc);
        let vs = generate_variants(&sc, &both_key(ElementKind::F32), &Cuda);
        let bt = vs
            .iter()
            .find(|v| v.tag == "bitonic")
            .expect("bitonic variant");
        let src = &bt.kernels[0].source;
        assert_eq!(
            bt.kernels[0].name,
            "baracuda_gen_fused_f32_rowsort_asc_stable_both_bitonic"
        );
        assert!(src.contains("float* __restrict__ out_val,"), "{}", src);
        assert!(src.contains("int* __restrict__ out_idx,"), "{}", src);
        // Both writeback stores (M6/M9 pins): value gathers raw bits; index writes sidx.
        assert!(
            src.contains("out_val[base + p] = in0[base + sidx[p]];"),
            "{}",
            src
        );
        assert!(src.contains("out_idx[base + p] = sidx[p];"), "{}", src);
        // Same smem staging as #8 (unchanged comparator/network/pad).
        assert!(
            src.contains("extern __shared__ unsigned char baracuda_sort_smem[];"),
            "{}",
            src
        );
        assert_eq!(bt.fidelity, crate::VariantFidelity::BitIdentical);
    }

    #[test]
    fn sort_both_symbol_is_distinct() {
        // The fused `_both` entry-point AND its `__device__` comparator stem collide
        // with neither the Values ("") nor the Indices ("_idx") symbols — critical
        // for the validator TU that #includes all three (M7: reverting the suffix to
        // "" makes the entry name AND the stem duplicate-link).
        let both = generate(
            &OpDef::row_sort_indices("fused", ElementKind::F32, SortOrder::Asc),
            &both_key(ElementKind::F32),
            &Cuda,
        );
        let val = generate(
            &OpDef::row_sort("fused", ElementKind::F32, SortOrder::Asc),
            &sort_key(ElementKind::F32),
            &Cuda,
        );
        let idx = generate(
            &OpDef::row_argsort("fused", ElementKind::F32, SortOrder::Asc),
            &argsort_key(ElementKind::F32),
            &Cuda,
        );
        assert_eq!(both.name, "baracuda_gen_fused_f32_rowsort_asc_stable_both");
        assert_ne!(both.name, val.name);
        assert_ne!(both.name, idx.name);
        assert_ne!(val.name, idx.name);
        // The comparator stems are distinct too (no ODR clash on include).
        assert!(
            both.source.contains("fused_f32_asc_both_pair_lt"),
            "{}",
            both.source
        );
        assert!(
            val.source.contains("fused_f32_asc_pair_lt"),
            "{}",
            val.source
        );
        assert!(
            idx.source.contains("fused_f32_asc_idx_pair_lt"),
            "{}",
            idx.source
        );
        // The Both stem is NOT a substring collision of the Values stem.
        assert!(
            !both.source.contains("fused_f32_asc_pair_lt"),
            "{}",
            both.source
        );
    }

    #[test]
    fn sort_values_golden_byte_identical() {
        // Regression (M11): the Values sort emits today's exact source — the single
        // `out` pointer, the raw-value store — and NEVER leaks out_val/out_idx/_both.
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let k = generate(&sc, &sort_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_sort_f32_rowsort_asc_stable");
        assert!(
            k.source.contains("float* __restrict__ out,"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("out[base + r] = in0[base + i];"),
            "{}",
            k.source
        );
        assert!(!k.source.contains("out_val"), "{}", k.source);
        assert!(!k.source.contains("out_idx"), "{}", k.source);
        assert!(!k.source.contains("_both"), "{}", k.source);
    }

    #[test]
    fn argsort_indices_golden_byte_identical() {
        // Regression (M11): the Indices sort emits today's exact source — the single
        // `int* out` pointer, the index store — and NEVER leaks out_val/out_idx/_both.
        let sc = OpDef::row_argsort("argsort", ElementKind::F32, SortOrder::Asc);
        let k = generate(&sc, &argsort_key(ElementKind::F32), &Cuda);
        assert_eq!(k.name, "baracuda_gen_argsort_f32_rowsort_asc_stable_idx");
        assert!(k.source.contains("int* __restrict__ out,"), "{}", k.source);
        assert!(k.source.contains("out[base + r] = (int)i;"), "{}", k.source);
        assert!(!k.source.contains("out_val"), "{}", k.source);
        assert!(!k.source.contains("out_idx"), "{}", k.source);
        assert!(!k.source.contains("_both"), "{}", k.source);
    }

    // ================= Increment 10 TOPK/BOTTOMK: runtime-k cap ==================

    #[test]
    fn topk_both_base_golden_f32() {
        // The capped fused base kernel: the 3-scalar `(n_out, k_in, k_out)` ABI, the
        // split bases, ONE shared pair_lt rank scan over `j < k_in`, and the store
        // GUARDED by `if (r < k_out)`. Pins M2 (guard), M3 (grid = n_out*k_in), M4
        // (out_base = row*k_out), M9 (store targets), M10 (out_idx is int*).
        let sc = OpDef::row_topk("topk", ElementKind::F32);
        let k = generate(&sc, &topk_both_key(ElementKind::F32, 64), &Cuda);
        assert_eq!(
            k.name,
            "baracuda_gen_topk_f32_rowsort_desc_stable_both_topk"
        );
        // 3-scalar signature + k_out==0 early-out (the Window ABI precedent).
        assert!(k.source.contains("long long n_out,"), "{}", k.source);
        assert!(k.source.contains("long long k_in,"), "{}", k.source);
        assert!(k.source.contains("long long k_out)"), "{}", k.source);
        assert!(k.source.contains("if (k_out == 0) return;"), "{}", k.source);
        // Full's single-`k` ABI is ABSENT (M1/M12: no byte-identity bleed the wrong way).
        assert!(!k.source.contains("long long k)"), "{}", k.source);
        // Grid over ALL inputs (M3), split bases (M4).
        assert!(
            k.source.contains("long long total = n_out * k_in;"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("long long in_base = row * k_in;"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("long long out_base = row * k_out;"),
            "{}",
            k.source
        );
        // ONE shared rank scan over k_in (M3: not k_out), ONE accumulator.
        assert_eq!(
            k.source
                .matches("if (topk_f32_desc_both_topk_pair_lt(kj, j, ki, i)) r++;")
                .count(),
            1,
            "{}",
            k.source
        );
        assert!(
            k.source.contains("for (long long j = 0; j < k_in; ++j) {"),
            "{}",
            k.source
        );
        // The store GUARD (M2) + both capped stores (M4/M9).
        assert!(k.source.contains("if (r < k_out) {"), "{}", k.source);
        assert!(
            k.source
                .contains("out_val[out_base + r] = in0[in_base + i];"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("out_idx[out_base + r] = (int)i;"),
            "{}",
            k.source
        );
        // Two-pointer signature; out_idx is int* (M10), never float*.
        assert!(
            k.source.contains("float* __restrict__ out_val,"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("int* __restrict__ out_idx,"),
            "{}",
            k.source
        );
        assert!(!k.source.contains("__restrict__ out,"), "{}", k.source);
        assert!(
            !k.source.contains("float* __restrict__ out_idx"),
            "{}",
            k.source
        );
        // Desc comparator argument order (kb,ka first) — topk = largest first (M8).
        assert!(
            k.source
                .contains("if (topk_f32_desc_both_topk_key_lt(kb, ka)) return true;"),
            "{}",
            k.source
        );
    }

    #[test]
    fn topk_both_bitonic_golden_f32() {
        // The capped fused bitonic variant: stage the FULL row (`pow2 < k_in`), sort,
        // then SHORTEN the writeback to `p < k_out` into `out*[out_base + p]`. Pins M5
        // (writeback bound), M6 (stage bound k_in), same network as the Both variant.
        let sc = OpDef::row_topk("topk", ElementKind::F32);
        let vs = generate_variants(&sc, &topk_both_key(ElementKind::F32, 64), &Cuda);
        let bt = vs
            .iter()
            .find(|v| v.tag == "bitonic")
            .expect("bitonic variant");
        let src = &bt.kernels[0].source;
        assert_eq!(
            bt.kernels[0].name,
            "baracuda_gen_topk_f32_rowsort_desc_stable_both_topk_bitonic"
        );
        // Stage the full input row over k_in (M6: not k_out).
        assert!(
            src.contains("long long pow2 = 1; while (pow2 < k_in) pow2 <<= 1;"),
            "{}",
            src
        );
        assert!(src.contains("long long base_in = row * k_in;"), "{}", src);
        assert!(src.contains("long long out_base = row * k_out;"), "{}", src);
        assert!(
            src.contains("if (p < k_in) { skey[p] = in0[base_in + p]; sidx[p] = (int)p; }"),
            "{}",
            src
        );
        // Writeback SHORTENED to k_out (M5), gathering through the permutation.
        assert!(
            src.contains("for (long long p = threadIdx.x; p < k_out; p += blockDim.x) {"),
            "{}",
            src
        );
        assert!(
            src.contains("out_val[out_base + p] = in0[base_in + sidx[p]];"),
            "{}",
            src
        );
        assert!(src.contains("out_idx[out_base + p] = sidx[p];"), "{}", src);
        assert_eq!(bt.fidelity, crate::VariantFidelity::BitIdentical);
        // The launch_note metadata must name the (k_in, k_out) TopK ABI — the
        // variant reads `limit`, so a stale Full note here (`next_pow2(k)`,
        // `REQUIRES k <= 1024`) would mis-describe the smem/occupancy contract.
        assert!(
            bt.launch_note.contains("next_pow2(k_in)")
                && bt.launch_note.contains("REQUIRES k_in <= 1024")
                && bt.launch_note.contains("top k_out"),
            "topk bitonic launch_note must describe the (k_in,k_out) ABI:\n{}",
            bt.launch_note
        );
        assert!(
            !bt.launch_note.contains("REQUIRES k <= 1024"),
            "topk bitonic launch_note must NOT carry the stale Full `k` contract:\n{}",
            bt.launch_note
        );
    }

    #[test]
    fn bottomk_both_base_golden_f32() {
        // bottomk = Asc + cap: the comparator argument order swaps to (ka,kb) first
        // (smallest first) while the cap is identical — proving `order` selects the
        // direction under TopK (M8: topk must NOT emit Asc). The `_topk` suffix + the
        // guard + the split bases match the topk golden.
        let sc = OpDef::row_bottomk("bottomk", ElementKind::F32);
        let k = generate(&sc, &topk_both_key(ElementKind::F32, 64), &Cuda);
        assert_eq!(
            k.name,
            "baracuda_gen_bottomk_f32_rowsort_asc_stable_both_topk"
        );
        // Asc comparator argument order (ka,kb first) — the M8 divergence from topk.
        assert!(
            k.source
                .contains("if (bottomk_f32_asc_both_topk_key_lt(ka, kb)) return true;"),
            "{}",
            k.source
        );
        // Same cap machinery as topk.
        assert!(k.source.contains("if (r < k_out) {"), "{}", k.source);
        assert!(
            k.source
                .contains("out_val[out_base + r] = in0[in_base + i];"),
            "{}",
            k.source
        );
        assert!(
            k.source.contains("long long total = n_out * k_in;"),
            "{}",
            k.source
        );
    }

    #[test]
    fn topk_values_base_golden_f32() {
        // The FREE `TopK`+`Values` single-output path (admitted-but-unexposed): a
        // single `float* out` pointer, the guarded `out[out_base + r] = in0[in_base +
        // i];` store — so the capped single-output path is tested, not dead. Built by
        // capping a Desc values-sort (no public ctor in v1).
        use crate::ir::{Access, SortLimit, SortOut};
        let mut sc = OpDef::row_sort("topk", ElementKind::F32, SortOrder::Desc);
        sc.access = Access::RowSort {
            order: SortOrder::Desc,
            stable: true,
            out: SortOut::Values,
            limit: SortLimit::TopK,
        };
        let k = generate(&sc, &topk_values_key(ElementKind::F32, 64), &Cuda);
        assert_eq!(k.name, "baracuda_gen_topk_f32_rowsort_desc_stable_topk");
        assert!(
            k.source.contains("float* __restrict__ out,"),
            "{}",
            k.source
        );
        assert!(k.source.contains("if (r < k_out) {"), "{}", k.source);
        assert!(
            k.source.contains("out[out_base + r] = in0[in_base + i];"),
            "{}",
            k.source
        );
        assert!(!k.source.contains("out_val"), "{}", k.source);
        assert!(!k.source.contains("out_idx"), "{}", k.source);
        assert!(!k.source.contains("_both"), "{}", k.source);
    }

    #[test]
    fn topk_symbol_is_distinct() {
        // M7: the `_topk` entry-point AND its `__device__` comparator stem collide
        // with NEITHER the full-sort Both (`_both`) NOR any other symbol in the
        // validator TU — reverting the suffix to "" would duplicate-link.
        let topk = generate(
            &OpDef::row_topk("op", ElementKind::F32),
            &topk_both_key(ElementKind::F32, 64),
            &Cuda,
        );
        let full_both = generate(
            &OpDef::row_sort_indices("op", ElementKind::F32, SortOrder::Desc),
            &both_key(ElementKind::F32),
            &Cuda,
        );
        assert_eq!(
            topk.name,
            "baracuda_gen_op_f32_rowsort_desc_stable_both_topk"
        );
        assert_eq!(
            full_both.name,
            "baracuda_gen_op_f32_rowsort_desc_stable_both"
        );
        assert_ne!(topk.name, full_both.name);
        // The comparator stems are distinct, and NOT substring collisions (the full
        // stem `op_f32_desc_both_pair_lt` is not inside the topk source).
        assert!(
            topk.source.contains("op_f32_desc_both_topk_pair_lt"),
            "{}",
            topk.source
        );
        assert!(
            !topk.source.contains("op_f32_desc_both_pair_lt"),
            "{}",
            topk.source
        );
        assert!(
            full_both.source.contains("op_f32_desc_both_pair_lt"),
            "{}",
            full_both.source
        );
    }

    #[test]
    fn full_sort_emits_no_topk_artifacts() {
        // M1/M12 REGRESSION (the #1 risk): every `Full` state (row_sort/row_argsort/
        // row_sort_indices), base AND bitonic, emits the single-`k` ABI and NEVER the
        // TopK split — no `_topk`, no `k_in`/`k_out`, no `if (r < k_out)` guard. This
        // is the load-bearing byte-identity guard: a mutation that emits the split
        // under Full fails HERE.
        let cases = [
            (
                OpDef::row_sort("s", ElementKind::F32, SortOrder::Asc),
                sort_key(ElementKind::F32),
            ),
            (
                OpDef::row_argsort("s", ElementKind::F32, SortOrder::Asc),
                argsort_key(ElementKind::F32),
            ),
            (
                OpDef::row_sort_indices("s", ElementKind::F32, SortOrder::Asc),
                both_key(ElementKind::F32),
            ),
        ];
        for (sc, key) in &cases {
            for v in generate_variants(sc, key, &Cuda) {
                for kern in &v.kernels {
                    assert!(kern.source.contains("long long k)"), "{}", kern.source);
                    assert!(
                        kern.source.contains("if (k == 0) return;"),
                        "{}",
                        kern.source
                    );
                    assert!(!kern.source.contains("_topk"), "{}", kern.source);
                    assert!(!kern.source.contains("k_in"), "{}", kern.source);
                    assert!(!kern.source.contains("k_out"), "{}", kern.source);
                    assert!(!kern.source.contains("in_base"), "{}", kern.source);
                    assert!(!kern.source.contains("out_base"), "{}", kern.source);
                    assert!(!kern.source.contains("if (r < k_out)"), "{}", kern.source);
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "out_idx (operand 2) must be forward-dense")]
    fn topk_both_emitter_backstop_rejects_broadcast_out_idx() {
        // M11 (Tier-2 independent emitter backstop): a hand-built `TopK`+`Both`
        // KernelPlan whose out_idx (operand 2) is broadcast (stride 0) — the SHARED
        // layout backstop (width-agnostic; it covers TopK too) must refuse it. If a
        // mutation gated the backstop on `limit == Full`, this TopK path would slip
        // through — this pins that it does not.
        use crate::backend::Backend;
        use crate::ir::{Access, BaseOffset, SortLimit, SortOut, WriteIndex};
        use crate::plan::{KernelPlan, Schedule};
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let ov = OperandDesc::new(2, &[256, 64], &[64, 1], ElementKind::F32, 256);
        let oi = OperandDesc::new(2, &[256, 64], &[0, 1], ElementKind::I32, 256); // broadcast
        let key = structure_key(OpCategory::UnaryElementwise, &[a, ov, oi], ArchSku::Sm89);
        let body = crate::ir::input(0).0;
        let access = Access::RowSort {
            order: SortOrder::Desc,
            stable: true,
            out: SortOut::Both,
            limit: SortLimit::TopK,
        };
        let plan = KernelPlan {
            op_name: "topk_bad",
            n_inputs: 1,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::RowSort {
                order: SortOrder::Desc,
                stable: true,
                out: SortOut::Both,
                limit: SortLimit::TopK,
            },
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &access,
            views: &[],
            read_index: &[],
            write_index: &WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    fn sort_both_is_an_honest_miss_no_contract() {
        // AOT-only, exactly like row_sort/row_argsort: `Both` is still
        // Access::RowSort (non-Elementwise), so `derive_pattern` rejects it as
        // NotElementwise BEFORE any body walk and `contract()` returns None. (This
        // lives here, not in contract.rs, to keep that file's diff empty — the
        // withhold path is a property of the RowSort access, unchanged by the state.)
        use crate::pattern::PatternError;
        let sc = OpDef::row_sort_indices("fused", ElementKind::F32, SortOrder::Asc);
        let k = generate(&sc, &both_key(ElementKind::F32), &Cuda);
        assert!(
            crate::contract(&sc, &both_key(ElementKind::F32), &k, "cuda").is_none(),
            "a fused two-output sort must emit NO contract (no Fuel Sort OpTag; AOT-only)"
        );
        assert!(matches!(
            crate::derive_pattern(&sc),
            Err(PatternError::NotElementwise)
        ));
    }

    #[test]
    fn topk_is_an_honest_miss_no_contract() {
        // Increment 10 AOT-only (unchanged from #8/#9): a capped topk is still
        // Access::RowSort (non-Elementwise), so `derive_pattern` rejects it as
        // NotElementwise and `contract()` returns None — the SAME withhold path,
        // ZERO contract.rs/pattern.rs change. Fuel has no top_k OpKind (MoE routes
        // densely), so AOT-only is the correct posture, not a limitation.
        use crate::pattern::PatternError;
        let sc = OpDef::row_topk("topk", ElementKind::F32);
        let k = generate(&sc, &topk_both_key(ElementKind::F32, 64), &Cuda);
        assert!(
            crate::contract(&sc, &topk_both_key(ElementKind::F32, 64), &k, "cuda").is_none(),
            "a topk must emit NO contract (no Fuel top_k OpTag; AOT-only)"
        );
        assert!(matches!(
            crate::derive_pattern(&sc),
            Err(PatternError::NotElementwise)
        ));
    }

    #[test]
    #[should_panic(expected = "key must carry n_inputs+2 operands")]
    fn both_emitter_backstop_rejects_two_operand_key() {
        // Tier-2 independent emitter backstop (M10): a hand-built `Both` KernelPlan
        // with a 2-operand key (missing out_idx) — build_plan's G2 would reject it,
        // but the emitter must ALSO refuse it (the two-pointer signature assumes the
        // third operand). Bypass build_plan and hand it straight to Cuda::lower.
        use crate::backend::Backend;
        use crate::ir::{Access, BaseOffset, SortLimit, SortOut, WriteIndex};
        use crate::plan::{KernelPlan, Schedule};
        let key = sort_key(ElementKind::F32); // only [in0, out_val] — 2 operands
        let body = crate::ir::input(0).0;
        let access = Access::RowSort {
            order: SortOrder::Asc,
            stable: true,
            out: SortOut::Both,
            limit: SortLimit::Full,
        };
        let plan = KernelPlan {
            op_name: "fused_bad",
            n_inputs: 1,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::RowSort {
                order: SortOrder::Asc,
                stable: true,
                out: SortOut::Both,
                limit: SortLimit::Full,
            },
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &access,
            views: &[],
            read_index: &[],
            write_index: &WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "out_idx (operand 2) must be forward-dense")]
    fn both_emitter_backstop_rejects_broadcast_out_idx() {
        // Tier-2 independent emitter backstop: a hand-built `Both` KernelPlan whose
        // out_idx (operand 2) is broadcast (stride 0) — the plan gate G3 would reject
        // it, but the emitter must ALSO refuse it (the forward-dense out_idx[base+r]
        // store assumes a full-width contiguous target). Pins the emitter's out_idx
        // LAYOUT re-check independently of the plan gate.
        use crate::backend::Backend;
        use crate::ir::{Access, BaseOffset, SortLimit, SortOut, WriteIndex};
        use crate::plan::{KernelPlan, Schedule};
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let ov = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let oi = OperandDesc::new(2, &[256, 128], &[0, 1], ElementKind::I32, 256); // broadcast
        let key = structure_key(OpCategory::UnaryElementwise, &[a, ov, oi], ArchSku::Sm89);
        let body = crate::ir::input(0).0;
        let access = Access::RowSort {
            order: SortOrder::Asc,
            stable: true,
            out: SortOut::Both,
            limit: SortLimit::Full,
        };
        let plan = KernelPlan {
            op_name: "fused_bad",
            n_inputs: 1,
            dtype: ElementKind::F32,
            out_dtype: ElementKind::F32,
            schedule: Schedule::RowSort {
                order: SortOrder::Asc,
                stable: true,
                out: SortOut::Both,
                limit: SortLimit::Full,
            },
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &access,
            views: &[],
            read_index: &[],
            write_index: &WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    fn bitonic_filter_declines_non_rowsort() {
        // A scan plan must not surface the sort bitonic variant.
        let sc = OpDef::scan_simple(
            "cum",
            &[ElementKind::F32],
            crate::ir::ReduceOp::Sum,
            1,
            false,
            false,
        );
        let vs = generate_variants(&sc, &sort_key(ElementKind::F32), &Cuda);
        assert!(
            vs.iter().all(|v| v.tag != "bitonic"),
            "scan must not offer the sort bitonic variant"
        );
    }

    // ======================= PARTIAL-SELECT TOPK (psel) =======================
    // The streaming tiled-bitonic top-k VARIANT (O(k_in·log²k_out), smem bounded
    // on k_out) — BitIdentical to the full-sort top-k_out. On-device numeric proof
    // (memcmp==0 vs the base across the §8 matrix + the 11 probes) is in
    // ondevice/sort_validate.cu; these pin the source shape + variant wiring.

    #[test]
    fn partial_select_variant_present_for_topk() {
        // A row_topk cell (small k_out, large k_in) OFFERS a `psel` variant:
        // BitIdentical, name ends `_topk_psel`.
        let sc = OpDef::row_topk("topk", ElementKind::F32);
        let vs = generate_variants(&sc, &topk_both_key(ElementKind::F32, 64), &Cuda);
        let ps = vs
            .iter()
            .find(|v| v.tag == "psel")
            .expect("psel variant present for a TopK cell");
        assert_eq!(ps.fidelity, crate::VariantFidelity::BitIdentical);
        assert_eq!(ps.kernels.len(), 1);
        assert!(
            ps.kernels[0].name.ends_with("_topk_psel"),
            "{}",
            ps.kernels[0].name
        );
        assert_eq!(
            ps.kernels[0].name,
            "baracuda_gen_topk_f32_rowsort_desc_stable_both_topk_psel"
        );
    }

    #[test]
    fn partial_select_declines_full() {
        // A row_sort (Full) offers NO psel — the streaming best-buffer is a
        // top-k_out selector with no whole-row form (the bitonic/base serve Full).
        let sc = OpDef::row_sort("sort", ElementKind::F32, SortOrder::Asc);
        let vs = generate_variants(&sc, &sort_key(ElementKind::F32), &Cuda);
        assert!(
            vs.iter().all(|v| v.tag != "psel"),
            "a Full row_sort must not offer the partial-select top-k variant"
        );
        // The fused Both Full sort likewise offers no psel.
        let fv = OpDef::row_sort_indices("fused", ElementKind::F32, SortOrder::Desc);
        let vf = generate_variants(&fv, &both_key(ElementKind::F32), &Cuda);
        assert!(vf.iter().all(|v| v.tag != "psel"));
    }

    #[test]
    fn partial_select_offers_regardless_of_kout_extent_free_key() {
        // BRIEF DEVIATION (flagged, not a silent work-around): the panel's
        // compile-time "advantage decline" (None when 2·next_pow2(k_out) ≥
        // next_pow2(k_in)) is NOT implementable here — the StructureKey carries
        // size CLASSES, never literal extents (the §1 non-negotiable), so NEITHER
        // k_in NOR k_out is recoverable at variant-selection time (verified:
        // OperandKey exposes only contig/bcast/vec_width/inner_div/flipped; the key
        // exposes only `work` = total-work class). This is the SAME limitation the
        // bitonic variant has with its k ≤ 1024 precondition. Both are BitIdentical
        // measured variants: psel is offered for EVERY TopK cell it can serve, the
        // launch_note documents the advantage regime, and the bench/dispatch layer
        // (which sees concrete extents) selects the measured default (ship-top-K).
        //
        // This test pins the ACTUAL behavior: a large-k_out TopK cell (where the
        // brief wanted a decline) still offers psel, and two keys that differ ONLY
        // in the literal k_out canonicalize identically ⇒ no gate could distinguish
        // them. If a literal-extent facet is ever added to the key, revisit §4.
        let sc = OpDef::row_topk("topk", ElementKind::F32);
        // Input inner extent 128 (next_pow2 = 128). k_out = 16 → the brief's
        // ADVANTAGE regime (2·next_pow2(16) = 32 < 128); k_out = 112 → the brief's
        // DECLINE regime (2·next_pow2(112) = 256 ≥ 128). Both are ÷16, so they
        // canonicalize to the SAME byte-equal key (only the divisibility BUCKET is
        // keyed, never the magnitude) — no gate could tell the advantage cell from
        // the no-advantage cell. This is the concrete proof §4's decline is
        // ungateable on the extent-free key.
        let k_small = topk_both_key(ElementKind::F32, 16);
        let k_large = topk_both_key(ElementKind::F32, 112);
        assert_eq!(
            k_small, k_large,
            "the extent-free structure key cannot distinguish k_out magnitude — so no \
             compile-time decline gate is possible (brief §4 deviation)"
        );
        for key in [&k_small, &k_large] {
            let vs = generate_variants(&sc, key, &Cuda);
            assert!(
                vs.iter().any(|v| v.tag == "psel"),
                "psel is offered for every TopK cell (no extent-based decline)"
            );
        }
    }

    #[test]
    fn partial_select_fidelity_bit_identical() {
        // psel fidelity == BitIdentical for topk (Desc) AND bottomk (Asc).
        for sc in [
            OpDef::row_topk("topk", ElementKind::F32),
            OpDef::row_bottomk("bottomk", ElementKind::F32),
        ] {
            let vs = generate_variants(&sc, &topk_both_key(ElementKind::F32, 64), &Cuda);
            let ps = vs.iter().find(|v| v.tag == "psel").expect("psel present");
            assert_eq!(
                ps.fidelity,
                crate::VariantFidelity::BitIdentical,
                "psel is BitIdentical to the full-sort top-k_out (composition lemma)"
            );
            // The launch_note names the k_out-bounded smem contract (NOT k_in) and
            // carries no `k_in <= 1024` cap — the streaming win.
            assert!(
                ps.launch_note.contains("2 * next_pow2(k_out)")
                    && ps.launch_note.contains("NOT k_in")
                    && ps.launch_note.contains("NO k_in <= 1024 cap"),
                "psel launch_note must describe the k_out-bounded streaming smem:\n{}",
                ps.launch_note
            );
        }
    }

    #[test]
    fn partial_select_emission_goldens() {
        // The `_psel` source contains the load-bearing shape: the 2m arena, the
        // in-kernel m ramp, the ladder bound, the GLOBAL-index store, the keep-lower
        // writeback, and sort_pad_lit-correct pads. (psel is a VARIANT, so it comes
        // off generate_variants, not the default `generate` lowering = the base.)
        let src = psel_src(
            &OpDef::row_topk("topk", ElementKind::F32),
            &topk_both_key(ElementKind::F32, 64),
        );
        let src = &src;
        // 2m-pair arena carved from ONE contiguous smem block (sidx after 2m keys).
        assert!(
            src.contains("int* sidx = (int*)(baracuda_sort_smem + (size_t)(2*m) * sizeof(float));"),
            "{src}"
        );
        // In-kernel m = next_pow2(k_out) (NOT next_pow2(k_in)).
        assert!(
            src.contains("long long m = 1; while (m < k_out) m <<= 1;"),
            "{src}"
        );
        // Full Batcher ladder over the 2m window.
        assert!(
            src.contains("for (long long kk = 2; kk <= 2*m; kk <<= 1) {"),
            "{src}"
        );
        assert!(
            src.contains("for (long long p = threadIdx.x; p < 2*m; p += blockDim.x) {"),
            "{src}"
        );
        // GLOBAL-index store for BOTH reals AND pads (the #1 bug surface).
        assert!(src.contains("long long g = t * m + (s - m);"), "{src}");
        assert!(src.contains("sidx[s] = (int)g;"), "{src}");
        // Keep-lower writeback reads sidx[p] for p < k_out (Both writes BOTH buffers).
        assert!(
            src.contains("for (long long p = threadIdx.x; p < k_out; p += blockDim.x) {"),
            "{src}"
        );
        assert!(
            src.contains("out_val[out_base + p] = in0[base_in + sidx[p]];"),
            "{src}"
        );
        assert!(src.contains("out_idx[out_base + p] = sidx[p];"), "{src}");
        // topk (Desc) pad = -inf (sort_pad_lit type-min), init index >= k_in,
        // SATURATED at INT_MAX so `k_in + p` cannot overflow `(int)` near the 2^31-1
        // cap (a wrapped-negative pad index would tie-beat a real — the review fix).
        assert!(
            src.contains(
                "skey[p] = __int_as_float(0xff800000u); long long pi = k_in + p; \
                 sidx[p] = (int)(pi < 2147483647LL ? pi : 2147483647LL);"
            ),
            "{src}"
        );
        assert!(
            src.contains("else       { skey[s] = __int_as_float(0xff800000u); }"),
            "{src}"
        );

        // bottomk (Asc) FP pad = qNaN (NOT +inf — a real NaN > +inf leaks a pad).
        let bottomk = psel_src(
            &OpDef::row_bottomk("bottomk", ElementKind::F32),
            &topk_both_key(ElementKind::F32, 64),
        );
        assert!(
            bottomk.contains("skey[p] = __int_as_float(0x7fc00000u);"),
            "bottomk FP pad must be a qNaN, never +inf:\n{bottomk}"
        );
        assert!(
            !bottomk.contains("0x7f800000u"),
            "bottomk must NOT emit a +inf pad (leaks before real NaNs):\n{bottomk}"
        );
        // bottomk (Asc) INTEGER pad = INT_MAX literal, NOT a NaN bitcast.
        let bottomk_i32 = psel_src(
            &OpDef::row_bottomk("bottomk", ElementKind::I32),
            &topk_both_key(ElementKind::I32, 64),
        );
        assert!(
            bottomk_i32.contains(
                "skey[p] = 2147483647; long long pi = k_in + p; \
                 sidx[p] = (int)(pi < 2147483647LL ? pi : 2147483647LL);"
            ),
            "bottomk integer pad must be INT_MAX (2147483647), never a NaN bitcast:\n{bottomk_i32}"
        );
        assert!(
            !bottomk_i32.contains("__int_as_float"),
            "an integer psel kernel must not reinterpret a NaN bit-pattern:\n{bottomk_i32}"
        );
    }

    #[test]
    fn partial_select_stores_global_not_slot() {
        // The single most likely bug (§2): the sidx store MUST be the ORIGINAL
        // GLOBAL index g = t*m + (s-m), never the local smem slot. The shipped
        // full-row bitonic stores `sidx[p] = (int)p` (slot == global there); the
        // STREAMING slot p is generally < k_in, so a slot store would break the
        // value gather, the tie-break, and put pad indices below k_in.
        let src = psel_src(
            &OpDef::row_topk("topk", ElementKind::F32),
            &topk_both_key(ElementKind::F32, 64),
        );
        let src = &src;
        // The global-index store is present.
        assert!(src.contains("sidx[s] = (int)g;"), "{src}");
        // The T-load must NOT store the local slot as the index.
        assert!(
            !src.contains("sidx[s] = (int)(s - m);"),
            "the T-load must store the GLOBAL index g, not the local slot (s-m):\n{src}"
        );
        // No bare `sidx[p] = (int)p;` (the shipped bitonic's slot store) leaks in.
        assert!(
            !src.contains("sidx[p] = (int)p;"),
            "the streaming variant must never store the local slot as the index:\n{src}"
        );
        // The init-B pad index is >= k_in (dominated, never output), not the slot,
        // and SATURATED at INT_MAX so `k_in + p` cannot overflow `(int)` to a
        // negative that would tie-beat a real (the review fix).
        assert!(
            src.contains(
                "long long pi = k_in + p; sidx[p] = (int)(pi < 2147483647LL ? pi : 2147483647LL);"
            ),
            "{src}"
        );
        // The bare unguarded `(int)(k_in + p)` overflow form must NOT remain.
        assert!(
            !src.contains("sidx[p] = (int)(k_in + p);"),
            "init-B pad index must be saturated at INT_MAX, not the overflowing (int)(k_in + p):\n{src}"
        );
    }

    /// Manual dump tool (not a wired assertion): regenerate the sort `.cu` sources
    /// the on-device validator `#include`s. Run with:
    ///   `cargo test -p baracuda-kernelgen dump_sort_sources -- --ignored --nocapture`
    /// then copy `ondevice/sort_validate.cu` beside the emitted files and `nvcc` it.
    #[test]
    #[ignore = "manual regeneration tool for ondevice/sort_validate.cu"]
    fn dump_sort_sources() {
        let out = std::env::var("SORT_OUT").unwrap_or_else(|_| ".".to_string());
        let write = |k: crate::GeneratedKernel| {
            std::fs::write(format!("{out}/{}.cu", k.name), &k.source).unwrap();
            println!("wrote {out}/{}.cu", k.name);
        };
        // Every base + bitonic cell the validator exercises: {f32,f64,i32} ×
        // {asc,desc} × {sort,argsort,fused-Both}, plus f16/bf16/i64/f32s asc (each).
        // Increment 9 FUSED_ARGSORT adds the `fused` (row_sort_indices) cell to each
        // dtype/order the validator's dual memcmp uses row_sort + row_argsort as the
        // oracle for; the fused key is the 3-operand `both_key`.
        let full = [ElementKind::F32, ElementKind::F64, ElementKind::I32];
        for dt in full {
            for order in [SortOrder::Asc, SortOrder::Desc] {
                let sv = OpDef::row_sort("sort", dt, order);
                for v in generate_variants(&sv, &sort_key(dt), &Cuda) {
                    for kern in v.kernels {
                        write(kern);
                    }
                }
                let av = OpDef::row_argsort("argsort", dt, order);
                for v in generate_variants(&av, &argsort_key(dt), &Cuda) {
                    for kern in v.kernels {
                        write(kern);
                    }
                }
                let fv = OpDef::row_sort_indices("fused", dt, order);
                for v in generate_variants(&fv, &both_key(dt), &Cuda) {
                    for kern in v.kernels {
                        write(kern);
                    }
                }
            }
        }
        // i64 asc sort + argsort + fused (base + bitonic each) — the wide-integer cell.
        {
            let sv = OpDef::row_sort("sort", ElementKind::I64, SortOrder::Asc);
            for v in generate_variants(&sv, &sort_key(ElementKind::I64), &Cuda) {
                for kern in v.kernels {
                    write(kern);
                }
            }
            let av = OpDef::row_argsort("argsort", ElementKind::I64, SortOrder::Asc);
            for v in generate_variants(&av, &argsort_key(ElementKind::I64), &Cuda) {
                for kern in v.kernels {
                    write(kern);
                }
            }
            let fv = OpDef::row_sort_indices("fused", ElementKind::I64, SortOrder::Asc);
            for v in generate_variants(&fv, &both_key(ElementKind::I64), &Cuda) {
                for kern in v.kernels {
                    write(kern);
                }
            }
        }
        // Half-precision + f32-strict asc (base + bitonic each) — the acc/convert
        // coverage cells. #8 dumped only the values sort here; increment 9 adds the
        // fused-Both cell AND the row_argsort oracle (the Both dual memcmp needs the
        // index oracle for these dtypes too).
        for dt in [ElementKind::F16, ElementKind::Bf16, ElementKind::F32Strict] {
            let sv = OpDef::row_sort("sort", dt, SortOrder::Asc);
            for v in generate_variants(&sv, &sort_key(dt), &Cuda) {
                for kern in v.kernels {
                    write(kern);
                }
            }
            let av = OpDef::row_argsort("argsort", dt, SortOrder::Asc);
            for v in generate_variants(&av, &argsort_key(dt), &Cuda) {
                for kern in v.kernels {
                    write(kern);
                }
            }
            let fv = OpDef::row_sort_indices("fused", dt, SortOrder::Asc);
            for v in generate_variants(&fv, &both_key(dt), &Cuda) {
                for kern in v.kernels {
                    write(kern);
                }
            }
        }
        // Increment 10 TOPK/BOTTOMK: row_topk (Desc) + row_bottomk (Asc), base +
        // bitonic, for every validator dtype — the capped `_topk` cells the
        // topk/bottomk acceptance matrix #includes. The out key is narrower
        // ([256,64]); the emitted source is k_out-independent (k_out is a launch
        // arg, never keyed), so one key per cell serves the whole boundary sweep.
        for dt in [
            ElementKind::F32,
            ElementKind::F64,
            ElementKind::I32,
            ElementKind::I64,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::F32Strict,
        ] {
            let tv = OpDef::row_topk("topk", dt);
            for v in generate_variants(&tv, &topk_both_key(dt, 64), &Cuda) {
                for kern in v.kernels {
                    write(kern);
                }
            }
            let bv = OpDef::row_bottomk("bottomk", dt);
            for v in generate_variants(&bv, &topk_both_key(dt, 64), &Cuda) {
                for kern in v.kernels {
                    write(kern);
                }
            }
        }
        // The DESC fused Both oracle for the topk (Desc) per-row cross-check on the
        // dtypes whose desc fused was NOT dumped above (i64/f16/bf16/f32s dumped asc
        // only for the #9 Both matrix). bottomk (Asc) reuses the asc fused Both.
        for dt in [
            ElementKind::I64,
            ElementKind::F16,
            ElementKind::Bf16,
            ElementKind::F32Strict,
        ] {
            let fv = OpDef::row_sort_indices("fused", dt, SortOrder::Desc);
            for v in generate_variants(&fv, &both_key(dt), &Cuda) {
                for kern in v.kernels {
                    write(kern);
                }
            }
        }
    }
}

#[cfg(test)]
mod select_tests {
    //! WHERE/SELECT emitter tests: emission goldens (the exact bitwise-pick
    //! spellings), the packed fallback (G7), the Tier-2 int backstop (G5),
    //! and the `#[ignore]`d source dump for `ondevice/select_validate.cu`.
    //! On-device numeric proof (triu/tril memcmp vs bespoke, NaN payloads,
    //! signed zeros) lives in that harness; these pin the source text.
    use crate::ir::{BinaryOp, OpDef, coord, input, konst, reduced};
    use crate::{Cuda, generate};
    use baracuda_kernel_vocab::{
        ArchSku, ElementKind, OpCategory, OperandDesc, StructureKey, structure_key,
    };

    /// `n_operands` contiguous rank-1 operands of `dt` at `align` (align 4/2/8
    /// defeats vectorization ⇒ the scalar golden; 256 keys V4/V8).
    fn sel_key(dt: ElementKind, n_operands: usize, align: u32) -> StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
        let ops: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(OpCategory::TernaryElementwise, &ops, ArchSku::Sm89)
    }

    /// [256,128] contiguous input → [256] output — reduce/row-reduce the last axis.
    fn red_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        let o = OperandDesc::new(1, &[256], &[1], dt, 256);
        structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89)
    }

    /// [256,128] in/out — a last-axis scan cell (same-shape output).
    fn scn_key(dt: ElementKind) -> StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    }

    /// [8,k_in] → [8,k_out] — a window/pool cell.
    fn win_key(dt: ElementKind, k_in: i64, k_out: i64) -> StructureKey {
        let a = OperandDesc::new(2, &[8, k_in], &[k_in, 1], dt, 256);
        let o = OperandDesc::new(2, &[8, k_out], &[k_out, 1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, o], ArchSku::Sm89)
    }

    /// `n_operands` odd-extent rank-1 operands (defeats V4 ⇒ the scalar-multi
    /// emitter) — a 3-in/2-out multi-output cell keys `n_operands = 5`.
    fn mo_key(dt: ElementKind, n_operands: usize) -> StructureKey {
        let a = OperandDesc::new(1, &[1_000_003], &[1], dt, 256);
        let ops: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
        structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89)
    }

    /// `cond.select(a, b)` over three inputs — the raw-pick shape.
    fn sel_op(dt: ElementKind) -> OpDef {
        OpDef::elementwise("sel", 3, &[dt], input(0).select(input(1), input(2)))
    }

    #[test]
    fn select_emission_golden_f32() {
        // The exact f32 ternary: cond decided `!= 0.0f` IN f32, both arms
        // `(float)`-pinned (identity on same-type operands; mandatory against
        // suffix-less double Const literals — M6's target), result f32.
        let k = generate(
            &sel_op(ElementKind::F32),
            &sel_key(ElementKind::F32, 4, 4),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_sel_f32_scalar");
        assert!(k.source.contains("const float* __restrict__ in2,"));
        assert!(
            k.source.contains(
                "out[i] = (((float)(in0[i])) != 0.0f ? (float)(in1[i]) : (float)(in2[i]));"
            ),
            "f32 select golden missing in:\n{}",
            k.source
        );
        // A Const arm stays inside the (float) pin: select(c, x, 0).
        let z = OpDef::elementwise(
            "selz",
            2,
            &[ElementKind::F32],
            input(0).select(input(1), konst(0.0)),
        );
        let kz = generate(&z, &sel_key(ElementKind::F32, 3, 4), &Cuda);
        assert!(
            kz.source
                .contains("out[i] = (((float)(in0[i])) != 0.0f ? (float)(in1[i]) : (float)(0.0));"),
            "f32 const-arm select golden missing in:\n{}",
            kz.source
        );
    }

    #[test]
    fn select_f64_golden_uses_double_literals() {
        let k = generate(
            &sel_op(ElementKind::F64),
            &sel_key(ElementKind::F64, 4, 8),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_sel_f64_scalar");
        assert!(
            k.source.contains(
                "out[i] = (((double)(in0[i])) != 0.0 ? (double)(in1[i]) : (double)(in2[i]));"
            ),
            "f64 select golden missing in:\n{}",
            k.source
        );
        assert!(!k.source.contains("0.0f"), "no float literals at f64");
    }

    #[test]
    fn select_f16_arms_are_raw_half_picks() {
        // Promote ONLY the cond (exact, the cmp-operand precedent); the ARMS
        // move as raw __half — never through the promote-demote wrapper (M7's
        // target). The `(__half)` arm cast is a same-type identity (no float
        // bridge, bits intact) that pins the ternary's type; a Const arm
        // converts once (double literal → half, +0.0 bits).
        let k = generate(
            &sel_op(ElementKind::F16),
            &sel_key(ElementKind::F16, 4, 2),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_sel_f16_scalar");
        assert!(
            k.source.contains(
                "out[i] = (__half2float(in0[i]) != 0.0f ? (__half)(in1[i]) : (__half)(in2[i]));"
            ),
            "f16 raw-pick select golden missing in:\n{}",
            k.source
        );
        // NO promote on either arm, NO demote wrapper around the pick.
        assert!(!k.source.contains("__half2float(in1[i])"), "{}", k.source);
        assert!(!k.source.contains("__half2float(in2[i])"), "{}", k.source);
        assert!(!k.source.contains("__float2half"), "{}", k.source);
        // Const arm at f16: converts ONCE via the half ctor (+0.0 bits), no
        // float bridge on the Input arm.
        let z = OpDef::elementwise(
            "selz",
            2,
            &[ElementKind::F16],
            input(0).select(input(1), konst(0.0)),
        );
        let kz = generate(&z, &sel_key(ElementKind::F16, 3, 2), &Cuda);
        assert!(
            kz.source.contains(
                "out[i] = (__half2float(in0[i]) != 0.0f ? (__half)(in1[i]) : (__half)(0.0));"
            ),
            "f16 const-arm select golden missing in:\n{}",
            kz.source
        );
    }

    #[test]
    fn select_bf16_arms_are_raw_picks() {
        let k = generate(
            &sel_op(ElementKind::Bf16),
            &sel_key(ElementKind::Bf16, 4, 2),
            &Cuda,
        );
        assert!(
            k.source.contains(
                "out[i] = (__bfloat162float(in0[i]) != 0.0f ? (__nv_bfloat16)(in1[i]) : \
                 (__nv_bfloat16)(in2[i]));"
            ),
            "bf16 raw-pick select golden missing in:\n{}",
            k.source
        );
        assert!(
            !k.source.contains("__bfloat162float(in1[i])"),
            "{}",
            k.source
        );
        assert!(
            !k.source.contains("__bfloat162float(in2[i])"),
            "{}",
            k.source
        );
        assert!(!k.source.contains("__float2bfloat16"), "{}", k.source);
    }

    #[test]
    fn select_vectorizes_per_lane_f32() {
        // All-Input select on a V4 cell: the whole body lowers once per lane
        // through the same select closure — per-lane ternary, bit-safe.
        let k = generate(
            &sel_op(ElementKind::F32),
            &sel_key(ElementKind::F32, 4, 256),
            &Cuda,
        );
        assert_eq!(k.name, "baracuda_gen_sel_f32_co_v4");
        for lane in ["x", "y", "z", "w"] {
            assert!(
                k.source.contains(&format!(
                    "vo.{lane} = (((float)(v0.{lane})) != 0.0f ? (float)(v1.{lane}) : \
                     (float)(v2.{lane}));"
                )),
                "lane {lane} select golden missing in:\n{}",
                k.source
            );
        }
    }

    #[test]
    fn select_with_nonleaf_body_falls_to_scalar_at_packed_cells() {
        // G7 (M12's target): `body_packs` returns false for Select, so a
        // packed-eligible f16 V8 cell with a select body takes the SCALAR
        // path — no packed pair struct, no pair splitting.
        let k = generate(
            &sel_op(ElementKind::F16),
            &sel_key(ElementKind::F16, 4, 256),
            &Cuda,
        );
        assert!(k.name.ends_with("_scalar"), "got {}", k.name);
        assert!(!k.source.contains("_vec"), "no packed vector struct");
        assert!(!k.source.contains("__halves2half2"), "no pair re-join");
        // The bf16 sibling too.
        let kb = generate(
            &sel_op(ElementKind::Bf16),
            &sel_key(ElementKind::Bf16, 4, 256),
            &Cuda,
        );
        assert!(kb.name.ends_with("_scalar"), "got {}", kb.name);
    }

    #[test]
    fn triu_select_strided_golden() {
        // The §6 acceptance-gate authoring: select(coord(1) >= coord(0) + k,
        // input(0), konst(0.0)) — the Coord cond forces Strided, the pick
        // stores bespoke-exact +0.0 on masked lanes (bit-proof on device).
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
        let k = generate(&triu, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_triu_sel_f32_strided_r2");
        assert!(
            k.source.contains(
                "out[oo] = (((float)(((float)(float)c1 >= (float)((float)c0 + 0.0) ? 1.0f : \
                 0.0f))) != 0.0f ? (float)(in0[o0]) : (float)(0.0));"
            ),
            "triu select strided golden missing in:\n{}",
            k.source
        );
        // The select pick replaces the 0d mask-MULTIPLY: no `*` touches the arm.
        assert!(
            !k.source.contains("in0[o0] *") && !k.source.contains("* in0[o0]"),
            "no mask-multiply on the data arm:\n{}",
            k.source
        );
    }

    #[test]
    fn shared_select_hoists_once_through_the_dag() {
        // DAG/CSE: a select value consumed twice hoists to ONE tmp (text stays
        // linear; the hoisted arm evaluates eagerly — value-neutral, §2).
        let sel = input(0).select(input(1), input(2));
        let body = sel.clone() + sel;
        let op = OpDef::elementwise("selshared", 3, &[ElementKind::F32], body);
        let k = generate(&op, &sel_key(ElementKind::F32, 4, 4), &Cuda);
        assert!(
            k.source.contains(
                "float tmp0 = (((float)(in0[i])) != 0.0f ? (float)(in1[i]) : (float)(in2[i]));"
            ),
            "shared select must hoist to tmp0:\n{}",
            k.source
        );
        assert!(k.source.contains("out[i] = (tmp0 + tmp0);"), "{}", k.source);
    }

    // ---- Non-elementwise select-closure emission goldens (review follow-up) ----
    //
    // The 3 elementwise select closures (emit_scalar/emit_strided/emit_vectorized)
    // are golden-pinned above, but Select is legal in EVERY Access arm at float
    // dtypes (G3) and 10 further `Lowering.select` closures emit it — reduction
    // pre/post, row-reduce epilogue, scan pre/post, window pre/post, and the three
    // multi-output emitters. G3's `masked_sum_reduction_with_a_select_body_builds`
    // proves only the PLAN builds (build_plan-direct); it never lowers. Without an
    // emission golden per family a SITE-LOCAL closure mutation — an `a`/`b` arm
    // swap (`select_f64(c, a, b)` → `(c, b, a)`) or a flipped `dbl` branch (f32
    // spelling truncating f64 arms) — silently inverts every masked
    // reduction/scan/window/multi-output kernel while the whole suite stays green
    // (the house dual-gate rule: every load-bearing emission decision gets a kill).
    //
    // Each golden uses distinguishable const arms `7.5`/`9.25` so the canonical
    // fragment `!= 0.0f ? (float)(7.5) : (float)(9.25)` pins BOTH the arm order
    // (a swap makes it `9.25 : 7.5`) AND the f32 speller (a `dbl`-flip makes it
    // `(double)`); one f64 sibling pins the double branch directly. Every kill is
    // mutation-checked (a site-local `select_f{32,64}(c, b, a)` swap or `dbl`-flip
    // fails ONLY its family's golden).
    //
    // These cover 10 of the 13 non-elementwise closures (reduction pre/post,
    // row-reduce, scan pre/post, window pre/post, and all three multi-output
    // emitters). The remaining THREE — reduction_splitk, contraction, and
    // contraction_splitk — are variant (`generate_variants` only, on specific
    // outer-axis/contraction cells) and contraction-epilogue (`Reduced(0)`-only)
    // paths that share the IDENTICAL `if dbl { select_f64 } else { select_f32 }`
    // closure body pinned here; a select reaching them is exotic (a masked
    // split-K reduction or a select in a matmul epilogue) and is a follow-up if
    // ever generated. The shared speller means any dtype-routing or arm-order bug
    // in `select_f32`/`select_f64`/`cuda_select` themselves is already killed by
    // the elementwise + these goldens.

    /// The canonical ordered f32 select fragment — arm order + f32 speller.
    fn assert_ordered_select_f32(src: &str, ctx: &str) {
        assert!(
            src.contains("!= 0.0f ? (float)(7.5) : (float)(9.25)"),
            "{ctx}: ordered f32 select fragment missing (arm swap or dbl-flip?):\n{src}"
        );
    }

    #[test]
    fn select_in_reduction_pre_body_emits_ordered_f32_and_f64() {
        use crate::ir::ReduceOp;
        // A masked pre-fold select (emit_reduction pre closure): the fold
        // accumulates `select(in0 > 0.5, 7.5, 9.25)` per element.
        let cond = input(0).binary(BinaryOp::CmpGt, konst(0.5));
        let op = OpDef::reduction(
            "mred",
            1,
            &[ElementKind::F32],
            cond.clone().select(konst(7.5), konst(9.25)),
            ReduceOp::Sum,
        );
        let k = generate(&op, &red_key(ElementKind::F32), &Cuda);
        assert_ordered_select_f32(&k.source, "reduction-pre f32");
        // f64 sibling pins the `dbl` branch (double literals, no float speller).
        let op64 = OpDef::reduction(
            "mred",
            1,
            &[ElementKind::F64],
            cond.select(konst(7.5), konst(9.25)),
            ReduceOp::Sum,
        );
        let k64 = generate(&op64, &red_key(ElementKind::F64), &Cuda);
        assert!(
            k64.source
                .contains("!= 0.0 ? (double)(7.5) : (double)(9.25)"),
            "reduction-pre f64 double-branch fragment missing:\n{}",
            k64.source
        );
        assert!(
            !k64.source.contains("(float)(7.5)"),
            "f64 reduction must not emit the f32 speller:\n{}",
            k64.source
        );
    }

    #[test]
    fn select_in_reduction_post_body_emits_ordered_f32() {
        use crate::ir::ReduceOp;
        // A masked epilogue select (emit_reduction post closure): the post reads
        // Reduced(0), so the cond derives from the fold result; arms are Const.
        let post = reduced(0)
            .binary(BinaryOp::CmpGt, konst(0.5))
            .select(konst(7.5), konst(9.25));
        let op = OpDef::reduction_post(
            "mredpost",
            1,
            &[ElementKind::F32],
            input(0),
            ReduceOp::Sum,
            post,
        );
        let k = generate(&op, &red_key(ElementKind::F32), &Cuda);
        assert_ordered_select_f32(&k.source, "reduction-post f32");
    }

    #[test]
    fn select_in_row_reduce_epilogue_emits_ordered_f32() {
        use crate::ir::{ReduceOp, ReduceStage};
        // A masked row-reduce epilogue select (emit_row_reduce_impl closure): one
        // Sum stage, epilogue `select(in0 > 0.5, 7.5, 9.25)` (reads Input(0)).
        let op = OpDef::row_reduce(
            "mrr",
            1,
            &[ElementKind::F32],
            vec![ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Sum,
            }],
            input(0)
                .binary(BinaryOp::CmpGt, konst(0.5))
                .select(konst(7.5), konst(9.25)),
        );
        let k = generate(&op, &red_key(ElementKind::F32), &Cuda);
        assert_ordered_select_f32(&k.source, "row-reduce epilogue f32");
    }

    #[test]
    fn select_in_scan_pre_and_post_emits_ordered_f32() {
        use crate::ir::ReduceOp;
        // Scan PRE closure (emit_scan_impl pre): masked cumulative sum, pre =
        // select(in0 > 0.5, 7.5, 9.25), post = reduced(0).
        let pre = OpDef::scan(
            "mscanpre",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
            input(0)
                .binary(BinaryOp::CmpGt, konst(0.5))
                .select(konst(7.5), konst(9.25)),
            reduced(0),
        );
        let kp = generate(&pre, &scn_key(ElementKind::F32), &Cuda);
        assert_ordered_select_f32(&kp.source, "scan-pre f32");
        // Scan POST closure (emit_scan_impl post): pre = input(0), post =
        // select(reduced(0) > 0.5, 7.5, 9.25).
        let post = OpDef::scan(
            "mscanpost",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            false,
            false,
            input(0),
            reduced(0)
                .binary(BinaryOp::CmpGt, konst(0.5))
                .select(konst(7.5), konst(9.25)),
        );
        let ks = generate(&post, &scn_key(ElementKind::F32), &Cuda);
        assert_ordered_select_f32(&ks.source, "scan-post f32");
    }

    #[test]
    fn select_in_window_pre_and_post_emits_ordered_f32() {
        use crate::ir::ReduceOp;
        // Window PRE closure (emit_window pre): masked pooling, pre =
        // select(in0 > 0.5, 7.5, 9.25), post = reduced(0).
        let pre = OpDef::window(
            "mpoolpre",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            3,
            2,
            1,
            1,
            1,
            false,
            input(0)
                .binary(BinaryOp::CmpGt, konst(0.5))
                .select(konst(7.5), konst(9.25)),
            reduced(0),
        );
        let kp = generate(&pre, &win_key(ElementKind::F32, 128, 64), &Cuda);
        assert_ordered_select_f32(&kp.source, "window-pre f32");
        // Window POST closure (emit_window post): pre = input(0), post =
        // select(reduced(0) > 0.5, 7.5, 9.25).
        let post = OpDef::window(
            "mpoolpost",
            1,
            &[ElementKind::F32],
            ReduceOp::Sum,
            1,
            3,
            2,
            1,
            1,
            1,
            false,
            input(0),
            reduced(0)
                .binary(BinaryOp::CmpGt, konst(0.5))
                .select(konst(7.5), konst(9.25)),
        );
        let ks = generate(&post, &win_key(ElementKind::F32, 128, 64), &Cuda);
        assert_ordered_select_f32(&ks.source, "window-post f32");
    }

    #[test]
    fn select_in_multi_output_body_emits_ordered_f32_all_three_emitters() {
        // A select in a SECOND output body reaches all three multi-output
        // `cuda_select` sites (emit_scalar_multi / emit_strided_multi /
        // emit_vectorized_multi) — each a distinct closure that a site-local arm
        // swap could survive. body0 = in0 + in1, body1 = select(in2 > 0.5, 7.5,
        // 9.25); routed by key to Scalar, Strided, and V4 respectively.
        let op = || {
            OpDef::elementwise_multi(
                "mosel",
                3,
                &[ElementKind::F32],
                vec![
                    input(0) + input(1),
                    input(2)
                        .binary(BinaryOp::CmpGt, konst(0.5))
                        .select(konst(7.5), konst(9.25)),
                ],
            )
        };
        // Scalar-multi (odd extent defeats V4).
        let ks = generate(&op(), &mo_key(ElementKind::F32, 5), &Cuda);
        assert_eq!(ks.name, "baracuda_gen_mosel_f32_mo2_scalar");
        assert_ordered_select_f32(&ks.source, "multi-output scalar f32");
        // Strided-multi (transposed operands force Strided).
        let t = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::F32, 256);
        let strided_key = structure_key(
            OpCategory::BinaryElementwise,
            &std::iter::repeat_n(t, 5).collect::<Vec<_>>(),
            ArchSku::Sm89,
        );
        let kt = generate(&op(), &strided_key, &Cuda);
        assert!(kt.name.contains("_mo2_strided"), "{}", kt.name);
        assert_ordered_select_f32(&kt.source, "multi-output strided f32");
        // Vectorized-multi (contig, aligned, even extent ⇒ per-lane V4).
        let v = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let vec_key = structure_key(
            OpCategory::BinaryElementwise,
            &std::iter::repeat_n(v, 5).collect::<Vec<_>>(),
            ArchSku::Sm89,
        );
        let kv = generate(&op(), &vec_key, &Cuda);
        assert!(
            kv.name.contains("_v4"),
            "expected a V4 multi kernel: {}",
            kv.name
        );
        assert_ordered_select_f32(&kv.source, "multi-output vectorized f32");
    }

    // ---- G5: the Tier-2 emitter backstop, independent of the plan gate ----

    #[test]
    #[should_panic(expected = "Select at an integer dtype")]
    fn int_dtype_select_is_refused_by_the_emitter_backstop() {
        use crate::backend::Backend;
        use crate::plan::{KernelPlan, Schedule};
        // build_plan never produces this plan (G1 rejects int select) — but
        // the backstop must hold INDEPENDENTLY (the 0a lesson: gate every
        // layer): a hand-built int-dtype select plan handed straight to
        // `Cuda::lower` must panic in `assert_no_int_div_or_const`'s Select
        // arm, never emit C that happens to compile.
        let key = sel_key(ElementKind::I32, 4, 4);
        let body = input(0).select(input(1), input(2)).0;
        let plan = KernelPlan {
            op_name: "backstop",
            n_inputs: 3,
            dtype: ElementKind::I32,
            out_dtype: ElementKind::I32,
            schedule: Schedule::Scalar,
            key: &key,
            body: &body,
            n_outputs: 1,
            extra_out_bodies: &[],
            extra_out_dtypes: &[],
            access: &crate::ir::Access::Elementwise,
            views: &[],
            read_index: &[],
            write_index: &crate::ir::WriteIndex::Direct,
            base_offsets: &[],
            out_base_offset: crate::ir::BaseOffset::Zero,
        };
        let _ = Cuda.lower(&plan);
    }

    // ---- source dump for ondevice/select_validate.cu ----

    /// Regenerate the generated kernels `ondevice/select_validate.cu` includes.
    /// Run with:
    ///   `SELECT_OUT=<outdir> cargo test -p baracuda-kernelgen dump_select_sources -- --ignored --nocapture`
    #[test]
    #[ignore = "writes generated sources for the on-device select harness"]
    fn dump_select_sources() {
        let out = std::env::var("SELECT_OUT").expect("set SELECT_OUT=<outdir>");
        let write = |k: crate::GeneratedKernel| {
            let path = format!("{out}/{}.cu", k.name);
            std::fs::write(&path, &k.source).unwrap();
            println!("wrote {path}");
        };
        // rank-2 strided cells for the triu/tril family (1 input + output).
        let key_r2 = |dt: ElementKind| {
            let a = OperandDesc::new(2, &[128, 256], &[256, 1], dt, 256);
            structure_key(OpCategory::BinaryElementwise, &[a, a], ArchSku::Sm89)
        };
        // Cell 1: the acceptance gate — bit-identical triu/tril vs bespoke.
        // triu keeps j >= i + k (CmpGe); tril keeps j <= i + k (CmpLe).
        let masked = |name: &str, dt: ElementKind, cmp: BinaryOp, k: f64| {
            OpDef::elementwise(
                name,
                1,
                &[dt],
                coord(1)
                    .binary(cmp, coord(0) + konst(k))
                    .select(input(0), konst(0.0)),
            )
        };
        for dt in [ElementKind::F32, ElementKind::F64] {
            for (suffix, k) in [("", 0.0), ("_km1", -1.0), ("_k2", 2.0)] {
                write(generate(
                    &masked(&format!("triu_sel{suffix}"), dt, BinaryOp::CmpGe, k),
                    &key_r2(dt),
                    &Cuda,
                ));
                write(generate(
                    &masked(&format!("tril_sel{suffix}"), dt, BinaryOp::CmpLe, k),
                    &key_r2(dt),
                    &Cuda,
                ));
            }
        }
        // Cells 2/3: fused-cmp select (4 inputs: a, b, x, y) + a composed cond,
        // scalar rank-1 cells (align 4/8 defeats vectorization).
        let selcmp = |name: &str, dt: ElementKind| {
            OpDef::elementwise(
                name,
                4,
                &[dt],
                input(0)
                    .binary(BinaryOp::CmpGe, input(1))
                    .select(input(2), input(3)),
            )
        };
        write(generate(
            &selcmp("sel_cmp", ElementKind::F32),
            &sel_key(ElementKind::F32, 5, 4),
            &Cuda,
        ));
        write(generate(
            &selcmp("sel_cmp", ElementKind::F64),
            &sel_key(ElementKind::F64, 5, 8),
            &Cuda,
        ));
        // Composed cond: (a >= b) * (a != 0) — the float-side conjunction the
        // any-expression cond decision exists for.
        write(generate(
            &OpDef::elementwise(
                "sel_and",
                4,
                &[ElementKind::F32],
                (input(0).binary(BinaryOp::CmpGe, input(1))
                    * input(0).binary(BinaryOp::CmpNe, konst(0.0)))
                .select(input(2), input(3)),
            ),
            &sel_key(ElementKind::F32, 5, 4),
            &Cuda,
        ));
        // Cell 3's device-computed u8 mask: the 0b-validated pred kernel.
        {
            let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 4);
            let o = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::U8, 4);
            let key = structure_key(OpCategory::BinaryElementwise, &[a, a, o], ArchSku::Sm89);
            write(generate(
                &OpDef::elementwise_pred(
                    "mask_ge",
                    2,
                    &[ElementKind::F32],
                    input(0).binary(BinaryOp::CmpGe, input(1)),
                ),
                &key,
                &Cuda,
            ));
        }
        // Cell 4: f16/bf16 raw-pick (leaf arms; scalar cells).
        write(generate(
            &sel_op(ElementKind::F16),
            &sel_key(ElementKind::F16, 4, 2),
            &Cuda,
        ));
        write(generate(
            &sel_op(ElementKind::Bf16),
            &sel_key(ElementKind::Bf16, 4, 2),
            &Cuda,
        ));
        // Cell 5: vectorized-vs-scalar pair (same op, two cells).
        write(generate(
            &sel_op(ElementKind::F32),
            &sel_key(ElementKind::F32, 4, 256),
            &Cuda,
        ));
        write(generate(
            &sel_op(ElementKind::F32),
            &sel_key(ElementKind::F32, 4, 4),
            &Cuda,
        ));
    }
}
