//! CUDA C++ lowering — the one backend-specific module (v1).
//!
//! Everything CUDA-shaped (the vector types, `__global__`, the
//! `blockIdx`/`blockDim` launch indexing, the fp16/bf16 headers) lives here. The
//! math itself is lowered by the language-neutral [`crate::backend::lower_expr`]
//! — and reused verbatim across dtypes, because CUDA overloads `+ - * /` for
//! `__half` / `__nv_bfloat16` the same as for `float`.

use crate::backend::{
    lower_dag, lower_dag_all, lower_expr, Backend, GeneratedKernel, Lowering, Variant,
    VariantFidelity,
};
use crate::ir::{Access, BinaryOp, ExprDag, ReduceOp, ScalarExpr, UnaryOp};
use crate::plan::{rr_role, KernelPlan, ReduceAxisClass, RrRole, Schedule};
use baracuda_kernels_types::{Contiguity, ElementKind, OperandKey};

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
        scalar_ctype(dtype).is_some()
    }

    fn lower_variants(&self, plan: &KernelPlan<'_>) -> Vec<Variant> {
        // Schedule variants per the backlog
        // (docs/planning/foundational/11-variant-generators-backlog.md):
        // split-K for the outer-axis reduction cell; the materialized row cache
        // for RowReduce cells whose epilogue recomputes a stage's per-element
        // values (cross-pass materialization — Softmax's shared exp).
        let mut vs = Vec::new();
        vs.extend(reduction_splitk_variant(plan));
        vs.extend(row_reduce_materialize_variant(plan));
        vs.extend(contraction_splitk_variant(plan));
        vs
    }

    fn lower(&self, plan: &KernelPlan<'_>) -> GeneratedKernel {
        let Some(ctype) = scalar_ctype(plan.dtype) else {
            panic!("cuda backend: unsupported dtype {:?}", plan.dtype);
        };
        assert!(
            params_used(plan.body).is_empty()
                || matches!(plan.dtype, ElementKind::F32 | ElementKind::F32Strict),
            "cuda backend v1: scalar params are f32-only for now (dtype {:?})",
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
        if crate::plan::is_int_dtype(plan.dtype) {
            let mut exprs: Vec<&ScalarExpr> = vec![plan.body];
            match plan.access {
                Access::RowReduce { stages, epilogue } => {
                    exprs.extend(stages.iter().map(|s| &s.pre));
                    exprs.push(epilogue);
                }
                Access::Contraction { epilogue, .. } => exprs.push(epilogue),
                // The 0e reduction post-expr lowers at the accumulator dtype too.
                Access::Reduction { post, .. } => exprs.push(post),
                Access::Elementwise => {}
            }
            for e in exprs {
                assert_no_int_div_or_const(e, plan.dtype);
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
            let mut exprs: Vec<&ScalarExpr> = vec![plan.body];
            match plan.access {
                Access::RowReduce { stages, epilogue } => {
                    exprs.extend(stages.iter().map(|s| &s.pre));
                    exprs.push(epilogue);
                }
                Access::Contraction { epilogue, .. } => exprs.push(epilogue),
                Access::Reduction { post, .. } => exprs.push(post),
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
        assert!(
            plan.out_dtype == plan.dtype
                || matches!(
                    plan.schedule,
                    Schedule::Scalar | Schedule::Strided | Schedule::Reduction { .. }
                ),
            "cuda backend: hetero output (out {:?}, key {:?}) lowers scalar/strided/reduction only; got {:?}",
            plan.out_dtype,
            plan.dtype,
            plan.schedule
        );
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
            Schedule::Strided => emit_strided(plan, ctype),
            Schedule::Reduction { op, .. } => emit_reduction(plan, ctype, op),
            Schedule::RowReduce { .. } => emit_row_reduce(plan, ctype),
            Schedule::Contraction => emit_contraction(plan, ctype),
        }
    }
}

/// CUDA scalar type for a dtype, or `None` if the backend can't lower it yet.
/// `U8` (increment 0b) is the comparison-predicate mask dtype — `unsigned char`
/// per the FKC §5 Bool→U8 pinning — and, since increment 0c, an audited
/// COMPUTE dtype (wrapping mod-256 C semantics), same class as the i32/i64
/// arms. `S8` (FKC `I8`, increment 0c) is `signed char` — two's-complement
/// wrapping via integer promotion + store truncation (see the ir.rs table).
fn scalar_ctype(dt: ElementKind) -> Option<&'static str> {
    Some(match dt {
        ElementKind::F32 | ElementKind::F32Strict => "float",
        ElementKind::F64 => "double",
        ElementKind::F16 => "__half",
        ElementKind::Bf16 => "__nv_bfloat16",
        ElementKind::I32 => "int",
        ElementKind::I64 => "long long",
        ElementKind::S8 => "signed char",
        ElementKind::U8 => "unsigned char",
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
        Schedule::Vectorized { width } => {
            if vector_type(plan.dtype, width).is_some()
                || (packed_kind(plan.dtype, width).is_some() && body_packs(plan.body))
            {
                width
            } else {
                1
            }
        }
        _ => 1,
    }
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
        ScalarExpr::Const(_) | ScalarExpr::Param(_) | ScalarExpr::Reduced(_)
        | ScalarExpr::Coord(_) => false,
        ScalarExpr::Unary(_, x) => body_packs(x),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Div(a, b)
        | ScalarExpr::Binary(_, a, b) => body_packs(a) && body_packs(b),
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
fn dtype_tag(dt: ElementKind) -> &'static str {
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
    s.push_str(&format!("    long long nv{})\n{{\n", param_args(plan.body)));
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

/// The output pointer's scalar C type — the input `ctype` for a uniform-dtype
/// plan, `unsigned char` for a u8-predicate plan (increment 0b).
fn out_ctype<'c>(plan: &KernelPlan<'_>, ctype: &'c str) -> &'c str {
    if plan.out_dtype == plan.dtype {
        ctype
    } else {
        scalar_ctype(plan.out_dtype).expect("validated out dtype has a scalar ctype")
    }
}

/// The store expression for the lowered body root. Uniform-dtype plans store
/// the root unchanged (byte-identical to pre-0b output). A u8-predicate plan
/// converts the exact 0.0/1.0 predicate to `unsigned char` — exact by
/// construction (`assert_valid_out_dtype` pinned the body root to a `Cmp*`).
/// f16/bf16 re-promote first: the root lowered in the house promote-demote
/// convention is the demoted `__float2half(<pred_f32>)` (byte-identical math
/// to a nested cmp, one speller, no special root path), and 1.0/0.0 round-trip
/// f32→half→f32 bit-exactly, so the extra conversion pair is value-exact (and
/// folded by ptxas). The direct `(unsigned char)__half` conversion operator is
/// deliberately avoided — it is a header-configuration-dependent C++ overload,
/// not a house-audited intrinsic.
fn store_expr(plan: &KernelPlan<'_>, root: String) -> String {
    if plan.out_dtype == plan.dtype {
        return root;
    }
    match plan.dtype {
        ElementKind::F16 => format!("(unsigned char)__half2float({root})"),
        ElementKind::Bf16 => format!("(unsigned char)__bfloat162float({root})"),
        _ => format!("(unsigned char){root}"),
    }
}

fn emit_scalar(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let name = format!("baracuda_gen_{}_{}_scalar", plan.op_name, dtype_tag(plan.dtype));
    let n = plan.n_inputs;
    let octype = out_ctype(plan, ctype);
    let mut s = header(plan, &name);
    for i in 0..n {
        s.push_str(&format!("    const {ctype}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {octype}* __restrict__ out,\n"));
    s.push_str(&format!("    long long n{})\n{{\n", param_args(plan.body)));
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
        },
    );
    let store = store_expr(plan, root);
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
    let name = format!(
        "baracuda_gen_{}_{}_strided_r{}",
        plan.op_name,
        dtype_tag(plan.dtype),
        rank
    );
    let octype = out_ctype(plan, ctype);
    let mut s = header(plan, &name);
    for i in 0..n {
        s.push_str(&format!("    const {ctype}* __restrict__ in{i},\n"));
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
    s.push_str(&format!("    long long n{})\n{{\n", param_args(plan.body)));
    // Hoist fully-broadcast inputs: their offset is loop-invariant, load once.
    for k in 0..n {
        if is_fully_broadcast(plan.key.operands[k], rank) {
            s.push_str(&format!("    {ctype} h{k} = in{k}[0];\n"));
        }
    }
    s.push_str("    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long step = (long long)gridDim.x * blockDim.x;\n");
    s.push_str("    for (; i < n; i += step) {\n");
    s.push_str("        long long lin = i;\n");
    // Row-major unravel (last axis fastest), unrolled over the iteration rank.
    for d in (0..rank).rev() {
        s.push_str(&format!(
            "        long long c{d} = lin % shape{d}; lin /= shape{d};\n"
        ));
    }
    for k in 0..n {
        if !is_fully_broadcast(plan.key.operands[k], rank) {
            let off = offset_expr(plan.key.operands[k], &format!("s{k}"), rank);
            s.push_str(&format!("        long long o{k} = {off};\n"));
        }
    }
    let oo = offset_expr(plan.key.operands[n], "so", rank);
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
        },
    );
    for decl in &prelude {
        s.push_str(&format!("        {decl}\n"));
    }
    s.push_str(&format!("        out[oo] = {};\n", store_expr(plan, root)));
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
fn emit_reduction(plan: &KernelPlan<'_>, ctype: &str, rop: ReduceOp) -> GeneratedKernel {
    let tag = match rop {
        ReduceOp::Sum => "sum",
        ReduceOp::Mean => "mean",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        ReduceOp::Prod => "prod",
    };
    let int_acc = matches!(plan.dtype, ElementKind::I32 | ElementKind::I64);
    assert!(
        matches!(
            plan.dtype,
            ElementKind::F16
                | ElementKind::Bf16
                | ElementKind::F32
                | ElementKind::F32Strict
                | ElementKind::F64
        ) || int_acc,
        "reduction: float or i32/i64 dtypes only (i8/u8 etc. not yet); got {:?}",
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
    let octype = out_ctype(plan, ctype);

    // Accumulate in double for f64 / f32-strict; float otherwise. Shared by both
    // paths — the leaf load up-converts, and the body is lowered in the acc width.
    // Integer reductions accumulate in `long long` (exact, overflow-resistant);
    // f64 / f32-strict in double; everything else in float. The leaf load is native
    // for int (the `_` arm below) and up-converts only f16/bf16/f32-strict.
    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let acc = if int_acc {
        "long long"
    } else if dbl {
        "double"
    } else {
        "float"
    };
    let zero = if int_acc { "0" } else if dbl { "0.0" } else { "0.0f" };
    // Prod identity (increment 0e): `acc = 1; acc *= elem` (matches the bespoke
    // `ProdReduce::init() = T(1)`). Additive combines keep the `0` identity.
    let one = if int_acc { "1" } else if dbl { "1.0" } else { "1.0f" };
    let load = |i: u8| match plan.dtype {
        ElementKind::F16 => format!("__half2float(in{i}[idx])"),
        ElementKind::Bf16 => format!("__bfloat162float(in{i}[idx])"),
        ElementKind::F32Strict => format!("(double)in{i}[idx]"),
        _ => format!("in{i}[idx]"), // f32 (float) / f64 (double) load natively
    };
    let elem = lower_expr(
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
            unary: &|op, x| if dbl { unary_f64(op, x) } else { unary_f32(op, x) },
            binary: &|op, a, b| {
                if dbl {
                    binary_f64(op, a, b)
                } else {
                    binary_f32(op, a, b)
                }
            },
        },
    );
    // Convert an accumulator-width value to the output store. Uniform-dtype is
    // byte-identical to pre-0e (f16/bf16 demote, else native). A 0e hetero-out
    // converts the accumulator to the output dtype: U8 for a boolean any/all
    // (the value is exactly 0.0/1.0 by the Cmp* post, so the cast is exact) and
    // I64 for a count/sum-widening (exact for an int accumulator; exact for a
    // float accumulator while count ≤ 2²⁴ — the documented caller precondition).
    let store = |v: String| -> String {
        if plan.out_dtype != plan.dtype {
            return match plan.out_dtype {
                ElementKind::U8 => format!("(unsigned char)({v})"),
                ElementKind::I64 => format!("(long long)({v})"),
                other => unreachable!("validated hetero out dtype {other:?}"),
            };
        }
        match plan.dtype {
            ElementKind::F16 => format!("__float2half({v})"),
            ElementKind::Bf16 => format!("__float2bfloat16({v})"),
            _ => v,
        }
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
        let posted = lower_expr(
            post,
            &Lowering {
                leaf: &|i| {
                    panic!(
                        "cuda backend: reduction post-expr Input({i}) reached the emitter \
                         — the post reads Reduced(0)/Const/Param only (validated by \
                         plan::assert_valid_reduction_post)"
                    )
                },
                reduced: &|s| {
                    assert_eq!(s, 0, "reduction post references Reduced({s}); only 0 exists");
                    "red0".to_string()
                },
                coord: &|d| panic!("cuda backend: reduction post-expr Coord({d}) is Elementwise-only"),
                unary: &|op, x| if dbl { unary_f64(op, x) } else { unary_f32(op, x) },
                binary: &|op, a, b| if dbl { binary_f64(op, a, b) } else { binary_f32(op, a, b) },
            },
        );
        (Some(format!("{acc} red0 = {finalized};")), store(posted))
    };

    if class == ReduceAxisClass::InnerContig {
        // ---------- Contiguous last-axis reduction: ONE BLOCK per output row. ----------
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
            param_args_multi(&[plan.body, post])
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
                let cmp = if matches!(rop, ReduceOp::Max) { ">" } else { "<" };
                let suf = if matches!(rop, ReduceOp::Max) { "max" } else { "min" };
                // `has` carries "this lane saw an element" so idle lanes inject nothing
                // (no ±inf seed, headerless); a NaN sticks via `e != e`.
                s.push_str(&format!("        {acc} acc = {zero}; int has = 0;\n"));
                s.push_str("        for (long long j = threadIdx.x; j < k; j += blockDim.x) {\n");
                s.push_str("            long long idx = base + j;\n");
                s.push_str(&format!("            {acc} e = {elem};\n"));
                s.push_str(&format!(
                    "            if (!has || e != e || e {cmp} acc) {{ acc = e; has = 1; }}\n"
                ));
                s.push_str("        }\n");
                s.push_str(&format!("        {acc} r = block_{suf}_{stem}(acc, has);\n"));
            }
        }
        // The block_* helpers broadcast the result to all threads; thread 0 applies
        // the 0e post-expr (identity ⇒ byte-identical) and writes.
        let (decl, rhs) = post_apply("r".to_string());
        match decl {
            None => s.push_str(&format!("        if (threadIdx.x == 0) out[row] = {rhs};\n")),
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
    let reduced: Vec<usize> = (0..rank).filter(|&d| axes.is_set(d as u8)).collect();
    let kept: Vec<usize> = (0..rank).filter(|&d| !axes.is_set(d as u8)).collect();
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
        "baracuda_gen_{}_{}_reduce_{tag}_ax{:x}{}",
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
        param_args_multi(&[plan.body, post])
    ));
    s.push_str("    long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;\n");
    s.push_str("    long long gstride = (long long)gridDim.x * blockDim.x;\n");
    s.push_str("    for (; o < n_out; o += gstride) {\n");
    // Unravel the output linear index over the kept axes (row-major, last kept axis
    // fastest). Reduced axes never enter the output enumeration. (Reduce-all ⇒ no
    // kept axes ⇒ n_out == 1, so no unravel and no `lin` — avoids an unused var.)
    if !kept.is_empty() {
        s.push_str("        long long lin = o;\n");
        for &a in kept.iter().rev() {
            s.push_str(&format!(
                "        long long ck{a} = lin % shape{a}; lin /= shape{a};\n"
            ));
        }
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
                s.push_str(&format!("            long long roff{} = roff{};\n", i + 1, i));
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
            let cmp = if matches!(rop, ReduceOp::Max) { ">" } else { "<" };
            // `has` seeds the first reduced element (all cr=0) without a ±∞ literal;
            // an empty reduced extent leaves `acc = 0` (matching the fast path).
            s.push_str(&format!("        {acc} acc = {zero};\n"));
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
            unary: &|op, x| if dbl { unary_f64(op, x) } else { unary_f32(op, x) },
            binary: &|op, a, b| {
                if dbl {
                    binary_f64(op, a, b)
                } else {
                    binary_f32(op, a, b)
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
        if matches!(rop, ReduceOp::Mean) { "mean" } else { "sum" },
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
            GeneratedKernel { name: pname.clone(), source: p },
            GeneratedKernel { name: cname.clone(), source: k },
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
        baracuda_kernels_types::SizeClass::Tiny,
        "contraction v1 emits the Tiny-M skinny schedule only; larger M classes \
         are the tiled variant's territory (and all-Large routes to the vendor)"
    );
    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
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
    // Loads up-convert to the accumulator width, exactly as the reductions do.
    let load = |expr: String| match plan.dtype {
        ElementKind::F16 => format!("__half2float({expr})"),
        ElementKind::Bf16 => format!("__bfloat162float({expr})"),
        ElementKind::F32Strict => format!("(double){expr}"),
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
            leaf: &|i| unreachable!("contraction v1 epilogue has no Input leaf: in{i}"),
            reduced: &red,
            coord: &|d| {
                panic!(
                    "cuda backend: Coord({d}) reached the contraction emitter — the \
                     (m, n) epilogue space is not the elementwise coordinate space; \
                     Coord is Elementwise-only"
                )
            },
            unary: &|op, x| if dbl { unary_f64(op, x) } else { unary_f32(op, x) },
            binary: &|op, a, b| {
                if dbl {
                    binary_f64(op, a, b)
                } else {
                    binary_f32(op, a, b)
                }
            },
        },
    );
    let stored = match plan.dtype {
        ElementKind::F16 => format!("__float2half({epi})"),
        ElementKind::Bf16 => format!("__float2bfloat16({epi})"),
        _ => epi,
    };

    let mut s = header(plan, &name);
    s.push_str(&format!("    const {ctype}* __restrict__ in0,\n")); // lhs [m,k]
    s.push_str(&format!("    const {ctype}* __restrict__ in1,\n")); // rhs [k,n]
    s.push_str(&format!("    {ctype}* __restrict__ out,\n")); // [m,n]
    s.push_str("    long long m,\n    long long n,\n    long long k)\n{\n");
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
        load("in1[kk * n + col]".to_string())
    ));
    s.push_str("            #pragma unroll\n");
    s.push_str("            for (int mm = 0; mm < 8; ++mm) {\n");
    s.push_str(&format!(
        "                if (mm < m) accs[mm] += {} * w;\n",
        load("in0[mm * k + kk]".to_string())
    ));
    s.push_str("            }\n        }\n");
    s.push_str("        #pragma unroll\n");
    s.push_str("        for (int mm = 0; mm < 8; ++mm) {\n");
    s.push_str("            if (mm < m) {\n");
    s.push_str(&format!("                {acc} r0 = accs[mm];\n"));
    s.push_str(&format!("                out[mm * n + col] = {stored};\n"));
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
    // build_plan admissibility already ran; these mirror emit_contraction.
    let c = plan.key.contraction?;
    if c.m != baracuda_kernels_types::SizeClass::Tiny {
        return None;
    }
    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
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
        ElementKind::F32Strict => format!("(double){expr}"),
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
    p.push_str(&format!("    for (int mm = 0; mm < 8; ++mm) accs[mm] = {zero};\n"));
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
    p.push_str(
        "        if (mm < m) ws[((long long)blockIdx.y * m + mm) * n + col] = accs[mm];\n",
    );
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
            unary: &|op, x| if dbl { unary_f64(op, x) } else { unary_f32(op, x) },
            binary: &|op, a, b| {
                if dbl {
                    binary_f64(op, a, b)
                } else {
                    binary_f32(op, a, b)
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
    kk.push_str(&format!("            {acc} r0 = ws[(long long)mm * n + col];\n"));
    kk.push_str(
        "            for (long long ch = 1; ch < n_chunks; ++ch) r0 += ws[(ch * m + mm) * n + col];\n",
    );
    kk.push_str(&format!("            out[mm * n + col] = {stored};\n"));
    kk.push_str("        }\n    }\n}\n");

    Some(Variant {
        tag: "splitk",
        kernels: vec![
            GeneratedKernel { name: pname.clone(), source: p },
            GeneratedKernel { name: cname.clone(), source: kk },
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

/// One-letter tag for a [`baracuda_kernels_types::SizeClass`] in symbol names.
fn size_tag(s: baracuda_kernels_types::SizeClass) -> char {
    match s {
        baracuda_kernels_types::SizeClass::Tiny => 't',
        baracuda_kernels_types::SizeClass::Small => 's',
        baracuda_kernels_types::SizeClass::Mid => 'm',
        baracuda_kernels_types::SizeClass::Large => 'l',
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
    emit_row_reduce_impl(plan, ctype, false)
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
    let k = emit_row_reduce_impl(plan, ctype, true);
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
    }
}

fn emit_row_reduce_impl(plan: &KernelPlan<'_>, ctype: &str, materialize: bool) -> GeneratedKernel {
    let Access::RowReduce { stages, epilogue } = plan.access else {
        unreachable!("emit_row_reduce requires Access::RowReduce");
    };
    let dbl = matches!(plan.dtype, ElementKind::F64 | ElementKind::F32Strict);
    let acc = if dbl { "double" } else { "float" };
    let zero = if dbl { "0.0" } else { "0.0f" };
    // Prod's multiplicative identity — passed through to `emit_block_reducers`
    // for its signature; a RowReduce Prod stage is rejected at the plan gate, so
    // `ops` never contains Prod here and no block_prod is emitted.
    let one = if dbl { "1.0" } else { "1.0f" };
    let n = plan.n_inputs;
    let vsuf = if materialize { "_smemrow" } else { "" };
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
        if materialize { "_sm" } else { "" }
    );

    // load(i), up-converting f16/bf16/f32-strict to the accumulate type. The index
    // is role-aware (validate guarantees the roles): a row-streamed input (`x`) loads
    // `in_i[idx]` (idx = base+j); a per-column weight/bias loads `in_i[j]` (the same
    // value every row). Both `idx` and `j` are in scope in every stage fold + the
    // epilogue loop. (Validate forbids column inputs inside a stage `pre`, so stage
    // folds only ever see row-streamed loads — byte-identical to the single-input path.)
    let load = |i: u8| {
        let pos = match rr_role(plan.key.operands[i as usize]) {
            RrRole::RowStreamed => "idx",
            RrRole::ColBroadcast => "j",
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
                unary: &|op, x| if dbl { unary_f64(op, x) } else { unary_f32(op, x) },
                binary: &|op, a, b| {
                    if dbl {
                        binary_f64(op, a, b)
                    } else {
                        binary_f32(op, a, b)
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
    emit_block_reducers(&mut s, acc, zero, one, &ops, &stem);
    s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
    for i in 0..n {
        s.push_str(&format!("    const {ctype}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {ctype}* __restrict__ out,\n"));
    s.push_str(&format!(
        "    long long n_out,\n    long long k{})\n{{\n",
        param_args(plan.body)
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
    s.push_str("    for (long long row = blockIdx.x; row < n_out; row += (long long)gridDim.x) {\n");
    s.push_str("        long long base = row * k;\n");

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
                s.push_str("        for (long long j = threadIdx.x; j < k; j += blockDim.x) {\n");
                s.push_str("            long long idx = base + j;\n");
                if cache {
                    s.push_str(&format!("            {acc} v = {pre};\n"));
                    s.push_str("            baracuda_row_smem[j] = v;\n");
                    s.push_str(&format!("            acc{i} += v;\n"));
                } else {
                    s.push_str(&format!("            acc{i} += {pre};\n"));
                }
                s.push_str("        }\n");
                // block_sum broadcasts the row sum to every thread; Mean divides by
                // k (k>0 guaranteed by the guard above) — uniform in every thread.
                let fin = if matches!(st.op, ReduceOp::Mean) {
                    format!("block_sum_{stem}(acc{i}) / ({acc})k")
                } else {
                    format!("block_sum_{stem}(acc{i})")
                };
                s.push_str(&format!("        {acc} r{i} = {fin};\n"));
            }
            ReduceOp::Max | ReduceOp::Min => {
                let cmp = if matches!(st.op, ReduceOp::Max) { ">" } else { "<" };
                let suf = if matches!(st.op, ReduceOp::Max) { "max" } else { "min" };
                // Carry a `has` flag so idle / short-row lanes inject nothing and no
                // ±inf seed is needed (headerless); NaN sticks via `e != e`.
                s.push_str(&format!("        {acc} acc{i} = {zero}; int has{i} = 0;\n"));
                s.push_str("        for (long long j = threadIdx.x; j < k; j += blockDim.x) {\n");
                s.push_str("            long long idx = base + j;\n");
                s.push_str(&format!("            {acc} e = {pre};\n"));
                if cache {
                    s.push_str("            baracuda_row_smem[j] = e;\n");
                }
                s.push_str(&format!(
                    "            if (!has{i} || e != e || e {cmp} acc{i}) {{ acc{i} = e; has{i} = 1; }}\n"
                ));
                s.push_str("        }\n");
                s.push_str(&format!("        {acc} r{i} = block_{suf}_{stem}(acc{i}, has{i});\n"));
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
    s.push_str("        for (long long j = threadIdx.x; j < k; j += blockDim.x) {\n");
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
    if ops.iter().any(|o| matches!(o, ReduceOp::Sum | ReduceOp::Mean)) {
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

/// `true` if every iteration axis of `o` is a broadcast axis — its offset is
/// loop-invariant, so the load hoists out of the loop.
fn is_fully_broadcast(o: OperandKey, rank: usize) -> bool {
    rank > 0 && (0..rank).all(|d| o.bcast.is_set(d as u8))
}

/// Element-offset expression for an operand, dropping the terms for broadcast
/// axes (whose stride is known 0 at compile time).
fn offset_expr(o: OperandKey, stride_arr: &str, rank: usize) -> String {
    let mut terms = Vec::new();
    for d in 0..rank {
        if o.bcast.is_set(d as u8) {
            continue;
        }
        // By-value scalar param spelling (extraction #1): `s0_1`, `so_0`, …
        terms.push(format!("c{d}*{stride_arr}_{d}"));
    }
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join(" + ")
    }
}

/// Spell a [`UnaryOp`] applied to an already-lowered f32 inner expression.
/// Inner strings are atomic or parenthesized, so the function-call forms need no
/// extra wrapping; the operator forms wrap themselves. (`Sigmoid`/`Gelu`/`Silu`
/// reference the inner twice — fine for an atomic load; a temp-binding pass to
/// avoid recompute on compound inners is a follow-up.)
fn unary_f32(op: UnaryOp, x: String) -> String {
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
fn unary_f64(op: UnaryOp, x: String) -> String {
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
            format!("__float2half({})", unary_f32(op, format!("__half2float({x})")))
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
fn binary_f32(op: BinaryOp, a: String, b: String) -> String {
    match op {
        BinaryOp::Max => format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} > {b} ? {a} : {b})))"),
        BinaryOp::Min => format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} < {b} ? {a} : {b})))"),
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
fn binary_f64(op: BinaryOp, a: String, b: String) -> String {
    match op {
        BinaryOp::Max => format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} > {b} ? {a} : {b})))"),
        BinaryOp::Min => format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} < {b} ? {a} : {b})))"),
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
fn binary_int(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> String {
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
    if matches!(dtype, ElementKind::F16 | ElementKind::Bf16)
        && matches!(op, BinaryOp::Nextafter)
    {
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
            binary_f32(op, format!("__half2float({a})"), format!("__half2float({b})"))
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
fn assert_no_int_div_or_const(e: &ScalarExpr, dtype: ElementKind) {
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
        ScalarExpr::Unary(_, x) => assert_no_int_div_or_const(x, dtype),
        ScalarExpr::Add(a, b)
        | ScalarExpr::Sub(a, b)
        | ScalarExpr::Mul(a, b)
        | ScalarExpr::Binary(_, a, b) => {
            assert_no_int_div_or_const(a, dtype);
            assert_no_int_div_or_const(b, dtype);
        }
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
    }
}

/// The trailing `, float p0, float p1, …` kernel-signature suffix for the op's
/// runtime scalar params (empty when the op has none).
fn param_args(e: &ScalarExpr) -> String {
    params_used(e)
        .iter()
        .map(|i| format!(", float p{i}"))
        .collect()
}

/// Like [`param_args`], but over the UNION of params used across several
/// expressions (increment 0e: a reduction's launch signature must declare params
/// referenced by the fold body OR the post-expr). Deduped + ascending via the
/// `BTreeSet` in [`params_used`]. For a body-only op (identity post) this is
/// byte-identical to `param_args(body)`.
fn param_args_multi(exprs: &[&ScalarExpr]) -> String {
    let mut set = std::collections::BTreeSet::new();
    for e in exprs {
        set.extend(params_used(e));
    }
    set.iter().map(|i| format!(", float p{i}")).collect()
}

#[cfg(test)]
mod tests {
    use crate::ir::{input, param, OpDef};
    use crate::{generate, Cuda};
    use baracuda_kernels_types::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

    fn add_op(dtypes: &[ElementKind]) -> OpDef {
        OpDef::elementwise("add", 2, dtypes, input(0) + input(1))
    }

    fn binary_key(dt: ElementKind) -> baracuda_kernels_types::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 256);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    }

    #[test]
    fn f32_contiguous_vectorizes_to_float4() {
        let k = generate(&add_op(&[ElementKind::F32]), &binary_key(ElementKind::F32), &Cuda);
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
        let k = generate(&add_op(&[ElementKind::F64]), &binary_key(ElementKind::F64), &Cuda);
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
        let k = generate(&add_op(&[ElementKind::F16]), &binary_key(ElementKind::F16), &Cuda);
        assert_eq!(k.name, "baracuda_gen_add_f16_co_v8");
        assert!(k.source.contains("#include <cuda_fp16.h>"));
        assert!(k.source.contains(
            "struct __align__(16) baracuda_gen_add_f16_co_v8_vec { __half2 a, b, c, d; };"
        ));
        assert!(k.source.contains("const baracuda_gen_add_f16_co_v8_vec* __restrict__ in0"));
        for f in ["a", "b", "c", "d"] {
            assert!(k.source.contains(&format!("__half2 tmp0 = (v0.{f} + v1.{f});")));
            assert!(k.source.contains(&format!("vo.{f} = tmp0;")));
        }
    }

    #[test]
    fn bf16_v8_packs_with_bfloat162() {
        let k = generate(&add_op(&[ElementKind::Bf16]), &binary_key(ElementKind::Bf16), &Cuda);
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
        assert!(k
            .source
            .contains("struct __align__(4) baracuda_gen_add_f16_co_v2_vec { __half2 a; };"));
        assert!(!k.source.contains("vo.b"));
    }

    #[test]
    fn splitk_variant_offered_for_outer_sum() {
        use crate::ir::ReduceOp;
        use crate::{generate_variants, VariantFidelity};
        use baracuda_kernels_types::AxisMask;
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
        assert_eq!(vs.len(), 2, "base + splitk");
        assert_eq!(vs[0].tag, "base");
        assert_eq!(vs[0].fidelity, VariantFidelity::BitIdentical);
        let sk = &vs[1];
        assert_eq!(sk.tag, "splitk");
        assert_eq!(sk.fidelity, VariantFidelity::ReassociatedDeterministic);
        assert_eq!(sk.kernels.len(), 2, "partial + combine, in launch order");
        // Partial: coalesced (adjacent threads, adjacent columns), chunked rows,
        // one workspace row per chunk. No atomics anywhere.
        let p = &sk.kernels[0];
        assert!(p.name.ends_with("_splitk_partial"));
        assert!(p.source.contains("long long idx = r * cols + c;"));
        assert!(p.source.contains("long long r1 = r0 + chunk_rows; if (r1 > rows) r1 = rows;"));
        assert!(p.source.contains("ws[(long long)blockIdx.y * cols + c] = acc;"));
        assert!(!p.source.contains("atomic"));
        // Combine: fixed chunk-order fold, no Mean divisor for Sum.
        let c = &sk.kernels[1];
        assert!(c.name.ends_with("_splitk_combine"));
        // Seeded from the first partial (not 0): keeps a -0.0 column total's
        // sign and makes the degenerate n_chunks=1 case bit-identical.
        assert!(c.source.contains("float acc = ws[c];"));
        assert!(c.source.contains("for (long long k = 1; k < n_chunks; ++k) acc += ws[k * cols + c];"));
        assert!(!c.source.contains("/ (float)rows"));
        assert!(sk.launch_note.contains("chunk_rows = ceil(rows/n_chunks)"));
    }

    #[test]
    fn splitk_mean_divides_in_combine_only() {
        use crate::ir::ReduceOp;
        use crate::generate_variants;
        use baracuda_kernels_types::AxisMask;
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
        assert_eq!(vs.len(), 2);
        let sk = &vs[1];
        assert!(!sk.kernels[0].source.contains("acc /"), "partial never divides");
        assert!(sk.kernels[1].source.contains("out[c] = acc / (float)rows;"));
    }

    #[test]
    fn contraction_emits_skinny_simt_kernel() {
        use crate::ir::{reduced, ContractionAxes, UnaryOp};
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
        assert!(k.source.contains("if (mm < m) accs[mm] += in0[mm * k + kk] * w;"));
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
        assert!(kh.source.contains("float w = __half2float(in1[kk * n + col]);"));
        assert!(kh.source.contains("out[mm * n + col] = __float2half(r0);"));
    }

    #[test]
    fn contraction_splitk_variant_offered_for_tiny_m_cell() {
        use crate::ir::{reduced, ContractionAxes};
        use crate::{generate_variants, VariantFidelity};
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
        assert!(p.source.contains("long long k1 = k0 + chunk_k; if (k1 > k) k1 = k;"));
        assert!(p
            .source
            .contains("if (mm < m) ws[((long long)blockIdx.y * m + mm) * n + col] = accs[mm];"));
        assert!(!p.source.contains("atomic"));
        let c = &sk.kernels[1];
        assert!(c.name.ends_with("_contract_tll_splitk_combine"));
        // Seeded from chunk 0 → degenerate n_chunks=1 is bit-identical to base.
        assert!(c.source.contains("float r0 = ws[(long long)mm * n + col];"));
        assert!(c
            .source
            .contains("for (long long ch = 1; ch < n_chunks; ++ch) r0 += ws[(ch * m + mm) * n + col];"));
        assert!(c.source.contains("out[mm * n + col] = r0;"));
        assert!(sk.launch_note.contains("chunk_k = ceil(k/n_chunks)"));
    }

    #[test]
    fn smemrow_variant_materializes_softmax_shared_exp() {
        use crate::ir::{reduced, ReduceOp, ReduceStage};
        use crate::{generate_variants, VariantFidelity};
        // Softmax: stage-2 pre = exp(x - r0) is recomputed verbatim in the
        // epilogue — the cross-pass shared value the variant caches.
        let softmax = OpDef::row_reduce(
            "softmax",
            1,
            &[ElementKind::F32],
            vec![
                ReduceStage { pre: input(0).0, op: ReduceOp::Max },
                ReduceStage { pre: (input(0) - reduced(0)).exp().0, op: ReduceOp::Sum },
            ],
            (input(0) - reduced(0)).exp() / reduced(1),
        );
        let x = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Softmax, &[x, x], ArchSku::Sm89);
        let vs = generate_variants(&softmax, &key, &Cuda);
        assert_eq!(vs.len(), 2, "base + smemrow");
        let sm = &vs[1];
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
        assert_eq!(vs[0].kernels[0].source.matches("expf").count(), 2, "base has both");
        // Helper symbols must not collide when base + variant share a TU.
        assert!(src.contains("block_sum_softmax_f32_sm"));
        assert!(vs[0].kernels[0].source.contains("block_sum_softmax_f32("));
        // Sanity: the base kernel is untouched by the variant machinery.
        assert!(!vs[0].kernels[0].source.contains("baracuda_row_smem"));
    }

    #[test]
    fn smemrow_not_offered_when_epilogue_does_not_recompute() {
        use crate::ir::{konst, reduced, ReduceOp, ReduceStage, UnaryOp};
        use crate::generate_variants;
        // RmsNorm: stage pre = x², epilogue = x·rsqrt(r0+eps) — the epilogue
        // never recomputes x², so there is nothing to materialize.
        let rmsnorm = OpDef::row_reduce(
            "rmsnorm",
            1,
            &[ElementKind::F32],
            vec![ReduceStage { pre: input(0).unary(UnaryOp::Sqr).0, op: ReduceOp::Mean }],
            input(0) * (reduced(0) + konst(1e-5)).unary(UnaryOp::Rsqrt),
        );
        let x = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Normalization, &[x, x], ArchSku::Sm89);
        assert_eq!(generate_variants(&rmsnorm, &key, &Cuda).len(), 1, "base only");
    }

    #[test]
    fn splitk_gate_refuses_flipped_param_and_malformed_cells() {
        use crate::ir::ReduceOp;
        use crate::{generate_variants, Backend};
        use baracuda_kernels_types::AxisMask;
        let op = OpDef::reduction_axes(
            "sum", 1, &[ElementKind::F32], input(0), ReduceOp::Sum, AxisMask(0b1), false,
        );
        let o = OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256);
        // Row-reversed dense view: keys Contig + flipped — the baseline serves
        // it via runtime strides, but the stride-free split-K ABI would read
        // out of bounds. The gate must refuse.
        let rev = OperandDesc::new(2, &[4096, 1024], &[-1024, 1], ElementKind::F32, 256);
        let k_rev = structure_key(OpCategory::Reduction, &[rev, o], ArchSku::Sm89);
        assert_eq!(generate_variants(&op, &k_rev, &Cuda).len(), 1, "flipped input: base only");
        // Param body: the fixed splitk signature has no p{i} slot — refused
        // (the emitted source would reference an undefined identifier).
        let wsum = OpDef::reduction_axes(
            "wsum", 1, &[ElementKind::F32],
            input(0) * crate::ir::param(0),
            ReduceOp::Sum, AxisMask(0b1), false,
        );
        let a = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let k_ok = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        assert_eq!(generate_variants(&wsum, &k_ok, &Cuda).len(), 1, "param body: base only");
        // Malformed 1-operand key: out_key would alias the input — refused.
        // (Exercised via lower_variants directly; generate() itself would also
        // reject such a key downstream.)
        let k_one = structure_key(OpCategory::Reduction, &[a], ArchSku::Sm89);
        let plan = crate::build_plan(&op, &k_one);
        assert!(Cuda.lower_variants(&plan).is_empty(), "1-operand key: no variant");
    }

    #[test]
    fn splitk_not_offered_for_lastaxis_max_or_int() {
        use crate::ir::ReduceOp;
        use crate::generate_variants;
        use baracuda_kernels_types::AxisMask;
        // Last-axis (InnerContig): already block-parallel — no split-K.
        let last = OpDef::reduction("s", 1, &[ElementKind::F32], input(0), ReduceOp::Sum);
        let a = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let o = OperandDesc::new(1, &[4096], &[1], ElementKind::F32, 256);
        let k = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        assert_eq!(generate_variants(&last, &k, &Cuda).len(), 1, "base only");
        // Outer Max: the has-flag NaN fold needs its own variant treatment.
        let mx = OpDef::reduction_axes(
            "amax", 1, &[ElementKind::F32], input(0), ReduceOp::Max, AxisMask(0b1), false,
        );
        let ao = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F32, 256);
        let oo = OperandDesc::new(1, &[1024], &[1], ElementKind::F32, 256);
        let km = structure_key(OpCategory::Reduction, &[ao, oo], ArchSku::Sm89);
        assert_eq!(generate_variants(&mx, &km, &Cuda).len(), 1, "base only");
        // Outer i32 Sum: int workspace is a follow-up.
        let is = OpDef::reduction_axes(
            "sum", 1, &[ElementKind::I32], input(0), ReduceOp::Sum, AxisMask(0b1), false,
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
        assert!(k.source.contains("long long c1 = lin % shape1; lin /= shape1;"));
        assert!(k.source.contains("long long o0 = c0*s0_0 + c1*s0_1;"));
        assert!(k.source.contains("out[oo] = (in0[o0] + in1[o1]);"));
    }

    #[test]
    fn shared_interior_is_hoisted_to_one_tmp() {
        use crate::ir::konst;
        // g = a*b; out = g / (g + 1). The shared product must be emitted ONCE as a
        // named tmp and referenced twice — not re-rendered (the recompute + source
        // blow-up the DAG rewrite exists to kill).
        let g = input(0) * input(1);
        let op = OpDef::elementwise("diamond", 2, &[ElementKind::F32], g.clone() / (g + konst(1.0)));
        // A transposed (strided) key routes through emit_strided (plain float infix).
        let t = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::F32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[t, t, t], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(
            k.source.matches("in0[o0] * in1[o1]").count(),
            1,
            "the shared product is emitted exactly once"
        );
        assert!(k.source.contains("float tmp0 = (in0[o0] * in1[o1]);"), "hoisted to a tmp");
        assert!(
            k.source.contains("out[oo] = (tmp0 / (tmp0 + 1.0));"),
            "and referenced twice, not recomputed"
        );
    }

    #[test]
    fn single_use_body_emits_no_tmp() {
        // Transparency guard: a body with no shared interior must produce zero
        // `tmp` declarations (byte-identical to the pre-DAG emitter).
        let k = generate(&add_op(&[ElementKind::F32]), &binary_key(ElementKind::F32), &Cuda);
        assert!(!k.source.contains("tmp"), "no hoisting for a single-use tree");
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
        assert!(k
            .source
            .contains("vo.x = ((v0.x + v1.x) < 0.0f ? 0.0f : (v0.x + v1.x));"));
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

    fn reduce_key(in_dt: ElementKind) -> baracuda_kernels_types::StructureKey {
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
        assert!(k.source.contains("for (long long row = blockIdx.x; row < n_out;"));
        assert!(k.source.contains("for (long long j = threadIdx.x; j < k; j += blockDim.x)"));
        assert!(k.source.contains("acc += (in0[idx]*in0[idx]);"));
        assert!(k.source.contains("float r = block_sum_ms_f32(acc) / (float)k;"));
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
        assert!(k.source.contains("for (long long j = threadIdx.x; j < k; j += blockDim.x)"));
        // NaN-propagating select (e != e forces the swap), torch.amax semantics.
        assert!(k.source.contains("if (!has || e != e || e > acc) { acc = e; has = 1; }"));
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
        assert!(k.source.contains("if (threadIdx.x == 0) out[row] = __float2half(r);"));
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
        use baracuda_kernels_types::AxisMask;
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
        assert!(k.source.contains("long long ck1 = lin % shape1; lin /= shape1;"));
        assert!(k.source.contains("long long base = ck1*s0_1;"));
        // Innermost reduced walk: int32 counter when the extent fits, with a
        // long long fallback nest; strength-reduced offsets (extraction #2 + #3).
        assert!(k.source.contains("if (shape0 <= 2147483647LL)"));
        assert!(k.source.contains("int ext0 = (int)shape0;"));
        assert!(k.source.contains("for (int cr0 = 0; cr0 < ext0; ++cr0)"));
        assert!(k.source.contains("for (long long cr0 = 0; cr0 < shape0; ++cr0)")); // fallback
        assert!(k.source.contains("long long roff0 = base;")
            && k.source.contains("long long idx = roff0;")
            && k.source.contains("roff0 += s0_0;"));
        assert!(k.source.contains("long long oo = ck1*so_0;"));
        assert!(k.source.contains("acc += in0[idx];"));
        assert!(k.source.contains("out[oo] ="));
    }

    #[test]
    fn reduction_multi_axis_mean_divisor_is_the_extent_product() {
        use crate::ir::ReduceOp;
        use baracuda_kernels_types::AxisMask;
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
        assert!(k.source.contains("for (long long cr0 = 0; cr0 < shape0; ++cr0)")); // outer stays ll
        assert!(k.source.contains("for (int cr1 = 0; cr1 < ext1; ++cr1)")); // innermost int32
        assert!(k.source.contains("if (shape1 <= 2147483647LL)"));
        assert!(k.source.contains("long long roff1 = roff0;")
            && k.source.contains("long long idx = roff1;")
            && k.source.contains("roff1 += s0_1;"));
        assert!(k.source.contains("acc / (float)(shape0 * shape1)"));
        assert!(k.source.contains("long long base = ck2*s0_2;")); // kept axis 2
    }

    #[test]
    fn reduction_keepdim_outer_axis_uses_input_axis_output_stride() {
        use crate::ir::ReduceOp;
        use baracuda_kernels_types::AxisMask;
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
        use baracuda_kernels_types::AxisMask;
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
        assert!(k
            .source
            .contains("acc = has ? ((e != e || e > acc) ? e : acc) : e; has = 1;"));
    }

    #[test]
    fn reduction_strided_last_axis_takes_the_general_path() {
        use crate::ir::ReduceOp;
        use baracuda_kernels_types::AxisMask;
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
        use baracuda_kernels_types::AxisMask;
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
        assert!(k.source.contains("v *= __shfl_down_sync(0xffffffffu, v, off);"));
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
    fn reduction_prod_general_axis_folds_from_one() {
        use crate::ir::ReduceOp;
        use baracuda_kernels_types::AxisMask;
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
        use crate::ir::{reduced, ReduceOp, UnaryOp};
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
        use crate::ir::{reduced, ReduceOp, UnaryOp};
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
        assert!(k.source.contains("float r = block_sum_rms_f32(acc) / (float)k;"));
        assert!(k.source.contains("float red0 = r;")); // post sees the mean
        assert!(k.source.contains("out[row] = sqrtf(red0);"));
    }

    #[test]
    fn reduction_post_identity_is_byte_identical_to_plain() {
        use crate::ir::{reduced, ReduceOp};
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
    fn reduce_key_hetero(out_dt: ElementKind) -> baracuda_kernels_types::StructureKey {
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let out = OperandDesc::new(1, &[256], &[1], out_dt, 256);
        structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89)
    }

    #[test]
    fn reduction_hetero_out_u8_any() {
        use crate::ir::{konst, reduced, BinaryOp, ReduceOp};
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
        assert!(k.source.contains("out[row] = (unsigned char)(((float)red0 > (float)0.0 ? 1.0f : 0.0f));"));
    }

    #[test]
    fn reduction_hetero_out_i64_count() {
        use crate::ir::{konst, BinaryOp, ReduceOp};
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
        assert!(k.source.contains("if (threadIdx.x == 0) out[row] = (long long)(r);"));
    }

    // ---- 0e gate-rejection tests (call build_plan DIRECTLY: the emitter would
    // silently truncate a bad hetero store rather than panic, so the plan gate is
    // the load-bearing wall — exercise it without the emitter in the way). ----

    #[test]
    #[should_panic(expected = "must not read Input")]
    fn reduction_post_reading_input_is_rejected_at_the_plan_gate() {
        use crate::ir::{reduced, ReduceOp};
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
        use crate::ir::{reduced, ReduceOp};
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
        use crate::ir::{reduced, ReduceOp, ReduceStage};
        // Prod is a 0e Access::Reduction combiner only; the fused row path has no
        // block_prod, so a Prod stage misses honestly at the gate.
        let op = OpDef::row_reduce(
            "gp",
            1,
            &[ElementKind::F32],
            vec![ReduceStage { pre: input(0).0, op: ReduceOp::Prod }],
            reduced(0),
        );
        let _ = crate::build_plan(&op, &rr_key(ElementKind::F32, OpCategory::Softmax));
    }

    fn rr_key(dt: ElementKind, cat: OpCategory) -> baracuda_kernels_types::StructureKey {
        // full-width fused op: input + output share the [256, 128] contiguous shape.
        let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
        structure_key(cat, &[a, a], ArchSku::Sm89)
    }

    fn rmsnorm_op(dt: ElementKind) -> OpDef {
        use crate::ir::{konst, reduced, ReduceOp, ReduceStage, UnaryOp};
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
        use crate::ir::{reduced, ReduceOp, ReduceStage};
        // exp(x - rowmax) / sum(exp(x - rowmax)) — numerically stable.
        OpDef::row_reduce(
            "softmax",
            1,
            &[dt],
            vec![
                ReduceStage { pre: input(0).0, op: ReduceOp::Max },
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
        assert!(k
            .source
            .contains("for (long long row = blockIdx.x; row < n_out; row += (long long)gridDim.x)"));
        assert!(k
            .source
            .contains("for (long long j = threadIdx.x; j < k; j += blockDim.x)"));
        // mean of squares -> rsqrt; full-width output; uniform empty-axis guard.
        assert!(k.source.contains("acc0 += (in0[idx]*in0[idx]);"));
        assert!(k.source.contains("block_sum_rmsnorm_f32(acc0) / (float)k"));
        assert!(k.source.contains("out[idx] = (in0[idx] * rsqrtf((r0 + 1e-5)));"));
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
        assert!(k.source.contains("out[idx] = (expf((in0[idx] - r0)) / r1);"));
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
        assert!(k
            .source
            .contains("acc0 += (__half2float(in0[idx])*__half2float(in0[idx]));"));
        assert!(k.source.contains(
            "out[idx] = __float2half((__half2float(in0[idx]) * rsqrtf((r0 + 1e-5))));"
        ));
    }

    #[test]
    #[should_panic(expected = "references a stage not yet produced")]
    fn rowreduce_forward_reduced_ref_panics() {
        use crate::ir::{reduced, ReduceOp, ReduceStage};
        // A stage 0 `pre` referencing Reduced(0) (its own not-yet-produced result)
        // is a mis-authored op — validate_row_reduce must reject it at build_plan.
        let bad = OpDef::row_reduce(
            "bad",
            1,
            &[ElementKind::F32],
            vec![ReduceStage { pre: reduced(0).0, op: ReduceOp::Sum }],
            input(0) * reduced(0),
        );
        let _ = generate(&bad, &rr_key(ElementKind::F32, OpCategory::Normalization), &Cuda);
    }

    // --- multi-input RowReduce: weighted-RmsNorm + LayerNorm ---

    fn mi_key(dt: ElementKind, n_col: usize) -> baracuda_kernels_types::StructureKey {
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
        use crate::ir::{konst, reduced, ReduceOp, ReduceStage, UnaryOp};
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
        use crate::ir::{konst, reduced, ReduceOp, ReduceStage, UnaryOp};
        // (x-mean)*rsqrt(var+eps)*weight + bias; in0=x, in1=weight[k], in2=bias[k].
        OpDef::row_reduce(
            "layernorm",
            3,
            &[dt],
            vec![
                ReduceStage { pre: input(0).0, op: ReduceOp::Mean },
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
        let k = generate(&wrmsnorm_op(ElementKind::F32), &mi_key(ElementKind::F32, 1), &Cuda);
        assert!(k.source.contains("const float* __restrict__ in1,")); // weight operand
        // stage reduces only the row-streamed x (in0[idx]); column weight only in
        // the epilogue, indexed in1[j] (per-column) not in1[idx] (per-element).
        assert!(k.source.contains("acc0 += (in0[idx]*in0[idx]);"));
        assert!(k.source.contains("out[idx] = ((in0[idx] * rsqrtf((r0 + 1e-5))) * in1[j]);"));
        assert!(!k.source.contains("in1[idx]"));
    }

    #[test]
    fn rowreduce_layernorm_stable_two_pass_with_weight_bias() {
        let k = generate(&layernorm_op(ElementKind::F32), &mi_key(ElementKind::F32, 2), &Cuda);
        assert!(k.source.contains("const float* __restrict__ in2,")); // bias operand
        // stage 0 mean; stage 1 = mean of squared DEVIATIONS (the stable two-pass
        // var, not the cancellation-prone mean(x^2)-mean(x)^2).
        assert!(k.source.contains("acc0 += in0[idx];"));
        assert!(k.source.contains("acc1 += ((in0[idx] - r0)*(in0[idx] - r0));"));
        // epilogue: (x-mean)*rsqrt(var+eps)*weight[j] + bias[j].
        assert!(k.source.contains(
            "out[idx] = ((((in0[idx] - r0) * rsqrtf((r1 + 1e-5))) * in1[j]) + in2[j]);"
        ));
        assert!(!k.source.contains("in1[idx]") && !k.source.contains("in2[idx]"));
    }

    #[test]
    #[should_panic(expected = "per-column")]
    fn rowreduce_bare_rank1_weight_rejected() {
        // The must-fix OOB guard: a weight passed as a BARE rank-1 [k] tensor (not a
        // rank-aligned [n_out,k] broadcast view) has an empty bcast mask -> would
        // misclassify as a second row-streamed input and read in1[row*k+j] past its
        // k elements. validate must reject it loudly.
        let x = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let bare_w = OperandDesc::new(1, &[128], &[1], ElementKind::F32, 256); // bare [K]
        let out = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::Normalization, &[x, bare_w, out], ArchSku::Sm89);
        let _ = generate(&wrmsnorm_op(ElementKind::F32), &key, &Cuda);
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

    // =======================================================================
    // Increment-0a scalar-fn vocabulary — emission goldens + dtype gates
    // =======================================================================

    /// Scalar (unvectorized) unary cell: align defeats vectorization so the
    /// emitted body is the bare `out[i] = <fn>(in0[i]);` golden.
    fn unary_scalar_key(dt: ElementKind, align: u32) -> baracuda_kernels_types::StructureKey {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    }

    fn binary_scalar_key(dt: ElementKind, align: u32) -> baracuda_kernels_types::StructureKey {
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
            let op = |dt: ElementKind| {
                OpDef::elementwise("v", 1, &[dt], input(0).unary(uop))
            };
            let kf = generate(&op(ElementKind::F32), &unary_scalar_key(ElementKind::F32, 4), &Cuda);
            assert!(
                kf.source.contains(&format!("out[i] = {f32_fn}(in0[i]);")),
                "{uop:?} f32 golden missing in:\n{}",
                kf.source
            );
            let kd = generate(&op(ElementKind::F64), &unary_scalar_key(ElementKind::F64, 8), &Cuda);
            assert!(
                kd.source.contains(&format!("out[i] = {f64_fn}(in0[i]);")),
                "{uop:?} f64 golden missing in:\n{}",
                kd.source
            );
            // f16 scalar: the existing promote-to-f32 transcendental path.
            let kh = generate(&op(ElementKind::F16), &unary_scalar_key(ElementKind::F16, 2), &Cuda);
            assert!(
                kh.source
                    .contains(&format!("out[i] = __float2half({f32_fn}(__half2float(in0[i])));")),
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
            let op = |dt: ElementKind| {
                OpDef::elementwise("v", 2, &[dt], input(0).binary(bop, input(1)))
            };
            let kf = generate(&op(ElementKind::F32), &binary_scalar_key(ElementKind::F32, 4), &Cuda);
            assert!(
                kf.source.contains(&format!("out[i] = {f32_fn}(in0[i], in1[i]);")),
                "{bop:?} f32 golden missing in:\n{}",
                kf.source
            );
            let kd = generate(&op(ElementKind::F64), &binary_scalar_key(ElementKind::F64, 8), &Cuda);
            assert!(
                kd.source.contains(&format!("out[i] = {f64_fn}(in0[i], in1[i]);")),
                "{bop:?} f64 golden missing in:\n{}",
                kd.source
            );
            if half_ok {
                let kh =
                    generate(&op(ElementKind::F16), &binary_scalar_key(ElementKind::F16, 2), &Cuda);
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
        assert!(!km.source.contains("fmaxf"), "Maximum must NOT become fmaxf");
        assert!(km.source.contains("!="), "Maximum keeps the NaN-propagating select");
        // Min side — BOTH directions (mutation-caught gap: Min silently
        // becoming fminf passed the suite when only FminIeee was pinned).
        let ieee_min = OpDef::elementwise(
            "fmini",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::FminIeee, input(1)),
        );
        assert!(generate(&ieee_min, &key, &Cuda).source.contains("fminf(in0[i], in1[i])"));
        let min = OpDef::elementwise("mn", 2, &[ElementKind::F32], input(0).min(input(1)));
        let kn = generate(&min, &key, &Cuda);
        assert!(!kn.source.contains("fminf"), "Minimum must NOT become fminf");
        assert!(kn.source.contains("!="), "Minimum keeps the NaN-propagating select");
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
        assert!(!kt.source.contains("floorf"), "RemTrunc must not take the floored form");
        let rem = OpDef::elementwise(
            "rem",
            2,
            &[ElementKind::F32],
            input(0).binary(BinaryOp::Rem, input(1)),
        );
        let kr = generate(&rem, &key, &Cuda);
        assert!(kr.source.contains("(in0[i] - floorf(in0[i] / in1[i]) * in1[i])"));
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
        assert!(!kt64.source.contains("floor("), "f64 RemTrunc must not take the floored form");
        let rem64 = OpDef::elementwise(
            "rem64",
            2,
            &[ElementKind::F64],
            input(0).binary(BinaryOp::Rem, input(1)),
        );
        let kr64 = generate(&rem64, &key64, &Cuda);
        assert!(kr64.source.contains("(in0[i] - floor(in0[i] / in1[i]) * in1[i])"));
        assert!(!kr64.source.contains("fmod("), "f64 Rem must not become fmod");
        let max64 = OpDef::elementwise("mx64", 2, &[ElementKind::F64], input(0).max(input(1)));
        let km64 = generate(&max64, &key64, &Cuda);
        assert!(!km64.source.contains("fmax("), "f64 Maximum must NOT become fmax");
        assert!(km64.source.contains("!="), "f64 Maximum keeps the NaN-propagating select");
        let min64 = OpDef::elementwise("mn64", 2, &[ElementKind::F64], input(0).min(input(1)));
        let kn64 = generate(&min64, &key64, &Cuda);
        assert!(!kn64.source.contains("fmin("), "f64 Minimum must NOT become fmin");
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
        use crate::ir::{konst, BinaryOp, ReduceOp};
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
        use crate::ir::{reduced, BinaryOp, ReduceOp, ReduceStage};
        let op = OpDef::row_reduce(
            "na_rr",
            1,
            &[ElementKind::F16],
            vec![ReduceStage { pre: input(0).0, op: ReduceOp::Max }],
            input(0).binary(BinaryOp::Nextafter, reduced(0)),
        );
        let x = OperandDesc::new(2, &[4096, 1024], &[1024, 1], ElementKind::F16, 256);
        let key = structure_key(OpCategory::Softmax, &[x, x], ArchSku::Sm89);
        let _ = generate(&op, &key, &Cuda);
    }

    #[test]
    #[should_panic(expected = "must miss honestly")]
    fn nextafter_f16_is_refused_in_a_contraction_epilogue() {
        use crate::ir::{konst, reduced, BinaryOp, ContractionAxes};
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
        let op = OpDef::elementwise("erfx", 1, &[ElementKind::F16], input(0).unary(UnaryOp::Erfc));
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

    use crate::ir::{konst, BinaryOp};

    /// Fully-aligned binary cell with a **U8 output** operand (inputs `dt`):
    /// the caller-side key shape of an `elementwise_pred` op. Note the aligned
    /// contiguous u8 output keys V8 on its own — the plan must still force the
    /// scalar path (no packed u8 store exists).
    fn pred_key(dt: ElementKind) -> baracuda_kernels_types::StructureKey {
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
            (BinaryOp::CmpEq, "cmp_eq", "((float)in0[i] == (float)in1[i] ? 1.0f : 0.0f)"),
            (BinaryOp::CmpNe, "cmp_ne", "((float)in0[i] != (float)in1[i] ? 1.0f : 0.0f)"),
            (BinaryOp::CmpLt, "cmp_lt", "((float)in0[i] < (float)in1[i] ? 1.0f : 0.0f)"),
            (BinaryOp::CmpLe, "cmp_le", "((float)in0[i] <= (float)in1[i] ? 1.0f : 0.0f)"),
            (BinaryOp::CmpGt, "cmp_gt", "((float)in0[i] > (float)in1[i] ? 1.0f : 0.0f)"),
            (BinaryOp::CmpGe, "cmp_ge", "((float)in0[i] >= (float)in1[i] ? 1.0f : 0.0f)"),
        ];
        for (op, name, ternary) in cases {
            let k = generate(&pred_op(name, *op, ElementKind::F32), &pred_key(ElementKind::F32), &Cuda);
            // The op name flows into the entry point (identity is (token, entry_point)).
            assert_eq!(k.name, format!("baracuda_gen_{name}_f32_scalar"));
            // Inputs stay the key dtype; ONLY the output is u8.
            assert!(k.source.contains("const float* __restrict__ in0"), "{name}");
            assert!(k.source.contains("unsigned char* __restrict__ out"), "{name}");
            // Exact predicate + exact store conversion (0.0f/1.0f -> 0/1).
            let store = format!("out[i] = (unsigned char){ternary};");
            assert!(k.source.contains(&store), "{name}: missing `{store}` in:\n{}", k.source);
        }
    }

    #[test]
    fn cmp_f64_golden_uses_double_literals() {
        let k = generate(&pred_op("cmp_lt", BinaryOp::CmpLt, ElementKind::F64), &pred_key(ElementKind::F64), &Cuda);
        assert_eq!(k.name, "baracuda_gen_cmp_lt_f64_scalar");
        assert!(k.source.contains("out[i] = (unsigned char)(in0[i] < in1[i] ? 1.0 : 0.0);"));
        assert!(!k.source.contains("1.0f"), "f64 predicate must not use float literals");
    }

    #[test]
    fn cmp_f16_promotes_to_f32_for_the_compare() {
        // f16 compares via promote-to-f32 — EXACT (half->f32 is a lossless,
        // order-preserving embedding, so the f32 compare decides identically to
        // a native half compare), and the store re-promotes the demoted 1.0/0.0
        // before the integer cast (exact round trip; see store_expr).
        let k = generate(&pred_op("cmp_lt", BinaryOp::CmpLt, ElementKind::F16), &pred_key(ElementKind::F16), &Cuda);
        assert_eq!(k.name, "baracuda_gen_cmp_lt_f16_scalar");
        assert!(k.source.contains("#include <cuda_fp16.h>"));
        assert!(k.source.contains("((float)__half2float(in0[i]) < (float)__half2float(in1[i]) ? 1.0f : 0.0f)"));
        assert!(k.source.contains("out[i] = (unsigned char)__half2float(__float2half("));
        assert!(k.source.contains("unsigned char* __restrict__ out"));
    }

    #[test]
    fn cmp_u8_out_falls_back_to_scalar_at_aligned_vector_cells() {
        // Packed-classifier fallback golden: a u8-output op at a fully-aligned
        // f32 (V4) or f16 (V8) cell must take the SCALAR path — no float4, no
        // packed half2 struct, no invented packed u8 store. (The aligned
        // contiguous u8 output operand itself keys V8, so without the plan
        // gate the min-width rule alone would still say "vectorize".)
        let kf = generate(&pred_op("cmp_ge", BinaryOp::CmpGe, ElementKind::F32), &pred_key(ElementKind::F32), &Cuda);
        assert!(kf.name.ends_with("_scalar"), "got {}", kf.name);
        assert!(!kf.source.contains("float4"));
        let kh = generate(&pred_op("cmp_eq", BinaryOp::CmpEq, ElementKind::F16), &pred_key(ElementKind::F16), &Cuda);
        assert!(kh.name.ends_with("_scalar"), "got {}", kh.name);
        // `__half2 ` (the pair TYPE, trailing space) must not appear — the
        // scalar promote fn `__half2float` legitimately does.
        assert!(!kh.source.contains("__half2 "), "no packed pair type: {}", kh.name);
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
            access: &crate::ir::Access::Elementwise,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    fn cmp_bf16_u8_store_bridges_through_bfloat162float() {
        // The direct __nv_bfloat16→integer conversion is a header-configuration-
        // dependent C++ overload (a headerless-nvrtc hazard) — the store must
        // bridge through __bfloat162float. (Review-caught gap: the Bf16 arm of
        // store_expr had zero coverage; only the F16 form was pinned.)
        let k = generate(&pred_op("cmp_lt", BinaryOp::CmpLt, ElementKind::Bf16), &pred_key(ElementKind::Bf16), &Cuda);
        assert_eq!(k.name, "baracuda_gen_cmp_lt_bf16_scalar");
        assert!(
            k.source.contains("out[i] = (unsigned char)__bfloat162float("),
            "bf16 u8 store must bridge through __bfloat162float:\n{}",
            k.source
        );
    }

    #[test]
    fn int_compute_dtypes_are_supported_after_the_0c_audit() {
        use crate::backend::Backend;
        // Increment 0c replaced the 0b uniform-u8 hold: U8 and S8 are audited
        // COMPUTE dtypes now (wrapping semantics + the int-only op set), so
        // supports_dtype says yes — and the per-OP legality lives in
        // dtype_compatible / the plan gate, NOT here (a uniform-U8 Div region
        // still declines; pinned in jit.rs). Both directions:
        for dt in [ElementKind::U8, ElementKind::S8, ElementKind::I32, ElementKind::I64] {
            assert!(Cuda.supports_dtype(dt), "{dt:?} is an audited compute dtype");
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
        let k = generate(&pred_op("cmp_gt", BinaryOp::CmpGt, ElementKind::F32), &pred_key(ElementKind::F32), &Cuda);
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
        let k = generate(&pred_op("cmp_ne", BinaryOp::CmpNe, ElementKind::F32), &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_cmp_ne_f32_strided_r2");
        assert!(k.source.contains("unsigned char* __restrict__ out"));
        assert!(k.source.contains("out[oo] = (unsigned char)((float)in0[o0] != (float)in1[o1] ? 1.0f : 0.0f);"));
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
        assert!(k.source.contains("out[i] = (in0[i] * ((float)in1[i] > (float)0.0 ? 1.0f : 0.0f));"));
        assert!(k.source.contains("float* __restrict__ out"));
        assert!(!k.source.contains("unsigned char"));
        // …and at a fully-aligned cell the SAME body vectorizes normally
        // (out_dtype None leaves the schedule untouched).
        let kv = generate(&op, &binary_scalar_key(ElementKind::F32, 256), &Cuda);
        assert_eq!(kv.name, "baracuda_gen_relu_bw_f32_co_v4");
        assert!(kv.source.contains("((float)v1.x > (float)0.0 ? 1.0f : 0.0f)"));
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
        assert!(k.source.contains("out[i] = ((float)in0[i] > (float)in1[i] ? 1.0f : 0.0f);"));
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
        use crate::ir::{reduced, ContractionAxes};
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
            assert!(ki.source.contains(&store), "{bop:?} i32: missing `{store}` in:\n{}", ki.source);
            let kl = generate(
                &int_op(name, bop, ElementKind::I64),
                &binary_scalar_key(ElementKind::I64, 8),
                &Cuda,
            );
            assert_eq!(kl.name, format!("baracuda_gen_{name}_i64_scalar"));
            assert!(kl.source.contains("const long long* __restrict__ in0"), "{bop:?}");
            assert!(kl.source.contains(&store), "{bop:?} i64: missing `{store}`");
            // Header-light: int kernels need no includes (nvrtc headerless).
            assert!(!ki.source.contains("#include"), "{bop:?} i32 must be headerless");
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
        assert!(ku.source.contains("out[i] = (in0[i] << in1[i]);"), "{}", ku.source);
        assert!(!ku.source.contains("(int)"), "no defeating casts — promotion is the contract");
        let ks = generate(
            &int_op("shr", BinaryOp::Shr, ElementKind::S8),
            &binary_scalar_key(ElementKind::S8, 1),
            &Cuda,
        );
        assert_eq!(ks.name, "baracuda_gen_shr_i8_scalar");
        assert!(ks.source.contains("const signed char* __restrict__ in0"));
        assert!(ks.source.contains("out[i] = (in0[i] >> in1[i]);"), "{}", ks.source);
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
            let k = generate(&int_op(name, bop, ElementKind::U8), &binary_scalar_key(ElementKind::U8, 1), &Cuda);
            assert_eq!(k.name, format!("baracuda_gen_{name}_u8_scalar"));
            assert!(k.source.contains("const unsigned char* __restrict__ in0"), "{bop:?}");
            assert!(k.source.contains(store), "{bop:?}: missing `{store}` in:\n{}", k.source);
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
        assert!(!ku.source.contains("float"), "no float detour in a u8 kernel");
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
        assert_eq!(ki.name, "baracuda_gen_band_i32_scalar", "aligned i32 stays scalar");
        assert!(!ki.source.contains("int4"), "no invented int vectorization");
        let addu = OpDef::elementwise("add", 2, &[ElementKind::U8], input(0) + input(1));
        let ku = generate(&addu, &binary_scalar_key(ElementKind::U8, 256), &Cuda);
        assert_eq!(ku.name, "baracuda_gen_add_u8_scalar", "aligned u8 stays scalar");
        assert!(!ku.source.contains("_vec"), "no packed vector struct");
    }

    #[test]
    fn int_strided_cell_unravels_with_the_same_operator() {
        // Transposed i32 views route to the strided emitter with the same raw
        // C operator at the unraveled offsets.
        let t = OperandDesc::new(2, &[8, 4], &[1, 8], ElementKind::I32, 256);
        let d = OperandDesc::new(2, &[8, 4], &[4, 1], ElementKind::I32, 256);
        let key = structure_key(OpCategory::BinaryElementwise, &[t, d, d], ArchSku::Sm89);
        let k = generate(&int_op("bxor", BinaryOp::BitXor, ElementKind::I32), &key, &Cuda);
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
            access: &crate::ir::Access::Elementwise,
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
            access: &crate::ir::Access::Elementwise,
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
            access: &crate::ir::Access::Elementwise,
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
            access: &crate::ir::Access::Elementwise,
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
            access: &crate::ir::Access::Elementwise,
        };
        let _ = Cuda.lower(&plan);
    }

    // ===== increment 0d: Coord(axis) — the iota/coordinate leaf =============

    use crate::ir::coord;

    /// Rank-2 contiguous, fully aligned cell — the shape whose ALIGNED inputs
    /// would normally vectorize (V4 at f32): exactly the cell the Coord
    /// routing must force onto Strided.
    fn coord_key_2d(dt: ElementKind, n_operands: usize) -> baracuda_kernels_types::StructureKey {
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
        let k = generate(&triu_mask_op("triu_mask", ElementKind::F32, 0.0), &coord_key_2d(ElementKind::F32, 2), &Cuda);
        assert_eq!(k.name, "baracuda_gen_triu_mask_f32_strided_r2");
        // Unravel present despite all-contiguous operands (last axis fastest).
        assert!(k.source.contains("long long c1 = lin % shape1; lin /= shape1;"));
        assert!(k.source.contains("long long c0 = lin % shape0; lin /= shape0;"));
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
        assert!(k.source.contains("long long c1 = lin % shape1; lin /= shape1;"));
        assert!(k.source.contains("long long oo = c0*so_0 + c1*so_1;"));
        assert!(k.source.contains("out[oo] = (float)c1;"), "{}", k.source);
        assert!(!k.source.contains("in0"), "a 0-input op has no input pointers");
    }

    #[test]
    fn coord_f64_golden_uses_double_cast_and_literals() {
        let k = generate(&triu_mask_op("triu_mask", ElementKind::F64, 0.0), &coord_key_2d(ElementKind::F64, 2), &Cuda);
        assert_eq!(k.name, "baracuda_gen_triu_mask_f64_strided_r2");
        assert!(
            k.source.contains(
                "out[oo] = (in0[o0] * ((double)c1 >= ((double)c0 + 0.0) ? 1.0 : 0.0));"
            ),
            "f64 triu-mask store golden missing in:\n{}",
            k.source
        );
        assert!(!k.source.contains("(float)"), "no float casts in the f64 kernel");
    }

    #[test]
    fn coord_alibi_body_with_launch_param_golden() {
        // (coord(1) - coord(0)) * param(0) — the alibi relative-position shape
        // with a runtime slope: zero tensor inputs, one f32 launch param.
        let op = OpDef::elementwise("alibi", 0, &[ElementKind::F32], (coord(1) - coord(0)) * param(0));
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
        let key = structure_key(OpCategory::UnaryElementwise, &[a], ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        assert_eq!(k.name, "baracuda_gen_alibi_f32_strided_r2");
        assert!(k.source.contains("long long n, float p0)"), "p0 rides the launch signature");
        assert!(
            k.source.contains("out[oo] = (((float)c1 - (float)c0) * p0);"),
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
        use crate::ir::{reduced, ReduceOp, ReduceStage};
        // The RowReduce epilogue iterates the (row, j) space, not the
        // elementwise output coordinate space.
        let op = OpDef::row_reduce(
            "rr_coord",
            1,
            &[ElementKind::F32],
            vec![ReduceStage { pre: input(0).0, op: ReduceOp::Sum }],
            reduced(0) + coord(0),
        );
        let key = rr_key(ElementKind::F32, OpCategory::Normalization);
        let _ = crate::build_plan(&op, &key);
    }

    #[test]
    #[should_panic(expected = "Elementwise-only in 0d")]
    fn coord_in_a_contraction_epilogue_is_rejected_at_the_plan_gate() {
        use crate::ir::{reduced, ContractionAxes};
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
            access: &crate::ir::Access::Elementwise,
        };
        let _ = Cuda.lower(&plan);
    }

    #[test]
    #[should_panic(expected = "under a non-Elementwise access")]
    fn coord_under_reduction_access_is_refused_by_the_emitter_backstop() {
        use crate::backend::Backend;
        use crate::ir::ReduceOp;
        use crate::plan::{KernelPlan, ReduceAxisClass, Schedule};
        use baracuda_kernels_types::AxisMask;
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
            access: &access,
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
            access: &crate::ir::Access::Elementwise,
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
            access: &crate::ir::Access::Elementwise,
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
}
