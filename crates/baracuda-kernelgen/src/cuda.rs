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
fn scalar_ctype(dt: ElementKind) -> Option<&'static str> {
    Some(match dt {
        ElementKind::F32 | ElementKind::F32Strict => "float",
        ElementKind::F64 => "double",
        ElementKind::F16 => "__half",
        ElementKind::Bf16 => "__nv_bfloat16",
        ElementKind::I32 => "int",
        ElementKind::I64 => "long long",
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
        ScalarExpr::Const(_) | ScalarExpr::Param(_) | ScalarExpr::Reduced(_) => false,
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

/// Packed binary speller — all four function ops are **Tier B** (pair-split
/// through the existing scalar speller): `__hmax2`/`__hmin2` are IEEE maxNum
/// (NaN-*suppressing*), which would break the house NaN-propagating Max/Min
/// convention, and `Pow`/`Rem` have no packed intrinsic.
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

fn emit_scalar(plan: &KernelPlan<'_>, ctype: &str) -> GeneratedKernel {
    let name = format!("baracuda_gen_{}_{}_scalar", plan.op_name, dtype_tag(plan.dtype));
    let n = plan.n_inputs;
    let mut s = header(plan, &name);
    for i in 0..n {
        s.push_str(&format!("    const {ctype}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {ctype}* __restrict__ out,\n"));
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
            unary: &|op, x| cuda_unary(op, x, plan.dtype),
            binary: &|op, a, b| cuda_binary(op, a, b, plan.dtype),
        },
    );
    if prelude.is_empty() {
        s.push_str(&format!("    for (; i < n; i += step) out[i] = {root};\n"));
    } else {
        // Shared interiors: hoist the `tmp` block inside the loop (its RHS reads
        // the per-`i` inputs), so a shared value is computed once per element.
        s.push_str("    for (; i < n; i += step) {\n");
        for decl in &prelude {
            s.push_str(&format!("        {decl}\n"));
        }
        s.push_str(&format!("        out[i] = {root};\n    }}\n"));
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
    let mut s = header(plan, &name);
    for i in 0..n {
        s.push_str(&format!("    const {ctype}* __restrict__ in{i},\n"));
    }
    s.push_str(&format!("    {ctype}* __restrict__ out,\n"));
    s.push_str("    const long long* __restrict__ shape,\n");
    for i in 0..n {
        s.push_str(&format!("    const long long* __restrict__ s{i},\n"));
    }
    s.push_str("    const long long* __restrict__ so,\n");
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
            "        long long c{d} = lin % shape[{d}]; lin /= shape[{d}];\n"
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
    let (prelude, root) = lower_dag(
        &ExprDag::from_expr(plan.body),
        ctype,
        &Lowering {
            leaf: &acc,
            reduced: &|i| unreachable!("no Reduced leaf outside RowReduce: red{i}"),
            unary: &|op, x| cuda_unary(op, x, plan.dtype),
            binary: &|op, a, b| cuda_binary(op, a, b, plan.dtype),
        },
    );
    for decl in &prelude {
        s.push_str(&format!("        {decl}\n"));
    }
    s.push_str(&format!("        out[oo] = {root};\n"));
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
    let (axes, keepdim) = match plan.access {
        Access::Reduction { axes, keepdim, .. } => (*axes, *keepdim),
        _ => unreachable!("emit_reduction on a non-reduction access"),
    };

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
        emit_block_reducers(&mut s, acc, zero, &ops, &stem);
        s.push_str(&format!("extern \"C\" __global__ void {name}(\n"));
        s.push_str(&format!("    const {ctype}* __restrict__ in0,\n"));
        s.push_str(&format!("    {ctype}* __restrict__ out,\n"));
        s.push_str(&format!(
            "    long long n_out,\n    long long k{})\n{{\n",
            param_args(plan.body)
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
        // The block_* helpers broadcast the result to all threads; one writes it.
        s.push_str(&format!(
            "        if (threadIdx.x == 0) out[row] = {};\n",
            store("r".to_string())
        ));
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
    s.push_str(&format!("    {ctype}* __restrict__ out,\n"));
    s.push_str("    const long long* __restrict__ shape,\n"); // per-input-axis extents
    s.push_str("    const long long* __restrict__ s0,\n"); // input strides
    s.push_str("    const long long* __restrict__ so,\n"); // output strides
    s.push_str(&format!("    long long n_out{})\n{{\n", param_args(plan.body)));
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
                "        long long ck{a} = lin % shape[{a}]; lin /= shape[{a}];\n"
            ));
        }
    }
    // Input base offset from the kept coords (a broadcast kept axis has stride 0
    // and drops out naturally).
    let base_expr = if kept.is_empty() {
        "0".to_string()
    } else {
        kept.iter()
            .map(|a| format!("ck{a}*s0[{a}]"))
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
            .map(|a| format!("ck{a}*so[{a}]"))
            .collect::<Vec<_>>()
            .join(" + ")
    } else {
        kept.iter()
            .enumerate()
            .map(|(j, a)| format!("ck{a}*so[{j}]"))
            .collect::<Vec<_>>()
            .join(" + ")
    };
    s.push_str(&format!("        long long oo = {oo_expr};\n"));
    // Reduced-axis offset for the fold: Σ cr{r}·s0[r]. A broadcast reduced axis
    // (stride 0) re-reads the same element `shape[r]` times — correct semantics.
    let red_off = reduced
        .iter()
        .map(|r| format!("cr{r}*s0[{r}]"))
        .collect::<Vec<_>>()
        .join(" + ");
    match rop {
        ReduceOp::Sum | ReduceOp::Mean => {
            s.push_str(&format!("        {acc} acc = {zero};\n"));
            for &r in &reduced {
                s.push_str(&format!(
                    "        for (long long cr{r} = 0; cr{r} < shape[{r}]; ++cr{r}) {{\n"
                ));
            }
            s.push_str(&format!("            long long idx = base + {red_off};\n"));
            s.push_str(&format!("            acc += {elem};\n"));
            for _ in &reduced {
                s.push_str("        }\n");
            }
        }
        ReduceOp::Max | ReduceOp::Min => {
            let cmp = if matches!(rop, ReduceOp::Max) { ">" } else { "<" };
            // `has` seeds the first reduced element (all cr=0) without a ±∞ literal;
            // an empty reduced extent leaves `acc = 0` (matching the fast path).
            s.push_str(&format!("        {acc} acc = {zero};\n"));
            s.push_str("        int has = 0;\n");
            for &r in &reduced {
                s.push_str(&format!(
                    "        for (long long cr{r} = 0; cr{r} < shape[{r}]; ++cr{r}) {{\n"
                ));
            }
            s.push_str(&format!("            long long idx = base + {red_off};\n"));
            s.push_str(&format!("            {acc} e = {elem};\n"));
            s.push_str(&format!(
                "            acc = has ? ((e != e || e {cmp} acc) ? e : acc) : e; has = 1;\n"
            ));
            for _ in &reduced {
                s.push_str("        }\n");
            }
        }
    }
    let finalized = if matches!(rop, ReduceOp::Mean) {
        // Mean divisor = product of the reduced extents (not just the last axis).
        let divisor = reduced
            .iter()
            .map(|r| format!("shape[{r}]"))
            .collect::<Vec<_>>()
            .join(" * ");
        format!("acc / ({acc})({divisor})")
    } else {
        "acc".to_string()
    };
    s.push_str(&format!("        out[oo] = {};\n", store(finalized)));
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
    let (axes, keepdim) = match plan.access {
        Access::Reduction { axes, keepdim, .. } => (*axes, *keepdim),
        _ => return None,
    };
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
        ScalarExpr::Input(_) | ScalarExpr::Const(_) | ScalarExpr::Param(_) | ScalarExpr::Reduced(_)
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
        | ScalarExpr::Reduced(_) => false,
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
        | ScalarExpr::Reduced(_) => e.clone(),
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
    emit_block_reducers(&mut s, acc, zero, &ops, &stem);
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
    ops: &std::collections::HashSet<ReduceOp>,
    stem: &str,
) {
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
        terms.push(format!("c{d}*{stride_arr}[{d}]"));
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
/// `maxNum`, NaN-*suppressing*) for a *separate* `Fmax`/`Fmin` op. So we emit the
/// compare-select, not `fmaxf`. (Operands appear 3× — the deferred temp-binding
/// pass, cf. relu/sigmoid, removes the recompute on compound inners.)
fn binary_f32(op: BinaryOp, a: String, b: String) -> String {
    match op {
        BinaryOp::Max => format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} > {b} ? {a} : {b})))"),
        BinaryOp::Min => format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} < {b} ? {a} : {b})))"),
        BinaryOp::Pow => format!("powf({a}, {b})"),
        // Floored remainder (torch.remainder, sign-of-divisor — Fuel's Op::Rem),
        // not C fmodf (sign-of-dividend). Operands appear twice (temp-binding TODO).
        BinaryOp::Rem => format!("({a} - floorf({a} / {b}) * {b})"),
    }
}

/// Same as [`binary_f32`] but with f64 math-function names.
fn binary_f64(op: BinaryOp, a: String, b: String) -> String {
    match op {
        BinaryOp::Max => format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} > {b} ? {a} : {b})))"),
        BinaryOp::Min => format!("({a} != {a} ? {a} : ({b} != {b} ? {b} : ({a} < {b} ? {a} : {b})))"),
        BinaryOp::Pow => format!("pow({a}, {b})"),
        BinaryOp::Rem => format!("({a} - floor({a} / {b}) * {b})"),
    }
}

/// Lower a non-infix binary op for `dtype` (f32/f64 native; f16/bf16 compute in
/// float). Mirrors [`cuda_unary`]; integer binary-function math is a follow-up.
fn cuda_binary(op: BinaryOp, a: String, b: String, dtype: ElementKind) -> String {
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
            ScalarExpr::Input(_) | ScalarExpr::Const(_) | ScalarExpr::Reduced(_) => {}
        }
    }
    let mut set = std::collections::BTreeSet::new();
    rec(e, &mut set);
    set.into_iter().collect()
}

/// The trailing `, float p0, float p1, …` kernel-signature suffix for the op's
/// runtime scalar params (empty when the op has none).
fn param_args(e: &ScalarExpr) -> String {
    params_used(e)
        .iter()
        .map(|i| format!(", float p{i}"))
        .collect()
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
        assert!(k.source.contains("long long c1 = lin % shape[1]; lin /= shape[1];"));
        assert!(k.source.contains("long long o0 = c0*s0[0] + c1*s0[1];"));
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
        assert!(k.source.contains("long long ck1 = lin % shape[1]; lin /= shape[1];"));
        assert!(k.source.contains("long long base = ck1*s0[1];"));
        assert!(k.source.contains("for (long long cr0 = 0; cr0 < shape[0]; ++cr0)"));
        assert!(k.source.contains("long long idx = base + cr0*s0[0];"));
        assert!(k.source.contains("long long oo = ck1*so[0];"));
        assert!(k.source.contains("acc += in0[idx];"));
        assert!(k.source.contains("out[oo] ="));
    }

    #[test]
    fn reduction_multi_axis_mean_divisor_is_the_extent_product() {
        use crate::ir::ReduceOp;
        use baracuda_kernels_types::AxisMask;
        // Reduce axes {0,1} of [2,3,4] → [4], Mean: divisor = shape[0] * shape[1].
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
        assert!(k.source.contains("for (long long cr0 = 0; cr0 < shape[0]; ++cr0)"));
        assert!(k.source.contains("for (long long cr1 = 0; cr1 < shape[1]; ++cr1)"));
        assert!(k.source.contains("long long idx = base + cr0*s0[0] + cr1*s0[1];"));
        assert!(k.source.contains("acc / (float)(shape[0] * shape[1])"));
        assert!(k.source.contains("long long base = ck2*s0[2];")); // kept axis 2
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
        assert!(k.source.contains("long long oo = ck1*so[1];")); // so[input-axis], not so[0]
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
        assert!(k.source.contains("long long idx = base + cr1*s0[1];")); // strided reduced fold
        assert!(k.source.contains("long long base = ck0*s0[0];"));
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
}
