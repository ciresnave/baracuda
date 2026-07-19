//! The **recipe** — a kernel's neutral KISS-Ops Semantics op-DAG (KISS-Contract
//! §2.3): the mandatory spine a recipe-verify importer runs to validate + register
//! an op it does not already know, and the decomposition that resolves an op to the
//! primitive floor.
//!
//! This re-bases the internal IR op names ([`crate::ir::ScalarExpr`] /
//! [`UnaryOp`] / [`BinaryOp`]) onto the single **KISS-Ops op-token set** — the same
//! op-vocab re-basing that lets the emitted contract carry a neutral op DAG instead
//! of a Fuel-private name. An op whose node has no confirmed KISS-Ops name yet is an
//! **honest miss**: [`semantics_dag`] returns `None` rather than fabricate a token.
//!
//! # Format — PROVISIONAL
//!
//! The Semantics text is a compact functional op-DAG — `op(arg, arg, …)`, with the
//! `Bind` leaf `in<i>` (kernel input) and the source-op leaves `const(<v>)`,
//! `iota(<axis>)`, `runtime_scalar(<slot>)` (attr in the parens) — e.g. a fused
//! `relu_add` is `add(relu(in0), in1)`. Fuel confirmed this functional text is a
//! valid **surface** the importer parses; Fuel flattens it to the §6.4-0009
//! `Op{op_name,op_attrs,child_edges} | Bind(input_index)` flat table and
//! canonicalizes on ingest (`docs/fuel-reply-recipe-{grammar,schema}-2026-07-15.md`).
//! The literal grammar is still a co-pin strawman, isolated here.
//!
//! # Scope
//!
//! Elementwise bodies over `Input`/`Const` leaves **plus** the co-pinned source ops
//! `Coord`→`iota` / `Param`→`runtime_scalar`; **contractions** (`matmul[<roles>]`
//! fold + the `Reduced(0)`→node epilogue, incl. fused bias/activation);
//! **reductions** (`reduce[<monoid>,<axes>,<keepdim>]` fold + post; float `Mean` =
//! `sum` fold ÷ `reduce_extent(<axes>)`, integer `Mean` an honest miss); **scans**
//! (`prefix_scan[<monoid>,<axis>,<excl>]`, reverse = flip ∘ scan
//! ∘ flip); **row-reductions** (staged `reduce[<monoid>,last,nokd]` folds
//! producing `Reduced(0..n)`, then a full-width epilogue over the stage scalars +
//! the row-streamed inputs — softmax/rmsnorm, a float `Mean` stage = `sum` fold ÷
//! `reduce_extent(last)`, integer `Mean` an honest miss); and
//! the **data-dependent indexed** ops that ride `op.read_index`/`op.write_index`
//! rather than an `Access` variant — a `gather[<axis>,<oob>,<index_dtype>](data,
//! index)` read and a `scatter[<axis>,<combine>,<oob>,<index_dtype>](value, index)`
//! write (Fuel's pinned `gather`/`scatter` node schemas; child order data/value
//! then index). Fuel's scatter floor is `ScatterAdd` = `scatter{atomic-add}` only,
//! so a bare-assign / atomic-max / atomic-min scatter is an honest miss. Pooling,
//! sort, and im2col remain follow-ups — an honest miss (`None`) until covered.

use crate::ir::{
    Access, AxisRole, BinaryOp, ContractionAxes, OobPolicy, OpDef, ReadIndex, ReduceOp, ScalarExpr,
    UnaryOp, WriteCombine, WriteIndex,
};
use baracuda_kernel_vocab::{AxisMask, ElementKind};

/// The KISS-Ops op-DAG (recipe) for `op`, or `None` if `op` is not yet expressible
/// as a neutral recipe (unsupported access, or a node with no confirmed KISS-Ops
/// name).
#[must_use]
pub fn semantics_dag(op: &OpDef) -> Option<String> {
    // Data-dependent GATHER / SCATTER ride `op.read_index` / `op.write_index`, NOT an
    // `Access` variant (a gather is `Access::Elementwise` WITH an `Indexed` read).
    // Detect them BEFORE the access match, whose `Elementwise` arm would emit the
    // plain body and SILENTLY DROP the runtime index — a wrong recipe for a gather
    // (it would read as the identity `in0`).
    if op.read_index.iter().any(|r| !r.is_direct()) {
        return gather_recipe(op);
    }
    if !op.write_index.is_direct() {
        return scatter_recipe(op);
    }
    match &op.access {
        // Elementwise: the body over Input/Const/source-op leaves; no fold node.
        Access::Elementwise => expr_to_recipe(&op.body, &[]),
        // Contraction: a `matmul[<roles>](in0, in1)` fold node, then the epilogue
        // over it. `Reduced(0)` in the epilogue = the K-sum = the matmul node (Fuel
        // co-pin: `Reduced` is a child_edge to the fold node), and a fused
        // bias/activation composes as ordinary elementwise nodes over it (the bias
        // rides `in2` = Bind(2)). Roles come from the op's `ContractionAxes` — no
        // rank/key dependency. `matmul` attr surface is a co-pin strawman
        // (docs/fuel-ask-recipe-copin-2026-07-16.md).
        Access::Contraction { axes, epilogue, .. } => {
            let node = format!("matmul[{}](in0, in1)", contraction_roles(axes));
            expr_to_recipe(epilogue, &[node])
        }
        // Reduction: a `reduce[<monoid>,<axes>,<keepdim>]` fold node over the
        // per-element pre-map (`op.body`), then the post epilogue over it
        // (`Reduced(0)`→node). `Mean` is not a monoid: a `sum` fold + a
        // `div`-by-extent finalize, the divisor being the shape-derived reduced-axis
        // extent leaf `reduce_extent(<axes>)` (Fuel-confirmed — NOT a literal `const`;
        // `StructureKey` carries size *classes*, not extents). INTEGER Mean stays an
        // honest miss (the emitter rejects `int_acc && Mean`, cuda.rs:2241 — an integer
        // average rounds, no such single-dtype cell).
        Access::Reduction {
            op: rop,
            axes,
            keepdim,
            post,
        } => {
            let pre = expr_to_recipe(&op.body, &[])?;
            let axes_code = reduce_axes_code(axes);
            let kd = if *keepdim { "kd" } else { "nokd" };
            let node = if matches!(rop, ReduceOp::Mean) {
                if matches!(op.dtypes.first(), Some(ElementKind::I32 | ElementKind::I64)) {
                    return None;
                }
                // The extent leaf carries the SAME axes token as the fold node
                // (byte-identical axis field per Fuel's refinement); `keepdim` shapes
                // only the fold's output, never the scalar divisor, so it is omitted.
                format!("div(reduce[sum,{axes_code},{kd}]({pre}), reduce_extent({axes_code}))")
            } else {
                let monoid = reduce_monoid(*rop)?;
                format!("reduce[{monoid},{axes_code},{kd}]({pre})")
            };
            expr_to_recipe(post, &[node])
        }
        // Scan: a `prefix_scan[<monoid>,<axis>,<excl>]` node over the pre-map, then
        // the post over it. Fuel co-pin: a `reverse` scan = flip ∘ prefix_scan ∘
        // flip (there is no reverse field). `Mean` is rejected at plan (not a
        // monoid).
        Access::Scan {
            op: rop,
            axis,
            reverse,
            exclusive,
            pre,
            post,
        } => {
            let monoid = reduce_monoid(*rop)?;
            let pre_r = expr_to_recipe(pre, &[])?;
            let excl = if *exclusive { "excl" } else { "incl" };
            let node = if *reverse {
                format!("flip[{axis}](prefix_scan[{monoid},{axis},{excl}](flip[{axis}]({pre_r})))")
            } else {
                format!("prefix_scan[{monoid},{axis},{excl}]({pre_r})")
            };
            expr_to_recipe(post, &[node])
        }
        // RowReduce (fused reduce → broadcast → elementwise over the last axis):
        // each stage `i` produces `Reduced(i)`; a later stage's `pre` may read earlier
        // stages' `Reduced(j<i)` (thread the already-built stage strings), and the
        // `epilogue` reads `Reduced(0..n)` AND the full-width row-streamed `Input`s. A
        // monoid stage is a `reduce[<monoid>,last,nokd]` fold; a `Mean` stage is a
        // `sum` fold ÷ `reduce_extent(last)` — the same divisor leaf as the Reduction
        // arm (RowReduce always folds the last axis, so `last`), which is exactly what
        // RmsNorm/LayerNorm internal means need. INTEGER Mean stays an honest miss.
        Access::RowReduce { stages, epilogue } => {
            let int_acc = matches!(op.dtypes.first(), Some(ElementKind::I32 | ElementKind::I64));
            let mut stage_strs: Vec<String> = Vec::with_capacity(stages.len());
            for st in stages {
                let pre = expr_to_recipe(&st.pre, &stage_strs)?;
                let node = if matches!(st.op, ReduceOp::Mean) {
                    if int_acc {
                        return None;
                    }
                    format!("div(reduce[sum,last,nokd]({pre}), reduce_extent(last))")
                } else {
                    let monoid = reduce_monoid(st.op)?;
                    format!("reduce[{monoid},last,nokd]({pre})")
                };
                stage_strs.push(node);
            }
            expr_to_recipe(epilogue, &stage_strs)
        }
        // Pooling (Window), sort (RowSort), im2col carry structure not yet covered —
        // a follow-up honest miss. (Gather/scatter are handled above, off the
        // read/write index roles, before this access match.)
        _ => None,
    }
}

/// The KISS-Ops monoid token for a [`ReduceOp`], or `None` for `Mean` — which is
/// NOT a monoid (Fuel: a `sum` fold + a `div`-by-extent epilogue), an honest miss
/// until the extent-div is expressible.
fn reduce_monoid(op: ReduceOp) -> Option<&'static str> {
    match op {
        ReduceOp::Sum => Some("sum"),
        ReduceOp::Prod => Some("prod"),
        ReduceOp::Max => Some("max"),
        ReduceOp::Min => Some("min"),
        ReduceOp::Mean => None,
    }
}

/// The reduced-axis attr for a `reduce[…]` node: `last` for the empty-mask
/// last-axis default (Fuel resolves it against the interface rank), else the raw
/// mask as `0x<hex>`. A co-pin strawman surface (the field is the pinned
/// `reduce_axes`).
fn reduce_axes_code(axes: &AxisMask) -> String {
    if axes.is_empty() {
        "last".to_string()
    } else {
        format!("0x{:x}", axes.0)
    }
}

/// Compact role-vector attr for a contraction — `<lhs>.<rhs>` over the axis-role
/// codes (`b`atch / free-`m` / free-`n` / contracted-`k`); e.g. rank-2 `mk.kn`,
/// batched `bmk.bkn`. The `{Batch,FreeM,FreeN,ContractedK}` vocabulary is Fuel's
/// proposed `matmul` op_attrs (co-pin pending).
fn contraction_roles(axes: &ContractionAxes) -> String {
    fn code(r: AxisRole) -> char {
        match r {
            AxisRole::Batch => 'b',
            AxisRole::FreeM => 'm',
            AxisRole::FreeN => 'n',
            AxisRole::ContractedK => 'k',
        }
    }
    let lhs: String = axes.lhs.iter().map(|r| code(*r)).collect();
    let rhs: String = axes.rhs.iter().map(|r| code(*r)).collect();
    format!("{lhs}.{rhs}")
}

/// The recipe node for a **gather** — `gather[<axis>,<oob>,<index_dtype>](<data>,
/// <index>)`, child order data-then-index (Fuel's pinned `gather` child_edges
/// `[data, index]`; the positional `index_operand` rides that order, not an attr).
/// `None` (honest miss) for an un-namable index dtype (only `u32`/`i32`/`i64` are in
/// Fuel's pinned `gather{…, index_dtype}` set) or a FUSED gather body: v1 covers the
/// identity read `data[index]` only — an elementwise-over-gather (e.g. `relu(gather)`)
/// is a follow-up, so a non-identity body is withheld rather than mis-described.
fn gather_recipe(op: &OpDef) -> Option<String> {
    // The one `Indexed` input is the gathered DATA operand (`data`); its role names
    // which operand supplies the index and along which axis. (v1 gathers a single
    // operand — `plan::assert_valid_gather`.)
    let (data, index_operand, axis, oob, index_dtype) =
        op.read_index
            .iter()
            .enumerate()
            .find_map(|(i, r)| match r {
                ReadIndex::Indexed {
                    index_operand,
                    axis,
                    oob,
                    index_dtype,
                } => Some((i, *index_operand, *axis, *oob, *index_dtype)),
                ReadIndex::Direct => None,
            })?;
    // The gathered body must be the identity read of the data operand; a fused body
    // is not yet expressible as a single gather node → honest miss.
    if op.body != ScalarExpr::Input(data as u8) {
        return None;
    }
    let idt = index_dtype_code(index_dtype)?;
    Some(format!(
        "gather[{axis},{},{idt}](in{data}, in{index_operand})",
        oob_code(oob)
    ))
}

/// The recipe node for a **scatter** — `scatter[<axis>,<combine>,<oob>,<index_dtype>]
/// (<value>, <index>)`, child order value-then-index. `None` (honest miss) for a
/// combine outside Fuel's floor (Fuel implements `ScatterAdd` = `atomic-add` ONLY;
/// bare-assign / atomic-max / atomic-min are Fuel-side gaps — an unresolvable recipe)
/// or an un-namable index dtype. The scattered `value` is the op body (`in0` for
/// scatter_add/index_add, `const(1)` for bincount).
fn scatter_recipe(op: &OpDef) -> Option<String> {
    let (index_operand, axis, combine, oob, index_dtype) = match &op.write_index {
        WriteIndex::ScatterIndexed {
            index_operand,
            axis,
            combine,
            oob,
            index_dtype,
        } => (*index_operand, *axis, *combine, *oob, *index_dtype),
        WriteIndex::Direct => return None,
    };
    let combine_code = scatter_combine_code(combine)?;
    let idt = index_dtype_code(index_dtype)?;
    let value = expr_to_recipe(&op.body, &[])?;
    Some(format!(
        "scatter[{axis},{combine_code},{},{idt}]({value}, in{index_operand})",
        oob_code(oob)
    ))
}

/// The KISS-Ops index-dtype token for a gather/scatter index operand, or `None` for
/// a dtype outside Fuel's pinned `index_dtype∈{u32,i32,i64}` set (an honest miss —
/// never a fabricated dtype token). Lowercase to match Fuel's pinned schema spelling.
fn index_dtype_code(dt: ElementKind) -> Option<&'static str> {
    match dt {
        ElementKind::U32 => Some("u32"),
        ElementKind::I32 => Some("i32"),
        ElementKind::I64 => Some("i64"),
        _ => None,
    }
}

/// The out-of-range policy token — matching the contract-side `oob_policy` field
/// spelling. **Exhaustive** over [`OobPolicy`] (a new policy forces a decision here).
fn oob_code(oob: OobPolicy) -> &'static str {
    match oob {
        OobPolicy::Skip => "skip",
        OobPolicy::Clamp => "clamp",
        OobPolicy::ZeroFill => "zero_fill",
    }
}

/// The KISS-Ops `scatter_combine` token, or `None` for a combine Fuel's floor lacks.
/// **Exhaustive** over [`WriteCombine`] (a new combine forces a decision). Fuel
/// implements `ScatterAdd` = `atomic-add` ONLY; a bare-`Assign` scatter (no Fuel
/// `Scatter` op), `AtomicMax`, and `AtomicMin` (no scatter-reduce op) are Fuel-side
/// gaps — an unresolvable recipe, so an honest miss rather than a fabricated token.
fn scatter_combine_code(combine: WriteCombine) -> Option<&'static str> {
    match combine {
        WriteCombine::AtomicAdd => Some("atomic-add"),
        WriteCombine::Assign | WriteCombine::AtomicMax | WriteCombine::AtomicMin => None,
    }
}

/// Serialize a scalar expression as a KISS-Ops recipe sub-DAG, or `None` if any
/// node has no confirmed KISS-Ops re-basing (never fabricates a token). `reduced`
/// is the ordered set of fold-node strings a `Reduced(i)` leaf resolves to — one
/// entry per stage inside a reduction/contraction/scan/RowReduce epilogue (a
/// single-fold arm passes a 1-element slice), and empty (`&[]`) for a bare
/// elementwise body. A `Reduced(i)` with no `i`-th entry is an honest miss.
fn expr_to_recipe(e: &ScalarExpr, reduced: &[String]) -> Option<String> {
    use ScalarExpr as E;
    Some(match e {
        E::Input(i) => format!("in{i}"),
        E::Const(c) => format!("const({})", const_repr(*c)),
        E::Add(a, b) => format!(
            "add({}, {})",
            expr_to_recipe(a, reduced)?,
            expr_to_recipe(b, reduced)?
        ),
        E::Sub(a, b) => format!(
            "sub({}, {})",
            expr_to_recipe(a, reduced)?,
            expr_to_recipe(b, reduced)?
        ),
        E::Mul(a, b) => format!(
            "mul({}, {})",
            expr_to_recipe(a, reduced)?,
            expr_to_recipe(b, reduced)?
        ),
        E::Div(a, b) => format!(
            "div({}, {})",
            expr_to_recipe(a, reduced)?,
            expr_to_recipe(b, reduced)?
        ),
        E::Unary(u, x) => format!("{}({})", unary_kiss_name(*u)?, expr_to_recipe(x, reduced)?),
        E::Binary(op, x, y) => format!(
            "{}({}, {})",
            binary_kiss_name(*op)?,
            expr_to_recipe(x, reduced)?,
            expr_to_recipe(y, reduced)?
        ),
        E::Select(c, a, b) => format!(
            "select({}, {}, {})",
            expr_to_recipe(c, reduced)?,
            expr_to_recipe(a, reduced)?,
            expr_to_recipe(b, reduced)?
        ),
        // Source ops (KISS-Ops leaves with an attr, no child edges — Fuel co-pin
        // 2026-07-15). The attr rides the parens, mirroring `const(v)`.
        E::Coord(axis) => format!("iota({axis})"),
        E::Param(i) => format!("runtime_scalar({i})"),
        // `Reduced(i)` resolves to stage `i`'s fold node (reduction/contraction/
        // scan/RowReduce epilogue); a bare elementwise body has no fold (`reduced`
        // empty) and an out-of-range stage index is an honest miss — never a
        // fabricated leaf.
        E::Reduced(i) => reduced.get(*i as usize).cloned()?,
    })
}

/// Readable KISS-Ops `const` value: the finite number, or a non-finite tag.
fn const_repr(c: f64) -> String {
    if c.is_finite() {
        format!("{c}")
    } else if c.is_nan() {
        "nan".to_string()
    } else if c > 0.0 {
        "inf".to_string()
    } else {
        "-inf".to_string()
    }
}

/// The confirmed KISS-Ops op token for a [`UnaryOp`], or `None` if not yet re-based.
/// **Exhaustive** (no catch-all) — a new [`UnaryOp`] variant forces a decision here
/// (compile error) rather than silently becoming an honest miss. Only tokens
/// verified present in the KISS-Ops op set are mapped; the rest return `None`.
fn unary_kiss_name(op: UnaryOp) -> Option<&'static str> {
    use UnaryOp as U;
    Some(match op {
        U::Neg => "neg",
        U::Abs => "abs",
        U::Sqr => "sqr",
        U::Sqrt => "sqrt",
        U::Rsqrt => "rsqrt",
        U::Recip => "recip",
        U::Exp => "exp",
        U::Log => "log",
        U::Tanh => "tanh",
        U::Sigmoid => "sigmoid",
        U::Relu => "relu",
        U::Erf => "erf",
        U::Gelu => "gelu",
        U::Silu => "silu",
        U::Sin => "sin",
        U::Cos => "cos",
        U::Floor => "floor",
        U::Ceil => "ceil",
        U::Sign => "sign",
        U::Step => "step",
        U::Erfc => "erfc",
        U::Trunc => "trunc",
        U::Exp2 => "exp2",
        U::Expm1 => "expm1",
        U::Log2 => "log2",
        U::Log10 => "log10",
        U::Log1p => "log1p",
        U::Sinh => "sinh",
        U::Cosh => "cosh",
        U::Tan => "tan",
        U::Asin => "asin",
        U::Acos => "acos",
        U::Atan => "atan",
        U::Asinh => "asinh",
        U::Acosh => "acosh",
        U::Atanh => "atanh",
        U::Cbrt => "cbrt",
        U::Lgamma => "lgamma",
        // `round` = round-half-to-even (RNE): the emitter is `rintf`/`rint` (Slang
        // `round`), the oracle is `round_ties_even` — an EXACT match for KISS-Ops
        // `round_even` (verified vs the emitter+oracle, not just the IR doc). A
        // discontinuous op, so the tie-rule fidelity is load-bearing (no tolerance).
        U::Round => "round_even",
    })
}

/// The confirmed KISS-Ops op token for a [`BinaryOp`], or `None` if not yet re-based.
/// **Exhaustive** — a new [`BinaryOp`] variant forces a decision here.
fn binary_kiss_name(op: BinaryOp) -> Option<&'static str> {
    use BinaryOp as B;
    Some(match op {
        // NaN-propagating elementwise minmax (`torch.maximum`/`minimum`) — the
        // KISS-Ops tokens are `max_prop`/`min_prop`, NOT bare `max`/`min` (those are
        // the reduce/scan MONOID enum, see `reduce_monoid`) and NOT the NaN-suppressing
        // `fmax_ieee`/`fmin_ieee` below. Never alias the two families (KISS-OPS-6.15-0001).
        B::Max => "max_prop",
        B::Min => "min_prop",
        B::Pow => "pow",
        B::Atan2 => "atan2",
        B::Copysign => "copysign",
        B::Nextafter => "nextafter",
        B::FmaxIeee => "fmax_ieee",
        B::FminIeee => "fmin_ieee",
        B::RemTrunc => "rem_trunc",
        // Floored remainder (sign-of-divisor, `torch.remainder`): emitter/oracle
        // `a - floor(a/b)*b` = KISS-Ops `rem_floor`'s reference decomposition
        // `sub(a, mul(floor(div(a,b)), b))` verbatim. Distinct from `rem_trunc`
        // (C `fmod`, sign-of-dividend) above — never alias the two.
        B::Rem => "rem_floor",
        B::CmpEq => "cmp_eq",
        B::CmpNe => "cmp_ne",
        B::CmpLt => "cmp_lt",
        B::CmpLe => "cmp_le",
        B::CmpGt => "cmp_gt",
        B::CmpGe => "cmp_ge",
        B::BitAnd => "bit_and",
        B::BitOr => "bit_or",
        B::BitXor => "bit_xor",
        B::Shl => "shl",
        B::Shr => "shr",
        B::LogicalAnd => "logical_and",
        B::LogicalOr => "logical_or",
        // `logical_xor` is genuinely not in the KISS-Ops set (only logical_and/or/not)
        // — honest miss, never a guessed name.
        B::LogicalXor => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinaryOp, Expr, OpDef, ScalarExpr, UnaryOp, input, konst, param, reduced};
    use baracuda_kernel_vocab::ElementKind::F32;

    fn unary_recipe(u: UnaryOp) -> Option<String> {
        let op = OpDef::elementwise(
            "t",
            1,
            &[F32],
            Expr(ScalarExpr::Unary(u, Box::new(ScalarExpr::Input(0)))),
        );
        semantics_dag(&op)
    }

    fn binary_recipe(b: BinaryOp) -> Option<String> {
        let op = OpDef::elementwise(
            "t",
            2,
            &[F32],
            Expr(ScalarExpr::Binary(
                b,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1)),
            )),
        );
        semantics_dag(&op)
    }

    #[test]
    fn primitive_add_recipe() {
        let op = OpDef::elementwise("add", 2, &[F32], input(0) + input(1));
        assert_eq!(semantics_dag(&op).as_deref(), Some("add(in0, in1)"));
    }

    #[test]
    fn fused_relu_add_recipe() {
        let op = OpDef::elementwise("relu_add", 2, &[F32], input(0).relu() + input(1));
        assert_eq!(semantics_dag(&op).as_deref(), Some("add(relu(in0), in1)"));
    }

    #[test]
    fn fma_recipe_nests() {
        let op = OpDef::elementwise("fma", 3, &[F32], input(0) * input(1) + input(2));
        assert_eq!(
            semantics_dag(&op).as_deref(),
            Some("add(mul(in0, in1), in2)")
        );
    }

    #[test]
    fn const_leaf_recipe() {
        let op = OpDef::elementwise("scale", 1, &[F32], input(0) * konst(0.5));
        assert_eq!(semantics_dag(&op).as_deref(), Some("mul(in0, const(0.5))"));
    }

    #[test]
    fn coord_and_param_map_to_the_resolved_source_ops() {
        // Fuel co-pin (docs/fuel-reply-recipe-schema-2026-07-15.md): coord → the
        // KISS-Ops `iota{axis}` source op, param → `runtime_scalar{slot}` — both
        // keep the node schema closed to Op|Bind (the attr rides the parens, as
        // `const(v)` already does). Previously honest misses.
        let coord_op = OpDef::elementwise("c", 1, &[F32], Expr(ScalarExpr::Coord(2)));
        assert_eq!(semantics_dag(&coord_op).as_deref(), Some("iota(2)"));
        let param_op = OpDef::elementwise("p", 1, &[F32], input(0) * param(0));
        assert_eq!(
            semantics_dag(&param_op).as_deref(),
            Some("mul(in0, runtime_scalar(0))")
        );
    }

    #[test]
    fn reduced_leaf_is_an_honest_miss_in_an_elementwise_body() {
        // A `Reduced(i)` is a fold-result child_edge (reduction/contraction access),
        // NOT an elementwise leaf — no recipe here yet → None, never a wrong token.
        let op = OpDef::elementwise("redy", 1, &[F32], Expr(ScalarExpr::Reduced(0)));
        assert_eq!(semantics_dag(&op), None);
    }

    #[test]
    fn round_and_rem_re_base_to_round_even_and_rem_floor() {
        // Both re-based to CONFIRMED kiss-ops-vocab tokens after verifying an EXACT
        // semantic match against the actual emitter + oracle (not just the IR doc):
        //   Round: emitter `rintf`/`rint` (cuda.rs) + Slang `round` = round-half-to-even
        //          (RNE) across all backends; oracle `round_ties_even`; KISS `round_even`
        //          is ties-to-even (2.5→2, -2.5→-2, -0.5→-0.0). Exact tie-rule match — a
        //          discontinuous op, so the tie-rule MUST match (no tolerance cushion).
        //   Rem:   emitter/oracle `a - floor(a/b)*b` (floored, sign-of-divisor); KISS
        //          `rem_floor` decomposes to `sub(a, mul(floor(div(a,b)), b))` — the
        //          identical formula on every backend (algebraic, no mode dependence).
        assert_eq!(
            unary_recipe(UnaryOp::Round).as_deref(),
            Some("round_even(in0)")
        );
        assert_eq!(
            binary_recipe(BinaryOp::Rem).as_deref(),
            Some("rem_floor(in0, in1)")
        );
    }

    #[test]
    fn extended_coverage_re_bases_the_newly_mapped_ops() {
        assert_eq!(unary_recipe(UnaryOp::Asinh).as_deref(), Some("asinh(in0)"));
        assert_eq!(unary_recipe(UnaryOp::Acosh).as_deref(), Some("acosh(in0)"));
        assert_eq!(unary_recipe(UnaryOp::Atanh).as_deref(), Some("atanh(in0)"));
        assert_eq!(unary_recipe(UnaryOp::Cbrt).as_deref(), Some("cbrt(in0)"));
        assert_eq!(
            unary_recipe(UnaryOp::Lgamma).as_deref(),
            Some("lgamma(in0)")
        );
        assert_eq!(
            binary_recipe(BinaryOp::Nextafter).as_deref(),
            Some("nextafter(in0, in1)")
        );
        assert_eq!(
            binary_recipe(BinaryOp::LogicalAnd).as_deref(),
            Some("logical_and(in0, in1)")
        );
        assert_eq!(
            binary_recipe(BinaryOp::LogicalOr).as_deref(),
            Some("logical_or(in0, in1)")
        );
    }

    #[test]
    fn nan_propagating_elementwise_minmax_map_to_max_prop_min_prop() {
        // `BinaryOp::Max`/`Min` are NaN-propagating (`torch.maximum`/`minimum` —
        // ir.rs:264), deliberately distinct from the NaN-suppressing `FmaxIeee`/
        // `FminIeee`. Their KISS-Ops elementwise tokens are `max_prop`/`min_prop`
        // (kiss-ops-vocab `MaxProp`/`MinProp`), NOT bare `max`/`min` — those exist
        // ONLY as the reduce/scan MONOID enum (`reduce[max,…]`). Emitting bare
        // `max`/`min` here would be unresolvable to a recipe-import peer, or silently
        // bind to the IEEE op and drop NaN-propagation (KISS-OPS-6.15-0001 forbids
        // aliasing the two families).
        assert_eq!(
            binary_recipe(BinaryOp::Max).as_deref(),
            Some("max_prop(in0, in1)")
        );
        assert_eq!(
            binary_recipe(BinaryOp::Min).as_deref(),
            Some("min_prop(in0, in1)")
        );
        // The NaN-suppressing IEEE variants stay distinct — never aliased to the above.
        assert_eq!(
            binary_recipe(BinaryOp::FmaxIeee).as_deref(),
            Some("fmax_ieee(in0, in1)")
        );
        assert_eq!(
            binary_recipe(BinaryOp::FminIeee).as_deref(),
            Some("fmin_ieee(in0, in1)")
        );
    }

    #[test]
    fn contraction_recipe_is_a_matmul_node_with_the_epilogue_over_it() {
        use crate::ir::ContractionAxes;
        // Plain matmul: the identity epilogue (`Reduced(0)`) IS the matmul node.
        let mm = OpDef::contraction("mm", &[F32], ContractionAxes::matmul(), reduced(0));
        assert_eq!(
            semantics_dag(&mm).as_deref(),
            Some("matmul[mk.kn](in0, in1)")
        );
        // Batched: the role vector carries the leading batch axis.
        let bmm = OpDef::contraction("bmm", &[F32], ContractionAxes::batched_matmul(), reduced(0));
        assert_eq!(
            semantics_dag(&bmm).as_deref(),
            Some("matmul[bmk.bkn](in0, in1)")
        );
        // Fused matmul + per-column bias + relu: the epilogue composes as ordinary
        // elementwise nodes over the matmul node; the bias is `in2` (Bind(2)).
        let mbr =
            OpDef::contraction_bias("mbr", &[F32], (reduced(0) + input(2)).unary(UnaryOp::Relu));
        assert_eq!(
            semantics_dag(&mbr).as_deref(),
            Some("relu(add(matmul[mk.kn](in0, in1), in2))")
        );
    }

    #[test]
    fn reduction_recipe_is_a_reduce_node_with_the_post_over_it() {
        use crate::ir::ReduceOp;
        // norm2 = sqrt(sum(sqr(x))): pre-map `sqr(in0)` feeds a `reduce[sum,…]`
        // fold node; the post `sqrt(Reduced(0))` composes over it (last-axis
        // default, no keepdim).
        let op = OpDef::reduction_post(
            "norm2",
            1,
            &[F32],
            input(0).unary(UnaryOp::Sqr),
            ReduceOp::Sum,
            reduced(0).sqrt(),
        );
        assert_eq!(
            semantics_dag(&op).as_deref(),
            Some("sqrt(reduce[sum,last,nokd](sqr(in0)))")
        );
        // INTEGER Mean stays an honest miss — the emitter rejects `int_acc && Mean`
        // (cuda.rs:2241; an integer average rounds, unrepresentable in a single-dtype
        // cell), so there is no such kernel to describe. Float Mean now emits the
        // sum+div-extent recipe (see the dedicated test below).
        use baracuda_kernel_vocab::ElementKind::I32;
        let int_mean = OpDef::reduction("imean", 1, &[I32], input(0), ReduceOp::Mean);
        assert_eq!(semantics_dag(&int_mean), None);
    }

    #[test]
    fn mean_reduction_recipe_is_a_sum_fold_divided_by_the_reduced_extent() {
        use crate::ir::ReduceOp;
        // Mean = a `sum` fold + a `div`-by-extent finalize (Fuel's `MeanDim`
        // decomposes exactly this way). The divisor is the shape-derived reduced-axis
        // extent — the `reduce_extent(<axes>)` source-op leaf CONFIRMED by Fuel
        // (docs/fuel-reply-reduce-extent-2026-07-18.md), NOT a literal `const`
        // (`StructureKey` carries size *classes*, not literal extents). The extent
        // leaf carries the SAME reduced-axis token as the fold node (byte-identical
        // axis field per Fuel's refinement).

        // Plain f32 Mean (identity post): the `div(sum, extent)` node IS the output;
        // last-axis default (`last`), no keepdim (`nokd`), matching `reduce_axes_code`.
        let mean = OpDef::reduction("mean", 1, &[F32], input(0), ReduceOp::Mean);
        assert_eq!(
            semantics_dag(&mean).as_deref(),
            Some("div(reduce[sum,last,nokd](in0), reduce_extent(last))")
        );

        // Fused post over Mean: the post's `Reduced(0)` resolves to the finalized
        // `div(...)` node (the "post sees the POST-Mean value" ordering pinned at
        // `plan::assert_valid_reduction_post` / the `Access::Reduction` doc), so a
        // `sqrt(Reduced(0))` post composes as `sqrt(div(sum, extent))`.
        let rms = OpDef::reduction_post(
            "mean_then_sqrt",
            1,
            &[F32],
            input(0).unary(UnaryOp::Sqr),
            ReduceOp::Mean,
            reduced(0).sqrt(),
        );
        assert_eq!(
            semantics_dag(&rms).as_deref(),
            Some("sqrt(div(reduce[sum,last,nokd](sqr(in0)), reduce_extent(last)))")
        );
    }

    #[test]
    fn scan_recipe_is_a_prefix_scan_node_with_reverse_as_flip() {
        use crate::ir::ReduceOp;
        // Forward inclusive cumsum on axis 1.
        let cs = OpDef::scan(
            "cumsum",
            1,
            &[F32],
            ReduceOp::Sum,
            1,
            false,
            false,
            input(0),
            reduced(0),
        );
        assert_eq!(
            semantics_dag(&cs).as_deref(),
            Some("prefix_scan[sum,1,incl](in0)")
        );
        // Reverse exclusive cumprod: Fuel's reverse = flip ∘ prefix_scan ∘ flip.
        let rc = OpDef::scan(
            "revcumprod",
            1,
            &[F32],
            ReduceOp::Prod,
            2,
            true,
            true,
            input(0),
            reduced(0),
        );
        assert_eq!(
            semantics_dag(&rc).as_deref(),
            Some("flip[2](prefix_scan[prod,2,excl](flip[2](in0)))")
        );
    }

    #[test]
    fn rowreduce_softmax_recipe_is_staged_reduces_with_the_epilogue_over_them() {
        use crate::ir::{ReduceOp, ReduceStage};
        // Softmax (2 stages): stage0 = row max, stage1 = sum(exp(x − max)). Each
        // stage folds over the LAST axis (`last`, `nokd`, row-broadcast). A later
        // stage's `pre` (and the epilogue) thread the earlier stages' strings —
        // stage1 reads `Reduced(0)`, the epilogue reads `Reduced(0..2)` AND the
        // full-width row-streamed `Input` (admissible here, unlike a Reduction).
        let stages = vec![
            ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Max,
            },
            ReduceStage {
                pre: (input(0) - reduced(0)).unary(UnaryOp::Exp).0,
                op: ReduceOp::Sum,
            },
        ];
        let epi = (input(0) - reduced(0)).unary(UnaryOp::Exp) / reduced(1);
        let op = OpDef::row_reduce("softmax", 1, &[F32], stages, epi);
        assert_eq!(
            semantics_dag(&op).as_deref(),
            Some(
                "div(exp(sub(in0, reduce[max,last,nokd](in0))), \
                 reduce[sum,last,nokd](exp(sub(in0, reduce[max,last,nokd](in0)))))"
            )
        );
    }

    #[test]
    fn rowreduce_rmsnorm_recipe_is_a_sum_of_squares_reduce_with_the_scale_epilogue() {
        use crate::ir::{ReduceOp, ReduceStage};
        // RmsNorm (1 stage): stage0 = sum(x²) over the last axis; the epilogue
        // `x · rsqrt(sum + eps)` scales the row-streamed input by the reduced
        // scalar. (The mean-of-squares form uses a `Mean` stage = sum-fold ÷ extent —
        // see rowreduce_mean_stage_is_a_sum_fold_divided_by_the_reduced_extent below.)
        let stages = vec![ReduceStage {
            pre: input(0).unary(UnaryOp::Sqr).0,
            op: ReduceOp::Sum,
        }];
        let epi = input(0) * (reduced(0) + konst(1e-5)).unary(UnaryOp::Rsqrt);
        let op = OpDef::row_reduce("rmsnorm", 1, &[F32], stages, epi);
        assert_eq!(
            semantics_dag(&op).as_deref(),
            Some("mul(in0, rsqrt(add(reduce[sum,last,nokd](sqr(in0)), const(0.00001))))")
        );
    }

    #[test]
    fn rowreduce_mean_stage_is_a_sum_fold_divided_by_the_reduced_extent() {
        use crate::ir::{ReduceOp, ReduceStage};
        // The canonical mean-of-squares RmsNorm: stage0 = mean(x²) over the last
        // axis, epilogue `x · rsqrt(mean + eps)`. A `Mean` stage folds `sum` then
        // divides by the reduced-axis extent — the SAME `reduce_extent(last)` leaf as
        // the Reduction arm (RowReduce always folds the last axis, so `last`). This is
        // the divisor Fuel flagged that RmsNorm/LayerNorm internal means need.
        let op = OpDef::row_reduce(
            "rmsnorm_mean",
            1,
            &[F32],
            vec![ReduceStage {
                pre: input(0).unary(UnaryOp::Sqr).0,
                op: ReduceOp::Mean,
            }],
            input(0) * (reduced(0) + konst(1e-5)).unary(UnaryOp::Rsqrt),
        );
        assert_eq!(
            semantics_dag(&op).as_deref(),
            Some(
                "mul(in0, rsqrt(add(div(reduce[sum,last,nokd](sqr(in0)), \
                 reduce_extent(last)), const(0.00001))))"
            )
        );
    }

    #[test]
    fn rowreduce_integer_mean_stage_stays_an_honest_miss() {
        use crate::ir::{ReduceOp, ReduceStage};
        use baracuda_kernel_vocab::ElementKind::I32;
        // INTEGER Mean stays an honest miss, as in the Reduction arm — an integer
        // average rounds (the emitter rejects int_acc && Mean), so no such kernel.
        let op = OpDef::row_reduce(
            "imean_rows",
            1,
            &[I32],
            vec![ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Mean,
            }],
            reduced(0) + input(0),
        );
        assert_eq!(semantics_dag(&op), None);
    }

    #[test]
    fn deliberately_unmapped_ops_stay_honest_misses() {
        // `logical_xor` is genuinely not in the KISS-Ops set (only logical_and/or/not) —
        // a recipe is withheld, never a guessed name. (Round/Rem re-based to
        // round_even/rem_floor — see round_and_rem_re_base_to_round_even_and_rem_floor.)
        assert_eq!(binary_recipe(BinaryOp::LogicalXor), None);
    }

    #[test]
    fn gather_recipe_is_a_gather_node_over_data_and_index() {
        use crate::ir::OobPolicy;
        use baracuda_kernel_vocab::ElementKind::{I32, I64, U32};
        // A gather rides `op.read_index` (Access::Elementwise WITH an Indexed read),
        // so its recipe is NOT the plain elementwise body `in0` — it is a `gather[…]`
        // node over (data, index), child order data-then-index (Fuel's pinned
        // `gather` child_edges `[data, index]`). Attrs = axis, oob, index dtype
        // (Fuel's pinned `gather{axis, oob_policy, index_operand, index_dtype}`; the
        // positional `index_operand` rides the child order). Index dtype ∈
        // {u32,i32,i64} (the pinned set) — all namable.
        let g = OpDef::gather("g", &[F32], 0, OobPolicy::Skip, I32);
        assert_eq!(
            semantics_dag(&g).as_deref(),
            Some("gather[0,skip,i32](in0, in1)")
        );
        // u32 index + a row gather (index_select / embedding) at axis 1, zero-fill OOB.
        let emb = OpDef::gather("emb", &[F32], 1, OobPolicy::ZeroFill, U32);
        assert_eq!(
            semantics_dag(&emb).as_deref(),
            Some("gather[1,zero_fill,u32](in0, in1)")
        );
        // i64 index — also in Fuel's pinned index-dtype set.
        let g64 = OpDef::gather("g64", &[F32], 0, OobPolicy::Skip, I64);
        assert_eq!(
            semantics_dag(&g64).as_deref(),
            Some("gather[0,skip,i64](in0, in1)")
        );
    }

    #[test]
    fn scatter_add_recipe_is_a_scatter_node_over_value_and_index() {
        use baracuda_kernel_vocab::ElementKind::{I32, U32};
        // scatter_add rides `op.write_index` (ScatterIndexed{combine: AtomicAdd}); the
        // recipe is a `scatter[…]` node over (value = body, index), child order
        // value-then-index. Fuel implements ScatterAdd = `scatter{atomic-add}`, so the
        // combine is resolvable. Attrs = axis, combine, oob, index dtype (Fuel's pinned
        // `scatter{axis, scatter_combine, oob_policy, index_operand, index_dtype}`).
        let sa = OpDef::scatter_add("sa", &[I32], 0, I32);
        assert_eq!(
            semantics_dag(&sa).as_deref(),
            Some("scatter[0,atomic-add,skip,i32](in0, in1)")
        );
        // bincount = scatter{atomic-add} of a const-1 value into a SELF-indexed dest
        // (index_operand 0 = the data itself); the value is the `const(1)` body.
        let bc = OpDef::bincount("bc", U32);
        assert_eq!(
            semantics_dag(&bc).as_deref(),
            Some("scatter[0,atomic-add,skip,u32](const(1), in0)")
        );
    }

    #[test]
    fn unresolvable_scatter_combines_and_index_dtypes_stay_honest_misses() {
        use crate::ir::{OobPolicy, WriteCombine, WriteIndex};
        use baracuda_kernel_vocab::ElementKind::{F32 as F32K, I32};
        // Fuel implements ScatterAdd (atomic-add) ONLY; a bare-assign scatter and
        // atomic-max / atomic-min are Fuel-side gaps — an unresolvable recipe, so an
        // honest miss (None), never a fabricated combine token.
        let assign = OpDef::scatter("s", &[I32], 0, I32); // WriteCombine::Assign
        assert_eq!(semantics_dag(&assign), None);
        let amax =
            OpDef::scatter("smax", &[I32], 0, I32).with_scatter(WriteIndex::ScatterIndexed {
                index_operand: 1,
                axis: 0,
                combine: WriteCombine::AtomicMax,
                oob: OobPolicy::Skip,
                index_dtype: I32,
            });
        assert_eq!(semantics_dag(&amax), None);
        // An un-namable index dtype (outside Fuel's pinned {u32,i32,i64}) → honest
        // miss, never a fabricated dtype token.
        let bad = OpDef::gather("gbad", &[F32K], 0, OobPolicy::Skip, F32K);
        assert_eq!(semantics_dag(&bad), None);
    }
}
