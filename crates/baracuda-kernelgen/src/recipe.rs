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
//! **reductions** (`reduce[<monoid>,<axes>,<keepdim>]` fold + post, `Mean` an honest
//! miss); and **scans** (`prefix_scan[<monoid>,<axis>,<excl>]`, reverse = flip ∘ scan
//! ∘ flip). RowReduce, gather/scatter, pooling, sort, and im2col are follow-ups — an
//! honest miss (`None`) until covered.

use crate::ir::{
    Access, AxisRole, BinaryOp, ContractionAxes, OpDef, ReduceOp, ScalarExpr, UnaryOp,
};
use baracuda_kernel_vocab::AxisMask;

/// The KISS-Ops op-DAG (recipe) for `op`, or `None` if `op` is not yet expressible
/// as a neutral recipe (unsupported access, or a node with no confirmed KISS-Ops
/// name).
#[must_use]
pub fn semantics_dag(op: &OpDef) -> Option<String> {
    match &op.access {
        // Elementwise: the body over Input/Const/source-op leaves; no fold node.
        Access::Elementwise => expr_to_recipe(&op.body, None),
        // Contraction: a `matmul[<roles>](in0, in1)` fold node, then the epilogue
        // over it. `Reduced(0)` in the epilogue = the K-sum = the matmul node (Fuel
        // co-pin: `Reduced` is a child_edge to the fold node), and a fused
        // bias/activation composes as ordinary elementwise nodes over it (the bias
        // rides `in2` = Bind(2)). Roles come from the op's `ContractionAxes` — no
        // rank/key dependency. `matmul` attr surface is a co-pin strawman
        // (docs/fuel-ask-recipe-copin-2026-07-16.md).
        Access::Contraction { axes, epilogue, .. } => {
            let node = format!("matmul[{}](in0, in1)", contraction_roles(axes));
            expr_to_recipe(epilogue, Some(&node))
        }
        // Reduction: a `reduce[<monoid>,<axes>,<keepdim>]` fold node over the
        // per-element pre-map (`op.body`), then the post epilogue over it
        // (`Reduced(0)`→node). `Mean` is not a monoid (Fuel: sum-fold + div) —
        // honest miss.
        Access::Reduction {
            op: rop,
            axes,
            keepdim,
            post,
        } => {
            let monoid = reduce_monoid(*rop)?;
            let pre = expr_to_recipe(&op.body, None)?;
            let node = format!(
                "reduce[{monoid},{},{}]({pre})",
                reduce_axes_code(axes),
                if *keepdim { "kd" } else { "nokd" }
            );
            expr_to_recipe(post, Some(&node))
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
            let pre_r = expr_to_recipe(pre, None)?;
            let excl = if *exclusive { "excl" } else { "incl" };
            let node = if *reverse {
                format!("flip[{axis}](prefix_scan[{monoid},{axis},{excl}](flip[{axis}]({pre_r})))")
            } else {
                format!("prefix_scan[{monoid},{axis},{excl}]({pre_r})")
            };
            expr_to_recipe(post, Some(&node))
        }
        // RowReduce, gather/scatter, pooling, sort, im2col carry structure not yet
        // covered — a follow-up honest miss.
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

/// Serialize a scalar expression as a KISS-Ops recipe sub-DAG, or `None` if any
/// node has no confirmed KISS-Ops re-basing (never fabricates a token). `reduced`
/// is the recipe string a `Reduced(0)` leaf resolves to — the fold node inside a
/// reduction/contraction epilogue; `None` for a bare elementwise body.
fn expr_to_recipe(e: &ScalarExpr, reduced: Option<&str>) -> Option<String> {
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
        // `Reduced(0)` resolves to the fold node inside a reduction/contraction
        // epilogue; a bare elementwise body has no fold (`reduced == None`) → honest
        // miss. `Reduced(i>0)` (a second stage) is not yet expressible.
        E::Reduced(0) => reduced.map(String::from)?,
        E::Reduced(_) => return None,
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
        // No confirmed KISS-Ops token yet (`round` is not in the op set) — honest miss.
        U::Round => return None,
    })
}

/// The confirmed KISS-Ops op token for a [`BinaryOp`], or `None` if not yet re-based.
/// **Exhaustive** — a new [`BinaryOp`] variant forces a decision here.
fn binary_kiss_name(op: BinaryOp) -> Option<&'static str> {
    use BinaryOp as B;
    Some(match op {
        B::Max => "max",
        B::Min => "min",
        B::Pow => "pow",
        B::Atan2 => "atan2",
        B::Copysign => "copysign",
        B::Nextafter => "nextafter",
        B::FmaxIeee => "fmax_ieee",
        B::FminIeee => "fmin_ieee",
        B::RemTrunc => "rem_trunc",
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
        // No confirmed KISS-Ops token yet: `Rem` (floored remainder) has no
        // confirmed name, and `logical_xor` is not in the op set — honest miss.
        B::Rem | B::LogicalXor => return None,
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
    fn round_has_no_confirmed_kiss_name_yet() {
        // `Round` is not (yet) a confirmed KISS-Ops token, so its recipe is withheld
        // rather than guessed.
        let op = OpDef::elementwise(
            "roundy",
            1,
            &[F32],
            Expr(ScalarExpr::Unary(
                UnaryOp::Round,
                Box::new(ScalarExpr::Input(0)),
            )),
        );
        assert_eq!(semantics_dag(&op), None);
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
        // Mean is not a monoid (Fuel: sum-fold + div-by-extent) → honest miss.
        let mean = OpDef::reduction("mean", 1, &[F32], input(0), ReduceOp::Mean);
        assert_eq!(semantics_dag(&mean), None);
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
    fn deliberately_unmapped_ops_stay_honest_misses() {
        // No confirmed KISS-Ops token for these yet — a recipe is withheld, never
        // a guessed name.
        assert_eq!(unary_recipe(UnaryOp::Round), None);
        assert_eq!(binary_recipe(BinaryOp::Rem), None);
        assert_eq!(binary_recipe(BinaryOp::LogicalXor), None);
    }
}
