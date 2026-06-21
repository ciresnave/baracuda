//! Derive a Fuel FKC `pattern:` block from an op's IR.
//!
//! For a *pattern-recognized* fused op (an elementwise epilogue Fuel discovers
//! inside a user's primitive subgraph), the FKC contract must carry a `pattern:`
//! tree describing the subgraph it replaces. Because the op's [`ScalarExpr`] body
//! *is* that subgraph, the pattern derives mechanically:
//!
//! - each arithmetic node → a graph-`Op` node (`Add`/`Sub`/`Mul`/`Div` — names
//!   already in the FKC §4.1 vocabulary);
//! - each [`ScalarExpr::Input`] → a `bind` leaf (`bind: i` → the fused op's
//!   `input[i]`);
//! - a reused input → a repeated `bind: i`, which is exactly the node-identity
//!   guard the spec requires (FKC §3.2) — for free, because a shared operand is
//!   literally the same `Input(i)`;
//! - interior nodes get `consumers: 1` (the sole-consumer fusion-safety rule,
//!   FKC §3a.4), the root gets the default (`any`).
//!
//! # Scope (v1)
//!
//! Pure-tensor elementwise bodies, plus runtime scalar params
//! ([`ScalarExpr::Param`]) which lower to `AddScalar`/`MulScalar` with an
//! `extract:` path (FKC §6). Compile-time scalar *constants*
//! ([`ScalarExpr::Const`]) have no graph-`Op` form yet and are rejected.
//!
//! Commutative operands (`Add`/`Mul`) are emitted in a deterministic internal
//! order, so two authorings of one expression (`a*b + c` vs `c + a*b`) emit
//! byte-identical YAML (clean diffs, caching, golden tests). This is **not** a
//! matching requirement: per FKC rev-4 §3a.2a Fuel canonicalizes *both* the
//! imported pattern and the user graph into one order before matching, so any
//! single emitted ordering matches regardless. The ordering key here
//! ([`canonicalize`]'s `sig`) is Baracuda-internal and need not equal Fuel's.

use crate::ir::{Access, BinaryOp, OpDef, ScalarExpr, UnaryOp};
use std::collections::BTreeSet;

/// A node in a derived FKC pattern tree (the v1 subset: `Op` + `bind`).
#[derive(Clone, Debug, PartialEq)]
pub enum PatternNode {
    /// A graph-`Op` node.
    Op {
        /// The FKC §4.1 graph-`Op` name.
        op: String,
        /// Ordered operand sub-patterns (one per tensor input).
        operands: Vec<PatternNode>,
        /// Consumer-count guard: `Some(1)` on interior nodes (sole-consumer),
        /// `None` (default `any`) on the root.
        consumers: Option<u32>,
        /// `extract:` entries `(field, path)` — one per scalar-param
        /// (`AddScalar`/`MulScalar`) value pulled into `op_params`. Collected
        /// onto the root (paths anchored there, per FKC §6); empty on most nodes.
        extract: Vec<(String, String)>,
    },
    /// A leaf binding the producer here as the fused op's `input[index]`.
    Bind(u8),
}

/// Why a body can't be expressed as a v1 FKC pattern.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PatternError {
    /// The op's access pattern isn't elementwise.
    NotElementwise,
    /// The body contains a scalar constant; scalar-param pattern emission
    /// (`AddScalar`/`MulScalar`/…) is a follow-up (needs the op-attribute bridge).
    ScalarParamUnsupported,
    /// The set of bound input indices isn't exactly `[0, n_inputs)`.
    BindSetMismatch {
        /// The op's declared input count.
        n_inputs: u8,
        /// The indices actually bound by the body, ascending.
        got: Vec<u8>,
    },
}

/// Derive the FKC pattern tree for `op`, or a [`PatternError`] if the body isn't
/// an expressible v1 pattern.
///
/// # Errors
/// See [`PatternError`].
pub fn derive_pattern(op: &OpDef) -> Result<PatternNode, PatternError> {
    if !matches!(op.access, Access::Elementwise) {
        return Err(PatternError::NotElementwise);
    }
    // Canonicalize commutative operands first so two authorings of one body
    // produce byte-identical paths/extracts/YAML (internal determinism); per
    // §3a.2a Fuel canonicalizes the pattern on import, so either order matches.
    let body = canonicalize(&op.body);
    let mut extracts = Vec::new();
    let mut root = walk(&body, true, &[], &mut extracts)?;
    // Scalar-param values are pulled via `extract:` on the root (FKC §6).
    if let PatternNode::Op { extract, .. } = &mut root {
        extracts.sort();
        *extract = extracts;
    }
    // The bound input indices must be exactly [0, n_inputs) — total only for a
    // body referencing each input exactly the right set; otherwise FKC §3.2's
    // "bind set MUST equal [0, N)" would reject it at import, so we reject here.
    let mut binds = BTreeSet::new();
    collect_binds(&root, &mut binds);
    let expected: BTreeSet<u8> = (0..op.n_inputs).collect();
    if binds != expected {
        return Err(PatternError::BindSetMismatch {
            n_inputs: op.n_inputs,
            got: binds.into_iter().collect(),
        });
    }
    Ok(root)
}

/// Rewrite a body into canonical form by sorting the operands of every
/// commutative node (`Add`/`Mul`) to a deterministic order, bottom-up
/// (children canonicalized first, so the sort key is stable). Non-commutative
/// nodes (`Sub`/`Div`/`Unary`) keep their operand order; leaves are unchanged.
///
/// Under FKC §3a.2a both sides canonicalize identically — Fuel canonicalizes the
/// imported pattern *and* the user graph before matching — so any single operand
/// order matches. We pick a deterministic one purely for reproducible output
/// (not for match correctness); the order *value* (via [`sig`]) need not equal
/// Fuel's key.
fn canonicalize(e: &ScalarExpr) -> ScalarExpr {
    match e {
        ScalarExpr::Add(a, b) => {
            let (lo, hi) = order2(canonicalize(a), canonicalize(b));
            ScalarExpr::Add(Box::new(lo), Box::new(hi))
        }
        ScalarExpr::Mul(a, b) => {
            let (lo, hi) = order2(canonicalize(a), canonicalize(b));
            ScalarExpr::Mul(Box::new(lo), Box::new(hi))
        }
        ScalarExpr::Sub(a, b) => {
            ScalarExpr::Sub(Box::new(canonicalize(a)), Box::new(canonicalize(b)))
        }
        ScalarExpr::Div(a, b) => {
            ScalarExpr::Div(Box::new(canonicalize(a)), Box::new(canonicalize(b)))
        }
        ScalarExpr::Binary(op, a, b) => {
            let (ca, cb) = (canonicalize(a), canonicalize(b));
            if matches!(op, BinaryOp::Max | BinaryOp::Min) {
                // commutative (like Add/Mul) — canonical operand order
                let (lo, hi) = order2(ca, cb);
                ScalarExpr::Binary(*op, Box::new(lo), Box::new(hi))
            } else {
                ScalarExpr::Binary(*op, Box::new(ca), Box::new(cb))
            }
        }
        ScalarExpr::Unary(op, x) => ScalarExpr::Unary(*op, Box::new(canonicalize(x))),
        ScalarExpr::Input(_) | ScalarExpr::Const(_) | ScalarExpr::Param(_) => e.clone(),
    }
}

/// Order two already-canonicalized operands ascending by [`sig`].
fn order2(a: ScalarExpr, b: ScalarExpr) -> (ScalarExpr, ScalarExpr) {
    if sig(&a) <= sig(&b) {
        (a, b)
    } else {
        (b, a)
    }
}

/// Deterministic structural sort key for internal ordering only (need not equal
/// Fuel's §3a.2a key). The `0` prefix on graph ops vs the `1` prefix on leaves
/// makes ops sort before leaves; within each, the recursive form / leaf index
/// breaks ties. Pure function of structure, so two authorings of the same
/// expression sort alike and emit identical YAML.
fn sig(e: &ScalarExpr) -> String {
    match e {
        ScalarExpr::Add(a, b) => format!("0Add({},{})", sig(a), sig(b)),
        ScalarExpr::Sub(a, b) => format!("0Sub({},{})", sig(a), sig(b)),
        ScalarExpr::Mul(a, b) => format!("0Mul({},{})", sig(a), sig(b)),
        ScalarExpr::Div(a, b) => format!("0Div({},{})", sig(a), sig(b)),
        ScalarExpr::Binary(op, a, b) => format!("0B{op:?}({},{})", sig(a), sig(b)),
        ScalarExpr::Unary(op, x) => format!("0U{op:?}({})", sig(x)),
        ScalarExpr::Input(i) => format!("1I{i:03}"),
        ScalarExpr::Param(i) => format!("1P{i:03}"),
        ScalarExpr::Const(v) => format!("1C{v:?}"),
    }
}

fn walk(
    e: &ScalarExpr,
    is_root: bool,
    path: &[usize],
    extracts: &mut Vec<(String, String)>,
) -> Result<PatternNode, PatternError> {
    let consumers = if is_root { None } else { Some(1) };
    match e {
        ScalarExpr::Input(i) => Ok(PatternNode::Bind(*i)),
        // A bare Const or a standalone Param (not the scalar of an Add/Mul) has
        // no graph-Op form.
        ScalarExpr::Const(_) | ScalarExpr::Param(_) => Err(PatternError::ScalarParamUnsupported),
        ScalarExpr::Add(a, b) => scalar_binop("Add", "AddScalar", a, b, path, consumers, extracts),
        ScalarExpr::Mul(a, b) => scalar_binop("Mul", "MulScalar", a, b, path, consumers, extracts),
        ScalarExpr::Sub(a, b) => plain_binop("Sub", a, b, path, consumers, extracts),
        ScalarExpr::Div(a, b) => plain_binop("Div", a, b, path, consumers, extracts),
        // The non-infix binary fns have no scalar-param form in §4.1 (no
        // MaxScalar/etc.), so a `Param` operand is rejected (plain_binop).
        ScalarExpr::Binary(op, a, b) => {
            plain_binop(binary_name(*op), a, b, path, consumers, extracts)
        }
        ScalarExpr::Unary(op, x) => Ok(op_node(
            unary_name(*op),
            vec![walk(x, false, &child(path, 0), extracts)?],
            consumers,
        )),
    }
}

/// FKC §4.1 graph-`Op` name for a [`BinaryOp`].
fn binary_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Max => "Maximum",
        BinaryOp::Min => "Minimum",
        BinaryOp::Pow => "Pow",
        BinaryOp::Rem => "Rem",
    }
}

/// FKC §4.1 graph-`Op` name for a [`UnaryOp`]. Per §4.1 (B6/E2 resolution) bare
/// `Gelu` is the **tanh** approximation and `GeluErf` is the **exact erf** form;
/// our [`UnaryOp::Gelu`] lowers to exact erf (see `cuda.rs`), so it emits
/// `GeluErf` — emitting bare `Gelu` would misroute it to tanh-GELU subgraphs.
fn unary_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "Neg",
        UnaryOp::Abs => "Abs",
        UnaryOp::Sqr => "Sqr",
        UnaryOp::Sqrt => "Sqrt",
        UnaryOp::Rsqrt => "Rsqrt",
        UnaryOp::Recip => "Recip",
        UnaryOp::Exp => "Exp",
        UnaryOp::Log => "Log",
        UnaryOp::Tanh => "Tanh",
        UnaryOp::Sigmoid => "Sigmoid",
        UnaryOp::Relu => "Relu",
        UnaryOp::Erf => "Erf",
        UnaryOp::Gelu => "GeluErf",
        UnaryOp::Silu => "Silu",
        UnaryOp::Sin => "Sin",
        UnaryOp::Cos => "Cos",
        UnaryOp::Floor => "Floor",
        UnaryOp::Ceil => "Ceil",
        UnaryOp::Round => "Round",
        UnaryOp::Sign => "Sign",
        UnaryOp::Step => "Step",
    }
}

/// `Add`/`Mul` with exactly one runtime `Param` operand is the FKC §4.1
/// scalar-param op (`AddScalar`/`MulScalar`): the tensor is the sole operand and
/// the scalar is pulled via `extract` (`<path-to-this-node>.value`, §6). Otherwise
/// a plain binary op.
fn scalar_binop(
    tensor_op: &str,
    scalar_op: &str,
    a: &ScalarExpr,
    b: &ScalarExpr,
    path: &[usize],
    consumers: Option<u32>,
    extracts: &mut Vec<(String, String)>,
) -> Result<PatternNode, PatternError> {
    let (tensor, pidx) = match (as_param(a), as_param(b)) {
        // Two scalars leave no tensor operand — not an elementwise op.
        (Some(_), Some(_)) => return Err(PatternError::ScalarParamUnsupported),
        (None, Some(i)) => (a, i),
        (Some(i), None) => (b, i),
        (None, None) => return plain_binop(tensor_op, a, b, path, consumers, extracts),
    };
    let operand = walk(tensor, false, &child(path, 0), extracts)?;
    extracts.push((format!("param{pidx}"), format!("{}.value", path_str(path))));
    Ok(op_node(scalar_op, vec![operand], consumers))
}

/// A binary op of two tensors (`Sub`/`Div`, or `Add`/`Mul` of two tensors). A
/// `Param` here has no §4.1 form (there is no `SubScalar`/`DivScalar`), so reject.
fn plain_binop(
    name: &str,
    a: &ScalarExpr,
    b: &ScalarExpr,
    path: &[usize],
    consumers: Option<u32>,
    extracts: &mut Vec<(String, String)>,
) -> Result<PatternNode, PatternError> {
    if as_param(a).is_some() || as_param(b).is_some() {
        return Err(PatternError::ScalarParamUnsupported);
    }
    let ca = walk(a, false, &child(path, 0), extracts)?;
    let cb = walk(b, false, &child(path, 1), extracts)?;
    Ok(op_node(name, vec![ca, cb], consumers))
}

fn op_node(op: &str, operands: Vec<PatternNode>, consumers: Option<u32>) -> PatternNode {
    PatternNode::Op {
        op: op.to_string(),
        operands,
        consumers,
        extract: Vec::new(),
    }
}

fn as_param(e: &ScalarExpr) -> Option<u8> {
    if let ScalarExpr::Param(i) = e {
        Some(*i)
    } else {
        None
    }
}

fn child(path: &[usize], j: usize) -> Vec<usize> {
    let mut p = path.to_vec();
    p.push(j);
    p
}

/// FKC §6 path from the root to a node at `path`: `self` for the root, else
/// `operand(j0).operand(j1)…`.
fn path_str(path: &[usize]) -> String {
    if path.is_empty() {
        "self".to_string()
    } else {
        path.iter()
            .map(|j| format!("operand({j})"))
            .collect::<Vec<_>>()
            .join(".")
    }
}

fn collect_binds(node: &PatternNode, out: &mut BTreeSet<u8>) {
    match node {
        PatternNode::Bind(i) => {
            out.insert(*i);
        }
        PatternNode::Op { operands, .. } => {
            for o in operands {
                collect_binds(o, out);
            }
        }
    }
}

/// Serialize a pattern tree to the FKC `pattern:` YAML block (FKC §3).
#[must_use]
pub fn to_fkc(root: &PatternNode) -> String {
    let mut out = String::from("pattern:\n  root:\n");
    for line in node_lines(root) {
        out.push_str("    ");
        out.push_str(&line);
        out.push('\n');
    }
    out
}

/// Relative (indent-0) YAML lines for a node. Operand list items are prefixed
/// `  - ` (the first line) and `    ` (continuations), so applying a single base
/// indent in [`to_fkc`] yields the FKC §3 two-space-per-level block style.
fn node_lines(node: &PatternNode) -> Vec<String> {
    match node {
        PatternNode::Bind(i) => vec![format!("bind: {i}")],
        PatternNode::Op {
            op,
            operands,
            consumers,
            extract,
        } => {
            let mut lines = vec![format!("op: {op}")];
            if let Some(c) = consumers {
                lines.push(format!("consumers: {c}"));
            }
            if !extract.is_empty() {
                let entries = extract
                    .iter()
                    .map(|(f, p)| format!("{f}: \"{p}\""))
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(format!("extract: {{ {entries} }}"));
            }
            if !operands.is_empty() {
                lines.push("operands:".to_string());
                for o in operands {
                    for (k, ol) in node_lines(o).into_iter().enumerate() {
                        if k == 0 {
                            lines.push(format!("  - {ol}"));
                        } else {
                            lines.push(format!("    {ol}"));
                        }
                    }
                }
            }
            lines
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{input, konst, param, OpDef};
    use baracuda_kernels_types::ElementKind;

    #[test]
    fn binary_add_pattern() {
        let op = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
        let pat = derive_pattern(&op).unwrap();
        assert_eq!(
            to_fkc(&pat),
            "pattern:\n  root:\n    op: Add\n    operands:\n      - bind: 0\n      - bind: 1\n"
        );
    }

    #[test]
    fn nested_fma_pattern_matches_fkc_style() {
        // y = a*b + c  ->  Add(Mul(bind0,bind1), bind2); Mul is interior (consumers: 1).
        let op = OpDef::elementwise(
            "fma",
            3,
            &[ElementKind::F32],
            input(0) * input(1) + input(2),
        );
        let pat = derive_pattern(&op).unwrap();
        let expected = "\
pattern:
  root:
    op: Add
    operands:
      - op: Mul
        consumers: 1
        operands:
          - bind: 0
          - bind: 1
      - bind: 2
";
        assert_eq!(to_fkc(&pat), expected);
    }

    #[test]
    fn reused_input_yields_repeated_bind_identity() {
        // y = (a - b) * a  -> a (input 0) appears twice => repeated `bind: 0`.
        let op = OpDef::elementwise(
            "x",
            2,
            &[ElementKind::F32],
            (input(0) - input(1)) * input(0),
        );
        let fkc = to_fkc(&derive_pattern(&op).unwrap());
        assert_eq!(fkc.matches("bind: 0").count(), 2); // node-identity, for free
        assert_eq!(fkc.matches("bind: 1").count(), 1);
    }

    #[test]
    fn scalar_constant_is_rejected_for_now() {
        let op = OpDef::elementwise("addk", 1, &[ElementKind::F32], input(0) + konst(0.5));
        assert_eq!(derive_pattern(&op), Err(PatternError::ScalarParamUnsupported));
    }

    #[test]
    fn missing_input_index_is_rejected() {
        // declares 3 inputs but the body binds only 0 and 1.
        let op = OpDef::elementwise("x", 3, &[ElementKind::F32], input(0) + input(1));
        assert_eq!(
            derive_pattern(&op),
            Err(PatternError::BindSetMismatch {
                n_inputs: 3,
                got: vec![0, 1]
            })
        );
    }

    #[test]
    fn activation_chain_pattern() {
        // y = silu(a + b)  ->  root Silu, interior Add(bind0, bind1).
        let op = OpDef::elementwise(
            "silu_add",
            2,
            &[ElementKind::F32],
            (input(0) + input(1)).silu(),
        );
        let expected = "\
pattern:
  root:
    op: Silu
    operands:
      - op: Add
        consumers: 1
        operands:
          - bind: 0
          - bind: 1
";
        assert_eq!(to_fkc(&derive_pattern(&op).unwrap()), expected);
    }

    #[test]
    fn binary_fn_ops_emit_names_and_canonicalize() {
        // max(a,b) -> "Maximum"; commutative, so max(a,b) and max(b,a) converge.
        let ab = OpDef::elementwise("m", 2, &[ElementKind::F32], input(0).max(input(1)));
        let ba = OpDef::elementwise("m", 2, &[ElementKind::F32], input(1).max(input(0)));
        let fa = to_fkc(&derive_pattern(&ab).unwrap());
        assert!(fa.contains("op: Maximum"));
        assert_eq!(fa, to_fkc(&derive_pattern(&ba).unwrap()));
        // Pow is non-commutative — the op name emits, order is positional.
        let p = OpDef::elementwise("p", 2, &[ElementKind::F32], input(0).pow(input(1)));
        assert!(to_fkc(&derive_pattern(&p).unwrap()).contains("op: Pow"));
    }

    #[test]
    fn binary_fn_with_param_operand_is_rejected() {
        // max(x, p0) has no §4.1 scalar form (no MaxScalar) — rejected, not faked.
        let op = OpDef::elementwise("mp", 1, &[ElementKind::F32], input(0).max(param(0)));
        assert_eq!(derive_pattern(&op), Err(PatternError::ScalarParamUnsupported));
    }

    #[test]
    fn gelu_emits_geluerf_exact_flavor() {
        // UnaryOp::Gelu lowers to exact erf (cuda.rs); FKC §4.1 names that flavor
        // `GeluErf`, while bare `Gelu` is the tanh approximation. A fused gelu must
        // emit GeluErf or Fuel would misroute it to tanh-GELU subgraphs.
        let op = OpDef::elementwise(
            "gelu_add",
            2,
            &[ElementKind::F32],
            (input(0) + input(1)).gelu(),
        );
        let fkc = to_fkc(&derive_pattern(&op).unwrap());
        assert!(fkc.contains("op: GeluErf"));
        assert!(!fkc.contains("op: Gelu\n")); // never the bare (tanh) name
    }

    #[test]
    fn commutative_authorings_converge() {
        // a*b + c  and  c + a*b  are the same math written two ways; our canonical-
        // ization emits one identical pattern for both (reproducible output).
        // Matching itself doesn't require this — §3a.2a has Fuel canonicalize the
        // imported pattern too — but byte-identical emission keeps diffs clean.
        let ab_c = OpDef::elementwise(
            "f",
            3,
            &[ElementKind::F32],
            input(0) * input(1) + input(2),
        );
        let c_ab = OpDef::elementwise(
            "f",
            3,
            &[ElementKind::F32],
            input(2) + input(0) * input(1),
        );
        assert_eq!(
            to_fkc(&derive_pattern(&ab_c).unwrap()),
            to_fkc(&derive_pattern(&c_ab).unwrap())
        );
        // commutative `a + b` vs `b + a` likewise.
        let ab = OpDef::elementwise("g", 2, &[ElementKind::F32], input(0) + input(1));
        let ba = OpDef::elementwise("g", 2, &[ElementKind::F32], input(1) + input(0));
        assert_eq!(
            to_fkc(&derive_pattern(&ab).unwrap()),
            to_fkc(&derive_pattern(&ba).unwrap())
        );
    }

    #[test]
    fn add_scalar_root_self_extract() {
        // y = x + p0  ->  root AddScalar over bind 0, scalar via `self.value`.
        let op = OpDef::elementwise("add_p", 1, &[ElementKind::F32], input(0) + param(0));
        let fkc = to_fkc(&derive_pattern(&op).unwrap());
        assert!(fkc.contains("op: AddScalar"));
        assert!(fkc.contains("extract: { param0: \"self.value\" }"));
        assert!(fkc.contains("- bind: 0"));
    }

    #[test]
    fn affine_relu_pattern_with_extracts() {
        // y = relu(x * p0 + p1)  ->  Relu(AddScalar(MulScalar(x))); the two
        // scalars are pulled onto the root via paths from it (FKC §6).
        let op = OpDef::elementwise(
            "affine_relu",
            1,
            &[ElementKind::F32],
            (input(0) * param(0) + param(1)).relu(),
        );
        let fkc = to_fkc(&derive_pattern(&op).unwrap());
        assert!(fkc.contains("op: Relu"));
        assert!(fkc.contains("op: AddScalar"));
        assert!(fkc.contains("op: MulScalar"));
        assert!(fkc.contains(
            "extract: { param0: \"operand(0).operand(0).value\", param1: \"operand(0).value\" }"
        ));
    }
}
