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
//! Pure-tensor elementwise bodies only. Scalar constants
//! ([`ScalarExpr::Const`]) map to Fuel's `AddScalar`/`MulScalar`/… graph ops
//! with an `extract:` path — deferred until the op-attribute bridge lands; the
//! walker rejects a `Const`-bearing body for now. Commutative operand ordering
//! is emitted in the body's natural order, pending Fuel's answer on whether it
//! canonicalizes commutative operands before matching (review item E1); when
//! that lands it is a one-line change to the canonical order here.

use crate::ir::{Access, OpDef, ScalarExpr, UnaryOp};
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
    let mut extracts = Vec::new();
    let mut root = walk(&op.body, true, &[], &mut extracts)?;
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
        ScalarExpr::Unary(op, x) => Ok(op_node(
            unary_name(*op),
            vec![walk(x, false, &child(path, 0), extracts)?],
            consumers,
        )),
    }
}

/// FKC §4.1 graph-`Op` name for a [`UnaryOp`]. (The bare-`Gelu` flavor mapping is
/// pending Fuel review item E2.)
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
        UnaryOp::Gelu => "Gelu",
        UnaryOp::Silu => "Silu",
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
