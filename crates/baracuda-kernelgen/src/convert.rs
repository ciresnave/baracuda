//! CUDA / Slang → IR converters (IR-hub Phase 4, `convert` feature) — the
//! source-language side of the translation hub, built on **tree-sitter**.
//!
//! Instead of a hand-written parser (the pilot in `lift.rs`), these frontends
//! reuse the mature `tree-sitter-cuda` / `tree-sitter-slang` grammars: parse
//! source into an error-tolerant CST, recognize the expressible idioms
//! (grid-stride / dispatch elementwise, …) into the neutral IR (`OpDef`), and
//! keep whatever doesn't map as a typed [`LiftError`] residue
//! ([`LiftError::Unrecognized`] / [`LiftError::Inexpressible`]).
//! Error tolerance is a feature — unrecognized constructs stay as intact subtrees
//! we refuse rather than mis-lift; portability scales with the lift fraction.
//!
//! Because `tree-sitter-slang` is built on `tree-sitter-cpp`, a Slang elementwise
//! store parses into the SAME `assignment_expression` / `subscript_expression` /
//! `binary_expression` node shapes as CUDA — so ONE CST walk serves both, differing
//! only in naming conventions (`out`/`in` are HLSL keywords, so Slang uses
//! `output`/`input{k}`) and the kernel/residue markers.

use crate::ir::{Expr, OpDef, ReduceOp, ScalarExpr, UnaryOp};
use crate::lift::{LiftError, Lifted, binary_fn, unary_fn};
use baracuda_kernel_vocab::ElementKind;
use tree_sitter::{Node, Parser, Tree};

/// Parse CUDA source into a tree-sitter CST (error-tolerant; unrecognized
/// constructs become `ERROR`/unhandled subtrees rather than failing).
pub fn parse_cuda(src: &str) -> Option<Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_cuda::LANGUAGE.into())
        .ok()?;
    parser.parse(src, None)
}

/// Parse Slang source into a tree-sitter CST.
pub fn parse_slang(src: &str) -> Option<Tree> {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_slang::LANGUAGE_SLANG.into())
        .ok()?;
    parser.parse(src, None)
}

/// CUDA constructs that aren't IR-expressible — their presence means the kernel
/// is hand-optimized (shared mem, atomics, barriers, library calls) and belongs
/// in the source language. Shared across the elementwise and reduction lifters:
/// a *naive* reduction (plain accumulator loop) has none of these; the generator
/// supplies the cooperative/block machinery.
const CUDA_RESIDUE: &[&str] = &[
    "__shared__",
    "atomicAdd",
    "atomicCAS",
    "__syncthreads",
    "cublas",
    "cudnn",
    "printf",
    "asm",
    "cp.async",
    "__shfl",
];

/// Slang analogue of [`CUDA_RESIDUE`] — group-shared memory, barriers, atomics.
const SLANG_RESIDUE: &[&str] = &[
    "groupshared",
    "GroupMemoryBarrier",
    "DeviceMemoryBarrier",
    "AllMemoryBarrier",
    "InterlockedAdd",
    "InterlockedCompareExchange",
    "RWByteAddressBuffer",
];

/// Lift a grid-stride elementwise **CUDA** kernel into an [`OpDef`] by walking
/// its tree-sitter CST. Recognizes `out[i] = <expr>;` where the body is
/// arithmetic over `inK[i]`, literals, and CUDA math intrinsics; refuses
/// anything else as a typed [`LiftError`] residue.
pub fn lift_elementwise_cuda(
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
) -> Result<Lifted, LiftError> {
    if !src.contains("__global__") {
        return Err(LiftError::NotAKernel);
    }
    reject_markers(src, CUDA_RESIDUE)?;
    let tree = parse_cuda(src).ok_or(LiftError::NotAKernel)?;
    lift_store(&tree, src, name, dtypes, "out", "in")
}

/// Lift a dispatch-elementwise **Slang** compute kernel into an [`OpDef`].
/// Recognizes `output[i] = <expr>;` where the body is arithmetic over
/// `input{K}[i]`, literals, and math intrinsics (`out`/`in` are HLSL keywords,
/// so the buffer convention is `output`/`input{K}`). Refuses group-shared
/// memory, atomics, and barriers as residue.
pub fn lift_elementwise_slang(
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
) -> Result<Lifted, LiftError> {
    if !src.contains("numthreads") {
        return Err(LiftError::NotAKernel);
    }
    reject_markers(src, SLANG_RESIDUE)?;
    let tree = parse_slang(src).ok_or(LiftError::NotAKernel)?;
    lift_store(&tree, src, name, dtypes, "output", "input")
}

/// Refuse the first construct we cannot express in the IR — it belongs in the
/// source language (residue).
fn reject_markers(src: &str, markers: &[&str]) -> Result<(), LiftError> {
    for m in markers {
        if src.contains(m) {
            return Err(LiftError::Inexpressible((*m).to_string()));
        }
    }
    Ok(())
}

/// Shared tail: find the `{out_name}[i] = <rhs>` store, walk the RHS into a
/// `ScalarExpr` (`{in_prefix}{K}[i]` → `Input(K)`), and build the [`OpDef`].
fn lift_store(
    tree: &Tree,
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
    out_name: &str,
    in_prefix: &str,
) -> Result<Lifted, LiftError> {
    let (idx_var, rhs) =
        find_out_store(tree.root_node(), src, out_name).ok_or(LiftError::NotElementwise)?;
    let mut w = Walk {
        src,
        idx_var,
        in_prefix,
        max_input: None,
    };
    let body = w.expr(rhs)?;
    let n_inputs = w.max_input.map_or(0, |m| m + 1);
    Ok(Lifted {
        op: OpDef::elementwise(name, n_inputs, dtypes, Expr(body)),
        n_inputs,
    })
}

/// Lift a naive accumulator-loop full **reduction** CUDA kernel into an
/// [`OpDef`]. Recognizes the *reference* reduction a human writes —
/// `T acc = <init>; for (...) acc <reduce> <in-expr>; out[0] = acc;` — where the
/// per-element `<in-expr>` is arithmetic over `inK[i]`. The generator turns it
/// into an optimized block/grid cooperative reduce. Reduce-op mapping:
/// `acc += e` / `acc = acc + e` → [`ReduceOp::Sum`]; `acc *= e` / `acc = acc * e`
/// → [`ReduceOp::Prod`]; `acc = fmaxf(acc, e)` → [`ReduceOp::Max`]; `fminf` →
/// [`ReduceOp::Min`]. This recognizes the IDIOM, not the loop bounds — the
/// reduced extent is the caller's `StructureKey`, set at generate time.
pub fn lift_reduction_cuda(
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
) -> Result<Lifted, LiftError> {
    if !src.contains("__global__") {
        return Err(LiftError::NotAKernel);
    }
    reject_markers(src, CUDA_RESIDUE)?;
    let tree = parse_cuda(src).ok_or(LiftError::NotAKernel)?;
    lift_reduce_store(&tree, src, name, dtypes, "out", "in")
}

/// Lift a naive accumulator-loop full **reduction** Slang compute kernel (same
/// shape as [`lift_reduction_cuda`]; `output`/`input{K}` conventions).
pub fn lift_reduction_slang(
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
) -> Result<Lifted, LiftError> {
    if !src.contains("numthreads") {
        return Err(LiftError::NotAKernel);
    }
    reject_markers(src, SLANG_RESIDUE)?;
    let tree = parse_slang(src).ok_or(LiftError::NotAKernel)?;
    lift_reduce_store(&tree, src, name, dtypes, "output", "input")
}

/// Lift a naive inclusive-forward **scan** (prefix cumulative) CUDA kernel into
/// an [`OpDef`]. Recognizes `T acc = init; for(...) { acc <reduce> <pre>;
/// out[i] = acc; }` — the running accumulator stored INSIDE the loop at `out[i]`
/// (vs a reduction's `out[0]` after it). Monoid from the update (`+=` → cumsum,
/// `*=` → cumprod, `fmaxf` → cummax, `fminf` → cummin); `<pre>` is the
/// per-element pre-map (reuses the elementwise walk). Lifts as an inclusive
/// forward scan on `axis = 0` (the 1-D reading of the flat loop); a higher-rank
/// row-scan sets the axis via a matching-rank `StructureKey`. `Mean` is not a
/// monoid; `fmaxf`/`fminf`/`+=`/`*=` never produce it, so it cannot arise here.
pub fn lift_scan_cuda(src: &str, name: &str, dtypes: &[ElementKind]) -> Result<Lifted, LiftError> {
    if !src.contains("__global__") {
        return Err(LiftError::NotAKernel);
    }
    reject_markers(src, CUDA_RESIDUE)?;
    let tree = parse_cuda(src).ok_or(LiftError::NotAKernel)?;
    lift_scan_store(&tree, src, name, dtypes, "out", "in")
}

/// Lift a naive inclusive-forward **scan** Slang compute kernel (same shape as
/// [`lift_scan_cuda`]; `output`/`input{K}` conventions).
pub fn lift_scan_slang(src: &str, name: &str, dtypes: &[ElementKind]) -> Result<Lifted, LiftError> {
    if !src.contains("numthreads") {
        return Err(LiftError::NotAKernel);
    }
    reject_markers(src, SLANG_RESIDUE)?;
    let tree = parse_slang(src).ok_or(LiftError::NotAKernel)?;
    lift_scan_store(&tree, src, name, dtypes, "output", "input")
}

/// Lift a **CUDA** kernel of unknown op-class: try elementwise, then reduction,
/// then scan (the three are mutually exclusive by store shape).
pub fn lift_cuda(src: &str, name: &str, dtypes: &[ElementKind]) -> Result<Lifted, LiftError> {
    lift_elementwise_cuda(src, name, dtypes)
        .or_else(|_| lift_reduction_cuda(src, name, dtypes))
        .or_else(|_| lift_scan_cuda(src, name, dtypes))
}

/// Lift a **Slang** kernel of unknown op-class: elementwise, then reduction, then scan.
pub fn lift_slang(src: &str, name: &str, dtypes: &[ElementKind]) -> Result<Lifted, LiftError> {
    lift_elementwise_slang(src, name, dtypes)
        .or_else(|_| lift_reduction_slang(src, name, dtypes))
        .or_else(|_| lift_scan_slang(src, name, dtypes))
}

/// Shared scan tail: find the in-loop `{out_name}[i] = acc` running store, find
/// the reduce update of `acc`, walk its per-element `pre`, and build an inclusive
/// forward [`OpDef::scan`] on axis 0 (`post = reduced(0)`, the plain prefix).
fn lift_scan_store(
    tree: &Tree,
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
    out_name: &str,
    in_prefix: &str,
) -> Result<Lifted, LiftError> {
    let acc = find_running_out_store(tree.root_node(), src, out_name).ok_or_else(|| {
        LiftError::Unrecognized(format!("no running `{out_name}[i] = acc` scan store"))
    })?;
    let (op, pre_node) = find_accumulation(tree.root_node(), src, &acc).ok_or_else(|| {
        LiftError::Unrecognized(format!("no recognizable scan update of `{acc}`"))
    })?;
    let idx_var = first_input_index(pre_node, src, in_prefix)
        .ok_or_else(|| LiftError::Unrecognized("scan body reads no inK[idx]".into()))?;
    let mut w = Walk {
        src,
        idx_var,
        in_prefix,
        max_input: None,
    };
    let pre = w.expr(pre_node)?;
    let n_inputs = w.max_input.map_or(0, |m| m + 1);
    Ok(Lifted {
        op: OpDef::scan(
            name,
            n_inputs,
            dtypes,
            op,
            0,     // axis — 1-D reading; caller sets higher rank via the key
            false, // reverse
            false, // exclusive — naive `acc op= in; out[i] = acc` is inclusive
            Expr(pre),
            crate::ir::reduced(0),
        ),
        n_inputs,
    })
}

/// Find an in-loop `{out_name}[<identifier>] = <identifier>` store (the running
/// accumulator written at the scanned axis) and return the accumulator ident.
/// The identifier index (vs. a numeric literal) distinguishes a scan's running
/// store from a reduction's scalar `out[0] = acc`; the bare-identifier RHS
/// distinguishes it from an elementwise `out[i] = <expr>`.
fn find_running_out_store(node: Node, src: &str, out_name: &str) -> Option<String> {
    if node.kind() == "assignment_expression"
        && let (Some(lhs), Some(rhs)) = (
            node.child_by_field_name("left"),
            node.child_by_field_name("right"),
        )
        && lhs.kind() == "subscript_expression"
        && rhs.kind() == "identifier"
        && lhs
            .child_by_field_name("argument")
            .and_then(|a| a.utf8_text(src.as_bytes()).ok())
            == Some(out_name)
        && subscript_index(lhs, src).is_some()
    {
        return Some(rhs.utf8_text(src.as_bytes()).ok()?.to_string());
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(hit) = find_running_out_store(child, src, out_name) {
            return Some(hit);
        }
    }
    None
}

/// Shared reduction tail: find `{out_name}[<lit>] = acc`, find the reduce update
/// of `acc`, walk the per-element body, and build a last-axis
/// [`OpDef::reduction`].
fn lift_reduce_store(
    tree: &Tree,
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
    out_name: &str,
    in_prefix: &str,
) -> Result<Lifted, LiftError> {
    let acc = find_scalar_out_store(tree.root_node(), src, out_name).ok_or_else(|| {
        LiftError::Unrecognized(format!("no scalar `{out_name}[..] = acc` store"))
    })?;
    let (op, body_node) = find_accumulation(tree.root_node(), src, &acc).ok_or_else(|| {
        LiftError::Unrecognized(format!("no recognizable reduce update of `{acc}`"))
    })?;
    let idx_var = first_input_index(body_node, src, in_prefix)
        .ok_or_else(|| LiftError::Unrecognized("reduction body reads no inK[idx]".into()))?;
    let mut w = Walk {
        src,
        idx_var,
        in_prefix,
        max_input: None,
    };
    let body = w.expr(body_node)?;
    let n_inputs = w.max_input.map_or(0, |m| m + 1);
    Ok(Lifted {
        op: OpDef::reduction(name, n_inputs, dtypes, Expr(body), op),
        n_inputs,
    })
}

/// Find a `{out_name}[<number>] = <ident>` store and return the stored
/// accumulator identifier. The numeric index (vs. an identifier) is what
/// distinguishes a scalar reduction store from an elementwise `out[i] = ...`.
fn find_scalar_out_store(node: Node, src: &str, out_name: &str) -> Option<String> {
    if node.kind() == "assignment_expression"
        && let (Some(lhs), Some(rhs)) = (
            node.child_by_field_name("left"),
            node.child_by_field_name("right"),
        )
        && lhs.kind() == "subscript_expression"
        && rhs.kind() == "identifier"
        && lhs
            .child_by_field_name("argument")
            .and_then(|a| a.utf8_text(src.as_bytes()).ok())
            == Some(out_name)
        && let Some(indices) = lhs.child_by_field_name("indices")
    {
        let mut c = indices.walk();
        let kids: Vec<Node> = indices.named_children(&mut c).collect();
        if kids.len() == 1 && kids[0].kind() == "number_literal" {
            return Some(rhs.utf8_text(src.as_bytes()).ok()?.to_string());
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(hit) = find_scalar_out_store(child, src, out_name) {
            return Some(hit);
        }
    }
    None
}

/// Find the reduce-update assignment of `acc` — `acc += e`, `acc *= e`,
/// `acc = acc <+|*> e`, or `acc = fmaxf/fminf(acc, e)` — returning the reduce op
/// and the per-element body node `e`.
fn find_accumulation<'t>(node: Node<'t>, src: &str, acc: &str) -> Option<(ReduceOp, Node<'t>)> {
    if node.kind() == "assignment_expression"
        && let (Some(lhs), Some(op), Some(rhs)) = (
            node.child_by_field_name("left"),
            node.child_by_field_name("operator"),
            node.child_by_field_name("right"),
        )
        && lhs.kind() == "identifier"
        && lhs.utf8_text(src.as_bytes()).ok() == Some(acc)
    {
        match op.utf8_text(src.as_bytes()).unwrap_or("") {
            "+=" => return Some((ReduceOp::Sum, rhs)),
            "*=" => return Some((ReduceOp::Prod, rhs)),
            "=" => {
                if let Some(hit) = classify_reduce_rhs(rhs, src, acc) {
                    return Some(hit);
                }
            }
            _ => {}
        }
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(hit) = find_accumulation(child, src, acc) {
            return Some(hit);
        }
    }
    None
}

/// Classify the RHS of `acc = <rhs>` as a reduce update: `acc <+|*> e` (acc on
/// either side) or `fmaxf/fminf(acc, e)`. Returns the op + the body `e`.
fn classify_reduce_rhs<'t>(rhs: Node<'t>, src: &str, acc: &str) -> Option<(ReduceOp, Node<'t>)> {
    let is_acc =
        |n: Node| n.kind() == "identifier" && n.utf8_text(src.as_bytes()).ok() == Some(acc);
    match rhs.kind() {
        "parenthesized_expression" => classify_reduce_rhs(rhs.named_child(0)?, src, acc),
        "binary_expression" => {
            let l = rhs.child_by_field_name("left")?;
            let r = rhs.child_by_field_name("right")?;
            let other = if is_acc(l) {
                r
            } else if is_acc(r) {
                l
            } else {
                return None;
            };
            match rhs
                .child_by_field_name("operator")?
                .utf8_text(src.as_bytes())
                .unwrap_or("")
            {
                "+" => Some((ReduceOp::Sum, other)),
                "*" => Some((ReduceOp::Prod, other)),
                _ => None,
            }
        }
        "call_expression" => {
            let fname = rhs
                .child_by_field_name("function")?
                .utf8_text(src.as_bytes())
                .ok()?;
            let args_node = rhs.child_by_field_name("arguments")?;
            let mut c = args_node.walk();
            let args: Vec<Node> = args_node.named_children(&mut c).collect();
            if args.len() != 2 {
                return None;
            }
            let other = if is_acc(args[0]) {
                args[1]
            } else if is_acc(args[1]) {
                args[0]
            } else {
                return None;
            };
            let op = match fname {
                "fmaxf" | "fmax" | "max" => ReduceOp::Max,
                "fminf" | "fmin" | "min" => ReduceOp::Min,
                _ => return None,
            };
            Some((op, other))
        }
        _ => None,
    }
}

/// The index identifier of the first `{in_prefix}{K}[idx]` read in `node` — the
/// reduced-axis loop variable the body iterates over.
fn first_input_index(node: Node, src: &str, in_prefix: &str) -> Option<String> {
    if node.kind() == "subscript_expression"
        && let Some(base) = node
            .child_by_field_name("argument")
            .and_then(|a| a.utf8_text(src.as_bytes()).ok())
        && base
            .strip_prefix(in_prefix)
            .and_then(|d| d.parse::<u8>().ok())
            .is_some()
    {
        return subscript_index(node, src);
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(idx) = first_input_index(child, src, in_prefix) {
            return Some(idx);
        }
    }
    None
}

/// Find the `{out_name}[<idx>] = <rhs>` store (an `assignment_expression` whose
/// LHS is a `subscript_expression` on `out_name`) and return `(idx_var, rhs)`.
fn find_out_store<'t>(node: Node<'t>, src: &str, out_name: &str) -> Option<(String, Node<'t>)> {
    if node.kind() == "assignment_expression"
        && let (Some(lhs), Some(rhs)) = (
            node.child_by_field_name("left"),
            node.child_by_field_name("right"),
        )
        && lhs.kind() == "subscript_expression"
        && lhs
            .child_by_field_name("argument")
            .and_then(|a| a.utf8_text(src.as_bytes()).ok())
            == Some(out_name)
        && let Some(idx) = subscript_index(lhs, src)
    {
        return Some((idx, rhs));
    }
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if let Some(hit) = find_out_store(child, src, out_name) {
            return Some(hit);
        }
    }
    None
}

/// The single index identifier inside a `subscript_expression`'s `indices`, or
/// `None` if it isn't a lone identifier.
fn subscript_index(sub: Node, src: &str) -> Option<String> {
    let indices = sub.child_by_field_name("indices")?;
    let mut cursor = indices.walk();
    let kids: Vec<Node> = indices.named_children(&mut cursor).collect();
    if kids.len() != 1 || kids[0].kind() != "identifier" {
        return None;
    }
    Some(kids[0].utf8_text(src.as_bytes()).ok()?.to_string())
}

struct Walk<'a> {
    src: &'a str,
    idx_var: String,
    in_prefix: &'a str,
    max_input: Option<u8>,
}

impl<'a> Walk<'a> {
    fn text(&self, n: Node) -> &'a str {
        n.utf8_text(self.src.as_bytes()).unwrap_or("")
    }

    fn expr(&mut self, n: Node) -> Result<ScalarExpr, LiftError> {
        match n.kind() {
            "parenthesized_expression" => {
                let inner = n
                    .named_child(0)
                    .ok_or_else(|| LiftError::Unrecognized("()".into()))?;
                self.expr(inner)
            }
            "binary_expression" => {
                let l = self.field(n, "left")?;
                let r = self.field(n, "right")?;
                let op = self.field(n, "operator")?;
                let (le, re) = (self.expr(l)?, self.expr(r)?);
                Ok(match self.text(op) {
                    "+" => ScalarExpr::Add(Box::new(le), Box::new(re)),
                    "-" => ScalarExpr::Sub(Box::new(le), Box::new(re)),
                    "*" => ScalarExpr::Mul(Box::new(le), Box::new(re)),
                    "/" => ScalarExpr::Div(Box::new(le), Box::new(re)),
                    other => return Err(LiftError::Unrecognized(format!("operator '{other}'"))),
                })
            }
            "unary_expression" => {
                let op = self.field(n, "operator")?;
                let a = self.field(n, "argument")?;
                match self.text(op) {
                    "-" => Ok(ScalarExpr::Unary(UnaryOp::Neg, Box::new(self.expr(a)?))),
                    other => Err(LiftError::Unrecognized(format!("unary '{other}'"))),
                }
            }
            "subscript_expression" => {
                let base = self.text(self.field(n, "argument")?);
                let k: u8 = base
                    .strip_prefix(self.in_prefix)
                    .and_then(|d| d.parse().ok())
                    .ok_or_else(|| {
                        LiftError::Unrecognized(format!(
                            "read '{base}[..]' (not {}K)",
                            self.in_prefix
                        ))
                    })?;
                let idx = subscript_index(n, self.src)
                    .ok_or_else(|| LiftError::Unrecognized("index".into()))?;
                if idx != self.idx_var {
                    return Err(LiftError::Unrecognized(format!(
                        "non-elementwise index [{idx}] (expected [{}])",
                        self.idx_var
                    )));
                }
                self.max_input = Some(self.max_input.map_or(k, |m| m.max(k)));
                Ok(ScalarExpr::Input(k))
            }
            "number_literal" => {
                let t = self
                    .text(n)
                    .trim_end_matches(['f', 'F', 'l', 'L', 'u', 'U']);
                t.parse::<f64>()
                    .map(ScalarExpr::Const)
                    .map_err(|_| LiftError::Unrecognized(format!("literal {t}")))
            }
            "call_expression" => {
                let fname = self.text(self.field(n, "function")?);
                let args_node = self.field(n, "arguments")?;
                let mut cursor = args_node.walk();
                let args: Vec<Node> = args_node.named_children(&mut cursor).collect();
                match args.len() {
                    1 => {
                        let op = unary_fn(fname)
                            .ok_or_else(|| LiftError::Unrecognized(format!("call '{fname}(_)'")))?;
                        Ok(ScalarExpr::Unary(op, Box::new(self.expr(args[0])?)))
                    }
                    2 => {
                        let op = binary_fn(fname).ok_or_else(|| {
                            LiftError::Unrecognized(format!("call '{fname}(_,_)'"))
                        })?;
                        Ok(ScalarExpr::Binary(
                            op,
                            Box::new(self.expr(args[0])?),
                            Box::new(self.expr(args[1])?),
                        ))
                    }
                    n => Err(LiftError::Unrecognized(format!(
                        "call '{fname}' with {n} args"
                    ))),
                }
            }
            "identifier" => Err(LiftError::Unrecognized(format!(
                "identifier '{}'",
                self.text(n)
            ))),
            other => Err(LiftError::Unrecognized(format!("node '{other}'"))),
        }
    }

    fn field<'t>(&self, n: Node<'t>, name: &str) -> Result<Node<'t>, LiftError> {
        n.child_by_field_name(name)
            .ok_or_else(|| LiftError::Unrecognized(format!("missing {name} in {}", n.kind())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32: &[ElementKind] = &[ElementKind::F32];

    fn cuda_body(src: &str) -> ScalarExpr {
        lift_elementwise_cuda(src, "x", F32).unwrap().op.body
    }

    const SLANG_MUL: &str = "StructuredBuffer<float> input0;\n\
        StructuredBuffer<float> input1;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256, 1, 1)]\n\
        void mul(uint3 tid : SV_DispatchThreadID) {\n\
            uint i = tid.x;\n\
            output[i] = input0[i] * input1[i];\n\
        }";

    fn reduce_op(l: &Lifted) -> Option<ReduceOp> {
        if let crate::ir::Access::Reduction { op, .. } = &l.op.access {
            Some(*op)
        } else {
            None
        }
    }

    fn scan_info(l: &Lifted) -> Option<(ReduceOp, &ScalarExpr, bool, bool)> {
        if let crate::ir::Access::Scan {
            op,
            pre,
            reverse,
            exclusive,
            ..
        } = &l.op.access
        {
            Some((*op, pre, *reverse, *exclusive))
        } else {
            None
        }
    }

    #[test]
    fn grammars_load_and_parse() {
        assert!(parse_cuda("__global__ void k(float* out){ out[i]=0.0f; }").is_some());
        assert!(parse_slang("float4 f(float x) { return x; }").is_some());
    }

    #[test]
    fn cuda_lifts_fused_multiply_add() {
        let src = "__global__ void mul(const float* in0, const float* in1, const float* in2, float* out, long long n) {\n\
            long long i = blockIdx.x*blockDim.x + threadIdx.x;\n\
            for (; i < n; i += gridDim.x*blockDim.x) { out[i] = in0[i] * in1[i] + in2[i]; }\n}";
        let lifted = lift_elementwise_cuda(src, "fma", F32).unwrap();
        assert_eq!(lifted.n_inputs, 3);
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Add(
                Box::new(ScalarExpr::Mul(
                    Box::new(ScalarExpr::Input(0)),
                    Box::new(ScalarExpr::Input(1)),
                )),
                Box::new(ScalarExpr::Input(2)),
            )
        );
    }

    #[test]
    fn cuda_lifts_unary_intrinsic() {
        let src = "__global__ void k(const float* in0, float* out, long long n){ out[i] = __expf(in0[i]); }";
        assert_eq!(
            cuda_body(src),
            ScalarExpr::Unary(UnaryOp::Exp, Box::new(ScalarExpr::Input(0)))
        );
    }

    #[test]
    fn cuda_fmaxf_is_ieee_not_torch_max() {
        let src = "__global__ void k(const float* in0, float* out, long long n){ out[i] = fmaxf(in0[i], 0.0f); }";
        assert_eq!(
            cuda_body(src),
            ScalarExpr::Binary(
                crate::ir::BinaryOp::FmaxIeee,
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Const(0.0)),
            )
        );
    }

    #[test]
    fn cuda_precedence_from_the_grammar() {
        let src = "__global__ void k(const float* in0, const float* in1, const float* in2, float* out){ out[i] = in0[i] + in1[i]*in2[i]; }";
        assert_eq!(
            cuda_body(src),
            ScalarExpr::Add(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Mul(
                    Box::new(ScalarExpr::Input(1)),
                    Box::new(ScalarExpr::Input(2)),
                )),
            )
        );
    }

    #[test]
    fn cuda_refuses_residue() {
        let neigh = "__global__ void k(const float* in0, float* out){ out[i] = in0[i+1]; }";
        assert!(matches!(
            lift_elementwise_cuda(neigh, "x", F32),
            Err(LiftError::Unrecognized(_))
        ));
        let smem = "__global__ void k(float* out){ __shared__ float s[32]; out[i] = s[0]; }";
        assert!(matches!(
            lift_elementwise_cuda(smem, "x", F32),
            Err(LiftError::Unrecognized(_))
        ));
    }

    #[test]
    fn slang_lifts_multiply() {
        let lifted = lift_elementwise_slang(SLANG_MUL, "slang_mul", F32).unwrap();
        assert_eq!(lifted.n_inputs, 2);
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(1)),
            )
        );
    }

    #[test]
    fn slang_refuses_groupshared() {
        let src = "groupshared float s[256];\n\
            [numthreads(256,1,1)]\n\
            void k(uint3 tid : SV_DispatchThreadID) { output[tid.x] = s[0]; }";
        assert!(matches!(
            lift_elementwise_slang(src, "x", F32),
            Err(LiftError::Unrecognized(_))
        ));
    }

    #[test]
    fn slang_and_cuda_lift_to_the_same_ir() {
        // A CUDA and a Slang kernel for the same math lift to the SAME OpDef body.
        let cuda = "__global__ void mul(const float* in0, const float* in1, float* out, long long n){ out[i] = in0[i] * in1[i]; }";
        let c = lift_elementwise_cuda(cuda, "m", F32).unwrap();
        let s = lift_elementwise_slang(SLANG_MUL, "m", F32).unwrap();
        assert_eq!(c.op.body, s.op.body);
    }

    #[test]
    fn cuda_lifts_sum_reduction() {
        let src = "__global__ void sum(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++) acc += in0[i]; out[0] = acc; }";
        let lifted = lift_reduction_cuda(src, "sum", F32).unwrap();
        assert_eq!(lifted.n_inputs, 1);
        assert_eq!(lifted.op.body, ScalarExpr::Input(0));
        assert_eq!(reduce_op(&lifted), Some(ReduceOp::Sum));
    }

    #[test]
    fn cuda_lifts_sum_of_squares() {
        let src = "__global__ void ss(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++) acc += in0[i]*in0[i]; out[0] = acc; }";
        let lifted = lift_reduction_cuda(src, "ss", F32).unwrap();
        assert_eq!(
            lifted.op.body,
            ScalarExpr::Mul(
                Box::new(ScalarExpr::Input(0)),
                Box::new(ScalarExpr::Input(0))
            )
        );
        assert_eq!(reduce_op(&lifted), Some(ReduceOp::Sum));
    }

    #[test]
    fn cuda_lifts_prod_and_acc_plus_forms() {
        let prod = "__global__ void pr(const float* in0, float* out, long long n){ float acc = 1.0f; for (long long i=0;i<n;i++) acc *= in0[i]; out[0] = acc; }";
        assert_eq!(
            reduce_op(&lift_reduction_cuda(prod, "pr", F32).unwrap()),
            Some(ReduceOp::Prod)
        );
        let sum2 = "__global__ void s2(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++) acc = acc + in0[i]; out[0] = acc; }";
        assert_eq!(
            reduce_op(&lift_reduction_cuda(sum2, "s2", F32).unwrap()),
            Some(ReduceOp::Sum)
        );
    }

    #[test]
    fn cuda_lifts_max_via_fmaxf() {
        let src = "__global__ void mx(const float* in0, float* out, long long n){ float acc = in0[0]; for (long long i=1;i<n;i++) acc = fmaxf(acc, in0[i]); out[0] = acc; }";
        let lifted = lift_reduction_cuda(src, "mx", F32).unwrap();
        assert_eq!(lifted.op.body, ScalarExpr::Input(0));
        assert_eq!(reduce_op(&lifted), Some(ReduceOp::Max));
    }

    #[test]
    fn elementwise_and_reduction_lifters_dont_cross() {
        // The elementwise lifter refuses a reduction (scalar store), and vice versa.
        let reduction = "__global__ void sum(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++) acc += in0[i]; out[0] = acc; }";
        assert!(lift_elementwise_cuda(reduction, "x", F32).is_err());
        let elementwise = "__global__ void k(const float* in0, float* out){ out[i] = in0[i]; }";
        assert!(lift_reduction_cuda(elementwise, "x", F32).is_err());
        // The unified entry point handles either class.
        assert!(lift_cuda(reduction, "x", F32).is_ok());
        assert!(lift_cuda(elementwise, "x", F32).is_ok());
    }

    #[test]
    fn slang_lifts_sum_reduction() {
        let src = "StructuredBuffer<float> input0;\n\
            RWStructuredBuffer<float> output;\n\
            [numthreads(256,1,1)]\n\
            void rsum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++) acc += input0[i]; output[0] = acc; }";
        let lifted = lift_reduction_slang(src, "rsum", F32).unwrap();
        assert_eq!(lifted.op.body, ScalarExpr::Input(0));
        assert_eq!(reduce_op(&lifted), Some(ReduceOp::Sum));
    }

    #[test]
    fn cuda_lifts_cumsum() {
        let src = "__global__ void cs(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++){ acc += in0[i]; out[i] = acc; } }";
        let lifted = lift_scan_cuda(src, "cs", F32).unwrap();
        assert_eq!(lifted.n_inputs, 1);
        let (op, pre, rev, exc) = scan_info(&lifted).unwrap();
        assert_eq!(op, ReduceOp::Sum);
        assert_eq!(pre, &ScalarExpr::Input(0));
        assert!(!rev && !exc);
    }

    #[test]
    fn cuda_lifts_cumprod_and_cummax() {
        let cp = "__global__ void cp(const float* in0, float* out, long long n){ float acc = 1.0f; for (long long i=0;i<n;i++){ acc *= in0[i]; out[i] = acc; } }";
        assert_eq!(
            scan_info(&lift_scan_cuda(cp, "cp", F32).unwrap())
                .unwrap()
                .0,
            ReduceOp::Prod
        );
        let cm = "__global__ void cm(const float* in0, float* out, long long n){ float acc = in0[0]; for (long long i=0;i<n;i++){ acc = fmaxf(acc, in0[i]); out[i] = acc; } }";
        assert_eq!(
            scan_info(&lift_scan_cuda(cm, "cm", F32).unwrap())
                .unwrap()
                .0,
            ReduceOp::Max
        );
    }

    #[test]
    fn scan_reduction_elementwise_are_disjoint() {
        // out[i]=acc (running) → scan; out[0]=acc → reduction; out[i]=expr → elementwise.
        let scan = "__global__ void cs(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++){ acc += in0[i]; out[i] = acc; } }";
        assert!(lift_scan_cuda(scan, "x", F32).is_ok());
        assert!(lift_reduction_cuda(scan, "x", F32).is_err());
        assert!(lift_elementwise_cuda(scan, "x", F32).is_err());
        assert!(matches!(
            lift_cuda(scan, "x", F32).unwrap().op.access,
            crate::ir::Access::Scan { .. }
        ));
    }

    #[test]
    fn slang_lifts_cumsum() {
        let src = "StructuredBuffer<float> input0;\n\
            RWStructuredBuffer<float> output;\n\
            [numthreads(256,1,1)]\n\
            void csum(uint3 tid : SV_DispatchThreadID){ float acc = 0.0f; for (uint i=0;i<256;i++){ acc += input0[i]; output[i] = acc; } }";
        let lifted = lift_scan_slang(src, "csum", F32).unwrap();
        assert_eq!(scan_info(&lifted).unwrap().0, ReduceOp::Sum);
    }
}
