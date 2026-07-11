//! CUDA / Slang → IR converters (IR-hub Phase 4, `convert` feature) — the
//! source-language side of the translation hub, built on **tree-sitter**.
//!
//! Instead of a hand-written parser (the pilot in `lift.rs`), these frontends
//! reuse the mature `tree-sitter-cuda` / `tree-sitter-slang` grammars: parse
//! source into an error-tolerant CST, recognize the expressible idioms
//! (grid-stride / dispatch elementwise, …) into the neutral IR (`OpDef`), and
//! keep whatever doesn't map as source **residue** ([`LiftError::Residue`]).
//! Error tolerance is a feature — unrecognized constructs stay as intact subtrees
//! we refuse rather than mis-lift; portability scales with the lift fraction.
//!
//! Because `tree-sitter-slang` is built on `tree-sitter-cpp`, a Slang elementwise
//! store parses into the SAME `assignment_expression` / `subscript_expression` /
//! `binary_expression` node shapes as CUDA — so ONE CST walk serves both, differing
//! only in naming conventions (`out`/`in` are HLSL keywords, so Slang uses
//! `output`/`input{k}`) and the kernel/residue markers.

use crate::ir::{Expr, OpDef, ScalarExpr, UnaryOp};
use crate::lift::{LiftError, Lifted, binary_fn, unary_fn};
use baracuda_kernels_types::ElementKind;
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

/// Lift a grid-stride elementwise **CUDA** kernel into an [`OpDef`] by walking
/// its tree-sitter CST. Recognizes `out[i] = <expr>;` where the body is
/// arithmetic over `inK[i]`, literals, and CUDA math intrinsics; refuses
/// anything else as [`LiftError::Residue`].
pub fn lift_elementwise_cuda(
    src: &str,
    name: &str,
    dtypes: &[ElementKind],
) -> Result<Lifted, LiftError> {
    if !src.contains("__global__") {
        return Err(LiftError::NotAKernel);
    }
    reject_markers(
        src,
        &[
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
        ],
    )?;
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
    reject_markers(
        src,
        &[
            "groupshared",
            "GroupMemoryBarrier",
            "DeviceMemoryBarrier",
            "AllMemoryBarrier",
            "InterlockedAdd",
            "InterlockedCompareExchange",
            "RWByteAddressBuffer",
        ],
    )?;
    let tree = parse_slang(src).ok_or(LiftError::NotAKernel)?;
    lift_store(&tree, src, name, dtypes, "output", "input")
}

/// Refuse the first construct we cannot express in the IR — it belongs in the
/// source language (residue).
fn reject_markers(src: &str, markers: &[&str]) -> Result<(), LiftError> {
    for m in markers {
        if src.contains(m) {
            return Err(LiftError::Residue((*m).to_string()));
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
                    .ok_or_else(|| LiftError::Residue("()".into()))?;
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
                    other => return Err(LiftError::Residue(format!("operator '{other}'"))),
                })
            }
            "unary_expression" => {
                let op = self.field(n, "operator")?;
                let a = self.field(n, "argument")?;
                match self.text(op) {
                    "-" => Ok(ScalarExpr::Unary(UnaryOp::Neg, Box::new(self.expr(a)?))),
                    other => Err(LiftError::Residue(format!("unary '{other}'"))),
                }
            }
            "subscript_expression" => {
                let base = self.text(self.field(n, "argument")?);
                let k: u8 = base
                    .strip_prefix(self.in_prefix)
                    .and_then(|d| d.parse().ok())
                    .ok_or_else(|| {
                        LiftError::Residue(format!("read '{base}[..]' (not {}K)", self.in_prefix))
                    })?;
                let idx = subscript_index(n, self.src)
                    .ok_or_else(|| LiftError::Residue("index".into()))?;
                if idx != self.idx_var {
                    return Err(LiftError::Residue(format!(
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
                    .map_err(|_| LiftError::Residue(format!("literal {t}")))
            }
            "call_expression" => {
                let fname = self.text(self.field(n, "function")?);
                let args_node = self.field(n, "arguments")?;
                let mut cursor = args_node.walk();
                let args: Vec<Node> = args_node.named_children(&mut cursor).collect();
                match args.len() {
                    1 => {
                        let op = unary_fn(fname)
                            .ok_or_else(|| LiftError::Residue(format!("call '{fname}(_)'")))?;
                        Ok(ScalarExpr::Unary(op, Box::new(self.expr(args[0])?)))
                    }
                    2 => {
                        let op = binary_fn(fname)
                            .ok_or_else(|| LiftError::Residue(format!("call '{fname}(_,_)'")))?;
                        Ok(ScalarExpr::Binary(
                            op,
                            Box::new(self.expr(args[0])?),
                            Box::new(self.expr(args[1])?),
                        ))
                    }
                    n => Err(LiftError::Residue(format!("call '{fname}' with {n} args"))),
                }
            }
            "identifier" => Err(LiftError::Residue(format!("identifier '{}'", self.text(n)))),
            other => Err(LiftError::Residue(format!("node '{other}'"))),
        }
    }

    fn field<'t>(&self, n: Node<'t>, name: &str) -> Result<Node<'t>, LiftError> {
        n.child_by_field_name(name)
            .ok_or_else(|| LiftError::Residue(format!("missing {name} in {}", n.kind())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const F32: &[ElementKind] = &[ElementKind::F32];

    fn cuda_body(src: &str) -> ScalarExpr {
        lift_elementwise_cuda(src, "x", F32).unwrap().op.body
    }

    // --- CUDA ---

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
            Err(LiftError::Residue(_))
        ));
        let smem = "__global__ void k(float* out){ __shared__ float s[32]; out[i] = s[0]; }";
        assert!(matches!(
            lift_elementwise_cuda(smem, "x", F32),
            Err(LiftError::Residue(_))
        ));
    }

    // --- Slang (the SAME walk, different conventions) ---

    const SLANG_MUL: &str = "StructuredBuffer<float> input0;\n\
        StructuredBuffer<float> input1;\n\
        RWStructuredBuffer<float> output;\n\
        [numthreads(256, 1, 1)]\n\
        void mul(uint3 tid : SV_DispatchThreadID) {\n\
            uint i = tid.x;\n\
            output[i] = input0[i] * input1[i];\n\
        }";

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
            Err(LiftError::Residue(_))
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
    fn round_trip_reemits_to_cuda_and_cpuc() {
        use crate::{CpuC, Cuda, generate};
        use baracuda_kernels_types::{ArchSku, OpCategory, OperandDesc, structure_key};
        // Lift a Slang kernel, re-emit to CUDA AND portable-C — cross-language port.
        let lifted = lift_elementwise_slang(SLANG_MUL, "ported", F32).unwrap();
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 4);
        let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
        let cuda = generate(&lifted.op, &key, &Cuda);
        let cpuc = generate(&lifted.op, &key, &CpuC);
        assert!(cuda.source.contains("in0[") && cuda.source.contains("in1["));
        assert!(cpuc.source.contains("for (long long i"));
    }
}
