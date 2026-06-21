//! JIT-on-request synthesis — Baracuda as the **synthesizer** (Kernel-Seam §5).
//!
//! The division of labor is fixed by the constitution (§5.1): **Fuel is the
//! strategist** — it chooses *which* primitive-subgraph region to fuse, *when*,
//! and whether to *adopt* the result (cost-gated); **Baracuda is the synthesizer**
//! — it builds the best kernel for the **Fuel-chosen** region and returns it. No
//! backend-side opportunity-finding: we never scan a graph to pick regions, we
//! only synthesize the one we're handed.
//!
//! A [`JitRequest`] carries that region (a graph-`Op` subgraph, the same shape
//! [`derive_pattern`] emits — read in reverse), the operand projection that keys
//! the schedule, and a target. [`synthesize`] turns it into a [`JitResponse`] =
//! `(kernel + FKC contract + recipe)`, exactly the §5 shape. The heavy lifting
//! reuses the AOT generator ([`generate`], [`contract`], [`derive_pattern`]); the
//! only new step is [`region_to_op`] (region → op IR) and the on-demand
//! [`Compiler`] seam.
//!
//! # Scope (increment 1)
//!
//! The elementwise-epilogue vocabulary the IR already covers ([`ScalarExpr`]):
//! `Add`/`Sub`/`Mul`/`Div`, the scalar-param ops `AddScalar`/`MulScalar`, and the
//! unary math/activations. The on-demand compiler is behind a trait with a stub
//! impl; the real nvrtc backend, the FFI wire surface (reconciling these Rust
//! types with Fuel's `JitRequest`/`JitResponse`), an inward e-graph optimizer
//! (§5.1 permits it), and the telemetry trigger are the growth path.

use crate::contract::contract;
use crate::ir::{Access, OpDef, ScalarExpr, UnaryOp};
use crate::pattern::{derive_pattern, to_fkc, PatternError, PatternNode};
use crate::{generate, Backend, Cuda};
use baracuda_kernels_types::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

/// A JIT synthesis request from Fuel (the strategist).
#[derive(Clone, Debug)]
pub struct JitRequest {
    /// The primitive subgraph to fuse — a graph-`Op` tree rooted at the sink,
    /// with `bind` leaves for the region's inputs (the §4.1 vocabulary; the same
    /// node shape [`derive_pattern`] produces).
    pub region: PatternNode,
    /// Region input count; `bind` indices must be exactly `[0, n_inputs)`.
    pub n_inputs: u8,
    /// Op taxonomy for the structure key (drives schedule legality).
    pub op_category: OpCategory,
    /// Operand descriptors (inputs then output) — Fuel's `FdxOperandDesc`
    /// projection, the input to [`structure_key`].
    pub operands: Vec<OperandDesc>,
    /// Target architecture.
    pub arch: ArchSku,
    /// Stable identity to register the synthesized fused op under.
    pub fused_op_id: String,
    /// Compile/resource budget (advisory; Fuel sets it).
    pub budget: JitBudget,
}

/// Compile-time / resource budget for a synthesis request.
#[derive(Copy, Clone, Debug)]
pub struct JitBudget {
    /// Soft ceiling on on-demand compilation time.
    pub max_compile_ms: u32,
}

/// The synthesizer's response — `(kernel + contract + recipe)`, the §5 shape.
#[derive(Clone, Debug)]
pub struct JitResponse {
    /// The synthesized kernel: entry-point symbol, source, and compiled artifact.
    pub kernel: SynthKernel,
    /// The full FKC contract for the kernel (front-matter-less per-kernel block).
    pub contract: String,
    /// The declarative recipe — `pattern:` (recognize the region) + `decompose:`
    /// (expand back to it). Both halves, per the rev-4 recipe principle.
    pub recipe: Recipe,
}

/// A synthesized kernel.
#[derive(Clone, Debug)]
pub struct SynthKernel {
    /// Linkable entry-point symbol (matches the contract's `entry_point`).
    pub entry_point: String,
    /// The generated backend source (`.cu`).
    pub source: String,
    /// The compiled artifact (PTX / cubin) from the on-demand [`Compiler`].
    pub artifact: Vec<u8>,
}

/// The two-directional recipe (rev-4 §1): both mandatory for a fused op.
#[derive(Clone, Debug)]
pub struct Recipe {
    /// The `pattern:` block — recognize the primitive subgraph.
    pub pattern: String,
    /// The `decompose:` block — expand the fused op back to that subgraph. For a
    /// JIT'd op this is, by construction, the region itself (we synthesized the op
    /// to be equivalent to exactly that subgraph). The declarative decompose
    /// *format* is §9-deferred by Fuel, so this is the provisional structural form.
    pub decompose: String,
}

/// Why a region can't be synthesized.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JitError {
    /// A region op name outside the increment-1 IR vocabulary.
    UnsupportedOp(String),
    /// Wrong tensor-operand arity for a region op.
    Arity {
        /// The op.
        op: String,
        /// Expected operand count.
        expected: usize,
        /// Actual operand count.
        got: usize,
    },
    /// The region's bind set isn't `[0, n_inputs)` (rejected by [`derive_pattern`]).
    Pattern(PatternError),
    /// The target dtype has no FKC §5 base-dtype spelling — no contract.
    UnsupportedDtype,
    /// On-demand compilation failed.
    Compile(String),
}

impl From<PatternError> for JitError {
    fn from(e: PatternError) -> Self {
        JitError::Pattern(e)
    }
}

/// The on-demand compilation seam: source → artifact (PTX/cubin). The production
/// impl drives nvrtc; tests use [`StubCompiler`]. Kept a trait so synthesis is
/// testable without a CUDA toolchain and so a pre-built-variant cache can slot in.
pub trait Compiler {
    /// Compile `source` (exposing `entry`) to a device artifact.
    ///
    /// # Errors
    /// Returns the compiler diagnostic string on failure.
    fn compile(&self, source: &str, entry: &str) -> Result<Vec<u8>, String>;
}

/// A no-toolchain stand-in compiler for tests / dry-runs.
#[derive(Copy, Clone, Debug, Default)]
pub struct StubCompiler;

impl Compiler for StubCompiler {
    fn compile(&self, source: &str, entry: &str) -> Result<Vec<u8>, String> {
        Ok(format!("// stub-ptx: {entry} from {}B source", source.len()).into_bytes())
    }
}

/// Synthesize a [`JitResponse`] for a Fuel-chosen region. The synthesizer core:
/// region → op IR → specialized kernel → on-demand compile → FKC contract +
/// recipe. The optimizer that §5.1 permits (an inward e-graph) would sit between
/// `region_to_op` and `generate`; increment 1 lowers the region directly.
///
/// # Errors
/// See [`JitError`] — an unsupported op/dtype or a compile failure.
pub fn synthesize(req: &JitRequest, compiler: &dyn Compiler) -> Result<JitResponse, JitError> {
    let dtype = req.operands.first().map_or(ElementKind::F32, |o| o.dtype);
    let op = region_to_op(&req.region, req.n_inputs, &req.fused_op_id, dtype)?;

    // The schedule cell is keyed from Fuel's operand projection — never re-derived.
    let key = structure_key(req.op_category, &req.operands, req.arch);
    let kernel = generate(&op, &key, &Cuda);

    let artifact = compiler
        .compile(&kernel.source, &kernel.name)
        .map_err(JitError::Compile)?;

    let contract = contract(&op, &key, &kernel, Cuda.name()).ok_or(JitError::UnsupportedDtype)?;
    let pattern = to_fkc(&derive_pattern(&op)?);
    let decompose = to_decompose(&req.region);

    Ok(JitResponse {
        kernel: SynthKernel {
            entry_point: kernel.name,
            source: kernel.source,
            artifact,
        },
        contract,
        recipe: Recipe { pattern, decompose },
    })
}

/// Translate a region (graph-`Op` subgraph) into the op IR — the inverse of
/// [`crate::pattern::derive_pattern`]'s walk. `bind: i` → `Input(i)`; a
/// scalar-param op (`AddScalar`/`MulScalar`) → the arithmetic op with a runtime
/// `Param` (the scalar is a launch arg, exactly as the AOT path treats it).
fn region_to_op(
    region: &PatternNode,
    n_inputs: u8,
    name: &str,
    dtype: ElementKind,
) -> Result<OpDef, JitError> {
    let mut next_param = 0u8;
    let body = node_to_expr(region, &mut next_param)?;
    let op = OpDef {
        name: name.to_string(),
        n_inputs,
        body,
        dtypes: vec![dtype],
        access: Access::Elementwise,
    };
    // Reuse the AOT bind-set / elementwise validation: a region whose bind set
    // isn't [0, n_inputs) is rejected here exactly as an imported pattern would be.
    derive_pattern(&op)?;
    Ok(op)
}

fn node_to_expr(n: &PatternNode, next_param: &mut u8) -> Result<ScalarExpr, JitError> {
    match n {
        PatternNode::Bind(i) => Ok(ScalarExpr::Input(*i)),
        PatternNode::Op { op, operands, .. } => synth_op(op, operands, next_param),
    }
}

fn synth_op(op: &str, operands: &[PatternNode], np: &mut u8) -> Result<ScalarExpr, JitError> {
    // Scalar-param ops: one tensor operand; the scalar becomes a runtime Param
    // (the AOT emitter's `extract:` pulls it back out — round-trip stable).
    if op == "AddScalar" || op == "MulScalar" {
        let t = unary_operand(op, operands, np)?;
        let p = ScalarExpr::Param(*np);
        *np += 1;
        return Ok(if op == "AddScalar" {
            ScalarExpr::Add(Box::new(t), Box::new(p))
        } else {
            ScalarExpr::Mul(Box::new(t), Box::new(p))
        });
    }
    if let Some(u) = region_unary(op) {
        let x = unary_operand(op, operands, np)?;
        return Ok(ScalarExpr::Unary(u, Box::new(x)));
    }
    // Binary tensor ops.
    let ctor: fn(Box<ScalarExpr>, Box<ScalarExpr>) -> ScalarExpr = match op {
        "Add" => ScalarExpr::Add,
        "Sub" => ScalarExpr::Sub,
        "Mul" => ScalarExpr::Mul,
        "Div" => ScalarExpr::Div,
        _ => return Err(JitError::UnsupportedOp(op.to_string())),
    };
    if operands.len() != 2 {
        return Err(JitError::Arity {
            op: op.to_string(),
            expected: 2,
            got: operands.len(),
        });
    }
    let a = node_to_expr(&operands[0], np)?;
    let b = node_to_expr(&operands[1], np)?;
    Ok(ctor(Box::new(a), Box::new(b)))
}

fn unary_operand(
    op: &str,
    operands: &[PatternNode],
    np: &mut u8,
) -> Result<ScalarExpr, JitError> {
    if operands.len() != 1 {
        return Err(JitError::Arity {
            op: op.to_string(),
            expected: 1,
            got: operands.len(),
        });
    }
    node_to_expr(&operands[0], np)
}

/// Inverse of [`crate::pattern`]'s `unary_name`. `GeluErf` → [`UnaryOp::Gelu`]
/// (our exact-erf flavor); bare `Gelu` (tanh approx) and ops with no IR form
/// (`Sin`/`Cos`/`Step`/…) are unsupported in increment 1.
fn region_unary(op: &str) -> Option<UnaryOp> {
    Some(match op {
        "Neg" => UnaryOp::Neg,
        "Abs" => UnaryOp::Abs,
        "Sqr" => UnaryOp::Sqr,
        "Sqrt" => UnaryOp::Sqrt,
        "Rsqrt" => UnaryOp::Rsqrt,
        "Recip" => UnaryOp::Recip,
        "Exp" => UnaryOp::Exp,
        "Log" => UnaryOp::Log,
        "Tanh" => UnaryOp::Tanh,
        "Sigmoid" => UnaryOp::Sigmoid,
        "Relu" => UnaryOp::Relu,
        "Erf" => UnaryOp::Erf,
        "GeluErf" => UnaryOp::Gelu,
        "Silu" => UnaryOp::Silu,
        _ => return None,
    })
}

/// Emit the `decompose:` block: the fused op expands to exactly the region we
/// fused. Same declarative node form as `pattern:`; only the header differs
/// (the declarative decompose format is §9-deferred — provisional).
fn to_decompose(region: &PatternNode) -> String {
    to_fkc(region).replacen("pattern:", "decompose:", 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn op_node(op: &str, operands: Vec<PatternNode>) -> PatternNode {
        PatternNode::Op {
            op: op.to_string(),
            operands,
            consumers: None,
            extract: Vec::new(),
        }
    }

    fn req(region: PatternNode, n_inputs: u8, dt: ElementKind, id: &str) -> JitRequest {
        let a = OperandDesc::new(2, &[128, 256], &[256, 1], dt, 256);
        let operands: Vec<_> = std::iter::repeat_n(a, (n_inputs + 1) as usize).collect();
        JitRequest {
            region,
            n_inputs,
            op_category: OpCategory::BinaryElementwise,
            operands,
            arch: ArchSku::Sm89,
            fused_op_id: id.to_string(),
            budget: JitBudget { max_compile_ms: 1000 },
        }
    }

    #[test]
    fn synthesize_fused_relu_add() {
        // The region Fuel would send for a relu(a+b) fusion.
        let region = op_node(
            "Relu",
            vec![op_node("Add", vec![PatternNode::Bind(0), PatternNode::Bind(1)])],
        );
        let r = req(region, 2, ElementKind::F32, "jit_relu_add");
        let resp = synthesize(&r, &StubCompiler).unwrap();

        // a real, compiled, contracted, recipe-carrying fused kernel.
        assert!(resp.kernel.entry_point.contains("jit_relu_add"));
        assert!(resp.kernel.source.contains("__global__"));
        assert!(!resp.kernel.artifact.is_empty());
        assert!(resp.contract.contains("fused_op: jit_relu_add"));
        assert!(resp.contract.contains("dtypes: [F32]"));
        assert!(resp.recipe.pattern.contains("op: Relu"));
        assert!(resp.recipe.decompose.starts_with("decompose:"));
        assert!(resp.recipe.decompose.contains("op: Relu"));
    }

    #[test]
    fn scalar_param_region_becomes_runtime_param() {
        // AddScalar(MulScalar(x)) — affine, both scalars are runtime params.
        let region = op_node(
            "AddScalar",
            vec![op_node("MulScalar", vec![PatternNode::Bind(0)])],
        );
        let r = req(region, 1, ElementKind::F32, "jit_affine");
        let resp = synthesize(&r, &StubCompiler).unwrap();
        // two op_params (the two scalars) + the source takes them as launch args.
        assert!(resp.contract.contains("name: param0"));
        assert!(resp.contract.contains("name: param1"));
        assert!(resp.recipe.pattern.contains("op: AddScalar"));
        assert!(resp.recipe.pattern.contains("op: MulScalar"));
    }

    #[test]
    fn geluerf_region_maps_to_exact_erf() {
        let region = op_node(
            "GeluErf",
            vec![op_node("Add", vec![PatternNode::Bind(0), PatternNode::Bind(1)])],
        );
        let resp = synthesize(&req(region, 2, ElementKind::F32, "jit_gelu"), &StubCompiler).unwrap();
        assert!(resp.recipe.pattern.contains("op: GeluErf"));
        assert!(resp.kernel.source.contains("erf")); // exact-erf math, not tanh
    }

    #[test]
    fn unsupported_op_is_rejected() {
        // Maximum has no increment-1 IR form.
        let region = op_node("Maximum", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
        let err = synthesize(&req(region, 2, ElementKind::F32, "x"), &StubCompiler).unwrap_err();
        assert_eq!(err, JitError::UnsupportedOp("Maximum".to_string()));
    }

    #[test]
    fn bare_gelu_tanh_flavor_is_rejected() {
        // We only synthesize exact-erf GELU; bare `Gelu` (tanh) has no IR form.
        let region = op_node("Gelu", vec![PatternNode::Bind(0)]);
        let err = synthesize(&req(region, 1, ElementKind::F32, "x"), &StubCompiler).unwrap_err();
        assert_eq!(err, JitError::UnsupportedOp("Gelu".to_string()));
    }

    #[test]
    fn compile_failure_propagates() {
        struct Failing;
        impl Compiler for Failing {
            fn compile(&self, _: &str, _: &str) -> Result<Vec<u8>, String> {
                Err("ptxas: synthetic failure".to_string())
            }
        }
        let region = op_node("Relu", vec![PatternNode::Bind(0)]);
        let err = synthesize(&req(region, 1, ElementKind::F32, "x"), &Failing).unwrap_err();
        assert!(matches!(err, JitError::Compile(m) if m.contains("synthetic failure")));
    }
}
