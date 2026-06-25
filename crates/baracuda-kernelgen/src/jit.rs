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
//! `(kernel + FKC contract + recipe + link row)`, the §5 shape. The heavy lifting
//! reuses the AOT generator ([`generate`], [`contract`], [`derive_pattern`]); the
//! only new step is [`region_to_op`] (region → op IR) and the on-demand
//! [`Compiler`] seam.
//!
//! # Scope (increment 1)
//!
//! The elementwise-epilogue vocabulary the IR already covers ([`ScalarExpr`]):
//! `Add`/`Sub`/`Mul`/`Div`, the scalar-param ops `AddScalar`/`MulScalar`, and the
//! unary math/activations, **single (uniform) dtype**. The on-demand compiler is
//! behind a trait with a stub impl; the real nvrtc backend, the FFI wire surface
//! (reconciling these Rust types with Fuel's `JitRequest`/`JitResponse`), an
//! inward e-graph optimizer (§5.1 permits it), per-operand dtypes, and the
//! telemetry trigger are the growth path.

use crate::contract::contract;
use crate::ir::{Access, BinaryOp, OpDef, ScalarExpr, UnaryOp};
use crate::link::{link_entry, LinkEntry};
use crate::optimize::optimize;
use crate::pattern::{derive_pattern, to_fkc, PatternError, PatternNode};
use crate::{generate, Backend};
use baracuda_kernels_types::{
    structure_key, ArchSku, ElementKind, OpCategory, OperandDesc, MAX_OPERANDS,
};

/// A JIT synthesis request from Fuel (the strategist).
#[derive(Clone, Debug)]
pub struct JitRequest {
    /// The primitive subgraph to fuse — a graph-`Op` tree rooted at the sink,
    /// with `bind` leaves for the region's inputs (the §4.1 vocabulary; the same
    /// node shape [`derive_pattern`] produces). Per-node `consumers`/`extract`
    /// are ignored — [`region_to_op`] regenerates them (see its docs).
    pub region: PatternNode,
    /// Region input count; `bind` indices must be exactly `[0, n_inputs)`, and
    /// [`Self::operands`] must hold exactly `n_inputs + 1` entries.
    pub n_inputs: u8,
    /// Op taxonomy for the structure key (drives schedule legality). Fuel's to
    /// choose (strategist); the synthesizer does not second-guess it.
    pub op_category: OpCategory,
    /// Operand descriptors (inputs then output) — Fuel's `FdxOperandDesc`
    /// projection, the input to [`structure_key`]. Increment 1 requires a single
    /// shared dtype across all operands.
    pub operands: Vec<OperandDesc>,
    /// Target compute capability — keys the schedule. The finer device identity
    /// (ordinal / exact SM / driver) that §5.2's `target.device` carries is
    /// folded into `arch` here; the real on-demand compiler (increment 2) will
    /// refine it where the artifact must be SM-specific.
    pub arch: ArchSku,
    /// Stable identity to register the synthesized fused op under.
    pub fused_op_id: String,
    /// Compile/resource budget (Fuel sets it). Threaded into [`Compiler::compile`].
    pub budget: JitBudget,
}

/// Compile-time / resource budget for a synthesis request.
#[derive(Copy, Clone, Debug)]
pub struct JitBudget {
    /// Ceiling on on-demand compilation time. Must be `> 0`.
    pub max_compile_ms: u32,
}

/// The synthesizer's response — `(kernel + contract + recipe + link)`, the §5 shape.
#[derive(Clone, Debug)]
pub struct JitResponse {
    /// The synthesized kernel: entry-point symbol, source, compiled artifact, and
    /// the artifact's provenance.
    pub kernel: SynthKernel,
    /// The full FKC contract for the kernel (the per-kernel block).
    pub contract: String,
    /// The declarative recipe — `pattern:` (recognize the region) + `decompose:`
    /// (expand back to it). Both halves, per the rev-4 recipe principle.
    pub recipe: Recipe,
    /// The `link_registry` row that resolves the kernel's `entry_point` to a
    /// `KernelRef` at load (FKC §12.6) — without it an adopted kernel is unbindable.
    pub link: LinkEntry,
}

/// A synthesized kernel.
#[derive(Clone, Debug)]
pub struct SynthKernel {
    /// Linkable entry-point symbol (matches the contract's `entry_point`).
    pub entry_point: String,
    /// The generated backend source (`.cu`).
    pub source: String,
    /// The compiled artifact (PTX / cubin / stub) from the on-demand [`Compiler`].
    pub artifact: Vec<u8>,
    /// What kind of artifact this is — a loader **must** refuse [`ArtifactKind::Stub`].
    pub kind: ArtifactKind,
}

/// Provenance of a [`SynthKernel::artifact`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArtifactKind {
    /// Compiled PTX (driver-JIT-linked; portable across SMs of the arch).
    Ptx,
    /// Compiled cubin (SM-specific machine code).
    Cubin,
    /// A stand-in artifact ([`StubCompiler`]) — **not loadable**; a device-module
    /// loader must refuse it rather than feed it to the driver.
    Stub,
}

/// The two-directional recipe (rev-4 §1): both mandatory for a fused op.
#[derive(Clone, Debug)]
pub struct Recipe {
    /// The `pattern:` block — recognize the primitive subgraph.
    pub pattern: String,
    /// The `decompose:` block — expand the fused op back to that subgraph. For a
    /// JIT'd op this is, by construction, the region itself. Derived from the same
    /// canonical pattern node as [`Recipe::pattern`], so the two halves are
    /// structurally identical and the scalar `extract:` routing is preserved. The
    /// declarative decompose *format* is §9-deferred by Fuel (provisional header).
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
    /// `operands.len()` isn't `n_inputs + 1`, or exceeds [`MAX_OPERANDS`] — the
    /// kernel signature and the `accept` predicate would describe different arities.
    OperandArity {
        /// Declared region input count.
        n_inputs: u8,
        /// Operand-projection length supplied.
        operands: usize,
    },
    /// Region operands don't all share one dtype (increment-1 is uniform-dtype) —
    /// rejected as an honest miss rather than mistyped.
    MixedDtype,
    /// The budget is meaningless (e.g. `max_compile_ms == 0`).
    Budget(String),
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
    /// Compile `source` (exposing `entry`) to a device artifact, within
    /// `max_compile_ms` (the request budget; an impl may cap optimization or
    /// abort on overrun).
    ///
    /// # Errors
    /// Returns the compiler diagnostic string on failure.
    fn compile(&self, source: &str, entry: &str, max_compile_ms: u32) -> Result<Vec<u8>, String>;

    /// Provenance of the artifacts this compiler emits. Defaults to
    /// [`ArtifactKind::Ptx`]; [`StubCompiler`] overrides to [`ArtifactKind::Stub`].
    fn artifact_kind(&self) -> ArtifactKind {
        ArtifactKind::Ptx
    }
}

/// A no-toolchain stand-in compiler for tests / dry-runs. Its artifact is tagged
/// [`ArtifactKind::Stub`] so it can never be mistaken for loadable code.
#[derive(Copy, Clone, Debug, Default)]
pub struct StubCompiler;

impl Compiler for StubCompiler {
    fn compile(&self, source: &str, entry: &str, _max_compile_ms: u32) -> Result<Vec<u8>, String> {
        Ok(format!("// stub-ptx: {entry} from {}B source", source.len()).into_bytes())
    }
    fn artifact_kind(&self) -> ArtifactKind {
        ArtifactKind::Stub
    }
}

/// The production on-demand compiler: nvrtc source → PTX. Feature-gated
/// (`--features nvrtc`) because it needs the nvrtc runtime; constructed per target
/// arch (the `--gpu-architecture` flag the schedule was keyed for).
#[cfg(feature = "nvrtc")]
#[derive(Copy, Clone, Debug)]
pub struct NvrtcCompiler {
    arch: ArchSku,
}

#[cfg(feature = "nvrtc")]
impl NvrtcCompiler {
    /// A compiler targeting `arch` (the request's `target` SKU).
    #[must_use]
    pub fn new(arch: ArchSku) -> Self {
        Self { arch }
    }
}

#[cfg(feature = "nvrtc")]
impl Compiler for NvrtcCompiler {
    fn compile(&self, source: &str, entry: &str, _max_compile_ms: u32) -> Result<Vec<u8>, String> {
        // nvrtc has no compile-deadline API; `max_compile_ms` gates optimization
        // depth / the inward e-graph's iteration count at a coarser grain (future).
        // Use the low-level path so a compilation error surfaces the nvrtc log.
        use baracuda_nvrtc::Program;
        let name = format!("{entry}.cu");
        let prog = Program::new(source, &name).map_err(|e| format!("nvrtc({entry}) create: {e}"))?;
        let arch = format!("--gpu-architecture={}", arch_flag(self.arch));
        let mut opts = vec![arch];
        // fp16/bf16 kernels `#include <cuda_fp16.h>`/`<cuda_bf16.h>`; headerless
        // nvrtc has no default search path, so point it at the CUDA include dir
        // (env-detected) — without this, f16/bf16 JIT fails to find the header even
        // though the AOT (nvcc) path compiles. Harmless for header-light f32 source.
        if let Some(inc) = cuda_include_dir() {
            opts.push(format!("-I{inc}"));
        }
        let opt_refs: Vec<&str> = opts.iter().map(String::as_str).collect();
        match prog.compile_raw(&opt_refs) {
            Ok(()) => prog
                .ptx()
                .map(String::into_bytes)
                .map_err(|e| format!("nvrtc({entry}) ptx: {e}")),
            Err(e) => {
                let log = prog.log().unwrap_or_default();
                Err(format!("nvrtc({entry}): {e}\n--- nvrtc log ---\n{}", log.trim()))
            }
        }
    }
    fn artifact_kind(&self) -> ArtifactKind {
        ArtifactKind::Ptx
    }
}

/// The CUDA toolkit `include/` directory (for nvrtc's `-I`), detected from the
/// usual environment vars. `None` if unset/missing — header-light (f32/f64/int)
/// kernels still compile; only the fp16/bf16 headers need it.
#[cfg(feature = "nvrtc")]
fn cuda_include_dir() -> Option<String> {
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(root) = std::env::var(var) {
            let inc = std::path::Path::new(&root).join("include");
            if inc.is_dir() {
                return Some(inc.to_string_lossy().into_owned());
            }
        }
    }
    None
}

/// `--gpu-architecture` flag for an [`ArchSku`].
#[cfg(feature = "nvrtc")]
fn arch_flag(arch: ArchSku) -> &'static str {
    match arch {
        ArchSku::Sm80 => "sm_80",
        ArchSku::Sm89 => "sm_89",
        ArchSku::Sm90a => "sm_90a",
    }
}

/// Synthesize a [`JitResponse`] for a Fuel-chosen region. The synthesizer core:
/// region → op IR → specialized kernel → on-demand compile → FKC contract +
/// recipe + link row. The optimizer that §5.1 permits (an inward e-graph) would
/// sit between `region_to_op` and `generate`; increment 1 lowers directly.
///
/// `backend` selects the lowering target (§5.2 `target.backend`); `compiler` is
/// the matching on-demand toolchain. Both are injected so the engine is
/// backend-agnostic and testable.
///
/// # Errors
/// See [`JitError`] — a malformed request (arity / mixed dtype / zero budget), an
/// unsupported op/dtype, or a compile failure.
pub fn synthesize(
    req: &JitRequest,
    backend: &dyn Backend,
    compiler: &dyn Compiler,
) -> Result<JitResponse, JitError> {
    // --- Trust-boundary validation of the Fuel-supplied request -------------
    let expected = usize::from(req.n_inputs) + 1;
    if req.operands.len() != expected || req.operands.len() > MAX_OPERANDS {
        return Err(JitError::OperandArity {
            n_inputs: req.n_inputs,
            operands: req.operands.len(),
        });
    }
    if req.budget.max_compile_ms == 0 {
        return Err(JitError::Budget("max_compile_ms must be > 0".to_string()));
    }
    let dtype = req.operands[0].dtype; // operands non-empty: len == n_inputs + 1 >= 1
    if req.operands.iter().any(|o| o.dtype != dtype) {
        // Increment 1 is uniform-dtype (StructureKey carries one dtype slot); a
        // mixed region would be mistyped + misdescribed, so miss honestly.
        return Err(JitError::MixedDtype);
    }

    // --- Synthesis ----------------------------------------------------------
    let (op, derived) = region_to_op(&req.region, req.n_inputs, &req.fused_op_id, dtype)?;
    synthesize_op(
        op,
        derived,
        &req.operands,
        req.op_category,
        req.arch,
        req.budget.max_compile_ms,
        backend,
        compiler,
    )
}

/// Core synthesis shared by [`synthesize`] (our [`PatternNode`] region) and the
/// `seam` front-end (Fuel's `fuel_kernel_seam_types::PatternNode` region): op IR +
/// its canonical recipe pattern → optimized kernel → on-demand compile → FKC
/// contract + recipe + link row. The §5.1 inward optimizer runs on the *kernel*
/// body; `derived` (the original region) carries the recipe.
fn synthesize_op(
    op: OpDef,
    derived: PatternNode,
    operands: &[OperandDesc],
    op_category: OpCategory,
    arch: ArchSku,
    max_compile_ms: u32,
    backend: &dyn Backend,
    compiler: &dyn Compiler,
) -> Result<JitResponse, JitError> {
    let dtype = operands.first().map_or(ElementKind::F32, |o| o.dtype);
    // Trust boundary, gate 1: the backend must be able to spell this dtype as a
    // scalar type at all. `dtype_compatible` (gate 2) only checks unary/binary-fn
    // float-ness and f32-only params, so it lets a PURE-INFIX body (Add/Sub/Mul/Div
    // over binds, no fn/param) through for ANY dtype — a Bool/S8/Complex Add region
    // would then panic in `scalar_ctype` during `generate`. Decline it here instead
    // (the Synthesizer trait must never unwind across the boundary).
    if !backend.supports_dtype(dtype) {
        return Err(JitError::UnsupportedDtype);
    }
    // CUDA backend dtype limits: a unary / binary-fn node needs a float dtype, and
    // scalar params are f32-only — honest miss rather than a lowering panic.
    if !dtype_compatible(&op.body, dtype) {
        return Err(JitError::UnsupportedDtype);
    }

    // The schedule cell is keyed from Fuel's operand projection — never re-derived.
    let key = structure_key(op_category, operands, arch);
    let kernel_op = OpDef {
        body: optimize(&op.body),
        ..op.clone()
    };
    let kernel = generate(&kernel_op, &key, backend);

    let artifact = compiler
        .compile(&kernel.source, &kernel.name, max_compile_ms)
        .map_err(JitError::Compile)?;
    let contract = contract(&op, &key, &kernel, backend.name()).ok_or(JitError::UnsupportedDtype)?;

    // Both recipe halves come from the SINGLE canonical pattern node, so they are
    // structurally identical and decompose carries the scalar `extract:` routing.
    let pattern = to_fkc(&derived);
    let decompose = to_fkc(&derived).replacen("pattern:", "decompose:", 1);

    let link = link_entry(&op, &key, &kernel);
    let kind = compiler.artifact_kind();

    Ok(JitResponse {
        kernel: SynthKernel {
            entry_point: kernel.name,
            source: kernel.source,
            artifact,
            kind,
        },
        contract,
        recipe: Recipe { pattern, decompose },
        link,
    })
}

/// Translate a region (graph-`Op` subgraph) into the op IR and its canonical
/// pattern — the inverse of [`crate::pattern::derive_pattern`]'s walk. `bind: i`
/// → `Input(i)`; a scalar-param op (`AddScalar`/`MulScalar`) → the arithmetic op
/// with a runtime `Param` (the scalar is a launch arg, exactly as the AOT path
/// treats it). The returned [`PatternNode`] is [`derive_pattern`]'s canonical
/// form (which also performs the bind-set / elementwise validation).
///
/// The region's per-node `consumers`/`extract` fields are ignored and regenerated
/// by `derive_pattern` under the sole-consumer rule — sound because the IR is a
/// pure tree (no shared interiors), so the only fusable shape (sole-consumer
/// interiors) is the only representable one.
fn region_to_op(
    region: &PatternNode,
    n_inputs: u8,
    name: &str,
    dtype: ElementKind,
) -> Result<(OpDef, PatternNode), JitError> {
    let mut next_param = 0u8;
    let body = node_to_expr(region, &mut next_param)?;
    let op = OpDef {
        name: name.to_string(),
        n_inputs,
        body,
        dtypes: vec![dtype],
        access: Access::Elementwise,
    };
    // Reuse the AOT bind-set / elementwise validation, and keep the canonical
    // pattern (so synthesize derives it exactly once).
    let pattern = derive_pattern(&op)?;
    Ok((op, pattern))
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
    // Non-infix binary fns (Maximum/Minimum/Pow/Rem) — two tensor operands.
    if let Some(bop) = region_binary(op) {
        let (a, b) = binary_operands(op, operands, np)?;
        return Ok(ScalarExpr::Binary(bop, Box::new(a), Box::new(b)));
    }
    // Infix binary tensor ops.
    let ctor: fn(Box<ScalarExpr>, Box<ScalarExpr>) -> ScalarExpr = match op {
        "Add" => ScalarExpr::Add,
        "Sub" => ScalarExpr::Sub,
        "Mul" => ScalarExpr::Mul,
        "Div" => ScalarExpr::Div,
        _ => return Err(JitError::UnsupportedOp(op.to_string())),
    };
    let (a, b) = binary_operands(op, operands, np)?;
    Ok(ctor(Box::new(a), Box::new(b)))
}

/// Resolve the exactly-two operands of a binary op, recursing into each.
fn binary_operands(
    op: &str,
    operands: &[PatternNode],
    np: &mut u8,
) -> Result<(ScalarExpr, ScalarExpr), JitError> {
    if operands.len() != 2 {
        return Err(JitError::Arity {
            op: op.to_string(),
            expected: 2,
            got: operands.len(),
        });
    }
    let a = node_to_expr(&operands[0], np)?;
    let b = node_to_expr(&operands[1], np)?;
    Ok((a, b))
}

/// Whether the CUDA backend can lower `body` at `dtype`: unary / binary-fn nodes
/// require a float dtype (`cuda_unary`/`cuda_binary` have no integer math), and a
/// runtime scalar `Param` is f32-only. Pure infix arithmetic works at any dtype.
fn dtype_compatible(body: &ScalarExpr, dtype: ElementKind) -> bool {
    let is_float = matches!(
        dtype,
        ElementKind::F16
            | ElementKind::Bf16
            | ElementKind::F32
            | ElementKind::F32Strict
            | ElementKind::F64
    );
    let f32_only = matches!(dtype, ElementKind::F32 | ElementKind::F32Strict);
    fn walk(e: &ScalarExpr, is_float: bool, f32_only: bool) -> bool {
        match e {
            // Reduced only appears in a RowReduce epilogue, which never reaches the
            // JIT path (region_to_op builds Elementwise only) — treat as a benign
            // float scalar leaf for exhaustiveness.
            ScalarExpr::Input(_) | ScalarExpr::Const(_) | ScalarExpr::Reduced(_) => true,
            ScalarExpr::Param(_) => f32_only,
            ScalarExpr::Unary(_, x) => is_float && walk(x, is_float, f32_only),
            ScalarExpr::Binary(_, a, b) => {
                is_float && walk(a, is_float, f32_only) && walk(b, is_float, f32_only)
            }
            ScalarExpr::Add(a, b)
            | ScalarExpr::Sub(a, b)
            | ScalarExpr::Mul(a, b)
            | ScalarExpr::Div(a, b) => walk(a, is_float, f32_only) && walk(b, is_float, f32_only),
        }
    }
    walk(body, is_float, f32_only)
}

/// Inverse of [`crate::pattern`]'s `binary_name`.
fn region_binary(op: &str) -> Option<BinaryOp> {
    Some(match op {
        "Maximum" => BinaryOp::Max,
        "Minimum" => BinaryOp::Min,
        "Pow" => BinaryOp::Pow,
        "Rem" => BinaryOp::Rem,
        _ => return None,
    })
}

fn unary_operand(op: &str, operands: &[PatternNode], np: &mut u8) -> Result<ScalarExpr, JitError> {
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
        "Sin" => UnaryOp::Sin,
        "Cos" => UnaryOp::Cos,
        "Floor" => UnaryOp::Floor,
        "Ceil" => UnaryOp::Ceil,
        "Round" => UnaryOp::Round,
        "Sign" => UnaryOp::Sign,
        "Step" => UnaryOp::Step,
        _ => return None,
    })
}

/// The direct-Rust §5 seam (`--features seam`): synthesize for a region in Fuel's
/// frozen grammar (`fuel_kernel_seam_types`). Fuel owns the region grammar
/// (`PatternNode`/`OpTag`); Baracuda owns the classifier input (`OperandDesc`).
/// We convert Fuel's node to our internal node form and reuse the exact native
/// `region_to_op` + core synthesis — no duplicated op logic.
#[cfg(feature = "seam")]
pub mod seam {
    use super::*;
    use fuel_kernel_seam_types::{OpTag, PatternNode as SeamNode};

    /// Synthesize a kernel for a Fuel-chosen `region`. `operands` is the
    /// inputs-then-output `OperandDesc` projection; `n_inputs = operands.len() - 1`.
    ///
    /// # Errors
    /// See [`JitError`] — a malformed request, an op/dtype outside the
    /// synthesizer's coverage (honest miss), or a compile failure.
    #[allow(clippy::too_many_arguments)]
    pub fn synthesize(
        region: &SeamNode,
        operands: &[OperandDesc],
        op_category: OpCategory,
        arch: ArchSku,
        fused_op_id: &str,
        max_compile_ms: u32,
        backend: &dyn Backend,
        compiler: &dyn Compiler,
    ) -> Result<JitResponse, JitError> {
        if operands.is_empty() || operands.len() > MAX_OPERANDS {
            return Err(JitError::OperandArity {
                n_inputs: 0,
                operands: operands.len(),
            });
        }
        if max_compile_ms == 0 {
            return Err(JitError::Budget("max_compile_ms must be > 0".to_string()));
        }
        let dtype = operands[0].dtype;
        if operands.iter().any(|o| o.dtype != dtype) {
            return Err(JitError::MixedDtype);
        }
        let n_inputs = (operands.len() - 1) as u8;

        let internal = to_internal(region)?;
        let (op, derived) = region_to_op(&internal, n_inputs, fused_op_id, dtype)?;
        synthesize_op(
            op, derived, operands, op_category, arch, max_compile_ms, backend, compiler,
        )
    }

    /// Max region nesting the seam will convert — a trust-boundary guard so a
    /// pathologically deep region from Fuel can't overflow the stack (an
    /// uncatchable abort, not a catchable panic) during the recursive conversion.
    /// Elementwise fusion regions are shallow; 64 is far above any real subgraph.
    const MAX_REGION_DEPTH: u32 = 64;

    /// Convert a Fuel `PatternNode` (region direction) to Baracuda's internal node
    /// (op vocabulary mapped by name). An `OpTag` the synthesizer doesn't cover and
    /// the matcher-only `SeeThrough`/`Any` are honest `UnsupportedOp` misses; a
    /// region nested past [`MAX_REGION_DEPTH`] is declined before it can overflow.
    fn to_internal(n: &SeamNode) -> Result<PatternNode, JitError> {
        to_internal_at(n, 0)
    }

    fn to_internal_at(n: &SeamNode, depth: u32) -> Result<PatternNode, JitError> {
        if depth > MAX_REGION_DEPTH {
            return Err(JitError::UnsupportedOp("region nested past MAX_REGION_DEPTH".to_string()));
        }
        match n {
            SeamNode::Bind { index } => Ok(PatternNode::Bind(*index)),
            SeamNode::Op { op, operands, .. } => {
                let name = optag_name(*op).ok_or_else(|| JitError::UnsupportedOp(format!("{op:?}")))?;
                let ops = operands
                    .iter()
                    .map(|o| to_internal_at(o, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(PatternNode::Op {
                    op: name.to_string(),
                    operands: ops,
                    consumers: None,
                    extract: Vec::new(),
                })
            }
            SeamNode::SeeThrough { .. } => Err(JitError::UnsupportedOp("SeeThrough".to_string())),
            SeamNode::Any => Err(JitError::UnsupportedOp("Any".to_string())),
        }
    }

    /// `OpTag` → Baracuda's emitter op-name (what `region_to_op` parses). `None`
    /// for any tag outside the increment-1 synthesizer coverage.
    fn optag_name(op: OpTag) -> Option<&'static str> {
        Some(match op {
            OpTag::Add => "Add",
            OpTag::Sub => "Sub",
            OpTag::Mul => "Mul",
            OpTag::Div => "Div",
            OpTag::Maximum => "Maximum",
            OpTag::Minimum => "Minimum",
            OpTag::Pow => "Pow",
            OpTag::Rem => "Rem",
            OpTag::Neg => "Neg",
            OpTag::Abs => "Abs",
            OpTag::Sqr => "Sqr",
            OpTag::Sqrt => "Sqrt",
            OpTag::Rsqrt => "Rsqrt",
            OpTag::Recip => "Recip",
            OpTag::Exp => "Exp",
            OpTag::Log => "Log",
            OpTag::Sin => "Sin",
            OpTag::Cos => "Cos",
            OpTag::Tanh => "Tanh",
            OpTag::Sigmoid => "Sigmoid",
            OpTag::Silu => "Silu",
            OpTag::GeluErf => "GeluErf",
            OpTag::Relu => "Relu",
            OpTag::Erf => "Erf",
            OpTag::Step => "Step",
            OpTag::Floor => "Floor",
            OpTag::Ceil => "Ceil",
            OpTag::Round => "Round",
            OpTag::Sign => "Sign",
            OpTag::AddScalar => "AddScalar",
            OpTag::MulScalar => "MulScalar",
            // Op::Gelu (tanh), PowI/Clamp, comparisons, Where/MaskedFill, reductions,
            // MatMul, shape/layout, indexing, LogSoftmaxLastDim, Iota — not synthesized.
            _ => return None,
        })
    }

    // ===== The live §5 call — the `fuel_kernel_seam::Synthesizer` Fuel invokes =====

    use crate::Cuda;
    // Alias the envelope types: their bare `JitRequest`/`JitResponse` names would
    // shadow our own internal `jit::{JitRequest, JitResponse}` (glob-imported via
    // `super::*`) across this whole module and silently retype the native core.
    use fuel_kernel_seam::{
        JitRequest as SeamRequest, JitResponse as SeamResponse, SynthesizedKernel, Synthesizer,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// A synthesized kernel's full artifact, retained for the seam-call site to load
    /// and bind. The envelope's [`SynthesizedKernel`] carries only the `entry_point`
    /// (the wire stays light); the PTX/source/contract/recipe/link ride here, keyed
    /// by it, fetched via [`BaracudaSynthesizer::take_kernel`].
    #[derive(Clone, Debug)]
    pub struct SynthArtifact {
        /// The generated backend (`.cu`) source.
        pub source: String,
        /// The compiled artifact (PTX / cubin / stub).
        pub artifact: Vec<u8>,
        /// Artifact provenance — a loader **must** refuse [`ArtifactKind::Stub`].
        pub kind: ArtifactKind,
        /// The full FKC contract for the kernel.
        pub contract: String,
        /// The two-directional recipe (`pattern:` + `decompose:`).
        pub recipe: Recipe,
        /// The `link_registry` row resolving `entry_point` → `KernelRef` (FKC §12.6).
        pub link: LinkEntry,
    }

    /// Baracuda's live JIT synthesizer — the [`Synthesizer`] Fuel calls (§5). Each
    /// [`Synthesizer::synthesize`] adapts the envelope [`JitRequest`] to the native
    /// [`synthesize`] core, returns a [`JitResponse`], and retains the compiled
    /// artifact under its `entry_point` for the seam-call site to fetch
    /// ([`Self::take_kernel`]) and bind (load PTX → `KernelRef` → Fuel's
    /// `adopt_runtime_fused`). **Never panics** (the trait's contract): an
    /// unbuildable / out-of-vocabulary / over-budget region is a typed
    /// [`JitResponse::Declined`], not an error or a crash across the boundary.
    #[derive(Debug)]
    pub struct BaracudaSynthesizer {
        max_compile_ms: u32,
        registry: Mutex<HashMap<String, SynthArtifact>>,
    }

    impl BaracudaSynthesizer {
        /// A synthesizer with the given on-demand compile budget (ms, must be `> 0`).
        #[must_use]
        pub fn new(max_compile_ms: u32) -> Self {
            Self {
                max_compile_ms,
                registry: Mutex::new(HashMap::new()),
            }
        }

        /// Fetch (and remove) the compiled artifact for `entry_point`. The seam-call
        /// site calls this after a [`JitResponse::Synthesized`] (once cost-gating
        /// adopts) to load the kernel and bind the `KernelRef`. Returns `None` if the
        /// entry point was never synthesized or was already taken.
        #[must_use]
        pub fn take_kernel(&self, entry_point: &str) -> Option<SynthArtifact> {
            self.registry.lock().unwrap_or_else(|e| e.into_inner()).remove(entry_point)
        }
    }

    impl Synthesizer for BaracudaSynthesizer {
        fn synthesize(&self, req: &SeamRequest) -> SeamResponse {
            // The on-demand compiler: real nvrtc when compiled in, else a stub (a
            // Stub artifact a loader must refuse — keeps the endpoint callable for
            // wiring/tests without a CUDA toolchain).
            #[cfg(feature = "nvrtc")]
            let compiler = NvrtcCompiler::new(req.arch);
            #[cfg(not(feature = "nvrtc"))]
            let compiler = StubCompiler;

            let n_inputs = req.operands.len().saturating_sub(1);
            // The envelope carries no op category; elementwise schedule is layout-
            // driven, so the category is a key tag — derive it from operand arity
            // (clamped at ternary for 4+-input fused regions).
            let op_category = match n_inputs {
                0 | 1 => OpCategory::UnaryElementwise,
                2 => OpCategory::BinaryElementwise,
                _ => OpCategory::TernaryElementwise,
            };
            let fused_op_id = region_op_id(&req.region, &req.operands);

            match crate::jit::seam::synthesize(
                &req.region,
                &req.operands,
                op_category,
                req.arch,
                &fused_op_id,
                self.max_compile_ms,
                &Cuda,
                &compiler,
            ) {
                Ok(resp) => {
                    let entry_point = resp.kernel.entry_point.clone();
                    // FKC cost-expr (Fuel's `cost_expr` core parses it, binding `n` =
                    // out elem count): the fused op's single-pass memory traffic =
                    // (inputs read + 1 output write) × n. Fuel cost-gates this against
                    // the multi-pass primitive path it would otherwise run.
                    let cost = format!("n * {}", n_inputs.max(1) + 1);
                    self.registry.lock().unwrap_or_else(|e| e.into_inner()).insert(
                        entry_point.clone(),
                        SynthArtifact {
                            source: resp.kernel.source,
                            artifact: resp.kernel.artifact,
                            kind: resp.kernel.kind,
                            contract: resp.contract,
                            recipe: resp.recipe,
                            link: resp.link,
                        },
                    );
                    SeamResponse::Synthesized(SynthesizedKernel {
                        entry_point,
                        pattern: req.region.clone(),
                        cost,
                    })
                }
                // Honest decline (the trait forbids panicking) — region op/dtype out
                // of vocabulary, malformed request, over budget, or compile failure.
                Err(e) => SeamResponse::Declined {
                    reason: format!("{e:?}"),
                },
            }
        }
    }

    /// A stable, readable kernel id for a region: `jit_<root-op>_<hash>`, where the
    /// full 64-bit hash covers the region structure **and** the operand projection,
    /// so distinct regions (or the same region at different layouts/dtypes) are
    /// collision-resistant on one `entry_point`. (The id keys the artifact
    /// `registry`, so a collision would silently overwrite a not-yet-taken kernel —
    /// hence the full u64, not a truncated prefix.)
    fn region_op_id(region: &SeamNode, operands: &[OperandDesc]) -> String {
        use std::hash::{Hash, Hasher};
        let root = match region {
            SeamNode::Op { op, .. } => format!("{op:?}").to_lowercase(),
            _ => "fused".to_string(),
        };
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{region:?}|{operands:?}").hash(&mut h);
        format!("jit_{root}_{:016x}", h.finish())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{Cuda, StubCompiler};
        use fuel_kernel_seam_types::OpAttrs;

        fn op(op: OpTag, operands: Vec<SeamNode>) -> SeamNode {
            SeamNode::Op {
                op,
                operands,
                attrs: OpAttrs::default(),
            }
        }

        fn operands(dt: ElementKind, n: usize) -> Vec<OperandDesc> {
            let a = OperandDesc::new(2, &[128, 256], &[256, 1], dt, 256);
            std::iter::repeat_n(a, n).collect()
        }

        #[test]
        fn synthesize_fuel_region() {
            // relu(a + b) in Fuel's grammar -> a synthesized fused kernel.
            let region = op(
                OpTag::Relu,
                vec![op(
                    OpTag::Add,
                    vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
                )],
            );
            let resp = synthesize(
                &region,
                &operands(ElementKind::F32, 3),
                OpCategory::BinaryElementwise,
                ArchSku::Sm89,
                "jit_relu_add",
                1000,
                &Cuda,
                &StubCompiler,
            )
            .unwrap();
            assert!(resp.contract.contains("fused_op: jit_relu_add"));
            assert!(resp.recipe.pattern.contains("op: Relu"));
            assert!(resp.kernel.source.contains("__global__"));
        }

        #[test]
        fn tanh_gelu_optag_is_unsupported() {
            // Op::Gelu (tanh-approx) is a distinct tag we don't synthesize.
            let region = op(OpTag::Gelu, vec![SeamNode::Bind { index: 0 }]);
            let err = synthesize(
                &region,
                &operands(ElementKind::F32, 2),
                OpCategory::UnaryElementwise,
                ArchSku::Sm89,
                "x",
                1000,
                &Cuda,
                &StubCompiler,
            )
            .unwrap_err();
            assert!(matches!(err, JitError::UnsupportedOp(_)));
        }

        #[test]
        fn synthesizer_synthesizes_and_retains_artifact() {
            // The live envelope path: a JitRequest -> Synthesized, with the compiled
            // artifact retained under its entry_point for the seam-call site.
            let region = op(
                OpTag::Relu,
                vec![op(
                    OpTag::Add,
                    vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
                )],
            );
            let synth = BaracudaSynthesizer::new(1000);
            let req = SeamRequest {
                region,
                operands: operands(ElementKind::F32, 3),
                arch: ArchSku::Sm89,
            };
            let SeamResponse::Synthesized(k) = synth.synthesize(&req) else {
                panic!("expected Synthesized");
            };
            assert!(k.entry_point.starts_with("baracuda_gen_jit_relu_")); // stable id
            assert_eq!(k.cost, "n * 3"); // 2 inputs + 1 output: a single fused pass
            assert_eq!(k.pattern, req.region); // the re-fuse pattern is the region
            // the artifact is retained for the seam-call site to load, then consumed.
            let art = synth.take_kernel(&k.entry_point).expect("artifact retained");
            assert!(art.source.contains("__global__"));
            assert!(synth.take_kernel(&k.entry_point).is_none()); // consumed
        }

        #[test]
        fn synthesizer_declines_never_panics() {
            // An out-of-vocabulary region (Op::Gelu tanh) is an honest Declined, never
            // an error or a panic across the trait boundary.
            let synth = BaracudaSynthesizer::new(1000);
            let req = SeamRequest {
                region: op(OpTag::Gelu, vec![SeamNode::Bind { index: 0 }]),
                operands: operands(ElementKind::F32, 2),
                arch: ArchSku::Sm89,
            };
            assert!(matches!(synth.synthesize(&req), SeamResponse::Declined { .. }));
        }

        #[test]
        fn synthesizer_declines_unlowerable_dtype_never_panics() {
            // Regression (adversarial review): a PURE-INFIX region (Add over binds —
            // no unary/binary-fn, no Param) at a dtype the CUDA backend can't spell as
            // a scalar (Bool / S8 / Complex64) used to PANIC in scalar_ctype during
            // `generate`, because dtype_compatible lets pure-infix bodies through for
            // ANY dtype. The backend dtype-lowerability gate must Decline, never unwind
            // across the trait boundary (which would crash the host).
            let synth = BaracudaSynthesizer::new(1000);
            let region = op(
                OpTag::Add,
                vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
            );
            for dt in [ElementKind::Bool, ElementKind::S8, ElementKind::Complex64] {
                let req = SeamRequest {
                    region: region.clone(),
                    operands: operands(dt, 3),
                    arch: ArchSku::Sm89,
                };
                assert!(
                    matches!(synth.synthesize(&req), SeamResponse::Declined { .. }),
                    "{dt:?} must Decline, not panic",
                );
            }
        }

        /// End-to-end live path on-device: a Fuel `JitRequest` through the
        /// `Synthesizer` impl yields a real nvrtc PTX artifact (retrievable for the
        /// seam-call site to bind). Ignored (needs nvrtc + CUDA).
        #[cfg(feature = "nvrtc")]
        #[test]
        #[ignore = "requires nvrtc runtime + CUDA install"]
        fn synthesizer_produces_real_ptx_on_device() {
            let region = op(
                OpTag::Relu,
                vec![op(
                    OpTag::Add,
                    vec![SeamNode::Bind { index: 0 }, SeamNode::Bind { index: 1 }],
                )],
            );
            let synth = BaracudaSynthesizer::new(5000);
            let req = SeamRequest {
                region,
                operands: operands(ElementKind::F32, 3),
                arch: ArchSku::Sm89,
            };
            let SeamResponse::Synthesized(k) = synth.synthesize(&req) else {
                panic!("expected Synthesized");
            };
            let art = synth.take_kernel(&k.entry_point).expect("artifact retained");
            assert_eq!(art.kind, ArtifactKind::Ptx);
            assert!(String::from_utf8(art.artifact).unwrap().contains(".entry"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cuda;

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
        let region = op_node(
            "Relu",
            vec![op_node("Add", vec![PatternNode::Bind(0), PatternNode::Bind(1)])],
        );
        let r = req(region, 2, ElementKind::F32, "jit_relu_add");
        let resp = synthesize(&r, &Cuda, &StubCompiler).unwrap();

        assert!(resp.kernel.entry_point.contains("jit_relu_add"));
        assert!(resp.kernel.source.contains("__global__"));
        assert_eq!(resp.kernel.kind, ArtifactKind::Stub); // provenance tagged
        assert!(!resp.kernel.artifact.is_empty());
        assert!(resp.contract.contains("fused_op: jit_relu_add"));
        assert!(resp.recipe.pattern.contains("op: Relu"));
        assert!(resp.recipe.decompose.starts_with("decompose:"));
        assert!(resp.recipe.decompose.contains("op: Relu"));
        // the link row makes entry_point resolvable at load.
        assert_eq!(resp.link.entry_point, resp.kernel.entry_point);
        assert!(resp.link.structure_key.starts_with("sk1|"));
    }

    #[test]
    fn scalar_param_region_and_decompose_carry_params() {
        let region = op_node(
            "AddScalar",
            vec![op_node("MulScalar", vec![PatternNode::Bind(0)])],
        );
        let r = req(region, 1, ElementKind::F32, "jit_affine");
        let resp = synthesize(&r, &Cuda, &StubCompiler).unwrap();
        assert!(resp.contract.contains("name: param0"));
        assert!(resp.contract.contains("name: param1"));
        assert!(resp.recipe.pattern.contains("op: AddScalar"));
        // decompose now derives from the same canonical node -> carries extract.
        assert!(resp.recipe.decompose.contains("extract:"));
        assert!(resp.recipe.decompose.contains("op: MulScalar"));
    }

    #[test]
    fn geluerf_region_maps_to_exact_erf() {
        let region = op_node(
            "GeluErf",
            vec![op_node("Add", vec![PatternNode::Bind(0), PatternNode::Bind(1)])],
        );
        let resp = synthesize(&req(region, 2, ElementKind::F32, "jit_gelu"), &Cuda, &StubCompiler)
            .unwrap();
        assert!(resp.recipe.pattern.contains("op: GeluErf"));
        assert!(resp.kernel.source.contains("erf"));
    }

    #[test]
    fn inward_optimizer_simplifies_kernel_but_keeps_the_recipe() {
        // Neg(Neg(x)) region: the inward e-graph (§5.1) cancels the double negation
        // for codegen, but the recipe (pattern/decompose) must still describe the
        // ORIGINAL region so Fuel's matcher recognizes it.
        let region = op_node("Neg", vec![op_node("Neg", vec![PatternNode::Bind(0)])]);
        let resp = synthesize(&req(region, 1, ElementKind::F32, "jit_negneg"), &Cuda, &StubCompiler)
            .unwrap();
        // kernel body is the optimized identity copy — no double negation emitted.
        assert!(!resp.kernel.source.contains("-(-("));
        // recipe still carries the original Neg subgraph.
        assert_eq!(resp.recipe.pattern.matches("op: Neg").count(), 2);
        assert!(resp.recipe.decompose.contains("op: Neg"));
    }

    #[test]
    fn unsupported_op_is_rejected() {
        // MatMul is not an elementwise op we synthesize.
        let region = op_node("MatMul", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
        let err = synthesize(&req(region, 2, ElementKind::F32, "x"), &Cuda, &StubCompiler)
            .unwrap_err();
        assert_eq!(err, JitError::UnsupportedOp("MatMul".to_string()));
    }

    #[test]
    fn broadened_ops_synthesize() {
        // The new binary fns + unary math now synthesize (no UnsupportedOp).
        for (region, n, id) in [
            (op_node("Maximum", vec![PatternNode::Bind(0), PatternNode::Bind(1)]), 2u8, "jit_max"),
            (op_node("Pow", vec![PatternNode::Bind(0), PatternNode::Bind(1)]), 2, "jit_pow"),
            (op_node("Floor", vec![PatternNode::Bind(0)]), 1, "jit_floor"),
            (op_node("Sin", vec![PatternNode::Bind(0)]), 1, "jit_sin"),
        ] {
            let resp = synthesize(&req(region, n, ElementKind::F32, id), &Cuda, &StubCompiler)
                .unwrap();
            assert!(resp.kernel.source.contains("__global__"));
        }
    }

    #[test]
    fn integer_unary_binary_is_honest_miss_not_panic() {
        // int + a unary/binary fn has no CUDA math -> honest miss, never a panic.
        let region = op_node("Maximum", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
        assert_eq!(
            synthesize(&req(region, 2, ElementKind::I32, "x"), &Cuda, &StubCompiler).unwrap_err(),
            JitError::UnsupportedDtype
        );
        // pure int Add (infix) is fine.
        let add = op_node("Add", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
        assert!(synthesize(&req(add, 2, ElementKind::I32, "x"), &Cuda, &StubCompiler).is_ok());
    }

    #[test]
    fn non_f32_scalar_param_is_honest_miss() {
        // scalar params are f32-only; an f64 AddScalar region misses honestly.
        let region = op_node("AddScalar", vec![PatternNode::Bind(0)]);
        assert_eq!(
            synthesize(&req(region, 1, ElementKind::F64, "x"), &Cuda, &StubCompiler).unwrap_err(),
            JitError::UnsupportedDtype
        );
    }

    #[test]
    fn unknown_binary_name_is_unsupported() {
        let region = op_node("Mod", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
        assert_eq!(
            synthesize(&req(region, 2, ElementKind::F32, "x"), &Cuda, &StubCompiler).unwrap_err(),
            JitError::UnsupportedOp("Mod".to_string())
        );
    }

    #[test]
    fn bare_gelu_tanh_flavor_is_rejected() {
        let region = op_node("Gelu", vec![PatternNode::Bind(0)]);
        let err = synthesize(&req(region, 1, ElementKind::F32, "x"), &Cuda, &StubCompiler)
            .unwrap_err();
        assert_eq!(err, JitError::UnsupportedOp("Gelu".to_string()));
    }

    #[test]
    fn compile_failure_propagates() {
        struct Failing;
        impl Compiler for Failing {
            fn compile(&self, _: &str, _: &str, _: u32) -> Result<Vec<u8>, String> {
                Err("ptxas: synthetic failure".to_string())
            }
        }
        let region = op_node("Relu", vec![PatternNode::Bind(0)]);
        let err =
            synthesize(&req(region, 1, ElementKind::F32, "x"), &Cuda, &Failing).unwrap_err();
        assert!(matches!(err, JitError::Compile(m) if m.contains("synthetic failure")));
    }

    #[test]
    fn operand_arity_mismatch_is_rejected() {
        // n_inputs says 2 (=> 3 operands expected) but only 2 operands supplied.
        let region = op_node("Add", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
        let mut r = req(region, 2, ElementKind::F32, "x");
        r.operands.pop(); // now 2 operands, not 3
        let err = synthesize(&r, &Cuda, &StubCompiler).unwrap_err();
        assert_eq!(err, JitError::OperandArity { n_inputs: 2, operands: 2 });
    }

    #[test]
    fn mixed_dtype_region_is_an_honest_miss() {
        let region = op_node("Add", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
        let mut r = req(region, 2, ElementKind::F32, "x");
        r.operands[1] = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F16, 256);
        assert_eq!(synthesize(&r, &Cuda, &StubCompiler).unwrap_err(), JitError::MixedDtype);
    }

    #[test]
    fn zero_budget_is_rejected() {
        let region = op_node("Relu", vec![PatternNode::Bind(0)]);
        let mut r = req(region, 1, ElementKind::F32, "x");
        r.budget.max_compile_ms = 0;
        assert!(matches!(
            synthesize(&r, &Cuda, &StubCompiler).unwrap_err(),
            JitError::Budget(_)
        ));
    }

    /// End-to-end on-device synthesis: region → kernel → real nvrtc PTX. Ignored
    /// by default (needs the nvrtc runtime + a CUDA install); run with
    /// `cargo test -p baracuda-kernelgen --features nvrtc -- --ignored`.
    #[cfg(feature = "nvrtc")]
    #[test]
    #[ignore = "requires nvrtc runtime + CUDA install"]
    fn nvrtc_compiles_a_synthesized_kernel() {
        let region = op_node(
            "Relu",
            vec![op_node("Add", vec![PatternNode::Bind(0), PatternNode::Bind(1)])],
        );
        let r = req(region, 2, ElementKind::F32, "jit_relu_add");
        let resp = synthesize(&r, &Cuda, &NvrtcCompiler::new(ArchSku::Sm89)).unwrap();
        assert_eq!(resp.kernel.kind, ArtifactKind::Ptx);
        let ptx = String::from_utf8(resp.kernel.artifact).expect("PTX is utf-8 text");
        assert!(ptx.contains(".entry"), "PTX should declare the kernel entry");
    }

    /// The broadened ops compile under nvrtc too: max(sin(a), b) exercises a new
    /// unary (`sinf`) and a new binary fn (`fmaxf`). Ignored (needs nvrtc + CUDA).
    #[cfg(feature = "nvrtc")]
    #[test]
    #[ignore = "requires nvrtc runtime + CUDA install"]
    fn nvrtc_compiles_broadened_ops() {
        let region = op_node(
            "Maximum",
            vec![op_node("Sin", vec![PatternNode::Bind(0)]), PatternNode::Bind(1)],
        );
        let r = req(region, 2, ElementKind::F32, "jit_max_sin");
        let resp = synthesize(&r, &Cuda, &NvrtcCompiler::new(ArchSku::Sm89)).unwrap();
        assert_eq!(resp.kernel.kind, ArtifactKind::Ptx);
        assert!(String::from_utf8(resp.kernel.artifact).unwrap().contains(".entry"));
    }

    /// The reduction schedule compiles headerless under nvrtc too: f32 mean-of-
    /// squares (no includes) and f16 sum (`__half2float` + the fp16 header nvrtc
    /// bundles). Numeric correctness is proven separately via an nvcc host harness;
    /// this guards the same headerless-portability property that the `cstdint`
    /// regression taught us. Ignored (needs nvrtc + CUDA).
    #[cfg(feature = "nvrtc")]
    #[test]
    #[ignore = "requires nvrtc runtime + CUDA install"]
    fn nvrtc_compiles_reduction_kernels() {
        use crate::ir::UnaryOp;
        use crate::{generate, input, ReduceOp};
        let cc = NvrtcCompiler::new(ArchSku::Sm89);
        let red_key = |dt: ElementKind| {
            let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
            let out = OperandDesc::new(1, &[256], &[1], dt, 256);
            structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89)
        };
        // f32 mean-of-squares (the RmsNorm core) — header-light source.
        let ms = OpDef::reduction(
            "ms",
            1,
            &[ElementKind::F32],
            input(0).unary(UnaryOp::Sqr),
            ReduceOp::Mean,
        );
        let kf32 = generate(&ms, &red_key(ElementKind::F32), &Cuda);
        let ptx = cc.compile(&kf32.source, &kf32.name, 5000).expect("f32 reduction compiles");
        assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
        // f16 sum — exercises __half2float + cuda_fp16.h under headerless nvrtc.
        let sum = OpDef::reduction("s", 1, &[ElementKind::F16], input(0), ReduceOp::Sum);
        let kf16 = generate(&sum, &red_key(ElementKind::F16), &Cuda);
        let ptx16 = cc.compile(&kf16.source, &kf16.name, 5000).expect("f16 reduction compiles");
        assert!(String::from_utf8(ptx16).unwrap().contains(".entry"));
    }

    /// The fused RowReduce kernels (RmsNorm / Softmax) compile headerless under
    /// nvrtc — the warp-shuffle/shared-mem block reduce, `rsqrtf`/`expf`, and (for
    /// f16) `__half2float` + the fp16 header. Numeric correctness is proven via the
    /// nvcc host harness; this guards headerless portability. Ignored (needs nvrtc).
    #[cfg(feature = "nvrtc")]
    #[test]
    #[ignore = "requires nvrtc runtime + CUDA install"]
    fn nvrtc_compiles_rowreduce_kernels() {
        use crate::ir::{konst, reduced, ReduceOp, ReduceStage, UnaryOp};
        use crate::{generate, input};
        let cc = NvrtcCompiler::new(ArchSku::Sm89);
        let key = |dt: ElementKind, cat: OpCategory| {
            let a = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
            structure_key(cat, &[a, a], ArchSku::Sm89)
        };
        let rms = |dt: ElementKind| {
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
        };
        let sm = |dt: ElementKind| {
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
        };
        for (op, dt, cat) in [
            (rms(ElementKind::F32), ElementKind::F32, OpCategory::Normalization),
            (sm(ElementKind::F32), ElementKind::F32, OpCategory::Softmax),
            // f16: exercises __half2float + cuda_fp16.h under headerless nvrtc.
            (rms(ElementKind::F16), ElementKind::F16, OpCategory::Normalization),
            // f64 / f32-strict: the double accumulator path relies on the `double`
            // __shfl_down_sync overload compiling headerless (the critics' flag).
            (rms(ElementKind::F64), ElementKind::F64, OpCategory::Normalization),
            (sm(ElementKind::F32Strict), ElementKind::F32Strict, OpCategory::Softmax),
        ] {
            let k = generate(&op, &key(dt, cat), &Cuda);
            let ptx = cc
                .compile(&k.source, &k.name, 5000)
                .unwrap_or_else(|e| panic!("{} failed to compile: {e}", k.name));
            assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
        }
    }
}
