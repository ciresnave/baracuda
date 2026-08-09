//! JIT synthesis tests that require a real backend (`&Cuda`) or the NVRTC compiler.
//!
//! Relocated from `baracuda-kernelgen` `src/jit.rs`'s `mod tests` during the
//! Unpopped carve (step 3). The generic JIT plumbing (`Compiler`/`StubCompiler`/
//! `synthesize`/`JitError`) stays neutral in `baracuda-kernelgen`; these tests
//! drive it through the CUDA backend, so they live with the CUDA backend crate.
//! The backend-free decline/dtype-predicate tests stayed neutral in kernelgen.

use baracuda_cuda_emit::Cuda;
use unpopped::jit::*;
use unpopped::pattern::*;
// The nvrtc-gated `nvrtc_compiles_*` tests build `OpDef`/`BinaryOp` IR directly;
// the non-nvrtc synthesis tests reach the IR only through the seam `PatternNode`
// path, so the `ir` glob is dead weight (and an unused-import warning) without it.
#[cfg(feature = "nvrtc")]
use baracuda_cuda_emit::NvrtcCompiler;
#[cfg(feature = "nvrtc")]
use unpopped::ir::*;
use unpopped_vocab::*;

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
        budget: JitBudget {
            max_compile_ms: 1000,
        },
    }
}

#[test]
fn synthesize_fused_relu_add() {
    let region = op_node(
        "Relu",
        vec![op_node(
            "Add",
            vec![PatternNode::Bind(0), PatternNode::Bind(1)],
        )],
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
    assert!(resp.link.structure_key.starts_with("sk3|"));
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
        vec![op_node(
            "Add",
            vec![PatternNode::Bind(0), PatternNode::Bind(1)],
        )],
    );
    let resp = synthesize(
        &req(region, 2, ElementKind::F32, "jit_gelu"),
        &Cuda,
        &StubCompiler,
    )
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
    let resp = synthesize(
        &req(region, 1, ElementKind::F32, "jit_negneg"),
        &Cuda,
        &StubCompiler,
    )
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
    let err = synthesize(&req(region, 2, ElementKind::F32, "x"), &Cuda, &StubCompiler).unwrap_err();
    assert_eq!(err, JitError::UnsupportedOp("MatMul".to_string()));
}

#[test]
fn broadened_ops_synthesize() {
    // The new binary fns + unary math now synthesize (no UnsupportedOp).
    for (region, n, id) in [
        (
            op_node("Maximum", vec![PatternNode::Bind(0), PatternNode::Bind(1)]),
            2u8,
            "jit_max",
        ),
        (
            op_node("Pow", vec![PatternNode::Bind(0), PatternNode::Bind(1)]),
            2,
            "jit_pow",
        ),
        (op_node("Floor", vec![PatternNode::Bind(0)]), 1, "jit_floor"),
        (op_node("Sin", vec![PatternNode::Bind(0)]), 1, "jit_sin"),
    ] {
        let resp = synthesize(&req(region, n, ElementKind::F32, id), &Cuda, &StubCompiler).unwrap();
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
fn uniform_int_regions_synthesize_the_audited_set_only() {
    // Increment 0c: U8/S8 are supported compute dtypes at the JIT boundary
    // — but ONLY for the audited op set. Legal: infix Add at U8/S8/I32
    // (wrapping)…
    for dt in [ElementKind::U8, ElementKind::S8, ElementKind::I32] {
        let add = op_node("Add", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
        let resp = synthesize(&req(add, 2, dt, "jit_add"), &Cuda, &StubCompiler)
            .unwrap_or_else(|e| panic!("{dt:?} Add must synthesize, got {e:?}"));
        assert!(resp.kernel.source.contains("__global__"));
    }
    // …illegal: Div at any int dtype (bespoke has no int elementwise div;
    // C `/0` is device-UB) — the gate consults op×dtype legality, not just
    // the dtype, so a uniform-U8 Div region still declines.
    for dt in [
        ElementKind::U8,
        ElementKind::S8,
        ElementKind::I32,
        ElementKind::I64,
    ] {
        let div = op_node("Div", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
        assert_eq!(
            synthesize(&req(div, 2, dt, "x"), &Cuda, &StubCompiler).unwrap_err(),
            JitError::UnsupportedDtype,
            "{dt:?} Div must decline typed"
        );
    }
    // …and a float fn at U8 declines exactly like the pre-0c I32 case.
    let mx = op_node("Maximum", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
    assert_eq!(
        synthesize(&req(mx, 2, ElementKind::U8, "x"), &Cuda, &StubCompiler).unwrap_err(),
        JitError::UnsupportedDtype
    );
}

#[test]
fn bitwise_names_are_not_region_reachable() {
    // fuel-kernel-seam-types 0.10.2 has no OpTag for the 0c ops, so no
    // region can name them — honest UnsupportedOp, and the mapping tables
    // were NOT extended speculatively (the invented-vocabulary trap).
    for name in [
        "BitAnd",
        "BitOr",
        "BitXor",
        "Shl",
        "Shr",
        "BitwiseAnd",
        "LeftShift",
        "RightShift",
        "LogicalAnd",
        "LogicalOr",
        "LogicalXor",
    ] {
        let region = op_node(name, vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
        assert_eq!(
            synthesize(&req(region, 2, ElementKind::I32, "x"), &Cuda, &StubCompiler).unwrap_err(),
            JitError::UnsupportedOp(name.to_string()),
            "{name} must be an honest region miss"
        );
        assert!(region_binary(name).is_none());
    }
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
fn nested_cmp_region_declines_awaiting_cast_vocabulary() {
    // relu-backward mask-multiply in region form: Mul(dy, Gt(x, z)) — the
    // kernel would be correct, but the contract's pattern block would be
    // unmatchable against any constructible Fuel graph (real graphs
    // interpose Cast(U8→float), outside the §4.1 grammar + see-through
    // set). Typed decline until Cast joins the vocabulary; AOT lowering
    // of the same body still works (contract withheld there too).
    let region = op_node(
        "Mul",
        vec![
            PatternNode::Bind(0),
            op_node("Gt", vec![PatternNode::Bind(1), PatternNode::Bind(2)]),
        ],
    );
    let r = req(region, 3, ElementKind::F32, "jit_relu_bw");
    let err = synthesize(&r, &Cuda, &StubCompiler).unwrap_err();
    assert!(
        matches!(&err, JitError::UnsupportedOp(m) if m.contains("interior comparison")),
        "expected the interior-cmp typed decline, got {err:?}"
    );
}

#[test]
fn where_region_passes_the_cmp_carve_out_and_declines_at_the_typed_pattern_miss() {
    // The [Gt, Where] region — Fuel's fused-select shape, cond consumed
    // DIRECTLY (no Cast). The interior-cmp carve-out must let it PAST the
    // "interior comparison" decline (the cmp is the cond child of the
    // Select); the v1 decline is then the TYPED pattern miss
    // (SelectUnsupported — the withheld Where advert), naming the real
    // blocker. Without the carve-out this would mis-decline as an
    // unreachable Cast problem.
    let region = op_node(
        "Where",
        vec![
            op_node("Gt", vec![PatternNode::Bind(0), PatternNode::Bind(1)]),
            PatternNode::Bind(2),
            PatternNode::Bind(3),
        ],
    );
    let err = synthesize(
        &req(region, 4, ElementKind::F32, "jit_where"),
        &Cuda,
        &StubCompiler,
    )
    .unwrap_err();
    assert_eq!(
        err,
        JitError::Pattern(PatternError::SelectUnsupported),
        "the decline must be the withheld Where advert, not the interior-cmp gate"
    );
    // A cmp in an ARM position keeps the interior-cmp decline (the
    // carve-out is the cond child ONLY).
    let cmp_in_arm = op_node(
        "Where",
        vec![
            op_node("Gt", vec![PatternNode::Bind(0), PatternNode::Bind(1)]),
            op_node("Ge", vec![PatternNode::Bind(2), PatternNode::Bind(3)]),
            PatternNode::Bind(2),
        ],
    );
    let err = synthesize(
        &req(cmp_in_arm, 4, ElementKind::F32, "x"),
        &Cuda,
        &StubCompiler,
    )
    .unwrap_err();
    assert!(
        matches!(&err, JitError::UnsupportedOp(m) if m.contains("interior comparison")),
        "a cmp ARM must keep the interior-cmp decline, got {err:?}"
    );
    // A cmp nested DEEPER inside a composed cond declines too (only the
    // cond-ROOT position is carved out).
    let deep_cond = op_node(
        "Where",
        vec![
            op_node(
                "Mul",
                vec![
                    op_node("Gt", vec![PatternNode::Bind(0), PatternNode::Bind(1)]),
                    PatternNode::Bind(2),
                ],
            ),
            PatternNode::Bind(2),
            PatternNode::Bind(3),
        ],
    );
    let err = synthesize(
        &req(deep_cond, 4, ElementKind::F32, "x"),
        &Cuda,
        &StubCompiler,
    )
    .unwrap_err();
    assert!(
        matches!(&err, JitError::UnsupportedOp(m) if m.contains("interior comparison")),
        "a composed-cond interior cmp must keep the decline, got {err:?}"
    );
}

#[test]
fn where_bound_cond_region_declines_typed() {
    // A Where whose cond is a bare BIND is Fuel's bound-cond shape — a U8
    // tensor operand ([U8,T,T] tuple), inexpressible under uniform-dtype
    // keying. The uniform all-T projection must decline TYPED at the
    // bound-cond gate (M11's target), never synthesize a key-dtype `!= 0`
    // kernel misdescribing the U8-cond op.
    let region = op_node(
        "Where",
        vec![
            PatternNode::Bind(0),
            PatternNode::Bind(1),
            PatternNode::Bind(2),
        ],
    );
    let err = synthesize(
        &req(region, 3, ElementKind::F32, "jit_where_bound"),
        &Cuda,
        &StubCompiler,
    )
    .unwrap_err();
    assert!(
        matches!(&err, JitError::UnsupportedOp(m) if m.contains("bound cond")),
        "expected the bound-cond typed decline, got {err:?}"
    );
}

#[test]
fn where_arity_mismatch_is_rejected() {
    // Where is strictly ternary — a 2-operand region is an arity error,
    // through the new ternary_operands helper.
    let region = op_node(
        "Where",
        vec![
            op_node("Gt", vec![PatternNode::Bind(0), PatternNode::Bind(1)]),
            PatternNode::Bind(2),
        ],
    );
    let err = synthesize(&req(region, 3, ElementKind::F32, "x"), &Cuda, &StubCompiler).unwrap_err();
    assert_eq!(
        err,
        JitError::Arity {
            op: "Where".to_string(),
            expected: 3,
            got: 2
        }
    );
}

#[test]
fn root_cmp_region_is_a_typed_decline() {
    // A region ROOTED at a comparison produces a U8 mask — hetero output
    // the increment-1 uniform-dtype keying can't express. Typed decline
    // for all six, never a panic and never a float-mask kernel advertised
    // as a §4.1 comparison.
    for name in ["Equal", "Ne", "Lt", "Le", "Gt", "Ge"] {
        let region = op_node(name, vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
        let err =
            synthesize(&req(region, 2, ElementKind::F32, "x"), &Cuda, &StubCompiler).unwrap_err();
        assert!(
            matches!(&err, JitError::UnsupportedOp(m) if m.contains("comparison at region root")),
            "{name}: expected the root-cmp typed decline, got {err:?}"
        );
    }
}

#[test]
fn increment_0a_names_are_not_region_reachable() {
    // The new scalar fns have no §4.1/OpTag vocabulary, so no region can name
    // them: honest UnsupportedOp, never a panic, and the mapping tables were
    // NOT extended speculatively.
    for (name, n) in [
        ("Erfc", 1u8),
        ("Lgamma", 1),
        ("Atan2", 2),
        ("Nextafter", 2),
        ("FmaxIeee", 2),
        ("RemTrunc", 2),
    ] {
        let operands = (0..n).map(PatternNode::Bind).collect();
        let region = op_node(name, operands);
        assert_eq!(
            synthesize(&req(region, n, ElementKind::F32, "x"), &Cuda, &StubCompiler).unwrap_err(),
            JitError::UnsupportedOp(name.to_string()),
            "{name} must be an honest region miss"
        );
    }
    assert!(region_unary("Erfc").is_none());
    assert!(region_binary("Atan2").is_none());
}

#[test]
fn bare_gelu_tanh_flavor_is_rejected() {
    let region = op_node("Gelu", vec![PatternNode::Bind(0)]);
    let err = synthesize(&req(region, 1, ElementKind::F32, "x"), &Cuda, &StubCompiler).unwrap_err();
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
    let err = synthesize(&req(region, 1, ElementKind::F32, "x"), &Cuda, &Failing).unwrap_err();
    assert!(matches!(err, JitError::Compile(m) if m.contains("synthetic failure")));
}

#[test]
fn operand_arity_mismatch_is_rejected() {
    // n_inputs says 2 (=> 3 operands expected) but only 2 operands supplied.
    let region = op_node("Add", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
    let mut r = req(region, 2, ElementKind::F32, "x");
    r.operands.pop(); // now 2 operands, not 3
    let err = synthesize(&r, &Cuda, &StubCompiler).unwrap_err();
    assert_eq!(
        err,
        JitError::OperandArity {
            n_inputs: 2,
            operands: 2
        }
    );
}

#[test]
fn mixed_dtype_region_is_an_honest_miss() {
    let region = op_node("Add", vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
    let mut r = req(region, 2, ElementKind::F32, "x");
    r.operands[1] = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F16, 256);
    assert_eq!(
        synthesize(&r, &Cuda, &StubCompiler).unwrap_err(),
        JitError::MixedDtype
    );
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
/// `cargo test -p baracuda-cuda-emit --features nvrtc -- --ignored`.
#[cfg(feature = "nvrtc")]
#[test]
#[ignore = "requires nvrtc runtime + CUDA install"]
fn nvrtc_compiles_a_synthesized_kernel() {
    let region = op_node(
        "Relu",
        vec![op_node(
            "Add",
            vec![PatternNode::Bind(0), PatternNode::Bind(1)],
        )],
    );
    let r = req(region, 2, ElementKind::F32, "jit_relu_add");
    let resp = synthesize(&r, &Cuda, &NvrtcCompiler::new(ArchSku::Sm89)).unwrap();
    assert_eq!(resp.kernel.kind, ArtifactKind::Ptx);
    let ptx = String::from_utf8(resp.kernel.artifact).expect("PTX is utf-8 text");
    assert!(
        ptx.contains(".entry"),
        "PTX should declare the kernel entry"
    );
}

/// The broadened ops compile under nvrtc too: max(sin(a), b) exercises a new
/// unary (`sinf`) and a new binary fn (`fmaxf`). Ignored (needs nvrtc + CUDA).
#[cfg(feature = "nvrtc")]
#[test]
#[ignore = "requires nvrtc runtime + CUDA install"]
fn nvrtc_compiles_broadened_ops() {
    let region = op_node(
        "Maximum",
        vec![
            op_node("Sin", vec![PatternNode::Bind(0)]),
            PatternNode::Bind(1),
        ],
    );
    let r = req(region, 2, ElementKind::F32, "jit_max_sin");
    let resp = synthesize(&r, &Cuda, &NvrtcCompiler::new(ArchSku::Sm89)).unwrap();
    assert_eq!(resp.kernel.kind, ArtifactKind::Ptx);
    assert!(
        String::from_utf8(resp.kernel.artifact)
            .unwrap()
            .contains(".entry")
    );
}

/// The increment-0a scalar-fn vocabulary compiles headerless under nvrtc —
/// the same implicit-device-math property the Exp/Tanh emission proves —
/// across f32, f16 (promote path + fp16 header), and f64. Erf/Log1p unary +
/// Atan2/RemTrunc binary are the representative sweep. Ignored (needs nvrtc).
#[cfg(feature = "nvrtc")]
#[test]
#[ignore = "requires nvrtc runtime + CUDA install"]
fn nvrtc_compiles_increment_0a_vocab() {
    use unpopped::generate;
    use unpopped::ir::{BinaryOp, UnaryOp, input};
    let cc = NvrtcCompiler::new(ArchSku::Sm89);
    let ukey = |dt: ElementKind| {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 256);
        structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89)
    };
    let bkey = |dt: ElementKind| {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 256);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    };
    for dt in [ElementKind::F32, ElementKind::F16, ElementKind::F64] {
        for uop in [UnaryOp::Erf, UnaryOp::Log1p] {
            let op = OpDef::elementwise("vu", 1, &[dt], input(0).unary(uop));
            let k = generate(&op, &ukey(dt), &Cuda);
            let ptx = cc
                .compile(&k.source, &k.name, 5000)
                .unwrap_or_else(|e| panic!("{uop:?} {dt:?} failed headerless nvrtc: {e}"));
            assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
        }
        for bop in [BinaryOp::Atan2, BinaryOp::RemTrunc] {
            let op = OpDef::elementwise("vb", 2, &[dt], input(0).binary(bop, input(1)));
            let k = generate(&op, &bkey(dt), &Cuda);
            let ptx = cc
                .compile(&k.source, &k.name, 5000)
                .unwrap_or_else(|e| panic!("{bop:?} {dt:?} failed headerless nvrtc: {e}"));
            assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
        }
    }
}

/// A MULTI-OUTPUT (increment 1) kernel compiles headerless under nvrtc: N
/// output pointers + N stores from a shared body-DAG are plain C (no
/// includes on the f32 path). mul_backward (2 outputs, scalar) and
/// fma_backward (3 outputs) are the representative cells; numeric
/// correctness is the nvcc host harness (`multi_output_validate.cu`), this
/// guards headerless portability. Ignored (needs nvrtc + CUDA).
#[cfg(feature = "nvrtc")]
#[test]
#[ignore = "requires nvrtc runtime + CUDA install"]
fn nvrtc_compiles_multi_output_kernel() {
    use unpopped::generate;
    use unpopped::ir::input;
    let cc = NvrtcCompiler::new(ArchSku::Sm89);
    let key = |n: usize| {
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        let ops: Vec<_> = std::iter::repeat_n(a, n).collect();
        structure_key(OpCategory::BinaryElementwise, &ops, ArchSku::Sm89)
    };
    // mul_backward: 3 inputs, 2 outputs (da=dy*b, db=dy*a).
    let mul = OpDef::elementwise_multi(
        "mul_backward",
        3,
        &[ElementKind::F32],
        vec![input(0) * input(2), input(0) * input(1)],
    );
    let k = generate(&mul, &key(5), &Cuda);
    let ptx = cc
        .compile(&k.source, &k.name, 5000)
        .unwrap_or_else(|e| panic!("mul_backward failed headerless nvrtc: {e}"));
    assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
    // fma_backward: 3 inputs, 3 outputs (one a plain copy).
    let fma = OpDef::elementwise_multi(
        "fma_backward",
        3,
        &[ElementKind::F32],
        vec![input(0) * input(2), input(0) * input(1), input(0)],
    );
    let k = generate(&fma, &key(6), &Cuda);
    let ptx = cc
        .compile(&k.source, &k.name, 5000)
        .unwrap_or_else(|e| panic!("fma_backward failed headerless nvrtc: {e}"));
    assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
}

/// The increment-0b comparison kernels compile headerless under nvrtc: the
/// f32-inputs/u8-mask-output predicate (the `unsigned char` signature + the
/// `(unsigned char)` store cast are plain C — no includes), plus the f16
/// promote form (fp16 header) and the nested mask-multiply. Numeric
/// correctness is proven by the nvcc host harness (`cmp_validate.cu`);
/// this guards headerless portability. Ignored (needs nvrtc + CUDA).
#[cfg(feature = "nvrtc")]
#[test]
#[ignore = "requires nvrtc runtime + CUDA install"]
fn nvrtc_compiles_cmp_u8_kernel() {
    use unpopped::generate;
    use unpopped::ir::{input, konst};
    let cc = NvrtcCompiler::new(ArchSku::Sm89);
    let pred_key = |dt: ElementKind| {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 256);
        let o = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::U8, 256);
        structure_key(OpCategory::BinaryElementwise, &[a, a, o], ArchSku::Sm89)
    };
    // f32 in, u8 mask out — the increment-0b headline cell.
    let lt = OpDef::elementwise_pred(
        "cmp_lt",
        2,
        &[ElementKind::F32],
        input(0).binary(BinaryOp::CmpLt, input(1)),
    );
    let k = generate(&lt, &pred_key(ElementKind::F32), &Cuda);
    let ptx = cc
        .compile(&k.source, &k.name, 5000)
        .expect("f32/u8 cmp compiles headerless");
    assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
    // f16 promote form (fp16 header under nvrtc).
    let lth = OpDef::elementwise_pred(
        "cmp_lt",
        2,
        &[ElementKind::F16],
        input(0).binary(BinaryOp::CmpLt, input(1)),
    );
    let kh = generate(&lth, &pred_key(ElementKind::F16), &Cuda);
    let ptxh = cc
        .compile(&kh.source, &kh.name, 5000)
        .expect("f16/u8 cmp compiles");
    assert!(String::from_utf8(ptxh).unwrap().contains(".entry"));
    // Nested mask-multiply (float out, no u8 machinery).
    let bw = OpDef::elementwise(
        "relu_bw",
        2,
        &[ElementKind::F32],
        input(0) * input(1).binary(BinaryOp::CmpGt, konst(0.0)),
    );
    let bkey = {
        let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    };
    let kb = generate(&bw, &bkey, &Cuda);
    let ptxb = cc
        .compile(&kb.source, &kb.name, 5000)
        .expect("mask-multiply compiles");
    assert!(String::from_utf8(ptxb).unwrap().contains(".entry"));
}

/// The increment-0c integer kernels compile headerless under nvrtc: a
/// bitwise i32 kernel (raw C `&`, `int` pointers — zero includes) and a
/// u8 wrapping add (`unsigned char` pointers, promotion + store-truncate —
/// plain C, zero includes). Numeric correctness is proven by the nvcc host
/// harness (`ondevice/int_validate.cu` — see the ondevice README's
/// "int ops (increment 0c)" section for the measured results); this
/// guards headerless portability. Ignored (needs nvrtc + CUDA).
#[cfg(feature = "nvrtc")]
#[test]
#[ignore = "requires nvrtc runtime + CUDA install"]
fn nvrtc_compiles_int_bitwise_and_u8_add() {
    use unpopped::generate;
    use unpopped::ir::input;
    let cc = NvrtcCompiler::new(ArchSku::Sm89);
    let bkey = |dt: ElementKind| {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, 256);
        structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89)
    };
    let band = OpDef::elementwise(
        "band",
        2,
        &[ElementKind::I32],
        input(0).binary(BinaryOp::BitAnd, input(1)),
    );
    let k = generate(&band, &bkey(ElementKind::I32), &Cuda);
    let ptx = cc
        .compile(&k.source, &k.name, 5000)
        .expect("i32 bitand compiles headerless");
    assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
    let addu = OpDef::elementwise("add", 2, &[ElementKind::U8], input(0) + input(1));
    let ku = generate(&addu, &bkey(ElementKind::U8), &Cuda);
    let ptxu = cc
        .compile(&ku.source, &ku.name, 5000)
        .expect("u8 add compiles headerless");
    assert!(String::from_utf8(ptxu).unwrap().contains(".entry"));
}

/// The WHERE/SELECT kernels compile headerless under nvrtc: the f32/f64
/// select is a plain identity-cast ternary (zero includes — no INFINITY
/// macro, no headers), the triu select composes the Coord cast + cmp cond
/// + pick in one strided body, and the f16 raw-pick form (cond promote +
/// `(__half)` arm casts, incl. the double-literal Const arm through the
/// half converting ctor) rides the fp16 header nvrtc bundles. Numeric
/// correctness is the nvcc host harness (`ondevice/select_validate.cu`);
/// this guards headerless portability. Ignored (needs nvrtc + CUDA).
#[cfg(feature = "nvrtc")]
#[test]
#[ignore = "requires nvrtc runtime + CUDA install"]
fn nvrtc_compiles_select_kernels() {
    use unpopped::generate;
    use unpopped::ir::{BinaryOp, coord, input, konst};
    let cc = NvrtcCompiler::new(ArchSku::Sm89);
    let skey = |dt: ElementKind, n: usize, align: u32| {
        let a = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
        let ops: Vec<_> = std::iter::repeat_n(a, n).collect();
        structure_key(OpCategory::TernaryElementwise, &ops, ArchSku::Sm89)
    };
    // f32 + f64 select with a Const arm (the identity-cast pin) — headerless.
    for (dt, align) in [(ElementKind::F32, 4u32), (ElementKind::F64, 8)] {
        let op = OpDef::elementwise("selz", 2, &[dt], input(0).select(input(1), konst(0.0)));
        let k = generate(&op, &skey(dt, 3, align), &Cuda);
        let ptx = cc
            .compile(&k.source, &k.name, 5000)
            .unwrap_or_else(|e| panic!("{dt:?} select failed headerless nvrtc: {e}"));
        assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
    }
    // The triu select (Coord cond, strided).
    let triu = OpDef::elementwise(
        "triu_sel",
        1,
        &[ElementKind::F32],
        coord(1)
            .binary(BinaryOp::CmpGe, coord(0) + konst(0.0))
            .select(input(0), konst(0.0)),
    );
    let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let tkey = structure_key(OpCategory::BinaryElementwise, &[a, a], ArchSku::Sm89);
    let kt = generate(&triu, &tkey, &Cuda);
    let ptxt = cc
        .compile(&kt.source, &kt.name, 5000)
        .expect("triu select compiles headerless");
    assert!(String::from_utf8(ptxt).unwrap().contains(".entry"));
    // f16/bf16 raw-pick (incl. the `(__half)(0.0)` Const-arm ctor).
    for dt in [ElementKind::F16, ElementKind::Bf16] {
        let op = OpDef::elementwise("selz", 2, &[dt], input(0).select(input(1), konst(0.0)));
        let k = generate(&op, &skey(dt, 3, 2), &Cuda);
        let ptx = cc
            .compile(&k.source, &k.name, 5000)
            .unwrap_or_else(|e| panic!("{dt:?} select failed nvrtc: {e}"));
        assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
    }
}

/// The increment-0d Coord kernels compile headerless under nvrtc: the
/// triu-mask strided kernel (the `(float)c1` coordinate cast + compute-
/// dtype compare — plain C, zero includes) and the zero-input f64 iota
/// (`(double)c1`, no input pointers). Numeric correctness is proven by the
/// nvcc host harness (`ondevice/coord_validate.cu` — see the ondevice
/// README's "coord ops (increment 0d)" section); this guards headerless
/// portability. Ignored (needs nvrtc + CUDA).
#[cfg(feature = "nvrtc")]
#[test]
#[ignore = "requires nvrtc runtime + CUDA install"]
fn nvrtc_compiles_coord_kernels() {
    use unpopped::generate;
    use unpopped::ir::{coord, input, konst};
    let cc = NvrtcCompiler::new(ArchSku::Sm89);
    let a32 = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let tkey = structure_key(OpCategory::BinaryElementwise, &[a32, a32], ArchSku::Sm89);
    let triu = OpDef::elementwise(
        "triu_mask",
        1,
        &[ElementKind::F32],
        input(0) * coord(1).binary(BinaryOp::CmpGe, coord(0) + konst(0.0)),
    );
    let k = generate(&triu, &tkey, &Cuda);
    let ptx = cc
        .compile(&k.source, &k.name, 5000)
        .expect("triu-mask compiles headerless");
    assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
    let a64 = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F64, 256);
    let ikey = structure_key(OpCategory::UnaryElementwise, &[a64], ArchSku::Sm89);
    let iota = OpDef::elementwise("iota1", 0, &[ElementKind::F64], coord(1));
    let ki = generate(&iota, &ikey, &Cuda);
    let ptxi = cc
        .compile(&ki.source, &ki.name, 5000)
        .expect("f64 iota compiles headerless");
    assert!(String::from_utf8(ptxi).unwrap().contains(".entry"));
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
    use unpopped::ir::UnaryOp;
    use unpopped::{ReduceOp, generate, input};
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
    let ptx = cc
        .compile(&kf32.source, &kf32.name, 5000)
        .expect("f32 reduction compiles");
    assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
    // f16 sum — exercises __half2float + cuda_fp16.h under headerless nvrtc.
    let sum = OpDef::reduction("s", 1, &[ElementKind::F16], input(0), ReduceOp::Sum);
    let kf16 = generate(&sum, &red_key(ElementKind::F16), &Cuda);
    let ptx16 = cc
        .compile(&kf16.source, &kf16.name, 5000)
        .expect("f16 reduction compiles");
    assert!(String::from_utf8(ptx16).unwrap().contains(".entry"));
    // 0e: Prod (block_prod cooperative reducer) compiles headerless.
    let prod = OpDef::reduction("p", 1, &[ElementKind::F32], input(0), ReduceOp::Prod);
    let kp = generate(&prod, &red_key(ElementKind::F32), &Cuda);
    let ptxp = cc
        .compile(&kp.source, &kp.name, 5000)
        .expect("f32 prod reduction compiles");
    assert!(String::from_utf8(ptxp).unwrap().contains(".entry"));
    // 0e: norm2 = Sqrt(Sum(Sqr(x))) — the fused post-expr (sqrtf on red0)
    // compiles headerless too.
    let norm2 = OpDef::reduction_post(
        "norm2",
        1,
        &[ElementKind::F32],
        input(0).unary(UnaryOp::Sqr),
        ReduceOp::Sum,
        unpopped::ir::reduced(0).sqrt(),
    );
    let kn = generate(&norm2, &red_key(ElementKind::F32), &Cuda);
    let ptxn = cc
        .compile(&kn.source, &kn.name, 5000)
        .expect("f32 norm2-post compiles");
    assert!(String::from_utf8(ptxn).unwrap().contains(".entry"));
}

/// The fused RowReduce kernels (RmsNorm / Softmax) compile headerless under
/// nvrtc — the warp-shuffle/shared-mem block reduce, `rsqrtf`/`expf`, and (for
/// f16) `__half2float` + the fp16 header. Numeric correctness is proven via the
/// nvcc host harness; this guards headerless portability. Ignored (needs nvrtc).
#[cfg(feature = "nvrtc")]
#[test]
#[ignore = "requires nvrtc runtime + CUDA install"]
fn nvrtc_compiles_rowreduce_kernels() {
    use unpopped::ir::{ReduceOp, ReduceStage, UnaryOp, konst, reduced};
    use unpopped::{generate, input};
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
                ReduceStage {
                    pre: input(0).0,
                    op: ReduceOp::Max,
                },
                ReduceStage {
                    pre: (input(0) - reduced(0)).exp().0,
                    op: ReduceOp::Sum,
                },
            ],
            (input(0) - reduced(0)).exp() / reduced(1),
        )
    };
    for (op, dt, cat) in [
        (
            rms(ElementKind::F32),
            ElementKind::F32,
            OpCategory::Normalization,
        ),
        (sm(ElementKind::F32), ElementKind::F32, OpCategory::Softmax),
        // f16: exercises __half2float + cuda_fp16.h under headerless nvrtc.
        (
            rms(ElementKind::F16),
            ElementKind::F16,
            OpCategory::Normalization,
        ),
        // f64 / f32-strict: the double accumulator path relies on the `double`
        // __shfl_down_sync overload compiling headerless (the critics' flag).
        (
            rms(ElementKind::F64),
            ElementKind::F64,
            OpCategory::Normalization,
        ),
        (
            sm(ElementKind::F32Strict),
            ElementKind::F32Strict,
            OpCategory::Softmax,
        ),
    ] {
        let k = generate(&op, &key(dt, cat), &Cuda);
        let ptx = cc
            .compile(&k.source, &k.name, 5000)
            .unwrap_or_else(|e| panic!("{} failed to compile: {e}", k.name));
        assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
    }

    // Multi-input: weighted-RmsNorm (x + weight) + LayerNorm (x + weight + bias)
    // exercise the per-column in_i[j] index headerless.
    let dt = ElementKind::F32;
    let x = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
    let col = OperandDesc::new(2, &[256, 128], &[0, 1], dt, 256);
    let out = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
    let wrms = OpDef::row_reduce(
        "wrmsnorm",
        2,
        &[dt],
        vec![ReduceStage {
            pre: input(0).unary(UnaryOp::Sqr).0,
            op: ReduceOp::Mean,
        }],
        input(0) * (reduced(0) + konst(1e-5)).unary(UnaryOp::Rsqrt) * input(1),
    );
    let ln = OpDef::row_reduce(
        "layernorm",
        3,
        &[dt],
        vec![
            ReduceStage {
                pre: input(0).0,
                op: ReduceOp::Mean,
            },
            ReduceStage {
                pre: (input(0) - reduced(0)).unary(UnaryOp::Sqr).0,
                op: ReduceOp::Mean,
            },
        ],
        (input(0) - reduced(0)) * (reduced(1) + konst(1e-5)).unary(UnaryOp::Rsqrt) * input(1)
            + input(2),
    );
    for (op, ops) in [(wrms, vec![x, col, out]), (ln, vec![x, col, col, out])] {
        let mk = structure_key(OpCategory::Normalization, &ops, ArchSku::Sm89);
        let k = generate(&op, &mk, &Cuda);
        let ptx = cc
            .compile(&k.source, &k.name, 5000)
            .unwrap_or_else(|e| panic!("{} failed to compile: {e}", k.name));
        assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
    }

    // Increment 2: compound backward — softmax bw (two row-streamed inputs) and
    // layer_norm bw dx (two row-streamed + two per-row saved-stat scalars — the
    // hoisted `in_i[row]` load compiles headerless).
    let stream = OperandDesc::new(2, &[256, 128], &[128, 1], dt, 256);
    let rowscalar = OperandDesc::new(2, &[256, 128], &[1, 0], dt, 256);
    let softmax_bw = OpDef::row_reduce(
        "softmax_bw",
        2,
        &[dt],
        vec![ReduceStage {
            pre: (input(0) * input(1)).0,
            op: ReduceOp::Sum,
        }],
        input(0) * (input(1) - reduced(0)),
    );
    let ln_x_hat = (input(0) - input(2)) * input(3);
    let layer_norm_bw = OpDef::row_reduce(
        "layer_norm_bw",
        4,
        &[dt],
        vec![
            ReduceStage {
                pre: input(1).0,
                op: ReduceOp::Mean,
            },
            ReduceStage {
                pre: (input(1) * ln_x_hat.clone()).0,
                op: ReduceOp::Mean,
            },
        ],
        input(3) * (input(1) - reduced(0) - ln_x_hat * reduced(1)),
    );
    for (op, ops, cat) in [
        (
            softmax_bw,
            vec![stream, stream, stream],
            OpCategory::Softmax,
        ),
        (
            layer_norm_bw,
            vec![stream, stream, rowscalar, rowscalar, stream],
            OpCategory::Normalization,
        ),
    ] {
        let mk = structure_key(cat, &ops, ArchSku::Sm89);
        let k = generate(&op, &mk, &Cuda);
        let ptx = cc
            .compile(&k.source, &k.name, 5000)
            .unwrap_or_else(|e| panic!("{} failed to compile: {e}", k.name));
        assert!(String::from_utf8(ptx).unwrap().contains(".entry"));
    }
}
