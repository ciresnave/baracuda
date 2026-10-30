//! Emitted kernels compile for Pascal (`cuda:sm61`) and the other pre-Ampere targets, no device needed.
//!
//! SA-13, hardware-free half: compiles a spread of generated kernels with NVRTC for `sm_61`, `sm_70`
//! and `sm_75` (and `sm_89` as the control) and prints a per-family table. It proves PTX generation
//! only; nothing here runs on a P40 (NOT run on device).
//!
//! CUDA 13.x NVRTC rejects `sm_61`, so this needs a 12.x NVRTC: set `BARACUDA_NVRTC_PATH` to
//! `nvrtc64_120_0.dll` of a CUDA 12.9 toolkit or redist (see `docs/sm61-nvrtc-toolkit-selection.md`).
//! Without one the test declares a skip instead of passing silently.
//!
//! Run with `cargo gpu-test -p baracuda-cuda-emit --features nvrtc,device-tests --test sm61_compiles`.
#![cfg(all(feature = "nvrtc", feature = "device-tests"))]

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::require_optional;
use unpopped::ir::{BinaryOp, Expr, OpDef, ReduceOp, UnaryOp, input, konst};
use unpopped::jit::Compiler;
use unpopped::try_generate;
use unpopped_vocab::{ElementKind, OpCategory, OperandDesc, TargetId, structure_key};

const TARGETS: [&str; 4] = ["cuda:sm61", "cuda:sm70", "cuda:sm75", "cuda:sm89"];
const FLOATS: [ElementKind; 4] = [
    ElementKind::F32,
    ElementKind::F64,
    ElementKind::F16,
    ElementKind::Bf16,
];

fn compiler(token: &str) -> NvrtcCompiler {
    NvrtcCompiler::new(TargetId::parse(token).expect("token")).expect("nvrtc target")
}

/// A trivial kernel on `sm_61`: `Err` means this NVRTC cannot target Pascal at all.
fn sm61_nvrtc_usable() -> Result<(), String> {
    let src = r#"extern "C" __global__ void k(int *x) { x[0] = __dp4a(x[0], x[1], 0); }"#;
    compiler("cuda:sm61").compile(src, "k", 5000).map(|_| ())
}

/// `(kernel name, arity, body)`; the name is a C identifier because the emitter embeds it.
fn elementwise_bodies() -> Vec<(&'static str, u8, Expr)> {
    let x = || input(0);
    let y = || input(1);
    vec![
        ("add1", 1, x() + konst(1.0)),
        ("mul", 2, x() * y()),
        ("max", 2, x().max(y())),
        ("relu", 1, x().relu()),
        ("silu", 1, x().silu()),
        ("gelu", 1, x().gelu()),
        ("sigmoid", 1, x().sigmoid()),
        ("tanh", 1, x().tanh()),
        ("exp", 1, x().exp()),
        ("sqrt", 1, x().sqrt()),
        ("sin", 1, x().sin()),
        ("pow", 2, x().pow(y())),
        ("floor", 1, x().unary(UnaryOp::Floor)),
        (
            "select",
            2,
            x().binary(BinaryOp::CmpGt, y()).select(x(), y()),
        ),
    ]
}

/// Generates and compiles one op; `Ok(None)` is a typed decline, `Ok(Some(()))` a compile.
fn run(op: &OpDef, cat: OpCategory, dt: ElementKind, token: &str) -> Result<Option<()>, String> {
    let (a, o) = match cat {
        OpCategory::Reduction => (
            OperandDesc::new(2, &[128, 4096], &[4096, 1], dt, 256),
            OperandDesc::new(1, &[128], &[1], dt, 256),
        ),
        _ => {
            let a = OperandDesc::new(1, &[20_000], &[1], dt, 256);
            (a, a)
        }
    };
    let operands: Vec<OperandDesc> = if cat == OpCategory::BinaryElementwise {
        vec![a, a, o]
    } else {
        vec![a, o]
    };
    let target = TargetId::parse(token).expect("token");
    let key = structure_key(cat, &operands, target);
    let Ok(kernel) = try_generate(op, &key, &Cuda) else {
        return Ok(None);
    };
    compiler(token)
        .compile(&kernel.source, &kernel.name, 5000)
        .map(|_| Some(()))
        .map_err(|e| format!("{token} {dt:?} {}: {e}\n{}", op.name, kernel.source))
}

#[derive(Default)]
struct Tally {
    compiled: u32,
    declined: u32,
    failed: Vec<String>,
}

fn sweep(token: &str) -> Vec<(String, Tally)> {
    let mut rows = Vec::new();
    for dt in FLOATS {
        let mut t = Tally::default();
        for (label, n, body) in elementwise_bodies() {
            let cat = if n == 2 {
                OpCategory::BinaryElementwise
            } else {
                OpCategory::UnaryElementwise
            };
            let op = OpDef::elementwise(label, n, &[dt], body);
            tally(&mut t, run(&op, cat, dt, token));
        }
        rows.push((format!("elementwise {dt:?}"), t));
        let mut t = Tally::default();
        for rop in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
            let op = OpDef::reduction("r", 1, &[dt], input(0), rop);
            tally(&mut t, run(&op, OpCategory::Reduction, dt, token));
        }
        rows.push((format!("reduction {dt:?}"), t));
    }
    rows
}

fn tally(t: &mut Tally, r: Result<Option<()>, String>) {
    match r {
        Ok(Some(())) => t.compiled += 1,
        Ok(None) => t.declined += 1,
        Err(e) => t.failed.push(e),
    }
}

#[test]
fn emitted_kernels_compile_for_pre_ampere_targets() {
    require_optional!(
        sm61_nvrtc_usable(),
        "an NVRTC that targets sm_61 (CUDA 12.x; set BARACUDA_NVRTC_PATH)"
    );
    let mut failures = Vec::new();
    let mut control_compiled = 0;
    for token in TARGETS {
        eprintln!("== {token}");
        for (family, t) in sweep(token) {
            eprintln!(
                "  {family:<22} compiled {:>2}  declined {:>2}  failed {:>2}",
                t.compiled,
                t.declined,
                t.failed.len()
            );
            if token == "cuda:sm89" {
                control_compiled += t.compiled;
            }
            failures.extend(t.failed);
        }
    }
    assert!(
        control_compiled > 0,
        "the sm_89 control compiled nothing: the sweep proves nothing"
    );
    assert!(
        failures.is_empty(),
        "{} generated kernels do not compile:\n{}",
        failures.len(),
        failures.join("\n---\n")
    );
}
