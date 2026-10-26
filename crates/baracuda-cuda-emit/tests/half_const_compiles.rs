//! Every `f16` / `bf16` elementwise body that contains a literal is either declined or compiles.
//!
//! The emitter used to spell `__half + 1.0` for such a body, which NVRTC rejects as an ambiguous
//! operator (measured on CUDA 13.3); the source-text goldens never compiled it. This compiles the
//! generated source of a spread of half bodies with literals, so an uncompilable spelling cannot
//! hide behind a golden string again.
//!
//! Run with `cargo gpu-test -p baracuda-cuda-emit --features nvrtc,device-tests --test half_const_compiles`.
#![cfg(all(feature = "nvrtc", feature = "device-tests"))]

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use unpopped::ir::{BinaryOp, Expr, OpDef, input, konst};
use unpopped::jit::Compiler;
use unpopped::try_generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn bodies() -> Vec<(&'static str, Expr)> {
    let x = || input(0);
    let gt0 = || x().binary(BinaryOp::CmpGt, konst(0.0));
    vec![
        ("x + 1", x() + konst(1.0)),
        ("x - 1", x() - konst(1.0)),
        ("x * 2", x() * konst(2.0)),
        ("x / 2", x() / konst(2.0)),
        ("1 + x", konst(1.0) + x()),
        ("1 / x", konst(1.0) / x()),
        ("max(x, 1)", x().max(konst(1.0))),
        ("min(x, 1)", x().min(konst(1.0))),
        ("x > 0", gt0()),
        ("select(x > 0, x, 0)", gt0().select(x(), konst(0.0))),
        ("select(x > 0, 1, 0)", gt0().select(konst(1.0), konst(0.0))),
        ("(x * x) + 1", (x() * x()) + konst(1.0)),
        ("x * x", x() * x()),
    ]
}

/// What happened to one body.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Declined,
    Compiled,
}

fn check(dt: ElementKind, label: &str, body: Expr, n: i64) -> Result<Outcome, String> {
    let op = OpDef::elementwise("hc", 1, &[dt], body);
    let a = OperandDesc::new(1, &[n], &[1], dt, 256);
    let key = structure_key(OpCategory::UnaryElementwise, &[a, a], ArchSku::Sm89);
    let kernel = match try_generate(&op, &key, &Cuda) {
        Ok(k) => k,
        Err(_) => return Ok(Outcome::Declined),
    };
    let compiler = NvrtcCompiler::new(ArchSku::Sm89.into()).expect("nvrtc target");
    compiler
        .compile(&kernel.source, &kernel.name, 5000)
        .map(|_| Outcome::Compiled)
        .map_err(|e| format!("{dt:?} [{label}] n={n}: nvrtc: {e}\n{}", kernel.source))
}

#[test]
fn half_bodies_with_literals_are_declined_or_compile() {
    let mut failures = Vec::new();
    let (mut declined, mut compiled) = (0, 0);
    for dt in [ElementKind::F16, ElementKind::Bf16] {
        // An odd length takes the scalar schedule; a multiple of 8 takes the packed path.
        for n in [20_001i64, 20_000, 1 << 20] {
            for (label, body) in bodies() {
                match check(dt, label, body, n) {
                    Ok(Outcome::Declined) => declined += 1,
                    Ok(Outcome::Compiled) => compiled += 1,
                    Err(e) => failures.push(e),
                }
            }
        }
    }
    eprintln!(
        "half const sweep: {compiled} compiled, {declined} declined, {} failed",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{} generated half kernels do not compile:\n{}",
        failures.len(),
        failures.join("\n---\n")
    );
    // Controls: the sweep must see both outcomes, or it proves nothing.
    assert!(
        declined > 0,
        "no body was declined: the literal arithmetic is not being gated"
    );
    assert!(
        compiled > 0,
        "no body compiled: the sweep is not compiling anything"
    );
}
