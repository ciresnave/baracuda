//! Hand-written-CUDA lifter round-trip tests that re-emit through real backends
//! (`&Cuda` and `&CpuC`). Relocated from baracuda-kernelgen `src/lift.rs`'s `mod
//! tests` during the Unpopped carve (step 3). The neutral lifter tests (parse /
//! refusal / KISS-Consume categories) stayed in kernelgen; only the two that
//! re-emit to CUDA moved here. Public-API only (no widening needed).
//!
//! The `&CpuC` leg (restored here) is what only this crate can carry: genuine
//! CUDA-plus-CpuC agreement on the same lifted IR. The published emitters'
//! own pairwise coverage lives in `unpopped-conformance`; this file does not
//! re-cover it.

use baracuda_cuda_emit::Cuda;
use unpopped::generate;
use unpopped::lift::lift_elementwise;
use unpopped_cpu_c::CpuC;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: &[ElementKind] = &[ElementKind::F32];

#[test]
fn round_trip_reemits_to_cuda_and_cpuc() {
    // A hand-written CUDA elementwise kernel a "copy-paster" might have.
    let src = "__global__ void mul(const float* in0, const float* in1, float* out, long long n) {\n\
               long long i = blockIdx.x*blockDim.x + threadIdx.x;\n\
               long long step = gridDim.x*blockDim.x;\n\
               for (; i < n; i += step) { out[i] = in0[i] * in1[i]; }\n}";
    let lifted = lift_elementwise(src, "lifted_mul", F32).unwrap();
    assert_eq!(lifted.n_inputs, 2);
    // Re-emit the SAME lifted IR to two different backends. `align = 4` yields the
    // Scalar (non-vectorized) schedule the CpuC v1 backend serves; F32 is inside
    // CpuC's compute allowlist (its positive control is implicit here — a CpuC that
    // declined f32 would fail the host-loop assertion below, not pass vacuously).
    let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 4);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
    let cuda = generate(&lifted.op, &key, &Cuda);
    let cpuc = generate(&lifted.op, &key, &CpuC);
    // Both backends read both inputs — the op survived source → IR → re-emit.
    for s in [&cuda.source, &cpuc.source] {
        assert!(
            s.contains("in0[") && s.contains("in1["),
            "missing an input:\n{s}"
        );
    }
    assert!(
        cuda.source.contains("__global__"),
        "not a CUDA kernel:\n{}",
        cuda.source
    );
    assert!(
        cpuc.source.contains("for (long long i"),
        "not a CpuC host loop:\n{}",
        cpuc.source
    );
    // Visible payoff (run with --nocapture).
    println!("=== re-emitted CUDA ===\n{}", cuda.source);
    println!("=== re-emitted CpuC ===\n{}", cpuc.source);
}

/// Dump the lift round-trip PAIR for the on-device differential validator
/// (`ondevice/lift_roundtrip_validate.cu`): the re-emitted CUDA kernel and a
/// hand-authored ORIGINAL with the SAME body. On device the two must produce
/// BIT-IDENTICAL output — the numerical proof that CUDA → IR → re-emit is
/// faithful. Mirrors `cuda::tests::dump_coord_unravel_helper` (env var +
/// `fs::write`). Run with:
///   `LIFT_OUT=<outdir> cargo test -p baracuda-cuda-emit --test lift dump_lift_roundtrip -- --ignored --nocapture`
#[test]
#[ignore = "writes the lift round-trip pair for the on-device differential validator"]
fn dump_lift_roundtrip() {
    let out = std::env::var("LIFT_OUT").unwrap_or_else(|_| ".".to_string());

    // A hand-written CUDA elementwise kernel with a couple of fused ops: a
    // per-element expression referencing TWO inputs, `in0` twice. The lifted
    // IR is re-emitted below; the same body is re-authored as `lift_orig`.
    let src = "__global__ void lift_src(const float* in0, const float* in1, float* out, long long n) {\n\
               long long i = blockIdx.x*blockDim.x + threadIdx.x;\n\
               long long step = gridDim.x*blockDim.x;\n\
               for (; i < n; i += step) { out[i] = in0[i] * in1[i] + in0[i]; }\n}";
    let lifted = lift_elementwise(src, "lift_rt", F32).unwrap();
    assert_eq!(lifted.n_inputs, 2);

    // Contiguous SCALAR cell: align = 4 forces Schedule::Scalar, so the
    // re-emitted signature is `(const float* in0, const float* in1, float*
    // out, long long n)` — the SAME shape as the hand-written original, so
    // the on-device harness launches both with identical args.
    let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 4);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
    let k = generate(&lifted.op, &key, &Cuda);

    // (a) the re-emitted CUDA kernel (its `.name` carries a cell suffix).
    let gen_path = format!("{out}/lift_gen.cu");
    std::fs::write(&gen_path, &k.source).unwrap();

    // (b) the ORIGINAL kernel, compilable, with the IDENTICAL body under the
    // fixed name `lift_orig` (the harness `#include`s both).
    let orig = "__global__ void lift_orig(const float* in0, const float* in1, float* out, long long n) {\n\
                long long i = blockIdx.x*blockDim.x + threadIdx.x;\n\
                long long step = gridDim.x*blockDim.x;\n\
                for (; i < n; i += step) { out[i] = in0[i] * in1[i] + in0[i]; }\n}\n";
    let orig_path = format!("{out}/lift_orig.cu");
    std::fs::write(&orig_path, orig).unwrap();

    println!("wrote {gen_path}");
    println!("wrote {orig_path}");
    // The harness must launch the generated kernel by THIS name.
    println!("GENERATED KERNEL NAME: {}", k.name);
    // Visible payoff (run with --nocapture): confirm the FP arithmetic matches.
    println!("=== re-emitted CUDA (lift_gen.cu) ===\n{}", k.source);
}
