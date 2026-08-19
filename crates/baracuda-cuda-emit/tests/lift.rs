//! Hand-written-CUDA lifter round-trip tests that re-emit through a real backend
//! (`&Cuda`; the `&CpuC` leg is DEFERRED — unpopped 0.2.0 `c584818`). Relocated from baracuda-kernelgen `src/lift.rs`'s `mod
//! tests` during the Unpopped carve (step 3). The neutral lifter tests (parse /
//! refusal / KISS-Consume categories) stayed in kernelgen; only the two that
//! re-emit to CUDA moved here. Public-API only (no widening needed).

use baracuda_cuda_emit::Cuda;
use unpopped::generate;
use unpopped::lift::lift_elementwise;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: &[ElementKind] = &[ElementKind::F32];

// DEFERRED COVERAGE (unpopped 0.2.0, commit c584818): the `&CpuC` leg — and the
// CUDA-vs-CpuC cross-backend agreement — was dropped when the CpuC/Slang emitters
// left `unpopped`'s core for the (currently unpublished) `unpopped-cpu-c` /
// `unpopped-slang` crates. A dev-dep on an unpublished crate would re-block the sk4
// merge. RESTORE when those publish: re-add `CpuC`, the `cpuc` re-emit, and the
// host-loop assertion. The CUDA round-trip below is unaffected.
#[test]
fn round_trip_reemits_to_cuda() {
    // A hand-written CUDA elementwise kernel a "copy-paster" might have.
    let src = "__global__ void mul(const float* in0, const float* in1, float* out, long long n) {\n\
               long long i = blockIdx.x*blockDim.x + threadIdx.x;\n\
               long long step = gridDim.x*blockDim.x;\n\
               for (; i < n; i += step) { out[i] = in0[i] * in1[i]; }\n}";
    let lifted = lift_elementwise(src, "lifted_mul", F32).unwrap();
    assert_eq!(lifted.n_inputs, 2);
    // `align = 4` yields the Scalar (non-vectorized) schedule.
    let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 4);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
    let cuda = generate(&lifted.op, &key, &Cuda);
    // The op survived source → IR → re-emit: both inputs read, real CUDA kernel.
    assert!(
        cuda.source.contains("in0[") && cuda.source.contains("in1["),
        "missing an input:\n{}",
        cuda.source
    );
    assert!(
        cuda.source.contains("__global__"),
        "not a CUDA kernel:\n{}",
        cuda.source
    );
    // Visible payoff (run with --nocapture).
    println!("=== re-emitted CUDA ===\n{}", cuda.source);
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
