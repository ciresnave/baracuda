//! Tree-sitter lifter round-trip tests that re-emit through a real backend
//! (`&Cuda`). Relocated from baracuda-kernelgen `src/convert.rs`'s `mod tests`
//! during the Unpopped carve (step 3): the neutral lifter tests (CUDA/Slang parse,
//! reductions, scans) stayed in kernelgen; only the two that re-emit to CUDA moved
//! here. Gated on kernelgen's `convert` feature (the cc-built tree-sitter grammars).
//! Public-API only (the `convert` lifters are already `pub`; no widening needed).
#![cfg(feature = "convert")]

use baracuda_cuda_emit::Cuda;
use unpopped::convert::{lift_elementwise_slang, lift_reduction_cuda};
use unpopped::generate;
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

const F32: &[ElementKind] = &[ElementKind::F32];

const SLANG_MUL: &str = "StructuredBuffer<float> input0;\n\
    StructuredBuffer<float> input1;\n\
    RWStructuredBuffer<float> output;\n\
    [numthreads(256, 1, 1)]\n\
    void mul(uint3 tid : SV_DispatchThreadID) {\n\
        uint i = tid.x;\n\
        output[i] = input0[i] * input1[i];\n\
    }";

// DEFERRED COVERAGE (unpopped 0.2.0, c584818): the `&CpuC` re-emit leg was dropped
// when the CpuC emitter left `unpopped`'s core for the (unpublished) `unpopped-cpu-c`
// crate. RESTORE when it publishes: re-add `CpuC` + the `cpuc` port assertion. The
// Slang→CUDA cross-language port below is unaffected — the Slang *lifter* is the
// `convert` feature's parser, not the emitter that left the core.
#[test]
fn round_trip_reemits_to_cuda() {
    // Lift a Slang kernel, re-emit to CUDA — cross-language port.
    let lifted = lift_elementwise_slang(SLANG_MUL, "ported", F32).unwrap();
    let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 4);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
    let cuda = generate(&lifted.op, &key, &Cuda);
    assert!(cuda.source.contains("in0[") && cuda.source.contains("in1["));
}

#[test]
fn reduction_round_trips_to_cuda() {
    // Lift a naive sum reduction and re-emit — the generator produces the
    // optimized cooperative reduce (`baracuda_gen_<name>_f32_reduce_sum`).
    let src = "__global__ void sum(const float* in0, float* out, long long n){ float acc = 0.0f; for (long long i=0;i<n;i++) acc += in0[i]; out[0] = acc; }";
    let lifted = lift_reduction_cuda(src, "sum", F32).unwrap();
    let a = OperandDesc::new(2, &[256, 128], &[128, 1], ElementKind::F32, 256);
    let out = OperandDesc::new(1, &[256], &[1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::Reduction, &[a, out], ArchSku::Sm89);
    let k = generate(&lifted.op, &key, &Cuda);
    assert!(k.name.contains("reduce_sum"), "name was {}", k.name);
}
