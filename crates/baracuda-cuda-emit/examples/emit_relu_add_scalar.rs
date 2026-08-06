//! Emit the `relu(add(a,b))` f32 SCALAR-schedule kernel and print its source +
//! symbol — the exact shape Fuel bisected to an all-zero output at alpha.78
//! (76/77 PASS, 78 FAIL). Used to discriminate emitter-body vs launch-ABI:
//! diff this against the alpha.77 emission (signature-change ⇒ ABI/marshalling,
//! body/store-change ⇒ emitter, source-identical ⇒ jit/launch path).
//!
//! `cargo run -p baracuda-cuda-emit --example emit_relu_add_scalar`

use baracuda_cuda_emit::Cuda;
use baracuda_kernel_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use baracuda_kernelgen::{OpDef, generate, input};

fn main() {
    let op = OpDef::elementwise(
        "relu_add",
        2,
        &[ElementKind::F32],
        (input(0) + input(1)).relu(),
    );
    // align = 4 forces Schedule::Scalar (the general one-thread-per-element path,
    // NOT float4-vectorized) — the shape Fuel's synth seam lands on for a scalar
    // fused elementwise region.
    let a = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 4);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
    let k = generate(&op, &key, &Cuda);
    println!("=== SCALAR NAME ===\n{}", k.name);
    println!("=== SCALAR SOURCE ===\n{}", k.source);
    if let Some(c) = baracuda_kernelgen::contract(&op, &key, &k, &Cuda) {
        println!("=== SCALAR CONTRACT ===\n{c}");
    }

    // The VECTORIZED (float4) path — align 16 → Schedule::Vectorized{4}, count_unit 4.
    let av = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 16);
    let keyv = structure_key(OpCategory::BinaryElementwise, &[av, av, av], ArchSku::Sm89);
    let kv = generate(&op, &keyv, &Cuda);
    println!("=== VEC NAME ===\n{}", kv.name);
    println!("=== VEC SOURCE ===\n{}", kv.source);
    if let Some(c) = baracuda_kernelgen::contract(&op, &keyv, &kv, &Cuda) {
        println!("=== VEC CONTRACT ===\n{c}");
    }
}
