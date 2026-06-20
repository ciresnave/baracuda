//! Emit a small FKC bundle to stdout — a worked example of the generator's
//! contract output (front-matter + a primitive, a fusion, and a scalar-param op).
//!
//! `cargo run -p baracuda-kernelgen --example emit_contract`

use baracuda_kernelgen::{
    contract, emit_link_registry, front_matter, generate, input, link_entry, param, Cuda, LinkEntry,
    OpDef,
};
use baracuda_kernels_types::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};

fn cell(n_operands: usize, op: OpCategory) -> baracuda_kernels_types::StructureKey {
    // [128, 256] row-major f32, 256-byte aligned (contiguous, float4-vectorizable).
    let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let operands: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
    structure_key(op, &operands, ArchSku::Sm89)
}

fn main() {
    print!("{}", front_matter("cuda", "feat/kernel-specialization@e3907f6"));
    println!();

    let add = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));
    let relu_add = OpDef::elementwise(
        "relu_add",
        2,
        &[ElementKind::F32],
        (input(0) + input(1)).relu(),
    );
    let affine_silu = OpDef::elementwise(
        "affine_silu",
        1,
        &[ElementKind::F32],
        (input(0) * param(0) + param(1)).silu(),
    );

    let mut registry: Vec<LinkEntry> = Vec::new();
    for (op, n) in [(&add, 3usize), (&relu_add, 3), (&affine_silu, 2)] {
        let key = cell(n, OpCategory::BinaryElementwise);
        let kernel = generate(op, &key, &Cuda);
        print!("{}", contract(op, &key, &kernel, "cuda"));
        println!();
        registry.push(link_entry(op, &key, &kernel));
    }

    // The link registry that resolves these entry_points at module load.
    println!("<!-- generated link_registry.rs -->");
    print!("{}", emit_link_registry(&registry));
}
