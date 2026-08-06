//! Emit a small FKC bundle to stdout — a worked example of the generator's
//! contract output (front-matter + a primitive; plus a fusion + a scalar-param op
//! whose CONTRACTS are generated but WITHHELD from the importable bundle).
//!
//! NOTE the two fused ops (`relu_add`, `affine_silu`) generate valid contracts,
//! but `bundle()` withholds them from the importable file: their `fused_op:` name
//! is not one of Fuel's FusedOps constants, and an unknown `fused_op:` is
//! bundle-FATAL (poisons every primitive beside it). So `sample_bundle.md` carries
//! only the `add` primitive — the fusions still generate + run AOT and ride the
//! JIT seam as bare blocks. See `contract::bundle`.
//!
//! `cargo run -p baracuda-cuda-emit --example emit_contract`

use baracuda_cuda_emit::Cuda;
use baracuda_kernel_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use baracuda_kernelgen::{
    LinkEntry, OpDef, bundle, contract, emit_link_registry, generate, input, link_entry, param,
};

fn cell(n_operands: usize, op: OpCategory) -> baracuda_kernel_vocab::StructureKey {
    // [128, 256] row-major f32, 256-byte aligned (contiguous, float4-vectorizable).
    let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let operands: Vec<_> = std::iter::repeat_n(a, n_operands).collect();
    structure_key(op, &operands, ArchSku::Sm89)
}

fn main() {
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

    // Collect the per-kernel contracts, then frame them into ONE importable
    // bundle via `bundle()` — front-matter + a `## <kernel>` heading per block
    // (headingless blocks are silently dropped by Fuel's parser).
    let mut contracts: Vec<String> = Vec::new();
    let mut registry: Vec<LinkEntry> = Vec::new();
    for (op, n) in [(&add, 3usize), (&relu_add, 3), (&affine_silu, 2)] {
        let key = cell(n, OpCategory::BinaryElementwise);
        let kernel = generate(op, &key, &Cuda);
        if let Some(c) = contract(op, &key, &kernel, &Cuda) {
            contracts.push(c);
        }
        registry.push(link_entry(op, &key, &kernel));
    }
    print!(
        "{}",
        bundle("cuda", "feat/kernel-specialization@e3907f6", &contracts)
    );
    println!();

    // The link registry that resolves these entry_points at module load.
    println!("<!-- generated link_registry.rs -->");
    print!("{}", emit_link_registry(&registry));
}
