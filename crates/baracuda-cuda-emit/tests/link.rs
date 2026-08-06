//! Link-registry tests that generate a real CUDA kernel (`&Cuda`). Relocated from
//! baracuda-kernelgen `src/link.rs`'s `mod tests` during the Unpopped carve (step
//! 3): both assert a CUDA-flavoured structure key / `baracuda_gen_` entry point, so
//! they live with the CUDA backend crate. Public-API only (no widening needed).

use baracuda_cuda_emit::Cuda;
use baracuda_kernel_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use baracuda_kernelgen::generate;
use baracuda_kernelgen::ir::{OpDef, input};
use baracuda_kernelgen::link::{LinkEntry, emit_link_registry, link_entry};

fn entry(name: &str) -> LinkEntry {
    let op = OpDef::elementwise(name, 2, &[ElementKind::F32], input(0) + input(1));
    let a = OperandDesc::new(2, &[128, 256], &[256, 1], ElementKind::F32, 256);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
    let kernel = generate(&op, &key, &Cuda);
    link_entry(&op, &key, &kernel)
}

#[test]
fn entry_mirrors_contract_fields() {
    let e = entry("add");
    assert!(e.entry_point.contains("add"));
    assert!(e.structure_key.starts_with("sk3|bin|f32|cuda:sm89|"));
    assert_ne!(e.revision_hash, 0);
}

#[test]
fn registry_is_sorted_deduped_valid_rust() {
    let mut es = vec![entry("zeta"), entry("alpha"), entry("alpha")];
    // shuffle so sorting is observable.
    es.reverse();
    let src = emit_link_registry(&es);
    assert!(src.contains("pub static BARACUDA_LINK_REGISTRY: &[(&str, &str, u64)] = &["));
    // alpha sorts before zeta; the duplicate alpha collapses.
    let a = src.find("alpha").unwrap();
    let z = src.find("zeta").unwrap();
    assert!(a < z, "entries must be sorted by entry_point");
    assert_eq!(
        src.matches("\"baracuda_gen_alpha").count(),
        1,
        "duplicate collapsed"
    );
    // closes cleanly.
    assert!(src.trim_end().ends_with("];"));
}
