//! `kernelgen` — thin CLI over the [`baracuda_kernelgen`] library.
//!
//! Usage: `kernelgen <out-dir>`. v1 emits the elementwise pilot cell (f32 `add`,
//! contiguous + V4) into `<out-dir>` via the CUDA backend. The spec-driven
//! matrix (ops × structure cells, eventually fed from Fuel telemetry) and a
//! `--backend` selector replace the hardcoded pilot next.

use baracuda_kernelgen::{derive_pattern, generate, input, to_fkc, Cuda, OpDef};
use baracuda_kernels_types::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};
use std::fs;

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "generated".to_string());
    fs::create_dir_all(&out_dir).expect("create out dir");

    // v1 pilot op: elementwise add, fanned out over a few dtype cells.
    let dtypes = [ElementKind::F32, ElementKind::F16, ElementKind::F64];
    let add = OpDef::elementwise("add", 2, &dtypes, input(0) + input(1));

    for dt in dtypes {
        // A contiguous 1-D cell, 256-byte aligned, extent %8.
        let operand = OperandDesc::new(1, &[1 << 20], &[1], dt, 256);
        let key = structure_key(
            OpCategory::BinaryElementwise,
            &[operand, operand, operand],
            ArchSku::Sm89,
        );
        let kernel = generate(&add, &key, &Cuda);
        let path = format!("{out_dir}/{}.cu", kernel.name);
        fs::write(&path, &kernel.source).expect("write kernel");
        println!("generated {path}  (cell {})", key.to_token());
    }

    // A broadcast cell: in1 is a fully-broadcast scalar over a 2-D f32 output.
    let a = OperandDesc::new(2, &[4, 8], &[8, 1], ElementKind::F32, 256);
    let b = OperandDesc::new(2, &[4, 8], &[0, 0], ElementKind::F32, 256);
    let out = OperandDesc::new(2, &[4, 8], &[8, 1], ElementKind::F32, 256);
    let bkey = structure_key(OpCategory::BinaryElementwise, &[a, b, out], ArchSku::Sm89);
    let bk = generate(&add, &bkey, &Cuda);
    let bpath = format!("{out_dir}/{}.cu", bk.name);
    fs::write(&bpath, &bk.source).expect("write kernel");
    println!("generated {bpath}  (cell {})", bkey.to_token());

    // Derive the FKC pattern: block for the (elementwise) op, alongside the .cu.
    if let Ok(pat) = derive_pattern(&add) {
        let ppath = format!("{out_dir}/add.fkc.pattern");
        fs::write(&ppath, to_fkc(&pat)).expect("write pattern");
        println!("derived FKC pattern -> {ppath}");
    }
}
