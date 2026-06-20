//! `kernelgen` — thin CLI over the [`baracuda_kernelgen`] library.
//!
//! Usage: `kernelgen <out-dir>`. v1 emits the elementwise pilot cell (f32 `add`,
//! contiguous + V4) into `<out-dir>` via the CUDA backend. The spec-driven
//! matrix (ops × structure cells, eventually fed from Fuel telemetry) and a
//! `--backend` selector replace the hardcoded pilot next.

use baracuda_kernelgen::{generate, input, Cuda, OpDef};
use baracuda_kernels_types::{structure_key, ArchSku, ElementKind, OpCategory, OperandDesc};
use std::fs;

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| "generated".to_string());
    fs::create_dir_all(&out_dir).expect("create out dir");

    // v1 pilot op: elementwise add over f32.
    let add = OpDef::elementwise("add", 2, &[ElementKind::F32], input(0) + input(1));

    // A contiguous 1-D f32 cell, 256-byte aligned, extent %4 → V4.
    let operand = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
    let key = structure_key(
        OpCategory::BinaryElementwise,
        &[operand, operand, operand],
        ArchSku::Sm89,
    );

    let kernel = generate(&add, &key, &Cuda);
    let path = format!("{out_dir}/{}.cu", kernel.name);
    fs::write(&path, &kernel.source).expect("write kernel");
    println!("generated {path}  (backend cuda, cell {})", key.to_token());
}
