//! `kernelgen` — thin CLI over the [`baracuda_kernelgen`] library.
//!
//! Usage: `kernelgen <out-dir>`. v1 emits the elementwise pilot cell (f32 `add`,
//! contiguous + V4) into `<out-dir>` via the CUDA backend. The spec-driven
//! matrix (ops × structure cells, eventually fed from Fuel telemetry) and a
//! `--backend` selector replace the hardcoded pilot next.

use baracuda_kernelgen::{derive_pattern, generate, input, param, to_fkc, Cuda, OpDef};
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

    // Activation-epilogue op: relu(a + b), across float dtypes.
    let relu_add = OpDef::elementwise(
        "relu_add",
        2,
        &[ElementKind::F32, ElementKind::F16, ElementKind::Bf16],
        (input(0) + input(1)).relu(),
    );
    for dt in [ElementKind::F32, ElementKind::F16, ElementKind::Bf16] {
        let ro = OperandDesc::new(1, &[1 << 20], &[1], dt, 256);
        let rkey = structure_key(OpCategory::BinaryElementwise, &[ro, ro, ro], ArchSku::Sm89);
        let rk = generate(&relu_add, &rkey, &Cuda);
        let rpath = format!("{out_dir}/{}.cu", rk.name);
        fs::write(&rpath, &rk.source).expect("write kernel");
        println!("generated {rpath}");
    }
    if let Ok(pat) = derive_pattern(&relu_add) {
        fs::write(format!("{out_dir}/relu_add.fkc.pattern"), to_fkc(&pat)).expect("write pattern");
    }

    // Parametric op: relu(x * p0 + p1), f32 — runtime scalar params.
    let affine_relu = OpDef::elementwise(
        "affine_relu",
        1,
        &[ElementKind::F32],
        (input(0) * param(0) + param(1)).relu(),
    );
    let ao = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, 256);
    let akey = structure_key(OpCategory::UnaryElementwise, &[ao, ao], ArchSku::Sm89);
    let ak = generate(&affine_relu, &akey, &Cuda);
    fs::write(format!("{out_dir}/{}.cu", ak.name), &ak.source).expect("write kernel");
    println!("generated {out_dir}/{}.cu", ak.name);
}
