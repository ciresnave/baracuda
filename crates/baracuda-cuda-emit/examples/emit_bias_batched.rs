//! Emit the B13/B14 contraction-growth cells for on-device numeric validation:
//! the **fused bias/activation epilogue** (B13) and the **batched
//! `[B,M,K]·[B,K,N]`** kernel (B14). Companion generator for
//! `ondevice/contract_bias_batched_validate.cu` — the analogue of how the item-10
//! `contract_validate.cu` cells come out of `bin/kernelgen`.
//!
//! Cells (the class extents pick the cell CLASS — Tiny-M, Large N/K — exactly as
//! the catalog's `contract_tll`; `m/n/k`/`B` are launch args, so the harness runs
//! these SAME kernels at small hand-checkable shapes):
//!   - `matmul_bias`       rank-2, `out = Σ_k lhs·rhs + bias[n]`     (bias load alone)
//!   - `matmul_bias_relu`  rank-2, `out = relu(Σ_k lhs·rhs + bias[n])` (fused epilogue)
//!   - `bmm`               rank-3, `out[b] = Σ_k lhs[b]·rhs[b]`      (batch strides alone)
//!   - `bmm_relu`          rank-3, `out[b] = relu(Σ_k lhs[b]·rhs[b])` (batch + epilogue)
//!
//! `cargo run -p baracuda-cuda-emit --example emit_bias_batched -- <out-dir>`

use baracuda_cuda_emit::Cuda;
use std::fs;
use unpopped::{ContractionAxes, OpDef, UnaryOp, generate, input, reduced};
use unpopped_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: emit_bias_batched <out-dir>");
    fs::create_dir_all(&out).expect("create out dir");
    let f32 = ElementKind::F32;

    let write = |k: unpopped::GeneratedKernel| {
        let path = format!("{out}/{}.cu", k.name);
        fs::write(&path, &k.source).expect("write kernel");
        println!("generated {path}");
    };

    // --- B13: fused matmul + per-column [N] bias, +/- relu (rank-2 Tiny-M cell). ---
    // The class operands match the catalog contraction cell: lhs [8,4096] (Tiny-M,
    // Large-K), rhs [4096,4096] (Large), bias [4096], out [8,4096].
    let lhs = OperandDesc::new(2, &[8, 4096], &[4096, 1], f32, 256);
    let rhs = OperandDesc::new(2, &[4096, 4096], &[4096, 1], f32, 256);
    let bias = OperandDesc::new(1, &[4096], &[1], f32, 256);
    let out2 = OperandDesc::new(2, &[8, 4096], &[4096, 1], f32, 256);
    let bias_key = structure_key(OpCategory::Gemm, &[lhs, rhs, bias, out2], ArchSku::Sm89);

    let matmul_bias = OpDef::contraction_bias("matmul_bias", &[f32], reduced(0) + input(2));
    let matmul_bias_relu = OpDef::contraction_bias(
        "matmul_bias_relu",
        &[f32],
        (reduced(0) + input(2)).unary(UnaryOp::Relu),
    );
    write(generate(&matmul_bias, &bias_key, &Cuda));
    write(generate(&matmul_bias_relu, &bias_key, &Cuda));

    // --- B14: batched [B,M,K]·[B,K,N], identity + relu epilogue (rank-3 cell). ---
    // Class operands: lhs [8,8,4096] (B Tiny, M Tiny, K Large), rhs [8,4096,4096],
    // out [8,8,4096].
    let blhs = OperandDesc::new(3, &[8, 8, 4096], &[8 * 4096, 4096, 1], f32, 256);
    let brhs = OperandDesc::new(3, &[8, 4096, 4096], &[4096 * 4096, 4096, 1], f32, 256);
    let bout = OperandDesc::new(3, &[8, 8, 4096], &[8 * 4096, 4096, 1], f32, 256);
    let bmm_key = structure_key(OpCategory::Gemm, &[blhs, brhs, bout], ArchSku::Sm89);

    let bmm = OpDef::contraction("bmm", &[f32], ContractionAxes::batched_matmul(), reduced(0));
    let bmm_relu = OpDef::contraction(
        "bmm_relu",
        &[f32],
        ContractionAxes::batched_matmul(),
        reduced(0).unary(UnaryOp::Relu),
    );
    write(generate(&bmm, &bmm_key, &Cuda));
    write(generate(&bmm_relu, &bmm_key, &Cuda));

    println!("cells: {}", bias_key.to_token());
    println!("cells: {}", bmm_key.to_token());
}
