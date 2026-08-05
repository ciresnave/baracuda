//! Emit the transposed-rhs and broadcast-batch-rhs contraction LAYOUT cells for
//! on-device numeric validation (sub-spec A, Tasks 5/8 emitter + Task 6/9 CPU
//! oracle; this is the on-device proof of the same emitted kernels). Companion
//! generator for `ondevice/contract_layout_validate.cu` — mirrors how
//! `emit_bias_batched` feeds `contract_bias_batched_validate.cu`.
//!
//! The layout CLASS comes from the operand STRIDE pattern alone
//! (`baracuda_kernel_vocab::structure_key`'s `classify_mat_layout` derives
//! `lhs_order`/`rhs_order` from strides, no `OpDef::with_views` needed —
//! `m`/`n`/`k`/`B` stay launch args, so the harness runs these SAME kernels at
//! small hand-checkable shapes, exactly as `emit_bias_batched` does):
//!
//!   - `bmm_transposed` batched `[B,M,K]·[B,K,N]`, rhs physically `[B,N,K]`
//!     (K inner per slice) — the SDPA `Q·Kᵀ` core.
//!   - `bmm_gqa` batched `[B,M,K]·[B,K,N]`, rhs BROADCAST over batch
//!     (stride 0) — GQA broadcast-KV, one KV slice shared by every
//!     batch/head group.
//!   - `bmm_gqa_t` both at once: broadcast-batch AND transposed rhs — the
//!     combined GQA+Kᵀ cell (Task 9's CPU-oracle cell).
//!
//! `cargo run -p baracuda-kernelgen --example emit_contract_layout -- <out-dir>`

use baracuda_kernel_vocab::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use baracuda_kernelgen::{ContractionAxes, Cuda, OpDef, generate, reduced};
use std::fs;

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: emit_contract_layout <out-dir>");
    fs::create_dir_all(&out).expect("create out dir");
    let f32 = ElementKind::F32;

    let write = |k: baracuda_kernelgen::GeneratedKernel| {
        let path = format!("{out}/{}.cu", k.name);
        fs::write(&path, &k.source).expect("write kernel");
        println!("generated {path}");
    };

    // Shared class shapes: B Tiny/Large-ish batch, M Tiny (the Tiny-M skinny
    // schedule ceiling), K/N Large — same class-picking shapes `emit_bias_batched`
    // uses for its B14 batched cell (`m/n/k`/`B` are launch args at runtime).
    let b = 8usize;
    let (m, k, n) = (8usize, 4096usize, 4096usize);

    // lhs [B,M,K] canonical row-major, shared by all three cells.
    let tlhs = OperandDesc::new(
        3,
        &[b as i64, m as i64, k as i64],
        &[(m * k) as i64, k as i64, 1],
        f32,
        256,
    );
    // out [B,M,N] canonical row-major, shared by all three cells.
    let tout = OperandDesc::new(
        3,
        &[b as i64, m as i64, n as i64],
        &[(m * n) as i64, n as i64, 1],
        f32,
        256,
    );

    // --- Cell 1: bmm_transposed — batched matmul, TRANSPOSED rhs. ---
    // rhs logical [B,K,N], physically stored [B,N,K] (K unit-stride, N strided
    // by k per slice): strides [n*k, 1, k].
    let trhs = OperandDesc::new(
        3,
        &[b as i64, k as i64, n as i64],
        &[(n * k) as i64, 1, k as i64],
        f32,
        256,
    );
    let tkey = structure_key(OpCategory::Gemm, &[tlhs, trhs, tout], ArchSku::Sm89);
    let bmm_t = OpDef::contraction(
        "bmm_transposed",
        &[f32],
        ContractionAxes::batched_matmul(),
        reduced(0),
    );
    write(generate(&bmm_t, &tkey, &Cuda));

    // --- Cell 2: bmm_gqa — batched matmul, BROADCAST-batch rhs (GQA KV). ---
    // rhs [B,K,N], canonical [K,N] order, B broadcast (stride 0): the KV slice
    // is shared across every batch/head group.
    let grhs = OperandDesc::new(
        3,
        &[b as i64, k as i64, n as i64],
        &[0, n as i64, 1],
        f32,
        256,
    );
    let gkey = structure_key(OpCategory::Gemm, &[tlhs, grhs, tout], ArchSku::Sm89);
    let bmm_g = OpDef::contraction(
        "bmm_gqa",
        &[f32],
        ContractionAxes::batched_matmul(),
        reduced(0),
    );
    write(generate(&bmm_g, &gkey, &Cuda));

    // --- Cell 3 (optional): bmm_gqa_t — broadcast-batch AND transposed rhs. ---
    // rhs [B,K,N], B broadcast (stride 0) AND K/N transposed (K unit-stride,
    // N strided by k): strides [0, 1, k]. The real combined GQA+Kᵀ cell.
    let gtrhs = OperandDesc::new(
        3,
        &[b as i64, k as i64, n as i64],
        &[0, 1, k as i64],
        f32,
        256,
    );
    let gtkey = structure_key(OpCategory::Gemm, &[tlhs, gtrhs, tout], ArchSku::Sm89);
    let bmm_gt = OpDef::contraction(
        "bmm_gqa_t",
        &[f32],
        ContractionAxes::batched_matmul(),
        reduced(0),
    );
    write(generate(&bmm_gt, &gtkey, &Cuda));

    println!("cells: {}", tkey.to_token());
    println!("cells: {}", gkey.to_token());
    println!("cells: {}", gtkey.to_token());
}
