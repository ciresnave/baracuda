//! `kernelgen` — thin CLI over the [`baracuda_kernelgen`] library.
//!
//! Usage: `kernelgen <out-dir>`. v1 emits the elementwise pilot cell (f32 `add`,
//! contiguous + V4) into `<out-dir>` via the CUDA backend. The spec-driven
//! matrix (ops × structure cells, eventually fed from Fuel telemetry) and a
//! `--backend` selector replace the hardcoded pilot next.

use baracuda_kernelgen::{
    derive_pattern, emit_dispatch_table, generate, input, konst, param, reduced, to_fkc, Cuda,
    OpDef, ReduceOp, ReduceStage, UnaryOp,
};
use baracuda_kernels_types::{
    seed_winner, structure_key, ArchSku, DispatchEntry, DispatchTable, ElementKind, OpCategory,
    OperandDesc,
};
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
        // Honest-miss invariant (§7): a generated cell must not also be a
        // vendor-routed cell. Elementwise cells never trip a seed, but we consult
        // the same oracle the GEMM path uses so the rule is enforced, not assumed.
        debug_assert!(
            seed_winner(&key).is_none(),
            "elementwise cell {} unexpectedly vendor-seeded",
            key.to_token()
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

    // --- Reductions + fused norms (contiguous last-axis float cells, [4096, 1024]) ---
    // A standalone mean reduction (the RmsNorm building block), f32 + f16.
    let mean = OpDef::reduction(
        "mean",
        1,
        &[ElementKind::F32, ElementKind::F16],
        input(0),
        ReduceOp::Mean,
    );
    for dt in [ElementKind::F32, ElementKind::F16] {
        let a = OperandDesc::new(2, &[4096, 1024], &[1024, 1], dt, 256);
        let o = OperandDesc::new(1, &[4096], &[1], dt, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let k = generate(&mean, &key, &Cuda);
        fs::write(format!("{out_dir}/{}.cu", k.name), &k.source).expect("write kernel");
        println!("generated {out_dir}/{}.cu", k.name);
    }

    // Fused norms: RmsNorm / Softmax (single input), weighted-RmsNorm / LayerNorm
    // (multi-input: x + per-column [k] weight/bias broadcast over the row axis).
    let dt = ElementKind::F32;
    let x = OperandDesc::new(2, &[4096, 1024], &[1024, 1], dt, 256);
    let col = OperandDesc::new(2, &[4096, 1024], &[0, 1], dt, 256); // weight/bias
    let full = OperandDesc::new(2, &[4096, 1024], &[1024, 1], dt, 256); // full-width output
    let rmsnorm = OpDef::row_reduce(
        "rmsnorm",
        1,
        &[dt],
        vec![ReduceStage {
            pre: input(0).unary(UnaryOp::Sqr).0,
            op: ReduceOp::Mean,
        }],
        input(0) * (reduced(0) + konst(1e-5)).unary(UnaryOp::Rsqrt),
    );
    let softmax = OpDef::row_reduce(
        "softmax",
        1,
        &[dt],
        vec![
            ReduceStage { pre: input(0).0, op: ReduceOp::Max },
            ReduceStage {
                pre: (input(0) - reduced(0)).exp().0,
                op: ReduceOp::Sum,
            },
        ],
        (input(0) - reduced(0)).exp() / reduced(1),
    );
    let wrmsnorm = OpDef::row_reduce(
        "wrmsnorm",
        2,
        &[dt],
        vec![ReduceStage {
            pre: input(0).unary(UnaryOp::Sqr).0,
            op: ReduceOp::Mean,
        }],
        input(0) * (reduced(0) + konst(1e-5)).unary(UnaryOp::Rsqrt) * input(1),
    );
    let layernorm = OpDef::row_reduce(
        "layernorm",
        3,
        &[dt],
        vec![
            ReduceStage { pre: input(0).0, op: ReduceOp::Mean },
            ReduceStage {
                pre: (input(0) - reduced(0)).unary(UnaryOp::Sqr).0,
                op: ReduceOp::Mean,
            },
        ],
        (input(0) - reduced(0)) * (reduced(1) + konst(1e-5)).unary(UnaryOp::Rsqrt) * input(1)
            + input(2),
    );
    for (op, ops, cat) in [
        (rmsnorm, vec![x, full], OpCategory::Normalization),
        (softmax, vec![x, full], OpCategory::Softmax),
        (wrmsnorm, vec![x, col, full], OpCategory::Normalization),
        (layernorm, vec![x, col, col, full], OpCategory::Normalization),
    ] {
        let key = structure_key(cat, &ops, ArchSku::Sm89);
        let k = generate(&op, &key, &Cuda);
        fs::write(format!("{out_dir}/{}.cu", k.name), &k.source).expect("write kernel");
        println!("generated {out_dir}/{}.cu  (cell {})", k.name, key.to_token());
    }

    // --- Dispatch table (item 07, §7 vendor-exclusion) --------------------------
    // Every cell generated above is an elementwise / reduction / norm class, none
    // of which trips a vendor-exclusion seed. The routing decisions kernelgen
    // records today are the *seeds*: cells we deliberately route to a vendor
    // library and therefore do NOT generate — the honest miss made explicit (no
    // `.cu`, no link-registry entry). The bench gate later merges measured rows
    // (`Provenance::Measured`) over these via `baracuda_kernels_types::merge`.
    let mut dispatch: Vec<DispatchEntry> = Vec::new();

    // Large aligned half-precision GEMM → cuBLAS. kernelgen cannot yet emit a
    // GEMM cell; this is the deliberate vendor route, not a forgotten kernel.
    for dt in [ElementKind::F16, ElementKind::Bf16] {
        let ga = OperandDesc::new(2, &[4096, 4096], &[4096, 1], dt, 256);
        let gb = OperandDesc::new(2, &[4096, 4096], &[4096, 1], dt, 256);
        let go = OperandDesc::new(2, &[4096, 4096], &[4096, 1], dt, 256);
        let gkey = structure_key(OpCategory::Gemm, &[ga, gb, go], ArchSku::Sm89);
        if let Some((winner, why)) = seed_winner(&gkey) {
            println!(
                "vendor-exclusion: route {} -> {} ({}); not generating",
                gkey.to_token(),
                winner.code(),
                why
            );
            dispatch.push(DispatchEntry::seeded(gkey.to_token(), winner));
        }
    }

    let table = DispatchTable::from_entries(dispatch);
    let dpath = format!("{out_dir}/dispatch_table.rs");
    fs::write(&dpath, emit_dispatch_table(&table)).expect("write dispatch table");
    println!(
        "emitted dispatch table -> {dpath}  ({} vendor-routed cell(s))",
        table.entries.len()
    );
}
