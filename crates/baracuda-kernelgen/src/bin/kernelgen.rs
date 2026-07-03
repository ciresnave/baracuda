//! `kernelgen` — thin CLI over the [`baracuda_kernelgen`] library.
//!
//! Usage: `kernelgen <out-dir>`. v1 emits the elementwise pilot cell (f32 `add`,
//! contiguous + V4) into `<out-dir>` via the CUDA backend. The spec-driven
//! matrix (ops × structure cells, eventually fed from Fuel telemetry) and a
//! `--backend` selector replace the hardcoded pilot next.

use baracuda_kernelgen::{
    derive_pattern, emit_dispatch_table, generate, generate_variants, input, konst, param,
    reduced, to_fkc, Cuda, OpDef, ReduceOp, ReduceStage, UnaryOp,
};
use baracuda_kernels_types::{
    seed_winner, structure_key, ArchSku, AxisMask, DispatchEntry, DispatchTable, ElementKind,
    OpCategory, OperandDesc,
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

    // --- General-path reductions (item 03): outer-axis, multi-axis, keepdim ---
    // These exercise the kept-unravel + strided reduced-fold emit (NOT the
    // contiguous `base=o*k` fast path), so nvcc-compiling + numeric-diffing these
    // .cu against a torch/strided oracle (brief §7) is what validates the general
    // path on device. Output operands are dense (collapse) or keepdim (size-1
    // reduced axis) so the store is injective.
    let f32 = ElementKind::F32;
    // (i) Reduce axis 0 of [4096,1024] -> [1024] (Sum, outer axis, collapse).
    let sum_ax0 =
        OpDef::reduction_axes("sum", 1, &[f32], input(0), ReduceOp::Sum, AxisMask(0b01), false);
    // (ii) Reduce axes {0,1} of [64,128,256] -> [256] (Mean, multi-axis, collapse).
    let mean_ax01 =
        OpDef::reduction_axes("mean", 1, &[f32], input(0), ReduceOp::Mean, AxisMask(0b011), false);
    // (iii) Reduce axis 0 of [4096,1024] -> [1,1024] (Sum, outer axis, keepdim).
    let sum_ax0_kd =
        OpDef::reduction_axes("sum", 1, &[f32], input(0), ReduceOp::Sum, AxisMask(0b01), true);
    // (iv) Max over axis 0 (the has-flag / NaN-propagating fold).
    let max_ax0 =
        OpDef::reduction_axes("amax", 1, &[f32], input(0), ReduceOp::Max, AxisMask(0b01), false);
    // (v) Middle axis of a rank-3 tensor -> two kept axes (0 and 2).
    let sum_mid =
        OpDef::reduction_axes("sum", 1, &[f32], input(0), ReduceOp::Sum, AxisMask(0b010), false);
    // (vi) Reduce-all of a rank-2 tensor -> scalar (kept axes empty).
    let sum_all =
        OpDef::reduction_axes("sum", 1, &[f32], input(0), ReduceOp::Sum, AxisMask(0b011), false);
    let general_reductions = [
        (
            &sum_ax0,
            OperandDesc::new(2, &[4096, 1024], &[1024, 1], f32, 256),
            OperandDesc::new(1, &[1024], &[1], f32, 256),
        ),
        (
            &mean_ax01,
            OperandDesc::new(3, &[64, 128, 256], &[128 * 256, 256, 1], f32, 256),
            OperandDesc::new(1, &[256], &[1], f32, 256),
        ),
        (
            &sum_ax0_kd,
            OperandDesc::new(2, &[4096, 1024], &[1024, 1], f32, 256),
            OperandDesc::new(2, &[1, 1024], &[1024, 1], f32, 256), // keepdim: size-1 axis 0
        ),
        (
            &max_ax0,
            OperandDesc::new(2, &[4096, 1024], &[1024, 1], f32, 256),
            OperandDesc::new(1, &[1024], &[1], f32, 256),
        ),
        (
            &sum_mid,
            OperandDesc::new(3, &[64, 128, 256], &[128 * 256, 256, 1], f32, 256),
            OperandDesc::new(2, &[64, 256], &[256, 1], f32, 256), // collapse -> [64,256]
        ),
        (
            &sum_all,
            OperandDesc::new(2, &[4096, 1024], &[1024, 1], f32, 256),
            OperandDesc::new(1, &[1], &[1], f32, 256), // scalar
        ),
    ];
    for (op, a, o) in general_reductions {
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let k = generate(op, &key, &Cuda);
        fs::write(format!("{out_dir}/{}.cu", k.name), &k.source).expect("write kernel");
        println!("generated {out_dir}/{}.cu  (cell {})", k.name, key.to_token());
    }

    // --- Schedule variants (phase 2, ship-top-K): split-K for the outer-axis cell ---
    // The baseline `_reduce_sum_ax1` (one thread per column, 118 GB/s measured) is
    // emitted above; the split-K pair is the first bench-gated variant. Every
    // variant ships (Fuel is the runtime selector); the base stays the default.
    {
        let a = OperandDesc::new(2, &[8192, 8192], &[8192, 1], f32, 256);
        let o = OperandDesc::new(1, &[8192], &[1], f32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        for v in generate_variants(&sum_ax0, &key, &Cuda) {
            if v.tag == "base" {
                continue; // the baseline cell is already written above
            }
            for k in &v.kernels {
                fs::write(format!("{out_dir}/{}.cu", k.name), &k.source).expect("write kernel");
                println!(
                    "generated {out_dir}/{}.cu  (cell {} | variant {})",
                    k.name,
                    key.to_token(),
                    v.tag
                );
            }
        }
    }

    // --- Integer reductions (item 04): i32 Sum/Max, exact `long long` accumulator ---
    let i32 = ElementKind::I32;
    let isum = OpDef::reduction("sum", 1, &[i32], input(0), ReduceOp::Sum);
    let imax = OpDef::reduction("amax", 1, &[i32], input(0), ReduceOp::Max);
    for op in [&isum, &imax] {
        let a = OperandDesc::new(2, &[4096, 1024], &[1024, 1], i32, 256);
        let o = OperandDesc::new(1, &[4096], &[1], i32, 256);
        let key = structure_key(OpCategory::Reduction, &[a, o], ArchSku::Sm89);
        let k = generate(op, &key, &Cuda);
        fs::write(format!("{out_dir}/{}.cu", k.name), &k.source).expect("write kernel");
        println!("generated {out_dir}/{}.cu  (cell {})", k.name, key.to_token());
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

    // --- Shared-interior DAG demo (item 02): diamond out = g / (g + 1), g = a*b ---
    // The product `g` feeds both the numerator and the denominator; the DAG
    // emitter hoists it to one `tmp` computed once. Two cells exercise both hoist
    // paths: a scalar cell (align 4 ⇒ no vectorize) and a vectorized cell (align
    // 256 ⇒ float4, per-lane `tmp` blocks).
    let g = input(0) * input(1);
    let diamond = OpDef::elementwise("diamond", 2, &[ElementKind::F32], g.clone() / (g + konst(1.0)));
    for align in [4u32, 256u32] {
        let o = OperandDesc::new(1, &[1 << 20], &[1], ElementKind::F32, align);
        let key = structure_key(OpCategory::BinaryElementwise, &[o, o, o], ArchSku::Sm89);
        let k = generate(&diamond, &key, &Cuda);
        fs::write(format!("{out_dir}/{}.cu", k.name), &k.source).expect("write kernel");
        println!("generated {out_dir}/{}.cu  (cell {})", k.name, key.to_token());
    }

    // --- Packed-vs-scalar differential pairs (item 09) --------------------------
    // f16/bf16 elementwise at 256-byte alignment key V8 → the packed half2 path;
    // the SAME ops at 2-byte alignment key Scalar → the bit-exactness oracle the
    // on-device validator diffs the packed kernels against.
    for dt in [ElementKind::F16, ElementKind::Bf16] {
        let addh = OpDef::elementwise("add", 2, &[dt], input(0) + input(1));
        let reluh = OpDef::elementwise("relu_add", 2, &[dt], (input(0) + input(1)).relu());
        for op in [&addh, &reluh] {
            for align in [256u32, 2u32] {
                let o = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
                let key = structure_key(OpCategory::BinaryElementwise, &[o, o, o], ArchSku::Sm89);
                let k = generate(op, &key, &Cuda);
                fs::write(format!("{out_dir}/{}.cu", k.name), &k.source).expect("write kernel");
                println!("generated {out_dir}/{}.cu  (cell {})", k.name, key.to_token());
            }
        }
        // Tier-A unary pairs: the sign-bit / exact-product intrinsics whose
        // bit-identity to the scalar float-round-trip path must hold for every
        // NaN payload — the validator sweeps all 16-bit patterns through them.
        for (nm, uop) in [
            ("negx", UnaryOp::Neg),
            ("absx", UnaryOp::Abs),
            ("sqrx", UnaryOp::Sqr),
        ] {
            let op = OpDef::elementwise(nm, 1, &[dt], input(0).unary(uop));
            for align in [256u32, 2u32] {
                let o = OperandDesc::new(1, &[1 << 20], &[1], dt, align);
                let key = structure_key(OpCategory::UnaryElementwise, &[o, o], ArchSku::Sm89);
                let k = generate(&op, &key, &Cuda);
                fs::write(format!("{out_dir}/{}.cu", k.name), &k.source).expect("write kernel");
                println!("generated {out_dir}/{}.cu  (cell {})", k.name, key.to_token());
            }
        }
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
