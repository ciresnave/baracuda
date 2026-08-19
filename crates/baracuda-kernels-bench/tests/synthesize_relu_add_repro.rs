//! ROOT-CAUSE repro for the alpha.78 synthesized-relu(add) all-zero (Fuel bisect:
//! 76/77 PASS, 78 FAIL). Unlike alpha78_relu_add_ondevice.rs (which drives the
//! `generate` path), this drives the ACTUAL path Fuel uses — `synthesize`
//! (region → region_to_op → build_plan → generate) — and launches each result
//! EXACTLY as Fuel's jit_cuda_load.rs does: it only launches a symbol ending in
//! `_scalar`, and passes n = output element count (scalar ABI: in0,in1,out,n).
//!
//! Run: `cargo test -p baracuda-kernels-bench --test synthesize_relu_add_repro -- --ignored --nocapture`

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::{DeviceBuffer, Module};
use baracuda_kernels_bench::setup_device;
use baracuda_kernels_types::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use unpopped::pattern::PatternNode;
use unpopped::{Compiler, JitBudget, JitRequest, synthesize};

fn op_node(op: &str, operands: Vec<PatternNode>) -> PatternNode {
    PatternNode::Op {
        op: op.to_string(),
        operands,
        consumers: None,
        extract: Vec::new(),
    }
}

fn relu_add_region() -> PatternNode {
    op_node(
        "Relu",
        vec![op_node(
            "Add",
            vec![PatternNode::Bind(0), PatternNode::Bind(1)],
        )],
    )
}

/// What a single config did — so the test can assert at least one config actually
/// launched and was verified, rather than silently passing on an all-declined /
/// all-vectorized sweep.
#[derive(PartialEq, Eq, Debug)]
enum Outcome {
    /// `synthesize` returned Err — no kernel produced.
    Declined,
    /// Synthesized a non-`_scalar` (vectorized) kernel — Fuel's scalar-only
    /// launcher declines it (this is the alpha.78 launch-path case).
    NonScalar,
    /// Launched a `_scalar` kernel AND asserted it byte-correct below.
    Launched,
}

/// Synthesize relu(add) for a given operand shape/align, then launch it the way
/// Fuel's scalar-only launcher would: skip unless the symbol ends in `_scalar`,
/// pass n = output element count. When a `_scalar` kernel launches, ASSERT it is
/// byte-correct — that is the correctness guarantee this guard exists to make.
fn run_case(
    ctx: &baracuda_driver::Context,
    stream: &baracuda_driver::Stream,
    rows: i64,
    cols: i64,
    align: u32,
) -> Outcome {
    let numel = rows * cols;
    let a = OperandDesc::new(2, &[rows, cols], &[cols, 1], ElementKind::F32, align);
    let operands = vec![a, a, a];
    let req = JitRequest {
        region: relu_add_region(),
        n_inputs: 2,
        op_category: OpCategory::BinaryElementwise,
        operands: operands.clone(),
        target: ArchSku::Sm89.into(),
        fused_op_id: "jit_relu_add".to_string(),
        budget: JitBudget {
            max_compile_ms: 5000,
        },
    };
    // Sanity: the structure key the synth builds for these operands.
    let _sk = structure_key(OpCategory::BinaryElementwise, &operands, ArchSku::Sm89);

    let resp = match synthesize(&req, &Cuda, &NvrtcCompiler::new(ArchSku::Sm89)) {
        Ok(r) => r,
        Err(e) => {
            println!("[rows={rows} cols={cols} align={align}] synthesize DECLINED: {e:?}");
            return Outcome::Declined;
        }
    };
    let name = resp.kernel.entry_point.clone();
    let is_scalar = name.ends_with("_scalar");
    println!(
        "\n[rows={rows} cols={cols} align={align}] synth name = {name}  (Fuel {})",
        if is_scalar {
            "LAUNCHES (scalar)"
        } else {
            "DECLINES (non-scalar suffix)"
        }
    );
    // Emit the signature line for the record.
    if let Some(sig) = resp.kernel.source.lines().find(|l| l.contains("void ")) {
        println!("  sig: {}", sig.trim());
    }
    if !is_scalar {
        return Outcome::NonScalar; // Fuel would decline — not the all-zero path.
    }

    // Fuel's launch: load PTX, n = output element count, scalar ABI.
    let ptx = NvrtcCompiler::new(ArchSku::Sm89)
        .compile(&resp.kernel.source, &name, 30_000)
        .unwrap_or_else(|e| panic!("nvrtc({name}) failed: {e}"));
    let ptx = String::from_utf8(ptx).unwrap();
    let module = Module::load_ptx(ctx, &ptx).expect("load");
    let f = module.get_function(&name).expect("func");

    let in0: Vec<f32> = (0..numel)
        .map(|i| (i as f32) - (numel as f32) / 2.0)
        .collect();
    let in1: Vec<f32> = vec![0.25f32; numel as usize];
    let expected: Vec<f32> = in0
        .iter()
        .zip(&in1)
        .map(|(&a, &b)| {
            let s = a + b;
            if s < 0.0 { 0.0 } else { s }
        })
        .collect();
    let d_in0 = DeviceBuffer::from_slice(ctx, &in0).unwrap();
    let d_in1 = DeviceBuffer::from_slice(ctx, &in1).unwrap();
    let d_out = DeviceBuffer::from_slice(ctx, &vec![f32::NAN; numel as usize]).unwrap();

    let n = numel; // Fuel: n = output element count
    let block = 256u32;
    let grid = ((numel as u32) + block - 1) / block;
    // SAFETY: scalar ABI (in0, in1, out, long long n) — Fuel's exact marshalling.
    unsafe {
        f.launch()
            .grid(grid)
            .block(block)
            .stream(stream)
            .arg(&d_in0)
            .arg(&d_in1)
            .arg(&d_out)
            .arg(&n)
            .launch()
            .unwrap_or_else(|e| panic!("launch: {e}"));
    }
    stream.synchronize().unwrap();
    let mut got = vec![0f32; numel as usize];
    d_out.copy_to_host(&mut got).unwrap();

    let n_nan = got.iter().filter(|x| x.is_nan()).count();
    let n_zero = got.iter().filter(|&&x| x == 0.0).count();
    let n_pos = got.iter().filter(|&&x| x > 0.0).count();
    let mism = got.iter().zip(&expected).filter(|(g, e)| g != e).count();
    println!(
        "  LAUNCHED n={n}: {n_nan} never-wrote, {n_zero} zero, {n_pos} positive; {mism} mismatches vs relu(a+b)  => {}",
        if n_pos == 0 && n_nan == 0 {
            "ALL-ZERO REGRESSION REPRODUCED"
        } else if mism == 0 {
            "CORRECT"
        } else {
            "WRONG (partial)"
        }
    );

    // THE CORRECTNESS GUARANTEE (previously only PRINTED — the defect this guard
    // carried): a _scalar kernel that launches MUST write relu(a+b) exactly. The
    // alpha.78 all-zero was Fuel's scalar-only launcher DECLINING a vectorized
    // kernel — a launch-path bug, not the emitter. A _scalar kernel that launches
    // here has no such excuse: never-written lanes or byte-wrongness = an emitter
    // regression, and the test must FAIL, not print "WRONG" and report ok.
    assert_eq!(
        n_nan, 0,
        "[rows={rows} cols={cols} align={align}] {name}: {n_nan}/{numel} lanes never written"
    );
    assert_eq!(
        mism, 0,
        "[rows={rows} cols={cols} align={align}] {name}: {mism}/{numel} lanes wrong vs relu(a+b) (n_zero={n_zero} n_pos={n_pos})"
    );
    Outcome::Launched
}

#[test]
#[ignore = "requires CUDA + nvrtc; on-device synthesize-path correctness guard"]
fn synthesize_relu_add_across_configs() {
    let (ctx, stream) = setup_device();
    // Sweep operand configs Fuel could plausibly send. align + contiguous-length
    // divisibility both gate the scalar-vs-vectorized decision. Each _scalar launch
    // is asserted byte-correct inside run_case.
    let outcomes = [
        run_case(&ctx, &stream, 128, 256, 256), // aligned, cols%4==0 -> likely vectorized
        run_case(&ctx, &stream, 128, 256, 4),   // align 4 -> scalar
        run_case(&ctx, &stream, 128, 255, 256), // cols%4 != 0 -> ?
        run_case(&ctx, &stream, 127, 129, 256), // odd dims
        run_case(&ctx, &stream, 1, 1000, 256),  // rank-2 row vector
        run_case(&ctx, &stream, 1000, 1, 256),  // column vector
    ];
    // A guard that verified nothing must not report ok. If every config declined or
    // synthesized a non-_scalar kernel, no correctness assertion ran — fail loud
    // rather than green vacuously.
    let launched = outcomes.iter().filter(|o| **o == Outcome::Launched).count();
    assert!(
        launched > 0,
        "no config synthesized a launchable _scalar kernel — the correctness guard \
         asserted nothing (outcomes: {outcomes:?})"
    );
    println!(
        "synthesize-path correctness guard: {launched}/{} configs launched _scalar + verified byte-correct",
        outcomes.len()
    );
}
