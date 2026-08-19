//! ON-DEVICE N2 verification: `Max` AND `Min` NaN-propagation survive the CUDA
//! compile chain (nvrtc -> ptxas -> driver JIT) on the RTX 4070.
//!
//! WHY this test exists (the gap a source golden cannot close):
//! KISS conformance N2 requires that if either operand is NaN, `Max`/`Min` yield
//! NaN. The emitter deliberately spells the NaN-PROPAGATING compare-select
//! (`Max`: `(a != a ? a : (b != b ? b : (a >= b ? a : b)))`; `Min`: the same shape
//! with `<`) and reserves `fmaxf`/`fminf` (NaN-SUPPRESSING) for the separate
//! `FmaxIeee`/`FminIeee` ops. But that spelling only DELIVERS N2 if nothing
//! downstream contracts the ternary back into a hardware `max.f32`/`min.f32`
//! (NaN-suppressing without the `.NaN` modifier). On CUDA that is decided by
//! ptxas / the driver JIT — a layer BELOW the emitted source, so every
//! source-text golden is structurally blind to it. Same "correct bytes, wrong
//! behavior" shape as the alpha.78 all-zero. The only way to know is to LAUNCH.
//!
//! `Max` and `Min` are covered SEPARATELY: they are different emitted expressions
//! (`>=` vs `<`), so verifying one does not verify the other — a transposed
//! comparison or operand-order slip in the `Min` arm is a near-miss `Max`'s lanes
//! cannot distinguish.
//!
//! Non-vacuity discipline (mirrors the portable-C `cpu_end_to_end`): production
//! optimization (nvrtc -> driver JIT default -O3, NOT -O0), SINGLE-NaN
//! discriminating lanes (a both-NaN lane still returns NaN under an fmaxf/fminf
//! mutation, so it cannot carry the test), finite positive-control lanes, and a
//! FINITE never-wrote sentinel (NaN is the valid N2 output here, so it cannot
//! double as a no-write marker), plus an assert-no-intrinsic-first so a passing
//! run means "the ternary survived", not "the suppressing intrinsic was there all
//! along and happened to be NaN-correct".
//!
//! NOTE: N2-survives-the-toolchain is a property of ONE target's optimizer, not of
//! the emitter — it does NOT transfer between targets and must be re-measured per
//! target. That is why this check migrates with the emitter into `unpopped-cuda`
//! rather than living in neutral core.
//!
//! Run under the gpu-run lock:
//!   pwsh scripts/gpu-run.ps1 -Project baracuda -- cargo test -p baracuda-kernels-bench \
//!       --test max_nan_propagation -- --ignored --nocapture

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::{Context, DeviceBuffer, Module, Stream};
use baracuda_kernels_bench::setup_device;
use baracuda_kernels_types::{ArchSku, ElementKind, OpCategory, OperandDesc};
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

/// Per-lane input pattern (period 5). Lanes 0 & 1 are the DISCRIMINATING
/// single-NaN cases; 4 is both-NaN (coverage, non-discriminating); 2 & 3 are
/// finite positive controls.
///   0: (NaN, 2.0)   -> N2: NaN            (in0 NaN)
///   1: (3.0, NaN)   -> N2: NaN            (in1 NaN — order independence)
///   2: (3.0, 5.0)   -> max 5.0 / min 3.0  (positive control)
///   3: (7.0, -1.0)  -> max 7.0 / min -1.0 (positive control)
///   4: (NaN, NaN)   -> NaN                (both-NaN, coverage only)
fn lane(i: usize) -> (f32, f32) {
    let nan = f32::NAN;
    match i % 5 {
        0 => (nan, 2.0),
        1 => (3.0, nan),
        2 => (3.0, 5.0),
        3 => (7.0, -1.0),
        _ => (nan, nan),
    }
}

fn expects_nan(i: usize) -> bool {
    matches!(i % 5, 0 | 1 | 4)
}

/// Emit `region_op` (a bare binary compare-select), run it through the real CUDA
/// production chain on-device, and assert N2 (single-NaN -> NaN) plus finite
/// positive controls. `is_max` selects the expected finite result and the
/// forbidden NaN-suppressing intrinsic.
fn check_op(ctx: &Context, stream: &Stream, region_op: &str, fused_id: &str, is_max: bool) {
    let forbidden = if is_max { "fmaxf" } else { "fminf" };

    // align=4 forces the SCALAR kernel (ABI: in0, in1, out, long long n). The NaN
    // concern applies to scalar and vectorized alike (both use the per-element
    // ternary); scalar just gives a known ABI.
    let rows = 1i64;
    let cols = 256i64;
    let numel = rows * cols;
    let n = numel as usize;
    let a = OperandDesc::new(2, &[rows, cols], &[cols, 1], ElementKind::F32, 4);
    let operands = vec![a, a, a];

    let region = op_node(region_op, vec![PatternNode::Bind(0), PatternNode::Bind(1)]);
    let req = JitRequest {
        region,
        n_inputs: 2,
        op_category: OpCategory::BinaryElementwise,
        operands,
        target: ArchSku::Sm89.into(),
        fused_op_id: fused_id.to_string(),
        budget: JitBudget {
            max_compile_ms: 5000,
        },
    };
    let resp = synthesize(&req, &Cuda, &NvrtcCompiler::new(ArchSku::Sm89))
        .unwrap_or_else(|e| panic!("synthesize {region_op}: {e:?}"));
    let name = resp.kernel.entry_point.clone();

    // Belt-and-suspenders: prove we are testing the NaN-PROPAGATING compare-select,
    // not the NaN-suppressing intrinsic (which would trivially "pass" a both-NaN
    // lane and fail single-NaN — the very thing under test).
    assert!(
        !resp.kernel.source.contains(forbidden),
        "{region_op} must emit the NaN-propagating ternary, not {forbidden}:\n{}",
        resp.kernel.source
    );
    assert!(
        name.ends_with("_scalar"),
        "expected a scalar kernel for align=4, got {name}"
    );

    // Compile the emitted source the way production does (nvrtc -> PTX -> driver
    // JIT at default -O3 on load). This is the exact chain where the feared
    // contraction lives.
    let ptx = NvrtcCompiler::new(ArchSku::Sm89)
        .compile(&resp.kernel.source, &name, 30_000)
        .unwrap_or_else(|e| panic!("nvrtc({name}) failed: {e}"));
    let ptx = String::from_utf8(ptx).unwrap();
    let module = Module::load_ptx(ctx, &ptx).expect("load PTX");
    let f = module.get_function(&name).expect("get function");

    let mut in0 = vec![0f32; n];
    let mut in1 = vec![0f32; n];
    for i in 0..n {
        let (x, y) = lane(i);
        in0[i] = x;
        in1[i] = y;
    }
    // FINITE, distinctive never-wrote sentinel (NaN is a VALID N2 output, so it
    // cannot double as a no-write marker).
    let sentinel = -777.0f32;
    let d_in0 = DeviceBuffer::from_slice(ctx, &in0).unwrap();
    let d_in1 = DeviceBuffer::from_slice(ctx, &in1).unwrap();
    let d_out = DeviceBuffer::from_slice(ctx, &vec![sentinel; n]).unwrap();

    let nn = numel; // scalar ABI: n = output element count (long long)
    let block = 256u32;
    let grid = ((numel as u32) + block - 1) / block;
    // SAFETY: scalar ABI (in0, in1, out, long long n) — the same marshalling the
    // relu(add) root-cause repro uses.
    unsafe {
        f.launch()
            .grid(grid)
            .block(block)
            .stream(stream)
            .arg(&d_in0)
            .arg(&d_in1)
            .arg(&d_out)
            .arg(&nn)
            .launch()
            .unwrap_or_else(|e| panic!("launch {region_op}: {e}"));
    }
    stream.synchronize().unwrap();
    let mut got = vec![0f32; n];
    d_out.copy_to_host(&mut got).unwrap();

    // Positive control: every lane must have been written (none left at sentinel).
    let unwritten: Vec<usize> = (0..n).filter(|&i| got[i] == sentinel).collect();
    assert!(
        unwritten.is_empty(),
        "{region_op}: kernel left {} lanes unwritten (== sentinel {sentinel}) — a \
         no-write kernel would falsely pass the NaN asserts; first few: {:?}",
        unwritten.len(),
        &unwritten[..unwritten.len().min(8)]
    );

    // N2: NaN-lanes must return NaN; finite lanes must equal the true max/min.
    let mut n2_violations: Vec<(usize, f32, f32, f32)> = Vec::new();
    for i in 0..n {
        if expects_nan(i) {
            if !got[i].is_nan() {
                n2_violations.push((i, in0[i], in1[i], got[i]));
            }
        } else {
            // Only reached for finite lanes (i % 5 in {2, 3}).
            let exp = if is_max {
                in0[i].max(in1[i])
            } else {
                in0[i].min(in1[i])
            };
            assert_eq!(
                got[i], exp,
                "{region_op} finite positive-control lane {i}: op({}, {}) = {} (expected {exp})",
                in0[i], in1[i], got[i]
            );
        }
    }

    let single_nan = (0..n).filter(|&i| matches!(i % 5, 0 | 1)).count();
    assert!(
        n2_violations.is_empty(),
        "N2 VIOLATED for {region_op} on {} lane(s): the op SUPPRESSED NaN — the \
         compile chain (nvrtc/ptxas/driver-JIT) contracted the NaN-propagating \
         ternary into a NaN-suppressing {forbidden}-class op. Silent wrong-output \
         bug on the primary target. Samples (idx, in0, in1, got): {:?}",
        n2_violations.len(),
        &n2_violations[..n2_violations.len().min(8)]
    );

    println!(
        "N2 VERIFIED ({region_op}) on the reference emitter (RTX 4070): single-NaN \
         lanes propagate NaN through nvrtc->ptxas->driver-JIT at -O3. {n} lanes \
         ({single_nan} discriminating single-NaN, rest finite/both-NaN controls); \
         kernel = {name}"
    );
}

#[test]
#[ignore = "requires CUDA + nvrtc; on-device N2 NaN-propagation check on the 4070"]
fn max_and_min_nan_propagate_through_cuda_compile_chain() {
    let (ctx, stream) = setup_device();
    // Serial (one context, one test) — never two GPU contexts in parallel.
    check_op(&ctx, &stream, "Maximum", "max_nan", true);
    check_op(&ctx, &stream, "Minimum", "min_nan", false);
}
