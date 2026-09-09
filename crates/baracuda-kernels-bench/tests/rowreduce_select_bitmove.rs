//! A `Select`-bearing all-move RowReduce, compiled and run, with the payload.
//!
//! #113 declined `Select` from the §6.16-0009 bit-move route. Measured with the
//! decline removed, the reason I gave was wrong: I said "a narrow payload path
//! with a promoted CONDITION beside it is a bigger change than the route", and
//! the condition was never the problem. The RowReduce lowering used
//! `select_f32`, which casts BOTH ARMS to `float` — and assigning that to a
//! narrow `out[idx]` re-introduces the round trip the clause forbids.
//!
//! ⚠️ The fix was already in the file: `cuda_select` is the dtype-aware speller
//! seven elementwise sites use, whose narrow arms are `(__nv_bfloat16)(x)` — an
//! identity no-op on an already-narrow operand. RowReduce was the one path not
//! using it.
//!
//! The emitter-side test (`cuda.rs` `select_bitmove_arms_are_raw_picks`) asserts
//! the TEXT. This runs it, because the emitted ternary mixes a narrow arm with a
//! float-valued condition and nothing had established that nvrtc accepts it.
//!
//! `#[ignore]`d — needs a CUDA device + nvrtc. Run with:
//! `cargo test -p baracuda-kernels-bench --test rowreduce_select_bitmove -- --ignored`

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::DeviceBuffer;
use baracuda_driver::{Context, Device, Module, Stream, require_optional};
use baracuda_kernels_bench::{current_hwstamp, setup_device};
use baracuda_kernels_types::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use unpopped::Compiler;
use unpopped::GeneratedKernel;
use unpopped::ir::{BinaryOp, ReduceOp, ReduceStage, UnaryOp, reduced};
use unpopped::{OpDef, generate, input};

const ROWS: i64 = 32;
const K: i64 = 128;
const NAN_COL: usize = 60;
const HI_COL: usize = 7;

// bf16: +1.0 3F80  +3.0 4040  qNaN payload 0x41 -> 7FC1
const LO: u16 = 0x3F80;
const HI: u16 = 0x4040;
const NAN_IN: u16 = 0x7FC1;

/// `x < rowmax ? rowmax : -x` over a `Max` fold — every transformation on the
/// path from input to output is a move, so §6.16-0009 governs the whole op.
fn emit() -> GeneratedKernel {
    let op = OpDef::row_reduce(
        "sel",
        1,
        &[ElementKind::Bf16],
        vec![ReduceStage {
            pre: input(0).0,
            op: ReduceOp::Max,
        }],
        input(0)
            .binary(BinaryOp::CmpLt, reduced(0))
            .select(reduced(0), input(0).unary(UnaryOp::Neg)),
    );
    let x = OperandDesc::new(2, &[ROWS, K], &[K, 1], ElementKind::Bf16, 256);
    let key = structure_key(OpCategory::Softmax, &[x, x], ArchSku::Sm89);
    let kernel = generate(&op, &key, &Cuda);
    // HARNESS CONTROL: a claim about the EMITTER, checked before any launch.
    // Without it every assertion below measures the promoted path under a
    // bit-move name.
    assert!(
        kernel.source.contains("__nv_bfloat16 e = in0[")
            && kernel.source.contains("? (__nv_bfloat16)("),
        "harness: the emitter did not take the bit-move Select route\n{}",
        kernel.source
    );
    kernel
}

fn run(ctx: &Context, stream: &Stream, kernel: &GeneratedKernel, host: &[u16]) -> Vec<u16> {
    // ⚠️ THE UNTESTED SHAPE. The emitted ternary has narrow arms and a
    // float-valued condition; if nvrtc rejects that mix, it fails here, loudly.
    let compiler = NvrtcCompiler::new(ArchSku::Sm89);
    let ptx = compiler
        .compile(&kernel.source, &kernel.name, 30_000)
        .unwrap_or_else(|e| {
            panic!(
                "nvrtc REJECTED the Select bit-move kernel: {e}\n{}",
                kernel.source
            )
        });
    let ptx = String::from_utf8(ptx).expect("ptx is text");
    let module = Module::load_ptx(ctx, &ptx).expect("module load");
    let f = module.get_function(&kernel.name).expect("get_function");
    let d_in = DeviceBuffer::from_slice(ctx, host).expect("d_in");
    let d_out = DeviceBuffer::<u16>::new(ctx, host.len()).expect("d_out");
    // SAFETY: the generated rowreduce signature is (in0, out, n_out, k); both
    // buffers outlive the launch, and u16 matches the bf16 storage width.
    unsafe {
        f.launch()
            .grid(ROWS as u32)
            .block(256u32)
            .stream(stream)
            .arg(&d_in)
            .arg(&d_out)
            .arg(&ROWS)
            .arg(&K)
            .launch()
            .expect("launch");
    }
    stream.synchronize().expect("sync");
    let mut out = vec![0u16; host.len()];
    d_out.copy_to_host(&mut out).expect("copy out");
    out
}

/// Row 0 carries the NaN; every row carries `HI` at `HI_COL` and `LO` elsewhere.
fn rows() -> Vec<u16> {
    let k = K as usize;
    let mut host = vec![LO; (ROWS * K) as usize];
    for r in 0..ROWS as usize {
        host[r * k + HI_COL] = HI;
    }
    host[NAN_COL] = NAN_IN;
    host
}

#[test]
#[ignore = "requires a CUDA device + nvrtc"]
fn select_bitmove_preserves_the_nan_payload() {
    let _serial = baracuda_kernels_bench::DEVICE_TIMING_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (ctx, stream) = setup_device();
    let device = Device::get(0).expect("device");
    let stamp = current_hwstamp(&device).expect("hwstamp");
    require_optional!(
        (stamp.target == ArchSku::Sm89.into()).then_some(()),
        "an sm89 device"
    );
    let out = run(&ctx, &stream, &emit(), &rows());
    let k = K as usize;

    // ROW 0 carries a NaN, so `rowmax` is NaN and `x < NaN` is FALSE for every
    // element (ordered compare). So every element takes the `-x` arm — which
    // means the NaN element's own output is the payload with its sign flipped.
    let want_nan = NAN_IN ^ 0x8000;
    eprintln!(
        "  NaN element:  want 0x{want_nan:04X}  got 0x{:04X}   (promoted path gives 0x{:04X})",
        out[NAN_COL],
        0x7FFFu16 ^ 0x8000
    );
    assert_eq!(
        out[NAN_COL],
        want_nan,
        "the NaN payload did not survive the Select arm; 0x{:04X} is what the \
         promoted path produces",
        0x7FFFu16 ^ 0x8000
    );
    // Same row, a finite element: also the `-x` arm, because nothing is < NaN.
    assert_eq!(
        out[0],
        LO ^ 0x8000,
        "row 0, finite element took the wrong arm"
    );

    // ⚠️ CONTROL — ROW 1 HAS NO NaN, SO IT EXERCISES THE OTHER ARM. Without it
    // every assertion above is satisfied by a kernel that ignores the fold and
    // always negates its input.
    eprintln!(
        "  row 1 LO elem: want 0x{HI:04X} got 0x{:04X}   HI elem: want 0x{:04X} got 0x{:04X}",
        out[k],
        HI ^ 0x8000,
        out[k + HI_COL]
    );
    assert_eq!(
        out[k], HI,
        "a below-max element must take the `rowmax` arm — the fold is not running"
    );
    assert_eq!(
        out[k + HI_COL],
        HI ^ 0x8000,
        "the max element itself is not < rowmax, so it must take the `-x` arm"
    );
}
