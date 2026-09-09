//! The §6.16-0009 bit-move RowReduce route, COMPILED AND RUN, with the payload
//! it exists to preserve.
//!
//! The emitter-side test (`cuda.rs`
//! `rowreduce_all_move_takes_the_bit_move_route_per_6_16_0009`) asserts the
//! emitted TEXT. Text is not a kernel:
//!
//! ⚠️ **The narrow route emits `__shfl_down_sync(0xffffffffu, v, off)` where `v`
//! is `__nv_bfloat16` / `__half`, and nothing had measured that that overload
//! resolves.** `narrow_minmax_semantics.rs` measured the narrow `!=`, `>` and
//! bit-copy semantics — it did NOT measure the warp shuffle, and I nearly
//! shipped the wiring on the assumption that "the block reducers need no
//! change" extended to an intrinsic I had not probed. It does not follow: the
//! reducer's ARITHMETIC being fine says nothing about its DATA MOVEMENT.
//!
//! And the payload is the whole point. The promoted path canonicalizes a bf16
//! NaN 0x7FC1 to 0x7FFF (measured in the sibling file); this asserts the
//! bit-move path does not, END TO END through a real fold, a real warp shuffle
//! and a real store — not through a hand-written probe.
//!
//! `#[ignore]`d — needs a CUDA device + nvrtc. Run with:
//! `cargo test -p baracuda-kernels-bench --test rowreduce_bitmove_ondevice -- --ignored`

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::DeviceBuffer;
use baracuda_driver::{Device, Module, require_optional};
use baracuda_kernels_bench::{current_hwstamp, setup_device};
use baracuda_kernels_types::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use unpopped::Compiler;
use unpopped::ir::{ReduceOp, ReduceStage, UnaryOp, reduced};
use unpopped::{OpDef, generate, input};

const ROWS: i64 = 64;
const K: i64 = 256;

/// bf16 bits: a qNaN with a NON-DEFAULT payload, so "canonicalized to 0x7FFF"
/// is distinguishable from "preserved".
const NAN_PAYLOAD: u16 = 0x7FC1;
/// A large finite bf16 (~3.0) that must NOT win against the NaN.
const FINITE_HI: u16 = 0x4040;
/// A small finite bf16 (~1.0).
const FINITE_LO: u16 = 0x3F80;

#[test]
#[ignore = "requires a CUDA device + nvrtc"]
fn bitmove_rowreduce_preserves_the_nan_payload_end_to_end() {
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

    // The all-move shape: a Max fold with a pure sign-edit epilogue. Every
    // transformation from input to output is a move, so §6.16-0009 governs.
    let all_move = OpDef::row_reduce(
        "bm",
        1,
        &[ElementKind::Bf16],
        vec![ReduceStage {
            pre: input(0).0,
            op: ReduceOp::Max,
        }],
        reduced(0).unary(UnaryOp::Neg),
    );
    let x = OperandDesc::new(2, &[ROWS, K], &[K, 1], ElementKind::Bf16, 256);
    let key = structure_key(OpCategory::Softmax, &[x, x], ArchSku::Sm89);
    let kernel = generate(&all_move, &key, &Cuda);

    // HARNESS CONTROL: this test is meaningless unless the emitter actually took
    // the bit-move route. Assert the mechanism before launching anything.
    assert!(
        kernel.source.contains("__nv_bfloat16 e = in0["),
        "harness: the emitter did NOT take the bit-move route, so this test would \
         be measuring the promoted path under a bit-move name\n{}",
        kernel.source
    );

    // ⚠️ THE UNMEASURED INTRINSIC. If `__shfl_down_sync` has no `__nv_bfloat16`
    // overload, this is where it fails — loudly, at compile, naming the line.
    let compiler = NvrtcCompiler::new(ArchSku::Sm89);
    let ptx = compiler
        .compile(&kernel.source, &kernel.name, 30_000)
        .unwrap_or_else(|e| panic!("nvrtc REJECTED the bit-move kernel: {e}\n{}", kernel.source));
    let ptx = String::from_utf8(ptx).expect("ptx is text");
    let module = Module::load_ptx(&ctx, &ptx).expect("module load");
    let f = module.get_function(&kernel.name).expect("get_function");

    // Row 0 carries the payload NaN plus finite values; every other row is
    // finite only. NaN-propagating Max must elect the NaN in row 0.
    let n = (ROWS * K) as usize;
    let mut host = vec![FINITE_LO; n];
    for r in 0..ROWS as usize {
        host[r * K as usize + 7] = FINITE_HI;
    }
    host[0 * K as usize + 100] = NAN_PAYLOAD;

    let d_in = DeviceBuffer::from_slice(&ctx, &host).expect("d_in");
    let d_out = DeviceBuffer::<u16>::new(&ctx, n).expect("d_out");
    // SAFETY: the generated rowreduce signature is (in0, out, n_out, k); both
    // buffers outlive the launch, and u16 matches the bf16 storage width.
    unsafe {
        f.launch()
            .grid(ROWS as u32)
            .block(256u32)
            .stream(&stream)
            .arg(&d_in)
            .arg(&d_out)
            .arg(&ROWS)
            .arg(&K)
            .launch()
            .expect("launch");
    }
    stream.synchronize().expect("sync");
    let mut out = vec![0u16; n];
    d_out.copy_to_host(&mut out).expect("copy out");

    // Row 0: Max elects the NaN, the Neg epilogue flips its sign bit, and the
    // payload rides through untouched. Expected = NaN_PAYLOAD ^ 0x8000.
    let want_row0 = NAN_PAYLOAD ^ 0x8000;
    let got = out[0];
    eprintln!(
        "  row 0 (NaN payload):  want 0x{want_row0:04X}  got 0x{got:04X}  \
         (input 0x{NAN_PAYLOAD:04X}; the promoted path would give 0x{:04X})",
        0x7FFFu16 ^ 0x8000
    );
    assert_eq!(
        got,
        want_row0,
        "the NaN payload did not survive the fold. 0x{:04X} is the CANONICALIZED \
         value the promoted path produces — if that is what came back, the \
         bit-move route is not actually being taken on device",
        0x7FFFu16 ^ 0x8000
    );

    // POSITIVE CONTROL on a row with no NaN: Max elects FINITE_HI and the sign
    // flips. Without this, a kernel that wrote `input ^ 0x8000` unconditionally
    // would pass the assertion above while doing no reduction at all.
    let want_row1 = FINITE_HI ^ 0x8000;
    let got1 = out[K as usize];
    eprintln!("  row 1 (finite only):  want 0x{want_row1:04X}  got 0x{got1:04X}");
    assert_eq!(
        got1, want_row1,
        "a finite row did not reduce to its maximum — the fold is not running"
    );

    // And every element of row 0 carries the row's reduced value (full-width
    // output), so a kernel writing only element 0 cannot pass.
    let stragglers = (0..K as usize).filter(|&j| out[j] != want_row0).count();
    assert_eq!(
        stragglers, 0,
        "{stragglers} of {K} elements in row 0 disagree with the reduced value"
    );
}
