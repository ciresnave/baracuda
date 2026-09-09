//! Piece 3: the §6.16-0009 bit-move RowReduce route across the cells piece 2
//! did not reach — f16 as well as bf16, `Min` as well as `Max`, `Abs` as well
//! as `Neg`.
//!
//! Piece 2 (`rowreduce_bitmove_ondevice.rs`) proved the route for exactly ONE
//! cell: bf16, a single `Max` fold, a `Neg` epilogue. That is one point, and
//! the emitter's behaviour varies along three axes it did not touch:
//!
//! * **dtype** — different bit width, different intrinsics
//!   (`__bfloat16_as_ushort` vs `__half_as_ushort`), different NaN encodings.
//! * **fold op** — `Min` emits `<` where `Max` emits `>`. ⚠️ A transposed
//!   comparison is a near-miss the `Max` lanes cannot distinguish, which is
//!   `max_nan_propagation.rs`'s own argument for covering both separately.
//! * **epilogue** — `Abs` CLEARS the sign bit where `Neg` FLIPS it. Different
//!   masks, and only one of them is idempotent.
//!
//! # ⚠️ Two design choices that make the assertions able to fail
//!
//! **1. The `Abs` cells use a NEGATIVE NaN and negative finite rows.** `Abs`
//! on a positive value is the identity, so a positive-only `Abs` cell cannot
//! distinguish "the epilogue ran" from "nothing ran". Every `Abs` cell here
//! feeds sign-bit-set inputs, so the expected output differs from the input.
//!
//! **2. Expected values are LITERALS, hand-derived, never computed.** A test
//! that recomputes the kernel's own logic to build its expectation shares any
//! bug with it. The tables below are what bf16/f16 bit patterns mean, written
//! out; if the emitter and this file disagree, one of them is wrong and the
//! test can say so.
//!
//! `#[ignore]`d — needs a CUDA device + nvrtc. Run with:
//! `cargo test -p baracuda-kernels-bench --test rowreduce_bitmove_matrix -- --ignored`

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::DeviceBuffer;
use baracuda_driver::{Context, Device, Module, Stream, require_optional};
use baracuda_kernels_bench::{current_hwstamp, setup_device};
use baracuda_kernels_types::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use unpopped::Compiler;
use unpopped::GeneratedKernel;
use unpopped::ir::{ReduceOp, ReduceStage, UnaryOp, reduced};
use unpopped::{OpDef, generate, input};

const ROWS: i64 = 32;
const K: i64 = 128;

/// One matrix cell. Every expectation is a hand-written bit pattern.
struct Cell {
    name: &'static str,
    dt: ElementKind,
    fold: ReduceOp,
    epi: UnaryOp,
    /// The NaN that row 0 carries — payload chosen non-default so a
    /// canonicalized result (0x7FFF / 0xFFFF) is distinguishable.
    nan_in: u16,
    /// The two finite values every row carries.
    finite: (u16, u16),
    /// Row 0 after fold+epilogue. The fold elects the NaN (propagating
    /// Max/Min), so this is the epilogue applied to `nan_in`.
    want_nan_row: u16,
    /// A NaN-free row after fold+epilogue.
    want_finite_row: u16,
}

// bf16 is the top 16 bits of f32: s eeeeeeee mmmmmmm
//   +1.0 3F80   -1.0 BF80   +3.0 4040   -3.0 C040
//   qNaN payload 0x41:  +7FC1   -FFC1
// f16 is s eeeee mmmmmmmmmm
//   +1.0 3C00   -1.0 BC00   +3.0 4200   -3.0 C200
//   qNaN payload 0x01:  +7E01   -FE01
const CELLS: &[Cell] = &[
    Cell {
        name: "bf16 Min Neg",
        dt: ElementKind::Bf16,
        fold: ReduceOp::Min,
        epi: UnaryOp::Neg,
        nan_in: 0x7FC1,
        // Min over {+1.0, -1.0} elects -1.0 (BF80); Neg flips the sign -> 3F80.
        finite: (0x3F80, 0xBF80),
        want_nan_row: 0x7FC1 ^ 0x8000,
        want_finite_row: 0x3F80,
    },
    Cell {
        name: "bf16 Max Abs",
        dt: ElementKind::Bf16,
        fold: ReduceOp::Max,
        epi: UnaryOp::Abs,
        // NEGATIVE NaN: Abs must CLEAR the sign, so input != output.
        nan_in: 0xFFC1,
        // Both finites negative, so Max elects -1.0 (BF80); Abs -> 3F80.
        // ⚠️ ORDER MATTERS AND THIS PAIR WAS ORIGINALLY REVERSED. With
        // `finite.0 = BF80`, element 0 of the finite row IS the winner, so
        // Abs(element 0) == the expected value and a NO-FOLD kernel passed the
        // control. `-3.0` first makes the winner a value element 0 does not
        // hold. `assert_controls_discriminate` now enforces this.
        finite: (0xC040, 0xBF80),
        want_nan_row: 0x7FC1,
        want_finite_row: 0x3F80,
    },
    Cell {
        name: "f16 Max Neg",
        dt: ElementKind::F16,
        fold: ReduceOp::Max,
        epi: UnaryOp::Neg,
        nan_in: 0x7E01,
        // Max over {+1.0, +3.0} elects +3.0 (4200); Neg -> C200.
        finite: (0x3C00, 0x4200),
        want_nan_row: 0x7E01 ^ 0x8000,
        want_finite_row: 0xC200,
    },
    Cell {
        name: "f16 Min Abs",
        dt: ElementKind::F16,
        fold: ReduceOp::Min,
        epi: UnaryOp::Abs,
        nan_in: 0xFE01,
        // Min over {-1.0, -3.0} elects -3.0 (C200); Abs -> 4200.
        finite: (0xBC00, 0xC200),
        want_nan_row: 0x7E01,
        want_finite_row: 0x4200,
    },
];

/// Emit the cell and assert the emitter took the bit-move route.
///
/// The control is here, with the emission: it is a claim about the EMITTER,
/// and without it every device assertion measures the promoted path under a
/// bit-move name.
fn emit(cell: &Cell) -> GeneratedKernel {
    let op = OpDef::row_reduce(
        "bm",
        1,
        &[cell.dt],
        vec![ReduceStage {
            pre: input(0).0,
            op: cell.fold,
        }],
        reduced(0).unary(cell.epi),
    );
    let x = OperandDesc::new(2, &[ROWS, K], &[K, 1], cell.dt, 256);
    let key = structure_key(OpCategory::Softmax, &[x, x], ArchSku::Sm89);
    let kernel = generate(&op, &key, &Cuda);
    // ⚠️ EXHAUSTIVE, NOT A FALL-THROUGH. This was `if Bf16 { .. } else { __half }`,
    // so ANY dtype that is neither would have been silently checked against
    // `__half` — in a matrix test whose entire purpose is distinguishing dtypes.
    // The failure would have read "emitter did NOT take the bit-move route",
    // blaming the emitter for the test's own mapping.
    //
    // The epilogue match below already panics on an unmodelled variant. Same
    // file, same author, two different standards; Sourcery caught the weaker
    // one. A dtype outside the narrow set has no bit-move route at all, so it
    // is not a cell this table can hold.
    let ctype = match cell.dt {
        ElementKind::Bf16 => "__nv_bfloat16",
        ElementKind::F16 => "__half",
        other => panic!(
            "{}: dtype {other:?} has no narrow bit-move route — `narrow_bit_casts`              returns None for it, so this table cannot hold a cell for it",
            cell.name
        ),
    };
    assert!(
        kernel.source.contains(&format!("{ctype} e = in0[")),
        "{}: emitter did NOT take the bit-move route\n{}",
        cell.name,
        kernel.source
    );
    assert!(
        !kernel.source.contains("2float(in0["),
        "{}: a promoting load survived\n{}",
        cell.name,
        kernel.source
    );
    kernel
}

/// The row the NaN goes in, and the column within it. Named because the flat
/// index alone (`host[60]`) states neither, and the assertions read `out[0]`
/// for "the NaN row" and `out[K]` for "a finite row" — a relationship the
/// constant has to preserve and could not express.
const NAN_ROW: usize = 0;
const NAN_COL: usize = 60;
/// The column carrying the second finite value in every row.
const ALT_COL: usize = 7;

/// Row `NAN_ROW` carries the NaN; every row carries both finite values.
fn rows(cell: &Cell) -> Vec<u16> {
    let k = K as usize;
    assert!(
        NAN_COL < k && ALT_COL < k,
        "fixture columns must lie inside a row"
    );
    let mut host = vec![cell.finite.0; (ROWS * K) as usize];
    for r in 0..ROWS as usize {
        host[r * k + ALT_COL] = cell.finite.1;
    }
    host[NAN_ROW * k + NAN_COL] = cell.nan_in;
    host
}

fn run(ctx: &Context, stream: &Stream, kernel: &GeneratedKernel, host: &[u16]) -> Vec<u16> {
    let compiler = NvrtcCompiler::new(ArchSku::Sm89);
    let ptx = compiler
        .compile(&kernel.source, &kernel.name, 30_000)
        .unwrap_or_else(|e| panic!("nvrtc REJECTED {}: {e}\n{}", kernel.name, kernel.source));
    let ptx = String::from_utf8(ptx).expect("ptx is text");
    let module = Module::load_ptx(ctx, &ptx).expect("module load");
    let f = module.get_function(&kernel.name).expect("get_function");
    let d_in = DeviceBuffer::from_slice(ctx, host).expect("d_in");
    let d_out = DeviceBuffer::<u16>::new(ctx, host.len()).expect("d_out");
    // SAFETY: the generated rowreduce signature is (in0, out, n_out, k); both
    // buffers outlive the launch, and u16 matches the f16/bf16 storage width.
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

fn check(cell: &Cell, out: &[u16]) {
    eprintln!(
        "  {:<14} NaN row: want 0x{:04X} got 0x{:04X}   finite row: want 0x{:04X} got 0x{:04X}",
        cell.name, cell.want_nan_row, out[0], cell.want_finite_row, out[K as usize]
    );
    assert_eq!(
        out[0], cell.want_nan_row,
        "{}: the NaN payload did not survive fold+epilogue (input 0x{:04X})",
        cell.name, cell.nan_in
    );
    // CONTROL: a NaN-free row. Without it, a kernel that only ever applied the
    // epilogue to its input would pass the assertion above while reducing
    // nothing at all.
    assert_eq!(
        out[K as usize], cell.want_finite_row,
        "{}: a finite row did not reduce correctly — the fold is not running",
        cell.name
    );
    // CONTROL: full-width output, so a kernel writing only element 0 fails.
    let stragglers = (0..K as usize)
        .filter(|&j| out[j] != cell.want_nan_row)
        .count();
    assert_eq!(
        stragglers, 0,
        "{}: {stragglers} of {K} elements in row 0 disagree with the reduced value",
        cell.name
    );
}

/// ⚠️ A CONTROL ON THE CONTROLS. The finite-row assertion exists to catch a
/// kernel that skipped the fold and applied the epilogue elementwise. It can
/// only do that if `epi(finite.0)` DIFFERS from `want_finite_row` — otherwise
/// the cheat produces the expected value and the control is inert.
///
/// One of the four cells shipped that way on the first draft (`bf16 Max Abs`,
/// where `Abs(-1.0) == Abs(max{-1.0,-3.0})`), and it passed. A control that
/// cannot fail is indistinguishable from one that holds, so this asserts the
/// discrimination instead of leaving it to whoever picks the next constants.
///
/// This is a driver-free property of the TABLE, so it runs without a device.
#[test]
fn controls_discriminate_a_no_fold_kernel() {
    for c in CELLS {
        // ⚠️ THIS CHEAT MODEL COVERS UNARY EPILOGUES ONLY. The bit-move route
        // now also admits `Select` (see `rowreduce_select_bitmove.rs`), which
        // this table cannot express — `Cell::epi` is a `UnaryOp`. The `other =>`
        // arm makes that a panic rather than a silent gap, which is the point:
        // adding a Select cell must FAIL here until someone models the cheat
        // for it, rather than quietly leaving the new cell uncontrolled.
        //
        // The reason lives in the code rather than in a tracking issue because
        // the issue that carried it was destroyed the moment the work landed —
        // and destroyed early, by a sentence merely PREDICTING its own closure.
        // A closing keyword has no tense.
        let cheat = match c.epi {
            UnaryOp::Neg => c.finite.0 ^ 0x8000,
            UnaryOp::Abs => c.finite.0 & 0x7FFF,
            other => panic!("{}: unmodelled epilogue {other:?}", c.name),
        };
        assert_ne!(
            cheat, c.want_finite_row,
            "{}: the finite-row control is VACUOUS. A kernel that skipped the fold and applied {:?} to element 0 would produce 0x{cheat:04X}, which is the expected value. Choose finite values whose winner is not element 0.",
            c.name, c.epi
        );
    }
}

#[test]
#[ignore = "requires a CUDA device + nvrtc"]
fn bitmove_rowreduce_matrix_preserves_payloads() {
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
    for cell in CELLS {
        check(cell, &run(&ctx, &stream, &emit(cell), &rows(cell)));
    }
}
