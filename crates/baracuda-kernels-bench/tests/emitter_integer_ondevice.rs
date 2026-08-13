//! ON-DEVICE end-to-end correctness for EMITTER-GENERATED INTEGER kernels
//! (emit -> NVRTC -> launch -> bit-exact vs a host reference), on the RTX 4070.
//!
//! WHY this test exists — the gap it closes:
//! Every standing on-device test for the CUDA emitter is FLOAT (relu_add, N2
//! NaN-propagation, the precision-first reduction). Integer emission is covered
//! only by SOURCE-STRING goldens — the exact same "bytes are pinned, but nothing
//! proved the LAUNCHED kernel writes the right answer" shape that let the
//! alpha.78 all-zero ship. No emitter-generated *integer* kernel has ever been
//! run end-to-end on real hardware. This closes that.
//!
//! Scope (deliberately narrow — the recon that motivated this test showed the
//! integer lowerings are source-faithful and deterministic: signed `>>` ->
//! `shr.s32`, unsigned `>>` -> `shr.u32`, native `+` -> two's-complement `add`,
//! all fixed by the C->PTX->ISA contract, so ptxas has NO latitude to diverge.
//! The value here is NOT "does the GPU respect integer semantics" — it's "does
//! an EMITTED integer kernel compute correctly end-to-end", which was untested):
//!   1. signed i32 `>>` — arithmetic (sign-replicating) shift;
//!   2. unsigned u8 `>>` — LOGICAL (zero-fill) shift. This is the ONE assertion
//!      that has never run on hardware: u8-logical-shift is U8-only in the
//!      emitter and was previously asserted only in a golden's prose comment;
//!   3. u8 `+` — wrapping (mod-2^8) add.
//!
//! Non-vacuity: the inputs are chosen so a WRONG lowering changes the answer.
//! i32 lanes are NEGATIVE (arithmetic `>>` stays negative; a logical `>>` would
//! go large-positive). u8 shift lanes have the HIGH BIT SET (logical `>>` zero-
//! fills; an arithmetic `>>` would replicate the sign — a different byte). u8 add
//! lanes overflow (wrapping differs from saturating/promoting). Each case also
//! carries an explicit discriminating control asserting the wrong semantics is
//! ruled out, so a green run means the right lowering, not a coincidence.
//!
//! Run under the gpu-run lock:
//!   cargo gpu-test -p baracuda-kernels-bench --test emitter_integer_ondevice -- --nocapture

use baracuda_cuda_emit::{Cuda, NvrtcCompiler};
use baracuda_driver::{Context, DeviceBuffer, Module, Stream};
use baracuda_kernels_bench::setup_device;
use baracuda_kernels_types::{ArchSku, ElementKind, OpCategory, OperandDesc, structure_key};
use baracuda_types::DeviceRepr;
use unpopped::Compiler;
use unpopped::generate;
use unpopped::ir::{BinaryOp, OpDef, input};

/// Emit `op` for `dt`, compile it through the real production chain, launch it
/// on-device over the `(in0, in1)` lanes, and assert the result is bit-exact
/// with `expected`. Returns the device output for case-specific controls.
/// `init` is a never-wrote sentinel chosen to differ from every expected value.
#[allow(clippy::too_many_arguments)]
fn run_binary_int<T>(
    ctx: &Context,
    stream: &Stream,
    op: &OpDef,
    dt: ElementKind,
    align: u32,
    in0: &[T],
    in1: &[T],
    expected: &[T],
    init: T,
    label: &str,
) -> Vec<T>
where
    T: DeviceRepr + Copy + PartialEq + std::fmt::Debug,
{
    assert_eq!(in0.len(), in1.len());
    assert_eq!(in0.len(), expected.len());
    let numel = in0.len() as i64;

    // Binary-elementwise structure key. Integer cells always take the scalar
    // path (no int4 vectorization), so alignment does not change the kernel.
    let a = OperandDesc::new(1, &[numel], &[1], dt, align);
    let key = structure_key(OpCategory::BinaryElementwise, &[a, a, a], ArchSku::Sm89);
    let k = generate(op, &key, &Cuda);
    assert!(
        k.name.ends_with("_scalar"),
        "{label}: expected a scalar integer kernel, got {}",
        k.name
    );

    let ptx = NvrtcCompiler::new(ArchSku::Sm89)
        .compile(&k.source, &k.name, 30_000)
        .unwrap_or_else(|e| panic!("{label}: nvrtc({}) failed: {e}", k.name));
    let ptx = String::from_utf8(ptx).unwrap();
    let module = Module::load_ptx(ctx, &ptx).expect("load PTX");
    let f = module.get_function(&k.name).expect("get function");

    let d0 = DeviceBuffer::from_slice(ctx, in0).unwrap();
    let d1 = DeviceBuffer::from_slice(ctx, in1).unwrap();
    let d_out = DeviceBuffer::from_slice(ctx, &vec![init; in0.len()]).unwrap();

    let n = numel; // scalar ABI: (in0, in1, out, long long n)
    let block = 256u32;
    let grid = ((numel as u32) + block - 1) / block;
    // SAFETY: scalar ABI (in0, in1, out, long long n) — same marshalling as the
    // f32 relu/N2 harnesses, with integer buffers.
    unsafe {
        f.launch()
            .grid(grid)
            .block(block)
            .stream(stream)
            .arg(&d0)
            .arg(&d1)
            .arg(&d_out)
            .arg(&n)
            .launch()
            .unwrap_or_else(|e| panic!("{label}: launch: {e}"));
    }
    stream.synchronize().unwrap();
    let mut got = vec![init; in0.len()];
    d_out.copy_to_host(&mut got).unwrap();

    // Positive control: every lane written (none left at the sentinel).
    assert!(
        !got.iter().any(|&g| g == init),
        "{label}: a lane was left at the never-wrote sentinel {init:?} — kernel did not write it"
    );
    // Bit-exact vs the host reference (the discriminating inputs make a wrong
    // lowering — logical-for-arithmetic, saturate-for-wrap — a byte mismatch).
    for i in 0..in0.len() {
        assert_eq!(
            got[i], expected[i],
            "{label} lane {i}: op({:?}, {:?}) = {:?} (expected {:?})",
            in0[i], in1[i], got[i], expected[i]
        );
    }
    got
}

#[test]
#[ignore = "requires CUDA + nvrtc; on-device emitter-integer correctness on the 4070"]
fn emitter_integer_kernels_are_correct_on_device() {
    let (ctx, stream) = setup_device();

    // ---- 1. signed i32 `>>` : ARITHMETIC (sign-replicating) shift ------------
    // Negative lanes discriminate: arithmetic `>>` stays negative; a logical
    // `>>` (treating the value as u32) would go large-positive.
    let shr_i32 = OpDef::elementwise(
        "shr",
        2,
        &[ElementKind::I32],
        input(0).binary(BinaryOp::Shr, input(1)),
    );
    let a_i32: Vec<i32> = vec![-8, -256, -1, 1024, i32::MIN, 255, -12345, 7];
    let b_i32: Vec<i32> = vec![1, 4, 3, 2, 8, 1, 4, 1];
    let e_i32: Vec<i32> = a_i32.iter().zip(&b_i32).map(|(&x, &s)| x >> s).collect();
    let got_i32 = run_binary_int(
        &ctx,
        &stream,
        &shr_i32,
        ElementKind::I32,
        4,
        &a_i32,
        &b_i32,
        &e_i32,
        i32::MAX,
        "i32 arithmetic >>",
    );
    // Discriminating control: every negative-input lane stays NEGATIVE — a
    // logical shift on those lanes would be a large positive value.
    for i in 0..a_i32.len() {
        if a_i32[i] < 0 {
            assert!(
                got_i32[i] < 0,
                "i32 >> lane {i}: {} >> {} = {} — arithmetic shift must keep the sign; a positive \
                 result means the emitter/toolchain used a LOGICAL shift",
                a_i32[i],
                b_i32[i],
                got_i32[i]
            );
        }
    }

    // ---- 2. unsigned u8 `>>` : LOGICAL (zero-fill) shift ---------------------
    // High-bit lanes discriminate: logical `>>` zero-fills; an arithmetic `>>`
    // (as if signed) would replicate bit 7 — a different byte.
    let shr_u8 = OpDef::elementwise(
        "shr",
        2,
        &[ElementKind::U8],
        input(0).binary(BinaryOp::Shr, input(1)),
    );
    let a_u8: Vec<u8> = vec![0x80, 0xFF, 0xC0, 0x81, 0x40, 0xAA, 0x01, 0xFE];
    let b_u8: Vec<u8> = vec![1, 4, 2, 7, 1, 3, 1, 1];
    let e_u8: Vec<u8> = a_u8.iter().zip(&b_u8).map(|(&x, &s)| x >> s).collect();
    let got_u8 = run_binary_int(
        &ctx,
        &stream,
        &shr_u8,
        ElementKind::U8,
        1,
        &a_u8,
        &b_u8,
        &e_u8,
        0x37,
        "u8 logical >>",
    );
    // Discriminating control: on each high-bit lane the LOGICAL result must
    // differ from the ARITHMETIC result (sign-replicating on the signed byte).
    for i in 0..a_u8.len() {
        if a_u8[i] & 0x80 != 0 {
            let arithmetic = ((a_u8[i] as i8) >> b_u8[i]) as u8;
            if e_u8[i] != arithmetic {
                assert_ne!(
                    got_u8[i], arithmetic,
                    "u8 >> lane {i}: {:#04x} >> {} came back {:#04x} — that is the ARITHMETIC \
                     (sign-extending) result; unsigned shift MUST be logical (zero-fill)",
                    a_u8[i], b_u8[i], got_u8[i]
                );
            }
        }
    }

    // ---- 3. u8 `+` : WRAPPING (mod-2^8) add ----------------------------------
    // Overflowing lanes discriminate: wrapping differs from saturating (which
    // would clamp to 0xFF) and from promoting (which would not truncate).
    let add_u8 = OpDef::elementwise("add", 2, &[ElementKind::U8], input(0) + input(1));
    let a_add: Vec<u8> = vec![200, 255, 128, 1, 100, 250, 0, 127];
    let b_add: Vec<u8> = vec![100, 1, 200, 254, 200, 10, 0, 129];
    let e_add: Vec<u8> = a_add
        .iter()
        .zip(&b_add)
        .map(|(&x, &y)| x.wrapping_add(y))
        .collect();
    let got_add = run_binary_int(
        &ctx,
        &stream,
        &add_u8,
        ElementKind::U8,
        1,
        &a_add,
        &b_add,
        &e_add,
        0x37,
        "u8 wrapping +",
    );
    // Discriminating control: an overflowing lane must NOT saturate to 0xFF.
    for i in 0..a_add.len() {
        if (a_add[i] as u16) + (b_add[i] as u16) > 0xFF {
            assert_ne!(
                got_add[i], 0xFF,
                "u8 + lane {i}: {} + {} came back 0xFF — saturating, not wrapping (expected {})",
                a_add[i], b_add[i], e_add[i]
            );
        }
    }

    println!(
        "EMITTER-INTEGER end-to-end VERIFIED on the RTX 4070: i32 arithmetic >>, u8 LOGICAL >> \
         (first time on hardware), and u8 wrapping + all compute bit-exact through \
         emit->nvrtc->launch. {} lanes across 3 kernels.",
        a_i32.len() + a_u8.len() + a_add.len()
    );
}
