//! #128: the type-0/1 MMVQ `-sys` launchers decline `ncols % 64 != 0`
//! themselves, so a caller who uses `baracuda-kernels-sys`'s raw
//! `extern "C"` symbols directly (bypassing `GgufMmvqPlan`, which already
//! declines this per #127) can no longer trigger the cross-row over-read.
//!
//! # The bug this closes, and why it needed a NUMERIC check, not just memcheck
//!
//! The shared type-0/1 kernel (`dequantize_mul_mat_vec` in
//! `baracuda_gguf.cuh`) always reads a full 64-column stride per iteration,
//! regardless of the caller's declared `ncols`. For `ncols=32` (one Q8_0
//! block per row), row `r`'s single iteration reads columns 0..63 — but
//! only columns 0..31 belong to row `r`; columns 32..63 land in row `r+1`'s
//! block (blocks are stored consecutively, `blocks_per_row=1`).
//!
//! For every row except the last, that second half is a real, *allocated*
//! read (row `r+1`'s actual block) — silently WRONG, but invisible to
//! compute-sanitizer memcheck, which only flags reads outside the
//! allocation. Only the LAST row's overrun goes past the whole allocation,
//! which is what the `#127`/`#128` memcheck evidence (32 invalid reads at
//! `nrows=2, ncols=32`) actually measured — the other half of the bug.
//!
//! This doc records the one-time manual measurement that proves the
//! wrong-result half existed (not just plausible) BEFORE this file's actual
//! test, `q8_0_ncols32_via_raw_ffi_declines`, which asserts the fix. That
//! pre-fix run is not itself a committed test — the whole point is it fails
//! against `main` — so this doc is the evidence trail instead.
//!
//! ## Pre-fix measurement (manual, one-time; not a committed test)
//!
//! Q8_0, `nrows=2, ncols=32` (buggy width), row 0's block real values
//! `d=1.0, qs[i]=1` for all `i`, row 1's block `d=1.0, qs[i]=1` for all `i`,
//! activation `y[0..32)=1.0` (the real 32 columns) and, so the read the bug
//! actually performs lands somewhere OBSERVABLE rather than in
//! caller-unallocated memory, `y[32..64)=10.0` (deliberately allocated by
//! this test, unlike a minimal real caller who wouldn't allocate them —
//! that's what makes the bug's OTHER half memcheck-invisible: the read is
//! in-bounds on whatever happens to follow the real 32-element buffer).
//!
//! Traced by hand from the kernel source (`dequantize_mul_mat_vec` +
//! `dequantize_q8_0`: `v = qs[iqs]*d`, `iter_stride=64`,
//! `vals_per_iter=2`, one iteration since `ncols=32 < 64`): for row 0,
//! `tid=0..15` reads block 0 (row 0's real block) at columns 0..31 against
//! `y[0..31]`; `tid=16..31` reads block 1 — which is `(row*ncols+col)/qk`
//! with `row=0, ncols=32, qk=32`, i.e. the NEXT global block — against
//! `y[32..63]`. Block 1 in this layout IS row 1's block.
//!
//! Predicted row-0 output = `Σ true (32 * 1.0*1*1.0 = 32.0)` +
//! `Σ contamination (32 * 1.0*1*10.0 = 320.0)` = **352.0**, against a
//! correct value of **32.0**. Measured on an RTX 4070 (`gpu-run`, this
//! crate's own on-device harness), decline check temporarily disabled to
//! reproduce the pre-fix launcher: **row 0 got exactly 352.0** — bit-exact
//! (all terms are small exact integers in f32, so no tolerance was needed).
//! Confirms the wrong-result half is real, not merely plausible from reading
//! the source. Row 1 (the LAST row) got 32.0, not a contaminated value: its
//! own second-half read (`ib=2`) goes past the whole 2-block weight
//! allocation, which is the memcheck-VISIBLE half of this same bug (already
//! caught by `#127`'s existing sanitizer evidence) rather than a second
//! silently-wrong number.
//!
//! AFTER the launcher patch (this file's actual test, decline check
//! re-enabled): the same call returns a nonzero status and never runs the
//! kernel — `dst` stays at its zero-initialized value, not 352.0 and not
//! (by chance) 32.0.

use core::ffi::c_void;

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::U8;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// One `block_q8_0`: `half d` (2 bytes) + `int8_t qs[32]`, all quants equal
/// to `q`, scale `d`. Matches `baracuda_kernels_sys`'s memory layout exactly
/// (no padding — `half` then 32 bytes).
fn q8_0_block_bytes(d: f32, q: i8) -> [u8; 34] {
    let mut out = [0u8; 34];
    out[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    out[2..34].copy_from_slice(&[q as u8; 32]);
    out
}

/// Calls `baracuda_kernels_mmvq_q8_0_run` directly — the raw `-sys` symbol,
/// bypassing `GgufMmvqPlan` entirely (the shape #128 is about: a caller who
/// never reaches the plan layer's own #127 decline).
fn call_raw_q8_0(
    ctx: &Context,
    stream: &Stream,
    ncols: i32,
    nrows: i32,
    weight_bytes: &[u8],
    activation: &[f32],
) -> (i32, Vec<f32>) {
    let host_weight: Vec<U8> = weight_bytes.iter().copied().map(U8).collect();
    let dev_weight = DeviceBuffer::from_slice(ctx, &host_weight).expect("up weight");
    let dev_activation = DeviceBuffer::from_slice(ctx, activation).expect("up act");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(ctx, nrows as usize).expect("alloc out");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_mmvq_q8_0_run(
            ncols,
            nrows,
            dev_weight.as_slice().as_raw().0 as *const c_void,
            dev_activation.as_slice().as_raw().0 as *const c_void,
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; nrows as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    (status, got)
}

/// **The fix.** `ncols=32` for Q8_0 (one block per row — a real GGUF width;
/// nothing about it is malformed except that the shared kernel's fixed
/// 64-column stride can't honor it) must now decline via the raw FFI
/// symbol, not just via `GgufMmvqPlan` (`#127`, already covered by
/// `tests/mmvq_ncols64_decline.rs`). `dst` must stay untouched (its
/// zero-init value) — not silently populated with the pre-fix contaminated
/// value (352.0, see the module doc) and not, by chance, the correct value
/// (32.0) either: a decline, not a lucky pass.
#[test]
#[ignore]
fn q8_0_ncols32_via_raw_ffi_declines() {
    let (ctx, stream) = setup();
    let nrows = 2;
    let ncols = 32;

    let mut weight = Vec::with_capacity(2 * 34);
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 1)); // row 0
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 1)); // row 1

    let mut activation = vec![1.0f32; 32]; // the real 32 columns
    activation.extend(std::iter::repeat_n(10.0f32, 32)); // deliberately allocated, see module doc

    let (status, got) = call_raw_q8_0(&ctx, &stream, ncols, nrows, &weight, &activation);

    assert_ne!(
        status, 0,
        "ncols=32 (not a multiple of 64) must decline, got status 0"
    );
    assert_eq!(
        got,
        vec![0.0, 0.0],
        "a declined launch must not have run the kernel at all — dst should still be zero-init, \
         got {got:?} (352.0 would mean the pre-fix contamination bug is back; 32.0 would mean it \
         silently computed anyway instead of declining)"
    );
}

/// Positive control: `ncols=64` (two Q8_0 blocks per row, a multiple of the
/// kernel's stride) must still be ACCEPTED via the raw FFI symbol and
/// produce the correct answer — the launcher must decline the bad width,
/// not everything.
#[test]
#[ignore]
fn q8_0_ncols64_via_raw_ffi_computes_correctly() {
    let (ctx, stream) = setup();
    let nrows = 2;
    let ncols = 64;

    // Row 0: two copies of (d=1.0, q=1). Row 1: two copies of (d=1.0, q=2).
    let mut weight = Vec::with_capacity(4 * 34);
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 1));
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 1));
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 2));
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 2));

    let activation = vec![1.0f32; 64];

    let (status, got) = call_raw_q8_0(&ctx, &stream, ncols, nrows, &weight, &activation);

    assert_eq!(
        status, 0,
        "ncols=64 is a valid width and must be accepted, got status {status}"
    );
    let expected = [64.0f32, 128.0f32]; // 64 cols * 1.0 * {1, 2}
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "row {i}: got {g}, expected {e} (all-integer inputs, should be exact)"
        );
    }
}
