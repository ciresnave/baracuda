//! #141: the batched MMVQ `-sys` launchers (`baracuda_kernels_mmvq_<qtype>
//! _batched[_<dtype>]_run` in `crates/baracuda-kernels-sys/kernels/gguf/
//! mmvq_batched.cu`) decline `ncols % 64 != 0` for the 5 type-0/1 formats
//! themselves now, mirroring `tests/mmvq_launcher_width_decline.rs`'s
//! (#128/#139) fix for the single-row raw launchers.
//!
//! # Why this needed resolving before it could be fixed (not just wired)
//!
//! Before this fix, `baracuda_kernels_mmvq_batched.cu`'s `_can_implement`
//! sidecars encoded `n_cols >= 64 && n_cols % 32 == 0` for type-0/1
//! formats -- genuinely looser than the `_run` launchers' (newly added)
//! `n_cols % 64 == 0` check (e.g. `n_cols = 96` passes the old sidecar
//! rule but fails the new one). Reading `mmvq_batched_type01_tmpl` in
//! `kernels/include/baracuda_mmvq_batched.cuh` settles which rule is
//! correct: it uses the SAME `iter_stride = 2 * GGML_CUDA_DMMV_X = 64`
//! fixed-column-stride loop as the single-row `dequantize_mul_mat_vec`
//! template #139 fixed, so it has the identical vulnerability and needs
//! the identical `% 64 == 0` rule -- not the sidecar's looser one. This
//! is independently confirmed by the Rust plan layer: `GgufMmvqBatchedPlan
//! ::select` (`crates/baracuda-kernels/src/quantize/gguf/mmvq_batched.rs`)
//! already declines exactly this way via its own `DMMV_ITER_STRIDE_COLS`
//! + `uses_dmmv_stride` (documented there against #127) -- the raw-FFI
//! `_can_implement` sidecars were simply out of sync with the plan layer's
//! already-correct, already-shipped rule. This fix brings the `.cu` layer
//! in line with the `.rs` layer rather than inventing a new rule.
//!
//! Before this fix, the raw `_run` launchers had NO width check at all
//! beyond non-null/positive (`validate_problem` is deliberately
//! format-agnostic, shared with the pure-FP variant which has no block
//! structure) -- a caller bypassing `GgufMmvqBatchedPlan` could pass ANY
//! `ncols`, including one not even a multiple of the block size, and get
//! silent cross-row/cross-expert contamination exactly like #128's
//! single-row bug, just reachable through a different entry point.

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

/// One `block_q8_0`: `half d` (2 bytes) + `int8_t qs[32]`.
fn q8_0_block_bytes(d: f32, q: i8) -> [u8; 34] {
    let mut out = [0u8; 34];
    out[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    out[2..34].copy_from_slice(&[q as u8; 32]);
    out
}

/// Calls `baracuda_kernels_mmvq_q8_0_batched_run` directly -- the raw
/// `-sys` symbol, bypassing `GgufMmvqBatchedPlan` entirely (the shape
/// #141 is about, mirroring #128's single-row raw-FFI gap).
///
/// Fixed shape: 1 expert, 1 dispatch (token 0 -> expert 0, `top_k=1`
/// store path, no routing-weight multiply).
fn call_raw_q8_0_batched(
    ctx: &Context,
    stream: &Stream,
    ncols: i32,
    n_rows_per_expert: i32,
    weight_bytes: &[u8],
    activation: &[f32],
) -> (i32, Vec<f32>) {
    let n_experts = 1;
    let m_total = 1;

    let host_weight: Vec<U8> = weight_bytes.iter().copied().map(U8).collect();
    let dev_weight = DeviceBuffer::from_slice(ctx, &host_weight).expect("up weight");
    let dev_activation = DeviceBuffer::from_slice(ctx, activation).expect("up act");
    let dev_sorted_token_ids = DeviceBuffer::from_slice(ctx, &[0i32]).expect("up tids");
    let dev_expert_offsets = DeviceBuffer::from_slice(ctx, &[0i32, 1i32]).expect("up offsets");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(ctx, n_rows_per_expert as usize).expect("alloc out");
    let mut dev_workspace: DeviceBuffer<i32> = DeviceBuffer::zeros(ctx, m_total).expect("ws");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_mmvq_q8_0_batched_run(
            n_experts,
            n_rows_per_expert,
            ncols,
            dev_weight.as_slice().as_raw().0 as *const c_void,
            dev_activation.as_slice().as_raw().0 as *const c_void,
            dev_sorted_token_ids.as_slice().as_raw().0 as *const i32,
            dev_expert_offsets.as_slice().as_raw().0 as *const i32,
            core::ptr::null(),
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            1, // top_k
            dev_workspace.as_slice_mut().as_raw().0 as *mut c_void,
            (m_total * core::mem::size_of::<i32>()) as usize,
            stream.as_raw() as *mut c_void,
        )
    };
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; n_rows_per_expert as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    (status, got)
}

/// **The fix.** `ncols=32` (one Q8_0 block per row) must decline via the
/// raw batched FFI symbol -- `dst` must stay at its zero-init value, not
/// silently populated with a cross-row-contaminated value.
#[test]
#[ignore]
fn q8_0_batched_ncols32_via_raw_ffi_declines() {
    let (ctx, stream) = setup();
    let n_rows_per_expert = 2;
    let ncols = 32;

    let mut weight = Vec::with_capacity(2 * 34);
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 1)); // row 0
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 1)); // row 1

    let mut activation = vec![1.0f32; 32];
    activation.extend(std::iter::repeat_n(10.0f32, 32)); // deliberately allocated

    let (status, got) = call_raw_q8_0_batched(
        &ctx,
        &stream,
        ncols,
        n_rows_per_expert,
        &weight,
        &activation,
    );

    assert_ne!(
        status, 0,
        "ncols=32 (not a multiple of 64) must decline via the batched launcher, got status 0"
    );
    assert_eq!(
        got,
        vec![0.0, 0.0],
        "a declined launch must not have run the kernel at all -- dst should still be zero-init, \
         got {got:?}"
    );
}

/// Positive control: `ncols=64` must still be ACCEPTED and compute the
/// correct answer via the raw batched FFI symbol.
#[test]
#[ignore]
fn q8_0_batched_ncols64_via_raw_ffi_computes_correctly() {
    let (ctx, stream) = setup();
    let n_rows_per_expert = 2;
    let ncols = 64;

    let mut weight = Vec::with_capacity(4 * 34);
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 1));
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 1));
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 2));
    weight.extend_from_slice(&q8_0_block_bytes(1.0, 2));

    let activation = vec![1.0f32; 64];

    let (status, got) = call_raw_q8_0_batched(
        &ctx,
        &stream,
        ncols,
        n_rows_per_expert,
        &weight,
        &activation,
    );

    assert_eq!(
        status, 0,
        "ncols=64 is a valid width and must be accepted, got status {status}"
    );
    let expected = [64.0f32, 128.0f32];
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!((g - e).abs() < 1e-3, "row {i}: got {g}, expected {e}");
    }
}

/// #141's direct caller-path counterpart: a width that's valid for the
/// K-quant batched path (any `ncols` multiple of `QK_K=256`) must stay
/// accepted after this fix -- the new type-0/1-only `DMMV_ITER_STRIDE_COLS`
/// check must not leak into the K-quant fanout (which uses `QK_K` as its
/// `qk`, not 32, and has `needs_dmmv=false`).
#[test]
#[ignore]
fn q2_k_batched_ncols256_via_raw_ffi_computes() {
    let (ctx, stream) = setup();
    let n_experts = 1;
    let n_rows_per_expert = 1;
    let ncols = 256; // QK_K, one block per row
    let m_total = 1;

    // block_q2_K: scales[QK_K/16=16] + qs[QK_K/4=64] + half2 dm(4 bytes) = 84 bytes.
    // Use an all-zero block (d=0): deterministic zero output, just proving
    // "accepted and ran" rather than exercising the full k-quant dequant math
    // (that's covered elsewhere; this test is purely about the width gate).
    let weight = vec![0u8; 84];
    let activation = vec![0.0f32; ncols as usize];

    let host_weight: Vec<U8> = weight.iter().copied().map(U8).collect();
    let dev_weight = DeviceBuffer::from_slice(&ctx, &host_weight).expect("up weight");
    let dev_activation = DeviceBuffer::from_slice(&ctx, &activation).expect("up act");
    let dev_sorted_token_ids = DeviceBuffer::from_slice(&ctx, &[0i32]).expect("up tids");
    let dev_expert_offsets = DeviceBuffer::from_slice(&ctx, &[0i32, 1i32]).expect("up offsets");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, n_rows_per_expert as usize).expect("alloc out");
    let mut dev_workspace: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, m_total).expect("ws");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_mmvq_q2_K_batched_run(
            n_experts,
            n_rows_per_expert,
            ncols,
            dev_weight.as_slice().as_raw().0 as *const c_void,
            dev_activation.as_slice().as_raw().0 as *const c_void,
            dev_sorted_token_ids.as_slice().as_raw().0 as *const i32,
            dev_expert_offsets.as_slice().as_raw().0 as *const i32,
            core::ptr::null(),
            dev_out.as_slice_mut().as_raw().0 as *mut c_void,
            1,
            dev_workspace.as_slice_mut().as_raw().0 as *mut c_void,
            (m_total * core::mem::size_of::<i32>()) as usize,
            stream.as_raw() as *mut c_void,
        )
    };
    stream.synchronize().expect("sync");

    assert_eq!(
        status, 0,
        "ncols=256 (QK_K, a multiple of the k-quant block size) must be accepted for q2_K, \
         got status {status} -- the type-0/1-only width check must not reject k-quant formats"
    );
}
