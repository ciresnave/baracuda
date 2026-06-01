//! Direct-FFI smoke tests for the **FA2 forward→saved-LSE→backward
//! saved-tensor contract** (Phase 63).
//!
//! This file is the load-bearing wiring proof that downstream autograd
//! frameworks (e.g. Fuel's `OpKind::FlashAttn` + `FlashAttnBackward{Q,K,V}`)
//! can carry baracuda's FA2 BW into a differentiable attention path.
//!
//! Tests fall into three groups:
//!
//! 1. **`lse_size` helper sanity** — confirms the byte-count formula
//!    matches what FA2 actually writes (host-only, no GPU needed).
//! 2. **FW → save LSE → BW roundtrip** — calls the raw
//!    `baracuda_kernels_fa2_sdpa_<dt>_run_v2` then
//!    `baracuda_kernels_fa2_sdpa_backward_<dt>_run` with the SAME LSE
//!    buffer the FW produced. Verifies status, finite gradients, and
//!    non-zero dQ/dK/dV (the BW would have to be totally broken to
//!    leave all three at zero given non-trivial Q/K/V/dO).
//! 3. **BW feature surface** — separate tests for sliding window,
//!    softcap, and ALiBi on the BW path. Each exercises the BW FFI
//!    with that feature enabled and verifies the gradient is finite +
//!    non-zero. (Numerical accuracy vs a CPU reference is a Tier-2
//!    follow-up — this layer of test just proves the feature surface
//!    is end-to-end callable on the BW path, which `fa2_backward_smoke.rs`
//!    leaves untested.)
//!
//! Marked `#[ignore]` per project convention; run with
//! `cargo test -p baracuda-kernels --features fa2,sm89 \
//!    --test fa2_fw_bw_saved_tensor_smoke -- --include-ignored`.

#![cfg(feature = "fa2")]

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn default_scale(d: i32) -> f32 {
    1.0 / (d as f32).sqrt()
}

fn gen_f16(n: usize, phase: f32) -> Vec<f16> {
    (0..n)
        .map(|i| f16::from_f32(((i as f32) * 0.013 + phase).sin() * 0.25))
        .collect()
}

fn gen_bf16(n: usize, phase: f32) -> Vec<bf16> {
    (0..n)
        .map(|i| bf16::from_f32(((i as f32) * 0.013 + phase).sin() * 0.25))
        .collect()
}

// =========================================================================
// Group 1: lse_size helper sanity (no GPU needed).
// =========================================================================

#[test]
fn lse_size_matches_dense_formula() {
    let sz = unsafe { baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_lse_size(2, 4, 128) };
    assert_eq!(sz, 2usize * 4 * 128, "lse_size = batch * num_heads * seq_q in f32 elements");
}

#[test]
fn lse_size_returns_zero_on_nonpositive() {
    for (b, h, q) in [(0, 4, 128), (2, 0, 128), (2, 4, 0), (-1, 4, 128)] {
        let sz = unsafe { baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_lse_size(b, h, q) };
        assert_eq!(sz, 0, "non-positive input ({b}, {h}, {q}) returns 0");
    }
}

#[test]
fn lse_size_large_values_dont_overflow() {
    // Choose dimensions just below usize::MAX / 4 (so byte count fits too).
    // 32-bit i32 inputs cap each dim at ~2.1G; we test a more realistic
    // long-context shape: batch=1, h=64, q=1M → 64M f32 elements = 256MB.
    let sz = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_lse_size(1, 64, 1_048_576)
    };
    assert_eq!(sz, 64 * 1_048_576);
}

// =========================================================================
// Group 2: FW → save LSE → BW roundtrip (f16, bf16).
//
// This is the load-bearing test that proves the saved-tensor contract
// works end-to-end. Same pattern Fuel will use in their autograd:
//   1. Allocate LSE via baracuda_kernels_fa2_sdpa_lse_size
//   2. Call FW with the LSE buffer
//   3. Save the same buffer alongside Q/K/V/O
//   4. Call BW with the same LSE buffer
//   5. Verify dQ/dK/dV are finite and non-zero
// =========================================================================

fn run_fw_bw_roundtrip_f16(dk: i32, is_causal: bool) {
    let (ctx, stream) = setup();
    let b: i32 = 1;
    let h: i32 = 4;
    let q: i32 = 128;
    let k: i32 = 128;
    let dv: i32 = dk;
    let scale = default_scale(dk);

    let n_qkv = (b * h * q * dk) as usize;
    let n_kv = (b * h * k * dk) as usize;
    let n_y = (b * h * q * dv) as usize;
    let n_lse_f32 = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_lse_size(b, h, q)
    };

    // Reasonable input fixtures — sinusoidal so the attention pattern
    // is non-trivial and dQ/dK/dV will be non-zero.
    let q_host = gen_f16(n_qkv, 0.1);
    let k_host = gen_f16(n_kv, 0.2);
    let v_host = gen_f16(n_kv, 0.3);
    let dy_host = gen_f16(n_y, 0.4);

    let dev_q = DeviceBuffer::from_slice(&ctx, &q_host).expect("upload q");
    let dev_k = DeviceBuffer::from_slice(&ctx, &k_host).expect("upload k");
    let dev_v = DeviceBuffer::from_slice(&ctx, &v_host).expect("upload v");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).expect("upload dy");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y");

    // === Step 1: pre-allocate LSE buffer via the size helper. ===
    let mut dev_lse: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_lse_f32).expect("alloc lse");

    // === Step 2: forward pass writes y AND lse. ===
    let fw_status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_f16_run(
            b, h, h, q, k, dk,
            scale,
            if is_causal { 1 } else { 0 },
            dev_q.as_slice().as_raw().0 as *const c_void,
            dev_k.as_slice().as_raw().0 as *const c_void,
            dev_v.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            dev_lse.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(fw_status, 0, "FW status");
    stream.synchronize().expect("sync fw");

    // Sanity: LSE should be finite and non-zero (softmax has computed
    // something meaningful given sinusoidal inputs).
    let mut h_lse = vec![0_f32; n_lse_f32];
    dev_lse.copy_to_host(&mut h_lse).expect("download lse");
    let nonzero_lse = h_lse.iter().filter(|&&x| x.is_finite() && x.abs() > 1e-6).count();
    assert!(nonzero_lse > n_lse_f32 / 2,
        "expected >50% of LSE cells non-zero finite, got {}/{}", nonzero_lse, n_lse_f32);

    // === Step 3: BW workspace allocation. ===
    let bw_ws_bytes = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_backward_workspace_size(b, h, q, dk)
    };
    assert!(bw_ws_bytes > 0, "BW workspace size must be non-zero");
    let mut dev_bw_ws: DeviceBuffer<u8> =
        DeviceBuffer::from_slice(&ctx, &vec![0u8; bw_ws_bytes]).expect("alloc bw ws");

    let mut dev_dq: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("alloc dq");
    let mut dev_dk: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_kv).expect("alloc dk");
    let mut dev_dv: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_kv).expect("alloc dv");

    // === Step 4: backward pass reads the SAME LSE we just wrote. ===
    let bw_status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_backward_f16_run(
            b, h, h, q, k, dk,
            scale,
            if is_causal { 1 } else { 0 },
            core::ptr::null(),     // alibi_slopes
            0,                     // alibi_batch_stride
            -1, -1,                // window_size_left/right (no window)
            0.0,                   // softcap
            dev_q.as_slice().as_raw().0 as *const c_void,
            dev_k.as_slice().as_raw().0 as *const c_void,
            dev_v.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice().as_raw().0 as *const c_void,
            dev_dy.as_slice().as_raw().0 as *const c_void,
            dev_lse.as_slice().as_raw().0 as *const c_void,
            dev_dq.as_slice_mut().as_raw().0 as *mut c_void,
            dev_dk.as_slice_mut().as_raw().0 as *mut c_void,
            dev_dv.as_slice_mut().as_raw().0 as *mut c_void,
            dev_bw_ws.as_slice_mut().as_raw().0 as *mut c_void,
            bw_ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(bw_status, 0, "BW status");
    stream.synchronize().expect("sync bw");

    // === Step 5: verify gradients are finite + non-zero. ===
    let mut h_dq = vec![f16::ZERO; n_qkv];
    let mut h_dk = vec![f16::ZERO; n_kv];
    let mut h_dv = vec![f16::ZERO; n_kv];
    dev_dq.copy_to_host(&mut h_dq).expect("download dq");
    dev_dk.copy_to_host(&mut h_dk).expect("download dk");
    dev_dv.copy_to_host(&mut h_dv).expect("download dv");

    let nz_dq = h_dq.iter().filter(|x| x.to_f32().abs() > 1e-4).count();
    let nz_dk = h_dk.iter().filter(|x| x.to_f32().abs() > 1e-4).count();
    let nz_dv = h_dv.iter().filter(|x| x.to_f32().abs() > 1e-4).count();
    assert!(nz_dq > n_qkv / 10, "dQ mostly zero ({}/{})", nz_dq, n_qkv);
    assert!(nz_dk > n_kv / 10, "dK mostly zero ({}/{})", nz_dk, n_kv);
    assert!(nz_dv > n_kv / 10, "dV mostly zero ({}/{})", nz_dv, n_kv);

    for (i, x) in h_dq.iter().enumerate() {
        let v = x.to_f32();
        assert!(v.is_finite(), "dQ[{i}] = {v} not finite");
    }
    for (i, x) in h_dk.iter().enumerate() {
        let v = x.to_f32();
        assert!(v.is_finite(), "dK[{i}] = {v} not finite");
    }
    for (i, x) in h_dv.iter().enumerate() {
        let v = x.to_f32();
        assert!(v.is_finite(), "dV[{i}] = {v} not finite");
    }
}

fn run_fw_bw_roundtrip_bf16(dk: i32, is_causal: bool) {
    let (ctx, stream) = setup();
    let b: i32 = 1;
    let h: i32 = 4;
    let q: i32 = 128;
    let k: i32 = 128;
    let dv: i32 = dk;
    let scale = default_scale(dk);

    let n_qkv = (b * h * q * dk) as usize;
    let n_kv = (b * h * k * dk) as usize;
    let n_y = (b * h * q * dv) as usize;
    let n_lse_f32 = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_lse_size(b, h, q)
    };

    let q_host = gen_bf16(n_qkv, 0.1);
    let k_host = gen_bf16(n_kv, 0.2);
    let v_host = gen_bf16(n_kv, 0.3);
    let dy_host = gen_bf16(n_y, 0.4);

    let dev_q = DeviceBuffer::from_slice(&ctx, &q_host).expect("upload q");
    let dev_k = DeviceBuffer::from_slice(&ctx, &k_host).expect("upload k");
    let dev_v = DeviceBuffer::from_slice(&ctx, &v_host).expect("upload v");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).expect("upload dy");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y");
    let mut dev_lse: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_lse_f32).expect("alloc lse");

    let fw_status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_bf16_run(
            b, h, h, q, k, dk,
            scale,
            if is_causal { 1 } else { 0 },
            dev_q.as_slice().as_raw().0 as *const c_void,
            dev_k.as_slice().as_raw().0 as *const c_void,
            dev_v.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            dev_lse.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(fw_status, 0);
    stream.synchronize().expect("sync fw");

    let bw_ws_bytes = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_backward_workspace_size(b, h, q, dk)
    };
    let mut dev_bw_ws: DeviceBuffer<u8> =
        DeviceBuffer::from_slice(&ctx, &vec![0u8; bw_ws_bytes]).expect("alloc bw ws");
    let mut dev_dq: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("alloc dq");
    let mut dev_dk: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_kv).expect("alloc dk");
    let mut dev_dv: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, n_kv).expect("alloc dv");

    let bw_status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_backward_bf16_run(
            b, h, h, q, k, dk,
            scale,
            if is_causal { 1 } else { 0 },
            core::ptr::null(), 0,
            -1, -1, 0.0,
            dev_q.as_slice().as_raw().0 as *const c_void,
            dev_k.as_slice().as_raw().0 as *const c_void,
            dev_v.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice().as_raw().0 as *const c_void,
            dev_dy.as_slice().as_raw().0 as *const c_void,
            dev_lse.as_slice().as_raw().0 as *const c_void,
            dev_dq.as_slice_mut().as_raw().0 as *mut c_void,
            dev_dk.as_slice_mut().as_raw().0 as *mut c_void,
            dev_dv.as_slice_mut().as_raw().0 as *mut c_void,
            dev_bw_ws.as_slice_mut().as_raw().0 as *mut c_void,
            bw_ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(bw_status, 0);
    stream.synchronize().expect("sync bw");

    let mut h_dq = vec![bf16::ZERO; n_qkv];
    let mut h_dk = vec![bf16::ZERO; n_kv];
    let mut h_dv = vec![bf16::ZERO; n_kv];
    dev_dq.copy_to_host(&mut h_dq).expect("download dq");
    dev_dk.copy_to_host(&mut h_dk).expect("download dk");
    dev_dv.copy_to_host(&mut h_dv).expect("download dv");

    let nz_dq = h_dq.iter().filter(|x| x.to_f32().abs() > 1e-3).count();
    let nz_dk = h_dk.iter().filter(|x| x.to_f32().abs() > 1e-3).count();
    let nz_dv = h_dv.iter().filter(|x| x.to_f32().abs() > 1e-3).count();
    assert!(nz_dq > n_qkv / 10, "bf16 dQ mostly zero ({}/{})", nz_dq, n_qkv);
    assert!(nz_dk > n_kv / 10, "bf16 dK mostly zero ({}/{})", nz_dk, n_kv);
    assert!(nz_dv > n_kv / 10, "bf16 dV mostly zero ({}/{})", nz_dv, n_kv);

    for x in &h_dq { assert!(x.to_f32().is_finite()); }
    for x in &h_dk { assert!(x.to_f32().is_finite()); }
    for x in &h_dv { assert!(x.to_f32().is_finite()); }
}

#[test] #[ignore] fn fw_bw_roundtrip_f16_d128_noncausal() { run_fw_bw_roundtrip_f16(128, false); }
#[test] #[ignore] fn fw_bw_roundtrip_f16_d128_causal()    { run_fw_bw_roundtrip_f16(128, true ); }
#[test] #[ignore] fn fw_bw_roundtrip_f16_d64_causal()     { run_fw_bw_roundtrip_f16(64,  true ); }
#[test] #[ignore] fn fw_bw_roundtrip_bf16_d128_noncausal(){ run_fw_bw_roundtrip_bf16(128, false); }
#[test] #[ignore] fn fw_bw_roundtrip_bf16_d128_causal()   { run_fw_bw_roundtrip_bf16(128, true ); }

// =========================================================================
// Group 3: BW feature surface — sliding window, softcap, ALiBi.
//
// fa2_backward_smoke.rs only tests the base path (alibi=None, no
// sliding window, no softcap). Phase 63 backfills the missing
// end-to-end BW exercise of these features.
// =========================================================================

/// Helper: run a FW with feature-enabled flags, capture LSE, then run
/// BW with the same flags. Asserts both calls succeed + grads are
/// finite. (Accuracy vs CPU oracle is deferred — this test layer just
/// proves the feature surface is callable on BW.)
fn run_fw_bw_with_features_f16(
    dk: i32,
    is_causal: bool,
    window_size_left: i32,
    window_size_right: i32,
    softcap: f32,
    alibi_slopes_host: Option<&[f32]>,
) {
    let (ctx, stream) = setup();
    let b: i32 = 1;
    let h: i32 = 4;
    let q: i32 = 128;
    let k: i32 = 128;
    let dv: i32 = dk;
    let scale = default_scale(dk);

    let n_qkv = (b * h * q * dk) as usize;
    let n_kv = (b * h * k * dk) as usize;
    let n_y = (b * h * q * dv) as usize;
    let n_lse_f32 = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_lse_size(b, h, q)
    };

    let q_host = gen_f16(n_qkv, 0.1);
    let k_host = gen_f16(n_kv, 0.2);
    let v_host = gen_f16(n_kv, 0.3);
    let dy_host = gen_f16(n_y, 0.4);

    let dev_q = DeviceBuffer::from_slice(&ctx, &q_host).expect("up q");
    let dev_k = DeviceBuffer::from_slice(&ctx, &k_host).expect("up k");
    let dev_v = DeviceBuffer::from_slice(&ctx, &v_host).expect("up v");
    let dev_dy = DeviceBuffer::from_slice(&ctx, &dy_host).expect("up dy");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_y).expect("alloc y");
    let mut dev_lse: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, n_lse_f32).expect("alloc lse");

    let dev_alibi: Option<DeviceBuffer<f32>> = alibi_slopes_host.map(|slopes| {
        DeviceBuffer::from_slice(&ctx, slopes).expect("up alibi slopes")
    });
    let (alibi_ptr, alibi_stride): (*const c_void, i32) = match &dev_alibi {
        Some(buf) => (buf.as_slice().as_raw().0 as *const c_void, 0), // per-head layout
        None => (core::ptr::null(), 0),
    };

    // FW with full v2 surface (carries ALiBi, sliding window, softcap).
    let fw_status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_f16_run_v2(
            b, h, h, q, k, dk,
            scale,
            if is_causal { 1 } else { 0 },
            alibi_ptr, alibi_stride,
            window_size_left, window_size_right, softcap,
            dev_q.as_slice().as_raw().0 as *const c_void,
            dev_k.as_slice().as_raw().0 as *const c_void,
            dev_v.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            dev_lse.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(fw_status, 0, "FW v2 status");
    stream.synchronize().expect("sync fw");

    let bw_ws_bytes = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_backward_workspace_size(b, h, q, dk)
    };
    let mut dev_bw_ws: DeviceBuffer<u8> =
        DeviceBuffer::from_slice(&ctx, &vec![0u8; bw_ws_bytes]).expect("alloc bw ws");
    let mut dev_dq: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_qkv).expect("alloc dq");
    let mut dev_dk: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_kv).expect("alloc dk");
    let mut dev_dv: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, n_kv).expect("alloc dv");

    let bw_status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_fa2_sdpa_backward_f16_run(
            b, h, h, q, k, dk,
            scale,
            if is_causal { 1 } else { 0 },
            alibi_ptr, alibi_stride,
            window_size_left, window_size_right, softcap,
            dev_q.as_slice().as_raw().0 as *const c_void,
            dev_k.as_slice().as_raw().0 as *const c_void,
            dev_v.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice().as_raw().0 as *const c_void,
            dev_dy.as_slice().as_raw().0 as *const c_void,
            dev_lse.as_slice().as_raw().0 as *const c_void,
            dev_dq.as_slice_mut().as_raw().0 as *mut c_void,
            dev_dk.as_slice_mut().as_raw().0 as *mut c_void,
            dev_dv.as_slice_mut().as_raw().0 as *mut c_void,
            dev_bw_ws.as_slice_mut().as_raw().0 as *mut c_void,
            bw_ws_bytes,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(bw_status, 0, "BW status");
    stream.synchronize().expect("sync bw");

    let mut h_dq = vec![f16::ZERO; n_qkv];
    let mut h_dk = vec![f16::ZERO; n_kv];
    let mut h_dv = vec![f16::ZERO; n_kv];
    dev_dq.copy_to_host(&mut h_dq).expect("dl dq");
    dev_dk.copy_to_host(&mut h_dk).expect("dl dk");
    dev_dv.copy_to_host(&mut h_dv).expect("dl dv");

    for x in &h_dq { assert!(x.to_f32().is_finite(), "dQ NaN/inf"); }
    for x in &h_dk { assert!(x.to_f32().is_finite(), "dK NaN/inf"); }
    for x in &h_dv { assert!(x.to_f32().is_finite(), "dV NaN/inf"); }

    let any_nz_dq = h_dq.iter().any(|x| x.to_f32().abs() > 1e-4);
    let any_nz_dk = h_dk.iter().any(|x| x.to_f32().abs() > 1e-4);
    let any_nz_dv = h_dv.iter().any(|x| x.to_f32().abs() > 1e-4);
    assert!(any_nz_dq, "dQ all zero — BW didn't produce gradient");
    assert!(any_nz_dk, "dK all zero");
    assert!(any_nz_dv, "dV all zero");
}

#[test]
#[ignore]
fn fw_bw_with_sliding_window_f16() {
    // Sliding window: each query attends to a 32-token left window (no right).
    // Causal-compatible: causal + left=32 == "last 32 past tokens, no future".
    run_fw_bw_with_features_f16(
        128,
        /*causal*/ true,
        /*window_left*/ 32,
        /*window_right*/ 0,  // causal already forces right=0 inside FA2
        /*softcap*/ 0.0,
        /*alibi*/ None,
    );
}

#[test]
#[ignore]
fn fw_bw_with_softcap_f16() {
    // Gemma-2 style tanh softcap on logits: tanh(s/cap) * cap.
    run_fw_bw_with_features_f16(
        128,
        /*causal*/ false,
        /*window_left*/ -1,
        /*window_right*/ -1,
        /*softcap*/ 30.0,  // typical Gemma-2 value
        /*alibi*/ None,
    );
}

#[test]
#[ignore]
fn fw_bw_with_alibi_f16() {
    // Per-head ALiBi slopes — 4 slopes for h=4.
    // Standard ALiBi geometric series: slope_i = 2^(-8*i/H) for i in 0..H.
    let slopes: Vec<f32> = (0..4)
        .map(|i| (2.0_f32).powf(-8.0 * (i as f32) / 4.0))
        .collect();
    run_fw_bw_with_features_f16(
        128,
        /*causal*/ false,
        /*window_left*/ -1,
        /*window_right*/ -1,
        /*softcap*/ 0.0,
        /*alibi*/ Some(&slopes),
    );
}

#[test]
#[ignore]
fn fw_bw_with_all_features_f16() {
    // Composition smoke: causal + sliding window + softcap + ALiBi all
    // enabled simultaneously. Validates the FW+BW feature plumbing
    // composes (no hidden mutual-exclusion bugs).
    let slopes: Vec<f32> = (0..4)
        .map(|i| (2.0_f32).powf(-8.0 * (i as f32) / 4.0))
        .collect();
    run_fw_bw_with_features_f16(
        128,
        /*causal*/ true,
        /*window_left*/ 64,
        /*window_right*/ 0,
        /*softcap*/ 30.0,
        /*alibi*/ Some(&slopes),
    );
}
