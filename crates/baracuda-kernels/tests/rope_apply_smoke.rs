//! Phase 36 (Fuel ask Gap 2) — direct-FFI smoke for RoPE-apply with
//! caller-supplied cos/sin tables.
//!
//! Equivalence-check: build cos/sin tables encoding the default
//! `θ = pos · base^(-2i/D)` schedule, then verify the apply path
//! produces the same output as `baracuda_kernels_rope_<dt>_run`.
//!
//! `#[ignore]` by default; needs a real CUDA device.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use half::{bf16, f16};

const ROPE_DEFAULT_BASE: f32 = 10000.0;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Build per-batch cos/sin tables for the default RoPE schedule:
///   `cos[s, pair] = cos(s · base^(-2·pair / d))`
/// Layout: `[seq, d/2]` — shared across all `bh` rows (`stride_b = 0`).
fn default_cs_tables(seq: usize, d: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let half_d = d / 2;
    let mut cos_t = vec![0f32; seq * half_d];
    let mut sin_t = vec![0f32; seq * half_d];
    let inv_d = 1.0f32 / (d as f32);
    for s in 0..seq {
        for pair in 0..half_d {
            let exponent = -((2 * pair) as f32) * inv_d;
            let freq = base.powf(exponent);
            let theta = (s as f32) * freq;
            cos_t[s * half_d + pair] = theta.cos();
            sin_t[s * half_d + pair] = theta.sin();
        }
    }
    (cos_t, sin_t)
}

#[test]
#[ignore]
fn rope_apply_f32_matches_default_rope() {
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 4i32;
    let seq = 8i32;
    let head_dim = 16i32;
    let bh = batch * heads;
    let td = seq * head_dim;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 - 0.7).sin() * 1.2)
        .collect();
    let (cos_t, sin_t) = default_cs_tables(seq as usize, head_dim as usize, ROPE_DEFAULT_BASE);
    assert_eq!(cos_t.len(), (seq * head_dim / 2) as usize);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y_apply: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc y apply");
    let mut dev_y_ref: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y ref");

    // Reference: derive-from-base RoPE FW.
    let status_ref = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_f32_run(
            batch,
            heads,
            seq,
            head_dim,
            ROPE_DEFAULT_BASE,
            1, // pos_default_flag
            dev_x.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_ref, 0, "rope_f32 returned status {status_ref}");

    // New: apply with precomputed cos/sin.
    let status_apply = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_f32_run(
            bh,
            td,
            head_dim,
            0, // stride_b = 0 — shared cos/sin
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_y_apply.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(
        status_apply, 0,
        "rope_apply_f32 returned status {status_apply}"
    );

    stream.synchronize().expect("sync");

    let mut got_apply = vec![0f32; numel];
    let mut got_ref = vec![0f32; numel];
    dev_y_apply.copy_to_host(&mut got_apply).expect("dl apply");
    dev_y_ref.copy_to_host(&mut got_ref).expect("dl ref");

    for (i, (a, r)) in got_apply.iter().zip(got_ref.iter()).enumerate() {
        // Same kernel arithmetic, same trig source — should be exact at
        // f32 (rope_f32_run uses __cosf/__sinf which is bit-identical
        // to the table values).
        let diff = (a - r).abs();
        let tol = 1e-4 * r.abs().max(1.0);
        assert!(
            diff <= tol,
            "rope_apply_f32 mismatch @ {i}: apply={a} ref={r} diff={diff}"
        );
    }
}

#[test]
#[ignore]
fn rope_apply_backward_f32_matches_default() {
    let (ctx, stream) = setup();
    let batch = 1i32;
    let heads = 2i32;
    let seq = 4i32;
    let head_dim = 8i32;
    let bh = batch * heads;
    let td = seq * head_dim;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.07 - 0.5).collect();
    let (cos_t, sin_t) = default_cs_tables(seq as usize, head_dim as usize, ROPE_DEFAULT_BASE);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_dx_apply: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc dx apply");
    let mut dev_dx_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc dx ref");

    let status_ref = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_backward_f32_run(
            batch,
            heads,
            seq,
            head_dim,
            ROPE_DEFAULT_BASE,
            1,
            dev_dy.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            dev_dx_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_ref, 0);

    let status_apply = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_backward_f32_run(
            bh,
            td,
            head_dim,
            0,
            dev_dy.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_dx_apply.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_apply, 0);
    stream.synchronize().expect("sync");

    let mut got_apply = vec![0f32; numel];
    let mut got_ref = vec![0f32; numel];
    dev_dx_apply.copy_to_host(&mut got_apply).expect("dl");
    dev_dx_ref.copy_to_host(&mut got_ref).expect("dl");

    for (i, (a, r)) in got_apply.iter().zip(got_ref.iter()).enumerate() {
        let diff = (a - r).abs();
        let tol = 1e-4 * r.abs().max(1.0);
        assert!(diff <= tol, "BW mismatch @ {i}: apply={a} ref={r}");
    }
}

#[test]
#[ignore]
fn rope_apply_f16_matches_default() {
    let (ctx, stream) = setup();
    let batch = 1i32;
    let heads = 2i32;
    let seq = 4i32;
    let head_dim = 8i32;
    let bh = batch * heads;
    let td = seq * head_dim;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_x: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32) * 0.05 - 0.3))
        .collect();
    let (cos_t, sin_t) = default_cs_tables(seq as usize, head_dim as usize, ROPE_DEFAULT_BASE);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y_apply: DeviceBuffer<f16> =
        DeviceBuffer::from_slice(&ctx, &vec![f16::ZERO; numel]).expect("alloc y apply");
    let mut dev_y_ref: DeviceBuffer<f16> =
        DeviceBuffer::from_slice(&ctx, &vec![f16::ZERO; numel]).expect("alloc y ref");

    let status_ref = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_f16_run(
            batch,
            heads,
            seq,
            head_dim,
            ROPE_DEFAULT_BASE,
            1,
            dev_x.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_ref, 0);

    let status_apply = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_f16_run(
            bh,
            td,
            head_dim,
            0,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_y_apply.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_apply, 0);
    stream.synchronize().expect("sync");

    let mut got_apply = vec![f16::ZERO; numel];
    let mut got_ref = vec![f16::ZERO; numel];
    dev_y_apply.copy_to_host(&mut got_apply).expect("dl");
    dev_y_ref.copy_to_host(&mut got_ref).expect("dl");

    for (i, (a, r)) in got_apply.iter().zip(got_ref.iter()).enumerate() {
        let af = a.to_f32();
        let rf = r.to_f32();
        let diff = (af - rf).abs();
        // f16 detours through f32 in both paths — but the table values
        // are bit-identical only modulo trig intrinsics. Use 1 ULP
        // worth of f16 tolerance.
        let tol = 1.0e-3 * rf.abs().max(1.0);
        assert!(
            diff <= tol,
            "rope_apply_f16 mismatch @ {i}: apply={af} ref={rf}"
        );
    }
}

#[test]
#[ignore]
fn rope_apply_bf16_matches_default() {
    let (ctx, stream) = setup();
    let batch = 1i32;
    let heads = 2i32;
    let seq = 4i32;
    let head_dim = 8i32;
    let bh = batch * heads;
    let td = seq * head_dim;
    let numel = (batch * heads * seq * head_dim) as usize;
    let host_x: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32) * 0.05 - 0.3))
        .collect();
    let (cos_t, sin_t) = default_cs_tables(seq as usize, head_dim as usize, ROPE_DEFAULT_BASE);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y_apply: DeviceBuffer<bf16> =
        DeviceBuffer::from_slice(&ctx, &vec![bf16::ZERO; numel]).expect("alloc y apply");
    let mut dev_y_ref: DeviceBuffer<bf16> =
        DeviceBuffer::from_slice(&ctx, &vec![bf16::ZERO; numel]).expect("alloc y ref");

    let status_ref = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_bf16_run(
            batch,
            heads,
            seq,
            head_dim,
            ROPE_DEFAULT_BASE,
            1,
            dev_x.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_ref, 0);

    let status_apply = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_bf16_run(
            bh,
            td,
            head_dim,
            0,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_y_apply.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_apply, 0);
    stream.synchronize().expect("sync");

    let mut got_apply = vec![bf16::ZERO; numel];
    let mut got_ref = vec![bf16::ZERO; numel];
    dev_y_apply.copy_to_host(&mut got_apply).expect("dl");
    dev_y_ref.copy_to_host(&mut got_ref).expect("dl");

    for (i, (a, r)) in got_apply.iter().zip(got_ref.iter()).enumerate() {
        let af = a.to_f32();
        let rf = r.to_f32();
        let diff = (af - rf).abs();
        // bf16 has only 7 mantissa bits — wider tolerance.
        let tol = 8.0e-3 * rf.abs().max(1.0);
        assert!(
            diff <= tol,
            "rope_apply_bf16 mismatch @ {i}: apply={af} ref={rf}"
        );
    }
}

#[test]
fn can_implement_rejects_negative_dims() {
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_rope_apply_f32_can_implement(-1, 8, 4, 0) };
    assert_ne!(s, 0, "should reject negative bh");
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_rope_apply_f32_can_implement(2, 8, 5, 0) };
    assert_ne!(s, 0, "should reject odd head_dim");
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_rope_apply_f32_can_implement(2, 7, 4, 0) };
    assert_ne!(s, 0, "should reject td % d != 0");
    let s = unsafe { baracuda_kernels_sys::baracuda_kernels_rope_apply_f32_can_implement(2, 16, 4, 0) };
    assert_eq!(s, 0, "valid: bh=2 td=16 d=4 stride_b=0");
}
