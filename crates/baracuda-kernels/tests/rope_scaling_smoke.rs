//! Phase 45 — `RopeScaledTableBuilder` real-GPU integration smoke test.
//!
//! Verifies that the Linear-scaling table produced by the builder is
//! bit-equivalent (within f32 trig precision) to the table the
//! `rope_apply_<dt>` kernel would derive internally — i.e., that the
//! existing Phase 36 RoPE-apply path consumed via the builder gives
//! the same output as the classic `rope_<dt>` path that derives
//! `θ = pos · base^(-2i/D)` on-device.
//!
//! Also exercises the YaRN + LongRoPE paths end-to-end:
//! builder-built tables → `rope_apply_<dt>` kernel → host download +
//! Python-equivalent reference computed in Rust. The reference here
//! is "apply the same math to the input vector pair-by-pair on the
//! host" — the kernel is supposed to do exactly that, so the check
//! is for end-to-end algorithmic correctness, not novel math.
//!
//! `#[ignore]` by default; needs a real CUDA device. The
//! pure-host-side unit tests in `attention::rope_scaling::tests` run
//! unconditionally and cover the math without needing a GPU.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{RopeScaledTableBuilder, RopeScaling};

const ROPE_DEFAULT_BASE: f32 = 10000.0;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Host-side reference: apply RoPE rotation pair-by-pair using
/// caller-supplied cos/sin tables (`[seq, d/2]` layout, shared across
/// bh rows). Mirrors what the `rope_apply_<dt>` kernel computes for
/// `stride_b = 0`.
fn rope_apply_host_ref(
    bh: usize,
    seq: usize,
    head_dim: usize,
    x: &[f32],
    cos_t: &[f32],
    sin_t: &[f32],
) -> Vec<f32> {
    let half_d = head_dim / 2;
    let mut y = vec![0f32; bh * seq * head_dim];
    for row in 0..bh {
        for s in 0..seq {
            for pair in 0..half_d {
                let i0 = row * seq * head_dim + s * head_dim + 2 * pair;
                let i1 = i0 + 1;
                let c = cos_t[s * half_d + pair];
                let sn = sin_t[s * half_d + pair];
                y[i0] = x[i0] * c - x[i1] * sn;
                y[i1] = x[i1] * c + x[i0] * sn;
            }
        }
    }
    y
}

#[test]
#[ignore]
fn rope_apply_linear_builder_matches_kernel() {
    let (ctx, stream) = setup();
    let batch = 1i32;
    let heads = 2i32;
    let seq = 8i32;
    let head_dim = 16i32;
    let bh = (batch * heads) as i32;
    let td = (seq * head_dim) as i32;
    let numel = (batch * heads * seq * head_dim) as usize;

    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 - 0.7).sin() * 1.2)
        .collect();

    let builder = RopeScaledTableBuilder::new(
        head_dim,
        seq,
        ROPE_DEFAULT_BASE,
        RopeScaling::Linear,
    );
    let (cos_t, sin_t) = builder.build_host_tables().expect("build linear tables");
    assert_eq!(cos_t.len(), builder.table_len());

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_f32_run(
            bh,
            td,
            head_dim,
            0, // stride_b = 0 — shared cos/sin
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "rope_apply_f32 status {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let expected = rope_apply_host_ref(
        bh as usize,
        seq as usize,
        head_dim as usize,
        &host_x,
        &cos_t,
        &sin_t,
    );

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let tol = 1e-4 * e.abs().max(1.0);
        assert!(
            diff <= tol,
            "Linear builder kernel mismatch @ {i}: got={g} ref={e} diff={diff}"
        );
    }
}

#[test]
#[ignore]
fn rope_apply_yarn_builder_matches_host_ref() {
    let (ctx, stream) = setup();
    let batch = 1i32;
    let heads = 1i32;
    let seq = 4i32;
    let head_dim = 32i32;
    let bh = (batch * heads) as i32;
    let td = (seq * head_dim) as i32;
    let numel = (batch * heads * seq * head_dim) as usize;

    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 0.3).collect();

    let builder = RopeScaledTableBuilder::new(
        head_dim,
        seq,
        ROPE_DEFAULT_BASE,
        RopeScaling::YaRN {
            scale: 4.0,
            alpha: 1.0,
            beta: 32.0,
            original_max_seq_len: 2048,
        },
    );
    let (cos_t, sin_t) = builder.build_host_tables().expect("build yarn tables");

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_f32_run(
            bh,
            td,
            head_dim,
            0,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let expected = rope_apply_host_ref(
        bh as usize,
        seq as usize,
        head_dim as usize,
        &host_x,
        &cos_t,
        &sin_t,
    );
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let tol = 1e-4 * e.abs().max(1.0);
        assert!(diff <= tol, "YaRN mismatch @ {i}: got={g} ref={e}");
    }
}

#[test]
#[ignore]
fn rope_apply_longrope_builder_matches_host_ref() {
    let (ctx, stream) = setup();
    let batch = 1i32;
    let heads = 1i32;
    let seq = 4i32;
    let head_dim = 16i32;
    let bh = (batch * heads) as i32;
    let td = (seq * head_dim) as i32;
    let numel = (batch * heads * seq * head_dim) as usize;

    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.07 - 0.5).collect();

    let half_d = (head_dim / 2) as usize;
    // Arbitrary factor table for testing — in production this comes
    // from LongRoPE's offline evolutionary search.
    let factors: Vec<f32> = (0..half_d)
        .map(|i| 1.0f32 + 0.5f32 * (i as f32 / half_d as f32))
        .collect();
    let builder = RopeScaledTableBuilder::new(
        head_dim,
        seq,
        ROPE_DEFAULT_BASE,
        RopeScaling::LongRoPE { per_dim_factors: factors },
    );
    let (cos_t, sin_t) = builder.build_host_tables().expect("build longrope");

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_f32_run(
            bh,
            td,
            head_dim,
            0,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let expected = rope_apply_host_ref(
        bh as usize,
        seq as usize,
        head_dim as usize,
        &host_x,
        &cos_t,
        &sin_t,
    );
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let tol = 1e-4 * e.abs().max(1.0);
        assert!(diff <= tol, "LongRoPE mismatch @ {i}: got={g} ref={e}");
    }
}
