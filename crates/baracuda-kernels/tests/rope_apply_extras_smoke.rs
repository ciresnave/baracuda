//! Phase 41 (Fuel Phase 6c.4 Gap 7 + 8) — direct-FFI smoke for the
//! RoPE-apply interleaved + THD-layout variants.
//!
//! - Interleaved: pair convention `(2k, 2k+1)`. The existing
//!   `rope_apply_<dt>_run` family already implements exactly this
//!   pairing — the interleaved symbols are re-exports under the
//!   Fuel-expected name, so equivalence with `rope_apply_<dt>_run`
//!   must be bit-identical (no special "degenerate fixture" needed).
//! - THD: operand layout `[T, H, D]`. We use `[T, H, D] == [seq, 1, D]`
//!   with `stride_b == D/2` so the per-t cs lookup reduces to a single
//!   linear walk, matching a CPU reference.
//!
//! `#[ignore]` by default — needs a real CUDA device.

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

/// Build per-seq cos/sin tables for the default RoPE schedule:
///   `cos[s, pair] = cos(s · base^(-2·pair / d))`
/// Layout: `[seq, d/2]` shared across all outer rows.
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

/// CPU reference for the interleaved RoPE-apply variant: pair
/// `(2k, 2k+1)`, cos/sin indexed by `pair = dim_idx >> 1`. Same
/// arithmetic as `rope_apply_fp_kernel`, so the reference exists
/// chiefly to give the smoke test a layer of math sanity beyond
/// "compare two FFI symbols that wrap the same kernel".
fn cpu_rope_apply_interleaved_f32(
    x: &[f32],
    cos_t: &[f32],
    sin_t: &[f32],
    bh: usize,
    td: usize,
    d: usize,
    stride_b: usize,
) -> Vec<f32> {
    let half_d = d / 2;
    let seq = td / d;
    let mut y = vec![0f32; bh * td];
    for bh_row in 0..bh {
        for s in 0..seq {
            for pair in 0..half_d {
                let cs_off = bh_row * stride_b + s * half_d + pair;
                let c = cos_t[cs_off];
                let si = sin_t[cs_off];
                let off_e = bh_row * td + s * d + 2 * pair;
                let off_o = off_e + 1;
                let x_e = x[off_e];
                let x_o = x[off_o];
                y[off_e] = x_e * c - x_o * si;
                y[off_o] = x_o * c + x_e * si;
            }
        }
    }
    y
}

/// CPU reference for the THD-layout RoPE-apply variant.
/// Operand layout `[T, H, D]`. cs is `cs[t * stride_b + pair]`.
fn cpu_rope_apply_thd_f32(
    x: &[f32],
    cos_t: &[f32],
    sin_t: &[f32],
    t_outer: usize,
    h_heads: usize,
    d: usize,
    stride_b: usize,
) -> Vec<f32> {
    let half_d = d / 2;
    let mut y = vec![0f32; t_outer * h_heads * d];
    for t_idx in 0..t_outer {
        for h in 0..h_heads {
            for pair in 0..half_d {
                let cs_off = t_idx * stride_b + pair;
                let c = cos_t[cs_off];
                let si = sin_t[cs_off];
                let off_e = t_idx * h_heads * d + h * d + 2 * pair;
                let off_o = off_e + 1;
                let x_e = x[off_e];
                let x_o = x[off_o];
                y[off_e] = x_e * c - x_o * si;
                y[off_o] = x_o * c + x_e * si;
            }
        }
    }
    y
}

// =============================================================================
// Interleaved variant — equivalence + CPU-ref tests
// =============================================================================

#[test]
#[ignore]
fn rope_apply_interleaved_f32_matches_canonical_apply() {
    // The interleaved variant is functionally equivalent to the existing
    // `rope_apply_f32_run` (both pair `(2k, 2k+1)`). The two symbols must
    // be bit-identical because they wrap the same kernel.
    let (ctx, stream) = setup();
    let batch = 2i32;
    let heads = 3i32;
    let seq = 5i32;
    let head_dim = 8i32;
    let bh = batch * heads;
    let td = seq * head_dim;
    let numel = (bh * td) as usize;

    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.13 - 0.7).sin() * 1.2)
        .collect();
    let (cos_t, sin_t) = default_cs_tables(seq as usize, head_dim as usize, ROPE_DEFAULT_BASE);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y_interleaved: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc y interleaved");
    let mut dev_y_canonical: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc y canonical");

    let status_canon = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_f32_run(
            bh,
            td,
            head_dim,
            0,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_y_canonical.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_canon, 0);

    let status_interleaved = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_interleaved_f32_run(
            bh,
            td,
            head_dim,
            0,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_y_interleaved.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_interleaved, 0);
    stream.synchronize().expect("sync");

    let mut got_inter = vec![0f32; numel];
    let mut got_canon = vec![0f32; numel];
    dev_y_interleaved.copy_to_host(&mut got_inter).expect("dl");
    dev_y_canonical.copy_to_host(&mut got_canon).expect("dl");

    // Both wrap the same kernel — must be bit-identical.
    for (i, (a, b)) in got_inter.iter().zip(got_canon.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "interleaved-vs-canonical bit mismatch @ {i}: inter={a} canon={b}"
        );
    }

    // Also cross-check against CPU reference.
    let ref_y = cpu_rope_apply_interleaved_f32(
        &host_x,
        &cos_t,
        &sin_t,
        bh as usize,
        td as usize,
        head_dim as usize,
        0,
    );
    for (i, (g, r)) in got_inter.iter().zip(ref_y.iter()).enumerate() {
        let diff = (g - r).abs();
        let tol = 1e-4 * r.abs().max(1.0);
        assert!(diff <= tol, "cpu-ref mismatch @ {i}: gpu={g} cpu={r}");
    }
}

#[test]
#[ignore]
fn rope_apply_interleaved_backward_f32_matches_canonical() {
    let (ctx, stream) = setup();
    let bh = 4i32;
    let seq = 4i32;
    let head_dim = 8i32;
    let td = seq * head_dim;
    let numel = (bh * td) as usize;
    let host_dy: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.07 - 0.5).collect();
    let (cos_t, sin_t) = default_cs_tables(seq as usize, head_dim as usize, ROPE_DEFAULT_BASE);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_dx_inter: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc");
    let mut dev_dx_canon: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let status_canon = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_backward_f32_run(
            bh,
            td,
            head_dim,
            0,
            dev_dy.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_dx_canon.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_canon, 0);

    let status_inter = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_interleaved_backward_f32_run(
            bh,
            td,
            head_dim,
            0,
            dev_dy.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_dx_inter.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_inter, 0);
    stream.synchronize().expect("sync");

    let mut got_inter = vec![0f32; numel];
    let mut got_canon = vec![0f32; numel];
    dev_dx_inter.copy_to_host(&mut got_inter).expect("dl");
    dev_dx_canon.copy_to_host(&mut got_canon).expect("dl");

    for (i, (a, b)) in got_inter.iter().zip(got_canon.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "interleaved BW vs canonical BW bit mismatch @ {i}"
        );
    }
}

#[test]
#[ignore]
fn rope_apply_interleaved_f16_matches_canonical_apply() {
    let (ctx, stream) = setup();
    let bh = 2i32;
    let seq = 4i32;
    let head_dim = 8i32;
    let td = seq * head_dim;
    let numel = (bh * td) as usize;
    let host_x: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32) * 0.05 - 0.3))
        .collect();
    let (cos_t, sin_t) = default_cs_tables(seq as usize, head_dim as usize, ROPE_DEFAULT_BASE);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y_inter: DeviceBuffer<f16> =
        DeviceBuffer::from_slice(&ctx, &vec![f16::ZERO; numel]).expect("alloc");
    let mut dev_y_canon: DeviceBuffer<f16> =
        DeviceBuffer::from_slice(&ctx, &vec![f16::ZERO; numel]).expect("alloc");

    let status_canon = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_f16_run(
            bh,
            td,
            head_dim,
            0,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_y_canon.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_canon, 0);

    let status_inter = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_interleaved_f16_run(
            bh,
            td,
            head_dim,
            0,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_y_inter.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status_inter, 0);
    stream.synchronize().expect("sync");

    let mut got_inter = vec![f16::ZERO; numel];
    let mut got_canon = vec![f16::ZERO; numel];
    dev_y_inter.copy_to_host(&mut got_inter).expect("dl");
    dev_y_canon.copy_to_host(&mut got_canon).expect("dl");

    for (i, (a, b)) in got_inter.iter().zip(got_canon.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "interleaved f16 vs canonical f16 bit mismatch @ {i}"
        );
    }
}

#[test]
fn interleaved_can_implement_validates_args() {
    let s = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_interleaved_f32_can_implement(-1, 8, 4, 0)
    };
    assert_ne!(s, 0, "should reject negative bh");
    let s = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_interleaved_f32_can_implement(2, 8, 5, 0)
    };
    assert_ne!(s, 0, "should reject odd head_dim");
    let s = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_interleaved_f32_can_implement(2, 7, 4, 0)
    };
    assert_ne!(s, 0, "should reject td % d != 0");
    let s = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_interleaved_f32_can_implement(2, 16, 4, 0)
    };
    assert_eq!(s, 0, "valid: bh=2 td=16 d=4 stride_b=0");
    let s = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_interleaved_backward_f32_can_implement(
            2, 16, 4, 0,
        )
    };
    assert_eq!(s, 0);
}

// =============================================================================
// THD-layout variant — CPU-ref tests
// =============================================================================

#[test]
#[ignore]
fn rope_apply_thd_f32_single_head_single_t_matches_cpu() {
    // `[T, H, D] == [1, 1, D]` reduces to a single RoPE vector. Drives
    // the per-t stride codepath with the minimal outer extent.
    let (ctx, stream) = setup();
    let t_outer = 1i32;
    let h_heads = 1i32;
    let head_dim = 8i32;
    let half_d = (head_dim / 2) as usize;
    let numel = (t_outer * h_heads * head_dim) as usize;

    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.1 - 0.2).collect();
    let (cos_t, sin_t) = default_cs_tables(t_outer as usize, head_dim as usize, ROPE_DEFAULT_BASE);
    assert_eq!(cos_t.len(), (t_outer as usize) * half_d);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_thd_f32_run(
            t_outer,
            h_heads,
            head_dim,
            half_d as i32, // stride_b = D/2 — per-t tables
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

    let ref_y = cpu_rope_apply_thd_f32(
        &host_x,
        &cos_t,
        &sin_t,
        t_outer as usize,
        h_heads as usize,
        head_dim as usize,
        half_d,
    );
    for (i, (g, r)) in got.iter().zip(ref_y.iter()).enumerate() {
        let diff = (g - r).abs();
        let tol = 1e-5 * r.abs().max(1.0);
        assert!(diff <= tol, "thd-f32 mismatch @ {i}: gpu={g} cpu={r}");
    }
}

#[test]
#[ignore]
fn rope_apply_thd_f32_full_extent_matches_cpu() {
    let (ctx, stream) = setup();
    let t_outer = 6i32;
    let h_heads = 4i32;
    let head_dim = 8i32;
    let half_d = (head_dim / 2) as usize;
    let numel = (t_outer * h_heads * head_dim) as usize;

    let host_x: Vec<f32> = (0..numel)
        .map(|i| ((i as f32) * 0.07 - 0.4).cos() * 0.9)
        .collect();
    let (cos_t, sin_t) = default_cs_tables(t_outer as usize, head_dim as usize, ROPE_DEFAULT_BASE);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_thd_f32_run(
            t_outer,
            h_heads,
            head_dim,
            half_d as i32,
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

    let ref_y = cpu_rope_apply_thd_f32(
        &host_x,
        &cos_t,
        &sin_t,
        t_outer as usize,
        h_heads as usize,
        head_dim as usize,
        half_d,
    );
    for (i, (g, r)) in got.iter().zip(ref_y.iter()).enumerate() {
        let diff = (g - r).abs();
        let tol = 1e-5 * r.abs().max(1.0);
        assert!(diff <= tol, "thd-f32 full-extent mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn rope_apply_thd_backward_f32_matches_cpu() {
    // The BW is the orthogonal-rotation reverse; we check that
    // `BW(FW(x)) == x` to confirm trig-sign correctness.
    let (ctx, stream) = setup();
    let t_outer = 3i32;
    let h_heads = 2i32;
    let head_dim = 8i32;
    let half_d = (head_dim / 2) as usize;
    let numel = (t_outer * h_heads * head_dim) as usize;

    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.13 - 0.5).collect();
    let (cos_t, sin_t) = default_cs_tables(t_outer as usize, head_dim as usize, ROPE_DEFAULT_BASE);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let mut dev_roundtrip: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, numel).expect("alloc roundtrip");

    let s1 = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_thd_f32_run(
            t_outer,
            h_heads,
            head_dim,
            half_d as i32,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(s1, 0);

    let s2 = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_thd_backward_f32_run(
            t_outer,
            h_heads,
            head_dim,
            half_d as i32,
            dev_y.as_slice().as_raw().0 as *const c_void,
            dev_cos.as_slice().as_raw().0 as *const c_void,
            dev_sin.as_slice().as_raw().0 as *const c_void,
            dev_roundtrip.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(s2, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_roundtrip.copy_to_host(&mut got).expect("dl");

    for (i, (g, x)) in got.iter().zip(host_x.iter()).enumerate() {
        let diff = (g - x).abs();
        let tol = 1e-5 * x.abs().max(1.0);
        assert!(diff <= tol, "thd round-trip mismatch @ {i}: got={g} x={x}");
    }
}

#[test]
#[ignore]
fn rope_apply_thd_bf16_smoke() {
    let (ctx, stream) = setup();
    let t_outer = 2i32;
    let h_heads = 2i32;
    let head_dim = 8i32;
    let half_d = (head_dim / 2) as usize;
    let numel = (t_outer * h_heads * head_dim) as usize;

    let host_x: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32) * 0.05 - 0.2))
        .collect();
    let (cos_t, sin_t) = default_cs_tables(t_outer as usize, head_dim as usize, ROPE_DEFAULT_BASE);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let dev_cos = DeviceBuffer::from_slice(&ctx, &cos_t).expect("up cos");
    let dev_sin = DeviceBuffer::from_slice(&ctx, &sin_t).expect("up sin");
    let mut dev_y: DeviceBuffer<bf16> =
        DeviceBuffer::from_slice(&ctx, &vec![bf16::ZERO; numel]).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_thd_bf16_run(
            t_outer,
            h_heads,
            head_dim,
            half_d as i32,
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

    // Convert host_x to f32, run CPU reference, then compare bf16 result.
    let host_x_f32: Vec<f32> = host_x.iter().map(|v| v.to_f32()).collect();
    let ref_y = cpu_rope_apply_thd_f32(
        &host_x_f32,
        &cos_t,
        &sin_t,
        t_outer as usize,
        h_heads as usize,
        head_dim as usize,
        half_d,
    );

    let mut got = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    for (i, (g, r)) in got.iter().zip(ref_y.iter()).enumerate() {
        let gf = g.to_f32();
        let diff = (gf - r).abs();
        let tol = 8.0e-3 * r.abs().max(1.0); // bf16 ~7 mantissa bits
        assert!(diff <= tol, "thd bf16 mismatch @ {i}: gpu={gf} ref={r}");
    }
}

#[test]
fn thd_can_implement_validates_args() {
    let s = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_thd_f32_can_implement(-1, 1, 4, 2)
    };
    assert_ne!(s, 0, "should reject negative t_outer");
    let s = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_thd_f32_can_implement(2, 1, 5, 2)
    };
    assert_ne!(s, 0, "should reject odd head_dim");
    let s = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_thd_f32_can_implement(2, 1, 4, 2)
    };
    assert_eq!(s, 0, "valid: t=2 h=1 d=4 stride_b=2");
    let s = unsafe {
        baracuda_kernels_sys::baracuda_kernels_rope_apply_thd_backward_f32_can_implement(2, 1, 4, 2)
    };
    assert_eq!(s, 0);
}
