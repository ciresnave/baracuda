//! Direct-FFI smoke test for the **Phase 65d in-place dispatch
//! contract** on BatchNorm. Proves that calling
//! `baracuda_kernels_batch_norm_<dt>_run` with `x_ptr == y_ptr`
//! produces the same result as the non-aliased call.
//!
//! BN's safety story is structurally different from the Phase 65b/c
//! normalizers: BN runs a two-kernel pipeline (stage-1 computes
//! per-group `(mean, inv_std)` into `saved_mean`/`saved_rstd`;
//! stage-2 reads each `x[i]` into a register then writes `y[i]`).
//! Stage-1 never touches `y`; stage-2 is single-read-then-write per
//! cell. Both stages are aliasing-safe **regardless of dtype**, so
//! this proof covers **f64 as well** — unlike LN/SM/LSM/RMSNorm
//! where f64 falls back to a multi-pass-global kernel.
//!
//! `#[ignore]` per project convention; run with
//! `cargo test -p baracuda-kernels --features sm89 \
//!    --test batch_norm_inplace_smoke -- --include-ignored`.

#![cfg(any(feature = "sm80", feature = "sm89", feature = "sm90a"))]

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

const N: i32 = 2;
const C: i32 = 4;
const S: i32 = 16;
const GROUP_KIND_BN: i32 = 0;
const NUM_GROUPS_BN: i32 = 1; // ignored for BN; kernel uses c_extent

fn host_x_f32() -> Vec<f32> {
    let n_total = (N * C * S) as usize;
    (0..n_total)
        .map(|i| ((i as f32) * 0.013 - 0.4).cos() * 0.9)
        .collect()
}

#[test]
#[ignore]
fn batch_norm_f32_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let host_x = host_x_f32();
    let eps: f32 = 1e-5;

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc");
    let mut dev_mean_ref: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, C as usize).expect("mean");
    let mut dev_rstd_ref: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, C as usize).expect("rstd");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_batch_norm_f32_run(
            N, C, S, NUM_GROUPS_BN, GROUP_KIND_BN, eps,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(), core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_mean_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rstd_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "non-aliased BN f32");
    stream.synchronize().expect("sync ref");
    let mut ref_out = vec![0_f32; host_x.len()];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_mean_ip: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, C as usize).expect("mean");
    let mut dev_rstd_ip: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, C as usize).expect("rstd");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_batch_norm_f32_run(
            N, C, S, NUM_GROUPS_BN, GROUP_KIND_BN, eps,
            p as *const c_void,
            core::ptr::null(), core::ptr::null(),
            p as *mut c_void,
            dev_mean_ip.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rstd_ip.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "aliased BN f32");
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; host_x.len()];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..host_x.len() {
        assert_eq!(aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "f32 in-place BN @ {i}: aliased={} non-aliased={}",
            aliased_out[i], ref_out[i]);
    }
}

#[test]
#[ignore]
fn batch_norm_f16_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let host_x: Vec<f16> = host_x_f32().iter().map(|&v| f16::from_f32(v)).collect();
    let eps: f32 = 1e-5;

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc");
    let mut dev_mean_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, C as usize).expect("mean");
    let mut dev_rstd_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, C as usize).expect("rstd");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_batch_norm_f16_run(
            N, C, S, NUM_GROUPS_BN, GROUP_KIND_BN, eps,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(), core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_mean_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rstd_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![f16::ZERO; host_x.len()];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_mean_ip: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, C as usize).expect("mean");
    let mut dev_rstd_ip: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, C as usize).expect("rstd");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_batch_norm_f16_run(
            N, C, S, NUM_GROUPS_BN, GROUP_KIND_BN, eps,
            p as *const c_void,
            core::ptr::null(), core::ptr::null(),
            p as *mut c_void,
            dev_mean_ip.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rstd_ip.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![f16::ZERO; host_x.len()];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..host_x.len() {
        assert_eq!(aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "f16 in-place BN @ {i}");
    }
}

#[test]
#[ignore]
fn batch_norm_bf16_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let host_x: Vec<bf16> = host_x_f32().iter().map(|&v| bf16::from_f32(v)).collect();
    let eps: f32 = 1e-5;

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc");
    let mut dev_mean_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, C as usize).expect("mean");
    let mut dev_rstd_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, C as usize).expect("rstd");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_batch_norm_bf16_run(
            N, C, S, NUM_GROUPS_BN, GROUP_KIND_BN, eps,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(), core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_mean_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rstd_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![bf16::ZERO; host_x.len()];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_mean_ip: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, C as usize).expect("mean");
    let mut dev_rstd_ip: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, C as usize).expect("rstd");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_batch_norm_bf16_run(
            N, C, S, NUM_GROUPS_BN, GROUP_KIND_BN, eps,
            p as *const c_void,
            core::ptr::null(), core::ptr::null(),
            p as *mut c_void,
            dev_mean_ip.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rstd_ip.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![bf16::ZERO; host_x.len()];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..host_x.len() {
        assert_eq!(aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "bf16 in-place BN @ {i}");
    }
}

#[test]
#[ignore]
fn batch_norm_f64_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let host_x: Vec<f64> = host_x_f32().iter().map(|&v| v as f64).collect();
    let eps: f32 = 1e-5;

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc");
    let mut dev_mean_ref: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, C as usize).expect("mean");
    let mut dev_rstd_ref: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, C as usize).expect("rstd");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_batch_norm_f64_run(
            N, C, S, NUM_GROUPS_BN, GROUP_KIND_BN, eps,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(), core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_mean_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rstd_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f64; host_x.len()];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_mean_ip: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, C as usize).expect("mean");
    let mut dev_rstd_ip: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, C as usize).expect("rstd");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_batch_norm_f64_run(
            N, C, S, NUM_GROUPS_BN, GROUP_KIND_BN, eps,
            p as *const c_void,
            core::ptr::null(), core::ptr::null(),
            p as *mut c_void,
            dev_mean_ip.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rstd_ip.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f64; host_x.len()];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..host_x.len() {
        assert_eq!(aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "f64 in-place BN @ {i}: aliased={} non-aliased={}",
            aliased_out[i], ref_out[i]);
    }
}
