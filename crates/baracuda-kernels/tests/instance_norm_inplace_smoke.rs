//! Direct-FFI smoke test for the **Phase 65d in-place dispatch
//! contract** on InstanceNorm. IN dispatches through GroupNorm with
//! `num_groups == c_extent`, so this proof covers the IN call path
//! end-to-end (the safe wrapper's `InstanceNormPlan` is sugar over
//! that same kernel).
//!
//! Same two-stage argument as BN/GN: stage-1 reads `x` and writes
//! only the saved buffers; stage-2 reads `x[i]` then writes `y[i]`
//! in the same thread. Aliasing-safe **regardless of dtype**.
//!
//! `#[ignore]` per project convention; run with
//! `cargo test -p baracuda-kernels --features sm89 \
//!    --test instance_norm_inplace_smoke -- --include-ignored`.

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
const GROUP_KIND_GN: i32 = 1;
const NUM_GROUPS_IN: i32 = C; // IN = GN with num_groups == c_extent
// group_count = N * C = 8 for IN

fn host_x_f32() -> Vec<f32> {
    let n_total = (N * C * S) as usize;
    (0..n_total)
        .map(|i| ((i as f32) * 0.017 - 0.1).cos() * 0.95)
        .collect()
}

#[test]
#[ignore]
fn instance_norm_f32_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let host_x = host_x_f32();
    let eps: f32 = 1e-5;
    let group_count = (N * NUM_GROUPS_IN) as usize;

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc");
    let mut dev_mean_ref: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, group_count).expect("mean");
    let mut dev_rstd_ref: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, group_count).expect("rstd");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_group_norm_f32_run(
            N, C, S, NUM_GROUPS_IN, GROUP_KIND_GN, eps,
            dev_x_ref.as_slice().as_raw().0 as *const c_void,
            core::ptr::null(), core::ptr::null(),
            dev_y_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_mean_ref.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rstd_ref.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "non-aliased IN f32");
    stream.synchronize().expect("sync");
    let mut ref_out = vec![0_f32; host_x.len()];
    dev_y_ref.copy_to_host(&mut ref_out).expect("dl");

    let mut dev_inplace = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_mean_ip: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, group_count).expect("mean");
    let mut dev_rstd_ip: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, group_count).expect("rstd");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_group_norm_f32_run(
            N, C, S, NUM_GROUPS_IN, GROUP_KIND_GN, eps,
            p as *const c_void,
            core::ptr::null(), core::ptr::null(),
            p as *mut c_void,
            dev_mean_ip.as_slice_mut().as_raw().0 as *mut c_void,
            dev_rstd_ip.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(), 0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "aliased IN f32");
    stream.synchronize().expect("sync");
    let mut aliased_out = vec![0_f32; host_x.len()];
    dev_inplace.copy_to_host(&mut aliased_out).expect("dl");

    for i in 0..host_x.len() {
        assert_eq!(aliased_out[i].to_bits(), ref_out[i].to_bits(),
            "f32 in-place IN @ {i}: aliased={} non-aliased={}",
            aliased_out[i], ref_out[i]);
    }
}

#[test]
#[ignore]
fn instance_norm_f16_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let host_x: Vec<f16> = host_x_f32().iter().map(|&v| f16::from_f32(v)).collect();
    let eps: f32 = 1e-5;
    let group_count = (N * NUM_GROUPS_IN) as usize;

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc");
    let mut dev_mean_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, group_count).expect("mean");
    let mut dev_rstd_ref: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, group_count).expect("rstd");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_group_norm_f16_run(
            N, C, S, NUM_GROUPS_IN, GROUP_KIND_GN, eps,
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
    let mut dev_mean_ip: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, group_count).expect("mean");
    let mut dev_rstd_ip: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, group_count).expect("rstd");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_group_norm_f16_run(
            N, C, S, NUM_GROUPS_IN, GROUP_KIND_GN, eps,
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
            "f16 in-place IN @ {i}");
    }
}

#[test]
#[ignore]
fn instance_norm_bf16_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let host_x: Vec<bf16> = host_x_f32().iter().map(|&v| bf16::from_f32(v)).collect();
    let eps: f32 = 1e-5;
    let group_count = (N * NUM_GROUPS_IN) as usize;

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc");
    let mut dev_mean_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, group_count).expect("mean");
    let mut dev_rstd_ref: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, group_count).expect("rstd");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_group_norm_bf16_run(
            N, C, S, NUM_GROUPS_IN, GROUP_KIND_GN, eps,
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
    let mut dev_mean_ip: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, group_count).expect("mean");
    let mut dev_rstd_ip: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, group_count).expect("rstd");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_group_norm_bf16_run(
            N, C, S, NUM_GROUPS_IN, GROUP_KIND_GN, eps,
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
            "bf16 in-place IN @ {i}");
    }
}

#[test]
#[ignore]
fn instance_norm_f64_inplace_matches_non_aliased() {
    let (ctx, stream) = setup();
    let host_x: Vec<f64> = host_x_f32().iter().map(|&v| v as f64).collect();
    let eps: f32 = 1e-5;
    let group_count = (N * NUM_GROUPS_IN) as usize;

    let dev_x_ref = DeviceBuffer::from_slice(&ctx, &host_x).expect("up");
    let mut dev_y_ref: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc");
    let mut dev_mean_ref: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, group_count).expect("mean");
    let mut dev_rstd_ref: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, group_count).expect("rstd");
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_group_norm_f64_run(
            N, C, S, NUM_GROUPS_IN, GROUP_KIND_GN, eps,
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
    let mut dev_mean_ip: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, group_count).expect("mean");
    let mut dev_rstd_ip: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, group_count).expect("rstd");
    let p = dev_inplace.as_slice_mut().as_raw().0;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_group_norm_f64_run(
            N, C, S, NUM_GROUPS_IN, GROUP_KIND_GN, eps,
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
            "f64 in-place IN @ {i}: aliased={} non-aliased={}",
            aliased_out[i], ref_out[i]);
    }
}
