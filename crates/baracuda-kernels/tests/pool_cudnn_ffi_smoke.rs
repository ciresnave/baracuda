#![cfg(feature = "cudnn")]
//! Smoke test for the Phase 19.1 cuDNN pool FFI facade.
//!
//! Verifies that the raw `extern "C"` FFI entry points in
//! `baracuda-kernels-sys` produce the same output as the equivalent
//! Rust plan layer (`MaxPool2dPlan`, `AvgPool2dPlan`, ...). This is the
//! load-bearing test for Phase 19.1: the FFI surface must be
//! functionally identical to the plan-layer dispatch.
//!
//! Each test runs the FFI symbol once, runs the matching Rust plan
//! once, and asserts byte-equivalence on the output buffers. cuDNN's
//! pooling kernel is deterministic for fixed input + descriptor, so
//! we expect exact bit-equality on the same hardware.

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AvgPool2dPlan, AvgPool3dPlan, ElementKind, MaxPool1dPlan, MaxPool2dPlan,
    PlanPreference, Pool1dDescriptor, Pool1dFwArgs, Pool2dBwArgs, Pool2dDescriptor, Pool2dFwArgs,
    Pool3dDescriptor, Pool3dFwArgs, PoolMode, TensorMut, TensorRef, Workspace,
};
use baracuda_kernels_sys::{
    baracuda_kernels_avg_pool_2d_bw_f32_run, baracuda_kernels_avg_pool_2d_fw_f32_run,
    baracuda_kernels_avg_pool_3d_fw_bf16_run, baracuda_kernels_max_pool_1d_fw_f16_run,
    baracuda_kernels_max_pool_2d_bw_f32_run, baracuda_kernels_max_pool_2d_fw_f32_run,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// =============================================================================
// MaxPool2d f32 — FW + BW vs Rust plan
// =============================================================================

#[test]
#[ignore]
fn max_pool_2d_f32_ffi_matches_plan() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (2i32, 3i32, 4i32, 4i32);
    let (kh, kw) = (2i32, 2i32);
    let (sh, sw) = (2i32, 2i32);
    let (ph, pw) = (0i32, 0i32);
    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let numel_x = (n * c * h_in * w_in) as usize;
    let numel_y = (n * c * h_out * w_out) as usize;
    let host_x: Vec<f32> = (0..numel_x).map(|i| ((i as f32) * 0.137).sin()).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");

    // --- FFI run ---
    let dev_y_ffi: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_ffi");
    let ffi_status = unsafe {
        baracuda_kernels_max_pool_2d_fw_f32_run(
            n,
            c,
            h_in,
            w_in,
            h_out,
            w_out,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            dev_x.as_raw().0 as *const c_void,
            dev_y_ffi.as_raw().0 as *mut c_void,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(ffi_status, 0, "ffi max_pool_2d_fw_f32 status");
    stream.synchronize().expect("sync ffi fw");
    let mut host_y_ffi = vec![0f32; numel_y];
    dev_y_ffi.copy_to_host(&mut host_y_ffi).expect("dl ffi y");

    // --- Plan run ---
    let mut dev_y_plan: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_plan");
    let desc = Pool2dDescriptor::new(
        n, c, h_in, w_in, kh, kw, PoolMode::Max, ElementKind::F32,
    )
    .with_padding(ph, pw)
    .with_stride(sh, sw);
    let plan = MaxPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out, w_out];
    plan.run_fw(
        &stream,
        Workspace::None,
        Pool2dFwArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            y: TensorMut {
                data: dev_y_plan.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("plan fw");
    stream.synchronize().expect("sync plan fw");
    let mut host_y_plan = vec![0f32; numel_y];
    dev_y_plan.copy_to_host(&mut host_y_plan).expect("dl plan y");

    for i in 0..numel_y {
        assert_eq!(
            host_y_ffi[i].to_bits(),
            host_y_plan[i].to_bits(),
            "MaxPool2d FW f32 mismatch @ {i}: ffi={} plan={}",
            host_y_ffi[i],
            host_y_plan[i],
        );
    }

    // --- BW ---
    let host_dy: Vec<f32> = (0..numel_y).map(|i| 1.0 + (i as f32) * 0.073).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");

    let dev_dx_ffi: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx_ffi");
    let ffi_bw_status = unsafe {
        baracuda_kernels_max_pool_2d_bw_f32_run(
            n,
            c,
            h_in,
            w_in,
            h_out,
            w_out,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            dev_y_ffi.as_raw().0 as *const c_void,
            dev_dy.as_raw().0 as *const c_void,
            dev_x.as_raw().0 as *const c_void,
            dev_dx_ffi.as_raw().0 as *mut c_void,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(ffi_bw_status, 0, "ffi max_pool_2d_bw_f32 status");
    stream.synchronize().expect("sync ffi bw");
    let mut host_dx_ffi = vec![0f32; numel_x];
    dev_dx_ffi.copy_to_host(&mut host_dx_ffi).expect("dl ffi dx");

    let mut dev_dx_plan: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx_plan");
    plan.run_bw(
        &stream,
        Workspace::None,
        Pool2dBwArgs {
            y: TensorRef {
                data: dev_y_plan.as_slice(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            dx: TensorMut {
                data: dev_dx_plan.as_slice_mut(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
        },
    )
    .expect("plan bw");
    stream.synchronize().expect("sync plan bw");
    let mut host_dx_plan = vec![0f32; numel_x];
    dev_dx_plan.copy_to_host(&mut host_dx_plan).expect("dl plan dx");

    for i in 0..numel_x {
        assert_eq!(
            host_dx_ffi[i].to_bits(),
            host_dx_plan[i].to_bits(),
            "MaxPool2d BW f32 mismatch @ {i}: ffi={} plan={}",
            host_dx_ffi[i],
            host_dx_plan[i],
        );
    }
}

// =============================================================================
// AvgPool2d f32 — FW + BW vs Rust plan (both count_include_pad modes)
// =============================================================================

fn avg_pool_2d_f32_one_mode(count_include_pad: bool) {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 2i32, 5i32, 5i32);
    let (kh, kw) = (3i32, 3i32);
    let (sh, sw) = (2i32, 2i32);
    let (ph, pw) = (1i32, 1i32); // non-zero padding to exercise the mode
    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let numel_x = (n * c * h_in * w_in) as usize;
    let numel_y = (n * c * h_out * w_out) as usize;
    let host_x: Vec<f32> = (0..numel_x).map(|i| 0.5 + (i as f32) * 0.21).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");

    let dev_y_ffi: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_ffi");
    let ffi_status = unsafe {
        baracuda_kernels_avg_pool_2d_fw_f32_run(
            n,
            c,
            h_in,
            w_in,
            h_out,
            w_out,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            if count_include_pad { 1 } else { 0 },
            dev_x.as_raw().0 as *const c_void,
            dev_y_ffi.as_raw().0 as *mut c_void,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(ffi_status, 0, "ffi avg_pool_2d_fw_f32 status");
    stream.synchronize().expect("sync ffi fw");
    let mut host_y_ffi = vec![0f32; numel_y];
    dev_y_ffi.copy_to_host(&mut host_y_ffi).expect("dl ffi y");

    let mut dev_y_plan: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_plan");
    let mode = if count_include_pad {
        PoolMode::AvgIncludePad
    } else {
        PoolMode::AvgExcludePad
    };
    let desc = Pool2dDescriptor::new(n, c, h_in, w_in, kh, kw, mode, ElementKind::F32)
        .with_padding(ph, pw)
        .with_stride(sh, sw);
    let plan = AvgPool2dPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");
    let x_shape = [n, c, h_in, w_in];
    let y_shape = [n, c, h_out, w_out];
    plan.run_fw(
        &stream,
        Workspace::None,
        Pool2dFwArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            y: TensorMut {
                data: dev_y_plan.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("plan fw");
    stream.synchronize().expect("sync plan fw");
    let mut host_y_plan = vec![0f32; numel_y];
    dev_y_plan.copy_to_host(&mut host_y_plan).expect("dl plan y");

    for i in 0..numel_y {
        assert_eq!(
            host_y_ffi[i].to_bits(),
            host_y_plan[i].to_bits(),
            "AvgPool2d FW f32 (cip={count_include_pad}) mismatch @ {i}: ffi={} plan={}",
            host_y_ffi[i],
            host_y_plan[i],
        );
    }

    // --- BW ---
    let host_dy: Vec<f32> = (0..numel_y).map(|i| 1.0 + (i as f32) * 0.073).collect();
    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("up dy");

    let dev_dx_ffi: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx_ffi");
    let ffi_bw_status = unsafe {
        baracuda_kernels_avg_pool_2d_bw_f32_run(
            n,
            c,
            h_in,
            w_in,
            h_out,
            w_out,
            kh,
            kw,
            sh,
            sw,
            ph,
            pw,
            if count_include_pad { 1 } else { 0 },
            dev_y_ffi.as_raw().0 as *const c_void,
            dev_dy.as_raw().0 as *const c_void,
            dev_x.as_raw().0 as *const c_void,
            dev_dx_ffi.as_raw().0 as *mut c_void,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(ffi_bw_status, 0, "ffi avg_pool_2d_bw_f32 status");
    stream.synchronize().expect("sync ffi bw");
    let mut host_dx_ffi = vec![0f32; numel_x];
    dev_dx_ffi.copy_to_host(&mut host_dx_ffi).expect("dl ffi dx");

    let mut dev_dx_plan: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel_x).expect("dx_plan");
    plan.run_bw(
        &stream,
        Workspace::None,
        Pool2dBwArgs {
            y: TensorRef {
                data: dev_y_plan.as_slice(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
            dy: TensorRef {
                data: dev_dy.as_slice(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            dx: TensorMut {
                data: dev_dx_plan.as_slice_mut(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
        },
    )
    .expect("plan bw");
    stream.synchronize().expect("sync plan bw");
    let mut host_dx_plan = vec![0f32; numel_x];
    dev_dx_plan.copy_to_host(&mut host_dx_plan).expect("dl plan dx");

    for i in 0..numel_x {
        assert_eq!(
            host_dx_ffi[i].to_bits(),
            host_dx_plan[i].to_bits(),
            "AvgPool2d BW f32 (cip={count_include_pad}) mismatch @ {i}: ffi={} plan={}",
            host_dx_ffi[i],
            host_dx_plan[i],
        );
    }
}

#[test]
#[ignore]
fn avg_pool_2d_f32_count_include_pad_true() {
    avg_pool_2d_f32_one_mode(true);
}

#[test]
#[ignore]
fn avg_pool_2d_f32_count_include_pad_false() {
    avg_pool_2d_f32_one_mode(false);
}

// =============================================================================
// MaxPool1d f16 — FW only vs Rust plan
// =============================================================================

#[test]
#[ignore]
fn max_pool_1d_f16_fw_ffi_matches_plan() {
    let (ctx, stream) = setup();
    let (n, c, l_in) = (1i32, 2i32, 16i32);
    let (kl, sl, pl) = (4i32, 2i32, 1i32);
    let l_out = (l_in + 2 * pl - kl) / sl + 1;

    let numel_x = (n * c * l_in) as usize;
    let numel_y = (n * c * l_out) as usize;
    let host_x: Vec<f16> = (0..numel_x)
        .map(|i| f16::from_f32(((i as f32) * 0.137).sin()))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");

    let dev_y_ffi: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_ffi");
    let ffi_status = unsafe {
        baracuda_kernels_max_pool_1d_fw_f16_run(
            n,
            c,
            l_in,
            l_out,
            kl,
            sl,
            pl,
            dev_x.as_raw().0 as *const c_void,
            dev_y_ffi.as_raw().0 as *mut c_void,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(ffi_status, 0, "ffi max_pool_1d_fw_f16 status");
    stream.synchronize().expect("sync ffi fw");
    let mut host_y_ffi = vec![f16::ZERO; numel_y];
    dev_y_ffi.copy_to_host(&mut host_y_ffi).expect("dl ffi y");

    let mut dev_y_plan: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_plan");
    let desc = Pool1dDescriptor::new(n, c, l_in, kl, PoolMode::Max, ElementKind::F16)
        .with_padding(pl)
        .with_stride(sl);
    let plan = MaxPool1dPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");
    let x_shape = [n, c, l_in];
    let y_shape = [n, c, l_out];
    plan.run_fw(
        &stream,
        Workspace::None,
        Pool1dFwArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            y: TensorMut {
                data: dev_y_plan.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("plan fw");
    stream.synchronize().expect("sync plan fw");
    let mut host_y_plan = vec![f16::ZERO; numel_y];
    dev_y_plan.copy_to_host(&mut host_y_plan).expect("dl plan y");

    for i in 0..numel_y {
        assert_eq!(
            host_y_ffi[i].to_bits(),
            host_y_plan[i].to_bits(),
            "MaxPool1d FW f16 mismatch @ {i}: ffi={} plan={}",
            host_y_ffi[i],
            host_y_plan[i],
        );
    }
}

// =============================================================================
// AvgPool3d bf16 — FW only vs Rust plan
// =============================================================================

#[test]
#[ignore]
fn avg_pool_3d_bf16_fw_ffi_matches_plan() {
    let (ctx, stream) = setup();
    let (n, c, d_in, h_in, w_in) = (1i32, 1i32, 4i32, 4i32, 4i32);
    let (kd, kh, kw) = (2i32, 2i32, 2i32);
    let (sd, sh, sw) = (2i32, 2i32, 2i32);
    let (pd, ph, pw) = (0i32, 0i32, 0i32);
    let d_out = (d_in + 2 * pd - kd) / sd + 1;
    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;

    let numel_x = (n * c * d_in * h_in * w_in) as usize;
    let numel_y = (n * c * d_out * h_out * w_out) as usize;
    let host_x: Vec<bf16> = (0..numel_x)
        .map(|i| bf16::from_f32(0.25 + (i as f32) * 0.01))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");

    let dev_y_ffi: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_ffi");
    let ffi_status = unsafe {
        baracuda_kernels_avg_pool_3d_fw_bf16_run(
            n,
            c,
            d_in,
            h_in,
            w_in,
            d_out,
            h_out,
            w_out,
            kd,
            kh,
            kw,
            sd,
            sh,
            sw,
            pd,
            ph,
            pw,
            0, // count_include_pad = false (PyTorch default)
            dev_x.as_raw().0 as *const c_void,
            dev_y_ffi.as_raw().0 as *mut c_void,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(ffi_status, 0, "ffi avg_pool_3d_fw_bf16 status");
    stream.synchronize().expect("sync ffi fw");
    let mut host_y_ffi = vec![bf16::ZERO; numel_y];
    dev_y_ffi.copy_to_host(&mut host_y_ffi).expect("dl ffi y");

    let mut dev_y_plan: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel_y).expect("y_plan");
    let desc = Pool3dDescriptor::new(
        n,
        c,
        d_in,
        h_in,
        w_in,
        kd,
        kh,
        kw,
        PoolMode::AvgExcludePad,
        ElementKind::Bf16,
    )
    .with_padding(pd, ph, pw)
    .with_stride(sd, sh, sw);
    let plan = AvgPool3dPlan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");
    let x_shape = [n, c, d_in, h_in, w_in];
    let y_shape = [n, c, d_out, h_out, w_out];
    plan.run_fw(
        &stream,
        Workspace::None,
        Pool3dFwArgs {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            y: TensorMut {
                data: dev_y_plan.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    )
    .expect("plan fw");
    stream.synchronize().expect("sync plan fw");
    let mut host_y_plan = vec![bf16::ZERO; numel_y];
    dev_y_plan.copy_to_host(&mut host_y_plan).expect("dl plan y");

    for i in 0..numel_y {
        assert_eq!(
            host_y_ffi[i].to_bits(),
            host_y_plan[i].to_bits(),
            "AvgPool3d FW bf16 mismatch @ {i}: ffi={} plan={}",
            host_y_ffi[i],
            host_y_plan[i],
        );
    }
}
