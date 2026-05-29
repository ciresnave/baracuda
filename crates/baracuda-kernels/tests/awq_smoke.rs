//! Real-GPU smoke test for `Int4AwqGemmPlan` — Phase 48 Goal B.
//!
//! Constructs a small AWQ GEMM with a programmatically built weight
//! tensor (all-zero zero-points, all-one scales, weights uniformly
//! set to the int4 value `0`), launches the AWQ kernel, and verifies
//! the kernel writes finite output.
//!
//! AWQ's asymmetric int4 with all-zero zeros + all-one scales
//! reconstructs the weight as `1.0 * (q - 0) = q`. With `q = 0`
//! everywhere the dequant is 0, so the GEMM output should be 0.
//!
//! `#[ignore]` by default — requires a real CUDA device + the `awq`
//! cargo feature on `baracuda-kernels-sys`.

#![cfg(feature = "awq")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Int4AwqGemmArgs, Int4AwqGemmDescriptor, Int4AwqGemmPlan, PlanPreference,
    TensorMut, TensorRef, Workspace,
};
use baracuda_kernels_sys::baracuda_kernels_int4_awq_gemm_f16_workspace_bytes;
use half::f16;

fn setup() -> Option<(Context, Stream)> {
    if init().is_err() {
        return None;
    }
    let device = Device::get(0).ok()?;
    let ctx = Context::new(&device).ok()?;
    let stream = Stream::new(&ctx).ok()?;
    Some((ctx, stream))
}

#[test]
fn awq_plan_select_rejects_invalid_descriptor() {
    let Some((_ctx, stream)) = setup() else {
        // No GPU on this host — skip the validation tests too because
        // select() needs a Stream.
        return;
    };

    // OC not divisible by 64.
    let bad_oc = Int4AwqGemmDescriptor::new(1, 256, 48);
    assert!(
        Int4AwqGemmPlan::<f16>::select(&stream, &bad_oc, PlanPreference::default()).is_err()
    );

    // group_size != 64/128.
    let bad_g = Int4AwqGemmDescriptor::new(1, 256, 64).with_group_size(32);
    assert!(
        Int4AwqGemmPlan::<f16>::select(&stream, &bad_g, PlanPreference::default()).is_err()
    );

    // IC not divisible by 32 * split_k_iters.
    let bad_ic = Int4AwqGemmDescriptor::new(1, 96, 64)
        .with_group_size(64)
        .with_split_k_iters(8);
    assert!(
        Int4AwqGemmPlan::<f16>::select(&stream, &bad_ic, PlanPreference::default()).is_err()
    );

    // Minimal valid.
    let ok = Int4AwqGemmDescriptor::new(1, 256, 64).with_group_size(128);
    assert!(
        Int4AwqGemmPlan::<f16>::select(&stream, &ok, PlanPreference::default()).is_ok()
    );
}

/// End-to-end AWQ GEMM smoke test on a real GPU.
///
/// Setup: M=1, IC=256, OC=64, group_size=128, split_k_iters=8.
/// All zeros = 0, all scales = 1.0, all weights = nibble 0 →
/// dequant = 0 → output should be all zeros.
#[test]
#[ignore]
fn awq_gemm_zero_weights_smoke() {
    let Some((ctx, stream)) = setup() else {
        eprintln!("awq_smoke: no CUDA device, skipping");
        return;
    };
    let m: i32 = 1;
    let ic: i32 = 256;
    let oc: i32 = 64;
    let group_size: i32 = 128;
    let split_k_iters: i32 = 8;

    // Activation [M, IC] = ones.
    let host_a: Vec<f16> = vec![f16::from_f32(1.0); (m * ic) as usize];
    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("up A");

    // Weight [OC, IC/8] = all zeros (every nibble = 0 → dequant = 0).
    let weight_len = (oc as usize) * (ic as usize) / 8;
    let host_w: Vec<i32> = vec![0i32; weight_len];
    let dev_w = DeviceBuffer::from_slice(&ctx, &host_w).expect("up W");

    // Scales [IC/g, OC] = all ones.
    let scales_len = ((ic / group_size) as usize) * (oc as usize);
    let host_s: Vec<f16> = vec![f16::from_f32(1.0); scales_len];
    let dev_s = DeviceBuffer::from_slice(&ctx, &host_s).expect("up S");

    // Zeros [IC/g, OC/8] = all zeros.
    let zeros_len = ((ic / group_size) as usize) * ((oc as usize) / 8);
    let host_z: Vec<i32> = vec![0i32; zeros_len];
    let dev_z = DeviceBuffer::from_slice(&ctx, &host_z).expect("up Z");

    // Output [M, OC].
    let mut dev_c: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, (m * oc) as usize).expect("alloc C");

    // Workspace size from FFI.
    let ws_bytes =
        unsafe { baracuda_kernels_int4_awq_gemm_f16_workspace_bytes(m, oc, split_k_iters) };
    assert!(ws_bytes > 0, "workspace bytes must be positive");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc workspace");

    let desc = Int4AwqGemmDescriptor::new(m, ic, oc)
        .with_group_size(group_size)
        .with_split_k_iters(split_k_iters);
    let plan = Int4AwqGemmPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("plan select");
    assert_eq!(plan.workspace_size(), ws_bytes, "plan/FFI workspace agree");

    let args = Int4AwqGemmArgs::<f16> {
        activation: TensorRef {
            data: dev_a.as_slice(),
            shape: [m, ic],
            stride: contiguous_stride([m, ic]),
        },
        weight_packed: TensorRef {
            data: dev_w.as_slice(),
            shape: [oc, ic / 8],
            stride: contiguous_stride([oc, ic / 8]),
        },
        scales: TensorRef {
            data: dev_s.as_slice(),
            shape: [ic / group_size, oc],
            stride: contiguous_stride([ic / group_size, oc]),
        },
        zeros: TensorRef {
            data: dev_z.as_slice(),
            shape: [ic / group_size, oc / 8],
            stride: contiguous_stride([ic / group_size, oc / 8]),
        },
        output: TensorMut {
            data: dev_c.as_slice_mut(),
            shape: [m, oc],
            stride: contiguous_stride([m, oc]),
        },
    };

    let ws = Workspace::Borrowed(dev_ws.as_slice_mut());
    plan.run(&stream, ws, args).expect("awq run");
    stream.synchronize().expect("sync");

    let mut host_c: Vec<f16> = vec![f16::ZERO; (m * oc) as usize];
    dev_c.copy_to_host(&mut host_c).expect("download C");

    // All weights = 0 → all outputs should be 0.
    let max_abs = host_c
        .iter()
        .map(|v| v.to_f32().abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_abs < 1e-3,
        "AWQ output with zero weights should be ~0, got max |out| = {max_abs}"
    );
}
