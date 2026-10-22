//! Descriptor-rejection test for `Int4AwqGemmPlan`.
//!
//! Numerical correctness of the AWQ kernel is covered by `awq_reference.rs`
//! (kernel vs a dequantise-then-GEMM reference at real shapes, with random
//! zero-points and a corrupted-scale control).
//!
//! Gated behind the `device-tests` feature (selecting a plan needs a Stream,
//! hence a device) and the `awq` cargo feature.

#![cfg(feature = "awq")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, Stream, init, require};
use baracuda_kernels::{Int4AwqGemmDescriptor, Int4AwqGemmPlan, PlanPreference};
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
    let (_ctx, stream) = require!(
        setup(),
        "a CUDA device + stream (AWQ select() needs a Stream)"
    );

    // OC not divisible by 64.
    let bad_oc = Int4AwqGemmDescriptor::new(1, 256, 48);
    assert!(Int4AwqGemmPlan::<f16>::select(&stream, &bad_oc, PlanPreference::default()).is_err());

    // group_size != 64/128.
    let bad_g = Int4AwqGemmDescriptor::new(1, 256, 64).with_group_size(32);
    assert!(Int4AwqGemmPlan::<f16>::select(&stream, &bad_g, PlanPreference::default()).is_err());

    // IC not divisible by 32 * split_k_iters.
    let bad_ic = Int4AwqGemmDescriptor::new(1, 96, 64)
        .with_group_size(64)
        .with_split_k_iters(8);
    assert!(Int4AwqGemmPlan::<f16>::select(&stream, &bad_ic, PlanPreference::default()).is_err());

    // Minimal valid.
    let ok = Int4AwqGemmDescriptor::new(1, 256, 64).with_group_size(128);
    assert!(Int4AwqGemmPlan::<f16>::select(&stream, &ok, PlanPreference::default()).is_ok());
}
