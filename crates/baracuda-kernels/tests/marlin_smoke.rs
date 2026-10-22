//! Descriptor-rejection test for `Int4MarlinGemmPlan`.
//!
//! Numerical correctness of the Marlin kernel and of `gptq_to_marlin_repack`
//! is covered by `marlin_reference.rs` (kernel vs a dequantise-then-GEMM
//! reference at real shapes, with a corrupted-scale control).
//!
//! Gated behind the `device-tests` feature (selecting a plan needs a Stream,
//! hence a device) and the `marlin` cargo feature.

#![cfg(feature = "marlin")]
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, Stream, init, require};
use baracuda_kernels::{Int4MarlinGemmDescriptor, Int4MarlinGemmPlan, PlanPreference};
use half::f16;

#[test]
fn marlin_plan_select_rejects_invalid_descriptor() {
    let _ = init();
    // select() does no GPU work, but reaching it needs a real Stream, which
    // needs a device — which MUST exist on the box.
    let dev = require!(
        Device::get(0),
        "a CUDA device (Marlin select() needs a Stream)"
    );
    let ctx = Context::new(&dev).expect("ctx");
    let stream = Stream::new(&ctx).expect("stream");

    // K not divisible by 128.
    let bad_k = Int4MarlinGemmDescriptor::new(1, 256, 64);
    assert!(
        Int4MarlinGemmPlan::<f16>::select(&stream, &bad_k, PlanPreference::default()).is_err(),
        "should reject K=64 (not divisible by 128)"
    );

    // N not divisible by 256.
    let bad_n = Int4MarlinGemmDescriptor::new(1, 128, 128);
    assert!(
        Int4MarlinGemmPlan::<f16>::select(&stream, &bad_n, PlanPreference::default()).is_err(),
        "should reject N=128 (not divisible by 256)"
    );

    // group_size = 64 unsupported.
    let bad_g = Int4MarlinGemmDescriptor::new(1, 256, 128).with_group_size(64);
    assert!(
        Int4MarlinGemmPlan::<f16>::select(&stream, &bad_g, PlanPreference::default()).is_err(),
        "should reject group_size=64"
    );

    // Minimal valid descriptor.
    let ok = Int4MarlinGemmDescriptor::new(1, 256, 128);
    assert!(
        Int4MarlinGemmPlan::<f16>::select(&stream, &ok, PlanPreference::default()).is_ok(),
        "minimal valid descriptor should succeed"
    );
}
