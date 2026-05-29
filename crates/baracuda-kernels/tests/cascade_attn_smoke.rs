//! Real-GPU smoke test for `CascadeAttentionPlan` — Phase 46
//! (FlashInfer LSE merge).
//!
//! Phase 46 Tier 1 trailblazer scope:
//!   - Plan-select validates head_dim + element gates.
//!
//! End-to-end LSE-merge correctness vs CPU reference is left to a
//! follow-up integration test once the DeviceBuffer testing helpers
//! are stabilized.
//!
//! `#[ignore]` by default — requires a real CUDA device + the
//! `flashinfer` cargo feature.

#![cfg(feature = "flashinfer")]

use baracuda_driver::{init, Context, Device, Stream};
use baracuda_kernels::{
    CascadeAttentionDescriptor, CascadeAttentionPlan, ElementKind, PlanPreference,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn cascade_attention_plan_select_validates_f32() {
    let (_ctx, stream) = setup();
    let desc = CascadeAttentionDescriptor {
        seq_len: 4,
        num_heads: 2,
        head_dim: 64,
        element: ElementKind::F32,
    };
    let plan = CascadeAttentionPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    assert_eq!(plan.workspace_size(), 0);

    // Bad head_dim must be rejected.
    let bad = CascadeAttentionDescriptor {
        head_dim: 96,
        ..desc
    };
    assert!(
        CascadeAttentionPlan::<f32>::select(&stream, &bad, PlanPreference::default()).is_err()
    );
}

#[test]
#[ignore]
fn cascade_attention_plan_select_validates_f16() {
    let (_ctx, stream) = setup();
    let desc = CascadeAttentionDescriptor {
        seq_len: 4,
        num_heads: 2,
        head_dim: 128,
        element: ElementKind::F16,
    };
    let plan =
        CascadeAttentionPlan::<half::f16>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    assert_eq!(plan.workspace_size(), 0);
}

#[test]
#[ignore]
fn cascade_attention_plan_select_validates_bf16() {
    let (_ctx, stream) = setup();
    let desc = CascadeAttentionDescriptor {
        seq_len: 4,
        num_heads: 2,
        head_dim: 256,
        element: ElementKind::Bf16,
    };
    CascadeAttentionPlan::<half::bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
}
