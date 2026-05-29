//! Real-GPU smoke test for `TopKTopPSamplingPlan` — Phase 46
//! (FlashInfer sort-free sampling).
//!
//! Phase 46 Tier 1 trailblazer scope:
//!   - Plan-select validates filter parameter ranges.
//!   - Each of the four sampler variants (TopK / TopP / MinP /
//!     TopKTopP) is selectable.
//!
//! End-to-end correctness vs naive `topk + softmax + multinomial`
//! is left to follow-up integration tests once the random-plan
//! testing harness can be shared.
//!
//! `#[ignore]` by default — requires a real CUDA device + the
//! `flashinfer` cargo feature.

#![cfg(feature = "flashinfer")]

use baracuda_driver::{init, Context, Device, Stream};
use baracuda_kernels::{
    PlanPreference, SamplerKind, TopKTopPSamplingDescriptor, TopKTopPSamplingPlan,
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
fn topk_topp_sampling_plan_select_validates() {
    let (_ctx, stream) = setup();
    let desc = TopKTopPSamplingDescriptor {
        batch_size: 4,
        vocab_size: 1024,
        sampler: SamplerKind::TopKTopP {
            top_k: 50,
            top_p: 0.9,
        },
        deterministic: true,
    };
    let plan = TopKTopPSamplingPlan::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    assert_eq!(plan.workspace_size(), 0);

    // Invalid top_k.
    let bad_topk = TopKTopPSamplingDescriptor {
        sampler: SamplerKind::TopK { top_k: 0 },
        ..desc
    };
    assert!(
        TopKTopPSamplingPlan::select(&stream, &bad_topk, PlanPreference::default()).is_err()
    );

    // Invalid top_p (out of (0, 1]).
    let bad_topp = TopKTopPSamplingDescriptor {
        sampler: SamplerKind::TopP { top_p: 1.5 },
        ..desc
    };
    assert!(
        TopKTopPSamplingPlan::select(&stream, &bad_topp, PlanPreference::default()).is_err()
    );

    // Invalid min_p (out of (0, 1]).
    let bad_minp = TopKTopPSamplingDescriptor {
        sampler: SamplerKind::MinP { min_p: 0.0 },
        ..desc
    };
    assert!(
        TopKTopPSamplingPlan::select(&stream, &bad_minp, PlanPreference::default()).is_err()
    );
}

#[test]
#[ignore]
fn topk_sampling_plan_selectable() {
    let (_ctx, stream) = setup();
    let desc = TopKTopPSamplingDescriptor {
        batch_size: 2,
        vocab_size: 64,
        sampler: SamplerKind::TopK { top_k: 5 },
        deterministic: true,
    };
    TopKTopPSamplingPlan::select(&stream, &desc, PlanPreference::default())
        .expect("top-k select");
}

#[test]
#[ignore]
fn topp_sampling_plan_selectable() {
    let (_ctx, stream) = setup();
    let desc = TopKTopPSamplingDescriptor {
        batch_size: 2,
        vocab_size: 64,
        sampler: SamplerKind::TopP { top_p: 0.9 },
        deterministic: true,
    };
    TopKTopPSamplingPlan::select(&stream, &desc, PlanPreference::default())
        .expect("top-p select");
}

#[test]
#[ignore]
fn min_p_sampling_plan_selectable() {
    let (_ctx, stream) = setup();
    let desc = TopKTopPSamplingDescriptor {
        batch_size: 2,
        vocab_size: 64,
        sampler: SamplerKind::MinP { min_p: 0.1 },
        deterministic: true,
    };
    TopKTopPSamplingPlan::select(&stream, &desc, PlanPreference::default())
        .expect("min-p select");
}

#[test]
#[ignore]
fn topk_topp_sampling_plan_selectable() {
    let (_ctx, stream) = setup();
    let desc = TopKTopPSamplingDescriptor {
        batch_size: 2,
        vocab_size: 64,
        sampler: SamplerKind::TopKTopP {
            top_k: 10,
            top_p: 0.9,
        },
        deterministic: true,
    };
    TopKTopPSamplingPlan::select(&stream, &desc, PlanPreference::default())
        .expect("top-k+top-p select");
}
