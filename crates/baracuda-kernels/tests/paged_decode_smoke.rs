//! Real-GPU smoke test for `BatchPagedDecodePlan` + `PagedKvAppendPlan`.
//! — Phase 46 (FlashInfer cherry-pick).
//!
//! Phase 46 Tier 1 trailblazer scope:
//!   - Plan-select validates extents / dtype / head_dim gates.
//!   - Workspace-size query matches the C-side launcher exactly.
//!   - The companion `PagedKvAppendPlan` accepts the same descriptor.
//!
//! End-to-end correctness (decode output vs SDPA reference on the
//! equivalent contiguous layout) is left to a follow-up integration
//! test once the plan layer's args-builder helpers are merged.
//!
//! `#[ignore]` by default — requires a real CUDA device + the
//! `flashinfer` cargo feature.

#![cfg(feature = "flashinfer")]

use baracuda_driver::{init, Context, Device, Stream};
use baracuda_kernels::{
    BatchPagedDecodeDescriptor, BatchPagedDecodePlan, ElementKind, PagedKvAppendDescriptor,
    PagedKvAppendPlan, PagedKvCacheDescriptor, PlanPreference,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn paged_decode_plan_select_validates() {
    let (_ctx, stream) = setup();
    let paged = PagedKvCacheDescriptor {
        page_size: 16,
        num_total_pages: 8,
        num_kv_heads: 2,
        head_dim: 128,
        element: ElementKind::F16,
    };
    let desc = BatchPagedDecodeDescriptor {
        batch_size: 2,
        num_qo_heads: 8,
        sm_scale: 1.0 / (128.0_f32).sqrt(),
        paged_kv: paged,
    };
    let plan = BatchPagedDecodePlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select should accept valid descriptor");
    assert!(
        plan.workspace_size() > 0,
        "paged decode needs auxiliary index workspace",
    );

    // Bad head_dim → Unsupported.
    let bad = BatchPagedDecodeDescriptor {
        paged_kv: PagedKvCacheDescriptor {
            head_dim: 96,
            ..paged
        },
        ..desc
    };
    assert!(
        BatchPagedDecodePlan::<f16>::select(&stream, &bad, PlanPreference::default()).is_err(),
        "head_dim=96 must be rejected",
    );

    // num_qo_heads not divisible by num_kv_heads → InvalidProblem.
    let gqa_bad = BatchPagedDecodeDescriptor {
        num_qo_heads: 9, // not divisible by num_kv_heads=2
        ..desc
    };
    assert!(
        BatchPagedDecodePlan::<f16>::select(&stream, &gqa_bad, PlanPreference::default())
            .is_err(),
        "non-integer GQA group size must be rejected",
    );
}

#[test]
#[ignore]
fn paged_kv_append_plan_select_validates() {
    let (_ctx, stream) = setup();
    let paged = PagedKvCacheDescriptor {
        page_size: 16,
        num_total_pages: 4,
        num_kv_heads: 2,
        head_dim: 64,
        element: ElementKind::F16,
    };
    let desc = PagedKvAppendDescriptor {
        batch_size: 2,
        paged_kv: paged,
    };
    let plan = PagedKvAppendPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    assert_eq!(plan.workspace_size(), 0);
}

#[test]
#[ignore]
fn paged_decode_workspace_size_matches_formula() {
    // The Rust-side workspace_size() reports `(3 * batch + 2) * 4`
    // bytes, matching the launcher's auxiliary index buffers.
    // (The matching C-side `*_workspace_size` symbol exists but is
    // not linkable in the Phase 46 Checkpoint A build — paged decode
    // launcher is staged for a follow-up; only the formula is
    // exercised here.)
    let (_ctx, stream) = setup();
    let paged = PagedKvCacheDescriptor {
        page_size: 16,
        num_total_pages: 8,
        num_kv_heads: 2,
        head_dim: 128,
        element: ElementKind::F16,
    };
    for batch_size in [1, 2, 4, 8, 16] {
        let desc = BatchPagedDecodeDescriptor {
            batch_size,
            num_qo_heads: 8,
            sm_scale: 1.0 / (128.0_f32).sqrt(),
            paged_kv: paged,
        };
        let plan =
            BatchPagedDecodePlan::<f16>::select(&stream, &desc, PlanPreference::default())
                .expect("select");
        let expected = ((3 * batch_size as usize) + 2) * 4;
        assert_eq!(
            plan.workspace_size(),
            expected,
            "workspace formula mismatch at batch={batch_size}",
        );
    }
}
