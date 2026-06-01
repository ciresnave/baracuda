//! Smoke tests for the FlashInfer sort-free sampling plan, via the
//! `baracuda-flashinfer` safe facade — Phase 66.
//!
//! Two layers:
//!   1. Plan-select gate validation across sampler variants + parameter
//!      ranges.
//!   2. A real-GPU end-to-end check: a one-hot probability row leaves
//!      exactly one admissible token, so every sampler variant must
//!      return that index regardless of the RNG seed. This closes the
//!      Phase 46 deferral ("sampler vs reference") for the deterministic
//!      case.
//!
//! `#[ignore]` by default — requires a real CUDA device + the
//! `flashinfer` cargo feature.

#![cfg(feature = "flashinfer")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_flashinfer::sampling::{
    SamplerKind, TopKTopPSamplingArgs, TopKTopPSamplingDescriptor, TopKTopPSamplingPlan,
};
use baracuda_flashinfer::{contiguous_stride, PlanPreference, TensorMut, TensorRef, Workspace};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn sampling_plan_select_validates() {
    let (_ctx, stream) = setup();
    let mk = |sampler| TopKTopPSamplingDescriptor {
        batch_size: 4,
        vocab_size: 32,
        sampler,
        deterministic: true,
    };

    // Each variant selects for valid parameters.
    for s in [
        SamplerKind::TopK { top_k: 8 },
        SamplerKind::TopP { top_p: 0.9 },
        SamplerKind::MinP { min_p: 0.05 },
        SamplerKind::TopKTopP { top_k: 8, top_p: 0.9 },
    ] {
        let plan = TopKTopPSamplingPlan::select(&stream, &mk(s), PlanPreference::default())
            .expect("valid sampler descriptor should select");
        assert_eq!(plan.workspace_size(), 0, "sampling is workspace-free");
    }

    // Out-of-range parameters are rejected.
    assert!(TopKTopPSamplingPlan::select(&stream, &mk(SamplerKind::TopK { top_k: 0 }), PlanPreference::default()).is_err());
    assert!(TopKTopPSamplingPlan::select(&stream, &mk(SamplerKind::TopK { top_k: 999 }), PlanPreference::default()).is_err());
    assert!(TopKTopPSamplingPlan::select(&stream, &mk(SamplerKind::TopP { top_p: 0.0 }), PlanPreference::default()).is_err());
    assert!(TopKTopPSamplingPlan::select(&stream, &mk(SamplerKind::TopP { top_p: 1.5 }), PlanPreference::default()).is_err());
    assert!(TopKTopPSamplingPlan::select(&stream, &mk(SamplerKind::MinP { min_p: 0.0 }), PlanPreference::default()).is_err());
}

/// A one-hot probability row admits exactly one token; every sampler
/// must return its index, for any seed.
#[test]
#[ignore]
fn sampling_one_hot_is_deterministic() {
    let (ctx, stream) = setup();

    // A realistic vocabulary size: FlashInfer's sampling kernels use
    // block-strided / vectorized loads sized for production vocabularies
    // and are not meant for degenerate tiny vocabs.
    const VOCAB: usize = 4096;
    const HOT: usize = 2718;
    let mut probs = vec![0.0f32; VOCAB];
    probs[HOT] = 1.0;
    let probs_dev = DeviceBuffer::from_slice(&ctx, &probs).expect("upload probs");

    for sampler in [
        SamplerKind::TopK { top_k: 1 },
        SamplerKind::TopP { top_p: 0.5 },
        SamplerKind::MinP { min_p: 0.5 },
        SamplerKind::TopKTopP { top_k: 1, top_p: 0.5 },
    ] {
        let desc = TopKTopPSamplingDescriptor {
            batch_size: 1,
            vocab_size: VOCAB as i32,
            sampler,
            deterministic: false,
        };
        let plan = TopKTopPSamplingPlan::select(&stream, &desc, PlanPreference::default())
            .expect("select");

        let mut out_dev: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc out");
        let mut valid_dev: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 1).expect("alloc valid");
        let probs_shape = [1, VOCAB as i32];
        plan.run(
            &stream,
            Workspace::None,
            TopKTopPSamplingArgs {
                probs: TensorRef {
                    data: probs_dev.as_slice(),
                    shape: probs_shape,
                    stride: contiguous_stride(probs_shape),
                },
                output: TensorMut { data: out_dev.as_slice_mut(), shape: [1], stride: [1] },
                valid: Some(TensorMut { data: valid_dev.as_slice_mut(), shape: [1], stride: [1] }),
                seed_val: 0x1234_5678,
                offset_val: 0,
            },
        )
        .expect("sampling run");
        stream.synchronize().expect("sync");

        let mut out_host = [0i32];
        out_dev.copy_to_host(&mut out_host).expect("download out");
        assert_eq!(
            out_host[0] as usize, HOT,
            "one-hot row must sample the hot index for sampler {sampler:?}",
        );
    }
}
