//! Smoke tests for the FlashInfer cascade-attention LSE-merge plan, via
//! the `baracuda-flashinfer` safe facade — Phase 66.
//!
//! Two layers:
//!   1. Plan-select gate validation (head_dim / dtype / extents).
//!   2. A real-GPU end-to-end check: when both partial states carry the
//!      same base-2 LSE, the merge weights are equal, so the merged `v`
//!      is the arithmetic mean of the two inputs and the merged LSE is
//!      `s + 1` (since `log2(1 + 1) = 1`). Closes the Phase 46 deferral
//!      ("cascade vs CPU LSE-merge formula") for the symmetric case.
//!
//! `#[ignore]` by default — requires a real CUDA device + the
//! `flashinfer` cargo feature.

#![cfg(feature = "flashinfer")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_flashinfer::attention::{
    CascadeAttentionArgs, CascadeAttentionDescriptor, CascadeAttentionPlan,
};
use baracuda_flashinfer::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
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
fn cascade_plan_select_validates() {
    let (_ctx, stream) = setup();
    let desc = CascadeAttentionDescriptor {
        seq_len: 1,
        num_heads: 4,
        head_dim: 128,
        element: ElementKind::F16,
    };
    let _plan = CascadeAttentionPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("valid descriptor should select");

    let mut bad = desc;
    bad.head_dim = 96;
    assert!(
        CascadeAttentionPlan::<f16>::select(&stream, &bad, PlanPreference::default()).is_err(),
        "head_dim 96 must be rejected",
    );
}

/// Equal base-2 LSE on both sides → merged `v` is the mean, merged LSE
/// is `s + 1`.
#[test]
#[ignore]
fn cascade_equal_lse_averages() {
    let (ctx, stream) = setup();

    const HEAD_DIM: usize = 128;
    let desc = CascadeAttentionDescriptor {
        seq_len: 1,
        num_heads: 1,
        head_dim: HEAD_DIM as i32,
        element: ElementKind::F16,
    };
    let plan = CascadeAttentionPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let v_h: Vec<f16> = (0..HEAD_DIM).map(|_| f16::from_f32(1.0)).collect();
    let v_other_h: Vec<f16> = (0..HEAD_DIM).map(|_| f16::from_f32(3.0)).collect();

    let mut v_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("upload v");
    let v_other_dev = DeviceBuffer::from_slice(&ctx, &v_other_h).expect("upload v_other");
    let mut s_dev = DeviceBuffer::from_slice(&ctx, &[0.0f32]).expect("upload s");
    let s_other_dev = DeviceBuffer::from_slice(&ctx, &[0.0f32]).expect("upload s_other");

    let v_shape = [1, 1, HEAD_DIM as i32];
    let s_shape = [1, 1];
    plan.run(
        &stream,
        Workspace::None,
        CascadeAttentionArgs {
            v: TensorMut { data: v_dev.as_slice_mut(), shape: v_shape, stride: contiguous_stride(v_shape) },
            s: TensorMut { data: s_dev.as_slice_mut(), shape: s_shape, stride: contiguous_stride(s_shape) },
            v_other: TensorRef { data: v_other_dev.as_slice(), shape: v_shape, stride: contiguous_stride(v_shape) },
            s_other: TensorRef { data: s_other_dev.as_slice(), shape: s_shape, stride: contiguous_stride(s_shape) },
        },
    )
    .expect("cascade merge run");
    stream.synchronize().expect("sync");

    let mut v_host = vec![f16::ZERO; HEAD_DIM];
    v_dev.copy_to_host(&mut v_host).expect("download v");
    let mut s_host = [0.0f32];
    s_dev.copy_to_host(&mut s_host).expect("download s");

    for (i, &got) in v_host.iter().enumerate() {
        let got = got.to_f32();
        assert!((got - 2.0).abs() < 2e-2, "v[{i}] = {got}, expected mean 2.0");
    }
    assert!(
        (s_host[0] - 1.0).abs() < 1e-3,
        "merged LSE = {}, expected s + log2(2) = 1.0",
        s_host[0],
    );
}
