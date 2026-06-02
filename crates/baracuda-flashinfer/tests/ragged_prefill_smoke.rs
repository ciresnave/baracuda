//! Smoke tests for ragged (non-paged) prefill — Phase 66 Tier 2.
//! `#[ignore]` + `flashinfer` feature.
//!
//! Same closed-form trick as paged prefill: identical keys across KV ⇒
//! uniform softmax ⇒ output = mean(attended V).

#![cfg(feature = "flashinfer")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_flashinfer::attention::{
    BatchRaggedPrefillArgs, BatchRaggedPrefillDescriptor, BatchRaggedPrefillPlan,
};
use baracuda_flashinfer::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::f16;

const HEAD_DIM: usize = 128;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn ragged_prefill_plan_select_validates() {
    let (_ctx, stream) = setup();
    let desc = BatchRaggedPrefillDescriptor {
        batch_size: 2,
        total_num_rows: 8,
        total_kv_rows: 12,
        num_qo_heads: 8,
        num_kv_heads: 2,
        head_dim: HEAD_DIM as i32,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        causal: true,
        enable_kv_split: false,
        element: ElementKind::F16,
    };
    let _plan = BatchRaggedPrefillPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("valid descriptor should select");
    let mut bad = desc;
    bad.head_dim = 96;
    assert!(BatchRaggedPrefillPlan::<f16>::select(&stream, &bad, PlanPreference::default()).is_err());
}

/// Non-causal, identical keys, 1 request, 2 q rows over 2 KV tokens ⇒ both
/// rows output mean(V).
#[test]
#[ignore]
fn ragged_prefill_uniform_key_is_mean() {
    let (ctx, stream) = setup();
    let qo_len = 2usize;
    let kv_len = 2usize;
    let v_vals = [1.0f32, 3.0];
    let mean = 2.0f32;

    let q_h: Vec<f16> = (0..qo_len * HEAD_DIM).map(|_| f16::from_f32(0.5)).collect();
    let k_h: Vec<f16> = (0..kv_len * HEAD_DIM).map(|_| f16::from_f32(0.1)).collect();
    let mut v_h: Vec<f16> = Vec::with_capacity(kv_len * HEAD_DIM);
    for &val in &v_vals {
        v_h.extend((0..HEAD_DIM).map(|_| f16::from_f32(val)));
    }

    let q_dev = DeviceBuffer::from_slice(&ctx, &q_h).expect("q");
    let q_indptr_dev = DeviceBuffer::from_slice(&ctx, &[0i32, qo_len as i32]).expect("q_indptr");
    let k_dev = DeviceBuffer::from_slice(&ctx, &k_h).expect("k");
    let v_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("v");
    let kv_indptr_dev = DeviceBuffer::from_slice(&ctx, &[0i32, kv_len as i32]).expect("kv_indptr");
    let mut o_dev: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, qo_len * HEAD_DIM).expect("o");
    let mut lse_dev: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, qo_len).expect("lse");

    let desc = BatchRaggedPrefillDescriptor {
        batch_size: 1,
        total_num_rows: qo_len as i32,
        total_kv_rows: kv_len as i32,
        num_qo_heads: 1,
        num_kv_heads: 1,
        head_dim: HEAD_DIM as i32,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        causal: false,
        enable_kv_split: false,
        element: ElementKind::F16,
    };
    let plan = BatchRaggedPrefillPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let qo_shape = [qo_len as i32, 1, HEAD_DIM as i32];
    let kv_shape = [kv_len as i32, 1, HEAD_DIM as i32];
    plan.run(
        &stream,
        Workspace::None,
        BatchRaggedPrefillArgs {
            q: TensorRef { data: q_dev.as_slice(), shape: qo_shape, stride: contiguous_stride(qo_shape) },
            q_indptr: TensorRef { data: q_indptr_dev.as_slice(), shape: [2], stride: [1] },
            k_data: TensorRef { data: k_dev.as_slice(), shape: kv_shape, stride: contiguous_stride(kv_shape) },
            v_data: TensorRef { data: v_dev.as_slice(), shape: kv_shape, stride: contiguous_stride(kv_shape) },
            kv_indptr: TensorRef { data: kv_indptr_dev.as_slice(), shape: [2], stride: [1] },
            o: TensorMut { data: o_dev.as_slice_mut(), shape: qo_shape, stride: contiguous_stride(qo_shape) },
            lse: TensorMut { data: lse_dev.as_slice_mut(), shape: [qo_len as i32, 1], stride: contiguous_stride([qo_len as i32, 1]) },
        },
    )
    .expect("ragged prefill run");
    stream.synchronize().expect("sync");

    let mut o_host = vec![f16::ZERO; qo_len * HEAD_DIM];
    o_dev.copy_to_host(&mut o_host).expect("download o");
    for r in 0..qo_len {
        let got = o_host[r * HEAD_DIM].to_f32();
        assert!((got - mean).abs() < 3e-2, "ragged row {r}: got {got}, expected mean {mean}");
    }
}
