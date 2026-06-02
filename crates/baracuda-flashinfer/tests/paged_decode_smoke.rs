//! Smoke tests for the FlashInfer paged-KV decode + append plans, via
//! the `baracuda-flashinfer` safe facade — Phase 66.
//!
//! Two layers:
//!   1. Plan-select gate validation (head_dim / dtype / GQA / extents),
//!      mirroring the Phase 46 kernels-crate smoke tests but exercised
//!      through `baracuda_flashinfer::attention`.
//!   2. A real-GPU end-to-end check: with a paged store holding exactly
//!      ONE KV token, the attention softmax is over a single score, so
//!      the decode output must equal that token's V vector (independent
//!      of Q / K). This is the Phase 46 deferral ("decode output vs
//!      reference") closed for the minimal case.
//!
//! `#[ignore]` by default — requires a real CUDA device + the
//! `flashinfer` cargo feature.

#![cfg(feature = "flashinfer")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_flashinfer::attention::{
    BatchPagedDecodeArgs, BatchPagedDecodeDescriptor, BatchPagedDecodePlan, PagedKvAppendDescriptor,
    PagedKvAppendPlan, PagedKvCacheDescriptor,
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
        .expect("valid descriptor should select");
    assert!(plan.workspace_size() > 0, "paged decode needs index workspace");

    // Bad head_dim → Unsupported.
    let mut bad = desc;
    bad.paged_kv.head_dim = 96;
    assert!(
        BatchPagedDecodePlan::<f16>::select(&stream, &bad, PlanPreference::default()).is_err(),
        "head_dim 96 must be rejected",
    );

    // Non-integer GQA grouping → InvalidProblem.
    let mut bad_gqa = desc;
    bad_gqa.num_qo_heads = 7; // 7 % 2 != 0
    assert!(
        BatchPagedDecodePlan::<f16>::select(&stream, &bad_gqa, PlanPreference::default()).is_err(),
        "non-integer GQA group must be rejected",
    );

    // The companion append plan accepts the same paged descriptor.
    let app_desc = PagedKvAppendDescriptor {
        batch_size: 2,
        paged_kv: paged,
    };
    let _app = PagedKvAppendPlan::<f16>::select(&stream, &app_desc, PlanPreference::default())
        .expect("append plan should select for valid descriptor");
}

/// With a single KV token in the paged store, softmax is over one score
/// (= 1.0), so the decode output equals that token's V vector exactly
/// (up to f16 rounding), regardless of Q / K. A clean closed-form check.
#[test]
#[ignore]
fn paged_decode_single_token_equals_v() {
    let (ctx, stream) = setup();

    const HEAD_DIM: usize = 128;
    let paged = PagedKvCacheDescriptor {
        page_size: 1,
        num_total_pages: 1,
        num_kv_heads: 1,
        head_dim: HEAD_DIM as i32,
        element: ElementKind::F16,
    };
    let desc = BatchPagedDecodeDescriptor {
        batch_size: 1,
        num_qo_heads: 1,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        paged_kv: paged,
    };
    let plan = BatchPagedDecodePlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    // Host fixtures.
    let q_h: Vec<f16> = (0..HEAD_DIM).map(|_| f16::from_f32(0.37)).collect();
    let k_h: Vec<f16> = (0..HEAD_DIM).map(|_| f16::from_f32(-0.11)).collect();
    // The reference: V values we expect to read back in `o`.
    let v_ref: Vec<f32> = (0..HEAD_DIM).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let v_h: Vec<f16> = v_ref.iter().map(|&x| f16::from_f32(x)).collect();

    let q_dev = DeviceBuffer::from_slice(&ctx, &q_h).expect("upload q");
    let k_dev = DeviceBuffer::from_slice(&ctx, &k_h).expect("upload k");
    let v_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("upload v");
    let indices_dev = DeviceBuffer::from_slice(&ctx, &[0i32]).expect("upload indices");
    let indptr_dev = DeviceBuffer::from_slice(&ctx, &[0i32, 1]).expect("upload indptr");
    let last_page_len_dev = DeviceBuffer::from_slice(&ctx, &[1i32]).expect("upload last_page_len");

    let mut o_dev: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, HEAD_DIM).expect("alloc o");
    let mut lse_dev: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc lse");
    let mut ws_dev: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, plan.workspace_size()).expect("alloc ws");

    let q_shape = [1, 1, HEAD_DIM as i32];
    let cache_shape = [1, 1, 1, HEAD_DIM as i32];

    plan.run(
        &stream,
        Workspace::Borrowed(ws_dev.as_slice_mut()),
        BatchPagedDecodeArgs {
            q: TensorRef { data: q_dev.as_slice(), shape: q_shape, stride: contiguous_stride(q_shape) },
            k_data: TensorRef { data: k_dev.as_slice(), shape: cache_shape, stride: contiguous_stride(cache_shape) },
            v_data: TensorRef { data: v_dev.as_slice(), shape: cache_shape, stride: contiguous_stride(cache_shape) },
            indices: TensorRef { data: indices_dev.as_slice(), shape: [1], stride: [1] },
            indptr: TensorRef { data: indptr_dev.as_slice(), shape: [2], stride: [1] },
            last_page_len: TensorRef { data: last_page_len_dev.as_slice(), shape: [1], stride: [1] },
            o: TensorMut { data: o_dev.as_slice_mut(), shape: q_shape, stride: contiguous_stride(q_shape) },
            lse: TensorMut { data: lse_dev.as_slice_mut(), shape: [1, 1], stride: contiguous_stride([1, 1]) },
        },
    )
    .expect("paged decode run");
    stream.synchronize().expect("sync");

    let mut o_host = vec![f16::ZERO; HEAD_DIM];
    o_dev.copy_to_host(&mut o_host).expect("download o");

    for (i, (&got, &want)) in o_host.iter().zip(v_ref.iter()).enumerate() {
        let got = got.to_f32();
        assert!(
            (got - want).abs() < 2e-2,
            "o[{i}] = {got}, expected V[{i}] = {want} (single-token decode must return V)",
        );
    }
}
