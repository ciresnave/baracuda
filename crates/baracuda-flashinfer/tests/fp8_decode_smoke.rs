//! Smoke tests for FP8 KV-cache paged decode — Phase 66 Tier 2.
//! `#[ignore]` + `flashinfer` feature.
//!
//! With a single KV token the softmax is over one score, so the decode
//! output equals that token's (dequantized) V vector. Storing V as the
//! e4m3 byte for 1.0 (`0x38`, exactly representable) gives a clean
//! reference of 1.0.

#![cfg(feature = "flashinfer")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_flashinfer::attention::{
    BatchPagedDecodeFp8Args, BatchPagedDecodeFp8Descriptor, BatchPagedDecodeFp8Plan, Fp8KvDtype,
};
use baracuda_flashinfer::{contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, Workspace};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const HEAD_DIM: usize = 128;
/// e4m3 encoding of 1.0: sign 0, exp = bias 7 (0b0111), mantissa 0 -> 0x38.
const E4M3_ONE: u8 = 0x38;

#[test]
#[ignore]
fn fp8_decode_plan_select_validates() {
    let (_ctx, stream) = setup();
    let desc = BatchPagedDecodeFp8Descriptor {
        batch_size: 2,
        num_qo_heads: 8,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        page_size: 16,
        num_total_pages: 8,
        num_kv_heads: 2,
        head_dim: HEAD_DIM as i32,
        kv_dtype: Fp8KvDtype::E4M3,
    };
    let plan = BatchPagedDecodeFp8Plan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("valid descriptor should select");
    assert!(plan.workspace_size() > 0);

    // f32 Q/O rejected.
    assert!(
        BatchPagedDecodeFp8Plan::<f32>::select(&stream, &desc, PlanPreference::default()).is_err(),
        "f32 Q/O must be rejected",
    );
    assert_eq!(plan.sku().aux_element, Some(ElementKind::Fp8E4M3));
}

#[test]
#[ignore]
fn fp8_decode_single_token_equals_v() {
    let (ctx, stream) = setup();
    let desc = BatchPagedDecodeFp8Descriptor {
        batch_size: 1,
        num_qo_heads: 1,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        page_size: 1,
        num_total_pages: 1,
        num_kv_heads: 1,
        head_dim: HEAD_DIM as i32,
        kv_dtype: Fp8KvDtype::E4M3,
    };
    let plan = BatchPagedDecodeFp8Plan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    let q_h: Vec<f16> = (0..HEAD_DIM).map(|_| f16::from_f32(0.5)).collect();
    // K + V stored as e4m3 1.0 (single token, so K is irrelevant to the result).
    let kv_h: Vec<u8> = vec![E4M3_ONE; HEAD_DIM];

    let q_dev = DeviceBuffer::from_slice(&ctx, &q_h).expect("q");
    let k_dev = DeviceBuffer::from_slice(&ctx, &kv_h).expect("k");
    let v_dev = DeviceBuffer::from_slice(&ctx, &kv_h).expect("v");
    let indices_dev = DeviceBuffer::from_slice(&ctx, &[0i32]).expect("indices");
    let indptr_dev = DeviceBuffer::from_slice(&ctx, &[0i32, 1]).expect("indptr");
    let last_page_len_dev = DeviceBuffer::from_slice(&ctx, &[1i32]).expect("lpl");
    let mut o_dev: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, HEAD_DIM).expect("o");
    let mut lse_dev: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("lse");
    let mut ws_dev: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, plan.workspace_size()).expect("ws");

    let q_shape = [1, 1, HEAD_DIM as i32];
    let cache_shape = [1, 1, 1, HEAD_DIM as i32];
    plan.run(
        &stream,
        Workspace::Borrowed(ws_dev.as_slice_mut()),
        BatchPagedDecodeFp8Args {
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
    .expect("fp8 decode run");
    stream.synchronize().expect("sync");

    let mut o_host = vec![f16::ZERO; HEAD_DIM];
    o_dev.copy_to_host(&mut o_host).expect("download o");
    for (i, &got) in o_host.iter().enumerate() {
        let got = got.to_f32();
        assert!((got - 1.0).abs() < 5e-2, "o[{i}] = {got}, expected dequant(V_fp8)=1.0");
    }
}
