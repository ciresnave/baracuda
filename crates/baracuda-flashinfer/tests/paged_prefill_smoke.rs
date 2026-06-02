//! Smoke tests for the FlashInfer paged-KV prefill plan — Phase 66 Tier 2.
//!
//! Closed-form correctness trick: if every KV token shares the SAME key
//! vector, then for any query the attention scores are equal across KV
//! positions, so softmax is uniform and the output is the mean of the
//! attended V vectors — independent of Q. That gives exact references for
//! both the non-causal (attend all) and causal (attend prefix) paths.
//!
//! `#[ignore]` by default — requires a real CUDA device + the
//! `flashinfer` cargo feature.

#![cfg(feature = "flashinfer")]

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_flashinfer::attention::{
    BatchPagedPrefillArgs, BatchPagedPrefillDescriptor, BatchPagedPrefillPlan,
    PagedKvCacheDescriptor,
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

const HEAD_DIM: usize = 128;

#[test]
#[ignore]
fn paged_prefill_plan_select_validates() {
    let (_ctx, stream) = setup();
    let paged = PagedKvCacheDescriptor {
        page_size: 16,
        num_total_pages: 8,
        num_kv_heads: 2,
        head_dim: HEAD_DIM as i32,
        element: ElementKind::F16,
    };
    let desc = BatchPagedPrefillDescriptor {
        batch_size: 2,
        total_num_rows: 10,
        num_qo_heads: 8,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        causal: true,
        enable_kv_split: false,
        paged_kv: paged,
    };
    let _plan = BatchPagedPrefillPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("valid descriptor should select");

    // f32 is rejected (prefill is mma-based).
    let mut f32_desc = desc;
    f32_desc.paged_kv.element = ElementKind::F32;
    assert!(
        BatchPagedPrefillPlan::<f32>::select(&stream, &f32_desc, PlanPreference::default()).is_err(),
        "f32 prefill must be rejected",
    );

    // Bad head_dim.
    let mut bad = desc;
    bad.paged_kv.head_dim = 96;
    assert!(
        BatchPagedPrefillPlan::<f16>::select(&stream, &bad, PlanPreference::default()).is_err(),
        "head_dim 96 must be rejected",
    );

    // Non-integer GQA.
    let mut bad_gqa = desc;
    bad_gqa.num_qo_heads = 7;
    assert!(
        BatchPagedPrefillPlan::<f16>::select(&stream, &bad_gqa, PlanPreference::default()).is_err(),
        "non-integer GQA group must be rejected",
    );
}

/// Build a single-request prefill fixture: `qo_len` query rows, `kv_len`
/// KV tokens in one page, identical keys (=0.1), V[token t] = t-dependent
/// value. Returns the device buffers + the host V values.
struct Fixture {
    q: DeviceBuffer<f16>,
    q_indptr: DeviceBuffer<i32>,
    k: DeviceBuffer<f16>,
    v: DeviceBuffer<f16>,
    kv_indices: DeviceBuffer<i32>,
    kv_indptr: DeviceBuffer<i32>,
    last_page_len: DeviceBuffer<i32>,
    o: DeviceBuffer<f16>,
    lse: DeviceBuffer<f32>,
}

fn make_fixture(ctx: &Context, qo_len: usize, kv_len: usize, v_vals: &[f32]) -> Fixture {
    assert_eq!(v_vals.len(), kv_len);
    let q_h: Vec<f16> = (0..qo_len * HEAD_DIM).map(|_| f16::from_f32(0.5)).collect();
    let k_h: Vec<f16> = (0..kv_len * HEAD_DIM).map(|_| f16::from_f32(0.1)).collect();
    let mut v_h: Vec<f16> = Vec::with_capacity(kv_len * HEAD_DIM);
    for &val in v_vals {
        v_h.extend((0..HEAD_DIM).map(|_| f16::from_f32(val)));
    }
    Fixture {
        q: DeviceBuffer::from_slice(ctx, &q_h).expect("q"),
        q_indptr: DeviceBuffer::from_slice(ctx, &[0i32, qo_len as i32]).expect("q_indptr"),
        k: DeviceBuffer::from_slice(ctx, &k_h).expect("k"),
        v: DeviceBuffer::from_slice(ctx, &v_h).expect("v"),
        kv_indices: DeviceBuffer::from_slice(ctx, &[0i32]).expect("kv_indices"),
        kv_indptr: DeviceBuffer::from_slice(ctx, &[0i32, 1]).expect("kv_indptr"),
        last_page_len: DeviceBuffer::from_slice(ctx, &[kv_len as i32]).expect("last_page_len"),
        o: DeviceBuffer::zeros(ctx, qo_len * HEAD_DIM).expect("o"),
        lse: DeviceBuffer::zeros(ctx, qo_len).expect("lse"),
    }
}

fn run_prefill(
    stream: &Stream, fx: &mut Fixture, qo_len: usize, kv_len: usize, causal: bool, enable_split: bool,
) -> Vec<f32> {
    let paged = PagedKvCacheDescriptor {
        page_size: kv_len as i32, // single page holds the whole history
        num_total_pages: 1,
        num_kv_heads: 1,
        head_dim: HEAD_DIM as i32,
        element: ElementKind::F16,
    };
    let desc = BatchPagedPrefillDescriptor {
        batch_size: 1,
        total_num_rows: qo_len as i32,
        num_qo_heads: 1,
        sm_scale: 1.0 / (HEAD_DIM as f32).sqrt(),
        causal,
        enable_kv_split: enable_split,
        paged_kv: paged,
    };
    let plan = BatchPagedPrefillPlan::<f16>::select(stream, &desc, PlanPreference::default())
        .expect("select");

    let qo_shape = [qo_len as i32, 1, HEAD_DIM as i32];
    let cache_shape = [1, 1, kv_len as i32, HEAD_DIM as i32];
    plan.run(
        stream,
        Workspace::None,
        BatchPagedPrefillArgs {
            q: TensorRef { data: fx.q.as_slice(), shape: qo_shape, stride: contiguous_stride(qo_shape) },
            q_indptr: TensorRef { data: fx.q_indptr.as_slice(), shape: [2], stride: [1] },
            k_data: TensorRef { data: fx.k.as_slice(), shape: cache_shape, stride: contiguous_stride(cache_shape) },
            v_data: TensorRef { data: fx.v.as_slice(), shape: cache_shape, stride: contiguous_stride(cache_shape) },
            kv_indices: TensorRef { data: fx.kv_indices.as_slice(), shape: [1], stride: [1] },
            kv_indptr: TensorRef { data: fx.kv_indptr.as_slice(), shape: [2], stride: [1] },
            last_page_len: TensorRef { data: fx.last_page_len.as_slice(), shape: [1], stride: [1] },
            o: TensorMut { data: fx.o.as_slice_mut(), shape: qo_shape, stride: contiguous_stride(qo_shape) },
            lse: TensorMut { data: fx.lse.as_slice_mut(), shape: [qo_len as i32, 1], stride: contiguous_stride([qo_len as i32, 1]) },
        },
    )
    .expect("prefill run");
    stream.synchronize().expect("sync");

    let mut o_host = vec![f16::ZERO; qo_len * HEAD_DIM];
    fx.o.copy_to_host(&mut o_host).expect("download o");
    // Return the first element of each query row (all dims share a value).
    (0..qo_len).map(|r| o_host[r * HEAD_DIM].to_f32()).collect()
}

/// Non-causal: every query attends all KV → output = mean(V) for all rows.
#[test]
#[ignore]
fn paged_prefill_uniform_key_non_causal_is_mean() {
    let (ctx, stream) = setup();
    let v_vals = [1.0f32, 3.0];
    let mean = (v_vals[0] + v_vals[1]) / 2.0; // 2.0
    let mut fx = make_fixture(&ctx, /*qo_len=*/ 2, /*kv_len=*/ 2, &v_vals);
    let got = run_prefill(&stream, &mut fx, 2, 2, /*causal=*/ false, /*enable_split=*/ false);
    for (r, &g) in got.iter().enumerate() {
        assert!(
            (g - mean).abs() < 3e-2,
            "non-causal row {r}: got {g}, expected mean(V) = {mean}",
        );
    }
}

/// Causal (right-aligned): with qo_len == kv_len, query row i attends
/// KV[0..=i]. With identical keys, row i's output = mean(V[0..=i]).
#[test]
#[ignore]
fn paged_prefill_uniform_key_causal_is_prefix_mean() {
    let (ctx, stream) = setup();
    let v_vals = [1.0f32, 3.0];
    let mut fx = make_fixture(&ctx, /*qo_len=*/ 2, /*kv_len=*/ 2, &v_vals);
    let got = run_prefill(&stream, &mut fx, 2, 2, /*causal=*/ true, /*enable_split=*/ false);
    // row 0 attends KV[0] only → V0; row 1 attends KV[0..=1] → mean.
    assert!((got[0] - 1.0).abs() < 3e-2, "causal row 0: got {}, expected V0 = 1.0", got[0]);
    assert!((got[1] - 2.0).abs() < 3e-2, "causal row 1: got {}, expected mean = 2.0", got[1]);
}

/// KV-split parallelism must produce the SAME result as the no-split path.
/// A single request with a long KV (1 q row, many KV tokens, 1 kv head)
/// is the regime the scheduler splits; uniform keys ⇒ output = mean(V),
/// independent of the split decision. We assert split == no-split == mean.
#[test]
#[ignore]
fn paged_prefill_kv_split_matches_no_split() {
    let (ctx, stream) = setup();
    // 1 query row, 512 KV tokens, identical keys, V[t] = (t % 4) as a value.
    let kv_len = 512usize;
    let v_vals: Vec<f32> = (0..kv_len).map(|t| (t % 4) as f32).collect();
    let mean: f32 = v_vals.iter().sum::<f32>() / kv_len as f32;

    let got_nosplit = {
        let mut fx = make_fixture(&ctx, 1, kv_len, &v_vals);
        run_prefill(&stream, &mut fx, 1, kv_len, /*causal=*/ false, /*enable_split=*/ false)
    };
    let got_split = {
        let mut fx = make_fixture(&ctx, 1, kv_len, &v_vals);
        run_prefill(&stream, &mut fx, 1, kv_len, /*causal=*/ false, /*enable_split=*/ true)
    };
    assert!((got_nosplit[0] - mean).abs() < 5e-2, "no-split: {} vs mean {mean}", got_nosplit[0]);
    assert!((got_split[0] - mean).abs() < 5e-2, "split: {} vs mean {mean}", got_split[0]);
    assert!(
        (got_split[0] - got_nosplit[0]).abs() < 2e-2,
        "split {} must match no-split {}",
        got_split[0], got_nosplit[0],
    );
}
