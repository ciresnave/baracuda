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
    CascadeAttentionArgs, CascadeAttentionDescriptor, CascadeAttentionPlan, CascadeMergeStatesArgs,
    CascadeMergeStatesDescriptor, CascadeMergeStatesPlan,
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

#[test]
#[ignore]
fn cascade_merge_states_select_validates() {
    let (_ctx, stream) = setup();
    let desc = CascadeMergeStatesDescriptor {
        num_index_sets: 3,
        seq_len: 1,
        num_heads: 4,
        head_dim: 128,
        element: ElementKind::F16,
    };
    let _plan = CascadeMergeStatesPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("valid descriptor should select");

    let mut bad = desc;
    bad.head_dim = 96;
    assert!(
        CascadeMergeStatesPlan::<f16>::select(&stream, &bad, PlanPreference::default()).is_err(),
        "head_dim 96 must be rejected",
    );
    let mut bad_fanin = desc;
    bad_fanin.num_index_sets = 0;
    assert!(
        CascadeMergeStatesPlan::<f16>::select(&stream, &bad_fanin, PlanPreference::default())
            .is_err(),
        "num_index_sets 0 must be rejected",
    );
}

/// Many-way merge of N equal-LSE partial states ⇒ arithmetic mean of the
/// N `v` vectors, merged LSE = s + log2(N). Closed-form check of the
/// `MergeStates` path (Phase 66).
#[test]
#[ignore]
fn cascade_merge_states_equal_lse_averages() {
    let (ctx, stream) = setup();

    const HEAD_DIM: usize = 128;
    const N: usize = 3; // fan-in
    let desc = CascadeMergeStatesDescriptor {
        num_index_sets: N as i32,
        seq_len: 1,
        num_heads: 1,
        head_dim: HEAD_DIM as i32,
        element: ElementKind::F16,
    };
    let plan = CascadeMergeStatesPlan::<f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");

    // Stacked v: [seq=1, idx=N, heads=1, head_dim]. Index set k = const (k+1).
    let vals = [1.0f32, 2.0, 4.0]; // mean = 7/3
    let mean = vals.iter().sum::<f32>() / N as f32;
    let mut v_h: Vec<f16> = Vec::with_capacity(N * HEAD_DIM);
    for &val in &vals {
        v_h.extend((0..HEAD_DIM).map(|_| f16::from_f32(val)));
    }
    // Stacked s: [seq=1, idx=N, heads=1] all equal base-2 LSE = 0.
    let s_h = vec![0.0f32; N];

    let v_dev = DeviceBuffer::from_slice(&ctx, &v_h).expect("upload v");
    let s_dev = DeviceBuffer::from_slice(&ctx, &s_h).expect("upload s");
    let mut v_merged_dev: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, HEAD_DIM).expect("alloc vm");
    let mut s_merged_dev: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).expect("alloc sm");

    let v_shape = [1, N as i32, 1, HEAD_DIM as i32];
    let s_shape = [1, N as i32, 1];
    let vm_shape = [1, 1, HEAD_DIM as i32];
    let sm_shape = [1, 1];
    plan.run(
        &stream,
        Workspace::None,
        CascadeMergeStatesArgs {
            v: TensorRef { data: v_dev.as_slice(), shape: v_shape, stride: contiguous_stride(v_shape) },
            s: TensorRef { data: s_dev.as_slice(), shape: s_shape, stride: contiguous_stride(s_shape) },
            v_merged: TensorMut { data: v_merged_dev.as_slice_mut(), shape: vm_shape, stride: contiguous_stride(vm_shape) },
            s_merged: TensorMut { data: s_merged_dev.as_slice_mut(), shape: sm_shape, stride: contiguous_stride(sm_shape) },
        },
    )
    .expect("merge_states run");
    stream.synchronize().expect("sync");

    let mut vm_host = vec![f16::ZERO; HEAD_DIM];
    v_merged_dev.copy_to_host(&mut vm_host).expect("download vm");
    let mut sm_host = [0.0f32];
    s_merged_dev.copy_to_host(&mut sm_host).expect("download sm");

    for (i, &got) in vm_host.iter().enumerate() {
        let got = got.to_f32();
        assert!(
            (got - mean).abs() < 3e-2,
            "v_merged[{i}] = {got}, expected mean {mean}",
        );
    }
    let want_s = (N as f32).log2(); // s + log2(N), s = 0
    assert!(
        (sm_host[0] - want_s).abs() < 1e-3,
        "merged LSE = {}, expected log2({N}) = {want_s}",
        sm_host[0],
    );
}
