//! Real-GPU smoke test for `SegmentSumPlan<T>` (Phase 7 7.6).
//!
//! Sorted-seg-ids `segment_sum`: `out[s, d] = Σ_{n : seg[n]==s} input[n, d]`.
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SegmentSumArgs, SegmentSumDescriptor,
    SegmentSumPlan, TensorMut, TensorRef, Workspace,
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
fn segment_sum_f32_basic() {
    let (ctx, stream) = setup();
    // N=8, D=3, num_segments=4. seg_ids: 0 0 1 1 2 2 2 3 (sorted).
    let n: i32 = 8;
    let d: i32 = 3;
    let ns: i32 = 4;
    let seg: Vec<i32> = vec![0, 0, 1, 1, 2, 2, 2, 3];
    let input: Vec<f32> = (0..(n * d) as usize)
        .map(|i| (i as f32) * 0.5 + 1.0)
        .collect();
    let mut expected = vec![0f32; (ns * d) as usize];
    for row in 0..n as usize {
        let s = seg[row] as usize;
        for col in 0..d as usize {
            expected[s * d as usize + col] += input[row * d as usize + col];
        }
    }
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up in");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (ns * d) as usize).expect("alloc out");

    let desc = SegmentSumDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan =
        SegmentSumPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = SegmentSumArgs::<f32> {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, d],
            stride: contiguous_stride([n, d]),
        },
        segment_ids: TensorRef {
            data: dev_seg.as_slice(),
            shape: [n],
            stride: contiguous_stride([n]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [ns, d],
            stride: contiguous_stride([ns, d]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (ns * d) as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "segment_sum f32 mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}

#[test]
#[ignore]
fn segment_sum_f64_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 6;
    let d: i32 = 2;
    let ns: i32 = 3;
    let seg: Vec<i32> = vec![0, 0, 0, 1, 2, 2];
    let input: Vec<f64> = (0..(n * d) as usize).map(|i| (i as f64) * 0.25).collect();
    let mut expected = vec![0f64; (ns * d) as usize];
    for row in 0..n as usize {
        let s = seg[row] as usize;
        for col in 0..d as usize {
            expected[s * d as usize + col] += input[row * d as usize + col];
        }
    }
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up in");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_out: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (ns * d) as usize).expect("alloc out");

    let desc = SegmentSumDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F64,
    };
    let plan =
        SegmentSumPlan::<f64>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = SegmentSumArgs::<f64> {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, d],
            stride: contiguous_stride([n, d]),
        },
        segment_ids: TensorRef {
            data: dev_seg.as_slice(),
            shape: [n],
            stride: contiguous_stride([n]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [ns, d],
            stride: contiguous_stride([ns, d]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; (ns * d) as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    let eps = f64::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "segment_sum f64 mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}
