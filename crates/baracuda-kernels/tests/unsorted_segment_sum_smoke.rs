//! Real-GPU smoke test for `UnsortedSegmentSumPlan<T>` (Phase 7 7.6).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef,
    UnsortedSegmentSumArgs, UnsortedSegmentSumDescriptor, UnsortedSegmentSumPlan, Workspace,
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
fn unsorted_segment_sum_f32_scrambled() {
    let (ctx, stream) = setup();
    let n: i32 = 8;
    let d: i32 = 3;
    let ns: i32 = 4;
    // Unsorted seg ids.
    let seg: Vec<i32> = vec![2, 0, 3, 1, 0, 2, 1, 2];
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
    // Output pre-populated with non-zero to verify the kernel
    // zero-fills.
    let init: Vec<f32> = (0..(ns * d) as usize).map(|i| (i as f32) * 100.0).collect();
    let mut dev_out = DeviceBuffer::from_slice(&ctx, &init).expect("alloc out");

    let desc = UnsortedSegmentSumDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan = UnsortedSegmentSumPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnsortedSegmentSumArgs::<f32> {
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
            "unsorted_segment_sum f32 mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}
