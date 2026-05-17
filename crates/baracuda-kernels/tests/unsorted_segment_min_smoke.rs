//! Real-GPU smoke test for `UnsortedSegmentMinPlan<T>` (Phase 7 7.6).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef,
    UnsortedSegmentMinArgs, UnsortedSegmentMinDescriptor, UnsortedSegmentMinPlan, Workspace,
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
fn unsorted_segment_min_f32_scrambled() {
    let (ctx, stream) = setup();
    let n: i32 = 6;
    let d: i32 = 2;
    let ns: i32 = 3;
    let seg: Vec<i32> = vec![2, 0, 1, 0, 2, 1];
    let input: Vec<f32> = vec![
        1.0, -2.0, 3.0, 4.0, 5.0, 1.0, 7.0, 0.0, 2.0, 9.0, -1.0, 5.0,
    ];
    let mut expected = vec![f32::INFINITY; (ns * d) as usize];
    for row in 0..n as usize {
        let s = seg[row] as usize;
        for col in 0..d as usize {
            let v = input[row * d as usize + col];
            let idx = s * d as usize + col;
            if v < expected[idx] {
                expected[idx] = v;
            }
        }
    }
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up in");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (ns * d) as usize).expect("alloc out");

    let desc = UnsortedSegmentMinDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan = UnsortedSegmentMinPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnsortedSegmentMinArgs::<f32> {
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
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g, e,
            "unsorted_segment_min f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
