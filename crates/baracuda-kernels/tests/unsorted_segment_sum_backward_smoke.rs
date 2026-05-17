//! Real-GPU smoke test for `UnsortedSegmentSumBackwardPlan<T>`
//! (Phase 7 7.6). `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef,
    UnsortedSegmentSumBackwardArgs, UnsortedSegmentSumBackwardDescriptor,
    UnsortedSegmentSumBackwardPlan, Workspace,
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
fn unsorted_segment_sum_backward_f32_scrambled() {
    let (ctx, stream) = setup();
    let n: i32 = 8;
    let d: i32 = 3;
    let ns: i32 = 4;
    let seg: Vec<i32> = vec![2, 0, 3, 1, 0, 2, 1, 2];
    let d_out: Vec<f32> = (0..(ns * d) as usize).map(|i| (i as f32) * 0.5).collect();
    let mut expected = vec![0f32; (n * d) as usize];
    for row in 0..n as usize {
        let s = seg[row] as usize;
        for col in 0..d as usize {
            expected[row * d as usize + col] = d_out[s * d as usize + col];
        }
    }
    let dev_dout = DeviceBuffer::from_slice(&ctx, &d_out).expect("up dout");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_dinput: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * d) as usize).expect("alloc din");

    let desc = UnsortedSegmentSumBackwardDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan = UnsortedSegmentSumBackwardPlan::<f32>::select(
        &stream,
        &desc,
        PlanPreference::default(),
    )
    .expect("select");
    let args = UnsortedSegmentSumBackwardArgs::<f32> {
        d_output: TensorRef {
            data: dev_dout.as_slice(),
            shape: [ns, d],
            stride: contiguous_stride([ns, d]),
        },
        segment_ids: TensorRef {
            data: dev_seg.as_slice(),
            shape: [n],
            stride: contiguous_stride([n]),
        },
        d_input: TensorMut {
            data: dev_dinput.as_slice_mut(),
            shape: [n, d],
            stride: contiguous_stride([n, d]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (n * d) as usize];
    dev_dinput.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g, e,
            "unsorted_segment_sum_backward f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
