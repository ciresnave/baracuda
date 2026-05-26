//! Real-GPU smoke test for `UnsortedSegmentMaxBackwardPlan<T>` (Phase 25).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef,
    UnsortedSegmentMaxBackwardArgs, UnsortedSegmentMaxBackwardDescriptor,
    UnsortedSegmentMaxBackwardPlan, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference: first-occurrence argmax tie-break across full input.
fn cpu_unsorted_max_bw(
    n: usize, d: usize, ns: usize,
    d_out: &[f32], input: &[f32], seg: &[i32],
) -> Vec<f32> {
    let mut din = vec![0f32; n * d];
    for s in 0..ns {
        for col in 0..d {
            let mut arg: Option<usize> = None;
            let mut best = f32::NEG_INFINITY;
            for m in 0..n {
                if seg[m] as usize != s {
                    continue;
                }
                let v = input[m * d + col];
                if arg.is_none() || v > best {
                    best = v;
                    arg = Some(m);
                }
            }
            if let Some(k) = arg {
                din[k * d + col] = d_out[s * d + col];
            }
        }
    }
    din
}

#[test]
#[ignore]
fn unsorted_segment_max_backward_f32_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 6;
    let d: i32 = 2;
    let ns: i32 = 3;
    // Unsorted seg ids.
    let seg: Vec<i32> = vec![1, 0, 2, 0, 1, 2];
    let input: Vec<f32> = vec![3.0, 7.0, -1.0, 2.0, 5.0, 0.5, 4.0, 1.0, 6.0, 8.0, -2.0, 9.0];
    let d_out: Vec<f32> = vec![0.5, -1.0, 1.5, 2.0, 0.25, -0.5];
    let expected = cpu_unsorted_max_bw(
        n as usize, d as usize, ns as usize, &d_out, &input, &seg,
    );

    let dev_dout = DeviceBuffer::from_slice(&ctx, &d_out).expect("up dout");
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up in");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_din: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * d) as usize).expect("alloc din");

    let desc = UnsortedSegmentMaxBackwardDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan = UnsortedSegmentMaxBackwardPlan::<f32>::select(
        &stream, &desc, PlanPreference::default(),
    )
    .expect("select");
    let args = UnsortedSegmentMaxBackwardArgs::<f32> {
        d_output: TensorRef {
            data: dev_dout.as_slice(),
            shape: [ns, d],
            stride: contiguous_stride([ns, d]),
        },
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
        d_input: TensorMut {
            data: dev_din.as_slice_mut(),
            shape: [n, d],
            stride: contiguous_stride([n, d]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (n * d) as usize];
    dev_din.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "unsorted_segment_max_backward f32 mismatch @ {i}: got {g} expected {e}");
    }
}
