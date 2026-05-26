//! Real-GPU smoke test for `SegmentMinBackwardPlan<T>` (Phase 25).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SegmentMinBackwardArgs,
    SegmentMinBackwardDescriptor, SegmentMinBackwardPlan, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn cpu_segment_min_backward_f32(
    n: usize,
    d: usize,
    ns: usize,
    d_out: &[f32],
    input: &[f32],
    seg: &[i32],
) -> Vec<f32> {
    let mut din = vec![0f32; n * d];
    for col in 0..d {
        let mut s_lo = vec![0usize; ns];
        let mut s_hi = vec![0usize; ns];
        let mut idx = 0;
        for s in 0..ns {
            s_lo[s] = idx;
            while idx < n && seg[idx] as usize == s {
                idx += 1;
            }
            s_hi[s] = idx;
        }
        for s in 0..ns {
            if s_lo[s] >= s_hi[s] {
                continue;
            }
            let mut arg = s_lo[s];
            let mut best = input[arg * d + col];
            for k in (s_lo[s] + 1)..s_hi[s] {
                let v = input[k * d + col];
                if v < best {
                    best = v;
                    arg = k;
                }
            }
            din[arg * d + col] = d_out[s * d + col];
        }
    }
    din
}

#[test]
#[ignore]
fn segment_min_backward_f32_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 6;
    let d: i32 = 2;
    let ns: i32 = 2;
    let seg: Vec<i32> = vec![0, 0, 0, 1, 1, 1];
    let input: Vec<f32> = vec![5.0, 2.0, 1.0, 7.0, 3.0, -1.0, 4.0, 8.0, 6.0, -2.0, 0.0, 5.0];
    let d_out: Vec<f32> = vec![0.5, -1.0, 1.5, 2.0];
    let expected = cpu_segment_min_backward_f32(
        n as usize, d as usize, ns as usize, &d_out, &input, &seg,
    );

    let dev_dout = DeviceBuffer::from_slice(&ctx, &d_out).expect("up dout");
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up in");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_din: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * d) as usize).expect("alloc din");

    let desc = SegmentMinBackwardDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan = SegmentMinBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = SegmentMinBackwardArgs::<f32> {
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
        assert_eq!(g, e, "segment_min_backward f32 mismatch @ {i}: got {g} expected {e}");
    }
}
