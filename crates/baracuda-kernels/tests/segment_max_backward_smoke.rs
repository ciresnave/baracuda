//! Real-GPU smoke test for `SegmentMaxBackwardPlan<T>` (Phase 25).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SegmentMaxBackwardArgs,
    SegmentMaxBackwardDescriptor, SegmentMaxBackwardPlan, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference: first-occurrence argmax tie-break (matches kernel).
fn cpu_segment_max_backward_f32(
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
        // Find segment ranges (assume sorted).
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
                if v > best {
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
fn segment_max_backward_f32_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 7;
    let d: i32 = 2;
    let ns: i32 = 3;
    let seg: Vec<i32> = vec![0, 0, 0, 1, 1, 2, 2];
    let input: Vec<f32> = vec![
        0.0, 5.0, 3.0, -1.0, 1.0, 4.0,    // seg 0
        7.0, 2.0, 6.0, 8.0,                // seg 1
        -2.0, 9.0, 0.5, 0.0,                // seg 2
    ];
    let d_out: Vec<f32> = (0..(ns * d) as usize)
        .map(|i| (i as f32) * 0.5 + 1.0)
        .collect();

    let expected = cpu_segment_max_backward_f32(
        n as usize, d as usize, ns as usize, &d_out, &input, &seg,
    );

    let dev_dout = DeviceBuffer::from_slice(&ctx, &d_out).expect("up dout");
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up in");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_din: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * d) as usize).expect("alloc din");

    let desc = SegmentMaxBackwardDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan = SegmentMaxBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = SegmentMaxBackwardArgs::<f32> {
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
        assert_eq!(g, e, "segment_max_backward f32 mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn segment_max_backward_f64_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 5;
    let d: i32 = 2;
    let ns: i32 = 2;
    let seg: Vec<i32> = vec![0, 0, 0, 1, 1];
    let input_f32: Vec<f32> = vec![1.0, 0.0, 2.0, 5.0, -1.0, 3.0, 4.0, 2.0, -7.0, 1.0];
    let input: Vec<f64> = input_f32.iter().map(|&x| x as f64).collect();
    let d_out: Vec<f64> = vec![0.5, 1.0, 2.0, -3.0];
    let expected_f32 = cpu_segment_max_backward_f32(
        n as usize, d as usize, ns as usize,
        &d_out.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        &input_f32, &seg,
    );
    let expected: Vec<f64> = expected_f32.iter().map(|&x| x as f64).collect();

    let dev_dout = DeviceBuffer::from_slice(&ctx, &d_out).expect("up dout");
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up in");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_din: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (n * d) as usize).expect("alloc din");

    let desc = SegmentMaxBackwardDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F64,
    };
    let plan = SegmentMaxBackwardPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = SegmentMaxBackwardArgs::<f64> {
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

    let mut got = vec![0f64; (n * d) as usize];
    dev_din.copy_to_host(&mut got).expect("dl");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "segment_max_backward f64 mismatch @ {i}: got {g} expected {e}");
    }
}
