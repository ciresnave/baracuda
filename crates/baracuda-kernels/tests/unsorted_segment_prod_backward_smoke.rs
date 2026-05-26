//! Real-GPU smoke test for `UnsortedSegmentProdBackwardPlan<T>` (Phase 25).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef,
    UnsortedSegmentProdBackwardArgs, UnsortedSegmentProdBackwardDescriptor,
    UnsortedSegmentProdBackwardPlan, Workspace,
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
fn unsorted_segment_prod_backward_f32_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 6;
    let d: i32 = 2;
    let ns: i32 = 3;
    let seg: Vec<i32> = vec![1, 0, 2, 0, 1, 2];
    let input: Vec<f32> = vec![2.0, 0.5, 1.5, -1.0, 4.0, 1.0, 0.5, 3.0, -2.0, 2.0, 1.0, 0.5];
    let d_out: Vec<f32> = vec![0.5, -1.0, 1.5, 2.0, 0.25, -0.5];

    let mut fw_out = vec![1f32; (ns * d) as usize];
    for k in 0..n as usize {
        let s = seg[k] as usize;
        for col in 0..d as usize {
            fw_out[s * d as usize + col] *= input[k * d as usize + col];
        }
    }
    let mut expected = vec![0f32; (n * d) as usize];
    for k in 0..n as usize {
        let s = seg[k] as usize;
        for col in 0..d as usize {
            let i = k * d as usize + col;
            let o = s * d as usize + col;
            expected[i] = d_out[o] * (fw_out[o] / input[i]);
        }
    }

    let dev_dout = DeviceBuffer::from_slice(&ctx, &d_out).expect("up dout");
    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up in");
    let dev_out = DeviceBuffer::from_slice(&ctx, &fw_out).expect("up output");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_din: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * d) as usize).expect("alloc din");

    let desc = UnsortedSegmentProdBackwardDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan = UnsortedSegmentProdBackwardPlan::<f32>::select(
        &stream, &desc, PlanPreference::default(),
    )
    .expect("select");
    let args = UnsortedSegmentProdBackwardArgs::<f32> {
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
        output: TensorRef {
            data: dev_out.as_slice(),
            shape: [ns, d],
            stride: contiguous_stride([ns, d]),
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
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 32.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "unsorted_segment_prod_backward f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
