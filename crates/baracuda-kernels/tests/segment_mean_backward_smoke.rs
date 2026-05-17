//! Real-GPU smoke test for `SegmentMeanBackwardPlan<T>` (Phase 7 7.6).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, SegmentMeanBackwardArgs,
    SegmentMeanBackwardDescriptor, SegmentMeanBackwardPlan, TensorMut, TensorRef, Workspace,
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
fn segment_mean_backward_f32_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 8;
    let d: i32 = 3;
    let ns: i32 = 4;
    let seg: Vec<i32> = vec![0, 0, 1, 1, 2, 2, 2, 3];
    let d_out: Vec<f32> = (0..(ns * d) as usize).map(|i| (i as f32) + 1.0).collect();
    // counts: [2, 2, 3, 1].
    let mut counts = vec![0i32; ns as usize];
    for &s in &seg {
        counts[s as usize] += 1;
    }
    let mut expected = vec![0f32; (n * d) as usize];
    for row in 0..n as usize {
        let s = seg[row] as usize;
        for col in 0..d as usize {
            expected[row * d as usize + col] =
                d_out[s * d as usize + col] / counts[s] as f32;
        }
    }
    let dev_dout = DeviceBuffer::from_slice(&ctx, &d_out).expect("up dout");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_dinput: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * d) as usize).expect("alloc din");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, (ns as usize) * core::mem::size_of::<i32>())
            .expect("alloc ws");

    let desc = SegmentMeanBackwardDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan = SegmentMeanBackwardPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = SegmentMeanBackwardArgs::<f32> {
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
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (n * d) as usize];
    dev_dinput.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "segment_mean_backward f32 mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}
