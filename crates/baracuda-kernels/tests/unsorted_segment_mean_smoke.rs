//! Real-GPU smoke test for `UnsortedSegmentMeanPlan<T>` (Phase 7 7.6).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef,
    UnsortedSegmentMeanArgs, UnsortedSegmentMeanDescriptor, UnsortedSegmentMeanPlan, Workspace,
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
fn unsorted_segment_mean_f32_scrambled() {
    let (ctx, stream) = setup();
    let n: i32 = 8;
    let d: i32 = 3;
    let ns: i32 = 4;
    let seg: Vec<i32> = vec![2, 0, 3, 1, 0, 2, 1, 2];
    let input: Vec<f32> = (0..(n * d) as usize).map(|i| (i as f32) * 0.5).collect();
    let mut counts = vec![0i32; ns as usize];
    let mut sums = vec![0f32; (ns * d) as usize];
    for row in 0..n as usize {
        let s = seg[row] as usize;
        counts[s] += 1;
        for col in 0..d as usize {
            sums[s * d as usize + col] += input[row * d as usize + col];
        }
    }
    let expected: Vec<f32> = (0..(ns * d) as usize)
        .map(|i| {
            let s = i / d as usize;
            if counts[s] > 0 {
                sums[i] / counts[s] as f32
            } else {
                0.0
            }
        })
        .collect();

    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up in");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (ns * d) as usize).expect("alloc out");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, (ns as usize) * core::mem::size_of::<i32>())
            .expect("alloc ws");

    let desc = UnsortedSegmentMeanDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan = UnsortedSegmentMeanPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnsortedSegmentMeanArgs::<f32> {
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
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; (ns * d) as usize];
    dev_out.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 8.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "unsorted_segment_mean f32 mismatch @ {i}: got {g} expected {e} tol {tol}"
        );
    }
}
