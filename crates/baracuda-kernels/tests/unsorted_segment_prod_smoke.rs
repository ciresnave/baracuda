//! Real-GPU smoke test for `UnsortedSegmentProdPlan<T>` (Phase 25).
//! `#[ignore]` by default.
//!
//! Validates atomicCAS-based prod FW. Uses small inputs so the multi-
//! threaded multiply is well-conditioned; non-determinism across runs
//! is tolerated by comparing to a small absolute tolerance.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef,
    UnsortedSegmentProdArgs, UnsortedSegmentProdDescriptor, UnsortedSegmentProdPlan, Workspace,
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
fn unsorted_segment_prod_f32_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 6;
    let d: i32 = 2;
    let ns: i32 = 3;
    // Unsorted seg ids.
    let seg: Vec<i32> = vec![1, 0, 2, 0, 1, 2];
    let input: Vec<f32> = vec![2.0, 0.5, 1.5, -1.0, 4.0, 1.0, 0.5, 3.0, -2.0, 2.0, 1.0, 0.5];
    let mut expected = vec![1f32; (ns * d) as usize];
    for k in 0..n as usize {
        let s = seg[k] as usize;
        for col in 0..d as usize {
            expected[s * d as usize + col] *= input[k * d as usize + col];
        }
    }

    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up in");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (ns * d) as usize).expect("alloc out");

    let desc = UnsortedSegmentProdDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan = UnsortedSegmentProdPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnsortedSegmentProdArgs::<f32> {
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
        // atomicCAS retry → multi-thread floating-point ordering may
        // diverge slightly from a single-threaded CPU ref. Tolerance
        // includes a ~per-segment scale factor.
        let tol = 64.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "unsorted_segment_prod f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn unsorted_segment_prod_f64_basic() {
    let (ctx, stream) = setup();
    let n: i32 = 4;
    let d: i32 = 1;
    let ns: i32 = 2;
    let seg: Vec<i32> = vec![1, 0, 0, 1];
    let input: Vec<f64> = vec![3.0, 2.5, -1.5, 2.0];
    let mut expected = vec![1f64; (ns * d) as usize];
    for k in 0..n as usize {
        let s = seg[k] as usize;
        for col in 0..d as usize {
            expected[s * d as usize + col] *= input[k * d as usize + col];
        }
    }

    let dev_in = DeviceBuffer::from_slice(&ctx, &input).expect("up in");
    let dev_seg = DeviceBuffer::from_slice(&ctx, &seg).expect("up seg");
    let mut dev_out: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (ns * d) as usize).expect("alloc out");

    let desc = UnsortedSegmentProdDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F64,
    };
    let plan = UnsortedSegmentProdPlan::<f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnsortedSegmentProdArgs::<f64> {
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
        let tol = 64.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "unsorted_segment_prod f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
