//! Real-GPU smoke test for `EmbeddingBagMaxBackwardPlan<T>` (Phase 25).
//!
//! BW pass: `dweight[output_index[b, d], d] += dout[b, d]` (atomicAdd).
//! Skip cells where `output_index[b, d] < 0`.
//!
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, EmbeddingBagMaxBackwardArgs,
    EmbeddingBagMaxBackwardDescriptor, EmbeddingBagMaxBackwardPlan, PlanPreference, TensorMut,
    TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn cpu_bag_max_bw_f32(
    v: usize,
    d: usize,
    num_bags: usize,
    dout: &[f32],
    out_index: &[i32],
) -> Vec<f32> {
    let mut dw = vec![0f32; v * d];
    for b in 0..num_bags {
        for col in 0..d {
            let row = out_index[b * d + col];
            if row < 0 || row >= v as i32 {
                continue;
            }
            dw[row as usize * d + col] += dout[b * d + col];
        }
    }
    dw
}

#[test]
#[ignore]
fn embedding_bag_max_backward_f32_basic() {
    let (ctx, stream) = setup();
    let v: usize = 5;
    let d: usize = 4;
    let num_bags = 3usize;
    let host_dout: Vec<f32> = (0..(num_bags * d))
        .map(|i| (i as f32) * 0.25 + 1.0)
        .collect();
    // Mixed out_index: real rows + a couple -1 sentinels.
    let host_oidx: Vec<i32> = vec![
        1, 3, 1, 3, // bag 0
        4, 0, 2, 4, // bag 1
        -1, -1, -1, -1, // bag 2 (all empty/padded)
    ];
    let expected = cpu_bag_max_bw_f32(v, d, num_bags, &host_dout, &host_oidx);

    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_oidx = DeviceBuffer::from_slice(&ctx, &host_oidx).expect("up oidx");
    let mut dev_dw: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, v * d).expect("alloc dweight");

    let desc = EmbeddingBagMaxBackwardDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        element: ElementKind::F32,
    };
    let plan = EmbeddingBagMaxBackwardPlan::<f32>::select(
        &stream, &desc, PlanPreference::default(),
    )
    .expect("select");
    let args = EmbeddingBagMaxBackwardArgs::<f32> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
        },
        output_index: TensorRef {
            data: dev_oidx.as_slice(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
        },
        dweight: TensorMut {
            data: dev_dw.as_slice_mut(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; v * d];
    dev_dw.copy_to_host(&mut got).expect("dl");
    let eps = f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 16.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_bag_max_backward f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn embedding_bag_max_backward_f64_basic() {
    let (ctx, stream) = setup();
    let v: usize = 4;
    let d: usize = 3;
    let num_bags = 2usize;
    let host_dout: Vec<f64> = vec![1.5, -0.5, 2.0, 0.25, 3.0, -1.0];
    let host_oidx: Vec<i32> = vec![0, 2, 3, 1, 0, -1];
    let mut expected = vec![0f64; v * d];
    for b in 0..num_bags {
        for col in 0..d {
            let row = host_oidx[b * d + col];
            if row < 0 {
                continue;
            }
            expected[row as usize * d + col] += host_dout[b * d + col];
        }
    }

    let dev_dout = DeviceBuffer::from_slice(&ctx, &host_dout).expect("up dout");
    let dev_oidx = DeviceBuffer::from_slice(&ctx, &host_oidx).expect("up oidx");
    let mut dev_dw: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, v * d).expect("alloc dweight");

    let desc = EmbeddingBagMaxBackwardDescriptor {
        num_embeddings: v as i32,
        embedding_dim: d as i32,
        num_bags: num_bags as i32,
        element: ElementKind::F64,
    };
    let plan = EmbeddingBagMaxBackwardPlan::<f64>::select(
        &stream, &desc, PlanPreference::default(),
    )
    .expect("select");
    let args = EmbeddingBagMaxBackwardArgs::<f64> {
        dout: TensorRef {
            data: dev_dout.as_slice(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
        },
        output_index: TensorRef {
            data: dev_oidx.as_slice(),
            shape: [num_bags as i32, d as i32],
            stride: contiguous_stride([num_bags as i32, d as i32]),
        },
        dweight: TensorMut {
            data: dev_dw.as_slice_mut(),
            shape: [v as i32, d as i32],
            stride: contiguous_stride([v as i32, d as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; v * d];
    dev_dw.copy_to_host(&mut got).expect("dl");
    let eps = f64::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = 16.0 * eps * e.abs().max(1.0);
        assert!(
            (g - e).abs() <= tol,
            "embedding_bag_max_backward f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
