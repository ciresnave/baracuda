//! Real-GPU smoke tests for Triu / Tril plans — Phase 13.4.
//!
//! Both ops are differentiable (`d_input = triu(d_output, diagonal)`,
//! `d_input = tril(d_output, diagonal)`); the backward plans reuse the
//! forward kernel, so the BW smokes are just the FW smokes with the
//! buffers renamed.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test triu_tril_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, TrilArgs,
    TrilBackwardArgs, TrilBackwardDescriptor, TrilBackwardPlan, TrilDescriptor, TrilPlan,
    TriuArgs, TriuBackwardArgs, TriuBackwardDescriptor, TriuBackwardPlan, TriuDescriptor,
    TriuPlan, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference: triu mask on a contiguous row-major tensor whose last
/// two dims are (M, N). The batch prefix is collapsed.
fn triu_ref_f32(input: &[f32], m: i32, n: i32, batch: i32, diagonal: i32) -> Vec<f32> {
    let mn = (m * n) as usize;
    let mut out = vec![0f32; input.len()];
    for b in 0..batch {
        let base = (b as usize) * mn;
        for i in 0..m {
            for j in 0..n {
                let idx = base + (i * n + j) as usize;
                out[idx] = if j >= i + diagonal { input[idx] } else { 0.0 };
            }
        }
    }
    out
}

fn tril_ref_f32(input: &[f32], m: i32, n: i32, batch: i32, diagonal: i32) -> Vec<f32> {
    let mn = (m * n) as usize;
    let mut out = vec![0f32; input.len()];
    for b in 0..batch {
        let base = (b as usize) * mn;
        for i in 0..m {
            for j in 0..n {
                let idx = base + (i * n + j) as usize;
                out[idx] = if j <= i + diagonal { input[idx] } else { 0.0 };
            }
        }
    }
    out
}

fn triu_ref_i32(input: &[i32], m: i32, n: i32, batch: i32, diagonal: i32) -> Vec<i32> {
    let mn = (m * n) as usize;
    let mut out = vec![0i32; input.len()];
    for b in 0..batch {
        let base = (b as usize) * mn;
        for i in 0..m {
            for j in 0..n {
                let idx = base + (i * n + j) as usize;
                out[idx] = if j >= i + diagonal { input[idx] } else { 0 };
            }
        }
    }
    out
}

#[test]
#[ignore]
fn triu_f32_3x3_diagonal_0() {
    let (ctx, stream) = setup();
    let shape = [3i32, 3];
    let host_x: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let expected = triu_ref_f32(&host_x, 3, 3, 1, 0);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 9).expect("alloc y");

    let desc = TriuDescriptor {
        shape,
        diagonal: 0,
        element: ElementKind::F32,
    };
    let plan =
        TriuPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TriuArgs::<f32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 9];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "triu_f32_3x3_d0 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn triu_f32_3x3_diagonal_1() {
    let (ctx, stream) = setup();
    let shape = [3i32, 3];
    let host_x: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let expected = triu_ref_f32(&host_x, 3, 3, 1, 1);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 9).expect("alloc y");

    let desc = TriuDescriptor {
        shape,
        diagonal: 1,
        element: ElementKind::F32,
    };
    let plan =
        TriuPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TriuArgs::<f32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 9];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "triu_f32_3x3_d1 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn triu_f32_3x4_diagonal_neg1() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4];
    let host_x: Vec<f32> = (1..=12).map(|i| i as f32).collect();
    let expected = triu_ref_f32(&host_x, 3, 4, 1, -1);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 12).expect("alloc y");

    let desc = TriuDescriptor {
        shape,
        diagonal: -1,
        element: ElementKind::F32,
    };
    let plan =
        TriuPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TriuArgs::<f32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 12];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "triu_f32_3x4_dneg1 mismatch @ {i}"
        );
    }
}

#[test]
#[ignore]
fn tril_f32_3x3_diagonal_0() {
    let (ctx, stream) = setup();
    let shape = [3i32, 3];
    let host_x: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let expected = tril_ref_f32(&host_x, 3, 3, 1, 0);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 9).expect("alloc y");

    let desc = TrilDescriptor {
        shape,
        diagonal: 0,
        element: ElementKind::F32,
    };
    let plan =
        TrilPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TrilArgs::<f32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 9];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "tril_f32_3x3_d0 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn tril_f32_3x3_diagonal_neg2() {
    let (ctx, stream) = setup();
    let shape = [3i32, 3];
    let host_x: Vec<f32> = (1..=9).map(|i| i as f32).collect();
    let expected = tril_ref_f32(&host_x, 3, 3, 1, -2);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 9).expect("alloc y");

    let desc = TrilDescriptor {
        shape,
        diagonal: -2,
        element: ElementKind::F32,
    };
    let plan =
        TrilPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TrilArgs::<f32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 9];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "tril_f32_3x3_dneg2 mismatch @ {i}"
        );
    }
}

#[test]
#[ignore]
fn triu_batched_f32_2x3x4_diagonal_0() {
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 4];
    let host_x: Vec<f32> = (0..24).map(|i| (i as f32) + 1.0).collect();
    let expected = triu_ref_f32(&host_x, 3, 4, 2, 0);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 24).expect("alloc y");

    let desc = TriuDescriptor {
        shape,
        diagonal: 0,
        element: ElementKind::F32,
    };
    let plan =
        TriuPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TriuArgs::<f32, 3> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 24];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "triu_batched mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn triu_bw_f32_3x3_diagonal_0() {
    let (ctx, stream) = setup();
    let shape = [3i32, 3];
    // Upstream gradient — arbitrary values; the BW just re-masks.
    let host_dy: Vec<f32> = (1..=9).map(|i| (i as f32) * 0.5).collect();
    let expected = triu_ref_f32(&host_dy, 3, 3, 1, 0);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 9).expect("alloc dx");

    let desc = TriuBackwardDescriptor {
        shape,
        diagonal: 0,
        element: ElementKind::F32,
    };
    let plan =
        TriuBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    let args = TriuBackwardArgs::<f32, 2> {
        grad_output: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        grad_input: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 9];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "triu_bw mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn tril_bw_f32_3x3_diagonal_neg1() {
    let (ctx, stream) = setup();
    let shape = [3i32, 3];
    let host_dy: Vec<f32> = (1..=9).map(|i| (i as f32) * 0.25).collect();
    let expected = tril_ref_f32(&host_dy, 3, 3, 1, -1);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 9).expect("alloc dx");

    let desc = TrilBackwardDescriptor {
        shape,
        diagonal: -1,
        element: ElementKind::F32,
    };
    let plan =
        TrilBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    let args = TrilBackwardArgs::<f32, 2> {
        grad_output: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        grad_input: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 9];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "tril_bw mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn triu_i32_3x3_diagonal_0() {
    let (ctx, stream) = setup();
    let shape = [3i32, 3];
    let host_x: Vec<i32> = (1..=9).collect();
    let expected = triu_ref_i32(&host_x, 3, 3, 1, 0);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, 9).expect("alloc y");

    let desc = TriuDescriptor {
        shape,
        diagonal: 0,
        element: ElementKind::I32,
    };
    let plan =
        TriuPlan::<i32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = TriuArgs::<i32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; 9];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "triu_i32 mismatch @ {i}");
    }
}
