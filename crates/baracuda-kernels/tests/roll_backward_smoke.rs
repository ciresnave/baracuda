//! Real-GPU smoke test for `RollBackwardPlan<T, N>` — backward of
//! `roll` is `roll` with negated shifts. Bit-exact, no math.
//!
//! Run: `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test roll_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, RollBackwardArgs, RollBackwardDescriptor,
    RollBackwardPlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// CPU reference: `dx[c] = dy[(c - shifts) mod shape]` per axis — same
// as forward roll because BW negates shifts and rolling by -(-s) = +s.
// Apply NEG-shifts here in the ref (mirrors what the plan does).
fn cpu_roll_ref<const N: usize, T: Copy>(
    src: &[T],
    shape: [i32; N],
    shifts: [i32; N],
) -> Vec<T> {
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let mut out = Vec::with_capacity(numel);
    let mut src_stride = [1usize; N];
    for d in (0..N).rev().skip(1) {
        src_stride[d] = src_stride[d + 1] * shape[d + 1] as usize;
    }
    for linear in 0..numel {
        let mut coord = [0i32; N];
        let mut rem = linear;
        for d in (0..N).rev() {
            coord[d] = (rem % shape[d] as usize) as i32;
            rem /= shape[d] as usize;
        }
        let mut src_idx = 0usize;
        for d in 0..N {
            let extent = shape[d] as i64;
            let s = shifts[d] as i64;
            let raw = (coord[d] as i64 - s) % extent;
            let cm = if raw < 0 { raw + extent } else { raw };
            src_idx += cm as usize * src_stride[d];
        }
        out.push(src[src_idx]);
    }
    out
}

#[test]
#[ignore]
fn roll_backward_f32_1d() {
    let (ctx, stream) = setup();
    let shape = [16i32];
    let shifts = [3i32];
    let host_dy: Vec<f32> = (0..16).map(|i| (i as f32) - 7.0).collect();
    // BW = forward with negated shifts.
    let neg = [-shifts[0]];
    let expected = cpu_roll_ref(&host_dy, shape, neg);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 16).expect("alloc");
    let desc = RollBackwardDescriptor {
        shape,
        shifts,
        element: ElementKind::F32,
    };
    let plan = RollBackwardPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = RollBackwardArgs::<f32, 1> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 16];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32 roll BW mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn roll_backward_f32_2d_negative_shift() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let shifts = [-2i32, 3];
    let host_dy: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5 - 5.0).collect();
    let neg = [-shifts[0], -shifts[1]];
    let expected = cpu_roll_ref(&host_dy, shape, neg);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 32).expect("alloc");
    let desc = RollBackwardDescriptor {
        shape,
        shifts,
        element: ElementKind::F32,
    };
    let plan = RollBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = RollBackwardArgs::<f32, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 32];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32 roll BW 2d mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn roll_backward_f16_2d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let shifts = [1i32, -3];
    let host_dy: Vec<f16> = (0..32)
        .map(|i| f16::from_f32((i as f32) * 0.25 - 4.0))
        .collect();
    let neg = [-shifts[0], -shifts[1]];
    let expected = cpu_roll_ref(&host_dy, shape, neg);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 32).expect("alloc");
    let desc = RollBackwardDescriptor {
        shape,
        shifts,
        element: ElementKind::F16,
    };
    let plan = RollBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = RollBackwardArgs::<f16, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::ZERO; 32];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f16 roll BW mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn roll_backward_bf16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 7];
    let shifts = [2i32, -1];
    let host_dy: Vec<bf16> = (0..21)
        .map(|i| bf16::from_f32((i as f32) * 0.5 - 5.0))
        .collect();
    let neg = [-shifts[0], -shifts[1]];
    let expected = cpu_roll_ref(&host_dy, shape, neg);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 21).expect("alloc");
    let desc = RollBackwardDescriptor {
        shape,
        shifts,
        element: ElementKind::Bf16,
    };
    let plan = RollBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = RollBackwardArgs::<bf16, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::ZERO; 21];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "bf16 roll BW mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn roll_backward_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 5];
    let shifts = [1i32, -2, 4];
    let host_dy: Vec<f64> = (0..30).map(|i| (i as f64) * 0.125 - 1.0).collect();
    let neg = [-shifts[0], -shifts[1], -shifts[2]];
    let expected = cpu_roll_ref(&host_dy, shape, neg);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 30).expect("alloc");
    let desc = RollBackwardDescriptor {
        shape,
        shifts,
        element: ElementKind::F64,
    };
    let plan = RollBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = RollBackwardArgs::<f64, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; 30];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f64 roll BW mismatch @ {i}");
    }
}
