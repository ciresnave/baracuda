//! Real-GPU smoke test for `FlipBackwardPlan<T, N>` — backward of
//! `flip` is the same op applied to `dy`. Bit-exact, no math.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test flip_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, FlipBackwardArgs, FlipBackwardDescriptor, FlipBackwardPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// CPU reference: `dx[c] = dy[c with flipped axes]`. Same formula as the
// forward — flip is involutive.
fn cpu_flip_ref<const N: usize, T: Copy>(
    src: &[T],
    shape: [i32; N],
    flip_axes: [bool; N],
) -> Vec<T> {
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let mut out = Vec::with_capacity(numel);
    let mut stride = [1usize; N];
    for d in (0..N).rev().skip(1) {
        stride[d] = stride[d + 1] * shape[d + 1] as usize;
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
            let c = if flip_axes[d] {
                shape[d] - 1 - coord[d]
            } else {
                coord[d]
            };
            src_idx += c as usize * stride[d];
        }
        out.push(src[src_idx]);
    }
    out
}

// -------- f32 --------

#[test]
#[ignore]
fn flip_backward_f32_2d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let flip_axes = [false, true];
    let host_dy: Vec<f32> = (0..32).map(|i| (i as f32) * 0.5 - 7.0).collect();
    let expected = cpu_flip_ref(&host_dy, shape, flip_axes);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 32).expect("alloc");

    let desc = FlipBackwardDescriptor {
        shape,
        flip_axes,
        element: ElementKind::F32,
    };
    let plan =
        FlipBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = FlipBackwardArgs::<f32, 2> {
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
        assert_eq!(g.to_bits(), e.to_bits(), "f32 flip BW mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn flip_backward_f32_3d_both() {
    let (ctx, stream) = setup();
    let shape = [2i32, 4, 8];
    let flip_axes = [true, false, true];
    let host_dy: Vec<f32> = (0..64).map(|i| (i as f32) * 0.25).collect();
    let expected = cpu_flip_ref(&host_dy, shape, flip_axes);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 64).expect("alloc");
    let desc = FlipBackwardDescriptor {
        shape,
        flip_axes,
        element: ElementKind::F32,
    };
    let plan =
        FlipBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = FlipBackwardArgs::<f32, 3> {
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

    let mut got = vec![0f32; 64];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32 flip BW 3d mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn flip_backward_f32_no_axes_is_copy() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let host_dy: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 32).expect("alloc");
    let desc = FlipBackwardDescriptor {
        shape,
        flip_axes: [false, false],
        element: ElementKind::F32,
    };
    let plan =
        FlipBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = FlipBackwardArgs::<f32, 2> {
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
    for (i, (g, e)) in got.iter().zip(host_dy.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32 flip BW no-axes mismatch @ {i}");
    }
}

// -------- f16 --------

#[test]
#[ignore]
fn flip_backward_f16_2d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let flip_axes = [true, true];
    let host_dy: Vec<f16> = (0..32)
        .map(|i| f16::from_f32((i as f32) * 0.25 - 4.0))
        .collect();
    let expected = cpu_flip_ref(&host_dy, shape, flip_axes);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 32).expect("alloc");
    let desc = FlipBackwardDescriptor {
        shape,
        flip_axes,
        element: ElementKind::F16,
    };
    let plan =
        FlipBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = FlipBackwardArgs::<f16, 2> {
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
        assert_eq!(g.to_bits(), e.to_bits(), "f16 flip BW mismatch @ {i}");
    }
}

// -------- bf16 --------

#[test]
#[ignore]
fn flip_backward_bf16_2d() {
    let (ctx, stream) = setup();
    let shape = [3i32, 7];
    let flip_axes = [true, false];
    let host_dy: Vec<bf16> = (0..21)
        .map(|i| bf16::from_f32((i as f32) * 0.5 - 5.0))
        .collect();
    let expected = cpu_flip_ref(&host_dy, shape, flip_axes);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 21).expect("alloc");
    let desc = FlipBackwardDescriptor {
        shape,
        flip_axes,
        element: ElementKind::Bf16,
    };
    let plan = FlipBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = FlipBackwardArgs::<bf16, 2> {
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
        assert_eq!(g.to_bits(), e.to_bits(), "bf16 flip BW mismatch @ {i}");
    }
}

// -------- f64 --------

#[test]
#[ignore]
fn flip_backward_f64_2d() {
    let (ctx, stream) = setup();
    let shape = [4i32, 8];
    let flip_axes = [true, true];
    let host_dy: Vec<f64> = (0..32).map(|i| (i as f64) * 0.125 - 2.0).collect();
    let expected = cpu_flip_ref(&host_dy, shape, flip_axes);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, 32).expect("alloc");
    let desc = FlipBackwardDescriptor {
        shape,
        flip_axes,
        element: ElementKind::F64,
    };
    let plan =
        FlipBackwardPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = FlipBackwardArgs::<f64, 2> {
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

    let mut got = vec![0f64; 32];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f64 flip BW mismatch @ {i}");
    }
}
