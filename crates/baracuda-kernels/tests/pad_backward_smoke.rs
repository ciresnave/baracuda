//! Real-GPU smoke test for `PadBackwardPlan<T, N>` — backward of
//! `pad(mode=Constant)` is a pure slice (`dx = dy[pad_low :
//! pad_low + input_shape]`). Bit-exact, no math.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test pad_backward_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PadBackwardArgs, PadBackwardDescriptor, PadBackwardPlan,
    PadMode, PlanPreference, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

// CPU reference: `dx[c] = dy[c + pad_low]` per axis.
fn cpu_slice_ref<const N: usize, T: Copy>(
    dy: &[T],
    input_shape: [i32; N],
    pad_low: [i32; N],
    dy_shape: [i32; N],
) -> Vec<T> {
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dy_stride = [1usize; N];
    for d in (0..N).rev().skip(1) {
        dy_stride[d] = dy_stride[d + 1] * dy_shape[d + 1] as usize;
    }
    let mut out = Vec::with_capacity(in_numel);
    for linear in 0..in_numel {
        let mut coord = [0i32; N];
        let mut rem = linear;
        for d in (0..N).rev() {
            coord[d] = (rem % input_shape[d] as usize) as i32;
            rem /= input_shape[d] as usize;
        }
        let mut dy_idx = 0usize;
        for d in 0..N {
            dy_idx += (coord[d] + pad_low[d]) as usize * dy_stride[d];
        }
        out.push(dy[dy_idx]);
    }
    out
}

// -------- f32 --------

#[test]
#[ignore]
fn pad_backward_f32_2d() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 5];
    let pad_low = [1i32, 2];
    let pad_high = [2i32, 3];
    let dy_shape = [
        input_shape[0] + pad_low[0] + pad_high[0],
        input_shape[1] + pad_low[1] + pad_high[1],
    ];
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| (i as f32) * 0.5 - 10.0).collect();
    let expected = cpu_slice_ref(&host_dy, input_shape, pad_low, dy_shape);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = PadBackwardDescriptor {
        mode: PadMode::Constant,
        input_shape,
        pad_low,
        pad_high,
        element: ElementKind::F32,
    };
    let plan =
        PadBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = PadBackwardArgs::<f32, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32 pad BW mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn pad_backward_f32_3d_zero_pad_is_copy() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 3, 5];
    let pad_low = [0i32, 0, 0];
    let pad_high = [0i32, 0, 0];
    let dy_shape = input_shape;
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f32> = (0..dy_numel).map(|i| (i as f32) * 0.1).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, dy_numel).expect("alloc");
    let desc = PadBackwardDescriptor {
        mode: PadMode::Constant,
        input_shape,
        pad_low,
        pad_high,
        element: ElementKind::F32,
    };
    let plan =
        PadBackwardPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = PadBackwardArgs::<f32, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; dy_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_dy.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32 pad BW zero-pad mismatch @ {i}");
    }
}

// -------- f16 --------

#[test]
#[ignore]
fn pad_backward_f16_2d() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 6];
    let pad_low = [2i32, 1];
    let pad_high = [1i32, 2];
    let dy_shape = [
        input_shape[0] + pad_low[0] + pad_high[0],
        input_shape[1] + pad_low[1] + pad_high[1],
    ];
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f16> = (0..dy_numel)
        .map(|i| f16::from_f32((i as f32) * 0.25 - 5.0))
        .collect();
    let expected = cpu_slice_ref(&host_dy, input_shape, pad_low, dy_shape);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dev_dx: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = PadBackwardDescriptor {
        mode: PadMode::Constant,
        input_shape,
        pad_low,
        pad_high,
        element: ElementKind::F16,
    };
    let plan =
        PadBackwardPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = PadBackwardArgs::<f16, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::ZERO; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f16 pad BW mismatch @ {i}");
    }
}

// -------- bf16 --------

#[test]
#[ignore]
fn pad_backward_bf16_2d() {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4];
    let pad_low = [1i32, 1];
    let pad_high = [2i32, 1];
    let dy_shape = [
        input_shape[0] + pad_low[0] + pad_high[0],
        input_shape[1] + pad_low[1] + pad_high[1],
    ];
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<bf16> = (0..dy_numel)
        .map(|i| bf16::from_f32((i as f32) * 0.5 - 3.0))
        .collect();
    let expected = cpu_slice_ref(&host_dy, input_shape, pad_low, dy_shape);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dev_dx: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = PadBackwardDescriptor {
        mode: PadMode::Constant,
        input_shape,
        pad_low,
        pad_high,
        element: ElementKind::Bf16,
    };
    let plan =
        PadBackwardPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = PadBackwardArgs::<bf16, 2> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::ZERO; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "bf16 pad BW mismatch @ {i}");
    }
}

// -------- f64 --------

#[test]
#[ignore]
fn pad_backward_f64_3d() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 3, 4];
    let pad_low = [1i32, 0, 2];
    let pad_high = [0i32, 2, 1];
    let dy_shape = [
        input_shape[0] + pad_low[0] + pad_high[0],
        input_shape[1] + pad_low[1] + pad_high[1],
        input_shape[2] + pad_low[2] + pad_high[2],
    ];
    let dy_numel: usize = dy_shape.iter().map(|&d| d as usize).product();
    let host_dy: Vec<f64> = (0..dy_numel).map(|i| (i as f64) * 0.125 - 1.0).collect();
    let expected = cpu_slice_ref(&host_dy, input_shape, pad_low, dy_shape);

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload");
    let dx_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, dx_numel).expect("alloc");
    let desc = PadBackwardDescriptor {
        mode: PadMode::Constant,
        input_shape,
        pad_low,
        pad_high,
        element: ElementKind::F64,
    };
    let plan =
        PadBackwardPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default()).expect("sel");
    let args = PadBackwardArgs::<f64, 3> {
        dy: TensorRef {
            data: dev_dy.as_slice(),
            shape: dy_shape,
            stride: contiguous_stride(dy_shape),
        },
        dx: TensorMut {
            data: dev_dx.as_slice_mut(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; dx_numel];
    dev_dx.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f64 pad BW mismatch @ {i}");
    }
}

// Validation guard — reject non-constant modes.
#[test]
#[ignore]
fn pad_backward_rejects_non_constant_mode() {
    let (_ctx, stream) = setup();
    let desc = PadBackwardDescriptor {
        mode: PadMode::Reflect,
        input_shape: [4i32, 4],
        pad_low: [1i32, 1],
        pad_high: [1i32, 1],
        element: ElementKind::F32,
    };
    let res = PadBackwardPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default());
    assert!(res.is_err(), "Reflect-mode BW select should fail");
}
