//! Real-GPU smoke test for the Pad trailblazer
//! (`PadPlan<f32, N> + PadMode::Constant`).
//!
//! Covers 1D / 2D / 3D constant-pad cases plus a "zero padding"
//! identity case (pad amounts all 0 → output == input). Bit-exact
//! against host reference — pad does no arithmetic.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test pad_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PadArgs, PadDescriptor, PadMode, PadPlan, PlanPreference,
    TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Host reference: write `value` to every output cell whose input
/// coord (after subtracting pad_low) falls outside [0, input_shape).
fn cpu_pad_constant<const N: usize>(
    x_host: &[f32],
    input_shape: [i32; N],
    pad_low: [i32; N],
    output_shape: [i32; N],
    value: f32,
) -> Vec<f32> {
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut out = vec![value; out_numel];
    let in_strides = contiguous_stride(input_shape);
    let out_strides = contiguous_stride(output_shape);
    // Walk every output cell.
    for i in 0..out_numel {
        let mut linear = i as i64;
        let mut coord = [0i64; N];
        for d in (0..N).rev() {
            let s = output_shape[d] as i64;
            coord[d] = if s == 0 { 0 } else { linear % s };
            if s != 0 {
                linear /= s;
            }
        }
        // Translate to input coord.
        let mut in_input = true;
        let mut in_coord = [0i64; N];
        for d in 0..N {
            let c = coord[d] - pad_low[d] as i64;
            if c < 0 || c >= input_shape[d] as i64 {
                in_input = false;
                break;
            }
            in_coord[d] = c;
        }
        if in_input {
            // Compute input linear offset (assuming contig input).
            let mut in_off: i64 = 0;
            for d in 0..N {
                in_off += in_coord[d] * in_strides[d];
            }
            let mut out_off: i64 = 0;
            for d in 0..N {
                out_off += coord[d] * out_strides[d];
            }
            out[out_off as usize] = x_host[in_off as usize];
        }
    }
    out
}

#[test]
#[ignore]
fn pad_constant_1d() {
    let (ctx, stream) = setup();
    let input_shape = [16i32];
    let pad_low = [3i32];
    let pad_high = [5i32];
    let value: f32 = -1.0;
    let output_shape = [input_shape[0] + pad_low[0] + pad_high[0]];

    let x: Vec<f32> = (0..16).map(|i| (i as f32) + 100.0).collect();
    let expected = cpu_pad_constant::<1>(&x, input_shape, pad_low, output_shape, value);

    let dev_x = DeviceBuffer::from_slice(&ctx, &x).expect("upload x");
    let out_numel = output_shape[0] as usize;
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = PadDescriptor {
        mode: PadMode::Constant,
        input_shape,
        pad_low,
        pad_high,
        value,
        element: ElementKind::F32,
    };
    let plan = PadPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = PadArgs::<f32, 1> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "1d pad mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn pad_constant_2d() {
    let (ctx, stream) = setup();
    let input_shape = [8i32, 12];
    let pad_low = [1, 2];
    let pad_high = [3, 4];
    let value: f32 = 0.0;
    let output_shape = [
        input_shape[0] + pad_low[0] + pad_high[0],
        input_shape[1] + pad_low[1] + pad_high[1],
    ];

    let in_numel = (input_shape[0] * input_shape[1]) as usize;
    let x: Vec<f32> = (0..in_numel).map(|i| (i as f32) * 0.5 - 2.0).collect();
    let expected = cpu_pad_constant::<2>(&x, input_shape, pad_low, output_shape, value);

    let dev_x = DeviceBuffer::from_slice(&ctx, &x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = PadDescriptor {
        mode: PadMode::Constant,
        input_shape,
        pad_low,
        pad_high,
        value,
        element: ElementKind::F32,
    };
    let plan = PadPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = PadArgs::<f32, 2> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "2d pad mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn pad_constant_3d() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 12];
    let pad_low = [1, 2, 1];
    let pad_high = [2, 1, 3];
    let value: f32 = 99.0;
    let output_shape = [
        input_shape[0] + pad_low[0] + pad_high[0],
        input_shape[1] + pad_low[1] + pad_high[1],
        input_shape[2] + pad_low[2] + pad_high[2],
    ];

    let in_numel = (input_shape[0] * input_shape[1] * input_shape[2]) as usize;
    let x: Vec<f32> = (0..in_numel).map(|i| (i as f32) * 0.125).collect();
    let expected = cpu_pad_constant::<3>(&x, input_shape, pad_low, output_shape, value);

    let dev_x = DeviceBuffer::from_slice(&ctx, &x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = PadDescriptor {
        mode: PadMode::Constant,
        input_shape,
        pad_low,
        pad_high,
        value,
        element: ElementKind::F32,
    };
    let plan = PadPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = PadArgs::<f32, 3> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "3d pad mismatch @ {i}: got {g} expected {e}");
    }
}

#[test]
#[ignore]
fn pad_zero_amounts_is_identity() {
    // pad_low = pad_high = 0 everywhere → output == input, no pad
    // region. Edge case for the pad-region branch.
    let (ctx, stream) = setup();
    let shape = [16i32, 16];
    let pad_low = [0, 0];
    let pad_high = [0, 0];
    let value: f32 = 999.0; // shouldn't appear anywhere

    let in_numel = (shape[0] * shape[1]) as usize;
    let x: Vec<f32> = (0..in_numel).map(|i| (i as f32) * 0.25 - 1.0).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, in_numel).expect("alloc y");

    let desc = PadDescriptor {
        mode: PadMode::Constant,
        input_shape: shape,
        pad_low,
        pad_high,
        value,
        element: ElementKind::F32,
    };
    let plan = PadPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = PadArgs::<f32, 2> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; in_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(x.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "zero-pad identity mismatch @ {i}");
    }
}
