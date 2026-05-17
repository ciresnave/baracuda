//! Real-GPU smoke tests for `PadPlan<{f16,bf16,f64}, N> +
//! PadMode::Constant` — dtype fanout of the f32 trailblazer in
//! `pad_smoke.rs`.
//!
//! Each test mirrors the structure of `pad_constant_3d` from the f32
//! smoke test. Pad does no arithmetic (pure element copy + constant
//! fill in the pad region), so the comparison is **bit-exact** —
//! checked via `to_bits()` against a CPU reference that does the same
//! coord-by-coord walk.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test pad_dtype_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PadArgs, PadDescriptor, PadMode, PadPlan, PlanPreference,
    TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Generic host pad-constant reference. Walks every output cell, copies
/// from input if the coord (minus pad_low) lands in [0, input_shape);
/// otherwise writes `value`. T must be `Copy + PartialEq`.
fn cpu_pad_constant<T: Copy, const N: usize>(
    x_host: &[T],
    input_shape: [i32; N],
    pad_low: [i32; N],
    output_shape: [i32; N],
    value: T,
) -> Vec<T> {
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut out = vec![value; out_numel];
    let in_strides = contiguous_stride(input_shape);
    let out_strides = contiguous_stride(output_shape);
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

/// 3D pad case shared across dtypes — same shape / pad amounts.
const INPUT_SHAPE_3D: [i32; 3] = [4, 8, 12];
const PAD_LOW_3D: [i32; 3] = [1, 2, 1];
const PAD_HIGH_3D: [i32; 3] = [2, 1, 3];

fn output_shape_3d() -> [i32; 3] {
    [
        INPUT_SHAPE_3D[0] + PAD_LOW_3D[0] + PAD_HIGH_3D[0],
        INPUT_SHAPE_3D[1] + PAD_LOW_3D[1] + PAD_HIGH_3D[1],
        INPUT_SHAPE_3D[2] + PAD_LOW_3D[2] + PAD_HIGH_3D[2],
    ]
}

#[test]
#[ignore]
fn pad_constant_3d_f16() {
    let (ctx, stream) = setup();
    let input_shape = INPUT_SHAPE_3D;
    let pad_low = PAD_LOW_3D;
    let pad_high = PAD_HIGH_3D;
    let output_shape = output_shape_3d();
    // Half-precision representable; not a special value.
    let pad_value_f32: f32 = -3.5;
    let pad_value_h = f16::from_f32(pad_value_f32);

    let in_numel = (input_shape[0] * input_shape[1] * input_shape[2]) as usize;
    // Stay in [-10, 10] to keep f16 representable.
    let host_x: Vec<f16> = (0..in_numel)
        .map(|i| f16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let expected = cpu_pad_constant::<f16, 3>(&host_x, input_shape, pad_low, output_shape, pad_value_h);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = PadDescriptor {
        mode: PadMode::Constant,
        input_shape,
        pad_low,
        pad_high,
        value: pad_value_f32,
        element: ElementKind::F16,
    };
    let plan = PadPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select f16");
    let args = PadArgs::<f16, 3> {
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
    plan.run(&stream, Workspace::None, args).expect("run f16");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "3d pad f16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn pad_constant_3d_bf16() {
    let (ctx, stream) = setup();
    let input_shape = INPUT_SHAPE_3D;
    let pad_low = PAD_LOW_3D;
    let pad_high = PAD_HIGH_3D;
    let output_shape = output_shape_3d();
    let pad_value_f32: f32 = 7.25;
    let pad_value_h = bf16::from_f32(pad_value_f32);

    let in_numel = (input_shape[0] * input_shape[1] * input_shape[2]) as usize;
    let host_x: Vec<bf16> = (0..in_numel)
        .map(|i| bf16::from_f32(((i % 37) as f32) * 0.25 - 4.5))
        .collect();
    let expected =
        cpu_pad_constant::<bf16, 3>(&host_x, input_shape, pad_low, output_shape, pad_value_h);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = PadDescriptor {
        mode: PadMode::Constant,
        input_shape,
        pad_low,
        pad_high,
        value: pad_value_f32,
        element: ElementKind::Bf16,
    };
    let plan = PadPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select bf16");
    let args = PadArgs::<bf16, 3> {
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
    plan.run(&stream, Workspace::None, args).expect("run bf16");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "3d pad bf16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn pad_constant_3d_f64() {
    let (ctx, stream) = setup();
    let input_shape = INPUT_SHAPE_3D;
    let pad_low = PAD_LOW_3D;
    let pad_high = PAD_HIGH_3D;
    let output_shape = output_shape_3d();
    let pad_value_f32: f32 = -1.5;
    // Pad's descriptor value is `f32`; dispatcher widens to `f64`.
    // Reference uses the same widening.
    let pad_value_d = pad_value_f32 as f64;

    let in_numel = (input_shape[0] * input_shape[1] * input_shape[2]) as usize;
    let host_x: Vec<f64> = (0..in_numel).map(|i| (i as f64) * 0.125 - 5.0).collect();
    let expected =
        cpu_pad_constant::<f64, 3>(&host_x, input_shape, pad_low, output_shape, pad_value_d);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = PadDescriptor {
        mode: PadMode::Constant,
        input_shape,
        pad_low,
        pad_high,
        value: pad_value_f32,
        element: ElementKind::F64,
    };
    let plan = PadPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select f64");
    let args = PadArgs::<f64, 3> {
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
    plan.run(&stream, Workspace::None, args).expect("run f64");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "3d pad f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
