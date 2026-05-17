//! Real-GPU smoke tests for the non-constant Pad modes: `Reflect`,
//! `Replicate`, `Circular`. 3 modes × 4 dtypes = 12 tests at a fixed
//! 2D shape.
//!
//! Pad does no arithmetic — every pad-region cell reads from a single
//! input cell (transformed coord), every "inside" cell copies one
//! input cell. Comparison is **bit-exact** via `to_bits()` for all
//! four dtypes.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test pad_modes_smoke -- --ignored`.

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

// Shape big enough that `pad_low + pad_high < input_extent` per axis
// (keeps Reflect well-defined without secondary mirrors).
const INPUT_SHAPE_2D: [i32; 2] = [4, 6];
const PAD_LOW_2D: [i32; 2] = [2, 1];
const PAD_HIGH_2D: [i32; 2] = [1, 3];

fn output_shape_2d() -> [i32; 2] {
    [
        INPUT_SHAPE_2D[0] + PAD_LOW_2D[0] + PAD_HIGH_2D[0],
        INPUT_SHAPE_2D[1] + PAD_LOW_2D[1] + PAD_HIGH_2D[1],
    ]
}

/// CPU reference: transform an output coord into the input coord for
/// the given pad mode. Per-axis. Returns the clamped input coord (in
/// `[0, extent)`) for the mode's rule.
fn map_coord(mode: PadMode, out_coord: i64, pad_low: i32, extent: i32) -> i64 {
    let c = out_coord - pad_low as i64;
    let e = extent as i64;
    match mode {
        PadMode::Constant => c, // caller checks bounds; not used here
        PadMode::Reflect => {
            if e <= 1 {
                0
            } else if c < 0 {
                let r = -c;
                if r >= e {
                    e - 1
                } else {
                    r
                }
            } else if c >= e {
                let r = 2 * e - 2 - c;
                if r < 0 {
                    0
                } else {
                    r
                }
            } else {
                c
            }
        }
        PadMode::Replicate => {
            if e <= 0 {
                0
            } else {
                c.max(0).min(e - 1)
            }
        }
        PadMode::Circular => {
            if e <= 0 {
                0
            } else {
                ((c % e) + e) % e
            }
        }
    }
}

/// Generic 2D pad reference for non-constant modes. `T: Copy`.
fn cpu_pad_mode_2d<T: Copy + Default>(
    x: &[T],
    input_shape: [i32; 2],
    pad_low: [i32; 2],
    output_shape: [i32; 2],
    mode: PadMode,
) -> Vec<T> {
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut out = vec![T::default(); out_numel];
    let in_strides = contiguous_stride(input_shape);
    let out_strides = contiguous_stride(output_shape);
    for i in 0..output_shape[0] {
        for j in 0..output_shape[1] {
            let in_i = map_coord(mode, i as i64, pad_low[0], input_shape[0]);
            let in_j = map_coord(mode, j as i64, pad_low[1], input_shape[1]);
            let in_off = in_i * in_strides[0] + in_j * in_strides[1];
            let out_off = (i as i64) * out_strides[0] + (j as i64) * out_strides[1];
            out[out_off as usize] = x[in_off as usize];
        }
    }
    out
}

// One generic helper per dtype that runs (mode, dtype) end-to-end.
// We use one function-per-test approach so each cell is its own
// `#[test]` for clearer pass/fail reporting.

fn run_f32(mode: PadMode) {
    let (ctx, stream) = setup();
    let input_shape = INPUT_SHAPE_2D;
    let pad_low = PAD_LOW_2D;
    let pad_high = PAD_HIGH_2D;
    let output_shape = output_shape_2d();
    let in_numel = (input_shape[0] * input_shape[1]) as usize;
    let host_x: Vec<f32> = (0..in_numel).map(|i| (i as f32) * 0.5 - 3.0).collect();
    let expected = cpu_pad_mode_2d::<f32>(&host_x, input_shape, pad_low, output_shape, mode);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = PadDescriptor {
        mode,
        input_shape,
        pad_low,
        pad_high,
        // Ignored for non-Constant modes.
        value: 0.0,
        element: ElementKind::F32,
    };
    let plan =
        PadPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
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
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "pad f32 mode {mode:?} mismatch @ {i}: got {g} expected {e}"
        );
    }
}

fn run_f16(mode: PadMode) {
    let (ctx, stream) = setup();
    let input_shape = INPUT_SHAPE_2D;
    let pad_low = PAD_LOW_2D;
    let pad_high = PAD_HIGH_2D;
    let output_shape = output_shape_2d();
    let in_numel = (input_shape[0] * input_shape[1]) as usize;
    let host_x: Vec<f16> = (0..in_numel)
        .map(|i| f16::from_f32((i as f32) * 0.25 - 2.0))
        .collect();
    let expected = cpu_pad_mode_2d::<f16>(&host_x, input_shape, pad_low, output_shape, mode);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = PadDescriptor {
        mode,
        input_shape,
        pad_low,
        pad_high,
        value: 0.0,
        element: ElementKind::F16,
    };
    let plan =
        PadPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = PadArgs::<f16, 2> {
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

    let mut got = vec![f16::from_f32(0.0); out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "pad f16 mode {mode:?} mismatch @ {i}: got {g} expected {e}"
        );
    }
}

fn run_bf16(mode: PadMode) {
    let (ctx, stream) = setup();
    let input_shape = INPUT_SHAPE_2D;
    let pad_low = PAD_LOW_2D;
    let pad_high = PAD_HIGH_2D;
    let output_shape = output_shape_2d();
    let in_numel = (input_shape[0] * input_shape[1]) as usize;
    let host_x: Vec<bf16> = (0..in_numel)
        .map(|i| bf16::from_f32((i as f32) * 0.25 - 2.0))
        .collect();
    let expected = cpu_pad_mode_2d::<bf16>(&host_x, input_shape, pad_low, output_shape, mode);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = PadDescriptor {
        mode,
        input_shape,
        pad_low,
        pad_high,
        value: 0.0,
        element: ElementKind::Bf16,
    };
    let plan =
        PadPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = PadArgs::<bf16, 2> {
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

    let mut got = vec![bf16::from_f32(0.0); out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "pad bf16 mode {mode:?} mismatch @ {i}: got {g} expected {e}"
        );
    }
}

fn run_f64(mode: PadMode) {
    let (ctx, stream) = setup();
    let input_shape = INPUT_SHAPE_2D;
    let pad_low = PAD_LOW_2D;
    let pad_high = PAD_HIGH_2D;
    let output_shape = output_shape_2d();
    let in_numel = (input_shape[0] * input_shape[1]) as usize;
    let host_x: Vec<f64> = (0..in_numel).map(|i| (i as f64) * 0.125 - 0.5).collect();
    let expected = cpu_pad_mode_2d::<f64>(&host_x, input_shape, pad_low, output_shape, mode);

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = PadDescriptor {
        mode,
        input_shape,
        pad_low,
        pad_high,
        value: 0.0,
        element: ElementKind::F64,
    };
    let plan =
        PadPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = PadArgs::<f64, 2> {
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

    let mut got = vec![0f64; out_numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "pad f64 mode {mode:?} mismatch @ {i}: got {g} expected {e}"
        );
    }
}

// 12 cells = 3 modes × 4 dtypes.

#[test]
#[ignore]
fn pad_reflect_2d_f32() {
    run_f32(PadMode::Reflect);
}
#[test]
#[ignore]
fn pad_reflect_2d_f16() {
    run_f16(PadMode::Reflect);
}
#[test]
#[ignore]
fn pad_reflect_2d_bf16() {
    run_bf16(PadMode::Reflect);
}
#[test]
#[ignore]
fn pad_reflect_2d_f64() {
    run_f64(PadMode::Reflect);
}

#[test]
#[ignore]
fn pad_replicate_2d_f32() {
    run_f32(PadMode::Replicate);
}
#[test]
#[ignore]
fn pad_replicate_2d_f16() {
    run_f16(PadMode::Replicate);
}
#[test]
#[ignore]
fn pad_replicate_2d_bf16() {
    run_bf16(PadMode::Replicate);
}
#[test]
#[ignore]
fn pad_replicate_2d_f64() {
    run_f64(PadMode::Replicate);
}

#[test]
#[ignore]
fn pad_circular_2d_f32() {
    run_f32(PadMode::Circular);
}
#[test]
#[ignore]
fn pad_circular_2d_f16() {
    run_f16(PadMode::Circular);
}
#[test]
#[ignore]
fn pad_circular_2d_bf16() {
    run_bf16(PadMode::Circular);
}
#[test]
#[ignore]
fn pad_circular_2d_f64() {
    run_f64(PadMode::Circular);
}
