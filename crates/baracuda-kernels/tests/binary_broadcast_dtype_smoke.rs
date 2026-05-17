//! Real-GPU dtype-fanout smoke tests for the strided / broadcast path
//! of `BinaryPlan<T, 2>` across all 15 remaining (op, dtype) cells.
//!
//! The trailblazer `binary_broadcast_smoke.rs` covers the five canonical
//! broadcast patterns (row-bias, col-bias, scalar, rank-3 seq-only,
//! transposed-B) for `Add × f32`. This file fans out one canonical
//! pattern (row-bias) across `{Add, Sub, Mul, Div} × {f32, f16, bf16,
//! f64}` minus the already-covered `Add × f32` — 15 (op, dtype) cells.
//!
//! Naming: `broadcast_<op>_<dtype>_row_bias`.
//!
//! Shape: `[64, 128]` with `a: [1, 128]` stride `[0, 1]` broadcast across
//! rows of `b: [64, 128]` into `y: [64, 128]`. Tolerance per dtype:
//! - f32 / f64: bit-exact.
//! - f16 / bf16: relative tolerance matching the contig dtype tests.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test binary_broadcast_dtype_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    BinaryArgs, BinaryDescriptor, BinaryKind, BinaryPlan, ElementKind, PlanPreference, TensorMut,
    TensorRef, Workspace,
};
use half::{bf16, f16};

// 1-ULP relative tolerance for f16 (mantissa = 10 bits → eps ≈ 2^-10).
const F16_EPS: f32 = 9.77e-4;
// 1-ULP relative tolerance for bf16 (mantissa = 7 bits → eps ≈ 2^-7).
const BF16_EPS: f32 = 7.81e-3;

const M: usize = 64;
const N_DIM: usize = 128;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Op selector for the host reference. The CPU reference walks the
/// broadcast offsets and applies the same scalar op the GPU kernel does.
#[derive(Copy, Clone)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

impl Op {
    fn kind(self) -> BinaryKind {
        match self {
            Op::Add => BinaryKind::Add,
            Op::Sub => BinaryKind::Sub,
            Op::Mul => BinaryKind::Mul,
            Op::Div => BinaryKind::Div,
        }
    }
    fn apply_f32(self, a: f32, b: f32) -> f32 {
        match self {
            Op::Add => a + b,
            Op::Sub => a - b,
            Op::Mul => a * b,
            Op::Div => a / b,
        }
    }
    fn apply_f64(self, a: f64, b: f64) -> f64 {
        match self {
            Op::Add => a + b,
            Op::Sub => a - b,
            Op::Mul => a * b,
            Op::Div => a / b,
        }
    }
}

// ----------------------------------------------------------------------------
// Input generators
// ----------------------------------------------------------------------------
//
// Row-bias broadcast: `a` is the bias of length N_DIM (logically [1, N]
// with stride [0, 1]); `b` is the full [M, N] contig tensor. Values are
// kept in [-10, 10] to stay well inside the f16 / bf16 dynamic range.
// For division, `b` is biased away from zero so the result stays finite
// for every cell.

fn gen_a_f32(non_div: bool) -> Vec<f32> {
    // Range roughly [-8, 7.875]; small magnitudes so f16/bf16 ULP errors
    // stay below 1 across all 8k cells.
    (0..N_DIM)
        .map(|i| {
            let v = (i as f32) * 0.125 - 8.0;
            if non_div {
                v
            } else {
                v
            }
        })
        .collect()
}

fn gen_b_f32_safe() -> Vec<f32> {
    // Range roughly [-5, 5] — fine for add/sub/mul, well away from zero
    // not required (these aren't div).
    (0..(M * N_DIM))
        .map(|i| (i as f32) * 0.0009765625 - 5.0)
        .collect()
}

fn gen_b_f32_div() -> Vec<f32> {
    // For div: keep b strictly positive and away from zero so the result
    // stays representable. Range roughly [1.5, 9.5] over the full grid.
    (0..(M * N_DIM))
        .map(|i| (i as f32) * 0.0009765625 + 1.5)
        .collect()
}

// ----------------------------------------------------------------------------
// Per-dtype kernel + compare drivers
// ----------------------------------------------------------------------------

fn run_f32(op: Op) {
    let (ctx, stream) = setup();
    let m = M as i32;
    let n = N_DIM as i32;

    let host_a = gen_a_f32(true);
    let host_b = if matches!(op, Op::Div) {
        gen_b_f32_div()
    } else {
        gen_b_f32_safe()
    };

    // CPU reference walks the row-bias pattern: y[i, j] = a[j] op b[i, j].
    let mut expected = vec![0f32; M * N_DIM];
    for i in 0..M {
        for j in 0..N_DIM {
            expected[i * N_DIM + j] = op.apply_f32(host_a[j], host_b[i * N_DIM + j]);
        }
    }

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let a_shape = [1i32, n];
    let a_stride = [0i64, 1];
    let b_shape = [m, n];
    let b_stride = [n as i64, 1];
    let y_shape = [m, n];
    let y_stride = [n as i64, 1];

    let desc = BinaryDescriptor {
        kind: op.kind(),
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = BinaryPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryArgs::<f32, 2> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: a_stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: b_stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: y_shape,
            stride: y_stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; M * N_DIM];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "row-bias broadcast f32 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

fn run_f64(op: Op) {
    let (ctx, stream) = setup();
    let m = M as i32;
    let n = N_DIM as i32;

    let host_a: Vec<f64> = (0..N_DIM).map(|i| (i as f64) * 0.125 - 8.0).collect();
    let host_b: Vec<f64> = if matches!(op, Op::Div) {
        (0..(M * N_DIM))
            .map(|i| (i as f64) * 0.0009765625 + 1.5)
            .collect()
    } else {
        (0..(M * N_DIM))
            .map(|i| (i as f64) * 0.0009765625 - 5.0)
            .collect()
    };

    let mut expected = vec![0f64; M * N_DIM];
    for i in 0..M {
        for j in 0..N_DIM {
            expected[i * N_DIM + j] = op.apply_f64(host_a[j], host_b[i * N_DIM + j]);
        }
    }

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let a_shape = [1i32, n];
    let a_stride = [0i64, 1];
    let b_shape = [m, n];
    let b_stride = [n as i64, 1];
    let y_shape = [m, n];
    let y_stride = [n as i64, 1];

    let desc = BinaryDescriptor {
        kind: op.kind(),
        shape: y_shape,
        element: ElementKind::F64,
    };
    let plan = BinaryPlan::<f64, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryArgs::<f64, 2> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: a_stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: b_stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: y_shape,
            stride: y_stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; M * N_DIM];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "row-bias broadcast f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

fn run_f16(op: Op) {
    let (ctx, stream) = setup();
    let m = M as i32;
    let n = N_DIM as i32;

    // Keep f16 magnitudes small.
    let host_a: Vec<f16> = (0..N_DIM)
        .map(|i| f16::from_f32(((i % 41) as f32) * 0.25 - 5.0))
        .collect();
    let host_b: Vec<f16> = if matches!(op, Op::Div) {
        (0..(M * N_DIM))
            .map(|i| f16::from_f32(((i % 37) as f32) * 0.0625 + 1.5))
            .collect()
    } else {
        (0..(M * N_DIM))
            .map(|i| f16::from_f32(((i % 37) as f32) * 0.125 - 4.0))
            .collect()
    };

    // Host reference: do the op in f16 to match the GPU one-step rounding.
    let mut expected = vec![f16::from_f32(0.0); M * N_DIM];
    for i in 0..M {
        for j in 0..N_DIM {
            let a = host_a[j];
            let b = host_b[i * N_DIM + j];
            expected[i * N_DIM + j] = match op {
                Op::Add => a + b,
                Op::Sub => a - b,
                Op::Mul => a * b,
                Op::Div => a / b,
            };
        }
    }

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let a_shape = [1i32, n];
    let a_stride = [0i64, 1];
    let b_shape = [m, n];
    let b_stride = [n as i64, 1];
    let y_shape = [m, n];
    let y_stride = [n as i64, 1];

    let desc = BinaryDescriptor {
        kind: op.kind(),
        shape: y_shape,
        element: ElementKind::F16,
    };
    let plan = BinaryPlan::<f16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryArgs::<f16, 2> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: a_stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: b_stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: y_shape,
            stride: y_stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); M * N_DIM];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let diff = (gf - ef).abs();
        let allow = ef.abs().max(1.0) * F16_EPS;
        assert!(
            diff <= allow,
            "row-bias broadcast f16 mismatch @ {i}: got {gf} (bits {:#x}) \
             expected {ef} (bits {:#x}), diff {diff}, allow {allow}",
            g.to_bits(),
            e.to_bits()
        );
    }
}

fn run_bf16(op: Op) {
    let (ctx, stream) = setup();
    let m = M as i32;
    let n = N_DIM as i32;

    let host_a: Vec<bf16> = (0..N_DIM)
        .map(|i| bf16::from_f32(((i % 41) as f32) * 0.25 - 5.0))
        .collect();
    let host_b: Vec<bf16> = if matches!(op, Op::Div) {
        (0..(M * N_DIM))
            .map(|i| bf16::from_f32(((i % 37) as f32) * 0.0625 + 1.5))
            .collect()
    } else {
        (0..(M * N_DIM))
            .map(|i| bf16::from_f32(((i % 37) as f32) * 0.125 - 4.0))
            .collect()
    };

    let mut expected = vec![bf16::from_f32(0.0); M * N_DIM];
    for i in 0..M {
        for j in 0..N_DIM {
            let a = host_a[j];
            let b = host_b[i * N_DIM + j];
            expected[i * N_DIM + j] = match op {
                Op::Add => a + b,
                Op::Sub => a - b,
                Op::Mul => a * b,
                Op::Div => a / b,
            };
        }
    }

    let dev_a = DeviceBuffer::from_slice(&ctx, &host_a).expect("upload a");
    let dev_b = DeviceBuffer::from_slice(&ctx, &host_b).expect("upload b");
    let mut dev_y: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, M * N_DIM).expect("alloc y");

    let a_shape = [1i32, n];
    let a_stride = [0i64, 1];
    let b_shape = [m, n];
    let b_stride = [n as i64, 1];
    let y_shape = [m, n];
    let y_stride = [n as i64, 1];

    let desc = BinaryDescriptor {
        kind: op.kind(),
        shape: y_shape,
        element: ElementKind::Bf16,
    };
    let plan = BinaryPlan::<bf16, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = BinaryArgs::<bf16, 2> {
        a: TensorRef {
            data: dev_a.as_slice(),
            shape: a_shape,
            stride: a_stride,
        },
        b: TensorRef {
            data: dev_b.as_slice(),
            shape: b_shape,
            stride: b_stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: y_shape,
            stride: y_stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); M * N_DIM];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let diff = (gf - ef).abs();
        let allow = ef.abs().max(1.0) * BF16_EPS;
        assert!(
            diff <= allow,
            "row-bias broadcast bf16 mismatch @ {i}: got {gf} (bits {:#x}) \
             expected {ef} (bits {:#x}), diff {diff}, allow {allow}",
            g.to_bits(),
            e.to_bits()
        );
    }
}

// ============================================================================
// Tests — 15 (op, dtype) cells, row-bias broadcast pattern.
// Naming: broadcast_<op>_<dtype>_row_bias.
// ============================================================================
//
// `Add × f32` is intentionally absent — covered by the trailblazer file
// `binary_broadcast_smoke.rs`.

#[test]
#[ignore]
fn broadcast_add_f16_row_bias() {
    run_f16(Op::Add);
}

#[test]
#[ignore]
fn broadcast_add_bf16_row_bias() {
    run_bf16(Op::Add);
}

#[test]
#[ignore]
fn broadcast_add_f64_row_bias() {
    run_f64(Op::Add);
}

#[test]
#[ignore]
fn broadcast_sub_f32_row_bias() {
    run_f32(Op::Sub);
}

#[test]
#[ignore]
fn broadcast_sub_f16_row_bias() {
    run_f16(Op::Sub);
}

#[test]
#[ignore]
fn broadcast_sub_bf16_row_bias() {
    run_bf16(Op::Sub);
}

#[test]
#[ignore]
fn broadcast_sub_f64_row_bias() {
    run_f64(Op::Sub);
}

#[test]
#[ignore]
fn broadcast_mul_f32_row_bias() {
    run_f32(Op::Mul);
}

#[test]
#[ignore]
fn broadcast_mul_f16_row_bias() {
    run_f16(Op::Mul);
}

#[test]
#[ignore]
fn broadcast_mul_bf16_row_bias() {
    run_bf16(Op::Mul);
}

#[test]
#[ignore]
fn broadcast_mul_f64_row_bias() {
    run_f64(Op::Mul);
}

#[test]
#[ignore]
fn broadcast_div_f32_row_bias() {
    run_f32(Op::Div);
}

#[test]
#[ignore]
fn broadcast_div_f16_row_bias() {
    run_f16(Op::Div);
}

#[test]
#[ignore]
fn broadcast_div_bf16_row_bias() {
    run_bf16(Op::Div);
}

#[test]
#[ignore]
fn broadcast_div_f64_row_bias() {
    run_f64(Op::Div);
}
