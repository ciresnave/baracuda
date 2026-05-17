//! Real-GPU smoke tests for `ReducePlan<T, 3> + ReduceKind::Mean`
//! across `{f32, f16, bf16, f64}`. One canonical case per dtype:
//! shape `[4, 16, 32]` with axis-1 reduction (extent 16).
//!
//! Mean = sum-along-axis / extent. Host reference does the same
//! in-order accumulation then divides by extent. Tolerance per dtype:
//! - `f32` / `f64`: relative `4 * eps` — the final divide is exact for
//!   integer extents (since `1/16` is power-of-two in this case
//!   bit-exact, but for safety we use a small tolerance to cover all
//!   reduce extents).
//! - `f16` / `bf16`: relative `4 * eps` (every accumulate step + the
//!   final divide detours through f32).
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test reduce_mean_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ReduceArgs, ReduceDescriptor, ReduceKind,
    ReducePlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

const F16_EPS: f32 = 9.77e-4;
const BF16_EPS: f32 = 7.81e-3;

const SHAPE: [i32; 3] = [4, 16, 32];
const AXIS: usize = 1;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn out_shape() -> [i32; 3] {
    let mut s = SHAPE;
    s[AXIS] = 1;
    s
}

fn in_numel() -> usize {
    SHAPE.iter().map(|&d| d as usize).product()
}

fn out_numel() -> usize {
    out_shape().iter().map(|&d| d as usize).product()
}

#[test]
#[ignore]
fn reduce_mean_f32() {
    let (ctx, stream) = setup();
    let host_x: Vec<f32> = (0..in_numel())
        .map(|i| (i as f32) * 0.0625 - 50.0)
        .collect();

    let in_strides = contiguous_stride(SHAPE);
    let out_strides = contiguous_stride(out_shape());
    let extent = SHAPE[AXIS];
    let mut expected = vec![0f32; out_numel()];
    for i in 0..out_numel() {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = out_shape()[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        let mut acc = 0f32;
        for k in 0..extent {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            acc += host_x[in_off as usize];
        }
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        expected[out_off as usize] = acc / (extent as f32);
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, out_numel()).expect("alloc y");

    let desc = ReduceDescriptor {
        kind: ReduceKind::Mean,
        input_shape: SHAPE,
        reduce_axis: AXIS as u8,
        element: ElementKind::F32,
        correction: 1,
    };
    let plan = ReducePlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceArgs::<f32, 3> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: SHAPE,
            stride: contiguous_stride(SHAPE),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: out_shape(),
            stride: contiguous_stride(out_shape()),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel()];
    dev_y.copy_to_host(&mut got).expect("download");

    let tol = 4.0 * f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1.0) * tol;
        assert!(
            diff <= allow,
            "reduce mean f32 mismatch @ {i}: got {g} expected {e}, diff {diff} allow {allow}"
        );
    }
}

#[test]
#[ignore]
fn reduce_mean_f64() {
    let (ctx, stream) = setup();
    let host_x: Vec<f64> = (0..in_numel())
        .map(|i| (i as f64) * 0.0625 - 50.0)
        .collect();

    let in_strides = contiguous_stride(SHAPE);
    let out_strides = contiguous_stride(out_shape());
    let extent = SHAPE[AXIS];
    let mut expected = vec![0f64; out_numel()];
    for i in 0..out_numel() {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = out_shape()[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        let mut acc = 0f64;
        for k in 0..extent {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            acc += host_x[in_off as usize];
        }
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        expected[out_off as usize] = acc / (extent as f64);
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, out_numel()).expect("alloc y");

    let desc = ReduceDescriptor {
        kind: ReduceKind::Mean,
        input_shape: SHAPE,
        reduce_axis: AXIS as u8,
        element: ElementKind::F64,
        correction: 1,
    };
    let plan = ReducePlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceArgs::<f64, 3> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: SHAPE,
            stride: contiguous_stride(SHAPE),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: out_shape(),
            stride: contiguous_stride(out_shape()),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; out_numel()];
    dev_y.copy_to_host(&mut got).expect("download");

    let tol = 4.0 * f64::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1.0) * tol;
        assert!(
            diff <= allow,
            "reduce mean f64 mismatch @ {i}: got {g} expected {e}, diff {diff} allow {allow}"
        );
    }
}

#[test]
#[ignore]
fn reduce_mean_f16() {
    let (ctx, stream) = setup();
    let host_x: Vec<f16> = (0..in_numel())
        .map(|i| f16::from_f32(((i % 41) as f32) * 0.125 - 2.0))
        .collect();

    let in_strides = contiguous_stride(SHAPE);
    let out_strides = contiguous_stride(out_shape());
    let extent = SHAPE[AXIS];
    let mut expected = vec![f16::from_f32(0.0); out_numel()];
    for i in 0..out_numel() {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = out_shape()[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        let mut acc = f16::from_f32(0.0);
        for k in 0..extent {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            acc = f16::from_f32(acc.to_f32() + host_x[in_off as usize].to_f32());
        }
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        expected[out_off as usize] = f16::from_f32(acc.to_f32() / (extent as f32));
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, out_numel()).expect("alloc y");

    let desc = ReduceDescriptor {
        kind: ReduceKind::Mean,
        input_shape: SHAPE,
        reduce_axis: AXIS as u8,
        element: ElementKind::F16,
        correction: 1,
    };
    let plan = ReducePlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceArgs::<f16, 3> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: SHAPE,
            stride: contiguous_stride(SHAPE),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: out_shape(),
            stride: contiguous_stride(out_shape()),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); out_numel()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let diff = (gf - ef).abs();
        let allow = ef.abs().max(1.0) * (4.0 * F16_EPS);
        assert!(
            diff <= allow,
            "reduce mean f16 mismatch @ {i}: got {gf} expected {ef}, diff {diff} allow {allow}"
        );
    }
}

#[test]
#[ignore]
fn reduce_mean_bf16() {
    let (ctx, stream) = setup();
    let host_x: Vec<bf16> = (0..in_numel())
        .map(|i| bf16::from_f32(((i % 41) as f32) * 0.125 - 2.0))
        .collect();

    let in_strides = contiguous_stride(SHAPE);
    let out_strides = contiguous_stride(out_shape());
    let extent = SHAPE[AXIS];
    let mut expected = vec![bf16::from_f32(0.0); out_numel()];
    for i in 0..out_numel() {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = out_shape()[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        let mut acc = bf16::from_f32(0.0);
        for k in 0..extent {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            acc = bf16::from_f32(acc.to_f32() + host_x[in_off as usize].to_f32());
        }
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        expected[out_off as usize] = bf16::from_f32(acc.to_f32() / (extent as f32));
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, out_numel()).expect("alloc y");

    let desc = ReduceDescriptor {
        kind: ReduceKind::Mean,
        input_shape: SHAPE,
        reduce_axis: AXIS as u8,
        element: ElementKind::Bf16,
        correction: 1,
    };
    let plan = ReducePlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceArgs::<bf16, 3> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: SHAPE,
            stride: contiguous_stride(SHAPE),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: out_shape(),
            stride: contiguous_stride(out_shape()),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); out_numel()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let gf = g.to_f32();
        let ef = e.to_f32();
        let diff = (gf - ef).abs();
        let allow = ef.abs().max(1.0) * (4.0 * BF16_EPS);
        assert!(
            diff <= allow,
            "reduce mean bf16 mismatch @ {i}: got {gf} expected {ef}, diff {diff} allow {allow}"
        );
    }
}
