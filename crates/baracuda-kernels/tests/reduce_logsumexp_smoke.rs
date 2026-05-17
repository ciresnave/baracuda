//! Real-GPU smoke tests for `ReducePlan<T, 3> + ReduceKind::LogSumExp`
//! across `{f32, f16, bf16, f64}`. Shape `[4, 16, 32]` with axis-1
//! reduction (matches the standard reduce smoke shape).
//!
//! LSE is the numerically stable
//! `y = log(sum(exp(x - max), axis=k)) + max`. We stress the kernel
//! on slightly negative inputs (max ≈ 0) so `exp(x - max)` stays
//! comfortably in range for every dtype.
//!
//! Tolerance: `8 * eps` relative for f32/f64 (two-pass walks + one
//! `exp` + one `log`). f16/bf16 absorb additional rounding at every
//! step — use `8 * dtype_eps`.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test reduce_logsumexp_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ReduceArgs, ReduceDescriptor, ReduceKind,
    ReducePlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

const F32_EPS: f32 = f32::EPSILON;
const F64_EPS: f64 = f64::EPSILON;
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
fn reduce_logsumexp_f32() {
    let (ctx, stream) = setup();

    // Inputs in roughly [-2, 0] — keeps exp(x - max) in [exp(-2), 1].
    let host_x: Vec<f32> = (0..in_numel())
        .map(|i| ((i % 41) as f32) * 0.05 - 2.0)
        .collect();

    let in_strides = contiguous_stride(SHAPE);
    let out_strides = contiguous_stride(out_shape());
    let mut expected = vec![0f32; out_numel()];
    for i in 0..out_numel() {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = out_shape()[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        // Pass 1: max.
        let mut m = f32::NEG_INFINITY;
        for k in 0..SHAPE[AXIS] {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            let v = host_x[in_off as usize];
            if v > m {
                m = v;
            }
        }
        // Pass 2: sum(exp(x - m)).
        let mut s = 0f32;
        for k in 0..SHAPE[AXIS] {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            s += (host_x[in_off as usize] - m).exp();
        }
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        expected[out_off as usize] = s.ln() + m;
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, out_numel()).expect("alloc y");

    let desc = ReduceDescriptor {
        kind: ReduceKind::LogSumExp,
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
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1.0) * (8.0 * F32_EPS);
        assert!(
            diff <= allow,
            "lse f32 mismatch @ {i}: got {g} expected {e}, diff {diff} allow {allow}"
        );
    }
}

#[test]
#[ignore]
fn reduce_logsumexp_f16() {
    let (ctx, stream) = setup();

    let host_x: Vec<f16> = (0..in_numel())
        .map(|i| f16::from_f32(((i % 17) as f32) * 0.05 - 1.0))
        .collect();

    let in_strides = contiguous_stride(SHAPE);
    let out_strides = contiguous_stride(out_shape());
    let mut expected = vec![f16::from_f32(0.0); out_numel()];
    for i in 0..out_numel() {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = out_shape()[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        let mut m = f32::NEG_INFINITY;
        for k in 0..SHAPE[AXIS] {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            let v = host_x[in_off as usize].to_f32();
            if v > m {
                m = v;
            }
        }
        let mut s = 0f32;
        for k in 0..SHAPE[AXIS] {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            s += (host_x[in_off as usize].to_f32() - m).exp();
        }
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        expected[out_off as usize] = f16::from_f32(s.ln() + m);
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> =
        DeviceBuffer::zeros(&ctx, out_numel()).expect("alloc y");

    let desc = ReduceDescriptor {
        kind: ReduceKind::LogSumExp,
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
        let allow = ef.abs().max(1.0) * (8.0 * F16_EPS);
        assert!(
            diff <= allow,
            "lse f16 mismatch @ {i}: got {gf} expected {ef}, diff {diff} allow {allow}"
        );
    }
}

#[test]
#[ignore]
fn reduce_logsumexp_bf16() {
    let (ctx, stream) = setup();

    let host_x: Vec<bf16> = (0..in_numel())
        .map(|i| bf16::from_f32(((i % 17) as f32) * 0.05 - 1.0))
        .collect();

    let in_strides = contiguous_stride(SHAPE);
    let out_strides = contiguous_stride(out_shape());
    let mut expected = vec![bf16::from_f32(0.0); out_numel()];
    for i in 0..out_numel() {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = out_shape()[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        let mut m = f32::NEG_INFINITY;
        for k in 0..SHAPE[AXIS] {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            let v = host_x[in_off as usize].to_f32();
            if v > m {
                m = v;
            }
        }
        let mut s = 0f32;
        for k in 0..SHAPE[AXIS] {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            s += (host_x[in_off as usize].to_f32() - m).exp();
        }
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        expected[out_off as usize] = bf16::from_f32(s.ln() + m);
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> =
        DeviceBuffer::zeros(&ctx, out_numel()).expect("alloc y");

    let desc = ReduceDescriptor {
        kind: ReduceKind::LogSumExp,
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
        let allow = ef.abs().max(1.0) * (8.0 * BF16_EPS);
        assert!(
            diff <= allow,
            "lse bf16 mismatch @ {i}: got {gf} expected {ef}, diff {diff} allow {allow}"
        );
    }
}

#[test]
#[ignore]
fn reduce_logsumexp_f64() {
    let (ctx, stream) = setup();

    let host_x: Vec<f64> = (0..in_numel())
        .map(|i| ((i % 41) as f64) * 0.05 - 2.0)
        .collect();

    let in_strides = contiguous_stride(SHAPE);
    let out_strides = contiguous_stride(out_shape());
    let mut expected = vec![0f64; out_numel()];
    for i in 0..out_numel() {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = out_shape()[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        let mut m = f64::NEG_INFINITY;
        for k in 0..SHAPE[AXIS] {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            let v = host_x[in_off as usize];
            if v > m {
                m = v;
            }
        }
        let mut s = 0f64;
        for k in 0..SHAPE[AXIS] {
            let mut in_coord = coord;
            in_coord[AXIS] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            s += (host_x[in_off as usize] - m).exp();
        }
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        expected[out_off as usize] = s.ln() + m;
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, out_numel()).expect("alloc y");

    let desc = ReduceDescriptor {
        kind: ReduceKind::LogSumExp,
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
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        let allow = e.abs().max(1.0) * (8.0 * F64_EPS);
        assert!(
            diff <= allow,
            "lse f64 mismatch @ {i}: got {g} expected {e}, diff {diff} allow {allow}"
        );
    }
}
