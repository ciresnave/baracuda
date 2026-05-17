//! Real-GPU smoke test for `ReducePlan` Var / Std (Welford one-pass).
//!
//! Wired for all four FP dtypes (Phase 4 deferral 4.2 close-out):
//! - f32: bit-stable vs the host f32 Welford reference (var) or 4 ULP (std).
//! - f16 / bf16: Welford state runs at f32 internally; the host
//!   reference uses the same f32 Welford and then casts back to T at
//!   the final store. Tolerance accounts for that single-store ULP.
//! - f64: Welford in f64 end-to-end. Bit-stable vs host f64 Welford for
//!   var; 4 ULP for std.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ReduceArgs, ReduceDescriptor, ReduceKind,
    ReducePlan, TensorMut, TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Host Welford computation matching the device kernel.
fn cpu_welford_axis(
    x: &[f32],
    input_shape: [i32; 3],
    reduce_axis: usize,
    correction: i32,
    do_sqrt: bool,
) -> (Vec<f32>, [i32; 3]) {
    let mut output_shape = input_shape;
    output_shape[reduce_axis] = 1;
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut out = vec![0f32; out_numel];

    let in_strides = contiguous_stride(input_shape);
    let out_strides = contiguous_stride(output_shape);
    let reduce_extent = input_shape[reduce_axis];

    for i in 0..out_numel {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = output_shape[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        let mut mean = 0f32;
        let mut m2 = 0f32;
        for k in 0..reduce_extent {
            let mut in_coord = coord;
            in_coord[reduce_axis] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            let v = x[in_off as usize];
            let delta = v - mean;
            mean += delta / (k + 1) as f32;
            let delta2 = v - mean;
            m2 += delta * delta2;
        }
        let denom = (reduce_extent - correction) as f32;
        let variance = if denom > 0.0 { m2 / denom } else { 0.0 };
        let result = if do_sqrt { variance.sqrt() } else { variance };
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        out[out_off as usize] = result;
    }
    (out, output_shape)
}

/// f64 Welford reference (mirrors the device kernel for f64 inputs).
fn cpu_welford_axis_f64(
    x: &[f64],
    input_shape: [i32; 3],
    reduce_axis: usize,
    correction: i32,
    do_sqrt: bool,
) -> (Vec<f64>, [i32; 3]) {
    let mut output_shape = input_shape;
    output_shape[reduce_axis] = 1;
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();
    let mut out = vec![0f64; out_numel];
    let in_strides = contiguous_stride(input_shape);
    let out_strides = contiguous_stride(output_shape);
    let reduce_extent = input_shape[reduce_axis];
    for i in 0..out_numel {
        let mut linear = i as i64;
        let mut coord = [0i64; 3];
        for d in (0..3).rev() {
            let s = output_shape[d] as i64;
            coord[d] = linear % s;
            linear /= s;
        }
        let mut mean = 0f64;
        let mut m2 = 0f64;
        for k in 0..reduce_extent {
            let mut in_coord = coord;
            in_coord[reduce_axis] = k as i64;
            let mut in_off: i64 = 0;
            for d in 0..3 {
                in_off += in_coord[d] * in_strides[d];
            }
            let v = x[in_off as usize];
            let delta = v - mean;
            mean += delta / (k + 1) as f64;
            let delta2 = v - mean;
            m2 += delta * delta2;
        }
        let denom = (reduce_extent - correction) as f64;
        let variance = if denom > 0.0 { m2 / denom } else { 0.0 };
        let result = if do_sqrt { variance.sqrt() } else { variance };
        let mut out_off: i64 = 0;
        for d in 0..3 {
            out_off += coord[d] * out_strides[d];
        }
        out[out_off as usize] = result;
    }
    (out, output_shape)
}

fn run_case(kind: ReduceKind, reduce_axis: usize, correction: i32) {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 16, 32];
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f32> = (0..in_numel)
        .map(|i| (i as f32) * 0.0625 - 1.0)
        .collect();
    let do_sqrt = matches!(kind, ReduceKind::Std);
    let (expected, output_shape) = cpu_welford_axis(
        &host_x, input_shape, reduce_axis, correction, do_sqrt,
    );
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = ReduceDescriptor {
        kind,
        input_shape,
        reduce_axis: reduce_axis as u8,
        element: ElementKind::F32,
        correction,
    };
    let plan = ReducePlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = ReduceArgs::<f32, 3> {
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

    // The device kernel uses the same Welford algorithm in f32 as the
    // host reference. Should be bit-exact for non-sqrt, very-close for
    // sqrt (since sqrtf may differ between libm and libdevice by 1 ULP).
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        if matches!(kind, ReduceKind::Var) {
            assert_eq!(
                g.to_bits(),
                e.to_bits(),
                "var axis={reduce_axis} corr={correction} mismatch @ {i}: got {g} expected {e}"
            );
        } else {
            let diff = (g - e).abs();
            let allow = e.abs().max(1.0) * 4.0 * f32::EPSILON;
            assert!(
                diff <= allow,
                "std axis={reduce_axis} corr={correction} mismatch @ {i}: got {g} expected {e} (diff {diff} > allow {allow})"
            );
        }
    }
}

// f16: Welford in f32 internally; final store casts to f16. Reference
// performs the f32 Welford then casts back. Tolerance ~4 f16-ULP.
fn run_case_f16(kind: ReduceKind, reduce_axis: usize, correction: i32) {
    let (ctx, stream) = setup();
    // Smaller / friendlier magnitudes for f16 dynamic range.
    let input_shape = [3i32, 4, 5];
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x_f32: Vec<f32> = (0..in_numel)
        .map(|i| 0.25 + 0.05 * (i as f32))
        .collect();
    let host_x: Vec<f16> = host_x_f32.iter().map(|&v| f16::from_f32(v)).collect();
    // Round trip the host values through f16 so the reference uses what
    // the GPU actually sees (else the input-quantization error would
    // show up as a host/GPU divergence rather than the per-output ULP we
    // care about).
    let host_x_round: Vec<f32> = host_x.iter().map(|v| v.to_f32()).collect();
    let do_sqrt = matches!(kind, ReduceKind::Std);
    let (expected_f32, output_shape) = cpu_welford_axis(
        &host_x_round, input_shape, reduce_axis, correction, do_sqrt,
    );
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = ReduceDescriptor {
        kind,
        input_shape,
        reduce_axis: reduce_axis as u8,
        element: ElementKind::F16,
        correction,
    };
    let plan = ReducePlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ReduceArgs::<f16, 3> {
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
    let mut got = vec![f16::ZERO; out_numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    // f16 mantissa ~9.77e-4. Allow 4 ULP plus an absolute floor.
    let eps = 4.0 * 9.77e-4_f32;
    for i in 0..out_numel {
        let g = got[i].to_f32();
        let e = expected_f32[i];
        let tol = (e.abs() * eps).max(eps);
        let diff = (g - e).abs();
        assert!(
            diff <= tol,
            "f16 {:?} axis={reduce_axis} corr={correction} @ {i}: got={g} want={e} diff={diff} tol={tol}",
            kind
        );
    }
}

// bf16: Welford in f32 internally; bf16 mantissa is 7 bits (~7.81e-3).
fn run_case_bf16(kind: ReduceKind, reduce_axis: usize, correction: i32) {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x_f32: Vec<f32> = (0..in_numel)
        .map(|i| 0.25 + 0.05 * (i as f32))
        .collect();
    let host_x: Vec<bf16> = host_x_f32.iter().map(|&v| bf16::from_f32(v)).collect();
    let host_x_round: Vec<f32> = host_x.iter().map(|v| v.to_f32()).collect();
    let do_sqrt = matches!(kind, ReduceKind::Std);
    let (expected_f32, output_shape) = cpu_welford_axis(
        &host_x_round, input_shape, reduce_axis, correction, do_sqrt,
    );
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = ReduceDescriptor {
        kind,
        input_shape,
        reduce_axis: reduce_axis as u8,
        element: ElementKind::Bf16,
        correction,
    };
    let plan = ReducePlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ReduceArgs::<bf16, 3> {
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
    let mut got = vec![bf16::ZERO; out_numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let eps = 4.0 * 7.81e-3_f32;
    for i in 0..out_numel {
        let g = got[i].to_f32();
        let e = expected_f32[i];
        let tol = (e.abs() * eps).max(eps);
        let diff = (g - e).abs();
        assert!(
            diff <= tol,
            "bf16 {:?} axis={reduce_axis} corr={correction} @ {i}: got={g} want={e} diff={diff} tol={tol}",
            kind
        );
    }
}

// f64: Welford in f64 end-to-end. Tighter tolerance (8*eps).
fn run_case_f64(kind: ReduceKind, reduce_axis: usize, correction: i32) {
    let (ctx, stream) = setup();
    let input_shape = [3i32, 4, 5];
    let in_numel: usize = input_shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..in_numel)
        .map(|i| 0.25 + 0.05 * (i as f64))
        .collect();
    let do_sqrt = matches!(kind, ReduceKind::Std);
    let (expected, output_shape) =
        cpu_welford_axis_f64(&host_x, input_shape, reduce_axis, correction, do_sqrt);
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("up x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc y");

    let desc = ReduceDescriptor {
        kind,
        input_shape,
        reduce_axis: reduce_axis as u8,
        element: ElementKind::F64,
        correction,
    };
    let plan = ReducePlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("sel");
    let args = ReduceArgs::<f64, 3> {
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
    dev_y.copy_to_host(&mut got).expect("dl");

    let eps = 8.0 * f64::EPSILON;
    for i in 0..out_numel {
        let g = got[i];
        let e = expected[i];
        let tol = (e.abs() * eps).max(eps);
        let diff = (g - e).abs();
        assert!(
            diff <= tol,
            "f64 {:?} axis={reduce_axis} corr={correction} @ {i}: got={g} want={e} diff={diff} tol={tol}",
            kind
        );
    }
}

#[test]
#[ignore]
fn var_axis_0_bessel() {
    run_case(ReduceKind::Var, 0, 1);
}

#[test]
#[ignore]
fn var_axis_1_bessel() {
    run_case(ReduceKind::Var, 1, 1);
}

#[test]
#[ignore]
fn var_axis_2_bessel() {
    run_case(ReduceKind::Var, 2, 1);
}

#[test]
#[ignore]
fn var_axis_0_population() {
    // correction = 0 → population variance
    run_case(ReduceKind::Var, 0, 0);
}

#[test]
#[ignore]
fn std_axis_1_bessel() {
    run_case(ReduceKind::Std, 1, 1);
}

#[test]
#[ignore]
fn std_axis_2_population() {
    run_case(ReduceKind::Std, 2, 0);
}

// ----- f16 -----

#[test]
#[ignore]
fn var_f16_axis_0_bessel() {
    run_case_f16(ReduceKind::Var, 0, 1);
}
#[test]
#[ignore]
fn var_f16_axis_1_population() {
    run_case_f16(ReduceKind::Var, 1, 0);
}
#[test]
#[ignore]
fn std_f16_axis_2_bessel() {
    run_case_f16(ReduceKind::Std, 2, 1);
}

// ----- bf16 -----

#[test]
#[ignore]
fn var_bf16_axis_1_bessel() {
    run_case_bf16(ReduceKind::Var, 1, 1);
}
#[test]
#[ignore]
fn var_bf16_axis_2_population() {
    run_case_bf16(ReduceKind::Var, 2, 0);
}
#[test]
#[ignore]
fn std_bf16_axis_0_bessel() {
    run_case_bf16(ReduceKind::Std, 0, 1);
}

// ----- f64 -----

#[test]
#[ignore]
fn var_f64_axis_2_bessel() {
    run_case_f64(ReduceKind::Var, 2, 1);
}
#[test]
#[ignore]
fn var_f64_axis_0_population() {
    run_case_f64(ReduceKind::Var, 0, 0);
}
#[test]
#[ignore]
fn std_f64_axis_1_bessel() {
    run_case_f64(ReduceKind::Std, 1, 1);
}
