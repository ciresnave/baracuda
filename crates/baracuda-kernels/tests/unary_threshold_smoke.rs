//! Real-GPU smoke test for `UnaryParamPlan + UnaryKind::Threshold` across
//! f32 / f16 / bf16 / f64.
//!
//! Forward: `y = (x > t) ? x : v`. Pure compare + select — no arithmetic,
//! so the result is bit-exact against a CPU reference for every dtype
//! (the matched branch returns `x` unchanged; the unmatched branch
//! returns `v` rounded to T exactly the same way on host and device).
//!
//! `#[ignore]` by default; run with `--ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryKind,
    UnaryParamArgs, UnaryParamDescriptor, UnaryParamPlan, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

const T: f32 = 0.5;
const V: f32 = -1.25;

#[test]
#[ignore]
fn threshold_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    // Inputs straddle the threshold T so both branches fire.
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.01 - 5.0).collect();
    let host_expected: Vec<f32> = host_x
        .iter()
        .map(|&x| if x > T { x } else { V })
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamDescriptor {
        kind: UnaryKind::Threshold,
        shape,
        element: ElementKind::F32,
        params: [T, V],
    };
    let plan = UnaryParamPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<f32, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    let mut first_mismatch: Option<(usize, f32, f32)> = None;
    let mut mismatches = 0usize;
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        if g.to_bits() != e.to_bits() {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((i, *g, *e));
            }
        }
    }
    if mismatches > 0 {
        let (i, g, e) = first_mismatch.unwrap();
        panic!(
            "threshold f32: {mismatches} mismatches / {numel}; first @ {i}: got {g} \
             (bits {:#x}) expected {e} (bits {:#x})",
            g.to_bits(), e.to_bits()
        );
    }
}

#[test]
#[ignore]
fn threshold_f64_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.01 - 5.0).collect();
    let host_expected: Vec<f64> = host_x
        .iter()
        .map(|&x| if x > T as f64 { x } else { V as f64 })
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamDescriptor {
        kind: UnaryKind::Threshold,
        shape,
        element: ElementKind::F64,
        params: [T, V],
    };
    let plan = UnaryParamPlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<f64, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "threshold f64 @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn threshold_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32((i as f32) * 0.01 - 5.0))
        .collect();
    // CPU reference matches the device branch: compare in f32, return
    // either the original f16 `x` or the f16-rounded `v`.
    let v_h = f16::from_f32(V);
    let host_expected: Vec<f16> = host_x
        .iter()
        .map(|x| if x.to_f32() > T { *x } else { v_h })
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamDescriptor {
        kind: UnaryKind::Threshold,
        shape,
        element: ElementKind::F16,
        params: [T, V],
    };
    let plan = UnaryParamPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<f16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "threshold f16 @ {i}: got {} (bits {:#x}) expected {} (bits {:#x})",
            g.to_f32(), g.to_bits(), e.to_f32(), e.to_bits()
        );
    }
}

#[test]
#[ignore]
fn threshold_bf16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 128, 128];
    let numel: usize = shape.iter().map(|&d| d as usize).product();
    let host_x: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32) * 0.01 - 5.0))
        .collect();
    let v_h = bf16::from_f32(V);
    let host_expected: Vec<bf16> = host_x
        .iter()
        .map(|x| if x.to_f32() > T { *x } else { v_h })
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");
    let stride = contiguous_stride(shape);
    let desc = UnaryParamDescriptor {
        kind: UnaryKind::Threshold,
        shape,
        element: ElementKind::Bf16,
        params: [T, V],
    };
    let plan = UnaryParamPlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryParamArgs::<bf16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");
    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "threshold bf16 @ {i}: got {} (bits {:#x}) expected {} (bits {:#x})",
            g.to_f32(), g.to_bits(), e.to_f32(), e.to_bits()
        );
    }
}
