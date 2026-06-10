//! Phase 31 — direct-FFI smoke for `unary_step_<dtype>_run`, plus the
//! Phase 74 plan-driven cases (`UnaryPlan<T, N> + UnaryKind::Step`)
//! added when the facade gained the `Step` variant.
//!
//! `y = (x > 0) ? 1 : 0`. `step(0.0) = step(-0.0) = 0` (`x > 0` is
//! false at both zeros); NaN → 0 (NaN > 0 is false). Outputs are
//! exactly representable in every FP format ({0, 1}), so comparisons
//! are bit-exact (same convention as `unary_sign_smoke`).

use core::ffi::c_void;

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, TensorMut, TensorRef, UnaryArgs,
    UnaryDescriptor, UnaryKind, UnaryPlan, Workspace,
};
use half::f16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn ffi_unary_step_f32_matches_cpu() {
    let (ctx, stream) = setup();
    // Span (-3.0, +3.0) plus a zero and an explicit NaN.
    let mut host_x: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 - 5.12).collect();
    host_x.push(0.0);
    host_x.push(f32::NAN);

    let expected: Vec<f32> = host_x.iter()
        .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_step_f32_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0, "unary_step_f32 returned status {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    // Step is exact (no math; just a compare). Use bit-equality.
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "step f32 @ {i} (x = {}): got {g} expected {e}",
            host_x[i],
        );
    }
}

#[test]
#[ignore]
fn ffi_unary_step_f64_matches_cpu() {
    let (ctx, stream) = setup();
    let host_x: Vec<f64> = (0..256).map(|i| (i as f64) * 0.02 - 2.56).collect();
    let expected: Vec<f64> = host_x.iter()
        .map(|&x| if x > 0.0 { 1.0 } else { 0.0 })
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, host_x.len()).expect("alloc y");

    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_unary_step_f64_run(
            host_x.len() as i64,
            dev_x.as_slice().as_raw().0 as *const c_void,
            dev_y.as_slice_mut().as_raw().0 as *mut c_void,
            core::ptr::null_mut(),
            0,
            stream.as_raw() as *mut c_void,
        )
    };
    assert_eq!(status, 0);
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; host_x.len()];
    dev_y.copy_to_host(&mut got).expect("download");

    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "step f64 mismatch @ {i}");
    }
}

// --- Phase 74 — plan-driven (UnaryPlan + UnaryKind::Step) ---------------------

// Step expected value. `0.0 > 0.0` and `-0.0 > 0.0` are both false
// → 0 at both zeros.
fn cpu_step(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

// Build an input distribution straddling zero, with exact `0.0`,
// `-0.0`, and `NaN` planted at fixed positions. `NaN > 0` is false,
// so step(NaN) = 0 (matching the kernel's `x > 0` branch).
fn step_input_f32(i: usize) -> f32 {
    match i % 16 {
        0 => 0.0,
        4 => f32::NAN,
        8 => -0.0,
        r => (r as f32) * 0.5 - 4.25, // spans [-3.75, +3.25], never zero
    }
}

#[test]
#[ignore]
fn plan_unary_step_f32_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 64, 64];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f32> = (0..numel).map(step_input_f32).collect();
    let expected: Vec<f32> = host_x.iter().map(|x| cpu_step(*x)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Step,
        shape,
        element: ElementKind::F32,
    };
    let plan = UnaryPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f32, 3>");
    let args = UnaryArgs::<f32, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("step f32 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "step f32 plan mismatch @ {i}: got {g} expected {e} (input was {})",
            host_x[i],
        );
    }
}

#[test]
#[ignore]
fn plan_unary_step_f16_3d() {
    let (ctx, stream) = setup();
    let shape = [8i32, 64, 64];
    let numel: usize = shape.iter().map(|&d| d as usize).product();

    let host_x: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(step_input_f32(i)))
        .collect();
    let host_expected: Vec<f16> = host_x
        .iter()
        .map(|x| f16::from_f32(cpu_step(x.to_f32())))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let stride = contiguous_stride(shape);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Step,
        shape,
        element: ElementKind::F16,
    };
    let plan = UnaryPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select UnaryPlan<f16, 3>");
    let args = UnaryArgs::<f16, 3> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("step f16 run");
    stream.synchronize().expect("sync");

    let mut host_got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut host_got).expect("download");
    for (i, (g, e)) in host_got.iter().zip(host_expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "step f16 plan mismatch @ {i}: got {} expected {} (input was {})",
            g.to_f32(), e.to_f32(), host_x[i].to_f32(),
        );
    }
}

#[test]
#[ignore]
fn plan_unary_step_f32_strided_transposed() {
    let (ctx, stream) = setup();
    const M: usize = 48;
    const N_DIM: usize = 32;
    let m = M as i32;
    let n = N_DIM as i32;

    let x_buf: Vec<f32> = (0..(N_DIM * M)).map(step_input_f32).collect();
    let x_shape = [m, n];
    let x_stride = [1i64, M as i64]; // transposed
    let y_shape = [m, n];
    let y_stride = contiguous_stride([m, n]);

    let numel = M * N_DIM;
    let mut expected = vec![0f32; numel];
    for i in 0..M {
        for j in 0..N_DIM {
            let x_lin = j * M + i;
            let y_lin = i * N_DIM + j;
            expected[y_lin] = cpu_step(x_buf[x_lin]);
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &x_buf).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = UnaryDescriptor {
        kind: UnaryKind::Step,
        shape: y_shape,
        element: ElementKind::F32,
    };
    let plan = UnaryPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = UnaryArgs::<f32, 2> {
        x: TensorRef { data: dev_x.as_slice(), shape: x_shape, stride: x_stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape: y_shape, stride: y_stride },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(), e.to_bits(),
            "step f32 strided plan mismatch @ {i}: got {g} expected {e}",
        );
    }
}
