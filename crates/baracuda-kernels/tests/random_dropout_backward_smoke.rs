//! Real-GPU smoke test for `DropoutBackwardPlan`.
//!
//! Verifies bit-exact (within rounding) `dx[i] = dy[i] * mask[i] * scale`.
//! No randomness involved — the mask is a deterministic input.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Bool, DropoutBackwardArgs, DropoutBackwardDescriptor, DropoutBackwardPlan,
    ElementKind, PlanPreference, TensorMut, TensorRef, Workspace,
};

const N: usize = 1024 * 1024;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn dropout_backward_f32_p_half() {
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    // Construct a deterministic mask (alternating-pattern) and a
    // deterministic dy.
    let host_dy: Vec<f32> = (0..N).map(|i| ((i as f32) * 0.125) - 4.0).collect();
    let host_mask: Vec<Bool> = (0..N).map(|i| Bool(if i % 3 == 0 { 0 } else { 1 })).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let dev_mask = DeviceBuffer::from_slice(&ctx, &host_mask).expect("upload mask");
    let mut dev_dx: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, N).expect("alloc dx");

    let p = 0.5f32;
    let scale = 1.0f32 / (1.0 - p);

    let desc = DropoutBackwardDescriptor {
        shape,
        element: ElementKind::F32,
        p,
    };
    let plan =
        DropoutBackwardPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
            .expect("select");
    let args = DropoutBackwardArgs::<f32, 1> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        mask: TensorRef { data: dev_mask.as_slice(), shape, stride },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).expect("dropout BW run");
    stream.synchronize().expect("sync");

    let mut host_dx = vec![0f32; N];
    dev_dx.copy_to_host(&mut host_dx).expect("download dx");

    let mut mismatches = 0usize;
    for (i, (got, (dyi, mi))) in host_dx
        .iter()
        .zip(host_dy.iter().zip(host_mask.iter()))
        .enumerate()
    {
        let expected = if mi.0 == 1 { dyi * scale } else { 0.0 };
        let diff = (got - expected).abs();
        let tol = expected.abs() * 1e-5 + 1e-6;
        if diff > tol {
            mismatches += 1;
            if mismatches <= 3 {
                eprintln!("cell {i}: got {got} expected {expected} (dy = {dyi}, mask = {})", mi.0);
            }
        }
    }
    assert_eq!(mismatches, 0, "{mismatches} BW cells off-tolerance");
}

#[test]
#[ignore]
fn dropout_backward_f64_p_one_third() {
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    let host_dy: Vec<f64> = (0..N).map(|i| ((i as f64) * 0.0625) - 7.0).collect();
    let host_mask: Vec<Bool> = (0..N).map(|i| Bool(if i % 5 < 2 { 0 } else { 1 })).collect();

    let dev_dy = DeviceBuffer::from_slice(&ctx, &host_dy).expect("upload dy");
    let dev_mask = DeviceBuffer::from_slice(&ctx, &host_mask).expect("upload mask");
    let mut dev_dx: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, N).expect("alloc dx");

    let p = 1.0f32 / 3.0;
    let scale = 1.0f64 / (1.0 - p as f64);

    let desc = DropoutBackwardDescriptor {
        shape,
        element: ElementKind::F64,
        p,
    };
    let plan =
        DropoutBackwardPlan::<f64, 1>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let args = DropoutBackwardArgs::<f64, 1> {
        dy: TensorRef { data: dev_dy.as_slice(), shape, stride },
        mask: TensorRef { data: dev_mask.as_slice(), shape, stride },
        dx: TensorMut { data: dev_dx.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::None, args).unwrap();
    stream.synchronize().unwrap();

    let mut host_dx = vec![0f64; N];
    dev_dx.copy_to_host(&mut host_dx).unwrap();

    for (i, (got, (dyi, mi))) in host_dx
        .iter()
        .zip(host_dy.iter().zip(host_mask.iter()))
        .enumerate()
    {
        let expected = if mi.0 == 1 { dyi * scale } else { 0.0 };
        let diff = (got - expected).abs();
        let tol = expected.abs() * 1e-12 + 1e-12;
        assert!(
            diff <= tol,
            "cell {i}: got {got} expected {expected} (mask {})",
            mi.0
        );
    }
}
