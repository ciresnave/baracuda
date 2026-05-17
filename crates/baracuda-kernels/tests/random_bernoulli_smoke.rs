//! Real-GPU smoke test for `RandomPlan + RandomKind::Bernoulli`.
//! Verifies the empirical mean of the Bool output matches the requested
//! probability `p` within a statistical tolerance.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Bool, ElementKind, PlanPreference, RandomBoolArgs, RandomDescriptor,
    RandomKind, RandomPlan, TensorMut, Workspace,
};

const N: usize = 1024 * 1024;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn run_case(p: f32, seed: u64) {
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    let mut dev_y: DeviceBuffer<Bool> = DeviceBuffer::zeros(&ctx, N).expect("alloc y");
    // Workspace = one f32 per cell.
    let ws_bytes = N * core::mem::size_of::<f32>();
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, ws_bytes).expect("alloc workspace");

    let desc = RandomDescriptor {
        kind: RandomKind::Bernoulli,
        shape,
        element: ElementKind::Bool,
        param1: p,
        param2: 0.0,
        seed,
    };
    let plan = RandomPlan::<Bool, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select RandomPlan<Bool>");
    let args = RandomBoolArgs::<1> {
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("bernoulli run");
    stream.synchronize().expect("sync");

    let mut host = vec![Bool(0); N];
    dev_y.copy_to_host(&mut host).expect("download");

    let mut ones: usize = 0;
    for b in &host {
        assert!(b.0 == 0 || b.0 == 1, "non-canonical bool byte {}", b.0);
        if b.0 == 1 {
            ones += 1;
        }
    }
    let empirical = ones as f64 / N as f64;
    // Bernoulli stderr = sqrt(p(1-p)/N) ≤ 0.5/sqrt(N) ≈ 4.9e-4 at N = 1M.
    // 3-stderr ≈ 1.5e-3; use 5e-3 tolerance.
    assert!(
        (empirical - p as f64).abs() < 5e-3,
        "bernoulli(p = {p}): empirical mean = {empirical}, expected ~{p}"
    );
}

#[test]
#[ignore]
fn bernoulli_p_half() {
    run_case(0.5, 0xCAFE_BABE_DEAD_BEEF);
}

#[test]
#[ignore]
fn bernoulli_p_quarter() {
    run_case(0.25, 0x1357_9BDF_2468_ACE0);
}

#[test]
#[ignore]
fn bernoulli_p_three_quarter() {
    run_case(0.75, 0xFEED_FACE_BAAD_F00D);
}

#[test]
#[ignore]
fn bernoulli_p_zero() {
    // All cells should be 0.
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    let mut dev_y: DeviceBuffer<Bool> = DeviceBuffer::zeros(&ctx, N).expect("alloc y");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, N * 4).expect("alloc workspace");

    let desc = RandomDescriptor {
        kind: RandomKind::Bernoulli,
        shape,
        element: ElementKind::Bool,
        param1: 0.0,
        param2: 0.0,
        seed: 42,
    };
    let plan = RandomPlan::<Bool, 1>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let args = RandomBoolArgs::<1> {
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args).unwrap();
    stream.synchronize().unwrap();

    let mut host = vec![Bool(0); N];
    dev_y.copy_to_host(&mut host).unwrap();
    // cuRAND samples in (0, 1] — every sample is strictly positive, so
    // (sample < 0) is always false. Every output byte must be 0.
    let nonzero = host.iter().filter(|b| b.0 != 0).count();
    assert_eq!(nonzero, 0, "p = 0 must yield all-zero output, got {nonzero} ones");
}
