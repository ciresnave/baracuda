//! Real-GPU smoke test for `DropoutPlan` (FW).
//!
//! Verifies:
//! 1. The mask is consistent with the output: `y == mask · x · (1/(1-p))`
//!    for every cell.
//! 2. The empirical fraction of kept cells matches `1 - p` within a
//!    statistical tolerance.
//! 3. Per-cell: `mask = 0 ⇒ y == 0`, `mask = 1 ⇒ y == x / (1 - p)`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Bool, DropoutArgs, DropoutDescriptor, DropoutPlan, ElementKind,
    PlanPreference, TensorMut, TensorRef, Workspace,
};

const N: usize = 1024 * 1024;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

fn run_case_f32(p: f32, seed: u64) {
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    // Deterministic non-pathological input — every cell has |x| ≥ 1 so
    // `mask = 0 ⇒ y == 0` is unambiguous (no x cell happens to equal zero).
    let host_x: Vec<f32> = (0..N).map(|i| ((i as f32) % 31.0) + 1.0).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, N).expect("alloc y");
    let mut dev_mask: DeviceBuffer<Bool> = DeviceBuffer::zeros(&ctx, N).expect("alloc mask");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, N * core::mem::size_of::<f32>()).expect("alloc workspace");

    let desc = DropoutDescriptor {
        shape,
        element: ElementKind::F32,
        p,
        seed,
    };
    let plan = DropoutPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default())
        .expect("select DropoutPlan<f32>");
    let args = DropoutArgs::<f32, 1> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
        mask: TensorMut { data: dev_mask.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args)
        .expect("dropout f32 run");
    stream.synchronize().expect("sync");

    let mut host_y = vec![0f32; N];
    let mut host_mask = vec![Bool(0); N];
    dev_y.copy_to_host(&mut host_y).expect("download y");
    dev_mask.copy_to_host(&mut host_mask).expect("download mask");

    let scale = 1.0_f32 / (1.0 - p);
    let mut kept: usize = 0;
    for (i, (xi, (yi, mi))) in host_x.iter().zip(host_y.iter().zip(host_mask.iter())).enumerate() {
        if mi.0 == 1 {
            kept += 1;
            let expected = xi * scale;
            assert!(
                (yi - expected).abs() <= expected.abs() * 1e-5 + 1e-6,
                "mask=1 cell {i}: y = {yi}, expected x*scale = {expected} (x = {xi})"
            );
        } else if mi.0 == 0 {
            assert!(
                *yi == 0.0,
                "mask=0 cell {i}: y = {yi}, expected 0"
            );
        } else {
            panic!("non-canonical mask byte at cell {i}: {}", mi.0);
        }
    }
    let empirical_keep = kept as f64 / N as f64;
    let expected_keep = 1.0 - p as f64;
    // stderr ≤ 0.5/sqrt(N) ≈ 4.9e-4; 3-stderr ≈ 1.5e-3; use 5e-3 tolerance.
    assert!(
        (empirical_keep - expected_keep).abs() < 5e-3,
        "dropout(p = {p}): kept fraction = {empirical_keep}, expected ~{expected_keep}"
    );
}

#[test]
#[ignore]
fn dropout_f32_p_half() {
    run_case_f32(0.5, 0xABAB_CDCD_EFEF_0101);
}

#[test]
#[ignore]
fn dropout_f32_p_one_tenth() {
    run_case_f32(0.1, 0x1111_2222_3333_4444);
}

#[test]
#[ignore]
fn dropout_f32_p_nine_tenths() {
    run_case_f32(0.9, 0x9999_8888_7777_6666);
}

#[test]
#[ignore]
fn dropout_f64_p_half() {
    let (ctx, stream) = setup();
    let shape = [N as i32];
    let stride = contiguous_stride(shape);

    let host_x: Vec<f64> = (0..N).map(|i| ((i as f64) % 31.0) + 1.0).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, N).expect("alloc y");
    let mut dev_mask: DeviceBuffer<Bool> = DeviceBuffer::zeros(&ctx, N).expect("alloc mask");
    let mut dev_ws: DeviceBuffer<u8> =
        DeviceBuffer::zeros(&ctx, N * 4).expect("alloc workspace");

    let desc = DropoutDescriptor {
        shape,
        element: ElementKind::F64,
        p: 0.5,
        seed: 0xC0DE_C0DE_F00D_F00D,
    };
    let plan = DropoutPlan::<f64, 1>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let args = DropoutArgs::<f64, 1> {
        x: TensorRef { data: dev_x.as_slice(), shape, stride },
        y: TensorMut { data: dev_y.as_slice_mut(), shape, stride },
        mask: TensorMut { data: dev_mask.as_slice_mut(), shape, stride },
    };
    plan.run(&stream, Workspace::Borrowed(dev_ws.as_slice_mut()), args).unwrap();
    stream.synchronize().unwrap();

    let mut host_y = vec![0f64; N];
    let mut host_mask = vec![Bool(0); N];
    dev_y.copy_to_host(&mut host_y).unwrap();
    dev_mask.copy_to_host(&mut host_mask).unwrap();

    let scale = 1.0_f64 / 0.5;
    let mut kept = 0usize;
    for (i, (xi, (yi, mi))) in host_x.iter().zip(host_y.iter().zip(host_mask.iter())).enumerate() {
        if mi.0 == 1 {
            kept += 1;
            let expected = xi * scale;
            assert!(
                (yi - expected).abs() <= expected.abs() * 1e-12 + 1e-12,
                "mask=1 cell {i}: y = {yi}, expected {expected}"
            );
        } else {
            assert_eq!(*yi, 0.0, "mask=0 cell {i}: y = {yi}, expected 0");
        }
    }
    let frac = kept as f64 / N as f64;
    assert!((frac - 0.5).abs() < 5e-3, "kept fraction = {frac}");
}
