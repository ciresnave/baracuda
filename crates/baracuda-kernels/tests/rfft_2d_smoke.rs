//! Real-GPU smoke tests for `RfftNdPlan` + `IrfftNdPlan` (rank 2,
//! R2C / C2R) — Milestone 6.8.
//!
//! Round-trip identity: `IRFFT2(RFFT2(x)) ≈ x` to within `1e-4` (f32) /
//! `1e-12` (f64). The complex-side last-axis extent is the
//! Hermitian-half (`W / 2 + 1`).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    Complex32, Complex64, ElementKind, IrfftNdArgs, IrfftNdDescriptor, IrfftNdPlan, PlanPreference,
    RfftNdArgs, RfftNdDescriptor, RfftNdPlan, Workspace,
};

const H: i32 = 4;
const W: i32 = 8;
const BATCH: i32 = 2;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn rfft2_irfft2_roundtrip_f32() {
    let (ctx, stream) = setup();
    let per_real = (H * W) as usize;
    let per_complex = (H * (W / 2 + 1)) as usize;
    let total_real = (BATCH as usize) * per_real;
    let total_complex = (BATCH as usize) * per_complex;

    let mut x_host = vec![0f32; total_real];
    for b in 0..BATCH as usize {
        for i in 0..H as usize {
            for j in 0..W as usize {
                x_host[b * per_real + i * W as usize + j] =
                    (i as f32) - 0.25 * (j as f32) + 0.5 * (b as f32);
            }
        }
    }

    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload x");
    let mut dev_y: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total_complex).expect("alloc y");
    let mut dev_xr: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, total_real).expect("alloc xr");

    let fwd_desc = RfftNdDescriptor {
        dims: [H, W, 0, 0],
        rank: 2,
        batch: BATCH,
        element: ElementKind::F32,
    };
    let fwd_plan = RfftNdPlan::<f32>::select(&stream, &fwd_desc, PlanPreference::default())
        .expect("select rfft2");
    {
        let args = RfftNdArgs::<f32, Complex32> {
            x: dev_x.as_slice(),
            y: dev_y.as_slice_mut(),
        };
        fwd_plan
            .run(&stream, Workspace::None, args)
            .expect("run rfft2");
    }

    let inv_desc = IrfftNdDescriptor {
        dims: [H, W, 0, 0],
        rank: 2,
        batch: BATCH,
        element: ElementKind::F32,
    };
    let inv_plan = IrfftNdPlan::<f32>::select(&stream, &inv_desc, PlanPreference::default())
        .expect("select irfft2");
    {
        let args = IrfftNdArgs::<f32, Complex32> {
            x: dev_y.as_slice(),
            y: dev_xr.as_slice_mut(),
        };
        inv_plan
            .run(&stream, Workspace::None, args)
            .expect("run irfft2");
    }
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; total_real];
    dev_xr.copy_to_host(&mut got).expect("download");
    for (idx, (g, w)) in got.iter().zip(x_host.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "[{idx}] roundtrip mismatch: got {g} want {w}"
        );
    }
}

#[test]
#[ignore]
fn rfft2_irfft2_roundtrip_f64() {
    let (ctx, stream) = setup();
    let per_real = (H * W) as usize;
    let per_complex = (H * (W / 2 + 1)) as usize;
    let total_real = (BATCH as usize) * per_real;
    let total_complex = (BATCH as usize) * per_complex;

    let mut x_host = vec![0f64; total_real];
    for b in 0..BATCH as usize {
        for i in 0..H as usize {
            for j in 0..W as usize {
                x_host[b * per_real + i * W as usize + j] =
                    (i as f64) - 0.25 * (j as f64) + 0.5 * (b as f64);
            }
        }
    }

    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload x");
    let mut dev_y: DeviceBuffer<Complex64> =
        DeviceBuffer::zeros(&ctx, total_complex).expect("alloc y");
    let mut dev_xr: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, total_real).expect("alloc xr");

    let fwd_desc = RfftNdDescriptor {
        dims: [H, W, 0, 0],
        rank: 2,
        batch: BATCH,
        element: ElementKind::F64,
    };
    let fwd_plan = RfftNdPlan::<f64>::select(&stream, &fwd_desc, PlanPreference::default())
        .expect("select rfft2 f64");
    {
        let args = RfftNdArgs::<f64, Complex64> {
            x: dev_x.as_slice(),
            y: dev_y.as_slice_mut(),
        };
        fwd_plan
            .run(&stream, Workspace::None, args)
            .expect("run rfft2 f64");
    }

    let inv_desc = IrfftNdDescriptor {
        dims: [H, W, 0, 0],
        rank: 2,
        batch: BATCH,
        element: ElementKind::F64,
    };
    let inv_plan = IrfftNdPlan::<f64>::select(&stream, &inv_desc, PlanPreference::default())
        .expect("select irfft2 f64");
    {
        let args = IrfftNdArgs::<f64, Complex64> {
            x: dev_y.as_slice(),
            y: dev_xr.as_slice_mut(),
        };
        inv_plan
            .run(&stream, Workspace::None, args)
            .expect("run irfft2 f64");
    }
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; total_real];
    dev_xr.copy_to_host(&mut got).expect("download");
    for (idx, (g, w)) in got.iter().zip(x_host.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-12,
            "[{idx}] roundtrip mismatch: got {g} want {w}"
        );
    }
}
