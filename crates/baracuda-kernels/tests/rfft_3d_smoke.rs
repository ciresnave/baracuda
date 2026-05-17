//! Real-GPU smoke tests for `RfftNdPlan` + `IrfftNdPlan` (rank 3,
//! R2C / C2R) — Milestone 6.8.
//!
//! Round-trip identity: `IRFFT3(RFFT3(x)) ≈ x`.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    Complex32, Complex64, ElementKind, IrfftNdArgs, IrfftNdDescriptor, IrfftNdPlan, PlanPreference,
    RfftNdArgs, RfftNdDescriptor, RfftNdPlan, Workspace,
};

const D: i32 = 4;
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
fn rfft3_irfft3_roundtrip_f32() {
    let (ctx, stream) = setup();
    let per_real = (D * H * W) as usize;
    let per_complex = (D * H * (W / 2 + 1)) as usize;
    let total_real = (BATCH as usize) * per_real;
    let total_complex = (BATCH as usize) * per_complex;

    let mut x_host = vec![0f32; total_real];
    for b in 0..BATCH as usize {
        for k in 0..D as usize {
            for i in 0..H as usize {
                for j in 0..W as usize {
                    let idx = b * per_real + k * (H * W) as usize + i * W as usize + j;
                    x_host[idx] = (i as f32) - 0.125 * (j as f32) + 0.25 * (k as f32)
                        + 0.5 * (b as f32);
                }
            }
        }
    }

    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload x");
    let mut dev_y: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total_complex).expect("alloc y");
    let mut dev_xr: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, total_real).expect("alloc xr");

    let fwd_desc = RfftNdDescriptor {
        dims: [D, H, W, 0],
        rank: 3,
        batch: BATCH,
        element: ElementKind::F32,
    };
    let fwd_plan = RfftNdPlan::<f32>::select(&stream, &fwd_desc, PlanPreference::default())
        .expect("select rfft3");
    {
        let args = RfftNdArgs::<f32, Complex32> {
            x: dev_x.as_slice(),
            y: dev_y.as_slice_mut(),
        };
        fwd_plan
            .run(&stream, Workspace::None, args)
            .expect("run rfft3");
    }

    let inv_desc = IrfftNdDescriptor {
        dims: [D, H, W, 0],
        rank: 3,
        batch: BATCH,
        element: ElementKind::F32,
    };
    let inv_plan = IrfftNdPlan::<f32>::select(&stream, &inv_desc, PlanPreference::default())
        .expect("select irfft3");
    {
        let args = IrfftNdArgs::<f32, Complex32> {
            x: dev_y.as_slice(),
            y: dev_xr.as_slice_mut(),
        };
        inv_plan
            .run(&stream, Workspace::None, args)
            .expect("run irfft3");
    }
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; total_real];
    dev_xr.copy_to_host(&mut got).expect("download");
    for (idx, (g, w)) in got.iter().zip(x_host.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-3,
            "[{idx}] roundtrip mismatch: got {g} want {w}"
        );
    }
}

#[test]
#[ignore]
fn rfft3_irfft3_roundtrip_f64() {
    let (ctx, stream) = setup();
    let per_real = (D * H * W) as usize;
    let per_complex = (D * H * (W / 2 + 1)) as usize;
    let total_real = (BATCH as usize) * per_real;
    let total_complex = (BATCH as usize) * per_complex;

    let mut x_host = vec![0f64; total_real];
    for b in 0..BATCH as usize {
        for k in 0..D as usize {
            for i in 0..H as usize {
                for j in 0..W as usize {
                    let idx = b * per_real + k * (H * W) as usize + i * W as usize + j;
                    x_host[idx] = (i as f64) - 0.125 * (j as f64) + 0.25 * (k as f64)
                        + 0.5 * (b as f64);
                }
            }
        }
    }

    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload x");
    let mut dev_y: DeviceBuffer<Complex64> =
        DeviceBuffer::zeros(&ctx, total_complex).expect("alloc y");
    let mut dev_xr: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, total_real).expect("alloc xr");

    let fwd_desc = RfftNdDescriptor {
        dims: [D, H, W, 0],
        rank: 3,
        batch: BATCH,
        element: ElementKind::F64,
    };
    let fwd_plan = RfftNdPlan::<f64>::select(&stream, &fwd_desc, PlanPreference::default())
        .expect("select rfft3 f64");
    {
        let args = RfftNdArgs::<f64, Complex64> {
            x: dev_x.as_slice(),
            y: dev_y.as_slice_mut(),
        };
        fwd_plan
            .run(&stream, Workspace::None, args)
            .expect("run rfft3 f64");
    }

    let inv_desc = IrfftNdDescriptor {
        dims: [D, H, W, 0],
        rank: 3,
        batch: BATCH,
        element: ElementKind::F64,
    };
    let inv_plan = IrfftNdPlan::<f64>::select(&stream, &inv_desc, PlanPreference::default())
        .expect("select irfft3 f64");
    {
        let args = IrfftNdArgs::<f64, Complex64> {
            x: dev_y.as_slice(),
            y: dev_xr.as_slice_mut(),
        };
        inv_plan
            .run(&stream, Workspace::None, args)
            .expect("run irfft3 f64");
    }
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; total_real];
    dev_xr.copy_to_host(&mut got).expect("download");
    for (idx, (g, w)) in got.iter().zip(x_host.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-10,
            "[{idx}] roundtrip mismatch: got {g} want {w}"
        );
    }
}
