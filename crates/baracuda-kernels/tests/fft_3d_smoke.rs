//! Real-GPU smoke tests for `FftNdPlan` (rank 3, C2C) — Milestone 6.8.
//!
//! Round-trip identity: `IFFT3(FFT3(x)) ≈ x`. Same harness shape as
//! the 2-D suite (`fft_2d_smoke`); rank-3 exercises cuFFT's
//! `cufftPlanMany` with three transform axes per batched entry.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    Complex32, Complex64, ElementKind, FftNdArgs, FftNdDescriptor, FftNdPlan, PlanPreference,
    Workspace,
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
fn fft3_ifft3_roundtrip_complex32() {
    let (ctx, stream) = setup();

    let per = (D * H * W) as usize;
    let total = (BATCH as usize) * per;
    let mut x_host = vec![Complex32::default(); total];
    for b in 0..BATCH as usize {
        for k in 0..D as usize {
            for i in 0..H as usize {
                for j in 0..W as usize {
                    let idx =
                        b * per + k * (H * W) as usize + i * W as usize + j;
                    x_host[idx] = Complex32::new(
                        (i as f32) + 0.25 * (j as f32) + 0.125 * (k as f32)
                            + 0.5 * (b as f32),
                        -(j as f32),
                    );
                }
            }
        }
    }

    let mut dev_x: DeviceBuffer<Complex32> =
        DeviceBuffer::from_slice(&ctx, &x_host).expect("upload x");
    let mut dev_y: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc y");
    let mut dev_xr: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc xr");

    let fwd_desc = FftNdDescriptor {
        dims: [D, H, W, 0],
        rank: 3,
        batch: BATCH,
        inverse: false,
        element: ElementKind::Complex32,
    };
    let fwd_plan = FftNdPlan::<Complex32>::select(&stream, &fwd_desc, PlanPreference::default())
        .expect("select fwd plan");
    {
        let args = FftNdArgs::<Complex32> {
            x: dev_x.as_slice(),
            y: dev_y.as_slice_mut(),
        };
        fwd_plan
            .run(&stream, Workspace::None, args)
            .expect("run fwd fft3");
    }

    let inv_desc = FftNdDescriptor {
        dims: [D, H, W, 0],
        rank: 3,
        batch: BATCH,
        inverse: true,
        element: ElementKind::Complex32,
    };
    let inv_plan = FftNdPlan::<Complex32>::select(&stream, &inv_desc, PlanPreference::default())
        .expect("select inv plan");
    {
        let args = FftNdArgs::<Complex32> {
            x: dev_y.as_slice(),
            y: dev_xr.as_slice_mut(),
        };
        inv_plan
            .run(&stream, Workspace::None, args)
            .expect("run inv fft3");
    }
    stream.synchronize().expect("sync");

    let mut got = vec![Complex32::default(); total];
    dev_xr.copy_to_host(&mut got).expect("download");

    for (idx, (got, want)) in got.iter().zip(x_host.iter()).enumerate() {
        let dre = (got.re - want.re).abs();
        let dim = (got.im - want.im).abs();
        assert!(
            dre < 1e-3 && dim < 1e-3,
            "[{idx}] roundtrip mismatch: got ({}, {}) want ({}, {})",
            got.re,
            got.im,
            want.re,
            want.im
        );
    }
}

#[test]
#[ignore]
fn fft3_ifft3_roundtrip_complex64() {
    let (ctx, stream) = setup();

    let per = (D * H * W) as usize;
    let total = (BATCH as usize) * per;
    let mut x_host = vec![Complex64::default(); total];
    for b in 0..BATCH as usize {
        for k in 0..D as usize {
            for i in 0..H as usize {
                for j in 0..W as usize {
                    let idx =
                        b * per + k * (H * W) as usize + i * W as usize + j;
                    x_host[idx] = Complex64::new(
                        (i as f64) + 0.25 * (j as f64) + 0.125 * (k as f64)
                            + 0.5 * (b as f64),
                        -(j as f64),
                    );
                }
            }
        }
    }

    let mut dev_x: DeviceBuffer<Complex64> =
        DeviceBuffer::from_slice(&ctx, &x_host).expect("upload x");
    let mut dev_y: DeviceBuffer<Complex64> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc y");
    let mut dev_xr: DeviceBuffer<Complex64> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc xr");

    let fwd_desc = FftNdDescriptor {
        dims: [D, H, W, 0],
        rank: 3,
        batch: BATCH,
        inverse: false,
        element: ElementKind::Complex64,
    };
    let fwd_plan = FftNdPlan::<Complex64>::select(&stream, &fwd_desc, PlanPreference::default())
        .expect("select fwd plan");
    {
        let args = FftNdArgs::<Complex64> {
            x: dev_x.as_slice(),
            y: dev_y.as_slice_mut(),
        };
        fwd_plan
            .run(&stream, Workspace::None, args)
            .expect("run fwd fft3 f64");
    }

    let inv_desc = FftNdDescriptor {
        dims: [D, H, W, 0],
        rank: 3,
        batch: BATCH,
        inverse: true,
        element: ElementKind::Complex64,
    };
    let inv_plan = FftNdPlan::<Complex64>::select(&stream, &inv_desc, PlanPreference::default())
        .expect("select inv plan");
    {
        let args = FftNdArgs::<Complex64> {
            x: dev_y.as_slice(),
            y: dev_xr.as_slice_mut(),
        };
        inv_plan
            .run(&stream, Workspace::None, args)
            .expect("run inv fft3 f64");
    }
    stream.synchronize().expect("sync");

    let mut got = vec![Complex64::default(); total];
    dev_xr.copy_to_host(&mut got).expect("download");

    for (idx, (got, want)) in got.iter().zip(x_host.iter()).enumerate() {
        let dre = (got.re - want.re).abs();
        let dim = (got.im - want.im).abs();
        assert!(
            dre < 1e-10 && dim < 1e-10,
            "[{idx}] roundtrip mismatch: got ({}, {}) want ({}, {})",
            got.re,
            got.im,
            want.re,
            want.im
        );
    }
}
