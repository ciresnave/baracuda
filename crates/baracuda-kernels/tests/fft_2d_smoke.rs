//! Real-GPU smoke tests for `FftNdPlan` (rank 2, C2C) — Milestone 6.8.
//!
//! Round-trip identity: `IFFT2(FFT2(x)) ≈ x` to within `1e-4` (f32) /
//! `1e-12` (f64) per cell. The plan's inverse branch applies the
//! `1/(H*W)` normalization at the plan layer so the round-trip is the
//! identity.
//!
//! `#[ignore]` by default — requires a real CUDA device. Run with:
//!
//! ```text
//! cargo test -p baracuda-kernels --release --features sm89 \
//!     --test fft_2d_smoke -- --ignored
//! ```

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    Complex32, Complex64, ElementKind, FftNdArgs, FftNdDescriptor, FftNdPlan, PlanPreference,
    Workspace,
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
fn fft2_ifft2_roundtrip_complex32() {
    let (ctx, stream) = setup();

    let per = (H * W) as usize;
    let total = (BATCH as usize) * per;
    let mut x_host = vec![Complex32::default(); total];
    for b in 0..BATCH as usize {
        for i in 0..H as usize {
            for j in 0..W as usize {
                let idx = b * per + i * W as usize + j;
                x_host[idx] = Complex32::new(
                    (i as f32) + 0.25 * (j as f32) + 0.5 * (b as f32),
                    -(j as f32),
                );
            }
        }
    }

    let mut dev_x: DeviceBuffer<Complex32> =
        DeviceBuffer::from_slice(&ctx, &x_host).expect("upload x");
    let mut dev_y: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc y");
    let mut dev_xr: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc xr");

    // Forward FFT2 — x → y.
    let fwd_desc = FftNdDescriptor {
        dims: [H, W, 0, 0],
        rank: 2,
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
            .expect("run fwd fft2");
    }

    // Inverse FFT2 — y → xr.
    let inv_desc = FftNdDescriptor {
        dims: [H, W, 0, 0],
        rank: 2,
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
            .expect("run inv fft2");
    }
    stream.synchronize().expect("sync");

    let mut got = vec![Complex32::default(); total];
    dev_xr.copy_to_host(&mut got).expect("download");

    for (idx, (got, want)) in got.iter().zip(x_host.iter()).enumerate() {
        let dre = (got.re - want.re).abs();
        let dim = (got.im - want.im).abs();
        assert!(
            dre < 1e-4 && dim < 1e-4,
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
fn fft2_ifft2_roundtrip_complex64() {
    let (ctx, stream) = setup();

    let per = (H * W) as usize;
    let total = (BATCH as usize) * per;
    let mut x_host = vec![Complex64::default(); total];
    for b in 0..BATCH as usize {
        for i in 0..H as usize {
            for j in 0..W as usize {
                let idx = b * per + i * W as usize + j;
                x_host[idx] = Complex64::new(
                    (i as f64) + 0.25 * (j as f64) + 0.5 * (b as f64),
                    -(j as f64),
                );
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
        dims: [H, W, 0, 0],
        rank: 2,
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
            .expect("run fwd fft2 f64");
    }

    let inv_desc = FftNdDescriptor {
        dims: [H, W, 0, 0],
        rank: 2,
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
            .expect("run inv fft2 f64");
    }
    stream.synchronize().expect("sync");

    let mut got = vec![Complex64::default(); total];
    dev_xr.copy_to_host(&mut got).expect("download");

    for (idx, (got, want)) in got.iter().zip(x_host.iter()).enumerate() {
        let dre = (got.re - want.re).abs();
        let dim = (got.im - want.im).abs();
        assert!(
            dre < 1e-12 && dim < 1e-12,
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
fn fft2_forward_constant_signal() {
    // FFT2 of a constant signal produces a single non-zero bin at
    // (0,0) with value `H * W * c`, all other bins zero. Confirms
    // the forward direction is unnormalized + the batch dim threads.
    let (ctx, stream) = setup();
    let per = (H * W) as usize;
    let total = (BATCH as usize) * per;
    let mut x_host = vec![Complex32::default(); total];
    for b in 0..BATCH as usize {
        let c = (b as f32) + 1.0;
        for k in 0..per {
            x_host[b * per + k] = Complex32::new(c, 0.0);
        }
    }
    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_y: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc y");

    let desc = FftNdDescriptor {
        dims: [H, W, 0, 0],
        rank: 2,
        batch: BATCH,
        inverse: false,
        element: ElementKind::Complex32,
    };
    let plan = FftNdPlan::<Complex32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = FftNdArgs::<Complex32> {
        x: dev_x.as_slice(),
        y: dev_y.as_slice_mut(),
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![Complex32::default(); total];
    dev_y.copy_to_host(&mut got).expect("dl");

    for b in 0..BATCH as usize {
        let c = (b as f32) + 1.0;
        let expected_dc = (H as f32) * (W as f32) * c;
        let dc = got[b * per];
        assert!(
            (dc.re - expected_dc).abs() < 1e-2,
            "batch {b} DC bin: got {} want {}",
            dc.re,
            expected_dc
        );
        for k in 1..per {
            let v = got[b * per + k];
            assert!(
                v.re.abs() < 1e-2 && v.im.abs() < 1e-2,
                "batch {b} bin {k}: expected zero, got ({}, {})",
                v.re,
                v.im
            );
        }
    }
}
