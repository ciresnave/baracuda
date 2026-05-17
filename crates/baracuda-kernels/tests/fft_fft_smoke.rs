//! Real-GPU smoke test for `FftPlan` (cuFFT C2C wrap).
//!
//! Round-trip identity: `IFFT(FFT(x)) ≈ x` to within `1e-4` (f32) /
//! `1e-12` (f64) per cell. Forward + inverse share the same plan
//! shape; the inverse branch applies the `1/n` normalization at the
//! plan layer so the round-trip is the identity.
//!
//! `#[ignore]` by default — requires a real CUDA device. Run with:
//!
//! ```text
//! cargo test -p baracuda-kernels --release --features sm89 \
//!     --test fft_fft_smoke -- --ignored
//! ```

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Complex32, Complex64, ElementKind, FftArgs, FftDescriptor, FftPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};

const N: i32 = 8;
const BATCH: i32 = 4;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn fft_ifft_roundtrip_complex32() {
    let (ctx, stream) = setup();

    let total = (BATCH * N) as usize;
    // Build a deterministic complex input — `x[b, i] = (i + 0.5*b, -i)`.
    let mut x_host = vec![Complex32::default(); total];
    for b in 0..BATCH as usize {
        for i in 0..N as usize {
            x_host[b * N as usize + i] =
                Complex32::new(i as f32 + 0.5 * b as f32, -(i as f32));
        }
    }

    let mut dev_x: DeviceBuffer<Complex32> =
        DeviceBuffer::from_slice(&ctx, &x_host).expect("upload x");
    let mut dev_y: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc y");
    let mut dev_xr: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc xr");

    let shape = [BATCH, N];
    let stride = contiguous_stride(shape);

    // Forward FFT — x → y.
    let fwd_desc = FftDescriptor {
        n: N,
        batch: BATCH,
        inverse: false,
        element: ElementKind::Complex32,
    };
    let fwd_plan = FftPlan::<Complex32>::select(&stream, &fwd_desc, PlanPreference::default())
        .expect("select fwd plan");
    {
        let args = FftArgs::<Complex32> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape,
                stride,
            },
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape,
                stride,
            },
        };
        fwd_plan
            .run(&stream, Workspace::None, args)
            .expect("run fwd fft");
    }

    // Inverse FFT — y → xr.
    let inv_desc = FftDescriptor {
        n: N,
        batch: BATCH,
        inverse: true,
        element: ElementKind::Complex32,
    };
    let inv_plan = FftPlan::<Complex32>::select(&stream, &inv_desc, PlanPreference::default())
        .expect("select inv plan");
    {
        let args = FftArgs::<Complex32> {
            x: TensorRef {
                data: dev_y.as_slice(),
                shape,
                stride,
            },
            y: TensorMut {
                data: dev_xr.as_slice_mut(),
                shape,
                stride,
            },
        };
        inv_plan
            .run(&stream, Workspace::None, args)
            .expect("run inv fft");
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
fn fft_ifft_roundtrip_complex64() {
    let (ctx, stream) = setup();

    let total = (BATCH * N) as usize;
    let mut x_host = vec![Complex64::default(); total];
    for b in 0..BATCH as usize {
        for i in 0..N as usize {
            x_host[b * N as usize + i] =
                Complex64::new(i as f64 + 0.5 * b as f64, -(i as f64));
        }
    }

    let mut dev_x: DeviceBuffer<Complex64> =
        DeviceBuffer::from_slice(&ctx, &x_host).expect("upload x");
    let mut dev_y: DeviceBuffer<Complex64> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc y");
    let mut dev_xr: DeviceBuffer<Complex64> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc xr");

    let shape = [BATCH, N];
    let stride = contiguous_stride(shape);

    let fwd_desc = FftDescriptor {
        n: N,
        batch: BATCH,
        inverse: false,
        element: ElementKind::Complex64,
    };
    let fwd_plan = FftPlan::<Complex64>::select(&stream, &fwd_desc, PlanPreference::default())
        .expect("select fwd plan");
    {
        let args = FftArgs::<Complex64> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape,
                stride,
            },
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape,
                stride,
            },
        };
        fwd_plan
            .run(&stream, Workspace::None, args)
            .expect("run fwd fft f64");
    }

    let inv_desc = FftDescriptor {
        n: N,
        batch: BATCH,
        inverse: true,
        element: ElementKind::Complex64,
    };
    let inv_plan = FftPlan::<Complex64>::select(&stream, &inv_desc, PlanPreference::default())
        .expect("select inv plan");
    {
        let args = FftArgs::<Complex64> {
            x: TensorRef {
                data: dev_y.as_slice(),
                shape,
                stride,
            },
            y: TensorMut {
                data: dev_xr.as_slice_mut(),
                shape,
                stride,
            },
        };
        inv_plan
            .run(&stream, Workspace::None, args)
            .expect("run inv fft f64");
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
fn fft_forward_constant_signal() {
    // FFT of a constant real signal (cast into the complex slot) yields
    // a single non-zero bin at k=0 with value `n * c`; all other bins
    // are zero. Validates that the forward direction is unnormalized
    // (no implicit divide) and the batch dimension threads through.
    let (ctx, stream) = setup();
    let total = (BATCH * N) as usize;
    let mut x_host = vec![Complex32::default(); total];
    for b in 0..BATCH as usize {
        let c = (b as f32) + 1.0;
        for i in 0..N as usize {
            x_host[b * N as usize + i] = Complex32::new(c, 0.0);
        }
    }
    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_y: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total).expect("alloc y");

    let shape = [BATCH, N];
    let stride = contiguous_stride(shape);
    let desc = FftDescriptor {
        n: N,
        batch: BATCH,
        inverse: false,
        element: ElementKind::Complex32,
    };
    let plan =
        FftPlan::<Complex32>::select(&stream, &desc, PlanPreference::default()).expect("select");
    let args = FftArgs::<Complex32> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![Complex32::default(); total];
    dev_y.copy_to_host(&mut got).expect("download");

    for b in 0..BATCH as usize {
        let c = (b as f32) + 1.0;
        let expected_dc = (N as f32) * c;
        assert!(
            (got[b * N as usize].re - expected_dc).abs() < 1e-3,
            "batch {b} DC bin: got {} want {}",
            got[b * N as usize].re,
            expected_dc
        );
        for i in 1..N as usize {
            let v = got[b * N as usize + i];
            assert!(
                v.re.abs() < 1e-3 && v.im.abs() < 1e-3,
                "batch {b} bin {i}: expected zero, got ({}, {})",
                v.re,
                v.im
            );
        }
    }
}
