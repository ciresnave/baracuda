//! Real-GPU smoke test for `RfftPlan` + `IrfftPlan` (cuFFT R2C / C2R wrap).
//!
//! Round-trip identity: `IRFFT(RFFT(x)) ≈ x` to within `1e-4` (f32) /
//! `1e-12` (f64) per cell. `RfftPlan` produces a Hermitian-half output;
//! `IrfftPlan` consumes the same and applies `1/n` normalization to
//! match PyTorch's `norm="backward"`.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Complex32, Complex64, ElementKind, IrfftArgs, IrfftDescriptor, IrfftPlan,
    PlanPreference, RfftArgs, RfftDescriptor, RfftPlan, TensorMut, TensorRef, Workspace,
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
fn rfft_irfft_roundtrip_f32() {
    let (ctx, stream) = setup();

    let total_real = (BATCH * N) as usize;
    let total_freq = (BATCH * (N / 2 + 1)) as usize;
    let mut x_host = vec![0f32; total_real];
    for b in 0..BATCH as usize {
        for i in 0..N as usize {
            x_host[b * N as usize + i] = (i as f32) - 0.25 * (b as f32);
        }
    }

    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload x");
    let mut dev_y: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total_freq).expect("alloc y");
    let mut dev_xr: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, total_real).expect("alloc xr");

    let real_shape = [BATCH, N];
    let real_stride = contiguous_stride(real_shape);
    let freq_shape = [BATCH, N / 2 + 1];
    let freq_stride = contiguous_stride(freq_shape);

    // Forward RFFT.
    let fwd_desc = RfftDescriptor {
        n: N,
        batch: BATCH,
        element: ElementKind::F32,
    };
    let fwd_plan =
        RfftPlan::<f32>::select(&stream, &fwd_desc, PlanPreference::default()).expect("select rfft");
    {
        let args = RfftArgs::<f32, Complex32> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: real_shape,
                stride: real_stride,
            },
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: freq_shape,
                stride: freq_stride,
            },
        };
        fwd_plan
            .run(&stream, Workspace::None, args)
            .expect("run rfft");
    }

    // Inverse IRFFT.
    let inv_desc = IrfftDescriptor {
        n: N,
        batch: BATCH,
        element: ElementKind::F32,
    };
    let inv_plan = IrfftPlan::<f32>::select(&stream, &inv_desc, PlanPreference::default())
        .expect("select irfft");
    {
        let args = IrfftArgs::<f32, Complex32> {
            x: TensorRef {
                data: dev_y.as_slice(),
                shape: freq_shape,
                stride: freq_stride,
            },
            y: TensorMut {
                data: dev_xr.as_slice_mut(),
                shape: real_shape,
                stride: real_stride,
            },
        };
        inv_plan
            .run(&stream, Workspace::None, args)
            .expect("run irfft");
    }
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; total_real];
    dev_xr.copy_to_host(&mut got).expect("download");

    for (idx, (got, want)) in got.iter().zip(x_host.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-4,
            "[{idx}] rfft/irfft mismatch: got {} want {}",
            got,
            want
        );
    }
}

#[test]
#[ignore]
fn rfft_irfft_roundtrip_f64() {
    let (ctx, stream) = setup();

    let total_real = (BATCH * N) as usize;
    let total_freq = (BATCH * (N / 2 + 1)) as usize;
    let mut x_host = vec![0f64; total_real];
    for b in 0..BATCH as usize {
        for i in 0..N as usize {
            x_host[b * N as usize + i] = (i as f64).cos() + 0.25 * (b as f64);
        }
    }

    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload x");
    let mut dev_y: DeviceBuffer<Complex64> =
        DeviceBuffer::zeros(&ctx, total_freq).expect("alloc y");
    let mut dev_xr: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, total_real).expect("alloc xr");

    let real_shape = [BATCH, N];
    let real_stride = contiguous_stride(real_shape);
    let freq_shape = [BATCH, N / 2 + 1];
    let freq_stride = contiguous_stride(freq_shape);

    let fwd_desc = RfftDescriptor {
        n: N,
        batch: BATCH,
        element: ElementKind::F64,
    };
    let fwd_plan =
        RfftPlan::<f64>::select(&stream, &fwd_desc, PlanPreference::default()).expect("select rfft");
    {
        let args = RfftArgs::<f64, Complex64> {
            x: TensorRef {
                data: dev_x.as_slice(),
                shape: real_shape,
                stride: real_stride,
            },
            y: TensorMut {
                data: dev_y.as_slice_mut(),
                shape: freq_shape,
                stride: freq_stride,
            },
        };
        fwd_plan
            .run(&stream, Workspace::None, args)
            .expect("run rfft f64");
    }

    let inv_desc = IrfftDescriptor {
        n: N,
        batch: BATCH,
        element: ElementKind::F64,
    };
    let inv_plan = IrfftPlan::<f64>::select(&stream, &inv_desc, PlanPreference::default())
        .expect("select irfft");
    {
        let args = IrfftArgs::<f64, Complex64> {
            x: TensorRef {
                data: dev_y.as_slice(),
                shape: freq_shape,
                stride: freq_stride,
            },
            y: TensorMut {
                data: dev_xr.as_slice_mut(),
                shape: real_shape,
                stride: real_stride,
            },
        };
        inv_plan
            .run(&stream, Workspace::None, args)
            .expect("run irfft f64");
    }
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; total_real];
    dev_xr.copy_to_host(&mut got).expect("download");

    for (idx, (got, want)) in got.iter().zip(x_host.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "[{idx}] rfft/irfft f64 mismatch: got {} want {}",
            got,
            want
        );
    }
}

#[test]
#[ignore]
fn rfft_dc_bin_equals_sum() {
    // For real input the DC bin (k=0) of RFFT equals the sum of the
    // signal (no normalization on forward). Sanity-check that the
    // R2C path emits the expected Hermitian-half layout.
    let (ctx, stream) = setup();
    let total_real = (BATCH * N) as usize;
    let total_freq = (BATCH * (N / 2 + 1)) as usize;
    let mut x_host = vec![0f32; total_real];
    for b in 0..BATCH as usize {
        for i in 0..N as usize {
            x_host[b * N as usize + i] = (i as f32) + 1.0;
        }
    }
    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_y: DeviceBuffer<Complex32> =
        DeviceBuffer::zeros(&ctx, total_freq).expect("alloc y");

    let real_shape = [BATCH, N];
    let real_stride = contiguous_stride(real_shape);
    let freq_shape = [BATCH, N / 2 + 1];
    let freq_stride = contiguous_stride(freq_shape);

    let desc = RfftDescriptor {
        n: N,
        batch: BATCH,
        element: ElementKind::F32,
    };
    let plan =
        RfftPlan::<f32>::select(&stream, &desc, PlanPreference::default()).expect("select rfft");
    let args = RfftArgs::<f32, Complex32> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: real_shape,
            stride: real_stride,
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: freq_shape,
            stride: freq_stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run rfft");
    stream.synchronize().expect("sync");

    let mut got = vec![Complex32::default(); total_freq];
    dev_y.copy_to_host(&mut got).expect("download");

    for b in 0..BATCH as usize {
        let expected: f32 = (0..N as usize).map(|i| (i as f32) + 1.0).sum();
        let dc = got[b * (N as usize / 2 + 1)];
        assert!(
            (dc.re - expected).abs() < 1e-3 && dc.im.abs() < 1e-3,
            "batch {b} DC: got ({}, {}) want ({}, 0)",
            dc.re,
            dc.im,
            expected
        );
    }
}
