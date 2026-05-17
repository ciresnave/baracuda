//! Real-GPU smoke test for `FftShiftPlan` (bespoke fftshift / ifftshift).
//!
//! Bit-exact — fftshift is a pure index permutation, no arithmetic
//! involved. The classic [0, 1, 2, 3] → [2, 3, 0, 1] permutation is
//! the canonical test; we also verify the odd-n behavior and that
//! `ifftshift(fftshift(x)) == x` for any `n`.
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Complex32, ElementKind, FftShiftArgs, FftShiftDescriptor, FftShiftPlan,
    PlanPreference, TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn fftshift_even_n_f32() {
    // [0, 1, 2, 3] -> [2, 3, 0, 1] (n=4).
    let (ctx, stream) = setup();
    let n: i32 = 4;
    let batch: i32 = 1;
    let x_host = vec![0f32, 1.0, 2.0, 3.0];
    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("alloc y");

    let shape = [batch, n];
    let stride = contiguous_stride(shape);
    let desc = FftShiftDescriptor {
        n,
        batch,
        inverse: false,
        element: ElementKind::F32,
    };
    let plan = FftShiftPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select fftshift");
    let args = FftShiftArgs::<f32> {
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
    plan.run(&stream, Workspace::None, args).expect("run shift");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 4];
    dev_y.copy_to_host(&mut got).expect("dl");
    assert_eq!(got, vec![2.0, 3.0, 0.0, 1.0]);
}

#[test]
#[ignore]
fn ifftshift_even_n_f32() {
    // For even n, ifftshift == fftshift.
    let (ctx, stream) = setup();
    let n: i32 = 4;
    let batch: i32 = 1;
    let x_host = vec![0f32, 1.0, 2.0, 3.0];
    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("alloc y");

    let shape = [batch, n];
    let stride = contiguous_stride(shape);
    let desc = FftShiftDescriptor {
        n,
        batch,
        inverse: true,
        element: ElementKind::F32,
    };
    let plan = FftShiftPlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select ifftshift");
    let args = FftShiftArgs::<f32> {
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
    plan.run(&stream, Workspace::None, args).expect("run shift");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 4];
    dev_y.copy_to_host(&mut got).expect("dl");
    assert_eq!(got, vec![2.0, 3.0, 0.0, 1.0]);
}

#[test]
#[ignore]
fn fftshift_odd_n_is_not_ifftshift() {
    // n=5 — NumPy / PyTorch convention.
    //   fftshift(x)  = roll(x,   n//2)  → y[i] = x[(i - n//2) % n]
    //                                  = x[(i + (n+1)/2) % n]
    //   ifftshift(x) = roll(x, -(n//2)) → y[i] = x[(i + n//2)     % n]
    //
    // For [0,1,2,3,4] this gives:
    //   fftshift  = [3, 4, 0, 1, 2]   (roll(+2))
    //   ifftshift = [2, 3, 4, 0, 1]   (roll(-2))
    //
    // Round-trip identity: `ifftshift(fftshift(x)) == x`. Verified by
    // the asymmetry between the two outputs for odd n.
    let (ctx, stream) = setup();
    let n: i32 = 5;
    let batch: i32 = 1;
    let x_host = vec![0f32, 1.0, 2.0, 3.0, 4.0];
    let dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 5).expect("alloc y");

    let shape = [batch, n];
    let stride = contiguous_stride(shape);

    // fftshift.
    let f_desc = FftShiftDescriptor {
        n,
        batch,
        inverse: false,
        element: ElementKind::F32,
    };
    let f_plan = FftShiftPlan::<f32>::select(&stream, &f_desc, PlanPreference::default())
        .expect("select fftshift");
    let args = FftShiftArgs::<f32> {
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
    f_plan
        .run(&stream, Workspace::None, args)
        .expect("run fftshift");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 5];
    dev_y.copy_to_host(&mut got).expect("dl");
    assert_eq!(got, vec![3.0, 4.0, 0.0, 1.0, 2.0]);

    // ifftshift: rebuild plan + buffer.
    let mut dev_y2: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 5).expect("alloc y2");
    let i_desc = FftShiftDescriptor {
        n,
        batch,
        inverse: true,
        element: ElementKind::F32,
    };
    let i_plan = FftShiftPlan::<f32>::select(&stream, &i_desc, PlanPreference::default())
        .expect("select ifftshift");
    let args2 = FftShiftArgs::<f32> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride,
        },
        y: TensorMut {
            data: dev_y2.as_slice_mut(),
            shape,
            stride,
        },
    };
    i_plan
        .run(&stream, Workspace::None, args2)
        .expect("run ifftshift");
    stream.synchronize().expect("sync");

    let mut got2 = vec![0f32; 5];
    dev_y2.copy_to_host(&mut got2).expect("dl");
    assert_eq!(got2, vec![2.0, 3.0, 4.0, 0.0, 1.0]);

    // Round-trip: ifftshift(fftshift(x)) == x.
    let mut dev_rt: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 5).expect("alloc rt");
    let args_rt = FftShiftArgs::<f32> {
        x: TensorRef {
            data: dev_y.as_slice(),
            shape,
            stride,
        },
        y: TensorMut {
            data: dev_rt.as_slice_mut(),
            shape,
            stride,
        },
    };
    i_plan
        .run(&stream, Workspace::None, args_rt)
        .expect("run ifftshift(fftshift)");
    stream.synchronize().expect("sync");

    let mut got_rt = vec![0f32; 5];
    dev_rt.copy_to_host(&mut got_rt).expect("dl rt");
    assert_eq!(got_rt, x_host);
}

#[test]
#[ignore]
fn fftshift_batched_complex32() {
    // Batch=2, n=4. Per-row shift, two rows independent.
    // row 0: [(0+0i), (1+0i), (2+0i), (3+0i)] -> [(2+0i), (3+0i), (0+0i), (1+0i)]
    // row 1: [(10+1i), (11+1i), (12+1i), (13+1i)] -> [(12+1i), (13+1i), (10+1i), (11+1i)]
    let (ctx, stream) = setup();
    let n: i32 = 4;
    let batch: i32 = 2;
    let x_host = vec![
        Complex32::new(0.0, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(10.0, 1.0),
        Complex32::new(11.0, 1.0),
        Complex32::new(12.0, 1.0),
        Complex32::new(13.0, 1.0),
    ];
    let mut dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_y: DeviceBuffer<Complex32> = DeviceBuffer::zeros(&ctx, 8).expect("alloc y");

    let shape = [batch, n];
    let stride = contiguous_stride(shape);
    let desc = FftShiftDescriptor {
        n,
        batch,
        inverse: false,
        element: ElementKind::Complex32,
    };
    let plan = FftShiftPlan::<Complex32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = FftShiftArgs::<Complex32> {
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

    let mut got = vec![Complex32::default(); 8];
    dev_y.copy_to_host(&mut got).expect("dl");
    let expected = vec![
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(1.0, 0.0),
        Complex32::new(12.0, 1.0),
        Complex32::new(13.0, 1.0),
        Complex32::new(10.0, 1.0),
        Complex32::new(11.0, 1.0),
    ];
    assert_eq!(got, expected);
}
