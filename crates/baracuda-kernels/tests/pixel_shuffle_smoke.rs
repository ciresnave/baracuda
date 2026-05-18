//! Real-GPU smoke for `PixelShufflePlan<T>` / `PixelUnshufflePlan<T>`
//! (Phase 9 Category T). Pure index permutation → bit-exact at every
//! dtype. Verifies inverse property (unshuffle(shuffle(x)) == x).
//! `#[ignore]` by default.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PixelShuffleArgs, PixelShuffleDescriptor, PixelShufflePlan,
    PixelUnshuffleArgs, PixelUnshuffleDescriptor, PixelUnshufflePlan, PlanPreference, TensorMut,
    TensorRef, Workspace,
};
use half::{bf16, f16};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

#[test]
#[ignore]
fn pixel_shuffle_roundtrip_f32() {
    let (ctx, stream) = setup();
    let (n, c, h, w, r) = (1, 2, 3, 3, 2);
    let cin = c * r * r;
    let host_in: Vec<f32> =
        (0..(n * cin * h * w)).map(|i| i as f32 * 0.5 + 1.0).collect();
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("");
    let mut dev_mid: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * h * r * w * r) as usize).expect("");
    let plan = PixelShufflePlan::<f32>::select(
        &stream,
        &PixelShuffleDescriptor {
            n, c, h, w, upscale_factor: r,
            element: ElementKind::F32,
        },
        PlanPreference::default(),
    ).expect("");
    plan.run(&stream, Workspace::None, PixelShuffleArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, cin, h, w],
            stride: contiguous_stride([n, cin, h, w]),
        },
        output: TensorMut {
            data: dev_mid.as_slice_mut(),
            shape: [n, c, h * r, w * r],
            stride: contiguous_stride([n, c, h * r, w * r]),
        },
    }).expect("");
    // Round-trip back.
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * cin * h * w) as usize).expect("");
    let uplan = PixelUnshufflePlan::<f32>::select(
        &stream,
        &PixelUnshuffleDescriptor {
            n, c, h, w, downscale_factor: r,
            element: ElementKind::F32,
        },
        PlanPreference::default(),
    ).expect("");
    uplan.run(&stream, Workspace::None, PixelUnshuffleArgs {
        input: TensorRef {
            data: dev_mid.as_slice(),
            shape: [n, c, h * r, w * r],
            stride: contiguous_stride([n, c, h * r, w * r]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n, cin, h, w],
            stride: contiguous_stride([n, cin, h, w]),
        },
    }).expect("");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; host_in.len()];
    dev_out.copy_to_host(&mut got).expect("");
    for (i, (g, e)) in got.iter().zip(host_in.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "round-trip f32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn pixel_shuffle_known_permutation_f32() {
    // Input: 1×4×1×1 (4 = 1·2·2) with channels {10, 20, 30, 40}.
    // Expected output: 1×1×2×2 with values laid out as
    //   [(0,0)=10, (0,1)=20, (1,0)=30, (1,1)=40]
    let (ctx, stream) = setup();
    let (n, c, h, w, r) = (1, 1, 1, 1, 2);
    let cin = c * r * r;
    let host_in: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("");
    let mut dev_out: DeviceBuffer<f32> =
        DeviceBuffer::zeros(&ctx, (n * c * h * r * w * r) as usize).expect("");
    let plan = PixelShufflePlan::<f32>::select(
        &stream,
        &PixelShuffleDescriptor {
            n, c, h, w, upscale_factor: r,
            element: ElementKind::F32,
        },
        PlanPreference::default(),
    ).expect("");
    plan.run(&stream, Workspace::None, PixelShuffleArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, cin, h, w],
            stride: contiguous_stride([n, cin, h, w]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n, c, h * r, w * r],
            stride: contiguous_stride([n, c, h * r, w * r]),
        },
    }).expect("");
    stream.synchronize().expect("sync");
    let mut got = vec![0f32; 4];
    dev_out.copy_to_host(&mut got).expect("");
    assert_eq!(got, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
#[ignore]
fn pixel_shuffle_roundtrip_f64() {
    let (ctx, stream) = setup();
    let (n, c, h, w, r) = (1, 1, 2, 2, 2);
    let cin = c * r * r;
    let host_in: Vec<f64> = (0..(n * cin * h * w)).map(|i| i as f64 * 0.5).collect();
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("");
    let mut dev_mid: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (n * c * h * r * w * r) as usize).expect("");
    PixelShufflePlan::<f64>::select(
        &stream,
        &PixelShuffleDescriptor {
            n, c, h, w, upscale_factor: r,
            element: ElementKind::F64,
        },
        PlanPreference::default(),
    ).expect("").run(&stream, Workspace::None, PixelShuffleArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, cin, h, w],
            stride: contiguous_stride([n, cin, h, w]),
        },
        output: TensorMut {
            data: dev_mid.as_slice_mut(),
            shape: [n, c, h * r, w * r],
            stride: contiguous_stride([n, c, h * r, w * r]),
        },
    }).expect("");
    let mut dev_out: DeviceBuffer<f64> =
        DeviceBuffer::zeros(&ctx, (n * cin * h * w) as usize).expect("");
    PixelUnshufflePlan::<f64>::select(
        &stream,
        &PixelUnshuffleDescriptor {
            n, c, h, w, downscale_factor: r,
            element: ElementKind::F64,
        },
        PlanPreference::default(),
    ).expect("").run(&stream, Workspace::None, PixelUnshuffleArgs {
        input: TensorRef {
            data: dev_mid.as_slice(),
            shape: [n, c, h * r, w * r],
            stride: contiguous_stride([n, c, h * r, w * r]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n, cin, h, w],
            stride: contiguous_stride([n, cin, h, w]),
        },
    }).expect("");
    stream.synchronize().expect("sync");
    let mut got = vec![0f64; host_in.len()];
    dev_out.copy_to_host(&mut got).expect("");
    for (i, (g, e)) in got.iter().zip(host_in.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "round-trip f64 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn pixel_shuffle_f16_bf16_smoke() {
    let (ctx, stream) = setup();
    let (n, c, h, w, r) = (1, 1, 1, 1, 2);
    let cin = c * r * r;
    // f16
    let host_in_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let host_in: Vec<f16> = host_in_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let dev_in = DeviceBuffer::from_slice(&ctx, &host_in).expect("");
    let mut dev_out: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, 4).expect("");
    PixelShufflePlan::<f16>::select(
        &stream,
        &PixelShuffleDescriptor {
            n, c, h, w, upscale_factor: r,
            element: ElementKind::F16,
        },
        PlanPreference::default(),
    ).expect("").run(&stream, Workspace::None, PixelShuffleArgs {
        input: TensorRef {
            data: dev_in.as_slice(),
            shape: [n, cin, h, w],
            stride: contiguous_stride([n, cin, h, w]),
        },
        output: TensorMut {
            data: dev_out.as_slice_mut(),
            shape: [n, c, h * r, w * r],
            stride: contiguous_stride([n, c, h * r, w * r]),
        },
    }).expect("");
    stream.synchronize().expect("sync");
    let mut got = vec![f16::from_f32(0.0); 4];
    dev_out.copy_to_host(&mut got).expect("");
    for (g, e) in got.iter().zip(host_in.iter()) {
        assert_eq!(g.to_bits(), e.to_bits());
    }

    // bf16
    let host_in_bf16: Vec<bf16> = host_in_f32.iter().map(|&x| bf16::from_f32(x)).collect();
    let dev_in_bf16 = DeviceBuffer::from_slice(&ctx, &host_in_bf16).expect("");
    let mut dev_out_bf16: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, 4).expect("");
    PixelShufflePlan::<bf16>::select(
        &stream,
        &PixelShuffleDescriptor {
            n, c, h, w, upscale_factor: r,
            element: ElementKind::Bf16,
        },
        PlanPreference::default(),
    ).expect("").run(&stream, Workspace::None, PixelShuffleArgs {
        input: TensorRef {
            data: dev_in_bf16.as_slice(),
            shape: [n, cin, h, w],
            stride: contiguous_stride([n, cin, h, w]),
        },
        output: TensorMut {
            data: dev_out_bf16.as_slice_mut(),
            shape: [n, c, h * r, w * r],
            stride: contiguous_stride([n, c, h * r, w * r]),
        },
    }).expect("");
    stream.synchronize().expect("sync");
    let mut got_bf16 = vec![bf16::from_f32(0.0); 4];
    dev_out_bf16.copy_to_host(&mut got_bf16).expect("");
    for (g, e) in got_bf16.iter().zip(host_in_bf16.iter()) {
        assert_eq!(g.to_bits(), e.to_bits());
    }
}
