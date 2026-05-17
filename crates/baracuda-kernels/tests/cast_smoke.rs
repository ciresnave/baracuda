//! Real-GPU smoke test for the Phase 3 cast kernel family
//! (`CastPlan<TIn, TOut>`).
//!
//! Exercises a handful of common cross-dtype casts against a CPU
//! reference. `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test cast_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, CastArgs, CastDescriptor, CastPlan, ElementKind, PlanPreference,
    TensorMut, TensorRef, Workspace,
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
fn cast_f32_to_f64_bit_exact() {
    let (ctx, stream) = setup();
    let numel = 1024usize;
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 17.25).collect();
    let expected: Vec<f64> = host_x.iter().map(|&x| x as f64).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = CastDescriptor {
        numel: numel as i32,
        input_element: ElementKind::F32,
        output_element: ElementKind::F64,
    };
    let plan = CastPlan::<f32, f64>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = CastArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32->f64 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn cast_f64_to_f32_bit_exact() {
    let (ctx, stream) = setup();
    let numel = 1024usize;
    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.5 - 17.25).collect();
    let expected: Vec<f32> = host_x.iter().map(|&x| x as f32).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = CastDescriptor {
        numel: numel as i32,
        input_element: ElementKind::F64,
        output_element: ElementKind::F32,
    };
    let plan = CastPlan::<f64, f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = CastArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f64->f32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn cast_f32_to_i32_truncates_toward_zero() {
    let (ctx, stream) = setup();
    // Mixed signs + half-integers so truncation behaviour is observable.
    let host_x: Vec<f32> = vec![
        0.0, 0.5, 0.9, -0.5, -0.9, 1.5, -1.5, 100.7, -100.7, 7777.25,
    ];
    let numel = host_x.len();
    // CPU reference: f32->i32 is truncation toward zero (C++ static_cast
    // semantics, matches GPU `static_cast<int32_t>(float)`).
    let expected: Vec<i32> = host_x.iter().map(|&x| x as i32).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = CastDescriptor {
        numel: numel as i32,
        input_element: ElementKind::F32,
        output_element: ElementKind::I32,
    };
    let plan = CastPlan::<f32, i32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = CastArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0i32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    assert_eq!(got, expected, "f32->i32");
}

#[test]
#[ignore]
fn cast_i32_to_f32_lossless_in_24bit_range() {
    let (ctx, stream) = setup();
    let numel = 256usize;
    // Stay within ±2^23 so f32 round-trip is bit-exact.
    let host_x: Vec<i32> = (0..numel).map(|i| (i as i32) - 128).collect();
    let expected: Vec<f32> = host_x.iter().map(|&x| x as f32).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = CastDescriptor {
        numel: numel as i32,
        input_element: ElementKind::I32,
        output_element: ElementKind::F32,
    };
    let plan = CastPlan::<i32, f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = CastArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "i32->f32 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn cast_f32_to_f16_round_to_nearest_even() {
    let (ctx, stream) = setup();
    let numel = 256usize;
    let host_x: Vec<f32> = (0..numel)
        .map(|i| (i as f32) * 0.0625 - 8.0) // step 1/16, well below f16 precision limit
        .collect();
    // CPU reference goes through half::f16::from_f32 (RNTE).
    let expected: Vec<f16> = host_x.iter().map(|&x| f16::from_f32(x)).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = CastDescriptor {
        numel: numel as i32,
        input_element: ElementKind::F32,
        output_element: ElementKind::F16,
    };
    let plan = CastPlan::<f32, f16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = CastArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "f32->f16 mismatch @ {i}");
    }
}

#[test]
#[ignore]
fn cast_bf16_to_f32_lossless() {
    let (ctx, stream) = setup();
    let numel = 256usize;
    let host_x: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32((i as f32) * 0.5 - 64.0))
        .collect();
    // bf16 -> f32 is bit-exact (zero-pad mantissa).
    let expected: Vec<f32> = host_x.iter().map(|x| x.to_f32()).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = CastDescriptor {
        numel: numel as i32,
        input_element: ElementKind::Bf16,
        output_element: ElementKind::F32,
    };
    let plan = CastPlan::<bf16, f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = CastArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "bf16->f32 mismatch @ {i}");
    }
}

#[test]
fn select_rejects_mismatched_descriptor_dtype() {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");

    // Descriptor says f32->f64 but we instantiate the plan with
    // CastPlan<f64, f32> — select must reject.
    let desc = CastDescriptor {
        numel: 16,
        input_element: ElementKind::F32,
        output_element: ElementKind::F64,
    };
    let err = CastPlan::<f64, f32>::select(&stream, &desc, PlanPreference::default());
    assert!(err.is_err(), "mismatched (TIn, TOut) must be rejected");
    let _ = ctx;
}
