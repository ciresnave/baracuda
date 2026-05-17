//! Real-GPU smoke tests for `PermutePlan<{f16,bf16,f64}, N>` — dtype
//! fanout of the f32 trailblazer in `permute_smoke.rs`.
//!
//! Permute does no math — bit-exact compare via `to_bits()`. Each test
//! mirrors `permute_3d_rotate` from the f32 file: `[A, B, C] -> [B, C,
//! A]` via dims=[1, 2, 0].
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test permute_dtype_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PermuteArgs, PermuteDescriptor, PermutePlan, PlanPreference,
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
fn permute_3d_rotate_f16() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 12];
    let dims = [1i32, 2, 0];
    let output_shape = [8i32, 12, 4];
    let numel = (4 * 8 * 12) as usize;

    let host_x: Vec<f16> = (0..numel)
        .map(|i| f16::from_f32(((i % 41) as f32) * 0.5 - 10.0))
        .collect();
    let mut expected = vec![f16::from_f32(0.0); numel];
    for a in 0..4 {
        for b in 0..8 {
            for c in 0..12 {
                let in_idx = (a * 96 + b * 12 + c) as usize;
                let out_idx = (b * 48 + c * 4 + a) as usize;
                expected[out_idx] = host_x[in_idx];
            }
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = PermuteDescriptor {
        input_shape,
        dims,
        element: ElementKind::F16,
    };
    let plan = PermutePlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select f16");
    let args = PermuteArgs::<f16, 3> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run f16");
    stream.synchronize().expect("sync");

    let mut got = vec![f16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "3d rotate f16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn permute_3d_rotate_bf16() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 12];
    let dims = [1i32, 2, 0];
    let output_shape = [8i32, 12, 4];
    let numel = (4 * 8 * 12) as usize;

    let host_x: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(((i % 37) as f32) * 0.25 - 4.5))
        .collect();
    let mut expected = vec![bf16::from_f32(0.0); numel];
    for a in 0..4 {
        for b in 0..8 {
            for c in 0..12 {
                let in_idx = (a * 96 + b * 12 + c) as usize;
                let out_idx = (b * 48 + c * 4 + a) as usize;
                expected[out_idx] = host_x[in_idx];
            }
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = PermuteDescriptor {
        input_shape,
        dims,
        element: ElementKind::Bf16,
    };
    let plan = PermutePlan::<bf16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select bf16");
    let args = PermuteArgs::<bf16, 3> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run bf16");
    stream.synchronize().expect("sync");

    let mut got = vec![bf16::from_f32(0.0); numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "3d rotate bf16 mismatch @ {i}: got {g} expected {e}"
        );
    }
}

#[test]
#[ignore]
fn permute_3d_rotate_f64() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 12];
    let dims = [1i32, 2, 0];
    let output_shape = [8i32, 12, 4];
    let numel = (4 * 8 * 12) as usize;

    let host_x: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.0625 - 2.0).collect();
    let mut expected = vec![0f64; numel];
    for a in 0..4 {
        for b in 0..8 {
            for c in 0..12 {
                let in_idx = (a * 96 + b * 12 + c) as usize;
                let out_idx = (b * 48 + c * 4 + a) as usize;
                expected[out_idx] = host_x[in_idx];
            }
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = PermuteDescriptor {
        input_shape,
        dims,
        element: ElementKind::F64,
    };
    let plan = PermutePlan::<f64, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select f64");
    let args = PermuteArgs::<f64, 3> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape: input_shape,
            stride: contiguous_stride(input_shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: output_shape,
            stride: contiguous_stride(output_shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run f64");
    stream.synchronize().expect("sync");

    let mut got = vec![0f64; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "3d rotate f64 mismatch @ {i}: got {g} expected {e}"
        );
    }
}
