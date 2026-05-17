//! Real-GPU smoke test for `PermutePlan<f32, N>`.
//!
//! Covers 2D transpose (dims=[1,0]) + 3D permutations (e.g. NHWC→NCHW
//! patterns from CV). Bit-exact compare — permute does no math.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test permute_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PermuteArgs, PermuteDescriptor, PermutePlan, PlanPreference,
    TensorMut, TensorRef, Workspace,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// 2D transpose: `output[i, j] = input[j, i]`. dims = [1, 0].
#[test]
#[ignore]
fn permute_2d_transpose() {
    let (ctx, stream) = setup();
    let input_shape = [16i32, 32];
    let dims = [1i32, 0];
    let output_shape = [32i32, 16];
    let numel = (16 * 32) as usize;

    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.125 - 5.0).collect();
    let mut expected = vec![0f32; numel];
    for i in 0..16 {
        for j in 0..32 {
            // input[i, j] (row-major) at index i*32+j → output[j, i] at index j*16+i
            expected[(j * 16 + i) as usize] = host_x[(i * 32 + j) as usize];
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = PermuteDescriptor {
        input_shape,
        dims,
        element: ElementKind::F32,
    };
    let plan = PermutePlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = PermuteArgs::<f32, 2> {
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
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "transpose mismatch @ {i}");
    }
}

/// 3D permute — `[A, B, C] -> [B, C, A]` via dims=[1, 2, 0]. Common
/// "shuffle axes" pattern.
#[test]
#[ignore]
fn permute_3d_rotate() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8, 12];
    let dims = [1i32, 2, 0];  // [A, B, C] -> [B, C, A]
    let output_shape = [8i32, 12, 4];
    let numel = (4 * 8 * 12) as usize;

    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.0625 - 2.0).collect();
    // input[a, b, c] = host_x[a*96 + b*12 + c]
    // output[b, c, a] = output_buf[b*48 + c*4 + a]
    let mut expected = vec![0f32; numel];
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
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = PermuteDescriptor {
        input_shape,
        dims,
        element: ElementKind::F32,
    };
    let plan = PermutePlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = PermuteArgs::<f32, 3> {
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
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "3d rotate mismatch @ {i}");
    }
}

/// 4D NHWC -> NCHW pattern: `[N, H, W, C] -> [N, C, H, W]`,
/// dims=[0, 3, 1, 2].
#[test]
#[ignore]
fn permute_4d_nhwc_to_nchw() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 8, 8, 4];   // [N=2, H=8, W=8, C=4]
    let dims = [0i32, 3, 1, 2];           // [N, C, H, W]
    let output_shape = [2i32, 4, 8, 8];
    let numel = (2 * 8 * 8 * 4) as usize;

    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.05 - 1.0).collect();
    // input[n, h, w, c] = host_x[n*256 + h*32 + w*4 + c]
    // output[n, c, h, w] = output_buf[n*256 + c*64 + h*8 + w]
    let mut expected = vec![0f32; numel];
    for n in 0..2 {
        for h in 0..8 {
            for w in 0..8 {
                for c in 0..4 {
                    let in_idx = (n * 256 + h * 32 + w * 4 + c) as usize;
                    let out_idx = (n * 256 + c * 64 + h * 8 + w) as usize;
                    expected[out_idx] = host_x[in_idx];
                }
            }
        }
    }

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = PermuteDescriptor {
        input_shape,
        dims,
        element: ElementKind::F32,
    };
    let plan = PermutePlan::<f32, 4>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = PermuteArgs::<f32, 4> {
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
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "NHWC→NCHW mismatch @ {i}");
    }
}

/// Identity permute: dims = [0, 1, 2, ...] — output == input.
#[test]
#[ignore]
fn permute_identity_is_copy() {
    let (ctx, stream) = setup();
    let shape = [4i32, 16, 8];
    let dims = [0i32, 1, 2];
    let numel = (4 * 16 * 8) as usize;

    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.5 - 100.0).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload x");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = PermuteDescriptor {
        input_shape: shape,
        dims,
        element: ElementKind::F32,
    };
    let plan = PermutePlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = PermuteArgs::<f32, 3> {
        x: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        y: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(host_x.iter()).enumerate() {
        assert_eq!(g.to_bits(), e.to_bits(), "identity permute mismatch @ {i}");
    }
}
