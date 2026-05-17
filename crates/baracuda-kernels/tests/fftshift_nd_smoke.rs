//! Real-GPU smoke tests for `FftShiftNdPlan` (N-D fftshift / ifftshift).
//!
//! fftshift is a pure index permutation — bit-exact across float types.
//! Each test pre-computes the expected output on the CPU by walking the
//! same index permutation the kernel computes and asserts element-wise
//! equality (no tolerance — the operation is purely a copy with rotated
//! indices).
//!
//! `#[ignore]` by default — requires a real CUDA device.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, Complex32, ElementKind, FftShiftNdArgs, FftShiftNdDescriptor,
    FftShiftNdPlan, PlanPreference, TensorMut, TensorRef, Workspace, FFTSHIFT_ND_MAX_RANK,
    FFTSHIFT_ND_MAX_SHIFT_AXES,
};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference: walk every output coord, transform shifted axes,
/// and recompose the source index.
fn cpu_fftshift_nd<T: Copy>(
    input: &[T],
    shape: &[i32],
    shift_axes: &[usize],
    inverse: bool,
) -> Vec<T> {
    let ndim = shape.len();
    let numel: i64 = shape.iter().map(|&d| d as i64).product();
    let numel = numel as usize;
    // Row-major contiguous strides.
    let mut stride = vec![1i64; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        stride[d] = stride[d + 1] * shape[d + 1] as i64;
    }
    // Per-axis shift amount.
    let mut shift = vec![0i32; ndim];
    for &a in shift_axes {
        let n = shape[a];
        let half = n / 2;
        shift[a] = if inverse { n - half } else { half };
    }
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let mut rem = i as i64;
        let mut src = 0i64;
        for d in 0..ndim {
            let s = stride[d];
            let n = shape[d];
            let out_coord = (rem / s) as i32;
            rem -= out_coord as i64 * s;
            let mut src_coord = out_coord + shift[d];
            if src_coord >= n {
                src_coord -= n;
            }
            src += src_coord as i64 * s;
        }
        out.push(input[src as usize]);
    }
    out
}

fn build_desc<const N: usize>(
    shape_arr: [i32; N],
    shift_axes_arr: [u8; N],
    num_shift_axes: u8,
    inverse: bool,
    element: ElementKind,
) -> FftShiftNdDescriptor {
    let mut shape = [0i32; FFTSHIFT_ND_MAX_RANK];
    let mut shift_axes = [0u8; FFTSHIFT_ND_MAX_SHIFT_AXES];
    shape[..N].copy_from_slice(&shape_arr);
    let n_axes = num_shift_axes as usize;
    shift_axes[..n_axes].copy_from_slice(&shift_axes_arr[..n_axes]);
    FftShiftNdDescriptor {
        shape,
        ndim: N as u8,
        shift_axes,
        num_shift_axes,
        inverse,
        element,
    }
}

// -------------------------------------------------------------------------
// 2-D shifts.
// -------------------------------------------------------------------------

#[test]
#[ignore]
fn fftshift_nd_2d_even_f32_both_axes() {
    // 4 x 4. Shift both axes (the canonical "center the DC peak" usage).
    let (ctx, stream) = setup();
    let shape = [4i32, 4];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let x_host: Vec<f32> = (0..numel).map(|i| i as f32).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = build_desc::<2>(shape, [0, 1], 2, false, ElementKind::F32);
    let plan = FftShiftNdPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let stride = contiguous_stride(shape);
    let args = FftShiftNdArgs::<f32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride,
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let expected = cpu_fftshift_nd(&x_host, &shape, &[0, 1], false);
    assert_eq!(got, expected);
}

#[test]
#[ignore]
fn fftshift_nd_2d_odd_f32_both_axes_fft_vs_ifft() {
    // 3 x 5 — odd on both axes. fftshift and ifftshift must differ;
    // their composition must be the identity.
    let (ctx, stream) = setup();
    let shape = [3i32, 5];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let x_host: Vec<f32> = (0..numel).map(|i| i as f32 * 0.5).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_fwd: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc fwd");
    let mut dev_rt: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc rt");

    let stride = contiguous_stride(shape);

    // fftshift.
    let f_desc = build_desc::<2>(shape, [0, 1], 2, false, ElementKind::F32);
    let f_plan = FftShiftNdPlan::<f32, 2>::select(&stream, &f_desc, PlanPreference::default())
        .expect("select fft");
    let f_args = FftShiftNdArgs::<f32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride,
        },
        output: TensorMut {
            data: dev_fwd.as_slice_mut(),
            shape,
            stride,
        },
    };
    f_plan.run(&stream, Workspace::None, f_args).expect("run fft");

    // ifftshift on the fftshift output -> identity.
    let i_desc = build_desc::<2>(shape, [0, 1], 2, true, ElementKind::F32);
    let i_plan = FftShiftNdPlan::<f32, 2>::select(&stream, &i_desc, PlanPreference::default())
        .expect("select ifft");
    let i_args = FftShiftNdArgs::<f32, 2> {
        input: TensorRef {
            data: dev_fwd.as_slice(),
            shape,
            stride,
        },
        output: TensorMut {
            data: dev_rt.as_slice_mut(),
            shape,
            stride,
        },
    };
    i_plan
        .run(&stream, Workspace::None, i_args)
        .expect("run ifft");
    stream.synchronize().expect("sync");

    let mut got_fwd = vec![0f32; numel];
    dev_fwd.copy_to_host(&mut got_fwd).expect("dl fwd");
    let expected_fwd = cpu_fftshift_nd(&x_host, &shape, &[0, 1], false);
    assert_eq!(got_fwd, expected_fwd);

    let mut got_rt = vec![0f32; numel];
    dev_rt.copy_to_host(&mut got_rt).expect("dl rt");
    assert_eq!(got_rt, x_host, "ifftshift(fftshift(x)) != x for odd 2-D");

    // Sanity: forward and inverse differ on odd shapes.
    let expected_inv = cpu_fftshift_nd(&x_host, &shape, &[0, 1], true);
    assert_ne!(expected_fwd, expected_inv);
}

#[test]
#[ignore]
fn fftshift_nd_2d_one_axis_only_complex32() {
    // 4 x 6 — Complex32. Shift only axis 1 (last axis); axis 0 is
    // pass-through. Equivalent to the 1-D plan with batch == shape[0],
    // but exercised through the N-D code path.
    let (ctx, stream) = setup();
    let shape = [4i32, 6];
    let numel: usize = (shape[0] * shape[1]) as usize;
    let x_host: Vec<Complex32> = (0..numel)
        .map(|i| Complex32::new(i as f32, (i as f32) * 0.25))
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_y: DeviceBuffer<Complex32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = build_desc::<2>(shape, [1, 0], 1, false, ElementKind::Complex32);
    let plan = FftShiftNdPlan::<Complex32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let stride = contiguous_stride(shape);
    let args = FftShiftNdArgs::<Complex32, 2> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride,
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![Complex32::default(); numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let expected = cpu_fftshift_nd(&x_host, &shape, &[1], false);
    assert_eq!(got, expected);
}

// -------------------------------------------------------------------------
// 3-D shifts.
// -------------------------------------------------------------------------

#[test]
#[ignore]
fn fftshift_nd_3d_mixed_parity_f32() {
    // 2 x 3 x 4 — mix of even and odd axes. Shift all three.
    let (ctx, stream) = setup();
    let shape = [2i32, 3, 4];
    let numel: usize = (shape[0] * shape[1] * shape[2]) as usize;
    let x_host: Vec<f32> = (0..numel).map(|i| i as f32 - 7.5).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = build_desc::<3>(shape, [0, 1, 2], 3, false, ElementKind::F32);
    let plan = FftShiftNdPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let stride = contiguous_stride(shape);
    let args = FftShiftNdArgs::<f32, 3> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride,
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let expected = cpu_fftshift_nd(&x_host, &shape, &[0, 1, 2], false);
    assert_eq!(got, expected);
}

#[test]
#[ignore]
fn fftshift_nd_3d_batched_inner_two_axes_f64() {
    // 3 x 4 x 5 — treat axis 0 as a batch axis (pass-through), shift
    // axes 1 and 2 only. Mixed parity (even / odd) on the shifted axes;
    // f64 to exercise the 8-byte cell path.
    let (ctx, stream) = setup();
    let shape = [3i32, 4, 5];
    let numel: usize = (shape[0] * shape[1] * shape[2]) as usize;
    let x_host: Vec<f64> = (0..numel).map(|i| (i as f64) * 1.5 + 0.25).collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_fwd: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc fwd");
    let mut dev_rt: DeviceBuffer<f64> = DeviceBuffer::zeros(&ctx, numel).expect("alloc rt");

    let stride = contiguous_stride(shape);

    let f_desc = build_desc::<3>(shape, [1, 2, 0], 2, false, ElementKind::F64);
    let f_plan = FftShiftNdPlan::<f64, 3>::select(&stream, &f_desc, PlanPreference::default())
        .expect("select fft");
    let f_args = FftShiftNdArgs::<f64, 3> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride,
        },
        output: TensorMut {
            data: dev_fwd.as_slice_mut(),
            shape,
            stride,
        },
    };
    f_plan.run(&stream, Workspace::None, f_args).expect("run fft");
    stream.synchronize().expect("sync");

    let mut got_fwd = vec![0f64; numel];
    dev_fwd.copy_to_host(&mut got_fwd).expect("dl fwd");
    let expected_fwd = cpu_fftshift_nd(&x_host, &shape, &[1, 2], false);
    assert_eq!(got_fwd, expected_fwd);

    // Round-trip via ifftshift.
    let i_desc = build_desc::<3>(shape, [1, 2, 0], 2, true, ElementKind::F64);
    let i_plan = FftShiftNdPlan::<f64, 3>::select(&stream, &i_desc, PlanPreference::default())
        .expect("select ifft");
    let i_args = FftShiftNdArgs::<f64, 3> {
        input: TensorRef {
            data: dev_fwd.as_slice(),
            shape,
            stride,
        },
        output: TensorMut {
            data: dev_rt.as_slice_mut(),
            shape,
            stride,
        },
    };
    i_plan
        .run(&stream, Workspace::None, i_args)
        .expect("run ifft");
    stream.synchronize().expect("sync");

    let mut got_rt = vec![0f64; numel];
    dev_rt.copy_to_host(&mut got_rt).expect("dl rt");
    assert_eq!(got_rt, x_host, "ifftshift(fftshift(x)) != x for 3-D batched");
}

#[test]
#[ignore]
fn fftshift_nd_3d_complex32_all_axes_odd() {
    // 3 x 5 x 7 — odd-odd-odd Complex32. Stresses the per-axis (n+1)/2
    // vs n/2 asymmetry on every shifted axis.
    let (ctx, stream) = setup();
    let shape = [3i32, 5, 7];
    let numel: usize = (shape[0] * shape[1] * shape[2]) as usize;
    let x_host: Vec<Complex32> = (0..numel)
        .map(|i| Complex32::new(i as f32, -(i as f32) * 0.125))
        .collect();
    let dev_x = DeviceBuffer::from_slice(&ctx, &x_host).expect("upload");
    let mut dev_y: DeviceBuffer<Complex32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc y");

    let desc = build_desc::<3>(shape, [0, 1, 2], 3, true, ElementKind::Complex32);
    let plan = FftShiftNdPlan::<Complex32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let stride = contiguous_stride(shape);
    let args = FftShiftNdArgs::<Complex32, 3> {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape,
            stride,
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape,
            stride,
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![Complex32::default(); numel];
    dev_y.copy_to_host(&mut got).expect("dl");

    let expected = cpu_fftshift_nd(&x_host, &shape, &[0, 1, 2], true);
    assert_eq!(got, expected);
}
