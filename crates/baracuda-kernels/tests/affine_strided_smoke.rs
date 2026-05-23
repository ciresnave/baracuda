//! Real-GPU smoke test for the Phase 14.1 strided affine sibling.
//!
//! Validates that [`AffinePlan::run`] dispatches between the contig
//! fast path and the strided FFI sibling based on the input / output
//! stride pattern, and that both paths produce identical results.
//!
//! `#[ignore]` like the rest of the GPU smoke suite. Run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test affine_strided_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, AffineArgs, AffineDescriptor, AffinePlan, ElementKind, PlanPreference,
    TensorMut, TensorRef, Workspace,
};
use half::bf16;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Sanity: contiguous rank-1 input / output still works through the
/// updated dispatch — the canonical-contig detector must hit the fast
/// path, and the result must match the CPU reference.
#[test]
#[ignore]
fn affine_f32_contig_path_still_correct() {
    let (ctx, stream) = setup();
    let numel = 512usize;
    let a: f32 = 1.5;
    let b: f32 = -0.25;
    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.125 - 8.0).collect();
    let expected: Vec<f32> = host_x.iter().map(|&x| a * x + b).collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = AffineDescriptor {
        numel: numel as i32,
        a,
        b,
        element: ElementKind::F32,
    };
    let plan = AffinePlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = AffineArgs {
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

    let tol_eps = 2.0f32 * f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        let tol = e.abs().max(1.0) * tol_eps;
        assert!(err <= tol, "affine_f32 contig @ {i}: got {g} expected {e}");
    }
}

/// Strided input (stride 2 over a buffer of 2*numel elements) — picks
/// every other element. The strided FFI sibling must fire and produce
/// `y[i] = a * x[2*i] + b`.
#[test]
#[ignore]
fn affine_f32_strided_input_stride2() {
    let (ctx, stream) = setup();
    let numel = 256usize;
    let stride: usize = 2;
    let buf_len = numel * stride;
    let a: f32 = 2.0;
    let b: f32 = 1.0;

    // Big buffer; we'll read every-other element.
    let host_x: Vec<f32> = (0..buf_len).map(|i| (i as f32) * 0.0625 - 4.0).collect();
    let expected: Vec<f32> = (0..numel)
        .map(|i| a * host_x[i * stride] + b)
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = AffineDescriptor {
        numel: numel as i32,
        a,
        b,
        element: ElementKind::F32,
    };
    let plan = AffinePlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = AffineArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: [stride as i64],
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

    let tol_eps = 2.0f32 * f32::EPSILON;
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        let tol = e.abs().max(1.0) * tol_eps;
        assert!(err <= tol, "affine_f32 stride2 @ {i}: got {g} expected {e}");
    }
}

/// Strided OUTPUT (writes every other slot in a 2*numel buffer). Verifies
/// the kernel writes only into the strided positions and leaves the
/// in-between slots untouched.
#[test]
#[ignore]
fn affine_f32_strided_output_stride2_interior_view() {
    let (ctx, stream) = setup();
    let numel = 128usize;
    let stride: usize = 2;
    let buf_len = numel * stride;
    let a: f32 = -1.0;
    let b: f32 = 0.5;

    let host_x: Vec<f32> = (0..numel).map(|i| (i as f32) * 0.25).collect();
    // Seed the destination buffer with a sentinel so we can verify the
    // un-strided slots are untouched after the kernel runs.
    const SENTINEL: f32 = 1234.5;
    let host_y_init: Vec<f32> = vec![SENTINEL; buf_len];

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y = DeviceBuffer::from_slice(&ctx, &host_y_init).expect("upload");

    let desc = AffineDescriptor {
        numel: numel as i32,
        a,
        b,
        element: ElementKind::F32,
    };
    let plan = AffinePlan::<f32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = AffineArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: [stride as i64],
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; buf_len];
    dev_y.copy_to_host(&mut got).expect("download");

    let tol_eps = 2.0f32 * f32::EPSILON;
    for i in 0..numel {
        let g = got[i * stride];
        let e = a * host_x[i] + b;
        let err = (g - e).abs();
        let tol = e.abs().max(1.0) * tol_eps;
        assert!(err <= tol, "affine_f32 out-stride2 @ {i}: got {g} expected {e}");
    }
    // Verify the in-between slots are untouched.
    for i in 0..numel {
        let off = i * stride + 1;
        assert_eq!(got[off], SENTINEL, "sentinel disturbed at off={off}");
    }
}

/// bf16 strided input — verifies the half-precision strided launcher
/// routes through f32 compute and stores as bf16, matching the contig
/// sibling's precision contract.
#[test]
#[ignore]
fn affine_bf16_strided_input_stride2() {
    let (ctx, stream) = setup();
    let numel = 256usize;
    let stride: usize = 2;
    let buf_len = numel * stride;
    let a: bf16 = bf16::from_f32(2.5);
    let b: bf16 = bf16::from_f32(-0.5);

    let host_x: Vec<bf16> = (0..buf_len)
        .map(|i| bf16::from_f32((i as f32) * 0.0625 - 4.0))
        .collect();
    let expected: Vec<f32> = (0..numel)
        .map(|i| a.to_f32() * host_x[i * stride].to_f32() + b.to_f32())
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<bf16> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = AffineDescriptor {
        numel: numel as i32,
        a,
        b,
        element: ElementKind::Bf16,
    };
    let plan = AffinePlan::<bf16>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = AffineArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: [stride as i64],
        },
        output: TensorMut {
            data: dev_y.as_slice_mut(),
            shape: [numel as i32],
            stride: contiguous_stride([numel as i32]),
        },
    };
    plan.run(&stream, Workspace::None, args).expect("run");
    stream.synchronize().expect("sync");

    let mut got_h = vec![bf16::ZERO; numel];
    dev_y.copy_to_host(&mut got_h).expect("download");
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();

    // bf16 mantissa is 7 bits — pick a tolerance generous enough for
    // single-rounding-at-store quantization plus the GPU's FFMA vs
    // CPU's mul-then-add 1-ulp slack.
    let tol_rel = 1.0 / 128.0f32; // ~ 2 * 2^-7
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let err = (g - e).abs();
        let tol = e.abs().max(1.0) * tol_rel;
        assert!(err <= tol, "affine_bf16 stride2 @ {i}: got {g} expected {e}");
    }
}

/// i32 strided correctness — pure integer multiply / add, no rounding.
#[test]
#[ignore]
fn affine_i32_strided_input_stride2() {
    let (ctx, stream) = setup();
    let numel = 128usize;
    let stride: usize = 2;
    let buf_len = numel * stride;
    let a: i32 = 3;
    let b: i32 = -7;

    let host_x: Vec<i32> = (0..buf_len).map(|i| (i as i32) - 64).collect();
    let expected: Vec<i32> = (0..numel)
        .map(|i| a.wrapping_mul(host_x[i * stride]).wrapping_add(b))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let desc = AffineDescriptor {
        numel: numel as i32,
        a,
        b,
        element: ElementKind::I32,
    };
    let plan = AffinePlan::<i32>::select(&stream, &desc, PlanPreference::default())
        .expect("select");
    let args = AffineArgs {
        input: TensorRef {
            data: dev_x.as_slice(),
            shape: [numel as i32],
            stride: [stride as i64],
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
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "affine_i32 stride2 @ {i}: got {g} expected {e}");
    }
}

/// u8 strided correctness — wraps mod 256 on overflow.
///
/// u8 is not in [`AffinePlan`]'s wired-dtype set today (the Plan layer
/// gates on `{f32, f64, f16, bf16, i32, i64}`); the strided FFI symbol
/// ships independently per Fuel's request, so this test exercises the
/// FFI symbol directly rather than going through the plan.
#[test]
#[ignore]
fn affine_u8_strided_input_stride2() {
    let (ctx, stream) = setup();
    let numel = 64usize;
    let stride: usize = 2;
    let buf_len = numel * stride;
    let a: u8 = 3;
    let b: u8 = 5;

    let host_x: Vec<u8> = (0..buf_len).map(|i| (i as u8).wrapping_mul(7)).collect();
    let expected: Vec<u8> = (0..numel)
        .map(|i| a.wrapping_mul(host_x[i * stride]).wrapping_add(b))
        .collect();

    let dev_x = DeviceBuffer::from_slice(&ctx, &host_x).expect("upload");
    let mut dev_y: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel).expect("alloc");

    let shape: [i32; 1] = [numel as i32];
    let stride_x: [i64; 1] = [stride as i64];
    let stride_y: [i64; 1] = [1];
    let x_ptr = dev_x.as_slice().as_raw().0 as *const core::ffi::c_void;
    let y_ptr = dev_y.as_slice_mut().as_raw().0 as *mut core::ffi::c_void;
    let stream_ptr = stream.as_raw() as *mut core::ffi::c_void;
    let status = unsafe {
        baracuda_kernels_sys::baracuda_kernels_affine_u8_strided_run(
            numel as i64,
            1,
            shape.as_ptr(),
            stride_x.as_ptr(),
            stride_y.as_ptr(),
            x_ptr,
            y_ptr,
            a,
            b,
            core::ptr::null_mut(),
            0,
            stream_ptr,
        )
    };
    assert_eq!(status, 0, "u8 strided FFI returned non-zero status: {status}");
    stream.synchronize().expect("sync");

    let mut got = vec![0u8; numel];
    dev_y.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "affine_u8 stride2 @ {i}: got {g} expected {e}");
    }
    let _ = ctx;
}
