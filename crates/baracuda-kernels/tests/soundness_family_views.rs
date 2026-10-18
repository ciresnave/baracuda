//! Soundness: one representative plan per op family must reject (`Err`) a
//! launch whose `TensorRef` / `TensorMut` / `MatrixRef` declares a shape that
//! reaches past the `DeviceSlice` backing it.
//!
//! Every test is the simplest smoke-test configuration of its family with
//! exactly one operand's slice shortened to half the elements its (unchanged)
//! shape needs. Binary elementwise, MSE loss and the bit-GEMM are covered in
//! `soundness_tensor_view_bounds.rs`.
//!
//! `cargo test -p baracuda-kernels --features sm89,device-tests --test soundness_family_views`
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    AlibiArgs, AlibiDescriptor, AlibiPlan, Bool, CholeskyArgs, CholeskyDescriptor, CholeskyPlan,
    Complex64, DequantizePerTensorArgs, DequantizePerTensorDescriptor, DequantizePerTensorPlan,
    DropoutArgs, DropoutDescriptor, DropoutPlan, ElementKind, EmbeddingArgs, EmbeddingDescriptor,
    EmbeddingPlan, EpilogueKind, FftArgs, FftDescriptor, FftPlan, FlipArgs, FlipDescriptor,
    FlipPlan, Fp8E4M3FN, Fp8GemmArgs, Fp8GemmDescriptor, Fp8GemmPlan, LayoutSku, MaskedFillArgs,
    MaskedFillDescriptor, MaskedFillPlan, MatrixMut, MatrixRef, PixelShuffleArgs,
    PixelShuffleDescriptor, PixelShufflePlan, PlanPreference, RMSNormArgs, RMSNormDescriptor,
    RMSNormPlan, ReduceArgs, ReduceDescriptor, ReduceKind, ReducePlan, S8, ScanArgs,
    ScanDescriptor, ScanKind, ScanPlan, SegmentSumArgs, SegmentSumDescriptor, SegmentSumPlan,
    SoftmaxArgs, SoftmaxDescriptor, SoftmaxKind, SoftmaxPlan, SortArgs, SortDescriptor, SortPlan,
    TensorMut, TensorRef, UnaryArgs, UnaryDescriptor, UnaryKind, UnaryPlan, Workspace,
    contiguous_stride,
};

#[cfg(feature = "mamba")]
use baracuda_kernels::{CausalConv1dArgs, CausalConv1dDescriptor, CausalConv1dPlan};
#[cfg(feature = "cudnn")]
use baracuda_kernels::{Im2ColArgs, Im2ColDescriptor, Im2ColPlan};

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// Zero-filled `f32` buffer of `n` elements.
fn zeros_f32(ctx: &Context, n: usize) -> DeviceBuffer<f32> {
    DeviceBuffer::zeros(ctx, n).expect("alloc f32")
}

/// Zero-filled `i32` buffer of `n` elements.
fn zeros_i32(ctx: &Context, n: usize) -> DeviceBuffer<i32> {
    DeviceBuffer::zeros(ctx, n).expect("alloc i32")
}

#[test]
fn attention_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let (batch, heads, q, k) = (2i32, 4i32, 6i32, 8i32);
    let numel = (batch * heads * q * k) as usize;
    let scores = zeros_f32(&ctx, numel / 2); // shape needs `numel`
    let slopes = zeros_f32(&ctx, heads as usize);
    let mut out = zeros_f32(&ctx, numel);
    let desc = AlibiDescriptor {
        batch_size: batch,
        num_heads: heads,
        query_len: q,
        key_len: k,
        element: ElementKind::F32,
    };
    let plan = AlibiPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let shape = [batch, heads, q, k];
    let res = plan.run(
        &stream,
        Workspace::None,
        AlibiArgs {
            scores: TensorRef {
                data: scores.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            slopes: TensorRef {
                data: slopes.as_slice(),
                shape: [heads],
                stride: [1],
            },
            out: TensorMut {
                data: out.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    );
    assert!(res.is_err(), "AlibiPlan::run accepted a short scores slice");
}

#[cfg(feature = "mamba")]
#[test]
fn causal_conv1d_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let (b, c, l, w) = (2i32, 4i32, 16i32, 4i32);
    let x = zeros_f32(&ctx, (b * c * l) as usize / 2); // shape needs b*c*l
    let weight = zeros_f32(&ctx, (c * w) as usize);
    let mut y = zeros_f32(&ctx, (b * c * l) as usize);
    let desc = CausalConv1dDescriptor {
        batch_size: b,
        channels: c,
        seq_len: l,
        width: w,
        use_silu: false,
        element: ElementKind::F32,
    };
    let plan = CausalConv1dPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let shape_x = [b, c, l];
    let shape_w = [c, w];
    let res = plan.run(
        &stream,
        Workspace::None,
        CausalConv1dArgs {
            x: TensorRef {
                data: x.as_slice(),
                shape: shape_x,
                stride: contiguous_stride(shape_x),
            },
            weight: TensorRef {
                data: weight.as_slice(),
                shape: shape_w,
                stride: contiguous_stride(shape_w),
            },
            bias: None,
            y: TensorMut {
                data: y.as_slice_mut(),
                shape: shape_x,
                stride: contiguous_stride(shape_x),
            },
        },
    );
    assert!(
        res.is_err(),
        "CausalConv1dPlan::run accepted a short x slice"
    );
}

#[cfg(feature = "cudnn")]
#[test]
fn conv_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let (n, c, h_in, w_in) = (1i32, 2i32, 4i32, 5i32);
    let (kh, kw) = (3i32, 3i32);
    let (h_out, w_out) = (2i32, 3i32);
    let x = zeros_f32(&ctx, (n * c * h_in * w_in) as usize / 2); // shape needs 40
    let y_shape = [n, c * kh * kw, h_out * w_out];
    let mut y = zeros_f32(&ctx, y_shape.iter().product::<i32>() as usize);
    let desc = Im2ColDescriptor {
        batch: n,
        channels: c,
        h_in,
        w_in,
        kh,
        kw,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
        dilation_h: 1,
        dilation_w: 1,
        element: ElementKind::F32,
    };
    let plan = Im2ColPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let x_shape = [n, c, h_in, w_in];
    let res = plan.run(
        &stream,
        Workspace::None,
        Im2ColArgs {
            input: TensorRef {
                data: x.as_slice(),
                shape: x_shape,
                stride: contiguous_stride(x_shape),
            },
            output: TensorMut {
                data: y.as_slice_mut(),
                shape: y_shape,
                stride: contiguous_stride(y_shape),
            },
        },
    );
    assert!(res.is_err(), "Im2ColPlan::run accepted a short input slice");
}

/// Unary has a `can_implement`, so it is checked there as well as in `run`.
#[test]
fn elementwise_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    const N: usize = 64;
    let shape = [N as i32];
    let x = zeros_f32(&ctx, N / 2);
    let mut y = zeros_f32(&ctx, N);
    let desc = UnaryDescriptor {
        kind: UnaryKind::Relu,
        shape,
        element: ElementKind::F32,
    };
    let plan = UnaryPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let args = UnaryArgs::<f32, 1> {
        x: TensorRef {
            data: x.as_slice(),
            shape,
            stride: contiguous_stride(shape),
        },
        y: TensorMut {
            data: y.as_slice_mut(),
            shape,
            stride: contiguous_stride(shape),
        },
    };
    assert!(
        plan.can_implement(&args).is_err(),
        "UnaryPlan::can_implement accepted a short x slice"
    );
    assert!(
        plan.run(&stream, Workspace::None, args).is_err(),
        "UnaryPlan::run accepted a short x slice"
    );
}

#[test]
fn embedding_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let (v, d, n) = (5i32, 4i32, 3i32);
    let weight = zeros_f32(&ctx, (v * d) as usize / 2); // shape needs v*d
    let indices = DeviceBuffer::from_slice(&ctx, &[1i32, 3, 0]).unwrap();
    let mut out = zeros_f32(&ctx, (n * d) as usize);
    let desc = EmbeddingDescriptor {
        num_embeddings: v,
        embedding_dim: d,
        num_indices: n,
        padding_idx: None,
        element: ElementKind::F32,
    };
    let plan = EmbeddingPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        EmbeddingArgs::<f32> {
            weight: TensorRef {
                data: weight.as_slice(),
                shape: [v, d],
                stride: contiguous_stride([v, d]),
            },
            indices: TensorRef {
                data: indices.as_slice(),
                shape: [n],
                stride: contiguous_stride([n]),
            },
            output: TensorMut {
                data: out.as_slice_mut(),
                shape: [n, d],
                stride: contiguous_stride([n, d]),
            },
        },
    );
    assert!(
        res.is_err(),
        "EmbeddingPlan::run accepted a short weight slice"
    );
}

#[test]
fn fft_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let (n, batch) = (16i32, 2i32);
    let total = (n * batch) as usize;
    let x: DeviceBuffer<Complex64> = DeviceBuffer::zeros(&ctx, total / 2).unwrap();
    let mut y: DeviceBuffer<Complex64> = DeviceBuffer::zeros(&ctx, total).unwrap();
    let desc = FftDescriptor {
        n,
        batch,
        inverse: false,
        element: ElementKind::Complex64,
    };
    let plan = FftPlan::<Complex64>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let shape = [batch, n];
    let res = plan.run(
        &stream,
        Workspace::None,
        FftArgs::<Complex64> {
            x: TensorRef {
                data: x.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            y: TensorMut {
                data: y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    );
    assert!(res.is_err(), "FftPlan::run accepted a short x slice");
}

/// FP8 GEMM: A's slice holds half the bytes the `[M, K]` matrix needs.
#[test]
fn gemm_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let (m, n, k) = (128i32, 128i32, 128i32);
    let a_short = DeviceBuffer::from_slice(&ctx, &vec![0u8; (m * k) as usize / 2]).unwrap();
    let b = DeviceBuffer::from_slice(&ctx, &vec![0u8; (k * n) as usize]).unwrap();
    let mut d: DeviceBuffer<Fp8E4M3FN> = DeviceBuffer::zeros(&ctx, (m * n) as usize).unwrap();
    let desc = Fp8GemmDescriptor {
        m,
        n,
        k,
        layout: LayoutSku::Rcr,
        epilogue: EpilogueKind::Identity,
    };
    let plan = Fp8GemmPlan::<Fp8E4M3FN>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        Fp8GemmArgs::<Fp8E4M3FN> {
            a: MatrixRef {
                data: a_short.view_as::<Fp8E4M3FN>(),
                rows: m,
                cols: k,
                ld: k as i64,
            },
            b: MatrixRef {
                data: b.view_as::<Fp8E4M3FN>(),
                rows: k,
                cols: n,
                ld: k as i64,
            },
            c: None,
            d: MatrixMut {
                data: d.as_slice_mut(),
                rows: m,
                cols: n,
                ld: n as i64,
            },
            bias: None,
            alpha: 1.0,
            beta: 0.0,
        },
    );
    assert!(
        res.is_err(),
        "Fp8GemmPlan::run accepted an A operand shorter than its footprint"
    );
}

#[test]
fn image_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let (n, c, h, w, r) = (1i32, 2i32, 3i32, 3i32, 2i32);
    let cin = c * r * r;
    let input = zeros_f32(&ctx, (n * cin * h * w) as usize / 2);
    let mut output = zeros_f32(&ctx, (n * c * h * r * w * r) as usize);
    let desc = PixelShuffleDescriptor {
        n,
        c,
        h,
        w,
        upscale_factor: r,
        element: ElementKind::F32,
    };
    let plan = PixelShufflePlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        PixelShuffleArgs {
            input: TensorRef {
                data: input.as_slice(),
                shape: [n, cin, h, w],
                stride: contiguous_stride([n, cin, h, w]),
            },
            output: TensorMut {
                data: output.as_slice_mut(),
                shape: [n, c, h * r, w * r],
                stride: contiguous_stride([n, c, h * r, w * r]),
            },
        },
    );
    assert!(
        res.is_err(),
        "PixelShufflePlan::run accepted a short input slice"
    );
}

#[test]
fn indexing_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let shape = [4i32, 6];
    let numel = 24usize;
    let src = zeros_f32(&ctx, numel / 2);
    let mask = DeviceBuffer::from_slice(&ctx, &vec![0u8; numel]).unwrap();
    let mut out = zeros_f32(&ctx, numel);
    let desc = MaskedFillDescriptor::new_f32(shape, 1.0);
    let plan = MaskedFillPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        MaskedFillArgs::<f32, 2> {
            src: TensorRef {
                data: src.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            mask: TensorRef {
                data: mask.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            out: TensorMut {
                data: out.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    );
    assert!(
        res.is_err(),
        "MaskedFillPlan::run accepted a short src slice"
    );
}

#[test]
fn linalg_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let n = 6i32;
    let mut a = zeros_f32(&ctx, (n * n) as usize / 2); // shape needs n*n
    let mut info = zeros_i32(&ctx, 1);
    let mut ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, 1 << 20).unwrap();
    let desc = CholeskyDescriptor {
        matrix_size: n,
        batch_size: 1,
        lower: true,
        element: ElementKind::F32,
    };
    let plan = CholeskyPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let shape = [1i32, n, n];
    let res = plan.run(
        &stream,
        Workspace::Borrowed(ws.as_slice_mut()),
        CholeskyArgs::<f32> {
            a: TensorMut {
                data: a.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
            info: TensorMut {
                data: info.as_slice_mut(),
                shape: [1i32],
                stride: [1],
            },
        },
    );
    assert!(res.is_err(), "CholeskyPlan::run accepted a short a slice");
}

#[test]
fn norm_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let shape = [3i32, 4, 8];
    let numel = 96usize;
    let x = zeros_f32(&ctx, numel / 2);
    let gamma = zeros_f32(&ctx, 8);
    let mut y = zeros_f32(&ctx, numel);
    let desc = RMSNormDescriptor {
        input_shape: shape,
        norm_axes_mask: 4,
        eps: 1e-5,
        has_gamma: true,
        element: ElementKind::F32,
    };
    let plan = RMSNormPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let rms_shape = desc.rms_shape();
    let mut rms = zeros_f32(&ctx, rms_shape.iter().product::<i32>() as usize);
    let res = plan.run(
        &stream,
        Workspace::None,
        RMSNormArgs {
            x: TensorRef {
                data: x.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            gamma: Some(TensorRef {
                data: gamma.as_slice(),
                shape: [8i32],
                stride: [1],
            }),
            y: TensorMut {
                data: y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
            rms: TensorMut {
                data: rms.as_slice_mut(),
                shape: rms_shape,
                stride: contiguous_stride(rms_shape),
            },
        },
    );
    assert!(res.is_err(), "RMSNormPlan::run accepted a short x slice");
}

#[test]
fn quantize_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let numel = 10i32;
    let q: DeviceBuffer<S8> = DeviceBuffer::zeros(&ctx, numel as usize / 2).unwrap();
    let mut x = zeros_f32(&ctx, numel as usize);
    let desc = DequantizePerTensorDescriptor {
        numel,
        input_element: ElementKind::F32,
        output_element: ElementKind::I8,
    };
    let plan =
        DequantizePerTensorPlan::<f32, S8>::select(&stream, &desc, PlanPreference::default())
            .unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        DequantizePerTensorArgs::<f32, S8> {
            input: TensorRef {
                data: q.as_slice(),
                shape: [numel],
                stride: contiguous_stride([numel]),
            },
            scale: 0.5,
            zero_point: 0,
            output: TensorMut {
                data: x.as_slice_mut(),
                shape: [numel],
                stride: contiguous_stride([numel]),
            },
        },
    );
    assert!(
        res.is_err(),
        "DequantizePerTensorPlan::run accepted a short input slice"
    );
}

#[test]
fn random_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    const N: usize = 256;
    let shape = [N as i32];
    let stride = contiguous_stride(shape);
    let x = zeros_f32(&ctx, N);
    let mut y = zeros_f32(&ctx, N);
    let mut mask: DeviceBuffer<Bool> = DeviceBuffer::zeros(&ctx, N / 2).unwrap();
    let mut ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, N * 4).unwrap();
    let desc = DropoutDescriptor {
        shape,
        element: ElementKind::F32,
        p: 0.5,
        seed: 1234,
    };
    let plan = DropoutPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::Borrowed(ws.as_slice_mut()),
        DropoutArgs::<f32, 1> {
            x: TensorRef {
                data: x.as_slice(),
                shape,
                stride,
            },
            y: TensorMut {
                data: y.as_slice_mut(),
                shape,
                stride,
            },
            mask: TensorMut {
                data: mask.as_slice_mut(),
                shape,
                stride,
            },
        },
    );
    assert!(res.is_err(), "DropoutPlan::run accepted a short mask slice");
}

#[test]
fn reduce_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 8];
    let output_shape = [4i32, 1];
    let x = zeros_f32(&ctx, 16); // shape needs 32
    let mut y = zeros_f32(&ctx, 4);
    let desc = ReduceDescriptor {
        kind: ReduceKind::Sum,
        input_shape,
        reduce_axis: 1,
        element: ElementKind::F32,
        correction: 1,
    };
    let plan = ReducePlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        ReduceArgs::<f32, 2> {
            x: TensorRef {
                data: x.as_slice(),
                shape: input_shape,
                stride: contiguous_stride(input_shape),
            },
            y: TensorMut {
                data: y.as_slice_mut(),
                shape: output_shape,
                stride: contiguous_stride(output_shape),
            },
        },
    );
    assert!(res.is_err(), "ReducePlan::run accepted a short x slice");
}

#[test]
fn scan_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let shape = [16i32];
    let x = zeros_f32(&ctx, 8);
    let mut y = zeros_f32(&ctx, 16);
    let desc = ScanDescriptor {
        kind: ScanKind::Cumsum,
        input_shape: shape,
        scan_axis: 0,
        reverse: false,
        element: ElementKind::F32,
    };
    let plan = ScanPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        ScanArgs::<f32, 1> {
            x: TensorRef {
                data: x.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            y: TensorMut {
                data: y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    );
    assert!(res.is_err(), "ScanPlan::run accepted a short x slice");
}

#[test]
fn segment_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let (n, d, ns) = (8i32, 3i32, 4i32);
    let input = zeros_f32(&ctx, (n * d) as usize / 2);
    let seg = DeviceBuffer::from_slice(&ctx, &[0i32, 0, 1, 1, 2, 2, 2, 3]).unwrap();
    let mut out = zeros_f32(&ctx, (ns * d) as usize);
    let desc = SegmentSumDescriptor {
        num_inputs: n,
        embedding_dim: d,
        num_segments: ns,
        element: ElementKind::F32,
    };
    let plan = SegmentSumPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        SegmentSumArgs::<f32> {
            input: TensorRef {
                data: input.as_slice(),
                shape: [n, d],
                stride: contiguous_stride([n, d]),
            },
            segment_ids: TensorRef {
                data: seg.as_slice(),
                shape: [n],
                stride: contiguous_stride([n]),
            },
            output: TensorMut {
                data: out.as_slice_mut(),
                shape: [ns, d],
                stride: contiguous_stride([ns, d]),
            },
        },
    );
    assert!(
        res.is_err(),
        "SegmentSumPlan::run accepted a short input slice"
    );
}

#[test]
fn shape_layout_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let shape = [16i32];
    let x = zeros_f32(&ctx, 8);
    let mut y = zeros_f32(&ctx, 16);
    let desc = FlipDescriptor {
        shape,
        flip_axes: [true],
        element: ElementKind::F32,
    };
    let plan = FlipPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        FlipArgs::<f32, 1> {
            x: TensorRef {
                data: x.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            y: TensorMut {
                data: y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    );
    assert!(res.is_err(), "FlipPlan::run accepted a short x slice");
}

#[test]
fn softmax_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let shape = [2i32, 5, 4];
    let x = zeros_f32(&ctx, 20); // shape needs 40
    let mut y = zeros_f32(&ctx, 40);
    let desc = SoftmaxDescriptor {
        kind: SoftmaxKind::Softmax,
        input_shape: shape,
        softmax_axis: 1,
        element: ElementKind::F32,
    };
    let plan = SoftmaxPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        SoftmaxArgs {
            x: TensorRef {
                data: x.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            y: TensorMut {
                data: y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    );
    assert!(res.is_err(), "SoftmaxPlan::run accepted a short x slice");
}

#[test]
fn sort_rejects_view_longer_than_its_slice() {
    let (ctx, stream) = setup();
    let (batch, row_len) = (3i32, 8i32);
    let numel = (batch * row_len) as usize;
    let input = zeros_f32(&ctx, numel / 2);
    let mut values = zeros_f32(&ctx, numel);
    let mut indices = zeros_i32(&ctx, numel);
    let desc = SortDescriptor {
        batch,
        row_len,
        descending: false,
        element: ElementKind::F32,
    };
    let plan = SortPlan::<f32>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let shape = [batch, row_len];
    let res = plan.run(
        &stream,
        Workspace::None,
        SortArgs::<f32> {
            input: TensorRef {
                data: input.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            values: TensorMut {
                data: values.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
            indices: TensorMut {
                data: indices.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    );
    assert!(res.is_err(), "SortPlan::run accepted a short input slice");
}
