//! Soundness: safe `BinaryPlan::run` must not read or write outside the
//! `DeviceSlice`s it is handed, whatever strides the (public-field)
//! `TensorRef` / `TensorMut` carry.
//!
//! Both OOB cases stay inside one larger allocation, so they corrupt or
//! leak neighbouring memory *without* faulting — the failure shape that
//! matters. Each test asserts the launch is rejected; if it is not, it
//! also reports what escaped.
//!
//! `cargo test -p baracuda-kernels --features sm89,device-tests --test soundness_tensor_view_bounds`
#![cfg(feature = "device-tests")]

use baracuda_driver::{Context, Device, DeviceBuffer, Stream, init};
use baracuda_kernels::{
    Bin, BinGemmArgs, BinGemmDescriptor, BinGemmPlan, BinaryArgs, BinaryDescriptor, BinaryKind,
    BinaryPlan, ContiguizeArgs, ContiguizeDescriptor, ContiguizePlan, ElementKind, LayoutSku,
    LossReduction, MatrixMut, MatrixRef, MseLossArgs, MseLossDescriptor, MseLossPlan,
    PlanPreference, TensorMut, TensorRef, Workspace, contiguous_stride,
};

const N: usize = 64;

fn setup() -> (Context, Stream, BinaryPlan<f32, 1>) {
    init().expect("driver init");
    let ctx = Context::new(&Device::get(0).expect("device 0")).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    let desc = BinaryDescriptor {
        kind: BinaryKind::Add,
        shape: [N as i32],
        element: ElementKind::F32,
    };
    let plan = BinaryPlan::<f32, 1>::select(&stream, &desc, PlanPreference::default()).unwrap();
    (ctx, stream, plan)
}

/// `y` is `big[0..N]` with element stride 2: the launch would write
/// `big[0], big[2], .. big[2N-2]` — half of it past the slice it was given.
#[test]
fn strided_output_view_cannot_write_past_its_slice() {
    let (ctx, stream, plan) = setup();
    let ones = DeviceBuffer::from_slice(&ctx, &vec![1.0f32; N]).unwrap();
    let mut big = DeviceBuffer::from_slice(&ctx, &vec![-1.0f32; 2 * N]).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        BinaryArgs {
            a: TensorRef {
                data: ones.as_slice(),
                shape: [N as i32],
                stride: [1],
            },
            b: TensorRef {
                data: ones.as_slice(),
                shape: [N as i32],
                stride: [1],
            },
            y: TensorMut {
                data: big.slice_mut(0..N),
                shape: [N as i32],
                stride: [2],
            },
        },
    );
    stream.synchronize().unwrap();
    let mut host = vec![0f32; 2 * N];
    big.copy_to_host(&mut host).unwrap();
    let clobbered = host[N..].iter().filter(|&&v| v != -1.0).count();
    assert!(
        res.is_err(),
        "run() accepted a y view whose strides reach past its slice; {clobbered} of {N} \
         elements outside the slice were overwritten"
    );
    assert_eq!(clobbered, 0, "memory outside y's slice was written");
}

/// `a` is `big[0..N]` with element stride 2: the launch reads
/// `big[0], big[2], .. big[2N-2]` — values it was never given.
#[test]
fn strided_input_view_cannot_read_past_its_slice() {
    let (ctx, stream, plan) = setup();
    // first half 0.0, second half 1000.0 (the "secret" neighbour memory)
    let mut init_host = vec![0.0f32; N];
    init_host.extend(vec![1000.0f32; N]);
    let big = DeviceBuffer::from_slice(&ctx, &init_host).unwrap();
    let zeros = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; N]).unwrap();
    let mut y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, N).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        BinaryArgs {
            a: TensorRef {
                data: big.slice(0..N),
                shape: [N as i32],
                stride: [2],
            },
            b: TensorRef {
                data: zeros.as_slice(),
                shape: [N as i32],
                stride: [1],
            },
            y: TensorMut {
                data: y.as_slice_mut(),
                shape: [N as i32],
                stride: [1],
            },
        },
    );
    stream.synchronize().unwrap();
    let mut host = vec![0f32; N];
    y.copy_to_host(&mut host).unwrap();
    let leaked = host.iter().filter(|&&v| v == 1000.0).count();
    assert!(
        res.is_err(),
        "run() accepted an a view whose strides reach past its slice; {leaked} of {N} \
         outputs contain values from outside the slice"
    );
    assert_eq!(leaked, 0, "values outside a's slice were read");
}

/// Plans with no `can_implement` (loss, conv, fft, linalg, ..) are guarded in
/// `run()` itself. `pred` claims shape `[2, 8]` but its slice holds 8 elements;
/// the stride-ignoring kernel would read 16.
#[test]
fn run_only_family_rejects_view_longer_than_its_slice() {
    let (ctx, stream, _) = setup();
    let shape = [2i32, 8];
    let numel = 16usize;
    let short = DeviceBuffer::from_slice(&ctx, &vec![1.0f32; numel / 2]).unwrap();
    let full = DeviceBuffer::from_slice(&ctx, &vec![1.0f32; numel]).unwrap();
    let mut y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 1).unwrap();
    let mut ws: DeviceBuffer<u8> = DeviceBuffer::zeros(&ctx, numel * 4).unwrap();
    let desc = MseLossDescriptor {
        input_shape: shape,
        reduction: LossReduction::Mean,
        element: ElementKind::F32,
    };
    let plan = MseLossPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::Borrowed(ws.as_slice_mut()),
        MseLossArgs {
            pred: TensorRef {
                data: short.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            target: TensorRef {
                data: full.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            out: TensorMut {
                data: y.as_slice_mut(),
                shape: [1, 1],
                stride: [1, 1],
            },
        },
    );
    stream.synchronize().unwrap();
    assert!(
        res.is_err(),
        "run() accepted a pred view of 16 elements over an 8-element slice"
    );
}

/// GEMM operands carry `rows` / `cols` / `ld` as public fields. `BinGemmPlan::run`
/// used to launch without any length check: here A's slice holds half the bytes
/// the `[M, K]` bit matrix needs.
#[test]
fn gemm_matrix_operand_shorter_than_its_footprint_is_rejected() {
    let (ctx, stream, _) = setup();
    let (m, n, k) = (128i32, 128i32, 512i32);
    let k_bytes = (k / 8) as usize;
    let a_short = DeviceBuffer::from_slice(&ctx, &vec![0u8; m as usize * k_bytes / 2]).unwrap();
    let b = DeviceBuffer::from_slice(&ctx, &vec![0u8; n as usize * k_bytes]).unwrap();
    let mut d: DeviceBuffer<i32> = DeviceBuffer::zeros(&ctx, (m * n) as usize).unwrap();
    let desc = BinGemmDescriptor {
        m,
        n,
        k,
        layout: LayoutSku::Rcr,
    };
    let plan = BinGemmPlan::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        BinGemmArgs {
            a: MatrixRef {
                data: a_short.view_as::<Bin>(),
                rows: m,
                cols: k,
                ld: k_bytes as i64,
            },
            b: MatrixRef {
                data: b.view_as::<Bin>(),
                rows: k,
                cols: n,
                ld: k_bytes as i64,
            },
            d: MatrixMut {
                data: d.as_slice_mut(),
                rows: m,
                cols: n,
                ld: n as i64,
            },
        },
    );
    stream.synchronize().unwrap();
    assert!(
        res.is_err(),
        "run() accepted an A operand shorter than its footprint"
    );
}

/// `ContiguizePlan` addresses `source_offset + sum(i * stride)` from the start
/// of its source slice. The offset lives in the (public) descriptor, so the
/// plan has to bound it: here it points 4 elements past the end of the slice.
#[test]
fn contiguize_source_offset_past_the_slice_is_rejected() {
    let (ctx, stream, _) = setup();
    let shape = [4i32, 3];
    let numel = 12usize;
    let src = DeviceBuffer::from_slice(&ctx, &vec![1.0f32; numel]).unwrap();
    let mut y: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, numel).unwrap();
    let desc = ContiguizeDescriptor::<2> {
        shape,
        source_strides: contiguous_stride(shape),
        source_offset: 4,
        element: ElementKind::F32,
    };
    let plan = ContiguizePlan::<f32, 2>::select(&stream, &desc, PlanPreference::default()).unwrap();
    let res = plan.run(
        &stream,
        Workspace::None,
        ContiguizeArgs::<f32, 2> {
            source: TensorRef {
                data: src.as_slice(),
                shape,
                stride: contiguous_stride(shape),
            },
            dest: TensorMut {
                data: y.as_slice_mut(),
                shape,
                stride: contiguous_stride(shape),
            },
        },
    );
    stream.synchronize().unwrap();
    assert!(
        res.is_err(),
        "run() accepted source_offset 4 over a 12-element slice needing 16"
    );
}
