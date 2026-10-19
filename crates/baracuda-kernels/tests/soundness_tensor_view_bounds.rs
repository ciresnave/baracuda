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

use baracuda_driver::{Context, Device, DeviceBuffer, DeviceSlice, DeviceSliceMut, Stream, init};
use baracuda_kernels::{
    Bin, BinGemmArgs, BinGemmDescriptor, BinGemmPlan, BinaryArgs, BinaryDescriptor, BinaryKind,
    BinaryPlan, ContiguizeArgs, ContiguizeDescriptor, ContiguizePlan, ElementKind, LayoutSku,
    LossReduction, MatrixMut, MatrixRef, MseLossArgs, MseLossDescriptor, MseLossPlan,
    PlanPreference, TensorMut, TensorRef, ViewError, Workspace, contiguous_stride,
};

/// Build a view WITHOUT the constructor's bounds check. These tests exist to
/// prove each plan rejects an out-of-bounds view on its own (defence in depth),
/// and `TensorRef::new` / `TensorMut::new` would refuse to build one.
fn bad_ref<'a, T: baracuda_types::DeviceRepr + Copy + 'static, const N: usize>(
    data: DeviceSlice<'a, T>,
    shape: [i32; N],
    stride: [i64; N],
) -> TensorRef<'a, T, N> {
    // SAFETY: deliberately out of bounds. The view is only ever handed to a plan
    // whose own check must return `Err` before any kernel is launched.
    unsafe { TensorRef::new_unchecked(data, shape, stride) }
}

/// Build a `MatrixRef` WITHOUT the constructor's slice-length check, so a test
/// can hand a plan an operand shorter than its footprint and prove the plan
/// rejects it on its own (defence in depth).
fn bad_matrix_ref<'a, T: baracuda_types::DeviceRepr + Copy + 'static>(
    data: DeviceSlice<'a, T>,
    rows: i32,
    cols: i32,
    ld: i64,
) -> MatrixRef<'a, T> {
    // SAFETY: deliberately too short. It is only ever handed to a plan whose own
    // footprint check must return `Err` before any kernel is launched.
    unsafe { MatrixRef::new_unchecked(data, rows, cols, ld) }
}

/// Mutable counterpart of [`bad_ref`].
fn bad_mut<'a, T: baracuda_types::DeviceRepr + Copy + 'static, const N: usize>(
    data: DeviceSliceMut<'a, T>,
    shape: [i32; N],
    stride: [i64; N],
) -> TensorMut<'a, T, N> {
    // SAFETY: as for [`bad_ref`].
    unsafe { TensorMut::new_unchecked(data, shape, stride) }
}

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
            a: bad_ref(ones.as_slice(), [N as i32], [1]),
            b: bad_ref(ones.as_slice(), [N as i32], [1]),
            y: bad_mut(big.slice_mut(0..N), [N as i32], [2]),
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
            a: bad_ref(big.slice(0..N), [N as i32], [2]),
            b: bad_ref(zeros.as_slice(), [N as i32], [1]),
            y: bad_mut(y.as_slice_mut(), [N as i32], [1]),
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
            pred: bad_ref(short.as_slice(), shape, contiguous_stride(shape)),
            target: bad_ref(full.as_slice(), shape, contiguous_stride(shape)),
            out: bad_mut(y.as_slice_mut(), [1, 1], [1, 1]),
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
            a: bad_matrix_ref(a_short.view_as::<Bin>(), m, k, k_bytes as i64),
            b: MatrixRef::new(b.view_as::<Bin>(), k, n, k_bytes as i64).unwrap(),
            d: MatrixMut::new(d.as_slice_mut(), m, n, n as i64).unwrap(),
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
            source: bad_ref(src.as_slice(), shape, contiguous_stride(shape)),
            dest: bad_mut(y.as_slice_mut(), shape, contiguous_stride(shape)),
        },
    );
    stream.synchronize().unwrap();
    assert!(
        res.is_err(),
        "run() accepted source_offset 4 over a 12-element slice needing 16"
    );
}

/// The constructors are now the only safe way to build a view, and they refuse
/// the shapes that made the plans above unsound: `big[0..N]` at stride 2 reaches
/// `2N - 1` elements, and a negative stride walks before element 0.
#[test]
fn view_constructors_reject_what_the_plans_used_to_accept() {
    let (ctx, _stream, _) = setup();
    let big = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; 2 * N]).unwrap();
    let mut out = DeviceBuffer::from_slice(&ctx, &vec![0.0f32; 2 * N]).unwrap();

    let strided = TensorRef::new(big.slice(0..N), [N as i32], [2]);
    assert!(
        matches!(strided, Err(ViewError::TooSmall { needed, got }) if needed == 2 * N - 1 && got == N),
        "read view reaching past its slice must be refused"
    );
    let strided = TensorMut::new(out.slice_mut(0..N), [N as i32], [2]);
    assert!(
        matches!(strided, Err(ViewError::TooSmall { .. })),
        "write view reaching past its slice must be refused"
    );
    let negative = TensorRef::new(big.slice(0..N), [N as i32], [-1]);
    assert!(
        matches!(negative, Err(ViewError::NotAddressable)),
        "negative stride reaches before the slice"
    );
    let negative_extent = TensorRef::new(big.slice(0..N), [-1], [1]);
    assert!(matches!(negative_extent, Err(ViewError::NotAddressable)));
    // Positive controls: the same slice with an in-bounds stride, and a 2x slice at stride 2.
    assert!(TensorRef::new(big.slice(0..N), [N as i32], [1]).is_ok());
    assert!(TensorRef::new(big.slice(0..2 * N), [N as i32], [2]).is_ok());
    assert!(TensorRef::contiguous(big.slice(0..N), [N as i32]).is_ok());
}
