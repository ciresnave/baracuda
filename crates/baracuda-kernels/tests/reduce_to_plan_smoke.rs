//! Real-GPU smoke test for the Phase 74 broadcast-reverse reduction
//! facade `ReduceToPlan<T, N>` — the plan-level sibling of the Phase 31
//! direct-FFI `reduce_sum_to_smoke` / Phase 37 `reduce_to_extras_smoke`.
//!
//! Covers all four ops on f32 (`[2, 3, 4] → [2, 1, 1]` — two reduced
//! dims), one f16 Sum case (accumulate-in-f32 tolerance), one f32
//! strided-input case (transposed view), and the Prod identity on an
//! empty reduce set (`[0, 4] → [1, 4]` ⇒ all 1s). CPU references are
//! computed in f64.
//!
//! `#[ignore]` by default; run with
//! `cargo test -p baracuda-kernels --release --features sm89 \
//!   --test reduce_to_plan_smoke -- --ignored`.

use baracuda_driver::{init, Context, Device, DeviceBuffer, Stream};
use baracuda_kernels::{
    contiguous_stride, ElementKind, PlanPreference, ReduceToArgs, ReduceToDescriptor,
    ReduceToOp, ReduceToPlan, TensorMut, TensorRef, Workspace,
};
use half::f16;

// 1-ULP relative-tolerance constant (same scheme as the unary smoke
// tests). Sum accumulates in f32 and narrows once on store, so the
// final f16 rounding dominates.
const F16_EPS: f32 = 9.77e-4;

fn setup() -> (Context, Stream) {
    init().expect("driver init");
    let device = Device::get(0).expect("device 0");
    let ctx = Context::new(&device).expect("context");
    let stream = Stream::new(&ctx).expect("stream");
    (ctx, stream)
}

/// CPU reference in f64: for each input cell, fold it into the output
/// cell it broadcasts to (matches the kernel semantics — output coord
/// is 0 on reduced dims, the input coord elsewhere).
fn cpu_reduce_to_f64(
    src: &[f64],
    in_shape: &[i32],
    in_stride: &[i64],
    out_shape: &[i32],
    op: ReduceToOp,
) -> Vec<f64> {
    let rank = in_shape.len();
    let out_numel: usize = out_shape.iter().map(|&d| d as usize).product();
    // Identity per op — matches the kernel's policies (finite extremes
    // for Max / Min, not ±inf; replaced by the first combine on any
    // non-empty broadcast set).
    let identity = match op {
        ReduceToOp::Sum => 0.0,
        ReduceToOp::Max => -f64::MAX,
        ReduceToOp::Min => f64::MAX,
        ReduceToOp::Prod => 1.0,
        _ => unreachable!("non-exhaustive ReduceToOp in test reference"),
    };
    let mut dst = vec![identity; out_numel];

    let in_numel: usize = in_shape.iter().map(|&d| d as usize).product();
    let out_contig: Vec<i64> = {
        let mut s = vec![0i64; rank];
        let mut acc: i64 = 1;
        for i in (0..rank).rev() {
            s[i] = acc;
            acc *= out_shape[i] as i64;
        }
        s
    };

    for in_lin in 0..in_numel {
        let mut lin = in_lin;
        let mut in_coord = vec![0i32; rank];
        for d in (0..rank).rev() {
            let s = in_shape[d] as usize;
            in_coord[d] = (lin % s) as i32;
            lin /= s;
        }
        let in_off: i64 = (0..rank)
            .map(|d| (in_coord[d] as i64) * in_stride[d])
            .sum();
        let mut out_lin: usize = 0;
        for d in 0..rank {
            let c = if out_shape[d] == 1 { 0 } else { in_coord[d] };
            out_lin += (c as usize) * (out_contig[d] as usize);
        }
        let v = src[in_off as usize];
        dst[out_lin] = match op {
            ReduceToOp::Sum => dst[out_lin] + v,
            ReduceToOp::Max => dst[out_lin].max(v),
            ReduceToOp::Min => dst[out_lin].min(v),
            ReduceToOp::Prod => dst[out_lin] * v,
            _ => unreachable!("non-exhaustive ReduceToOp in test reference"),
        };
    }
    dst
}

/// Drive the plan for one f32 (op, fixture) cell on `[2, 3, 4] →
/// [2, 1, 1]` (reduces dims 1 and 2) and compare against the f64
/// reference.
fn run_f32_3d(op: ReduceToOp, host_src: &[f32]) {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 3, 4];
    let output_shape = [2i32, 1, 1];
    let in_stride = contiguous_stride(input_shape);
    let out_stride = contiguous_stride(output_shape);
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();

    let src_f64: Vec<f64> = host_src.iter().map(|&v| v as f64).collect();
    let expected = cpu_reduce_to_f64(&src_f64, &input_shape, &in_stride, &output_shape, op);

    let dev_src = DeviceBuffer::from_slice(&ctx, host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc dst");

    let desc = ReduceToDescriptor {
        op,
        input_shape,
        output_shape,
        element: ElementKind::F32,
    };
    let plan = ReduceToPlan::<f32, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select ReduceToPlan<f32, 3>");
    assert_eq!(plan.workspace_size(), 0, "reduce_to needs no workspace");
    let args = ReduceToArgs::<f32, 3> {
        x: TensorRef { data: dev_src.as_slice(), shape: input_shape, stride: in_stride },
        y: TensorMut { data: dev_dst.as_slice_mut(), shape: output_shape, stride: out_stride },
    };
    plan.run(&stream, Workspace::None, args).expect("reduce_to f32 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; out_numel];
    dev_dst.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (*g as f64 - e).abs();
        let allow = e.abs().max(1.0) * 1e-5;
        assert!(diff <= allow,
            "reduce_to {op:?} f32 [2,3,4]→[2,1,1] @ {i}: got {g} expected {e} \
             (diff {diff} > allow {allow})");
    }
}

#[test]
#[ignore]
fn plan_reduce_to_sum_f32_3d() {
    let host_src: Vec<f32> = (0..24).map(|i| (i as f32) * 0.05 - 0.5).collect();
    run_f32_3d(ReduceToOp::Sum, &host_src);
}

#[test]
#[ignore]
fn plan_reduce_to_max_f32_3d() {
    let host_src: Vec<f32> = (0..24).map(|i| (i as f32) * 0.13 - 1.5).collect();
    run_f32_3d(ReduceToOp::Max, &host_src);
}

#[test]
#[ignore]
fn plan_reduce_to_min_f32_3d() {
    let host_src: Vec<f32> = (0..24).map(|i| (i as f32) * 0.13 - 1.5).collect();
    run_f32_3d(ReduceToOp::Min, &host_src);
}

#[test]
#[ignore]
fn plan_reduce_to_prod_f32_3d() {
    // Keep values close to 1 so the 12-element product stays in a
    // numerically safe range.
    let host_src: Vec<f32> = (0..24).map(|i| 0.85 + (i as f32) * 0.01).collect();
    run_f32_3d(ReduceToOp::Prod, &host_src);
}

/// f16 Sum: the kernel accumulates in f32 (PyTorch convention) and
/// narrows once on store, so the tolerance is the f16 ULP of the
/// result, not the f16 ULP compounded over the 12-element walk.
#[test]
#[ignore]
fn plan_reduce_to_sum_f16_3d() {
    let (ctx, stream) = setup();
    let input_shape = [2i32, 3, 4];
    let output_shape = [2i32, 1, 1];
    let in_stride = contiguous_stride(input_shape);
    let out_stride = contiguous_stride(output_shape);
    let out_numel: usize = output_shape.iter().map(|&d| d as usize).product();

    let host_src_f32: Vec<f32> = (0..24).map(|i| (i as f32) * 0.05 - 0.5).collect();
    let host_src: Vec<f16> = host_src_f32.iter().map(|&v| f16::from_f32(v)).collect();

    // Reference in f64 over the f16-rounded inputs (what the kernel
    // actually sees).
    let src_f64: Vec<f64> = host_src.iter().map(|v| v.to_f32() as f64).collect();
    let expected = cpu_reduce_to_f64(
        &src_f64, &input_shape, &in_stride, &output_shape, ReduceToOp::Sum,
    );

    let dev_src = DeviceBuffer::from_slice(&ctx, &host_src).expect("upload src");
    let mut dev_dst: DeviceBuffer<f16> = DeviceBuffer::zeros(&ctx, out_numel).expect("alloc dst");

    let desc = ReduceToDescriptor {
        op: ReduceToOp::Sum,
        input_shape,
        output_shape,
        element: ElementKind::F16,
    };
    let plan = ReduceToPlan::<f16, 3>::select(&stream, &desc, PlanPreference::default())
        .expect("select ReduceToPlan<f16, 3>");
    let args = ReduceToArgs::<f16, 3> {
        x: TensorRef { data: dev_src.as_slice(), shape: input_shape, stride: in_stride },
        y: TensorMut { data: dev_dst.as_slice_mut(), shape: output_shape, stride: out_stride },
    };
    plan.run(&stream, Workspace::None, args).expect("reduce_to sum f16 run");
    stream.synchronize().expect("sync");

    let mut got_f16 = vec![f16::from_f32(0.0); out_numel];
    dev_dst.copy_to_host(&mut got_f16).expect("download");
    for (i, (g, e)) in got_f16.iter().zip(expected.iter()).enumerate() {
        let g_f32 = g.to_f32();
        let diff = (g_f32 as f64 - e).abs();
        let allow = e.abs().max(1.0) * (4.0 * F16_EPS) as f64;
        assert!(diff <= allow,
            "reduce_to sum f16 @ {i}: got {g_f32} expected {e} (diff {diff} > allow {allow})");
    }
}

/// Strided input: reduce a transposed view. `x` is `[m, n]` with
/// stride `[1, m]` (column-major over a contiguous `[n, m]` buffer);
/// dim 0 reduces → `[1, n]`.
#[test]
#[ignore]
fn plan_reduce_to_sum_f32_strided_transposed() {
    let (ctx, stream) = setup();
    const M: usize = 48;
    const N_DIM: usize = 32;
    let input_shape = [M as i32, N_DIM as i32];
    let output_shape = [1i32, N_DIM as i32];
    let in_stride = [1i64, M as i64]; // transposed view
    let out_stride = contiguous_stride(output_shape);

    let x_buf: Vec<f32> = (0..(N_DIM * M)).map(|i| (i as f32) * 0.01 - 1.75).collect();
    let src_f64: Vec<f64> = x_buf.iter().map(|&v| v as f64).collect();
    let expected = cpu_reduce_to_f64(
        &src_f64, &input_shape, &in_stride, &output_shape, ReduceToOp::Sum,
    );

    let dev_src = DeviceBuffer::from_slice(&ctx, &x_buf).expect("upload src");
    let mut dev_dst: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, N_DIM).expect("alloc dst");

    let desc = ReduceToDescriptor {
        op: ReduceToOp::Sum,
        input_shape,
        output_shape,
        element: ElementKind::F32,
    };
    let plan = ReduceToPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select ReduceToPlan<f32, 2>");
    let args = ReduceToArgs::<f32, 2> {
        x: TensorRef { data: dev_src.as_slice(), shape: input_shape, stride: in_stride },
        y: TensorMut { data: dev_dst.as_slice_mut(), shape: output_shape, stride: out_stride },
    };
    plan.run(&stream, Workspace::None, args).expect("reduce_to strided run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; N_DIM];
    dev_dst.copy_to_host(&mut got).expect("download");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (*g as f64 - e).abs();
        let allow = e.abs().max(1.0) * 1e-5;
        assert!(diff <= allow,
            "reduce_to sum f32 strided @ {i}: got {g} expected {e} \
             (diff {diff} > allow {allow})");
    }
}

/// Stride-0 broadcast INPUT view: `x` is logically `[4, 5]` but its
/// dim-0 stride is 0 over a 5-element buffer (a broadcast view flowing
/// back into the broadcast-reverse reduction — the doubly-broadcast
/// gradient case). `numel` (20) exceeds the storage extent (5), so
/// this locks in the span-based `can_implement` bound (a naive
/// `len >= numel` check would falsely reject it). Sum over the
/// reduced dim 0 hits the same buffer row 4× ⇒ `out[j] = 4·src[j]`.
#[test]
#[ignore]
fn plan_reduce_to_sum_f32_stride0_broadcast_input() {
    let (ctx, stream) = setup();
    let input_shape = [4i32, 5];
    let output_shape = [1i32, 5];
    let in_stride = [0i64, 1]; // dim 0 broadcast over a 5-element buffer
    let out_stride = contiguous_stride(output_shape);

    let x_buf: Vec<f32> = (0..5).map(|i| (i as f32) * 0.75 - 1.5).collect();
    let dev_src = DeviceBuffer::from_slice(&ctx, &x_buf).expect("upload src");
    let mut dev_dst: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 5).expect("alloc dst");

    let desc = ReduceToDescriptor {
        op: ReduceToOp::Sum,
        input_shape,
        output_shape,
        element: ElementKind::F32,
    };
    let plan = ReduceToPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select ReduceToPlan<f32, 2>");
    let args = ReduceToArgs::<f32, 2> {
        x: TensorRef { data: dev_src.as_slice(), shape: input_shape, stride: in_stride },
        y: TensorMut { data: dev_dst.as_slice_mut(), shape: output_shape, stride: out_stride },
    };
    plan.run(&stream, Workspace::None, args).expect("reduce_to stride-0 run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 5];
    dev_dst.copy_to_host(&mut got).expect("download");
    for (i, (g, &x)) in got.iter().zip(x_buf.iter()).enumerate() {
        let e = 4.0 * x;
        assert!(
            (g - e).abs() <= e.abs().max(1.0) * 1e-6,
            "reduce_to stride-0 broadcast @ {i}: got {g} expected {e}",
        );
    }
}

/// Prod over an empty reduce set: `[0, 4] → [1, 4]` — dim 0 is reduced
/// but has extent 0, so every output cell gets the identity `1`.
#[test]
#[ignore]
fn plan_reduce_to_prod_f32_empty_reduce_set_identity() {
    let (ctx, stream) = setup();
    let input_shape = [0i32, 4];
    let output_shape = [1i32, 4];
    let in_stride = contiguous_stride(input_shape);
    let out_stride = contiguous_stride(output_shape);

    // The source has zero logical elements — a 1-element dummy
    // allocation keeps the upload path happy; the kernel never reads it.
    let dev_src = DeviceBuffer::from_slice(&ctx, &[0f32]).expect("upload dummy src");
    let mut dev_dst: DeviceBuffer<f32> = DeviceBuffer::zeros(&ctx, 4).expect("alloc dst");

    let desc = ReduceToDescriptor {
        op: ReduceToOp::Prod,
        input_shape,
        output_shape,
        element: ElementKind::F32,
    };
    let plan = ReduceToPlan::<f32, 2>::select(&stream, &desc, PlanPreference::default())
        .expect("select ReduceToPlan<f32, 2>");
    let args = ReduceToArgs::<f32, 2> {
        x: TensorRef { data: dev_src.as_slice(), shape: input_shape, stride: in_stride },
        y: TensorMut { data: dev_dst.as_slice_mut(), shape: output_shape, stride: out_stride },
    };
    plan.run(&stream, Workspace::None, args).expect("reduce_to prod empty run");
    stream.synchronize().expect("sync");

    let mut got = vec![0f32; 4];
    dev_dst.copy_to_host(&mut got).expect("download");
    for (i, g) in got.iter().enumerate() {
        assert_eq!(
            g.to_bits(), 1.0f32.to_bits(),
            "reduce_to prod empty set @ {i}: got {g} expected identity 1.0",
        );
    }
}
