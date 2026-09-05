//! Loss-family throughput — baracuda vs PyTorch (Wave-1).
//!
//! Phase 29 cross-impl bench for the loss family: `mse`, `l1`,
//! `cross_entropy`, `nll`. Each op is timed against the frozen PyTorch
//! baseline (`bench-baselines/pytorch_*.json`), looked up by the shared
//! `R{rows}_C{cols}` shape key produced by
//! `tools/refresh_pytorch_baseline.py`.
//!
//! Dtypes: `f32` / `f16` / `bf16`. The narrow dtypes are a genuinely NEW
//! measurement, not a scaled `f32` one — baracuda promotes each element to
//! f32, computes, and narrows on store, so the work per cell differs in
//! kind and not only in width. Their baselines are generated on the box
//! alongside the f32 ones rather than derived from them.
//!
//! Reduction is `Mean` on both sides (baracuda `LossReduction::Mean`,
//! PyTorch default `reduction='mean'`) so the two time the same work: a
//! per-cell / per-row pass plus a tree reduction to a scalar. Timing is
//! data-independent (dense kernels), so deterministic fills suffice; the
//! fills are chosen non-degenerate (loss ≠ 0) so `assert_cell_live` proves
//! a live scalar rather than an all-zero one.

use baracuda_driver::DeviceBuffer;
use baracuda_kernels::{
    CrossEntropyLossArgs, CrossEntropyLossDescriptor, CrossEntropyLossPlan, CrossEntropyTargetKind,
    L1LossArgs, L1LossDescriptor, L1LossPlan, LossReduction, MseLossArgs, MseLossDescriptor,
    MseLossPlan, NllLossArgs, NllLossDescriptor, NllLossPlan, PlanPreference, TensorMut, TensorRef,
    Workspace, contiguous_stride,
};
use baracuda_kernels_bench::{
    LOSS_COL_SWEEP, LOSS_ROW_SWEEP, LiveScalar, PhaseTwentyNineRow, PytorchBaseline,
    append_csv_row, assert_cell_live, measure_median_ns, setup_device, time_with_events, warmup,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use half::{bf16, f16};

const BENCH_NAME: &str = "loss";

/// Build a bench fill from an `f32` literal. The three dtypes need three
/// different constructors, and the fills are the only place the op bodies
/// differ across dtypes — so this keeps the benches themselves generic.
trait FillFrom: Copy {
    fn fill(v: f32) -> Self;
}
impl FillFrom for f32 {
    fn fill(v: f32) -> Self {
        v
    }
}
impl FillFrom for f16 {
    fn fill(v: f32) -> Self {
        f16::from_f32(v)
    }
}
impl FillFrom for bf16 {
    fn fill(v: f32) -> Self {
        bf16::from_f32(v)
    }
}

/// Everything a loss bench needs of its element type.
trait LossElem: baracuda_kernels::Element + Copy + 'static + LiveScalar + FillFrom {}
impl<T> LossElem for T where T: baracuda_kernels::Element + Copy + 'static + LiveScalar + FillFrom {}

/// Shared tail: warm the just-populated scalar, assert it is live, record
/// the baracuda-vs-PyTorch CSV row, then run the criterion timing. `$run`
/// is the launch block (re-expanded per closure; borrows the device
/// buffers in scope). `$dev_y` is the output scalar (numel 1).
macro_rules! bench_tail {
    ($ctx:expr, $stream:expr, $group:expr, $op:literal, $dt:expr, $shape:expr, $dev_y:expr, $baseline:expr, $run:block) => {{
        warmup(&$stream, || $run);
        // Liveness (NOT a reference): the warmup launch populated the scalar
        // loss; assert it is finite before timing it — a bench over a
        // NaN/garbage scalar is fast and meaningless.
        assert_cell_live(&format!("{}/{}/{}", $op, $dt, $shape), &$dev_y, 1);
        let baracuda_ns = measure_median_ns(&$ctx, &$stream, 11, 50, || $run);
        append_csv_row(
            BENCH_NAME,
            &PhaseTwentyNineRow {
                op: $op,
                shape: ($shape).clone(),
                dtype: $dt,
                baracuda_ns,
                reference_ns: None,
                reference: "",
                pytorch_ns: $baseline.and_then(|b| b.lookup($op, &$shape, $dt)),
            },
        );
        $group.bench_with_input(BenchmarkId::from_parameter(&$shape), &(), |bb, _| {
            bb.iter_custom(|iters| time_with_events(&$ctx, &$stream, iters, || $run))
        });
    }};
}

/// `mse` / `l1` share one API shape (per-cell `pred`/`target` → scalar).
macro_rules! percell_loss_bench {
    ($fn_name:ident, $op:literal, $Plan:ident, $Desc:ident, $Args:ident) => {
        fn $fn_name<T: LossElem>(
            c: &mut Criterion,
            dt: &'static str,
            baseline: Option<&PytorchBaseline>,
        ) {
            let (ctx, stream) = setup_device();
            let mut group = c.benchmark_group(format!("{}/{}", $op, dt));
            for &rows in LOSS_ROW_SWEEP {
                for &cols in LOSS_COL_SWEEP {
                    let shape = format!("R{rows}_C{cols}");
                    let ishape = [rows, cols];
                    let st = contiguous_stride(ishape);
                    let numel = (rows * cols) as usize;

                    let dev_p = match DeviceBuffer::from_slice(&ctx, &vec![T::fill(1.0); numel]) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let dev_t = match DeviceBuffer::from_slice(&ctx, &vec![T::fill(0.5); numel]) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, 1) {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let desc = $Desc::<2> {
                        input_shape: ishape,
                        reduction: LossReduction::Mean,
                        element: T::KIND,
                    };
                    let plan =
                        match $Plan::<T, 2>::select(&stream, &desc, PlanPreference::default()) {
                            Ok(p) => p,
                            Err(_) => continue,
                        };
                    let mut dev_ws: DeviceBuffer<u8> =
                        match DeviceBuffer::zeros(&ctx, plan.workspace_size().max(1)) {
                            Ok(b) => b,
                            Err(_) => continue,
                        };
                    bench_tail!(ctx, stream, group, $op, dt, shape, dev_y, baseline, {
                        plan.run(
                            &stream,
                            Workspace::Borrowed(dev_ws.as_slice_mut()),
                            $Args::<T, 2> {
                                pred: TensorRef {
                                    data: dev_p.as_slice(),
                                    shape: ishape,
                                    stride: st,
                                },
                                target: TensorRef {
                                    data: dev_t.as_slice(),
                                    shape: ishape,
                                    stride: st,
                                },
                                out: TensorMut {
                                    data: dev_y.as_slice_mut(),
                                    shape: [1, 1],
                                    stride: [1, 1],
                                },
                            },
                        )
                        .expect(concat!("baracuda ", $op));
                    });
                }
            }
            group.finish();
        }
    };
}

percell_loss_bench!(
    bench_mse,
    "mse",
    MseLossPlan,
    MseLossDescriptor,
    MseLossArgs
);
percell_loss_bench!(bench_l1, "l1", L1LossPlan, L1LossDescriptor, L1LossArgs);

/// Class-index targets `[n_rows]`, values in `[0, class_extent)`.
fn class_targets(n_rows: i32, class_extent: i32) -> Vec<i64> {
    (0..n_rows as usize)
        .map(|i| (i as i64) % (class_extent as i64))
        .collect()
}

/// CrossEntropy: logits `[n_rows, class_extent]` + i64 class-index target.
fn bench_cross_entropy<T: LossElem>(
    c: &mut Criterion,
    dt: &'static str,
    baseline: Option<&PytorchBaseline>,
) {
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("cross_entropy/{dt}"));
    for &rows in LOSS_ROW_SWEEP {
        for &cols in LOSS_COL_SWEEP {
            let (n_rows, class_extent) = (rows, cols);
            let shape = format!("R{rows}_C{cols}");
            let ishape = [n_rows, class_extent];
            let ist = contiguous_stride(ishape);
            let tshape = [n_rows];
            let tst = contiguous_stride(tshape);
            let numel = (n_rows * class_extent) as usize;

            let dev_inp = match DeviceBuffer::from_slice(&ctx, &vec![T::fill(0.1); numel]) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_t = match DeviceBuffer::from_slice(&ctx, &class_targets(n_rows, class_extent)) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, 1) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let desc = CrossEntropyLossDescriptor {
                n_rows,
                class_extent,
                reduction: LossReduction::Mean,
                target_kind: CrossEntropyTargetKind::ClassIndex,
                element: T::KIND,
            };
            let plan = match CrossEntropyLossPlan::<T>::select(
                &stream,
                &desc,
                PlanPreference::default(),
            ) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let mut dev_ws: DeviceBuffer<u8> =
                match DeviceBuffer::zeros(&ctx, plan.workspace_size().max(1)) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
            bench_tail!(
                ctx,
                stream,
                group,
                "cross_entropy",
                dt,
                shape,
                dev_y,
                baseline,
                {
                    plan.run(
                        &stream,
                        Workspace::Borrowed(dev_ws.as_slice_mut()),
                        CrossEntropyLossArgs {
                            input: TensorRef {
                                data: dev_inp.as_slice(),
                                shape: ishape,
                                stride: ist,
                            },
                            target: Some(TensorRef {
                                data: dev_t.as_slice(),
                                shape: tshape,
                                stride: tst,
                            }),
                            soft_target: None,
                            out: TensorMut {
                                data: dev_y.as_slice_mut(),
                                shape: [1],
                                stride: [1],
                            },
                        },
                    )
                    .expect("baracuda cross_entropy");
                }
            );
        }
    }
    group.finish();
}

/// NLL: log-probabilities `[n_rows, class_extent]` + i64 class-index target.
fn bench_nll<T: LossElem>(c: &mut Criterion, dt: &'static str, baseline: Option<&PytorchBaseline>) {
    let (ctx, stream) = setup_device();
    let mut group = c.benchmark_group(format!("nll/{dt}"));
    for &rows in LOSS_ROW_SWEEP {
        for &cols in LOSS_COL_SWEEP {
            let (n_rows, class_extent) = (rows, cols);
            let shape = format!("R{rows}_C{cols}");
            let ishape = [n_rows, class_extent];
            let ist = contiguous_stride(ishape);
            let tshape = [n_rows];
            let tst = contiguous_stride(tshape);
            let numel = (n_rows * class_extent) as usize;

            // Input is log-probabilities (≤ 0); a constant -1.0 is a valid
            // log-prob and yields a finite non-zero NLL.
            let dev_inp = match DeviceBuffer::from_slice(&ctx, &vec![T::fill(-1.0); numel]) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let dev_t = match DeviceBuffer::from_slice(&ctx, &class_targets(n_rows, class_extent)) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let mut dev_y: DeviceBuffer<T> = match DeviceBuffer::zeros(&ctx, 1) {
                Ok(b) => b,
                Err(_) => continue,
            };
            let desc = NllLossDescriptor {
                n_rows,
                class_extent,
                reduction: LossReduction::Mean,
                element: T::KIND,
            };
            let plan = match NllLossPlan::<T>::select(&stream, &desc, PlanPreference::default()) {
                Ok(p) => p,
                Err(_) => continue,
            };
            let mut dev_ws: DeviceBuffer<u8> =
                match DeviceBuffer::zeros(&ctx, plan.workspace_size().max(1)) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
            bench_tail!(ctx, stream, group, "nll", dt, shape, dev_y, baseline, {
                plan.run(
                    &stream,
                    Workspace::Borrowed(dev_ws.as_slice_mut()),
                    NllLossArgs {
                        input: TensorRef {
                            data: dev_inp.as_slice(),
                            shape: ishape,
                            stride: ist,
                        },
                        target: TensorRef {
                            data: dev_t.as_slice(),
                            shape: tshape,
                            stride: tst,
                        },
                        out: TensorMut {
                            data: dev_y.as_slice_mut(),
                            shape: [1],
                            stride: [1],
                        },
                    },
                )
                .expect("baracuda nll");
            });
        }
    }
    group.finish();
}

/// Every loss op at one dtype.
fn all_ops<T: LossElem>(c: &mut Criterion, dt: &'static str, b: Option<&PytorchBaseline>) {
    bench_mse::<T>(c, dt, b);
    bench_l1::<T>(c, dt, b);
    bench_cross_entropy::<T>(c, dt, b);
    bench_nll::<T>(c, dt, b);
}

/// Top-level criterion entry - invoked by criterion_main!.
fn benches(c: &mut Criterion) {
    let baseline = PytorchBaseline::load_default();
    let b = baseline.as_ref();
    all_ops::<f32>(c, "f32", b);
    all_ops::<f16>(c, "f16", b);
    all_ops::<bf16>(c, "bf16", b);
}

// `criterion_group!` expands into a `pub fn benches_grp` whose signature
// is fixed by the macro - can't doc-comment it directly, so suppress the
// workspace `missing_docs = deny` lint on the generated fn.
#[allow(missing_docs)]
mod criterion_glue {
    use super::*;
    criterion_group!(benches_grp, benches);
}
criterion_main!(criterion_glue::benches_grp);
